"""Tableau de bord du monitoring : suivi opérationnel de l'API et dérive des données en production.

Relit le journal des appels à chaque ouverture (cache de 10 minutes) et le compare à la référence d'entraînement.
    uv run --group monitoring python -m streamlit run monitoring/tableau_de_bord.py
"""

# --- IMPORT MODULES ----------------------------------

import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from monitoring import analyse

# --- CONSTANTES ----------------------------------

# Fenêtre d'observation : les N derniers appels réussis (1 000 = minimum fiable mesuré dans le notebook)
TAILLES_FENETRE = [500, 1000, 1500, 2000]
FENETRE_DEFAUT = 1000

# Variables surveillées : les plus importantes pour le modèle, avec le score
N_VARIABLES = 20

# Durée de vie du cache du journal (secondes)
TTL_JOURNAL = 600

# Suivi : médianes glissantes des temps de réponse, écart de taux de refus toléré face à la référence
GLISSEMENT = 50
ECART_REFUS_MAX = 0.05

# Charte graphique (reprise du notebook)
PURPLE = "#C9BCE3"
GOLD_DARK = "#FFE186"
GRIS = "#929292"

# --- FONCTIONS ----------------------------------

@st.cache_data(ttl=TTL_JOURNAL, show_spinner="Lecture du journal...")
def charger_journal() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Journal des appels et, pour les appels réussis, variables et score à plat."""
    lignes = analyse.charger_journal()
    return lignes, analyse.entrees_a_plat(lignes)

@st.cache_data(show_spinner="Lecture de la référence...")
def charger_reference() -> pd.DataFrame:
    """Échantillon d'entraînement (10 000 clients) avec le score du modèle."""
    return pd.read_parquet(analyse.FICHIER_REFERENCE)

@st.cache_data
def charger_seuil() -> float:
    """Seuil de décision exporté avec le modèle."""
    return json.loads((analyse.DOSSIER_MODELE / "metadonnees.json").read_text(encoding="utf-8"))["seuil"]

@st.cache_data(show_spinner="Chargement du modèle...")
def variables_surveillees() -> list[str]:
    """Les N_VARIABLES plus importantes du modèle en production (gain LightGBM)."""
    return list(analyse.importance_variables().head(N_VARIABLES).index)

def densites(groupes: dict[str, pd.Series], bornes: np.ndarray) -> pd.DataFrame:
    """Histogrammes en densité de plusieurs groupes sur des classes communes, au format long
    (une ligne par classe et par groupe) attendu par Altair."""
    centres = (bornes[:-1] + bornes[1:]) / 2
    return pd.concat([
        pd.DataFrame({"valeur": centres, "densite": np.histogram(serie.dropna(), bornes, density=True)[0], "groupe": nom})
        for nom, serie in groupes.items()
    ])

# --- MAIN ----------------------------------

st.set_page_config(page_title="Monitoring du scoring crédit", layout="wide")
analyse.charger_env()

lignes, entrees = charger_journal()
reference = charger_reference()
seuil = charger_seuil()
colonnes = [*variables_surveillees(), "score"]
if entrees.empty:
    st.warning("Aucun appel réussi dans le journal.")
    st.stop()

# Barre latérale : réglages et source des données
with st.sidebar:
    st.header("Réglages")
    taille = st.select_slider("Fenêtre (appels réussis)", TAILLES_FENETRE, value=FENETRE_DEFAUT)
    if st.button("Rafraîchir les données"):
        charger_journal.clear()
        st.rerun()
    st.caption(f"Base : {'Supabase (Postgres)' if os.getenv('DATABASE_URL') else 'SQLite locale'}")
    st.caption(f"Dernier appel : {lignes['horodatage'].max():%d/%m/%Y %H:%M} UTC")

# Fenêtre courante : les derniers appels réussis, et tous les appels reçus depuis son début
fenetre = entrees.tail(taille)
appels_fenetre = lignes[lignes["horodatage"] >= lignes.loc[fenetre.index[0], "horodatage"]]

st.title("Monitoring du scoring crédit")
st.caption(f"Fenêtre courante : {len(fenetre)} derniers appels réussis, depuis le "
           f"{appels_fenetre['horodatage'].min():%d/%m/%Y %H:%M} UTC. Référence : {len(reference):,} clients d'entraînement.".replace(",", " "))

# Indicateurs clés de la fenêtre courante
refus_fenetre = (fenetre["score"] >= seuil).mean()
refus_reference = (reference["score"] >= seuil).mean()
kpi = st.columns(4)
kpi[0].metric("Appels journalisés", f"{len(lignes):,}".replace(",", " "))
kpi[1].metric("Taux d'erreur", f"{(appels_fenetre['statut_http'] != 200).mean():.1%}")
kpi[2].metric("Latence p95", f"{appels_fenetre['latence_ms'].quantile(0.95):.0f} ms")
kpi[3].metric("Taux de refus", f"{refus_fenetre:.1%}", f"{(refus_fenetre - refus_reference) * 100:+.1f} pts vs référence",
              delta_color="inverse")

onglet_suivi, onglet_drift = st.tabs(["Suivi opérationnel", "Dérive des données"])

with onglet_suivi:
    # --- Erreurs : appels refusés (422) ou en échec (5xx) dans la fenêtre, par cause
    st.subheader("Erreurs")
    erreurs = appels_fenetre[appels_fenetre["statut_http"] != 200]
    gauche, droite = st.columns([1, 3])
    gauche.metric("Appels en erreur", len(erreurs))
    gauche.metric("Pannes serveur (5xx)", int((erreurs["statut_http"] >= 500).sum()))
    if erreurs.empty:
        droite.info("Aucune erreur dans la fenêtre.")
    else:
        par_cause = erreurs["erreur"].map(analyse.cause_erreur).value_counts().rename_axis("cause").reset_index(name="appels")
        droite.bar_chart(par_cause, x="cause", y="appels", horizontal=True, x_label="", y_label="Appels en erreur", color=PURPLE)

    # --- Temps de réponse : latence de l'API et temps d'inférence du modèle seul, appels réussis
    st.subheader("Temps de réponse")
    reussis = appels_fenetre[appels_fenetre["statut_http"] == 200]
    gauche, droite = st.columns([1, 3])
    resume = reussis[["latence_ms", "temps_inference_ms"]].quantile([0.5, 0.95, 1]).round(0)
    resume.index = ["Médiane", "95e centile", "Maximum"]
    gauche.dataframe(resume.rename(columns={"latence_ms": "Latence (ms)", "temps_inference_ms": "Inférence (ms)"}))
    gauche.metric("Part du modèle dans la latence", f"{(reussis['temps_inference_ms'] / reussis['latence_ms']).median():.0%}")
    # Courbes glissantes au fil des appels : la médiane donne le régime courant, le 95e centile les pics
    durees = reussis.set_index("horodatage")
    glissant = pd.DataFrame({
        "Latence, 95e centile": durees["latence_ms"].rolling(GLISSEMENT).quantile(0.95),
        "Latence, médiane": durees["latence_ms"].rolling(GLISSEMENT).median(),
        "Inférence, médiane": durees["temps_inference_ms"].rolling(GLISSEMENT).median(),
    }).dropna()
    droite.line_chart(glissant, x_label="Heure (UTC)", y_label=f"ms (sur {GLISSEMENT} appels glissants)",
                      color=[GOLD_DARK, GRIS, PURPLE])

    # --- Scores : distribution de la fenêtre face à la référence, et taux de refus
    st.subheader("Scores")
    gauche, droite = st.columns([1, 3])
    ecart = refus_fenetre - refus_reference
    gauche.metric("Taux de refus de la référence", f"{refus_reference:.1%}")
    gauche.metric("Score moyen", f"{fenetre['score'].mean():.3f}",
                  f"{fenetre['score'].mean() - reference['score'].mean():+.3f} vs référence", delta_color="inverse")
    if abs(ecart) > ECART_REFUS_MAX:
        gauche.warning(f"Taux de refus à {ecart * 100:+.1f} points de la référence (tolérance {ECART_REFUS_MAX * 100:.0f} points) : "
                       "la population a changé, voir l'onglet Dérive.")
    else:
        gauche.success(f"Taux de refus dans la tolérance de {ECART_REFUS_MAX * 100:.0f} points autour de la référence.")
    # Histogrammes en densité, scores écrêtés à 0,6 pour la lisibilité, et seuil de décision
    histogrammes = densites({"Référence": reference["score"].clip(upper=0.6), "Fenêtre courante": fenetre["score"].clip(upper=0.6)},
                            np.linspace(0, 0.6, 61))
    courbes = alt.Chart(histogrammes).mark_line(interpolate="step").encode(
        x=alt.X("valeur:Q", title="Score (probabilité de défaut)"),
        y=alt.Y("densite:Q", title="Densité"),
        color=alt.Color("groupe:N", title=None, scale=alt.Scale(domain=["Référence", "Fenêtre courante"], range=[GRIS, GOLD_DARK])),
    )
    ligne_seuil = alt.Chart(pd.DataFrame({"valeur": [seuil]})).mark_rule(color=GRIS, strokeDash=[4, 4]).encode(x="valeur:Q")
    droite.altair_chart(courbes + ligne_seuil, width="stretch")
    droite.caption(f"Ligne pointillée : seuil de décision ({seuil}), refus au-dessus.")

with onglet_drift:
    st.info("En construction : PSI des variables surveillées.")
