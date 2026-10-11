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
ROUGE = "#D1495B"
COULEURS_NIVEAU = {"stable": PURPLE, "vigilance": GOLD_DARK, "critique": ROUGE}

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

def graphe_distributions(histogrammes: pd.DataFrame, titre_x: str, seuil: float | None = None) -> alt.LayerChart | alt.Chart:
    """Densités de la référence et de la fenêtre courante en marches, avec un seuil vertical facultatif."""
    courbes = alt.Chart(histogrammes).mark_line(interpolate="step").encode(
        x=alt.X("valeur:Q", title=titre_x),
        y=alt.Y("densite:Q", title="Densité"),
        color=alt.Color("groupe:N", title=None, scale=alt.Scale(domain=["Référence", "Fenêtre courante"], range=[GRIS, GOLD_DARK])),
    )
    if seuil is None:
        return courbes
    return courbes + alt.Chart(pd.DataFrame({"valeur": [seuil]})).mark_rule(color=GRIS, strokeDash=[4, 4]).encode(x="valeur:Q")

def valeurs(donnees: pd.DataFrame, colonne: str) -> pd.Series:
    """Valeurs renseignées d'une colonne, l'âge en années remplaçant DAYS_BIRTH (jours avant la demande, négatifs)."""
    serie = donnees[colonne].dropna()
    return -serie / 365.25 if colonne == "DAYS_BIRTH" else serie

def persistance(psi_courant: pd.Series, psi_precedent: pd.Series | None) -> pd.Series:
    """Persistance de chaque colonne en dérive : confirmée si la fenêtre précédente dépassait aussi le seuil,
    à confirmer sinon, non évaluable sans fenêtre précédente complète. Vide pour une colonne stable."""
    if psi_precedent is None:
        statut = pd.Series("non évaluable", index=psi_courant.index)
    else:
        statut = (psi_precedent >= analyse.SEUIL_PSI).map({True: "confirmée", False: "à confirmer"})
    return statut.where(psi_courant >= analyse.SEUIL_PSI, "")

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
    # Révision déployée (commit) : la plus récente par défaut, pour ne pas mélanger les versions de l'API.
    # Appels sans révision : API lancée en local
    revisions = lignes.groupby(lignes["revision"].fillna("locale"))["horodatage"].min().sort_values(ascending=False)
    revision = st.selectbox("Révision de l'API", ["Toutes", *revisions.index], index=1 if len(revisions) else 0,
                            format_func=lambda r: r if r == "Toutes" else f"{r[:7]} (depuis le {revisions[r]:%d/%m %H:%M})")
    if revision != "Toutes":
        lignes = lignes[lignes["revision"].fillna("locale") == revision]
        entrees = entrees[entrees.index.isin(lignes.index)]
    taille = st.select_slider("Fenêtre (appels réussis)", TAILLES_FENETRE, value=FENETRE_DEFAUT)
    if st.button("Rafraîchir les données"):
        charger_journal.clear()
        st.rerun()
    st.caption(f"Base : {'Supabase (Postgres)' if os.getenv('DATABASE_URL') else 'SQLite locale'}")
    st.caption(f"Appels journalisés : {len(lignes):,}".replace(",", " "))
    st.caption(f"Dernier appel : {lignes['horodatage'].max():%d/%m/%Y %H:%M} UTC")

# Révision trop peu appelée : PSI sans valeur (voire incalculable si une colonne n'a aucune valeur)
if len(entrees) < TAILLES_FENETRE[0]:
    st.warning(f"{len(entrees)} appel(s) réussi(s) pour cette révision : trop peu pour l'analyse "
               f"(minimum {TAILLES_FENETRE[0]}). Choisir une autre révision dans la barre latérale.")
    st.stop()

# Fenêtres disjointes d'appels réussis : la dernière est la fenêtre courante. 
# On identifie aussi tous les appels reçus depuis son début (erreurs comprises)
fenetres = analyse.fenetres_disjointes(entrees, taille)
# La plus ancienne fenêtre, souvent incomplète, a un PSI trop bruité : on l'écarte s'il en reste d'autres
if len(fenetres) > 1 and len(fenetres[0]) < taille:
    fenetres = fenetres[1:]
fenetre = fenetres[-1]
appels_fenetre = lignes[lignes["horodatage"] >= lignes.loc[fenetre.index[0], "horodatage"]]
reussis = appels_fenetre[appels_fenetre["statut_http"] == 200]

st.title("Monitoring du scoring crédit")
st.caption(f"Fenêtre courante : {len(fenetre)} derniers appels réussis, depuis le "
           f"{appels_fenetre['horodatage'].min():%d/%m/%Y %H:%M} UTC. Référence : {len(reference):,} clients d'entraînement.")

# PSI de chaque fenêtre face à la référence : une colonne par fenêtre, datée par son dernier appel
historique = pd.DataFrame({lignes.loc[f.index[-1], "horodatage"]: analyse.psi_colonnes(f, reference, colonnes) for f in fenetres})
psi_courant = historique.iloc[:, -1]
# La fenêtre précédente (complète) confirme ou non une alerte
psi_precedent = historique.iloc[:, -2] if len(fenetres) > 1 else None
niveaux = psi_courant.map(analyse.niveau_psi)
statuts = persistance(psi_courant, psi_precedent)

# Indicateurs clés de la fenêtre courante
refus_fenetre = (fenetre["score"] >= seuil).mean()
refus_reference = (reference["score"] >= seuil).mean()
kpi = st.columns(5)
kpi[0].metric("Appels dans la fenêtre", f"{len(appels_fenetre):,}".replace(",", " "))
kpi[1].metric("Taux d'erreur", f"{(appels_fenetre['statut_http'] != 200).mean():.1%}")
kpi[2].metric("Latence p95", f"{reussis['latence_ms'].quantile(0.95):.0f} ms")
kpi[3].metric("Taux de refus", f"{refus_fenetre:.1%}", f"{(refus_fenetre - refus_reference) * 100:+.1f} pts vs référence",
              delta_color="inverse")
kpi[4].metric("Colonnes en dérive", f"{(niveaux != 'stable').sum()} / {len(colonnes)}",
              f"{(statuts == 'confirmée').sum()} confirmée(s)", delta_color="off", delta_arrow="off")

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
    gauche.metric("Taux de refus", f"{refus_fenetre:.1%}", f"{(refus_fenetre - refus_reference) * 100:+.1f} pts vs référence", delta_color="inverse")
    gauche.metric("Score moyen", f"{fenetre['score'].mean():.3f}",
                  f"{fenetre['score'].mean() - reference['score'].mean():+.3f} vs référence", delta_color="inverse")
    if abs(ecart) > ECART_REFUS_MAX:
        gauche.warning(f"Taux de refus à {ecart * 100:+.1f} points de la référence (tol. {ECART_REFUS_MAX * 100:.0f} points) : la population a changé.")
    else:
        gauche.success(f"Taux de refus dans la tolérance de {ECART_REFUS_MAX * 100:.0f} points autour de la référence.")
    # Histogrammes en densité, scores écrêtés à 0,5 pour la lisibilité, et seuil de décision
    histogrammes = densites({"Référence": reference["score"].clip(upper=0.5), "Fenêtre courante": fenetre["score"].clip(upper=0.5)},
                            np.linspace(0, 0.5, 51))
    droite.altair_chart(graphe_distributions(histogrammes, "Score (probabilité de défaut)", seuil), width="stretch")
    droite.caption(f"Ligne pointillée : seuil de décision ({seuil}), refus au-dessus.")

with onglet_drift:
    # --- Fiabilité de la fenêtre : sous 1 000 appels, le bruit du PSI dépasse le seuil (étude du notebook)
    if len(fenetre) < taille:
        st.warning(f"Fenêtre incomplète : {len(fenetre)} appels réussis sur {taille}.")
    if len(fenetre) < FENETRE_DEFAUT:
        st.caption(f"Sous {FENETRE_DEFAUT} appels, le PSI est bruité : fausses alertes fréquentes (40 % des fenêtres à 500 appels).")

    # --- Synthèse des alertes
    critiques, vigilances = list(niveaux.index[niveaux == "critique"]), list(niveaux.index[niveaux == "vigilance"])
    if critiques:
        st.error(f"Dérive critique (PSI ≥ {analyse.SEUIL_PSI_CRITIQUE}) : {', '.join(critiques)}")
    if vigilances:
        st.warning(f"Dérive à surveiller (PSI ≥ {analyse.SEUIL_PSI}) : {', '.join(vigilances)}")
    if not critiques and not vigilances:
        st.success(f"Aucune colonne en dérive : PSI maximal {psi_courant.max():.3f}.")

    # --- PSI de la fenêtre courante : barres colorées par niveau, seuils en pointillés
    st.subheader("PSI de la fenêtre courante")
    gauche, droite = st.columns([3, 2])
    tableau = pd.DataFrame({"psi": psi_courant, "niveau": niveaux, "persistance": statuts}).rename_axis("colonne").reset_index()
    barres = alt.Chart(tableau).mark_bar().encode(
        x=alt.X("psi:Q", title="PSI"),
        y=alt.Y("colonne:N", sort="-x", title=None),
        color=alt.Color("niveau:N", title="Niveau",
                        scale=alt.Scale(domain=list(COULEURS_NIVEAU), range=list(COULEURS_NIVEAU.values()))),
        tooltip=["colonne", alt.Tooltip("psi:Q", format=".3f"), "niveau", "persistance"],
    )
    lignes_seuil = alt.Chart(pd.DataFrame({"psi": [analyse.SEUIL_PSI, analyse.SEUIL_PSI_CRITIQUE]})).mark_rule(
        color=GRIS, strokeDash=[4, 4]).encode(x="psi:Q")
    gauche.altair_chart(barres + lignes_seuil, width="stretch")
    # Détail des colonnes en dérive : la fenêtre précédente confirme ou non l'alerte
    en_derive = tableau[tableau["niveau"] != "stable"].set_index("colonne").sort_values("psi", ascending=False)
    if not en_derive.empty:
        en_derive["psi précédent"] = psi_precedent[en_derive.index] if psi_precedent is not None else np.nan
        en_derive["renseignées"] = fenetre[en_derive.index].notna().mean()
        droite.dataframe(en_derive[["psi", "psi précédent", "niveau", "persistance", "renseignées"]],
                         column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                                        "psi précédent": st.column_config.NumberColumn("PSI précédent", format="%.3f"),
                                        "renseignées": st.column_config.NumberColumn("Renseignées", format="percent")})
    droite.caption(f"Vigilance à partir de {analyse.SEUIL_PSI}, critique à partir de {analyse.SEUIL_PSI_CRITIQUE}. "
                   "Une alerte est confirmée quand la fenêtre précédente (disjointe et complète) dépassait aussi le seuil. "
                   "Les colonnes peu renseignées sont les plus bruitées.")

    # --- Historique : PSI des fenêtres disjointes successives
    st.subheader("Historique par fenêtre")
    if len(fenetres) < 2:
        st.info("Une seule fenêtre : pas encore d'historique.")
    else:
        choix = st.multiselect("Colonnes", colonnes, default=list(en_derive.index) or list(psi_courant.nlargest(3).index))
        st.line_chart(historique.loc[choix].T, x_label="Fin de la fenêtre (UTC)", y_label="PSI")
        st.caption(f"{len(fenetres)} fenêtres complètes de {taille} appels réussis.")

    # --- Distribution d'une colonne : référence face à la fenêtre courante
    st.subheader("Distribution d'une colonne")
    colonne = st.selectbox("Colonne", colonnes, index=colonnes.index(psi_courant.idxmax()))
    bornes = np.histogram_bin_edges(valeurs(reference, colonne), bins=30)
    histogrammes = densites({"Référence": valeurs(reference, colonne), "Fenêtre courante": valeurs(fenetre, colonne)}, bornes)
    st.altair_chart(graphe_distributions(histogrammes, "Âge (années)" if colonne == "DAYS_BIRTH" else colonne), width="stretch")
    st.caption(f"PSI {psi_courant[colonne]:.3f}, valeurs renseignées : {fenetre[colonne].notna().mean():.0%} "
               f"de la fenêtre contre {reference[colonne].notna().mean():.0%} de la référence.")

    # --- Rapport Evidently de la fenêtre courante : généré seulement au clic (environ 1 s, 5 Mo), sans relancer la page
    st.subheader("Rapport Evidently")
    fin_fenetre = lignes.loc[fenetre.index[-1], "horodatage"]
    st.download_button(
        "Télécharger le rapport HTML",
        data=lambda: analyse.rapport_drift(fenetre, reference, colonnes[:-1]).get_html_str(as_iframe=False),
        file_name=f"rapport_drift_{fin_fenetre:%Y%m%d_%H%M}.html",
        mime="text/html",
        on_click="ignore",
    )
    st.caption("Rapport complet d'Evidently sur la fenêtre courante (mêmes colonnes et même PSI) : distributions et "
               "test de chaque colonne. Le verdict global d'Evidently (50 % de colonnes en dérive) est à lire avec "
               "prudence : une dérive concentrée sur quelques variables ne le déclenche pas.")
