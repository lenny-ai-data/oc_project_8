"""Tableau de bord du monitoring : suivi opérationnel de l'API et dérive des données en production.

Relit le journal des appels à chaque ouverture (cache de 10 minutes) et le compare à la référence d'entraînement.
    uv run --group monitoring python -m streamlit run monitoring/tableau_de_bord.py
"""

# --- IMPORT MODULES ----------------------------------

import json
import os

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
