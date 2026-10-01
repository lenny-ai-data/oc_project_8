"""Chargement des données du monitoring : journal des appels, phases de la simulation, référence.

Partagé par le notebook d'analyse du drift et le tableau de bord. Dépendances : groupe `monitoring`
    uv sync --group monitoring
"""

# --- IMPORT MODULES ----------------------------------

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import select

from api import journal

# --- CONSTANTES ----------------------------------

# Chemins depuis la racine du dépôt
RACINE = Path(__file__).resolve().parent.parent
FICHIER_ENV = RACINE / ".env"
FICHIER_SIMULATION = RACINE / "data" / "simulation.json"
FICHIER_REFERENCE = RACINE / "monitoring" / "reference.parquet"
DOSSIER_MODELE = RACINE / "model"

# Écart toléré entre l'horloge du poste et celle du serveur
MARGE_HORLOGE_S = 10

# Dérive : une colonne dérive si son PSI atteint le seuil ; le jeu de données dérive si la part de
# colonnes en dérive atteint DRIFT_SHARE (valeurs par défaut d'Evidently)
SEUIL_PSI = 0.1
DRIFT_SHARE = 0.5

# --- FONCTIONS SUPPORT ----------------------------------

def charger_env() -> None:
    """Lit le fichier .env et définit les variables absentes de l'environnement (DATABASE_URL en particulier).
    Un notebook ne lit pas .env tout seul."""
    if not FICHIER_ENV.exists():
        return
    for ligne in FICHIER_ENV.read_text(encoding="utf-8").splitlines():
        nom, separateur, valeur = ligne.partition("=")
        if separateur and not nom.lstrip().startswith("#"):
            os.environ.setdefault(nom.strip(), valeur.strip())

def charger_journal() -> pd.DataFrame:
    """Journal des appels de /predict, du plus ancien au plus récent."""
    lignes = pd.read_sql(select(journal.TABLE).order_by(journal.TABLE.c.horodatage), journal.creer_moteur())
    # SQLite renvoie des dates sans fuseau, Postgres en UTC : tout en UTC
    lignes["horodatage"] = pd.to_datetime(lignes["horodatage"], utc=True)
    return lignes

def charger_phases() -> list[dict]:
    """Phases de la dernière simulation."""
    return json.loads(FICHIER_SIMULATION.read_text(encoding="utf-8"))["phases"]

def ajouter_phase(lignes: pd.DataFrame, phases: list[dict]) -> pd.DataFrame:
    """Copie du journal avec la colonne `phase` (les appels hors simulation sont marqués « hors simulation »).

    Les horaires viennent de deux horloges (poste et serveur), décalées de quelques secondes : les phases,
    sont donc découpées par ordre et par effectif."""
    lignes = lignes.copy()
    lignes["phase"] = "hors simulation"
    # Appels de la simulation avec fenêtre élargie
    debut = pd.Timestamp(phases[0]["debut"]) - pd.Timedelta(seconds=MARGE_HORLOGE_S)
    fin = pd.Timestamp(phases[-1]["fin"]) + pd.Timedelta(seconds=MARGE_HORLOGE_S)
    simulation = lignes.index[lignes["horodatage"].between(debut, fin)]
    position = 0
    for phase in phases:
        # Une requête sans réponse ("reseau") n'a pas laissé de ligne dans le journal
        effectif = sum(nombre for code, nombre in phase["codes"].items() if code != "reseau")
        lignes.loc[simulation[position:position + effectif], "phase"] = phase["nom"]
        position += effectif
    return lignes

def entrees_a_plat(lignes: pd.DataFrame) -> pd.DataFrame:
    """Variables et score des appels réussis : une ligne par appel, une colonne par variable
    (non renseignées à NaN) et une colonne `score`."""
    reussis = lignes[lignes["statut_http"] == 200]
    entrees = pd.DataFrame(reussis["entrees"].tolist(), index=reussis.index)
    entrees["score"] = reussis["score"]
    return entrees

def cause_erreur(erreur) -> str:
    """Cause d'une erreur du journal : catégories de l'API (422 métier, dict), type incorrect (422 Pydantic,
    liste) ou erreur serveur (500 et 503, message texte)."""
    if isinstance(erreur, dict):
        return " + ".join(sorted(erreur))
    return "type incorrect" if isinstance(erreur, list) else "erreur serveur"

# --- FONCTIONS DE MONITORING ----------------------------------

def importance_variables() -> pd.Series:
    """Part de l'importance (gain) de chaque variable dans le modèle en production, de la plus forte à la plus faible.
    On regarde directement le gain associé à chaque variable dans le modèle, ce n'est pas une estimation SHAP."""
    pipeline = mlflow.sklearn.load_model(str(DOSSIER_MODELE))
    gain = pd.Series(pipeline.named_steps["modele"].booster_.feature_importance(importance_type="gain"),index=pipeline.feature_names_in_)
    return (gain / gain.sum()).sort_values(ascending=False)

def rapport_drift(actuel: pd.DataFrame, reference: pd.DataFrame, variables: list[str]):
    """Rapport Evidently : dérive (test PSI) de chaque variable et du score de actuel par rapport à reference."""
    colonnes = [*variables, "score"]
    rapport = Report([DataDriftPreset(columns=colonnes, method="psi", threshold=SEUIL_PSI, drift_share=DRIFT_SHARE)])
    return rapport.run(actuel[colonnes], reference[colonnes])

def psi_par_colonne(rapport) -> pd.Series:
    """PSI de chaque colonne d'un rapport de dérive."""
    metriques = [m for m in rapport.dict()["metrics"] if m["config"]["type"].endswith("ValueDrift")]
    return pd.Series({m["config"]["column"]: m["value"] for m in metriques})

def _part_des_classes(comptes: np.ndarray) -> np.ndarray:
    """Part de clients par classe : une classe vide prend une petite valeur plutôt que 0 (log défini)."""
    parts = comptes / comptes.sum()
    plancher = parts[parts > 0].min() / 10**6 if parts[parts > 0].min() <= 0.0001 else 0.0001
    return np.where(parts == 0, plancher, parts)

def psi(reference: pd.Series, actuel: pd.Series) -> float:
    """PSI d'une colonne, similaire à Evidently mais sans passer par un rapport (plus rapide)."""
    ref, cur = reference.dropna().to_numpy(), actuel.dropna().to_numpy()
    # Variables continues
    if len(np.unique(ref)) > 20:
        bornes = np.histogram_bin_edges(np.concatenate([ref, cur]), bins="sturges")
        comptes_ref, comptes_cur = np.histogram(ref, bornes)[0], np.histogram(cur, bornes)[0]
    else:
        valeurs = np.union1d(ref, cur)
        comptes_ref = np.bincount(np.searchsorted(valeurs, ref), minlength=len(valeurs))
        comptes_cur = np.bincount(np.searchsorted(valeurs, cur), minlength=len(valeurs))
    part_ref, part_cur = _part_des_classes(comptes_ref), _part_des_classes(comptes_cur)
    return float(np.sum((part_ref - part_cur) * np.log(part_ref / part_cur)))

def psi_colonnes(actuel: pd.DataFrame, reference: pd.DataFrame, colonnes: list[str]) -> pd.Series:
    """PSI de chaque colonne de `actuel` par rapport à `reference`."""
    return pd.Series({colonne: psi(reference[colonne], actuel[colonne]) for colonne in colonnes})

def verdict_global(rapport) -> dict:
    """Nombre et part de colonnes en dérive, et verdict sur le jeu de données (part >= DRIFT_SHARE)."""
    metrique = next(m for m in rapport.dict()["metrics"] if m["config"]["type"].endswith("DriftedColumnsCount"))
    return {"en_derive": int(metrique["value"]["count"]), "part": metrique["value"]["share"],
            "derive_globale": metrique["value"]["share"] >= DRIFT_SHARE}
