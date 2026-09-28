"""Exporte le modèle champion du registry de P6 dans le dépôt (dossier `model/`).

A lancer depuis la racine de P8, avec l'environnement de P6 qui contient MLflow et le registry :
    uv run --project ../P6 python scripts/exporter_modele.py
"""

# --- IMPORT MODULES ----------------------------------
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

# --- CONSTANTES ----------------------------------
# Registry de P6
TRACKING_URI = "sqlite:///../P6/mlflow.db"
NOM_MODELE = "scoring-credit"
ALIAS = "champion"

# Destination dans le dépôt
DOSSIER_MODELE = Path("model")
FICHIER_METADONNEES = DOSSIER_MODELE / "metadonnees.json"
FICHIER_BORNES = DOSSIER_MODELE / "bornes.json"

# Jeu d'entraînement de P6 : même split que le notebook de modélisation (test_size, seed, stratification)
DATASET = Path("../P6/data/processed/dataset.parquet")
TARGET = "TARGET"
TEST_SIZE = 0.2
SEED = 42

# Fichiers exportés inutiles pour P8 : environnement complet de P6 (les dépendances
# de P8 sont dans pyproject.toml) et méta du registry (reprises dans metadonnees.json)
FICHIERS_INUTILES = ("requirements.txt", "conda.yaml", "python_env.yaml", "registered_model_meta")

# --- FONCTIONS ----------------------------------
def exporter_modele() -> dict:
    """Copie les artefacts du champion dans `model/` et y ajoute seuil et version."""
    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(NOM_MODELE, ALIAS)

    # Repartir d'un dossier vide pour ne pas mélanger deux versions
    if DOSSIER_MODELE.exists():
        shutil.rmtree(DOSSIER_MODELE)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{NOM_MODELE}@{ALIAS}", dst_path=str(DOSSIER_MODELE)
    )
    for nom in FICHIERS_INUTILES:
        (DOSSIER_MODELE / nom).unlink(missing_ok=True)

    metadonnees = {
        "nom": NOM_MODELE,
        "alias": ALIAS,
        "version": int(version.version),
        "run_id": version.run_id,
        "seuil": float(version.tags["seuil"]),
        "tags": version.tags,
    }
    FICHIER_METADONNEES.write_text(json.dumps(metadonnees, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadonnees

def exporter_bornes() -> dict:
    """Écrit le min et le max de chaque variable sur le jeu d'entraînement (référence des valeurs plausibles)."""
    tab = pd.read_parquet(DATASET)
    X_train, _ = train_test_split(tab.drop(columns=TARGET), test_size=TEST_SIZE, random_state=SEED, stratify=tab[TARGET])
    bornes = {nom: [float(mini), float(maxi)] for nom, (mini, maxi) in X_train.agg(["min", "max"]).T.iterrows()}
    FICHIER_BORNES.write_text(json.dumps(bornes), encoding="utf-8")
    return {"clients": len(X_train), "variables": len(bornes)}

# --- MAIN ----------------------------------
if __name__ == "__main__":
    print(exporter_modele())
    print(exporter_bornes())
