"""Exporte le modèle champion du registry de P6 dans le dépôt (dossier `model/`), avec sa conversion ONNX
servie par l'API, et exporte les résultats complets pour 10,000 échantillons pour servir de référence au monitoring.

A lancer depuis la racine de P8, avec l'environnement de P6 qui contient MLflow et le registry,
complété à la volée par onnxmltools (conversion ONNX) :
    uv run --project ../P6 --with onnxmltools python scripts/exporter_modele.py
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

# Classifieur converti en ONNX, servi par l'API : entrée float32 nommée comme dans api/main.py
FICHIER_ONNX = DOSSIER_MODELE / "model.onnx"
ENTREE_ONNX = "X"

# Référence du monitoring : échantillon d'entraînement et son score.
FICHIER_REFERENCE = Path("monitoring/reference.parquet")
TAILLE_REFERENCE = 10000

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

def exporter_onnx(metadonnees: dict) -> dict:
    """Convertit le classifieur LightGBM du modèle exporté en ONNX (`model/model.onnx`), marqué de la version
    du registry et du run MLflow : l'API refuse de démarrer si ces marques diffèrent de metadonnees.json.

    L'étape de cast float32 du pipeline devient l'entrée float32 du graphe. Les seuils des arbres passent
    en float32 : écarts de score rares, mesurés par scripts/profiler_api.py.
    """
    # Import local : simuler_trafic.py importe ce module dans l'environnement de P6, sans onnxmltools
    from onnx.helper import set_model_props
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    modele = mlflow.pyfunc.load_model(str(DOSSIER_MODELE))
    colonnes = modele.metadata.get_input_schema().input_names()
    # zipmap=False : probabilités en tableau numpy plutôt qu'en liste de dictionnaires
    graphe = convert_lightgbm(modele.get_raw_model()[-1], initial_types=[(ENTREE_ONNX, FloatTensorType([None, len(colonnes)]))], zipmap=False)
    # Traçabilité jusqu'au registry de P6
    graphe.model_version = metadonnees["version"]
    set_model_props(graphe, {"nom": metadonnees["nom"], "version": str(metadonnees["version"]), "run_id": metadonnees["run_id"]})
    FICHIER_ONNX.write_bytes(graphe.SerializeToString())
    return {"fichier": str(FICHIER_ONNX), "taille_mo": round(FICHIER_ONNX.stat().st_size / 1e6, 1), "variables": len(colonnes)}

def decouper_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variables des jeux d'entraînement et de test de P6 (même split que le notebook de modélisation)."""
    tab = pd.read_parquet(DATASET)
    return train_test_split(tab.drop(columns=TARGET), test_size=TEST_SIZE, random_state=SEED, stratify=tab[TARGET])

def exporter_bornes() -> dict:
    """Écrit le min et le max de chaque variable sur le jeu d'entraînement (référence des valeurs plausibles)."""
    X_train, _ = decouper_dataset()
    bornes = {nom: [float(mini), float(maxi)] for nom, (mini, maxi) in X_train.agg(["min", "max"]).T.iterrows()}
    FICHIER_BORNES.write_text(json.dumps(bornes), encoding="utf-8")
    return {"clients": len(X_train), "variables": len(bornes)}

def exporter_reference() -> dict:
    """Écrit la référence du monitoring : un échantillon du jeu d'entraînement, toutes variables, avec le score
    du modèle exporté (colonne `score`)."""
    X_train, _ = decouper_dataset()
    reference = X_train.sample(TAILLE_REFERENCE, random_state=SEED).reset_index(drop=True)
    modele = mlflow.pyfunc.load_model(str(DOSSIER_MODELE))
    colonnes = modele.metadata.get_input_schema().input_names()
    reference["score"] = modele.predict(reference.reindex(columns=colonnes).astype(float))[:, 1]
    FICHIER_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(FICHIER_REFERENCE)
    return {"clients": len(reference), "colonnes": reference.shape[1], "score_moyen": round(float(reference["score"].mean()), 4)}

# --- MAIN ----------------------------------
if __name__ == "__main__":
    metadonnees = exporter_modele()
    print(metadonnees)
    print(exporter_onnx(metadonnees))
    print(exporter_bornes())
    print(exporter_reference())
