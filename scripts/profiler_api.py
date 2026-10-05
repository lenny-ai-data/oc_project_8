"""Profile la route /predict de l'API et trace les gains de l'optimisation

Trois mesures, sur le vrai modèle :
- chronométrage de l'inférence, de l'état initial à l'état optimisé, pas à pas, puis avec ONNX Runtime
- précision d'ONNX Runtime (calcul en float32) sur les 10 000 clients de la référence du monitoring
- cProfile de requêtes complètes (TestClient) sur l'API actuelle

ONNX Runtime est mesuré mais non déployé : l'API sert le modèle MLflow.
Le journal des appels est écrit dans un SQLite temporaire, rien n'est envoyé en base.
Le profil cProfile est sauvegardé dans `data/profil_predict.prof`, pour l'explorer avec snakeviz.

A lancer depuis la racine du projet (groupe optimisation) :
    uv run --group optimisation python -m scripts.profiler_api
    uv run --group optimisation snakeviz data/profil_predict.prof
"""

# --- IMPORT MODULES ----------------------------------
import cProfile
import os
import pstats
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi.testclient import TestClient
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from api import main

# --- CONSTANTES ----------------------------------
# Jeton d'accès forcé, propre au profilage
JETON_PROFIL = "jeton-de-profil"

# Requêtes de chauffe (premiers appels plus lents), puis requêtes mesurées
CHAUFFE = 20
REPETITIONS = 200

# Nombre de fonctions affichées par cProfile, et profil complet sauvegardé pour snakeviz
LIGNES_PROFIL = 25
FICHIER_PROFIL = Path("data/profil_predict.prof")

# Clients d'entraînement et leur score par le modèle MLflow (voir scripts/exporter_modele.py)
FICHIER_REFERENCE = Path("monitoring/reference.parquet")

# ONNX : seule la sortie des probabilités est demandée (la sortie "label" déclenche un avertissement de forme)
SORTIE_ONNX = ["probabilities"]

# Écart de score au-delà duquel un client est compté comme mal reproduit par ONNX
ECART_NOTABLE = 1e-4

# Variantes d'inférence, de l'état initial à l'état optimisé, pas à pas
# (libellé, threads LightGBM, entrée du modèle, appel du modèle)
VARIANTES = [
    ("Initial (DataFrame + MLflow)", -1, "dataframe", lambda X: main.modele.predict(X)),
    ("+ retrait validation MLflow", -1, "dataframe", lambda X: main.pipeline.predict_proba(X)),
    ("+ passage en numpy", -1, "numpy", lambda X: main.pipeline.predict_proba(X)),
    ("+ un seul thread", 1, "numpy", lambda X: main.pipeline.predict_proba(X)),
]

# --- FONCTIONS ----------------------------------
def chronometrer(fonction, *arguments) -> tuple[float, float]:
    """Médiane et p95 du temps d'exécution de `fonction(*arguments)`, en ms, après une chauffe."""
    for _ in range(CHAUFFE):
        fonction(*arguments)
    durees = []
    for _ in range(REPETITIONS):
        debut = time.perf_counter()
        fonction(*arguments)
        durees.append((time.perf_counter() - debut) * 1000)
    return float(np.median(durees)), float(np.percentile(durees, 95))

def entree_modele(features: dict, format_entree: str):
    """Entrée du modèle construite comme dans l'API : DataFrame (avant) ou ligne numpy (après)."""
    if format_entree == "dataframe":
        return pd.DataFrame([features]).reindex(columns=main.colonnes).astype(float)
    return np.array([[features.get(nom) for nom in main.colonnes]], dtype=float)

def inferer(predire, features: dict, format_entree: str):
    """Préparation de l'entrée puis appel du modèle : ce que la route fait pour scorer un client."""
    return predire(entree_modele(features, format_entree))

def regler_threads(n_jobs: int) -> None:
    """Nombre de threads de LightGBM (-1 : tous les coeurs), partagé par MLflow et le pipeline."""
    main.pipeline[-1].set_params(n_jobs=n_jobs)

def convertir_onnx() -> ort.InferenceSession:
    """Classifieur LightGBM converti en ONNX, servi par ONNX Runtime sur un seul thread (comme l'API).

    L'étape de cast float32 du pipeline est remplacée par l'entrée float32 du graphe ONNX.
    """
    graphe = convert_lightgbm(
        main.pipeline[-1], initial_types=[("X", FloatTensorType([None, len(main.colonnes)]))], zipmap=False
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(graphe.SerializeToString(), options, providers=["CPUExecutionProvider"])

def chronometrer_variantes(features: dict, session: ort.InferenceSession) -> None:
    """Temps de préparation de l'entrée + inférence, pour chaque variante, et gain par rapport à l'état initial."""
    print(f"\n=== Inférence (entrée + modèle) : {REPETITIONS} répétitions ===")
    print(f"{'':28} {'médiane':>9} {'p95':>9} {'gain':>6}")
    reference = None
    for libelle, n_jobs, format_entree, predire in VARIANTES:
        regler_threads(n_jobs)
        mediane, p95 = chronometrer(inferer, predire, features, format_entree)
        reference = reference or mediane
        print(f"{libelle:28} {mediane:6.2f} ms {p95:6.2f} ms {reference / mediane:5.1f}x")
    # ONNX Runtime, mesuré mais non déployé : même ligne numpy, convertie en float32
    mediane, p95 = chronometrer(inferer, lambda X: session.run(SORTIE_ONNX, {"X": X.astype(np.float32)}), features, "numpy")
    print(f"{'(ONNX Runtime, non déployé)':28} {mediane:6.2f} ms {p95:6.2f} ms {reference / mediane:5.1f}x")
    # Plancher : le booster LightGBM seul, sur une entrée déjà convertie en float32 par le pipeline
    booster = main.pipeline[-1].booster_
    X_cast = main.pipeline[:-1].transform(entree_modele(features, "numpy"))
    mediane, p95 = chronometrer(booster.predict, X_cast)
    print(f"{'(booster LightGBM seul)':28} {mediane:6.2f} ms {p95:6.2f} ms")

def comparer_precision_onnx(session: ort.InferenceSession) -> None:
    """Écart de score entre ONNX Runtime (float32) et le pipeline servi par l'API, sur les clients de la référence."""
    X = pd.read_parquet(FICHIER_REFERENCE)[main.colonnes].to_numpy(dtype=float)
    api = main.pipeline.predict_proba(X)[:, 1]
    onnx = session.run(SORTIE_ONNX, {"X": X.astype(np.float32)})[0][:, 1]
    ecarts = np.abs(onnx - api)
    seuil = main.metadonnees["seuil"]
    print(f"\n=== Précision d'ONNX Runtime : {len(X)} clients de la référence ===")
    print(f"écart de score : médiane {np.median(ecarts):.1e}, p99 {np.percentile(ecarts, 99):.1e}, max {ecarts.max():.1e}")
    print(f"clients à plus de {ECART_NOTABLE:.0e} : {int((ecarts > ECART_NOTABLE).sum())}")
    print(f"décisions changées au seuil {seuil:.3f} : {int(((api >= seuil) != (onnx >= seuil)).sum())}")
    print(f"clients à moins de {ecarts.max():.1e} du seuil (décision fragile) : {int((np.abs(api - seuil) < ecarts.max()).sum())}")

def profiler_requetes(client: TestClient, corps: dict) -> None:
    """Chronométrage et cProfile de requêtes complètes à /predict sur l'API actuelle."""
    mediane, p95 = chronometrer(lambda: client.post("/predict", json=corps))
    print(f"\n=== Requête complète (API actuelle, écriture du journal comprise) : {REPETITIONS} répétitions ===")
    print(f"médiane {mediane:.2f} ms   p95 {p95:.2f} ms")

    profil = cProfile.Profile()
    profil.enable()
    for _ in range(REPETITIONS):
        client.post("/predict", json=corps)
    profil.disable()
    print(f"\n=== cProfile : {REPETITIONS} requêtes /predict, tri par temps cumulé ===")
    pstats.Stats(profil).strip_dirs().sort_stats("cumulative").print_stats(LIGNES_PROFIL)
    FICHIER_PROFIL.parent.mkdir(parents=True, exist_ok=True)
    profil.dump_stats(FICHIER_PROFIL)
    print(f"Profil complet : {FICHIER_PROFIL}")

# --- MAIN ----------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as dossier:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(dossier) / 'logs.db'}"
        main.JETON = JETON_PROFIL
        with TestClient(main.app, headers={"Authorization": f"Bearer {JETON_PROFIL}"}) as client:
            corps = {"features": main.EXEMPLES["client_1"]["value"]["features"]}
            print(f"Client d'exemple 1 : {len(corps['features'])} variables renseignées sur {len(main.colonnes)}")
            session = convertir_onnx()
            chronometrer_variantes(corps["features"], session)
            # Réglage de l'API remis en place avant la comparaison et le profilage des requêtes
            regler_threads(1)
            comparer_precision_onnx(session)
            profiler_requetes(client, corps)
            # Libère le fichier SQLite avant la suppression du dossier temporaire (Windows)
            main.moteur.dispose()
