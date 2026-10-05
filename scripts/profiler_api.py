"""Profile la route /predict de l'API et trace les gains de l'optimisation

Trois mesures, sur le vrai modèle :
- chronométrage de l'inférence, de l'état initial à l'état optimisé, pas à pas.
- cProfile de requêtes complètes (TestClient) sur l'API actuelle

Le journal des appels est écrit dans un SQLite temporaire, rien n'est envoyé en base.
Le profil cProfile est sauvegardé dans `data/profil_predict.prof`, pour l'explorer avec snakeviz.

A lancer depuis la racine du projet :
    uv run python -m scripts.profiler_api
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
import pandas as pd
from fastapi.testclient import TestClient

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

def chronometrer_variantes(features: dict) -> None:
    """Temps de préparation de l'entrée + inférence, pour chaque variante, et gain par rapport à l'état initial."""
    print(f"\n=== Inférence (entrée + modèle) : {REPETITIONS} répétitions ===")
    print(f"{'':28} {'médiane':>9} {'p95':>9} {'gain':>6}")
    reference = None
    for libelle, n_jobs, format_entree, predire in VARIANTES:
        regler_threads(n_jobs)
        mediane, p95 = chronometrer(inferer, predire, features, format_entree)
        reference = reference or mediane
        print(f"{libelle:28} {mediane:6.2f} ms {p95:6.2f} ms {reference / mediane:5.1f}x")
    # Plancher : le booster LightGBM seul, sur une entrée déjà convertie en float32 par le pipeline
    booster = main.pipeline[-1].booster_
    X_cast = main.pipeline[:-1].transform(entree_modele(features, "numpy"))
    mediane, p95 = chronometrer(booster.predict, X_cast)
    print(f"{'(booster LightGBM seul)':28} {mediane:6.2f} ms {p95:6.2f} ms")

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
            chronometrer_variantes(corps["features"])
            # Réglage de l'API remis en place avant de profiler les requêtes
            regler_threads(1)
            profiler_requetes(client, corps)
            # Libère le fichier SQLite avant la suppression du dossier temporaire (Windows)
            main.moteur.dispose()
