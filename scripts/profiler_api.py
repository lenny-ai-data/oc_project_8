"""Profile la route /predict de l'API : où passe le temps d'une requête ?

Deux mesures, sur le vrai modèle et un client d'exemple de Swagger :
- cProfile de requêtes complètes (TestClient) : fonctions les plus coûteuses en temps cumulé
- chronométrage de chaque étape de l'inférence : médiane et p95

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

# --- FONCTIONS ----------------------------------
def chronometrer(fonction, repetitions: int = REPETITIONS) -> str:
    """Médiane et p95 du temps d'exécution d'une fonction, en ms, après une chauffe."""
    for _ in range(CHAUFFE):
        fonction()
    durees = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        fonction()
        durees.append((time.perf_counter() - debut) * 1000)
    return f"médiane {np.median(durees):7.2f} ms   p95 {np.percentile(durees, 95):7.2f} ms"

def profiler_requetes(client: TestClient, corps: dict) -> None:
    """cProfile de requêtes complètes à /predict, trié par temps cumulé."""
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

def chronometrer_etapes(client: TestClient, corps: dict) -> None:
    """Temps de la requête complète, puis de chaque étape de l'inférence telle que l'API la fait."""
    features = corps["features"]
    X = pd.DataFrame([features]).reindex(columns=main.colonnes).astype(float)
    pipeline = main.modele.get_raw_model()
    X_cast = pipeline[:-1].transform(X)
    booster = pipeline[-1].booster_

    print(f"\n=== Chronométrage : {REPETITIONS} répétitions ===")
    print("requête complète      ", chronometrer(lambda: client.post("/predict", json=corps)))
    print("validation du client  ", chronometrer(lambda: main.verif_client(features)))
    print("DataFrame d'entrée    ", chronometrer(lambda: pd.DataFrame([features]).reindex(columns=main.colonnes).astype(float)))
    print("modèle MLflow (pyfunc)", chronometrer(lambda: main.modele.predict(X)))
    print("  pipeline sklearn    ", chronometrer(lambda: pipeline.predict_proba(X)))
    print("  booster LightGBM    ", chronometrer(lambda: booster.predict(X_cast)))

# --- MAIN ----------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as dossier:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(dossier) / 'logs.db'}"
        main.JETON = JETON_PROFIL
        with TestClient(main.app, headers={"Authorization": f"Bearer {JETON_PROFIL}"}) as client:
            corps = {"features": main.EXEMPLES["client_1"]["value"]["features"]}
            print(f"Client d'exemple 1 : {len(corps['features'])} variables renseignées sur {len(main.colonnes)}")
            chronometrer_etapes(client, corps)
            profiler_requetes(client, corps)
            # Libère le fichier SQLite avant la suppression du dossier temporaire (Windows)
            main.moteur.dispose()
