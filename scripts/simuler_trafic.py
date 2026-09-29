"""Simule du trafic de production : rejoue des clients du jeu de test sur l'API.

Deux phases de même taille, pour agrémenter le tableau de monitoring :
- une phase stable : clients tirés au hasard dans le test set
- une phase de dérive : clients tirés avec un biais vers les plus jeunes et les EXT_SOURCE_2 les plus faibles

Les horaires des phases sont écrits dans `data/simulation.json`, pour les retrouver sur le dashboard.

Pour intégrer des incidents, environ 5 % des requêtes sont volontairement rendues invalides. 

A lancer depuis la racine du projet, avec l'environnement de P6 (lecture du parquet) et le fichier .env (API_TOKEN) :
    uv run --project ../P6 --env-file .env python -m scripts.simuler_trafic --url https://credit-scoring-swvm.onrender.com/ --appels 1000
"""

# --- IMPORT MODULES ----------------------------------
import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.exporter_modele import FICHIER_BORNES, SEED, decouper_dataset

# --- CONSTANTES ----------------------------------
FICHIER_SIMULATION = Path("data/simulation.json")

# Nombre de requetes par phase
NBR_REQUETE = 1000

# Taux de requetes rendues invalides
TAUX_INVALIDES = 0.05

# Variables utilisées pour forcer l'invalidité
VARIABLE_OBLIGATOIRE = "AMT_GOODS_PRICE"
VARIABLE_BORNEE = "AMT_CREDIT"

# Le premier démarrage de Render peut prendre 1 minute.
DELAI_MAX_S = 120

# Dérive : les clients surpondérés (ici 25 % des plus jeunes ou des EXT_SOURCE_2 les plus faibles, profils plus risqués)
# ont chacun 5 fois plus de chances d'être tirés
QUANTILE_DERIVE = 0.25
POIDS_DERIVE = 5

# --- FONCTIONS ----------------------------------
def tirer_clients(X_test: pd.DataFrame, appels: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clients de la phase stable (tirage uniforme), puis de la phase de dérive (tirage biaisé), sans doublon."""
    stable = X_test.sample(appels, random_state=SEED)
    reste = X_test.drop(stable.index)
    # Identification des lignes a surpondere (bool)
    surponderes = (reste["DAYS_BIRTH"] > reste["DAYS_BIRTH"].quantile(1 - QUANTILE_DERIVE)) | (
        reste["EXT_SOURCE_2"] < reste["EXT_SOURCE_2"].quantile(QUANTILE_DERIVE)
    )
    # Tirage avec poids = 5 de la population surponderes vs 1 pour le reste
    derive = reste.sample(appels, weights=1 + (POIDS_DERIVE - 1) * surponderes, random_state=SEED)
    return stable, derive

def variables_renseignees(client: pd.Series) -> dict:
    """Variables d'un client au format de l'API : les valeurs manquantes sont omises."""
    return {nom: float(valeur) for nom, valeur in client.dropna().items()}

def generer_invalidite(features: dict, bornes: dict, hasard: random.Random) -> dict:
    """Copie de la requête avec un défaut au hasard : variable manquante, inconnue, hors plage ou de mauvais type."""
    features = dict(features)
    defaut = hasard.choice(["manquante", "inconnue", "hors_plage", "texte"])
    if defaut == "manquante":
        features.pop(VARIABLE_OBLIGATOIRE, None)
    elif defaut == "inconnue":
        features["variable_inventee"] = 1.0
    elif defaut == "hors_plage":
        features[VARIABLE_BORNEE] = bornes[VARIABLE_BORNEE][1] + 1
    else:
        features[VARIABLE_BORNEE] = "12"
    return features

def appeler(url: str, jeton: str, features: dict) -> str:
    """Envoie la requête à /predict et renvoie le code HTTP ("reseau" si l'API ne répond pas)."""
    requete = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"features": features}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {jeton}"},
    )
    try:
        with urllib.request.urlopen(requete, timeout=DELAI_MAX_S) as reponse:
            return str(reponse.status)
    except urllib.error.HTTPError as erreur:
        return str(erreur.code)
    except (urllib.error.URLError, TimeoutError):
        return "reseau"

def rejouer(nom: str, clients: pd.DataFrame, url: str, jeton: str, bornes: dict, hasard: random.Random) -> dict:
    """Envoie les clients un par un et renvoie l'horaire de la phase avec le décompte des codes HTTP."""
    debut = datetime.now(UTC)
    codes = Counter()
    for numero, (_, client) in enumerate(clients.iterrows(), start=1):
        # Mise en forme des features
        features = variables_renseignees(client)
        # Generation des requetes invalides
        if hasard.random() < TAUX_INVALIDES:
            features = generer_invalidite(features, bornes, hasard)
        # Prediction
        codes[appeler(url, jeton, features)] += 1
        if numero % 100 == 0:
            print(f"  {nom} : {numero}/{len(clients)}")
    return {"nom": nom, "debut": debut.isoformat(), "fin": datetime.now(UTC).isoformat(),
            "appels": len(clients), "codes": dict(sorted(codes.items()))}

# --- MAIN ----------------------------------

if __name__ == "__main__":
    # Parseur des arguments
    analyseur = argparse.ArgumentParser(description="Simule du trafic de production sur l'API de scoring.")
    analyseur.add_argument("--url", default="http://127.0.0.1:8000", help="adresse de l'API, sans /predict")
    analyseur.add_argument("--appels", type=int, default=NBR_REQUETE, help="nombre d'appels par phase")
    arguments = analyseur.parse_args()

    # Recuperation jeton
    jeton = os.getenv("API_TOKEN")
    if not jeton:
        sys.exit("API_TOKEN absent : lancer avec --env-file .env")

    # Generation des jeux stable et derive
    bornes = json.loads(FICHIER_BORNES.read_text(encoding="utf-8"))
    _, X_test = decouper_dataset()
    stable, derive = tirer_clients(X_test, arguments.appels)

    # Prediction
    hasard = random.Random(SEED)
    phases = []
    for nom, clients in (("stable", stable), ("derive", derive)):
        print(f"Phase {nom} : {len(clients)} appels vers {arguments.url}")
        phases.append(rejouer(nom, clients, arguments.url.rstrip("/"), jeton, bornes, hasard))

    # Enregistrement metadonees simulation
    FICHIER_SIMULATION.parent.mkdir(parents=True, exist_ok=True)
    FICHIER_SIMULATION.write_text(json.dumps({"url": arguments.url, "phases": phases}, indent=2), encoding="utf-8")
    for phase in phases:
        print(phase["nom"], phase["codes"])
