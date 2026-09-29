"""API FastAPI, expose le modèle de scoring crédit.

Usage : uv run uvicorn api.main:app --reload   (documentation interactive sur /docs)
"""

# --- IMPORT MODULES ----------------------------------

import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictFloat

# --- CONSTANTES ----------------------------------

# Modèle champion de P6 (voir scripts/exporter_modele.py)
DOSSIER_MODELE = Path("model")
FICHIER_METADONNEES = DOSSIER_MODELE / "metadonnees.json"
FICHIER_BORNES = DOSSIER_MODELE / "bornes.json"
FICHIER_EXEMPLES = DOSSIER_MODELE / "input_example.json"

# Commit déployé, injecté par Render ; absent en local
REVISION = os.getenv("RENDER_GIT_COMMIT")

# --- INITIALISATION ----------------------------------

modele = None
colonnes: list[str] = []
obligatoires: list[str] = []
bornes: dict[str, list[float]] = {}
metadonnees: dict = {}
erreur_demarrage: str | None = None
charge_le: datetime | None = None
duree_chargement: float | None = None

# --- MODÈLES DE DONNÉES ----------------------------------

class Client(BaseModel):
    """Données d'un client : variables du modèle, les facultatives non fournies valent NaN."""

    # StrictFloat : un texte ("12") ou un booléen est rejeté au lieu d'être converti
    features: dict[str, StrictFloat | None] = Field(description="Valeur de chaque variable, indexée par son nom")

class Score(BaseModel):
    """Score de défaut et décision au seuil métier."""

    score: float = Field(description="Probabilité de défaut estimée par le modèle")
    seuil: float = Field(description="Seuil métier : au-delà, le crédit est refusé")
    decision: str = Field(description="OK ou NOK")
    nom_modele: str
    alias: str
    version_modele: int = Field(description="Version au registry MLflow de P6")

class Health(BaseModel):
    """État de l'API."""

    status: str = Field(description="ok si le modèle est chargé")
    # Modèle
    nom_modele: str
    # Démarrage et déploiement
    en_ligne_depuis: datetime = Field(description="Date de chargement du modèle (démarrage de l'API)")
    revision: str | None = Field(default=None, description="Commit déployé")

class Metadata(BaseModel):
    """Description du modèle en service et périmètre des données."""

    # Modèle
    nom_modele: str
    alias: str
    version_modele: int = Field(description="Version au registry MLflow de P6")
    mlflow_run_id: str = Field(description="Run MLflow d'entraînement, pour remonter à l'expérimentation")
    # Entrée attendue
    variables: int = Field(description="Nombre de variables du modèle")
    variables_obligatoires: int = Field(description="Les autres sont facultatives et complétées à NaN")
    # Règle de décision et performances mesurées en P6
    seuil: float = Field(description="Seuil métier : au-delà, le crédit est refusé")
    ratio_cout_fn_fp: int = Field(description="Coût d'un faux négatif rapporté à celui d'un faux positif")
    auc_test: float
    cout_metier_test: float = Field(description="Coût métier moyen par client sur le jeu de test")
    train_set: str = Field(description="Taille du jeu d'entraînement")

# --- FONCTIONS ----------------------------------

def verif_modele() -> None:
    """Erreur 503 si le modèle n'a pas pu être chargé au démarrage."""
    if modele is None:
        raise HTTPException(503, f"Modèle indisponible : {erreur_demarrage}")

def verif_client(features: dict[str, float | None]) -> None:
    """Erreur 422 si la requête ne peut pas être scorée : variables obligatoires manquantes,
    noms inconnus ou valeurs hors des bornes du jeu d'entraînement."""
    # Valeur nulle ou NaN = variable non renseignée
    renseignees = {nom: valeur for nom, valeur in features.items() if valeur is not None and not math.isnan(valeur)}
    manquantes = [nom for nom in obligatoires if nom not in renseignees]
    inconnues = sorted(set(features) - set(colonnes))
    hors_plage = [
        f"{nom} = {valeur} (entraînement : de {bornes[nom][0]} à {bornes[nom][1]})"
        for nom, valeur in renseignees.items()
        if nom in bornes and not bornes[nom][0] <= valeur <= bornes[nom][1]
    ]
    if manquantes or inconnues or hors_plage:
        # Seules les catégories en erreur sont renvoyées
        detail = {"manquantes": manquantes, "inconnues": inconnues, "hors_plage": hors_plage}
        raise HTTPException(422, {cle: valeurs for cle, valeurs in detail.items() if valeurs})

def charger_exemples() -> dict:
    """Clients d'exemple proposés dans Swagger, tirés de l'exemple d'entrée du modèle.

    Les variables non renseignées sont omises. Sans fichier d'exemples, Swagger n'en propose pas.
    """
    try:
        exemples = json.loads(FICHIER_EXEMPLES.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {
        f"client_{numero}": {
            "summary": f"Client d'exemple {numero}",
            "value": {"features": {nom: valeur for nom, valeur in zip(exemples["columns"], ligne)
                                   if valeur is not None and not math.isnan(valeur)}},
        }
        for numero, ligne in enumerate(exemples["data"], start=1)
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle, ses métadonnées et les bornes au démarrage.

    Un modèle absent ne fait pas échouer le démarrage : les routes répondent 503 avec la cause
    """
    global modele, colonnes, obligatoires, bornes, metadonnees, en_ligne_depuis, erreur_demarrage
    try:
        modele = mlflow.pyfunc.load_model(str(DOSSIER_MODELE))
        schema = modele.metadata.get_input_schema()
        colonnes = schema.input_names()
        obligatoires = schema.required_input_names()
        bornes = json.loads(FICHIER_BORNES.read_text(encoding="utf-8"))
        metadonnees = json.loads(FICHIER_METADONNEES.read_text(encoding="utf-8"))
        en_ligne_depuis = datetime.now().astimezone()
    except Exception as erreur:  # noqa: BLE001 - toute panne au démarrage doit être signalée
        modele, erreur_demarrage = None, f"{type(erreur).__name__} : {erreur}"
    yield

# --- API ----------------------------------

EXEMPLES = charger_exemples()

app = FastAPI(
    title="Prêt à dépenser - API de scoring",
    description="Renvoie la probabilité de défaut d'un client et la décision d'octroi du crédit.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", summary="État de l'API")
def health() -> Health:
    """Vérifie que le modèle est chargé et le status de l'API."""
    verif_modele()
    return Health(
        status="OK",
        nom_modele=metadonnees["nom"],
        en_ligne_depuis=en_ligne_depuis,
        revision=REVISION,
    )

@app.get("/metadata", summary="Description du modèle en service")
def metadata() -> Metadata:
    """Décrit le modèle en service."""
    verif_modele()
    tags = metadonnees["tags"]
    return Metadata(
        nom_modele=metadonnees["nom"],
        alias=metadonnees["alias"],
        version_modele=metadonnees["version"],
        mlflow_run_id=metadonnees["run_id"],
        variables=len(colonnes),
        variables_obligatoires=len(obligatoires),
        seuil=metadonnees["seuil"],
        ratio_cout_fn_fp=int(tags["ratio_cout_fn_fp"]),
        auc_test=float(tags["auc_test"]),
        cout_metier_test=float(tags["cout_metier_test"]),
        train_set=tags["entrainement"],
    )

@app.post(
    "/predict",
    summary="Scorer un client",
    responses={
        422: {"description": "Requête invalide : variables obligatoires manquantes, noms inconnus, valeurs hors des bornes d'entraînement ou types incorrects"},
        503: {"description": "Modèle indisponible"},
    },
)
def predict(client: Annotated[Client, Body(openapi_examples=EXEMPLES)]) -> Score:
    """Calcule la probabilité de défaut du client et la compare au seuil métier.

    Les 147 variables obligatoires doivent être renseignées, les 380 autres, si absentes ou nulles, sont traitées comme manquantes par le modèle.
    La liste « Examples » propose 5 echantillons réels.
    """
    verif_modele()
    verif_client(client.features)
    # Colonnes dans l'ordre du modèle, variables non fournies à NaN
    X = pd.DataFrame([client.features]).reindex(columns=colonnes).astype(float)
    score = float(modele.predict(X)[0, 1])
    seuil = metadonnees["seuil"]
    return Score(
        score=round(score,3),
        seuil=round(seuil,3),
        decision="NOK" if score >= seuil else "OK",
        nom_modele=metadonnees["nom"],
        alias=metadonnees["alias"],
        version_modele=metadonnees["version"],
    )
