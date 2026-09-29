"""Tests de l'API de scoring : modèle réel embarqué dans model/.

Usage : uv run pytest   (couverture de api/ affichée)
"""

# --- IMPORT MODULES ----------------------------------

import pytest
from fastapi.testclient import TestClient

from api import main

# --- CONSTANTES ----------------------------------

# Requêtes invalides (422).
# Les lambdas lisent main.* à l'appel : le modèle n'est chargé qu'au démarrage de l'API.
CAS_422 = {
    "aucune variable": (lambda f: {}, "manquantes"),
    "obligatoire nulle": (lambda f: f | {main.obligatoires[0]: None}, "manquantes"),
    "nom inconnu": (lambda f: f | {"variable_inventee": 1.0}, "inconnues"),
    "hors plage": (lambda f: f | {main.obligatoires[0]: main.bornes[main.obligatoires[0]][1] + 1}, "hors_plage"),
    "texte": (lambda f: f | {main.obligatoires[0]: "12"}, None),
}

# --- FIXTURES ----------------------------------

@pytest.fixture(scope="module")
def api():
    """Client de test, le `with` déclenche le chargement du modèle (lifespan)."""
    with TestClient(main.app) as client:
        yield client

def features_valides(numero: int = 1) -> dict:
    """Variables d'un client d'exemple de Swagger."""
    return dict(main.EXEMPLES[f"client_{numero}"]["value"]["features"])

# --- TESTS ----------------------------------

def test_health_et_metadata(api):
    sante = api.get("/health")
    assert sante.status_code == 200
    assert sante.json()["status"] == "OK"
    assert sante.json()["nom_modele"] == main.metadonnees["nom"]
    assert {"en_ligne_depuis", "revision"} <= set(sante.json())

    meta = api.get("/metadata")
    assert meta.status_code == 200
    # Valeurs du modèle embarqué
    assert meta.json()["variables"] == len(main.colonnes)
    assert meta.json()["variables_obligatoires"] == len(main.obligatoires)
    assert meta.json()["seuil"] == main.metadonnees["seuil"]

@pytest.mark.parametrize("numero", range(1, len(main.EXEMPLES) + 1))
def test_predict_exemples(api, numero):
    reponse = api.post("/predict", json={"features": features_valides(numero)})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert 0 <= corps["score"] <= 1
    # Décision cohérente avec le seuil, quel que soit le modèle
    assert corps["decision"] == ("NOK" if corps["score"] >= corps["seuil"] else "OK")
    assert corps["seuil"] == pytest.approx(main.metadonnees["seuil"], abs=1e-3)
    assert corps["version_modele"] == main.metadonnees["version"]

def test_predict_variables_facultatives(api):
    # Absentes ou nulles, les variables facultatives donnent le même score
    features = features_valides()
    facultatives = set(main.colonnes) - set(main.obligatoires)
    nulles = features | {nom: None for nom in facultatives}
    absentes = {nom: valeur for nom, valeur in features.items() if nom not in facultatives}
    scores = [api.post("/predict", json={"features": f}).json()["score"] for f in (nulles, absentes)]
    assert scores[0] == scores[1]

@pytest.mark.parametrize(("modifier", "categorie"), CAS_422.values(), ids=CAS_422.keys())
def test_predict_requete_invalide(api, modifier, categorie):
    reponse = api.post("/predict", json={"features": modifier(features_valides())})
    assert reponse.status_code == 422
    detail = reponse.json()["detail"]
    if categorie is None:
        assert isinstance(detail, list)
    else:
        # Seule la catégorie en erreur est renvoyée
        assert set(detail) == {categorie}

def test_predict_sans_corps(api):
    assert api.post("/predict", json={}).status_code == 422

def test_modele_indisponible(api, monkeypatch):
    monkeypatch.setattr(main, "modele", None)
    monkeypatch.setattr(main, "erreur_demarrage", "modèle absent")
    for reponse in (api.get("/health"), api.get("/metadata"), api.post("/predict", json={"features": features_valides()})):
        assert reponse.status_code == 503
        assert "modèle absent" in reponse.json()["detail"]

def test_demarrage_sans_fichiers_du_modele(tmp_path, monkeypatch):
    # Le démarrage n'échoue pas : l'erreur est gardée et les routes répondent 503
    monkeypatch.setattr(main, "modele", main.modele)
    monkeypatch.setattr(main, "erreur_demarrage", None)
    monkeypatch.setattr(main, "DOSSIER_MODELE", tmp_path / "absent")
    monkeypatch.setattr(main, "FICHIER_EXEMPLES", tmp_path / "absent" / "input_example.json")
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 503
    assert main.erreur_demarrage
    # Sans fichier d'exemples, Swagger n'en propose pas
    assert main.charger_exemples() == {}
