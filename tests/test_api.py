"""Tests de l'API de scoring : modèle réel embarqué dans model/.

Usage : uv run pytest   (couverture de api/ affichée)
"""

# --- IMPORT MODULES ----------------------------------

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select

from api import journal, main

# --- CONSTANTES ----------------------------------

# Jeton d'accès à /predict, propre aux tests
JETON_TEST = "jeton-de-test"

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

@pytest.fixture(scope="module", autouse=True)
def base_temporaire(tmp_path_factory):
    """Journal des appels dans un SQLite temporaire pour ne rien polluer, et jeton de test."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DATABASE_URL", f"sqlite:///{tmp_path_factory.mktemp('journal') / 'logs.db'}")
        mp.setattr(main, "JETON", JETON_TEST)
        yield

@pytest.fixture(scope="module")
def api(base_temporaire):
    """Client de test authentifié, le `with` déclenche le chargement du modèle et du journal (lifespan).
    Une exception imprévue de l'API donne une réponse 500 au lieu de faire échouer le test."""
    with TestClient(main.app, raise_server_exceptions=False, headers={"Authorization": f"Bearer {JETON_TEST}"}) as client:
        yield client

def features_valides(numero: int = 1) -> dict:
    """Variables d'un client d'exemple de Swagger."""
    return dict(main.EXEMPLES[f"client_{numero}"]["value"]["features"])

def lignes_journal() -> list[dict]:
    """Lignes du journal, de la plus ancienne à la plus récente."""
    with main.moteur.connect() as connexion:
        return [ligne._asdict() for ligne in connexion.execute(select(journal.TABLE).order_by(journal.TABLE.c.id))]

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
    features = features_valides(numero)
    reponse = api.post("/predict", json={"features": features})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert 0 <= corps["score"] <= 1
    # Décision cohérente avec le seuil, quel que soit le modèle
    assert corps["decision"] == ("NOK" if corps["score"] >= corps["seuil"] else "OK")
    assert corps["seuil"] == pytest.approx(main.metadonnees["seuil"], abs=1e-3)
    assert corps["version_modele"] == main.metadonnees["version"]
    # Non-régression de l'optimisation : score non arrondi (journal) identique à celui du modèle MLflow complet
    X = pd.DataFrame([features]).reindex(columns=main.colonnes).astype(float)
    assert lignes_journal()[-1]["score"] == main.modele.predict(X)[0, 1]

def test_jeton_requis(api, monkeypatch):
    avant = len(lignes_journal())
    corps = {"features": features_valides()}
    monkeypatch.delitem(api.headers, "Authorization")
    # Sans jeton, mauvais jeton ou mauvais schéma : 401. Les sondes restent ouvertes.
    for entetes in ({}, {"Authorization": "Bearer mauvais"}, {"Authorization": f"Basic {JETON_TEST}"}):
        reponse = api.post("/predict", json=corps, headers=entetes)
        assert reponse.status_code == 401
        assert reponse.headers["www-authenticate"] == "Bearer"
    # Même un corps illisible, refusé avant le contrôle du jeton (422)
    assert api.post("/predict", content="pas du json", headers={"content-type": "application/json"}).status_code == 422
    assert api.get("/health").status_code == 200
    # Les appels sans jeton valide ne sont pas journalisés
    assert len(lignes_journal()) == avant
    # API sans jeton configuré : route fermée, même avec un jeton
    monkeypatch.setattr(main, "JETON", None)
    reponse = api.post("/predict", json=corps, headers={"Authorization": f"Bearer {JETON_TEST}"})
    assert reponse.status_code == 503
    assert "non configuré" in reponse.json()["detail"]

def test_journal_appel_reussi(api):
    avant = len(lignes_journal())
    # Seul /predict est journalisé, pas les sondes de santé
    api.get("/health")
    reponse = api.post("/predict", json={"features": features_valides()}).json()
    lignes = lignes_journal()
    assert len(lignes) == avant + 1
    ligne = lignes[-1]
    assert ligne["statut_http"] == 200
    assert ligne["entrees"] == features_valides()
    assert ligne["score"] == pytest.approx(reponse["score"], abs=1e-3)
    assert ligne["decision"] == reponse["decision"]
    assert ligne["version_modele"] == main.metadonnees["version"]
    # Le modèle est une partie de la latence totale
    assert 0 < ligne["temps_inference_ms"] <= ligne["latence_ms"]
    assert ligne["erreur"] is None

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
    # L'erreur est journalisée avec sa cause, sans score
    ligne = lignes_journal()[-1]
    assert ligne["statut_http"] == 422
    assert ligne["erreur"] == detail
    assert ligne["score"] is None

def test_predict_corps_inexploitable(api):
    # Ni JSON, ni objet de variables : 422, journalisé sans entrées
    requetes = [
        {"content": "pas du json", "headers": {"content-type": "application/json"}},
        {"json": {}},
        {"json": {"features": 3}},
    ]
    for requete in requetes:
        assert api.post("/predict", **requete).status_code == 422
        assert lignes_journal()[-1]["entrees"] is None

def test_panne_imprevue_journalisee(api, monkeypatch):
    def modele_casse(_):
        raise RuntimeError("modèle cassé")

    monkeypatch.setattr(main.pipeline, "predict_proba", modele_casse)
    assert api.post("/predict", json={"features": features_valides()}).status_code == 500
    ligne = lignes_journal()[-1]
    assert ligne["statut_http"] == 500
    assert "modèle cassé" in ligne["erreur"]
    assert ligne["score"] is None

def test_base_en_panne(api, tmp_path, monkeypatch):
    # Le dossier n'existe pas : l'écriture échoue, le client a quand même son score
    monkeypatch.setattr(main, "moteur", create_engine(f"sqlite:///{tmp_path / 'absent' / 'logs.db'}"))
    assert api.post("/predict", json={"features": features_valides()}).status_code == 200

def test_choix_de_la_base(tmp_path, monkeypatch):
    # URL Supabase : pilote psycopg, sans connexion tant qu'on n'écrit pas
    moteur = journal.creer_moteur("postgresql://utilisateur:mot_de_passe@localhost/base")
    assert (moteur.dialect.name, moteur.dialect.driver) == ("postgresql", "psycopg")
    # Sans DATABASE_URL : fichier SQLite local, dossier créé au besoin
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setattr(journal, "FICHIER_SQLITE", tmp_path / "data" / "logs.db")
    assert journal.creer_moteur().dialect.name == "sqlite"
    assert (tmp_path / "data").is_dir()

def test_journal_impossible_a_creer(tmp_path, monkeypatch):
    # Un fichier occupe la place du dossier SQLite local : pas de journal, mais pas d'exception
    (tmp_path / "data").write_text("")
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setattr(journal, "FICHIER_SQLITE", tmp_path / "data" / "logs.db")
    assert journal.preparer() is None

def test_modele_indisponible(api, monkeypatch):
    monkeypatch.setattr(main, "modele", None)
    monkeypatch.setattr(main, "erreur_demarrage", "modèle absent")
    for reponse in (api.get("/health"), api.get("/metadata"), api.post("/predict", json={"features": features_valides()})):
        assert reponse.status_code == 503
        assert "modèle absent" in reponse.json()["detail"]

def test_demarrage_sans_fichiers_du_modele_ni_base(tmp_path, monkeypatch):
    # Le démarrage n'échoue pas : l'erreur est gardée et les routes répondent 503
    monkeypatch.setattr(main, "modele", main.modele)
    monkeypatch.setattr(main, "pipeline", main.pipeline)
    monkeypatch.setattr(main, "moteur", main.moteur)
    monkeypatch.setattr(main, "erreur_demarrage", None)
    # Base injoignable (dossier absent) : la création de la table échoue sans bloquer le démarrage
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'absent' / 'logs.db'}")
    monkeypatch.setattr(main, "DOSSIER_MODELE", tmp_path / "absent")
    monkeypatch.setattr(main, "FICHIER_EXEMPLES", tmp_path / "absent" / "input_example.json")
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 503
    assert main.erreur_demarrage
    # Sans fichier d'exemples, Swagger n'en propose pas
    assert main.charger_exemples() == {}
