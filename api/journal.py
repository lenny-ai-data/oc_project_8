"""Journal des appels à /predict : une ligne par appel, dans Postgres (Supabase) ou SQLite en local.

La base est choisie par la variable d'environnement DATABASE_URL (voir .env)
"""

# --- IMPORT MODULES ----------------------------------

import logging
import os
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
)
from sqlalchemy.dialects.postgresql import JSONB

# --- CONSTANTES ----------------------------------

# Base locale, utilisée quand DATABASE_URL est absente
FICHIER_SQLITE = Path("data/logs.db")

# JSONB sous Postgres (interrogeable), JSON générique sous SQLite
JSON_SOUPLE = JSON().with_variant(JSONB(), "postgresql")

# Une ligne par appel à /predict
TABLE = Table(
    "predictions",
    MetaData(),
    Column("id", Integer, primary_key=True),
    Column("horodatage", DateTime(timezone=True), nullable=False),
    Column("statut_http", Integer, nullable=False),
    Column("erreur", JSON_SOUPLE),
    Column("entrees", JSON_SOUPLE),
    Column("score", Float),
    Column("decision", String(3)),
    Column("temps_inference_ms", Float),
    Column("latence_ms", Float, nullable=False),
    Column("version_modele", Integer),
    Column("revision", String(40)),
)

journal = logging.getLogger("uvicorn.error")

# --- FONCTIONS ----------------------------------

def creer_moteur(url: str | None = None) -> Engine:
    """Moteur SQLAlchemy vers DATABASE_URL (Postgres), ou vers le fichier SQLite local à défaut."""
    url = url or os.getenv("DATABASE_URL")
    if not url:
        FICHIER_SQLITE.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(f"sqlite:///{FICHIER_SQLITE}")
    # Supabase avec le pilote psycopg (v3) installé
    return create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1), pool_pre_ping=True)

def creer_table(moteur: Engine) -> None:
    """Crée la table `predictions` si elle n'existe pas encore."""
    TABLE.metadata.create_all(moteur)

def preparer() -> Engine | None:
    """Moteur prêt à écrire, appelé au démarrage de l'API. Un journal indisponible est signalé dans
    les logs sans empêcher le démarrage. Base injoignable : chaque écriture retentera sa chance.
    Pas de moteur (ex. dossier SQLite impossible à créer) : les appels ne sont pas journalisés."""
    moteur = None
    try:
        moteur = creer_moteur()
        creer_table(moteur)
    except Exception:  # noqa: BLE001 - le journal ne doit jamais faire échouer l'API
        journal.exception("Journal des appels indisponible")
    return moteur

def enregistrer(moteur: Engine, ligne: dict) -> None:
    """Ajoute un appel au journal. Une panne de la base est tracée dans les logs
    mais n'empeche pas le client d'avoir sa réponse."""
    try:
        with moteur.begin() as connexion:
            connexion.execute(insert(TABLE).values(**ligne))
    except Exception:  # noqa: BLE001 - le journal ne doit jamais faire échouer l'API
        journal.exception("Écriture dans le journal impossible")
