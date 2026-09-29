# Image de l'API de scoring "Pret a depenser".
# Build : docker build -t scoring-api .
# Run   : docker run --rm -p 8000:8000 scoring-api

# Image python avec uv
FROM ghcr.io/astral-sh/uv:0.12.6-python3.13-trixie-slim

# Bytecode précompilé et copie des paquets
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# LightGBM a besoin d'OpenMP, absent de l'image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Couche des dependances, seulement quand le lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups

# Utilisateur non privilégié
RUN useradd --create-home --uid 1000 app

# Seulement l'API et le modèle exporté (lecture seule à l'exécution)
COPY api/ api/
COPY model/ model/

USER app

# Port du conteneur
EXPOSE 8000

# exec pour que uvicorn devienne PID 1 et reçoive les signaux d'arrêt.
# PORT fourni par l'hébergeur ou 8000 en local.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
