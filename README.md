# Prêt à dépenser : API de scoring crédit en production

Mise en production du modèle de scoring pour le département « Crédit Express » : une API qui renvoie la probabilité de défaut d'un client et la décision d'octroi du crédit, conteneurisée avec Docker et déployée automatiquement par CI/CD.

Stack : **FastAPI**, modèle **LightGBM** chargé via **MLflow**, **Docker**, **GitHub Actions**, hébergement **Render**.

## Le modèle

Le modèle cible du registry MLflow (`scoring-credit`, alias `champion`) est exporté dans `model/` et versionné avec le code : le conteneur n'a pas accès au registry.

| | |
|---|---|
| Entrée | 527 variables, dont 147 obligatoires (les 380 autres, si absentes, sont traitées comme manquantes) |
| Sortie | probabilité de défaut (`score`) |
| Règle de décision | `score >= seuil` : crédit refusé (`NOK`), sinon accordé (`OK`) |
| Seuil | optimisé pour un coût d'un faux négatif égal à 10 fois celui d'un faux positif |

Le dossier `model/` contient le modèle MLflow (`MLmodel`, `model.skops`), `metadonnees.json` (version, run MLflow, seuil, performances) et `bornes.json` (min et max des variables sur le jeu d'entraînement, utilisés pour rejeter les valeurs aberrantes).

## Arborescence

```
P8/
├── api/
│   └── main.py                   # API FastAPI
├── model/                        # Modèle exporté (modèle, métadonnées, bornes)
├── scripts/
│   └── exporter_modele.py        # Export du modèle depuis le registry -> model/
├── tests/
│   └── test_api.py               # Tests de l'API
├── .github/workflows/ci-cd.yml   # Lint, tests, build de l'image, déploiement
├── Dockerfile                    # Image Docker
├── .dockerignore
└── pyproject.toml / uv.lock      # Dépendances
```

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et Docker.

```bash
# Récupération du projet
git clone https://github.com/lenny-ai-data/oc_project_8.git
cd oc_project_8
# Création de l'environnement virtuel
uv sync
```

## Lancer l'API

```bash
# En local
uv run uvicorn api.main:app

# Via Docker
docker build -t scoring-api .
docker run --rm -p 8000:8000 scoring-api
```

La documentation interactive (Swagger) est sur <http://127.0.0.1:8000/docs>.

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | État de l'API : modèle chargé, date de démarrage, commit déployé |
| GET | `/metadata` | Modèle en service : version, run MLflow, nombre de variables, seuil, performances mesurées |
| POST | `/predict` | Score et décision d'un client |

## Requête de prédiction

Le corps est un objet `features` qui associe le nom de chaque variable à sa valeur. Pour la démonstration, 5 échantillons (tirés de `model/input_example.json`)) sont préconfigurés dans la liste **Examples**.

En Python, avec le premier client d'exemple :

```python
import json
import urllib.request

# Récupération d'un jeu d'exemple, à remplacer par les valeurs à prédire
exemple = json.load(open("model/input_example.json", encoding="utf-8"))
features = {nom: valeur for nom, valeur in zip(exemple["columns"], exemple["data"][0]) if valeur is not None and valeur == valeur}

# Requete
requete = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps({"features": features}).encode(),
    headers={"Content-Type": "application/json"},
)

# Sortie
print(urllib.request.urlopen(requete).read().decode())
```

Réponse :

```json
{"score": 0.103, "seuil": 0.09, "decision": "NOK", "nom_modele": "scoring-credit", "alias": "champion", "version_modele": 2}
```

### Erreurs

L'API refuse une requête que le modèle ne peut pas scorer correctement.

| Code | Cas | Réponse |
|---|---|---|
| 422 | Variables obligatoires absentes ou nulles | `{"detail": {"manquantes": [...]}}` |
| 422 | Noms de variables inconnus | `{"detail": {"inconnues": [...]}}` |
| 422 | Valeur hors de l'intervalle [min, max] défini | `{"detail": {"hors_plage": [...]}}` |
| 422 | Type incorrect (texte, booléen) | détail Pydantic |
| 503 | Modèle non chargé au démarrage | la cause de l'échec |

Seules les catégories en erreur sont renvoyées. Une valeur `null` ou `NaN` compte comme non renseignée : refusée pour une variable obligatoire, acceptée pour une facultative.

## Export du modèle depuis le registry

Après avoir promu un nouveau modèle (avec son registry et son jeu de données) :

```bash
# Réexporte le modèle, les métadonnées et les bornes dans model/
uv run python scripts/exporter_modele.py
```
Les versions de `scikit-learn`, `lightgbm`, `skops` et `pandas` sont figées dans `pyproject.toml` : le format `skops` est lié à la version de scikit-learn.

## Tests et CI/CD

```bash
# Tests et couverture de api/ 
uv run pytest
```

Les tests utilisent le vrai modèle versionné, sans besoin de réseau ni secret. Ils couvrent les trois routes, les 5 exemples, chaque cas d'erreur 422, le 503 et le démarrage sans les fichiers du modèle.

À chaque push, `.github/workflows/ci-cd.yml` enchaîne trois jobs de CI, puis un 4ème de CD sur `main`, chacun bloquant le suivant :

1. **qualité** : lint `ruff`
2. **tests** : `pytest`, avec 90 % de couverture requis, rapport HTML publié en artefact
3. **image** : build de l'image Docker
4. **déploiement** : reconstruction de l'image par Render directement depuis GitHub. La CD appelle le *deploy hook* avec le SHA du commit vérifié, puis attend que `/health` renvoie cette révision.
