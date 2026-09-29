# Prêt à dépenser : API de scoring crédit en production

Mise en production du modèle de scoring pour le département « Crédit Express » : une API qui renvoie la probabilité de défaut d'un client et la décision d'octroi du crédit, conteneurisée avec Docker et déployée automatiquement par CI/CD.

Stack : **FastAPI**, modèle **LightGBM** chargé via **MLflow**, **Docker**, **GitHub Actions**, hébergement **Render**, journal des appels dans **Postgres (Supabase)**.

> 🔌 API Render : https://credit-scoring-swvm.onrender.com/docs

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
│   ├── main.py                   # API FastAPI
│   └── journal.py                # Journal des appels (table, écriture en base)
├── model/                        # Modèle exporté (modèle, métadonnées, bornes)
├── scripts/
│   └── exporter_modele.py        # Export du modèle depuis le registry -> model/
├── tests/
│   └── test_api.py               # Tests de l'API
├── .github/workflows/ci-cd.yml   # Lint, tests, build de l'image, déploiement
├── .env.example                  # Modèle du fichier .env (variables d'environnement)
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
# Renseigner les variables d'environnement
cp .env.example .env
```

| Variable .env | Rôle | Si absente |
|---|---|---|
| `API_TOKEN` | jeton d'accès à `/predict` | `/predict` répond 503 |
| `DATABASE_URL` | URL Postgres du journal des appels | redirection vers base locale SQLite `data/logs.db` |

## Lancer l'API

```bash
# En local
uv run --env-file .env uvicorn api.main:app

# Via Docker
docker build -t scoring-api .
docker run --rm -p 8000:8000 --env-file .env scoring-api
```

La documentation interactive (Swagger) est sur <http://127.0.0.1:8000/docs>.

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | État de l'API : modèle chargé, date de démarrage, commit déployé |
| GET | `/metadata` | Modèle en service : version, run MLflow, nombre de variables, seuil, performances mesurées |
| POST | `/predict` | Score et décision d'un client, protégée par jeton |

## Requête de prédiction

`/predict` exige un jeton d'accès, fourni dans l'en-tête `Authorization: Bearer <jeton>`. Dans Swagger, le bouton **Authorize** le mémorise pour les essais. `/health` et `/metadata` restent ouvertes.

Le corps est un objet `features` qui associe le nom de chaque variable à sa valeur. Pour la démonstration, 5 échantillons (tirés de `model/input_example.json`)) sont préconfigurés dans la liste **Examples**.

En Python, avec le premier client d'exemple :

```python
import json
import os
import urllib.request

# Récupération d'un jeu d'exemple, à remplacer par les valeurs à prédire
exemple = json.load(open("model/input_example.json", encoding="utf-8"))
features = {nom: valeur for nom, valeur in zip(exemple["columns"], exemple["data"][0]) if valeur is not None and valeur == valeur}

# Requete
requete = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps({"features": features}).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['API_TOKEN']}"},
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
| 401 | Jeton absent ou invalide | `{"detail": "Jeton d'accès absent ou invalide"}` |
| 422 | Variables obligatoires absentes ou nulles | `{"detail": {"manquantes": [...]}}` |
| 422 | Noms de variables inconnus | `{"detail": {"inconnues": [...]}}` |
| 422 | Valeur hors de l'intervalle [min, max] défini | `{"detail": {"hors_plage": [...]}}` |
| 422 | Type incorrect (texte, booléen) | détail Pydantic |
| 503 | Modèle non chargé au démarrage | la cause de l'échec |
| 503 | `API_TOKEN` non défini sur l'API : la route reste fermée | `{"detail": "Jeton d'accès non configuré sur l'API"}` |

Seules les catégories en erreur sont renvoyées. Une valeur `null` ou `NaN` compte comme non renseignée : refusée pour une variable obligatoire, acceptée pour une facultative.

## Journal des appels

Chaque appel authentifié à `/predict` (succès comme erreurs) est enregistré dans la table `predictions`. Le monitoring se concentrant autour du modèle, les deux autres routes ne sont pas incluses dans le cadre de ce livrable. Ce journal est la matière première du monitoring : dérive des données, distribution des scores, latence et taux d'erreur.

| Colonne | Contenu | Sert à |
|---|---|---|
| `horodatage` | date et heure de l'appel (UTC) | séries temporelles, fenêtres de comparaison |
| `statut_http`, `erreur` | code de réponse et cause (JSON)| taux d'erreur et leurs causes |
| `entrees` | variables **renseignées** de la requête (JSON) | data drift face au jeu d'entraînement |
| `score`, `decision` | sortie du modèle | distribution des scores, taux de refus |
| `temps_inference_ms` | durée du modèle seul | identifier un goulot d'étranglement |
| `latence_ms` | durée de traitement par l'API | temps de réponse |
| `version_modele`, `revision` | version du modèle et commit déployé | tracer et vérifier un déploiement |

La mesure est faite via un **middleware** (`journaliser` dans `api/main.py`) et l'écriture est réalisée en tâche de fond, sans impact sur la latence mesurée ou le temps d'attente du client. Une base injoignable est signalée dans les logs mais ne fait pas échouer la requête.

### Stratégie de stockage

- **Base de données** : Postgres hébergé par **Supabase**, région UE. Postgres permet d'interroger le journal en SQL et le type JSONB permet de stocker l'ensemble des variables d'entrée sans impacter le nombre de colonnes.
  - **Base de test** : Pour du développement local, ou la réalisation des tests, on peut rediriger les écritures vers une base SQLite locale `data/logs.db`.
  - **Choix de la base** : Le choix de la base se fait via la variable d'environnement `DATABASE_URL`. Si elle n'est pas renseignée avec l'URL Supabase, alors l'API redirige automatiquement vers la base SQLite.
- **Accès** : seule la connexion Postgres, protégée par le mot de passe de la base, lit et écrit. L'API REST publique de Supabase est désactivée et la table a la sécurité au niveau des lignes (RLS) activée (refus par défaut).
- **RGPD** : l'API ne demande aucun identifiant de client et ne conserve ni adresse IP ni en-tête HTTP, seules les variables renseignées de la requête sont stockées. Points de vigilance : le jeton unique protège l'écriture du journal (les appels sans jeton valide n'y sont pas enregistrés) mais ne distingue pas les utilisateurs. Une gestion de jetons par utilisateur, avec rotation, serait la suite en production réelle.
- **Conservation** : 90 jours, par purge manuelle : `DELETE FROM predictions WHERE horodatage < now() - interval '90 days';`, à automatiser pour la production.

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

Les tests utilisent le vrai modèle versionné et un SQLite temporaire pour le journal, sans besoin de réseau ni secret. Ils couvrent les trois routes, les 5 exemples, chaque cas d'erreur 422, le 503, le démarrage sans les fichiers du modèle, et le journal : une ligne par appel, erreurs et panne imprévue (500) tracées, base indisponible sans effet sur la réponse.

À chaque push, `.github/workflows/ci-cd.yml` enchaîne trois jobs de CI, puis un 4ème de CD sur `main`, chacun bloquant le suivant :

1. **qualité** : lint `ruff`
2. **tests** : `pytest`, avec 90 % de couverture requis, rapport HTML publié en artefact
3. **image** : build de l'image Docker
4. **déploiement** : reconstruction de l'image par Render directement depuis GitHub. La CD appelle le *deploy hook* avec le SHA du commit vérifié, puis attend que `/health` renvoie cette révision.
