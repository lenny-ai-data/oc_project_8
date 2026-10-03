# Prêt à dépenser : API de scoring crédit en production

Mise en production du modèle de scoring pour le département « Crédit Express » : une API qui renvoie la probabilité de défaut d'un client et la décision d'octroi du crédit, conteneurisée avec Docker et déployée automatiquement par CI/CD.

Stack : **FastAPI**, modèle **LightGBM** chargé via **MLflow**, **Docker**, **GitHub Actions**, hébergement **Render**, journal des appels dans **Postgres (Supabase)**, tableau de bord de monitoring **Streamlit**.

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
├── monitoring/
│   ├── tableau_de_bord.py        # Tableau de bord Streamlit
│   ├── analyse.py                # Chargement du journal, PSI, fenêtres (partagé avec le notebook)
│   └── reference.parquet         # Référence d'entraînement (10 000 clients et leur score)
├── scripts/
│   ├── exporter_modele.py        # Export du modèle depuis le registry -> model/ et référence du monitoring
│   └── simuler_trafic.py         # Simulation de trafic de production sur l'API
├── tests/
│   └── test_api.py               # Tests de l'API
├── .github/workflows/ci-cd.yml   # Lint, tests, build de l'image, déploiement
├── Jacquinot_Lenny_6_notebook_drift_102026.ipynb  # Étude du suivi opérationnel et du drift
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

## Tableau de bord de monitoring

Le tableau de bord Streamlit (`monitoring/tableau_de_bord.py`) relit le journal des appels et fait une synthèse de la santé de l'API et des dérive des données.

```bash
# Dépendances du monitoring (Streamlit, Evidently...)
uv sync --group monitoring

# Lancement : base Supabase si renseignée, sinon base SQLite locale
uv run --group monitoring python -m streamlit run monitoring/tableau_de_bord.py
```

### Données comparées

| | |
|---|---|
| Production | journal des appels, relu toutes les 10 minutes ou via le bouton **Rafraîchir les données** |
| Référence | `monitoring/reference.parquet` : 10 000 clients tirés du jeu d'entraînement, avec le score du modèle, écrit par `scripts/exporter_modele.py` |
| Variables surveillées | les 20 variables les plus importantes du modèle (gain LightGBM) et le score |

Les métriques portent sur la **fenêtre courante** : les 1 000 derniers appels réussis (réglable dans la barre latérale), et toutes les erreurs reçues sur la même période. Le journal est découpé en fenêtres disjointes en partant du plus récent : chaque fenêtre est un échantillon indépendant, ce qui permet de confirmer une alerte par la fenêtre précédente et de suivre le PSI dans le temps.

1 000 appels est le minimum fiable déterminé dans le notebook d'analyse.

### Lecture du tableau de bord

Les métriques principales affichées en tête résument la fenêtre courante : nombre d'appels, taux d'erreur, latence p95 des appels réussis, taux de refus et nombre de colonnes en dérive. Deux onglets complètent l'analyse :

**Suivi opérationnel :**

- **Erreurs** : appels en erreur et répartition par cause (données manquantes, inconnues ou hors plages, type incorrect, erreur serveur...).
- **Temps de réponse** : Latence globale et temps d'inférence avec courbe glissante sur 50 appels.
- **Suivi des scores** : taux de refus, score moyen et distribution.

**Dérive des données**

- **Analyse des PSI** : PSI (Population Stability Index) par colonne, évalué selon les seuils d'alerte prédéfinis (stable < 0.10 < vigilance < 0.25 < critique)
- **Historique temporel** : suivi des PSI pour les fenêtres successives.
- **Distribution d'une variable** : visualisation de la distribution pour une variable cible.

Une alerte est **confirmée** si la fenêtre précédente dépassait aussi 0,1, **à confirmer** sinon (pic isolé possible), et **non évaluable** tant qu'il n'y a qu'une fenêtre. La plus ancienne fenêtre, incomplète, est écartée : sur quelques appels, son PSI n'a pas de sens.

La dérive signale que la population des clients change, pas que le modèle se trompe : la cible (défaut réel) n'est connue que des mois plus tard.

### Simulation de trafic

En l'absence de vrais clients, `scripts/simuler_trafic.py` rejoue sur l'API des clients du jeu de test, en deux phases de même taille :
1. **stable** : clients tirés au hasard
2. **dérive** : tirage biaisé vers les clients les plus jeunes et les `EXT_SOURCE_2` les plus faibles (profils plus risqués)

Environ 5 % des requêtes sont volontairement rendues invalides pour alimenter le suivi des erreurs. Les horaires des phases sont écrits dans `data/simulation.json`, pour l'analyse du notebook.

```bash
# Depuis la racine, avec l'environnement de P6 (lecture du jeu de données) et le jeton de .env
uv run --project ../P6 --env-file .env python -m scripts.simuler_trafic --url https://credit-scoring-swvm.onrender.com/ --appels 5000
```

## Interpréter le monitoring

L'étude complète (choix des variables, du PSI et de la taille de fenêtre) est dans le notebook `Jacquinot_Lenny_6_notebook_drift_102026.ipynb`. Cette section résume comment lire les signaux et ce que la simulation a montré.

### Que faire face à un signal

| Signal | Lecture | Action |
|---|---|---|
| Taux d'erreur au-dessus de 5 à 6 % | les requêtes reçues ne respectent plus le contrat de l'API | lire les causes : `manquantes` ou `inconnues` → format des données en amont, `hors_plage` → population sortant des bornes d'entraînement |
| Pannes serveur (5xx) | panne du modèle ou de l'API | logs Render et `/health`|
| Latence p95 en hausse | serveur saturé ou ralenti | comparer latence globale et temps d'inférence pour identifier la source |
| Taux de refus à plus de 5 points de la référence | la population a changé, ou le modèle juge différemment | ouvrir l'onglet Dérive pour trouver les variables en cause |
| PSI en vigilance, **à confirmer** | pic possible sur une seule fenêtre, surtout pour une colonne peu renseignée | attendre la fenêtre suivante |
| PSI en vigilance, **confirmée** | changement réel et durable de la population sur cette variable | analyse métier : nouveau canal, nouvelle offre, méthode de calcul des données |
| PSI **critique** confirmé | changement important sur une variable influente | réévaluer le modèle dès que des défauts réels sont disponibles et envisager de recalibrer le seuil ou de réentraîner |

### Résultats sur la simulation

Deux phases de 4 500 appels (journal du 02/10/2026), comparées à la référence :

- **Service stable** : aucune panne serveur, 5,3 à 5,7 % d'erreurs pour 5 % injectées. L'écart vient d'environ 0,6 % de clients réels refusés (variable obligatoire absente ou hors bornes).
- **L'inférence est le goulot d'étranglement** : 87 % de la latence, avec environ 2 % d'appels autour de 1 s liés au processeur partagé de Render.
- **Phase stable sans fausse alerte** : PSI maximal de 0,019 sur les 21 colonnes, taux de refus de 28,1 % contre 28,5 % pour la référence.
- **Phase de dérive détectée** : les deux variables déplacées par la simulation ressortent, `EXT_SOURCE_2` (PSI de 0,205) et `DAYS_BIRTH` (0,183). Le taux de refus gagne 8,3 points (36,8 %), le PSI du score reste à 0,061.
- **Détection en 290 à 340 appels** après le début de la dérive, avec des fenêtres glissantes de 500 appels.
- **Dans le tableau de bord** (fenêtre courante de 1 000 appels, en fin de phase de dérive) : `EXT_SOURCE_2` (0,213) et `DAYS_BIRTH` (0,165) sont en vigilance **confirmée**. `EXT_SOURCE_1` (0,101, renseignée à 46 %) est **à confirmer** : elle était à 0,089 sur la fenêtre précédente. Elle n'est pas visée par la simulation, mais elle est corrélée à l'âge.

### Points de vigilance

- **Taille de fenêtre** : sous 1 000 appels, le bruit du PSI dépasse le seuil de 0,1. Une fenêtre plus grande est plus fiable mais réagit plus tard : une dérive récente n'y apparaît qu'une fois une bonne part de ses appels renouvelée.
- **Colonnes peu renseignées** (`EXT_SOURCE_1`, `OWN_CAR_AGE`, `CC_CNT_DRAWINGS_ATM_CURRENT_MEAN`) portent les fausses alertes, d'où la confirmation sur deux fenêtres.
- **Verdict global d'Evidently** (dérive si 50 % des colonnes dérivent) : il ne se déclenche pas pour une dérive concentrée sur 2 colonnes sur 21. Le tableau de bord évalue donc colonne par colonne.
- **Dérive n'est pas performance** : sans les défauts réels, connus des mois plus tard, on détecte un changement de population, pas une dégradation du modèle. Le réentraînement se décide sur une dérive persistante **et** une performance mesurée en baisse.
- **Simulation simple** : dérive franche et brutale, sur deux variables. Les dérives lentes, multivariées ou portant sur les valeurs manquantes ne sont pas testées.
- **Seuils à calibrer** : l'écart de 5 points sur le taux de refus et le niveau critique de 0,25 sont des conventions, à ajuster sur du trafic réel.

### Suites possibles en production

- **Historiser les PSI** calculés (aucune donnée personnelle) dans une table dédiée, pour suivre la dérive au-delà de la purge des appels bruts à 90 jours.
- **Calcul automatique** tous les 1 000 appels, ou une fois par jour si le trafic est faible, avec notification au lieu d'une consultation manuelle du tableau de bord.
- **Performance réelle** : rapprocher les scores des défauts constatés quand ils sont connus (AUC, coût métier), seul moyen de savoir si le modèle se dégrade.

## Export du modèle depuis le registry

Après avoir promu un nouveau modèle (avec son registry et son jeu de données) :

```bash
# Réexporte le modèle, les métadonnées et les bornes dans model/, et la référence du monitoring
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
