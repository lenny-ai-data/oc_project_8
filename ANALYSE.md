# Analyse du monitoring et des performances

Ce document retrace l'évolution du service de scoring en production : de la première version mise en ligne (02/10/2026) à la version servie par ONNX Runtime (05/10/2026), en passant par l'analyse et l'optimisation de l'inférence en local.

Les chiffres de production viennent du journal des appels (Supabase) au moment des mesures ; ils sont repris ici pour garder la trace de l'évolution, indépendamment du contenu de la base. Le fonctionnement de la solution et la grille de lecture des signaux du monitoring sont dans le [README](README.md).

## 1. Protocole

- **Trafic** : en l'absence de vrais clients, `scripts/simuler_trafic.py` rejoue sur l'API déployée des clients du jeu de test de P6, en deux phases de même taille : **stable** (tirage au hasard), puis **dérive** (tirage biaisé vers les clients les plus jeunes et les `EXT_SOURCE_2` les plus faibles). Environ 5 % des requêtes sont volontairement rendues invalides.
- **Mesures** : une ligne par appel dans le journal, avec la **latence** (mesurée par le middleware de l'API), le **temps d'inférence** (modèle seul) et la **révision** déployée (commit), qui permet de comparer les versions entre elles.
- **Dérive** : PSI des 20 variables les plus importantes du modèle et du score, par fenêtres de 1 000 appels, face à une référence de 10 000 clients d'entraînement. L'étude complète (choix des variables, du test et de la taille de fenêtre) est dans le notebook `Jacquinot_Lenny_6_notebook_drift_102026.ipynb`.

| Version | Révision | Date | Appels simulés | Inférence |
|---|---|---|---|---|
| Initiale | `1bb27d2` | 02/10 | 2 × 4 500 | modèle MLflow (`pyfunc.predict`) sur un DataFrame |
| Code optimisé | `614eec2` | 05/10 | 2 × 500 | pipeline sklearn appelé directement, entrée numpy, un thread |
| ONNX | `1ba5399` | 05/10 | 2 × 500 | ONNX Runtime, un thread |

Les deux dernières simulations envoient exactement les mêmes clients (graine fixe) : 948 appels réussis chacune.

## 2. État initial (02/10)

### Service et erreurs

- **Aucune panne serveur**, 5,3 % d'erreurs en phase stable et 5,7 % en phase de dérive, pour 5 % injectées.
- L'écart vient d'environ **0,6 % de clients réels refusés** : variables obligatoires absentes (`OBS/DEF_30/60_CNT_SOCIAL_CIRCLE`, `EXT_SOURCE_2`) ou quelques valeurs hors des bornes d'entraînement (variables `BURO_`, `PREV_`, `CC_`). C'est la limite assumée de la validation stricte de l'API.

### Temps de réponse

| Appels réussis (8 504) | Médiane | p95 | p99 |
|---|---|---|---|
| Latence | 113 ms | 209 ms | 1 047 ms |
| Inférence | 97 ms | 191 ms | 1 020 ms |

- **L'inférence est le goulot d'étranglement** : 97 ms sur 113 en médiane, le reste de l'API (validation, journal) ne prenant que 13 ms environ.

### Dérive des données

- **Phase stable sans fausse alerte** : PSI maximal de 0,019 sur les 21 colonnes, taux de refus de 28,1 % contre 28,5 % pour la référence.
- **Phase de dérive détectée** : les deux variables déplacées par la simulation ressortent, `EXT_SOURCE_2` (PSI de 0,205) et `DAYS_BIRTH` (0,183). Le taux de refus gagne 8,3 points (36,8 %), alors que le PSI du score reste à 0,061, sous le seuil.
- **Détection en 290 à 340 appels** après le début de la dérive, avec des fenêtres glissantes de 500 appels.
- **Dans le tableau de bord** (fenêtre courante de 1 000 appels, en fin de phase de dérive) : `EXT_SOURCE_2` (0,213) et `DAYS_BIRTH` (0,165) sont en vigilance **confirmée**. `EXT_SOURCE_1` (0,101, renseignée à 46 %) est **à confirmer** : elle était à 0,089 sur la fenêtre précédente. Elle n'est pas visée par la simulation, mais elle est corrélée à l'âge.

## 3. Analyse et optimisation en local

### Méthode

`scripts/profiler_api.py` mesure la route `/predict` sur le vrai modèle, en local :

- **chronométrage pas à pas** : préparation de l'entrée et inférence, en ajoutant une optimisation à la fois (médiane et p95 sur 200 répétitions, après une chauffe) ;
- **cProfile** de 200 requêtes complètes via le `TestClient`, profil sauvegardé pour **snakeviz**.

```bash
# Lancement profiler
uv run python -m scripts.profiler_api

# Exploration graphique du profil
uv run --group optimisation snakeviz data/profil_predict.prof
```

### Goulots identifiés

1. **La validation de schéma de MLflow** (`_enforce_schema`) : 10,0 s sur les 13,5 s passées dans la route pour 200 requêtes sous cProfile, soit environ 75 %. `pyfunc.predict` contrôle les 527 colonnes une par une et reconstruit un DataFrame, en doublon de la validation de l'API (Pydantic et `verif_client`).
2. **Le DataFrame pandas d'une seule ligne** : environ 5 ms au total, 5x le temps du modèle lui-même.
3. **Les threads de LightGBM** (`n_jobs=-1`) : autant de threads que de cœurs de la machine hôte pour scorer une seule ligne.

### Optimisations pas à pas

Evolution de l'optimisation, chaque ligne ajoutant une brique à la précédente :

| Optimisation | Médiane | p95 | Gain |
|---|---|---|---|
| **Initial** : DataFrame et `pyfunc.predict` | 21,5 ms | 26,6 ms | 1× |
| Pipeline sklearn appelé directement, sans la validation MLflow | 5,8 ms | 6,6 ms | 3,7× |
| Entrée en ligne numpy plutôt qu'en DataFrame | 0,97 ms | 1,02 ms | 22× |
| LightGBM sur un seul thread (**Version optimisée**) | 0,63 ms | 0,79 ms | 34× |
| Classifieur converti en ONNX, servi par ONNX Runtime (**Version ONNX**) | 0,07 ms | 0,12 ms | ≈ 330× |

- ONNX Runtime fait mieux que le booster LightGBM appelé seul : il supprime l'enrobage Python qui domine pour une seule ligne.
- La requête complète passe de 36 ms à 13 ms en local (écriture du journal comprise, que le `TestClient` attend) : le modèle n'y pèse presque plus rien.

### Précision

- **Version optimisé** : scores identiques au modèle MLflow (écart nul sur les 10 000 clients de la référence), le pipeline gardant sa conversion en float32.
- **Version ONNX** : le convertisseur stocke les seuils des arbres en float32, contre float64 dans LightGBM, ce qui entraine une quantification subie des paramètres. Un client dont une variable tombe sur un seuil arrondi change de branche. Sur les 10 000 clients de la référence on relève un p99 de 3,5·10⁻⁷, maximal de 1,4·10⁻³, et aucune décision changée au seuil métier. 
- **Tests** : `test_non_regression_reference` compare le modèle servi au score du modèle MLflow exporté, sur les 10 000 clients. Il accepte l'approximation dans des limites fixées : écart sous 5·10⁻³ et au plus 0,1 % de décisions inversées. Rejoué en CI, il bloque un réexport ou une mise à jour d'ONNX qui dégraderait les scores.

## 4. Version optimisée en production (05/10, `614eec2`)

Les trois optimisations du code, déployées ensemble par la CI/CD :

| Appels réussis | Initiale | Code optimisé |
|---|---|---|
| Latence médiane | 113 ms | **8,2 ms** |
| Latence p95 | 209 ms | 11,5 ms |
| Latence p99 | 1 047 ms | **17,4 ms** |
| Inférence médiane | 97 ms | **1,5 ms** |
| Inférence p99 | 1 020 ms | 3,4 ms |
| Hors inférence (médiane) | 13 ms | 6,7 ms |
| Appels au-delà de 500 ms | 2,4 % | 0,1 % (1 appel) |
| Part de l'inférence dans le temps total | 81 % | 17 % |

- **Le gain est plus fort sur Render qu'en local** : l'inférence est divisée par 64, contre 34 en local. Le processeur partagé amplifie le coût de la validation MLflow, de pandas et des threads.
- **Les appels ralenti autour de 1 s disparaissent**. Le passage à un seul thread en est la cause la plus probable.
- **Le goulot change de place** : le temps hors modèle représente désormais 83 % de la latence.
- **Mesure reproductible** : une seconde simulation, 30 minutes plus tard, donne les mêmes chiffres à 0,2 ms près en médiane.

## 5. Version ONNX en production (05/10, `1ba5399`)

Le classifieur converti en ONNX et servi par ONNX Runtime, avec les mêmes clients que la version précédente :

| Appels réussis | Initiale | Code optimisé | ONNX |
|---|---|---|---|
| Latence médiane | 113 ms | 8,2 ms | **7,2 ms** |
| Latence p95 | 209 ms | 11,5 ms | **10,4 ms** |
| Latence p99 | 1 047 ms | 17,4 ms | 24,1 ms |
| Inférence médiane | 97 ms | 1,5 ms | **0,33 ms** |
| Inférence p99 | 1 020 ms | 3,4 ms | 0,63 ms |
| Hors inférence (médiane) | 13 ms | 6,7 ms | 6,9 ms |
| Appels au-delà de 500 ms | 2,4 % | 0,1 % | 0,1 % |
| Part de l'inférence dans le temps total | 81 % | 17 % | **5 %** |

- **ONNX divise encore l'inférence par 5** et gagne environ 1 ms de latence médiane (13 %). Un gain résiduel après l'optimisation du code, mais au-delà de la variation entre deux simulations (0,2 ms).
- **Le p99 plus élevé n'est pas significatif** : sur 948 appels, il se joue sur une dizaine d'appels, à la merci des variations de Render.
- **Version conservée** : c'est la plus rapide, et son approximation est mesurée, bornée par les tests et sans effet observé sur les décisions. Contreparties : les scores ne sont plus exactement ceux du modèle validé en P6, et onnxruntime s'ajoute à l'image. Pour éviter qu'un `model.onnx` désynchronisé passe en production, il est produit par le même export que le modèle MLflow, marqué de sa version et de son run : s'ils diffèrent, l'API ne sert aucun score et ses routes répondent 503 avec la cause.

En résumé, la latence médiane passe de **113 à 7 ms** et le p99 d'environ **1 s à moins de 25 ms**, sans régression des décisions.

## 6. Limites et pistes

- **Le goulot est désormais hors du modèle** : environ 7 ms par requête sur Render pour la validation Pydantic des 527 champs, `verif_client` et le middleware, qui relit le corps JSON pour le journal.
- **Scalabilité** : un seul processus uvicorn sur un processeur partagé (offre gratuite de Render). Pour plus de débit, plusieurs workers ou instances ; pour une latence stable, un processeur dédié (offre payante). Pour du scoring par lots, ONNX Runtime traite tout un lot de clients en un seul appel.
- **Dérive** : la simulation teste une dérive franche et brutale sur deux variables ; les dérives lentes, multivariées ou portant sur les valeurs manquantes ne sont pas testées.
