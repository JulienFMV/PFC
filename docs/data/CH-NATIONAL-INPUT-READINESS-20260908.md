# D316 — Qualification des données nationales CH

Observation du 8 septembre 2026. Référence de code :
`dee652bc919d06345f71304d1f1eaacf0edf7bc7` (D315).

Les sources nationales existent. Le travail restant concerne leur préparation,
leur sens physique et leurs hypothèses futures. Les profils des clients FMV ne
sont ni des substituts au système suisse ni un préalable à cette qualification.

## Résultats et périmètre

Quatre SELECT bornés ont été soumis : un inventaire détaillé tronqué à 2 001
lignes a été rejeté, puis trois résultats complets ont été conservés. Le
dictionnaire comporte 94 groupes couvrant 21 familles ; le profil historique
contient 260 groupes et 3 124 297 lignes de versions ; le profil futur contient
14 groupes. Ce sont des agrégats de métadonnées, pas des exports de valeurs.
Les trois résultats ont été rapprochés indépendamment des réponses JSON brutes.

Le périmètre porte sur CH, les échanges depuis/vers CH et les prix day-ahead
CH/AT/DE_LU/FR/IT_NORD. Les tables consultées sont
`prd.gold.dimentsoeseries` et
`prd.silver.ge_power_entsoe_time_series_vintages`.
Les requêtes successives ne constituent pas un snapshot Delta atomique.

| Domaine | Présence constatée | Qualification encore nécessaire |
|---|---|---|
| Charge, production, prix, échanges | Historiques et familles nationales identifiés | Grille suisse complète, expansion des blocs, révisions, signes, unités, couverture physique |
| Hydro | ENTSO-E hebdomadaire ; source OFEN déjà préparée dans D300 | Réutiliser les 922 semaines OFEN jusqu'au 31 août 2026 ; la série ENTSO-E examinée finit plus tôt |
| Capacités installées | Types B10/B11/B12/B14 ; données par unité allant jusqu'à fin 2028 | Parc complet, PV/éolien et trajectoires 2027–2029 ; une date maximale ne prouve pas une couverture nationale |
| Indisponibilités | 4 179 lignes de versions futures, trois familles | Identité d'actif absente du champ supérieur dans ces lignes ; inspecter les détails imbriqués/raw avant jointure et dédoublonnage |
| Capacités frontalières | Familles NTC jour/semaine/mois/année | Capacités commerciales par direction et saison ; le profil NTC annuel examiné s'arrête à fin 2026 |
| Prévisions opérationnelles | Familles historiques présentes | Aucune ligne future dans les quatre familles charge/production/renouvelables et la fenêtre interrogée ; ne pas généraliser à tous les flux ni extrapoler vers trois ans |

La production réalisée inclut B10/B11/B12/B14/B16/B19 : pompage-turbinage,
fil de l'eau, réservoir, nucléaire, solaire et éolien terrestre. Cette
interprétation est recoupée dans la [liste officielle ENTSO-E v36, historique](https://www.entsoe.eu/Documents/EDI/Library/Core/entso-e-code-list-v36r0.pdf).
La [bibliothèque actuelle](https://www.entsoe.eu/publications/electronic-data-interchange-edi-library/)
annonce une version plus récente ; son ZIP n'a pas été lu par l'outil web.
La v36 n'est donc pas présentée comme la norme actuelle complète.

Pour les indisponibilités, une quantité publiée peut exprimer la **capacité
encore disponible** pendant l'événement. La transformer en MW perdus demande
l'identité de l'installation et la capacité de référence. Les déclarations
production/génération doivent aussi être réconciliées avant agrégation.
[Définition officielle ENTSO-E](https://transparencyplatform.zendesk.com/hc/en-us/articles/16652173943828-Planned-Unavailability-Changes-in-Actual-Availability-of-Generation-Production-Units-15-1-A-15-1-B-15-1-C-15-1-D).
Un code PT1M dans ces événements ne constitue pas une prévision nationale
continue à la minute. L'absence d'identifiant dans le champ supérieur ne prouve
pas son absence dans les objets imbriqués ou le document source.

## Ce que ces profils ne prouvent pas

La fenêtre historique sélectionne les **débuts** d'intervalles avant le
31 août 2026 à 22:00 UTC. Vingt groupes contiennent des intervalles dont la fin
dépasse cette limite. Malgré le nom local `closed-history`, ce résultat n'est
pas un jeu d'apprentissage entièrement fermé à cette date. Il faut appliquer
la règle de coupure à la matérialisation avant tout entraînement.

Les comptes de versions ne mesurent ni les heures, ni l'énergie, ni les trous
de couverture. Les labels de partitions annuelles ne prouvent pas davantage
une année suisse complète. Les indicateurs stockés signalent zéro valeur
nulle et zéro échec DQ dans ce profil ; cela n'admet pas automatiquement la
sémantique, la représentativité ou la continuité des séries.

La première observation Databricks enregistrée dans les groupes examinés est
le 7 août 2026 à 07:03:13 UTC. Un historique de livraison 2019 téléchargé en
2026 n'est pas une preuve indépendante de ce qui était connu en 2019. Il peut
servir à des expériences explicitement qualifiées d'historique révisé ; une
adoption réclamera des origines futures réellement archivées et un nouveau
holdout indépendant. T057 reste fermé.

## Ordre d'exécution proposé

1. Réutiliser D300 : 35 040 quarts d'heure physiques suisses couvrant septembre
   2025 à août 2026, ses règles d'expansion/révision et son hydro OFEN. Les six
   fichiers préparés et les six pièces hydro ont été revérifiés par SHA-256.
2. Qualifier les grilles nationales supplémentaires et le rapprochement des
   indisponibilités ; figer les hypothèses futures manquantes avec sources,
   unités, date d'observation et horizon explicites.
3. Ajouter le registre d'événements décrit dans
   [la proposition réseau](CH-STRUCTURAL-EVENT-REGISTRY-PROPOSAL.md), en priorité
   pour les changements touchant réellement les livraisons 2026–2029.
4. Avant résultats, figer un benchmark CPU limité contre D304 : même recette
   sur tous les mois, niveau et forme séparés, seuils pré-origine, saisons,
   années, horizons, négatifs/proches de zéro, rampes et pointes. Réutiliser
   l'assembleur et la projection EEX, puis vérifier les contraintes finales.

Aucun nouveau modèle, aucune courbe ni export candidat n'a été fabriqué dans
D316. Les candidats existants et D305–D307 sont préservés. Le solveur reste
l'unique autorité de niveau mensuel. Production, promotion, scientific_admission,
trading, externally_registered et countable_origin restent tous `false`.
CPU seulement ; zéro démarrage Warehouse, aucun changement de table/config.
Dernier état Warehouse observé : RUNNING, arrêt automatique 45 minutes ; arrêt
effectif non confirmé. Aucun fichier client supplémentaire n'est demandé.

Preuves locales : `build/lt-national-readiness-20260908/analysis-v1/`,
`review-v1/verification.json`, les SQL et reçus dans `national-*/`.
Le checkpoint de code a passé 591 tests, avec 5 tests ignorés ; le présent lot
ajoute seulement de la documentation et des diagnostics locaux.
