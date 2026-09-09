# D318 — garde-fous et benchmark de forme conditionnée au niveau

**D304 reste la référence. Les deux recettes sont rejetées.** Le protocole
[figé avant résultats](LT-LEVEL-CONDITIONED-EXPERIMENT-20260908.md) teste une
pente calendaire signée contre les niveaux solveur archivés, avec ridge 1/10.
Aucune moyenne réalisée n'est utilisée comme covariable de niveau.

| Recette | Gain MAE forme | Gain RMSE forme | Origines gagnées | Pertes origine/régime | Dont sans support multi-origine |
|---|---:|---:|---:|---:|---:|
| ridge 1 | −4,268087 % | −1,677367 % | 2/4 | 37 | 2 |
| ridge 10 | −0,505495 % | −0,031569 % | 2/4 | 4 | 0 |

Un gain négatif est une dégradation. Ces pertes sont des cellules origine ×
régime × erreur, pas des origines indépendantes. Les quatre origines 2023–2026
sont déjà exposées ; leurs livraisons se recouvrent. Aucun holdout ni admission.
Le résultat rejette les deux recettes, pas toute hypothèse de conditionnement.

F-01 exige maintenant chaque clé mensuelle BASE à la frontière d'évaluation,
sans toucher l'assembleur gelé. F-06 impose les populations ordonnées identiques
dans le runner et le vérificateur de maturité. F-10 remplace le modèle horaire
futur par une sentinelle sans état dans les recherches signées concernées.
F-05 expose les pertes sans support et impose le veto par origine ; le nouveau
benchmark applique le veto jusque dans chaque régime de chaque origine.

Le premier lancement s'est arrêté avant fit/score parce que la signature de la
sentinelle ne satisfaisait pas le constructeur. Correction ciblée et test du
constructeur réel, puis nouveau gel avant fit. Recettes, données et critères
inchangés. Plan exécuté SHA-256 :
`5668b23c6536ecbb7d6724c17307461ab9a9fb57db511a4a2044c9cb1dea5d4e`.

Validation : **736 tests réussis, 5 skips, 0 échec**, 40 modules, 127,239 s.
Douze fits et 21 assemblages ; 2021 reste explicitement sans paire et à pente
nulle. Les sept ablations D304 sont exactement égales. Rejeu arithmétique
indépendant de 3 288 métriques, coefficients, paires, populations, projections
et exports. Écart maximal des moyennes solveur : `2,3988e-11 EUR/MWh`.
Les vérifications sont locales ; aucune nouvelle CI distante n'est revendiquée.

Les 1 063 artefacts antérieurs et 2 290 liaisons d'entrée sont revérifiés.
Le pilote reste à 1/20. D305–D307, l'assembleur et la projection EEX sont
préservés. Six nouveaux CSV couvrent octobre 2026–décembre 2032, avec 54 817
heures et 219 268 quarts d'heure répétés par recette. Parmi les 75 mois,
39 dépassent la maturité maximale d'apprentissage. Les anciens exports restent
intacts ; la valorisation du 7 septembre n'est pas présentée comme actualisée.

Preuves numériques locales : `build/lt-level-conditioned-20260908/`, notamment
`RAPPORT-BENCHMARK.md`, `exports.json`, `summary.json`, `preservation.json`,
`independent-guards/verification.json`, `independent-v1/verification.json`,
`run-v2/plan.json`, `run-v2/results/decision.json`, `tests-matrix.xml`.
Le rapport détaille niveau/forme avant et après projection, couverture annuelle
exacte, saisons, maturités, régimes et extrapolation. Ces artefacts locaux ne
sont pas rendus publiquement reproductibles par la seule présence de ce document.

Limites ouvertes : sources rétrospectives non PIT, petite archive d'origines,
conflits de cotes sans politique signée admise, trajectoires nationales et futur
holdout à qualifier. Aucun troisième ridge ni choix par mois au vu des résultats.
Toutes les autorités restent false : production, promotion, scientific_admission,
trading, externally_registered, countable_origin. Aucun SQL/Warehouse/GPU,
AFRY, T057 ou changement de données protégées.
