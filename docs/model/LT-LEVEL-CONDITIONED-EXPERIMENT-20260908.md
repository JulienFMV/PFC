# D318 — protocole CPU avant résultats

La référence reste D304 ; D305–D307 sont conservés comme comparaisons. Aucun
choix par mois, aucune modification du solveur, assembleur ou projection EEX.
Toutes les six autorités restent false. Aucun AFRY, T057, SQL, GPU ou nouveau
jour pilote. Les exports sont expérimentaux, avec quarts d'heure répétés.

## Challenge et recette figée

Tester `raw = D304 + b(calendrier) * B_solveur / 100`, puis centrage sur les
heures réellement présentes de chaque mois suisse. Le calendrier comporte
trois harmoniques sinus/cosinus, par saison et type de jour (54 coefficients).
Deux seules pénalisations ridge globales : 1 et 10, Gram normalisé par la masse
des poids. Aucun intercept supplémentaire : la pente nulle reproduit D304.
L'échelle fixe 100 EUR/MWh n'est pas une division par le prix. Aucun oracle
réalisé, clipping, sélection de mois, recherche supplémentaire ou réglage
après résultats. La correction linéaire peut extrapoler dangereusement :
publier le support des niveaux et maturités, sans le masquer par un plafond.

Apprendre les résidus signés par rapport au D304 de chaque ancienne origine,
avec son niveau solveur archivé ; uniquement les mois complets disponibles
dans les cibles closes de l'origine externe. Pondérer chaque heure livrée
par l'inverse de son nombre de paires. Les origines annuelles archivées sont
peu nombreuses : sans paire, pente nulle et statut UNSUPPORTED_NO_PAIRS.
Ne pas fabriquer de niveaux solveur pré-2021 pour augmenter le support.
Ces reconstructions utilisent des sources rétrospectives latest-observed ;
elles ne prouvent pas une disponibilité historique PIT indépendante.

## Population et critères

Les six origines 2021–2026 et l'export courant du 7 septembre hérités de D308
sont tous `development_exposed` (courant descriptif). Les quatre origines
2023–2026 forment seulement un screening exploratoire. Reprendre les seuils
pré-origine D308, les sérialiser et les recalculer avant fit. Avant résultats,
lier plan, code, paires, seuils et entrées par SHA-256 dans un dossier neuf.
Maximum 14 fits et 21 assemblages (dont sept contrôles ablation), CPU 4 threads.

Comparer avant et après projection, niveaux et formes séparés, sur des
populations UTC ordonnées identiques entre tous les candidats et stages.
Rapporter couverture exacte des mois, années, saisons, maturités, négatifs,
proches de zéro, rampes, pointes et queues signées. Une année partielle n'est
jamais déclarée complète ; les mois sans vérité restent hors score et visibles.
Les segments vides ont une métrique absente, jamais PASS ; une métrique non
finie d'un segment non vide arrête le screening.

Critères fixés : gains MAE et RMSE forme >=2%, au moins trois origines gagnées,
aucune régression >5% MAE ou RMSE forme/rampe dans aucun segment d'aucune origine.
Un régime robuste exige >=168 heures et >=2 origines ; les pertes sans ce
support restent explicitement listées et opposent aussi un veto. Les trois
queues signées doivent avoir ce support. L'absence de perte ne vaut pas preuve
de gain. Même un screening favorable ne change pas la référence ni l'admission.
Vérifier indépendamment coefficients, calendriers, populations, métriques,
projection et conservation des niveaux, ainsi que les anciennes liaisons.

## Holdout séparé

La fenêtre future octobre 2026–septembre 2027 reste un projet non enregistré
indépendamment : sources de vérité, modalités de constat et dépositaire à
qualifier. Aucun résultat de ce lot n'est un holdout ; aucune origine passée
ni date de commit ne peut devenir `countable_origin=true`.
