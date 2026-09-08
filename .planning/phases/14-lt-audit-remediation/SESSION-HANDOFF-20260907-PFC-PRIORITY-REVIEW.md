# D303 — Recentrage des lots sur la qualité de la PFC FMV

## Décision et état final

La mission est d'améliorer la PFC FMV pour chaque année de livraison et à chaque
nouvelle version. Une courbe conforme aux marchés, une forme prédictive et une
représentation explicite des changements futurs sont trois exigences distinctes.
Un algorithme récent ou une simulation physique correcte ne garantit pas, seul,
une meilleure PFC.

La revue demandée est terminée. L'ordre est ajusté : l'intégration d'une forme
signée et sa comparaison dans la PFC finale passent avant un noyau isolé de
stockage/flexibilité. Cela précise la suite de D302 et remplace son indication
« prochain lot : noyau chronologique ». L'inventaire D302 et ses cibles restent
utiles aux deux voies, statistique et physique. Aucun nouveau modèle, adaptateur,
entraînement ou calcul de PFC n'a été exécuté dans cette revue.

## Preuves recalculées

Diagnostic local après observation des résultats D301, sans nouvelle sélection.
Trois comparateurs, quatre origines 2023–2026, 70 101 couples origine/heure et
32 135 heures distinctes. Les horizons se recouvrent et les données sont des
historiques révisés : cette analyse reste une preuve de développement.

MAE de forme mensuellement centrée, EUR/MWh, moyenne donnant le même poids à
chaque origine disponible :

| Livraison | MLP actuel | LightGBM | Référence saisonnière | Couverture |
| --- | ---: | ---: | ---: | --- |
| Toutes les périodes | 22,817767 | 22,915155 | 23,113075 | 4 origines, 96 mois-origines |
| Première année | 22,794260 | 22,565877 | 22,980426 | 4 origines, 44 mois-origines |
| Deuxième année | 22,982806 | 22,815993 | 23,231179 | 3 origines, 32 mois-origines |
| Troisième année | 23,515624 | 23,830649 | 23,823004 | 2 origines, 20 mois-origines |

Ces années ne partagent pas la même population. La dernière année disponible
peut être partielle ; cette table ne prouve pas une dégradation causée par
l'échéance. Elle ne justifie aucun changement automatique de modèle par horizon.
Le résultat global D301 et le modèle de référence restent inchangés.

Pour le MLP actuel, les 2 250 observations dont le prix réalisé est négatif
représentent 3,209655 % des couples origine/heure, 10,656619 % de la somme des
erreurs absolues et 25,091520 % de la somme des erreurs au carré. Les trois
comparateurs ne prédisent aucune heure négative sur la population évaluée.
Les proportions correspondantes d'erreur au carré sont 27,545689 % pour LightGBM
et 25,531356 % pour la référence. Ce sont des contributions sur la population
regroupée, et non des moyennes de pourcentages par origine ou des gains possibles.

Sur l'entraînement précédant l'origine 2026, 805 heures de prix négatifs sont
présentes : 116 appartiennent à des journées exclues par la moyenne <=5 EUR/MWh,
689 sont conservées mais leur ratio est ramené au plancher positif. La variable
de maturité est constante à zéro dans les six matrices d'entraînement.

Cela motive une expérience sur la représentation de la cible. Cela n'isole pas
la cause de toute l'erreur : incertitude, covariables, niveaux à terme, modèles
et assemblage interviennent aussi. Une prévision centrale peut rester positive
alors que certaines réalisations sont négatives. Ne jamais imposer à la PFC
centrale une fréquence de prix négatifs copiée de l'historique ; la distribution
et les trajectoires de risque nécessitent une évaluation séparée.

La dernière projection des produits EEX réduit la MAE du MLP de 20,968306 à
20,399366 en origine 2024, de 22,568078 à 21,055968 en 2025 et de 28,335057 à
25,426517 en 2026. Effet numérique nul en 2023, où les PEAK sont absents.
Comparer une nouvelle forme avant cette projection seulement serait insuffisant.
Les contraintes restent obligatoires, indépendamment de leur effet statistique.
Les QUOTE_CONFLICT/UNSUPPORTED D301 ne sont ni résolus ni dispensés par ce calcul.

## Ordre d'exécution adopté

1. **Prochain lot : expérience locale de forme signée jusqu'à la PFC finale.**
   Reprendre la cible D302, conserver les prix négatifs et les mois complets
   antérieurs à chaque origine. Faire évoluer explicitement l'interface de
   l'assembleur existant avec une nouvelle identité d'exécution ; conserver les
   sources et résultats de référence avant toute évolution de leurs dépendances.
   Ne pas masquer des EUR/MWh dans `f_H`, diviser par un niveau potentiellement
   nul, détourner la correction hydro ou compter deux fois la forme hebdomadaire.
   La forme mensuelle signée contient déjà les écarts entre jours.
   Vérifier d'abord l'intégration : signes, niveaux zéro/négatifs, DST, mois
   bissextiles, BASE du solveur et PEAK supportés, chemin historique inchangé
   lorsque la nouvelle entrée est absente. Puis exécuter un benchmark borné
   avec une référence signée simple et un challenger CPU rapide, en gardant
   constants données, covariables, budgets et contraintes. Rapporter séparément
   performance de la forme proposée et performance après projection EEX.
   Mesurer les première/deuxième/troisième années, régimes et couverture avant
   toute décision. Aucun gain n'est acquis d'avance ; conserver le MLP si le
   candidat échoue. Ne pas simultanément changer pondération, maturité et 15 min.

2. **Préparer maintenant les hypothèses du démonstrateur physique 2030.**
   Le propriétaire modèle doit expliciter un rapprochement de scénarios publics,
   les capacités, usages électriques, profils météo, disponibilité et hypothèses
   d'exploitation. Le data engineer assure sources, unités, révisions et dates ;
   il ne décide ni de la pénétration future des batteries ni de la formulation
   du marché. Une hypothèse d'exploitation décidée par le modélisateur peut être
   testée et identifiée comme telle ; elle ne doit pas être fabriquée comme une
   donnée PRD attestée. Les lacunes D302 limitent les affirmations sur 2030,
   sans interdire le développement local ni les sensibilités explicites.

3. **Démonstrateur physique réduit, relié aux prix et à la PFC.**
   Après qualification de l'interface commune et formulation d'un cas cohérent,
   traiter une année de référence et un cas 2030 : demande, production, stockage,
   flexibilité, hydro et échanges aux frontières dans une chronologie cohérente,
   avec une règle de formation des prix. Tester les bilans, limites de puissance,
   rendements, récupération de demande et conditions de début/fin de stockage.
   Mesurer les effets PV/BESS/demande flexible et la déformation finale après
   contraintes de marché. Un noyau de batterie isolé sans ces relations ne
   constitue pas ce livrable. Des fixtures peuvent valider le code mais ne
   remplacent jamais les sources EEX/ENTSO-E ou une prévision de 2030.
   Étendre ensuite les années et météos ; pas de grand modèle d'investissement
   paneuropéen préalable. Pour un scénario modifiant les niveaux, utiliser le
   produit fondamental séparé, jamais écraser les niveaux de la PFC centrale.

4. **Complexité supplémentaire selon le problème encore mesuré.**
   Les comparateurs NBEATSx et les pilotes CPU Chronos-2/TiRex-2 restent dans la
   suite, après stabilisation de la cible et des entrées qu'ils compareront.
   Une maturité apprise exige des exemples historiques de prévision à différentes
   échéances, avec information disponible à l'origine ; ajouter seulement une
   colonne `years_ahead` ne suffit pas. L'intraday, les trajectoires probabilistes
   et l'utilité pour la couverture/hydro FMV ont leurs expériences propres.
   La mise en production et les affirmations indépendantes gardent leurs gates.

À chaque incrément : nouvelle courbe comparable, gain/échec visible, explication
des changements, conservation de la meilleure version démontrée. Le futur
holdout indépendant sert à confirmer les choix après développement ; les années
D301 déjà examinées ne retrouvent pas leur indépendance par renommage.

## Rapport à l'état de l'art au 31 août 2026

Deux sources primaires ont été revérifiées en lecture seule le 7 septembre.
[FETS v2, 17 juillet 2026](https://arxiv.org/abs/2604.22328v2) motive la mise à
l'essai de Chronos-2/TiRex-2 avec covariables, par ses résultats agrégés en séries
énergétiques. Il ne démontre pas leur supériorité sur notre PFC suisse à trois ans.
[REMIND–PyPSA-Eur v1, 5 octobre 2025](https://arxiv.org/abs/2510.04388v1), publié
en 2026, illustre le couplage de trajectoires sectorielles et d'exploitation
horaire avec stockage et flexibilité. Notre séquencement réduit est une décision
d'ingénierie fondée sur les lacunes locales ; ces articles ne le prescrivent pas
et ne prouvent pas un gain FMV.

## Fichiers, exécution, vérification

Fichiers documentaires modifiés : `.planning/HANDOFF.md`, le `DECISION-LOG.md`
Phase 14, `docs/model/LT-STRUCTURAL-SHAPING-CONTRACT.md` et ce nouveau handoff.
Aucun fichier de code produit, aucun adaptateur ni aucune donnée de bureau modifié.

Racine de travail : `build/lt-priority-review-20260907/`.

- `review.py` : vérification des empreintes, effets de filtrage de cible,
  décomposition par année, contribution des heures négatives, projection EEX.
- `results-v2/` : `scope.json`, `review.json`, `manifest.json`, six CSV
  (`training-effects`, `by-origin-year`, `delivery-year-summary`,
  `negative-hours-by-origin`, `negative-error-contributions`, `projection-effects`).
- `verify.py`, `verification.json`, `verification.log` : vérification distincte
  des sommes via `math.fsum`, population horaire, partitions annuelles et sources.
- `execution.log`, `results/scope.json` : première tentative conservée.
  `execution-v2.log` : exécution finale réussie.

Commandes après contrôle exact cwd et Git top-level canonique, avec
`build/conda-runtime-v41-model-source/python.exe -B` :

1. `build/lt-priority-review-20260907/review.py`
2. `build/lt-priority-review-20260907/verify.py`
3. `git diff --check` et contrôle ciblé des nouveaux fichiers documentaires.

Les journaux sont capturés par `Tee-Object` dans cette racine. TEMP/TMP, APPDATA,
LOCALAPPDATA, MPLCONFIGDIR, XDG_CACHE_HOME, NUMBA_CACHE_DIR, JOBLIB_TEMP_FOLDER,
PYTHONUSERBASE et PIP_CACHE_DIR sont placés sous sa sous-arborescence `runtime/`.
PYTHONDONTWRITEBYTECODE=1, OMP/OPENBLAS/MKL_NUM_THREADS=4,
CUDA_VISIBLE_DEVICES=-1. Aucun paquet installé.

Première tentative : assertion trop forte dans le script de diagnostic,
exigeant des prévisions identiques dans les quatre quarts d'heure. Cette
contrainte est correcte pour les observations CH transportées, mais pas pour
les prévisions, dont la maturité varie dans l'heure. Correction limitée au
diagnostic : quatre quarts d'heure requis, moyenne horaire des prévisions,
égalité stricte conservée pour les observations. Résultats v2 dans un nouveau
répertoire ; aucun résultat D301 réécrit. Un premier chemin de lecture du runner
avait également été mal nommé (`run_local_lt_benchmark.py`) ; corrigé par
recherche vers `scripts/run_lt_local_benchmark.py`, sans effet sur l'exécution.

Vérification finale : PASS, 139 fichiers d'entrée et 8 sorties contrôlés,
12 couples modèle/origine vérifiés indépendamment, 6 origines d'entraînement.
Partitions annuelles exhaustives ; scores globaux D301 reproduits. Les
empreintes des sources/code D301, de la courbe/manifeste préparé D300 et des
artefacts D302 restent identiques. Pas de nouveaux tests produit nécessaires
pour ce diagnostic sans modification du produit ; les matrices antérieures
ne sont pas présentées comme réexécutées.

Manifeste de diagnostic SHA-256 :
`5eeb5831a8bc00287b7598915a508de747494effa5bcca572f43dde30460655a`.

Toutes les autorités production/promotion/scientific_admission/trading/
externally_registered/countable_origin restent false. Zéro fit, nouvelle
prévision, appel Warehouse ou GPU. Aucune valeur AFRY consommée, aucun T057,
aucun contact externe ou promotion. Aucune nouvelle permission nécessaire
pour reprendre le prochain lot local déjà autorisé.
