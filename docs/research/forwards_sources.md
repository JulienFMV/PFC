# Phase 0 — Cadrage des sources EEX forwards

**Date :** 2026-05-05
**Owner :** E1 (Forward markets EEX) — reviewer E5 (Data architecture)
**Branche :** `claude/audit-pfc-forwards-q73iC`

---

## Mise à jour 2026-08-05 — EEX `prd.gold` et travail local

La source EEX effectivement disponible pour le travail LT est désormais la
jointure en lecture seule de :

- `prd.gold.facteexpricedaily` ;
- `prd.gold.dimeexproduct` ;
- `prd.gold.dimeexdeliveryperiod`.

Une extraction CH POWER unique a été conservée sous
`build/databricks-eex-daily/2026-08-05/`. Elle contient 82 552 cotations du
2019-01-02 au 2026-08-04. Le prix retenu est `SettlementPrice`, en EUR/MWh ;
`LastPrice` reste une information annexe et n'est jamais utilisé comme
fallback. Les prix nuls et négatifs sont conservés.

La normalisation locale est implémentée dans
`pfc_shaping/data/databricks_eex_daily_snapshot.py`. Elle sépare explicitement
deux couches : 72 175 cotations encore négociables pour tous les produits
DAY/WEEK/WEEKEND/MONTH/QUARTER/YEAR, et une vue CAL/Q/M de 34 105 lignes pour
le solveur mensuel. DAY, WEEK et WEEKEND sont donc conservés pour le shaping et
les diagnostics ; ils ne deviennent pas automatiquement des contraintes de
niveau mensuel. La dernière date observée contient 74 contrats vivants, dont
38 CAL/Q/M.

Les conventions de livraison sont contrôlées contre le fichier public EEX
« Contract Details » du 2026-07-22. Pour les contrats suisses courts, DAY PEAK
livre de 08:00 à 20:00 tous les jours, y compris samedi et dimanche ; WEEK PEAK
livre du lundi au vendredi, soit 60 heures au total ; WEEKEND PEAK livre 12
heures le samedi et 12 heures le dimanche, soit 24 heures au total. Il ne
s'agit pas d'une bande continue de 24 heures par jour. Cette définition des
familles courtes est distincte du calendrier PEAK mensuel limité aux jours
ouvrés. WEEK BASE couvre lundi-dimanche et WEEKEND BASE samedi-dimanche ; leurs
tailles 167/168/169 et 47/48/49 heures reflètent le DST Europe/Zurich.

Le fichier officiel est lié par SHA-256 et 14 082 lignes suisses
DAY/WEEK/WEEKEND sont rejouées : zéro incohérence de taille de contrat et 1 564
lignes DAY PEAK livrées un samedi ou un dimanche. Toute cotation dont la date
n'est pas strictement antérieure au
début de livraison est exclue de la couche vivante. Le contrôle isole 10 376
de ces lignes non vivantes et une anomalie de borne WEEK PEAK, sans les effacer
de la quarantaine traçable.

La normalisation sélectionnée est le bundle local adressé par contenu
`2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`.

Cette capture remplace les requêtes Databricks répétées pour le développement
local et fournit des identités compatibles avec le solveur mensuel. Elle ne
prouve toutefois pas l'heure historique de disponibilité des cotations : elle
n'est donc ni une preuve PIT signée, ni une autorisation de rolling-origin, de
sélection ou de promotion. L'ancien EEX local ne doit pas être réintroduit
comme substitut empirique. ENTSO-E Databricks, le catalogue de vintages signé
et un nouveau holdout futur indépendant restent requis.

AFRY et OMPEX demeurent des benchmarks de diagnostic uniquement. Ils ne
déterminent ni les niveaux mensuels, ni la sélection du modèle. Le solveur
mensuel CH reste l'unique autorité de niveau.

### Audit local de la surface historique EEX

L'audit hors ligne `pfc_shaping/validation/databricks_eex_surface_audit.py`
est figé dans le bundle adressé par contenu
`bb1a09932b4bbff31dfdbb4ada561befb02050413ee819b03bf6c28f4858ab54`.
Il rejoue exactement les 34 105 cotations CAL/Q/M vivantes sur 1 939 dates, sans
requête ni démarrage de Warehouse Databricks.

La couverture PEAK ne commence que le 2023-06-26. Elle est absente avant cette
date, puis devient quotidienne en 2024 et 2025 ; elle ne peut donc pas soutenir
une calibration ou une validation PEAK directe avant mi-2023. La surface du
2026-08-04 est complète sur ses 19 identités BASE/PEAK.

L'audit v2 exige une partition enfant complète et non chevauchante. Pour chaque
trimestre d'un CAL, il utilise les trois mois uniquement si les trois sont
présents ; sinon il utilise le trimestre complet. Un ou deux mois ne peuvent
jamais être combinés avec la cotation du même trimestre comme si celle-ci ne
représentait que le mois restant.

Parmi 3 255 parents vivants comparables, aucun écart ne dépasse
0,01 EUR/MWh. Les 1 615 comparaisons CAL ont un maximum absolu de
0,005370 EUR/MWh ; les 1 640 comparaisons trimestre-vers-mois ont un maximum de
0,004433 EUR/MWh. Les 437 conflits du premier audit étaient des faux positifs
créés par une partition chevauchante. L'exclusion indépendante des cotations
dont la livraison avait déjà commencé reste correcte pour définir la couche
forward vivante, mais elle n'explique pas à elle seule la disparition des
conflits : l'audit corrigé trouve également zéro conflit sur l'ancien historique
CAL/Q/M non filtré.

La politique solveur reste stricte et inchangée. La dernière surface BASE est
acceptée sur 77 mois avec 18 contraintes actives, deux parents redondants
cohérents et un résidu maximal de `8,5265e-14` EUR/MWh. Aucun mécanisme de
contournement des conflits n'a été ajouté.

Les 10 766 paires BASE/PEAK produisent un OFFPEAK implicite sans aucun échec de
recomposition et sans OFFPEAK implicite négatif observé. Ce contrôle valide
l'identité algébrique locale, pas l'historique de disponibilité PIT. Le bundle
reste donc interdit pour rolling-origin, sélection, promotion et production.

### Couverture locale des cotations par horizon

L'audit hors ligne `pfc_shaping/validation/databricks_eex_horizon_audit.py`
rejoue les mêmes 34 105 cotations CAL/Q/M vivantes depuis le Parquet local.
Le bundle adressé par contenu est
`435ecbc737f95268f03f7f347dfafc4163f5b5a4cb5b8dc9cec87e05f1645108`
(manifest SHA-256
`949a1598710ad6198b569fd4cb00a99750707e162dbc0c709fefaa04f28e0ed1`).
Aucune requête Databricks et aucun démarrage de SQL Warehouse ne sont
nécessaires pour le reproduire.

L'historique contient 1 939 dates de cotation et 220 cycles de vie produits.
Les contrats mensuels directs atteignent au maximum 185 jours en BASE et 187
jours en PEAK. Seulement 81 cotations mensuelles se situent au-delà de 180
jours et aucune au-delà de 365 jours. La forme mensuelle directe est donc
observable jusqu'à environ six mois, mais pas à l'horizon annuel ou plus
lointain ; les horizons LT doivent alors s'appuyer sur des priors de forme sans
modifier les moyennes mensuelles imposées par le solveur.

Un seul cycle de vie dépasse le seuil descriptif de 20 jours ouvrés manquants :
le BASE `2023-Q2`, avec 172 jours ouvrés proxy entre le 2021-07-29 et le
2022-03-29. Ce constat demande une explication amont mais ne prouve pas une
inactivité du marché, car le calendrier utilisé est un proxy lundi-vendredi
sans jours fériés EEX.

Enfin, `FactLoadTimestamp` ne constitue pas une preuve de disponibilité chez
le fournisseur : 91,15 % des lignes ont été chargées plus de 30 jours après
leur date de cotation et 65,07 % plus d'un an après. La couverture est donc
descriptive, sans fill-forward et sans utiliser les prix dans ses métriques ;
elle n'autorise ni backtest PIT, ni rolling-origin, ni sélection, ni promotion.

### Audit local DAY/WEEK/WEEKEND

L'audit hors ligne
`pfc_shaping/validation/databricks_eex_short_tenor_audit.py` est figé dans le
bundle adressé par contenu
`b09eb3250df5a3c0616eb169c512319c514ddf540251b405023d9351bd5d8bde`
(manifest SHA-256
`5978fd0383c64138cac0486f891361a07fb4160ee0e6fb22465ffe60ef5bb63d`).
Il consomme la normalisation D212 et sa quarantaine, sans requête Databricks,
sans fill-forward et sans mélange avec les contraintes mensuelles.

Le roast D216 durcit la réconciliation avec la quarantaine : seule une ligne
CH POWER, BASE/PEAK, avec raison autorisée, prix fini, timestamps cohérents et
produit canonique valide peut expliquer un trou. Une ligne d'un autre pays,
d'une autre commodity ou portant une raison inconnue fait échouer l'audit.
Les six Parquets analytiques et le résumé sont identiques à D215 ; seule la
preuve d'implémentation renforcée change.

Les 38 070 cotations vivantes couvrent des horizons strictement courts : DAY
de J-1 à J-13, WEEK de J-3 à J-28 et WEEKEND de J-1 à J-12. PEAK commence le
2023-06-26. L'appariement BASE/PEAK historique total est de 46,99 % uniquement
parce que PEAK est absent avant cette date ; DAY et WEEK sont complets en 2024,
puis les trois familles restent environ à 99-100 % en 2025-2026, avec quelques
absences WEEKEND documentées.

Les contrôles same-vintage trouvent 4 900 bandes WEEK/WEEKEND comparables à
une bande DAY complète et zéro conflit au seuil de 0,01 EUR/MWh. Les 12 170
paires BASE/PEAK se recomposent toutes exactement. Deux OFFPEAK implicites
négatifs concernent le même DAY dominical au lancement de PEAK ; ils sont
économiquement possibles, conservés et ne constituent pas une erreur
algébrique.

La continuité est contrôlée avec le calendrier férié officiel EEX du
2025-06-19, déclaré applicable à toutes les années jusqu'à nouvel avis. Sur 53
diagnostics initiaux, un est expliqué par la ligne WEEK PEAK déjà mise en
quarantaine. Il reste 52 absences candidates sur six dates : 30 identités le
2026-07-08, 18 identités et toute la couche courte absentes le 2026-07-20, puis
une identité chacune le 2020-05-20 et les 2025-09-26, 2025-09-29 et
2025-09-30. Elles ne sont ni remplies ni automatiquement qualifiées de
corruption.

Points à faire confirmer par le data engineer :

- cause de l'absence complète du 2026-07-20 et du trou multi-produit du
  2026-07-08 ;
- statut des quatre absences isolées restantes ;
- conservation future des clés produit/période, dates de cotation et raisons
  de quarantaine ;
- fourniture d'un catalogue de vintages signé avant tout rolling-origin ou
  sélection de modèle.

Ces contrats peuvent devenir des diagnostics de forme jour/semaine/weekend à
court horizon. Ils ne sont ni une vérité horaire/15 minutes, ni une autorité de
niveau mensuel, ni une preuve de supériorité sur OMPEX.

### Contrat dormant de forme court terme

Le contrat
`.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-SHAPE-CONTRACT-V1.json`
et le module LT `pfc_shaping/lt/model/short_tenor_shape_contract.py` définissent
la frontière mathématique d'une future utilisation. Ils ne construisent pas
encore de feature à partir des prix et ne sont reliés ni à l'assembleur ni au
pipeline de production.

Une future feature DAY/WEEK/WEEKEND devra être additive en EUR/MWh, construite
avec une même date de cotation, des bandes complètes et non chevauchantes, et
rester constante au grain maximal `jour local × bloc 08:00-20:00`. Aucune
forme horaire ou quart-horaire ne peut être inventée à l'intérieur de ces
blocs. Les mois locaux doivent être complets, y compris les journées DST à 23
ou 25 heures.

Le module projette ensuite ce signal dans le noyau exact des contraintes CH
BASE/PEAK/OFFPEAK actives. Les moyennes mensuelles du solveur et les
contraintes PEAK déjà acceptées restent donc inchangées à `1e-9` EUR/MWh près.
L'opération est additive afin de rester définie lorsque les prix sont nuls ou
négatifs.

Le bundle de preuve mathématique D219 sélectionné est
`4da441b09c5559d772fd507214b2989489314bd9114a9328752a150f3957450a`
(manifest SHA-256
`b6892ba9b4de7c165ad5f70a01800cdd2a471b51f21fa6b94a7ebbacc97d5bb6`).
Il conserve le contrat D217 et le lie à la normalisation D212 ainsi qu'à
l'audit durci D216. Ses cinq cas couvrent une année horaire, les deux mois DST
en 15 minutes, des contraintes PEAK partielles et un cas BASE-only. Le résidu
maximal des contraintes est `6,78e-15 EUR/MWh` et celui des moyennes
mensuelles `2,99e-15 EUR/MWh`.

Le roast prouve en plus que la projection dépend uniquement de la géométrie
des contraintes, jamais des valeurs numériques des forwards, et que son ajout
à une courbe solver déjà admissible conserve exactement ses contraintes. Le
module reste absent de l'orchestrateur LT. Les cas horaires/15 minutes, DST,
prix négatifs, mois partiels, index irréguliers et variations intrabloc sont
couverts par `14 passed`; la matrice adjacente donne `256 passed, 4 skipped`.

Ce résultat est uniquement une preuve de neutralité mathématique. L'activation
reste interdite tant que les vintages EEX PIT signés, ENTSO-E gouverné, le
rolling-origin préenregistré et un nouveau holdout futur indépendant ne sont
pas disponibles. AFRY et OMPEX ne sont utilisés ni comme entrée, ni comme
cible, ni comme critère de sélection.

### Construction locale des contrastes courts

D220 ajoute le constructeur pur
`pfc_shaping/lt/model/short_tenor_contrasts.py`. Il transforme les cotations
DAY/WEEK/WEEKEND d'une même date en composantes additives séparées : jour dans
la semaine, jour dans le weekend, weekend contre semaine et BASE/PEAK avec
OFFPEAK implicite. Une bande incomplète ou incohérente produit uniquement un
diagnostic ; aucune cotation n'est complétée, reportée ou interpolée.

Le bundle local sélectionné est
`ae7b962ca5f85ff75223ad45fd578c29b49a1d66dfbdd418b941c337dda38147`
(manifest SHA-256
`fd5af41eb51154dfa1a3406feebe830cac7fb11d90603edce266116b3278b390`).
Il contient les mêmes Parquet de contrastes, diagnostics et synthèse que le
bundle descriptif antérieur ; la sélection ci-dessus conserve la preuve issue
du runtime gouverné et son inventaire de tests exact.
Il contient 17 635 composantes acceptées et 95 783 cellules au grain maximal
`jour local × bloc court`, toutes centrées à zéro à mieux que
`8,53e-14 EUR/MWh`. Le maximum d'écart parent/enfants accepté est
`0,00438 EUR/MWh`, sous le seuil de `0,01 EUR/MWh` ; aucun conflit n'est émis.

La couverture n'est pas uniforme : BASE/PEAK accepte 46,99 % des paires
possibles, DAY dans WEEK 10,32 %, DAY dans WEEKEND 98,85 % et WEEKEND contre
WEEKDAY BASE 7,32 %. Les refus correspondent à des composantes absentes à la
même date de cotation et restent absents. Le taux global augmente fortement
après le lancement de PEAK, mais ne constitue toujours pas une preuve PIT.

Une seule composante peut être matérialisée sur des mois Europe/Zurich
complets, à l'heure ou au quart d'heure, puis passée au projecteur D219. Le
roast de 336 projections, incluant DST et semaines à cheval sur deux mois,
reste sous `2,67e-14 EUR/MWh` sur les contraintes et `5,48e-15 EUR/MWh` sur
les moyennes mensuelles. Les composantes ne sont jamais additionnées
automatiquement et le pipeline LT ne les importe pas. Les tests donnent `19
passed` puis `200 passed, 4 skipped` sur la matrice adjacente.

Cette construction reste descriptive et inactive. Elle ne fixe ni amplitude,
ni clipping, ni choix de feature, et n'utilise ni OMPEX, ni AFRY, ni T057.
L'exécution a lu uniquement les Parquet locaux : zéro requête Databricks, zéro
démarrage de Warehouse, zéro appel réseau et zéro écriture distante.

### Frontière de combinaison explicite

D222 ajoute le contrat
`.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-COMBINATION-CONTRACT-V1.json`
et le module
`pfc_shaping/lt/model/short_tenor_combination_contract.py`. Cette couche ne
choisit aucun poids : elle exige une correspondance exacte entre les noms des
composantes et des coefficients fournis, ainsi que trois plafonds explicites
(coefficient, contribution individuelle et signal combiné). Une valeur
absente, booléenne, non finie ou hors plafond provoque un refus ; aucun
clipping silencieux n'est effectué.

Chaque composante est projetée séparément par D219, leur somme pondérée est
calculée, puis la somme repasse par D219. Cette seconde projection doit être
identique à la combinaison linéaire à `1e-9 EUR/MWh` près. Il est interdit de
clipper après la projection, car cela pourrait réécrire les contraintes BASE,
PEAK/OFFPEAK ou les moyennes mensuelles du solveur.

Le design suit les recommandations de Lago et al. (2021,
doi `10.1016/j.apenergy.2021.116983`) sur les comparaisons hors échantillon
rigoureuses contre des baselines fortes, et de Ziel & Weron (2018,
doi `10.1016/j.eneco.2017.12.016`) sur la régularisation et l'intérêt potentiel
des combinaisons lorsqu'aucune structure ne domine toutes les saisons et
heures. Il n'en déduit aucun coefficient : ces valeurs devront être choisies
à l'intérieur de chaque rolling-origin gouverné, avec données d'entraînement
seulement, embargo, mêmes cibles/masques et ablations de groupes de features.

Le bundle de preuve algébrique sélectionné est
`21c557df75260ce162c15af6fbe0c91de4c8a6fcd233564403a04e7b1fc53ad0`
(manifest SHA-256
`c2cddd1ba59b3f0cf77ff16f5debcbf915c46b9373dd6a36e238d144d58bb21c`).
Six cas persistés et un roast aléatoire de 24 cas couvrent 1h/15min, DST,
BASE-only, coefficients positifs/négatifs/nuls et niveaux algébriques négatifs.
Le résidu de linéarité maximal est nul ; les résidus de contrainte et mensuels
restent respectivement sous `1,45e-14` et `4,82e-15 EUR/MWh`. Les tests donnent
`33 passed`, puis `209 passed, 4 skipped` sur la matrice adjacente. Le validateur
de politique reste strictement outcome-blind : les poids, plafonds et grilles
numériques sont nuls et toute exécution empirique demeure interdite.

D222 reste une preuve mathématique inactive. Les identifiants de reçus sont
liés mais non authentifiés ; ils ne donnent aucun droit d'entraînement ou
d'assemblage. Aucun prix fournisseur, poids ou plafond sélectionné n'est
persisté dans la preuve. Zéro requête Databricks, démarrage de Warehouse, appel
réseau ou écriture distante n'a été effectué.

### Supplément outcome-blind pour la future sélection

D223 durcit D222 en liant la frontière de combinaison au bundle D220 gouverné
`ae7b962ca5f85ff75223ad45fd578c29b49a1d66dfbdd418b941c337dda38147`
et au successor core CH v3. Le contrat obtient le statut
`PASS_LOCAL_STRUCTURE_ONLY_NO_EMPIRICAL_EXECUTION` et le content ID
`00a2b2087589d14ef1330cfa0de109fe9dfcce81436e0a0265fb96067d10fbb6`.
Il ne constitue pas encore un préenregistrement exécutable ou signé.

Deux familles candidates sont désormais fermées avant résultat : un modèle
linéaire/GAM robuste régularisé et un gradient boosting monotone. Elles devront
être comparées au seasonal-naive contraint par le marché et à l'incumbent CH LT
gelé, sur les mêmes origines, cibles et masques. La sélection reste imbriquée
dans chaque origine ; la perte primaire est la MAE de forme intra-mensuelle au
régime natif. Biais, RMSE, erreur médiane, rampes, queues, prix négatifs,
peak/offpeak, weekday/weekend/jours fériés, saisons, DST, horizon et stabilité
de vintage restent des diagnostics obligatoires.

Le nombre minimal d'origines, le nombre effectif de clusters, l'embargo, la
grille et le cap des coefficients, les caps de contribution/signal, les marges
de supériorité/non-infériorité, les effets de puissance et le nombre requis
d'origines restent explicitement `null`. Ils ne pourront être fixés qu'à partir
d'un plan de puissance CH gelé avant les résultats. Les comparaisons communes
par origine utilisent un bootstrap studentisé à blocs adapté au chevauchement
et un contrôle Holm de l'erreur familiale ; un model confidence set reste un
diagnostic et ne remplace jamais les gates confirmatoires.

Ce design complète les références déjà retenues par D222 avec les tests de
capacité prédictive conditionnelle de
[Giacomini et White (2006)](https://doi.org/10.1111/j.1468-0262.2006.00718.x),
le contrôle des comparaisons multiples dépendantes de
[Romano et Wolf (2005)](https://doi.org/10.1111/j.1468-0262.2005.00615.x)
et l'incertitude de sélection de
[Hansen, Lunde et Nason (2011)](https://doi.org/10.3982/ECTA5771).

Le validateur refuse le bundle D220 descriptif non sélectionné, un nombre
d'origines ou une marge prématurés, une nouvelle famille ou métrique, une
ouverture du holdout, toute utilisation d'OMPEX pour entraîner/sélectionner et
toute autorité modèle ou production. Le combiner vérifie aussi le cap du signal
après la projection finale D219, et pas seulement avant celle-ci.

Le nouveau bundle de preuve sans valeur de marché est
`c06dfcf5fbda16cd5bab04005d4581e35216116c30c6c9df6cef94f2d949f67b`.
Les six cas 1h/15min/DST conservent un résidu de linéarité nul ; les résidus de
contrainte et mensuels restent sous `8,32e-15` et `2,70e-15 EUR/MWh`. Les tests
ciblés donnent `33 passed`, la matrice D219/D220/OMPEX/ENTSO-E/LT donne `127
passed, 1 skipped`, et Ruff passe. Aucun appel Databricks, démarrage de
Warehouse, appel réseau ou écriture distante n'a été effectué.

OMPEX reste fermé jusqu'au gel complet de la candidate et ne servira ensuite
que de benchmark contre la même vérité indépendante. AFRY reste descriptif,
T057 scellé, les substituts ENTSO-E legacy/synthétiques interdits et le solveur
mensuel l'unique autorité de niveau.

### Réconciliation de la chaîne de preuves courtes

D225 supersède uniquement les identités d'artefacts retenues par D220-D223.
Les bundles de contrastes `ae7b962c...`, `7a4c2a60...` et `309b9f07...`
contiennent exactement les mêmes Parquet de contrastes et diagnostics ainsi que
la même synthèse. Ils diffèrent seulement par les octets de tests liés dans
leur manifest. Aucun prix ou score de performance n'a donc servi à les
départager.

Le bundle courant est
`309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5`
(manifest SHA-256
`57b571bef311cf22955edc415e650f55671ad2f243141d4f334ccac425d8bb0f`).
Il lie le contrat, le module et les tests actuellement présents, et deux
replays avec le runtime local gouverné ont retourné le même content ID.

La preuve de combinaison courante devient
`122afd7292549b00a946a2893487d344e459c420a8bbdf99c11714c7308a8cfd`.
Elle lie exactement `309b9f07...` et la politique outcome-blind
`189e7ab9d460daefbaa689f1dc8ae4c3bb269f7b5bcb5eae95e77b6d0d4d3e43`.
Les preuves `21c557df...` et `c06dfcf...` restent conservées mais sont
supersédées, car elles pointent vers les anciennes identités de bundle.

Le registre
`EEX-CH-SHORT-TENOR-EVIDENCE-SELECTION-V1.json` possède le content ID
`792149a7d46834b17507ea4ec6a15dbfeb2ab9233b058b9783673a58a57c26aa`.
Son validateur vérifie les six manifests exacts, l'identité des payloads
scientifiques, les liaisons amont et l'absence de toute autorité empirique. Le
bundle de preuve local est `ca099abf...`; les tests donnent `66 passed`, puis
`244 passed, 4 skipped` sur la matrice adjacente.

Cette correction ne modifie aucune formule, composante, amplitude ou courbe.
Elle ne prouve ni PIT, ni qualité prédictive et n'autorise aucun entraînement.
L'exécution est restée strictement locale : zéro requête Databricks, zéro
démarrage de Warehouse, zéro appel réseau et zéro écriture distante.

### Reçus structurels de training et sélection short-tenor

Le contrat `EEX-CH-SHORT-TENOR-RECEIPT-CONTRACT-V1.json` ferme maintenant la
frontière recommandée par D223/D225. Pour chaque origin externe futur, le reçu
de training lie le catalogue PIT EEX, le catalogue PIT ENTSO-E, l'inventaire
origin/target, la grille de candidats, les folds internes, le runtime et les
preuves D219-D225. Chaque fold interne possède ses propres hashes d'entrée,
de ligne d'inventaire et de cutoff PIT EEX/ENTSO-E ; l'origin externe est
interdit dans les ensembles internes de training et de validation.

Le reçu de sélection doit reprendre exactement le reçu de training, l'origin,
la grille, les folds et l'engagement hashé des pertes internes. Le candidat
retenu doit appartenir à l'allowlist gelée, et la règle reste
`ONE_STANDARD_ERROR_THEN_LOWEST_LATTICE_RANK_WITH_FIXED_FAMILY_TIE_ORDER`.
Les accès à la vérité externe, OMPEX, AFRY numérique, T057, au holdout futur
ou aux données postérieures à l'origin restent explicitement faux.

Le profil suit l'[in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/blob/main/spec/README.md)
avec `Statement/v1` et enveloppe DSSE, et reprend le modèle de provenance
[SLSA v1.2](https://slsa.dev/spec/v1.2/) : outputs nommés et hashés,
dépendances résolues hashées, paramètres externes, identité du builder,
timestamps d'exécution et byproducts liés. Le profil local est volontairement
plus strict que le parseur générique SLSA : champs inconnus, clés JSON
dupliquées, chemins ambigus et noms non relatifs POSIX sont rejetés.

Le roast ciblé compte 55 tests passants et la matrice LT adjacente
`329 passed, 4 skipped`. La preuve locale reproductible porte le content ID
`398b6ff48128a577f06a0106473a3cf71cdc6d411e3d1e877270d433cc6091cc`.
Son manifeste enregistre explicitement zéro requête Databricks, zéro démarrage
de Warehouse, zéro appel réseau et zéro écriture distante.

Il ne s'agit toujours que d'un schéma local non authentifié : aucune signature
indépendante, aucun temps de confiance, aucun catalogue PIT gouverné et aucun
training réel ne sont admis. Toutes les autorités de training, sélection,
modèle, assemblage, promotion et production restent donc fausses. Aucun prix,
coefficient, cap, perte ou marge numérique n'est émis par ces reçus
structurels.

### Replay signé local des reçus short-tenor

D228 ajoute un replay signé, strictement local et sans prix, au-dessus de D227.
Le reçu exact est encapsulé dans une
[Statement in-toto v1](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)
avec le prédicat officiel [SLSA provenance v1](https://slsa.dev/provenance/v1),
puis signé dans une enveloppe
[DSSE](https://github.com/in-toto/attestation/blob/main/spec/v1/envelope.md).
Le subject reprend l'output exact du reçu ; les paramètres externes embarquent
le reçu canonique ; les dépendances sont rejouées selon le rôle training ou
sélection ; builder, invocation, heures et byproducts doivent correspondre.

Sept fichiers sont fournis par chemin absolu et hash attendu : deux enveloppes
de reçu, trois clés publiques PEM et deux observations temporelles. Le training
est signé par la clé d'exécution, la sélection par la clé de gouvernance, et les
deux observations par une troisième clé de temps. Les chemins doivent être des
fichiers mono-liens distincts, les PEM et JSON canoniques, les clés séparées et
les deux attestations d'exécution différentes. Toute substitution de reçu,
dépendance, byproduct, clé, heure ou liaison training-sélection échoue fermée.

Cette preuve vérifie uniquement l'intégrité cryptographique locale et la
cohérence des liens. Une clé fournie par l'appelant ne prouve ni son propriétaire
organisationnel, ni son cycle de vie ; l'observation signée ne devient pas un
temps indépendant de confiance. Les catalogues PIT EEX/ENTSO-E et leurs
artefacts ne sont pas ouverts. Training, sélection, entrée modèle, assemblage,
promotion et production restent donc faux.

Le roast ciblé donne `27 passed` et la matrice adjacente élargie
`455 passed, 2 skipped`; Ruff passe sur le contrat, le validateur, les tests et
le matérialiseur. La preuve synthétique déterministe est
`eed94d79109a5f196f49cf2bb11950b79889c8384cc82c1ef70a33a69cdebc5e`.
Elle persiste zéro prix/coefficient/cap/perte/marge et enregistre zéro requête
Databricks, zéro démarrage de Warehouse, zéro appel réseau et zéro écriture
distante.

### Registre externe de confiance et cycle de vie des clés

D229 ferme la lacune locale laissée par D228 avec un registre sans valeur de
marché, signé par un quorum Ed25519 `2-sur-3`. La chaîne commence à la séquence
1, lie chaque payload précédent, interdit les sauts de version et le rollback,
expire au plus après 31 jours et exige un checkpoint de tête fourni par
l'appelant. Les trois clés de gouvernance restent extérieures aux clés de rôle.

Sept rôles sont obligatoires et cryptographiquement distincts : acquisition et
temps source EEX, acquisition et temps source ENTSO-E, exécution du training,
gouvernance de la sélection et temps de confiance. Chaque clé possède une
validité finie et un historique append-only `ISSUED`, `ACTIVATED`, puis
éventuellement `RETIRED`, `REVOKED` ou `COMPROMISED`. Les rotations planifiées
sont continues et sans chevauchement. Après révocation ou compromission, un
intervalle sans clé est admis mais échoue fermé ; aucun chevauchement avec la
clé de remplacement n'est accepté. L'API de résolution historique exige que
l'appelant affirme avoir vérifié indépendamment le temps de signature, mais
D229 ne vérifie pas cette autorité externe et ne la revendique jamais.

Le profil reprend les principes de cycle de vie de
[NIST SP 800-57 Part 1 Rev. 5](https://doi.org/10.6028/NIST.SP.800-57pt1r5),
les notions de quorum/version/expiration de
[TUF 1.0.33](https://theupdateframework.github.io/specification/v1.0.33/),
et adapte les sémantiques d'invalidité de
[RFC 5280](https://www.rfc-editor.org/rfc/rfc5280). Il ne revendique toutefois
ni conformité TUF, ni PKI X.509, ni identité organisationnelle externe : les
racines, propriétaires et temps restent fournis localement par l'appelant.

Le roast ciblé donne `34 passed`; la matrice adjacente purement locale étendue
donne `466 passed, 1 skipped`. Deux matérialisations identiques produisent la preuve
`72483a8aee28241db716a07355ed27ac4065b049d74e42f1c2224809821cbf61`.
Elle contient zéro ancre externe réelle, zéro clé privée, zéro prix et enregistre
zéro requête Databricks, démarrage de Warehouse, appel réseau, accès `H:` ou
écriture distante. Training, sélection, entrée modèle, assemblage, promotion et
production restent strictement faux.

### Liaison des signataires D228 au registre D229

D230 compose les deux contrôles précédents sans ouvrir de valeur de marché. Les
trois fichiers de clé publique utilisés pour vérifier D228 doivent être exactement
les mêmes chemins absolus et les mêmes octets ancrés dans D229 pour les rôles
`training_execution`, `selection_governance` et `trusted_time`. Le registre est
rejoué deux fois afin que la clé de temps soit résolue aux deux
`observed_at_utc`; les clés d'exécution et de gouvernance sont résolues au temps
d'observation associé à leur reçu. La tête du registre doit être publiée après
ces temps déclarés, rester valide au temps de référence de l'appelant et ne pas
dépasser 31 jours de retard local.

Le roast a toutefois confirmé une limite qu'il serait dangereux de masquer :
`observed_at_utc` date l'observation de l'attestation d'exécution, pas la création
cryptographique de la signature DSSE. C'est une assertion signée dans le payload,
pas un jeton d'horodatage externe. D230 prouve donc la liaison locale de l'identité
de clé et sa résolution au temps déclaré, mais laisse explicitement faux le temps
réel de signature, la résistance au backdating et le temps indépendant de
confiance. Les clés source EEX/ENTSO-E restent également non liées à des reçus
d'acquisition nommés.

Le validateur recontrôle aussi les identités de contrat, les flags d'intégrité,
les autorités fausses et tous les compteurs d'accès nuls renvoyés par D228 et
D229 : un sous-validateur monkeypatché ne peut donc pas faire passer une autorité
ou un accès caché dans une réponse localement verte.

Le roast ciblé donne `25 passed`; la matrice locale étendue
EEX/PIT/mensuel/ENTSO-E/OMPEX/LT donne `491 passed, 1 skipped`. Ruff passe et
deux matérialisations identiques produisent
la preuve
`db365ca3045f989cd7257594f37ccec0c8b306475e9f4149b525f1732fff2ad3`.
Elle enregistre zéro requête Databricks, démarrage de Warehouse, appel réseau,
accès `H:` ou écriture distante et ne persiste aucune clé privée ni valeur de
marché. Aucun droit de training, sélection, entrée modèle, promotion ou production
n'en découle.

### Plan d'acquisition Databricks sans exécution

D231 gèle la prochaine étape sans lancer Databricks. Pour EEX, le plan réutilise
la capture locale existante de `prd.gold` (82 552 lignes, `EUR/MWh`) : aucune
nouvelle requête n'est nécessaire. Pour ENTSO-E, il prépare seulement un
`SELECT` explicite sur `dev.information_schema.columns`, limité à 1 024 lignes,
afin d'identifier les colonnes physiques des trois tables `dev.gold`. Cette
requête n'a pas été exécutée et son exécution future ne pourra pas être qualifiée
de gratuite : elle exige une nouvelle autorisation explicite et un budget d'une
seule instruction avec timeout de 60 secondes.

Avant cette admission du schéma, le budget de requête de données ENTSO-E reste
à zéro et aucun SQL de valeurs n'est généré. Après admission, les contrôles
obligatoires couvriront complétude, unicité, validité, cohérence, intégrité
référentielle, fraîcheur, volumes/granularité et absence de fuite
point-in-time via `as_of_utc`. Une capture locale immuable, ses nombres de
lignes, identifiants de requête et SHA-256 seront requis avant tout usage modèle.

Le roast ciblé donne `25 passed`; la matrice locale adjacente donne
`127 passed, 1 skipped`. Ruff passe et deux matérialisations identiques produisent
la preuve
`127506f29101c98738d4fc876fb428295722a8253c4ee290e820b55ef67d3a83`.
Elle enregistre zéro requête Databricks, démarrage de Warehouse, appel réseau,
accès `H:` ou écriture distante. Aucun droit de training, sélection, entrée
modèle, assemblage, promotion ou production n'en découle.

### Admission locale du futur résultat de métadonnées ENTSO-E

D232 ajoute le sas qui manquait entre la requête préparée et toute génération
de SQL de données. Le résultat futur devra être un CSV canonique lié par
SHA-256 à un reçu exact : identifiant d'instruction UUID, état `SUCCEEDED`,
heure UTC, nombre de lignes, absence de troncature, lecture de métadonnées seule
et zéro écriture distante. Le reçu doit être contrôlé dans l'heure. Un résultat
de 1 024 lignes est rejeté, car il pourrait avoir atteint la limite.

Le validateur impose les trois tables exactes, un ordre déterministe, des
positions ordinales positives et contiguës, des noms de colonnes uniques sans
ambiguïté de casse, des types simples et `is_nullable` dans `YES`/`NO`. Ces
invariants suivent les clés et définitions officielles de
[Databricks `INFORMATION_SCHEMA.COLUMNS`](https://docs.databricks.com/aws/en/sql/language-manual/information-schema/columns).
La requête D231 filtre maintenant explicitement `table_catalog = 'dev'`, le
schéma et les trois tables. Databricks précise que le pushdown de `LIMIT` n'est
pas pris en charge pour l'information schema : la limite borne le résultat mais
ne garantit pas un travail ou un coût nul. Les filtres sélectifs et
l'autorisation séparée restent donc les vrais garde-fous de coût
([documentation Information Schema](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-information-schema)).

Ce batch ne contient qu'une fixture synthétique sans valeur destinée à roaster
le sas. Il n'existe encore aucun reçu réel, aucune ligne de métadonnées réelle,
aucun mapping de colonnes physiques et aucun SQL de données. Le manque de
capture réelle reste un blocage critique pour l'usage analytique ou modèle ; le
reçu local ne constituerait par ailleurs pas une preuve de temps externe.

Le roast ciblé donne `40 passed`; la matrice locale adjacente donne
`167 passed, 1 skipped`. Ruff passe et deux matérialisations identiques produisent
la preuve
`9a270975187e9ff334d80afba308ad0021f0df97a4269348943a62d655bc4147`.
Elle enregistre zéro requête Databricks, démarrage de Warehouse, appel réseau,
accès `H:` ou écriture distante. Le vrai schéma Databricks et toutes les
autorités modèle/production restent faux.

### Paquet gouverné EEX/ENTSO-E avant acquisition

D233 formalise le livrable local attendu du data engineer sans lancer aucune
requête. EEX est repris depuis l'artefact local exact de D231 : aucune nouvelle
instruction EEX. Après admission d'un vrai schéma par D232, ENTSO-E devra
fournir trois artefacts normalisés distincts : dimension des séries, valeurs
latest et vintages. Leurs grains, champs, unités, résolutions natives, lignage,
hashes, nombres de lignes et bornes temporelles sont imposés par contrat.

Le plafond futur reste inactif sans nouvelle autorisation explicite et chiffrée :
une capture, un démarrage de Warehouse, trois `SELECT` en lecture seule,
300 secondes par instruction, 900 secondes au total, zéro retry et 64 GiB
maximum en local. Un dépassement arrête le lot ; aucune troncature silencieuse
n'est admise. Ce plafond borne les ressources, mais ne constitue pas une
promesse de coût nul. La voie préférée reste un export immuable publié par le
data engineer sous `build/governed-source-intake/`, ce qui maintient le coût
Databricks de la PFC à zéro.

Quatre enveloppes DSSE/in-toto séparées sont requises pour les acquisitions EEX
et ENTSO-E et leurs observations temporelles. D233 minimise l'opération de temps
externe en plaçant les quatre hashes dans une racine de Merkle RFC 9162, avec
une preuve d'inclusion par enveloppe, puis en horodatant cette racine via RFC
3161 ou un journal de transparence gouverné. RFC 3161 lie l'empreinte demandée
et fournit notamment `genTime`; RFC 9162 définit les preuves d'inclusion et de
cohérence d'un journal append-only
([RFC 3161](https://www.rfc-editor.org/rfc/rfc3161),
[RFC 9162](https://www.rfc-editor.org/rfc/rfc9162)). Cette preuve établit une
existence au plus tard à l'heure externe, pas l'heure exacte de signature, et
ne rend pas rétroactivement PIT les lignes historiques antérieures au challenge.

Le roast D233 ciblé donne `44 passed`; la matrice combinée D231+D232+D233 donne
`109 passed` et la matrice élargie acquisition/PIT/ENTSO-E donne
`418 passed, 3 skipped`. Ruff passe. Deux matérialisations identiques produisent la preuve
`314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53`.
Elle enregistre zéro requête ou écriture Databricks, démarrage de Warehouse,
appel réseau, accès `H:`, écriture distante ou ligne de prix ouverte. Les vraies
enveloppes, le temps externe, les valeurs ENTSO-E, le PIT et toutes les autorités
modèle/production restent absents.

### Compilation hors ligne du mapping physique ENTSO-E

D234 compose explicitement le paquet gouverné D233, l'admission de métadonnées
D232 et le contrat d'intake ENTSO-E. Un manifeste de mapping doit lier le hash
du CSV de métadonnées, son empreinte de schéma et son identifiant d'instruction,
puis associer sans ambiguïté chaque champ normalisé à une colonne physique
réellement admise. Le mapping est injectif par table : une colonne physique ne
peut pas remplir deux rôles normalisés.

Le compilateur vérifie les neuf champs de dimension, les sept champs latest et
les sept champs vintages. Il contrôle les classes de types et impose l'égalité
des types partagés entre tables pour `series_id`, `target_ts_utc`, `value`,
`quality_flag`, `revision_number` et `meta_load_timestamp_utc`. Aucun cast,
forward-fill, upsampling ou statut PIT de la table latest n'est autorisé. Les
sémantiques UTC restent une assertion du propriétaire du mapping, pas une preuve
déduite du seul type SQL.

Trois templates Databricks SQL sont compilés en mémoire avec projections et
alias explicites, identifiants qualifiés et délimités, ordre déterministe et
limites fixes. Les facts utilisent une fenêtre cible semi-ouverte
`:start_utc` / `:end_utc`; les vintages ajoutent
`:as_of_cutoff_utc`. Databricks documente que les paramètres nommés sont pris en
charge par la Statement Execution API et séparent les valeurs de la structure
SQL, ce qui réduit le risque d'injection
([parameter markers](https://docs.databricks.com/gcp/en/sql/language-manual/sql-ref-parameter-marker)).
La clause `LIMIT` borne les lignes retournées et doit être combinée à
`ORDER BY` pour un résultat déterministe
([LIMIT](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-qry-select-limit));
elle ne constitue toujours pas une garantie de coût. Une future exécution devra
valider séparément les paramètres, rester sous 31 jours et rejeter tout résultat
qui atteint la limite.

Le roast utilise uniquement un schéma et un mapping synthétiques sans valeur.
Les textes SQL synthétiques ne sont pas persistés dans la preuve, seulement
leurs SHA-256, paramètres et plafonds. Le roast ciblé donne `47 passed`; la
matrice D231-D234 donne `156 passed` et la matrice locale élargie
`465 passed, 3 skipped`. Deux matérialisations identiques produisent la preuve
`bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766`.
Elle lie aussi la preuve D233 et enregistre zéro requête Databricks, démarrage de
Warehouse, appel réseau, accès `H:` ou écriture distante. Le vrai mapping, les
valeurs, le PIT et toutes les autorités modèle/production restent absents.

### Vérification locale du batch d'horodatage externe

D235 implémente la partie cryptographique locale qui manquait au paquet D233,
après liaison du compilateur physique D234. Les quatre rôles source sont placés
dans un ordre fixe. Chaque feuille contient le même identifiant de capture et
le même challenge, plus le rôle et le SHA-256 de l'enveloppe DSSE. Le hash de
feuille est `SHA256(0x00 || payload)` et celui d'un nœud
`SHA256(0x01 || gauche || droite)`. Cette séparation de domaine et la règle de
découpage à la plus grande puissance de deux suivent exactement la définition
du Merkle Tree Hash de [RFC 9162, section 2.1.1](https://www.rfc-editor.org/rfc/rfc9162#section-2.1.1).

Le vérificateur génère et contrôle une preuve d'inclusion par enveloppe selon
[RFC 9162, section 2.1.3](https://www.rfc-editor.org/rfc/rfc9162#section-2.1.3).
La direction gauche/droite est dérivée de l'index et de la taille de l'arbre ;
elle n'est jamais acceptée du caller. Le roast couvre aussi tous les arbres de
1 à 16 feuilles afin de vérifier les formes équilibrées et non équilibrées,
même si le contrat gouverné en exige exactement quatre.

Pour RFC 3161, D235 gèle le futur `TimeStampReq` : SHA-256, racine Merkle exacte
dans `messageImprint`, nonce positif, politique TSA explicite et demande du
certificat. Le futur reçu comporte 46 champs et devra être obtenu par parsing et
rejeu indépendants des bytes DER/CMS et `TSTInfo`, avec vérification de
l'empreinte, nonce, politique, statut, chaîne, EKU time-stamping exclusive et
critique, validité, révocation, racine pré-épinglée et fenêtre temporelle.
RFC 3161 impose notamment la correspondance de `messageImprint` et, si le nonce
était demandé, sa présence dans la réponse
([RFC 3161, sections 2.4.1-2.4.2](https://www.rfc-editor.org/rfc/rfc3161#section-2.4.1)).

Le batch actuel reste entièrement synthétique : aucun DER, aucune requête TSA,
aucune politique réelle, aucun token et aucune enveloppe DSSE réelle. Le roast
ciblé donne `50 passed`; la matrice adjacente D229/D231-D235 donne
`265 passed`; Ruff passe. Deux matérialisations identiques produisent la preuve
`93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1`.
Elle enregistre zéro connexion/instruction Databricks, démarrage de Warehouse,
appel réseau/TSA, accès `H:`, écriture distante ou ligne de prix ouverte. Le
temps externe indépendant et toutes les autorités modèle/production restent
faux.

### Paramètres et reçu ENTSO-E bornés sans exécution

D236 ajoute le dernier contrôle local avant toute éventuelle extraction. Une
proposition doit fixer une fenêtre UTC semi-ouverte de 1 à 31 jours, identique
pour les tables latest et vintage, avec un cutoff `as_of` qui ne dépasse pas la
date de photographie. Elle doit aussi déclarer une estimation de coût en
décimal fixe, un plafond dur au moins égal, la taille du Warehouse, l'arrêt
automatique à cinq minutes, un seul démarrage maximal futur, trois lectures,
zéro retry, 300 secondes par requête, 900 secondes pour le lot et les plafonds
de lignes/octets par rôle. Ces champs sont des barrières de refus, pas une
autorisation ni une garantie de coût.

Le reçu testé est uniquement synthétique. Le validateur lie chaque rôle au
hash du template D234 et au hash exact de ses paramètres, impose l'ordre des
trois lectures, réconcilie durées et octets, et refuse une limite atteinte, une
troncature, un dépassement, un retry, une écriture distante ou un chemin hors
de `build/governed-source-intake/`. Il n'ouvre aucun artefact ni aucune valeur
ENTSO-E et ne peut pas admettre un reçu réel.

Le roast ciblé donne `31 passed`; la matrice locale inclusive D231-D236 donne
`237 passed`; Ruff passe. Deux matérialisations identiques produisent la preuve
`772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247`.
Tous les compteurs connexion/instruction Databricks, démarrage de Warehouse,
appel réseau, accès `H:` et écriture distante sont nuls. Une future exécution
réelle nécessitera une autorisation utilisateur distincte, liée au contenu
exact de la proposition et à son plafond financier explicite.

### DER canonique de la requête RFC 3161

D237 ferme l'ambiguïté laissée par le schéma D235. La donnée effectivement
horodatée est le préimage de 65 octets du nœud racine RFC 9162 :
`0x01 || hash_sous_arbre_gauche || hash_sous_arbre_droit`. Son SHA-256 est
exactement la racine Merkle D235 placée dans `messageImprint`. On ne prétend
donc ni que la racine se hache elle-même, ni qu'un double hash implicite existe.

Le générateur produit un `TimeStampReq` DER de 89 octets : version 1, OID
SHA-256, paramètres absents, racine de 32 octets, politique synthétique, nonce
positif et `certReq=TRUE`. Le rejeu refuse longueurs BER indéfinies ou non
minimales, INTEGER négatif/non minimal, OID non canonique, paramètres SHA-256
présents, champ manquant, extension et octet final. La règle de génération
avec paramètres absents suit
[RFC 5754, section 2](https://www.rfc-editor.org/rfc/rfc5754#section-2).
Cette même norme demande cependant à un parseur général d'accepter aussi
`NULL` : D237 ne revendique donc que le rejeu exact de ses propres bytes, pas
un parseur CMS/TSP interopérable complet.

Le préflight de dépendances reste `NO_GO` pour un vrai token.
`cryptography==47.0.0` fournit les primitives SHA-256, X.509 et de signature,
mais l'extraction des certificats PKCS#7 n'est pas la vérification de
`SignedData`, `SignerInfo`, `TSTInfo`, `SigningCertificate(V2)`, du chemin TSA
et de la révocation à `genTime`. Les candidats `asn1crypto==1.5.1` et
`pyHanko==0.35.2` ne sont ni installés ni admis. Les exigences de signature CMS
restent celles de
[RFC 5652, sections 5.3-5.6](https://www.rfc-editor.org/rfc/rfc5652#section-5.3),
et `SigningCertificateV2` celle de
[RFC 5816](https://www.rfc-editor.org/rfc/rfc5816).

Le roast ciblé donne `71 passed`, la matrice adjacente D229/D231-D237
`367 passed`, la matrice du garde d'exécution `150 passed`, et Ruff passe.
Deux matérialisations finales donnent la preuve
`53e2222392f71541d28e05e1dfc912361c02003b4278f2770109827319d4e9c1`.
Les dix compteurs Databricks, Warehouse, réseau, TSA, `H:`, écriture distante,
enveloppe/token réels et valeurs marché sont nuls. La seule autorité vraie est
le vecteur DER synthétique ; aucune requête réelle n'est autorisée.

---

## 1. Décision sources

Deux fichiers Excel maîtres maintenus par le desk FMV dans
`H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\` :

| Fichier | Couverture marché | Profondeur historique | Rôle |
|---|---|---|---|
| `Price_Report_EEX_Yearly.xlsx` | DE, CH, AT, IT, FR | snapshot courant + ~quelques mois | **source as-of multi-marché** — alimente la calibration PFC du jour |
| `Price_Report_EEX_CH_DE_Historique2019.xlsx` | CH, DE | depuis 2019 | **profondeur historique** — alimente fits saisonniers, peak ratios, structural break, backtest |

Le fichier legacy `Price_Report_EEX.xlsx` (codes `Y01_2027_BASE`) reste lisible en
mode rétro-compatible (déploiements existants), mais n'est plus la source canonique.

### Politique seasonal/peak ratios par marché

`ContractCascader.fit_seasonal_ratios()` et `fit_peak_ratios()` sont alimentés
par le **spot history** (EPEX Swiss / EEX DE) et non par les forwards. La
politique reste donc :

| Marché | Source spot pour fits | Politique de fallback |
|---|---|---|
| CH | EPEX Swiss (energy-charts) | — |
| DE | EPEX DE-LU (energy-charts/SMARD) | — |
| AT | EPEX AT (energy-charts si dispo) | fallback DE-LU (zone fortement couplée historiquement) |
| FR | EPEX FR (energy-charts) | fallback DE-LU (basis stable depuis CRE 2024) |
| IT | EPEX IT_NORD (GME via energy-charts si dispo) | fallback ratios maison à figer après Phase 2 |

Décision : **Phase 0 ne touche pas aux fits saisonniers**. La couverture spot
multi-marché est traitée en Phase 3 (activation production) avec des tests de
fallback explicites.

---

## 2. Contrat parser (entrée Phase 1)

### Format `Yearly` (nouveau, observé sur screenshot 2026-05-05)

Onglets : `DE`, `CH`, `AT`, `IT`, `FR` (un par marché).

Layout par feuille :

| Ligne | Colonne A | Colonnes B+ |
|---|---|---|
| 1 | _(vide)_ | ISIN EEX, ex. `DE000EEX0D8L1` |
| 2 | _(vide)_ | Libellé FR, ex. `Fév 27 BASE`, `Sep 27 BASE`, `Cal 28 PEAK` |
| 3 | `Date` | _(vide)_ |
| 4+ | `01.05.2024` (DD.MM.YYYY) | prix EUR/MWh, `0` si non quoté |

**Mapping libellé FR → code interne** (à implémenter en Phase 1) :

```
Mois FR → numéro :
{Jan:1, Fév:2, Mar:3, Avr:4, Mai:5, Juin:6,
 Juil:7, Aoû:8, Sep:9, Oct:10, Nov:11, Déc:12}

Trimestre FR → numéro :
{T1:1, T2:2, T3:3, T4:4}     (à confirmer Phase 1 avec snapshot réel)

Calendar :
"Cal 28 BASE" → "2028"
"Cal 28 PEAK" → "2028-Peak"

Trimestre :
"T2 27 BASE" → "2027-Q2"     (notation supposée — à vérifier)

Mois :
"Fév 27 BASE" → "2027-02"
"Fév 27 PEAK" → "2027-02-Peak"
```

⚠️ **Action Phase 1** : confirmer la notation des trimestres et la convention
d'année (`27` = `2027`, jamais `1927`) sur snapshot réel de la feuille DE.

### Format `Historique2019` (à confirmer Phase 1)

Hypothèse de travail : même layout que `Yearly` mais avec uniquement DE et CH
en onglets, et historique de marks remontant à 2019. À vérifier dès accès au
fichier.

### Format legacy (existant, à préserver)

`pfc_shaping/data/ingest_forwards.py:39-40` :
```
_EEX_PRODUCT_PATTERN = ^(Y01|Q\d{2}|M\d{2})_(\d{4})_(BASE|PEAK)$
```

Layout legacy :
- Ligne 1 = code produit `Y01_2027_BASE` directement (pas d'ISIN)
- Ligne 4+ = marks

### Sniff de format (logique Phase 1)

```
Lire ligne 1 colonnes B+ d'une feuille.
Si > 50% des cellules matchent regex ^[A-Z]{2}[0-9A-Z]{10}$  → ISIN → format `yearly`
Sinon, si > 50% matchent _EEX_PRODUCT_PATTERN              → format `legacy`
Sinon                                                        → ValueError explicite
```

En mode `yearly`, on lit la **ligne 2** comme source du libellé produit (pas
la ligne 1). L'ISIN est conservé comme clé secondaire pour traçabilité.

---

## 3. Inventaire à compléter sur poste FMV (action utilisateur)

Cette phase ne peut pas être validée à 100% depuis l'environnement Linux du
repo : il faut un sniff réel des deux fichiers depuis un poste avec accès
au lecteur `H:\` ou au chemin UNC `\\fmvfs1\data\Energy\…`.

**Script de sniff** (à lancer côté FMV avec `ppa_env` Python) :

```powershell
$env:PATH='C:\Users\jbattaglia\.conda\ppa_env;C:\Users\jbattaglia\.conda\ppa_env\Library\bin;C:\Users\jbattaglia\.conda\ppa_env\Scripts;' + $env:PATH
C:\Users\jbattaglia\.conda\ppa_env\python.exe scripts/phase0_sniff_forwards.py
```

Le script :
1. Ouvre les deux fichiers Excel (Yearly + Historique2019).
2. Pour chaque feuille, dump les 3 premières lignes × 10 premières colonnes en
   JSON Lines vers `docs/research/phase0_forwards_snapshot.jsonl`.
3. Détecte automatiquement le format (sniff ISIN vs legacy).
4. Vérifie présence des onglets attendus (`DE/CH/AT/IT/FR` pour Yearly,
   `DE/CH` pour Historique2019).
5. Logge les couvertures de dates par feuille.

Ce snapshot **fixe le contrat de tests unitaires** de la Phase 1.

---

## 4. Risques identifiés

| Risque | Impact | Mitigation |
|---|---|---|
| Notation trimestre non standard (`T1` vs `Q1` vs `Mars 27 BASE Q1`) | parser Phase 1 partiel | snapshot Phase 0 obligatoire avant codage |
| Onglets renommés par le desk (`DE-LU` au lieu de `DE`) | crash production | sniff insensible à la casse + alias `{"DE-LU":"DE", "ITN":"IT"}` |
| Cellules `0` = non-quoté vs vrai `0` €/MWh (prix négatifs prompt) | filtrage agressif perd info | Phase 1 ne filtre que `NaN` et `< -500`, pas le strict `> 0` actuel |
| Year-2-digit ambigu (`27` = 1927 ou 2027) | mapping faux Y+10 | clamp `year < 2000 → year + 2000` + assertion `2020 ≤ year ≤ 2040` |
| ISIN non présent en historique 2019 (codes EEX changés) | parser yearly inopérant sur Historique | accepter le mode `legacy` aussi sur ce fichier |

---

## 5. Définition of Done — Phase 0

- [x] Document de cadrage `docs/research/forwards_sources.md` (ce fichier)
- [x] Script de sniff `scripts/phase0_sniff_forwards.py` (livré côté repo)
- [ ] **Snapshot JSONL exécuté côté FMV** (`docs/research/phase0_forwards_snapshot.jsonl`)
- [ ] Validation du contrat parser (mois/trimestres/Cal en libellé FR confirmés)
- [ ] Décision actée par E5 sur la double-source (Yearly + Historique) en parquet historique unifié

Les deux dernières cases nécessitent l'exécution du script sur le poste FMV.

---

## 6. Sortie de phase → entrée Phase 1

Phase 1 ne démarre que si :
1. Snapshot JSONL est commit dans `docs/research/`.
2. Mapping mois/trimestre FR confirmé sur snapshot réel.
3. Couverture historique de `Historique2019` confirmée (date min, date max, nombre de produits).

Sinon Phase 1 risque de coder un parser sur des hypothèses fausses.

### Réservation quotidienne ENTSO-E hors ligne

D238 rend testable la limite d'une capture par journée `Europe/Zurich`. La
journée est consommée dès la réservation autorisée, avant tout démarrage de
Warehouse. Les états réservé, démarré, réussi et échoué comptent tous : un
échec ne permet donc ni retry ni seconde capture le même jour. Le registre
refuse aussi les journées, identifiants de réservation, lots et propositions
réutilisés, ainsi que les horodatages non strictement croissants.

Le candidat embarque la proposition D236 complète, y compris son plafond
financier, et la journée est dérivée de son heure de création convertie en
`Europe/Zurich`, pas du jour UTC. Le lot reste purement synthétique : aucune
persistance atomique, réservation réelle, autorisation ou exécution n'est
accordée. Le roast donne `20 passed`; la matrice adjacente D233-D238 donne
`261 passed`; deux matérialisations identiques produisent la preuve
`2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8`.
Tous les compteurs Databricks, Warehouse, réseau, `H:` et écriture distante
restent nuls.

### Profil qualité normalisé ENTSO-E hors ligne

D239 transforme le contrat ENTSO-E en contrôles reproductibles sans ouvrir un
artefact réel. Les trois rôles sont vérifiés à leurs grains exacts : dimension
par `series_id`, latest par `series_id, target_ts_utc`, vintages par
`series_id, target_ts_utc, as_of_utc`. Les clés dupliquées, champs requis nuls,
valeurs non finies, révisions négatives, séries orphelines et couvertures
dimension/faits incomplètes sont refusés.

Le profil exige les groupes suisses, les six composantes de production et les
quatre frontières CH-DE/FR/IT/AT pour flux, échanges programmés et NTC. Il
contrôle `MW`, `MWh`, `EUR/MWh`, la cadence native, la convention de signe et
la grille UTC. Les informations prévisionnelles, prix, programmes et capacités
doivent être disponibles au plus tard à la livraison ; les actuals et le
stockage ne peuvent être disponibles avant leur période. Les révisions et
timestamps de chargement doivent progresser, et latest doit égaler la dernière
vintage sur chaque clé commune.

La sortie par série ne contient aucune valeur brute : seulement slots attendus
et observés, trous, jours UTC complets, première vintage fiable, backfill et
couverture latest/vintage. Les 730 jours complets restent une condition du
diagnostic saisonnier, distincte de la propreté structurelle. Le rejeu
rolling-origin ne sélectionne que `as_of_utc <= origin_utc`; latest ne remplace
jamais une historique PIT.

Le roast ciblé donne `16 passed`, la chaîne D231-D239 `344 passed` et la
matrice qualité locale `31 passed`. Deux matérialisations identiques produisent
la preuve
`5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b`.

### Préflight d'intégrité du paquet ENTSO-E hors ligne

D240 vérifie les octets avant de laisser D239 analyser les tables. Le paquet
accepté ici est exclusivement synthétique et repo-local : trois fichiers
NDJSON ordonnés, avec chemins, SHA-256, tailles, nombres de lignes, colonnes,
types et schémas exacts. BOM, CRLF, ligne finale manquante, ligne vide, clé JSON
dupliquée, valeur non finie, type ou ordre faux, fichier supplémentaire,
chemin dangereux, troncature et altération des preuves sources sont refusés.

Après succès d'intégrité seulement, les 33 lignes de dimension, 792 lignes
latest et 792 lignes vintage sont profilées en mémoire par D239. La preuve
persistée ne contient aucune valeur brute. Elle distingue explicitement
l'intégrité technique, la qualité analytique et l'autorité réelle : seule la
première est verte sur cette fixture. Le streaming des gros Parquet réels et
l'admission d'un reçu réel ne sont pas encore implémentés.

Le roast ciblé donne `14 passed` et la chaîne D231-D240 `358 passed`. Deux
matérialisations identiques produisent la preuve
`5595a20c5b997485bbaa0e3aa41f90b131190e3e89d812b1acfdfaaebc88536b`.
Tous les compteurs Databricks, Warehouse, réseau, `H:` et écriture distante
restent nuls. Aucun artefact réel n'a été ouvert et aucun droit de modèle,
sélection ou production n'est accordé.
La fixture d'algorithme contient 33 séries et 792 lignes par table de faits,
mais elle reste synthétique et non empirique. Un appelant ne peut pas convertir
des booléens « hash/reçu vérifié » en admission réelle. Tous les compteurs
Databricks, Warehouse, réseau, `H:` et écriture distante restent nuls ; modèle,
sélection et production restent interdits.

### Préflight Parquet ENTSO-E borné et hors ligne

D244 prépare le lecteur des futurs artefacts ENTSO-E normalisés sans lire les
tables Databricks. Il lie le constat réel D241 : les schémas bruts Unity
Catalog à 11/8/10 colonnes ne correspondent pas encore à l'interface cible.
Les champs manquants de cadence, lignage, qualité, révision, signe et temps PIT
canonique restent donc bloquants et ne sont jamais fabriqués.

Le vérificateur synthétique traite les Parquet par lots Arrow, sans pandas ni
chargement complet des octets. Il calcule le SHA-256 avant et après le scan sur
le même descripteur, vérifie magic/footer, version/producteur, schémas, codecs,
row groups et checksums de pages. Les tailles comprimées/décomprimées, ratio de
décompression, limites Thrift, lignes/lots et chaînes sont bornés. Nulls,
valeurs non finies et révisions négatives sont refusés.

La fixture mécanique contient trois lignes de dimension et 8 193 lignes dans
chacun des deux faits ; ses sept lots sont tous visités. Elle n'est ni un
profil de qualité complet ni une preuve empirique. Le roast ciblé donne
`16 passed`, la chaîne D231-D244 `374 passed`, et deux matérialisations
identiques produisent la preuve
`42c1065bf66117a2be2c792424f08d08511056d0df5f3b4b706ac57af1fcf564`.
D244 effectue zéro connexion/requête Databricks, zéro démarrage de Warehouse,
zéro appel réseau, zéro accès `H:` et zéro écriture distante. Données réelles,
PIT, qualité incrémentale, modèle, sélection et production restent interdits.

### Profil qualité ENTSO-E incrémental et borné

D245 ajoute la couche analytique qui manquait à D244, toujours entièrement en
local et sur données synthétiques. Après la vérification des octets, schémas et
lots Parquet par D244, un second passage Arrow calcule les contrôles de qualité
sans pandas et sans charger une table complète. Un index SQLite temporaire,
limité à 1 Gio, conserve uniquement les clés structurelles, les timestamps et
des empreintes SHA-256 des valeurs ; aucun prix, volume, load ou niveau de
production brut n'y est stocké.

Le profil refuse les grains dupliqués, séries orphelines, trous sur la grille
native, unités ou conventions de signe invalides, disponibilités incohérentes,
révisions non chronologiques, familles suisses/frontières/technologies
manquantes et écarts entre latest et dernière vintage. La cadence est vérifiée
par série : une série historiquement horaire n'est pas considérée défectueuse
du seul fait que la PFC cible est au quart d'heure. La sortie sépare les taux et
constats de qualité de l'autorité réelle des données.

La fixture compte 33 séries, 792 lignes latest et 792 vintages, avec un jour UTC
complet et 792 clés de recouvrement. Elle valide l'algorithme mais ne satisfait
pas le seuil empirique de 730 jours. Le roast ciblé donne `12 passed`, la chaîne
D231-D245 `386 passed`, et deux matérialisations identiques produisent la preuve
`a5840e92c5ea783d9931069a8725352398e882831faf4e3fd6beb8f80a9653a1`.
Tous les fichiers SQLite temporaires ont été supprimés. D245 effectue zéro
connexion ou requête Databricks, zéro démarrage de Warehouse, zéro appel réseau,
zéro accès `H:` et zéro écriture distante. Les schémas réels D241, l'inventaire
D243, le PIT réel, l'entraînement, la sélection et la production restent
bloqués.

### Remédiation du mapping temporel ENTSO-E réel

D250 lie le schéma Unity Catalog observé par D241 à une règle de
normalisation explicite. Les commentaires des deux tables de faits définissent
`DateTimeUtc` comme la borne droite de l'intervalle. Notre interface PFC définit
au contraire `target_ts_utc` comme le début de l'intervalle : un renommage direct
est donc interdit et la transformation doit être
`DateTimeUtc - native_resolution`.

Cette règle respecte le guide officiel ENTSO-E, où chaque point est placé à
`period start + (position - 1) * resolution`. Elle ne suppose ni une cadence
horaire ni une cadence 15 minutes : `native_resolution` doit être publiée par
`SeriesID` et la cadence historique propre à chaque série doit être conservée.
Références primaires : [bibliothèque EDI ENTSO-E](https://www.entsoe.eu/publications/electronic-data-interchange-edi-library/)
et [guide officiel des séries temporelles](https://www.entsoe.eu/Documents/EDI/Library/Introduction_of_different_time_series_v1.4.1.pdf).

Les trois timestamps `PublicationTimestampUtc`, `PullTimestampUtc` et
`Meta_Load_Timestamp` sont conservés séparément. La proposition conservatrice
`as_of_utc = max(publication, pull, load)` n'est admissible que si les trois
valeurs sont présentes et si le propriétaire des données approuve leur
sémantique ; elle reste inactive pour l'instant. `latest` ne constitue jamais
une preuve historique point-in-time. Le guide de publication ENTSO-E précise
que `createdDateTime` date la création du document, ce qui ne suffit pas à
décrire sa disponibilité dans notre système : [modèle officiel de publication](https://www.entsoe.eu/Documents/EDI/Library/cim_based/schema/Publication_document_UML_model_and_schema_v1.3.pdf).

Le contrat refuse aussi de transformer `DocumentType` en identifiant de
document, `VintageID` en révision source, ou d'inventer une qualité `OK`, une
révision zéro et une convention de signe. La preuve D250
`5269f46ac8bc078eacbd57e41ab6a447f9153ce9662880e99c846bc3774d3576`
n'effectue aucun SQL, ne démarre aucun Warehouse, ne lit aucune valeur de table
et n'accorde aucune autorité PIT, modèle ou production.

### Audit de couverture ENTSO-E pour la PFC CH

D252 conclut qu'il est encore impossible d'affirmer que tous les types de
données nécessaires existent dans `dev.gold`. Les schémas réels et 11
macro-signaux observés le 3 août sont encourageants, mais l'inventaire exact et
courant de la dimension n'a pas été exécuté. Le forecast de charge distinct,
les horizons renouvelables, les technologies, frontières/directions, cadences,
unités, signes, lignage, qualité, révisions et PIT restent non prouvés.

L'audit ajoute une exigence auparavant insuffisamment explicite : qualifier les
zones couplées CH, DE_LU, FR, IT_NORTH et AT pour les prix day-ahead, puis leurs
charges, forecasts et productions. Il ajoute aussi comme candidates les prix
intraday, batteries/stockage, FCR, impact des outages réseau sur les positions
nettes, paramètres flow-based long terme, prévisions d'adéquation, outages de
grandes consommations et historique zones/EIC. Seul ce dernier est un prérequis
de métadonnées ; les autres restent non bloquants pour la première baseline et
doivent gagner leur place par rolling-origin.

Le contrat D252 a l'identifiant
`a032c630091b491adc2925b7432ed6a916080f10f21ba631975291b7f8b1470c` et
son roast ciblé donne `10 passed`. Le rapport détaillé est
`docs/research/ENTSOE-DEV-PFC-DATA-COVERAGE-REPORT-20260806.md`.
D252 utilise zéro connexion/requête Databricks, zéro démarrage de Warehouse,
zéro écriture, zéro ligne ENTSO-E réelle et zéro accès `H:`.

### Régimes de cadence ENTSO-E effectifs dans le temps

D260 corrige une faiblesse du contrat qualité initial : une unique
`native_resolution` dans la dimension ne suffit pas à décrire tout
l'historique lorsqu'une même série passe, par exemple, de 60 à 15 minutes.
Le sidecar cible est au grain `series_id, valid_from_utc`, avec intervalle
gauche-fermé/droite-ouverte, `valid_to_utc`, résolution, document source,
endpoint et date de confirmation propriétaire.

La grille attendue est reconstruite morceau par morceau. Un historique horaire
n'est donc pas pénalisé parce qu'un régime ultérieur devient quart-horaire ; à
l'inverse, aucun quart d'heure antérieur n'est créé par upsampling. Les trous
sont comptés sans fill. Plusieurs familles peuvent conserver des cadences
différentes au même instant, par exemple prix day-ahead et NTC.

Cette prudence est nécessaire car les sources officielles décrivent des
évolutions distinctes par marché et par frontière. EPEX SPOT traite
l'introduction suisse séparément du SDAC et documente la coexistence de
produits/indices 15, 30 et 60 minutes : [EPEX SPOT, 15-minute MTU](https://www.epexspot.com/en/new-15-minute-products-market-coupling).
Swissgrid publie un calendrier de déploiement différencié pour les enchères
suisses et les frontières ; ces dates de roadmap ne prouvent pas à elles seules
la date de changement d'une série physique : [Swissgrid, Balancing Roadmap 2026–2030](https://www.swissgrid.ch/content/dam/swissgrid/about-us/newsroom/publications/balancing-roadmap-fr.pdf).

Le roast couvre une série prix horaire puis 15 minutes, une NTC restant
horaire, un jour DST de 23 heures, les trous, faux quarts d'heure historiques,
gaps, overlaps, bornes de transition, sources non HTTPS, timestamps naïfs et
tentatives d'auto-admission réelle. Il donne `18 passed`; la matrice adjacente
D239/D245/D250/D251/D260 donne `92 passed`. Deux matérialisations identiques
produisent la preuve
`740de518f8f4afc2b135f7cb3217f301c883897d39b78fa8a83d80b61fbf2739`.
Le profil ne contient ni valeurs ni identifiants de séries en clair. D260
effectue zéro requête Databricks, zéro démarrage de Warehouse, zéro appel
réseau, zéro accès `H:` et n'accorde aucune autorité réelle, PIT, modèle ou
production.

### Fenêtre explicite pour la complétude des cadences ENTSO-E

D261 supersède le calcul de complétude D260 qui bornait implicitement le
contrôle par la première et la dernière observation. Cette convention pouvait
laisser passer une série tronquée : supprimer les premières ou dernières
lignes rétrécissait la grille attendue au lieu de produire des manquants.

Le profil v2 exige désormais trois instants UTC gouvernés :
`assessment_start_utc`, `assessment_end_utc` avec une fenêtre gauche-fermée et
droite-ouverte, puis `metadata_as_of_utc >= assessment_end_utc`. La grille
native attendue est construite sur toute cette fenêtre, indépendamment des
lignes présentes. Une série sans aucune observation est donc conservée dans
le diagnostic avec tous ses slots manquants ; une ligne hors fenêtre est
rejetée.

La résolution courante de la dimension est comparée au régime actif au
`metadata_as_of_utc`, pas au dernier point disponible. Cet instant de preuve
n'a pas à coïncider avec un pas de livraison. En revanche, une confirmation
propriétaire postérieure à cet as-of et une provenance HTTPS ambiguë ou
contenant une query/credential sont refusées.

Le contrat v2 a l'identifiant canonique
`450d3628bf7dc83e16d630fe001709b58561f74bdad66d73b155b199f5d96821`.
Le roast ciblé donne `30 passed` et la matrice adjacente ENTSO-E `131 passed`.
Après formatage mécanique, une matrice finale liée aux hashes courants couvre
cadence, zones, binding série-zone, familles et couverture : `88 passed`.
D261 effectue zéro requête Databricks, zéro démarrage de Warehouse, zéro appel
réseau, zéro accès `H:` et n'accorde aucune autorité réelle, PIT, modèle ou
production.

### Liaison de la cadence au paquet Parquet ENTSO-E

D270 conserve le paquet D244/D245 à trois rôles et publie les régimes de
cadence dans un sidecar Parquet séparé. Son manifeste est contenu-adressé et
lié au `snapshot_id`, au manifeste de base, au contexte qualité, à la fenêtre
d'évaluation et au `metadata_as_of_utc`. D245 doit passer avant le contrôle de
cadence.

Le contrôle D270 ne décode que les identifiants, groupes, résolutions et
timestamps nécessaires. Les clés et régimes sont indexés dans un SQLite local
éphémère, puis les observations triées sont comparées en flux aux slots natifs
attendus. La preuve synthétique couvre 33 séries et 792 timestamps ; une
mutation élargissant la fenêtre d'une heure de chaque côté rapporte exactement
66 slots manquants, sans fill ni resampling. La matrice D244/D245/D261/D270
donne `70 passed`. Cette preuve n'accorde toujours aucune autorité réelle, PIT,
modèle ou production.

### Rôles ENTSO-E sans fuite temporelle

D273 sépare sept rôles d'information. Les valeurs réalisées (charge,
production, flux, prix) peuvent servir à l'entraînement uniquement après la
fin de leur intervalle et leur disponibilité réelle ; elles ne peuvent pas
décrire l'intervalle en cours de prévision. Une erreur de forecast devient une
feature uniquement avec retard, après disponibilité du forecast et de la
vérité.

Un forecast opérationnel doit avoir été publié avant l'origine et couvrir la
cible dans son horizon déclaré. Il ne peut donc pas être étiré jusqu'à N+3.
Pour cet horizon, seules les informations connues à l'origine sont
structurellement admissibles : calendrier, climatologie gelée avant chaque
fold et éventuelle shape de scénario gouvernée déjà publiée. Cela ne prouve
pas leur valeur prédictive.

Les backfills ne gagnent jamais une disponibilité rétroactive. Une série
absente ou non supportée reste `NULL`, sans zéro ni valeur neutre. Toute
contribution ENTSO-E non calendaire reste une shape de moyenne mensuelle nulle
sous l'autorité du solveur. Le roast complet D273 donne `588 passed`; il n'a
ouvert aucune valeur réelle et n'a émis aucune requête Databricks.

### Admission composite des entrées physiques ENTSO-E

D280 assemble enfin les preuves sans les confondre : même paquet Parquet
D244/D245 lié à sa cadence D270, même dimension physique liée aux signatures
D243/D253 par D272, mêmes zones temporelles D254/D255 et mêmes demandes de
features qualifiées par D273. Une sélection directe n'accepte que les rôles
`REALIZED_ACTUAL`, `OPERATIONAL_FORECAST` et `LAGGED_REALIZED`, exactement une
fois, avec une cible comprise dans la fenêtre de cadence et une origine
antérieure à la sélection et à la référence.

Les erreurs de forecast restent interdites tant qu'un lignage de transformation
séparé ne lie pas leurs deux primitives. Calendrier, climatologie gelée et
shape de scénario restent hors de la sélection physique ENTSO-E. Toute
contribution non calendaire demeure de moyenne mensuelle nulle sous l'autorité
du solveur BASE.

Le roast synthétique couvre 82 séries, 1 968 créneaux horaires et trois
primitives, sans créneau manquant ni blocage. Le replay identique deux fois
produit la preuve
`c9821d3666f1f402c1214ad49f62dcb1a26d2661a18e16fd5fd5ce553a129753`.
La matrice adjacente donne `200 passed`. Ce PASS prouve seulement le câblage
fail-closed : données réelles, PIT réel, valeur prédictive, entrée modèle,
candidat et production restent explicitement non autorisés. D280 effectue
zéro requête Databricks, zéro démarrage de Warehouse, zéro écriture, zéro appel
réseau et zéro accès `H:`.

### Lignage des features ENTSO-E dérivées

D281 ferme le premier maillon des transformations dérivées sans ouvrir les
valeurs. L'erreur retardée de forecast de charge est définie sans ambiguïté par
`réalisé − forecast`, donc positive lorsque la charge réelle dépasse la
prévision. L'output est lié par hash aux deux enregistrements D273 exacts, dans
l'ordre de la formule, avec la même cible et la même origine.

La disponibilité de l'erreur est la plus tardive des deux disponibilités
primitives. Le calcul ne peut pas être déclaré avant cet instant ; les nulls
restent nulls. Dépendance manquante, doublée, inversée, auto-référente ou output
non lié échouent. La lecture stable du JSON rejette aussi les clés dupliquées.

Le roast D281 donne `19 passed`; la matrice D244 à D281 donne `219 passed`.
Deux replays identiques produisent la preuve
`7bbeab342f8d687068ec7591c69ad5690f63196e1916dd0d8c68fcb2794b0495`.
Ce PASS ne prouve volontairement ni unité, ni zone, ni cadence physique, ni
arithmétique sur les valeurs. D280 continue donc à bloquer cette feature tant
qu'un composite ultérieur n'a pas relié D281 aux séries physiques exactes.
D281 effectue zéro requête Databricks, zéro démarrage de Warehouse, zéro
écriture, zéro appel réseau et zéro accès `H:`.

### Compatibilité physique des features ENTSO-E dérivées

D282 compose les deux preuves sans les fusionner. Chaque primitive D281 est
retrouvée par son hash dans la sélection D280, puis reliée à la série, à la
signature sémantique, à l'unité canonique et à la zone temporelle exactes. Pour
l'erreur de forecast de charge, les séries réalisée et prévue doivent être
distinctes, toutes deux en `MW`, dans la même zone logique et valides sur toute
la cible.

D281 impose une cible identique aux deux primitives. D280 a déjà prouvé que
cette cible correspond exactement au slot de cadence natif effectif de chaque
série. La durée commune prouve donc la même cadence sans upsampling,
downsampling ni remplissage implicite. Le cutoff UTC et tous les IDs de contenu
sont liés au résultat ; une mutation pendant la composition échoue.

Le roast D282 donne `20 passed`; la matrice adjacente D244 à D282 donne
`239 passed`. Deux replays identiques produisent la preuve
`5961fba69241f0de74d14d3d0db062c03182eeb9b4748a2540f6553c0b9d4092`.
La matrice contractuelle ENTSO-E complète jusqu'à D282 donne `646 passed`.
Ce PASS décharge structurellement le seul blocage de lignage D280, mais
n'autorise toujours ni lecture des valeurs, ni calcul arithmétique, ni entrée
modèle, ni conclusion prédictive. Toute contribution future restera de moyenne
mensuelle nulle sous l'autorité exclusive du solveur BASE.

### Exécution décimale synthétique des features ENTSO-E dérivées

D283 exécute enfin la formule liée par D281 et physiquement qualifiée par D282,
mais uniquement sur des valeurs synthétiques. Les deux opérandes sont des
artefacts séparés, liés au contenu exact de l'assessment D282 et joints par les
hashes exacts des records. Chaque ligne doit reprendre la série, la zone,
l'unité `MW`, la cadence native et l'intervalle déjà prouvés par D282.

Les valeurs sont encodées en texte décimal canonique, jamais en float binaire.
Le calcul utilise une précision décimale de 34 chiffres et applique exactement
`actual - operational forecast`. Une entrée `NULL` produit `NULL`; aucun zéro,
remplissage, interpolation ou suppression de ligne n'est autorisé. Le zéro
négatif est normalisé à `0`, les sorties sont triées par hash du record dérivé
et l'artefact complet est content-addressé.

Le roast D283 donne `30 passed`, la chaîne D280 à D283 `88 passed` et la
matrice adjacente ENTSO-E jusqu'à D283 `269 passed`. Les 504 tests courants
nommés `test_entsoe_*` passent également. Deux replays identiques
produisent la preuve
`a2304c18e44f785a4f9e66bc200b0c50decb57e9f1646473ee0e25e218b10631`.
Le manifeste ne conserve aucune valeur, seulement les hashes et compteurs. Ce
PASS ne prouve ni les valeurs réelles, ni leur PIT, ni l'utilité prédictive et
n'autorise toujours aucune entrée modèle.

### Adaptateur Parquet synthétique borné des opérandes ENTSO-E

D284 ferme la frontière fichier qui restait entre une future livraison
normalisée et l’exécuteur D283. Le package contient un manifeste strict et un
seul Parquet frère. Avant tout décodage Arrow, l’adaptateur vérifie des lectures
stables mono-lien, le hash et la taille déclarés, les budgets de lignes,
colonnes, cellules et row groups, les types physiques et la compression.

Le schéma Arrow est exact et ordonné : hashes de record/série/zone, rôle
explicite, début et fin UTC, unité, cadence native et valeur décimale en chaîne
nullable. La clé est `rôle + feature_record_id + intervalle cible`. Un float,
une coercition de timestamp, une colonne nullable inattendue, un rôle inféré,
un doublon, un orphelin, une substitution ou une mutation pendant l’exécution
échoue. Les lignes admises sont séparées en deux artefacts D283, sans dupliquer
ni affaiblir son calcul décimal et ses contrôles D282.

Le roast D284 donne `25 passed`, la chaîne D280 à D284 `113 passed`, la matrice
adjacente `342 passed` et tous les tests `test_entsoe_*` `529 passed`. Deux
matérialisations identiques produisent la preuve
`860b652a495089234fbc546fb5743a63f0d6bcc2b8cba45f6a7dee2c974921fd`.
Le manifeste de preuve ne conserve ni Parquet, ni opérande, ni valeur de sortie.
D284 reste synthétique : données réelles, PIT réel, valeur prédictive, entrée
modèle, sélection et production restent non autorisés. Aucun accès Databricks,
réseau ou `H:` et aucune écriture distante n’ont eu lieu.

### Enveloppe complète de livraison ENTSO-E

D285 ferme l’écart entre les trois artefacts centraux déjà contrôlés par D244
et l’ensemble des preuves effectivement demandées au data engineer. Une
livraison complète contient maintenant dix rôles obligatoires : dimension,
historique des cadences, historique zones/EIC, valeurs latest, vintages,
résumé et détail qualité, inventaire des familles, rapport de trous et
réconciliation source. `excluded_series.parquet` devient un onzième rôle
obligatoire exactement lorsque le compteur d’exclusions est positif.

Le manifeste est content-addressé sans auto-référence, son dossier porte le
`snapshot_id`, l’inventaire est plat et exact, et chaque fichier déclare son
hash, sa taille, son compte logique, son schéma et les bornes de ses timestamps
pertinents. Le vérificateur effectue une première passe de hash streaming sur
tous les fichiers, puis une seconde passe globale, et recontrôle le manifeste
et l’inventaire. Il ne décode aucun Parquet ni JSON : intégrité locale de
l’enveloppe, authenticité source, schéma, qualité, PIT et aptitude modèle
restent des verdicts séparés.

Le roast D285 donne `34 passed`, la matrice adjacente D244–D285 `376 passed` et
tous les tests `test_entsoe_*` `563 passed`. Deux matérialisations identiques
produisent la preuve
`9b0f83b111eab55eb65e77fd340a38b248c882090ab5bbbbe129f58309839f84`.
Seule l’intégrité locale de l’enveloppe est vraie ; toutes les autorités data,
PIT, prédictives et modèle restent fausses. Aucun accès Databricks, réseau ou
`H:` et aucune écriture distante n’ont eu lieu.

### Première solution mensuelle CH ancrée sur la surface EEX courante

D286 matérialise la première couche de niveau mensuel réellement fondée sur
les octets EEX CH locaux sélectionnés par D212. Le solveur couvre 76 mois, de
septembre 2026 à décembre 2032. Août 2026 est volontairement exclu car il est
déjà en livraison : aucun niveau LT partiel n'est inventé.

Les 19 cotations BASE CAL/Q/M alimentent l'unique solveur de niveau. Le système
contient 17 contraintes indépendantes et deux cotations redondantes cohérentes.
Le repricing des contraintes actives est exact à la précision numérique et
toutes les cotations BASE affichées restent sous la tolérance de conflit de
`0.01 EUR/MWh`. Les 19 cotations PEAK sont conservées séparément pour un futur
gate de shape horaire ; elles ne peuvent pas modifier les moyennes mensuelles
BASE. DAY, WEEK et WEEKEND restent également hors de l'autorité mensuelle.

L'historique CH intervient seulement comme prior de forme recentré à moyenne
nulle dans chaque parent contraint. Aucun niveau voisin, ENTSO-E, AFRY ou OMPEX
n'est injecté. Un test de mutation confirme qu'un déplacement global du niveau
historique ne change pas la solution, tandis qu'une mutation PEAK ne change que
son sidecar.

Les Parquet contenant les prix restent sous `build/`; la preuve durable ne
contient que des hashes, compteurs, diagnostics de tolérance et autorités
fausses. Les roasts donnent `31 passed` sur D286, `54 passed` sur la chaîne EEX
locale et `153 passed, 1 skipped` sur la matrice solveur/LT adjacente. Le bundle
local porte l'identifiant
`a2e27c4e78515d2e7473769e60d7fa9767756fd2078ce5698388880e3a329a5b` et
la preuve price-free
`5db497336ddc218cb9256809a3f3005fbce7d475cfc2d048004163f461d2c8bd`.

Ce résultat est une couche de recherche et de repricing mécanique, pas encore
une PFC complète : la provenance PIT, la valeur prédictive, l'entrée modèle,
la candidature, la supériorité sur OMPEX, la promotion et la production restent
explicitement non autorisées. Le prochain saut utile est la shape horaire CH
sur vérité spot gouvernée, puis l'apport ENTSO-E PIT de moyenne mensuelle nulle.
