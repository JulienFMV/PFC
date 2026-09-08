# PFC CH — Réponse à l'audit Claude, D317

8 septembre 2026. Rapport indépendant conservé sans modification :
[audit des 33 constats](PFC-CH-AUDIT-REPORT-20260908.md), commit source
`719a18975229d5d13477c78a45c1aff6e3eebcfe`, importé par cherry-pick documentaire.
Le code audité reste `dee652bc919d06345f71304d1f1eaacf0edf7bc7`.

Les défauts démontrés de collecte EEX et de causalité temporelle ont été
reproduits puis corrigés. La recette D304, le solveur, l'assembleur et la
projection EEX sont inchangés. Aucun nouvel ajustement de modèle, aucune
capture réelle du jour 2 et aucune admission ne sont réalisés dans D317.

## Résultat vérifié et portée

- Matrice de 38 modules : **716 réussites, 0 échec, 5 skips**, 117,42 s.
  Elle reprend le checkpoint et ajoute l'intégration snapshot Databricks/v4
  et les contrats publisher. Ce n'est pas la suite entière du dépôt ni une
  exécution sur Linux. Un avertissement préexistant concerne une conversion
  de fuseau dans le chargeur legacy Energy Charts.
- Rejeu indépendant local de cinq sondes sur les anciennes fonctions réelles
  puis sur les fonctions corrigées : fuite d'epoch, grille décalée de 0,5 s,
  création documentaire non causale, départage par observation future,
  effacement d'une date EEX entièrement quarantainée. Les anciens comportements
  sont reproduits et les nouvelles fonctions les rejettent.
- Les valeurs de la fixture PIT valide sont identiques avant/après. Le rejeu
  des entrées réelles gelées du premier jour reproduit **75 niveaux mensuels,
  écart maximal 0,0 EUR/MWh**. Toutes les liaisons SHA-256 de la recette D304
  restent identiques. Cela ne revalide pas les performances D304–D310 depuis Git.
- Les régressions synthétiques sont publiables et rejouables depuis Git.
  Les données fournisseurs, exports candidats et preuves numériques locales
  restent dans `build/`. Aucun chiffre fournisseur n'est présenté comme une
  preuve indépendante accessible au seul lecteur GitHub.

Traces locales : `build/lt-audit-response-20260908/`, notamment
`before.json`, `regressions-before.xml`, `tests-matrix.xml` (premier échec
d'intégration conservé), `tests-matrix-v2.xml`, `tests-matrix-v2-command.json`,
`independent-v1/verification.json` et `independent-v1/day1-base-quote-diagnostics.csv`.

## Traitement des 33 constats

« Corrigé » désigne le périmètre testé ci-dessous. « À traiter » et « partiel »
ne valent pas fermeture. Aucun constat ne transforme une autorité en `true`.

| Constat | Décision et preuve / travail restant |
|---|---|
| F-01 | À traiter avant le prochain benchmark : rejet explicite des clés BASE mensuelles manquantes. Pas de modification de l'assembleur gelé dans D317. |
| F-02 | Retenu : l'asymétrie de réajustement des candidats ratio doit être déclarée et contrôlée dans une nouvelle comparaison. |
| F-03 | Retenu comme hypothèse : conditionner l'amplitude signée au niveau peut aider ; aucune attribution causale des régressions 2024 n'est établie. |
| F-04 | À traiter dans la qualification de couverture du prochain benchmark : ne pas assimiler un contrôle au 28 février à une preuve annuelle complète. |
| F-05 | À traiter avant le prochain benchmark : afficher les régimes défavorables sans support multi-origine et ajouter un veto par origine, sans les compter comme validation robuste. |
| F-06 | À traiter avant le prochain benchmark : assert d'identité des populations, heures et maturités entre candidats. |
| F-07 | Retenu : bord NaN D307 documenté ; ne pas réécrire ses anciens résultats. Le prochain protocole rejette explicitement toute métrique non finie. |
| F-08 | Affirmation précisée : seuils D307 calculés avec données pré-origine pendant le run ; ils n'étaient pas tous sérialisés dans le plan initial. |
| F-09 | À traiter dans le prochain plan : politique explicite `development_exposed`, registre des origines déjà vues et distinction du futur holdout. |
| F-10 | À traiter avant le prochain benchmark : sentinelle prouvant qu'aucun fit futur transmis mais supposé inerte n'est consommé. |
| F-11 | Retenu : ancrer tous les âges à l'origine ; le décalage constant s'annule dans les poids normalisés du cas actuel. |
| F-12 | Retenu : paramètres/counters/origines et métriques asymétriques à expliciter dans la prochaine version, sans embellir les anciens rapports. |
| F-13 | Corrigé : le test dépendant d'un fichier EEX local fait un skip explicite si ce fichier manque ; son hash reste contrôlé s'il existe. |
| F-14 | Recette CI corrigée : installation du projet et de l'extra `test`, qui apporte pandas et les dépendances de conftest. Contrats publisher réussis localement ; résultat Actions distant à vérifier après push. |
| F-15 | Ouvert : pas de nettoyage global des 264 violations rapportées par Claude. Le runtime local utilisé ne contient ni ruff ni pip ; aucune réussite lint globale revendiquée. |
| F-16 | Corrigé dans la lane PIT : `SOURCE_DOCUMENT_CREATED` est rejeté ; un horodatage de document ne prouve pas la publication originale. Sonde ancienne/nouvelle reproductible. |
| F-17 | Affirmation corrigée dans README et les deux contrats Databricks : PIT atomique testé sur fixtures seulement. Les blocs PRD et `publication > first_seen` restent à qualifier ; aucun contournement du rejet. |
| F-18 | Corrigé : timestamps explicitement zonés exigés dans toutes les colonnes temporelles Silver ; epochs numériques et dates naïves rejetés. Cas de fuite tardive reproduit indépendamment. |
| F-19 | Corrigé : écarts en nanosecondes et alignement modulo la résolution native, sur spot et ENTSO-E. Rejet des décalages cohérents de 0,5 s. |
| F-20 | Corrigé aux frontières visées : snapshot v4 de calibration exige le mode Silver PIT ; le rejeu générique contrôle les métadonnées Databricks et les lie au hash de frame. Aucun label seul ne vaut preuve PRD. |
| F-21 | Corrigé dans le matérialiseur PIT : départage par première observation, jamais dernière observation future. Des valeurs différentes à ordre causal égal restent ambiguës et sont rejetées. Le SQL legacy n'est pas qualifié par ce correctif. |
| F-22 | Différence de contrat conservée et documentée : latest-observed et SQL realized ne sont pas déclarés interchangeables. Harmonisation éventuelle exige un rejeu séparé des preuves D300. |
| F-23 | Partiel : provenance de cadence native spot, comptages d'expansion, rejet des fausses vérités 15 min dans rejeu et qualité snapshot. La représentation/pondération des séries physiques ENTSO-E répétées reste ouverte. |
| F-24 | Notes retenues : ancrage CH, sémantique des hashes et provenance d'origine restent des points de qualification des sources ; aucune nouvelle admission. |
| F-25 | Corrigé : seules les dates acceptées sont remplacées ; rejet si une date historique renvoyée perd toutes ses lignes acceptées ou si la dernière date brute devient un repli vers une date antérieure. Quarantaine conservée avant rejet. |
| F-26 | Affirmation corrigée, politique ouverte : le solveur élimine les parents redondants sous 0,01 EUR/MWh. Le nouveau reçu exporte tous les diagnostics et cette règle non signée. « Aucun conflit accepté » désigne uniquement la gate d'audit, pas une absence d'arbitrage numérique du solveur. |
| F-27 | Corrigé pour les nouveaux jours : dates EEX brute/normalisée/finale dans la requête et le registre v2 ; égalité, cohérence avec le Parquet lié et date d'observation contrôlées. Lecture v1 conservée, sans inventer de dates historiques. |
| F-28 | Partiel : provenance PRD locale et restriction d'autorité explicites à côté du manifeste legacy. Son label interne `TEST_FIXTURE` reste inchangé pour conserver l'interface gelée ; il ne décrit pas l'origine réelle des cotes. |
| F-29 | Corrigé : reçus fondés sur les actes de cette invocation (`stop_requests=0`, `manual_stop_attempted=false`), sans recycler un ancien HTTP 403. |
| F-30 | Corrigé : échec de lecture finale du Warehouse conservé séparément ; il ne masque plus l'exception primaire de capture. Test de la vraie fonction avec transport injecté. |
| F-31 | Corrigé après validation du répertoire local de sortie : erreurs précoces de registre/configuration produisent un reçu d'échec et manifeste. Une sortie non autorisée reste rejetée avant toute écriture. |
| F-32 | Corrigé : dès le jour 2, historique du dernier enregistrement lié, schéma exact et recontrôle SHA-256 avant restitution de la requête ; la graine de configuration sert seulement au registre vide. |
| F-33 | Limites conservées : identité du propriétaire du verrou, concurrence des invocations manuelles, inventaire exhaustif et bornes transport. Le pilote n'est pas présenté comme un service autonome qualifié. |

## Ordre de travail après D317

1. Vérifier indépendamment ces corrections et la CI ; poursuivre la collecte
   de vrais jours lorsque disponibles. Jour 2 utilise les contrôles corrigés,
   la même recette et un nouveau répertoire. Aucun jour artificiel ni origine
   rétrospectivement comptable. Résoudre la politique des conflits avant toute
   admission, sans changer la recette du pilote silencieusement.
2. Qualifier les entrées nationales CH : sémantique, couverture, révisions et
   origine des blocs Silver, capacités commerciales directionnelles, PV,
   hydro et indisponibilités. Les 17 séries clients FMV ne représentent pas
   la Suisse et ne constituent pas un prérequis de la PFC nationale.
3. Corriger F-01/F-05/F-06/F-10 et figer un benchmark CPU restreint de forme
   signée conditionnée au niveau. D304 reste la référence ; D305–D307 restent
   des comparaisons. Le plan doit précéder tout résultat ; aucun choix par mois.
4. Tester ensuite un signal structurel réseau/PV, avec le registre documentaire
   comme provenance et des séries quantitatives qualifiées comme entrées.

## Challenge du modèle proposé

Une famille additive du type `forme = a(calendrier) + b(calendrier) × niveau`
est une piste raisonnable. L'ablation `b = 0` doit reproduire D304 exactement.
Conserver très peu de recettes globales, régulariser fortement la pente et
centrer avec les durées réelles du mois suisse avant l'assembleur/projection.
La projection BASE/PEAK peut retirer du signal : mesurer l'effet avant ET après.

Attention au niveau utilisé pour l'apprentissage : une moyenne mensuelle
réalisée n'est pas la prévision solveur disponible à l'origine. L'utiliser
comme variable d'apprentissage puis substituer le niveau solveur en production
introduit un écart entre entraînement et usage. Préférer des reconstructions
du solveur connues à l'origine ; isoler le niveau réalisé comme diagnostic
oracle, jamais comme entrée future disponible. L'erreur de niveau et l'erreur
de forme doivent être publiées séparément.

Le protocole à figer devra contrôler années de livraison, saisons, horizons,
prix négatifs/proches de zéro, rampes, pointes et support des régimes, avec
seuils pré-origine, populations communes et veto par origine. Pas de division
par un prix proche de zéro, pas de sélection au vu d'un seul millésime 2024.
La pertinence économique de la piste ne prouve pas qu'elle battra D304.

Une archive Git publique constitue un témoin utile. Elle ne suffit pas à
transformer les dates de commit, une branche réinscriptible ou des données
déjà vues en holdout prospectif indépendant. Figer aussi fenêtre future,
critères, données disponibles, mode de constat des réalisations et registre
de sélection avant ouverture. Les prix cibles et les covariables n'ont pas
la même règle de disponibilité, mais une collecte actuelle ne prouve pas
leur disponibilité à une ancienne origine.

## Événements et correction des horizons

La veille D316 oubliait des événements utiles. La révision documentaire
[V2](../data/CH-STRUCTURAL-EVENT-WATCHLIST-20260908-V2.json) conserve les quatre
observations initiales et ajoute l'accord CH–UE, le MTU aux frontières et les
deux retraits de Beznau. Aucun effet EUR/MWh, MW commercial ou probabilité
n'est inventé ; aucun collecteur récurrent n'est installé.

L'accord électricité a été signé le 2 mars 2026 et le message transmis en
mars. La source officielle consultée ne fixe pas ici une entrée en vigueur
certaine en 2028–2030. Son rang de facteur « dominant » demande une analyse
quantitative et des hypothèses de mise en œuvre. [OFEN](https://www.bfe.admin.ch/fr/accord-sur-lelectricite-suisse-ue).

JAO a annoncé le 30 juin 2026 un calendrier révisé visant 2027 pour les
enchères de capacité aux frontières suisses à pas de 15 minutes, sous
dépendances techniques. Cela ne donne ni jour ferme ni preuve du lancement
du marché day-ahead domestique suisse au même pas. [JAO](https://www.jao.eu/news/update-st-auctions-swiss-borders-15-min-mtu-and).

L'exploitant prévoit Beznau 2 jusqu'en 2032 et Beznau 1 jusqu'en 2033,
sous conditions de sûreté ; l'année ne doit pas devenir arbitrairement
un arrêt au 1er janvier. [IFSN/ENSI](https://ensi.admin.ch/en/2024/12/06/ensi-takes-note-of-the-decision-to-operate-beznau-npp-up-to-2033/).

**Erratum D316** : « horizon du candidat 2026–2029 » confondait la fenêtre
centrale de comparaison avec l'export étendu. Les entrées gelées du solveur
D312 couvrent **octobre 2026–décembre 2032**. Bickigen–Chippis (année annoncée
2031), Saint-Gothard (2030) et Beznau 2 (2032) peuvent donc concerner cette
extension. Cela ne prouve ni une admission de cet horizon ni une couverture
équivalente chez OMPEX/LSEG. La V2 ajoute cette lecture sans réécrire la V1.

Une régression de forme sur capacités JAO/ENTSO-E et PV reste exploratoire :
la capacité disponible dépend elle-même du réseau, de la météo et des arrêts.
Contrôler saison, charge, hydro, disponibilité voisine et dates de publication ;
ne pas assimiler corrélation à effet causal d'un ouvrage. Les transformations
et scénarios de capacité doivent être connus à l'origine. Le niveau mensuel
reste exclusivement au solveur, pour éviter de compter deux fois une annonce
déjà reflétée dans les forwards.

## Autorités et limites

`production=false`, `promotion=false`, `scientific_admission=false`,
`trading=false`, `externally_registered=false`, `countable_origin=false`.
CPU local uniquement dans cette réponse ; aucun SQL, Warehouse, GPU, AFRY
ou T057. Aucun export candidat antérieur n'est remplacé. Le handoff D317 et
la clôture locale détaillent fichiers, commandes, hashes et résultats distants.
