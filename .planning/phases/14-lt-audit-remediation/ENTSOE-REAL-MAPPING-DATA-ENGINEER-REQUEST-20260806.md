# ENTSO-E — contrat de publication Gold pour le data engineer

## Objectif

Publier des tables ENTSO-E Gold comprenant la dimension des séries, les valeurs
courantes, l'historique des vintages et les forecasts. Elles doivent préserver
l'information réellement disponible à chaque date (`PIT`, point-in-time) et
permettre à notre équipe de produire ensuite son propre snapshot local.

Le travail demandé est un travail d'extraction, de mapping technique, de
traçabilité et de contrôle qualité. Aucune feature ni transformation de
modélisation n'est demandée.

## Sources identifiées

- `dev.gold.dimentsoeseries`
- `dev.gold.factentsoetimeserieslatest`
- `dev.gold.factentsoetimeseriesvintages`

Cibles à publier en `prd.gold` : les trois surfaces ci-dessus, plus
`dimentsoeseriesresolutionhistory` et `dimentsoezonehistory`. Les rapports de
qualité, trous, familles et réconciliation seront calculés par notre exporteur
local à partir de ces Gold ; ils ne nécessitent pas chacun une table métier
supplémentaire.

La dimension Gold doit être publiée sans filtre. Si certaines séries sont
exclues des faits pour des raisons de volume ou de droits, publier leur
identifiant et le motif d'exclusion dans une surface de contrôle ; aucune
omission silencieuse.

## Périmètre minimal des faits

- prix day-ahead pour `CH`, `DE_LU`, `FR`, `IT_NORTH` et `AT` ;
- charge réelle et forecast de charge ;
- production réelle et forecast par technologie, notamment solaire, éolien,
  nucléaire, fil de l'eau, réservoir et pompage ;
- forecasts renouvelables day-ahead et intraday, distingués explicitement ;
- stockage des réservoirs hydro ;
- flux physiques, échanges programmés et NTC
  month-ahead/day-ahead-D2/intraday/year-ahead pour
  `CH-DE`, `CH-FR`, `CH-IT` et `CH-AT`, dans les deux directions ;
- outages de production/réseau et capacités installées/disponibles lorsqu'ils
  existent dans la source.

Exporter tout l'historique disponible. Si un export unique est trop volumineux,
le partitionner sans réduire la couverture, par exemple par année/mois.

Pour le backfill, viser au minimum `2019-01-01T00:00:00Z` pour chaque série
dont la source officielle possède cet historique. Un backfill historique ne
reconstitue pas les anciennes vintages : ajouter `capture_mode = BACKFILL` et
conserver comme `first_observed_at_utc` la date à laquelle notre chaîne a
réellement reçu la ligne. Ne jamais antidater cette disponibilité.

Les trous doivent être recherchés à nouveau dans la source officielle. Une
valeur retrouvée est livrée avec son document et ses timestamps source ; une
valeur toujours indisponible reste `NULL`. Ne pas compléter un trou par
interpolation, `forward-fill`, duplication d'une période voisine ou zéro.
Livrer la liste des trous par série avec `start_utc`, `end_utc`, nombre de slots
manquants, nouvelle tentative effectuée et résultat.

Retourner également un inventaire `présent`, `absent`, `ambigu` ou
`non_applicable` pour les familles ENTSO-E complémentaires suivantes, sans les
fabriquer si elles ne sont pas déjà collectées : prix intraday ; capacité de
réserve FCR/aFRR/mFRR/RR et prix de capacité ; énergie d'équilibrage aFRR,
mFRR et RR, volumes et prix, hausse/baisse ; prix de déséquilibre et
déséquilibre système ; redispatch/countertrading ; positions nettes ; capacité
cross-zonale intraday ; stockage par batteries ; paramètres flow-based.

Pour les NTC, ne pas considérer ces timeframes comme interchangeables. Le PDF
Swissgrid mensuel est un forecast indicatif ; les publications D-2 et intraday
sont des familles séparées. Réconcilier la source Swissgrid avec ENTSO-E et
conserver les deux provenances lorsqu'elles coexistent. Swissgrid indique que
les NTC CH-IT/IT-CH sont calculées dans la région Italy North et publiées sur
la Transparency Platform ENTSO-E.

## Surfaces requises pour notre export local

### Package que notre équipe produira ensuite

Le data engineer ne doit pas écrire sur notre `C:`. Les Gold publiées doivent
permettre à notre exporteur de créer un dossier plat nommé par son
`snapshot_id`, sans sous-dossier ni fichier supplémentaire, contenant
exactement :

Les noms de fichiers ci-dessous définissent notre mapping de sortie local. Du
côté Databricks, le besoin du data engineer est d'exposer les colonnes et
historiques nécessaires dans les Gold correspondantes.

- `manifest.json` ;
- `series_dimension.parquet` ;
- `series_resolution_history.parquet` ;
- `zone_history.parquet` ;
- `latest_values.parquet` ;
- `vintage_values.parquet` ;
- `quality_summary.json` ;
- `series_quality.parquet` ;
- `family_inventory.json` pour l'inventaire `présent / absent / ambigu /
  non_applicable` demandé ci-dessus ;
- `gap_report.parquet` pour les trous et nouvelles tentatives ;
- `source_reconciliation.parquet` pour les comptes source, livrés, exclus et
  rejetés par table et filtre ;
- `excluded_series.parquet`, avec une ligne par série exclue et son motif. Si
  aucune série n'est exclue, livrer ce Parquet avec le schéma attendu et zéro
  ligne ; le manifeste doit également déclarer le compteur à zéro.

Pour chaque fichier, `manifest.json` doit déclarer dans cet ordre logique : le
rôle, le nom relatif exact, le media type, le SHA-256, la taille, le nombre
d'enregistrements logiques, le hash du schéma et les bornes min/max des champs
timestamp pertinents. Le `snapshot_id` est le SHA-256 canonique du manifeste
sans le champ `snapshot_id` ; il engage donc tout l'inventaire sans
auto-référence.

### 1. `series_dimension.parquet`

Grain attendu : une ligne par `series_id`.

Colonnes de sortie et mapping source exact :

- `series_id = SeriesID`, `group_name = GroupName`,
  `field_name = FieldName` ;
- `document_type = DocumentType`, `business_type = BusinessType`,
  `process_type = ProcessType`, `psr_type = PsrType` ;
- `from_zone = FromZone`, `to_zone = ToZone`, `unit = Unit` ;
- `meta_load_timestamp_utc = Meta_Load_Timestamp` ;

Les valeurs source doivent être conservées telles quelles. Les éventuels
libellés normalisés sont ajoutés dans des colonnes séparées.

Livrer cet historique gouverné dans `zone_history.parquet` pour les zones et
codes EIC utilisés dans `FromZone` et `ToZone`, avec `source_zone_id`,
`eic_code`, `domain_kind`,
`valid_from_utc`, `valid_to_utc`, document et endpoint source. Une nouvelle
ligne est requise à chaque changement de code ou de périmètre ; aucun alias ne
doit être déduit d'un libellé.

Ajouter, lorsqu'ils existent dans la source amont :

- identifiant immuable du document source et endpoint source ;
- convention de signe, notamment pour les flux et échanges ;
- qualité source et numéro de révision source.

Un champ absent en amont reste `NULL` avec un motif explicite. Ne pas inventer
un endpoint, une qualité `OK`, une révision `0`, une unité ou un signe.

Contrôle demandé : pour chaque combinaison exacte des champs sémantiques
ci-dessus, le `COUNT(DISTINCT SeriesID)` doit se réconcilier avec le détail des
lignes exportées.

### 2. `series_resolution_history.parquet`

Grain attendu : `series_id, valid_from_utc`.

Champs requis :

- `series_id`, `valid_from_utc`, `valid_to_utc`, avec des intervalles
  `[valid_from_utc, valid_to_utc)` ;
- `native_resolution` au format ISO 8601, par exemple `PT60M` ou `PT15M` ;
- source de la résolution et date de confirmation.

Les intervalles doivent être non chevauchants. La résolution doit provenir de
métadonnées gouvernées, pas être déduite des écarts entre lignes. Une série
historique horaire est valide si sa cadence native était horaire ; aucun
upsampling vers 15 minutes. Si l'historique de résolution n'existe pas en
amont, signaler explicitement le blocage ; ne pas le reconstruire à partir des
valeurs.

Ne pas appliquer une date globale de passage à 15 minutes. La cadence est
propre à chaque série et à chaque produit : certaines publications suisses
sont déjà en 15 minutes alors que le prix day-ahead suisse reste régi par son
propre calendrier de marché. Vérification des sources publiques officielles au
6 août 2026 : la [feuille de route Swissgrid
2026–2030](https://www.swissgrid.ch/dam/jcr%3Aeaa2aa50-deb1-4579-92c6-dacb66429480/balancing-roadmap-en.pdf)
annonce les produits 15 minutes des enchères explicites day-ahead et intraday
au troisième trimestre 2026, sans publier de jour précis. Elle indique aussi
que les produits continus 15 minutes existent déjà aux frontières CH-DE et
CH-AT, et prévoit CH-IT en 2027 puis CH-FR en 2029. La date du 3 novembre 2026
n'est donc pas confirmée publiquement et ne doit pas devenir une règle globale.

Si le 3 novembre 2026 provient d'une communication non publique adressée aux
participants de marché, livrer sa référence, le produit concerné et la liste
exacte des séries affectées. Sinon, conserver la date effective comme inconnue
jusqu'à une annonce officielle plus précise.

### 3. `latest_values.parquet`

Grain cible : `series_id, target_end_utc`. Toute duplication à ce grain doit
être signalée et expliquée, pas supprimée arbitrairement.

Champs requis :

- `series_id` ;
- `target_end_utc = DateTimeUtc`, conservé comme borne droite ;
- `target_start_utc = target_end_utc - native_resolution`, en utilisant le
  régime source applicable à l'intervalle de livraison ;
- `value = FieldValue` ;
- `publication_timestamp_utc = PublicationTimestampUtc` ;
- `is_historical = IsHistorical` ;
- `meta_load_timestamp_utc = Meta_Load_Timestamp`.
- `capture_mode`, égal à `BACKFILL` ou `LIVE`, et
  `first_observed_at_utc`, correspondant à la première réception réelle dans
  notre chaîne.

Si la résolution effective manque, conserver `target_start_utc = NULL` avec
un motif. Ne pas supposer une heure ou 15 minutes.

`latest` représente l'état courant. Il ne doit jamais être présenté comme
l'information disponible à une ancienne date de calcul.

### 4. `vintage_values.parquet`

Conserver le grain source jusqu'à confirmation de sa clé officielle. Les
colonnes d'identification minimales sont :

`source_vintage_id, series_id, target_end_utc`.

Ne faire aucun `DISTINCT` ni dédoublonnage tant que la clé source officielle
n'est pas confirmée. Les collisions éventuelles doivent rester visibles dans
le rapport qualité.

Champs requis :

- `source_vintage_id = VintageID`, sans l'interpréter comme un numéro de
  révision métier ;
- `series_id`, `target_end_utc = DateTimeUtc`, `target_start_utc` ;
- `value = FieldValue` ;
- `publication_timestamp_utc = PublicationTimestampUtc` ;
- `pull_timestamp_utc = PullTimestampUtc` ;
- `meta_load_timestamp_utc = Meta_Load_Timestamp` ;
- `is_historical = IsHistorical` ;
- `capture_mode` et `first_observed_at_utc`, sans antidatage du backfill ;
- numéro de révision et qualité source s'ils existent réellement en amont.

Conserver les trois timestamps séparément. Ajouter `available_at_utc`
uniquement avec une règle documentée et approuvée par le propriétaire des
données. À défaut, livrer `available_at_utc = NULL`, `pit_eligible = false` et
un `pit_reason` explicite ; ne pas supprimer la ligne.

Les chargements historiques tardifs restent dans l'export. Lorsqu'ils sont
identifiables comme backfills source, les signaler explicitement et conserver
leur vraie date de chargement ; ne jamais leur attribuer une disponibilité
rétroactive.

Les forecasts doivent conserver toutes leurs vintages. Leur type
day-ahead/intraday, leur horizon, leur zone et leur technologie doivent rester
identifiables. Ajouter `forecast_issue_utc` seulement si son mapping vers un
timestamp source est confirmé ; sinon le laisser `NULL` avec un motif. Ne pas
remplacer les anciennes vintages par la dernière version connue.

### 5. `manifest.json`, `quality_summary.json` et `series_quality.parquet`

Le manifeste doit contenir :

- `snapshot_id`, date UTC de création et tables sources ;
- filtres et fenêtre temporelle exacts ;
- schéma, taille, nombre de lignes et SHA-256 de chaque fichier ;
- minimum/maximum des timestamps principaux ;
- version du job ou de la requête ayant produit l'export.

Le résumé qualité et son détail par série doivent contenir :

- lignes, doublons au grain attendu et taux de nulls ;
- première/dernière cible et minimum/maximum de chacun des trois timestamps de
  vintage ;
- couverture de jointure vers la dimension, attendue à 100 % ;
- slots attendus/observés/manquants selon la résolution native effective ;
- unités, zones, directions et catégories nouvelles ou disparues ;
- écarts entre `latest` et la dernière vintage sur les clés communes, calculés
  séparément selon `PublicationTimestampUtc` et `PullTimestampUtc` tant que la
  règle canonique de disponibilité n'est pas approuvée ;
- lignes exclues, rejetées ou non éligibles PIT, avec leur motif.

Pour chaque table et chaque filtre, le nombre de lignes lu à la source doit se
réconcilier exactement avec les lignes livrées et explicitement exclues.

## Règles non négociables

- Tous les timestamps sont en UTC et typés comme timestamps, pas comme textes.
- Les identifiants sont des chaînes, les indicateurs des booléens et `value`
  est numérique. Les valeurs non finies (`NaN`, `+Inf`, `-Inf`) sont signalées.
- Les prix sont explicitement en `EUR/MWh` ; les puissances/flux en `MW` et les
  énergies/stockages en `MWh` selon la sémantique source.
- Un prix négatif ou un flux signé n'est pas une erreur si la convention source
  l'autorise et la documente.
- Les séries `CH → voisin` et `voisin → CH` restent distinctes. Une direction
  absente reste `NULL`, jamais zéro.
- Aucun `forward-fill`, remplissage rétrospectif, `coalesce`, resampling,
  upsampling, dédoublonnage ou drop silencieux.
- Les valeurs manquantes restent `NULL` ; seules les vraies valeurs physiques
  nulles sont égales à zéro.
- L'export est un nouveau snapshot immuable. Ne pas fusionner avec un ancien
  cache local et ne pas écraser une livraison précédente.
- Lecture seule des tables sources ; aucune modification de `dev.gold`.

## Points à confirmer avec le propriétaire des données

1. clé primaire officielle et sémantique exacte de `VintageID` ;
2. signification et ordre chronologique attendu de
   `PublicationTimestampUtc`, `PullTimestampUtc` et `Meta_Load_Timestamp` ;
3. timestamp qui représente réellement la première disponibilité d'une ligne ;
4. source officielle de l'historique des résolutions par série ;
5. disponibilité des identifiants de document, révisions, qualités et
   conventions de signe dans la chaîne amont.
6. pour chaque changement de résolution, référence de la source, périmètre
   produit, date effective et séries touchées. La source publique Swissgrid
   actuelle ne donne que le troisième trimestre 2026 pour les enchères
   explicites day-ahead et intraday. N'utiliser le `3 novembre 2026` que si une
   communication officielle plus précise est fournie avec la livraison.

Une ambiguïté sur ces six points doit être documentée dans la livraison, pas
résolue par une valeur par défaut.
