# ENTSO-E `dev.gold` - inventaire sans démarrage du Warehouse

Date : 2026-08-06  
Périmètre : métadonnées Unity Catalog et état des SQL Warehouses uniquement.

## Verdict

Les trois tables ENTSO-E attendues existent comme tables Delta managées dans
`dev.gold`, mais leur schéma réel ne satisfait pas encore le contrat PFC LT.
L'existence exacte de toutes les familles métier requises n'est pas prouvée :
elle nécessite l'inventaire des valeurs distinctes de `GroupName` et
`FieldName`, donc une lecture SQL de la petite dimension.

Aucune requête SQL n'a été exécutée. Les deux Warehouses visibles sont des
Warehouses classiques `2X-Small`, arrêtés, avec auto-stop à 45 minutes. Ils
n'ont pas été démarrés.

## Schémas réels observés

### `dev.gold.dimentsoeseries`

`SeriesID`, `FieldName`, `GroupName`, `DocumentType`, `BusinessType`,
`ProcessType`, `PsrType`, `FromZone`, `ToZone`, `Unit`,
`Meta_Load_Timestamp`.

### `dev.gold.factentsoetimeserieslatest`

`SeriesID`, `DateTimeUtc`, `DateUtc`, `FieldValue`, `Epoch`,
`PublicationTimestampUtc`, `IsHistorical`, `Meta_Load_Timestamp`.

### `dev.gold.factentsoetimeseriesvintages`

`VintageID`, `SeriesID`, `DateTimeUtc`, `DateUtc`, `FieldValue`, `Epoch`,
`PullTimestampUtc`, `PublicationTimestampUtc`, `IsHistorical`,
`Meta_Load_Timestamp`.

`DateTimeUtc` est documenté comme la borne droite UTC de l'intervalle ENTSO-E.

## Écarts bloquants du schéma

- `native_resolution` absent : impossible de construire et contrôler la grille
  native par série sans inférence.
- `source_endpoint` et identifiant de document source absents ;
  `DocumentType` décrit un type, pas le document reçu.
- `quality_flag` absent des deux faits.
- `revision_number` absent des deux faits.
- aucune convention de signe explicite par série pour les flux/échanges.
- `GroupName`, `FieldName`, zones, unité et timestamps de publication/chargement
  sont déclarés nullables ; leurs taux réels doivent être contrôlés.
- la sémantique PIT reste à figer : `PublicationTimestampUtc` et
  `PullTimestampUtc` existent dans les vintages, mais aucune colonne canonique
  `as_of_utc` n'indique laquelle gouverne l'information disponible.
- `latest` reste non-PIT et ne peut pas remplacer les vintages historiques.

## Familles minimales déjà prévues

- charge réelle et forecast de charge ;
- prix day-ahead en `EUR/MWh` ;
- production réelle et forecast, avec solaire, éolien, nucléaire, fil de l'eau,
  réservoir et pompage ;
- forecasts renouvelables day-ahead et intraday ;
- stockage des réservoirs hydro ;
- flux physiques, échanges programmés et NTC day/month/year-ahead sur
  CH-DE, CH-FR, CH-IT et CH-AT.

Le contrôle du 2026-08-03 indiquait que les groupes CH utiles étaient présents,
mais ne constitue pas un inventaire exhaustif et rejouable des valeurs de la
dimension au 2026-08-06.

## Familles supplémentaires à rechercher

Priorité haute pour expliquer les pointes, creux et régimes de forme :

1. indisponibilités de production et de réseau, planifiées et fortuites ;
2. capacité installée par technologie et disponibilité des unités ;
3. prix/volumes d'énergie d'équilibrage activée (`aFRR`, `mFRR`, `RR`), sens
   hausse/baisse, prix de déséquilibre et déséquilibre système ;
4. capacité de réserve achetée et son prix ;
5. redispatch, countertrading et autres actions de congestion ;
6. positions nettes/allocations implicites et évolution de la capacité
   cross-zonale intraday.

Variables dérivées à construire seulement après admission PIT : erreur de
forecast charge/solaire/éolien, charge résiduelle, rampes, facteurs de capacité,
net imports et indicateurs de rareté. Elles ne doivent pas devenir de nouvelles
sources ni modifier le niveau mensuel du solveur.

## Prochaine lecture minimale proposée

Une seule requête sur `dimentsoeseries`, sans faits numériques, regroupée par
`GroupName`, `FieldName`, types ENTSO-E, zones et unité, avec nombre de séries
et min/max de `Meta_Load_Timestamp`. Cette lecture permettrait de classer chaque
famille comme présente, absente ou ambiguë.

Elle n'a pas été lancée car le Warehouse configuré et le Warehouse DEV sont
tous deux arrêtés, classiques `2X-Small`, avec auto-stop à 45 minutes. Un
plafond de coût explicite ou un Warehouse déjà actif est requis avant lancement.

## Preuve locale scellée D241

Le schéma ci-dessus a été recapturé une seule fois par trois appels GET Unity
Catalog et validé hors ligne. La preuve locale est :

- ID de preuve :
  `d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544` ;
- contenu capturé :
  `d69fdab73ba1d9c55f70f77925f2253d583564d06d922f2b41e035763bca176f` ;
- statut : `FAIL_REAL_CONTROL_PLANE_SCHEMA_INCOMPATIBLE_NO_MODEL_AUTHORITY` ;
- compteurs : 3 GET, 0 SQL, 0 démarrage de Warehouse, 0 ligne ouverte et
  0 écriture Databricks ;
- colonnes observées : 11 dans la dimension, 8 dans `latest`, 10 dans
  `vintages` ;
- dérive de type sur les champs directement mappables : aucune.

Cette preuve qualifie uniquement la structure réelle. Elle ne prouve ni les
familles présentes dans `GroupName`/`FieldName`, ni la couverture, la fraîcheur,
les valeurs, les trous ou la disponibilité point-in-time.
