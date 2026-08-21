# Databricks — inventaire spot et météo

Date : 6 août 2026  
Méthode : métadonnées Unity Catalog uniquement, sans SQL ni lecture de ligne.

## Verdict

- **Spot** : une surface spot granulaire candidate existe dans `dev`, mais sa
  source est nommée `Euler` et non EEX/EPEX dans le catalogue. Aucune table spot
  équivalente n'est visible dans `prd`. Son origine de marché, sa licence et sa
  couverture CH doivent donc être confirmées avant usage.
- **Météo** : les observations et forecasts MeteoSwiss/Open-Meteo existent
  bien dans `prd`, avec température, pluie, vent, irradiation, neige, nuages,
  humidité et pression. La couverture géographique et la disponibilité
  point-in-time des forecasts ne sont pas encore prouvées.
- Les deux SQL Warehouses visibles étaient `STOPPED` (`2X-Small`, auto-stop
  45 minutes). Aucun Warehouse n'a été démarré.

## Inventaire observé

Le catalogue accessible contient 534 tables/vues dans `dev` et 406 dans `prd`,
réparties dans les schémas `bronze`, `silver`, `gold`, `data_science` et
`data_quality`.

### Prix spot

| Table | Ce que le schéma prouve | Limite actuelle |
|---|---|---|
| `dev.silver.ge_market_euler_spot` | Intervalles `ts_start_utc`/`ts_end_utc`, fréquence native, produit/courbe, date et timestamp de quotation, prix décimal et unité | Source primaire, courbes CH, historique, fraîcheur et licence non prouvés |
| `dev.gold.dimspotproduct` | Dimension produit/courbe et fréquence | Contenu des produits non lu |
| `dev.gold.factspotpricemonthly` | Moyenne, minimum, maximum et nombre de points par mois | Agrégé mensuel : insuffisant pour apprendre la forme horaire/15 minutes |

`prd.gold.facteexpricedaily` et les autres tables `facteex*` sont des prix de
règlement de produits à terme EEX. Elles ne constituent pas des prix spot.
Aucune table `spot` ou `euler_spot` n'a été observée dans les 406 objets de
`prd`.

### Météo en production

| Table | Contenu utile observé dans le schéma | Risque à fermer |
|---|---|---|
| `prd.gold.factweather` | Mesures et forecasts MeteoSwiss/Open-Meteo : température, précipitations, vent, irradiation, neige, nuages, humidité, pression | Table mixte ; séparer mesure/forecast et confirmer les lieux |
| `prd.gold.factweatherforecasthistms` | Forecasts MeteoSwiss historiques/futurs : température, pluie, irradiation, niveau de gel, `MinLeadTime` | Pas de timestamp d'émission explicite dans le schéma |
| `prd.gold.factweatherforecasthistom` | Forecasts Open-Meteo historiques/futurs avec variables météo riches et `MinLeadTime` | Même risque PIT ; `MinLeadTime` est typé `string` |
| `prd.silver.weather_meteoswiss_measurement` | Mesures longues avec source, fréquence, unité, champ, valeur et cible UTC | Couverture géographique et trous non lus |
| `prd.silver.weather_meteoswiss_forecast` | Forecasts longs avec `lead_time`, indicateur historique et cible UTC | Origine/vintage de forecast non explicite |
| `prd.silver.weather_openmeteo_forecast` | Même grain long pour Open-Meteo | Origine/vintage de forecast non explicite |

Les vues `prd.data_science.ds_weather_*` montrent notamment Martigny, Evionnaz,
Blatten, Ferden et Steg. Cette liste de vues ne prouve pas que les tables Gold
sont limitées à ces sites, mais elle alerte sur un risque de couverture trop
locale pour une PFC nationale CH et ses voisins.

## Admission minimale avant modélisation

### Spot

1. confirmer contractuellement que `Euler` redistribue bien les prix EPEX
   utilisés et documenter la licence ;
2. inventorier les courbes/pays/produits et isoler `CH`, `DE_LU`, `FR`,
   `IT_NORTH` et `AT` ;
3. mesurer les bornes historiques, la fréquence native, les trous, doublons,
   unités et fraîcheur ;
4. conserver le prix spot granulaire ; ne jamais apprendre la forme depuis la
   seule moyenne mensuelle.

### Météo

1. inventorier `LocationID`, coordonnées/altitude, source, algorithme,
   fréquence et période couverte ;
2. distinguer observations et forecasts ;
3. obtenir un `forecast_issued_at_utc` ou une règle propriétaire confirmée pour
   le reconstruire ; sans cela, les forecasts ne sont pas admissibles en
   rolling-origin ;
4. profiler trous, doublons, révisions et fraîcheur par lieu/variable/source ;
5. sélectionner une couverture spatiale représentative de CH et des zones
   voisines, puis démontrer la valeur prédictive hors échantillon.

La météo ne pourra agir que comme forme zéro-moyenne à l'intérieur du mois. Le
solveur mensuel CH reste l'unique autorité de niveau.

## Coût et preuve

- appels de plan de contrôle : 28 (`18` inventaire, `9` schémas ciblés,
  `1` état des Warehouses) ;
- instructions SQL : `0` ; lignes métier ouvertes : `0` ;
- démarrages de Warehouse : `0` ; écritures Databricks : `0`.

Preuves locales ignorées par Git :

- `build/databricks-eex-daily/2026-08-06/catalog-surface-prd-dev.json` —
  SHA-256 `d2532463f25ca29dd1b868ee300389fa9cc62a495395331b81ae780596ff9aa9` ;
- `build/databricks-eex-daily/2026-08-06/selected-spot-weather-table-schemas.json`
  — SHA-256
  `27a40f03373900229488ab699187265a40c7581773248057f2fdecc2f6b3be5f`.

Prochaine action distante : aucune tant que le Warehouse reste arrêté. Quand il
sera déjà actif pour un besoin métier autorisé, exécuter un seul petit batch de
profiling en lecture seule, rapatrier le résultat sous `build/`, puis poursuivre
tous les contrôles localement.

Les deux requêtes préparées et non exécutées sont :

- `docs/data/sql/databricks_dev_spot_profile.sql` pour les courbes, fréquences,
  unités, intervalles, bornes temporelles et nulls spot ;
- `docs/data/sql/databricks_prd_weather_profile.sql` pour les lieux/points,
  variables, sources, fréquences, lead times, bornes temporelles et nulls météo.

Elles n'inventent ni origine de forecast ni disponibilité PIT. Les contrôles de
doublons au grain, de grille complète et de rolling-origin seront faits
localement sur l'export, où ils ne génèrent plus de coût Databricks.
