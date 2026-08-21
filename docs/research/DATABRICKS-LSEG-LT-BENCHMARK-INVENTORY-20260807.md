# Databricks LSEG — candidat benchmark PFC LT

Date : 2026-08-07  
Statut : **HPFC suisse confirmée dans le dépôt ; publication PROD absente**

## Verdict

Le dépôt `FMVSA/lseg-lakehouse` confirme une HPFC suisse dans le groupe
`continuous_forward` : mesure `110181967`, série
`ContPwrPriceForward.forward.Price`, pays/zone `CHE`, prix électrique horaire,
EUR/MWh, publication quotidienne et scénario 0. Le pull demande jusqu'à quatre
ans de valeurs futures.

La chaîne DEV dispose de la dimension, de la dernière prévision et de
l'historique des vintages. Elle est donc adaptée à un benchmark de courbe long
terme. Elle n'est pas publiée dans `prd` : `main` est 44 commits derrière
`dev`, aucun PR `dev -> main` ni tag `vX.Y.Z` n'existe, et aucun workflow PROD
n'a été exécuté.

L'usage initial recommandé est **benchmark externe indépendant uniquement**.
La courbe ne doit pas
réécrire les moyennes mensuelles du solveur EEX. Si elle devient plus tard une
entrée ou un teacher du modèle, elle ne pourra plus servir simultanément de
benchmark indépendant sur les mêmes origines et échéances.

## Éléments confirmés sans SQL

- `dev.gold.dimlsegcurves` : 17 courbes, 9 pays, une commodity, deux
  fréquences et trois unités ;
- la HPFC LT suisse est la courbe `110181967`, horaire, EUR/MWh et réémise
  quotidiennement ;
- la courbe suisse `pmt_spot_forecast` est une autre série, limitée à 16 jours,
  et relève du court terme ;
- les séries `epex_actuals` sont des réalisations day-ahead, utiles comme vérité
  de scoring mais pas comme prévision LT ;
- `dev.gold.factlsegcurvevalueslatest` : 633 893 points, environ 10,5 MB,
  échéances du 2022-12-31 au 2028-12-31 ;
- `dev.gold.factlsegcurvevaluevintages` : 164 269 954 points, environ 11,94 GB,
  dates de forecast du 2022-01-01 au 2026-08-04 ;
- le site LSEG expose séparément `Forecast Date`, `Updated` et
  `Corrected Date`. Le pipeline ne parse actuellement que `forecastDate`,
  stocké comme `ForecastDateTimeUtc`; il ne conserve pas les deux autres
  timestamps. `ForecastDateTimeUtc` ne prouve donc pas seul l'heure réelle de
  disponibilité ou de correction ;
- `KnownAtTimestampUtc` est dérivé de l'ingestion Bronze puis du pull. Il ne
  commence que le 2026-06-15 : les vintages antérieurs sont un backfill vendor
  et non une preuve de disponibilité historique dans le lakehouse FMV ;
- la dernière échéance au 2028-12-31 ne couvre pas intégralement un horizon
  glissant N+3 depuis août 2026, malgré une fenêtre de requête configurée à
  quatre ans ;
- les facts Gold/Silver de vintages ne sont ni partitionnés ni effectivement
  clusterisés. Un export intégral serait une mauvaise première opération ;
- `dev.bronze.lseg_curve_value_points` est partitionnée par `_year`, `_month`
  et conserve `source_timestamp`, `pull_ts_utc`, l'identité source et les
  contrôles DQ. Son `source_timestamp` minimum vaut 2000-01-01, valeur
  potentiellement sentinelle à exclure ou expliquer.

## Contrôle minimal restant

Lors d'un Warehouse déjà actif :

1. profiler uniquement la courbe `110181967` pour vérifier granularité, DST,
   couverture, trous et horizon effectivement livré ;
2. expliquer pourquoi les valeurs s'arrêtent en 2028 alors que le pull demande
   quatre ans ;
3. ajouter et documenter les timestamps source `Updated` et `Corrected Date`
   lorsqu'ils sont fournis, en conservant les valeurs nulles ;
4. ne lire les vintages qu'après définition d'un filtre de courbe et de fenêtre
   temporelle, ainsi que d'un plan de scan acceptable ;
5. confirmer le droit contractuel de conserver un snapshot LSEG sur le poste
   local.

## Exécution et preuve

- Unity Catalog/warehouse control-plane GET cumulés : 28 ;
- SQL : 0 ; lignes métier ouvertes : 0 ; démarrage de Warehouse : 0 ;
  écriture Databricks : 0 ;
- Warehouse associé : `STOPPED`, Classic `2X-Small` ;
- capture de schémas locale ignorée par Git :
  `build/databricks-lseg/2026-08-07/lseg-selected-table-schemas.json` ;
- SHA-256 :
  `55383924184e2ca3bb3f94bf29151f67589fb5f7709acd0c23d55a3ad1271701`.
- dépôt local ignoré par Git :
  `build/data-engineer-repos/lseg-lakehouse`, branche `dev`, commit
  `ebc3f23ff0a7e62e65471d861e4be993f35fdde1` ;
- `main` : `89abd94cd50e7827dd4094d5a0944cfebfa864d8`, 44 commits derrière
  `dev` ; tags de release : aucun.
