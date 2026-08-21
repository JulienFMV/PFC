# Analyse des repositories data — besoins PFC

Date : 2026-08-06

## Conclusion

Ne pas reconstruire toutes les données en parallèle sur le poste local. Le data
engineer publie les tables Databricks ; l'équipe PFC réalisera ensuite elle-même
un export Parquet local, sélectif et en lecture seule. En attendant, nous
réutilisons les fichiers déjà présents localement et développons les adapters à
partir des schémas des repositories.

Repositories analysés sur leur branche `dev` :

- `epi-lakehouse` : `a91b2e17c903454d1d1ec420f0a1dbd1b49243cc` ;
- `sdl-lakehouse` : `905a8abbff06caab2f9d9da1d22f2265616af2e9` ;
- `opendata-lakehouse` : `10b25187f0b56473a70adcbdd675932c0838d27a`.

## Besoins réels par repository

### `epi-lakehouse`

Le spot Euler est déjà produit au grain horaire dans
`{catalog}.silver.ge_market_euler_spot`, avec `ts_start_utc`, `ts_end_utc`,
`price`, `price_unit`, `curve_id` et `quotation_date`. La Gold actuelle
`FactSpotPriceMonthly` agrège ces points au mois et ne convient donc pas au
shaping de la PFC.

Besoin : ajouter une Gold additive au grain intervalle, par exemple
`FactSpotPriceInterval`, sans modifier `FactSpotPriceMonthly`. Elle doit garder
le début/fin UTC, le prix EUR/MWh, le produit, la zone de marché, la date de
quotation et une vraie information de publication/observation. Confirmer aussi
la source métier exacte derrière Euler et le mapping CH, DE_LU, FR, AT et
IT_NORTH. Le trigger Euler est actuellement configuré `PAUSED` en PROD : la
fraîcheur PROD doit être confirmée.

Deux corrections sont nécessaires avant cette Gold :

- `quotation_ts` est actuellement alimenté avec `curve_start_utc`, donc avec le
  début de livraison et non l'heure de quotation ; utiliser le timestamp source
  s'il existe, sinon `_ingest_ts` comme heure d'observation conservatrice ;
- le spot est forcé à `H`/`60` dans `10_bronze_to_silver`. Conserver la
  granularité source lorsqu'elle deviendra 15 minutes, sans changer
  rétroactivement les courbes horaires.

### `sdl-lakehouse`

Ce repository couvre uniquement les résultats d'appels d'offres SDL Swissgrid
depuis 2023. Il ne contient ni balancing, ni NTC, ni flux transfrontaliers.

Besoin : indiquer le repository ou job qui produit
`FactSwissgridBalancingQuarterHourly` et les autres données Swissgrid observées
dans Databricks. Pour les tenders, conserver dans la Gold le timestamp du
snapshot source et les composantes typées d'origine (`PowerPrice`,
`EnergyPrice`, unités, volumes), pas seulement `AssignedPowerPrice`.

### `opendata-lakehouse`

La branche `dev` est nettement plus avancée que `main`. Elle configure 22
familles ENTSO-E, notamment prix day-ahead, charge, production, hydro, forecasts,
échanges, flux, NTC day/week/month/year, balancing, outages et capacités. Elle
produit déjà les trois objets attendus : `DimEntsoeSeries`,
`FactEntsoeTimeSeriesLatest` et `FactEntsoeTimeSeriesVintages`.

Avant utilisation PFC, il reste à :

1. propager `resolution` de Bronze vers Silver et Gold afin de reconstruire
   correctement l'intervalle à partir du timestamp de borne droite ;
2. propager vers Gold la traçabilité PIT déjà disponible en Silver : première
   observation, dernière observation, `source_document_mrid`, révision du
   document et lien vers la vintage retenue ;
3. corriger l'identité de série : `SeriesID = xxhash64(field_name)` et la clé
   latest `field_name x Date_Time_UTC` peuvent fusionner plusieurs
   `TimeSeries` ENTSO-E. Conserver au minimum `source_time_series_id` et, pour
   les familles unitaires/outages, l'identifiant de la ressource ;
4. corriger les prix : le parseur choisit un seul champ parmi
   `currency_Unit.name` et `price_Measure_Unit.name`. Conserver les deux puis
   publier l'unité canonique `EUR/MWh` ;
5. ajouter la famille NTC intraday. La configuration actuelle contient
   day/week/month/year-ahead seulement. Les échanges programmés sont aussi
   limités au contrat `A01` ; ajouter une famille distincte si les échanges
   intraday sont requis ;
6. exécuter le backfill depuis 2019 et le notebook
   `90_post_backfill_validation`, puis fournir la couverture par famille, zone,
   direction et résolution ;
7. valider la cadence des familles intraday et des vintages. Le job unique à
   06:00 ne capture pas toutes les révisions intraday ;
8. promouvoir ensuite la version validée vers PROD. Le schedule PROD est encore
   `PAUSED`.

La convention actuelle de borne droite (`DateTimeUtc`) peut être conservée
pour compatibilité FMV, mais Gold doit exposer explicitement
`IntervalStartUtc`, `IntervalEndUtc` et `Resolution` afin d'éviter toute
ambiguïté.

### Météo et Swissgrid balancing

Aucun pipeline météo ni Swissgrid balancing n'apparaît dans les trois
repositories fournis. Demander leurs repositories ou jobs sources avant de
spécifier une modification. Ne pas demander de les reconstruire à l'aveugle.

## Qualification technique observée

- Les 21 notebooks des trois branches `dev` sont des JSON valides.
- Les modules Python ENTSO-E se compilent.
- Les trois worktrees locaux sont propres.
- Les CI des trois repositories exécutent `yamllint` et la validation du bundle,
  mais aucun test de logique des notebooks/pipelines.
- Les 7 tests unitaires ENTSO-E présents ne démarrent pas hors Databricks : le
  stub `pyspark.sql.types` ne fournit pas `StructType`. Ce point doit être corrigé
  et les tests ajoutés à la CI avant qualification de production.
- Aucune connexion, requête, lecture métier ou écriture Databricks n'a été
  effectuée pour cette analyse.

## Commentaire Jira proposé

> Merci pour les repositories. Après lecture des branches `dev`, voici les
> écarts précis pour la PFC :
>
> - `epi-lakehouse` : ajouter `FactSpotPriceInterval` depuis
>   `silver.ge_market_euler_spot`. Garder `ts_start_utc`, `ts_end_utc`, le prix
>   `EUR/MWh`, la granularité source et `_ingest_ts`. Ne pas utiliser
>   `curve_start_utc` comme `quotation_ts` ;
> - `opendata-lakehouse` : propager en Gold `resolution`, intervalle start/end,
>   `first_seen_pull_ts_utc`, document/révision et identité `TimeSeries`/unité.
>   La clé basée uniquement sur `field_name` ne doit pas fusionner plusieurs
>   séries. Composer correctement les prix en `EUR/MWh`, ajouter
>   `ntc_intraday`, puis terminer le backfill 2019 et
>   `90_post_backfill_validation` ;
> - dans `sdl-lakehouse`, les tenders sont couverts, mais pas le balancing ni les
>   NTC : merci d'indiquer le repository/job qui les produit ;
> - merci également d'indiquer le repository du pipeline météo.
>
> La livraison attendue reste dans Databricks. L'équipe PFC réalisera elle-même
> le snapshot Parquet local, en lecture seule, une fois les Gold validées.
