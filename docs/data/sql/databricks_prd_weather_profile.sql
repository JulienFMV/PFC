-- Profiling exact des familles météo candidates en production.
-- Ne pas exécuter si le Warehouse est arrêté. Cette requête ne reconstruit pas
-- un forecast_issue_utc : elle expose seulement les champs réellement présents.
-- NON AUTORISÉ EN L'ÉTAT : LIMIT borne les lignes retournées, pas les octets lus.
-- Confirmer taille Delta, partitions et plan estimé avant toute exécution.

WITH weather_long AS (
  SELECT
    'prd.silver.weather_meteoswiss_measurement' AS `source_table`,
    'MEASUREMENT' AS `information_role`,
    `measurement`,
    `data_entity`,
    CAST(NULL AS STRING) AS `data_point`,
    `data_point_type`,
    `data_source`,
    `sampling_frequency`,
    `unit`,
    CAST(NULL AS STRING) AS `lead_time`,
    `field`,
    `field_value`,
    `Date_Time_UTC` AS `target_ts_utc`,
    `Is_Historical` AS `is_historical`,
    `Meta_Load_Timestamp` AS `meta_load_timestamp_utc`
  FROM `prd`.`silver`.`weather_meteoswiss_measurement`

  UNION ALL

  SELECT
    'prd.silver.weather_meteoswiss_forecast' AS `source_table`,
    'FORECAST' AS `information_role`,
    `measurement`,
    `data_entity`,
    `data_point_category` AS `data_point`,
    `data_point_type`,
    `data_source`,
    `sampling_frequency`,
    `unit`,
    `lead_time`,
    `field`,
    `field_value`,
    `Date_Time_UTC` AS `target_ts_utc`,
    `Is_Historical` AS `is_historical`,
    `Meta_Load_Timestamp` AS `meta_load_timestamp_utc`
  FROM `prd`.`silver`.`weather_meteoswiss_forecast`

  UNION ALL

  SELECT
    'prd.silver.weather_openmeteo_forecast' AS `source_table`,
    'FORECAST' AS `information_role`,
    `measurement`,
    `data_entity`,
    `data_point`,
    `data_point_type`,
    `data_source`,
    `sampling_frequency`,
    `unit`,
    `lead_time`,
    `field`,
    `field_value`,
    `Date_Time_UTC` AS `target_ts_utc`,
    `Is_Historical` AS `is_historical`,
    `Meta_Load_Timestamp` AS `meta_load_timestamp_utc`
  FROM `prd`.`silver`.`weather_openmeteo_forecast`
), weather_profile AS (
  SELECT
    `source_table`,
    `information_role`,
    NULLIF(TRIM(CAST(`measurement` AS STRING)), '') AS `measurement`,
    NULLIF(TRIM(CAST(`data_entity` AS STRING)), '') AS `data_entity`,
    NULLIF(TRIM(CAST(`data_point` AS STRING)), '') AS `data_point`,
    NULLIF(TRIM(CAST(`data_point_type` AS STRING)), '') AS `data_point_type`,
    NULLIF(TRIM(CAST(`data_source` AS STRING)), '') AS `data_source`,
    NULLIF(TRIM(CAST(`sampling_frequency` AS STRING)), '')
      AS `sampling_frequency`,
    NULLIF(TRIM(CAST(`unit` AS STRING)), '') AS `unit`,
    NULLIF(TRIM(CAST(`lead_time` AS STRING)), '') AS `lead_time`,
    NULLIF(TRIM(CAST(`field` AS STRING)), '') AS `field`,
    `is_historical`,
    COUNT(*) AS `row_count`,
    COUNT_IF(`field_value` IS NULL) AS `null_value_count`,
    COUNT_IF(`target_ts_utc` IS NULL) AS `null_target_count`,
    COUNT_IF(`meta_load_timestamp_utc` IS NULL) AS `null_meta_load_count`,
    MIN(`target_ts_utc`) AS `min_target_ts_utc`,
    MAX(`target_ts_utc`) AS `max_target_ts_utc`,
    MIN(`meta_load_timestamp_utc`) AS `min_meta_load_timestamp_utc`,
    MAX(`meta_load_timestamp_utc`) AS `max_meta_load_timestamp_utc`,
    MIN(`field_value`) AS `min_value`,
    MAX(`field_value`) AS `max_value`
  FROM weather_long
  GROUP BY
    `source_table`,
    `information_role`,
    NULLIF(TRIM(CAST(`measurement` AS STRING)), ''),
    NULLIF(TRIM(CAST(`data_entity` AS STRING)), ''),
    NULLIF(TRIM(CAST(`data_point` AS STRING)), ''),
    NULLIF(TRIM(CAST(`data_point_type` AS STRING)), ''),
    NULLIF(TRIM(CAST(`data_source` AS STRING)), ''),
    NULLIF(TRIM(CAST(`sampling_frequency` AS STRING)), ''),
    NULLIF(TRIM(CAST(`unit` AS STRING)), ''),
    NULLIF(TRIM(CAST(`lead_time` AS STRING)), ''),
    NULLIF(TRIM(CAST(`field` AS STRING)), ''),
    `is_historical`
)
SELECT
  *,
  COUNT(*) OVER () AS `profile_group_count`,
  SUM(`row_count`) OVER () AS `total_row_count`
FROM weather_profile
ORDER BY
  `source_table`,
  `information_role`,
  `measurement` ASC NULLS FIRST,
  `data_entity` ASC NULLS FIRST,
  `data_point` ASC NULLS FIRST,
  `field` ASC NULLS FIRST,
  `lead_time` ASC NULLS FIRST
LIMIT 20001
