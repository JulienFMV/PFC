-- Profiling exact du bilan de zone et de l'énergie d'équilibrage Swissgrid.
-- Ne pas exécuter si le Warehouse est arrêté. Les unités et la convention de
-- signe doivent être confirmées par le propriétaire ; cette requête ne les infère pas.
-- NON AUTORISÉ EN L'ÉTAT : LIMIT borne les lignes retournées, pas les octets lus.
-- Confirmer taille Delta, partitions et plan estimé avant toute exécution.

WITH cab AS (
  SELECT
    `ts_start_utc`,
    `afrr_plus`,
    `afrr_moins`,
    `nrv_plus_import`,
    `nrv_moins_export`,
    `sa_mfrr_plus`,
    `sa_mfrr_moins`,
    `da_mfrr_plus`,
    `da_mfrr_moins`,
    `rr_plus`,
    `rr_moins`,
    `frce_plus_import`,
    `frce_moins_export`,
    `total_system_imbalance`,
    `ae_price_long`,
    `ae_price_short`,
    `ae_price_annual`,
    `_file_date`,
    `_file_time`,
    `_ingest_ts`,
    `_silver_ts`
  FROM `prd`.`silver`.`ge_power_swissgrid_cab`
), duplicate_targets AS (
  SELECT
    `ts_start_utc`,
    COUNT(*) AS `row_count`
  FROM cab
  GROUP BY `ts_start_utc`
  HAVING COUNT(*) > 1
), duplicate_summary AS (
  SELECT
    COUNT(*) AS `duplicated_target_count`,
    COALESCE(SUM(`row_count` - 1), 0) AS `excess_duplicate_row_count`
  FROM duplicate_targets
)
SELECT
  COUNT(*) AS `row_count`,
  MIN(`ts_start_utc`) AS `min_target_start_utc`,
  MAX(`ts_start_utc`) AS `max_target_start_utc`,
  MIN(`_ingest_ts`) AS `min_ingest_ts_utc`,
  MAX(`_ingest_ts`) AS `max_ingest_ts_utc`,
  MIN(`_silver_ts`) AS `min_silver_ts_utc`,
  MAX(`_silver_ts`) AS `max_silver_ts_utc`,
  COUNT_IF(`ts_start_utc` IS NULL) AS `null_target_count`,
  COUNT_IF(
    `ts_start_utc` IS NOT NULL
    AND PMOD(UNIX_MICROS(`ts_start_utc`), 900000000) <> 0
  ) AS `invalid_quarter_hour_alignment_count`,
  COUNT_IF(`_ingest_ts` IS NULL) AS `null_ingest_count`,
  COUNT_IF(`afrr_plus` IS NOT NULL) AS `afrr_plus_observed_count`,
  COUNT_IF(`afrr_moins` IS NOT NULL) AS `afrr_minus_observed_count`,
  COUNT_IF(`sa_mfrr_plus` IS NOT NULL) AS `scheduled_mfrr_plus_observed_count`,
  COUNT_IF(`sa_mfrr_moins` IS NOT NULL) AS `scheduled_mfrr_minus_observed_count`,
  COUNT_IF(`da_mfrr_plus` IS NOT NULL) AS `direct_mfrr_plus_observed_count`,
  COUNT_IF(`da_mfrr_moins` IS NOT NULL) AS `direct_mfrr_minus_observed_count`,
  COUNT_IF(`rr_plus` IS NOT NULL) AS `rr_plus_observed_count`,
  COUNT_IF(`rr_moins` IS NOT NULL) AS `rr_minus_observed_count`,
  COUNT_IF(`total_system_imbalance` IS NULL) AS `null_system_imbalance_count`,
  COUNT_IF(`ae_price_long` IS NOT NULL) AS `long_price_observed_count`,
  COUNT_IF(`ae_price_short` IS NOT NULL) AS `short_price_observed_count`,
  COUNT_IF(`ae_price_annual` IS NOT NULL) AS `one_price_observed_count`,
  MIN(`total_system_imbalance`) AS `min_system_imbalance`,
  MAX(`total_system_imbalance`) AS `max_system_imbalance`,
  MIN(`ae_price_annual`) AS `min_one_price`,
  MAX(`ae_price_annual`) AS `max_one_price`,
  COALESCE(MAX(duplicate_summary.`duplicated_target_count`), 0)
    AS `duplicated_target_count`,
  COALESCE(MAX(duplicate_summary.`excess_duplicate_row_count`), 0)
    AS `excess_duplicate_row_count`
FROM cab
CROSS JOIN duplicate_summary
LIMIT 1
