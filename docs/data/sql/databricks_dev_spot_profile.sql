-- Profiling exact de la surface spot candidate.
-- Ne pas exécuter si le Warehouse est arrêté. Une exécution = un scan borné
-- aux colonnes ci-dessous ; rapatrier ensuite le résultat et travailler localement.
-- NON AUTORISÉ EN L'ÉTAT : LIMIT borne les lignes retournées, pas les octets lus.
-- Confirmer taille Delta, partitions et plan estimé avant toute exécution.

WITH spot_profile AS (
  SELECT
    NULLIF(TRIM(CAST(`source_product` AS STRING)), '') AS `source_product`,
    NULLIF(TRIM(CAST(`curve_id` AS STRING)), '') AS `curve_id`,
    NULLIF(TRIM(CAST(`frequency_code` AS STRING)), '') AS `frequency_code`,
    `frequency_minutes`,
    NULLIF(TRIM(CAST(`price_unit` AS STRING)), '') AS `price_unit`,
    COUNT(*) AS `row_count`,
    COUNT(DISTINCT `source_row_id`) AS `distinct_source_row_id_count`,
    COUNT_IF(`source_row_id` IS NULL) AS `null_source_row_id_count`,
    COUNT_IF(`price` IS NULL) AS `null_price_count`,
    COUNT_IF(`frequency_minutes` IS NULL OR `frequency_minutes` <= 0)
      AS `invalid_frequency_count`,
    COUNT_IF(`ts_start_utc` IS NULL OR `ts_end_utc` IS NULL)
      AS `null_interval_count`,
    COUNT_IF(
      `frequency_minutes` IS NOT NULL
      AND `ts_start_utc` IS NOT NULL
      AND `ts_end_utc` IS NOT NULL
      AND `ts_end_utc`
        <> TIMESTAMPADD(MINUTE, `frequency_minutes`, `ts_start_utc`)
    ) AS `interval_duration_mismatch_count`,
    MIN(`quotation_ts`) AS `min_quotation_ts_utc`,
    MAX(`quotation_ts`) AS `max_quotation_ts_utc`,
    MIN(`ts_start_utc`) AS `min_delivery_start_utc`,
    MAX(`ts_end_utc`) AS `max_delivery_end_utc`,
    MIN(`_ingest_ts`) AS `min_ingest_ts_utc`,
    MAX(`_ingest_ts`) AS `max_ingest_ts_utc`,
    MIN(`price`) AS `min_price`,
    MAX(`price`) AS `max_price`
  FROM `dev`.`silver`.`ge_market_euler_spot`
  GROUP BY
    NULLIF(TRIM(CAST(`source_product` AS STRING)), ''),
    NULLIF(TRIM(CAST(`curve_id` AS STRING)), ''),
    NULLIF(TRIM(CAST(`frequency_code` AS STRING)), ''),
    `frequency_minutes`,
    NULLIF(TRIM(CAST(`price_unit` AS STRING)), '')
)
SELECT
  *,
  COUNT(*) OVER () AS `profile_group_count`,
  SUM(`row_count`) OVER () AS `total_row_count`
FROM spot_profile
ORDER BY
  `source_product` ASC NULLS FIRST,
  `curve_id` ASC NULLS FIRST,
  `frequency_minutes` ASC NULLS FIRST,
  `price_unit` ASC NULLS FIRST
LIMIT 10001
