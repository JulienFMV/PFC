-- Inventaire exact des produits, pays, unités et snapshots des appels d'offres
-- Swissgrid SDL. Ne pas exécuter si le Warehouse est arrêté.
-- NON AUTORISÉ EN L'ÉTAT : LIMIT borne les lignes retournées, pas les octets lus.
-- Confirmer taille Delta, partitions et plan estimé avant toute exécution.

WITH tender_profile AS (
  SELECT
    `source_year`,
    NULLIF(TRIM(CAST(`ausschreibung` AS STRING)), '') AS `tender_code`,
    NULLIF(TRIM(CAST(`beschreibung` AS STRING)), '') AS `description`,
    NULLIF(TRIM(CAST(`land` AS STRING)), '') AS `country`,
    NULLIF(TRIM(CAST(`angebotenes_volumen_einheit` AS STRING)), '')
      AS `offered_volume_unit`,
    NULLIF(TRIM(CAST(`zugesprochenes_volumen_einheit` AS STRING)), '')
      AS `awarded_volume_unit`,
    NULLIF(TRIM(CAST(`leistungspreis_einheit` AS STRING)), '')
      AS `capacity_price_unit`,
    NULLIF(TRIM(CAST(`preis_einheit` AS STRING)), '') AS `price_unit`,
    NULLIF(TRIM(CAST(`angebotspreis_einheit` AS STRING)), '')
      AS `offer_price_unit`,
    COUNT(*) AS `offer_row_count`,
    COUNT_IF(`tender_row_id` IS NULL) AS `null_tender_row_id_count`,
    COUNT(DISTINCT `tender_row_id`) AS `distinct_tender_row_id_count`,
    COUNT_IF(`dq_tender_key_null`) AS `dq_tender_key_null_count`,
    COUNT_IF(`dq_missing_business_fields`) AS `dq_missing_business_fields_count`,
    COUNT_IF(`dq_negative_offer_volume`) AS `dq_negative_offer_volume_count`,
    COUNT_IF(`dq_negative_awarded_volume`) AS `dq_negative_awarded_volume_count`,
    COUNT_IF(`dq_negative_prices`) AS `dq_negative_prices_count`,
    COUNT_IF(`is_awarded`) AS `awarded_row_count`,
    MIN(`source_snapshot_ts`) AS `min_source_snapshot_ts_utc`,
    MAX(`source_snapshot_ts`) AS `max_source_snapshot_ts_utc`,
    MIN(`_ingest_ts`) AS `min_ingest_ts_utc`,
    MAX(`_ingest_ts`) AS `max_ingest_ts_utc`,
    MIN(`angebotenes_volumen`) AS `min_offered_volume`,
    MAX(`angebotenes_volumen`) AS `max_offered_volume`,
    MIN(`zugesprochenes_volumen`) AS `min_awarded_volume`,
    MAX(`zugesprochenes_volumen`) AS `max_awarded_volume`,
    MIN(`leistungspreis`) AS `min_capacity_price`,
    MAX(`leistungspreis`) AS `max_capacity_price`
  FROM `prd`.`silver`.`ge_market_swissgrid_sdl_tenders`
  GROUP BY
    `source_year`,
    NULLIF(TRIM(CAST(`ausschreibung` AS STRING)), ''),
    NULLIF(TRIM(CAST(`beschreibung` AS STRING)), ''),
    NULLIF(TRIM(CAST(`land` AS STRING)), ''),
    NULLIF(TRIM(CAST(`angebotenes_volumen_einheit` AS STRING)), ''),
    NULLIF(TRIM(CAST(`zugesprochenes_volumen_einheit` AS STRING)), ''),
    NULLIF(TRIM(CAST(`leistungspreis_einheit` AS STRING)), ''),
    NULLIF(TRIM(CAST(`preis_einheit` AS STRING)), ''),
    NULLIF(TRIM(CAST(`angebotspreis_einheit` AS STRING)), '')
)
SELECT
  *,
  COUNT(*) OVER () AS `profile_group_count`,
  SUM(`offer_row_count`) OVER () AS `total_offer_row_count`
FROM tender_profile
ORDER BY
  `source_year`,
  `tender_code` ASC NULLS FIRST,
  `country` ASC NULLS FIRST,
  `capacity_price_unit` ASC NULLS FIRST
LIMIT 20001
