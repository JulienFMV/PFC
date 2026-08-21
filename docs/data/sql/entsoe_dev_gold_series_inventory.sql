WITH series_signatures AS (
  SELECT
    NULLIF(TRIM(CAST(`GroupName` AS STRING)), '') AS `group_name`,
    NULLIF(TRIM(CAST(`FieldName` AS STRING)), '') AS `field_name`,
    NULLIF(TRIM(CAST(`DocumentType` AS STRING)), '') AS `document_type`,
    NULLIF(TRIM(CAST(`BusinessType` AS STRING)), '') AS `business_type`,
    NULLIF(TRIM(CAST(`ProcessType` AS STRING)), '') AS `process_type`,
    NULLIF(TRIM(CAST(`PsrType` AS STRING)), '') AS `psr_type`,
    NULLIF(TRIM(CAST(`FromZone` AS STRING)), '') AS `from_zone`,
    NULLIF(TRIM(CAST(`ToZone` AS STRING)), '') AS `to_zone`,
    NULLIF(TRIM(CAST(`Unit` AS STRING)), '') AS `unit`,
    COUNT(*) AS `series_count`,
    MIN(`Meta_Load_Timestamp`) AS `min_meta_load_timestamp_utc`,
    MAX(`Meta_Load_Timestamp`) AS `max_meta_load_timestamp_utc`
  FROM `dev`.`gold`.`dimentsoeseries`
  GROUP BY
    NULLIF(TRIM(CAST(`GroupName` AS STRING)), ''),
    NULLIF(TRIM(CAST(`FieldName` AS STRING)), ''),
    NULLIF(TRIM(CAST(`DocumentType` AS STRING)), ''),
    NULLIF(TRIM(CAST(`BusinessType` AS STRING)), ''),
    NULLIF(TRIM(CAST(`ProcessType` AS STRING)), ''),
    NULLIF(TRIM(CAST(`PsrType` AS STRING)), ''),
    NULLIF(TRIM(CAST(`FromZone` AS STRING)), ''),
    NULLIF(TRIM(CAST(`ToZone` AS STRING)), ''),
    NULLIF(TRIM(CAST(`Unit` AS STRING)), '')
), bounded_inventory AS (
  SELECT
    *,
    COUNT(*) OVER () AS `total_signature_count`,
    SUM(`series_count`) OVER () AS `total_series_count`
  FROM series_signatures
)
SELECT
  `group_name`,
  `field_name`,
  `document_type`,
  `business_type`,
  `process_type`,
  `psr_type`,
  `from_zone`,
  `to_zone`,
  `unit`,
  `series_count`,
  `min_meta_load_timestamp_utc`,
  `max_meta_load_timestamp_utc`,
  `total_signature_count`,
  `total_series_count`
FROM bounded_inventory
ORDER BY
  `group_name` ASC NULLS FIRST,
  `field_name` ASC NULLS FIRST,
  `document_type` ASC NULLS FIRST,
  `business_type` ASC NULLS FIRST,
  `process_type` ASC NULLS FIRST,
  `psr_type` ASC NULLS FIRST,
  `from_zone` ASC NULLS FIRST,
  `to_zone` ASC NULLS FIRST,
  `unit` ASC NULLS FIRST
LIMIT 10001
