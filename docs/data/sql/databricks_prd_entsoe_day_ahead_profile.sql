-- Value-blind PRD acceptance profile for the five ENTSO-E day-ahead prices.
-- Parameters are bound by the Databricks client; do not interpolate strings.
-- The explicit year/month bounds are a cost fence for the Silver Delta partitions.
WITH parameters AS (
  SELECT
    CAST(:start_utc AS TIMESTAMP) AS start_utc,
    CAST(:end_utc AS TIMESTAMP) AS end_utc,
    CAST(:delivery_year AS INT) AS delivery_year,
    CAST(:delivery_month AS INT) AS delivery_month
),
gold_scope AS (
  SELECT
    SeriesID,
    SeriesKey,
    FieldName,
    ClassificationSequence,
    Unit,
    DocumentType
  FROM prd.gold.dimentsoeseries
  WHERE GroupName = 'day_ahead_prices'
    AND FieldName IN ('ch_price', 'at_price', 'de_lu_price', 'fr_price', 'it_nord_price')
),
silver_scope AS (
  SELECT
    v.SK_ge_power_entsoe_time_series_vintages AS vintage_id,
    v.series_key,
    v.field_name,
    v.classification_sequence,
    v.field_value,
    v.IntervalStartUtc AS interval_start_utc,
    v.IntervalEndUtc AS interval_end_utc,
    v.Date_Time_UTC AS date_time_utc,
    v.resolution,
    v.publication_timestamp_utc,
    v.first_seen_pull_ts_utc,
    v.last_seen_pull_ts_utc,
    v.availability_known,
    v.availability_timestamp_utc,
    v.dq_failed
  FROM prd.silver.ge_power_entsoe_time_series_vintages AS v
  CROSS JOIN parameters AS p
  WHERE v.group_name = 'day_ahead_prices'
    AND v.field_name IN ('ch_price', 'at_price', 'de_lu_price', 'fr_price', 'it_nord_price')
    AND v._year = p.delivery_year
    AND v._month = p.delivery_month
    AND v.IntervalStartUtc >= p.start_utc
    AND v.IntervalStartUtc < p.end_utc
),
series_stats AS (
  SELECT
    field_name,
    series_key,
    classification_sequence,
    resolution,
    COUNT(*) AS vintage_row_count,
    COUNT(DISTINCT interval_start_utc) AS distinct_interval_count,
    MIN(interval_start_utc) AS min_interval_start_utc,
    MAX(interval_end_utc) AS max_interval_end_utc,
    SUM(CASE WHEN field_value IS NULL THEN 1 ELSE 0 END) AS null_value_count,
    SUM(CASE WHEN COALESCE(dq_failed, TRUE) THEN 1 ELSE 0 END) AS dq_failed_count,
    SUM(
      CASE
        WHEN COALESCE(availability_known, FALSE) = FALSE
          OR availability_timestamp_utc IS NULL
        THEN 1 ELSE 0
      END
    ) AS unknown_availability_count,
    SUM(
      CASE
        WHEN publication_timestamp_utc IS NULL
          OR first_seen_pull_ts_utc IS NULL
          OR last_seen_pull_ts_utc IS NULL
          OR publication_timestamp_utc > first_seen_pull_ts_utc
          OR first_seen_pull_ts_utc > last_seen_pull_ts_utc
        THEN 1 ELSE 0
      END
    ) AS invalid_availability_order_count,
    SUM(
      CASE
        WHEN interval_start_utc IS NULL
          OR interval_end_utc IS NULL
          OR date_time_utc IS NULL
          OR interval_end_utc <> date_time_utc
          OR interval_start_utc >= interval_end_utc
          OR UNIX_TIMESTAMP(interval_end_utc) - UNIX_TIMESTAMP(interval_start_utc)
            <> CASE resolution
              WHEN 'PT15M' THEN 900
              WHEN 'PT30M' THEN 1800
              WHEN 'PT60M' THEN 3600
              WHEN 'PT1H' THEN 3600
              ELSE -1
            END
        THEN 1 ELSE 0
      END
    ) AS invalid_interval_count,
    SUM(
      CASE
        WHEN series_key <> CONCAT(
          'day_ahead_prices||',
          field_name,
          CASE
            WHEN classification_sequence IS NOT NULL
              AND TRIM(classification_sequence) <> ''
            THEN CONCAT('||', TRIM(classification_sequence))
            ELSE ''
          END
        )
        THEN 1 ELSE 0
      END
    ) AS canonical_series_key_mismatch_count
  FROM silver_scope
  GROUP BY field_name, series_key, classification_sequence, resolution
),
duplicate_vintage AS (
  SELECT COALESCE(SUM(rows_per_key - 1), 0) AS duplicate_vintage_key_count
  FROM (
    SELECT vintage_id, COUNT(*) AS rows_per_key
    FROM silver_scope
    GROUP BY vintage_id
    HAVING COUNT(*) > 1
  )
),
orphan_series AS (
  SELECT COUNT(*) AS orphan_series_key_count
  FROM silver_scope AS s
  LEFT ANTI JOIN gold_scope AS g ON s.series_key = g.SeriesKey
),
gold_duplicates AS (
  SELECT COALESCE(SUM(rows_per_key - 1), 0) AS gold_series_key_duplicate_count
  FROM (
    SELECT SeriesKey, COUNT(*) AS rows_per_key
    FROM gold_scope
    GROUP BY SeriesKey
    HAVING COUNT(*) > 1
  )
),
latest_duplicates AS (
  SELECT COALESCE(SUM(rows_per_grain - 1), 0) AS latest_grain_duplicate_count
  FROM (
    SELECT
      l.SeriesID,
      l.IntervalStartUtc,
      l.DateTimeUtc,
      COUNT(*) AS rows_per_grain
    FROM prd.gold.factentsoetimeserieslatest AS l
    INNER JOIN gold_scope AS g ON l.SeriesID = g.SeriesID
    CROSS JOIN parameters AS p
    WHERE l.IntervalStartUtc >= p.start_utc
      AND l.IntervalStartUtc < p.end_utc
    GROUP BY l.SeriesID, l.IntervalStartUtc, l.DateTimeUtc
    HAVING COUNT(*) > 1
  )
),
legacy_new_overlap AS (
  SELECT COUNT(*) AS legacy_new_overlap_interval_count
  FROM (
    SELECT field_name, interval_start_utc, date_time_utc
    FROM silver_scope
    GROUP BY field_name, interval_start_utc, date_time_utc
    HAVING MAX(
      CASE
        WHEN (classification_sequence IS NULL OR TRIM(classification_sequence) = '')
          AND series_key = CONCAT('day_ahead_prices||', field_name)
        THEN 1 ELSE 0
      END
    ) = 1
    AND MAX(
      CASE
        WHEN classification_sequence IS NOT NULL
          AND TRIM(classification_sequence) <> ''
          AND series_key = CONCAT(
            'day_ahead_prices||', field_name, '||', TRIM(classification_sequence)
          )
        THEN 1 ELSE 0
      END
    ) = 1
  )
),
profile_rows AS (
  SELECT
    g.FieldName AS field_name,
    g.SeriesKey AS series_key,
    g.ClassificationSequence AS classification_sequence,
    g.Unit AS unit,
    g.DocumentType AS document_type,
    g.SeriesID AS series_id,
    s.resolution,
    COALESCE(s.vintage_row_count, 0) AS vintage_row_count,
    COALESCE(s.distinct_interval_count, 0) AS distinct_interval_count,
    s.min_interval_start_utc,
    s.max_interval_end_utc,
    COALESCE(s.null_value_count, 0) AS null_value_count,
    COALESCE(s.dq_failed_count, 0) AS dq_failed_count,
    COALESCE(s.unknown_availability_count, 0) AS unknown_availability_count,
    COALESCE(s.invalid_availability_order_count, 0) AS invalid_availability_order_count,
    COALESCE(s.invalid_interval_count, 0) AS invalid_interval_count,
    COALESCE(s.canonical_series_key_mismatch_count, 0)
      AS canonical_series_key_mismatch_count
  FROM gold_scope AS g
  LEFT JOIN series_stats AS s ON g.SeriesKey = s.series_key
),
global_checks AS (
  SELECT
    COUNT(*) AS profile_row_count,
    d.duplicate_vintage_key_count,
    o.orphan_series_key_count,
    g.gold_series_key_duplicate_count,
    l.latest_grain_duplicate_count,
    n.legacy_new_overlap_interval_count
  FROM profile_rows
  CROSS JOIN duplicate_vintage AS d
  CROSS JOIN orphan_series AS o
  CROSS JOIN gold_duplicates AS g
  CROSS JOIN latest_duplicates AS l
  CROSS JOIN legacy_new_overlap AS n
  GROUP BY ALL
)
SELECT
  p.*,
  c.profile_row_count,
  c.duplicate_vintage_key_count,
  c.orphan_series_key_count,
  c.gold_series_key_duplicate_count,
  c.latest_grain_duplicate_count,
  c.legacy_new_overlap_interval_count
FROM profile_rows AS p
CROSS JOIN global_checks AS c
ORDER BY p.field_name, p.series_key, p.resolution
LIMIT 101
