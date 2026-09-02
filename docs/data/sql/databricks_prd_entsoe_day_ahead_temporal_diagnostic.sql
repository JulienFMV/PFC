-- Value-blind temporal root-cause diagnostic for ENTSO-E day-ahead prices.
-- Parameters are bound natively; the exact year/month predicates prune Silver.
WITH parameters AS (
  SELECT
    CAST(:start_utc AS TIMESTAMP) AS start_utc,
    CAST(:end_utc AS TIMESTAMP) AS end_utc,
    CAST(:delivery_year AS INT) AS delivery_year,
    CAST(:delivery_month AS INT) AS delivery_month
),
silver_scope AS (
  SELECT
    v.field_name,
    v.series_key,
    v.classification_sequence,
    v.resolution,
    v.IntervalStartUtc AS interval_start_utc,
    v.IntervalEndUtc AS interval_end_utc,
    v.Date_Time_UTC AS date_time_utc,
    v.publication_timestamp_utc,
    v.first_seen_pull_ts_utc,
    v.last_seen_pull_ts_utc
  FROM prd.silver.ge_power_entsoe_time_series_vintages AS v
  CROSS JOIN parameters AS p
  WHERE v.group_name = 'day_ahead_prices'
    AND v.field_name IN ('ch_price', 'at_price', 'de_lu_price', 'fr_price', 'it_nord_price')
    AND v._year = p.delivery_year
    AND v._month = p.delivery_month
    AND v.IntervalStartUtc >= p.start_utc
    AND v.IntervalStartUtc < p.end_utc
),
diagnostic AS (
  SELECT
    field_name,
    series_key,
    classification_sequence,
    resolution,
    COUNT(*) AS row_count,
    SUM(CASE WHEN publication_timestamp_utc IS NULL THEN 1 ELSE 0 END)
      AS publication_timestamp_null_count,
    SUM(CASE WHEN first_seen_pull_ts_utc IS NULL THEN 1 ELSE 0 END)
      AS first_seen_null_count,
    SUM(CASE WHEN last_seen_pull_ts_utc IS NULL THEN 1 ELSE 0 END)
      AS last_seen_null_count,
    SUM(
      CASE
        WHEN publication_timestamp_utc IS NOT NULL
          AND first_seen_pull_ts_utc IS NOT NULL
          AND publication_timestamp_utc > first_seen_pull_ts_utc
        THEN 1 ELSE 0
      END
    ) AS publication_after_first_seen_count,
    SUM(
      CASE
        WHEN first_seen_pull_ts_utc IS NOT NULL
          AND last_seen_pull_ts_utc IS NOT NULL
          AND first_seen_pull_ts_utc > last_seen_pull_ts_utc
        THEN 1 ELSE 0
      END
    ) AS first_seen_after_last_seen_count,
    SUM(
      CASE
        WHEN publication_timestamp_utc IS NOT NULL
          AND interval_start_utc IS NOT NULL
          AND publication_timestamp_utc > interval_start_utc
        THEN 1 ELSE 0
      END
    ) AS publication_after_delivery_start_count,
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
    MIN(
      UNIX_TIMESTAMP(first_seen_pull_ts_utc) - UNIX_TIMESTAMP(publication_timestamp_utc)
    ) AS publication_to_first_seen_min_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(first_seen_pull_ts_utc) - UNIX_TIMESTAMP(publication_timestamp_utc),
      0.5,
      10000
    ) AS publication_to_first_seen_p50_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(first_seen_pull_ts_utc) - UNIX_TIMESTAMP(publication_timestamp_utc),
      0.95,
      10000
    ) AS publication_to_first_seen_p95_seconds,
    MAX(
      UNIX_TIMESTAMP(first_seen_pull_ts_utc) - UNIX_TIMESTAMP(publication_timestamp_utc)
    ) AS publication_to_first_seen_max_seconds,
    MIN(
      UNIX_TIMESTAMP(last_seen_pull_ts_utc) - UNIX_TIMESTAMP(first_seen_pull_ts_utc)
    ) AS first_seen_to_last_seen_min_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(last_seen_pull_ts_utc) - UNIX_TIMESTAMP(first_seen_pull_ts_utc),
      0.5,
      10000
    ) AS first_seen_to_last_seen_p50_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(last_seen_pull_ts_utc) - UNIX_TIMESTAMP(first_seen_pull_ts_utc),
      0.95,
      10000
    ) AS first_seen_to_last_seen_p95_seconds,
    MAX(
      UNIX_TIMESTAMP(last_seen_pull_ts_utc) - UNIX_TIMESTAMP(first_seen_pull_ts_utc)
    ) AS first_seen_to_last_seen_max_seconds,
    MIN(
      UNIX_TIMESTAMP(interval_start_utc) - UNIX_TIMESTAMP(publication_timestamp_utc)
    ) AS publication_to_delivery_min_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(interval_start_utc) - UNIX_TIMESTAMP(publication_timestamp_utc),
      0.5,
      10000
    ) AS publication_to_delivery_p50_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(interval_start_utc) - UNIX_TIMESTAMP(publication_timestamp_utc),
      0.95,
      10000
    ) AS publication_to_delivery_p95_seconds,
    MAX(
      UNIX_TIMESTAMP(interval_start_utc) - UNIX_TIMESTAMP(publication_timestamp_utc)
    ) AS publication_to_delivery_max_seconds,
    SUM(CASE WHEN interval_start_utc IS NULL THEN 1 ELSE 0 END)
      AS interval_start_null_count,
    SUM(CASE WHEN interval_end_utc IS NULL THEN 1 ELSE 0 END)
      AS interval_end_null_count,
    SUM(CASE WHEN date_time_utc IS NULL THEN 1 ELSE 0 END)
      AS date_time_null_count,
    SUM(
      CASE
        WHEN interval_end_utc IS NOT NULL
          AND date_time_utc IS NOT NULL
          AND interval_end_utc <> date_time_utc
        THEN 1 ELSE 0
      END
    ) AS interval_end_datetime_mismatch_count,
    SUM(
      CASE
        WHEN interval_start_utc IS NOT NULL
          AND interval_end_utc IS NOT NULL
          AND interval_start_utc >= interval_end_utc
        THEN 1 ELSE 0
      END
    ) AS interval_nonpositive_count,
    SUM(
      CASE WHEN resolution NOT IN ('PT15M', 'PT30M', 'PT60M', 'PT1H') OR resolution IS NULL
        THEN 1 ELSE 0
      END
    ) AS unsupported_resolution_count,
    SUM(
      CASE
        WHEN interval_start_utc IS NOT NULL
          AND interval_end_utc IS NOT NULL
          AND resolution IN ('PT15M', 'PT30M', 'PT60M', 'PT1H')
          AND UNIX_TIMESTAMP(interval_end_utc) - UNIX_TIMESTAMP(interval_start_utc)
            <> CASE resolution
              WHEN 'PT15M' THEN 900
              WHEN 'PT30M' THEN 1800
              WHEN 'PT60M' THEN 3600
              WHEN 'PT1H' THEN 3600
            END
        THEN 1 ELSE 0
      END
    ) AS duration_mismatch_count,
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
    MIN(UNIX_TIMESTAMP(interval_end_utc) - UNIX_TIMESTAMP(interval_start_utc))
      AS interval_duration_min_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(interval_end_utc) - UNIX_TIMESTAMP(interval_start_utc),
      0.5,
      10000
    ) AS interval_duration_p50_seconds,
    PERCENTILE_APPROX(
      UNIX_TIMESTAMP(interval_end_utc) - UNIX_TIMESTAMP(interval_start_utc),
      0.95,
      10000
    ) AS interval_duration_p95_seconds,
    MAX(UNIX_TIMESTAMP(interval_end_utc) - UNIX_TIMESTAMP(interval_start_utc))
      AS interval_duration_max_seconds
  FROM silver_scope
  GROUP BY field_name, series_key, classification_sequence, resolution
)
SELECT *
FROM diagnostic
ORDER BY field_name, series_key, resolution
LIMIT 101
