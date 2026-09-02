-- One-month, five-series, point-in-time ENTSO-E day-ahead extraction.
-- A result with 20,001 rows is a rejection sentinel, not a usable extract.
WITH parameters AS (
  SELECT
    CAST(:start_utc AS TIMESTAMP) AS start_utc,
    CAST(:end_utc AS TIMESTAMP) AS end_utc,
    CAST(:as_of_utc AS TIMESTAMP) AS as_of_utc,
    CAST(:delivery_year AS INT) AS delivery_year,
    CAST(:delivery_month AS INT) AS delivery_month
),
selected_series AS (
  SELECT *
  FROM VALUES
    ('ch_price', :ch_series_key),
    ('at_price', :at_series_key),
    ('de_lu_price', :de_lu_series_key),
    ('fr_price', :fr_series_key),
    ('it_nord_price', :it_nord_series_key)
  AS selected(field_name, series_key)
),
eligible AS (
  SELECT
    s.field_name,
    v.series_key,
    v.IntervalStartUtc AS interval_start_utc,
    v.IntervalEndUtc AS interval_end_utc,
    v.resolution,
    CAST(v.field_value AS DOUBLE) AS price_eur_per_mwh,
    v.availability_timestamp_utc,
    v.source_document_mrid,
    v.source_document_revision_number,
    v.last_seen_pull_ts_utc,
    v.SK_ge_power_entsoe_time_series_vintages AS vintage_id
  FROM prd.silver.ge_power_entsoe_time_series_vintages AS v
  INNER JOIN selected_series AS s
    ON v.series_key = s.series_key
    AND v.field_name = s.field_name
  CROSS JOIN parameters AS p
  WHERE v.group_name = 'day_ahead_prices'
    AND v._year = p.delivery_year
    AND v._month = p.delivery_month
    AND v.IntervalStartUtc >= p.start_utc
    AND v.IntervalStartUtc < p.end_utc
    AND v.availability_known = TRUE
    AND v.availability_timestamp_utc IS NOT NULL
    AND v.availability_timestamp_utc <= p.as_of_utc
    AND COALESCE(v.dq_failed, TRUE) = FALSE
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY field_name, series_key, interval_start_utc, interval_end_utc
      ORDER BY
        availability_timestamp_utc DESC,
        source_document_revision_number DESC NULLS LAST,
        last_seen_pull_ts_utc DESC,
        vintage_id DESC
    ) AS vintage_rank
  FROM eligible
)
SELECT
  field_name,
  series_key,
  interval_start_utc,
  interval_end_utc,
  resolution,
  price_eur_per_mwh,
  availability_timestamp_utc,
  source_document_mrid,
  source_document_revision_number
FROM ranked
WHERE vintage_rank = 1
ORDER BY field_name, interval_start_utc, interval_end_utc
LIMIT 20001
