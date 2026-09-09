-- Consumer-complete, one-month ENTSO-E day-ahead export known at one origin.
-- A result with 20,001 rows is a rejection sentinel, not a usable extract.
WITH parameters AS (
  SELECT
    CAST(:start_utc AS TIMESTAMP) AS start_utc,
    CAST(:end_utc AS TIMESTAMP) AS end_utc,
    CAST(:as_of_utc AS TIMESTAMP) AS as_of_utc,
    CAST(:partition_start_year AS STRING) AS partition_start_year,
    LPAD(CAST(:partition_start_month AS STRING), 2, '0') AS partition_start_month,
    CAST(:partition_end_year AS STRING) AS partition_end_year,
    LPAD(CAST(:partition_end_month AS STRING), 2, '0') AS partition_end_month
),
selected_series AS (
  SELECT field_name, series_key
  FROM VALUES
    ('ch_price', :ch_series_key),
    ('at_price', :at_series_key),
    ('de_lu_price', :de_lu_series_key),
    ('fr_price', :fr_series_key),
    ('it_nord_price', :it_nord_series_key)
  AS selected(field_name, series_key)
  WHERE series_key IS NOT NULL
),
eligible AS (
  SELECT
    s.field_name,
    v.series_key,
    v.classification_sequence,
    v.IntervalStartUtc AS interval_start_utc,
    v.Date_Time_UTC AS date_time_utc,
    v.IntervalEndUtc AS interval_end_utc,
    v.resolution,
    CAST(v.field_value AS DOUBLE) AS price_eur_per_mwh,
    v.publication_timestamp_utc,
    v.first_seen_pull_ts_utc,
    v.availability_basis,
    v.availability_known,
    v.availability_timestamp_utc,
    v.Is_Historical AS is_historical,
    v.dq_failed,
    v.source_time_series_id,
    v.source_document_mrid,
    v.source_document_revision_number,
    v.source_snapshot_id,
    v.source_file_path,
    v.SK_ge_power_entsoe_time_series_vintages AS vintage_id
  FROM prd.silver.ge_power_entsoe_time_series_vintages AS v
  INNER JOIN selected_series AS s
    ON v.series_key = s.series_key
    AND v.field_name = s.field_name
  CROSS JOIN parameters AS p
  WHERE v.group_name = 'day_ahead_prices'
    AND v.document_type = 'A44'
    AND v.unit = 'EUR/MWh'
    AND (
      (v._year = p.partition_start_year AND v._month = p.partition_start_month)
      OR
      (v._year = p.partition_end_year AND v._month = p.partition_end_month)
    )
    AND v.IntervalStartUtc >= p.start_utc
    AND v.IntervalStartUtc < p.end_utc
    AND v.IntervalEndUtc <= p.end_utc
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
        vintage_id DESC
    ) AS vintage_rank
  FROM eligible
)
SELECT
  field_name,
  series_key,
  classification_sequence,
  interval_start_utc,
  date_time_utc,
  interval_end_utc,
  resolution,
  price_eur_per_mwh,
  publication_timestamp_utc,
  first_seen_pull_ts_utc,
  availability_basis,
  availability_known,
  availability_timestamp_utc,
  is_historical,
  dq_failed,
  source_time_series_id,
  source_document_mrid,
  source_document_revision_number,
  source_snapshot_id,
  source_file_path,
  vintage_id
FROM ranked
WHERE vintage_rank = 1
ORDER BY field_name, interval_start_utc, interval_end_utc
LIMIT 20001
