-- Bounded four-curve LSEG EPEX latest-observation extraction.
-- A result with 20,001 rows is a rejection sentinel, not a usable extract.
WITH parameters AS (
  SELECT
    CAST(:start_utc AS TIMESTAMP) AS start_utc,
    CAST(:end_utc AS TIMESTAMP) AS end_utc,
    CAST(:assessed_at_utc AS TIMESTAMP) AS assessed_at_utc,
    CAST(:start_value_date AS DATE) AS start_value_date,
    CAST(:end_value_date AS DATE) AS end_value_date
),
selected_curves AS (
  SELECT *
  FROM VALUES
    ('CH', '115688058', '1h', 'PT1H'),
    ('AT', '165444048', '15m', 'PT15M'),
    ('DE_LU', '165349556', '15m', 'PT15M'),
    ('FR', '165442712', '15m', 'PT15M')
  AS selected(market_zone, curve_id, expected_value_frequency, resolution)
)
SELECT
  s.market_zone,
  v.curve_id,
  v.value_start_timestamp AS interval_start_utc,
  v.value_timestamp AS interval_end_utc,
  s.resolution,
  CAST(v.value AS DOUBLE) AS price_eur_per_mwh,
  v.pipeline_first_seen_at_utc,
  v.pull_ts_utc,
  SHA2(
    CONCAT_WS(
      '||',
      COALESCE(CAST(v.curve_id AS STRING), '__NULL__'),
      COALESCE(CAST(v.scenario_id AS STRING), '__NULL__'),
      COALESCE(DATE_FORMAT(v.forecast_issued_at_utc, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"), '__NULL__'),
      COALESCE(DATE_FORMAT(v.value_start_timestamp, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"), '__NULL__'),
      COALESCE(DATE_FORMAT(v.value_timestamp, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"), '__NULL__'),
      COALESCE(CAST(v.value AS STRING), '__NULL__'),
      COALESCE(CAST(v.unit AS STRING), '__NULL__')
    ),
    256
  ) AS curve_value_vintage_id
FROM prd.silver.ge_market_lseg_curve_values AS v
INNER JOIN selected_curves AS s
  ON v.curve_id = s.curve_id
  AND v.value_frequency = s.expected_value_frequency
CROSS JOIN parameters AS p
WHERE v.value_date >= p.start_value_date
  AND v.value_date <= p.end_value_date
  AND v.value_start_timestamp >= p.start_utc
  AND v.value_start_timestamp < p.end_utc
  AND v.group_name = 'epex_actuals'
  AND v.schedule_group = 'daily'
  AND v.market = 'auction_day_ahead'
  AND v.value_type = 'price'
  AND v.unit = 'EUR/MWh'
  AND v.provider = 'EPEX Spot SE'
  AND v.scenario_id = 0
  AND v.forecast_issued_at_utc IS NULL
  AND v.pipeline_first_seen_at_utc IS NOT NULL
  AND v.pipeline_first_seen_at_utc <= p.assessed_at_utc
  AND v.pull_ts_utc <= p.assessed_at_utc
  AND v._silver_updated_ts <= p.assessed_at_utc
  AND COALESCE(v.dq_missing_curve_id, TRUE) = FALSE
  AND COALESCE(v.dq_missing_curve_name, TRUE) = FALSE
  AND COALESCE(v.dq_missing_value_timestamp, TRUE) = FALSE
  AND COALESCE(v.dq_missing_value, TRUE) = FALSE
  AND COALESCE(v.dq_missing_vendor_timezone, TRUE) = FALSE
ORDER BY s.market_zone, v.value_start_timestamp, v.value_timestamp
LIMIT 20001
