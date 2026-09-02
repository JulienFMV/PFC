-- One-month, four-curve, point-in-time LSEG EPEX actual-price extraction.
-- A result with 20,001 rows is a rejection sentinel, not a usable extract.
WITH parameters AS (
  SELECT
    CAST(:start_utc AS TIMESTAMP) AS start_utc,
    CAST(:end_utc AS TIMESTAMP) AS end_utc,
    CAST(:as_of_utc AS TIMESTAMP) AS as_of_utc,
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
),
eligible AS (
  SELECT
    s.market_zone,
    v.curve_id,
    v.value_start_timestamp AS interval_start_utc,
    v.value_timestamp AS interval_end_utc,
    s.resolution,
    CAST(v.value AS DOUBLE) AS price_eur_per_mwh,
    v.pipeline_first_seen_at_utc,
    v.pull_ts_utc,
    v._silver_updated_ts,
    v._curve_value_vintage_id AS curve_value_vintage_id
  FROM prd.silver.ge_market_lseg_curve_value_vintages AS v
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
    AND v.pipeline_first_seen_at_utc <= p.as_of_utc
    AND COALESCE(v.dq_missing_curve_id, TRUE) = FALSE
    AND COALESCE(v.dq_missing_curve_name, TRUE) = FALSE
    AND COALESCE(v.dq_missing_value_timestamp, TRUE) = FALSE
    AND COALESCE(v.dq_missing_value, TRUE) = FALSE
    AND COALESCE(v.dq_missing_vendor_timezone, TRUE) = FALSE
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY market_zone, curve_id, interval_start_utc, interval_end_utc
      ORDER BY
        pipeline_first_seen_at_utc DESC,
        pull_ts_utc DESC,
        _silver_updated_ts DESC,
        curve_value_vintage_id DESC
    ) AS vintage_rank
  FROM eligible
)
SELECT
  market_zone,
  curve_id,
  interval_start_utc,
  interval_end_utc,
  resolution,
  price_eur_per_mwh,
  pipeline_first_seen_at_utc,
  pull_ts_utc,
  curve_value_vintage_id
FROM ranked
WHERE vintage_rank = 1
ORDER BY market_zone, interval_start_utc, interval_end_utc
LIMIT 20001
