# PFC QA - Report specification

Objectif: auditer rapidement la qualite d'une HFC/PFC CH horaire calibree EEX
BASE+PEAK, sans Excel.

Source principale:

```text
powerbi\data\hfc_hourly_powerbi.csv
```

## Page 1 - Executive QA

### Slicers

| slicer | table | champ |
|---|---|---|
| Delivery year | `HFC_Hourly` | `year` |
| Season | `HFC_Hourly` | `season` |
| Month | `HFC_Hourly` | `month_label` |

### Cards

| titre | mesure |
|---|---|
| Avg weighted price | `Avg Price Weighted` |
| Min weighted price | `Min Weighted Price` |
| Weighted negative hours | `Weighted Negative Hours` |
| Fast negative hours | `Fast Negative Hours` |
| EEX BASE residual | `EEX Max BASE Residual` |
| EEX PEAK residual | `EEX Max PEAK Residual` |
| Peak/offpeak spread | `Peak Offpeak Spread` |
| Evening/midday spread | `Evening Midday Spread` |

### Table

Table `Summary_Metrics`:

- `metric`
- `value`

Conditional colors:

- `EEX Max BASE Residual` > 0.01: red
- `EEX Max PEAK Residual` > 0.01: red
- `Weighted Negative Hours` > 0: red
- `shape_score_10` < 8.5: orange/red
- `hfc_vs_spot_score_10` < 8.5: orange/red

## Page 2 - EEX Calibration

### Matrix

Table: `EEX_Residuals`

Rows:

- `load_type`
- `product`

Values:

- `target_eex_eur_mwh`
- `csv_mean_eur_mwh`
- `abs_error_eur_mwh`
- `rows`

Conditional formatting:

- `abs_error_eur_mwh` green <= 0.01
- orange > 0.01 and <= 0.10
- red > 0.10

### Bar chart

- Axis: `EEX_Residuals[product]`
- Legend: `EEX_Residuals[load_type]`
- Values: `EEX_Residuals[abs_error_eur_mwh]`

## Page 3 - Duck Curves

### Monthly duck curve

Line chart:

- X axis: `Duck_Month_Hour[hour]`
- Legend: `Duck_Month_Hour[month]`
- Y: average `price_weighted_mean_eur_mwh`
- Slicer: `Duck_Month_Hour[year]`

### Scenario duck curve

Line chart:

- X axis: `HFC_Hourly[hour]`
- Y:
  - average `price_slow_eur_mwh`
  - average `price_central_eur_mwh`
  - average `price_fast_eur_mwh`
  - average `price_weighted_mean_eur_mwh`
- Slicers: `year`, `month`, `season`

## Page 4 - Seasonality

### Monthly means

Line/column chart:

- X axis: `Period_Means[month]`
- Legend: `Period_Means[year]`
- Y: average `price_weighted_mean_eur_mwh`

### Annual shape table

Table: `Annual_Shape`

Fields:

- `year`
- `mean_eur_mwh`
- `evening_minus_midday_eur_mwh`
- `weekend_minus_weekday_eur_mwh`
- `structural_width_mean_eur_mwh`
- `structural_width_p95_eur_mwh`

## Page 5 - Negative Tail

### Negative-hour heatmap

Matrix:

- Rows: `Negative_Low_Hours[month]`
- Columns: `Negative_Low_Hours[hour]`
- Values:
  - `fast_negative_hours`
  - `p10_negative_hours`

Conditional formatting:

- 0: white
- >0: red scale

### Low-price diagnostics

Table: `Negative_Low_Hours`

Fields:

- `year`
- `month`
- `hour`
- `fast_negative_hours`
- `p10_negative_hours`
- `min_fast_eur_mwh`
- `min_p10_eur_mwh`

## Page 6 - Structural Fan

### P10/P50/P90 over time

Line chart:

- X axis: `HFC_Hourly[timestamp_ch_iso]`
- Y:
  - average `structural_p10_eur_mwh`
  - average `structural_p50_eur_mwh`
  - average `structural_p90_eur_mwh`
  - average `price_weighted_mean_eur_mwh`
- Slicers: `year`, `month`

### Structural width by hour

Line chart:

- X axis: `HFC_Hourly[hour]`
- Y: average `structural_width_eur_mwh`
- Legend: `season`
- Slicer: `year`

## Page 7 - Holiday / EEX Peak

### Peak vs offpeak

Clustered column chart:

- Axis: `year`
- Legend: `is_eex_peak`
- Values: average `price_weighted_mean_eur_mwh`

### Weekend effect

Clustered column chart:

- Axis: `year`
- Legend: `is_weekend`
- Values: average `price_weighted_mean_eur_mwh`

## Page 8 - Data

Table visual over `HFC_Hourly`, with filters enabled:

- `timestamp_ch_iso`
- `price_weighted_mean_eur_mwh`
- `price_central_eur_mwh`
- `price_fast_eur_mwh`
- `structural_p10_eur_mwh`
- `structural_p90_eur_mwh`
- `is_eex_peak`
- `is_fast_negative`

This page is for drill-down only.
