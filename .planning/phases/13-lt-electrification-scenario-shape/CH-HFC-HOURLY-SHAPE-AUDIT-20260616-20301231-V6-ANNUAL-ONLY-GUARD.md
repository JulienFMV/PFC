# CH Hourly PFC Shape Audit

* CSV: `output\ch_hfc_hourly_20260616_20301231_v6_annual_only_guard.csv`
* score: `8.25/10`
* scope: `local/test only`
* production approval: `NO`

## Metrics

| score_10 | finite_ok | no_negative | bounded_negative_ok | min_price_eur_mwh | weighted_negative_hours | p10_negative_hours | quantile_order | max_eex_error_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh | duck_2030_evening_minus_midday_eur_mwh | weekend_2030_minus_weekday_eur_mwh | ramp_abs_p99_eur_mwh | ramp_abs_max_eur_mwh | boundary_jump_abs_p95_eur_mwh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.250000 | 1.000000 | 0.000000 | 1.000000 | -2.281212 | 0.000000 | 45.000000 | 1.000000 | 0.000000 | 5.928600 | 17.436172 | 24.168514 | -5.366417 | 27.632721 | 56.267610 | 18.592879 |

## Annual Shape

| year | mean_eur_mwh | evening_minus_midday_eur_mwh | weekend_minus_weekday_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh |
|---|---|---|---|---|---|
| 2026.000000 | 112.463747 | 43.692997 | -26.249959 | 5.694990 | 13.073276 |
| 2027.000000 | 95.803783 | 35.705518 | -16.002975 | 5.531274 | 13.846984 |
| 2028.000000 | 79.980000 | 24.576267 | -8.455037 | 6.148032 | 18.486546 |
| 2029.000000 | 71.900000 | 24.543884 | -6.018135 | 6.094592 | 18.635138 |
| 2030.000000 | 68.860000 | 24.168514 | -5.366417 | 6.067294 | 18.723712 |

## EEX Residuals

| product | target_eex_base_eur_mwh | csv_mean_eur_mwh | abs_error_eur_mwh | rows |
|---|---|---|---|---|
| 2026-06 | 95.590000 | 95.590000 | 0.000000 | 360 |
| 2026-07 | 93.950000 | 93.950000 | 0.000000 | 744 |
| 2026-08 | 92.860000 | 92.860000 | 0.000000 | 744 |
| 2026-09 | 105.480000 | 105.480000 | 0.000000 | 720 |
| 2026-10 | 116.680000 | 116.680000 | 0.000000 | 745 |
| 2026-11 | 138.000000 | 138.000000 | 0.000000 | 720 |
| 2026-12 | 136.570000 | 136.570000 | 0.000000 | 744 |
| 2027-Q1 | 128.380000 | 128.380000 | 0.000000 | 2159 |
| 2027-Q2 | 72.540000 | 72.540000 | 0.000000 | 2184 |
| 2027-Q3 | 75.480000 | 75.480000 | 0.000000 | 2208 |
| 2027-Q4 | 107.280000 | 107.280000 | 0.000000 | 2209 |
| 2028 | 79.980000 | 79.980000 | 0.000000 | 8784 |
| 2029 | 71.900000 | 71.900000 | 0.000000 | 8760 |
| 2030 | 68.860000 | 68.860000 | 0.000000 | 8760 |

## Notes

* Score >= 8.5 is a local/test quality threshold, not production approval.
* Remaining high boundary/ramp metrics require upstream smoothing before production use.
