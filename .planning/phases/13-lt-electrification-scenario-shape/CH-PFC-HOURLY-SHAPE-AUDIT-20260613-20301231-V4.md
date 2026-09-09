# CH Hourly PFC Shape Audit

* CSV: `output\ch_pfc_hourly_20260613_20301231_v4_shape.csv`
* score: `9.00/10`
* scope: `local/test only`
* production approval: `NO`

## Metrics

| score_10 | finite_ok | no_negative | quantile_order | max_eex_error_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh | duck_2030_evening_minus_midday_eur_mwh | weekend_2030_minus_weekday_eur_mwh | ramp_abs_p99_eur_mwh | ramp_abs_max_eur_mwh | boundary_jump_abs_p95_eur_mwh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 8.676187 | 19.865102 | 20.308157 | -4.773017 | 27.707147 | 62.387532 | 18.945026 |

## Annual Shape

| year | mean_eur_mwh | evening_minus_midday_eur_mwh | weekend_minus_weekday_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh |
|---|---|---|---|---|---|
| 2026.000000 | 116.058435 | 45.406851 | -26.914490 | 9.342043 | 19.705343 |
| 2027.000000 | 97.186247 | 35.677409 | -16.022511 | 8.946145 | 20.759534 |
| 2028.000000 | 80.440000 | 22.408508 | -8.096602 | 8.669942 | 20.175567 |
| 2029.000000 | 72.640000 | 20.970468 | -5.261753 | 8.419745 | 19.839178 |
| 2030.000000 | 69.290000 | 20.308157 | -4.773017 | 8.300355 | 19.680366 |

## EEX Residuals

| product | target_eex_base_eur_mwh | csv_mean_eur_mwh | abs_error_eur_mwh | rows |
|---|---|---|---|---|
| 2026-06 | 96.880000 | 96.880000 | 0.000000 | 432 |
| 2026-07 | 97.440000 | 97.440000 | 0.000000 | 744 |
| 2026-08 | 95.510000 | 95.510000 | 0.000000 | 744 |
| 2026-09 | 110.500000 | 110.500000 | 0.000000 | 720 |
| 2026-10 | 120.470000 | 120.470000 | 0.000000 | 745 |
| 2026-11 | 142.990000 | 142.990000 | 0.000000 | 720 |
| 2026-12 | 141.260000 | 141.260000 | 0.000000 | 744 |
| 2027-Q1 | 132.610000 | 132.610000 | 0.000000 | 2159 |
| 2027-Q2 | 72.700000 | 72.700000 | 0.000000 | 2184 |
| 2027-Q3 | 75.820000 | 75.820000 | 0.000000 | 2208 |
| 2027-Q4 | 108.130000 | 108.130000 | 0.000000 | 2209 |
| 2028 | 80.440000 | 80.440000 | 0.000000 | 8784 |
| 2029 | 72.640000 | 72.640000 | 0.000000 | 8760 |
| 2030 | 69.290000 | 69.290000 | 0.000000 | 8760 |

## Notes

* Score >= 8.5 is a local/test quality threshold, not production approval.
* Remaining high boundary/ramp metrics require upstream smoothing before production use.
