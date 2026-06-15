# CH Hourly PFC Shape Audit

* CSV: `output\ch_pfc_hourly_20260613_20301231_v5_negative_capture.csv`
* score: `9.00/10`
* scope: `local/test only`
* production approval: `NO`

## Metrics

| score_10 | finite_ok | no_negative | bounded_negative_ok | min_price_eur_mwh | weighted_negative_hours | p10_negative_hours | quantile_order | max_eex_error_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh | duck_2030_evening_minus_midday_eur_mwh | weekend_2030_minus_weekday_eur_mwh | ramp_abs_p99_eur_mwh | ramp_abs_max_eur_mwh | boundary_jump_abs_p95_eur_mwh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.000000 | 1.000000 | 0.000000 | 1.000000 | -2.503237 | 0.000000 | 82.000000 | 1.000000 | 0.000000 | 9.290561 | 23.459181 | 23.161095 | -5.261520 | 27.894435 | 62.387531 | 18.945026 |

## Annual Shape

| year | mean_eur_mwh | evening_minus_midday_eur_mwh | weekend_minus_weekday_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh |
|---|---|---|---|---|---|
| 2026.000000 | 116.058435 | 45.476579 | -26.973787 | 9.446614 | 19.657782 |
| 2027.000000 | 97.186247 | 35.880795 | -16.091681 | 9.232539 | 20.817575 |
| 2028.000000 | 80.440000 | 24.537391 | -8.546115 | 9.440830 | 24.566624 |
| 2029.000000 | 72.640000 | 23.519924 | -5.754671 | 9.251381 | 24.037435 |
| 2030.000000 | 69.290000 | 23.161095 | -5.261520 | 9.150699 | 23.375722 |

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
