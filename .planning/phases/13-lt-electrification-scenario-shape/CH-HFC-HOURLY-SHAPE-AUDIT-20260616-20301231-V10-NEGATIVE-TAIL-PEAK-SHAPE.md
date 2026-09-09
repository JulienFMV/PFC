# CH Hourly PFC Shape Audit

* CSV: `output\ch_hfc_hourly_20260616_20301231_v10_negative_tail_peak_shape.csv`
* score: `9.00/10`
* scope: `local/test only`
* production approval: `NO`

## Metrics

| score_10 | finite_ok | no_negative | bounded_negative_ok | min_price_eur_mwh | weighted_negative_hours | p10_negative_hours | quantile_order | max_eex_error_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh | duck_2030_evening_minus_midday_eur_mwh | weekend_2030_minus_weekday_eur_mwh | ramp_abs_p99_eur_mwh | ramp_abs_max_eur_mwh | boundary_jump_abs_p95_eur_mwh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.000000 | 1.000000 | 0.000000 | 1.000000 | -10.765951 | 0.000000 | 377.000000 | 1.000000 | 0.000000 | 9.738211 | 25.810730 | 25.836095 | -6.393944 | 28.568806 | 56.267610 | 18.926799 |

## Annual Shape

| year | mean_eur_mwh | evening_minus_midday_eur_mwh | weekend_minus_weekday_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh |
|---|---|---|---|---|---|
| 2026.000000 | 112.463747 | 44.698658 | -27.114608 | 9.228374 | 19.148539 |
| 2027.000000 | 95.803783 | 36.707208 | -16.867808 | 9.027498 | 19.895183 |
| 2028.000000 | 79.980000 | 26.306037 | -9.377026 | 10.229088 | 26.259616 |
| 2029.000000 | 71.900000 | 26.280835 | -7.023884 | 10.034068 | 26.877754 |
| 2030.000000 | 68.860000 | 25.836095 | -6.393944 | 9.938869 | 26.138945 |

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
| 2028-Q1 | 110.760000 | 110.760000 | 0.000000 | 2183 |
| 2028-RESIDUAL | 69.800824 | 69.800824 | 0.000000 | 6601 |
| 2029 | 71.900000 | 71.900000 | 0.000000 | 8760 |
| 2030 | 68.860000 | 68.860000 | 0.000000 | 8760 |

## Notes

* Score >= 8.5 is a local/test quality threshold, not production approval.
* Remaining high boundary/ramp metrics require upstream smoothing before production use.
