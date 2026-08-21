# CH Hourly PFC Shape Audit

* CSV: `output\ch_hfc_hourly_20260616_20301231_v6_annual_only_guard_wide.csv`
* score: `9.00/10`
* scope: `local/test only`
* production approval: `NO`

## Metrics

| score_10 | finite_ok | no_negative | bounded_negative_ok | min_price_eur_mwh | weighted_negative_hours | p10_negative_hours | quantile_order | max_eex_error_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh | duck_2030_evening_minus_midday_eur_mwh | weekend_2030_minus_weekday_eur_mwh | ramp_abs_p99_eur_mwh | ramp_abs_max_eur_mwh | boundary_jump_abs_p95_eur_mwh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.000000 | 1.000000 | 0.000000 | 1.000000 | -2.317183 | 0.000000 | 45.000000 | 1.000000 | 0.000000 | 6.607993 | 19.194015 | 24.074753 | -5.350308 | 27.601906 | 56.267610 | 18.592879 |

## Annual Shape

| year | mean_eur_mwh | evening_minus_midday_eur_mwh | weekend_minus_weekday_eur_mwh | structural_width_mean_eur_mwh | structural_width_p95_eur_mwh |
|---|---|---|---|---|---|
| 2026.000000 | 112.463747 | 43.692997 | -26.249959 | 6.465032 | 14.880139 |
| 2027.000000 | 95.803783 | 35.705518 | -16.002975 | 6.293576 | 15.870430 |
| 2028.000000 | 79.980000 | 24.540296 | -8.423986 | 6.823196 | 19.925778 |
| 2029.000000 | 71.900000 | 24.477874 | -5.993351 | 6.733312 | 20.680978 |
| 2030.000000 | 68.860000 | 24.074753 | -5.350308 | 6.659258 | 19.871915 |

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
