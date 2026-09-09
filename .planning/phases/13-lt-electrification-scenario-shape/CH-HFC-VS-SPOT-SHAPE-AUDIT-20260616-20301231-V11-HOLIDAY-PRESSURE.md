# CH HFC vs Spot Shape Audit

* HFC CSV: `output\ch_hfc_hourly_20260616_20301231_v11_holiday_pressure.csv`
* spot source: `data\epex_hourly.parquet`
* score: `9.00/10`
* scope: `local/test quality audit`
* production approval: `NO`

## Summary

| score_10 | spot_start | spot_end | spot_rows | hfc_start | hfc_end | hfc_rows | latest_hfc_year | latest_hfc_peak_offpeak_spread_eur_mwh | latest_hfc_evening_midday_spread_eur_mwh | latest_hfc_winter_summer_spread_eur_mwh | latest_hfc_jan_oct_spread_eur_mwh | latest_hfc_shape_corr_vs_spot | fast_negative_hours | weighted_negative_hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.000000 | 2023-01-01 01:00:00+01:00 | 2026-03-15 23:00:00+01:00 | 28079.000000 | 2026-06-16 00:00:00+02:00 | 2030-12-31 23:00:00+01:00 | 39841.000000 | 2030.000000 | 1.011562 | 25.827874 | 37.393421 | 18.038101 | 0.920612 | 356.000000 | 0.000000 |

## Annual Metrics

| source | year | mean_eur_mwh | min_eur_mwh | negative_hours | negative_share_pct | peak_offpeak_spread_eur_mwh | evening_midday_spread_eur_mwh | morning_night_spread_eur_mwh | weekend_weekday_spread_eur_mwh | winter_summer_spread_eur_mwh | spring_autumn_spread_eur_mwh | jan_oct_spread_eur_mwh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot_actual | 2023 | 107.500094 | -142.880000 | 76 | 0.867679 | 16.903423 | 31.573097 | 22.133623 | -25.305499 | 40.391589 | 6.816440 | 52.001660 |
| spot_actual | 2024 | 75.924083 | -427.510000 | 292 | 3.324226 | 12.175772 | 32.749028 | 15.459615 | -21.278783 | 45.237951 | -30.644026 | 0.657201 |
| spot_actual | 2025 | 101.686706 | -262.210000 | 303 | 3.458904 | 5.702123 | 48.936722 | 14.412198 | -21.749541 | 54.436399 | -15.720239 | 30.187471 |
| spot_actual | 2026 | 125.460835 | 1.500000 | 0 | 0.000000 | 19.547738 | 21.734378 | 24.152044 | -17.191795 | nan | nan | nan |
| hfc_weighted | 2026 | 112.463747 | 18.848858 | 0 | 0.000000 | 9.876858 | 44.667074 | 21.653618 | -27.069492 | 42.739351 | nan | nan |
| hfc_weighted | 2027 | 95.803783 | 9.289052 | 0 | 0.000000 | 0.550998 | 36.679495 | 9.729018 | -16.784419 | 61.399594 | -8.512345 | 38.750151 |
| hfc_weighted | 2028 | 79.980000 | 2.338152 | 0 | 0.000000 | 2.392511 | 26.299838 | 8.701066 | -9.204644 | 62.265277 | -26.234515 | 20.769567 |
| hfc_weighted | 2029 | 71.900000 | 8.941645 | 0 | 0.000000 | 1.144868 | 26.262187 | 7.864488 | -6.851749 | 38.754368 | -13.622269 | 18.842991 |
| hfc_weighted | 2030 | 68.860000 | 8.214856 | 0 | 0.000000 | 1.011562 | 25.827874 | 8.022087 | -6.221914 | 37.393421 | -13.089619 | 18.038101 |

## Normalized Month-Hour Profile vs Spot

| year | month_hour_shape_corr_vs_spot | month_hour_shape_mae_vs_spot_eur_mwh | cells |
|---|---|---|---|
| 2026.000000 | 0.931553 | 5.958018 | 168.000000 |
| 2027.000000 | 0.897282 | 6.483763 | 288.000000 |
| 2028.000000 | 0.907853 | 6.691149 | 288.000000 |
| 2029.000000 | 0.916690 | 6.636926 | 288.000000 |
| 2030.000000 | 0.920612 | 6.733884 | 288.000000 |

## Negative-Price Diagnostics

| series | negative_hours | min_eur_mwh | spring_summer_midday_share_pct |
|---|---|---|---|
| price_fast_eur_mwh | 356 | -9.405115 | 100.000000 |
| structural_p10_eur_mwh | 356 | -9.405115 | 100.000000 |
| price_weighted_mean_eur_mwh | 0 | 2.338152 | 0.000000 |

## Interpretation

* Shape correlation is computed on month-hour profiles de-meaned by month, not on outright price levels.
* Historical spot is used as a plausibility anchor; EEX forwards remain the level anchor for the HFC.
* A positive peak/offpeak spread, positive winter/summer spread, positive January/October spread and evening-over-midday duck spread are expected for a credible CH LT HFC.
* Negative prices are expected mainly in the lower structural tail, not necessarily in the weighted FMV curve.
