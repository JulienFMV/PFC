# CH HFC vs Spot Shape Audit

* HFC CSV: `output\ch_hfc_hourly.csv`
* spot source: `data\epex_hourly.parquet`
* score: `9.00/10`
* scope: `local/test quality audit`
* production approval: `NO`

## Summary

| score_10 | spot_start | spot_end | spot_rows | hfc_start | hfc_end | hfc_rows | latest_hfc_year | latest_hfc_peak_offpeak_spread_eur_mwh | latest_hfc_evening_midday_spread_eur_mwh | latest_hfc_winter_summer_spread_eur_mwh | latest_hfc_jan_oct_spread_eur_mwh | latest_hfc_shape_corr_vs_spot | fast_negative_hours | weighted_negative_hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.000000 | 2023-01-01 01:00:00+01:00 | 2026-03-15 23:00:00+01:00 | 28079.000000 | 2026-06-18 00:00:00+02:00 | 2030-12-31 23:00:00+01:00 | 39793.000000 | 2030.000000 | 1.560308 | 25.521348 | 30.685082 | 19.647104 | 0.921875 | 74.000000 | 0.000000 |

## Annual Metrics

| source | year | mean_eur_mwh | min_eur_mwh | negative_hours | negative_share_pct | peak_offpeak_spread_eur_mwh | evening_midday_spread_eur_mwh | morning_night_spread_eur_mwh | weekend_weekday_spread_eur_mwh | winter_summer_spread_eur_mwh | spring_autumn_spread_eur_mwh | jan_oct_spread_eur_mwh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spot_actual | 2023 | 107.500094 | -142.880000 | 76 | 0.867679 | 16.903423 | 31.573097 | 22.133623 | -25.305499 | 40.391589 | 6.816440 | 52.001660 |
| spot_actual | 2024 | 75.924083 | -427.510000 | 292 | 3.324226 | 12.175772 | 32.749028 | 15.459615 | -21.278783 | 45.237951 | -30.644026 | 0.657201 |
| spot_actual | 2025 | 101.686706 | -262.210000 | 303 | 3.458904 | 5.702123 | 48.936722 | 14.412198 | -21.749541 | 54.436399 | -15.720239 | 30.187471 |
| spot_actual | 2026 | 125.460835 | 1.500000 | 0 | 0.000000 | 19.547738 | 21.734378 | 24.152044 | -17.191795 | nan | nan | nan |
| hfc_weighted | 2026 | 110.227555 | 24.289429 | 0 | 0.000000 | 6.215249 | 45.209753 | 19.419690 | -25.037231 | 37.300400 | nan | nan |
| hfc_weighted | 2027 | 93.541651 | 1.388031 | 0 | 0.000000 | 3.830578 | 35.079231 | 11.078813 | -18.169105 | 60.903196 | -5.750911 | 34.774746 |
| hfc_weighted | 2028 | 79.890000 | 8.918024 | 0 | 0.000000 | 2.826210 | 26.079459 | 8.881471 | -9.430468 | 53.224758 | -5.584179 | 28.510499 |
| hfc_weighted | 2029 | 71.960000 | 15.042174 | 0 | 0.000000 | 2.270667 | 25.820467 | 8.391193 | -7.375135 | 31.165711 | -6.285411 | 22.009825 |
| hfc_weighted | 2030 | 68.830000 | 13.545766 | 0 | 0.000000 | 1.560308 | 25.521348 | 8.260202 | -6.462839 | 30.685082 | -7.865531 | 19.647104 |

## Normalized Month-Hour Profile vs Spot

| year | month_hour_shape_corr_vs_spot | month_hour_shape_mae_vs_spot_eur_mwh | cells |
|---|---|---|---|
| 2026.000000 | 0.939587 | 6.222044 | 168.000000 |
| 2027.000000 | 0.900157 | 6.161967 | 288.000000 |
| 2028.000000 | 0.909826 | 6.745708 | 288.000000 |
| 2029.000000 | 0.918964 | 6.640484 | 288.000000 |
| 2030.000000 | 0.921875 | 6.751521 | 288.000000 |

## Negative-Price Diagnostics

| series | negative_hours | min_eur_mwh | spring_summer_midday_share_pct |
|---|---|---|---|
| price_fast_eur_mwh | 74 | -8.044606 | 100.000000 |
| structural_p10_eur_mwh | 74 | -8.044606 | 100.000000 |
| price_weighted_mean_eur_mwh | 0 | 1.388031 | 0.000000 |

## Interpretation

* Shape correlation is computed on month-hour profiles de-meaned by month, not on outright price levels.
* Historical spot is used as a plausibility anchor; EEX forwards remain the level anchor for the HFC.
* A positive peak/offpeak spread, positive winter/summer spread, positive January/October spread and evening-over-midday duck spread are expected for a credible CH LT HFC.
* Negative prices are expected mainly in the lower structural tail, not necessarily in the weighted FMV curve.
