# Perfect-Foresight Shaping Diagnostic — Delivery Cal 2025

**Config** : `bowl_on_floors_off`

Additive, non-gating. Isolates **shaping** quality from **forward-forecast** error by re-anchoring on realized ex-post settlements (perfect foresight) and de-levelling intra-period profiles. Methodology: Fleten-Lemming (2003), Benth et al. (2007), Kiesel et al. (2019), Lago et al. (2021); CH-physical decomposition after Bevilacqua et al. (2022).

## 1. Granularity ladder (best-trained vintage)

Monthly-shape correlation vs realized, as the anchor granularity coarsens from the full market quotes down to a single realized annual level. The anchor key sets are listed alongside so the comparison is transparent — note in particular that at late-vintage delivery the market may already quote at month granularity for the near horizon, so `market` is not always at coarser granularity than `pf_cal_quarter`.

| anchor | monthly_corr | n_keys | keys |
|---|---|---|---|
| pf_cal | 0.8241 | 1 | 2025 |
| pf_cal_quarter | 0.9275 | 5 | 2025, 2025-Q1, 2025-Q2, 2025-Q3, 2025-Q4 |
| market | 0.8584 | 29 | 2025-01, 2025-02, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-Q2, 2025-Q3, 2025-Q4, 2026, 2026-Q1, 2026-Q2, 2026-Q3, 2026-Q4, 2027, 2027-Q1, 2027-Q2, 2027-Q3, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035 |

## 2. Training-maturity sweep (seasonal-shape decomposition)

`pf_cal_corr` = monthly-shape correlation under **perfect annual-level foresight**. `market_corr` = with real traded quotes. `shaping_residual` = 1 − pf_cal_corr. **Important caveat**: `fit_seasonal_ratios` only counts full calendar years, so many adjacent vintages share identical seasonal_ratios → the sweep's effective sample size is **6 distinct seasonal-ratio regimes**, not the row count. The median/p10/p90 aggregates below should be read with that collapse in mind.

| vintage | train_years | market_corr | pf_cal_corr | pf_cal_spearman | forecast_gap | shaping_residual | n_months |
|---|---|---|---|---|---|---|---|
| 2024-01-31 | 1.08 | 0.9569 | 0.9184 | 0.8811 | -0.0385 | 0.0816 | 12 |
| 2024-02-29 | 1.16 | 0.9148 | 0.9184 | 0.8811 | 0.0036 | 0.0816 | 12 |
| 2024-03-29 | 1.24 | 0.9174 | 0.9184 | 0.8811 | 0.0009 | 0.0816 | 12 |
| 2024-04-30 | 1.33 | 0.8948 | 0.9184 | 0.8811 | 0.0236 | 0.0816 | 12 |
| 2024-05-31 | 1.41 | 0.9059 | 0.9184 | 0.8811 | 0.0125 | 0.0816 | 12 |
| 2024-06-28 | 1.49 | 0.9082 | 0.9184 | 0.8811 | 0.0102 | 0.0816 | 12 |
| 2024-07-31 | 1.58 | 0.8705 | 0.9178 | 0.8811 | 0.0473 | 0.0822 | 12 |
| 2024-08-30 | 1.66 | 0.8801 | 0.918 | 0.8811 | 0.0378 | 0.082 | 12 |
| 2024-09-30 | 1.75 | 0.8741 | 0.918 | 0.8811 | 0.0439 | 0.082 | 12 |
| 2024-10-31 | 1.83 | 0.921 | 0.9173 | 0.8811 | -0.0037 | 0.0827 | 12 |
| 2024-11-29 | 1.91 | 0.8925 | 0.8606 | 0.8462 | -0.0319 | 0.1394 | 12 |
| 2024-12-31 | 2.0 | 0.8817 | 0.8735 | 0.8741 | -0.0083 | 0.1265 | 12 |


**Aggregates** — pf_cal_corr median **0.918** [min 0.861, max 0.918]; market_corr median **0.900** (distinct seasonal-ratio regimes: 6).

## 3. De-levelled intra-day shape (cosine + demeaned RMSE)

Computed on the **delivery-year window only** (year == 2025), so the score is not contaminated by extrapolated hours outside the anchored window. Cosine = pattern fidelity (additive- AND multiplicative-invariant on the profile); demeaned RMSE = amplitude fidelity (additive-invariant only — responds to multiplicative rescale by design, since amplitude should).

| vintage | train_years | diurnal_cosine | diurnal_demeaned_rmse | summer_cosine | summer_demeaned_rmse |
|---|---|---|---|---|---|
| 2024-01-31 | 1.08 | 0.9166 | 8.804 | 0.8974 | 16.179 |
| 2024-02-29 | 1.16 | 0.9282 | 8.435 | 0.8971 | 16.137 |
| 2024-03-29 | 1.24 | 0.9424 | 7.704 | 0.8942 | 16.186 |
| 2024-04-30 | 1.33 | 0.9663 | 5.863 | 0.8939 | 16.144 |
| 2024-05-31 | 1.41 | 0.9762 | 5.189 | 0.8932 | 16.121 |
| 2024-06-28 | 1.49 | 0.9864 | 4.514 | 0.9689 | 12.237 |
| 2024-07-31 | 1.58 | 0.99 | 3.606 | 0.98 | 9.087 |
| 2024-08-30 | 1.66 | 0.9936 | 2.592 | 0.9867 | 6.303 |
| 2024-09-30 | 1.75 | 0.9951 | 1.98 | 0.9721 | 7.767 |
| 2024-10-31 | 1.83 | 0.995 | 1.778 | 0.9721 | 7.771 |
| 2024-11-29 | 1.91 | 0.9928 | 2.287 | 0.973 | 8.377 |
| 2024-12-31 | 2.0 | 0.9892 | 2.839 | 0.9697 | 8.587 |

## 4. Swiss-physical sub-KPIs (model vs realized)

Best-trained vintage `2024-12-31`.

| sub-KPI | model | realized |
|---|---|---|
| winter/summer ratio | 1.557 | 1.699 |
| solar-bowl depth | 0.428 | 0.558 |
| peak/off-peak spread (EUR/MWh) | 7.158 | 6.407 |

## 5. Figures
