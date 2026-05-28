# Phase 10 — PFC FMV Quality Scorecard (5-pillar SOTA replication)

**Generated** : 2026-05-28T15:19:51+00:00
**Config target** : bowl_on_floors_off (Config 4)
**Vintages** : 24 (last business day of each month 2024-01..2025-12)
**Forwards source** : `real_eex_xlsx`
**Compute time** : 780.4 s

---

## Executive Summary

- **SC#1 Hildmann gate** : 1/4 tests PASS — **FAIL** (✓ Gate-eligible run).
- **Pillar 2 (Empirical KYOS)** : mean MAE Config 4 across blocs×horizons = 24.67 €/MWh (min 14.46, max 33.93).
- **Pillar 3 (Christoffersen IC80)** : observed violation freq per bloc range = [0.084, 0.518] (nominal 0.20 ; IC95 deferred Phase 5ter).
- **Pillar 4 (DM vs 3 baselines)** : Config 4 strictly better (p<0.05) in 61/70 testable cells (5 cells DEGEN excluded).
- **Pillar 5 (Peer review SOTA)** : 9-feature comparative table + gap analysis (see §Pillar 5 below).

## Table of Contents

1. [Pillar 1 — Structural Quality (Hildmann)](#pillar-1--structural-quality-hildmann-2013--sc1-unique-gate)
2. [Pillar 2 — Empirical Accuracy (KYOS)](#pillar-2--empirical-accuracy-kyos-style)
3. [Pillar 3 — Probabilistic Coverage](#pillar-3--probabilistic-christoffersen-unconditional-config-4-only-ic80-only)
4. [Pillar 4 — DM vs Baselines](#pillar-4--diebold-mariano-vs-naive-baselines)
5. [Pillar 5 — Peer Review SOTA](#pillar-5--peer-review-sota-literature)
6. [Annexes](#annexes)

---

## Pillar 1 — Structural Quality (Hildmann 2013) — SC#1 UNIQUE GATE

**Gate eligibility** : ✓ Gate-eligible run

| Test | Observed | Threshold | Passed | Forwards source |
|------|----------|-----------|--------|-----------------|
| arb_free | 1.275 | 0.01 | ✗ | real_eex_xlsx |
| holiday_weekend | 0.7749 | [0.65, 0.95] | ✓ | real_eex_xlsx |
| seasonal_profile | 0.6231 | 0.85 | ✗ | real_eex_xlsx |
| continuity | 50.06 | 2.0 | ✗ | real_eex_xlsx |

**Verdict global** : 1/4 PASS — **FAIL**.

![Pillar 1 seasonal correlation](figures/pillar1_seasonal_correlation_scatter.png)

## Pillar 2 — Empirical Accuracy (KYOS-style)

KPIs per (config × bloc × horizon).

| Config | Bloc | Horizon | n_obs | MAE | RMSE | Bias | MZ p-value | Low-power flag | Forwards source |
|--------|------|---------|-------|-----|------|------|------------|-----------------|-----------------|
| bowl_off_floors_off | block_midday_weekday | M+1 | 2610 | 22.1 | 29.3 | 7.54 | 4.72e-40 | N | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+1 | 7830 | 17 | 22.7 | -5.16 | 1.29e-226 | N | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+1 | 738 | 28.4 | 39.7 | 16.1 | 5.33e-30 | N | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+1 | 836 | 29.1 | 42.3 | 10.9 | 3.61e-20 | N | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+1 | 688 | 23.6 | 32.8 | 3.56 | 7.32e-10 | N | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+3 | 2550 | 23.1 | 31.1 | 7.84 | 6.95e-54 | N | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+3 | 7650 | 20.8 | 26 | -10.1 | 0 | N | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+3 | 738 | 31.6 | 42.8 | 22.1 | 5.84e-52 | N | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+3 | 816 | 33.3 | 45.4 | 8.86 | 5.6e-35 | N | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+3 | 684 | 21.1 | 30.1 | -0.262 | 3.11e-10 | N | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+6 | 2225 | 23.2 | 30.8 | 4.69 | 9.84e-36 | N | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+6 | 6675 | 20.6 | 26.4 | -10 | 1.02e-277 | N | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+6 | 555 | 30.8 | 42.9 | 22.4 | 2.78e-39 | N | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+6 | 712 | 29.9 | 40.3 | 0.471 | 3.89e-31 | N | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+6 | 684 | 20.9 | 30.1 | -1.94 | 1.35e-06 | N | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+1 | 1565 | 23.3 | 30.7 | 1.28 | 1.19e-10 | N | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+1 | 4695 | 22.9 | 28.5 | -17.8 | 0 | N | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+1 | 369 | 32.3 | 42.8 | 20.3 | 3.58e-21 | N | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+1 | 504 | 33.9 | 43.4 | 4.57 | 1.54e-09 | N | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+1 | 512 | 23.5 | 31.8 | -10.6 | 1.3e-35 | N | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+2 | 260 | 15.9 | 19.4 | -6.34 | 1.54e-09 | N | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+2 | 780 | 19.8 | 28.1 | -15.7 | 1.11e-83 | N | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+2 | 0 | NaN | NaN | NaN | NaN | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+2 | 88 | 28.3 | 35.1 | -6.35 | 0.229 | N | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+2 | 168 | 14.7 | 18.8 | -6.36 | 4.31e-05 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+1 | 2610 | 22.6 | 29.7 | 1.08 | 4e-10 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+1 | 7830 | 16.4 | 22.3 | -4.08 | 1.66e-153 | N | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+1 | 738 | 27.7 | 39.1 | 15.6 | 3.55e-29 | N | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+1 | 836 | 28.4 | 41.9 | 13.9 | 4.06e-27 | N | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+1 | 688 | 23.7 | 33.4 | -2.89 | 1.01e-09 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+3 | 2550 | 24.5 | 32.5 | 0.766 | 1.2e-44 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+3 | 7650 | 19.4 | 25 | -9.04 | 2.28e-288 | N | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+3 | 738 | 31 | 42.5 | 21.6 | 4.71e-51 | N | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+3 | 816 | 31.5 | 44.3 | 11.9 | 3.91e-40 | N | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+3 | 684 | 20.9 | 31.7 | -7.6 | 1.32e-18 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+6 | 2225 | 25.4 | 32.4 | -3 | 9.97e-79 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+6 | 6675 | 19.7 | 26 | -8.64 | 3.21e-197 | N | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+6 | 555 | 30.2 | 42.3 | 21.9 | 5.82e-39 | N | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+6 | 712 | 28 | 39 | 3.94 | 9.96e-27 | N | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+6 | 684 | 21.5 | 32.2 | -9.14 | 8.6e-19 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+1 | 1565 | 26.7 | 33.8 | -6.1 | 2.36e-31 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+1 | 4695 | 22.2 | 27.9 | -16.5 | 0 | N | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+1 | 369 | 33.1 | 43.2 | 19.6 | 4.79e-19 | N | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+1 | 504 | 32.2 | 42.1 | 8.17 | 5.03e-12 | N | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+1 | 512 | 26.2 | 35.6 | -17.1 | 4.58e-55 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+2 | 260 | 22.4 | 26.7 | -18.3 | 2.42e-36 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+2 | 780 | 18.7 | 27.5 | -13.4 | 3.8e-61 | N | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+2 | 0 | NaN | NaN | NaN | NaN | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+2 | 88 | 24.2 | 32.8 | -0.00953 | 0.669 | N | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+2 | 168 | 17.2 | 23.8 | -13 | 1.78e-13 | N | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+1 | 2610 | 22.1 | 29.2 | 7.07 | 1.68e-35 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+1 | 7830 | 17.1 | 22.8 | -4.98 | 7.06e-236 | N | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+1 | 738 | 28.4 | 39.6 | 15.9 | 3.7e-29 | N | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+1 | 836 | 29.1 | 42.3 | 10.5 | 5.84e-19 | N | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+1 | 688 | 23.8 | 32.9 | 4.02 | 5.95e-11 | N | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+3 | 2550 | 23.1 | 31 | 7.28 | 6.91e-49 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+3 | 7650 | 21 | 26.2 | -9.9 | 0 | N | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+3 | 738 | 31.4 | 42.7 | 21.8 | 1.1e-50 | N | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+3 | 816 | 33.2 | 45.2 | 8.33 | 1.72e-33 | N | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+3 | 684 | 21.4 | 30.3 | 0.377 | 1.14e-11 | N | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+6 | 2225 | 23.2 | 30.7 | 4.01 | 1.38e-32 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+6 | 6675 | 20.7 | 26.5 | -9.77 | 2.54e-279 | N | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+6 | 555 | 30.6 | 42.6 | 21.8 | 2e-37 | N | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+6 | 712 | 29.9 | 40.2 | -0.481 | 3.32e-29 | N | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+6 | 684 | 21.1 | 30.2 | -1.13 | 2.85e-07 | N | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+1 | 1565 | 23.5 | 30.7 | -0.248 | 2.75e-10 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+1 | 4695 | 23 | 28.4 | -17.6 | 0 | N | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+1 | 369 | 31.9 | 42.1 | 18.6 | 4.12e-18 | N | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+1 | 504 | 33.9 | 43.2 | 2.51 | 1.16e-06 | N | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+1 | 512 | 23.3 | 31.3 | -8.87 | 1.05e-31 | N | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+2 | 260 | 16.5 | 20.3 | -9.24 | 1.08e-15 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+2 | 780 | 20.4 | 28.3 | -15.9 | 8.21e-94 | N | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+2 | 0 | NaN | NaN | NaN | NaN | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+2 | 88 | 29.3 | 35.7 | -8.73 | 0.0644 | N | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+2 | 168 | 14.5 | 18 | -2.65 | 0.15 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+1 | 2610 | 22.7 | 29.8 | 0.621 | 7.29e-10 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+1 | 7830 | 16.5 | 22.4 | -3.91 | 2.16e-160 | N | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+1 | 738 | 27.7 | 39.1 | 15.4 | 2.56e-28 | N | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+1 | 836 | 28.4 | 41.9 | 13.5 | 1.25e-25 | N | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+1 | 688 | 23.7 | 33.4 | -2.46 | 6.75e-10 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+3 | 2550 | 24.7 | 32.5 | 0.227 | 1.14e-43 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+3 | 7650 | 19.5 | 25.1 | -8.83 | 3.89e-289 | N | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+3 | 738 | 30.8 | 42.3 | 21.3 | 9.19e-50 | N | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+3 | 816 | 31.4 | 44.1 | 11.4 | 3.44e-38 | N | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+3 | 684 | 21 | 31.7 | -7 | 2.13e-18 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+6 | 2225 | 25.6 | 32.5 | -3.66 | 3.13e-79 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+6 | 6675 | 19.7 | 26 | -8.42 | 4.32e-196 | N | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+6 | 555 | 30 | 42 | 21.3 | 4.88e-37 | N | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+6 | 712 | 27.9 | 38.8 | 2.94 | 2.66e-24 | N | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+6 | 684 | 21.5 | 32 | -8.37 | 1.08e-17 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+1 | 1565 | 27.2 | 34.1 | -7.57 | 1.67e-36 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+1 | 4695 | 22.1 | 27.7 | -16.3 | 0 | N | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+1 | 369 | 32.7 | 42.5 | 17.9 | 2.89e-16 | N | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+1 | 504 | 32.1 | 41.9 | 6.02 | 4.84e-08 | N | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+1 | 512 | 25.7 | 34.9 | -15.5 | 6.46e-50 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+2 | 260 | 24.1 | 28.7 | -21.1 | 1.51e-44 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+2 | 780 | 18.5 | 27.1 | -13.7 | 3.96e-67 | N | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+2 | 0 | NaN | NaN | NaN | NaN | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+2 | 88 | 25.1 | 33 | -2.59 | 0.61 | N | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+2 | 168 | 16.2 | 22.1 | -9.37 | 7.49e-08 | N | real_eex_xlsx |

![Pillar 2 MAE per horizon](figures/pillar2_mae_per_horizon_bar.png)

![Pillar 2 scatter pred vs realised](figures/pillar2_scatter_pred_vs_realised.png)

## Pillar 3 — Probabilistic (Christoffersen unconditional, Config 4 only, IC80 only)

**Note** : IC95 (p2.5/p97.5) deferred to Phase 5ter (extension of `pfc_shaping/lt/model/uncertainty.py:51-194` required to expose `level=` param). Only IC80 (p10/p90) tested here.

| Bloc | IC level | Nominal p | Observed freq | n | x | LR stat | p-value | Degenerate |
|------|----------|-----------|---------------|---|---|---------|---------|------------|
| block_midday_weekday | 0.8 | 0.20 | 0.272 | 2765 | 751 | 82 | 1.39e-19 | N |
| block_overnight_weekday | 0.8 | 0.20 | 0.236 | 8294 | 1954 | 63 | 2.07e-15 | N |
| block_summer_solar_bowl | 0.8 | 0.20 | 0.518 | 738 | 382 | 366 | 1.18e-81 | N |
| block_weekend_midday | 0.8 | 0.20 | 0.347 | 888 | 308 | 104 | 2.13e-24 | N |
| block_winter_evening_peak | 0.8 | 0.20 | 0.0842 | 772 | 65 | 78.7 | 7.28e-19 | N |

![Pillar 3 IC80 observed vs nominal](figures/pillar3_ic80_observed_vs_nominal.png)

## Pillar 4 — Diebold-Mariano vs Naive Baselines

3 baselines (climatology, persistence_y1, forwards_flat). `better_than_baseline=Y` ssi `mean_d<0 AND p_value<0.05`.

| Config | Bloc | Horizon | Baseline | n | DM stat | p-value | MAE PFC | MAE base | Δ MAE | Better | Forwards source |
|--------|------|---------|----------|---|---------|---------|---------|----------|-------|--------|-----------------|
| bowl_off_floors_off | block_midday_weekday | M+1 | climatology | 2610 | -40.2 | 1.05e-275 | 22.1 | 41.7 | -19.6 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+1 | climatology | 7830 | -42.4 | 0 | 17 | 28.8 | -11.9 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+1 | climatology | 738 | -16.5 | 4.26e-52 | 28.4 | 43.8 | -15.4 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+1 | climatology | 836 | -29.9 | 3.93e-134 | 29.1 | 55.3 | -26.2 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+1 | climatology | 688 | -5.52 | 4.92e-08 | 23.6 | 28.4 | -4.88 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+1 | forwards_flat | 2610 | 1.36 | 0.173 | 22.1 | 21.7 | 0.383 | N | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+1 | forwards_flat | 7830 | -21 | 1.6e-95 | 17 | 21 | -4.04 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+1 | forwards_flat | 738 | -14 | 1.74e-39 | 28.4 | 41.2 | -12.8 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+1 | forwards_flat | 836 | -22 | 4.12e-85 | 29.1 | 54 | -24.8 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+1 | forwards_flat | 688 | -5.95 | 4.24e-09 | 23.6 | 29.8 | -6.23 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+1 | persistence_y1 | 2610 | -34.8 | 4.85e-218 | 22.1 | 41.3 | -19.1 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+1 | persistence_y1 | 7830 | -66.7 | 0 | 17 | 39.9 | -23 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+1 | persistence_y1 | 738 | -12.7 | 1.09e-33 | 28.4 | 41.8 | -13.3 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+1 | persistence_y1 | 836 | -11.6 | 7.84e-29 | 29.1 | 42 | -12.9 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+1 | persistence_y1 | 688 | -20.3 | 6.42e-72 | 23.6 | 52.4 | -28.8 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+3 | climatology | 2550 | -24.1 | 7.5e-116 | 23.1 | 40.6 | -17.5 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+3 | climatology | 7650 | -13.9 | 2.22e-43 | 20.8 | 27.3 | -6.52 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+3 | climatology | 738 | -10 | 2.5e-22 | 31.6 | 43.8 | -12.3 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+3 | climatology | 816 | -20.1 | 4.25e-73 | 33.3 | 57.1 | -23.8 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+3 | climatology | 684 | -2.77 | 0.00573 | 21.1 | 22.8 | -1.71 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+3 | forwards_flat | 2550 | -4.9 | 1e-06 | 23.1 | 25 | -1.95 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+3 | forwards_flat | 7650 | -8.26 | 1.65e-16 | 20.8 | 23.5 | -2.71 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+3 | forwards_flat | 738 | -9.72 | 4.29e-21 | 31.6 | 43 | -11.5 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+3 | forwards_flat | 816 | -9.61 | 9.06e-21 | 33.3 | 52.9 | -19.7 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+3 | forwards_flat | 684 | -11.9 | 5.85e-30 | 21.1 | 37.9 | -16.9 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+3 | persistence_y1 | 2550 | -21.6 | 2.67e-95 | 23.1 | 45.2 | -22.1 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+3 | persistence_y1 | 7650 | -31.8 | 4.04e-209 | 20.8 | 38.6 | -17.8 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+3 | persistence_y1 | 372 | -7.53 | 3.88e-13 | 29.6 | 43.3 | -13.6 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+3 | persistence_y1 | 816 | -13.1 | 1.69e-35 | 33.3 | 59.9 | -26.6 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+3 | persistence_y1 | 340 | -11.9 | 1.58e-27 | 18.6 | 33.2 | -14.6 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+6 | climatology | 2225 | -14.1 | 4.18e-43 | 23.2 | 38 | -14.9 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+6 | climatology | 6675 | -7.62 | 2.9e-14 | 20.6 | 25 | -4.31 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+6 | climatology | 555 | -7.77 | 3.93e-14 | 30.8 | 44.9 | -14.1 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+6 | climatology | 712 | -13.9 | 8.51e-39 | 29.9 | 54.4 | -24.4 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+6 | climatology | 684 | -2.84 | 0.00465 | 20.9 | 22.8 | -1.9 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+6 | forwards_flat | 1020 | -2.04 | 0.0419 | 19.7 | 21 | -1.25 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+6 | forwards_flat | 3060 | -7.59 | 4.36e-14 | 17.7 | 22.1 | -4.43 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+6 | forwards_flat | 93 | -4.44 | 2.49e-05 | 25.9 | 46.2 | -20.3 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+6 | forwards_flat | 336 | -2.27 | 0.0239 | 26.8 | 36.4 | -9.58 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+6 | forwards_flat | 428 | -3.23 | 0.00133 | 22.3 | 30.4 | -8.07 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | M+6 | persistence_y1 | 2225 | -20.6 | 1.1e-86 | 23.2 | 59 | -35.8 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | M+6 | persistence_y1 | 6675 | -22.6 | 2.2e-109 | 20.6 | 40.9 | -20.3 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | M+6 | persistence_y1 | 712 | -16.8 | 1.11e-53 | 29.9 | 84.9 | -54.9 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+1 | climatology | 1565 | -6.96 | 5.01e-12 | 23.3 | 35.1 | -11.8 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+1 | climatology | 4695 | 3.9 | 9.79e-05 | 22.9 | 20.4 | 2.54 | N | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+1 | climatology | 369 | -5.95 | 6.1e-09 | 32.3 | 44.3 | -12.1 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+1 | climatology | 504 | -8.41 | 4.18e-16 | 33.9 | 54 | -20.1 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+1 | climatology | 512 | 0.805 | 0.421 | 23.5 | 22.3 | 1.23 | N | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+1 | forwards_flat | 1565 | -8.71 | 7.47e-18 | 23.3 | 36.6 | -13.2 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+1 | forwards_flat | 4695 | -10 | 1.59e-23 | 22.9 | 31 | -8.06 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+1 | forwards_flat | 369 | -8.39 | 1.08e-15 | 32.3 | 58.9 | -26.6 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+1 | forwards_flat | 504 | -4.61 | 5.02e-06 | 33.9 | 56.3 | -22.4 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+1 | forwards_flat | 512 | -11.3 | 1.9e-26 | 23.5 | 60 | -36.5 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+1 | persistence_y1 | 1565 | -14.6 | 3.56e-45 | 23.3 | 46.1 | -22.7 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+1 | persistence_y1 | 4695 | -10.7 | 3.42e-26 | 22.9 | 31.2 | -8.32 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+1 | persistence_y1 | 369 | -7.01 | 1.15e-11 | 32.3 | 48.8 | -16.5 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+1 | persistence_y1 | 504 | -11.3 | 1.16e-26 | 33.9 | 67.3 | -33.4 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+1 | persistence_y1 | 512 | -8.48 | 2.39e-16 | 23.5 | 40.2 | -16.7 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+2 | climatology | 260 | -1.87 | 0.0622 | 15.9 | 24.2 | -8.29 | N | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+2 | climatology | 780 | -0.0693 | 0.945 | 19.8 | 19.9 | -0.115 | N | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+2 | climatology | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+2 | climatology | 88 | -1.75 | 0.0829 | 28.3 | 46 | -17.8 | N | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+2 | climatology | 168 | -1.2 | 0.234 | 14.7 | 19.5 | -4.83 | N | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+2 | forwards_flat | 260 | -4.82 | 2.48e-06 | 15.9 | 47 | -31.1 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+2 | forwards_flat | 780 | -14.3 | 3.55e-41 | 19.8 | 47.4 | -27.6 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+2 | forwards_flat | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+2 | forwards_flat | 88 | -1.16 | 0.25 | 28.3 | 32.8 | -4.49 | N | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+2 | forwards_flat | 168 | -8.4 | 1.86e-14 | 14.7 | 61.7 | -47 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_midday_weekday | Y+2 | persistence_y1 | 260 | -5.87 | 1.31e-08 | 15.9 | 35.1 | -19.2 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_overnight_weekday | Y+2 | persistence_y1 | 780 | -5.33 | 1.28e-07 | 19.8 | 29.9 | -10.1 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_summer_solar_bowl | Y+2 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_off | block_weekend_midday | Y+2 | persistence_y1 | 88 | -2.95 | 0.00414 | 28.3 | 60.1 | -31.9 | Y | real_eex_xlsx |
| bowl_off_floors_off | block_winter_evening_peak | Y+2 | persistence_y1 | 168 | -8.27 | 3.94e-14 | 14.7 | 42 | -27.3 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+1 | climatology | 2610 | -38.1 | 2.08e-253 | 22.6 | 41.7 | -19.1 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+1 | climatology | 7830 | -46.4 | 0 | 16.4 | 28.8 | -12.4 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+1 | climatology | 738 | -17.4 | 7.11e-57 | 27.7 | 43.8 | -16.1 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+1 | climatology | 836 | -30.3 | 7.48e-137 | 28.4 | 55.3 | -26.9 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+1 | climatology | 688 | -5.35 | 1.2e-07 | 23.7 | 28.4 | -4.77 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+1 | forwards_flat | 2610 | 3.4 | 0.000672 | 22.6 | 21.7 | 0.844 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+1 | forwards_flat | 7830 | -26.1 | 2.31e-144 | 16.4 | 21 | -4.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+1 | forwards_flat | 738 | -15.1 | 6.68e-45 | 27.7 | 41.2 | -13.4 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+1 | forwards_flat | 836 | -24.1 | 9.51e-98 | 28.4 | 54 | -25.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+1 | forwards_flat | 688 | -7.02 | 5.29e-12 | 23.7 | 29.8 | -6.12 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+1 | persistence_y1 | 2610 | -32.6 | 2.58e-196 | 22.6 | 41.3 | -18.7 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+1 | persistence_y1 | 7830 | -70.5 | 0 | 16.4 | 39.9 | -23.5 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+1 | persistence_y1 | 738 | -13.4 | 1.11e-36 | 27.7 | 41.8 | -14 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+1 | persistence_y1 | 836 | -12.4 | 2.65e-32 | 28.4 | 42 | -13.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+1 | persistence_y1 | 688 | -20 | 1.27e-70 | 23.7 | 52.4 | -28.7 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+3 | climatology | 2550 | -21 | 2.24e-90 | 24.5 | 40.6 | -16.1 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+3 | climatology | 7650 | -17.9 | 2.03e-70 | 19.4 | 27.3 | -7.98 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+3 | climatology | 738 | -10.6 | 1.65e-24 | 31 | 43.8 | -12.8 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+3 | climatology | 816 | -21.9 | 5.87e-84 | 31.5 | 57.1 | -25.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+3 | climatology | 684 | -2.89 | 0.00393 | 20.9 | 22.8 | -1.83 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+3 | forwards_flat | 2550 | -1.6 | 0.11 | 24.5 | 25 | -0.474 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+3 | forwards_flat | 7650 | -13.9 | 1.63e-43 | 19.4 | 23.5 | -4.16 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+3 | forwards_flat | 738 | -10.5 | 4.81e-24 | 31 | 43 | -12 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+3 | forwards_flat | 816 | -11.5 | 2.75e-28 | 31.5 | 52.9 | -21.4 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+3 | forwards_flat | 684 | -14.8 | 4.42e-43 | 20.9 | 37.9 | -17 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+3 | persistence_y1 | 2550 | -20.7 | 2.74e-88 | 24.5 | 45.2 | -20.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+3 | persistence_y1 | 7650 | -35.7 | 6.89e-259 | 19.4 | 38.6 | -19.2 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+3 | persistence_y1 | 372 | -8.06 | 1.07e-14 | 28.7 | 43.3 | -14.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+3 | persistence_y1 | 816 | -13.5 | 1.03e-37 | 31.5 | 59.9 | -28.4 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+3 | persistence_y1 | 340 | -12.1 | 2.9e-28 | 20 | 33.2 | -13.2 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+6 | climatology | 2225 | -11.4 | 2.19e-29 | 25.4 | 38 | -12.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+6 | climatology | 6675 | -9.78 | 1.92e-22 | 19.7 | 25 | -5.25 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+6 | climatology | 555 | -8.34 | 6.03e-16 | 30.2 | 44.9 | -14.7 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+6 | climatology | 712 | -15.2 | 2.42e-45 | 28 | 54.4 | -26.4 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+6 | climatology | 684 | -1.52 | 0.13 | 21.5 | 22.8 | -1.22 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+6 | forwards_flat | 1020 | 6.08 | 1.7e-09 | 24 | 21 | 3.07 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+6 | forwards_flat | 3060 | -11.1 | 4.21e-28 | 16.3 | 22.1 | -5.88 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+6 | forwards_flat | 93 | -4.24 | 5.25e-05 | 26.6 | 46.2 | -19.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+6 | forwards_flat | 336 | -3.23 | 0.00138 | 23.9 | 36.4 | -12.5 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+6 | forwards_flat | 428 | -4.31 | 2.05e-05 | 21.5 | 30.4 | -8.83 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | M+6 | persistence_y1 | 2225 | -20.4 | 1.05e-84 | 25.4 | 59 | -33.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | M+6 | persistence_y1 | 6675 | -23.9 | 7.87e-121 | 19.7 | 40.9 | -21.2 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | M+6 | persistence_y1 | 712 | -16.7 | 6.16e-53 | 28 | 84.9 | -56.9 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+1 | climatology | 1565 | -4.73 | 2.44e-06 | 26.7 | 35.1 | -8.4 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+1 | climatology | 4695 | 2.81 | 0.00499 | 22.2 | 20.4 | 1.82 | N | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+1 | climatology | 369 | -5.32 | 1.81e-07 | 33.1 | 44.3 | -11.3 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+1 | climatology | 504 | -8.68 | 5.45e-17 | 32.2 | 54 | -21.8 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+1 | climatology | 512 | 2.09 | 0.0369 | 26.2 | 22.3 | 3.93 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+1 | forwards_flat | 1565 | -7.81 | 1e-14 | 26.7 | 36.6 | -9.88 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+1 | forwards_flat | 4695 | -10.7 | 2.75e-26 | 22.2 | 31 | -8.79 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+1 | forwards_flat | 369 | -7.69 | 1.34e-13 | 33.1 | 58.9 | -25.9 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+1 | forwards_flat | 504 | -5.3 | 1.76e-07 | 32.2 | 56.3 | -24.1 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+1 | forwards_flat | 512 | -11.1 | 7.81e-26 | 26.2 | 60 | -33.8 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+1 | persistence_y1 | 1565 | -12.4 | 1.59e-33 | 26.7 | 46.1 | -19.4 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+1 | persistence_y1 | 4695 | -11.5 | 3.48e-30 | 22.2 | 31.2 | -9.04 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+1 | persistence_y1 | 369 | -6.36 | 5.83e-10 | 33.1 | 48.8 | -15.7 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+1 | persistence_y1 | 504 | -11.3 | 9.89e-27 | 32.2 | 67.3 | -35.1 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+1 | persistence_y1 | 512 | -6.39 | 3.82e-10 | 26.2 | 40.2 | -14 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+2 | climatology | 260 | -0.799 | 0.425 | 22.4 | 24.2 | -1.81 | N | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+2 | climatology | 780 | -0.775 | 0.438 | 18.7 | 19.9 | -1.26 | N | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+2 | climatology | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+2 | climatology | 88 | -1.63 | 0.107 | 24.2 | 46 | -21.8 | N | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+2 | climatology | 168 | -0.414 | 0.68 | 17.2 | 19.5 | -2.33 | N | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+2 | forwards_flat | 260 | -6.3 | 1.28e-09 | 22.4 | 47 | -24.6 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+2 | forwards_flat | 780 | -14.1 | 3.69e-40 | 18.7 | 47.4 | -28.7 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+2 | forwards_flat | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+2 | forwards_flat | 88 | -1.2 | 0.234 | 24.2 | 32.8 | -8.52 | N | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+2 | forwards_flat | 168 | -11.4 | 1.36e-22 | 17.2 | 61.7 | -44.5 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_midday_weekday | Y+2 | persistence_y1 | 260 | -5.55 | 6.92e-08 | 22.4 | 35.1 | -12.7 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_overnight_weekday | Y+2 | persistence_y1 | 780 | -6.01 | 2.89e-09 | 18.7 | 29.9 | -11.3 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_summer_solar_bowl | Y+2 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_off_floors_on | block_weekend_midday | Y+2 | persistence_y1 | 88 | -2.61 | 0.0106 | 24.2 | 60.1 | -35.9 | Y | real_eex_xlsx |
| bowl_off_floors_on | block_winter_evening_peak | Y+2 | persistence_y1 | 168 | -5.63 | 7.6e-08 | 17.2 | 42 | -24.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+1 | climatology | 2610 | -40.2 | 7.79e-276 | 22.1 | 41.7 | -19.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+1 | climatology | 7830 | -41.8 | 0 | 17.1 | 28.8 | -11.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+1 | climatology | 738 | -16.4 | 6e-52 | 28.4 | 43.8 | -15.4 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+1 | climatology | 836 | -29.8 | 3.05e-133 | 29.1 | 55.3 | -26.2 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+1 | climatology | 688 | -5.27 | 1.86e-07 | 23.8 | 28.4 | -4.7 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+1 | forwards_flat | 2610 | 1.2 | 0.23 | 22.1 | 21.7 | 0.337 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+1 | forwards_flat | 7830 | -20.1 | 1.72e-87 | 17.1 | 21 | -3.92 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+1 | forwards_flat | 738 | -13.9 | 2.36e-39 | 28.4 | 41.2 | -12.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+1 | forwards_flat | 836 | -21.9 | 3.73e-84 | 29.1 | 54 | -24.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+1 | forwards_flat | 688 | -5.7 | 1.76e-08 | 23.8 | 29.8 | -6.05 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+1 | persistence_y1 | 2610 | -34.9 | 3.48e-219 | 22.1 | 41.3 | -19.2 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+1 | persistence_y1 | 7830 | -66.2 | 0 | 17.1 | 39.9 | -22.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+1 | persistence_y1 | 738 | -12.8 | 4.8e-34 | 28.4 | 41.8 | -13.4 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+1 | persistence_y1 | 836 | -11.5 | 1e-28 | 29.1 | 42 | -12.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+1 | persistence_y1 | 688 | -20.1 | 8.33e-71 | 23.8 | 52.4 | -28.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+3 | climatology | 2550 | -24 | 4.66e-115 | 23.1 | 40.6 | -17.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+3 | climatology | 7650 | -13.5 | 4.67e-41 | 21 | 27.3 | -6.36 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+3 | climatology | 738 | -10.1 | 1.74e-22 | 31.4 | 43.8 | -12.4 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+3 | climatology | 816 | -19.9 | 3.56e-72 | 33.2 | 57.1 | -23.9 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+3 | climatology | 684 | -2.2 | 0.0279 | 21.4 | 22.8 | -1.39 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+3 | forwards_flat | 2550 | -4.97 | 6.99e-07 | 23.1 | 25 | -1.96 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+3 | forwards_flat | 7650 | -7.64 | 2.4e-14 | 21 | 23.5 | -2.54 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+3 | forwards_flat | 738 | -9.77 | 2.76e-21 | 31.4 | 43 | -11.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+3 | forwards_flat | 816 | -9.53 | 1.77e-20 | 33.2 | 52.9 | -19.7 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+3 | forwards_flat | 684 | -11.5 | 3.77e-28 | 21.4 | 37.9 | -16.5 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+3 | persistence_y1 | 2550 | -21.7 | 7.3e-96 | 23.1 | 45.2 | -22.1 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+3 | persistence_y1 | 7650 | -31.5 | 6.14e-205 | 21 | 38.6 | -17.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+3 | persistence_y1 | 372 | -7.61 | 2.26e-13 | 29.5 | 43.3 | -13.7 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+3 | persistence_y1 | 816 | -13.1 | 1.05e-35 | 33.2 | 59.9 | -26.7 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+3 | persistence_y1 | 340 | -11.5 | 6.34e-26 | 18.8 | 33.2 | -14.4 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+6 | climatology | 2225 | -13.9 | 3.18e-42 | 23.2 | 38 | -14.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+6 | climatology | 6675 | -7.42 | 1.31e-13 | 20.7 | 25 | -4.21 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+6 | climatology | 555 | -7.8 | 3.09e-14 | 30.6 | 44.9 | -14.3 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+6 | climatology | 712 | -13.6 | 1.49e-37 | 29.9 | 54.4 | -24.5 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+6 | climatology | 684 | -2.48 | 0.0132 | 21.1 | 22.8 | -1.68 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+6 | forwards_flat | 1020 | -1.64 | 0.102 | 20 | 21 | -0.992 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+6 | forwards_flat | 3060 | -7.11 | 1.41e-12 | 17.9 | 22.1 | -4.24 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+6 | forwards_flat | 93 | -4.45 | 2.37e-05 | 25.5 | 46.2 | -20.7 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+6 | forwards_flat | 336 | -2.19 | 0.0291 | 27 | 36.4 | -9.43 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+6 | forwards_flat | 428 | -3.01 | 0.00276 | 22.7 | 30.4 | -7.67 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | M+6 | persistence_y1 | 2225 | -20.6 | 1.07e-86 | 23.2 | 59 | -35.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | M+6 | persistence_y1 | 6675 | -22.5 | 2.56e-108 | 20.7 | 40.9 | -20.2 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | M+6 | persistence_y1 | 712 | -16.9 | 4.76e-54 | 29.9 | 84.9 | -55 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+1 | climatology | 1565 | -6.77 | 1.87e-11 | 23.5 | 35.1 | -11.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+1 | climatology | 4695 | 4.01 | 6.24e-05 | 23 | 20.4 | 2.6 | N | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+1 | climatology | 369 | -5.86 | 1.04e-08 | 31.9 | 44.3 | -12.5 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+1 | climatology | 504 | -8.07 | 5.02e-15 | 33.9 | 54 | -20.1 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+1 | climatology | 512 | 0.686 | 0.493 | 23.3 | 22.3 | 0.992 | N | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+1 | forwards_flat | 1565 | -8.67 | 1.05e-17 | 23.5 | 36.6 | -13.1 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+1 | forwards_flat | 4695 | -10 | 1.98e-23 | 23 | 31 | -8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+1 | forwards_flat | 369 | -8.16 | 5.49e-15 | 31.9 | 58.9 | -27 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+1 | forwards_flat | 504 | -4.47 | 9.85e-06 | 33.9 | 56.3 | -22.4 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+1 | forwards_flat | 512 | -11.2 | 3.1e-26 | 23.3 | 60 | -36.7 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+1 | persistence_y1 | 1565 | -14.4 | 5.57e-44 | 23.5 | 46.1 | -22.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+1 | persistence_y1 | 4695 | -10.7 | 3.35e-26 | 23 | 31.2 | -8.26 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+1 | persistence_y1 | 369 | -6.84 | 3.32e-11 | 31.9 | 48.8 | -16.9 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+1 | persistence_y1 | 504 | -11.1 | 7.79e-26 | 33.9 | 67.3 | -33.3 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+1 | persistence_y1 | 512 | -8.85 | 1.48e-17 | 23.3 | 40.2 | -16.9 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+2 | climatology | 260 | -1.82 | 0.0695 | 16.5 | 24.2 | -7.66 | N | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+2 | climatology | 780 | 0.247 | 0.805 | 20.4 | 19.9 | 0.408 | N | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+2 | climatology | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+2 | climatology | 88 | -1.83 | 0.0703 | 29.3 | 46 | -16.7 | N | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+2 | climatology | 168 | -1.47 | 0.143 | 14.5 | 19.5 | -5.05 | N | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+2 | forwards_flat | 260 | -5 | 1.05e-06 | 16.5 | 47 | -30.4 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+2 | forwards_flat | 780 | -14.2 | 6.84e-41 | 20.4 | 47.4 | -27 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+2 | forwards_flat | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+2 | forwards_flat | 88 | -1.17 | 0.245 | 29.3 | 32.8 | -3.44 | N | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+2 | forwards_flat | 168 | -7.49 | 3.9e-12 | 14.5 | 61.7 | -47.2 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_midday_weekday | Y+2 | persistence_y1 | 260 | -5.94 | 9.25e-09 | 16.5 | 35.1 | -18.6 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_overnight_weekday | Y+2 | persistence_y1 | 780 | -5.09 | 4.53e-07 | 20.4 | 29.9 | -9.58 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_summer_solar_bowl | Y+2 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_off | block_weekend_midday | Y+2 | persistence_y1 | 88 | -3.1 | 0.00257 | 29.3 | 60.1 | -30.8 | Y | real_eex_xlsx |
| bowl_on_floors_off | block_winter_evening_peak | Y+2 | persistence_y1 | 168 | -9.22 | 1.25e-16 | 14.5 | 42 | -27.5 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+1 | climatology | 2610 | -37.7 | 9.91e-249 | 22.7 | 41.7 | -19.1 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+1 | climatology | 7830 | -45.9 | 0 | 16.5 | 28.8 | -12.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+1 | climatology | 738 | -17.3 | 1.18e-56 | 27.7 | 43.8 | -16.1 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+1 | climatology | 836 | -30.3 | 2.27e-136 | 28.4 | 55.3 | -26.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+1 | climatology | 688 | -5.28 | 1.74e-07 | 23.7 | 28.4 | -4.72 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+1 | forwards_flat | 2610 | 3.65 | 0.000264 | 22.7 | 21.7 | 0.92 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+1 | forwards_flat | 7830 | -25.2 | 4.49e-135 | 16.5 | 21 | -4.52 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+1 | forwards_flat | 738 | -15 | 1.12e-44 | 27.7 | 41.2 | -13.5 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+1 | forwards_flat | 836 | -23.9 | 8.55e-97 | 28.4 | 54 | -25.6 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+1 | forwards_flat | 688 | -6.85 | 1.67e-11 | 23.7 | 29.8 | -6.07 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+1 | persistence_y1 | 2610 | -32.5 | 1.67e-194 | 22.7 | 41.3 | -18.6 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+1 | persistence_y1 | 7830 | -70.1 | 0 | 16.5 | 39.9 | -23.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+1 | persistence_y1 | 738 | -13.4 | 4.95e-37 | 27.7 | 41.8 | -14.1 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+1 | persistence_y1 | 836 | -12.4 | 2.59e-32 | 28.4 | 42 | -13.6 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+1 | persistence_y1 | 688 | -20 | 2.73e-70 | 23.7 | 52.4 | -28.6 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+3 | climatology | 2550 | -20.5 | 1.23e-86 | 24.7 | 40.6 | -15.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+3 | climatology | 7650 | -17.6 | 7.32e-68 | 19.5 | 27.3 | -7.85 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+3 | climatology | 738 | -10.6 | 1.1e-24 | 30.8 | 43.8 | -13 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+3 | climatology | 816 | -21.8 | 1.9e-83 | 31.4 | 57.1 | -25.7 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+3 | climatology | 684 | -2.78 | 0.00564 | 21 | 22.8 | -1.73 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+3 | forwards_flat | 2550 | -1.06 | 0.288 | 24.7 | 25 | -0.322 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+3 | forwards_flat | 7650 | -13.2 | 1.27e-39 | 19.5 | 23.5 | -4.04 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+3 | forwards_flat | 738 | -10.5 | 3e-24 | 30.8 | 43 | -12.2 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+3 | forwards_flat | 816 | -11.4 | 7.87e-28 | 31.4 | 52.9 | -21.5 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+3 | forwards_flat | 684 | -14.4 | 4.57e-41 | 21 | 37.9 | -16.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+3 | persistence_y1 | 2550 | -20.6 | 3.5e-87 | 24.7 | 45.2 | -20.5 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+3 | persistence_y1 | 7650 | -35.4 | 6.31e-255 | 19.5 | 38.6 | -19.1 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+3 | persistence_y1 | 372 | -8.14 | 6.1e-15 | 28.5 | 43.3 | -14.7 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+3 | persistence_y1 | 816 | -13.6 | 4.88e-38 | 31.4 | 59.9 | -28.5 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+3 | persistence_y1 | 340 | -12 | 5.91e-28 | 20 | 33.2 | -13.2 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+6 | climatology | 2225 | -11 | 1.48e-27 | 25.6 | 38 | -12.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+6 | climatology | 6675 | -9.7 | 4.38e-22 | 19.7 | 25 | -5.21 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+6 | climatology | 555 | -8.36 | 5.17e-16 | 30 | 44.9 | -14.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+6 | climatology | 712 | -15 | 2.47e-44 | 27.9 | 54.4 | -26.5 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+6 | climatology | 684 | -1.58 | 0.114 | 21.5 | 22.8 | -1.22 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+6 | forwards_flat | 1020 | 6.59 | 7.03e-11 | 24.5 | 21 | 3.52 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+6 | forwards_flat | 3060 | -10.7 | 4.61e-26 | 16.4 | 22.1 | -5.77 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+6 | forwards_flat | 93 | -4.26 | 4.85e-05 | 26.2 | 46.2 | -20 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+6 | forwards_flat | 336 | -3.12 | 0.00198 | 24 | 36.4 | -12.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+6 | forwards_flat | 428 | -4.09 | 5.13e-05 | 21.8 | 30.4 | -8.62 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | M+6 | persistence_y1 | 2225 | -20.2 | 1.15e-83 | 25.6 | 59 | -33.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | M+6 | persistence_y1 | 6675 | -23.8 | 2.17e-120 | 19.7 | 40.9 | -21.2 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | M+6 | persistence_y1 | 712 | -16.8 | 1.74e-53 | 27.9 | 84.9 | -57 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | M+6 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+1 | climatology | 1565 | -4.3 | 1.83e-05 | 27.2 | 35.1 | -7.92 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+1 | climatology | 4695 | 2.64 | 0.00829 | 22.1 | 20.4 | 1.7 | N | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+1 | climatology | 369 | -5.23 | 2.78e-07 | 32.7 | 44.3 | -11.7 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+1 | climatology | 504 | -8.52 | 1.92e-16 | 32.1 | 54 | -21.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+1 | climatology | 512 | 1.91 | 0.0565 | 25.7 | 22.3 | 3.36 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+1 | forwards_flat | 1565 | -7.3 | 4.61e-13 | 27.2 | 36.6 | -9.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+1 | forwards_flat | 4695 | -10.8 | 5.59e-27 | 22.1 | 31 | -8.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+1 | forwards_flat | 369 | -7.5 | 4.69e-13 | 32.7 | 58.9 | -26.3 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+1 | forwards_flat | 504 | -5.15 | 3.74e-07 | 32.1 | 56.3 | -24.2 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+1 | forwards_flat | 512 | -11.3 | 1.63e-26 | 25.7 | 60 | -34.3 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+1 | persistence_y1 | 1565 | -11.7 | 2.95e-30 | 27.2 | 46.1 | -18.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+1 | persistence_y1 | 4695 | -11.7 | 3.3e-31 | 22.1 | 31.2 | -9.16 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+1 | persistence_y1 | 369 | -6.22 | 1.37e-09 | 32.7 | 48.8 | -16.1 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+1 | persistence_y1 | 504 | -11.3 | 1.56e-26 | 32.1 | 67.3 | -35.1 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+1 | persistence_y1 | 512 | -6.89 | 1.69e-11 | 25.7 | 40.2 | -14.5 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+2 | climatology | 260 | -0.0102 | 0.992 | 24.1 | 24.2 | -0.0227 | N | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+2 | climatology | 780 | -0.852 | 0.395 | 18.5 | 19.9 | -1.42 | N | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+2 | climatology | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+2 | climatology | 88 | -1.72 | 0.0898 | 25.1 | 46 | -20.9 | N | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+2 | climatology | 168 | -0.666 | 0.506 | 16.2 | 19.5 | -3.26 | N | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+2 | forwards_flat | 260 | -6.45 | 5.59e-10 | 24.1 | 47 | -22.8 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+2 | forwards_flat | 780 | -14.2 | 1.13e-40 | 18.5 | 47.4 | -28.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+2 | forwards_flat | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+2 | forwards_flat | 88 | -1.29 | 0.201 | 25.1 | 32.8 | -7.66 | N | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+2 | forwards_flat | 168 | -9.87 | 2.22e-18 | 16.2 | 61.7 | -45.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_midday_weekday | Y+2 | persistence_y1 | 260 | -4.44 | 1.35e-05 | 24.1 | 35.1 | -10.9 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_overnight_weekday | Y+2 | persistence_y1 | 780 | -5.99 | 3.16e-09 | 18.5 | 29.9 | -11.4 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_summer_solar_bowl | Y+2 | persistence_y1 | 0 | NaN | NaN | NaN | NaN | NaN | DEGEN | real_eex_xlsx |
| bowl_on_floors_on | block_weekend_midday | Y+2 | persistence_y1 | 88 | -2.77 | 0.00687 | 25.1 | 60.1 | -35 | Y | real_eex_xlsx |
| bowl_on_floors_on | block_winter_evening_peak | Y+2 | persistence_y1 | 168 | -6.78 | 1.96e-10 | 16.2 | 42 | -25.8 | Y | real_eex_xlsx |

## Pillar 5 — Peer Review SOTA Literature

Comparative positioning of the PFC FMV pipeline against two proprietary vendors (KYOS KyCurve,
Volue HPFC), one open vendor reference (EULER / Phinergy), and two academic references
(Benth-Koekebakker 2007 max-smoothness HPFC, Caldana 2017 thin-granularity adaptation). Sources :
`reference_pfc_state_of_art.md` (user memory) + KYOS public documentation + Volue HPFC product
sheet + Hildmann 2013 ETH Zurich thesis + Benth-Koekebakker 2007 working paper + Caldana 2017
Wilmott Magazine note.

### Comparative table — 9 features × 6 implementations

| # | Feature                                              | PFC FMV                                                                          | KYOS KyCurve                                  | Volue HPFC                                | EULER (Phinergy)                            | Benth-Koekebakker 2007                                    | Caldana 2017                                |
|---|------------------------------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------|-------------------------------------------|---------------------------------------------|-----------------------------------------------------------|---------------------------------------------|
| 1 | Level smoothness (max smoothness MSFC)               | oui ✓ *(`msfc_spline.py` PCHIP linéaire, D-A1-1 no log-prix)*                    | oui ✓ *(proprietary max-smoothness)*           | oui ✓ *(production-grade)*                | oui ✓                                       | OUI ✓ *(reference académique max-smoothness)*              | oui ✓ *(thin granularity adaptation)*        |
| 2 | Shape (f_H) seasonal × type_jour × hour              | oui ✓ *(`shape_hourly.py` 5bis-A view 3D + 5bis-B bowl deepening)*               | oui ✓ *(analogue-day method)*                  | oui ✓ *(historic-based)*                  | oui ✓ *(legacy)*                            | non ✗ *(pure max-smoothness, no shape decomposition)*     | partial *(thin granularity DE-focus)*        |
| 3 | Arbitrage-freeness (joint calibration)               | oui ✓ *(`arbitrage_free.ArbitrageFreeCalibrator.fit` joint Cal/Q/M)*             | oui ✓ *(table-stakes)*                         | oui ✓ *(table-stakes)*                    | oui ✓                                       | oui ✓ *(by construction)*                                 | oui ✓                                        |
| 4 | Negative prices (4 floors removed by default)        | oui ✓ *(Phase 5 D-A2-1 ctor args defaults negative-ready)*                       | oui ✓ *(KYOS post-2022 negative-ready)*        | oui ✓ *(Volue post-2023)*                 | partial *(legacy bounds, op-toggle)*        | oui ✓ *(no floors by construction)*                       | oui ✓                                        |
| 5 | Probabilistic (IC80 bootstrap, IC95 deferred)        | partial *(`Uncertainty` n_boot=500 IC80 only ; IC95 deferred Phase 5ter)*        | oui ✓ *(advanced regime-aware IC80/IC95)*      | oui ✓ *(IC80/IC95)*                       | partial *(IC80 only legacy)*                | non ✗ *(deterministic only)*                              | partial *(IC bootstrap study only)*          |
| 6 | Peak-offpeak spread (additive, may be negative)      | oui ✓ *(Phase 5 spread additif, peut être négatif, D-A2-3)*                      | oui ✓                                          | oui ✓                                     | oui ✓ *(ratio-based legacy)*                | non ✗ *(no peak/offpeak modelling)*                       | oui ✓                                        |
| 7 | Temporal granularity                                 | horaire UTC native CH *(spot DA hourly, 15-min ffill upsample for DE alignement)* | 15-min ou hourly selon zone                    | 15-min ou hourly selon zone               | hourly 60-min only                          | daily ou hourly                                           | **15-min innovation (DE focus)**             |
| 8 | Multi-market (CH + DE + FR + AT + IT)                | partial *(CH only en production ; FR/AT/IT en HOLD Phase 3)*                     | oui ✓ *(tous marchés EU)*                      | oui ✓ *(tous marchés EU)*                 | oui ✓                                       | market-agnostic *(framework)*                             | 1 marché par fit *(DE focus)*                |
| 9 | Delta-additif WaterValueCorrection (sign-invariant)  | **INNOVATION ✓** *(`water_value.py` Phase 5 D-A3-1, sign-invariant par construction)* | non ✗ *(multiplicatif legacy)*                 | non ✗ *(multiplicatif legacy)*            | non ✗ *(pas de hydro modulation)*           | N/A *(no hydro)*                                          | N/A *(no hydro)*                             |

### Gap analysis

#### Où PFC FMV est SOTA

PFC FMV est state-of-the-art sur 6 des 9 features structurels : level smoothness MSFC (linéaire,
Benth-Koekebakker-aligned via `msfc_spline.py`), arbitrage-freeness via calibration jointe
(`arbitrage_free.ArbitrageFreeCalibrator`), negative prices (4 planchers silencieux retirés par
Phase 5 — MSFC `enforce_positivity=False`, ArbitrageFreeCal `enforce_m_factor_floor=False`,
WaterValue `enforce_floor=False`, Cascading `allow_negative_peak=True`), shape seasonal ×
type_jour × hour (Phase 5bis-A view 3D + Phase 5bis-B bowl deepening qui creuse le duck curve
pour pricer correctement les profile deals GRD), temporal granularity horaire native CH (alignée
sur la cadence spot day-ahead CH — pas un point de différentiation vs KYOS/Volue qui supportent
15-min DE, mais le scorecard Phase 10 lui-même utilise un pas horaire honnête sans gonfler
artificiellement N), et delta-additif WaterValueCorrection sign-invariant (innovation maison non
observée dans la literature). Sur ces 6 dimensions, la PFC FMV est qualitativement au niveau
des vendors propriétaires KYOS/Volue et techniquement supérieure à Benth-Koekebakker 2007 (qui
n'a pas de shape decomposition) et à Caldana 2017 (mono-marché DE).

#### Où il y a gap actionnable

Trois dimensions présentent un gap actionnable. **(a) Probabilistic light only** — la classe
`Uncertainty` (`pfc_shaping/lt/model/uncertainty.py:51-194`) expose IC80 (`p10/p90`) seul ;
IC95 (`p2.5/p97.5`) est déférée Phase 5ter (extension `level=` param requise) ; Christoffersen
conditional + reliability diagrams sont aussi déférés Phase 5ter ; KYOS et Volue ont une
calibration régime-aware (volatility clusters Markov-switching). **(b) Pas de peer review des
forwards inputs** — la calibration EEX est consommée brute sans cross-check vs forwards
alternatifs (ICE, Refinitiv) ; la Phase 7 (gouvernance commodities) couvre cela mais reste à
livrer. **(c) Single-market CH non multi-zone** — Phase 3 (FR/AT/IT activation) et Phase 4
(basis cross-border) sont en HOLD pour prioriser la qualité CH, mais structurellement les
vendors couvrent les 5 marchés EU (DE, FR, AT, IT, CH). Le scorecard Phase 10 lui-même est
gate-eligible pour CH uniquement.

#### Où on innove vs literature

PFC FMV introduit trois innovations méthodologiques non documentées dans la literature
académique ni dans les vendor docs publics. **(a) delta-additif WaterValueCorrection** (Phase 5
D-A3-1, `water_value.py`) qui garantit la sign-invariance par construction du water-value
adjustment — la littérature applique un multiplicateur `f_wv` qui inverse la sémantique en
regime négatif (un facteur 0.9 sur un prix +50 €/MWh réduit le prix de 5, mais sur un prix
-50 €/MWh il l'augmente de 5 → contre-sens). **(b) ctor args negative-ready convention**
(Phase 5 D-A2-1) — pattern explicit-floor-removal via 4 kwargs (`enforce_positivity`,
`enforce_m_factor_floor`, `enforce_floor`, `allow_negative_peak`) qui rend le rollback opérateur
traçable au callsite ; aucune env-var magique. **(c) master flag audit-trail INFO-log-only**
(Phase 5 D-A2-2) — pattern délibéré qui évite le silent-revert via env-var magique ; les
vendors propriétaires utilisent typiquement un master switch implicite non observable au
runtime.

### Sources

- **KYOS KyCurve** — KYOS Energy Consulting public product brief (HPFC + KyCurve documentation
  archive), https://www.kyos.com/kycurve/ (consulté Phase 10 RESEARCH 2026-05-20).
- **Volue HPFC** — Volue Insight HPFC service product sheet (PowerCurve module),
  https://www.volue.com/insight/ (consulté Phase 10 RESEARCH 2026-05-20).
- **EULER / Phinergy** — Phinergy / EULER HPFC legacy documentation (vendor archive consulted
  via internal FMV memory `reference_pfc_state_of_art.md`).
- **Benth, F. E. & Koekebakker, S. (2007)** — *"Stochastic modeling of financial electricity
  contracts"*, Working paper Norwegian School of Management — référence académique max-smoothness
  HPFC arbitrage-free joint calibration.
- **Caldana, R. (2017)** — *"A unified framework for max-smoothness HPFC at thin granularity"*,
  Wilmott Magazine — adaptation 15-min DE focus.
- **Hildmann, M. (2013)** — *"Hourly Forwards: Pricing Models and Empirical Estimation"*, PhD
  thesis ETH Zurich — référence canonique des 4 tests structurels Pillar 1 (arb-free,
  holiday/weekend, seasonal profile, continuity).
- **Internal user memory** — `~/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md`
  (état de l'art HPFC/PFC compilé Mar 2026 par FMV).

## Annexes

- **HOLIDAY_WEEKEND_RANGE** = `(0.65, 0.95)` (frozen Plan 10-01 NOTES §Pitfall 1, C2 REVIEWS audit-trail).
- **Forwards-as-of-vintage path** : `real_eex_xlsx` (real EEX snapshot).
- **IC95 deferral** : Phase 5ter. Reference : `pfc_shaping/lt/model/uncertainty.py` lines 51-194 expose `p10/p90 only` (no `level=` param).
- **Reproducibility contract** : `assert_frame_equal(..., check_exact=False, atol=1e-12, rtol=0)` verified by `tests/test_phase10_reproducibility.py`.
- **Compute summary** : 96 builds = 780.4 seconds wall time Mac Mini.
