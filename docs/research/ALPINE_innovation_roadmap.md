# ALPINE — Adaptive Learning Pipeline for INtelligent Energy pricing

## Innovation Roadmap for FMV SA PFC Short-Term Forecaster
**Date: 2026-03-22**
**Based on: 4 expert research agents, 7 research PDFs, 10+ web searches, full codebase audit**

---

## Architecture Overview

```
Layer 4:  QRA Meta-Learner (optimal combination + calibrated quantiles)
            |           |            |
Layer 3:  LEAR     Chronos-2     Regime Router
          causal    + covariates   (spike/normal)
            |           |            |
Layer 2:  Swiss Market Intelligence Features
            |
Layer 1:  Data (EPEX CH/DE, ENTSO-E, hydro, commodities, calendar)
```

### What makes this novel (not in existing literature):

1. **Causal LEAR**: LEAR + causal normalization (from Toto) + exponential weighting (St-Gallen Swiss paper) + QRA instead of weighted average
2. **Chronos-2 with European covariates**: First FM with DE prices + load + wind + solar as native covariates for Swiss market
3. **Regime Router**: spike/normal detection -> routing to specialized models
4. **Swiss Market Intelligence**: novel features (CH-DE congestion binary, hydro fill delta, fuel stack proxy, French nuclear)

---

## Prioritized Implementation Plan

### TIER 0 — Quick Wins (< 1 day each, total ~2 days)

| # | Technique | Source | Expected MAE Impact | Implementation |
|---|-----------|--------|-----------|------|
| 1 | Exponential sample weighting (half-life 180d) | Paraschiv, Fleten & Schuerle (U. St-Gallen, Swiss specific) | -3 to 7% | 1 line: `model.fit(X, y, sample_weight=weights)` |
| 2 | Holiday feature from `calendar_ch.py` | EPF literature consensus | -2 to 5% on holidays | Binary feature in `_build_features()` |
| 3 | Fuel stack proxy: `gas * 2.0 + CO2 * 0.37` | Weron 2014, Maciejowska 2019 | -3 to 5% | 1 composite feature |
| 4 | Window pruning: zero-weight windows with MAE > 1.5x best | Marcjasz et al. 2023 | -2 to 4% | 5 lines in `_weighted_model_average()` |
| 5 | Lag-7 AR error correction (same-day-of-week) | Lago 2021, Ziel & Weron 2018 | -2 to 4% | Trivial extension of existing AR |

### TIER 1 — High Impact, Moderate Effort (1-3 days each, total ~2 weeks)

| # | Technique | Source | Expected MAE Impact | Details |
|---|-----------|--------|-----------|---------|
| 6 | QRA (Quantile Regression Averaging) | Nowotarski & Weron 2015, Marcjasz 2023 | -5 to 8% point, -10 to 20% intervals | Replaces `_weighted_model_average()` + conformal |
| 7 | Upgrade Chronos-Bolt to Chronos-2 | Agent research | +covariates native | `pip install "chronos-forecasting>=2.0"`, feed DE/load/wind/solar |
| 8 | Causal Instance Normalization (from Toto) | Toto paper ablation study | -3 to 8% | Replace fixed asinh with rolling causal mean/std |
| 9 | Swiss Market Intelligence features | Domain expertise + literature | -3 to 5% | CH-DE congestion binary, hydro fill delta, FR nuclear |

### TIER 2 — Advanced Architecture (1-2 weeks each)

| # | Technique | Source | Expected MAE Impact | Details |
|---|-----------|--------|-----------|---------|
| 10 | Regime Router (spike detection) | Janczura & Weron 2012 | -10 to 15% on spikes | Classifier spike/normal -> specialized models |
| 11 | Meta-learner stacking (XGBoost/QRA) | Olivares 2023 | -5 to 10% | Combines LEAR + FM + disagreement features |
| 12 | LightGBM per-hour for h17-h19 | Lago et al. 2021 | -10 to 15% peak MAE | Replace LASSO for nonlinear hours |
| 13 | Fine-tune Chronos-2 (LoRA) on CH+DE EPEX | Foundation model research | -10 to 20% FM accuracy | 2-4h on Apple MPS |

### TIER 3 — Long-term Innovation

| # | Technique | Source | Impact |
|---|-----------|--------|--------|
| 14 | Student-T mixture head for MLP | Toto paper | Better probabilistic calibration |
| 15 | Multi-FM ensemble (Chronos-2 + Toto + TiRex) | Research synthesis | Maximum diversity |

---

## Key Research Findings

### Foundation Model Comparison (March 2026)

| Model | Params | Covariates? | Probabilistic? | EPF tested? | pip? | License | Recommendation |
|-------|--------|-------------|---------------|-------------|------|---------|---------------|
| **Chronos-2** | 120M | **YES** (past+future) | YES (quantiles) | YES | YES | Apache 2.0 | **PRIMARY** |
| Timer-XL | 84M | YES (theory) | NO | YES (best) | NO | MIT | Wait for multivariate release |
| Toto | 151M | YES | YES (Student-T) | Demand only | YES | Apache 2.0 | Experimental |
| TiRex | 35M | NO | YES (9 quantiles) | NO | YES | NXAI Community | Backup |
| TimesFM 2.0 | 200M | NO | YES | NO | YES | Apache 2.0 | Wait for ICF |
| MOIRAI 1.1 | 300M | YES | YES | NO | YES | CC-BY-NC | License issue |

### Timer-XL EPF Benchmark Results (from paper Table 2)

| Market | MSE | MAE |
|--------|-----|-----|
| NP | 0.234 | 0.262 |
| PJM | 0.089 | 0.187 |
| BE | 0.371 | 0.243 |
| FR | 0.381 | 0.204 |
| **DE** | **0.434** | **0.415** |
| Average | 0.302 | 0.262 |

### Swiss Market Specificities

- Switzerland is NOT in SDAC (EU day-ahead coupling) since 2021
- Cross-border features are CRITICAL (CH-DE spread, NTC, French nuclear)
- Hydro fill rate (delta, not just level) captures filling/emptying dynamics
- Run-of-river ~40% of Swiss hydro acts as must-run
- Peak/off-peak dynamics differ from DE (less extreme)
- Water value creates positive feedback loop with forwards

### Key Academic References

1. **Lago et al. (2021)** — "Forecasting Day-Ahead Electricity Prices: A Review" — IJF
2. **Ziel & Weron (2018)** — "Day-Ahead EPF with High-Dimensional Structures" — Energy Economics
3. **Nowotarski & Weron (2015)** — QRA for EPF — EJOR
4. **Paraschiv, Fleten & Schuerle** — Exponential weighting for Swiss market — Energy Economics
5. **Marcjasz, Uniejewski & Weron (2023)** — Updated QRA and multi-window benchmarks
6. **Janczura & Weron (2012)** — Regime-switching for electricity prices
7. **Uniejewski et al. (2019)** — Renewable features reduce MAE 4-8%
8. **Das et al. (ICML 2025)** — TimesFM-ICF: +6.8% via in-context examples

---

## Estimated Cumulative Impact

If Tier 0 + Tier 1 implemented (~2 weeks of work):

| Metric | Current | Estimated Post-ALPINE | Improvement |
|----------|---------|----------------------|-------------|
| MAE | ~10.5 | ~7.5-8.5 | -20 to 28% |
| RMSE | ~16.7 | ~12-14 | -16 to 28% |
| Correlation | 0.82 | 0.88-0.92 | +7 to 12% |
| Peak MAE (h11-14) | ~18 | ~12-14 | -22 to 33% |
