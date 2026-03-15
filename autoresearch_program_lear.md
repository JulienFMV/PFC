# LEAR Autoresearch — Program

Autonomous AI-driven improvement of the LEAR (LASSO Estimated AutoRegressive)
short-term electricity price forecaster.

## Context

FMV SA uses a hybrid LEAR+MLP model for day-ahead (D+1 to D+10) electricity
price forecasting on the Swiss (CH) EPEX spot market. The model predicts
24 hourly prices and is evaluated via rolling 30-day out-of-sample backtest.

Current architecture:
- 24 independent LASSO regressions (one per delivery hour)
- Asinh variance-stabilizing transformation
- Multi-window calibration averaging (42, 56, 84, 365 days)
- ~40 features: reduced CH price lags, DE cross-border prices, load/solar/wind,
  commodities, hydro fill, calendar dummies
- StandardScaler normalization before LASSO fitting
- Per-hour variance recalibration (with 1.2-std clamp)
- AR error correction (coefficient 0.4, lag-1)
- Expanding-window bias correction (strength 0.7, 14-day window)
- MLP ensemble member (128-64 neurons, 60/40 LASSO/MLP weight) — predict() only
- Conformal prediction intervals

## The One Editable File

`pfc_shaping/model/lear_forecaster.py` — this is your "train.py".
You may ONLY modify this file.

## DO NOT Modify

- `autoresearch_eval_lear.py` — Fixed evaluation harness
- `run_pfc_production.py` — Production runner
- `autoresearch_program_lear.md` — This file
- `pfc_shaping/data/` — Data ingestion code
- `dashboard/` — Dashboard code

## Evaluation

Run the evaluation:
```bash
python3 autoresearch_eval_lear.py > eval_lear.log 2>&1
```

Read the key metrics:
```bash
grep "^mae:\|^rmse:\|^score:\|^status:" eval_lear.log
```

Primary metric: **score** (composite, lower is better):
```
score = 0.35*(MAE/15.0) + 0.30*(RMSE/22.3) + 0.20*(MAPE/30.9) + 0.15*(1-corr)
```

Secondary metrics (all from 30-day D+1 backtest):
- `mae` — Mean Absolute Error (EUR/MWh), target < 10
- `rmse` — Root Mean Square Error, target < 15
- `mape` — Mean Absolute Percentage Error (%), target < 18
- `corr` — Pearson correlation forecast vs actual, target > 0.85
- `mae_peak` — Peak hours (h07-h19) MAE, currently the bottleneck
- `mae_offpeak` — Off-peak MAE, already good (~7)
- `bias` — Systematic error, target ~0

## Current Best (2026-03-15)

| Metric | Baseline | Current Best |
|--------|----------|-------------|
| MAE | 15.0 | 10.5 |
| RMSE | 22.3 | 16.7 |
| MAPE | 30.9% | 19.4% |
| corr | 0.629 | 0.811 |
| score | 1.000 | 0.661 |

## Known Bottlenecks

1. **Peak hours h11-h14**: MAE ~18, corr ~0.5 — model is nearly random at midday
2. **Systematic peak bias**: +4 to +7 EUR overestimation at peak hours
3. **Solar-driven midday prices**: Hard to predict without day-ahead solar forecasts
4. **MLP not in backtest**: MLP ensemble only used in predict(), not validated in backtest

## Experiment Ideas (Prioritized by Expected Impact)

### Tier 1 — High Impact (5-15% MAE reduction)
- **N-PIT (Normal Probability Integral Transform)** instead of asinh for variance stabilization
  - Recent literature shows 14.6% error reduction for LEAR
- **Quantile Regression Averaging (QRA)**: combine LEAR point forecasts across windows
  using quantile regression instead of simple averaging
- **Peak-specific models**: use different features/windows for peak vs off-peak hours
- **ElasticNet** instead of pure LASSO: better handles correlated features (l1_ratio=0.5)
- **Exponential weighting** of calibration samples (recent data weighted more)

### Tier 2 — Medium Impact (2-5% MAE reduction)
- **Daily LASSO λ recalibration**: reselect lambda daily within each window
- **Adjacent hour features**: prices at h±1, h±2 as features (strong peak correlation)
- **Price level features**: is current price in top/bottom quartile historically?
- **Separate models for weekdays vs weekends** (different dynamics)
- **Rolling volatility features**: 3d, 7d, 14d rolling std per hour
- **Optimize AR correction coefficient per hour** (currently fixed 0.4)
- **Temperature features** from ENTSO-E data if available
- **Intraday price features**: if 15min data has more signal than hourly

### Tier 3 — Incremental (0-2% MAE reduction)
- **Calibration window grid search**: test [35, 42, 56, 70, 84, 180, 365]
- **Variance ratio cap optimization**: currently 2.5, grid search [1.5, 2.0, 2.5, 3.0]
- **Variance clamp optimization**: currently 1.2 std, grid search [0.8, 1.0, 1.2, 1.5]
- **Bias correction strength**: currently 0.7, grid search [0.3, 0.5, 0.7, 0.9]
- **AR coefficient optimization**: currently 0.4, grid search [0.2, 0.3, 0.4, 0.5, 0.6]
- **Feature subset selection**: systematically remove features and test impact
- **MLP architecture**: try (64, 32), (256, 128, 64), different activations
- **Box-Cox transform** instead of asinh

### Tier 4 — Structural/Ambitious
- **Recurrent Neural Networks with Linear Structures** (SOTA 2025: -12% RMSE vs LEAR)
- **Online learning with exponential decay** (α=0.95-0.98) for LASSO update
- **CEEMDAN decomposition** of prices before forecasting (decompose → forecast → recompose)
- **Regime-switching model**: detect high-vol vs low-vol periods, use different models
- **Transformer-based attention** for capturing long-range dependencies
- **Distributional output**: predict full price distribution, not just mean

## Key Literature

- Lago et al. (2021): LEAR reference, multi-window averaging, epftoolbox
- Ziel & Weron (2018): variance-stabilizing transforms for EPF
- Kath & Ziel (2021): conformal prediction for EPF
- El Mahtout & Ziel (2026): cross-border asynchronous market data (9-22% MAE gain)
- Marcjasz et al. (2025): N-PIT transform (14.6% LEAR improvement)
- Uniejewski & Weron: calibration window selection, NOT change-point detection

## Constraints

- Each evaluation takes ~100-120 seconds (backtest is compute-heavy)
- ~30 experiments per hour, ~200+ overnight
- Model must remain interpretable (LASSO coefficients are examined by traders)
- Production code: changes must not break `run_pfc_production.py`
- Keep code clean: no dead code, no commented-out experiments

## Simplicity Criterion

All else being equal, simpler is better:
- A 0.1 MAE improvement from a clean change? Keep.
- A 0.3 MAE improvement from 200 lines of spaghetti? Not worth it.
- Removing code and getting equal results? Great — simplification win.

## NEVER STOP

Once the loop begins, do NOT pause to ask permission.
Run experiments until manually stopped.
If stuck, re-read lear_forecaster.py, try combining near-misses,
look at per-hour error patterns, think harder.
