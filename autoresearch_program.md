# PFC Autoresearch — Program

Autonomous AI-driven improvement of the PFC (Price Forward Curve) 15min model.
Inspired by Karpathy's autoresearch loop, adapted for energy forward curves.

## Context

FMV SA builds a PFC at 15-minute granularity for Swiss electricity markets.
The model decomposes price into multiplicative shape factors:

```
P(t) = B × f_S × f_W × f_H × f_Q × f_WV
```

The goal: **minimize RMSE vs realized EPEX spot prices** on an out-of-sample
test window (last 2 months).

## In-Scope Files

You may modify any file under `pfc_shaping/`. The key components are:

| File | Role | What to improve |
|---|---|---|
| `pfc_shaping/model/shape_hourly.py` | f_H hourly shape + f_W day-of-week | Smoothing, estimation, fallback logic |
| `pfc_shaping/model/shape_intraday.py` | f_Q 15min intra-hour shape | Huber regression, Ridge correction, feature engineering |
| `pfc_shaping/model/assembler.py` | PFC assembly P(t) | Spline interpolation, factor composition |
| `pfc_shaping/model/uncertainty.py` | Bootstrap confidence intervals | Bootstrap calibration, horizon widening |
| `pfc_shaping/model/water_value.py` | f_WV hydro reservoir correction | Regression, seasonal sensitivity |
| `pfc_shaping/data/forward_proxy.py` | Base price B derivation | Anchor window, decay estimation, seasonality |
| `pfc_shaping/data/calendar_ch.py` | Holiday + season classification | Day types, season boundaries |
| `pfc_shaping/data/ingest_epex.py` | EPEX spot data loading | Data quality, gap handling |
| `pfc_shaping/data/ingest_entso.py` | Load + generation features | Feature engineering, solar_regime |
| `pfc_shaping/config.yaml` | All tunable parameters | Any config value |

## DO NOT Modify

- `autoresearch_eval.py` — This is the fixed evaluation harness (like prepare.py in Karpathy's version)
- `dashboard/` — Dashboard code is out of scope
- `autoresearch_program.md` — This file

## Evaluation

Run the evaluation:
```bash
python3 autoresearch_eval.py > eval.log 2>&1
```

Read the key metric:
```bash
grep "^rmse:" eval.log
```

The primary metric is **rmse** (lower is better). Secondary metrics:
- `rmse_shape` — level-adjusted RMSE (isolates shape quality from level prediction)
- `mae` — mean absolute error
- `bias` — systematic over/under prediction
- `ic80_coverage` — should be ~0.80 for well-calibrated intervals

## Experiment Ideas

Here are directions to explore (non-exhaustive — be creative):

### Quick wins (parameter tuning)
- Gaussian sigma for f_H smoothing (currently 0.5, autoresearch found 1.17 is better)
- Ridge alpha for Layer 2 corrections
- R² threshold for accepting exogenous corrections (currently > 0)
- Lookback window length
- f_W smoothing sigma

### Structural improvements
- Better seasonal boundary handling (smooth transitions instead of sharp month cuts)
- Improved forward proxy anchor (rolling window vs fixed 6-month)
- Better holiday treatment (continuous weighting instead of binary)
- Temperature/solar features if available in ENTSO-E data
- Improved f_Q estimation (different regression approaches)
- Cross-validation for Ridge alpha selection
- Weighted regression (recent data more important)
- Better outlier handling (MAD instead of percentile-based)

### Architectural changes
- Different decomposition structure
- Additive components instead of/alongside multiplicative
- Log-space modeling
- Regime-switching (high-vol vs low-vol markets)
- Ensemble approaches

## Simplicity Criterion

All else being equal, simpler is better:
- A small RMSE improvement that adds ugly complexity? Not worth it.
- Removing code and getting equal results? Great — that's a simplification win.
- A 0.1 EUR/MWh improvement from a clean, simple change? Definitely keep.

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue.
The human might be away. You are autonomous.
Run experiments until manually stopped.
If you run out of ideas, think harder — re-read the model code,
try combining previous near-misses, try more radical changes.
