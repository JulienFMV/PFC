# CODEX AGENT BRIEF — PFC Project (FMV SA)

## Context

This is the PFC (Price Forward Curve) project for FMV SA, a Swiss hydro electricity producer in Sion (Valais). The project forecasts short-term (D+1 to D+10) and long-term (M+1 to Y+3) electricity prices for the Swiss EPEX market.

**Current best results (March 2026, crisis conditions — Iran war):**
- MAE: 9.24 EUR/MWh (down from 13.42 baseline)
- RMSE: 15.43
- Correlation: 0.843
- Score: 0.5481

## Repository Structure

```
pfc_shaping/
  model/
    lear_forecaster.py      # CORE — LEAR short-term model (~1400 lines)
    foundation_forecaster.py # Chronos-2/Bolt foundation model integration
    shape_hourly.py          # Long-term hourly shape factors
    shape_intraday.py        # 15-min intra-hourly factors
    assembler.py             # PFC assembly (B x f_S x f_W x f_H x f_Q x f_WV)
    water_value.py           # Hydro reservoir correction
    uncertainty.py           # Bootstrap confidence intervals
    msfc_spline.py           # Maximum smoothness forward curve
  calibration/
    arbitrage_free.py        # Arbitrage-free calibration (MSFC QP)
    cascading.py             # Forward contract cascading (Cal/Q/M)
  data/
    *.parquet                # Data files (EPEX, ENTSO-E, hydro, outages, etc.)
  pipeline/
    rolling_update.py        # Production pipeline
dashboard/                   # Streamlit dashboard (14 pages)
scripts/
  finetune_chronos2.py       # Chronos-2 LoRA fine-tuning script (READY)
autoresearch_eval_lear.py    # Fixed evaluation harness
autoresearch_program_lear.md # Research program with experiment ideas
```

## What Has Been Done (Claude Code sessions, March 2026)

### Model improvements
- ElasticNetCV (l1_ratio=0.1) with 5 calibration windows (42/56/84/180/365 days)
- AR error correction (lag-1: 0.60, lag-2: 0.15, lag-7: 0.20)
- Variance recalibration (cap 1.8)
- HistGradientBoostingRegressor for peak hours (h07-h19)
- Chronos-2 foundation model with covariates (DE price, load, wind, solar)
- Regime router (spike/volatile/normal)
- QRA (Quantile Regression Averaging) for probabilistic intervals

### Features (current ~83 per hour)
- CH price lags (d-1, d-2, d-3, d-7) at target hour + daily aggregates
- DE price lags + DE daily mean
- CH-DE spread + directional congestion (import vs export)
- DE renewable forecasts (wind, solar) d+0 and d-1
- CH exogenous (load, solar, wind, outages) d-1 and d-7 + daily aggregates
- Commodities (TTF gas, CO2 EUA, Brent) d-2
- Fuel stack proxy (gas * 2.0 + CO2 * 0.37)
- Hydro fill + 7-day delta
- **NEW (Phase 1):** Wallis-specific fill, hydro_pumped_mw, hydro_ror_mw, nuclear_available_ch_mw, nant_de_drance_online, winterreserve_active, directional congestion, spot_dist_to_q90/q10, spot ramps, spot volatility 7d
- Calendar: DOW dummies, weekend, month sin/cos, holidays CH+DE, regime dummies

### Deep audit (91 findings, 31 fixed)
- P0: 14/14 fixed (data leakage, reference_date, GBM weight bug, quadruple bias correction, etc.)
- P1: 17/22 fixed
- Full report: `AUDIT_DEEP_2026-03-23.md`

## YOUR MISSION — 3 Tasks for GPU Machine

### Task 1: LightGBM per-hour (REPLACES HistGradientBoosting)

**Why:** LightGBM is faster, handles missing data natively, provides feature importance. Lago et al. (2021) showed LEAR+GBRT reduces MAE 5-10%.

**What to do:**
1. `pip install lightgbm`
2. In `lear_forecaster.py`, replace `HistGradientBoostingRegressor` with `lightgbm.LGBMRegressor` in `_fit_gbm_for_hour()`:

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    verbose=-1,
    n_jobs=-1,  # Use all cores on GPU machine
)
# Chronological split for early stopping (NOT random)
n_val = max(7, int(len(y_arr) * 0.15))
X_fit, X_val = X_arr[:-n_val], X_arr[-n_val:]
y_fit, y_val = y_arr[:-n_val], y_arr[-n_val:]
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(20, verbose=False)])
```

3. Extend GBM to ALL 24 hours (not just PEAK_HOURS). In backtest, change `for hour in PEAK_HOURS:` to `for hour in range(24):` in the GBM pre-training block (~line 1598).

4. Run backtest: `python autoresearch_eval_lear.py`

**IMPORTANT:** On macOS ARM this causes a segfault. On Windows x64 with CUDA it should work fine.

### Task 2: Fine-tune Chronos-2 with LoRA

**Why:** Zero-shot Chronos-2 is mediocre on EPF. Fine-tuning on 2-3 years of CH+DE data gives 20-40% improvement on FM MAE.

**What to do:**
1. `pip install "chronos-forecasting>=2.0" autogluon-timeseries peft`
2. Run: `python scripts/finetune_chronos2.py`
3. The script:
   - Loads CH + DE EPEX hourly prices
   - Adds DE wind/solar as covariates
   - Fine-tunes `amazon/chronos-2-base` with LoRA via AutoGluon
   - Saves to `pfc_shaping/model/chronos2_finetuned/`
4. After fine-tuning, update `foundation_forecaster.py` line 65:
   ```python
   CHRONOS2_MODEL = "pfc_shaping/model/chronos2_finetuned"  # local fine-tuned
   ```
5. Run backtest to compare.

**GPU requirement:** ~2-4 hours on NVIDIA GPU. The script auto-detects CUDA.

### Task 3: Run autoresearch loop

Use the `/autoresearch-lear` skill or manually:
1. Create branch: `git checkout -b autoresearch/codex-gpu`
2. Baseline: `python autoresearch_eval_lear.py`
3. Experiment loop:
   - Modify `lear_forecaster.py` with ONE focused change
   - Run eval, compare score
   - Keep if improved, revert if not
   - Log in `results_lear.tsv`

**Experiment ideas (prioritized):**
- LightGBM all hours (Task 1)
- Fine-tuned Chronos-2 (Task 2)
- Ridge stacker combining LEAR + GBM predictions
- FR/AT/IT neighbor prices from ENTSO-E (need to extend `ingest_entso.py`)
- NTC day-ahead cross-border features

## Evaluation

**Fixed harness — DO NOT MODIFY:** `autoresearch_eval_lear.py`

```bash
python autoresearch_eval_lear.py > eval_lear.log 2>&1
grep "^mae:\|^rmse:\|^score:\|^status:" eval_lear.log
```

**Composite score (lower is better):**
```
score = 0.35*(MAE/15.0) + 0.30*(RMSE/22.3) + 0.20*(MAPE/30.9) + 0.15*(1-corr)
```

**Current best: score=0.5481, MAE=9.24**

## Key Constraints
- Only modify `pfc_shaping/model/lear_forecaster.py` for model experiments
- Keep code interpretable — no black-box hacks
- Physical bounds: `np.clip(forecast, -500, 1000)`
- Each eval takes ~100-200s
- Commit every experiment (keep or revert)

## Expert Consensus (4 experts, March 2026)
1. "Biggest gap is DATA, not models" — add FR/AT/IT prices, NTC, FR nuclear
2. LightGBM = #1 model addition
3. QRA per-hour = optimal combination method
4. Stop optimizing LASSO hyperparams — diminishing returns
5. Profile correlation (Corr-f) > MAE for hydro dispatch
6. MAE targets: 5-7 normal, 8-10 volatile

---

## TOP 5 PRIORITIES TO REDUCE MAE (ordered by impact)

### Priority 1: Add FR/AT/IT neighbor prices to LEAR (Impact: -1 to 3 EUR/MWh)

The #1 consensus across all 4 experts. The LEAR model only sees DE prices, but CH is
coupled with 4 neighbors. The data is available via the same ENTSO-E API.

**Implementation:**
1. In `ingest_entso.py`, add queries for FR, AT, IT-Nord area codes:
   - FR: `10YFR-RTE------C`
   - AT: `10YAT-APG------L`
   - IT-Nord: `10Y1001A1001A73I`
2. In `lear_forecaster.py` `_build_features()`, add the same lag structure (d-1, d-7)
   for FR, AT, IT prices — same as the existing DE features.
3. In `_build_prediction_row()`, add the corresponding lookups.

Expected: ~6-8 new features per hour. Combined with existing DE, this gives the model
a full picture of all neighbors, which is what Axpo/Alpiq use.

### Priority 2: LightGBM native on ALL 24 hours (Impact: -5 to 10%)

Replace `HistGradientBoostingRegressor` (sklearn) with native `lightgbm.LGBMRegressor`.
Extend from peak-only to ALL 24 hours.

**Implementation:**
```python
import lightgbm as lgb

def _fit_gbm_for_hour(self, hour, X, y):
    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1,
        verbose=-1, n_jobs=-1,
    )
    # Chronological split (NOT random) for early stopping
    n_val = max(7, int(len(y) * 0.15))
    X_fit, X_val = X[:-n_val], X[-n_val:]
    y_fit, y_val = y[:-n_val], y[-n_val:]
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
    return model
```

In the backtest GBM pre-training block (~line 1598), change:
```python
# BEFORE: for hour in PEAK_HOURS:
# AFTER:
for hour in range(24):
```

### Priority 3: Fine-tune Chronos-2 with LoRA (Impact: -10 to 20% on FM)

The model weights are now in `models/chronos-2/` (downloaded via Git LFS).

**Implementation:**
1. In `scripts/finetune_chronos2.py`, update the model path:
   ```python
   CHRONOS2_MODEL = "models/chronos-2"  # local, no SSL needed
   ```
2. Run: `python scripts/finetune_chronos2.py`
3. After fine-tuning (2-4h on NVIDIA GPU), the model saves to
   `pfc_shaping/model/chronos2_finetuned/`
4. Update `foundation_forecaster.py` line 65:
   ```python
   CHRONOS2_MODEL = "pfc_shaping/model/chronos2_finetuned"
   ```
5. Run backtest to compare zero-shot vs fine-tuned.

### Priority 4: Ridge stacker (LEAR + LightGBM + FM) (Impact: -5 to 8%)

Instead of fixed weights or inverse-MAE, train a Ridge regression on the last 60 days
of out-of-sample predictions from each model. The stacker learns which model is best
for which hour and regime.

**Implementation:**
After `_weighted_model_average()` and `_fit_gbm_for_hour()` produce their predictions,
add a stacking layer:

```python
from sklearn.linear_model import Ridge

def _fit_stacker(self, lear_preds, gbm_preds, fm_preds, actuals, hours):
    """Train per-hour Ridge stacker on last 60 OOS days."""
    stackers = {}
    for h in range(24):
        mask = hours == h
        if mask.sum() < 14:
            continue
        X_stack = np.column_stack([
            lear_preds[mask], gbm_preds[mask], fm_preds[mask]
        ])
        y_stack = actuals[mask]
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_stack, y_stack)
        stackers[h] = ridge
    return stackers
```

Use in predict(): `final = stacker[hour].predict([[lear, gbm, fm]])`

### Priority 5: FR nuclear availability (Impact: -5 to 10% in volatile)

The most market-moving variable in Continental Europe. When FR nuclear drops from
61 GW to 45 GW, France imports from CH and CH prices spike by 20-40 EUR/MWh.

**Implementation:**
1. In `ingest_entso.py` or a new `ingest_remit.py`, add:
   ```python
   # Query French nuclear unavailability
   client.query_unavailability_of_generation_units(
       'FR', start, end, doctype='A77'
   )
   ```
2. Aggregate by fuel_type == "Nuclear", sum unavailable MW
3. Create feature: `fr_nuclear_available_mw = 61000 - unavailable_nuclear_fr`
4. Add as LEAR feature with d-1 lag (the schedule is published D-1)

---

## WHAT NOT TO DO

- Do NOT optimize AR coefficients further (diminishing returns, confirmed by experts)
- Do NOT add complex DL models (TFT, N-HiTS) before LightGBM + stacker are working
- Do NOT modify `autoresearch_eval_lear.py` (fixed evaluation harness)
- Do NOT touch the PFC long-term model (stable, well-audited)
- Do NOT use random train/test splits (always chronological for time series)
- Do NOT add more than ~120 features to LEAR (multicollinearity degrades LASSO)

## TARGET

| Metric | Current | Post-Codex Target |
|--------|---------|-------------------|
| MAE | 9.24 | **6.5-8.0** |
| MAE peak | ~13 | **9-11** |
| Correlation | 0.84 | **0.90+** |
| Score | 0.5481 | **<0.45** |

Priorities 1+2+4 alone should bring MAE below 8 in normal conditions.
Fine-tuned Chronos-2 (priority 3) is the bonus for further gains.
