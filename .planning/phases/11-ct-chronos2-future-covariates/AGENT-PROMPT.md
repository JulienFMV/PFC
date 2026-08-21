# Expert task — Covariate-informed Chronos-2 inference for the CT short-term forecaster

## 0. Who you are / repo orientation
You are a senior ML engineer in the **PFC** repo (Swiss/German electricity price
forecasting; long-term HPFC + short-term "CT"). Work from the repo root. The CT
stack is in `pfc_shaping/ct/model/`. Read these in full before editing:
- `pfc_shaping/ct/model/foundation_forecaster.py` — Chronos-2/Bolt wrapper.
- `pfc_shaping/ct/model/lear_forecaster.py` — LEAR; consumes the foundation model.
- `scripts/eval_lear_feature_ab.py` — the reproducible A/B harness you will reuse.
- `scripts/finetune_chronos2.py` — LoRA fine-tune (AutoGluon-based; for §7 only).

Use the repo's Python env; run tests with `pytest`. The foundation model needs
`torch` + `chronos-forecasting>=2.0`. If absent, `FoundationForecaster.available`
is False and LEAR runs standalone — your changes MUST preserve that graceful
degradation (LEAR must never hard-depend on Chronos).

Relevant data files (paths are exact):
- `pfc_shaping/data/epex_15min.parquet` (CH), `pfc_shaping/data/epex_de_15min.parquet` (DE)
- `pfc_shaping/data/entso_15min.parquet` (load/solar/wind, CH+DE)
- `pfc_shaping/data/de_renewable_forecast.parquet` (`forecast_wind_de_mw`, `forecast_solar_de_mw`)

## 1. Objective
Implement **covariate-informed inference** for Chronos-2: hand it the *genuinely
known-future* covariates over the forecast horizon via the
`predict_df(future_df=...)` channel. This is the top-ROI lever from *Five Ways to
Fine-Tune Chronos-2* — in the article the covariate-informed setup gave the
largest accuracy gains (zero-shot WAPE 4.0% → 2.8%, −30%; portfolio+covariate
8.4% → 2.8%, −66.8%). Our code currently passes Chronos-2 **no** future
covariates at all.

Scope is **inference only** — no retraining is required for the first win. A
fine-tune follow-up is optional and separately gated (§7).

## 2. Verified current state (confirm, then build — don't re-derive)
- `foundation_forecaster.py::_forecast_chronos2` (~l.177-250) builds a DataFrame
  `df` with columns `timestamp, item_id, target` plus covariate columns, then
  calls `self._pipeline.predict_df(df, target="target", timestamp_column=...,
  id_column=..., prediction_length=horizon, quantile_levels=[...])`.
  **No `future_df` is passed** → all covariates are PAST-only.
- **Synthetic-grid trap**: `df`'s timestamps are NOT the real timestamps. The code
  uses `ts = pd.date_range(end="2025-01-01", periods=n, freq="h")` to avoid DST
  "could not infer frequency" errors. Therefore any `future_df` you build MUST
  continue *this synthetic grid* (the next `horizon` hourly steps after the last
  synthetic `ts`), NOT the real calendar — else timestamps won't align and
  Chronos-2 will error or silently misalign. Map real-calendar future covariate
  values (is_weekend, DE forecast, …) onto these synthetic future steps
  **positionally by horizon index**.
- `lear_forecaster.py` predict path (~l.1332-1358) and backtest path
  (~l.1707-1738) assemble the covariate dict from neighbor prices
  (`_neighbor_price_series_h_`) and `load_mw, solar_mw, wind_mw` (`self.exog_`),
  passed as `covariates=` (past).
- `de_renewable_forecast.parquet` is ALREADY a LEAR exog feature
  (`lear_forecaster.py` ~l.234-255) but is NEVER handed to Chronos-2.
- Foundation output is blended per-hour into LEAR with weight `w_fm`
  (~l.1464-1482). **Do not change the blend** in this task.

## 3. CRITICAL correctness rule — what may / may not be a future covariate
A future covariate must be **known at forecast time across the whole horizon**
with no realized target-period leakage. Note Chronos-2 also expects a future
covariate to be present in the PAST context (it learns the past covariate↔target
relation, then conditions on the known future) — so a valid future covariate is
one you can supply for BOTH the context window AND the horizon. Classify rigorously:
- ✅ **Calendar** — `is_weekend`, hour-of-day, day-of-week, CH+DE holiday flags.
  Deterministic for all time → safe. Provide for context AND horizon.
- ⚠️ **DE renewable day-ahead forecast** (`forecast_wind_de_mw`,
  `forecast_solar_de_mw`) — nominally known-future, BUT the file has **no `as_of`
  column** and a prior probe found near-zero-lag perfect alignment with realized,
  i.e. it may be **realization-stamped (leakage)**. Treat as SUSPECT: gate behind
  its own sub-flag; in the backtest, for each forecast cutoff use ONLY rows with
  timestamp ≥ cutoff (never realized target-window data); document the caveat in
  code and report.
- ❌ **Realized CH `load_mw, solar_mw, wind_mw` and realized neighbor prices** —
  NOT known over the future horizon. They MUST stay **past-only**. Passing them as
  future covariates is leakage and will inflate scores — do not.

Net: keep existing past covariates as past; add a SEPARATE `future_covariates`
channel containing only calendar (+ optionally the gated DE forecast), each also
present in the past context.

## 4. Implementation plan (ordered, exact targets)
1. **`FoundationForecaster.forecast(...)`** — add
   `future_covariates: dict[str, pd.Series] | None = None`; thread to
   `_forecast_chronos2`. Bolt path ignores it (debug-log once).
2. **`_forecast_chronos2(...)`** (chronos2 backend only):
   - First **verify the installed `chronos` API**: confirm `predict_df` accepts
     `future_df` and its expected column schema (inspect the installed version's
     signature/docstring; the article uses `predict_df(..., future_df=...)`). If
     the installed version differs, adapt or stop per §11.
   - Ensure each future covariate is also a PAST column in `df` (add it if not).
   - Build `future_df` with: `id_column` (same `item_id`), `timestamp_column`
     (the `horizon` synthetic future steps continuing the `pd.date_range(
     end="2025-01-01", ...)` grid), and one column per future covariate — **no
     `target` column**. Align covariate values positionally by horizon index.
   - Pass `future_df=future_df` to `predict_df(...)`.
   - Robustness: if a future covariate is missing/short/NaN, skip it with a
     warning; the method must still return the same dict shape or `None` as today
     (never crash the pipeline).
3. **`LEARForecaster`** — add constructor flags
   `use_future_covariates: bool = False` and `use_de_renewable_future: bool = False`
   (both default OFF). When `use_future_covariates` is ON, build a
   `future_covariates` dict spanning context+horizon:
   - calendar features (reuse LEAR's existing calendar/holiday helper if present;
     else derive from a `pd.date_range` over the horizon in the model tz);
   - if `use_de_renewable_future` is ON and the file is available, add the DE
     forecast with the §3 leakage guard.
   Pass it into BOTH `forecast()` call sites (predict ~l.1346, backtest ~l.1726).
   Past covariates unchanged.
4. Keep everything **flag-OFF by default**; production path unchanged.

## 5. Reproducibility / non-regression contract
- With both flags OFF, behavior MUST be **byte-identical** to current `main`.
  Prove it: run the LEAR backtest before and after, flags OFF, assert identical
  forecasts (atol=0); report the result.
- Do not touch the foundation blend weights, the LEAR feature set, or any
  HPFC/long-term code. This task is additive and CT-only.

## 6. Validation protocol (a primary deliverable)
Reuse / extend `scripts/eval_lear_feature_ab.py`:
- **A** = baseline (flags OFF); **B** = `use_future_covariates=True` (calendar
  only); **C** = B + `use_de_renewable_future=True`.
- **Evaluate LEAR's final blended forecast** (the production output) — that is the
  metric that matters. ALSO log the foundation-model-only WAPE per arm so you can
  attribute how much of any gain comes from the covariates vs the blend.
- Metrics: **WAPE** = Σ|y−ŷ| / Σ|y|, and **MAE** — overall, split by
  peak/off-peak, and by horizon day (D+1..D+10); also per-hour MAE.
- Backtest over the longest fully-covered span; fix the seed; record input
  SHA-256 (harness already does this).
- Significance: block-bootstrap CI on ΔMAE (harness supports it).
- **Ship-recommendation gate**: B and/or C improve overall WAPE vs A AND the
  bootstrap CI on ΔMAE excludes zero AND no peak-hour regression beyond noise.
  If C > B, explicitly flag that C's edge may be partly spurious if the DE file is
  realization-stamped. A null result (nothing beats A) is a valid, reportable
  outcome — say so honestly; do not p-hack.
- Write a concise markdown report (under `.planning/` or the repo's experiment
  location) with the metric tables, the flags-OFF byte-identity check, and the
  leakage caveats.

## 7. Optional follow-up (only if §6 shows a clear win; SEPARATE commit)
Align fine-tune with inference: `scripts/finetune_chronos2.py` uses AutoGluon
while production loads the raw `Chronos2Pipeline`. If you fine-tune, mirror the
article's LoRA config (`r=8, lora_alpha=16, lr=2e-5`; targets
`self_attention.{q,k,v,o}` + `output_patch_embedding.output_layer`) and confirm
the produced checkpoint loads via `FoundationForecaster`. Not in the §4 commit.

## 8. Environment realities
- Assume no guaranteed internet for model downloads; rely on the local-path logic
  already in `FoundationForecaster` (`models/chronos-2` → `chronos2_finetuned` →
  `amazon/chronos-2`).
- GPU optional; CPU works but is slow — keep the backtest window reasonable.
- If `torch`/`chronos` aren't installed: install per the `foundation_forecaster.py`
  header; if you truly cannot, at minimum run the A/B with the foundation model
  disabled to prove flags-OFF byte-identity, and state that clearly.
- **Do NOT commit model weights/checkpoints** or large artifacts (respect
  `.gitignore`); commit only code, tests, and the small markdown report.

## 9. Git workflow
- Feature branch (e.g. `feat/ct-chronos2-future-covariates`).
- Descriptive commits; §4 and §7 in separate commits.
- Run the existing test suite; introduce no failures.
- Push the branch. **Do NOT open a PR** unless explicitly asked.

## 10. Definition of done
- [ ] `future_covariates` plumbed through `FoundationForecaster` into
      `predict_df(future_df=...)`, on the synthetic grid, with robust fallbacks.
- [ ] Future covariates present in BOTH the past context and `future_df`;
      `future_df` has no `target` column.
- [ ] LEAR builds horizon calendar (+ gated DE forecast) and passes it; both flags
      default OFF.
- [ ] Flags-OFF byte-identity proven (atol=0) and reported.
- [ ] A/B/C backtest on the final blended output (+ foundation-only attribution)
      with WAPE/MAE breakdowns + bootstrap CI; markdown report committed.
- [ ] Realized covariates kept past-only (no leakage); DE-forecast leakage caveat
      documented in code + report.
- [ ] A unit test covering the new `future_covariates` plumbing (e.g. shapes,
      flag-OFF no-op, missing-covariate fallback).
- [ ] Tests green; branch pushed; no PR.

## 11. If blocked
If a covariate's leakage status is genuinely ambiguous, or the installed
`chronos.predict_df` `future_df` schema doesn't match the article, STOP and report
exactly what you found (error text, signature, your proposed resolution). Never
guess in a way that could introduce leakage or silently misalign timestamps.
