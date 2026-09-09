# Onboarding — get fully up to speed on the PFC project before any work

You are an expert ML/quant engineer joining the **PFC** repository (Swiss/German
electricity price forecasting). Before writing any code, get completely up to date
with the project, its conventions, and ALL work done so far. Follow this brief.

## 0. First action — sync to the exact branch (worktree-safe)
All recent work lives on **`claude/clean-lt-ct-integration`** (NOT yet merged into
`main`). That branch may already be checked out in another git **worktree** on
this machine (e.g. `PFC_phase10`); a direct `checkout` of it from a second
worktree will fail — that is expected git behaviour. Do NOT touch other worktrees
or their local/uncommitted changes. Create your own feature branch from the
up-to-date remote instead:

```
git fetch origin
git checkout -b feat/ct-chronos2-future-covariates origin/claude/clean-lt-ct-integration
# if it already exists: git checkout -B feat/ct-chronos2-future-covariates origin/claude/clean-lt-ct-integration
git log --oneline -8        # confirm the agent-brief commits are present
```

If `git log` does not show the recent agent-brief commits (e.g.
`8647b1e Add agent prompt: covariate-informed Chronos-2 inference (CT)` or newer),
STOP and resolve the sync before proceeding — everything below assumes that state.

## 1. What this project is
PFC builds price-forward curves and forecasts for the Swiss (CH) power market,
tightly coupled to Germany (DE). Two largely independent stacks:
- **LT / HPFC (long-term)** — `pfc_shaping/lt/`, assembled by
  `pfc_shaping/lt/model/assembler.py` (`PFCAssembler`). Produces an hourly
  forward curve = `B × f_S × f_W × f_H × f_Q × f_WV` then arbitrage-free
  calibration. Validation/scorecard in `pfc_shaping/validation/`.
- **CT (short-term)** — `pfc_shaping/ct/model/`. Main model `LEARForecaster`
  (LASSO-per-hour + MLP + conformal), with an optional **Chronos-2 foundation
  model** member (`FoundationForecaster`) blended in.

Read these orientation files (they are the source of truth, not your priors):
- `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`,
  `.planning/REQUIREMENTS.md`, `.planning/CONTEXT.md`
- `.planning/phases/` — one folder per delivered phase; newest is
  `10-pfc-fmv-quality-scorecard/` and `11-ct-chronos2-future-covariates/`.

## 2. Hard project conventions (do not violate)
- **Reproducibility contract**: several pillars are guarded to `atol=1e-12`
  (`tests/test_phase10_reproducibility.py`). New behaviour must ship **behind a
  feature flag that defaults OFF**, so the existing pipeline stays byte-identical
  when the flag is off. This is the single most important rule in the repo.
- **No leakage, ever**: anything used at a vintage/forecast cutoff must use only
  data strictly before it. Forward/future values must be projected, never read
  from realized future. Leakage guards are expected to be intrinsic to the
  module, not left to the caller.
- **Additive changes**: prefer new modules + flags over editing guarded code.
  Don't touch LT code when doing CT work and vice-versa.
- **Tests**: run `pytest`; never introduce new failures. Know that SOME tests
  already fail in a fresh clone purely because large data parquets are
  git-ignored/absent (e.g. `data/epex_hourly.parquet`,
  `data/forwards_history_phase10.parquet`) — confirm a failure reproduces on the
  parent commit before attributing it to your change.
- **Git**: work on a feature branch, descriptive commits, push; **do not open a
  PR unless explicitly asked**.

## 3. Environment realities
- `numpy/pandas/scipy/pyarrow/sklearn/holidays/matplotlib/pytest` are needed; a
  fresh clone may require installing them. There is a broken system
  `cryptography` (missing `_cffi_backend`) — if PDF/crypto libs fail, `pip
  install --force-reinstall cffi`.
- The foundation model needs `torch` + `chronos-forecasting>=2.0`; if absent,
  `FoundationForecaster.available` is False and LEAR runs standalone — this
  graceful degradation must be preserved.
- Data: `pfc_shaping/data/*.parquet` ships `entso_15min`, `epex_15min`,
  `epex_de_15min`, `de_renewable_forecast`, `hydro_reservoir`, `outages_15min`.
  Some `data/*.parquet` (top-level) are git-ignored and may be absent.
- Calendar/holidays helper: `pfc_shaping/data/calendar_ch.py`
  (`enrich_15min_index`).

## 4. Work already completed (so you don't redo or contradict it)

### 4a. LT — Solar-aware intra-day shape correction (Phase 10 §4quater) — SHIPPED
Commit `642ce82`. Adds an exogenous, leak-free post-processing layer on the
hourly shape factor `f_H` to close the residual peak/off-peak amplitude gap from
the 2025 CH solar-regime shift.
- New module `pfc_shaping/lt/model/solar_modulation.py`:
  `SolarPenetrationFeature` (monthly `solar_pen_m = Σsolar/Σload`, leak-free,
  forward months via climatology+trend capped at p99),
  `SolarBlockedFHCorrection` (ridge block-pooled per-hour β: 4 hour-blocks ×
  season × workday/weekend, fitted on de-levelled midday residual), and
  `solar_modulate()` (the assembler hook).
- `PFCAssembler(enable_solar_modulation=False)` — new kwarg, default OFF →
  byte-identical; when ON, applies `solar_modulate` right after
  `ShapeHourly.apply()` and re-normalises per local day to `mean_h f_H = 1`.
- `pfc_shaping/validation/perfect_foresight.py` — new estimator `sota_solar`
  (`_sota_solar_estimator()`), composes the SOTA swaps + the solar flag.
- `pfc_shaping/validation/scorecard.py::build_one` — stashes leak-free
  `epex_train` on `ShapeHourly._solar_epex_hist`.
- `scripts/run_perfect_foresight.py --ab` — benchmarks baseline/sota/sota_solar.
- `tests/test_solar_modulation.py` — 9 pass / 3 skip (skips need the absent EPEX
  + forwards parquets).
- Doc: `.planning/phases/10-pfc-fmv-quality-scorecard/10-PERFECT-FORESIGHT-SHAPING.md`
  §4quater (includes a data caveat: full Cal-2025 KPI numbers must be produced
  where the realized EPEX/forwards parquets exist; they were NOT measurable in the
  cloud clone).
- Research specs the layer consumes:
  `.planning/phases/10-pfc-fmv-quality-scorecard/solar_research/`.

### 4b. CT — Next task is specified, NOT yet implemented
`.planning/phases/11-ct-chronos2-future-covariates/AGENT-PROMPT.md` is the
detailed task brief: give Chronos-2 genuinely known-future covariates at
inference (`predict_df(future_df=...)`), the top-ROI lever from the article
"Five Ways to Fine-Tune Chronos-2". Read it in full — it has the verified
file:line map of the CT code, the leakage classification (calendar + a *gated,
suspect* DE renewable forecast vs past-only realized series), the synthetic-grid
`future_df` trap, the flag-OFF byte-identity contract, the A/B/C validation
protocol, and the definition of done. **That is the work to do next.**

## 5. CT code map (for the upcoming task)
- `pfc_shaping/ct/model/foundation_forecaster.py` — Chronos-2/Bolt wrapper;
  `_forecast_chronos2` currently passes covariates as PAST only (no `future_df`),
  and uses a SYNTHETIC timestamp grid to dodge DST gaps.
- `pfc_shaping/ct/model/lear_forecaster.py` — builds the covariate dict and blends
  the foundation forecast per-hour; already ingests `de_renewable_forecast` as a
  LEAR exog feature, but never hands it to Chronos-2.
- `scripts/eval_lear_feature_ab.py` — reproducible A/B harness (seeds, input
  SHA-256, block-bootstrap CI) — reuse it for validation.
- `scripts/finetune_chronos2.py` — LoRA fine-tune (AutoGluon-based); only relevant
  to the optional §7 follow-up.

## 6. How to proceed
1. Do §0 (sync) and confirm HEAD = `8647b1e`.
2. Read the orientation files (§1) and the two phase folders (10 and 11).
3. Skim `solar_modulation.py` + its test to absorb the house style for
   flag-gated, leak-free, reproducible additive features (mirror it for CT).
4. Then open `.planning/phases/11-ct-chronos2-future-covariates/AGENT-PROMPT.md`
   and execute it.
5. If anything in your local tree contradicts this brief (different HEAD, missing
   files, different branch), STOP and report before coding.

## 7. Confirm understanding before coding
Reply with a 5-line summary: (a) the branch + HEAD you are on, (b) the OFF-by-default
reproducibility rule, (c) the leakage rule, (d) what was already shipped (4a),
(e) what you are about to do (4b). Only then start.
