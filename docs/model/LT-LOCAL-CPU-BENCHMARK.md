# Local CH hourly CPU benchmark v1 — 7 September 2026

This user-authorized development experiment asks whether another hourly model
improves the FMV market-constrained central curve. It is a retrospective replay
of latest-observed PRD history, not a historical point-in-time backtest or an
independent prospective holdout. Scientific protocol v6 stays unchanged.

## Frozen design before fitting

- Retained D300 CH price, SFOE fill and normalized CH EEX history, hash verified.
  No extraction, Warehouse, GPU, AFRY, CT or T057 access. No D300 fitted weights.
- Tuning origins: 31 December 2020 and 2021, 12:00 UTC; delivery the following
  January–December. Full bounded grids from v6; average origin MAE selects one
  configuration per family, ties broken by canonical parameter JSON. Test data
  never select a configuration, stopping tolerance, training window or seed.
- Assessment origins: 31 December 2022, 2023, 2024 and 2025, 12:00 UTC. Forecast
  at most the following 36 whole months. Score only complete truth months through
  August 2026. Overlapping deliveries across origins are dependent, not extra
  independent observations. Report equal origin weighting and coverage per lead.
- Expanding training begins with the first complete Swiss day in January 2019;
  only complete days ending before the simulated origin are eligible. Preserve
  the native daily mean >5 EUR/MWh filter, ratio clip [0.2,3], recency-weighted
  local-hour aggregation and autumn repeated-clock-hour merge. These learning
  exclusions never remove negative-price days from evaluation.
- Nine common features: native calendar encoding, historical hydro fill,
  pre-origin hydro week climatology for forecasts and native years_ahead.
  No future realized physical/hydro/outage predictor. Hydro climatologies and
  all scalers are fitted again per origin. Source publication/vintage timing is
  unproven; delivery cutoffs do not turn revised values into historical PIT.
- Corrected local incumbent: HydroAlignedShapeHourlyMLP, native fit/apply.
  Original frozen MLP: separately reported diagnostic, not the corrected model.
  The native MLP uses an internal random early-stopping subset of pre-origin
  training only. It is not a chronological tuning fold or future test data.
- Four challengers: true 180-day weighted 64x64 MLP (unchanged L-BFGS-B,
  maxiter=500/tol=1e-8), standardized Ridge, additive cubic spline Ridge, exact
  deterministic CPU LightGBM 4.6.0. Existing bounded grids and seed 42.
  Weighted MLP non-convergence is a failed candidate, never a hidden approximate
  success. No retries with outcome-chosen parameters. Other families may still
  be compared on the unchanged common population if one candidate fails.
- Untuned local seasonal reference: arithmetic mean training target by Swiss
  season/day-type/clock hour, missing-cell fallback season/hour → hour → global
  mean. Report fallback counts. Outside the model selection inventory.
- Forecast f_H: native floor .1, Swiss daily normalization, final clip [.4,2],
  exactly once. A new local implementation is parity tested; no future registry
  slot or synthetic provenance is fabricated to invoke a restricted boundary.
- EEX surface: latest single quotation date strictly before the origin's Swiss
  date, no per-product forward filling. Keep wholly undelivered products fully
  contained in the requested 12/36-month window. The existing monthly solver
  consumes only quotation history before that cutoff, with D300 configuration.
  Its unsigned research lane is explicit; nested TEST_FIXTURE is the library's
  sentinel, not the real source identity. Keep raw quote conflicts visible.
- Existing PFCAssembler and existing hourly-factor injection seam only. Common
  fitted f_W from the native incumbent; f_Q=1, water value absent/delta=0,
  physical forecast absent. Preserve native horizon damping, near-term bridge,
  monthly solver authority and final product projection. All candidates use
  identical common layers and market constraints. This isolates hourly effects;
  it does not evaluate D300's full intraday/water-value configuration.
- CH source is native hourly expanded to quarter-hours. Score the assembled
  EUR/MWh prices at complete UTC-hour grain, one MWh weight per hour at 1 MW.
  Monthly neutralization separately removes prediction and actual Swiss-month
  means before errors/segments. Incomplete truth months are excluded for all
  candidates. Forecast NaNs fail; they never shrink the comparison mask.
- Primary MAE; secondary RMSE, bias, weighted P95 absolute centered error.
  Report lead buckets 1–6, 7–12, 13–24, 25–36; seasons, clock peak/offpeak,
  weekends/holidays and negative truth. Centered aggregate bias is mechanically
  zero, not evidence of accurate forecasting. Negative-truth segmentation is
  descriptive and cannot support a causal regime claim.
- Mean origin MAE/RMSE/P95 are explicit averages of origin metrics. The report
  also shows each origin. No p-value or confidence interval treats overlapped
  hours as independent replicates. No economic/PnL/tail-risk/probabilistic or
  intraday superiority claim. Policy goals (2% MAE/RMSE gain, 70% origins with
  positive MAE gain, regime degradation ≤2%) are descriptive checks, not an
  automatic promotion or proof of statistical significance.

## Execution and review

`scripts/run_lt_local_benchmark.py` freezes source/code/config/runtime hashes,
builds per-origin solver inputs, fits the declared models, stores failures and
predictions, and writes comparative metrics. All runtime outputs are below
`build/local-lt-benchmark-20260907/`; the original D300 artifacts stay unchanged.
The run checks the frozen plan before using cached work. Production, promotion,
scientific admission and trading authorities remain false. The local execution
manifest truthfully records real-data fits/scoring and local tuning performed.

Verification covers native feature/target/postprocessing parity including DST,
strict training cutoffs, common masks, monthly invariance, missing coverage,
seasonal fallback and no synthetic-container calls. Independent recomputation
of saved forecast errors and market repricing accompanies the final report.

Consumer/model team owns code, leakage checks, common inputs and metrics. Data
engineering owns gaps and historical availability/revision semantics. Product
owner owns loss priorities. Externally registered origins, admitted timing and
an independent future holdout remain requirements for scientific confirmation.

The next architecture experiment should address the positive target/clipping
and the native maturity feature (zero throughout training); neither a different
regressor nor a stronger retrospective score establishes negative-price or
far-horizon extrapolation validity. Intraday regularization is a separate test
with explicit coverage, since native DE quarter-hour history is under one year.
