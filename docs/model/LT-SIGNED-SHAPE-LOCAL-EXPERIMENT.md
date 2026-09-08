# Local signed hourly experiment v1 — 7 September 2026

This D304 protocol is fixed before its fits. It implements D303's next act:
compare a signed hourly representation through the existing PFCAssembler and
final BASE/PEAK projection. No scientific origin, admission or promotion occurs.
Production/promotion/scientific_admission/trading/externally_registered/
countable_origin remain false. CPU only, retained D300/D301 inputs, no Warehouse,
GPU, AFRY, T057, new assembly adapter or protected-data write.

## Fixed experiment

- Six D301 origins and their exact saved delivery windows and monthly solver
  outputs; historical diagnostic years 2021–2022, development comparison
  2023–2026. The latter were already examined in D301/D303. No new independent
  assessment, tuning, outcome-selected horizon switch or hidden holdout access.
- Both representations start from the same complete consecutive Swiss hourly
  training months ending before each origin. Incomplete edge/open months are
  excluded. Original EPEX quarter-hours must agree four-to-one before transport
  is reduced to observed hours; both autumn hours remain distinct. Hourly training
  does not merge the repeated clock hour as the original native MLP did.
- Signed labels: price minus its own historical Swiss monthly mean, EUR/MWh.
  Every finite negative-price hour remains. Future realized means are forbidden
  as predictors. Prediction centering uses predicted values only.
- Ratio control: same underlying history and hourly covariates, daily mean >5,
  target clipped to [0.2,3], prediction floor0.1, daily normalization, clip[0.4,2].
  Native weekly factors are fitted on this same closed-month history using the
  existing180-day weighting and daily filter. Existing multiplicative damping
  and prompt bridge are retained in that lane. Therefore the contrast concerns
  the representation/composition package, not clipping alone.
- Two methods per representation: untuned arithmetic mean by season/day-type/
  hour (fallback season/hour, hour, global), and LightGBM4.6.0 on the existing
  nine features. Both use the same calendar conventions, per-origin historical
  hydro mapper and pre-origin hydro climatology for predictions. Maturity stays
  zero in training and uses the existing forecast feature; this experiment does
  not claim learned maturity. No outages/physical/CT covariates added.
- LightGBM parameters fixed from the previous declared selected configuration,
  with no search here: regression_l1, learning_rate0.05, min_data_in_leaf100,
  n_estimators300, num_leaves31, deterministic CPU/force_col_wise, n_jobs1,
  random/bagging/data/feature_fraction seeds42. Reuse existing fit/predict kernels.
  No sample weighting for either LightGBM fit. Twelve estimator fits total,
  twelve reference calculations, twenty-four new candidate curves.
- Saved D301 corrected MLP remains the global comparison anchor. Six cache-loaded
  native models must reproduce their saved prices with the evolved assembler
  within1e-10 EUR/MWh, with no refit. Six additional signed-input round trips use
  the old pre-projection hourly price shape; their final hourly means must match
  the old final hourly means within1e-9. This isolates integration compatibility
  from model skill. These controls are never new model candidates.

## Signed assembly contract

`PFCAssembler.build(..., signed_hourly_shape=series)` accepts a real numeric,
finite, monthly-neutral EUR/MWh Series on whole Swiss hourly months and the
matching full quarter-hour delivery grid. An explicit solver BASE is required
for each month. One-to-four expansion adds no learned intrahour resolution.

The signed prior is B plus the supplied shape. It replaces f_W and f_H together,
with no hidden division by B, clipping, maturity damping or legacy bridge. The
existing solver monthly conservation and final hard BASE/PEAK projection apply.
Reported multiplicative factors are neutral; the signed contribution has its
own EUR/MWh column. Intraday factors must be exactly neutral. Physical/hydro/
outage forecast contexts, water-value/uncertainty components, active modulation
or floor flags are rejected, rather than silently combined. This first contract
does not qualify additive intraday, hydro or probabilistic composition.

## Verification and interpretation

Score final and pre-projection prices separately using unchanged D301 hourly
monthly-centered metrics and complete truth-month masks, ending at September1
2026 Swiss time. Report MAE/RMSE/P95, negative truth, seasons, peak/offpeak,
weekend/holiday and delivery years1/2/3 with exact coverage. Average origin
metrics with equal origin weight; overlapping years remain dependent.

Require finite output, identical candidate scoring populations, solver monthly
residual <=1e-9 EUR/MWh, supported hard-product residual <=1e-6 and no CRITICAL
product gate. Preserve all raw QUOTE_CONFLICT/UNSUPPORTED statuses and their
source identity; no waiver. Failed/non-converged candidates are retained and
excluded from a complete comparison. No retry with outcome-dependent settings.

Check the charter's2% MAE/RMSE,70% origin wins and critical-regime criteria
descriptively. Improvements against the ratio control and D301 incumbent have
different meanings; neither establishes independent confirmation. A central
forecast is not required to reproduce historical negative-price frequency.
Historical benchmarks do not validate2030 structural assumptions or FMV PnL.

The D30175-file source/config capture precedes the assembler edit:
`build/lt-signed-benchmark-20260907/baseline/manifest.json`, SHA-256
f51fb0e62b13a9eba6c86e55d8f3dd654ee3edf98264b981ae69d683c54ab129.
D300/D301/D302 artifacts and scientific pins stay unchanged. The live assembler
source now has a new hash; an old current-source hash check correctly rejects
it. Do not relabel the evolved checkout as the frozen scientific v6 runtime.
Local compatibility checks establish numerical regression evidence only.
