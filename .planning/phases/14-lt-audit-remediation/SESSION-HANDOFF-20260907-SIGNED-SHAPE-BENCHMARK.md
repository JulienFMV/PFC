# D304 — Signed shape integration and local CPU comparison

## Final status — complete

The authorized lot is implemented, executed and independently reviewed. No
active process remains. This status supersedes the historical checkpoint and
its remaining-work list below. Root HANDOFF, this session note,
`docs/model/LT-STRUCTURAL-SHAPING-CONTRACT.md` and D304 in DECISION-LOG record
the resulting sequence; the pre-fit experiment spec remains frozen.

Primary delivery: `build/lt-signed-benchmark-20260907/RAPPORT-COMPARATIF.md`.
The task root also contains `comparison-verified.csv`, `origins-verified.csv`,
`delivery-years-verified.csv`, `verification.json`, `verify_and_report.py` and
`review-console.log`. Raw curves, models, stage errors, product gates, inputs,
plan and complete manifest are under `run-v1/`. A closure manifest inventories
the final task artifacts and documentation; its path is `closure.json`.

### Results and decision

Equal-origin averages of monthly-centered final hourly-price errors on the
four previously exposed development origins2023–2026, EUR/MWh:

| Candidate | MAE | RMSE | MAE gain versus D301 MLP |
| --- | ---: | ---: | ---: |
| D301 corrected MLP |22.817767|33.604667|0%|
| Ratio seasonal, matched closed months |23.092102|33.856670|-1.202284%|
| Ratio LightGBM, matched closed months |22.892232|33.751600|-0.326347%|
| Signed seasonal |20.345234|30.180010|+10.836000%|
| Signed LightGBM |23.866017|34.617472|-4.594007%|

The signed seasonal reference is the promising development candidate. Its RMSE
gain is10.191016%, with MAE wins4/4 origins:26.51%,4.84%,5.12%,5.35%. All declared
aggregate segments improve; the smallest MAE gain is3.752764% in months25–36.
Negative-truth-hour MAE improves80.1430 to75.3280 (about6.01%). Prediction
negativity is not an optimization target: eligible negative predicted hours
by origin are0/20/38/38, without any frequency-matching rule.

Delivery-year MAE for MLP versus signed seasonal: year1 22.7943 vs18.6927
(four origins), year2 22.9828 vs20.1342 (three), year3 23.5156 vs22.6331
(two). Exact unrounded scores are in the CSVs. Unequal/incomplete year coverage
prevents interpreting these as a causal maturity experiment. The common score
population is70,101 origin-hour pairs and32,135 distinct hours.

The signed seasonal comparison improves before and after projection; its
pre-projection MAE is21.218055. All candidates retain the same raw market gate
counts as D301, including absent PEAK and QUOTE_CONFLICT, with no CRITICAL or
waiver. Monthly solver residual max2.518163455533795e-11 EUR/MWh. Signed LightGBM
fails to improve despite the signed target, so increased model complexity is
not adopted by default. No model production flag or current D300 curve changed.

The measured contrast is the representation/composition package. Ratio controls
have the same underlying closed-month observations and nine covariates, but
retain their declared positive-day filter, ratio clipping, weekday factor,
damping and bridge. This does not isolate the causal contribution of a single
clip. All candidate settings were fixed before fitting; no outcome-driven retry.

Next local act: qualify this candidate in a comparison with the components needed
for the complete PFC. The current signed API intentionally rejects active 15min,
water, uncertainty and exogenous overlays; expanding an hour into four equal
quarters does not qualify those components. D303's coherent public2030 scenario
preparation remains relevant; no isolated storage kernel should displace the
next measurable final-PFC improvement. These results do not establish2030 skill,
historical PIT availability, full charter compliance or independent confirmation.

### Execution and verification evidence

Run:13:49:20.639594Z–13:51:49.718143Z, about149seconds.12 actual LightGBM CPU fits,
12 reference calculations,24 new candidate curves,6 cached MLP comparisons,
6 signed integration controls. No failed candidate or refit of native MLP.
The native weekday tables were fitted on the common closed-month history;
that context preparation is recorded separately from estimator-fit counts.

`python -B build/lt-signed-benchmark-20260907/verify_and_report.py` independently
reconstructed all60 pre/final error arrays using saved prices and direct monthly
arithmetic, maxresidual1.9895196601282805e-13 EUR/MWh. All12 serialized models
reproduce saved predictions exactly. The actual run review verifies
125 pinned inputs,81 code/config files and276 output artifacts; counts are in
`verification.json`. Closed-month boundaries, label arithmetic, retained negative
observations, target/feature timestamps and common populations are checked.

All six native quarterly-grid curves reproduce D301 exactly (maxerror0).
Six signed round-trip controls preserve hourly means within6.252776074688882e-13.
Default behavior is therefore regression-checked on the retained real cases.
The live assembler SHA-256 is now
`e30c2f8d30aca02149b40ead0e36349bb42251d60d87c1b1969ed428859f887b`.
Scientific v6 pins/old benchmark artifacts remain unmodified, but this changed
live dependency must not be described as the old frozen runtime.

Tests remain44 new-unit passes; full selected matrix161passes,5skips,2proven
pre-existing legacy-fixture failures. Four skips concern optional CT imports
(LightGBM is exposed only in the dedicated benchmark process, not installed in
the base interpreter; Torch/TensorFlow absent); the fifth is the existing gated
Phase5 synthetic summer-bowl acceptance test. These skips/failures are unchanged,
not new approvals or proof that all product gates passed. After the selected
matrix, only the two failing tests were rerun to attribute the legacy failures.
Final `git diff --check` and new-file whitespace checks pass.

| Artifact below task root | SHA-256 |
| --- | --- |
|run-v1/plan.json|3125c1d04daf894fa34568f42e4aa4df47727e7b53bb3184265852287f908f94|
|run-v1/manifest.json|446c940cc1a62845bbde664bdc61509078efca802467ccfd7451982031f134af|
|run-v1/complete.json|1812e78355128d5b85c8ca4e213ea64b342ac994e2a1a0df6bd05f73df471c6b|
|comparison-verified.csv|b74ceb142f84747a71cab9567d2b80a7b03663959f779b5c10fab7b02f79988f|
|verification.json|cdb25a6fb5464adf9b25068939997f51a12cb247bb8edce465d3d8bf6171920b|
|RAPPORT-COMPARATIF.md|4ca2b4dc031508f5897701abf5c62dff4abb4e0fb9f8998c6a4431182e1aafa8|
|tests-matrix-v1.xml|a3ad5f9b595e63f24828eca429d1aa1da635fb9a5a5c6d5f639aa24abcceba7d|

All production/promotion/scientific_admission/trading/externally_registered/
countable_origin authorities remain false. No Warehouse/GPU/AFRY/T057, no new
dependency install, no CT or protected-data mutation, no external communication.

## Execution checkpoint — 7 September 2026

User authorized continuing D303's immediate lot. No further permission is
required. Objective: qualify a signed hourly EUR/MWh input in the existing
assembler, then execute the fixed two-representation/two-method comparison
through final market-constrained prices. No new assembly adapter or promotion.

Implemented files:

- `pfc_shaping/lt/model/assembler.py`: optional keyword-only signed hourly shape,
  validation and private assembly path using the same final BASE/PEAK projection.
  Full Swiss months, explicit monthly solver BASE, neutral ancillary layers;
  shape includes weekday variation and replaces multiplicative f_W/f_H.
- `pfc_shaping/lt/signed_benchmark.py`: closed-month targets and common unclipped
  calendar-cell reference. Negative observations and repeated DST hours retained.
- `scripts/run_lt_signed_benchmark.py`: fixed local experiment, source pins,
  baseline compatibility, CPU fits, pre/final projection scores and artifacts.
- `tests/test_signed_hourly_assembly.py`, `tests/test_signed_benchmark.py`.
- Frozen pre-fit spec: `docs/model/LT-SIGNED-SHAPE-LOCAL-EXPERIMENT.md`.
- Root handoff, this session handoff and Phase14 decision log at closure.

Task root: `build/lt-signed-benchmark-20260907/`. Before any assembler edit,
75 D301 source/config bytes were copied under `baseline/sources/`; manifest
SHA-256 `f51fb0e62b13a9eba6c86e55d8f3dd654ee3edf98264b981ae69d683c54ab129`.
Old assembler SHA-256 remains captured as
`32b62c8c0174e226c616e0ab8fc392fbadc5642127ae100ff6efeea421f41e76`.
The live assembler is deliberately a new source version. Existing D301/scientific
pins are not rewritten; old current-source hash checks should reject this evolved
checkout. Other pinned dependencies and D300/D301/D302 artifacts stay unchanged.

## Commands and settings

Every shell command verified exact canonical cwd and Git top-level first.
Interpreter `build/conda-runtime-v41-model-source/python.exe -B`; mutable runtime
TEMP/TMP, APPDATA/LOCALAPPDATA, MPLCONFIGDIR, XDG_CACHE_HOME, NUMBA_CACHE_DIR,
JOBLIB_TEMP_FOLDER, PYTHONUSERBASE and PIP_CACHE_DIR under task-root `runtime/`.
PYTHONDONTWRITEBYTECODE=1, OMP/OPENBLAS/MKL_NUM_THREADS=4,
CUDA_VISIBLE_DEVICES=-1. Existing pinned LightGBM4.6.0 dependency tree from D301
is read without a new installation. LightGBM uses one thread and seed42.

Tests: `python -B -m pytest` with repo-local basetemp/cache and XML under task root.
New units plus required LT suites and relevant existing assembly/negative/target
tests were executed. Matrix v1:161 passed,5 skipped,2 legacy fixture failures.
The standalone new-unit run v3 passed44 tests.

The two failures in `tests/test_phase05_negative_prices.py` are
`test_phase05_baseline_regression` and `test_phase05_baseline_5bisA_via_enforce_true`.
`check_legacy_fixtures.py` loads the exact captured pre-edit assembler and reruns
these tests. Both fail with identical assertion messages. Proof retained in
`tests-preexisting-v1.xml`, `preexisting-fixtures.log` and
`preexisting-fixtures.json`. Fixtures were not changed or waived. These are
pre-existing Phase5 golden-curve mismatches; no claim that the full suite passes.

Initial new-unit failures were test-fixture mistakes: the sentinel hourly model
omitted the explicit `reference_date` signature required by the constructor
(38 failures), then `np.repeat(Series)` retained duplicate labels in the malformed
quarter-grid fixture (37 passed,1 failed). Fixed the test fixture only; v3 passes.
An early read command also mistyped `-First sixty`; corrected with no mutation.

Execution command:

`python -B -m scripts.run_lt_signed_benchmark --output build/lt-signed-benchmark-20260907/run-v1`

Console retained by `Tee-Object` in `run-v1-console.log`. Plan frozen before fits,
SHA-256 `3125c1d04daf894fa34568f42e4aa4df47727e7b53bb3184265852287f908f94`.
Six D301 origins;12 planned LightGBM fits,12 seasonal reference calculations,
24 new candidates plus6 cached MLP comparisons and6 signed round-trip controls.
No tuning here. Regression_l1,learning_rate.05,min_data_in_leaf100,
n_estimators300,num_leaves31; native helper fixes deterministic settings/seeds.
Same closed-month underlying observations, nine features and fitted hydro
climatology per origin. Ratio filtering/composition and signed composition are
explicitly different packages, not a clipping-only causal experiment.

## Remaining work at this checkpoint

Wait for the running command, inspect failures and exact results, independently
verify saved predictions/models/input hashes and market/monthly constraints,
write the comparative report, update this handoff/root/decision log and run
targeted whitespace checks. Do not modify any frozen experiment code/spec while
the run is active. Numerical native regression and signed round-trip controls
must pass before each fold's new fits.

No Warehouse, GPU, AFRY values, T057, CT overlay, protected-data changes, source
attestation fabrication or production/promotion/scientific/trading authorities.
Known revised-history availability, dependent development origins, missing PEAK,
quote conflicts and neutral intraday/water scope remain visible.
