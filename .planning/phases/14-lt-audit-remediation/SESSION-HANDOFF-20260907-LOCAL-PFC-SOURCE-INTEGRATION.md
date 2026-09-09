# Local FMV PFC source integration — 7 September 2026

## Active objective and authorization

The user explicitly authorized everything necessary for the best achievable
local FMV PFC, including native CPU calibration. This supersedes the earlier
read-only/no-Warehouse scope for this local construction task. It does not
grant scientific or production admission, promotion, T057 opening or AFRY use.
Do not ask for CPU calibration authorization again. No new assembly adapter:
reuse `PFCAssembler` and the existing monthly BASE solver.

Completed continuation, decision **D-20260907-300**. A local curve and reviewed
exports now exist. Main delivery is **October 2026-December 2029**, 114,052
quarter-hours; the full native solver horizon reaches December 2032.
Start with `build/local-pfc-source-preflight-20260907/README-PFC-FMV.md`.
No independent predictive superiority or production admission is claimed.

## Exact source evidence

All task artifacts are below `build/local-pfc-source-preflight-20260907/`.
The canonical checkout remains `C:\Users\jbattaglia\PFC_LT`.

- Warehouse: one start accepted at `2026-09-07T07:33:31.5009211Z`; observed
  RUNNING at `07:45:38.731Z`, existing `2X-Small`, classic/non-serverless,
  no resize/create. One stop request at `08:02:54.582Z` returned HTTP 403.
  State then remained RUNNING, auto-stop 45 minutes. A final read-only GET
  at **08:46:08.331Z** confirmed **STOPPED**. Do not repeat start/stop.
  Receipts: `warehouse-preflight.json`, `warehouse-observation-*.json`,
  `warehouse-finish.json`, `warehouse-final-observation.json`. Acquisition
  is finished; continue offline.
- Current EEX: existing `build/databricks-eex-daily/capture.ps1 -Execute`, one
  exact CH/POWER three-Gold-table SELECT, statement
  `01f1aa90-a1dc-1fba-975f-b86bd22911eb`. Capture is in
  `build/databricks-eex-daily/2026-09-07/`, 84,370 source rows. Query SHA-256
  remains `54a2e7e1752af4506673d2b5cbc2666f0deea45ec96e6d82561e4b265c78797a`.
  Offline `eex-replay/` contains 34,977 solver-history rows, quotations
  2019-01-02 through **2026-09-04**, retaining normalization quarantine.
- ENTSO-E: observed Silver table 1,551,865,334 bytes / 137 files, `_year` and
  `_month` partitions; current Gold dictionary 507,147 bytes. Exact dictionary
  query returned 22 rows, including the 11 physical series needed by LT.
  Silver Delta **version 55** was frozen through `DESCRIBE HISTORY`.
- One value-blind coverage SELECT returned 149 aggregate rows. One subsequent
  frozen-version Arrow export returned **239,288 rows / 8 chunks**,
  149,110,952 Arrow bytes, source Parquet SHA-256
  `33e1ff143e119f0c2e5cb0954b5f2198b340ab32b132095ad8d66cbd4845e2b4`.
  Exact SQL, Arrow chunks, redacted API responses and manifest are in
  `entsoe-export-arrow/`. Temporary Azure SAS URLs were never printed or
  persisted; the Databricks bearer token was not sent to Azure downloads.
  This is latest observed evidence, not retrospective PIT or final truth.
- Independent LSEG control: current Silver `ge_market_lseg_curve_values`
  contains 17,945,191 bytes / 3 files; frozen Delta version **2911**. One
  29,792-row bounded SELECT, statement
  `01f1aa92-198b-138f-bdc4-2fd612232df7`, source SHA-256
  `b50e8ace985ad101d08b3a0813570cf8021baa46ec3b7456e05691a14bcd0624`.
  The German EPEX reference equals ENTSO-E sequence 1 exactly on **29,768**
  quarter-hours from 2025-10-01 local through 2026-09-07 00:00Z excluding July.
  Sequence 2 MAE is 11.081942018274658 EUR/MWh. July was not re-compared;
  D282's existing 2,976-quarter-hour equality evidence is retained.
  `lseg-control/reconciliation.json` closes the local DE selection over the
  full calibration period. The other 24 LSEG rows cover the missing CH day;
  they are diagnostic only, not silently substituted into training.
- Hydro: national SFOE OGD17 HTTP 200 capture at
  `2026-09-07T07:40:14.9074208Z`, source SHA-256
  `793fd38b216d547c30c098d93f5ee1e886bf33290570bec590e851f82ba255c0`.
  `hydro/` retains source bytes, real HTTP headers, unsigned raw envelope,
  parser config, raw/derived Parquets and replay audit. Window 2009-01-01
  through 2026-09-01 yields 922 weeks, through Monday 31 August local.
  The existing recent-eight-week water-value support check passes.

There were 14 accepted SQL statements: 13 successful, one failed during
timestamp-travel analysis. One additional INLINE submission was rejected by
HTTP 400 without a statement ID. Table byte counts are planning proxies;
actual query-history scan/DBU counters have not yet been captured.

## Real findings and consumer treatment

- CH price history lacks the Swiss market day 3 September; consumption lacks
  seven hours in September. Do not interpolate or silently fill these gaps.
- Several physical series contain overlapping variable-length blocks.
  `input-audit/overlap-diagnostic.json` separates identical repeated values
  from revised values. No conflicting equal source ordering was found in the
  consumed full-year physical window after the new local replay policy.
- Physical replay expands blocks first, then selects by last observation,
  revision, first observation; tied contradictory values fail. It requires a
  complete explicit delivered window, correct native alignment, finite
  nonnegative values, exact dictionary binding, units and signed flow direction.
- PRD physical `FieldName` is descriptive (`ch_actual_load`, etc.), not the
  synthetic fixture's literal `quantity`; A11 flow `ProcessType` is null.
  The consumer now accepts those exact semantic cases without changing the
  original PIT policy. Arrow microsecond timestamps must be converted to
  nanoseconds before quarter-hour timestamp arithmetic.
- SFOE regional reconciliation now applies to consumed weeks. Global finite,
  nonnegative, capacity, unique-date and future-date checks remain. The 2004
  and 2008 one-GWh source discrepancies are outside the consumed window;
  no source values or reconciliation tolerance were changed.

Prepared local frames are in `prepared-inputs/`, with file hashes and code
hashes in `manifest.json`:

- CH: 268,984 transport quarter-hours, native hourly source, 2019-01-01
  through 2026-09-02 22:00Z (before the missing Swiss day).
- DE: 32,160 native quarter-hours, 2025-09-30 22:00Z through
  2026-08-31 22:00Z; exact selected sequence 1.
- Physical inputs: 35,040 quarter-hours, the full Swiss year September 2025
  through August 2026. Four derived features reuse `build_entso_features`.
  105,642 eligible raw intervals expand to 406,294 transport rows;
  20,854 superseded/repeated transport rows are resolved explicitly.
- Hydro: 922 weekly observations. Three native CPU components were fitted
  once; their original fitted bytes remain unchanged after the loader fix.

## Code and verification

Changed in this continuation, preserving prior dirty work:

- `pfc_shaping/data/governed_lt_acquisition.py`: selected-window SFOE totals.
- `tests/test_governed_lt_acquisition.py`: in-window rejection/off-window acceptance.
- `pfc_shaping/data/databricks_lt_materialization.py`: exact PRD quantity
  semantics, a separate `materialize_entsoe_latest_observed_features` local
  function, and Arrow timestamp-unit correction. No signed replay mode added.
- `tests/test_databricks_lt_materialization.py`: local revision, overlap,
  chronology, window, DQ, mapping and microsecond timestamp regressions.
- `pfc_shaping/lt/model/shape_hourly_mlp_hydro.py`: small explicit subclass
  preserves the frozen scientific incumbent while correcting Swiss civil-week
  climatology and forward alignment of actual weekly observations. Network,
  fit and assembly APIs are inherited, with separate local model identity.
- `tests/test_shape_hourly_mlp_hydro.py`: two actual-observation/DST regressions.
- `pfc_shaping/lt/model/shape_intraday.py`: load drops Parquet null padding of
  absent optional correction keys; it no longer creates NaN factors.
- `tests/test_intraday_persistence.py`: finite and exactly identical `get` and
  `apply` outputs before/after saving cells with different sparse coefficients.
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`: local current-observation
  contract and consumer/data-engineering/evaluation ownership.
- `.planning/HANDOFF.md`, `DECISION-LOG.md`, this handoff.

The original `shape_hourly_mlp.py` was restored after the frozen-source test
caught an attempted direct fix. There is **no textual diff** in that file;
its normalized hash remains
`8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef`.
Git may list it due to line endings. Do not update frozen evaluation pins.

Commands use canonical cwd/Git-root guards and repo-local Python
`build/conda-runtime-v41-model-source/python.exe -B`. Early tests used
`scripts.run_workspace_local`; later tests called the same interpreter's
`-m pytest` directly with all mutable caches and temporary paths under build.
The convenience harness repeatedly inventoried the full runtime for over
eight minutes. Its owned worker was terminated only after receipt/PID creation
identity and repo-local executable checks; see `test-preflight-stop.json`.
No elevation, shell override or external mutable directory was used.

- `d300red`: SFOE reproduction, 1 failed / 1 passed.
- `d300green`: SFOE subset, 7 passed.
- `d300matrix`: acquisition + required LT minimum, **126 passed, 5 skipped**,
  one existing Energy Charts timezone warning.
- `d300observed`: 8 passed, 2 failed (microsecond timestamp defect and a
  deliberately conflicting fixture ordering corrected before the next run).
- `d300obsgreen`: whole materialization file, **45 passed**.
- `d300mlpred`: stopped during harness preflight, not a completed test result.
- `mlp-red.xml`: 2 failed, proving silent neutral hydro and wrong ISO weeks.
- `tests-final.xml`: 203 passed, 5 skipped, 1 failed (frozen MLP hash).
- `tests-final-v2.xml`: 204 passed, 5 skipped after isolating the correction.
- `si-red.xml`: 1 failed, reproducing NaN factors on sparse Parquet round trip.
- **`tests-final-v3.xml`: 217 passed, 5 skipped, one pre-existing Energy Charts
  timezone warning, 7.73 seconds.** This 13-file matrix includes the required
  LT minimum, materializers/acquisition, hydro, intraday, frozen target/assembly/
  execution and ENTSO-E contracts. Four skips are optional CT packages; one
  requires an exact audited LT wheel/dependency root not supplied to this source
  test run. No dependency was installed to silence a skip. Installed-wheel/CI
  qualification is not claimed by these library tests.

Runtime scikit-learn is 1.6.1; `MLPRegressor.fit` lacks `sample_weight`.
The existing MLP's hourly target averaging cancels its nominal age weights;
do not claim that the network fit applies exponential age weighting. Do not
silently upgrade runtime or introduce a different estimator/bootstrap policy.

## Errors retained, not hidden

- Timestamp-as-of later than the last Delta commit failed before data scan;
  corrected using observed Delta version 55, preserving failed receipt.
- Initial 200-MiB INLINE request returned HTTP 400; corrected to the documented
  EXTERNAL_LINKS/Arrow path, with bounded download and redacted temporary URLs.
- Python SFOE TLS attempts failed; Windows default TLS succeeded without any
  trust bypass or certificate/security-policy change.
- EEX replay artifacts were written successfully before an optional console
  projection referenced nonexistent `segment`; that display was removed.
- A test run-id over 16 characters was rejected before execution; corrected.
- Several read-only `rg` wildcard/path mistakes were corrected. Use `rg -g`,
  not Windows wildcard path arguments.
- First assembly failed on non-finite optimizer inputs. Task-only tracing in
  `curve-diagnostic/nonfinite-diagnostic.json` isolated 7,304 NaNs in `f_Q`;
  every other shape component was finite. The loader regression/fix corrected
  this without refitting or filling missing curve prices.
- Initial task manifest incorrectly labeled the absence of CRITICAL as all
  product gates passing. The existing independent auditor established nine
  unaccepted quote conflicts. The corrected manifest says
  `LOCAL_PFC_GENERATED_WITH_QUOTE_CONFLICTS`, `all_product_gates_pass=false`;
  the original receipt is preserved in the review directory. No price changed.
- Plotting libraries were absent. A copied repo-local pip wheel installed only
  plotting dependencies under the task's `plot-libs/` with repo-local caches;
  `plot-install.json` records exact wheel URLs/hashes. The model interpreter,
  dependencies and fitted bytes were not changed. Initial review imports and
  the auditor's legacy timestamp format were corrected in task scripts.
- A diff check with `core.autocrlf=false` incorrectly treated existing CRLF
  line endings as trailing whitespace. The normal repository-configured
  `git diff --check` passes; no unrelated line-ending rewrite was performed.

## Monthly solver, calibration and generation

`monthly-solver/inputs.json` retains the exact 38 quotes, source/config hashes
and settings. Existing defaults were used: lambda prior 1e-6, month smoothness
0.1, year-over-year smoothness 150, shape 100, neighbor shrinkage 0.5, six-year
history, structural template amplitude 150, panel/history/structural weights
1/0.5/1, constraint tolerance 1e-9, quote-conflict tolerance 0.01 and
stationarity tolerance 1e-7. The source path is the current repo-local replay.

Existing `solve_monthly_level_authority(..., allow_unverified_inputs=True)`
was used in its explicitly allowed non-promotional research lane. Its inner
unsigned provenance placeholder says `TEST_FIXTURE`; this is a library
sentinel, **not the input data**. Actual input identity is bound in the outer
`inputs.json`, `source_hashes`, EEX acquisition and replay manifests. Do not
promote this nested placeholder, its run date or its eligibility to signed
source evidence. No synthetic/legacy replacement or forged ForwardSnapshot
was provided. The observed quotation date is 4 September, not the placeholder's
7 September run date.

The existing solver's 75-month grid is October 2026-December 2032. KKT maximum
constraint residual is 5.6843e-14 EUR/MWh; stationarity 1.0158e-11, condition
2231.44, rank 16, nullspace 59, no ridge/lstsq fallback. Panel is UNSUPPORTED,
history PARTIAL_HISTORY_FORWARD, structural prior STRUCTURAL_TEMPLATE. These
are declared coverage limits, not hidden critical gates.

`fitted-models/manifest.json` binds prepared inputs, software, source code and
three fitted components, completed 08:26:58.357373Z:

- Hourly `HydroAlignedShapeHourlyMLP`: 66,856 hourly samples after the native
  positive-day filter, 322 iterations, random_state 42, 58.141 seconds. Training
  loss 0.01529754 and in-sample factor RMSE 0.1748 are not future scores.
- Intraday native `ShapeIntraday`: 480 cells, three sparse correction cells,
  4.265 seconds. Existing experimental sparse regularization remains disabled.
- Water value: 93 months, native bounded beta -0.1, 0.078 seconds.

Four CPU/BLAS threads; no GPU, AFRY, CT overlay, uncertainty fit, challenger
selection or holdout access. Outages are explicitly absent in the optional
native interface, not claimed to have been observed as zero.

`loader-fix.json` preserves the original fit manifest hash and binds the
corrected intraday loader. The archived fit-time source hash is verified;
AST comparison proves all code except `load()` unchanged. Additional fits: 0.
The original fit manifest and all fitted model files remain unchanged.

Existing `PFCAssembler` used solver monthly authority, both legacy level
cascades disabled, same-first peak policy, ArbitrageFreeCalibrator smoothness
1.0/tolerance 0.01, no raw-calibration fallback, monthly tolerance 1e-9.
Physical forecasts use the complete observed year's existing climatology;
hydro anomaly follows the existing linear attenuation assumption, starting
from the last observed week. All supporting forecast frames are retained.

Generation completed at 08:36:48.525949Z. Final product projection has 91
constraints and maximum residual 8.214e-12. Main output is an exact time slice
of the full native curve, with no post-solver level patch.

## Delivered artifacts and review

All paths below are relative to the task root:

| Artifact | SHA-256 |
| --- | --- |
| `prepared-inputs/manifest.json` | `36f27a705bdf56ff723d778d226ed3b5f4ae594f20e54b402cc3d6385088b3ef` |
| `fitted-models/manifest.json` | `209a427339159a0e39ca9c7705ad55f522ceb943534f6038efade157b842c5ca` |
| `loader-fix.json` | `1b9d623faf0754de0b608bebe6d9f45dbe60d842e31d2abb5f2488c8d3a3340e` |
| `monthly-solver/result.json` | `250cc0d89b32a7e1c92a4358f9b6a6d7aba800d0da9e02223468e5cee0b42aad` |
| `curve-final/pfc-fmv-ch-15min.csv` | `d35599fa0aaf51b2e89b85857f2cf4a814e624dbef51375cbf52ae59d3f16b02` |
| `curve-final/pfc-fmv-ch-15min.parquet` | `c094ea178235154e52b1a897cd81104a8a6c6bca8e167c370b6b2afd5a33b2ca` |
| `curve-final/pfc-full-native-horizon.parquet` | `e9e0afb9844b4206b9402784dbb5f5c00e18e9d14216d5aec2520af211b890e4` |
| `curve-final/manifest.json` after status review | `ec9680351a79fac993f12b6c023e5bbba194ca887eff885dc02feee4159acae4` |

`curve-review-final/` contains a separate quote repricer, existing strict
product audit from reread hourly CSV, monthly distributions, shape diagnostics,
PNG/SVG and `review.json`. PNG visually inspected: labels readable, monthly
levels and a full week at 15 minutes shown. Its shaded monthly percentiles
describe the distribution of deterministic prices, not forecast uncertainty.

- Main output: 114,052 rows, 39 whole months; native full output: 219,268 rows.
- UTC grids exact, finite prices, no gaps/duplicates. Swiss days: 1,181 normal
  96-row days, four 100-row fall-back days, three 92-row spring-forward days.
- CSV versus Parquet maximum absolute error 5.0000892e-11 EUR/MWh; main versus
  full curve is byte-value exact. Monthly means within 8.0717655e-12 EUR/MWh.
- All 38 raw BASE/PEAK quotes independently repriced at quarter-hour grain;
  largest error 0.00322317791 EUR/MWh, below existing solver tolerance 0.01.
- Strict product audit: **80 PASS, 9 QUOTE_CONFLICT, no CRITICAL/UNSUPPORTED**.
  Six redundant parent BASE/PEAK quotes concern Q4 2026, Q1 2027 and calendar
  2027; three further conflicts are their implied OFFPEAK identities. Largest
  implied conflict is 0.00584044 EUR/MWh. Rounding is a compatible explanation,
  not independently proven source causation. No conflicts were waived; the
  signed production hierarchy policy is absent, `all_gates_pass=false`.
- Main prices range 11.3889-297.7348 EUR/MWh, no negative quarter-hours.
  Native `p10`/`p90` are null; `confidence` is an uncalibrated horizon heuristic.

The amplitude inspection found no additional assembly error. Historical DE
intrahour relative-range p99 is 2.2811 versus 0.6184 in the local curve, using
absolute hourly mean floored at 10 EUR/MWh as the denominator. This comparison
is descriptive, across different periods and horizons, not a predictive test.
Some extreme base profiles have only 9-13 parent hours and fallback copies.
Do not smooth them ad hoc or declare the existing experimental regularizer a
winner without evaluation. The native positive construction, short DE support,
missing outages and canceled neural age weighting remain material limits.

## Commands and reproducibility

Every shell command checked both cwd and Git top-level against
`C:\Users\jbattaglia\PFC_LT`. Mutable environment variables TEMP, TMP,
APPDATA, LOCALAPPDATA, MPLCONFIGDIR, XDG_CACHE_HOME, NUMBA_CACHE_DIR,
JOBLIB_TEMP_FOLDER, PYTHONUSERBASE and PIP_CACHE_DIR were redirected under
the task's `runtime/`; PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES=-1,
OMP/OPENBLAS/MKL_NUM_THREADS=4. No system prefix or administrator change.

Task scripts retain the exact SQL and local actions. Acquisition scripts:
`preflight.ps1`, `metadata_sql.py`, `profile_inputs.py`, `export_entsoe.py`,
`lseg_preflight.py`, `capture_lseg_control.py`, `capture_hydro.ps1` and the
existing EEX capture script. Follow-up source commands: `reconcile_de.py`,
`audit_inputs.py`, `diagnose_overlaps.py`, `replay_eex.py`, `hydro_replay.py`,
`prepare_local_inputs.py`. Model/export commands used the same repo interpreter
with `-B` followed by task-root scripts `solve_monthly.py`, `fit_components.py`,
`bind_loader_fix.py`, `generate_curve.py`, `review_curve.py`, `inspect_shape.py`.
`finish_warehouse.py` made the sole stop request; `observe_warehouse_final.py`
made only the final GET. Do not rerun capture/fit scripts into existing outputs.

Final test command after the environment/cwd guards:

```powershell
& .\build\conda-runtime-v41-model-source\python.exe -B -m pytest tests/test_intraday_persistence.py tests/test_intraday_amplitude.py tests/test_shape_hourly_mlp_hydro.py tests/test_databricks_lt_materialization.py tests/test_governed_lt_acquisition.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_lt_evaluation_target_builder.py tests/test_lt_evaluation_curve_assembly.py tests/test_lt_evaluation_execution_contract.py tests/test_lt_entsoe_data_contract.py -q -o cache_dir=build/local-pfc-source-preflight-20260907/pytest-cache --basetemp=build/local-pfc-source-preflight-20260907/pytest-final-v3 --junitxml=build/local-pfc-source-preflight-20260907/tests-final-v3.xml
```

Final `git diff --check` passes. Existing dirty work was preserved; no commit,
production publication, protected heavy desk-data mutation or external message.
`session-artifacts.json` inventories retained task evidence (excluding caches,
temporary test trees and plotting packages), hashes the exact changed source/
test/governance files and repeats final model/input/export hash verification.
`close_session.py` is the read-only verifier that creates this local inventory;
it performs no network calls, fit or generation.

## Next useful work and remaining external blockers

The immediate local-generation objective is achieved. Reuse these artifacts
for desk review. The smallest next model act is a scoped diagnostic/evaluation
design for the already implemented sparse intraday regularization and explicit
neural age weighting, with unchanged EEX/monthly constraints and frozen input
identity. Negative-price behaviour and unavailable outage inputs also require
an explicit model/data treatment before calling this the best predictive curve.

Data engineering owns producer gaps, revision/publication meanings and
platform permissions. The consumer owns exact units, calendars, selected
windows, feature semantics, numerical correctness and exported-curve repricing.
Do not offload our consumer defects to producer certification.

The scientific runner still lacks externally countable origin registration,
the required signed/PIT source admissions and an independently frozen future
holdout. Its smallest authorized external-evidence act remains receiving the
actual owner profile/attestations and verifying their exact byte/origin/window
binding read-only. A locally frozen October 2026-September 2027 plan is not
independent admission; T057 remains sealed. These blockers did not prevent
local generation, and no artifact here clears scientific or production gates.
