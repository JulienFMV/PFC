# D318 — safeguarded level-conditioned signed CPU benchmark

Completed locally on 2026-09-08, canonical workspace only. Starting/public HEAD
is `d07aca6c1745e4c052fc51e0df3fed3834d6c2e4`; no commit, push or new remote CI.
Read AGENTS.md, .planning/HANDOFF.md, D317 audit response and D318 decision.
User authorized guards, frozen benchmark, tests, independent checks, report,
experimental exports and governance. All requested local work is complete.

## Decision and result

D304 retained, D305–D307 artifacts/comparisons preserved. Both global slope
recipes fail. Shape MAE/RMSE gains (macro mean over four exposed origins):
ridge1 −4.268087%/−1.677367%; ridge10 −0.505495%/−0.031569%.
Each wins2/4origins.37/4adverse origin×regime×error cells, including2/0without
multi-origin support; all adverse cells veto. No month-specific selection.
This rejects these recipes, not every level-conditioning hypothesis.

Model: D304 raw plus54calendar-harmonic slopes×archived solver B/100;
global ridge1/10, inverse-pairs-per-delivery weights, duration-aware monthly
centering before unchanged assembler/projection. Historical solver vintages
are retrospective latest-observed reconstructions, NOT independently PIT.
Realized monthly means never replace the solver-level training covariable.
2021has no earlier solver pairs: explicit zero slope/UNSUPPORTED_NO_PAIRS.
2022has only one inner origin and all delivery levels outside training range.
Current extension has39months beyond training maturity, despite level support.

12actual fits,21assemblies;7D304 ablations are bit-for-bit equal in prices
(max error0.0). All six authorities remain false. No SQL, Warehouse, GPU,
AFRY, T057, CT, protected data mutation or new pilot capture. Pilot remains1/20.
Assembler, arbitrage-free calibrator and EEX contract selection match D308
hashes; no solver code change. Maximum monthly solver residual2.3987922759260982e-11.

## Guards

- F-01: `_validated_prices` now requires all delivered monthly BASE keys,
  before common multiplicative evaluation reaches the unchanged assembler.
  Quarter/year/previous-year fallback cannot enter through this boundary.
  This does not change the standalone legacy assembler fallback contract.
- F-05: shared stability screening records unsupported adverse segments and
  rejects populated nonfinite metrics; per-origin veto retained. D318 applies
  stronger vetoes to every origin/regime, including sparse adverse cells.
- F-06: maturity runner and independent verifier assert exact ordered hourly
  populations; D318 additionally checks the same population across stages.
- F-10: signed recency/stability/revision/maturity runners pass a sentinel
  with no fitted state. Real signed construction works; deliberate future
  hourly-state consumption fails. Old artifacts and source snapshots untouched.

## Freeze and failures

Local evidence root `build/lt-level-conditioned-20260908/`.
Protocol: `docs/model/LT-LEVEL-CONDITIONED-EXPERIMENT-20260908.md`.
Run-v1 plan SHA256:
`6455e9b27c0316dc10ae874fdb8d6888f1fe15621c7bc94d4d09823c42946a1f`.
Run-v1 stopped before first fit/assembly/score: constructor requires explicit
`reference_date` in sentinel.apply signature. Fixed only that signature and
strengthened the test to instantiate the actual assembler with the sentinel.
Failure retained in `run-v1/failure.json`; no abandoned performance results.
Run-v2 refroze unchanged recipes, thresholds, inputs and pairs before fit.
Executed plan SHA256:
`5668b23c6536ecbb7d6724c17307461ab9a9fb57db511a4a2044c9cb1dea5d4e`.
Plans bind input/source/pair hashes and archive source snapshots. Do not edit
them or the completed results. One unmatched documentation apply_patch failed
without mutation; a targeted patch subsequently appended D318 successfully.

## Verification and artifacts

- First focused tests62pass; second focused tests57pass; corrected constructor
  and model tests20pass. Final40-module matrix736pass/5skip/0fail/0error,
  127.239s. Exact argv in `tests-matrix-command.json`; JUnit and log retained.
  One pre-existing timezone warning from legacy Energy Charts loader.
- `independent-guards/verification.json`: actual checkpoint functions extracted
  and executed on synthetic adversarial probes for F-01/F-05; population shift
  rejection; real assembler sentinel and deliberate-consumption mutation probe.
- `independent-v1/verification.json`: separate spectral coefficient solver,
  separately implemented calendar basis, source/pair reconciliation,3288metric
  checks, exact populations/calendar coverage, decomposition, EEX product
  gates, monthly means, ablation and CSV readback. Max coefficient error
  1.354472090042691e-14; raw prediction error4.796163466380676e-14.
  Local independent arithmetic implementation, not independent human/remote CI.
- `preservation.json`:1063previous artifact bindings and2290input bindings
  verified after execution, same one-record pilot and chain. Protected source
  hashes included. `ablation-verification.json`: seven exact price replays.
- `RAPPORT-BENCHMARK.md`: full local report; `comparison.csv` reports before/
  after projection, full/shape/level/ramp errors and all regimes.
  `annual-coverage.csv`: actual full Swiss years versus partial/unsupported.
  `month-support.csv`: level/maturity support; no year falsely validated at Feb28.
- `run-v2/results/`: all curves/raw shapes/fit coefficients, diagnostics,
  per-origin regime gates, decision, receipts and output manifest.
  Existing quote conflicts stay visible; zero CRITICAL product gates.
- `exports.json`: six experimental CSV hashes and authorities; each recipe
  has54817native hourly predictions and219268repeated-QH rows, October2026
  throughDecember2032. Values retained at7September valuation, not repriced.
  Paths: `run-v2/results/current/{signed-equal,level-ridge1,level-ridge10}/`
  `pfc-experimental-{1h,15min}.csv`. Old candidate/pilot exports preserved.
- `summary.json`, `closure.json`: machine-readable outcome/final file hashes.

## Exact changed repository files

- `pfc_shaping/lt/evaluation_curve_assembly.py`
- `pfc_shaping/lt/benchmark_safeguards.py` (new)
- `pfc_shaping/lt/level_conditioned_shape.py` (new)
- `scripts/run_lt_hourly_recency.py`
- `scripts/run_lt_hourly_stability.py`
- `scripts/audit_lt_hourly_revisions.py`
- `scripts/run_lt_maturity.py`
- `scripts/verify_lt_maturity.py`
- `scripts/verify_lt_hourly_stability.py`
- `scripts/run_lt_level_conditioned.py` (new)
- `scripts/verify_lt_level_conditioned.py` (new)
- `tests/test_benchmark_safeguards.py` (new)
- `tests/test_level_conditioned_shape.py` (new)
- `docs/model/LT-LEVEL-CONDITIONED-EXPERIMENT-20260908.md` (new)
- `docs/model/PFC-CH-LEVEL-CONDITIONED-RESULTS-20260908.md` (new)
- `docs/model/PFC-CH-AUDIT-RESPONSE-20260908.md` (D318 follow-up link only)
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D318 appended)
- this handoff (new)

Six files were already dirty at entry: HANDOFF.md, DECISION-LOG.md, Phase14
QUALITY-FOUNDATION and LT-VOLUME-DISCOVERY handoffs, and their two docs/model
contracts. Only the two shared governance files receive D318 additions.
Four other private files were not edited. Never stage these six files wholesale.
Old research CLI runs bind old code: replay with their archived source snapshots,
not by rewriting old manifests to accept current guards.

## Commands and runtime configuration

Every shell call first checks `(Get-Location).Path` and `git rev-parse
--show-toplevel` equal `C:\Users\jbattaglia\PFC_LT`. No sandbox override.
Interpreter `build/conda-runtime-v41-model-source/python.exe -B`.
Set each of TEMP,TMP,APPDATA,LOCALAPPDATA,MPLCONFIGDIR,XDG_CACHE_HOME,
NUMBA_CACHE_DIR,JOBLIB_TEMP_FOLDER,PYTHONUSERBASE,PIP_CACHE_DIR to its own
`build/lt-level-conditioned-20260908/runtime/<NAME>` directory.
PYTHONDONTWRITEBYTECODE=1,PYTHONIOENCODING=utf-8,CUDA_VISIBLE_DEVICES=-1,
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4. No installs/browser/exe builds.

Executed with that interpreter and configuration:

1. `-m pytest tests/test_benchmark_safeguards.py tests/test_signed_hourly_assembly.py tests/test_lt_maturity.py -q`
   with basetemp `pytest-first`, JUnit `tests-first.xml` below task root.
2. `-m pytest tests/test_benchmark_safeguards.py tests/test_level_conditioned_shape.py tests/test_lt_evaluation_curve_assembly.py tests/test_hourly_stability.py tests/test_lt_maturity.py -q`
   with basetemp `pytest-second`, JUnit `tests-second.xml`.
3. `-m scripts.run_lt_level_conditioned freeze --output build/lt-level-conditioned-20260908/run-v1`
4. Same module `run --output build/lt-level-conditioned-20260908/run-v1` (constructor failure).
5. `-m pytest tests/test_benchmark_safeguards.py tests/test_level_conditioned_shape.py -q`
   with basetemp `pytest-sentinel`, JUnit `tests-sentinel.xml`.
6. Freeze then run in separate calls with `--output build/lt-level-conditioned-20260908/run-v2`.
7. `build/lt-level-conditioned-20260908/independent_guards.py`.
8. `build/lt-level-conditioned-20260908/run_checks.py` (exact40-module argv saved).
9. `-m scripts.verify_lt_level_conditioned --run build/lt-level-conditioned-20260908/run-v2 --output build/lt-level-conditioned-20260908/independent-v1`.
10. `build/lt-level-conditioned-20260908/prepare_report.py`.
11. `build/lt-level-conditioned-20260908/finalize.py` (final QA and closure).

All pytest invocations also use `-o cache_dir=build/lt-level-conditioned-20260908/pytest-cache`;
the basetemp and JUnit arguments are absolute/relative below the same task root.
Read-only Git status/diff/check and checkpoint show probes were also executed.
No new global lint success is claimed; the prior repository lint debt remains.

## Remaining work, not an instruction to tune

Signed quote-conflict policy, PRD block/PIT qualification, physical cadence
weighting, qualified national future trajectories and an independently registered
future holdout remain open. October2026–September2027 remains draft; neither
this plan hash nor past exposed data grants countable origin/admission authority.
Do not add a third ridge, per-month rules, AFRY or T057 as a response to losses.
Continue pilot only with a real new governed observation, without altering recipe.
