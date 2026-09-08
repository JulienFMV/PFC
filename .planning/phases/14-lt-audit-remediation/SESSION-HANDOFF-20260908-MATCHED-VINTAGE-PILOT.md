# D312 — common-cutoff first day complete; 19 real future days pending

Read AGENTS.md, .planning/HANDOFF.md, then this handoff. D304–D311 prior evidence
preserved. Report build/lt-matched-vintages-20260908/RAPPORT-PILOTE.md.

## Result and honest completion boundary

First actual pilot snapshot at2026-09-08T09:45:22.774073UTC, candidate committed
09:45:38.513948UTC,1record/20targetdays. Remaining19calendar days cannot be
collected now and are explicitly pending. No scheduled task/background daemon
installed. Daily registry is local hash continuity only, not external custody.
Independent countable origins0. All6authoritiesfalse. No production, trading,
scientific admission or superiority claim.

Common cutoff means every selected local input was observed before valuation;
it does not mean identical vendor issue timestamps or authenticated historicPIT.
D311 external prices were already exposed. The D304 hourly recipe is unchanged
and neither OMPEX nor LSEG enters fitting, priors, solver or model selection.
Sources from D311 are reused by exact hash, retaining original issues/receipts.

## Sources / execution / unchanged invariants

Warehouse wasRUNNING at initial and final control-planeGET; auto-stop45min.
One additional boundedEEXSELECT on already-running Warehouse; no start, stop
retry, configuration or table write. The earlier manual stop403 remains; no
escalation asked. Auto-stop is the available termination mechanism; shutdown
is not claimed confirmed. Last receipt warehouse-final.json.
User Warehouse/GPU permission persists; CPU4threads used, no GPU compute.

EEX delta164rows on quotations20260904/20260907; source predicateCH/POWER and
dateID20260904..20260908. Existing materializer normalizes/quarantines rows.
Historical normalized data beforeSep4 retained; replacement delta contains
fresh Sep4/Sep7. Latest selected quote dateSep7, compared withSep4 inD311.
Same monthly solver parameters and same PFCAssembler/projection;75months
Oct2026–Dec2032,219268QH. No post-solver monthly patch. D304 raw seasonal signed
shape is identical value-for-value to previous reference; training uses same
closed CH monthly targets, no September open-month additions.

Solver residual1.0203393685515039e-11EUR/MWh;80PASS/9QUOTE_CONFLICT/0CRITICAL.
Quotes conflict as before; no gate suppressed. Projection constraint residual
1.1340262062731199e-11,stationarity9.947598300641403e-14.
MainCSV exports28513hours/114052QH throughDec2029, both independently verified.
Curve SHA25613bca031c6364d71ba32277d2db54459d6a0cc61551912b0af862f34abddc263.
Recipe SHA256afeefbb98ca6f0349efe3ed7f275aae0e374d88140f687a07af4d21519c82c3f.
Solver SHA25607c955b5ab9f975c36d219005ef5d801abf641e32987bcdbf0770629d9af1d1d.

## Diagnostics

27common complete months/19753hours: OMPEX levelMAD2.893846,shapeMAD11.479800;
LSEG16.334161/20.506840EUR/MWh.39months/28513hours OMPEX3.170473/10.636364.
These are forecast-to-forecast distances, not accuracy. LSEG2029UNSUPPORTED.
Report separates years,FMVseasons,horizons,negative/nearzero forecasts,pre-origin
shape/ramp thresholds and seams; no lowest-error timestamp alignment.
D311→D312 transition54817hours: mean absolute total1.000500,level1.000500,
preprojection shape9.66e-15,projection0.237422. Signed components sum; absolute
means do not. This transition is not two prospective pilot snapshots. Daily
revision stability remainsUNSUPPORTED_ONE_PILOT_SNAPSHOT.

## New files / artifacts / hashes

New source/tests/protocol:
- pfc_shaping/validation/lt_benchmark_snapshots.py
- scripts/register_lt_benchmark_snapshot.py
- scripts/run_lt_benchmark_day.py
- tests/test_lt_benchmark_snapshots.py
- tests/test_lt_benchmark_day.py
- docs/model/LT-MATCHED-VINTAGE-PILOT.md
New handoff:.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260908-MATCHED-VINTAGE-PILOT.md.
Updated shared governance:.planning/HANDOFF.md and Phase14/DECISION-LOG.md.
No product assembler/solver/shape changes. No protected deskdata,CT,PowerBI,
AFRY,T057 or prior source/artifact modifications. Existing dirty work preserved.

Taskroot build/lt-matched-vintages-20260908/:
- capture_eex.py,eex-capture-v1/query.sql,plan,response,source.parquet,receipt,manifest.
- build_first_snapshot.py,recipe.json,snapshot-20260908/plan.json,eex-delta.parquet,
  eex-delta-audit.json,eex-history.parquet,eex-surface.parquet,solver-inputs.json,
  solver.json,D304/(curve.parquet,raw.parquet,receipt.json,product-gates.csv),
  candidate-commitment.json,OMPEX.parquet,LSEG.parquet,2CSVexports,
  registry-entry.json,complete.json,manifest.json,runtime.log.
- registry/0001-2026-09-08.json (future records must not alter this record).
- analyze.py,analysis-v1/(distances-by-regime.csv,monthly-levels.csv,summary.csv,
  pairedParquets,external-seams.parquet,external-seam-metrics.csv,
  D311-to-D312-transition.parquet,transition.json,pilot-status.json,coverage.json,manifest.json).
- verify.py,review-v1/(verification.json,product-gates.csv,manifest.json).
- finish.py,pilot-calendar.json,next-day-request-template.json,
  future-truth-contract.json,RAPPORT-PILOTE.md,report-evidence.json.
- close.py,validate_closure.py,closure.json,closure-validation.json,
  tests-initial.xml,tests-matrix.xml,tests-day.xml,warehouse-before.json,
  warehouse-final.json; task-local runtime/pytest dirs.

Snapshot manifest SHA256 0f246e1524d35e2ce8625427851cfb943c8ccafb103a8db7a4cc4e93507ccb8c.
Review manifest SHA256 99b700c21e04e3c5d779c0434d81a185ed3f7ebc404e853ba0e66a953d511ec5.
Report SHA256 fb326174d9dcad8e4ed1ebcbf82234ac743125ec6eb573f6080f58eda9511b06.
Final closure binds source/docs/governance and task files, not runtime caches.

## Verification and commands

21newtests:16registry/calendar/scoring +5daily-preflight/numeric tests.
Main matrix154pass/4skip; separate daily suite5pass, total159pass/4skip.
No test failures. Initial documentation closure failed on unescaped literal
braces in an f-string (NameError); dependent validation found no closure file.
Both failures are preserved in closure-first-failure.json. Only documentation
rendering was corrected; no model, tests, scores or frozen source changed.
Numerical tests isolate constant level error from zero shape error,
reject future truth/late commitment/incomplete months; registry tests reject
post-cutoff inputs, positive authority, same day duplicates, broken chains,
outside-build synthetic paths and>20records. DST March743/October745hours.
Daily preflight requires fresh daily observations for EEX/OMPEX/LSEG, keeps
closed CH history reusable, preserves model recipe, refuses external model input.
Independent replay reconstructs D304 calendar means, monthly neutrality,
solver/EEX gates, external distances and transition components.746D308/294D309/
357D310/68D311 frozen files unchanged, excluding shared handoff/decisionlog.

Every shell command checks cwd and Git top-level exactly C:\Users\jbattaglia\PFC_LT.
Runtime TEMP,TMP,APPDATA,LOCALAPPDATA,MPLCONFIGDIR,XDG_CACHE_HOME,NUMBA_CACHE_DIR,
JOBLIB_TEMP_FOLDER,PYTHONUSERBASE,PIP_CACHE_DIR under task/runtime/<name>;
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4;
PYTHONDONTWRITEBYTECODE=1;CUDA_VISIBLE_DEVICES=-1. No external mutable paths.
AllPython below uses & build/conda-runtime-v41-model-source/python.exe -B:
- capture_eex.py undertask :oneboundedSELECT,164rows,zeroWarehouse starts.
- -m pytest tests/test_lt_benchmark_snapshots.py -q --tb=short -o cache_dir=build/lt-matched-vintages-20260908/pytest-cache --basetemp=build/lt-matched-vintages-20260908/pytest-initial --junitxml=build/lt-matched-vintages-20260908/tests-initial.xml :16pass.
- build/lt-matched-vintages-20260908/build_first_snapshot.py :one solver/assembler, COMPLETE.
- New-Item -ItemType Directory -Path build/lt-matched-vintages-20260908/registry.
- -m scripts.register_lt_benchmark_snapshot --entry build/lt-matched-vintages-20260908/snapshot-20260908/registry-entry.json --registry build/lt-matched-vintages-20260908/registry :oneentry.
- -m pytest tests/test_lt_benchmark_snapshots.py tests/test_ch_lt_prospective_hourly_scoring.py tests/test_ch_lt_prospective_capture_ledger.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_signed_hourly_assembly.py tests/test_databricks_eex_daily_snapshot.py tests/test_ompex_benchmark.py -q --tb=short -o cache_dir=build/lt-matched-vintages-20260908/pytest-cache --basetemp=build/lt-matched-vintages-20260908/pytest-matrix --junitxml=build/lt-matched-vintages-20260908/tests-matrix.xml :154pass/4skip.
- -m pytest tests/test_lt_benchmark_day.py -q --tb=short -o cache_dir=build/lt-matched-vintages-20260908/pytest-cache --basetemp=build/lt-matched-vintages-20260908/pytest-day --junitxml=build/lt-matched-vintages-20260908/tests-day.xml :5pass.
- build/lt-matched-vintages-20260908/analyze.py :commoncutoffdiagnostics.
- build/lt-matched-vintages-20260908/verify.py :VERIFIED_LOCAL_COMMON_CUTOFF.
- build/lt-matched-vintages-20260908/finish.py :report/calendar/truthdraft/template.
- build/lt-matched-vintages-20260908/close.py and validate_closure.py :closure; git diff --check.
Read-only initial/final warehouseGET saved only id/state/auto-stop; credentials
loaded in memory through existing helper, not printed. Other reads limited to
contracts, relevant source/interfaces and saved artifacts. No source H: reread.

## Next actual day and future truth

Next slot2026-09-09, then daily throughSep27 provisionally. Missed/unavailable
days remain pending, never backfilled as real-time observations. No automatic
schedule or source refresh is installed. On the next real day, capture/reobserve
source bytes with actual timestamps, hash-bind them belowbuild and fill a new
day-YYYYMMDD-request.json using the intentionally incomplete template. EEX input
must be normalized fullhistory, CH input closed signed targets; external
Parquets remain benchmark-only. Then run fromcanonicalroot/task-local runtime:

python -B -m scripts.run_lt_benchmark_day --request build/lt-matched-vintages-20260908/day-YYYYMMDD-request.json --registry build/lt-matched-vintages-20260908/registry --output build/lt-matched-vintages-20260908/snapshot-YYYYMMDD

Use the repo-local Python prefix above, not systemPython. Command owns no
network/schedule and cannot manufacture earlier valuations. It is not the
governed production entrypoint. With a second genuine day, compare common
delivery timestamps and sourceissue/capture metadata; no independent-N inflation.

Future truth contract frozen beforeOctdelivery. Earliest first monthly close
2026-10-31T23:00Z. Provider/series semantic receipt, independent custodian and
finalization/availability authority still absent; no futuretruth loaded/scored.
Local score_closed_months reuses existing monthly scoring math, refuses future
or incomplete truth and keeps all authoritiesfalse. Full price metrics follow
the explicit level+shape identity; scientific margins/multiplicity/power remain
unregistered, so no superiorityclaim. Do not reuse T057 or vendor forecasts as
truth. Warehouse manualstop403 remains external operational limitation; do not
claimSTOPPED until a later observation proves it.
