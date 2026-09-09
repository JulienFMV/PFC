# D309 hourly revisions and calendar seams — 8 September 2026

Complete. No active process, adoption, new predictive model, commit/push or
external publication. User accepted the post-D308 expert next step: audit
revisions/seams, attribution and a conditional single-challenger proposal.
The proposal trigger fails, so no smoothing model is launched. This completes
that next lot, not the separate physical/uncertainty research roadmap.

## Frozen scope and findings

Protocol docs/model/LT-HOURLY-REVISION-SEAM-AUDIT.md written before results.
D304 primary; D305–D308 and existing dirty work preserved. Same assembler and
EEX projection, no product-code edits or solver-level changes. CPU only, all
six authorities false; no Warehouse/GPU/AFRY/T057/CT/protected-data mutation.

Plan frozen 2026-09-08T08:19:08.664207+00:00; completed 2026-09-08T08:20:12.566516+00:00.
All7 threshold sets inside plan before seam/revision results: native midnight
and month-boundary absolute ramp p95, signed-target boundary p95 and support.
Pseudo-update budget compares full pre-origin D304 history with the same history
excluding its last12closed months on a deterministic next12month calendar;
requires24closed months.12 calendar calculations plus4 old-history replays,
zero statistical estimator fits. Budgets are diagnostic scales, not trading gates.

Calendar contract: winter Nov–Mar, spring Apr–May, summer Jun–Sep, autumn Oct.
Seasons change Apr/Jun/Oct/Nov, not meteorological quarters. Existing source
unchanged. A new test first used the wrong expectation; corrected before freeze.

56 curves,59120 midnight/month/season events,11136 seam metric rows. Full/shape
ramp MAE EUR/MWh on exposed2023–2026 origins:

| Group | Events | Full MAE | Shape MAE | Level-step MAE |
|---|---:|---:|---:|---:|
| Seasonal boundaries |30|18.264819|13.446529|19.413253|
| Other month boundaries |62|18.982272|16.858420|15.927846|
| Within-month midnights |2825|6.914071|6.914071|0|

Ratios seasonal/other0.962204(full),0.797615(shape), below the fixed>1.10 on
BOTH errors. Support holds across4origins; raw absolute contribution share
89.916% exceeds50% but is insufficient alone. No smooth-calendar hypothesis
trigger; no result-driven relaxation. Month boundaries are harder than ordinary
midnights, but this does not prove an avoidable calendar defect.

Exact signed step identities separate monthly level, pre-projection shape and
EEX projection. Signed curves additionally expose raw calendar and centering.
Truth means are evaluation labels only, both adjacent hours must lie in complete
eligible months. No MSE orthogonality claim on seam subsets. All year, horizon,
season, negative/near-zero, high-price and shape-tail tables remain available;
unsupported segments never become passed gates.

## Revision attribution and availability

D301 explicitly LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT. One current valuation
in admitted D300–D308 lineage; variants are not vintages. Local directory-level
inventory revealed no extra admitted D304 release series. Scope does not claim
no archive exists elsewhere. Real daily-vintage stability remains UNSUPPORTED.

2021->2022 and2022->2023:0hours common, UNSUPPORTED. Four valid pairs:

| Pair | Hours | Mean abs final revision | Mean abs level change | Mean abs history effect after projection |
|---|---:|---:|---:|---:|
|2023->2024|17544|106.133800|106.133800|1.917505|
|2024->2025|17520|8.430539|8.040707|1.209925|
|2025->2026|17520|8.974489|8.106808|1.344830|
|2026->current|19753|34.831455|34.790291|1.338851|

Values EUR/MWh. Absolute contributions need not sum. Intermediate replays use
older D304 targets with newer levels/quotes/grid/reference date, preserving
projection interactions. Grid-induced old-shape differences on common full
months are below1e-9. Attribution72337hour-pairs;32 comparison revision series,
416 summary rows,1024 additional horizon/season summary rows. This is controlled
retrospective attribution, not causal attribution to physical fundamentals.

## Outputs and independent verification

Task root build/lt-hourly-revision-audit-20260908/.
French RAPPORT-AUDIT.md. run-v1/ retains plan/source-snapshot/inventory,
events.parquet+CSV, seam-metrics.csv, revision-summary.csv, revisions/*.parquet,
pair-status.json, recommendation.json, counterfactuals/*, prospective draft,
complete.json and manifest.json. Original D304–D308 artifacts unchanged.

Four counterfactual folders2023-to-2024,2024-to-2025,2025-to-2026,2026-to-current
contain curve.parquet,raw.parquet,product-gates.csv,receipt.json,attribution.parquet
and explicit COUNTERFACTUAL_NOT_ARCHIVED_VINTAGE metadata. Current folder exports
counterfactual-ch-1h.csv28513rows and counterfactual-ch-15min.csv114052repeated
QH rows Oct2026–Dec2029;219268QH full Parquet through2032. Retained7September2026
08Z valuation,4September quotes. Current80PASS/9QUOTE_CONFLICT/0CRITICAL;
other missing/conflicting product statuses explicit, no waiver.

review-v1 independent array reconstruction:1796 input hashes/104 source hashes/
173 output hashes,59120 event rows/11136 metrics,32 revision series/416metrics,
4 raw history replays and72337 attribution hours. Max solver residual2.4102e-11,
boundary error identity1.1369e-13.733 D301 files unchanged. Separate verifier
recomputes pre-origin budgets, timestamp populations, errors, regimes and EEX.

additional-review-v1/ verifies246 direct BASE/PEAK/OFFPEAK means;5 residual
gates are independently rerun through the main gate helper.746 D308 closure-bound
files rehashed unchanged (two evolving shared handoff/decision docs excluded).
inventory56rows, tests.json exact known failures, attribution-summary.csv,
revision-regimes.csv,D304-seam-summary.csv,current-D304-boundaries.csv,
verification.json and manifest.json. Current74future month boundaries,18scale
exceedances, unscored and not proof of forecast defects. Full breakdown retained.

Prospective draft binds current D304 forecast Oct2026–Sep2027; future labels
not opened. DRAFT_NOT_INDEPENDENTLY_REGISTERED: independent custodian and governed
truth finalization/availability missing. No registration/countable origin or
three-year horizon qualification. T057 stays sealed.

## Tests, failures and source changes

90 focused tests pass after correcting the pre-freeze new season assertion;
first attempt89pass/1fail retained in tests-focused-v1.xml. Final focused-v2
and matrix use the corrected17new tests. Matrix272pass/5skip/2 pre-existing
Phase5 failures with exact D308 messages. No golden/fixture changes to existing
tests. Audit run-v1, review-v1 and additional review all succeed first try.
git diff --check passes with pre-existing LF/CRLF warnings only.

Exact repository changes this session:

- scripts/audit_lt_hourly_revisions.py (new pure audit helpers and bounded run)
- scripts/verify_lt_hourly_revisions.py (new independent reconstruction)
- tests/test_hourly_revisions.py (17new tests)
- docs/model/LT-HOURLY-REVISION-SEAM-AUDIT.md (new preregistration)
- .planning/HANDOFF.md (D309 insertion)
- .planning/phases/14-lt-audit-remediation/DECISION-LOG.md (D309 append)
- .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260908-HOURLY-REVISION-AUDIT.md (new)
- Task-local complete_review.py,close_session.py,validate_closure.py and artifacts.

## Runtime and exact commands

Before every shell action verify Get-Location.Path and Git top-level normalized
to backslashes both equal C:\Users\jbattaglia\PFC_LT. Every shell uses that cwd.
Runtime build/conda-runtime-v41-model-source/python.exe -B. TEMP,TMP,APPDATA,
LOCALAPPDATA,MPLCONFIGDIR,XDG_CACHE_HOME,NUMBA_CACHE_DIR,JOBLIB_TEMP_FOLDER,
PYTHONUSERBASE,PIP_CACHE_DIR each points to task-root runtime/<NAME> under build.
CUDA_VISIBLE_DEVICES=-1; OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4;
PYTHONDONTWRITEBYTECODE=1. No external mutable paths or approval override.

After those guards/settings:

```powershell
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_hourly_revisions.py tests/test_hourly_stability.py tests/test_hourly_recency.py tests/test_signed_hourly_assembly.py -q --tb=short -o cache_dir=build/lt-hourly-revision-audit-20260908/pytest-cache --basetemp=build/lt-hourly-revision-audit-20260908/pytest-focused-v1 --junitxml=build/lt-hourly-revision-audit-20260908/tests-focused-v1.xml
# Correct the new season-expectation test against existing FMV calendar, before freeze.
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_hourly_revisions.py tests/test_hourly_stability.py tests/test_hourly_recency.py tests/test_signed_hourly_assembly.py -q --tb=short -o cache_dir=build/lt-hourly-revision-audit-20260908/pytest-cache --basetemp=build/lt-hourly-revision-audit-20260908/pytest-focused-v2 --junitxml=build/lt-hourly-revision-audit-20260908/tests-focused-v2.xml
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.audit_lt_hourly_revisions --output build/lt-hourly-revision-audit-20260908/run-v1
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_hourly_revisions.py tests/test_hourly_stability.py tests/test_hourly_recency.py tests/test_price_conditioned_intraday.py tests/test_signed_hourly_assembly.py tests/test_signed_composition.py tests/test_signed_benchmark.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_lt_evaluation_curve_assembly.py tests/test_lt_local_benchmark.py tests/test_assembler_profile_type.py tests/test_phase05_negative_prices.py tests/test_lt_structural_readiness.py tests/test_intraday_persistence.py -q --tb=short -o cache_dir=build/lt-hourly-revision-audit-20260908/pytest-cache --basetemp=build/lt-hourly-revision-audit-20260908/pytest-matrix-v1 --junitxml=build/lt-hourly-revision-audit-20260908/tests-matrix-v1.xml
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.verify_lt_hourly_revisions --run build/lt-hourly-revision-audit-20260908/run-v1 --output build/lt-hourly-revision-audit-20260908/review-v1
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-revision-audit-20260908/complete_review.py
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-revision-audit-20260908/close_session.py
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-revision-audit-20260908/validate_closure.py
git diff --check
```

SHA256:

- plan: 180946ad9482960c10249781e263d2e4e988a3db96c4744ac3b3fbc6cb1fa9a4
- run manifest: f3038565ee5022ffecddfaf12cf1f5a4c04836e7a4b91f21e09d236d129986e8
- verification: 0b0301aaf94be4cf9712410608028da23f7db8a7afd9159b963cd7d29449b28b
- review manifest: 87e4383bd5058e10d694015926b82ffa00f2f3d63be3ad4d41d058a85129fbb8
- additional verification: b4292f27fc37e92210002d167b3e79f8dcffb90e1b76e70da1103a7eeb5356cf
- additional manifest: c4b727630a681308c5d4f9e10f73c49b053388fc339511a95d1e676f3d76b8f4
- report: 95ade5b7815ce87b75a6931c4d64ff358433bc93acc73e89016de38f334aae94


Never reuse output destinations. closure.json binds final source/docs/tests,
reports, failed first test evidence and all substantive run/review artifacts;
runtime caches excluded. closure-validation.json checks the final binding.

## Next boundary

The accepted audit lot is complete. D304 remains primary; no smooth-calendar
challenger is justified by the frozen seasonal-seam trigger. Do not replace this
negative result with ad hoc weights or month patches. Successive governed local
forecast snapshots and independent prospective truth are needed to qualify real
update stability. A next shape-improvement hypothesis should address causal
future drivers/maturity with a separately fixed data/feature protocol, before
more complex models. The separate physical/uncertainty roadmap is not executed
or qualified here. Read restricted-data/source contracts before scenario work;
AFRY remains excluded. No implied permission for Warehouse/GPU/T057 or promotion.
