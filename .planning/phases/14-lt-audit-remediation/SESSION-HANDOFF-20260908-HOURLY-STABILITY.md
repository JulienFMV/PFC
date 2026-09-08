# D308 hourly CH stability — 8 September 2026

Complete. No active process, adoption, commit, push or external publication.
User authorized local CPU, bounded blends and all verification/export/handoff
work. No agents, Warehouse, GPU, AFRY, T057, CT or protected-data changes.
Existing dirty work preserved; no product code modified this session.

## Frozen design and outcome

Read docs/model/LT-HOURLY-STABILITY-LOCAL-EXPERIMENT.md (written before results).
D304 primary signed reference; D305/D306 retained separate composition
comparisons; D307 one/two-year recency plus MLP are exact hourly controls.
Four fixed blends25%/50% against each half-life; no fits or model/month switch.
Reuse pinned raw predictions, center per Swiss month, call unchanged assembler
and EEX projection. No blend or repair of solver levels.

run-v1 frozen 2026-09-08T07:19:14.442868+00:00; completed 2026-09-08T07:22:11.680899+00:00.
28 new assemblies,28 cached control curves,0 statistical model fits.
Thresholds for all seven origins are inside plan.json before assembly/scoring:
raw-price p95, within-month contiguous-ramp absolute p95, signed monthly-target
p05/p95 and absolute p95. Pre-origin complete training months only. Same six
historical origins,2021–2022 diagnostic and2023–2026 exposed development.

| Candidate | Shape MAE | Shape RMSE | MAE gain % | Regime / origin regressions |
|---|---:|---:|---:|---:|
| blend-hl365-a25 | 20.155702 | 29.713319 | 0.931579 | 0 / 0 |
| blend-hl365-a50 | 20.059879 | 29.386879 | 1.402563 | 2 / 2 |
| blend-hl730-a25 | 20.213340 | 29.873700 | 0.648277 | 0 / 0 |
| blend-hl730-a50 | 20.110168 | 29.610510 | 1.155386 | 0 / 0 |

Reference D304 shape MAE20.345234/RMSE30.180010. All four blends win3/4 origins,
lose origin2023 and miss2% MAE gain. Three have no>5% supported regressions;
half-life1year at50% fails delivery2024 (shape+5.675%,ramp+8.141%) and origin2023
(shape+7.242%,ramp+8.457%). No favorable full local screen or adoption.

Shape-tail support across four origins: low3822/high1751/absolute1888
hours-origin. HIGH_PRICE0 remains UNSUPPORTED, never passed spike evidence.
Every year/season/horizon/negative/near-zero/ramp/tail comparison retained.
Monthly level MAE47.260956/RMSE55.960385 identical across all eight candidates;
MSE(full)=MSE(shape)+MSE(level) per complete month independently verified.
Monthly forward-spot error is distinct from solver constraint accuracy.

## Artifacts and verification

Task root: build/lt-hourly-stability-20260908/.
RAPPORT-COMPARATIF.md combines editorial conclusion/tests with immutable
review-v1/RAPPORT-COMPARATIF.md. review-v1/ includes comparison.csv (equal-origin
and pooled), regime-gates.csv, origin-gates.csv, shape-mae-by-origin.csv,
verified-metrics.csv, verified-monthly-decomposition.csv, current-profiles-by-year.csv,
screening.json, verification.json and manifest.json.

Four candidate folders under run-v1/current/:

- blend-hl365-a25
- blend-hl365-a50
- blend-hl730-a25
- blend-hl730-a50

Each pfc-fmv-ch-1h.csv28513rows and pfc-fmv-ch-15min.csv114052repeated-QH rows,
Oct2026–Dec2029; curve.parquet219268QH through2032. D304/D307/MLP controls also
copied and checked. Retained valuation2026-09-07T08Z,4September quotes; no fresh
September8 revaluation. Signed current gates80PASS/9QUOTE_CONFLICT/0CRITICAL.

Independent verifier:1173 input hashes/100 source hashes/622 output hashes,
8768 metric rows,1920 monthly decompositions;21 independent NumPy/dictionary
component calculations,49 raw signed prediction/centered-input comparisons.
Max raw error4.2633e-14; diagnostic2.2737e-13; monthly solver2.4102e-11.
733 D301 inventory files unchanged; D304–D307 input/output/source pins retained.
Supplemental additional-review-v2 verifies28 affine pre/final assembly identities
(max 3.41060513165e-12) and
1160 direct BASE/PEAK/OFFPEAK means.
Residual bucket gates rely on the separately rerun main product gate verifier.
23 D307 closure-bound files rehashed unchanged;
only the two shared evolving handoff/decision docs are excluded from that replay.

Tests:78 focused pass (21 new), matrix255pass/5skip/2 known Phase5 failures.
additional-review-v2/known-fixture-comparison.json proves exact D307 assertion
message equality. No golden edit, waiver or claim of a fully green suite.
git diff --check passes with existing LF/CRLF warnings only.

Failures: focused-v1 initially emitted a pandas future dtype warning in a new
synthetic test; fixed fixture dtype before plan freeze, matrix has final bytes.
Benchmark run-v1 and principal review-v1 passed on first execution. Supplemental
check_final_evidence-v1.py omitted the OFFPEAK mask and failed its direct gate
assertion after saving known-fixture-comparison.json. Fixed only the checker;
check_final_evidence.py writes fresh additional-review-v2. No benchmark rerun,
threshold change or result-dependent candidate change. Failed source retained.

## Exact changed files

- scripts/run_lt_hourly_stability.py (new)
- scripts/verify_lt_hourly_stability.py (new; reuses independent D307 helpers)
- tests/test_hourly_stability.py (new)
- docs/model/LT-HOURLY-STABILITY-LOCAL-EXPERIMENT.md (new preregistration)
- .planning/HANDOFF.md (D308 current-state insertion)
- .planning/phases/14-lt-audit-remediation/DECISION-LOG.md (D308 appended)
- .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260908-HOURLY-STABILITY.md (new)
- Task-local prepare_verifier.py (pre-score scaffolding; do not rerun),
  check_final_evidence-v1.py (failed source), check_final_evidence.py (corrected),
  finish_session.py, source snapshots, runtime/test artifacts, reports,
  additional-review-v2, closure.json and closure validation receipts.

## Runtime and exact execution commands

Before every shell action, verify Get-Location.Path equals
C:\Users\jbattaglia\PFC_LT and git rev-parse --show-toplevel normalized to
backslashes equals the same canonical root. Every shell uses that workdir.
No override/elevation. Existing local runtime:
build/conda-runtime-v41-model-source/python.exe -B.
TEMP,TMP,APPDATA,LOCALAPPDATA,MPLCONFIGDIR,XDG_CACHE_HOME,NUMBA_CACHE_DIR,
JOBLIB_TEMP_FOLDER,PYTHONUSERBASE,PIP_CACHE_DIR each points to the task-root
runtime/<NAME> below build/; directories created before Python execution.
CUDA_VISIBLE_DEVICES=-1; OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4;
PYTHONDONTWRITEBYTECODE=1. No mutable path outside the canonical workspace.

Commands after those guards/settings (existing destinations must not be reused):

```powershell
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-stability-20260908/prepare_verifier.py
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_hourly_stability.py tests/test_hourly_recency.py tests/test_signed_benchmark.py tests/test_signed_hourly_assembly.py -q --tb=short -o cache_dir=build/lt-hourly-stability-20260908/pytest-cache --basetemp=build/lt-hourly-stability-20260908/pytest-focused-v1 --junitxml=build/lt-hourly-stability-20260908/tests-focused-v1.xml
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_hourly_stability --output build/lt-hourly-stability-20260908/run-v1
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_hourly_stability.py tests/test_hourly_recency.py tests/test_price_conditioned_intraday.py tests/test_signed_hourly_assembly.py tests/test_signed_composition.py tests/test_signed_benchmark.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_lt_evaluation_curve_assembly.py tests/test_lt_local_benchmark.py tests/test_assembler_profile_type.py tests/test_phase05_negative_prices.py tests/test_lt_structural_readiness.py tests/test_intraday_persistence.py -q --tb=short -o cache_dir=build/lt-hourly-stability-20260908/pytest-cache --basetemp=build/lt-hourly-stability-20260908/pytest-matrix-v1 --junitxml=build/lt-hourly-stability-20260908/tests-matrix-v1.xml
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.verify_lt_hourly_stability --run build/lt-hourly-stability-20260908/run-v1 --output build/lt-hourly-stability-20260908/review-v1
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-stability-20260908/check_final_evidence.py
Copy-Item -LiteralPath build/lt-hourly-stability-20260908/check_final_evidence.py -Destination build/lt-hourly-stability-20260908/check_final_evidence-v1.py
# Apply the saved OFFPEAK/output-directory checker fix, then:
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-stability-20260908/check_final_evidence.py
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-stability-20260908/finish_session.py
git diff --check
```

SHA256:

- plan: 60ae52b021d605d0ba535ecab5792af0c6c52031028b6488bbd6ad7930cd9fc7
- run manifest: 1f131effbce0af30e418aa9f47d72bb097c04746e23c7cd68feec9ffeecb0628
- review verification: 7da715192b71fb41357ecd38cdd830559f1e141d951217c48b159ca19051c4ee
- review manifest: 13a349a241ad83295b4fe7a4d37d09f7cca242ad865138df001c1a1f2e89a727
- additional verification: f43007b4f4c11510e7bbb90de729e0d04aae4a4fb36cdadf5d4a0368908e7a9b
- additional manifest: a1cc5a06543725ef89ed88ba0b4631218a0956fd1da242251b523082f6043cce

closure.json binds final code/docs/tests/reports/manifests, plus existing task
helper sources and failed-attempt evidence. Runtime caches are excluded.

## Next boundary

D304 remains primary local signed reference, D305–D307 comparisons intact.
The limited blend grid is exhausted; no extra weight, month switch or weakened
gate from these results. Some shrinkage variants meet comparative stability
vetoes but do not meet the frozen gain threshold. A new experiment would need
a distinct pre-origin hypothesis/protocol and fresh independent holdout before
scientific admission. Do not relabel the six overlapping exposed origins.
Keep monthly risk-premium/forward-spot gaps separate from hourly-shape quality;
no solver-month patch. Native CH15min truth, future2030 physical evolution,
PnL and uncertainty qualification remain unresolved. All six production,
promotion, scientific, trading and registry authority values stay false.
