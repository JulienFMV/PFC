# D306 price-conditioned intraday — 7 September 2026

Complete; no active process, adoption, commit, push or external publication.
User authorized local CPU work through tests, independent saved-result review,
comparative report and candidate exports. No agents, Warehouse, GPU, AFRY, T057,
CT, protected desk-data mutation or authority change. D304/D305 retained.

## Outcome

First reuse of the existing component: ShapeIntraday now exposes
`price_conditioned_residual(hourly_prices, timestamps, calendar_df,
reference_date=..., entso_df=None)`. Native factors become
`hourly_price * (factor - 1)`, centered within each UTC hour. Exactly four
quarters, aligned finite hourly prices/context and aware timestamps required.
No division by price, estimator change, new adapter, assembler change, monthly
patch, hydro addition or default replacement. At zero price this stays flat.
AST verification proves all existing ShapeIntraday methods unchanged.

Protocol written before execution in
`docs/model/LT-PRICE-CONDITIONED-INTRADAY-EXPERIMENT.md`; run plan frozen
before opening new truth/scoring. Eight pre-origin D305 native and regularized
fits reused; each Jan-Aug2026 origin predicts all remaining months through
August:36 paired populations/105588 QH-origin, with repeated observations.
Eight lead-zero populations reproduce D305 within1e-9. Observed-parent DE
decomposition and effective DE calendar-price forecast are separate. The latter
uses only pre-origin hourly calendar means, with existing CH calendar semantics,
not future observed price, monthly truth or a DE EEX solver. This transparent
proxy does not establish DE production forecast quality.

DE lead-zero equal-month MAE/RMSE:

| Candidate | Conditional | Effective forecast |
|---|---|---|
| flat | 7.926110 /13.803879 |36.086184 /50.130863 |
| native |6.571542 /12.322029 |35.624156 /49.566120 |
| regularized |6.487984 /12.287431 |35.691182 /49.676545 |
| unconditional additive |6.225856 /11.424343 |35.586281 /49.514262 |

On all36 populations, pooled conditional native MAE7.993026 versus flat8.491463
(-5.87%), but RMSE15.394107 versus15.243350 (+0.99%) and negative-parent
MAE3.353739 versus2.991296 (+12.12%). Regularized negative-parent MAE3.428422
(+14.61%). Unconditional additive near-zero MAE4.903779 versus flat0.919231.
Native/regularized/additive fail12/13/10 conditional gates (including horizon
and regime comparisons); no global adoption. Effective native forecast MAE
49.959645 versus flat50.143042 (-0.365746%); negative-truth errors around99
EUR/MWh show that hourly forecast error dominates. Zero >5% forecast gate
regressions is not evidence of accurate negative-price forecasts.

Masks fixed before scores: negative true parent, abs(parent)<=5 (intentional
overlap), parent>5, predicted negative/near-zero, four seasons and leads0,
1-2,3-5,6-7 months. Report equal-fold and pooled errors, counts and every fold.
Autumn and horizons>7 months are unsupported. Local gate: >5% MAE or RMSE
regression versus flat with>=96 QH; regularized also compared to native.
Dependent exposed origins imply no significance or independent promotion claim.

CH composition replays six D304 signed curves plus current. Five historical
origins2021-2025 have no DE training history: explicitly unsupported flat,
not invented data. Two fits at origin2026 on8792 QH strictly before
2025-12-31T12Z; two current fits on32160 QH through2026-08-31T21:45Z, before
valuation2026-09-07T08Z. Four new statistical fits total;14 CH compositions.
Hourly model object is existing; the explicit signed lane consumes saved D305
signed shape and solver levels. Conditioning price is saved final signed PFC
hourly price. Existing assembler and final projection preserve this baseline.

Two current CSVs,114052 QH each Oct2026-Dec2029:

- `build/lt-price-conditioned-20260907/run-v1/current/native/pfc-fmv-ch-15min.csv`
- `build/lt-price-conditioned-20260907/run-v1/current/native-regularized/pfc-fmv-ch-15min.csv`

Each sibling `curve.parquet`:219268 QH full horizon through2032.
Both80 PASS/9 QUOTE_CONFLICT/noCRITICAL; no hierarchy waiver. Native minimum
-30.989868, regularized-32.667008; both480 negative QH. D305 signed minimum
-29.747102/480negative, additive-66.651528/541negative, MLP11.388921/0negative
on same main horizon. These are descriptive forecasts, not accuracy/frequency
targets. CH observations remain hourly repeated; hourly scores unchanged,
native15min transfer accuracy and structural2030 accuracy unproven.

## Verification and artifacts

Task root `build/lt-price-conditioned-20260907/`.
Canonical successful benchmark `run-v1`, ended2026-09-07T15:27:17.321205Z.
Final independent verifier `review-v3`; reviews-v1/v2 retained as failed
verification attempts, no benchmark rerun or changed predictions.

- 619 inputs/90 source-config pins/178 run outputs verified.
- 2880 metric rows independently recomputed and288 saved prediction series
  replayed from native factors/arithmetic; max error5.684341886080802e-14.
- 14 CH saved curves independently checked against factors, monthly solver,
  baseline hourly scores and regenerated EEX product gates; CSV/Parquet parity.
- Max hourly drift1.7905676941154525e-12 EUR/MWh;
  max monthly solver residual2.3931079340400174e-11.
- D304/D305 manifests/output pins unchanged; all733 D301 inventory files
  independently hash checked again by `finalize_review.py`.
- Source snapshots in `run-v1/source-snapshot/`; pre-edit intraday component
  in `baseline/shape_intraday.py` SHA256
  `94d0cec097bdad51ab6092ff85ef804103f86502acdabd8d08276e7a910b03a8`.

Main French report: `RAPPORT-COMPARATIF.md` in task root. Raw independent
report and comparison/gate/fold/annual/CH-invariance CSVs in `review-v3/`.
`current-controls-comparison.csv` compares the five current profiles.
`source-and-prior-review.json` records AST and exact known-failure comparison.

SHA256:

- run plan: `a379ed7e8786483737a1bdeeac76211f44389a28698b5ebfb2dbacec060b8c5e`
- run manifest: `87ddf8e4f38fa78558d8877f7494e555a92270b99770c5cc55b88abb5d02111a`
- verification: `4d6c48ec0fc2743e90116ce0858a89909bb8fc7a384f0c3283924135245c9ed3`
- review manifest: `243c6da10b2e05202e2bdbced69cf4489f8a0d724e6de28a6b005a98678d73e2`

Tests99 focused pass (25 new), matrix221pass/5skip/2 known Phase5 failures.
Exact XML failure messages equal D305; no golden alteration or waiver.
`git diff --check` passes with existing LF/CRLF warnings.

Verifier failures: v1 exposed floating summation ordering at exact-zero regime
boundaries. Independent numpy parent means match saved pandas means within
1e-12; masks then use the verified frozen common parent to preserve exact
populations. v2 failed EEX text-column comparison because empty CSV strings
reload as NaN. Normalize missing object text only; numeric fields remain
checked. v3 passed. No model/data result was changed to make review pass.

## Changed files and exact commands

Substantive changed/added files this turn:

- `pfc_shaping/lt/model/shape_intraday.py` (one additional method only).
- `scripts/run_lt_price_conditioned_intraday.py`.
- `scripts/verify_lt_price_conditioned_intraday.py`.
- `tests/test_price_conditioned_intraday.py`.
- `docs/model/LT-PRICE-CONDITIONED-INTRADAY-EXPERIMENT.md`.
- `.planning/HANDOFF.md`, Phase14 `DECISION-LOG.md`, this handoff.
- Task-local `finalize_review.py`, reports, baseline/source snapshots and
  runtime/test/result artifacts. Other dirty user work preserved.

Before every shell action: assert current directory
`C:\Users\jbattaglia\PFC_LT` and Git top-level
`C:/Users/jbattaglia/PFC_LT`; canonical workdir only. Runtime prefix
`build/conda-runtime-v41-model-source/python.exe -B`. Each of TEMP,TMP,APPDATA,
LOCALAPPDATA,MPLCONFIGDIR,XDG_CACHE_HOME,NUMBA_CACHE_DIR,JOBLIB_TEMP_FOLDER,
PYTHONUSERBASE,PIP_CACHE_DIR points to task-local `runtime/<NAME>`;
CUDA_VISIBLE_DEVICES=-1;OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4;
PYTHONDONTWRITEBYTECODE=1. No external mutable path or approval override.

Commands (after those guards/environment settings):

```powershell
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_price_conditioned_intraday.py tests/test_signed_composition.py tests/test_signed_hourly_assembly.py tests/test_intraday_persistence.py -q --tb=short -o cache_dir=build/lt-price-conditioned-20260907/pytest-cache --basetemp=build/lt-price-conditioned-20260907/pytest-v1 --junitxml=build/lt-price-conditioned-20260907/tests-v1.xml
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_price_conditioned_intraday --output build/lt-price-conditioned-20260907/run-v1
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.verify_lt_price_conditioned_intraday --run build/lt-price-conditioned-20260907/run-v1 --output build/lt-price-conditioned-20260907/review-v3
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_price_conditioned_intraday.py tests/test_signed_hourly_assembly.py tests/test_signed_composition.py tests/test_signed_benchmark.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_lt_evaluation_curve_assembly.py tests/test_lt_local_benchmark.py tests/test_assembler_profile_type.py tests/test_phase05_negative_prices.py tests/test_lt_structural_readiness.py tests/test_intraday_persistence.py -q --tb=short -o cache_dir=build/lt-price-conditioned-20260907/pytest-cache --basetemp=build/lt-price-conditioned-20260907/pytest-matrix-v1 --junitxml=build/lt-price-conditioned-20260907/tests-matrix-v1.xml
& build/conda-runtime-v41-model-source/python.exe -B build/lt-price-conditioned-20260907/finalize_review.py
& build/conda-runtime-v41-model-source/python.exe -B build/lt-price-conditioned-20260907/close_session.py
git diff --check
```

Verifier v1/v2 used same command with review-v1/review-v2 output names. Do not
rerun into existing completed destinations. closure.json binds final docs,
source, test evidence and artifact manifests; caches/scratch excluded.

## Next

No robust global winner. Preserve signed D304 hourly reference and D305 hydro/
unconditional-additive conclusions. Next useful local experiment is explicit
price-regime/support conditioning with a flat fallback, preregistered before
new results; evaluate parent forecast quality independently. This is a proposal,
not a selected new model. Do not tune month switches on exposed folds. No
scientific/promotion/production/trading authority. CH native15min truth,
independent future holdout and coherent2030 physics remain separate gaps.
