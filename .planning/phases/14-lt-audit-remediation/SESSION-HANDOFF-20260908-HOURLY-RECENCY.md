# D307 hourly CH recency — 8 September 2026

Complete. No active process, adoption, commit, push or external publication.
User requested official web verification of Swiss hourly pricing through2027
and continuation of the proposed hourly benchmark. Local CPU remained authorized.
No agents, Warehouse, GPU, AFRY, T057, CT/protected-data changes or authority gain.

## Official calendar finding

EPEX index specification, July2026, p40 defines CH day-ahead60min instruments.
JAO30June2026 announcement delays Swiss-border15min MTU/block-bid go-live
from the provisional September2026 schedule to2027, with no month specified.
**End2027 is not confirmed.** Cross-border capacity allocation and domestic
EPEX energy-auction resolution are related but distinct; no firm domestic
launch date was established. Do not hard-code a date or confuse continuous
intraday15min products with retained CH hourly day-ahead truth.

Sources and interpretation are in
`docs/data/CH-DAY-AHEAD-RESOLUTION-20260908.md`. Primary direct links:

- https://www.epexspot.com/sites/default/files/download_center_files/EPEX%20SPOT%20Indices%202019-05_final.pdf
- https://www.jao.eu/news/update-st-auctions-swiss-borders-15-min-mtu-and
- https://www.jao.eu/news/st-auctions-swiss-borders-15-min-mtu
- https://www.jao.eu/news/tsos-survey-introduction-15-minutes-mtu

Older Swissgrid Q3 2026 roadmap must be read with the later JAO update. APG's
2July link confirmed the JAO page. EPEX search-result parameter URLs sometimes
returned a different area/modality and were excluded as resolution evidence.
Web open and EPEX page40 screenshot succeeded. Local raw-source capture failed
on Python Windows certificate-store ASN1 NOT_ENOUGH_DATA before any file was
downloaded; no TLS bypass or policy change. Task retains source-review.json
with links and verified paraphrases, not claimed raw PDF/HTML hashes.

## Fixed benchmark and outcome

Protocol `docs/model/LT-HOURLY-RECENCY-LOCAL-EXPERIMENT.md` written before
new results. Same six D301/D304 origins and delivery/solver/quote surfaces;
2021–2022 diagnostic and2023–2026 already-exposed development assessment.
Compare saved MLP, equal-history signed D304, exponential365.25day and730.5day
half-lives on identical complete pre-origin training months. All negative
observations retained. No new estimator family, hydro, clipping, tuning or
outcome-selected month/horizon switch. Weight ages are relative to the latest
eligible training hour; uniform default unchanged. Existing cell/backoff
reference, signed monthly centering, assembler and EEX projection reused.

One optional `sample_weight` argument added to `calendar_cell_reference` in
`pfc_shaping/lt/signed_benchmark.py`; requires exactly aligned finite positive
weights, normalized by their maximum before weighted aggregation. Default
arithmetic branch preserved. Only this function's AST changed; closed-month
labels untouched. Source pre-edit copy retained; old scientific/source pins
must not be relabelled to accept current code.

Successful `run-v2` froze2026-09-08T06:51:17.103216Z and completed
2026-09-08T06:55:01.962262Z. Twenty-one seasonal calculations/assembled curves
(three per six historical origins plus current), seven copied MLP controls;
zero ML estimator fits. Raw predictions, weights, cell effective support,
pre/final curves/errors, monthly decompositions and EEX gates saved.

Equal-origin assessment results:

| Candidate | Shape MAE/RMSE | Full-price MAE/RMSE | Ramp MAE/RMSE |
|---|---|---|---|
| MLP |22.817767/33.604667 |55.641665/68.442451 |7.923017/13.610352 |
| D304 equal |20.345234/30.180010 |54.847770/66.956943 |6.887920/12.254894 |
| half-life1year |20.134905/29.144896 |53.869560/65.897657 |7.090081/12.067216 |
| half-life2years |19.991763/29.214678 |54.207832/66.181119 |6.954345/12.046375 |

Two-year versus D304: MAE gain1.737363%, RMSE gain3.198581%,3/4 origin wins,
but4 declared regime regressions. One-year gains1.033799%/3.429803%,2/4wins,
12regressions. Both fail preregistered screening (>=2% both metrics,>=3/4wins,
no>5% supported shape/ramp regression). **No adoption** and no month switching.

Two-year regressions: delivery YEAR_2024 shape MAE+6.25%, ramp MAE+9.58%;
autumn ramp MAE+6.79%; near-zero ramp MAE+5.37%. Delivery2024 is distinct from
origin2024. Origin2023 loses versus D304 (19.239701 versus17.923703), while
origins2024,2025,2026 improve. All per-origin and segment results preserved.

Monthly forward-spot level MAE47.260956/RMSE55.960385 is identical across all
four candidates. Independently verified per complete month:
MSE(full)=MSE(shape)+MSE(level). This level discrepancy includes market
information/risk-premium effects and is not evidence of broken solver
constraints; no recentering to future actual levels. Ramps exclude UTC gaps
and Swiss month boundaries. Native D304 score_curves is preserved separately.

Masks: actual<0,abs(actual)<=5,actual>5, pre-origin historical95th percentile
high-price and absolute-ramp thresholds, four seasons, peak/offpeak,
weekend/holiday, horizons1–6/7–12/13–24/25–36months, each delivery year and
COMMON_FIRST8. Support for regression gates>=168hours across>=2origins.
**HIGH_PRICE has zero assessment observations and remains UNSUPPORTED.**
Do not lower that threshold after results or claim spike accuracy. Future
protocol should separate shape-tail events from absolute levels. Overlapping
exposed years and revised extracts imply no independent significance claim.

## Exports and verification

Task root: `build/lt-hourly-recency-20260908/`.
French main report `RAPPORT-COMPARATIF.md`; detailed immutable review in
`review-v1/`. Its comparisons, regime gates, verified-metrics, monthly
decomposition and current profiles by year are CSVs.

Three signed alternatives under `run-v2/current/`:

- `signed-equal/pfc-fmv-ch-1h.csv` (D304 control).
- `signed-hl365/pfc-fmv-ch-1h.csv`.
- `signed-hl730/pfc-fmv-ch-1h.csv`.

Each hourly CSV28513rows, Oct2026–Dec2029. Sibling15min CSV114052rows repeats
each hourly signed price four times. Each `curve.parquet` retains219268QH
through2032. `mlp-control/` also exports its retained hourly average and native
QH control. Retained valuation2026-09-07T08Z, September4 quotes: **not a fresh
September8 revaluation**. All signed current variants80PASS/9QUOTE_CONFLICT,
0CRITICAL; no waiver. No qualified CH native15min accuracy,2030 structural
accuracy, FMV PnL or calibrated risk bands.

Independent verification succeeded on first reviewer execution:

- 799 input pins/95 source-config pins/373 outputs hash verified.
- 21 raw calendar-reference replays use separate NumPy weighted averages and
  explicit dictionary backoff; max error4.263256414560601e-14 EUR/MWh.
- 48 saved hourly stages,3808 metric rows,960 monthly decompositions verified;
  max diagnostic numerical difference2.2737367544323206e-13.
- EEX gates independently recalculated, CSV timestamp/price parity and hourly
  repeated transport verified; max monthly solver residual2.4101609596982598e-11.
- 12096 support-cell calculations independently checked against saved weights
  by separate boolean masks;733 D301 files rehashed unchanged; D304–D306
  artifacts/pins unchanged. Seven signed control assemblies reproduce D304/D305.
- 57 focused tests pass (13 new). Matrix234pass/5skip/2 known Phase5 failures;
  exact XML failure messages equal D306. No golden/fixture edits or waiver.
- `git diff --check` passes, existing LF/CRLF warnings only.

Hashes SHA256:

- pre-edit reference source: `acd322f20e27d6cf1b83c14f2040e99ac5425996bd98eb2d446528bf38e26228`
- run-v2 plan: `0822169468cb9fd1ff59ec250a4fcdda98886e7bab8b8c0807b967f61a53052e`
- run manifest: `05a44fbda7f3619d129a88943e68cf76950ac3564ad25c0fc1b286023152d8ab`
- verification: `2871463cdca19d26f0a1ed3f553e5b7450ad1b54c2d1b6d9f9830b6c9898f3dd`
- review manifest: `3375774ff06ecc7261f7cc4fa85d380501e413914b79ff7a1789f14602d70b19`

Failures: local capture SSL issue above; run-v1 stopped before scoring at
target equality because pandas frequency metadata was Hour versus None in
retained Parquet. Changed only check_freq=False; exact timestamps/values
remain checked. Run-v1 and its frozen plan/source snapshot retained. No scored
candidate-dependent tuning or model rerun after results. Review-v1 succeeded.

## Exact files and commands

Repository changes this turn:

- `pfc_shaping/lt/signed_benchmark.py` optional weight argument only.
- `scripts/run_lt_hourly_recency.py` fixed execution and diagnostics.
- `scripts/verify_lt_hourly_recency.py` independent reconstruction/scoring/report.
- `tests/test_hourly_recency.py`13 new tests.
- `docs/model/LT-HOURLY-RECENCY-LOCAL-EXPERIMENT.md` preregistration.
- `docs/data/CH-DAY-AHEAD-RESOLUTION-20260908.md` official-source review.
- `.planning/HANDOFF.md`, Phase14 `DECISION-LOG.md`, this handoff.
- Task-local `capture_web.py` (failed raw capture), `web-sources/source-review.json`,
  `check_prior_evidence.py`, `finish_review.py`, `close_session.py`, snapshots,
  tests/curves/reports and closure. Main report received an explicit editorial
  interpretation on missing spike support and level-versus-shape limitations;
  original reviewer report remains immutable. Other dirty user work preserved.

Before **every** shell action check current directory exactly
`C:\Users\jbattaglia\PFC_LT` and Git top-level exactly
`C:/Users/jbattaglia/PFC_LT`. All commands use this canonical workdir.
Runtime is `build/conda-runtime-v41-model-source/python.exe -B`.
TEMP,TMP,APPDATA,LOCALAPPDATA,MPLCONFIGDIR,XDG_CACHE_HOME,NUMBA_CACHE_DIR,
JOBLIB_TEMP_FOLDER,PYTHONUSERBASE,PIP_CACHE_DIR each points to
`build/lt-hourly-recency-20260908/runtime/<NAME>`; CUDA_VISIBLE_DEVICES=-1;
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4;
PYTHONDONTWRITEBYTECODE=1. No external mutable paths or approval override.

Commands after these guards/settings:

```powershell
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-recency-20260908/capture_web.py
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_hourly_recency.py tests/test_signed_benchmark.py tests/test_signed_hourly_assembly.py -q --tb=short -o cache_dir=build/lt-hourly-recency-20260908/pytest-cache --basetemp=build/lt-hourly-recency-20260908/pytest-v1 --junitxml=build/lt-hourly-recency-20260908/tests-v1.xml
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_hourly_recency --output build/lt-hourly-recency-20260908/run-v1
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_hourly_recency --output build/lt-hourly-recency-20260908/run-v2
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_hourly_recency.py tests/test_price_conditioned_intraday.py tests/test_signed_hourly_assembly.py tests/test_signed_composition.py tests/test_signed_benchmark.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_lt_evaluation_curve_assembly.py tests/test_lt_local_benchmark.py tests/test_assembler_profile_type.py tests/test_phase05_negative_prices.py tests/test_lt_structural_readiness.py tests/test_intraday_persistence.py -q --tb=short -o cache_dir=build/lt-hourly-recency-20260908/pytest-cache --basetemp=build/lt-hourly-recency-20260908/pytest-matrix-v1 --junitxml=build/lt-hourly-recency-20260908/tests-matrix-v1.xml
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-recency-20260908/check_prior_evidence.py
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.verify_lt_hourly_recency --run build/lt-hourly-recency-20260908/run-v2 --output build/lt-hourly-recency-20260908/review-v1
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-recency-20260908/finish_review.py
& build/conda-runtime-v41-model-source/python.exe -B build/lt-hourly-recency-20260908/close_session.py
git diff --check
```

Existing output destinations must not be reused. closure.json binds final
source/docs/tests/review manifests/report hashes; runtime caches excluded.

## Next boundary

Keep D304 equal-history signed reference as primary local hourly control and
MLP as incumbent; retain one/two-year candidates as alternatives. Do not adopt
recency by aggregate gain. Next should address stability across delivery years
and near-zero/autumn ramps, with a pre-origin shape-tail control whose support
is explicit. No outcome-selected month switching. Monthly forward-spot gaps
must be interpreted separately from solver constraint correctness and cannot
be repaired by post-solver patches. Independent holdout/CH15min truth/2030
physical evolution remain separate unresolved qualifications. All six model,
production, trading and registry authority fields remain false.
