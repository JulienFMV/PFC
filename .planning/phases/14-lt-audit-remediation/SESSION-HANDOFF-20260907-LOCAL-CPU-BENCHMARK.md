# Local CPU benchmark — 7 September 2026

## Final status — complete

The authorized implementation, CPU execution, independent numerical review and
comparative report are complete. No active worker remains. The historical
checkpoint below is retained for provenance; its remaining-work list is now
superseded by this final status.

Primary delivery below `build/local-lt-benchmark-20260907/`:

- `rapport-comparatif.html`: canonical portable read-only report,22blocks,
  eight tables, one horizontal comparison chart; no external publication.
- `RAPPORT-COMPARATIF.md`: readable text companion.
- `comparative-summary.csv`, `assessment-by-origin.csv`,
  `assessment-by-segment.csv`, `sensitivity-common-horizons.csv`, `all-metrics.csv`.
- `benchmark-review.ipynb`: executed review cell reading verified results;
  detailed reproducible arithmetic in `verify_and_summarize.py`.
- `verification.json`, `final-review.json`, `report-delivery.json`,
  `report-notes.json`, `artifact.json` and final `session-artifacts.json`.

### Results and their limits

Equal-origin average scores in EUR/MWh; tuning2021–2022, assessment2023–2026:

| Model | MAE | RMSE | Mean origin P95 | MAE change versus corrected MLP |
| --- | ---: | ---: | ---: | ---: |
| Current hydro-corrected MLP |22.817767|33.604667|70.028154|0%|
| Original frozen MLP diagnostic |22.927858|33.700336|70.321930|+0.482480%|
| LightGBM CPU |22.915155|33.843064|71.349182|+0.426809%|
| Seasonal reference |23.113075|33.925311|71.166450|+1.294200%|
| GAM |25.024331|36.638965|78.400960|+9.670376%|
| Ridge |25.930839|37.630382|80.721992|+13.643191%|
| Weighted MLP |UNSUPPORTED|UNSUPPORTED|UNSUPPORTED|two failed fits|

No global replacement is justified. Corrected MLP improves reference MAE by
1.277665%, RMSE by0.945146%, and MAE in4/4origins; the combined2% target is
not attained. LightGBM improves reference MAE on1/4origins. It is much faster
to fit: mean0.653811s per assessment origin versus66.967042s native MLP
(native fit includes its preprocessing/f_W; common materialization/assembly
is excluded from these fit timings).

The ranking depends on horizon: first6months/fourorigins gives LightGBM
MAE23.309350 versus MLP23.439637; first12months/threeolderorigins gives
21.366360 versus21.916840, a2.511676% LightGBM MAE gain. These are sensitivity
views of saved predictions, not authorization for outcome-chosen horizon
switches or independent confirmation. Original frozen MLP remains diagnostic,
not a silently replaced baseline. A global near-tie is not a statistically
established equivalence result.

There are32,135unique assessment hours and70,101origin-hour pairs (dependent,
overlapping deliveries), with36/32/20/8complete months. Historical native CH
quarter-hours are transport; scoring counts actual hours. Negatives remain in
truth: negative-hour centered MAE80.142965 for current MLP,84.566125 LightGBM,
81.798654 reference. Overall centered bias is algebraically nearzero, not
predictive skill. Future years_ahead is unlearned because training values are
allzero. Physical/QH/WV layers are neutral and their superiority is untested.

Weighted MLP exhausted the fixed500iteration/tol1e-8 budget in both tuning
origins,17.435588s and26.945987s. Both failures and traceback files remain;
no relaxed tolerance, substituted model or approximate test score. Selection
was locked at09:53:11.696955Z before assessment: Ridge alpha100; GAM alpha100,
5knots; LightGBM learning_rate.05/min_data_in_leaf100/n_estimators300/num_leaves31;
weighted MLPnull. Final review independently reconstructs that choice from
tuning metrics and verifies every assessment configuration matches.

### Execution and market checks

Main execution ran approximately09:49:20–10:02:24.944447Z, about13minutes.
104reference/candidate attempts:102scored,2nonconverged;98actual estimator fits
including12native fits, with6untuned reference calculations. There were no
new Databricks/warehouse calls, GPU, CT overlay, AFRY, T057 or protected desk-data
writes. Dependency provisioning downloaded only the explicit LightGBM wheel.
Scientific v6, all D300 prepared/fitted/curve hashes and all negative operational
authorities remain unchanged. No scientific truth/admission/registration is
manufactured by this retrospective run.

Saved per-model audits have noCRITICAL. Reference product counts by origin
(all models use the same source constraints within an origin):

| Origin | PASS | QUOTE_CONFLICT | UNSUPPORTED |
| --- | ---: | ---: | ---: |
|2021|12|0|6|
|2022|12|0|6|
|2023|22|0|11|
|2024|45|2|4|
|2025|44|0|2|
|2026|65|6|0|

UNSUPPORTED here means missing required PEAK quotes, not missing hourly truth.
Retained BASE/PEAK counts are6/0,6/0,11/0,12/8,10/8,15/15. The2024conflicts are
Q2BASE plus impliedOFFPEAK;2026has Q2 andCAL2027BASE/PEAK/impliedOFFPEAK.
No waiver or all-PASS interpretation. This variation limits the comparison's
generality to fully PEAK-constrained deployment.

After observing that coverage change, a separately frozen explanatory
`base-only-sensitivity/` reran only assembly on all four assessment origins
for reference/currentMLP/LightGBM.12additional predictions,0fits, unchanged
models/monthlyBASE, pre-damped factors reconstructed from saved models (never
double-damped from exported f_H). Same existing assembly seam. BASE-only mean
MAE: MLP24.065164, LightGBM24.131923, reference24.624267. This does not reverse
the global ranking and does not replace primary scores. Exact recipe/plan:
`base_only_sensitivity.py`, `base-only-sensitivity/plan.json` and complete.json.

### Verification and final artifact identities

108tests passed,4optional CT skips. The102primary predictions were independently
recentered/scored with pandas arithmetic from saved prices; maxerror residual
1.9895196601282805e-13EUR/MWh. Every shared scoring population agrees. Maximum
monthly solver residual2.4101609596982598e-11EUR/MWh. Independent BASE-only
recalculation:12curves, maxMAE residual7.105427357601002e-15. Twelve native model
hashes, all frozen source/input hashes, current selection, original D300 model
files/curve and normalized frozen MLP hash reverified. `git diff --check`
passes; untracked new source/test/spec files also checked for whitespace.

| Artifact | SHA-256 |
| --- | --- |
|plan.json|d0ea1b8fc21725e74da9cdf1234e4e473c3f877f0b9c39c8a1856028a74f56e1|
|selection.json|df47ff4e9df72a42157051eb37688e69131f3fd80c298ab4ae8f30efde6bbc9e|
|comparative-summary.csv|b5f47240e033658e242f55129bbe6a54c15b62f33974b7af4b4ffb97f1f7de0b|
|verification.json|198f7d078ef45cd99bf32a5c1d993bb1ecac6506dfc11b651a80f622515401dd|
|final-review.json|518d15160a7379fa840fdeed3d30396ee745de15b638d118833504088f1aff34|
|artifact.json|b7c00602b72d16416a4497f729c9be4631b91438b06fe7f4f06c4adb293ce4d4|
|rapport-comparatif.html|16ef74c77f316fc6aa59031a75b1e93062d4f86b38801608f9e6a629ff0d1b6a|
|tests-matrix-v1.xml|37f3e7f4aeac77fcc6ecff976e26979fee96b48ea0e65a6b7c31cc0bbe5341c7|

New task-only commands after the same cwd/Git-root/runtime guards:
`python -B verify_and_summarize.py`, `python -B base_only_sensitivity.py`,
`python -B prepare_report.py`, copied repo-local Node with`deliver_report.mjs`,
then`python -B final_review.py`. Prefix task-script paths with the task root.
No primary model/source code was changed after plan freeze. Report and
independent-review scripts live underbuild and are inventoried at closure.

Report packaging reused hash-verified plugin code/assets (36files) and a
repo-local copy of Node24.18.0, SHA
9a4eb5f1c29c6a2e93852ead46b999e284a6a5ca8bab4d4e241d587d025a52de.
First packaging rejected Python text in the SQL provenance slot. Correction
restored the actual original ENTSO-E acquisition SQL, explicitly labelled
acquisition-only, while preserving separate exact Python scoring/verification
paths and EEX/SFOE lineage. No query or metric was fabricated to pass validation.
Final canonical validation/package/payload structural verification pass. Browser
QA was intentionally not run because AGENTS prohibits workstation browser
runtimes: existing renderer dependency injection returns a truthful policy
unavailability reason; source dialogs/layouts are not browser-qualified. HTML
retains enhanced reader plus semantic no-script tables. No npm/system mutation,
custom replacement renderer, browser launch or external publication.

### Smallest useful next act

Keep the current MLP for the global local curve. The next bounded modeling act
is to specify and qualify weighted-MLP convergence (and an optimizer-matched
unweighted ablation), plus a predeclared LightGBM maturity/PEAK-constraint
comparison. Do not choose horizon switches on these test outcomes. Negative
price targets and intraday regularization deserve separate experiments with
their own coverage. Data engineering owns historical PEAK availability and
producer publication/revision meanings; consumer/model team owns these code
and evaluation defects. Externally registered origins, admitted timing and
independent future holdout still block scientific confirmation, not this
completed local engineering benchmark. No further acquisition is required to
inspect these results.

## Historical checkpoint (execution was in progress)

User explicitly requested implementation and execution through the comparative
report. Authorization is already present. No new permission loop. Read AGENTS,
root HANDOFF, the benchmark-next-step note and D300 source integration handoff.

New source files: `pfc_shaping/lt/local_benchmark.py`,
`scripts/run_lt_local_benchmark.py`, `tests/test_lt_local_benchmark.py`,
`docs/model/LT-LOCAL-CPU-BENCHMARK.md`. Handoff/root/decision log also updated.
Frozen scientific modules and the original MLP are unchanged. No new assembly
adapter: local execution uses existing `_InjectedHourlyFactors`,
`_build_from_template`, `_validate_common_assembly` and PFCAssembler. Numerical
weighted MLP/scoring kernels are reused directly, without SyntheticTrainingSet,
synthetic manifests, forged admissions or prospective origin substitution.

Work root: `build/local-lt-benchmark-20260907/`. Interpreter:
`build/conda-runtime-v41-model-source/python.exe -B`. All mutable environment
paths under this task's runtime, 4 BLAS/CPU threads, CUDA_VISIBLE_DEVICES=-1.
Exact LightGBM4.6.0 wheel installed --no-deps/--no-compile into task dependencies,
via existing copied pip24 wheel; `lightgbm-install.json` retains URL/hash.

Commands, after canonical cwd and Git-root guards and repo-local cache env:

1. pip module with --isolated --disable-pip-version-check install
   --only-binary=:all: --no-deps --no-compile --target task/dependencies
   --cache-dir task/runtime/pip_cache_dir --report task/lightgbm-install.json
   lightgbm==4.6.0.
2. `-m pytest tests/test_lt_local_benchmark.py -q` with local cache/basetemp/JUnit:
   8 passed, `tests-unit-v1.xml`.
3. `-m scripts.run_lt_local_benchmark freeze`: six origin preparations succeed,
   no fit. Plan frozen 09:48:34.871618Z, SHA
   `d0ea1b8fc21725e74da9cdf1234e4e473c3f877f0b9c39c8a1856028a74f56e1`.
4. Matrix of local tests, required four LT files, evaluation target/assembly/
   execution/challengers/engine: **108 passed, four optional CT skips**,
   `tests-matrix-v1.xml`. No installed-wheel/CI qualification claim.
5. `-m scripts.run_lt_local_benchmark run *> task/run-console.log`, active
   tool exec session 91263 at this checkpoint; execution.log + per-model JSON.
   Poll before launching another run. Do not mutate frozen source while active.

## Design and actual support

Full design is frozen in docs/model/LT-LOCAL-CPU-BENCHMARK.md and plan.json.
Origins: 31 December previous year 12:00 UTC. Two tuning delivery years
2021/2022; expanding training starts first complete Swiss day January2019.
Four assessment forecasts2023/2024/2025/2026, max36months, truth through August
2026. Complete observed months36/32/20/8. Corresponding training hourly targets:
17,398 /26,133 /34,892 /43,627 /52,290 /61,025. EEX retained quote counts:
6/6/11/20/18/30. Own per-origin history strictly quotation-date filtered.

Each fold binds training prices/hydro, common native-equivalent matrix,
prediction matrix, EEX surface/history, monthly solver output, coverage and
hashes in inputs.json. All learned components are refit inside each origin.
CH hourly source quarter-hours are transport; scoring counts native hours.
Native positive-day target exclusion applies to learning only, never truth.
No values are interpolated. Incomplete truth months are uncountable for every
candidate. Common f_Q=1, WV absent/delta0, physical context absent: hourly-effect
experiment, not a test of D300's full intraday/WV configuration.

Reference: untuned seasonal/daytype/hour mean target, explicit fallback counts.
Native corrected MLP plus original frozen diagnostic. Four challenger grids
1/5/15/16 configurations. Select by mean two-tuning-origin MAE before assessment;
failed configurations are ineligible. Exact weighted MLPmaxiter500/tol1e-8;
first2021 fit failed at 500iterations after17.436s, preserved as
FAILED_NOT_RANKABLE. Do not silently relax convergence or substitute a model.

All central predictions use the existing common full-price assembly and
market product audit. Quote conflicts are retained; critical gates disqualify
scores. Score monthly-centered EUR/MWh errors, overall and horizon/regime masks.
Equal origin averaging; overlapping forecast deliveries are dependent. No
historical PIT, scientific significance, independent holdout or promotion claim.
All production/promotion/scientific-admission/trading authorities stay false.
No Databricks/Warehouse/GPU/AFRY/CT/T057 or protected desk-data mutation.

## Remaining work

Finish execution, inspect all results/failures and per-origin market gates.
Implement independent recomputation from saved predictions and original prices,
verify common layers/monthly levels/source hashes, then build the comparative
report with exact per-origin/horizon/regime numbers and policy-target status.
Do not choose test-dependent model hyperparameters or call a retrospective
development leader a validated winner. Update this checkpoint with final hashes,
reports, tests, caveats and smallest next act.

Skills applied: validate-data and its analyze-data-quality companion, followed
by build-report technical specification for the model-comparison deliverable.
Selected report mode portable HTML in this Codex runtime, with supporting
audit/code/CSV artifacts. Skill renderer's browser verification conflicts with
AGENTS's explicit prohibition on workstation browser runtimes; use canonical
structural validation only and disclose that limit, never launch a browser.
No external publishing is authorized by this local task. Renderer support may
be copied hash-verified below build; no npm/system mutation outside workspace.
