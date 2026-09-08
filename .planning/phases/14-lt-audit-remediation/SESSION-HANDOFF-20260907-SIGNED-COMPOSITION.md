# D305 signed PFC composition — 7 September 2026

## Final status

Complete. **run-v4** is the final canonical execution, ending
2026-09-07T15:00:02.429140Z. No active process remains. All run-v3 numerical
outcomes below are reproduced in v4;156 CSV/Parquet files are byte-identical
(`reexecution-comparison.json`). No model adoption or operational promotion.
The user was told that the next lot can use a new window and this handoff.

Final evidence:151 pinned inputs,87 source/config files,203 outputs;48 independently
recomputed pre/final hourly error series, maximum1.7053025658242404e-13 EUR/MWh.
12 saved water contributions and16 saved factor-model predictions replay exactly;
eight additive reference calculations replay exactly. Native D300/D301 controls
are exact; signed D304 replay maximum5.684341886080802e-14. The added quarter
residual changes hourly means by at most1.9326762412674725e-12. Monthly solver
residual maximum2.4101609596982598e-11. All five current alternatives keep80PASS/
9QUOTE_CONFLICT, noCRITICAL; authority fields remain false.

Across this turn:44 successful statistical fits (22 each in v3/v4), plus one
run-v1 water-fit call taking its n_obs=0 fallback before failure. Run-v2 had no
fit. Each successful run also computes eight historical and one current additive
reference, plus one current hourly reference; no current saved-model refits.
Verifier replays are predictions/arithmetic only, with no new statistical fit.

Report: `build/lt-signed-composition-20260907/RAPPORT-COMPARATIF.md`.
Verified comparisons: `review-v4/hydro-comparison.csv`, `hydro-by-origin.csv`,
`de-comparison.csv`, `current-profile-comparison.csv`, `verification.json`.
Five current CSVs: `run-v4/current/{mlp-current,signed,signed-hydro,
signed-intraday,signed-both}/pfc-fmv-ch-15min.csv`; same folders' `curve.parquet`
retain the full native horizon. Additive variants are explicitly rejected as
unconditional model choices despite their aggregate DE gain.

Source preservation:81 pre-edit D304 files under `baseline/`;87 executed source/
config files under `source-snapshot/`. The AST comparison confirms only `build`
and `_build_signed_hourly_shape` changed among existing assembler methods.
See `assembler-d305.diff`, `assembler-change-review.json`. All733 D301 inventory
files and276 D304 run outputs remain hash-identical (`prior-artifacts-verified.json`).
Other dirty user work was preserved. No commit/push or external publication.

Hashes (SHA256):

- final plan: `2b1eda9e4e3d03f142aaf8056922e2c01551dfcea3be3dfc88fd3304e46904ff`
- final run manifest: `72509923aab50948116544ec3d3f85c6bcce4366896d938174899df068b87df2`
- verified review: `c08409f30ac0121b7c34ef7472433d22fada0a50b9ae1b33a0e1950a61029972`
- source snapshot manifest: `e702659ac36ae30b7b3f390779af98e8ce4eecfcdbc07fe57a9093a02b96f64a`
- live assembler: `61a5e5d4cb64661da56d681f4337c3bb59d5f9b1a17d591fd709a1b809219da7`
- selected matrix XML: `c85054fa99e087b9ff587d235ab505d7c53202ba40cd18449d0274e8cec2d046`

`closure.json` records final task/doc/source hashes, including this handoff and
the report; runtime caches and pytest scratch directories are excluded.

Exact substantive commands, after canonical cwd/Git guards and runtime variables
described below (all mutable arguments are below the task root):

```powershell
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_signed_composition --output build/lt-signed-composition-20260907/run-v1
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_signed_composition --output build/lt-signed-composition-20260907/run-v2
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_signed_composition --output build/lt-signed-composition-20260907/run-v3
& build/conda-runtime-v41-model-source/python.exe -B -m scripts.run_lt_signed_composition --output build/lt-signed-composition-20260907/run-v4
& build/conda-runtime-v41-model-source/python.exe -B build/lt-signed-composition-20260907/verify_and_report.py
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_signed_hourly_assembly.py tests/test_signed_composition.py tests/test_signed_benchmark.py -q --tb=short -o cache_dir=build/lt-signed-composition-20260907/pytest-cache --basetemp=build/lt-signed-composition-20260907/pytest-v2 --junitxml=build/lt-signed-composition-20260907/tests-v2.xml
& build/conda-runtime-v41-model-source/python.exe -B -m pytest tests/test_signed_hourly_assembly.py tests/test_signed_composition.py tests/test_signed_benchmark.py tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py tests/test_lt_evaluation_curve_assembly.py tests/test_lt_local_benchmark.py tests/test_assembler_profile_type.py tests/test_phase05_negative_prices.py tests/test_lt_structural_readiness.py tests/test_intraday_persistence.py -q --tb=short -o cache_dir=build/lt-signed-composition-20260907/pytest-cache --basetemp=build/lt-signed-composition-20260907/pytest-matrix-v1 --junitxml=build/lt-signed-composition-20260907/tests-matrix-v1.xml
git diff --check
```

Targeted v1 used the same command with `pytest-v1`/`tests-v1.xml`. The initial
successful verifier ran on v3 and wrote root review files; its exact source was
retained as `verify_run_v3_source.py`. The final verifier targets v4 and writes a
fresh `review-v4/`. Never rerun into those completed destinations. Runtime logs
are retained in each run; XML captures test outcomes. Diff check passes with
existing benign LF-to-CRLF warnings; no unrelated formatting change made.

Resume objective: retain the signed hourly gain, compose the existing intraday
component in price space first, and test any refinement against negative and
near-zero price segments before accepting it. Do not switch models month by
month using the exposed scores. CH native15min truth is an empirical data gap,
not a blocker to further local engineering. The next physical2030 lot must still
connect coherent scenarios, chronology, flexible operation and final PFC shape.

## Historical checkpoint (superseded by the final status above)

User requested continuing and asked whether a new window was needed. Local CPU
execution remains authorized; no further permission requested. This session
continues D304 with hydro ablation, native DE intrahour diagnostics and current
PFC alternatives. No agents, Warehouse, GPU, CT, AFRY or T057. All authorities false.

Task root: `build/lt-signed-composition-20260907/`.
Canonical workspace and Git-root guards precede every shell action.
Runtime: `build/conda-runtime-v41-model-source/python.exe -B`, all ten cache/temp
variables below task `runtime/`, OMP/OPENBLAS/MKL threads4, CUDA_VISIBLE_DEVICES=-1.

Changed/added this turn:

- `pfc_shaping/lt/model/assembler.py`: extends the existing signed lane with
  optional `signed_intraday_shape` EUR/MWh, exactly aligned and already neutral
  within every native hour. Existing non-neutral f_Q remains rejected. Supports
  the existing additive water API with a complete finite forecast, rejecting
  orphan model/context, floors, invalid/nonneutral deltas. One final projection;
  no new adapter or changes to native multiplicative calculations.
- `pfc_shaping/lt/signed_intraday.py`: unconditional DE additive seasonal residual
  reference, four explicit calendar backoffs and final parent-hour centering.
- `scripts/run_lt_signed_composition.py`: fixed experiment, retained-source
  hash checks, local runtime/output guards, six hydro origins, eight DE months,
  five current PFC alternatives, saved fits/curves/errors/gates/receipts.
- `tests/test_signed_composition.py`:34 new tests.
- `tests/test_signed_hourly_assembly.py`: one expected error-message match
  broadened for now-explicit paired hydro/model guard.
- `docs/model/LT-SIGNED-COMPOSITION-LOCAL-EXPERIMENT.md`: protocol fixed before
  outcomes; pre-score refinement for unsupported ISO week53 hydro anomalies.
- `docs/model/LT-STRUCTURAL-SHAPING-CONTRACT.md`: D305 outcome and remaining scope.
- Mandatory current handoff, this session handoff and D305 decision log.

Before edits,81 D304 source/config files were hash checked and captured under
`baseline/`; manifest SHA256
`c0e3705a9f6b1dfc7a20a6955e1d04e7a57e921fa4375c8777a4bb614114b8fc`.

Execution history / failures:

1. Targeted tests v1 failed collection because a sibling test import lacked
   the `tests.` package prefix. Fixed test import; v2 **78 passed**.
2. Experiment run-v1 froze its plan, called water fit once with fill_pct-only
   D301 history (native fallback, n_obs=0), then failed constructing the anomaly
   forecast. No scored curve. Fixed by calling the existing causal
   `build_hydro_water_value` on pre-origin history and requiring n_obs>=12.
3. Run-v2 stopped before fitting: ISO week53 at origin2021 had no supported
   anomaly. Refined preparation before any score: exclude unsupported neutral
   sentinels and take latest supported observation no more than14 days old.
   Origin2021 uses2020-12-20T23Z (10.54 days old); no future-data substitution.
4. Run-v3 completed14:49:46.019300Z–14:54:10.352726Z. Six water fits,16 intraday
   factor fits, eight historical and one current additive calculations, one
   current hourly reference calculation,29 curves. Independent verification:
   48 pre/final error stages (max1.7053e-13),12 water and16 factor replays exact,
   eight additive replays exact;149 inputs/85 direct source pins/203 outputs.
5. Review found two newly used helpers only transitively named by D300 receipts,
   not directly checked before execution: `lt_replay_transforms.py` and
   `production_phases.py`. Added explicit code checks against those pre-existing
   D300 hashes. Run-v4 repeats the unchanged numerical experiment with complete
   direct binding; plan SHA256
   `2b1eda9e4e3d03f142aaf8056922e2c01551dfcea3be3dfc88fd3304e46904ff`.
   At this checkpoint run-v4 is active (exec session22463). Final work: verify
   run-v4, compare to run-v3, write French report and closure; do not change
   its pinned source/spec while running. Prior outputs remain intact.

Selected matrix: **196 passed,5 skipped,2 existing Phase5 fixture failures** in
44.90s, XML `tests-matrix-v1.xml`. The two exact assertion messages match D304
(`preexisting-fixtures-comparison.json`), which already reproduced both against
pre-D304 assembler bytes. No golden fixture alteration or waiver. Four optional
CT imports and the existing synthetic summer-bowl test account for skips.

Run-v3 outcomes (await exact run-v4 closure): equal-origin final centered hourly
MAE/RMSE, four exposed assessment origins2023–2026:
MLP22.817767/33.604667; MLP+hydro22.821732/33.610301;
signed20.345234/30.180010; signed+hydro20.352302/30.184130.
Hydro adds no demonstrated improvement. Existing water fit still groups in UTC
months, so Swiss edge-month aggregation is an explicit unchanged limitation.

DE conditional disaggregation (actual parent-hour price supplied to factor
models; NOT a future price forecast):23,324 native QH, eight monthly tests.
MAE/RMSE flat7.926110/13.803879, native6.571542/12.322029,
regularized6.487984/12.287431, additive6.225856/11.424343.
But negative-parent MAE: native2.152440 versus additive4.707404 over1,728 QH.
Additive wins only5/8 months; reject unconditional adoption despite mean gain.
User already informed of this decisive negative-price regression.

Current alternatives Oct2026–Dec2029 all114,052 QH; native full horizon to
Dec2032 is219,268 QH. Same solver levels, all80 PASS/9 QUOTE_CONFLICT, noCRITICAL.
Native control minimum11.3889/0negative QH; signed-only minimum-29.7471/480;
signed+hydro minimum-29.7725/480; signed+intraday minimum-66.6515/541;
signed+both minimum-66.6769/542. These are descriptive forecast outputs, not
future accuracy or desired negative-frequency evidence. All remain alternatives;
no automatic replacement of D300, no risk bands, no structural2030 qualification.

Next useful lot: price-conditioned intrahour residuals with explicit negative
and near-zero regime gates, rather than unconditional DE residual transfer or
automatic hydro activation. CH native15min truth remains needed to establish
CH transfer accuracy. Keep the signed hourly reference and monthly authority.
