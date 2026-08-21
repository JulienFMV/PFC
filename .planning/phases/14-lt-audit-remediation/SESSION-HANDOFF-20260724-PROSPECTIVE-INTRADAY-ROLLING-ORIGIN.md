# Session handoff — prospective intraday rolling-origin and local PFC replay

Date: 2026-07-24  
Branch: `fix/lt-audit-remediation`  
HEAD inspected: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Workspace: `C:\Users\jbattaglia\PFC_LT`  
Production status: **STRICT NO-GO**

## Objective and invariants

This session resumed the permanent objective of a production-ready Swiss FMV
LT PFC, with the immediate focus on a robust local run and visible scientific
quality. The following invariants were preserved:

- no production promotion or commit;
- no reset, clean, or restore of the intentionally dirty worktree;
- no edit or staging of `data/eex_forwards_history.parquet`;
- no change under `pfc_shaping/ct/` or Power BI;
- the monthly solver remains the level authority;
- OMPEX remains benchmark-only and is never a model input;
- experimental intraday regularisation remains default-off.

## Fresh public-data capture and quarantine

Six Energy Charts acquisitions were captured under
`output/phase14/prospective_public_capture_20260724`. They are unsigned local
quarantines, not production evidence. Independent audit status for all six is
`VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION`.

| Acquisition | Manifest SHA-256 |
|---|---|
| CH recent | `160aed6566e8edb8d4fdb7edcad6a0ff54e67419255bce2cd7a284b47792a372` |
| DE spring | `07e0c3897015e2c8634d77d9c3e1895694d9f79616d5feb20d426632d94d342d` |
| DE summer | `c630d175a00c1115e7fcde5ca6e6cff611cb82267cba541e764acf1c6aa8a49d` |
| gap 1 | `56ca229b27e105fd7be2f028a84e923f27c7cd050adad09e6f7dd9032e659178` |
| gap 2 | `2e6079767a6ff36e07d8669a8affa151ab4857b1f7dd0553a52b91bac9b4cb95` |
| gap 3 | `f85f5d5dc5865330602f500e35946e0e12a58c51fb0e651618d6617c991e5bf5` |

The consolidated diagnostic panel is:

- path: `output/phase14/prospective_intraday_panel_20260724/epex_de_15min_complete.parquet`;
- SHA-256: `e5b0a6d6e3c6837ae1728f25ea25bdbee9278d8eefd414f4476669d7286268ab`;
- manifest SHA-256: `af8232e7045eb9aa685d946770a56a108e438a6a71e9cbdab47614abcc5a9e4f`;
- coverage: 28,416 contiguous quarter-hours from 2025-10-01 through
  2026-07-23 23:45 UTC;
- status: `COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO`.

Known authority limitations are explicit: the base segment is ungoverned, the
workstation clock is not a trusted timestamp authority, and use of DE spot data
as a proxy for CH intraday shaping has not been scientifically validated.

## Intraday diagnosis and rolling-origin decision

The extreme incumbent cells were traced to very sparse parent-hour support.
The worst diagnosed example was `Printemps/Samedi/h09`, based on nine parent
hours, with quarter-hour factors
`[2.260017, 1.123301, 0.459689, 0.156992]` and maximum deviation
`1.26001738`.

A sparse-cell candidate was implemented for evaluation only: price-space WLS
for cells with at most 24 parent hours, contracted to the maximum deviation
envelope observed on dense training cells. Its authoritative evaluation is:

- summary:
  `output/phase14/intraday_shape_rolling_origin_20260724_v3/summary.json`;
- summary SHA-256:
  `523169da91a3ec3ab46f341608274a449f035488bf671d2af82481112a3d8f09`;
- fold CSV SHA-256:
  `910dd2dbe0a2c87f345f05431be14769d45e4113e1f54301176ba25fc7ecfe45`;
- 16 rolling-origin folds and 21,504 evaluated quarter-hours;
- candidate MAE `5.68178684` versus incumbent `5.78312867`;
- candidate RMSE `11.31838592` versus incumbent `11.53373319`;
- fold outcomes: six wins, six losses, four ties;
- latest evaluable fold: candidate inferior.

Decision: `LOCAL_DIAGNOSTIC_CANDIDATE_REJECTED_NOT_PRODUCTION`. The aggregate
improvement is insufficiently stable across origins. The feature is therefore
default-off in `ShapeIntraday` and can only be activated by the explicit local
runner switch `--enable-experimental-sparse-intraday-regularization`.

This rejection does not validate the incumbent. In the current replay, its
quarter-hour factor range remains `0.3812253366` to `1.9248631083`, with 67
future quarter-hours satisfying `|fQ - 1| > 0.5`. That tail has not received a
proper out-of-sample test on new, unused data.

## Integrity and TOCTOU closures

The local runner was hardened while preserving its non-production boundary:

- output-prefix traversal outside the governed root is rejected;
- the intraday provenance JSON is bound to the exact panel path and SHA-256;
- inventory, governance YAML, hourly, forwards, intraday, and provenance inputs
  are captured as bytes and the exact captured bytes are consumed;
- the monthly solver accepts the already captured forwards payload, eliminating
  the verification/import re-open window;
- output parents are revalidated around Parquet directory creation;
- completion is published last and remains explicitly local, unsigned, and
  non-promotable;
- sparse-model diagnostics survive save/load;
- the test harness now rejects permissive false-PASS cases.

## Current authoritative local replay

The current source state was replayed twice without the rejected experimental
regularisation:

- run 8:
  `output/phase14/local_pfc_prospective_intraday_20260724_run8`;
  completion-manifest SHA-256
  `de2ba171a803a322a8dab69e6f7467ed2d7aa466e9bfcd488483676f745b61bc`;
- run 9:
  `output/phase14/local_pfc_prospective_intraday_20260724_run9`;
  completion-manifest SHA-256
  `fd07fc0c11f2fe545c738912cabddfe7513cf8c5e2d3587a21ea7c4abc11c939`.

The command form was:

```powershell
python scripts\build_local_test_ch_pfc.py `
  --intraday-data output\phase14\prospective_intraday_panel_20260724\epex_de_15min_complete.parquet `
  --intraday-provenance output\phase14\prospective_intraday_panel_20260724\panel-manifest.json `
  --output-root output\phase14\local_pfc_prospective_intraday_20260724_<run>
```

Code hashes recorded by both runs:

- runner: `3d463e4d84f27d73f1e92a9f2cd3c944a5b0f409816510ecf37269f0774ea87b`;
- PFC builder: `055945dbdbd660c1cef2a3efe59b3ec4ace1db137f8c1c8c4c7e2015df4042bd`;
- intraday model: `1b2171f181597c510db9b257ce6b8977595d0b7bf24621809c72067f8c332a4e`.

Eight material outputs are byte-identical between runs. The completion and
quality documents remain path-bound by design. Central PFC diagnostics:

- mean: `85.48362297717911 EUR/MWh`;
- minimum: `4.468397139799421 EUR/MWh`;
- maximum: `237.39736787425164 EUR/MWh`;
- maximum hourly quarter-factor neutrality residual:
  `2.220446049250313e-16`;
- maximum monthly solver KKT residual:
  `2.842170943040401e-14`;
- source classification: `TEST_FIXTURE`;
- promotion: `false`;
- probabilistic P10/P90: absent.

The run is reproducible and useful for local quality inspection, but it is not
a production candidate.

## Verification matrices

Final executed results on the current source state:

- Ruff targeted checks: `All checks passed!`;
- focused delta suite: `58 passed in 9.38s`;
- LT/model/monthly-solver suite: `191 passed, 1 skipped in 86.52s`;
- packaging/runtime/acquisition suite:
  `165 passed, 14 skipped in 66.25s`;
- publication/external-CAS suite:
  `240 passed, 2 skipped in 225.81s`;
- explicit captured-root-only `sys.path`: `3 passed in 0.27s`;
- real optimized zipapp test with the handoff paths:
  `1 passed, 53 deselected in 260.65s`.

The optimized zipapp used:

- wheelhouse:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252`;
- dependency root:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages`;
- receipt:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages\dependency-closure-receipt.json`.

Only the captured dependency root entered `sys.path`; the host site-packages
path did not.

## Independent readonly roasts

- Security: no remaining demonstrated P0/P1 in the session delta after the
  path-confinement, byte-capture, provenance-binding, and parent-revalidation
  fixes. Production remains NO-GO.
- IT/Operations: GO for the local diagnostic software delta; no demonstrated
  P0/P1. The only code-style finding was an unused variable, subsequently
  removed and Ruff re-run successfully. Production remains NO-GO.
- Quant/Data: rejection of the sparse regularisation candidate is correct.
  Scientific P1s remain: preregister and test any new sparse policy on unused
  fresh data, validate DE-to-CH transfer, and quantify the incumbent sparse
  tail before candidate status.

## Files changed by this focused continuation

- `pfc_shaping/lt/model/shape_intraday.py`
- `pfc_shaping/pipeline/monthly_curve_authority.py`
- `scripts/build_ep2050_multi_scenario_pfc.py`
- `scripts/build_local_test_ch_pfc.py`
- `scripts/capture_public_energy_charts_lt.py`
- `scripts/audit_provider_acquisition_quarantine.py`
- `scripts/build_local_intraday_calibration_panel.py`
- `scripts/backtest_intraday_shape_estimator.py`
- `tests/test_intraday_amplitude.py`
- `tests/test_build_ep2050_multi_scenario_pfc_script.py`
- `tests/test_build_local_test_ch_pfc_script.py`
- `tests/test_capture_public_energy_charts_lt_script.py`
- `tests/test_audit_provider_acquisition_quarantine_script.py`
- `tests/test_build_local_intraday_calibration_panel_script.py`
- `tests/test_backtest_intraday_shape_estimator_script.py`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- this handoff.

The repository contains many intentional pre-existing dirty changes outside
this focused continuation. They were preserved.

## Next best actions

1. Acquire a genuinely governed, signed, fresh point-in-time CH intraday panel
   with trusted capture time and independent provenance.
2. Preregister the sparse-tail policy and thresholds before observing the next
   holdout; evaluate it rolling-origin and on unused fresh data.
3. Establish or reject DE-to-CH transfer with direct CH evidence and explicit
   capture-factor/product-loss metrics.
4. Re-run the locked T057 decision path only under its one-shot governance and
   then build a new auditable CH candidate if all gates pass.
5. Add calibrated probabilistic outputs/scenarios and coverage diagnostics;
   current P10/P90 are absent.
6. Keep production, promotion, and monthly production flags disabled until
   independent real manifests, external CAS/service identity, operational DR,
   and all scientific gates are complete.
