# Phase E Behind-Flag Integration Summary - 2026-06-19

## Scope

This patch wires the monthly BASE solver behind an explicit OFF-by-default flag.
It does not promote the solver to production default and does not relax CH EEX
quote calibration.

## Implemented

- Added `forwards.monthly_curve_solver.enabled: false` in `pfc_shaping/config.yaml`.
- Added `pfc_shaping.pipeline.monthly_curve_authority` as the shared integration
  layer for production and local-test paths.
- Added assembler switches:
  - `monthly_level_authority`
  - `skip_legacy_level_cascade`
  - `skip_legacy_base_smoothing`
- In solver mode, `PFCAssembler` skips legacy cascade and MSFC BASE smoothing.
- In solver mode, final calibration contracts are built from the same monthly
  non-overlap constraint system, so `CAL + Q1` becomes `Q1 + residual`, not
  twelve synthetic monthly hard constraints.
- Synthetic monthly BASE keys are used as level input but excluded from
  `quoted_keys` unless they were genuinely quoted.
- Local-test build/export exposes `--enable-monthly-forward-curve-solver`.
- Local-test export refuses mutating legacy post-processors when the monthly
  solver is enabled.
- Local-test build writes a `*.monthly_curve_manifest.json` sidecar containing
  `monthly_solution_hash` and `active_constraints_hash`.

## Audit-Relevant Invariants

- Flag OFF remains the configured default.
- Solver mode uses the monthly solver as BASE level authority.
- Legacy cascade and MSFC level smoothing are bypassed only in solver mode.
- Final calibrator does not treat synthetic solver months as traded products.
- Production and local-test paths consume the same `MonthlyLevelAuthority`
  object shape.
- No new LT import from `pfc_shaping.ct.*` or deprecated `pfc_shaping.model.*`.

## Verification

```text
pytest tests/test_monthly_forward_curve_integration.py -q
6 passed

pytest tests/test_monthly_forward_curve_*.py tests/test_monthly_curve_lambda_calibration.py -q
52 passed

pytest tests/test_export_local_test_ch_hourly_csv_script.py tests/test_lt_ct_imports.py -q
43 passed, 1 skipped

python -m py_compile \
  pfc_shaping/pipeline/monthly_curve_authority.py \
  pfc_shaping/pipeline/production_phases.py \
  pfc_shaping/lt/model/assembler.py \
  scripts/build_local_test_ch_pfc.py \
  scripts/build_ep2050_multi_scenario_pfc.py \
  scripts/export_local_test_ch_hourly_csv.py
OK
```

Sparse-year proof remains unchanged:

```text
max_abs_constraint_residual=2.132e-13
neighbor_level_leakage_max_abs=1.421e-13
gate_summary={'PASS': 19, 'WARNING': 1}
panel_status=PARTIAL_MONTHLY_PANEL
history_status=PARTIAL_HISTORY_FORWARD
structural_status=UNSUPPORTED
fused_status=PARTIAL_MONTHLY_PANEL
```

## Remaining Limits

- Full byte-identical production artifact comparison is not performed in unit
  tests; the flag is OFF by default and existing export/import tests cover the
  unchanged script path.
- Production/export hash parity is implemented at the shared helper level.
  A full dry-run parity fixture should be added next.
- Phase F numerical gates are still required before any solver promotion.
- Lambda calibration remains fail-closed unless the full grid identifies a
  better configuration under the existing guard.
