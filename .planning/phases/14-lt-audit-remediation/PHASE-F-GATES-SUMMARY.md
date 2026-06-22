# Phase F Gates Summary - 2026-06-19

## Scope

This patch starts Phase F governance for the CH LT monthly BASE solver. It is
not a full Phase F completion and does not promote the solver to production
default.

## Gates Delivered

- `historical_thresholds.csv` builder
  - Added `build_monthly_curve_historical_thresholds(...)` and
    `scripts/build_monthly_curve_historical_thresholds.py`.
  - Uses only EEX history rows point-in-time at or before `run_timestamp`.
  - Emits Phase F threshold schema rows with `PASS` only when
    `n_snapshots >= min_required_n`; otherwise `UNSUPPORTED`.
- `hard_monthly_curve_repricing`
  - Uses active monthly constraint residuals.
  - `PASS <= 1e-8`, otherwise `CRITICAL`.
- `neighbor_level_leakage`
  - Reuses the sparse-proof neighbor +1000 EUR/MWh invariance check.
  - Written into `audit_gates.csv` when the leakage metric is supplied.
- `same_month_rank_consistency`
  - Compares same-month cross-year deviations from comparable parent blocks.
  - Requires historical P90/P97.5 thresholds; otherwise `UNSUPPORTED`.
- `residual_vs_implied_comparable_block`
  - Compares residual-vs-calendar years through month deviations from each
    comparable parent block.
  - Does not compare Apr-Dec residual level directly with a full calendar
    level.
  - Requires historical P90/P97.5 thresholds; otherwise `UNSUPPORTED`.
- `calendar_spread_seasonal_decomposition`
  - Verifies the calendar spread equals the duration-weighted spread implied
    by the two solved monthly calendars.
  - Uses the same actual month-hour weights as the constraint system.
  - This is a hard algebraic identity gate: `PASS <= 1e-8`, otherwise
    `CRITICAL`.
- `monthly_shape_regression_2028_2030`
  - Aggregates targeted 2028-2030 same-month, comparable-block and calendar
    decomposition rows.
  - `CRITICAL` if any targeted row is critical; `UNSUPPORTED` if required
    threshold evidence is missing.
- `point_in_time_data_contract`
  - Added as a governance gate row in the sparse-year proof.
  - Checks that supplied quotes and EEX history rows are available at or before
    `run_timestamp`.
  - Any future or unverifiable input is `CRITICAL`.
- `lambda_calibration_artifact_present`
  - Row builder added for strict promotion/audit packages.
  - Compares active config hash with selected lambda artifact hash.
  - Missing or mismatched hashes are `CRITICAL` when the row is requested.
  - `run_monthly_curve_sparse_year_proof.py` can now emit this row via
    `--require-lambda-artifact`, using either `--selected-config-hash` or
    `--selected-config-artifact`.
- `production_export_path_parity`
  - Row builder added for strict promotion/audit packages.
  - Compares production/export `monthly_solution_hash` and
    `active_constraints_hash`.
  - Missing or mismatched hashes are `CRITICAL` when the row is requested.
  - `run_monthly_curve_sparse_year_proof.py` can now emit this row via
    `--require-path-parity`, using explicit production/export solution and
    constraint hashes.
- Manifest-backed promotion capstone
  - Added `scripts/check_monthly_curve_promotion_from_manifests.py`.
  - Reads production/export monthly manifests and the selected lambda config
    artifact, derives the strict governance gates, appends them to
    `audit_gates.csv`, then runs the same promotion evaluator.
  - This avoids treating manually supplied proof-local hashes as promotion
    evidence.
  - Monthly authority manifests now include `active_config_hash` in the same
    hash scheme as the selected lambda artifact; older manifests can still be
    checked by recomputing the hash from `solver_config`.

All emitted rows follow the Phase F machine-readable schema:

```text
gate_id,status,severity,market,load_type,year,month,product,
parent_block_id,parent_block_type,parent_hours,parent_mean,month_price,
month_deviation_from_parent,metric_name,metric_value,threshold_warning,
threshold_critical,threshold_source,n_history,n_neighbors,evidence,
remediation_hint
```

## Tests

Commands run:

```powershell
pytest tests/test_monthly_forward_curve_audit.py -q
pytest tests/test_monthly_forward_curve_audit.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_curve_lambda_calibration.py -q
python -m compileall pfc_shaping/calibration/monthly_curve_audit.py scripts/run_monthly_curve_sparse_year_proof.py
python scripts/run_monthly_curve_sparse_year_proof.py --forwards data/eex_forwards_history.parquet --output-dir output/monthly_curve_sparse_year_proof --no-plot
$files = Get-ChildItem tests -Filter 'test_monthly_forward_curve_*.py' | ForEach-Object { $_.FullName }; pytest @files tests/test_monthly_curve_lambda_calibration.py tests/test_lt_ct_imports.py -q
python scripts/build_monthly_curve_historical_thresholds.py --forwards data/eex_forwards_history.parquet --output output/monthly_curve_calibration/historical_thresholds.csv --run-timestamp 2026-06-17 --lookback-years 6 --min-required-n 24
python scripts/run_monthly_curve_sparse_year_proof.py --forwards data/eex_forwards_history.parquet --output-dir output/monthly_curve_sparse_year_proof --historical-thresholds output/monthly_curve_calibration/historical_thresholds.csv --no-plot
python scripts/check_monthly_curve_promotion.py --audit-gates output/monthly_curve_sparse_year_proof/audit_gates.csv --historical-thresholds output/monthly_curve_sparse_year_proof/historical_thresholds.csv --manifest output/monthly_curve_sparse_year_proof/manifest.json --run-timestamp 2026-06-17
python scripts/check_monthly_curve_promotion.py --audit-gates output/monthly_curve_sparse_year_proof/audit_gates.csv --historical-thresholds output/monthly_curve_sparse_year_proof/historical_thresholds.csv --manifest output/monthly_curve_sparse_year_proof/manifest.json --run-timestamp 2026-06-17 --require-governance-gates lambda_calibration_artifact_present,production_export_path_parity
python scripts/run_monthly_curve_sparse_year_proof.py --forwards data/eex_forwards_history.parquet --output-dir output/monthly_curve_sparse_year_proof_strict_missing --historical-thresholds output/monthly_curve_calibration/historical_thresholds.csv --require-lambda-artifact --require-path-parity --allow-critical-gates --no-plot
python scripts/run_monthly_curve_sparse_year_proof.py --forwards data/eex_forwards_history.parquet --output-dir output/monthly_curve_sparse_year_proof_strict_pass --historical-thresholds output/monthly_curve_calibration/historical_thresholds.csv --require-lambda-artifact --selected-config-hash <active_config_hash> --require-path-parity --production-monthly-solution-hash <hash> --export-monthly-solution-hash <hash> --production-active-constraints-hash <hash> --export-active-constraints-hash <hash> --no-plot
pytest tests/test_run_monthly_curve_sparse_year_proof_script.py -q
$files = Get-ChildItem tests -Filter 'test_monthly_forward_curve_*.py' | ForEach-Object { $_.FullName }; pytest @files tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_lt_ct_imports.py -q
pytest tests/test_check_monthly_curve_promotion_from_manifests.py -q
$files = Get-ChildItem tests -Filter 'test_monthly_forward_curve_*.py' | ForEach-Object { $_.FullName }; pytest @files tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_lt_ct_imports.py -q
python scripts/check_monthly_curve_promotion_from_manifests.py --audit-gates output/monthly_curve_sparse_year_proof/audit_gates.csv --historical-thresholds output/monthly_curve_sparse_year_proof/historical_thresholds.csv --manifest output/monthly_curve_sparse_year_proof/manifest.json --production-manifest output/monthly_curve_sparse_year_proof/manifest.json --export-manifest output/monthly_curve_sparse_year_proof/manifest.json --selected-config-artifact output/monthly_curve_sparse_year_proof/selected_config_from_manifest.json --run-timestamp 2026-06-17 --augmented-audit-gates output/monthly_curve_sparse_year_proof/audit_gates_manifest_capstone.csv
```

Results:

- Monthly audit unit tests: `6 passed`
- Audit/solver/lambda targeted tests: `28 passed`
- Monthly family plus LT/CT import guard: `76 passed, 1 skipped`
- After threshold-builder patch:
  - Monthly audit unit tests: `9 passed`
  - Monthly family plus LT/CT import guard: `79 passed, 1 skipped`
- Historical threshold generation:
  - `rows=26`
  - `same_month_rank_consistency`: `13 PASS`
  - `residual_vs_implied_comparable_block`: `13 UNSUPPORTED`
- Sparse proof:
  - `max_abs_constraint_residual=2.132e-13`
  - `neighbor_level_leakage_max_abs=1.421e-13`
  - `gate_summary={'UNSUPPORTED': 22, 'PASS': 9}`
  - `panel_status=PARTIAL_MONTHLY_PANEL`
  - `history_status=PARTIAL_HISTORY_FORWARD`
  - `structural_status=UNSUPPORTED`
  - `fused_status=PARTIAL_MONTHLY_PANEL`
- Sparse proof with historical thresholds:
  - `max_abs_constraint_residual=2.132e-13`
  - `neighbor_level_leakage_max_abs=1.421e-13`
  - before point-in-time governance row:
    `gate_summary={'PASS': 21, 'UNSUPPORTED': 10}`
  - after point-in-time governance row:
    `gate_summary={'PASS': 22, 'UNSUPPORTED': 10}`
  - after calendar-spread decomposition row:
    `gate_summary={'PASS': 23, 'UNSUPPORTED': 10}`
  - `same_month_rank_consistency`: `12 PASS`
  - `residual_vs_implied_comparable_block`: `9 UNSUPPORTED`
- Promotion checker:
  - standard sparse evidence: `PROMOTION_EVIDENCE_PASS`
  - strict governance mode: `BLOCKED` with two expected blockers until lambda
    artifact and prod/export parity hashes are supplied
- Strict sparse-proof governance emission:
  - missing required governance proofs:
    `gate_summary={'PASS': 23, 'UNSUPPORTED': 10, 'CRITICAL': 2}`
  - supplied matching config/parity hashes:
    `gate_summary={'PASS': 25, 'UNSUPPORTED': 10}`
  - checker with required governance gates passes only for the supplied-hash
    package.
- Sparse proof governance wiring unit test: `2 passed`
- Monthly family plus LT/CT import guard after strict wiring:
  `92 passed, 1 skipped`
- Manifest-backed capstone unit tests: `2 passed`
- Monthly family plus LT/CT import guard after manifest-backed capstone:
  `94 passed, 1 skipped`
- Manifest-backed local proof check:
  `PROMOTION_EVIDENCE_PASS`, `audit_gate_status_counts={'PASS': 25, 'UNSUPPORTED': 10}`

The `UNSUPPORTED` sparse-proof shape gates are intentional at this stage:
historical P90/P97.5 threshold artifacts are not yet calibrated for promotion.
The gates fail closed instead of reporting `PASS` without sufficient evidence.

After adding threshold generation, the same-month rank gate is active on the
current local history. The residual-vs-calendar comparable-block gate remains
`UNSUPPORTED` because the local CH history does not contain enough monthly
truth for residual Apr-Dec versus full-calendar comparisons.

## Remaining Limits

- Historical threshold generation now exists, but residual/calendar thresholds
  remain unsupported on the local parquet due to insufficient observable
  monthly market evidence.
- The same-month gate currently uses absolute comparable-parent shape delta
  against calibrated thresholds. Full conditional rank/z-score logic remains a
  later Phase F extension.
- `historical_quantile_shape_outlier` remains to be implemented as a Phase F
  gate.
- `lambda_calibration_artifact_present` and `production_export_path_parity`
  are emitted by the sparse proof when requested, but full candidate approval
  must still pass hashes from the selected lambda artifact and the real
  prod/export path manifests. The manifest-backed capstone script is now the
  preferred promotion entry point because it reads those manifests directly.
- Power BI sidecar integration for the new `audit_gates.csv` rows remains to be
  wired.
- Production approval remains blocked while required near-horizon or otherwise
  historically calibrable gates are `UNSUPPORTED`.
- Far-horizon `UNSUPPORTED` should not be treated as an automatic permanent
  blocker if point-in-time threshold calibration proves insufficient monthly
  market evidence. It must be documented as residual model risk and must not
  hide any `CRITICAL` on hard gates or known-bad fixtures.
