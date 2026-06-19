# Phase F Gates Summary - 2026-06-19

## Scope

This patch starts Phase F governance for the CH LT monthly BASE solver. It is
not a full Phase F completion and does not promote the solver to production
default.

## Gates Delivered

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
- `monthly_shape_regression_2028_2030`
  - Aggregates targeted 2028-2030 same-month and comparable-block rows.
  - `CRITICAL` if any targeted row is critical; `UNSUPPORTED` if required
    threshold evidence is missing.

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
```

Results:

- Monthly audit unit tests: `6 passed`
- Audit/solver/lambda targeted tests: `28 passed`
- Monthly family plus LT/CT import guard: `76 passed, 1 skipped`
- Sparse proof:
  - `max_abs_constraint_residual=2.132e-13`
  - `neighbor_level_leakage_max_abs=1.421e-13`
  - `gate_summary={'UNSUPPORTED': 22, 'PASS': 9}`
  - `panel_status=PARTIAL_MONTHLY_PANEL`
  - `history_status=PARTIAL_HISTORY_FORWARD`
  - `structural_status=UNSUPPORTED`
  - `fused_status=PARTIAL_MONTHLY_PANEL`

The `UNSUPPORTED` sparse-proof shape gates are intentional at this stage:
historical P90/P97.5 threshold artifacts are not yet calibrated for promotion.
The gates fail closed instead of reporting `PASS` without sufficient evidence.

## Remaining Limits

- Historical threshold generation and `historical_thresholds.csv` production
  are still missing.
- The same-month gate currently uses absolute comparable-parent shape delta
  against calibrated thresholds. Full conditional rank/z-score logic remains a
  later Phase F extension.
- `calendar_spread_seasonal_decomposition`,
  `historical_quantile_shape_outlier`,
  `lambda_calibration_artifact_present`, `point_in_time_data_contract`, and
  `production_export_path_parity` remain to be implemented as Phase F gates.
- Power BI sidecar integration for the new `audit_gates.csv` rows remains to be
  wired.
- Production approval remains blocked while required near-horizon or otherwise
  historically calibrable gates are `UNSUPPORTED`.
- Far-horizon `UNSUPPORTED` should not be treated as an automatic permanent
  blocker if point-in-time threshold calibration proves insufficient monthly
  market evidence. It must be documented as residual model risk and must not
  hide any `CRITICAL` on hard gates or known-bad fixtures.
