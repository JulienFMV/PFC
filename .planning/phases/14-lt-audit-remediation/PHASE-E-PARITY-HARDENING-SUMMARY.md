# Phase E Parity Hardening Summary - 2026-06-19

## Scope

This patch tightens the Phase E monthly-forward-curve-solver integration after
external audit feedback. It does not promote the solver to production default
and does not claim Phase F governance gates are complete.

## Changes

- Added an explicit OFF-path regression test proving the final calibrator keeps
  the legacy monthly contract construction when `skip_legacy_level_cascade` is
  false.
- Added a direct-vs-history monthly authority parity test, covering the two
  construction styles used by production-style current inputs and local export
  history inputs.
- Made production monthly authority use the forward-history snapshot timestamp
  when history is available, rather than wall-clock `utcnow`, so point-in-time
  diagnostics and prod/export parity are not timestamp-dependent.
- Made delivery-month grid construction respect the exclusive end of a
  timezone-aware delivery window.
- Made solver-mode final-calibration quote filtering ignore PEAK/OFFPEAK keys
  case-insensitively.
- Added `forward_snapshot_date` and `solver_config_hash` to the monthly solver
  manifest.

## Verified

Commands run:

```powershell
pytest tests/test_monthly_forward_curve_integration.py -q
$files = Get-ChildItem tests -Filter 'test_monthly_forward_curve_*.py' | ForEach-Object { $_.FullName }; pytest @files tests/test_monthly_curve_lambda_calibration.py -q
pytest tests/test_export_local_test_ch_hourly_csv_script.py tests/test_lt_ct_imports.py -q
python scripts/run_monthly_curve_sparse_year_proof.py --forwards data/eex_forwards_history.parquet --output-dir output/monthly_curve_sparse_year_proof --no-plot
```

Results:

- Monthly integration: `10 passed`
- Monthly solver/calibration family: `56 passed`
- Export/import LT-CT guard: `43 passed, 1 skipped`
- Sparse-year proof:
  - `max_abs_constraint_residual=2.132e-13`
  - `neighbor_level_leakage_max_abs=1.421e-13`
  - `gate_summary={'PASS': 19, 'WARNING': 1}`
  - `panel_status=PARTIAL_MONTHLY_PANEL`
  - `history_status=PARTIAL_HISTORY_FORWARD`
  - `structural_status=UNSUPPORTED`
  - `fused_status=PARTIAL_MONTHLY_PANEL`

## Residual Limits

- This is still Phase E hardening. Phase F numerical governance gates
  (`same_month_rank_consistency`, comparable-block decomposition and
  historical P90/P97.5 thresholds) remain the next required work.
- OFF byte identity is now guarded at the contract-selection layer. A full
  frozen artifact byte-for-byte regression remains useful once a compact
  production/export fixture is finalized.
- The lambda calibration horizon mismatch remains documented: h+2/h+3 sparse
  deployment years still lack direct CH monthly truth.
