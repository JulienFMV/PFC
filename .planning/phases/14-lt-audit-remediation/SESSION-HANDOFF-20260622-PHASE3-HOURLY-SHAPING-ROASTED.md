# Session Handoff - 2026-06-22 - Phase 3 Hourly Shaping Roasted

## Scope

Phase 3 delivered-hourly-shape remediation after Lambda evidence hardening.

Goal: improve and diagnose the delivered CH HFC graph, not the isolated monthly
BASE solver.

No CT code was touched. No `powerbi/data/*` files were written. Diagnostic CSVs
and sidecars were written only under `output/phase3_*`.

## Expert Roasts

Three read-only expert roasts were run before and during coding.

PEAK roast:

- Monthly BASE solver is not the PEAK fix layer.
- `calibrate_hourly_to_eex_base_peak(...)` already exists but was disabled by
  default.
- Correct layer is final delivered-hourly projection before CSV write.
- PEAK fix is necessary but not sufficient for promotion.

Quantile/width roast:

- Inverted Phase 2 quantiles were an export bridge bug.
- Source fan chart has ordered `structural_scenario_low/high` columns.
- Export and Power BI fallbacks treated `slow/fast` labels as P10/P90.
- Ordered bridge repair fixes inversion but true width remains too small.

Cross-year Q4 roast:

- Q4 defect belongs to monthly solver prior/objective diagnostics, not
  assembler.
- Current Q4 critical rows partly use full-CAL parent space where comparable
  Apr-Dec parent space is needed.
- Do not apply naive `CAL28 > CAL29 => every 2028 month > 2029 month`.
- Next model work should add comparable-block allocation diagnostics/penalties
  using CAL, Q, residual Apr-Dec, parent means and deviations.

## Code Changes

Hourly export:

- `scripts/export_local_test_ch_hourly_csv.py`
  - Final `--enable-eex-peak-calibration` projection now runs immediately
    before CSV write, after all mutating hourly options including seam
    smoothing.
  - `calibrate_hourly_to_eex_base_peak(...)` now solves final offpeak level
    from target BASE energy and target PEAK energy, instead of preserving a
    possibly-damaged current BASE level.
  - `to_hourly_csv_frame(...)` maps ordered
    `structural_scenario_low/central/high/spread` fan columns to structural
    compatibility columns before legacy `structural_p10/p50/p90`.
  - Final structural fallback computes ordered row-wise low/median/high from
    scenario prices instead of assigning slow/central/fast as ordered quantiles.

Power BI export:

- `scripts/build_powerbi_exports.py`
  - Structural fallback now computes ordered row-wise low/median/high from
    scenario prices when structural columns are absent.

Tests:

- `tests/test_export_local_test_ch_hourly_csv_script.py`
  - Added ordered fan bracket export regression.
  - Added final-mutator PEAK regression: a monkeypatched final seam step
    damages PEAK, and final CSV still matches quoted BASE and PEAK when
    `--enable-eex-peak-calibration` is active.
- `tests/test_build_powerbi_exports_script.py`
  - Added Power BI loader fallback regression for crossed scenario labels.

## Diagnostic Runs

Source fan chart reused to isolate export-layer effects:

`output/phase2_20260622_solver_probe/phase2_20260622_solver_probe_structural_fan_chart.parquet`

Bridge-only regeneration:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'
python scripts/export_local_test_ch_hourly_csv.py `
  --skip-build `
  --valuation-date 2026-06-22 `
  --local-start-date 2026-06-22 `
  --local-end-date 2032-12-31 `
  --prefix phase3_20260622_bridge_probe `
  --fan-chart-output output/phase2_20260622_solver_probe/phase2_20260622_solver_probe_structural_fan_chart.parquet `
  --output output/phase3_20260622_bridge_probe/ch_hfc_hourly_20260622_20321231_phase3_bridge.csv `
  --report output/phase3_20260622_bridge_probe/CH-HFC-HOURLY-CSV-20260622-20321231-PHASE3-BRIDGE.md `
  --skip-powerbi-refresh
```

Result:

- rows: `57241`
- shape score: `4.75/10`
- quantile_order: `1`
- bad_quantile_rows: `0`
- negative_width_rows: `0`
- width mean: `0.511877`
- width p95: `1.330235`
- max BASE residual: `0.000000`
- max PEAK residual: `17.497926`

PEAK-calibrated regeneration:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'
python scripts/export_local_test_ch_hourly_csv.py `
  --skip-build `
  --valuation-date 2026-06-22 `
  --local-start-date 2026-06-22 `
  --local-end-date 2032-12-31 `
  --prefix phase3_20260622_peakcal_probe_v2 `
  --fan-chart-output output/phase2_20260622_solver_probe/phase2_20260622_solver_probe_structural_fan_chart.parquet `
  --output output/phase3_20260622_peakcal_probe_v2/ch_hfc_hourly_20260622_20321231_phase3_peakcal_v2.csv `
  --report output/phase3_20260622_peakcal_probe_v2/CH-HFC-HOURLY-CSV-20260622-20321231-PHASE3-PEAKCAL-V2.md `
  --enable-eex-peak-calibration `
  --skip-powerbi-refresh
```

Result:

- rows: `57241`
- shape score: `6.75/10`
- quantile_order: `1`
- bad_quantile_rows: `0`
- negative_width_rows: `0`
- width mean: `0.511877`
- width p95: `1.330235`
- max BASE residual: `0.000000`
- max PEAK residual: `0.000000`

Power BI strict export on PEAK v2:

```powershell
python scripts/build_powerbi_exports.py `
  --csv output/phase3_20260622_peakcal_probe_v2/ch_hfc_hourly_20260622_20321231_phase3_peakcal_v2.csv `
  --forwards data/eex_forwards_history.parquet `
  --spot data/epex_hourly.parquet `
  --output-dir output/phase3_20260622_peakcal_probe_v2/powerbi_strict
```

Result: blocked, as desired:

```text
Power BI export blocked by quality gates. Use --allow-failed-gates only for explicitly diagnostic sidecars.
- shape_score_10=6.75 < 8.50
- monthly_split_critical_flags=1
- cross_year_month_shape_critical_flags=3
```

## Tests

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_export_local_test_ch_hourly_csv_script.py -q
```

Result: `28 passed in 17.05s`.

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_build_powerbi_exports_script.py tests/test_export_local_test_ch_hourly_csv_script.py -q
```

Result: `70 passed in 18.46s`.

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_monthly_forward_curve_solver.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_audit_ch_pfc_hourly_shape_script.py tests/test_build_powerbi_exports_script.py -q
```

Result: `26 passed in 3.75s`.

Broad guardrail:

```powershell
$env:PYTHONPATH='.'; $files = Get-ChildItem tests -Filter 'test_monthly_forward_curve_*.py' | ForEach-Object { $_.FullName }; python -m pytest $files tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_long_term_branch.py tests/test_lt_ct_imports.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_pfc_hourly_shape_script.py -q
```

Result: `165 passed, 1 skipped, 1 warning in 93.15s`.

Warning:

- `RuntimeWarning: All-NaN slice encountered` in insufficient-history prior
  diagnostics. Existing warning, not a failing gate.

## Current Truth State

Improved:

- PEAK residual can be made exact and is now protected as a final export
  invariant when `--enable-eex-peak-calibration` is active.
- Quantile inversion / negative width bridge bug is fixed.
- Power BI fallback no longer fabricates inverted structural P10/P90 from
  crossed scenario labels.

Still not promotion-ready:

- Best current diagnostic score is `6.75/10`, below `8.50`.
- Structural width is still too small: mean `0.511877`, p95 `1.330235`, versus
  audit band mean `6-11`, p95 `18-32`.
- `monthly_split_critical_flags=1`.
- `cross_year_month_shape_critical_flags=3`.
- Power BI strict export remains blocked without `--allow-failed-gates`.

## Next Phase

Phase 4 should not touch the assembler or CT.

Recommended order:

1. Curate and push the reproducible code/test patch; do not commit generated
   data, Power BI layout churn or heavy local desk files accidentally.
2. Add comparable-block cross-year diagnostics/penalty in the monthly solver:
   residual Apr-Dec vs comparable Apr-Dec, parent means and deviations, not full
   CAL shortcuts.
3. Add a real structural width calibration, or explicitly define a governed
   structural fan-width model. The bridge is now honest, but the fan is still
   too narrow.
4. Regenerate a fresh candidate with solver ON and final PEAK calibration, then
   rerun strict Power BI export.
