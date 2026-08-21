# Session Handoff - 2026-06-23 - Candidate Audit Blockers

## Scope

Continued Phase 14 from the solver/export governance hardening handoff. Goal
was to generate a fresh local-test CH LT candidate with monthly solver ON and
final PEAK calibration ON, then run delivered-product and strict Power BI
gates without promoting production.

## Code Changes

- `pfc_shaping/pipeline/monthly_curve_authority.py`
  - Added `delivery_months_for_local_window(...)`.
  - Purpose: derive inclusive solver months from the intended local delivered
    artifact window.
- `scripts/build_local_test_ch_pfc.py`
  - Added optional `--monthly-solver-delivery-local-start-date` and
    `--monthly-solver-delivery-local-end-date`.
  - Uses the local-window helper when both are supplied; otherwise preserves
    existing `start_date`/`horizon_days` behavior.
- `scripts/export_local_test_ch_hourly_csv.py`
  - Passes the requested local CSV start/end dates to the local build script
    whenever `--enable-monthly-forward-curve-solver` is active.
- `tests/test_monthly_forward_curve_integration.py`
  - Added coverage proving the rounded UTC build horizon includes `2031-01`
    while the intended artifact window ends at `2030-12`.
- `tests/test_export_local_test_ch_hourly_csv_script.py`
  - Added CLI wiring coverage proving the exporter passes the local delivery
    window to the build script under monthly solver mode.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added D-20260623-10 for the local-export solver-month contract.

No `pfc_shaping/ct/*` files were touched.

## Read-Only Roaster Results

Two read-only subagents reviewed the first generation blocker before code
edits.

- Erdos: root cause is `_utc_build_window(...)` ceiling the local CSV window
  into whole UTC `horizon_days`, causing the monthly solver grid to include
  `2031-01`. Recommended passing local artifact months to the solver and not
  relaxing `monthly_forward_curve.py`.
- Carver: governance recommendation was to fail/document unless the candidate
  horizon is product-complete. Acceptable path is to align solver quote
  selection with the delivered artifact window, while keeping partially
  overlapping products fail-closed.

## Commands And Results

Initial candidate attempt, strict solver tolerance:

```powershell
python scripts/export_local_test_ch_hourly_csv.py `
  --enable-monthly-forward-curve-solver `
  --enable-eex-peak-calibration `
  --monthly-solver-constraint-tolerance 0.000000001 `
  --required-forward-date 2026-06-17 `
  --output output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --report output/phase14/20260623_solver_peak_candidate/export_report.md `
  --fan-chart-output output/phase14/20260623_solver_peak_candidate/ch_hfc_fan_chart_solver_peak_20260613_20301231.csv `
  --skip-powerbi-refresh
```

Result after local-window patch: blocked by quote consistency tolerance:

```text
ValueError: inconsistent quoted product 2026-Q3: target=95.25,
implied=95.2469565217, diff=-0.00304347826086
```

This diff is below the local export CLI default tolerance `0.01`, so the
candidate was regenerated using the existing CLI default.

Tests:

```powershell
python -m pytest `
  tests/test_export_local_test_ch_hourly_csv_script.py::test_monthly_solver_build_receives_intended_local_delivery_window `
  tests/test_monthly_forward_curve_integration.py::test_delivery_months_for_local_window_uses_intended_artifact_months -q
```

Result:

```text
2 passed in 2.52s
```

```powershell
python -m pytest `
  tests/test_export_local_test_ch_hourly_csv_script.py `
  tests/test_monthly_forward_curve_integration.py `
  tests/test_monthly_forward_curve_constraints.py -q
```

Result:

```text
57 passed in 21.72s
```

Candidate generation:

```powershell
python scripts/export_local_test_ch_hourly_csv.py `
  --enable-monthly-forward-curve-solver `
  --enable-eex-peak-calibration `
  --required-forward-date 2026-06-17 `
  --output output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --report output/phase14/20260623_solver_peak_candidate/export_report.md `
  --fan-chart-output output/phase14/20260623_solver_peak_candidate/ch_hfc_fan_chart_solver_peak_20260613_20301231.csv `
  --skip-powerbi-refresh
```

Result:

```text
[hourly-csv] rows=39913
[hourly-csv] output -> output\phase14\20260623_solver_peak_candidate\ch_hfc_hourly_solver_peak_20260613_20301231.csv
[hourly-csv] report -> output/phase14/20260623_solver_peak_candidate/export_report.md
```

Delivered-product audit:

```powershell
python scripts/audit_ch_product_normalization.py `
  --csv output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --forwards data/eex_forwards_history.parquet `
  --required-forward-date 2026-06-17 `
  --price-column price_weighted_mean_eur_mwh `
  --output-csv output/phase14/20260623_solver_peak_candidate/delivered_product_normalization_gates.csv `
  --summary-json output/phase14/20260623_solver_peak_candidate/delivered_product_normalization_summary.json
```

Result: exit 1, fail-closed.

```text
all_gates_pass=false
critical_count=9
unsupported_count=9
supported_hard_gate_max_abs_residual_eur_mwh=0.10292307435898351
status_counts={PASS: 70, UNSUPPORTED: 9, CRITICAL: 9}
```

Strict Power BI export:

```powershell
python scripts/build_powerbi_exports.py `
  --csv output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --forwards data/eex_forwards_history.parquet `
  --spot data/epex_hourly.parquet `
  --output-dir output/phase14/20260623_solver_peak_candidate/powerbi_strict
```

Result: exit 1, strict block.

```text
Power BI export blocked by quality gates.
- shape_score_10=6.75 < 8.50
- monthly_split_critical_flags=1
```

Additional diagnostics:

```powershell
python scripts/audit_ch_hfc_seasonal_coherence.py `
  --csv output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --forwards data/eex_forwards_history.parquet `
  --price-column price_weighted_mean_eur_mwh `
  --report output/phase14/20260623_solver_peak_candidate/seasonal_coherence_report.md `
  --monthly-output output/phase14/20260623_solver_peak_candidate/seasonal_monthly.csv `
  --hour-month-output output/phase14/20260623_solver_peak_candidate/seasonal_hour_month.csv `
  --monthly-split-output output/phase14/20260623_solver_peak_candidate/seasonal_monthly_split.csv `
  --monthly-path-output output/phase14/20260623_solver_peak_candidate/seasonal_monthly_path.csv `
  --cross-year-output output/phase14/20260623_solver_peak_candidate/seasonal_cross_year.csv `
  --calendar-output output/phase14/20260623_solver_peak_candidate/seasonal_calendar.csv
```

Result:

```text
[seasonal-audit] critical=1 warning=3
```

```powershell
python scripts/audit_ch_pfc_hourly_shape.py `
  --csv output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --forwards data/eex_forwards_history.parquet `
  --report output/phase14/20260623_solver_peak_candidate/hourly_shape_report.md
```

Result:

```text
[shape-audit] score=6.75/10
```

## Candidate Artifact Paths

Local ignored artifacts:

- `output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv`
- `output/phase14/20260623_solver_peak_candidate/export_report.md`
- `output/phase14/20260623_solver_peak_candidate/ch_hfc_fan_chart_solver_peak_20260613_20301231.csv`
  - Note: despite `.csv` suffix, this is the existing build script's parquet
    fan-chart payload because `_write_parquet(...)` is used.
- `output/phase14/20260623_solver_peak_candidate/ch_hfc_fan_chart_solver_peak_20260613_20301231.monthly_curve_manifest.json`
- `output/phase14/20260623_solver_peak_candidate/delivered_product_normalization_gates.csv`
- `output/phase14/20260623_solver_peak_candidate/delivered_product_normalization_summary.json`
- `output/phase14/20260623_solver_peak_candidate/seasonal_coherence_report.md`
- `output/phase14/20260623_solver_peak_candidate/hourly_shape_report.md`
- `output/phase14/20260623_solver_peak_candidate/seasonal_*.csv`

The failed strict Power BI run did not leave files under
`output/phase14/20260623_solver_peak_candidate/powerbi_strict`.

Generated tracked Phase 13 build markdown was restored before handoff:

- `.planning/phases/13-lt-electrification-scenario-shape/LOCAL_TEST_CH_PFC_20260613_20301231-BUILD.md`

## Hashes And Config Values

Forward snapshot:

- `required_forward_date`: `2026-06-17`
- `forwards_sha256`: `c4bedaeb4cf7a04324bcf667be35ef9f92eeb2118c431109220076b114f9a3c5`

Delivered product audit:

- `input_csv_sha256`: `4d79737ae985a227e5f81498a512b54259a7090cc4a77cbb9abe6cfb7e3c32fe`
- `audit_script_sha256`: `8b20f0843335eea7c8b2c829bfae920d0f43f9cb0b1fcac98d3f12f441a50059`
- `hard_tolerance_eur_mwh`: `1e-06`

Monthly solver manifest:

- `monthly_level_authority`: `solver`
- `active_config_hash`: `9d3bc8f93d7099fa0af659d4e98b092debc8d4ca507a50d87362040c9a2c05f1`
- `solver_config_hash`: `0ed2c14d74676911aebd5bf1ae5205cff8f770fb1fb5a6c52ccd826f52a84b19`
- `active_constraints_hash`: `efb01468d31e43f9c6cd66102ad5c573a9aacc8913e26b1a20139358174144cf`
- `monthly_solution_hash`: `88cc7e292ebd090835930912b1ed925e6dc006887bf7878edba0a1ae01a63c5f`
- `constraint_tolerance`: `0.01`
- `lambda_smooth_month`: `1.0`
- `lambda_smooth_yoy`: `0.25`
- `lambda_shape`: `1.0`
- `neighbor_shrinkage`: `0.5`
- `allow_template_structural_fallback`: `true`
- `structural_amplitude_eur_mwh`: `110.0`
- `panel_weight`: `1.0`
- `history_weight`: `0.5`
- `structural_weight`: `1.0`
- `fused_status`: `PARTIAL_MONTHLY_PANEL`
- `history_status`: `PARTIAL_HISTORY_FORWARD`
- `panel_status`: `PARTIAL_MONTHLY_PANEL`
- `structural_status`: `STRUCTURAL_TEMPLATE`
- `solver_kkt.max_abs_constraint_residual`: `1.4210854715202004e-14`
- `solver_kkt.stationarity_residual`: `3.0698023235065095e-13`
- `solver_kkt.active_constraint_rank`: `15`
- `solver_kkt.nullspace_dimension`: `40`

## Gate Findings

Delivered-product audit:

- `quote_aware_base_bucket_repricing`: all `14 PASS`.
- `quote_aware_peak_bucket_repricing`: all `14 PASS`.
- Direct hard quoted-product gates fail on redundant parent products:
  - BASE `2026-Q3`: `-0.003043478260863708`
  - BASE `2026-Q4`: `0.003811679492983444`
  - BASE `2027`: `0.0026426940639368013`
  - PEAK `2026-Q3`: `-0.0013636363636351234`
  - PEAK `2026-Q4`: `-0.10292307692310487`
  - PEAK `2027`: `-0.0071428571428668874`
  - Implied OFFPEAK also fails for those parent products.
- Unsupported products:
  - `2026-06` is partial because the delivered window starts
    `2026-06-13`.
  - `2031` and `2032` are outside the delivered artifact window.

Snapshot consistency check confirmed those direct-parent residuals match
inconsistencies in the EEX snapshot between finer quotes and parent quotes.
The candidate cannot satisfy both at `1e-6` without a quote hierarchy policy or
different source data.

Seasonal/Power BI gates:

- `monthly_split_critical_flags=1`
  - `2027 Q2 BASE` vs `DE`
  - `split_corr=0.093065`
  - `amplitude_ratio=1.788180`
  - reason: `CH monthly split is weakly aligned with neighbor shape`
- Cross-year warnings:
  - `2028 -> 2029`, month `5`, near-cloned same-month values.
  - `2028 -> 2029`, Apr-Dec seasonal slope delta `-8.785457`.
- Hourly shape score:
  - `score_10=6.75`
  - `structural_width_mean_eur_mwh=0.539105`
  - `structural_width_p95_eur_mwh=1.453780`
  - `ramp_abs_p99_eur_mwh=24.189157`
  - `boundary_jump_abs_p95_eur_mwh=18.903215`
  - quantile ordering passed; finite checks passed; no negative weighted
    prices.

## Risks / Next Work

- This is still production NO-GO. The fresh candidate exists but does not pass
  delivered-product or strict Power BI gates.
- The product-normalization blocker is partly a governance/spec question:
  direct parent quotes in the snapshot conflict with finer quotes, while
  quote-aware non-overlapping buckets pass. Before changing audit semantics,
  run read-only roasters specifically on whether direct parent rows should be
  `CRITICAL`, `UNSUPPORTED/QUOTE_CONFLICT`, or separated from promotion hard
  gates when the snapshot is internally inconsistent.
- Do not solve this by patching individual months or by relaxing the low-level
  partial-product check.
- If the desk wants `2031/2032` in promotion scope, regenerate a declared
  full-horizon candidate and audit that exact horizon.
- Investigate `2027 Q2 BASE` monthly split in model/prior/objective space.
  The current solver KKT is numerically clean, so the issue is likely objective
  weighting/prior evidence, not a failed solve.
- Structural width remains too narrow for the Power BI threshold. Any change
  must preserve quantile ordering and manifest/hash traceability.
