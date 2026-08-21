# Session Handoff - Phase 14 Monthly Reform - 2026-06-22 Phase 2 Diagnostic

## Session Scope

Branch: `fix/lt-audit-remediation`

Goal: generate a fresh diagnostic CH LT candidate with the monthly solver ON
and prove or refute whether the delivered monthly graph is corrected.

Result: **not green for promotion**. The monthly BASE solver layer is coherent
for the candidate config, but the delivered local-test curve still fails
quality gates. The layer comparison shows `solver == B == price_shape == CSV`
for BASE monthly means within numerical tolerance, so the remaining red flags
are not caused by monthly mean drift after the solver. They are delivered-curve
quality/audit failures, especially PEAK residuals and cross-year seasonal
allocation.

## Read-Only Expert Challenges

Three read-only agents challenged Phase 2 before/while running diagnostics:

- quant/prior:
  - sparse proof alone is not promotion proof;
  - `active_config_hash` omits material prior knobs;
  - template structural fallback is wired and must not be treated as calibrated
    market evidence;
  - point-in-time neighbor path can mask timestamp leakage if neighbor latest
    snapshots differ.
- pipeline/assembler:
  - solver mode mostly survives through assembler;
  - standard parquet exposes `B` and `price_shape`, not `price_raw`;
  - final calibration uses original quoted keys, not synthetic monthly keys;
  - local hourly CSV cannot by itself prove `B` preservation.
- audit/Power BI:
  - current Power BI sidecars are legacy diagnostics, not Phase F proof
    sidecars;
  - `--allow-failed-gates` must remain diagnostic-only;
  - promotion requires manifest-backed proof artifacts and selected lambda
    artifact checks.

## Commands And Results

Initial state commands:

```powershell
git status --short
git diff --stat
git ls-files -o --exclude-standard
git diff --name-only --diff-filter=A
```

Important observed state at start:

- branch was `fix/lt-audit-remediation`;
- worktree was already dirty with code, data, output, Power BI and test
  changes;
- Phase1 handoff files were present as staged/intended additions in the first
  status view;
- no existing changes were reverted.

Fresh candidate build, strict tolerance first:

```powershell
$env:PYTHONPATH='.'
python scripts/build_local_test_ch_pfc.py `
  --start-date '2026-06-21 22:00:00' `
  --horizon-days 2386 `
  --output-dir output/phase2_20260622_solver_probe `
  --output-prefix phase2_20260622_solver_probe `
  --expanded-output output/phase2_20260622_solver_probe/scenario_expanded.parquet `
  --features-output output/phase2_20260622_solver_probe/hpfc_scenario_features.parquet `
  --fan-chart-output output/phase2_20260622_solver_probe/phase2_20260622_solver_probe_structural_fan_chart.parquet `
  --summary output/phase2_20260622_solver_probe/LOCAL-TEST-CH-PFC-BUILD.md `
  --governance-report output/phase2_20260622_solver_probe/LOCAL-TEST-GOVERNANCE-GATE.md `
  --enable-monthly-forward-curve-solver `
  --monthly-solver-constraint-tolerance 0.000000001 `
  --monthly-solver-lambda-smooth-month 1.0 `
  --monthly-solver-lambda-smooth-yoy 0.25 `
  --monthly-solver-lambda-shape 1.0 `
  --monthly-solver-neighbor-shrinkage 0.5 `
  --monthly-solver-structural-amplitude 110.0 `
  --monthly-solver-allow-template-structural-fallback
```

Result: failed fast on quote consistency:

```text
ValueError: inconsistent quoted product 2026-Q3: target=98.35,
implied=98.3485869565, diff=-0.00141304347825
```

Diagnostic tolerance candidate build:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'
python scripts/build_local_test_ch_pfc.py `
  --start-date '2026-06-21 22:00:00' `
  --horizon-days 2386 `
  --output-dir output/phase2_20260622_solver_probe `
  --output-prefix phase2_20260622_solver_probe `
  --expanded-output output/phase2_20260622_solver_probe/scenario_expanded.parquet `
  --features-output output/phase2_20260622_solver_probe/hpfc_scenario_features.parquet `
  --fan-chart-output output/phase2_20260622_solver_probe/phase2_20260622_solver_probe_structural_fan_chart.parquet `
  --summary output/phase2_20260622_solver_probe/LOCAL-TEST-CH-PFC-BUILD.md `
  --governance-report output/phase2_20260622_solver_probe/LOCAL-TEST-GOVERNANCE-GATE.md `
  --enable-monthly-forward-curve-solver `
  --monthly-solver-constraint-tolerance 0.01 `
  --monthly-solver-lambda-smooth-month 1.0 `
  --monthly-solver-lambda-smooth-yoy 0.25 `
  --monthly-solver-lambda-shape 1.0 `
  --monthly-solver-neighbor-shrinkage 0.5 `
  --monthly-solver-structural-amplitude 110.0 `
  --monthly-solver-allow-template-structural-fallback
```

Result:

```text
[local-test-pfc] slow -> output\phase2_20260622_solver_probe\phase2_20260622_solver_probe_slow.parquet
[local-test-pfc] central -> output\phase2_20260622_solver_probe\phase2_20260622_solver_probe_central.parquet
[local-test-pfc] fast -> output\phase2_20260622_solver_probe\phase2_20260622_solver_probe_fast.parquet
[local-test-pfc] fan chart -> output\phase2_20260622_solver_probe\phase2_20260622_solver_probe_structural_fan_chart.parquet
[local-test-pfc] monthly curve manifest -> output\phase2_20260622_solver_probe\phase2_20260622_solver_probe_structural_fan_chart.monthly_curve_manifest.json
[local-test-pfc] weighted_mean=78.49 scenario_spread_mean=0.5117
```

Hourly export from the fresh fan chart, without rebuild:

```powershell
python scripts/export_local_test_ch_hourly_csv.py `
  --skip-build `
  --valuation-date 2026-06-22 `
  --local-start-date 2026-06-22 `
  --local-end-date 2032-12-31 `
  --prefix phase2_20260622_solver_probe `
  --fan-chart-output output/phase2_20260622_solver_probe/phase2_20260622_solver_probe_structural_fan_chart.parquet `
  --output output/phase2_20260622_solver_probe/ch_hfc_hourly_20260622_20321231_phase2.csv `
  --report output/phase2_20260622_solver_probe/CH-HFC-HOURLY-CSV-20260622-20321231-PHASE2.md `
  --skip-powerbi-refresh
```

Result:

```text
[hourly-csv] rows=57241
```

Power BI sidecar build without override:

```powershell
python scripts/build_powerbi_exports.py `
  --csv output/phase2_20260622_solver_probe/ch_hfc_hourly_20260622_20321231_phase2.csv `
  --forwards data/eex_forwards_history.parquet `
  --spot data/epex_hourly.parquet `
  --output-dir output/phase2_20260622_solver_probe/powerbi
```

Result: blocked, as desired:

```text
Power BI export blocked by quality gates. Use --allow-failed-gates only for explicitly diagnostic sidecars.
- shape_score_10=3.25 < 8.50
- max_eex_peak_error_eur_mwh=17.497926 > 0.010000
- monthly_split_critical_flags=1
- cross_year_month_shape_critical_flags=3
```

Diagnostic-only Power BI sidecars:

```powershell
python scripts/build_powerbi_exports.py `
  --csv output/phase2_20260622_solver_probe/ch_hfc_hourly_20260622_20321231_phase2.csv `
  --forwards data/eex_forwards_history.parquet `
  --spot data/epex_hourly.parquet `
  --output-dir output/phase2_20260622_solver_probe/powerbi_diagnostic_allow_failed `
  --allow-failed-gates
```

Result: sidecars written under `output/.../powerbi_diagnostic_allow_failed`.

Seasonal coherence audit:

```powershell
python scripts/audit_ch_hfc_seasonal_coherence.py `
  --csv output/phase2_20260622_solver_probe/ch_hfc_hourly_20260622_20321231_phase2.csv `
  --forwards data/eex_forwards_history.parquet `
  --report output/phase2_20260622_solver_probe/seasonal_audit_phase2.md `
  --monthly-output output/phase2_20260622_solver_probe/seasonal_monthly.csv `
  --hour-month-output output/phase2_20260622_solver_probe/seasonal_hour_month.csv `
  --monthly-split-output output/phase2_20260622_solver_probe/monthly_split_diagnostics.csv `
  --monthly-path-output output/phase2_20260622_solver_probe/monthly_path_diagnostics.csv `
  --cross-year-output output/phase2_20260622_solver_probe/cross_year_month_shape_diagnostics.csv `
  --calendar-output output/phase2_20260622_solver_probe/calendar_coherence.csv
```

Result:

```text
[seasonal-audit] critical=4 warning=4
```

Hourly shape audit:

```powershell
python scripts/audit_ch_pfc_hourly_shape.py `
  --csv output/phase2_20260622_solver_probe/ch_hfc_hourly_20260622_20321231_phase2.csv `
  --forwards data/eex_forwards_history.parquet `
  --report output/phase2_20260622_solver_probe/hourly_shape_audit_phase2.md
```

Result:

```text
[shape-audit] score=3.25/10
```

Sparse proof with candidate prior weights:

```powershell
python scripts/run_monthly_curve_sparse_year_proof.py `
  --forwards data/eex_forwards_history.parquet `
  --output-dir output/phase2_20260622_solver_probe/monthly_curve_sparse_year_proof_candidate_config `
  --market CH `
  --start 2026-06 `
  --end 2032-12 `
  --quote-consistency-tolerance 0.01 `
  --lambda-smooth-month 1.0 `
  --lambda-smooth-yoy 0.25 `
  --lambda-shape 1.0 `
  --neighbor-shrinkage 0.5 `
  --structural-amplitude-eur-mwh 110.0 `
  --allow-template-structural-fallback `
  --panel-weight 1.0 `
  --history-weight 0.5 `
  --structural-weight 1.0
```

Result:

```text
max_abs_constraint_residual=5.684e-14
neighbor_level_leakage_max_abs=2.842e-14
gate_summary={'UNSUPPORTED': 22, 'PASS': 20}
panel_status=PARTIAL_MONTHLY_PANEL history_status=PARTIAL_HISTORY_FORWARD structural_status=STRUCTURAL_TEMPLATE fused_status=PARTIAL_MONTHLY_PANEL
```

## Key Artifacts

All generated outputs are under:

```text
output/phase2_20260622_solver_probe/
```

Important files:

- `phase2_20260622_solver_probe_{slow,central,fast}.parquet`
- `phase2_20260622_solver_probe_structural_fan_chart.parquet`
- `phase2_20260622_solver_probe_structural_fan_chart.monthly_curve_manifest.json`
- `ch_hfc_hourly_20260622_20321231_phase2.csv`
- `powerbi_diagnostic_allow_failed/summary_metrics.csv`
- `monthly_curve_sparse_year_proof_candidate_config/audit_gates.csv`
- `monthly_curve_sparse_year_proof_candidate_config/manifest.json`
- `phase2_monthly_layer_comparison_candidate_config.csv`
- `phase2_layer_drift_summary_candidate_config.csv`
- `phase2_same_month_cross_year_spreads_candidate_config.csv`
- `phase2_ws_ratio_monthly_range_2028_2032_candidate_config.csv`
- `phase2_eex_residuals_sorted.csv`

Hashes:

```text
2289abbbb313a016c4b537e6d1c60c7785dd945e12cd42be064f49fa6c7cbd19  phase2_20260622_solver_probe_structural_fan_chart.monthly_curve_manifest.json
e68cc7d0aa5c60005eb966f7bf069d3ecf779304abe18a04231517e065a125ef  ch_hfc_hourly_20260622_20321231_phase2.csv
ce5f7a337247375e4121e399732c67a45b6ace7f770ea6f372a432ed9febe311  phase2_monthly_layer_comparison_candidate_config.csv
b0d649eddfe6f51f57f763944b6c79435ed6b2d243e4218ff9005ca72a0f4ca3  phase2_layer_drift_summary_candidate_config.csv
624f8a77a8cfb65bbb47c8da16c0320bb51462c4c418a48e4545bd7d01423f2a  phase2_same_month_cross_year_spreads_candidate_config.csv
925b706005ea95713bb5516e10d0311ca890a2db392238362501842dc3bbde07  phase2_ws_ratio_monthly_range_2028_2032_candidate_config.csv
4a6d4eaa04d75da56b1328319a95e4a956e606e37bbb9bc681b558ffe0bc488f  phase2_eex_residuals_sorted.csv
8f4f06bedb231027fdc907a3dab269a483e7390e8a828741e0bf081300db0682  monthly_curve_sparse_year_proof_candidate_config/audit_gates.csv
560779dd86038251368f8270e9795d29c5726ecb217b811a03b089c687b36c2a  monthly_curve_sparse_year_proof_candidate_config/manifest.json
5cbdd4e11029981872533e282c0aca6cf52f514542b3e8b6a314567f9615ba16  powerbi_diagnostic_allow_failed/summary_metrics.csv
```

Candidate monthly manifest:

```text
monthly_solution_hash=1cd6a845dd5d1f0134d08e31184fa97bb0ec86679785a3e69d852f2efd355e0f
active_constraints_hash=554cfae0e419da72a190cdfb9ce4db9149383abcda1e57ec6d3ee5a036a62c18
active_config_hash=cb11dea390965ecc5895f494163e1b9cf50ff776fde645bac1019b5b0d3cce7b
panel_status=PARTIAL_MONTHLY_PANEL
history_status=PARTIAL_HISTORY_FORWARD
structural_status=STRUCTURAL_TEMPLATE
fused_status=PARTIAL_MONTHLY_PANEL
```

Sparse proof manifest with matching prior weights:

```text
active_constraints_hash=554cfae0e419da72a190cdfb9ce4db9149383abcda1e57ec6d3ee5a036a62c18
active_config_hash=58c9874cc04f1e03e0e19df97fa0ef8088711b8aa6e39424830e363b48f17e0f
monthly_solution_hash=13323bcbadcfff24d141fa695a35d51c19c1265bb087f862764a93677a86b2f1
```

Note: the sparse proof and export manifest active config hashes differ because
the hash contracts are not identical and do not cover the full prior stack.
Do not use this as promotion proof without fixing the hash contract.

## Diagnostic Findings

Layer drift:

```text
max abs B - solver: 5.684342e-14 EUR/MWh
max abs price_shape - B: 2.842171e-14 EUR/MWh
max abs CSV weighted - fan weighted: 5.053732e-07 EUR/MWh
```

Interpretation: the Phase1 fix appears to preserve monthly BASE means through
assembler and hourly CSV export for this diagnostic run. `price_raw` was not
verified because standard artifacts do not expose it.

Power BI / delivered curve:

```text
shape_score_10=3.25
max_eex_base_error_eur_mwh=0.000000
max_eex_peak_error_eur_mwh=17.50
seasonal_warning_flags=1
monthly_split_critical_flags=1
monthly_split_warning_flags=1
cross_year_month_shape_critical_flags=3
cross_year_month_shape_warning_flags=2
powerbi_quality_gate_status=FAILED_DIAGNOSTIC
```

Worst EEX residuals:

```text
BASE max abs residual: 1.111111e-08 EUR/MWh
PEAK max abs residual: 17.497926 EUR/MWh
Worst PEAK rows:
2026-07 17.497926
2026-08 16.897451
2026-10 16.285176
2026-09 13.203526
2026-11 11.420811
```

Cross-year failures:

```text
2028 vs 2029 month 10 critical: opposite sign to parent bucket spread
2028 vs 2029 month 11 critical: opposite sign to parent bucket spread
2028 vs 2029 month 12 critical: opposite sign to parent bucket spread
2028 vs 2029 month 6 warning: near-cloned despite non-zero parent spread
2028 vs 2029 Apr-Dec seasonal slope warning
```

W/S ratio and monthly range 2028-2032 from
`phase2_ws_ratio_monthly_range_2028_2032_candidate_config.csv`:

```text
2028 solver/B/price_shape/csv W/S=1.468856 range=54.643526
2029 solver/B/price_shape/csv W/S=1.226730 range=29.244951
2030 solver/B/price_shape/csv W/S=1.209775 range=25.572712
2031 solver/B/price_shape/csv W/S=1.207214 range=24.924486
2032 solver/B/price_shape/csv W/S=1.213137 range=24.816803
```

## Decisions Recorded

Added to `DECISION-LOG.md`:

- `D-20260622-05`: candidate evidence must be layered; sparse solver proof
  alone cannot declare the delivered graph fixed.
- `D-20260622-06`: lambda/prior hashes are not yet promotion proof because
  material prior knobs are omitted.

## Files Changed By This Session

Planning only:

- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-PHASE2-DIAGNOSTIC.md`

No code files were edited in this Phase 2 diagnostic session.

Generated diagnostic outputs under `output/phase2_20260622_solver_probe/`.
No writes were made to `powerbi/data`.

## Open Risks / Next Actions

- Delivered curve is not promotion-ready: PEAK residuals and cross-year
  month-shape critical flags remain.
- The monthly BASE level chain is not the apparent culprit for this candidate;
  `solver == B == price_shape == CSV` within tolerance.
- Standard artifacts do not expose `price_raw`; add a diagnostic artifact or
  temporary trace if Phase 2 still requires explicit pre-calibration proof.
- Fix or quarantine PEAK calibration behavior before claiming dashboard quality.
- Implement a stronger promotion/hash contract covering prior weights,
  structural fallback knobs, selected lambda artifact status, and known-bad /
  coherent fixture summaries.
- Run a sensitivity candidate with lower structural amplitude or
  `--no-allow-template-structural-fallback` to quantify template dependence.
