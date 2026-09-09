# Session Handoff - 2026-07-07 - Daily Generation As-Of 2026-07-06

## Scope

Generated and audited the CH LT PFC for Tuesday 2026-07-07, bound to the real
EEX quote snapshot `2026-07-06`. Work stayed in LT/Phase 14 scope. No
`pfc_shaping/ct/*`, `powerbi/data/*`, or `powerbi/PFC_QA.*` files were edited.

Important date convention: the workbook was available on 2026-07-07, but the
latest usable CH/DE/FR quote rows inside it are dated 2026-07-06. All current
evidence is therefore bound to `forward_snapshot_date=2026-07-06`.

## Data Refresh

Workbook used:

`H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx`

Observed workbook timestamp: `2026-07-07 05:03:28` local filesystem time.

Local forward history was appended with
`pfc_shaping.data.ingest_forwards.update_forwards_parquet(...)`.

Coverage after append:

- rows: `147496`
- CH latest BASE date: `2026-07-06`
- DE latest BASE date: `2026-07-06`
- FR latest BASE date: `2026-07-06`
- AT latest BASE date: `2026-06-17`
- IT latest BASE date: `2026-06-17`

AT/IT were not present in the daily workbook; exact-as-of neighbor evidence for
the 2026-07-06 solve used DE/FR only.

## Final Candidate

Final promotion-ready candidate:

`output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/`

Rationale: the earlier `amp200` candidate passed strict Power BI but failed
monthly sparse proof on 2028-06 vs 2029-06. Solver probes showed
`structural_amplitude_eur_mwh=150` removes the sparse-proof CRITICAL gate while
preserving visible seasonal shape. No individual month was patched after the
solver.

Local export command:

```powershell
python scripts/export_local_test_ch_hourly_csv.py --valuation-date 2026-07-07 --local-start-date 2026-07-01 --local-end-date 2032-12-31 --enable-monthly-forward-curve-solver --enable-eex-peak-calibration --monthly-solver-lambda-shape 100 --monthly-solver-lambda-smooth-month 0.1 --monthly-solver-lambda-smooth-yoy 150 --monthly-solver-structural-amplitude 150 --enable-structural-shape-upgrade --structural-shape-upgrade-intensity 0.5 --structural-scenario-spread-intensity 1.26 --required-forward-date 2026-07-06 --output output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260706_lshape100_yoy150_amp150_2032.csv --report output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/export_report.md --fan-chart-output output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/fan_asof20260706_lshape100_yoy150_amp150_2032.parquet --skip-powerbi-refresh
```

Result:

- rows: `57025`
- weighted mean: `78.98`
- scenario spread mean: `0.5168`
- CSV sha256: `4349f4774a4faace76f2022f2a6c39970eb40eb0916099a82801739d53381668`
- export manifest sha256: `0a048fbe07ea3e1afc32de1c95662708cd058b539790c75336978cd85c728a74`

Manifest values:

- `active_config_hash=f95e81bf8987174eb8b553406de296fc8cfb67a3dfde35f006b88cc006a66469`
- `active_constraints_hash=451f9a23a4388973addabc7b467b535df2352bec74e69af34dcf4c28cdbeb15d`
- `monthly_solution_hash=a505fd3a07cb5573adc2a76061caa52f300ee90043766e10cfe4819107171a04`

## Strict Gates

Power BI strict command:

```powershell
python scripts/build_powerbi_exports.py --csv output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260706_lshape100_yoy150_amp150_2032.csv --forwards data/eex_forwards_history.parquet --spot data/epex_hourly.parquet --output-dir output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/powerbi_strict
```

Result: exit `0`, `powerbi_quality_gate_status=PASS`.

Key metrics:

- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `seasonal_critical_flags=0`
- `seasonal_warning_flags=0`
- `monthly_path_critical_flags=0`
- `monthly_path_warning_flags=4`
- `cross_year_month_shape_critical_flags=0`
- `cross_year_month_shape_warning_flags=0`
- `latest_hfc_winter_summer_spread_eur_mwh=24.90`

Delivered-product audit strict command:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260706_lshape100_yoy150_amp150_2032.csv --forwards data/eex_forwards_history.parquet --required-forward-date 2026-07-06 --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260706_lshape100_yoy150_amp150_2032.json --output-csv output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/product_normalization_audit_policy.csv --summary-json output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/product_normalization_audit_policy.json
```

Result: exit `0`, `all_gates_pass=true`.

Key metrics:

- `PASS=90`
- `QUOTE_CONFLICT=6`
- `accepted_quote_conflict_count=6`
- `blocking_quote_conflict_count=0`
- `UNSUPPORTED=0`
- `critical_count=0`
- `delivered_curve_drift_count=0`
- `quote_conflict_identity_hash=a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`
- source hierarchy policy sha256: `9dca2c8940c11408a00ab8c6fdf62fcfbdb96796543dd22d9503a9455065020f`

## Production Generation

`pfc_shaping/config.yaml` was updated locally for the final run:

- `forwards.eex_as_of_date="2026-07-06"`
- `monthly_curve_solver.lambda_smooth_yoy=150.0`
- `monthly_curve_solver.structural_amplitude_eur_mwh=150.0`

Commit hygiene note: `eex_as_of_date="2026-07-06"` was a run pin for this
evidence package. It should not be committed as the durable default; after the
run, the config should return to `eex_as_of_date=null` so future daily runs use
the latest available quote row unless explicitly pinned.

Production LT save command used `load_inputs`, `run_long_term_phase`, and
`save_long_term_outputs` from `pfc_shaping.pipeline.production_phases`.

Result: exit `0`.

- `LT_PRODUCTION_SAVE_OK`
- today: `2026-07-07`
- markets: `CH`, `DE`
- CH rows: `227424`, source `EEX history CH as-of 2026-07-06 (20 keys)`
- DE rows: `227424`, source `EEX XLSX DE (71 keys)`

Production outputs:

- `pfc_shaping/output/pfc_15min_2026-07-07.csv`
- `pfc_shaping/output/pfc_15min_2026-07-07.parquet`
- `pfc_shaping/output/pfc_de_15min_2026-07-07.csv`
- `pfc_shaping/output/pfc_de_15min_2026-07-07.parquet`
- `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`

Production manifest sha256:

`d0a130a5651a01a39001918a88f217b4e73f61388d501006be7e194915170f56`

Production manifest values:

- `active_config_hash=f95e81bf8987174eb8b553406de296fc8cfb67a3dfde35f006b88cc006a66469`
- `active_constraints_hash=451f9a23a4388973addabc7b467b535df2352bec74e69af34dcf4c28cdbeb15d`
- `monthly_solution_hash=a505fd3a07cb5573adc2a76061caa52f300ee90043766e10cfe4819107171a04`

## Promotion Evidence

Selected config artifact:

`.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260706_lshape100_yoy150_amp150_2032.json`

Selected config sha256:

`2698f1dfe26ad350ff71878c5114f04593789831812625c67ec88324b029b62d`

Quote conflict policy artifact:

`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260706_lshape100_yoy150_amp150_2032.json`

Sparse proof:

`output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/monthly_curve_sparse_year_proof_full_export_window_strict/`

Sparse proof status counts:

- `PASS=33`
- `WARNING=2`
- `UNSUPPORTED=7`
- `CRITICAL=0`

Capstone command:

```powershell
python scripts/check_monthly_curve_promotion_from_manifests.py --audit-gates output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/monthly_curve_sparse_year_proof_full_export_window_strict/audit_gates.csv --historical-thresholds output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/monthly_curve_sparse_year_proof_full_export_window_strict/historical_thresholds.csv --manifest output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/monthly_curve_sparse_year_proof_full_export_window_strict/manifest.json --production-manifest pfc_shaping/model/artifacts/production_monthly_curve_manifest.json --export-manifest output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/fan_asof20260706_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json --selected-config-artifact .planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260706_lshape100_yoy150_amp150_2032.json --run-timestamp 2026-07-06 --augmented-audit-gates output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/promotion_triad_real_prod_check/audit_gates_real_prod_triad.csv --output output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad.json --details-output output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad_details.csv
```

Result:

- `approved=true`
- `status=PROMOTION_EVIDENCE_PASS`
- `blocking_count=0`
- decision sha256: `091105ba9bc313b36364a75a9dd88ab9e3eaa9e740151c44c8a40c42cce1048c`

## PNG Diagnostics

Command:

```powershell
python scripts/plot_ch_hfc_diagnostics.py --csv output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260706_lshape100_yoy150_amp150_2032.csv --forwards data/eex_forwards_history.parquet --output-dir output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/png_diagnostics
```

Key PNGs:

- `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/png_diagnostics/01_monthly_means_by_year.png`
- `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/png_diagnostics/05_heatmap_month_hour_2030.png`
- `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/png_diagnostics/09_executive_qa_summary.png`

Visual inspection: EEX residual panel is zero for BASE/PEAK; monthly curves keep
visible winter/summer seasonality while avoiding the `amp200` sparse-proof Q2
cross-year critical.

## OMPEX Benchmark

OMPEX benchmark is read-only and was not used in model inputs, priors,
objectives, calibration, or candidate selection.

Latest benchmark file observed:

`H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260707_101700.xlsx`

The legacy `python -m pfc_shaping.validation.compare_hfc` failed with
`No overlapping timestamps between PFC and HFC` because of timestamp convention
mismatch. A read-only local-hour benchmark was run against the final candidate
using `timestamp_ch`.

Output:

`output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/ompex_benchmark_read_only/benchmark_metrics.json`

Metrics:

- overlap: `2026-07-01 00:00:00` -> `2031-01-01 00:00:00`
- points: `39473`
- MAE: `14.0163` EUR/MWh
- RMSE: `18.4226` EUR/MWh
- bias: `0.6167` EUR/MWh
- p95 absolute error: `37.8807` EUR/MWh
- correlation: `0.8330`
- `ompex_used_in_model=false`

## Tests

Command:

```powershell
python -m pytest tests/test_monthly_forward_curve_integration.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_build_powerbi_exports_script.py tests/test_plot_ch_hfc_diagnostics_script.py -q -p no:cacheprovider
```

Result: `62 passed, 13 warnings`.

`git diff --check` returned exit `0` with only line-ending warnings.

## Read-Only Roasters/MIT Audit

Three independent read-only subagent audits were run after the capstone.

Verdict: `GO` from all three auditors. No P0/P1 blocker found.

Audited areas:

- governance triad: production/export/selected hashes and capstone
- quantitative shape: strict Power BI, sparse proof, product normalization,
  PNG existence/nonblank checks, EEX residuals
- contamination/commit hygiene: no OMPEX model usage, benchmark read-only,
  generated files not staged

Accepted P2 findings:

- `export_report.md` in the generated candidate directory still contains
  local-test boilerplate saying production governance remains NO-GO. It is not
  the selected governance artifact and is superseded by the selected config
  plus capstone.
- Sparse-proof standalone manifest is not self-explanatory because it records
  its own diagnostic solution hash and `production_approved=false`; the
  capstone resolves promotion using the real production/export/selected
  manifests.
- Sparse-proof manifest points at the historical thresholds source path used
  during the run. The strict proof folder also contains copied thresholds; audit
  comparison found only floating-format drift, not materially different
  thresholds.
- Residual accepted warnings remain: 2 sparse-proof P2 warnings, 7 far-horizon
  `UNSUPPORTED`, and 4 Power BI monthly path warnings. None hide a `CRITICAL`.

Commit hygiene finding:

- `data/eex_forwards_history.parquet` is modified local evidence and must not
  be committed.
- Avoid broad `git add -A`; stage only curated code/docs/governance artifacts.

## Worktree / Commit Hygiene

Generated or refreshed local evidence, not default commit targets:

- `data/eex_forwards_history.parquet`
- `pfc_shaping/output/*2026-07-07*`
- `pfc_shaping/model/artifacts/*`
- `output/phase14/20260707_*`
- local benchmark outputs under `ompex_benchmark_read_only/`

The export script modified a tracked Phase 13 generated report; it was restored
from HEAD to avoid cross-phase pollution.

Commit candidates for Phase 14, subject to final curation:

- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260707-DAILY-GENERATION-ASOF20260706.md`
- `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260706_lshape100_yoy150_amp150_2032.json`
- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260706_lshape100_yoy150_amp150_2032.json`
- `pfc_shaping/config.yaml`
- existing code/test changes already in the worktree, after review.
