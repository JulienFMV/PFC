# Session Handoff - 2026-06-24 - Shape PNG Remediation Pass

## Scope

Continued Phase 14 after the user rejected the monthly shape visible in PNG
diagnostics as unsatisfactory. Work stayed in LT/audit scope. No
`pfc_shaping/ct/*`, `powerbi/data/*`, or `powerbi/PFC_QA.*` files were edited.

Important date convention: the EEX workbook was available on 2026-06-24, but
the latest usable CH/DE/FR quote rows in the workbook are dated 2026-06-23.
All current promotion evidence is bound to `forward_snapshot_date=2026-06-23`,
not to 2026-06-24.

## Current Verdict

Current candidate:

`output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/`

This supersedes:

`output/phase14/20260624_asof20260623_yoy50_2032/`

Reason for supersession: the older `yoy50` candidate passed strict gates and
manifest parity, but PNG diagnostics showed annual-only years that were too
flat. The replacement keeps solver-level authority and restores seasonality by
changing solver objective/spec settings, not by patching individual months.

## Changed Files

Code/config/tests:

- `pfc_shaping/config.yaml`
- `pfc_shaping/pipeline/monthly_curve_authority.py`
- `pfc_shaping/pipeline/production_phases.py`
- `pfc_shaping/lt/model/assembler.py`
- `tests/test_monthly_forward_curve_integration.py`

Phase 14 governance artifacts:

- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260623_lshape100_yoy10_amp200_2032.json`
- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260623_lshape100_yoy10_amp200_2032.json`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-SHAPE-PNG-REMEDIATION-PASS.md`

Generated/local evidence, not commit targets by default:

- `data/eex_forwards_history.parquet`
- `pfc_shaping/output/pfc_15min_2026-06-24.*`
- `pfc_shaping/output/pfc_de_15min_2026-06-24.*`
- `pfc_shaping/model/artifacts/*.pkl`
- `pfc_shaping/model/artifacts/*.parquet`
- `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`
- `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/**`

## Data Refresh

EEX workbook used:

`H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx`

Workbook file timestamp observed:

`2026-06-24 05:03:29`

Updated local forward history:

- before: `145302` rows
- after: `146023` rows
- CH latest BASE date: `2026-06-23`
- DE latest BASE date: `2026-06-23`
- FR latest BASE date: `2026-06-23`
- AT latest BASE date: `2026-06-17`
- IT latest BASE date: `2026-06-17`

AT/IT are skipped when an exact 2026-06-23 neighbor snapshot is requested.
They are not mixed into the CH 2026-06-23 monthly solve.

## Code Changes

`monthly_curve_authority.py`:

- `latest_base_prices_by_market(...)` now accepts `as_of_date`.
- `solve_monthly_level_authority_from_history(...)` now accepts `as_of_date`.
- `_latest_base_prices(...)` now raises if the exact requested snapshot is
  missing.
- neighbor prices in the history path are pinned to the CH run timestamp.

`production_phases.py`:

- Reads `forwards.eex_as_of_date`.
- When set, CH BASE prices are loaded from the exact history snapshot and the
  production source string records `EEX history CH as-of ...`.
- Neighbor markets are loaded at the same as-of date; missing exact-date
  neighbors are skipped.

`config.yaml`:

- `forwards.eex_markets=["CH","DE","FR"]`
- `forwards.eex_as_of_date="2026-06-23"`
- monthly solver enabled for CH
- `lambda_shape=100.0`
- `lambda_smooth_month=0.1`
- `lambda_smooth_yoy=10.0`
- `structural_amplitude_eur_mwh=200.0`
- `constraint_tolerance=0.01`

`assembler.py`:

- Fixed production MLP path by replacing direct `self.sh._use_seasonal_hourly`
  access with `getattr(..., False)`.

Regression test:

- `test_solver_monthly_level_accepts_hourly_model_without_seasonal_flag`

## Candidate Generation

Full-horizon local candidate command:

```powershell
python scripts/export_local_test_ch_hourly_csv.py --local-end-date 2032-12-31 --enable-monthly-forward-curve-solver --enable-eex-peak-calibration --monthly-solver-lambda-shape 100 --monthly-solver-lambda-smooth-month 0.1 --monthly-solver-lambda-smooth-yoy 10 --monthly-solver-structural-amplitude 200 --enable-structural-shape-upgrade --structural-shape-upgrade-intensity 0.5 --structural-scenario-spread-intensity 1.26 --required-forward-date 2026-06-23 --output output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/ch_hfc_hourly_asof20260623_lshape100_yoy10_amp200_2032.csv --report output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/export_report.md --fan-chart-output output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/fan_asof20260623_lshape100_yoy10_amp200_2032.parquet --skip-powerbi-refresh
```

Result:

- rows: `57457`
- weighted mean: `78.95`
- scenario spread mean: `0.5165`
- `forward_snapshot_date=2026-06-23`

Local export manifest:

`output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/fan_asof20260623_lshape100_yoy10_amp200_2032.monthly_curve_manifest.json`

Key manifest values:

- `active_config_hash=f4b64f88919149a42a85693135c047b442ffa099011ce17e41c1cfe8782db88e`
- `active_constraints_hash=a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262`
- `monthly_solution_hash=d717a426f5fee7fe62abf294a0e44311040115fd4edb6a3a118f06bf7243832e`
- `panel_status=PARTIAL_MONTHLY_PANEL`
- `history_status=PARTIAL_HISTORY_FORWARD`
- `structural_status=STRUCTURAL_TEMPLATE`
- `fused_status=PARTIAL_MONTHLY_PANEL`

## PNG Diagnostics

Command:

```powershell
python scripts/plot_ch_hfc_diagnostics.py --csv output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/ch_hfc_hourly_asof20260623_lshape100_yoy10_amp200_2032.csv --forwards data/eex_forwards_history.parquet --output-dir output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/png_diagnostics
```

Key PNGs:

- `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/png_diagnostics/01_monthly_means_by_year.png`
- `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/png_diagnostics/05_heatmap_month_hour_2030.png`
- `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/png_diagnostics/09_executive_qa_summary.png`

Monthly shape metrics from `monthly_diagnostics.csv`:

| year | amplitude | Q1-Q3 | Jan-Oct |
|---|---:|---:|---:|
| 2027 | 63.36 | 48.55 | 31.08 |
| 2028 | 60.82 | 45.73 | 31.46 |
| 2029 | 44.87 | 29.27 | 22.43 |
| 2030 | 39.75 | 24.23 | 17.92 |
| 2031 | 39.33 | 23.81 | 17.55 |
| 2032 | 39.33 | 23.76 | 17.52 |

For comparison, the superseded `yoy50` candidate had only about `27-30`
EUR/MWh amplitude in 2029-2032.

## Production Dry Run

Command used inline:

```powershell
@'
import logging, os
from pathlib import Path
from pfc_shaping.pipeline.production_phases import load_inputs, run_long_term_phase, save_long_term_outputs
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('phase14-production-dryrun-lshape100-yoy10-amp200')
project_root = Path.cwd()
inputs = load_inputs(project_root, logger)
lt = run_long_term_phase(project_root=project_root, inputs=inputs, peak_source_policy=os.environ.get('PFC_PEAK_SOURCE_POLICY','same_first'), logger=logger)
save_long_term_outputs(lt, logger)
print('LT_DRYRUN_SAVE_OK')
'@ | python -
```

Result: exit `0`; production outputs and manifest were written.

Production manifest:

`pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`

Key values:

- `forward_snapshot_date=2026-06-23`
- `active_config_hash=f4b64f88919149a42a85693135c047b442ffa099011ce17e41c1cfe8782db88e`
- `active_constraints_hash=a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262`
- `monthly_solution_hash=d717a426f5fee7fe62abf294a0e44311040115fd4edb6a3a118f06bf7243832e`
- manifest sha256: `46c519216aa2db8fbbb0772ecae67b3d5bebc0a4f8f6be1408bec696224ce9fc`

## Delivered-Product Audit

Source hierarchy policy:

`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260623_lshape100_yoy10_amp200_2032.json`

Policy sha256:

`f5c2723dc2c05d484bba421cf4640c35e7a70d3362d35d792dfb21f63d095313`

Command:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/ch_hfc_hourly_asof20260623_lshape100_yoy10_amp200_2032.csv --forwards data/eex_forwards_history.parquet --required-forward-date 2026-06-23 --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260623_lshape100_yoy10_amp200_2032.json --output-csv output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/product_normalization_audit_policy.csv --summary-json output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/product_normalization_audit_policy.json
```

Result: exit `0`.

Summary:

- `all_gates_pass=true`
- `PASS=80`
- `QUOTE_CONFLICT=9`
- `accepted_quote_conflict_count=9`
- `blocking_quote_conflict_count=0`
- `UNSUPPORTED=0`
- `OUT_OF_SCOPE=3`
- `critical_count=0`
- `delivered_curve_drift_count=0`
- `input_csv_sha256=fba35ff2b007da57e561a6ccad3114a790be73bb58143f45ac23b3a8abf88172`
- `forwards_sha256=ca0d2ee05e97a119ad8a434e18a71322928b42376d7765ff06659544040241d0`
- `quote_conflict_identity_hash=b13d9c87813f9cbf9c43d8cbbe0bb533b029e0845b80e363aee7aaa2946a66f9`

## Strict Power BI Export

Command:

```powershell
python scripts/build_powerbi_exports.py --csv output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/ch_hfc_hourly_asof20260623_lshape100_yoy10_amp200_2032.csv --forwards data/eex_forwards_history.parquet --spot data/epex_hourly.parquet --output-dir output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/powerbi_strict
```

Result: exit `0`.

Key `summary_metrics.csv` values:

- `powerbi_quality_gate_status=PASS`
- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `negative_gate_status=PASS`
- `seasonal_warning_flags=0`
- `seasonal_critical_flags=0`
- `monthly_split_critical_flags=0`
- `monthly_path_critical_flags=0`
- `cross_year_month_shape_warning_flags=0`
- `calendar_critical_flags=0`
- `latest_hfc_winter_summer_spread_eur_mwh=31.10`

## Selected Config Artifact

Created:

`.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260623_lshape100_yoy10_amp200_2032.json`

sha256:

`1eac4023b40cc3ac46158b70f5b6729aec95bdec4eddee878911f9d384ce7514`

Key values:

- `schema_version=monthly_curve_selected_config.v1`
- `forward_snapshot_date=2026-06-23`
- `production_approved=true`
- `production_promotion_approved=true`
- `selection_status=PRODUCTION_APPROVED`
- `promotion_scope=PRODUCTION_EXPORT_SELECTED_TRIAD`
- `production_manifest_triad_validated=true`
- `config_hash=f4b64f88919149a42a85693135c047b442ffa099011ce17e41c1cfe8782db88e`
- `active_constraints_hash=a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262`
- `monthly_solution_hash=d717a426f5fee7fe62abf294a0e44311040115fd4edb6a3a118f06bf7243832e`

## Promotion Capstone

Sparse proof:

```powershell
python scripts/run_monthly_curve_sparse_year_proof.py --forwards data/eex_forwards_history.parquet --output-dir output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/monthly_curve_sparse_year_proof --historical-thresholds output/phase14/20260624_asof20260623_yoy50_2032/monthly_curve_calibration/historical_thresholds.csv --lambda-shape 100 --lambda-smooth-month 0.1 --lambda-smooth-yoy 10 --neighbor-markets DE,FR,AT,IT --require-lambda-artifact --active-config-hash f4b64f88919149a42a85693135c047b442ffa099011ce17e41c1cfe8782db88e --selected-config-hash f4b64f88919149a42a85693135c047b442ffa099011ce17e41c1cfe8782db88e --selected-config-artifact .planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260623_lshape100_yoy10_amp200_2032.json --require-path-parity --production-monthly-solution-hash d717a426f5fee7fe62abf294a0e44311040115fd4edb6a3a118f06bf7243832e --export-monthly-solution-hash d717a426f5fee7fe62abf294a0e44311040115fd4edb6a3a118f06bf7243832e --production-active-constraints-hash a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262 --export-active-constraints-hash a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262 --no-plot
```

Result:

- `max_abs_constraint_residual=5.684e-14`
- `neighbor_level_leakage_max_abs=8.527e-14`
- `gate_summary={'PASS': 25, 'UNSUPPORTED': 10}`
- no CRITICAL gates

Real production/export/selected capstone:

```powershell
python scripts/check_monthly_curve_promotion_from_manifests.py --audit-gates output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/monthly_curve_sparse_year_proof/audit_gates.csv --historical-thresholds output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/monthly_curve_sparse_year_proof/historical_thresholds.csv --manifest output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/monthly_curve_sparse_year_proof/manifest.json --production-manifest pfc_shaping/model/artifacts/production_monthly_curve_manifest.json --export-manifest output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/fan_asof20260623_lshape100_yoy10_amp200_2032.monthly_curve_manifest.json --selected-config-artifact .planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260623_lshape100_yoy10_amp200_2032.json --run-timestamp 2026-06-23 --augmented-audit-gates output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/promotion_triad_real_prod_check/audit_gates_real_prod_triad.csv --output output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad.json --details-output output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad_details.csv
```

Result: exit `0`.

Decision summary:

```json
{
  "approved": true,
  "audit_gate_status_counts": {
    "PASS": 27,
    "UNSUPPORTED": 10
  },
  "blocking_count": 0,
  "manifest_gate_summary": {
    "PASS": 25,
    "UNSUPPORTED": 10
  },
  "status": "PROMOTION_EVIDENCE_PASS",
  "threshold_status_counts": {
    "PASS": 13,
    "UNSUPPORTED": 13
  }
}
```

Capstone sha256:

`c4b3f025be134bd3e815aa98216e9b785e3e7e0e40a8092ddd39be52079ca48d`

## Tests

Targeted LT/audit suite:

```powershell
python -m pytest tests/test_monthly_forward_curve_integration.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_build_powerbi_exports_script.py tests/test_plot_ch_hfc_diagnostics_script.py -q -p no:cacheprovider
```

Result:

`62 passed, 13 warnings in 21.62s`

## Residual Risks

- `data/eex_forwards_history.parquet` is locally refreshed evidence and should
  not be committed unless explicitly requested.
- Production output CSV/parquet and model artifacts are generated evidence and
  should not be committed by default.
- AT/IT were not present at 2026-06-23; exact-as-of governance skips them.
- Sparse proof still reports documented `UNSUPPORTED` threshold rows due
  insufficient historical support; capstone accepts them because there are no
  `CRITICAL` gates and the unsupported rows are explicit.
- `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/export_report.md`
  is a local-test export report and should not be read as the production
  promotion decision. The authoritative promotion evidence is the
  manifest-backed capstone
  `promotion_triad_real_prod_check/promotion_decision_real_prod_triad.json`,
  which reports `approved=true` and `PROMOTION_EVIDENCE_PASS`.
- Production/export `solver_config_hash` differs even though the enforced
  `active_config_hash`, `active_constraints_hash`, and `monthly_solution_hash`
  match. This is not blocking under current policy; the active config hash is
  the canonical governance hash for selected-lambda parity.
