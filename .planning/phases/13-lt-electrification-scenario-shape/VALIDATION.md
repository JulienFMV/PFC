# Phase 13 VALIDATION: Electrification-Scenario Shape

## Scope

Branch: `feat/lt-next-sota`.

Implemented foundation:

* `pfc_shaping/lt/model/electrification_shape.py`
* `pfc_shaping/data/electrification_scenarios.py`
* `PFCAssembler(enable_electrification_shape=False, ...)`
* perfect-foresight estimator string `sota_electrification`
* `scripts/validate_electrification_scenarios.py`
* `scripts/build_hpfc_scenario_features.py`
* `scripts/import_ep2050_hourly.py`
* `scripts/enrich_electrification_scenarios.py`
* `scripts/import_bfe_structural_actuals.py`
* `scripts/build_first_ep2050_pfc.py`
* `scripts/build_ep2050_multi_scenario_pfc.py`
* `scripts/build_local_test_ch_pfc.py`
* `scripts/export_local_test_ch_hourly_csv.py`
* `scripts/materialize_lt_data_contract.py`
* `tests/test_electrification_shape.py`
* `tests/test_build_ep2050_multi_scenario_pfc_script.py`
* `tests/test_materialize_lt_data_contract_script.py`
* multi-scenario application and weighted structural fan chart utilities
* production data schema validation and as-of helpers
* gold penetration-feature derivation for HPFC scenario modelling
* local OFEN EP2050+ hourly extraction into silver/gold parquet tables
* Databricks table key `electrification_scenarios`

The layer is OFF by default. When enabled without an official scenario table, it
returns identity and emits a warning:

```text
Electrification scenario store is empty; returning f_H unchanged
```

This is intentional for local diagnostics. A production/gate run must provide a
versioned `data/electrification_scenarios.parquet` or explicit scenario path.
Set `require_electrification_scenarios=True` on `PFCAssembler`, or use the
inventory validator script, when missing scenario data must fail the run.

## Commands Run

```powershell
$env:PYTHONPATH='.'; python -m py_compile pfc_shaping/data/electrification_scenarios.py pfc_shaping/data/__init__.py pfc_shaping/lt/model/electrification_shape.py pfc_shaping/lt/model/assembler.py pfc_shaping/validation/perfect_foresight.py scripts/run_perfect_foresight.py scripts/validate_electrification_scenarios.py
$env:PYTHONPATH='.'; pytest tests/test_electrification_scenarios_data.py -q
$env:PYTHONPATH='.'; pytest tests/test_electrification_shape.py -q
$env:PYTHONPATH='.'; pytest tests/test_intraday_amplitude.py -q
$env:PYTHONPATH='.'; pytest tests/test_validate_electrification_scenarios_script.py -q
$env:PYTHONPATH='.'; pytest tests/test_build_hpfc_scenario_features_script.py -q
$env:PYTHONPATH='.'; pytest tests/test_ep2050_import.py -q
$env:PYTHONPATH='.'; pytest tests/test_ep2050_import.py tests/test_electrification_scenarios_data.py tests/test_electrification_shape.py tests/test_intraday_amplitude.py tests/test_validate_electrification_scenarios_script.py tests/test_build_hpfc_scenario_features_script.py -q
$env:PYTHONPATH='.'; pytest tests/test_perfect_foresight.py::test_build_curve_rejects_unknown_estimator_string tests/test_perfect_foresight.py::test_sota_swap_restores_on_exception tests/test_perfect_foresight.py::test_sota_swap_patches_and_restores_shapehourly_halflife -q
git diff --check
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/run_perfect_foresight.py --estimator sota_electrification --no-figures
$env:PYTHONPATH='.'; python scripts/import_ep2050_hourly.py --cache-dir C:\Users\jbattaglia\pfc_local_data\ep2050 --years 2025,2030,2035 --scenarios ZERO_Basis,WWB --skip-download
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/enrich_electrification_scenarios.py --input data/electrification_scenarios_ep2050.parquet --output data/electrification_scenarios_ep2050_enriched.parquet --hourly data/silver_ep2050_hourly.parquet --assumption-publication-date 2026-06-05
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_first_ep2050_pfc.py --market CH --scenario ZERO_Basis --start-date 2030-01-01 --horizon-days 365 --output output/first_ep2050_pfc_2030_zero_basis.parquet --summary .planning/phases/13-lt-electrification-scenario-shape/FIRST-PFC.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path output/first_ep2050_pfc_2030_zero_basis_scenario_expanded.parquet --country CH --scenarios ZERO_Basis --years 2030,2031 --vintage 2026-06-05 --report .planning/phases/13-lt-electrification-scenario-shape/FIRST-PFC-SCENARIO-GATE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_first_ep2050_pfc.py --market CH --scenario ZERO_Basis --start-date 2030-01-01 --horizon-days 365 --scenario-path data/electrification_scenarios_ep2050_enriched.parquet --output output/first_ep2050_pfc_2030_zero_basis_enriched.parquet --summary .planning/phases/13-lt-electrification-scenario-shape/FIRST-PFC-ENRICHED.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path output/first_ep2050_pfc_2030_zero_basis_enriched_scenario_expanded.parquet --country CH --scenarios ZERO_Basis --years 2030,2031 --vintage 2026-06-05 --require-recommended --report .planning/phases/13-lt-electrification-scenario-shape/FIRST-PFC-ENRICHED-SCENARIO-GATE.md
$env:PYTHONPATH='.'; python -m py_compile scripts/build_ep2050_multi_scenario_pfc.py
$env:PYTHONPATH='.'; pytest tests/test_build_ep2050_multi_scenario_pfc_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_ep2050_multi_scenario_pfc.py --scenario-input data/electrification_scenarios_ep2050_enriched.parquet --scenario-output data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet --features-output data/hpfc_scenario_features_ep2050_enriched_slow_central_fast.parquet --market CH --start-date 2030-01-01 --horizon-days 365 --output-prefix ep2050_pfc_2030 --fan-chart-output output/ep2050_pfc_2030_structural_fan_chart.parquet --summary .planning/phases/13-lt-electrification-scenario-shape/MULTI-SCENARIO-PFC.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path output/ep2050_pfc_2030_scenario_expanded.parquet --country CH --scenarios slow,central,fast --years 2030,2031 --vintage 2026-06-05 --require-recommended --report .planning/phases/13-lt-electrification-scenario-shape/MULTI-SCENARIO-GATE.md
$env:PYTHONPATH='.'; python -m py_compile scripts/materialize_lt_data_contract.py
$env:PYTHONPATH='.'; pytest tests/test_materialize_lt_data_contract_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/materialize_lt_data_contract.py --local-root C:\Users\jbattaglia\pfc_local_data --scenario-source data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet --scenario-output data/electrification_scenarios.parquet --features-output data/hpfc_scenario_features.parquet --years 2027-2031 --vintage 2026-06-05 --country CH --scenarios slow,central,fast --report .planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-LOCAL.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path data/electrification_scenarios.parquet --country CH --scenarios slow,central,fast --years 2027-2031 --vintage 2026-06-05 --require-recommended --report .planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-GATE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path data/electrification_scenarios.parquet --country CH --scenarios slow,central,fast --years 2027-2031 --vintage 2026-06-05 --require-recommended --require-production --required-countries CH,DE,FR,IT,AT --report .planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-PRODUCTION-GATE.md
$env:PYTHONPATH='.'; python -m py_compile scripts/import_tyndp2024_supply_inputs.py
$env:PYTHONPATH='.'; python -m py_compile scripts/import_tyndp2024_demand_outputs.py
$env:PYTHONPATH='.'; pytest tests/test_import_tyndp2024_supply_inputs_script.py tests/test_import_tyndp2024_demand_outputs_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/import_tyndp2024_supply_inputs.py --workbook "C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\20231103-Final-Supply-Inputs-for-TYNDP-2024-Scenarios.xlsx\20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx" --output data/electrification_scenarios_tyndp2024_supply.parquet --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-SUPPLY-IMPORT.md --countries CH,DE,FR,IT,AT --scenarios slow,central,fast --years 2030,2040 --publication-date 2024-05-31 --ingested-at-utc 2026-06-11
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path data/electrification_scenarios_tyndp2024_supply.parquet --country CH --scenarios slow,central,fast --years 2030,2040 --vintage 2026-06-11 --require-recommended --require-production --required-countries CH,DE,FR,IT,AT --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-SUPPLY-PRODUCTION-GATE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/import_tyndp2024_demand_outputs.py --workbook "C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb" --output data/electrification_scenarios_tyndp2024_demand.parquet --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-DEMAND-IMPORT.md --countries AT,DE,FR,IT --publication-date 2024-05-31 --ingested-at-utc 2026-06-11
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path data/electrification_scenarios_tyndp2024_demand.parquet --country DE --scenarios tyndp_distributed_energy,tyndp_global_ambition --years 2040,2050 --vintage 2026-06-11 --require-recommended --require-production --required-countries AT,DE,FR,IT --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-DEMAND-PRODUCTION-GATE.md
$env:PYTHONPATH='.'; python -m py_compile scripts/build_tyndp2024_neighbor_demand_bridge.py scripts/audit_ntc_baseline_inputs.py
$env:PYTHONPATH='.'; pytest tests/test_build_tyndp2024_neighbor_demand_bridge_script.py tests/test_audit_ntc_baseline_inputs_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_tyndp2024_neighbor_demand_bridge.py --workbook "C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb" --entso pfc_shaping/data/entso_15min.parquet --output data/electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-NEIGHBOR-DEMAND-BRIDGE.md --countries AT,DE,FR,IT --vintage 2026-06-11 --publication-date 2026-06-11 --ingested-at-utc 2026-06-11
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/audit_ntc_baseline_inputs.py --entso pfc_shaping/data/entso_15min.parquet --report .planning/phases/13-lt-electrification-scenario-shape/NTC-BASELINE-AUDIT.md
$env:PYTHONPATH='.'; python -m py_compile scripts/compose_lt_scenario_inventory.py
$env:PYTHONPATH='.'; pytest tests/test_compose_lt_scenario_inventory_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/compose_lt_scenario_inventory.py --tyndp-supply data/electrification_scenarios_tyndp2024_supply.parquet --ch-ep2050 data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet --neighbor-demand data/electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet --output data/electrification_scenarios_composed_partial_2030.parquet --features-output data/hpfc_scenario_features_composed_partial_2030.parquet --report .planning/phases/13-lt-electrification-scenario-shape/COMPOSED-PARTIAL-INVENTORY.md --years 2030 --vintage 2026-06-11 --countries CH,DE,FR,IT,AT --scenarios slow,central,fast
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_electrification_scenarios.py --path data/electrification_scenarios_composed_partial_2030.parquet --country CH --scenarios slow,central,fast --years 2030 --vintage 2026-06-11 --require-recommended --require-production --required-countries CH,DE,FR,IT,AT --report .planning/phases/13-lt-electrification-scenario-shape/COMPOSED-PARTIAL-PRODUCTION-GATE.md
$env:PYTHONPATH='.'; python -m py_compile scripts/validate_scenario_governance.py
$env:PYTHONPATH='.'; pytest tests/test_validate_scenario_governance_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_scenario_governance.py --inventory data/electrification_scenarios_prod_candidate_neutralized_2030.parquet --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml --vintage 2026-06-12 --countries CH,DE,FR,IT,AT --scenarios slow,central,fast --years 2030 --report .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-VALIDATION.md
$env:PYTHONPATH='.'; python -m py_compile scripts/build_lt_data_gap_register.py
$env:PYTHONPATH='.'; pytest tests/test_build_lt_data_gap_register_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_lt_data_gap_register.py --inventory data/electrification_scenarios_prod_candidate_neutralized_2030.parquet --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml --vintage 2026-06-12 --countries CH,DE,FR,IT,AT --scenarios slow,central,fast --years 2030 --csv-output data/lt_scenario_governance_gap_register.csv --report .planning/phases/13-lt-electrification-scenario-shape/LT-DATA-GAP-REGISTER.md
$env:PYTHONPATH='.'; python -m py_compile scripts/build_scenario_governance_approval_pack.py
$env:PYTHONPATH='.'; pytest tests/test_build_scenario_governance_approval_pack_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_scenario_governance_approval_pack.py --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml --governance-report .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-VALIDATION.md --gap-register data/lt_scenario_governance_gap_register.csv --expert-reviews .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-EXPERT-REVIEWS.md --output .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-APPROVAL-PACK.md
$env:PYTHONPATH='.'; python -m py_compile scripts/bridge_lt_p0_structural_fields.py
$env:PYTHONPATH='.'; pytest tests/test_bridge_lt_p0_structural_fields_script.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/bridge_lt_p0_structural_fields.py --input data/electrification_scenarios_composed_partial_2030.parquet --output data/electrification_scenarios_composed_p0_bridge_2030.parquet --features-output data/hpfc_scenario_features_composed_p0_bridge_2030.parquet --report .planning/phases/13-lt-electrification-scenario-shape/P0-STRUCTURAL-BRIDGE.md --entso pfc_shaping/data/entso_15min.parquet --demand-workbook C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb --countries CH,DE,FR,IT,AT
$env:PYTHONPATH='.'; python -m py_compile scripts/apply_swissgrid_ntc_baseline.py scripts/apply_ember_yearly_baseline.py
$env:PYTHONPATH='.'; pytest tests/test_apply_swissgrid_ntc_baseline_script.py tests/test_apply_ember_yearly_baseline_script.py -q
$env:PYTHONPATH='.'; pytest tests/test_phase10_reproducibility.py -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/apply_swissgrid_ntc_baseline.py --input data/electrification_scenarios_composed_p0_bridge_2030.parquet --swissgrid-csv C:\Users\jbattaglia\pfc_local_data\scenarios\swissgrid_ntc\Grenzfluesse-2026.csv --output data/electrification_scenarios_composed_p0_real_sources_2030.parquet --features-output data/hpfc_scenario_features_composed_p0_real_sources_2030.parquet --component-output data/electrification_scenarios_swissgrid_ntc_baseline_2026.parquet --report .planning/phases/13-lt-electrification-scenario-shape/SWISSGRID-NTC-BASELINE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/apply_ember_yearly_baseline.py --input data/electrification_scenarios_composed_p0_real_sources_2030.parquet --ember-csv C:\Users\jbattaglia\pfc_local_data\scenarios\ember_yearly\yearly_full_release_long_format.csv --output data/electrification_scenarios_composed_p0_public_sources_2030.parquet --features-output data/hpfc_scenario_features_composed_p0_public_sources_2030.parquet --component-output data/electrification_scenarios_ember_yearly_baseline_2026.parquet --report .planning/phases/13-lt-electrification-scenario-shape/EMBER-YEARLY-BASELINE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/apply_lt_neutralization_policy.py --input data/electrification_scenarios_composed_p0_public_sources_2030.parquet --output data/electrification_scenarios_prod_candidate_neutralized_2030.parquet --feature-output data/hpfc_scenario_features_prod_candidate_neutralized_2030.parquet --audit-output data/electrification_scenarios_neutralization_audit_2030.csv --report .planning/phases/13-lt-electrification-scenario-shape/LT-NEUTRALIZATION-POLICY.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/import_bfe_energiedashboard.py --raw-dir C:\Users\jbattaglia\pfc_local_data\scenarios\bfe_energiedashboard --output data/bfe_energiedashboard_daily.parquet --report .planning/phases/13-lt-electrification-scenario-shape/BFE-ENERGYDASHBOARD.md --ingested-at-utc 2026-06-12T00:00:00Z
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/audit_bfe_opendata_catalog.py --output data/bfe_opendata_catalog_audit.csv --report .planning/phases/13-lt-electrification-scenario-shape/BFE-OPENDATA-CATALOG-AUDIT.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/import_bfe_structural_actuals.py --raw-dir C:\Users\jbattaglia\pfc_local_data\scenarios\bfe_structural_actuals --ingested-at-utc 2026-06-12T00:00:00Z
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/validate_scenario_governance.py --inventory data/electrification_scenarios_prod_candidate_neutralized_2030.parquet --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml --vintage 2026-06-12 --countries CH,DE,FR,IT,AT --scenarios slow,central,fast --years 2030 --mode local-test --report .planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-GOVERNANCE-GATE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_local_test_ch_pfc.py --inventory data/electrification_scenarios_prod_candidate_neutralized_2030.parquet --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml --vintage 2026-06-12 --market CH --start-date 2030-01-01 --horizon-days 365 --summary .planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-CH-PFC.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/rebuild_forwards_history.py --history "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Hist.xlsx" --yearly "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx"
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/export_local_test_ch_hourly_csv.py --valuation-date 2026-06-12 --local-start-date 2026-06-13 --local-end-date 2030-12-31 --output output/ch_pfc_hourly_20260613_20301231.csv --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-CSV-20260613-20301231.md --prefix local_test_ch_pfc_20260613_20301231
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/export_local_test_ch_hourly_csv.py --valuation-date 2026-06-12 --local-start-date 2026-06-13 --local-end-date 2030-12-31 --output output/ch_pfc_hourly_20260613_20301231_v4_shape.csv --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-CSV-20260613-20301231-V4-SHAPE.md --prefix local_test_ch_pfc_20260613_20301231_v2_shape --skip-build --enable-structural-shape-upgrade --structural-shape-upgrade-intensity 1.0 --structural-scenario-spread-intensity 2.0
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/audit_ch_pfc_hourly_shape.py --csv output/ch_pfc_hourly_20260613_20301231_v4_shape.csv --forwards data/eex_forwards_history.parquet --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-SHAPE-AUDIT-20260613-20301231-V4.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/export_local_test_ch_hourly_csv.py --valuation-date 2026-06-12 --local-start-date 2026-06-13 --local-end-date 2030-12-31 --output output/ch_pfc_hourly_20260613_20301231_v5_negative_capture.csv --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-CSV-20260613-20301231-V5-NEGATIVE-CAPTURE.md --prefix local_test_ch_pfc_20260613_20301231_v2_shape --skip-build --enable-structural-shape-upgrade --structural-shape-upgrade-intensity 1.0 --structural-scenario-spread-intensity 2.0 --enable-negative-price-capture --negative-price-capture-intensity 1.0 --negative-price-floor -30
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/audit_ch_pfc_hourly_shape.py --csv output/ch_pfc_hourly_20260613_20301231_v5_negative_capture.csv --forwards data/eex_forwards_history.parquet --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-SHAPE-AUDIT-20260613-20301231-V5-NEGATIVE-CAPTURE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_ch_hfc_validation_workbook.py --csv output/ch_hfc_hourly_20260616_20301231_v5_negative_capture.csv --forwards data/eex_forwards_history.parquet --output output/ch_hfc_hourly_20260616_20301231_validation.xlsx --report .planning/phases/13-lt-electrification-scenario-shape/CH-HFC-HOURLY-VALIDATION-WORKBOOK-20260616-20301231.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/export_local_test_ch_hourly_csv.py --valuation-date 2026-06-15 --local-start-date 2026-06-16 --local-end-date 2030-12-31 --output output/ch_hfc_hourly_20260616_20301231_v10_negative_tail_peak_shape.csv --report .planning/phases/13-lt-electrification-scenario-shape/CH-HFC-HOURLY-CSV-20260616-20301231-V10-NEGATIVE-TAIL-PEAK-SHAPE.md --prefix local_test_ch_hfc_20260616_20301231_v10_negative_tail_peak_shape --skip-build --fan-chart-output output/local_test_ch_hfc_20260616_20301231_v6_annual_only_guard_structural_fan_chart.parquet --enable-structural-shape-upgrade --structural-shape-upgrade-intensity 1.0 --structural-scenario-spread-intensity 1.9 --enable-negative-price-capture --negative-price-capture-intensity 1.0 --negative-price-floor -30 --enable-post-calibration-negative-rebalancer --post-calibration-negative-rebalancer-intensity 0.75 --enable-post-calibration-peak-shape-rebalancer --post-calibration-peak-shape-intensity 1.0 --max-weighted-negative-hours 0 --disable-cascade-trend-for-annual-only
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/audit_ch_pfc_hourly_shape.py --csv output/ch_hfc_hourly_20260616_20301231_v10_negative_tail_peak_shape.csv --forwards data/eex_forwards_history.parquet --report .planning/phases/13-lt-electrification-scenario-shape/CH-HFC-HOURLY-SHAPE-AUDIT-20260616-20301231-V10-NEGATIVE-TAIL-PEAK-SHAPE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/audit_ch_hfc_seasonal_coherence.py --csv output/ch_hfc_hourly_20260616_20301231_v10_negative_tail_peak_shape.csv --forwards data/eex_forwards_history.parquet --report .planning/phases/13-lt-electrification-scenario-shape/CH-HFC-SEASONAL-COHERENCE-AUDIT-20260616-20301231-V10-NEGATIVE-TAIL-PEAK-SHAPE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/audit_ch_hfc_vs_spot_shape.py --csv output/ch_hfc_hourly_20260616_20301231_v10_negative_tail_peak_shape.csv --spot data/epex_hourly.parquet --report .planning/phases/13-lt-electrification-scenario-shape/CH-HFC-VS-SPOT-SHAPE-AUDIT-20260616-20301231-V10-NEGATIVE-TAIL-PEAK-SHAPE.md
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/build_ch_hfc_validation_workbook.py --csv output/ch_hfc_hourly_20260616_20301231_v10_negative_tail_peak_shape.csv --forwards data/eex_forwards_history.parquet --output output/ch_hfc_hourly_20260616_20301231_validation_charts_v10_negative_tail_peak_shape.xlsx --report .planning/phases/13-lt-electrification-scenario-shape/CH-HFC-HOURLY-VALIDATION-WORKBOOK-20260616-20301231-V10-NEGATIVE-TAIL-PEAK-SHAPE.md --no-tables
```

## Results

| check | result |
|---|---:|
| `test_electrification_scenarios_data.py` | 13 passed |
| `test_electrification_shape.py` | production guard tests extended for missing critical values |
| `test_intraday_amplitude.py` | 7 passed |
| `test_validate_electrification_scenarios_script.py` | production gate tests extended for missing critical values |
| `test_build_hpfc_scenario_features_script.py` | 1 passed |
| `test_build_ep2050_multi_scenario_pfc_script.py` | 3 passed |
| `test_materialize_lt_data_contract_script.py` | 2 passed |
| `test_import_tyndp2024_supply_inputs_script.py` | 2 passed |
| `test_import_tyndp2024_demand_outputs_script.py` | 2 passed |
| `test_build_tyndp2024_neighbor_demand_bridge_script.py` | 1 passed |
| `test_audit_ntc_baseline_inputs_script.py` | 2 passed |
| `test_compose_lt_scenario_inventory_script.py` | 2 passed |
| `test_validate_scenario_governance_script.py` | 10 passed |
| `test_build_lt_data_gap_register_script.py` | 1 passed |
| `test_build_scenario_governance_approval_pack_script.py` | 1 passed |
| `test_bridge_lt_p0_structural_fields_script.py` | 1 passed |
| `test_apply_swissgrid_ntc_baseline_script.py` | 2 passed |
| `test_apply_ember_yearly_baseline_script.py` | 2 passed |
| `test_apply_lt_neutralization_policy_script.py` | 2 passed |
| `test_import_bfe_energiedashboard_script.py` | 2 passed |
| `test_audit_bfe_opendata_catalog_script.py` | 2 passed |
| `test_import_bfe_structural_actuals_script.py` | 2 passed |
| `test_build_local_test_ch_pfc_script.py` | 2 passed |
| `test_export_local_test_ch_hourly_csv_script.py` | 7 passed |
| `test_audit_ch_pfc_hourly_shape_script.py` | 1 passed |
| `test_audit_ch_hfc_vs_spot_shape_script.py` | 1 passed |
| `test_build_ch_hfc_validation_workbook_script.py` | 1 passed |
| `test_phase10_reproducibility.py` | 1 passed; flag-OFF reproducibility gate `atol=1e-12, rtol=0` |
| `test_ep2050_import.py` | 1 passed |
| `test_enrich_electrification_scenarios_script.py` | 3 passed |
| combined Phase 13 prod-readiness tests | 107 passed with EP2050, enrichment, multi-scenario, materialization, production gates, TYNDP importers, neighbour demand bridge, NTC audit, partial composer, scenario governance, local-test agent governance, gap register, approval pack, source-provenance, zero/bounds governance, P0 structural bridge, Swissgrid NTC baseline, Ember yearly baseline, explicit neutralisation policy, BFE Energiedashboard import, full BFE opendata.swiss catalogue audit, BFE structural actuals import, local-test CH PFC runner, hourly CSV export, EEX previous-business-day freshness guard, hourly shape audit, bounded negative-price capture and Excel validation workbook |
| selected perfect-foresight tests | 3 passed, 1 existing `pytest.mark.slow` warning |
| py_compile | OK |
| `git diff --check` | OK, CRLF warnings only |
| `sota_electrification --no-figures` | completed; scenario store empty so layer no-op |
| `import_ep2050_hourly.py` local run | completed; 946,080 hourly rows and 6 annual/gold rows |
| `enrich_electrification_scenarios.py` local run | completed; 6 proxy-enriched rows |
| `build_first_ep2050_pfc.py` local run | completed; 35,040 rows, no NaN, complete 15-min UTC grid |
| first PFC scenario gate | OK; 2 as-of rows covering years 2030 and 2031 |
| first enriched PFC scenario gate | OK with `--require-recommended`; no missing recommended columns |
| multi-scenario PFC runner | completed; 3 x 35,040 rows plus structural fan chart |
| multi-scenario scenario gate | OK with `--require-recommended`; 6 as-of rows covering slow/central/fast for 2030 and 2031 |
| local canonical LT data materialization | completed; canonical scenario/features paths written for 2027-2031 |
| local canonical LT data gate | OK with `--require-recommended`; 15 as-of rows covering slow/central/fast for 2027-2031 |
| local canonical LT production gate | FAILED as expected; local proxy is CH-only and not final FMV production data |
| TYNDP 2024 Supply import | completed; 30 official partial rows covering CH/DE/FR/IT/AT x slow/central/fast x 2030/2040 |
| TYNDP 2024 Supply production gate | FAILED as expected; official supply component still lacks demand, NTC, hydro, EV, heat-pump, battery power and flexibility fields |
| TYNDP 2024 Demand import | completed; 16 official partial rows covering AT/DE/FR/IT x DE/GA x 2040/2050 |
| TYNDP 2024 Demand production gate | FAILED as expected; CH demand is absent and raw DE/GA scenarios are not mapped to governed slow/central/fast |
| neighbour 2030 demand bridge | completed; 12 rows for AT/DE/FR/IT x slow/central/fast x 2030; peak/winter remain null because local neighbour load columns are empty |
| NTC baseline audit | completed; local ENTSO-E NTC usable=NO because all local NTC columns have zero observations |
| Swissgrid NTC baseline | completed; 60 NTC cells filled from official observed 2026 Swissgrid CSV using conservative symmetric medians |
| Ember yearly baseline | completed; 69 cells filled for hydro, net imports, gas/coal and dispatchable lower-bound where public source values are present and non-zero |
| composed partial inventory | completed; 15 rows covering CH/DE/FR/IT/AT x slow/central/fast x 2030, now with neighbour `demand_twh` |
| composed partial production gate | coverage OK, production gate FAILED as expected on remaining critical gaps and partial/proxy flags |
| scenario governance gate | FAILED as expected; manifest is draft, approval metadata missing and flags are partial/proxy; no critical numeric null blockers remain on the neutralized candidate |
| LT data gap register | completed; 18 blocking rows on the neutralized candidate: 15 proxy/partial quality flags and 3 governance decision items |
| P0 structural bridge | completed; 96 cells filled for peak/winter, PV/Wind TWh, battery power, EV/PAC and nuclear lower-bound dispatchable |
| explicit LT neutralization policy | completed; 66 P1 values neutralized with field-level `*_zero_justification`, no overwrite of non-null source values |
| BFE Energiedashboard import | completed; 70,610 official daily rows for CH production, consumption, gross import/export flows, net imports, spot base and SDSC forecast snapshot |
| BFE opendata.swiss catalogue audit | completed; 146 BFE datasets scanned into `data/bfe_opendata_catalog_audit.csv`; classification found 54 P0, 39 P1, 3 P2 and 50 P3 candidates; P0/P1 candidates documented for reservoir, WASTA, production plants, electricity balances, EV charging and PV datasets |
| BFE structural actuals import | completed; 20,700 weekly reservoir rows, 327,223 production-plant rows, 728 WASTA hydro rows and 12 model-facing CH structural actuals |
| local-test agent governance gate | OK; 15 effective rows, proxy/partial flags allowed only for local/test and `approved_for_production=false` |
| local-test CH PFC runner | completed; 3 x 35,040 CH curves plus 35,040-row weighted structural fan chart |
| EEX forwards refresh | completed; `data/eex_forwards_history.parquet` rebuilt from desk files with CH/DE/FR/AT/IT latest BASE date 2026-06-11 |
| hourly CH CSV export | completed; 39,913 Europe/Zurich hourly rows from 2026-06-13 00:00 to 2030-12-31 23:00, post-calibrated to required 2026-06-11 EEX CH BASE monthly/quarterly/calendar products |
| hourly CH CSV V4 shape upgrade | completed; local/test-only overlay ON, 39,913 rows, EEX residual 0.000000, score 9.00/10 |
| hourly CH CSV V5 negative-price capture | completed; local/test-only capture ON, 82 structural p10 negative hours, 0 weighted-mean negative hours, EEX residual 0.000000, score 9.00/10 |
| hourly CH HFC V10 negative tail and peak shape | completed; 39,841 Europe/Zurich hourly rows from 2026-06-16 00:00 to 2030-12-31 23:00, EEX date 2026-06-12, 377 fast/P10 negative hours, 0 weighted-mean negative hours, min fast/P10 -10.765951 EUR/MWh, EEX residual 0.000000, shape score 9.00/10 |
| HFC V10 seasonal coherence audit | completed; critical 0, warning 0, January remains above October for annual-only 2029/2030 |
| HFC V10 vs CH spot shape audit | completed; score 9.00/10, 2030 normalized month-hour correlation vs historical CH spot 0.914, positive peak/offpeak, winter/summer, January/October and evening/midday spreads |
| HFC V10 validation workbook | completed; `output/ch_hfc_hourly_20260616_20301231_validation_charts_v10_negative_tail_peak_shape.xlsx`, 15 sheets, 4 charts, 0 Excel structured tables, hidden chart data sheet |
| HFC validation workbook | completed; `output/ch_hfc_hourly_20260616_20301231_validation.xlsx`, 13 sheets, 4 charts, Excel-native date cells and hidden chart data sheet |
| expert reviews | completed; quant, data-engineering and model-risk agents all recommend NO-GO prod and OK controlled diagnostic |
| approval pack | completed; recommendation `NO-GO`, agents recorded as non-voting reviewers |
| source component links | OK on neutralized candidate; rows cite scenario source components plus baseline and neutralization components |
| zero/bounds governance | enabled in manifest; zero values require explicit justification except configured allowed fields |
| human data contract HTML | OK; `ENTSOEINGESTIONSPEC.html` parses with 12 sections including scenario governance |

## Agent Audit Findings

Two independent agents audited the current build.

| area | finding | action |
|---|---|---|
| neighbour demand bridge | acceptable only as `PARTIAL/PROXY/NON-PROD`; mapping DE/GA to slow/fast must be explicit | implemented with `internal_tyndp_demand_bridge_partial_proxy`; prod gate rejects it |
| NTC baseline | historical local NTC baseline would be proxy-only and `ntc_total` can overstate usable capacity | local ENTSO-E NTC rejected; Swissgrid CSV baseline added with `min(median export, median import)` by border |
| Ember yearly baseline | historical generation/capacity values are not 2030 scenario paths | added as `official_historical_baseline_proxy`; governed net import now satisfies the conditional cross-border balance; gross import/export remain optional diagnostics |
| neutralization policy | missing P1 flex and coal values cannot become silent zeros | added separate prod-candidate artefact with `*_zero_justification` columns and audit CSV |
| BFE Energiedashboard | useful historical official CH actuals, not LT scenarios | added daily normalized parquet and report; forecast feed marked ingest-time snapshot only |
| all-NaN demand import | summing all missing cells could silently write `0.0` | fixed to preserve `NaN`; regression test added |
| production gate semantics | null checks used full frame instead of effective latest as-of rows | fixed to validate latest scoped rows; regression test added |
| partial features | feature output from partial rows is diagnostic-only | composed inventory/report keeps `partial/proxy` flags and strict prod gate fails |

Single-run diagnostic granularity ladder under empty scenario store:

| anchor | monthly_corr |
|---|---:|
| `pf_cal` | 0.8241 |
| `pf_cal_quarter` | 0.9275 |
| `market` | 0.8624 |

The diagnostic proves wiring and graceful no-op behavior. It does not validate
structural 2027/2030 uplift because no official scenario inventory is loaded.

## Unit Properties Covered

* `publication_date > vintage` rows are excluded by `asof(vintage)`.
* A store containing only future-published rows raises on lookup.
* missing required scenario columns fail fast.
* production data validator rejects bad delivery months, negative capacities and
  non-finite capacities.
* production data validator rejects `managed_charging_share` outside `[0, 1]`.
* gold HPFC feature derivation normalizes raw PV/wind/battery/EV/PAC/hydro/import
  drivers into penetration indicators.
* scenario coverage helper accepts year-level rows and reports missing scenarios.
* empty scenario store returns identity.
* empty scenario store raises when scenario data is explicitly required.
* scenario inventory script returns `0` for covered vintages and `2` for missing
  coverage while writing a Markdown gate report without optional dependencies.
* gold feature build script writes a vintage-filtered parquet and excludes
  future-published scenario rows.
* OFEN EP2050+ annual aggregation converts workbook `GWh/h` sums to `*_twh`
  columns by dividing by 1000.
* yearly scenario rows are linearly interpolated on the EP2050/TYNDP 5-year grid
  while preserving publication/scenario/source provenance and marking
  interpolated quality flags.
* the first-PFC runner forbids clamp/extrapolation outside bracketing published
  years; 2031 is accepted only because 2030 and 2035 are both loaded.
* local enrichment preserves official source values, fills only missing
  recommended fields, stamps `proxy_enriched`, and validates with
  `require_recommended=True`.
* PV-only scenario lowers the synthetic midday block relative to night.
* battery energy/power refills midday and compresses evening vs PV-only.
* wind lowers the winter night block in a controlled fixture.
* CH hydro flexibility compresses the winter evening ramp.
* managed EV charging reduces the EV evening uplift.
* local-day mean of `f_H` is preserved to `< 1e-12`.
* signed `f_H` values keep their sign under modulation.
* DST short-day input means are preserved rather than forced to 1.0.
* weighted structural fan chart quantiles are ordered and weighted mean is correct.
* 2030 slow/central/fast scenario divergence is wider than 2027 on the targeted
  midday block in a controlled fixture.
* enriched OFEN `WWB` and `ZERO_Basis` rows are mapped to governed
  `slow/central/fast` without silent zero filling; `central` is explicitly marked
  as an internal midpoint.
* multi-scenario fan chart output contains `curve_slow`, `curve_central`,
  `curve_fast`, `weighted_mean`, `structural_p10`, `structural_p50`,
  `structural_p90`, and ordered structural width.
* canonical Phase 13 model paths can be materialized locally from the governed
  proxy inventory and validated strictly before use.
* final-production gate rejects local proxy/internal scenario rows and missing
  multi-country production columns plus incomplete country/scenario/year
  coverage while leaving smoke/prod-readiness gates unchanged.
* model-side production guard is OFF by default and rejects non-production
  scenario inventories, including missing country/scenario/year rows, when
  explicitly required by `PFCAssembler`.
* `PFCAssembler.enable_electrification_shape` defaults to `False`.

## Local EP2050+ Extraction Smoke Run

Raw cache:

```text
C:\Users\jbattaglia\pfc_local_data\ep2050
```

Generated local outputs:

| file | rows |
|---|---:|
| `data/silver_ep2050_hourly.parquet` | 946,080 |
| `data/electrification_scenarios_ep2050.parquet` | 6 |
| `data/hpfc_scenario_features_ep2050.parquet` | 6 |
| `data/electrification_scenarios_ep2050_enriched.parquet` | 6 |
| `data/hpfc_scenario_features_ep2050_enriched.parquet` | 6 |

Coverage: scenarios `ZERO_Basis`, `WWB`; years `2025`, `2030`, `2035`.

## First Local PFC Smoke Run

Generated local outputs:

| file | rows |
|---|---:|
| `output/first_ep2050_pfc_2030_zero_basis.parquet` | 35,040 |
| `output/first_ep2050_pfc_2030_zero_basis_scenario_expanded.parquet` | 2 |

The first curve is CH, delivery year 2030, scenario `ZERO_Basis`, latest EEX
BASE snapshot `2026-06-05`, with `enable_electrification_shape=True`,
`require_electrification_scenarios=True`, and
`enable_intraday_amplitude_shrinkage=True`. The expanded scenario table contains
official 2030 assumptions plus an interpolated 2031 boundary row so the UTC
delivery horizon covers the Europe/Zurich local-year crossover without silently
dropping scenario coverage.

Summary:

| metric | value |
|---|---:|
| rows | 35,040 |
| mean | 68.7398 |
| min | 19.2718 |
| p05 | 31.9446 |
| p95 | 105.5806 |
| max | 122.7220 |
| midday_mean | 61.4799 |
| evening_mean | 73.4628 |
| night_mean | 69.2726 |

The runner now fails fast if the output is not a complete 15-minute UTC grid,
contains non-finite `price_shape` values, misses `price_shape`, or violates
`p10 <= p90` when structural fan-chart columns are present. Full details are in
`FIRST-PFC.md`; the scenario coverage gate is in `FIRST-PFC-SCENARIO-GATE.md`.

## First Enriched Local PFC

Generated local outputs:

| file | rows |
|---|---:|
| `output/first_ep2050_pfc_2030_zero_basis_enriched.parquet` | 35,040 |
| `output/first_ep2050_pfc_2030_zero_basis_enriched_scenario_expanded.parquet` | 2 |

This curve uses `data/electrification_scenarios_ep2050_enriched.parquet`, profile
`ch_first_pfc_proxy_v0`, and validates the expanded scenario table with
`--require-recommended`.

Summary:

| metric | value |
|---|---:|
| rows | 35,040 |
| mean | 68.7398 |
| min | 19.2875 |
| p05 | 32.5818 |
| p95 | 105.9458 |
| max | 124.4428 |
| midday_mean | 63.1455 |
| evening_mean | 70.7542 |
| night_mean | 69.2224 |

Impact versus the OFEN-only first PFC:

| metric | value |
|---|---:|
| mean_diff | ~0.0000 |
| min_diff | -4.6723 |
| max_diff | 3.3901 |
| mean_abs_diff | 1.0420 |
| p95_abs_diff | 3.2999 |

## Multi-Scenario EP2050 PFC and Structural Fan Chart

Generated local outputs:

| file | rows |
|---|---:|
| `data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet` | 9 |
| `data/hpfc_scenario_features_ep2050_enriched_slow_central_fast.parquet` | 9 |
| `output/ep2050_pfc_2030_scenario_expanded.parquet` | 6 |
| `output/ep2050_pfc_2030_slow.parquet` | 35,040 |
| `output/ep2050_pfc_2030_central.parquet` | 35,040 |
| `output/ep2050_pfc_2030_fast.parquet` | 35,040 |
| `output/ep2050_pfc_2030_structural_fan_chart.parquet` | 35,040 |

Mapping profile: `ep2050_slow_central_fast_mapping_v0`.

| target scenario | source rule | weight |
|---|---|---:|
| `slow` | enriched OFEN `WWB` alias | 0.25 |
| `central` | explicit midpoint between enriched `WWB` and `ZERO_Basis` | 0.50 |
| `fast` | enriched OFEN `ZERO_Basis` alias | 0.25 |

Scenario gate:

| check | result |
|---|---|
| expanded table vintage | `2026-06-05` |
| scenarios | `slow`, `central`, `fast` |
| delivery years | `2030`, `2031` |
| `--require-recommended` | OK |
| missing recommended columns | none |

Curve summary:

| scenario | mean | min | p05 | p95 | max | midday_mean | evening_mean | night_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `slow` | 68.7398 | 19.2896 | 32.6851 | 106.0907 | 124.3149 | 63.3973 | 70.5822 | 69.1367 |
| `central` | 68.7398 | 19.2885 | 32.6541 | 105.9442 | 124.3791 | 63.2712 | 70.6682 | 69.1797 |
| `fast` | 68.7398 | 19.2875 | 32.5818 | 105.9458 | 124.4428 | 63.1455 | 70.7542 | 69.2224 |

Structural fan chart:

| metric | value |
|---|---:|
| rows | 35,040 |
| weighted mean | 68.7398 |
| mean structural p10 | 68.6747 |
| mean structural p90 | 68.8048 |
| mean structural width | 0.1301 |
| p95 structural width | 0.3822 |
| max structural width | 0.5289 |

Full report: `MULTI-SCENARIO-PFC.md`. Strict scenario gate:
`MULTI-SCENARIO-GATE.md`.

## Local Canonical Data Contract

Generated local outputs:

| file | rows |
|---|---:|
| `data/electrification_scenarios.parquet` | 15 |
| `data/hpfc_scenario_features.parquet` | 15 |

The materialization script also creates the local cache tree under:

```text
C:\Users\jbattaglia\pfc_local_data
```

Report:
`.planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-LOCAL.md`.

Strict gate:
`.planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-GATE.md`.

Final-production gate:
`.planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-PRODUCTION-GATE.md`.
It intentionally fails on the current local proxy with:

| issue | detail |
|---|---|
| missing production columns | `track`, `measurement_date`, `scenario_edition`, `ingested_at_utc`, `ntc_ch_at_gw`, `gas_eur_mwh`, `coal_eur_mwh`, `co2_eur_t`, `electrolysis_twh`, `p2x_gw`, `dsm_gw` |
| missing countries | `DE`, `FR`, `IT`, `AT` |
| missing country/scenario/year coverage | `DE`, `FR`, `IT`, `AT` for `slow`, `central`, `fast` across 2027-2031 |
| unacceptable quality flags | all current `proxy` / `internal` rows |

The default scenario store path is now usable locally by the model when
`enable_electrification_shape=True` and no explicit
`electrification_scenario_path` is provided. A smoke check with
`require_scenario_data=True` loaded `central` for 2030 from the canonical file
and preserved the one-day mean:

| metric | value |
|---|---:|
| rows | 96 |
| mean | 1.0000 |
| min | 0.9108 |
| max | 1.0566 |

The stricter model-side guard is exposed as:

```python
PFCAssembler(
    ...,
    require_production_electrification_scenarios=True,
)
```

It remains default-OFF. Dedicated tests confirm that governed multi-country rows
pass, missing country/scenario/year rows fail, proxy quality flags fail, and the
assembler constructor default is `False`.

## Audit Fixes

An expert read-only audit flagged two important issues in the first
implementation:

1. signed or negative `f_H` values were implicitly forced positive by
   `log(clip(f_H))`;
2. local-day normalization forced the output mean to 1.0 instead of preserving
   the incoming day mean.

Both were fixed by applying the adjustment in signed-log magnitude space and
renormalizing to the input local-day mean. Dedicated regression tests now cover
negative `f_H` and a 23-hour DST day.

## Known Limitations

This is a governed infrastructure slice, not yet a final Phase 13 quant signal:

* coefficients are conservative monotone defaults, not fitted on a curated
  multi-country driver history;
* weighted scenario fan-chart utility is implemented, but production reporting
  and stochastic-structural uncertainty combination remain future work;
* the local `slow/central/fast` workflow is bounded by two OFEN scenarios
  (`WWB`, `ZERO_Basis`); the `central` path is an internal midpoint, not an
  official third OFEN publication;
* official OFEN/TYNDP/Pronovo/BNetzA scenario ingestion is not yet loaded;
* full `--ab` has not been extended to include `sota_electrification` as a fifth
  line because, without scenario data, it would be a no-op and add runtime only.

Next gate: load a versioned scenario inventory through local parquet or
Databricks, run the inventory validator at the pricing vintage, add faux-future
validation, then decide whether the layer graduates from infrastructure to
signal.
