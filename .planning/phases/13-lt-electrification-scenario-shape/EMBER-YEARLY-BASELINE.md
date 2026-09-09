# Ember Yearly Baseline

* raw source: `C:\Users\jbattaglia\pfc_local_data\scenarios\ember_yearly\yearly_full_release_long_format.csv`
* scenario output: `data\electrification_scenarios_composed_p0_public_sources_2030.parquet`
* component output: `data\electrification_scenarios_ember_yearly_baseline_2026.parquet`
* status: `NON-PRODUCTION / OFFICIAL HISTORICAL BASELINE / PROXY`
* publication date used for vintage checks: `2026-04-24`
* filled fields: `hydro_twh`, `hydro_capacity_gw`, `net_import_twh`, `gas_gw`, `coal_gw`, `dispatchable_gw` where source values are present and non-zero.
* deliberately not filled: `import_twh`, `export_twh`, `hydro_reservoir_twh` because Ember yearly data does not provide governed gross flows or reservoir energy capacity.

## Baseline Values

| publication_date | source | component_id | country | area | field | value | source_year | method | quality_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | AT | Austria | dispatchable_gw | 4.92 | 2024 | latest positive gas+coal+nuclear capacity lower-bound | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | AT | Austria | gas_gw | 4.92 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | AT | Austria | hydro_capacity_gw | 15.07 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | AT | Austria | hydro_twh | 37.93 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | AT | Austria | net_import_twh | 2.62 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | CH | Switzerland | dispatchable_gw | 3.19 | 2024 | latest positive gas+coal+nuclear capacity lower-bound | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | CH | Switzerland | gas_gw | 0.23 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | CH | Switzerland | hydro_capacity_gw | 16.22 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | CH | Switzerland | hydro_twh | 33.99 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | CH | Switzerland | net_import_twh | 0.1 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | DE | Germany | coal_gw | 32.34 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | DE | Germany | dispatchable_gw | 66.27000000000001 | 2024 | latest positive gas+coal+nuclear capacity lower-bound | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | DE | Germany | gas_gw | 33.93 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | DE | Germany | hydro_capacity_gw | 5.84 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | DE | Germany | hydro_twh | 19.56 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | DE | Germany | net_import_twh | 19.8 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | FR | France | coal_gw | 1.91 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | FR | France | dispatchable_gw | 81.64 | 2024 | latest positive gas+coal+nuclear capacity lower-bound | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | FR | France | gas_gw | 18.33 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | FR | France | hydro_capacity_gw | 24.64 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | FR | France | hydro_twh | 59.41 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | FR | France | net_import_twh | -92.68 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | IT | Italy | coal_gw | 5.19 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | IT | Italy | dispatchable_gw | 62.39 | 2024 | latest positive gas+coal+nuclear capacity lower-bound | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | IT | Italy | gas_gw | 57.2 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | IT | Italy | hydro_capacity_gw | 18.97 | 2024 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | IT | Italy | hydro_twh | 41.7 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |
| 2026-04-24 00:00:00+00:00 | Ember yearly electricity data | ember_yearly_2026_baseline | IT | Italy | net_import_twh | 46.9 | 2025 | latest reported Ember yearly value | official_historical_baseline_proxy |

## Changed Scenario Cells

| country | scenario | delivery_year | column | value | source_year | method |
| --- | --- | --- | --- | --- | --- | --- |
| CH | central | 2030 | gas_gw | 0.23 | 2024 | latest reported Ember yearly value |
| CH | fast | 2030 | gas_gw | 0.23 | 2024 | latest reported Ember yearly value |
| CH | slow | 2030 | gas_gw | 0.23 | 2024 | latest reported Ember yearly value |
| DE | central | 2030 | hydro_twh | 19.56 | 2025 | latest reported Ember yearly value |
| DE | fast | 2030 | hydro_twh | 19.56 | 2025 | latest reported Ember yearly value |
| DE | slow | 2030 | hydro_twh | 19.56 | 2025 | latest reported Ember yearly value |
| DE | central | 2030 | hydro_capacity_gw | 5.84 | 2024 | latest reported Ember yearly value |
| DE | fast | 2030 | hydro_capacity_gw | 5.84 | 2024 | latest reported Ember yearly value |
| DE | slow | 2030 | hydro_capacity_gw | 5.84 | 2024 | latest reported Ember yearly value |
| DE | central | 2030 | net_import_twh | 19.8 | 2025 | latest reported Ember yearly value |
| DE | fast | 2030 | net_import_twh | 19.8 | 2025 | latest reported Ember yearly value |
| DE | slow | 2030 | net_import_twh | 19.8 | 2025 | latest reported Ember yearly value |
| DE | central | 2030 | gas_gw | 33.93 | 2024 | latest reported Ember yearly value |
| DE | fast | 2030 | gas_gw | 33.93 | 2024 | latest reported Ember yearly value |
| DE | slow | 2030 | gas_gw | 33.93 | 2024 | latest reported Ember yearly value |
| DE | central | 2030 | coal_gw | 32.34 | 2024 | latest reported Ember yearly value |
| DE | fast | 2030 | coal_gw | 32.34 | 2024 | latest reported Ember yearly value |
| DE | slow | 2030 | coal_gw | 32.34 | 2024 | latest reported Ember yearly value |
| DE | central | 2030 | dispatchable_gw | 66.27000000000001 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| DE | fast | 2030 | dispatchable_gw | 66.27000000000001 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| DE | slow | 2030 | dispatchable_gw | 66.27000000000001 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| FR | central | 2030 | hydro_twh | 59.41 | 2025 | latest reported Ember yearly value |
| FR | fast | 2030 | hydro_twh | 59.41 | 2025 | latest reported Ember yearly value |
| FR | slow | 2030 | hydro_twh | 59.41 | 2025 | latest reported Ember yearly value |
| FR | central | 2030 | hydro_capacity_gw | 24.64 | 2024 | latest reported Ember yearly value |
| FR | fast | 2030 | hydro_capacity_gw | 24.64 | 2024 | latest reported Ember yearly value |
| FR | slow | 2030 | hydro_capacity_gw | 24.64 | 2024 | latest reported Ember yearly value |
| FR | central | 2030 | net_import_twh | -92.68 | 2025 | latest reported Ember yearly value |
| FR | fast | 2030 | net_import_twh | -92.68 | 2025 | latest reported Ember yearly value |
| FR | slow | 2030 | net_import_twh | -92.68 | 2025 | latest reported Ember yearly value |
| FR | central | 2030 | gas_gw | 18.33 | 2024 | latest reported Ember yearly value |
| FR | fast | 2030 | gas_gw | 18.33 | 2024 | latest reported Ember yearly value |
| FR | slow | 2030 | gas_gw | 18.33 | 2024 | latest reported Ember yearly value |
| FR | central | 2030 | coal_gw | 1.91 | 2024 | latest reported Ember yearly value |
| FR | fast | 2030 | coal_gw | 1.91 | 2024 | latest reported Ember yearly value |
| FR | slow | 2030 | coal_gw | 1.91 | 2024 | latest reported Ember yearly value |
| IT | central | 2030 | hydro_twh | 41.7 | 2025 | latest reported Ember yearly value |
| IT | fast | 2030 | hydro_twh | 41.7 | 2025 | latest reported Ember yearly value |
| IT | slow | 2030 | hydro_twh | 41.7 | 2025 | latest reported Ember yearly value |
| IT | central | 2030 | hydro_capacity_gw | 18.97 | 2024 | latest reported Ember yearly value |
| IT | fast | 2030 | hydro_capacity_gw | 18.97 | 2024 | latest reported Ember yearly value |
| IT | slow | 2030 | hydro_capacity_gw | 18.97 | 2024 | latest reported Ember yearly value |
| IT | central | 2030 | net_import_twh | 46.9 | 2025 | latest reported Ember yearly value |
| IT | fast | 2030 | net_import_twh | 46.9 | 2025 | latest reported Ember yearly value |
| IT | slow | 2030 | net_import_twh | 46.9 | 2025 | latest reported Ember yearly value |
| IT | central | 2030 | gas_gw | 57.2 | 2024 | latest reported Ember yearly value |
| IT | fast | 2030 | gas_gw | 57.2 | 2024 | latest reported Ember yearly value |
| IT | slow | 2030 | gas_gw | 57.2 | 2024 | latest reported Ember yearly value |
| IT | central | 2030 | coal_gw | 5.19 | 2024 | latest reported Ember yearly value |
| IT | fast | 2030 | coal_gw | 5.19 | 2024 | latest reported Ember yearly value |
| IT | slow | 2030 | coal_gw | 5.19 | 2024 | latest reported Ember yearly value |
| IT | central | 2030 | dispatchable_gw | 62.39 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| IT | fast | 2030 | dispatchable_gw | 62.39 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| IT | slow | 2030 | dispatchable_gw | 62.39 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| AT | central | 2030 | hydro_twh | 37.93 | 2025 | latest reported Ember yearly value |
| AT | fast | 2030 | hydro_twh | 37.93 | 2025 | latest reported Ember yearly value |
| AT | slow | 2030 | hydro_twh | 37.93 | 2025 | latest reported Ember yearly value |
| AT | central | 2030 | hydro_capacity_gw | 15.07 | 2024 | latest reported Ember yearly value |
| AT | fast | 2030 | hydro_capacity_gw | 15.07 | 2024 | latest reported Ember yearly value |
| AT | slow | 2030 | hydro_capacity_gw | 15.07 | 2024 | latest reported Ember yearly value |
| AT | central | 2030 | net_import_twh | 2.62 | 2025 | latest reported Ember yearly value |
| AT | fast | 2030 | net_import_twh | 2.62 | 2025 | latest reported Ember yearly value |
| AT | slow | 2030 | net_import_twh | 2.62 | 2025 | latest reported Ember yearly value |
| AT | central | 2030 | gas_gw | 4.92 | 2024 | latest reported Ember yearly value |
| AT | fast | 2030 | gas_gw | 4.92 | 2024 | latest reported Ember yearly value |
| AT | slow | 2030 | gas_gw | 4.92 | 2024 | latest reported Ember yearly value |
| AT | central | 2030 | dispatchable_gw | 4.92 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| AT | fast | 2030 | dispatchable_gw | 4.92 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |
| AT | slow | 2030 | dispatchable_gw | 4.92 | 2024 | latest positive gas+coal+nuclear capacity lower-bound |

## Production Interpretation

This reduces numeric gaps with a true public source, but it remains a baseline projection proxy. Strict production governance must still fail until 2030 scenario values or explicit committee neutralisations are approved.
