# Local-Test CH PFC 2030

* status: `agent-approved local/test only`
* production approval: `NO`
* production activation allowed: `NO`
* governance report: `.planning\phases\13-lt-electrification-scenario-shape\LOCAL-TEST-GOVERNANCE-GATE.md`
* source inventory: `data/electrification_scenarios_prod_candidate_neutralized_2030.parquet`
* expanded scenario path: `output/local_test_ch_pfc_2030_scenario_expanded.parquet`
* feature output: `data/hpfc_scenario_features_local_test_2030.parquet`
* fan chart: `output\local_test_ch_pfc_2030_structural_fan_chart.parquet`
* market: `CH`
* start date UTC: `2030-01-01`
* horizon days: `365`
* EEX forward date: `2026-06-05`
* scenario weights: `slow=0.2500, central=0.5000, fast=0.2500`

## PFC Outputs

| scenario | path |
|---|---|
| slow | output\local_test_ch_pfc_2030_slow.parquet |
| central | output\local_test_ch_pfc_2030_central.parquet |
| fast | output\local_test_ch_pfc_2030_fast.parquet |

## Price Summary

| scenario | mean | min | p05 | p95 | max | midday_mean | evening_mean | night_mean |
|---|---|---|---|---|---|---|---|---|
| slow | 68.7398 | 19.2840 | 32.4119 | 105.9044 | 124.1458 | 62.7839 | 71.2973 | 69.2571 |
| central | 68.7398 | 19.2811 | 32.2993 | 105.9026 | 124.2926 | 62.4696 | 71.5419 | 69.3596 |
| fast | 68.7398 | 19.2780 | 32.1647 | 105.8288 | 124.4613 | 62.1221 | 71.8032 | 69.4787 |

## Structural Fan Chart

| rows | weighted_mean | structural_width_mean | structural_width_p95 | structural_width_max |
|---|---|---|---|---|
| 35040.0000 | 68.7398 | 0.3445 | 0.9886 | 1.3482 |

## Expanded CH Scenario Rows

| scenario | delivery_year | scenario_weight | quality_flag | demand_twh | peak_load_gw | pv_gw | pv_twh | wind_gw | wind_twh | battery_power_gw | battery_energy_gwh | ev_twh | heatpump_twh | hydro_reservoir_twh | net_import_twh | ntc_ch_de_gw | ntc_ch_fr_gw | ntc_ch_it_gw | ntc_ch_at_gw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| central | 2030 | 0.5000 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | 70.3842 | 14.9389 | 9.7703 | 10.2705 | 0.3100 | 0.5431 | 2.0000 | 0.6106 | 0.1924 | 3.8496 | 3.9709 | 9.0550 | 0.9500 | 1.3000 | 1.8100 | 0.9000 |
| central | 2031 | 0.5000 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit_interpolated | 70.3842 | 14.9389 | 9.7703 | 10.2705 | 0.3100 | 0.5431 | 2.0000 | 0.6106 | 0.1924 | 3.8496 | 3.9709 | 9.0550 | 0.9500 | 1.3000 | 1.8100 | 0.9000 |
| fast | 2030 | 0.2500 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | 70.4407 | 15.8620 | 12.2100 | 12.8352 | 0.3100 | 0.5431 | 2.0000 | 0.7631 | 0.2719 | 4.4931 | 3.9781 | 7.4853 | 0.9500 | 1.3000 | 1.8100 | 0.9000 |
| fast | 2031 | 0.2500 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit_interpolated | 70.4407 | 15.8620 | 12.2100 | 12.8352 | 0.3100 | 0.5431 | 2.0000 | 0.7631 | 0.2719 | 4.4931 | 3.9781 | 7.4853 | 0.9500 | 1.3000 | 1.8100 | 0.9000 |
| slow | 2030 | 0.2500 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | 70.3276 | 14.0159 | 7.6500 | 8.0417 | 0.1800 | 0.3154 | 2.0000 | 0.6106 | 0.1128 | 3.2061 | 3.9636 | 10.6247 | 0.9500 | 1.3000 | 1.8100 | 0.9000 |
| slow | 2031 | 0.2500 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit_interpolated | 70.3276 | 14.0159 | 7.6500 | 8.0417 | 0.1800 | 0.3154 | 2.0000 | 0.6106 | 0.1128 | 3.2061 | 3.9636 | 10.6247 | 0.9500 | 1.3000 | 1.8100 | 0.9000 |

## Limitations

* This curve is suitable for local validation, diagnostics and model review only.
* Agent approval replaces human approval only for local/test work.
* Production FMV use still requires the production governance gate with accountable human sign-off.
* Proxy/partial/internal quality flags remain visible and are not relabelled as production-governed.
