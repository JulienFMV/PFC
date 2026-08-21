# EP2050 Multi-Scenario PFC 2030

* market: `CH`
* delivery start: `2030-01-01`
* horizon days: `365`
* EEX forward date: `2026-06-05`
* source enriched inventory: `data/electrification_scenarios_ep2050_enriched.parquet`
* mapped inventory: `data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet`
* gold features: `data/hpfc_scenario_features_ep2050_enriched_slow_central_fast.parquet`
* expanded scenario path: `output\ep2050_pfc_2030_scenario_expanded.parquet`
* fan chart: `output\ep2050_pfc_2030_structural_fan_chart.parquet`
* electrification shape: `ON`
* intraday amplitude shrinkage: `ON`
* missing scenario behavior: `fail-fast`
* scenario weights: `slow=0.2500, central=0.5000, fast=0.2500`
* mapping profile: `ep2050_slow_central_fast_mapping_v0`

## PFC Outputs

| scenario | path |
|---|---|
| slow | output\ep2050_pfc_2030_slow.parquet |
| central | output\ep2050_pfc_2030_central.parquet |
| fast | output\ep2050_pfc_2030_fast.parquet |

## Price Summary

| scenario | mean | min | p05 | p95 | max | midday_mean | evening_mean | night_mean |
|---|---|---|---|---|---|---|---|---|
| slow | 68.7398 | 19.2896 | 32.6851 | 106.0907 | 124.3149 | 63.3973 | 70.5822 | 69.1367 |
| central | 68.7398 | 19.2885 | 32.6541 | 105.9442 | 124.3791 | 63.2712 | 70.6682 | 69.1797 |
| fast | 68.7398 | 19.2875 | 32.5818 | 105.9458 | 124.4428 | 63.1455 | 70.7542 | 69.2224 |

## Structural Fan Chart Summary

| rows | weighted_mean | structural_p10_mean | structural_p90_mean | structural_width_mean | structural_width_p95 | max_width |
|---|---|---|---|---|---|---|
| 35040.0000 | 68.7398 | 68.6747 | 68.8048 | 0.1301 | 0.3822 | 0.5289 |

## Scenario Inventory Used

| scenario | original_scenario | delivery_year | scenario_weight | quality_flag | demand_twh | pv_twh | wind_twh | battery_power_gw | battery_energy_gwh | ev_twh | heatpump_twh | managed_charging_share | hydro_reservoir_twh | net_import_twh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| central | midpoint(WWB,ZERO_Basis) | 2030 | 0.5000 | internal_midpoint_proxy_enriched | 70.3842 | 7.7341 | 0.4524 | 2.0000 | 5.0000 | 0.1924 | 3.8496 | 0.3500 | 3.9709 | 9.0550 |
| fast | ZERO_Basis | 2030 | 0.2500 | official_proxy_enriched_scenario_mapped | 70.4407 | 8.6778 | 0.5780 | 2.0000 | 5.0000 | 0.2719 | 4.4931 | 0.3500 | 3.9781 | 7.4853 |
| slow | WWB | 2030 | 0.2500 | official_proxy_enriched_scenario_mapped | 70.3276 | 6.7904 | 0.3268 | 2.0000 | 5.0000 | 0.1128 | 3.2061 | 0.3500 | 3.9636 | 10.6247 |

## Governance Notes

* `slow` aliases OFEN `WWB` after the local proxy enrichment.
* `fast` aliases OFEN `ZERO_Basis` after the local proxy enrichment.
* `central` is an explicit midpoint between `WWB` and `ZERO_Basis`, stamped as an internal proxy assumption.
* All scenario rows keep `publication_date <= vintage`; production/smoke builds use `require_electrification_scenarios=True`.
* This is still a local proxy workflow until TYNDP, Pronovo, MaStR and governed NTC feeds are connected.
