# Electrification Scenario Inventory Validation

* vintage: `2026-06-11`
* country: `CH`
* scenarios: `slow, central, fast`
* delivery years: `2030`
* coverage: `OK`
* production gate: `FAILED`

## Summary

| rows_total | rows_asof | countries | scenarios | publication_min | publication_max | delivery_years_country |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 15 | AT, CH, DE, FR, IT | central, fast, slow | 2026-06-05 00:00:00+00:00 | 2026-06-11 00:00:00+00:00 | 2030 |

## Missing Recommended Columns

_none_

## Production Gate Issues

### missing_critical_production_values

| column | missing_rows |
| --- | --- |
| battery_power_gw | 12 |
| coal_gw | 15 |
| dispatchable_gw | 15 |
| dsm_gw | 15 |
| electrolysis_twh | 15 |
| ev_twh | 12 |
| export_twh | 12 |
| gas_gw | 15 |
| heatpump_twh | 12 |
| hydro_capacity_gw | 12 |
| hydro_reservoir_twh | 12 |
| hydro_twh | 12 |
| import_twh | 12 |
| managed_charging_share | 15 |
| net_import_twh | 12 |
| ntc_ch_at_gw | 15 |
| ntc_ch_de_gw | 15 |
| ntc_ch_fr_gw | 15 |
| ntc_ch_it_gw | 15 |
| p2x_gw | 15 |
| peak_load_gw | 12 |
| pv_twh | 15 |
| wind_twh | 15 |
| winter_demand_twh | 12 |

### unacceptable_quality_flags

| country | scenario | delivery_year | quality_flag | rows |
| --- | --- | --- | --- | --- |
| AT | central | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| AT | fast | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| AT | slow | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| CH | central | 2030 | official_component_partial_ch_ep2050_proxy | 1 |
| CH | fast | 2030 | official_component_partial_ch_ep2050_proxy | 1 |
| CH | slow | 2030 | official_component_partial_ch_ep2050_proxy | 1 |
| DE | central | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| DE | fast | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| DE | slow | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| FR | central | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| FR | fast | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| FR | slow | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| IT | central | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| IT | fast | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
| IT | slow | 2030 | official_component_partial_neighbor_demand_proxy | 1 |
