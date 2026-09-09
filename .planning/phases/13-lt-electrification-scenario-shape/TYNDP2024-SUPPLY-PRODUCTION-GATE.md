# Electrification Scenario Inventory Validation

* vintage: `2026-06-11`
* country: `CH`
* scenarios: `slow, central, fast`
* delivery years: `2030, 2040`
* coverage: `OK`
* production gate: `FAILED`

## Summary

| rows_total | rows_asof | countries | scenarios | publication_min | publication_max | delivery_years_country |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 30 | AT, CH, DE, FR, IT | central, fast, slow | 2024-05-31 00:00:00+00:00 | 2024-05-31 00:00:00+00:00 | 2030, 2040 |

## Missing Recommended Columns

_none_

## Production Gate Issues

### missing_critical_production_values

| column | missing_rows |
| --- | --- |
| battery_power_gw | 30 |
| coal_gw | 30 |
| demand_twh | 30 |
| dispatchable_gw | 30 |
| dsm_gw | 30 |
| electrolysis_twh | 30 |
| ev_twh | 30 |
| export_twh | 30 |
| gas_gw | 30 |
| heatpump_twh | 30 |
| hydro_capacity_gw | 30 |
| hydro_reservoir_twh | 30 |
| hydro_twh | 30 |
| import_twh | 30 |
| managed_charging_share | 30 |
| net_import_twh | 30 |
| ntc_ch_at_gw | 30 |
| ntc_ch_de_gw | 30 |
| ntc_ch_fr_gw | 30 |
| ntc_ch_it_gw | 30 |
| p2x_gw | 30 |
| peak_load_gw | 30 |
| pv_twh | 30 |
| wind_twh | 30 |
| winter_demand_twh | 30 |

### unacceptable_quality_flags

| country | scenario | delivery_year | quality_flag | rows |
| --- | --- | --- | --- | --- |
| AT | central | 2030 | official_tyndp_supply_partial | 1 |
| AT | central | 2040 | official_tyndp_supply_partial | 1 |
| AT | fast | 2030 | official_tyndp_supply_partial | 1 |
| AT | fast | 2040 | official_tyndp_supply_partial | 1 |
| AT | slow | 2030 | official_tyndp_supply_partial | 1 |
| AT | slow | 2040 | official_tyndp_supply_partial | 1 |
| CH | central | 2030 | official_tyndp_supply_partial | 1 |
| CH | central | 2040 | official_tyndp_supply_partial | 1 |
| CH | fast | 2030 | official_tyndp_supply_partial | 1 |
| CH | fast | 2040 | official_tyndp_supply_partial | 1 |
| CH | slow | 2030 | official_tyndp_supply_partial | 1 |
| CH | slow | 2040 | official_tyndp_supply_partial | 1 |
| DE | central | 2030 | official_tyndp_supply_partial | 1 |
| DE | central | 2040 | official_tyndp_supply_partial | 1 |
| DE | fast | 2030 | official_tyndp_supply_partial | 1 |
| DE | fast | 2040 | official_tyndp_supply_partial | 1 |
| DE | slow | 2030 | official_tyndp_supply_partial | 1 |
| DE | slow | 2040 | official_tyndp_supply_partial | 1 |
| FR | central | 2030 | official_tyndp_supply_partial | 1 |
| FR | central | 2040 | official_tyndp_supply_partial | 1 |
| FR | fast | 2030 | official_tyndp_supply_partial | 1 |
| FR | fast | 2040 | official_tyndp_supply_partial | 1 |
| FR | slow | 2030 | official_tyndp_supply_partial | 1 |
| FR | slow | 2040 | official_tyndp_supply_partial | 1 |
| IT | central | 2030 | official_tyndp_supply_partial | 1 |
| IT | central | 2040 | official_tyndp_supply_partial | 1 |
| IT | fast | 2030 | official_tyndp_supply_partial | 1 |
| IT | fast | 2040 | official_tyndp_supply_partial | 1 |
| IT | slow | 2030 | official_tyndp_supply_partial | 1 |
| IT | slow | 2040 | official_tyndp_supply_partial | 1 |
