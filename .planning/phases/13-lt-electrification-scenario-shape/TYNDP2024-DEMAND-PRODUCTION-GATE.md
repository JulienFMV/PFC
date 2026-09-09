# Electrification Scenario Inventory Validation

* vintage: `2026-06-11`
* country: `DE`
* scenarios: `tyndp_distributed_energy, tyndp_global_ambition`
* delivery years: `2040, 2050`
* coverage: `OK`
* production gate: `FAILED`

## Summary

| rows_total | rows_asof | countries | scenarios | publication_min | publication_max | delivery_years_country |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 16 | AT, DE, FR, IT | tyndp_distributed_energy, tyndp_global_ambition | 2024-05-31 00:00:00+00:00 | 2024-05-31 00:00:00+00:00 | 2040, 2050 |

## Missing Recommended Columns

_none_

## Production Gate Issues

### missing_critical_production_values

| column | missing_rows |
| --- | --- |
| battery_energy_gwh | 16 |
| battery_power_gw | 16 |
| co2_eur_t | 16 |
| coal_eur_mwh | 16 |
| coal_gw | 16 |
| dispatchable_gw | 16 |
| dsm_gw | 16 |
| electrolysis_twh | 16 |
| ev_twh | 16 |
| export_twh | 16 |
| gas_eur_mwh | 16 |
| gas_gw | 16 |
| heatpump_twh | 16 |
| hydro_capacity_gw | 16 |
| hydro_reservoir_twh | 16 |
| hydro_twh | 16 |
| import_twh | 16 |
| managed_charging_share | 16 |
| net_import_twh | 16 |
| ntc_ch_at_gw | 16 |
| ntc_ch_de_gw | 16 |
| ntc_ch_fr_gw | 16 |
| ntc_ch_it_gw | 16 |
| nuclear_gw | 16 |
| p2x_gw | 16 |
| peak_load_gw | 16 |
| pv_gw | 16 |
| pv_twh | 16 |
| scenario_weight | 16 |
| wind_gw | 16 |
| wind_twh | 16 |
| winter_demand_twh | 16 |

### unacceptable_quality_flags

| country | scenario | delivery_year | quality_flag | rows |
| --- | --- | --- | --- | --- |
| AT | tyndp_distributed_energy | 2040 | official_tyndp_demand_partial | 1 |
| AT | tyndp_distributed_energy | 2050 | official_tyndp_demand_partial | 1 |
| AT | tyndp_global_ambition | 2040 | official_tyndp_demand_partial | 1 |
| AT | tyndp_global_ambition | 2050 | official_tyndp_demand_partial | 1 |
| DE | tyndp_distributed_energy | 2040 | official_tyndp_demand_partial | 1 |
| DE | tyndp_distributed_energy | 2050 | official_tyndp_demand_partial | 1 |
| DE | tyndp_global_ambition | 2040 | official_tyndp_demand_partial | 1 |
| DE | tyndp_global_ambition | 2050 | official_tyndp_demand_partial | 1 |
| FR | tyndp_distributed_energy | 2040 | official_tyndp_demand_partial | 1 |
| FR | tyndp_distributed_energy | 2050 | official_tyndp_demand_partial | 1 |
| FR | tyndp_global_ambition | 2040 | official_tyndp_demand_partial | 1 |
| FR | tyndp_global_ambition | 2050 | official_tyndp_demand_partial | 1 |
| IT | tyndp_distributed_energy | 2040 | official_tyndp_demand_partial | 1 |
| IT | tyndp_distributed_energy | 2050 | official_tyndp_demand_partial | 1 |
| IT | tyndp_global_ambition | 2040 | official_tyndp_demand_partial | 1 |
| IT | tyndp_global_ambition | 2050 | official_tyndp_demand_partial | 1 |

### scenario_weight_sum_not_one

| country | delivery_year | scenario_count | weight_sum |
| --- | --- | --- | --- |
| DE | 2040 | 2 | 0.0 |
| DE | 2050 | 2 | 0.0 |
