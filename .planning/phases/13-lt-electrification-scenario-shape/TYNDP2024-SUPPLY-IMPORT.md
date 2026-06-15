# TYNDP 2024 Supply Import

* workbook: `C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\20231103-Final-Supply-Inputs-for-TYNDP-2024-Scenarios.xlsx\20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx`
* status: `PARTIAL - not production-complete`
* source: ENTSO-E/ENTSOG TYNDP 2024 Supply Inputs workbook
* rule: missing demand, NTC, battery power and flexibility fields remain null; no silent zero filling

## Coverage

| rows | countries | scenarios | years |
| --- | --- | --- | --- |
| 30 | AT, CH, DE, FR, IT | central, fast, slow | 2030, 2040 |

## Non-null Critical Fields

| column | non_null_rows |
| --- | --- |
| battery_energy_gwh | 30 |
| battery_power_gw | 0 |
| co2_eur_t | 30 |
| coal_eur_mwh | 30 |
| coal_gw | 0 |
| demand_twh | 0 |
| dispatchable_gw | 0 |
| dsm_gw | 0 |
| electrolysis_twh | 0 |
| ev_twh | 0 |
| export_twh | 0 |
| gas_eur_mwh | 30 |
| gas_gw | 0 |
| heatpump_twh | 0 |
| hydro_capacity_gw | 0 |
| hydro_reservoir_twh | 0 |
| hydro_twh | 0 |
| import_twh | 0 |
| ingested_at_utc | 30 |
| managed_charging_share | 0 |
| net_import_twh | 0 |
| ntc_ch_at_gw | 0 |
| ntc_ch_de_gw | 0 |
| ntc_ch_fr_gw | 0 |
| ntc_ch_it_gw | 0 |
| nuclear_gw | 30 |
| p2x_gw | 0 |
| peak_load_gw | 0 |
| pv_gw | 30 |
| pv_twh | 0 |
| quality_flag | 30 |
| scenario_edition | 30 |
| scenario_weight | 30 |
| source | 30 |
| track | 30 |
| wind_gw | 30 |
| wind_twh | 0 |
| winter_demand_twh | 0 |

## Production Limitation

This file is an official supply-side component. It must be merged with governed demand, NTC, hydro, EV, heat-pump and flexibility feeds before `--require-production` can pass.
