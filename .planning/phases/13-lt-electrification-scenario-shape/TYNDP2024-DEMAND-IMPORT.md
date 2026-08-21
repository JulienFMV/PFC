# TYNDP 2024 Demand Import

* workbook: `C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb`
* status: `PARTIAL - not production-complete`
* source: ENTSO-E/ENTSOG TYNDP 2024 Demand Outputs workbook
* rule: raw official DE/GA scenario labels are preserved; no slow/central/fast mapping is inferred

## Coverage

| rows | countries | scenarios | years |
| --- | --- | --- | --- |
| 16 | AT, DE, FR, IT | tyndp_distributed_energy, tyndp_global_ambition | 2040, 2050 |

## Non-null Critical Fields

| column | non_null_rows |
| --- | --- |
| battery_energy_gwh | 0 |
| battery_power_gw | 0 |
| co2_eur_t | 0 |
| coal_eur_mwh | 0 |
| coal_gw | 0 |
| demand_twh | 16 |
| dispatchable_gw | 0 |
| dsm_gw | 0 |
| electrolysis_twh | 0 |
| ev_twh | 0 |
| export_twh | 0 |
| gas_eur_mwh | 0 |
| gas_gw | 0 |
| heatpump_twh | 0 |
| hydro_capacity_gw | 0 |
| hydro_reservoir_twh | 0 |
| hydro_twh | 0 |
| import_twh | 0 |
| ingested_at_utc | 16 |
| managed_charging_share | 0 |
| net_import_twh | 0 |
| ntc_ch_at_gw | 0 |
| ntc_ch_de_gw | 0 |
| ntc_ch_fr_gw | 0 |
| ntc_ch_it_gw | 0 |
| nuclear_gw | 0 |
| p2x_gw | 0 |
| peak_load_gw | 0 |
| pv_gw | 0 |
| pv_twh | 0 |
| quality_flag | 16 |
| scenario_edition | 16 |
| scenario_weight | 0 |
| source | 16 |
| track | 16 |
| wind_gw | 0 |
| wind_twh | 0 |
| winter_demand_twh | 0 |

## Production Limitation

This component supplies demand for available neighbouring countries only. CH demand remains sourced from OFEN/EP2050, and final production still requires governed scenario mapping, NTC, hydro, EV, heat-pump and flexibility feeds.
