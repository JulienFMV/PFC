# Composed Partial LT Scenario Inventory

* scenario output: `data\electrification_scenarios_composed_partial_2030.parquet`
* feature output: `data\hpfc_scenario_features_composed_partial_2030.parquet`
* rows: `15`
* feature rows: `15`
* status: `PARTIAL - production gate must fail`
* rule: no missing critical value is converted to zero

## Coverage

| country | scenario | years |
| --- | --- | --- |
| AT | central | 2030 |
| AT | fast | 2030 |
| AT | slow | 2030 |
| CH | central | 2030 |
| CH | fast | 2030 |
| CH | slow | 2030 |
| DE | central | 2030 |
| DE | fast | 2030 |
| DE | slow | 2030 |
| FR | central | 2030 |
| FR | fast | 2030 |
| FR | slow | 2030 |
| IT | central | 2030 |
| IT | fast | 2030 |
| IT | slow | 2030 |

## Remaining Critical Gaps

| column | missing_rows |
| --- | --- |
| coal_gw | 15 |
| dispatchable_gw | 15 |
| dsm_gw | 15 |
| electrolysis_twh | 15 |
| gas_gw | 15 |
| managed_charging_share | 15 |
| ntc_ch_at_gw | 15 |
| ntc_ch_de_gw | 15 |
| ntc_ch_fr_gw | 15 |
| ntc_ch_it_gw | 15 |
| p2x_gw | 15 |
| pv_twh | 15 |
| wind_twh | 15 |
| battery_power_gw | 12 |
| ev_twh | 12 |
| export_twh | 12 |
| heatpump_twh | 12 |
| hydro_capacity_gw | 12 |
| hydro_reservoir_twh | 12 |
| hydro_twh | 12 |
| import_twh | 12 |
| net_import_twh | 12 |
| peak_load_gw | 12 |
| winter_demand_twh | 12 |

## Overlay Provenance

| country | scenario | delivery_year | source | column |
| --- | --- | --- | --- | --- |
| AT | central | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| AT | fast | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| AT | slow | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| CH | central | 2030 | CH EP2050 enriched slow/central/fast | battery_power_gw, demand_twh, ev_twh, export_twh, heatpump_twh, hydro_capacity_gw, hydro_reservoir_twh, hydro_twh, import_twh, net_import_twh, peak_load_gw, winter_demand_twh |
| CH | fast | 2030 | CH EP2050 enriched slow/central/fast | battery_power_gw, demand_twh, ev_twh, export_twh, heatpump_twh, hydro_capacity_gw, hydro_reservoir_twh, hydro_twh, import_twh, net_import_twh, peak_load_gw, winter_demand_twh |
| CH | slow | 2030 | CH EP2050 enriched slow/central/fast | battery_power_gw, demand_twh, ev_twh, export_twh, heatpump_twh, hydro_capacity_gw, hydro_reservoir_twh, hydro_twh, import_twh, net_import_twh, peak_load_gw, winter_demand_twh |
| DE | central | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| DE | fast | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| DE | slow | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| FR | central | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| FR | fast | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| FR | slow | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| IT | central | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| IT | fast | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |
| IT | slow | 2030 | TYNDP2024 neighbour demand bridge | demand_twh |

## Decision

This composed file is a stronger smoke/prod-readiness inventory than the CH-only proxy because it carries official multi-country TYNDP supply rows, bridged neighbour 2030 demand when provided, and CH EP2050 overlays. It remains non-production because neighbour peak/winter load, NTC, dispatchable capacity, hydro, EV/heat-pump/flex and committee-approved scenario mapping are still incomplete.
