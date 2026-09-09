# P0 Structural Bridge

* output: `data\electrification_scenarios_composed_p0_bridge_2030.parquet`
* status: `NON-PRODUCTION / PARTIAL / PROXY`
* rule: fills only traceable P0 numeric gaps; does not fill NTC, hydro or cross-border balance without source

## Filled Rows By Column

| column | filled_rows |
| --- | --- |
| battery_power_gw | 12 |
| dispatchable_gw | 6 |
| ev_twh | 12 |
| heatpump_twh | 12 |
| peak_load_gw | 12 |
| pv_twh | 15 |
| wind_twh | 15 |
| winter_demand_twh | 12 |

## Methodology

* `peak_load_gw`: CH historical peak/average load ratio from local ENTSO-E cache.
* `winter_demand_twh`: CH historical Nov-Mar demand share from local ENTSO-E cache.
* `pv_twh` / `wind_twh`: explicit country capacity-factor bridge; proxy only.
* `battery_power_gw`: four-hour duration bridge from battery energy.
* `dispatchable_gw`: nuclear-only lower-bound bridge where missing.
* neighbour `ev_twh` / `heatpump_twh`: TYNDP Demand REF2019-to-2040 bridge.

## Changed Cells

| country | scenario | delivery_year | column | value | method |
| --- | --- | --- | --- | --- | --- |
| AT | central | 2030 | peak_load_gw | 18.88149924358273 | CH historical peak/average load ratio |
| AT | central | 2030 | winter_demand_twh | 43.615124156679805 | CH historical winter demand share |
| AT | central | 2030 | pv_twh | 12.614670158400001 | country PV capacity factor bridge |
| AT | central | 2030 | wind_twh | 19.710002192535544 | country wind capacity factor bridge |
| AT | central | 2030 | battery_power_gw | 0.0254 | 4h battery duration bridge |
| AT | central | 2030 | ev_twh | 11.505927572154638 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| AT | central | 2030 | heatpump_twh | 7.653078564080287 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| AT | fast | 2030 | peak_load_gw | 19.526583019067026 | CH historical peak/average load ratio |
| AT | fast | 2030 | winter_demand_twh | 45.10522876099345 | CH historical winter demand share |
| AT | fast | 2030 | pv_twh | 31.535999999999998 | country PV capacity factor bridge |
| AT | fast | 2030 | wind_twh | 24.30024 | country wind capacity factor bridge |
| AT | fast | 2030 | battery_power_gw | 0.06349864007079181 | 4h battery duration bridge |
| AT | fast | 2030 | ev_twh | 12.106978029891422 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| AT | fast | 2030 | heatpump_twh | 7.975384861096888 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| AT | slow | 2030 | peak_load_gw | 18.236415468098443 | CH historical peak/average load ratio |
| AT | slow | 2030 | winter_demand_twh | 42.12501955236617 | CH historical winter demand share |
| AT | slow | 2030 | pv_twh | 12.6144 | country PV capacity factor bridge |
| AT | slow | 2030 | wind_twh | 16.425001827112958 | country wind capacity factor bridge |
| AT | slow | 2030 | battery_power_gw | 0.0254 | 4h battery duration bridge |
| AT | slow | 2030 | ev_twh | 10.904877114417854 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| AT | slow | 2030 | heatpump_twh | 7.330772267063686 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| CH | central | 2030 | pv_twh | 10.270506322285712 | country PV capacity factor bridge |
| CH | central | 2030 | wind_twh | 0.5431199999999999 | country wind capacity factor bridge |
| CH | central | 2030 | dispatchable_gw | 1.19 | nuclear-only firm dispatchable lower-bound bridge |
| CH | fast | 2030 | pv_twh | 12.835152 | country PV capacity factor bridge |
| CH | fast | 2030 | wind_twh | 0.5431199999999999 | country wind capacity factor bridge |
| CH | fast | 2030 | dispatchable_gw | 1.19 | nuclear-only firm dispatchable lower-bound bridge |
| CH | slow | 2030 | pv_twh | 8.04168 | country PV capacity factor bridge |
| CH | slow | 2030 | wind_twh | 0.31536 | country wind capacity factor bridge |
| CH | slow | 2030 | dispatchable_gw | 1.19 | nuclear-only firm dispatchable lower-bound bridge |
| DE | central | 2030 | peak_load_gw | 144.68957496744414 | CH historical peak/average load ratio |
| DE | central | 2030 | winter_demand_twh | 334.22418924318805 | CH historical winter demand share |
| DE | central | 2030 | pv_twh | 207.1759272 | country PV capacity factor bridge |
| DE | central | 2030 | wind_twh | 356.1114849600001 | country wind capacity factor bridge |
| DE | central | 2030 | battery_power_gw | 14.2347 | 4h battery duration bridge |
| DE | central | 2030 | ev_twh | 78.6229836529754 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| DE | central | 2030 | heatpump_twh | 58.60810975645089 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| DE | fast | 2030 | peak_load_gw | 153.64048401511494 | CH historical peak/average load ratio |
| DE | fast | 2030 | winter_demand_twh | 354.90024914674643 | CH historical winter demand share |
| DE | fast | 2030 | pv_twh | 207.17399999999998 | country PV capacity factor bridge |
| DE | fast | 2030 | wind_twh | 356.10976800000003 | country wind capacity factor bridge |
| DE | fast | 2030 | battery_power_gw | 14.2347 | 4h battery duration bridge |
| DE | fast | 2030 | ev_twh | 87.98792308046694 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| DE | fast | 2030 | heatpump_twh | 59.426241100351604 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| DE | slow | 2030 | peak_load_gw | 135.7386659197734 | CH historical peak/average load ratio |
| DE | slow | 2030 | winter_demand_twh | 313.5481293396297 | CH historical winter demand share |
| DE | slow | 2030 | pv_twh | 207.17399999999998 | country PV capacity factor bridge |
| DE | slow | 2030 | wind_twh | 356.10976800000003 | country wind capacity factor bridge |
| DE | slow | 2030 | battery_power_gw | 14.2347 | 4h battery duration bridge |
| DE | slow | 2030 | ev_twh | 69.25804422548387 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| DE | slow | 2030 | heatpump_twh | 57.78997841255017 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| FR | central | 2030 | peak_load_gw | 102.70687421799703 | CH historical peak/average load ratio |
| FR | central | 2030 | winter_demand_twh | 237.24668327304101 | CH historical winter demand share |
| FR | central | 2030 | pv_twh | 51.854877816000005 | country PV capacity factor bridge |
| FR | central | 2030 | wind_twh | 83.19590923261696 | country wind capacity factor bridge |
| FR | central | 2030 | battery_power_gw | 0.235 | 4h battery duration bridge |
| FR | central | 2030 | dispatchable_gw | 61.761 | nuclear-only firm dispatchable lower-bound bridge |
| FR | central | 2030 | ev_twh | 49.64271034656343 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| FR | central | 2030 | heatpump_twh | 65.8232352936242 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| FR | fast | 2030 | peak_load_gw | 103.50044238588856 | CH historical peak/average load ratio |
| FR | fast | 2030 | winter_demand_twh | 239.07977786594748 | CH historical winter demand share |
| FR | fast | 2030 | pv_twh | 71.1312 | country PV capacity factor bridge |
| FR | fast | 2030 | wind_twh | 102.98653690663603 | country wind capacity factor bridge |
| FR | fast | 2030 | battery_power_gw | 0.3223579478735609 | 4h battery duration bridge |
| FR | fast | 2030 | dispatchable_gw | 63.0 | nuclear-only firm dispatchable lower-bound bridge |
| FR | fast | 2030 | ev_twh | 50.72643578478261 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| FR | fast | 2030 | heatpump_twh | 65.9325051155661 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| FR | slow | 2030 | peak_load_gw | 101.91330605010546 | CH historical peak/average load ratio |
| FR | slow | 2030 | winter_demand_twh | 235.41358868013455 | CH historical winter demand share |
| FR | slow | 2030 | pv_twh | 28.2072 | country PV capacity factor bridge |
| FR | slow | 2030 | wind_twh | 75.39075000000001 | country wind capacity factor bridge |
| FR | slow | 2030 | battery_power_gw | 0.235 | 4h battery duration bridge |
| FR | slow | 2030 | dispatchable_gw | 59.4 | nuclear-only firm dispatchable lower-bound bridge |
| FR | slow | 2030 | ev_twh | 48.55898490834426 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| FR | slow | 2030 | heatpump_twh | 65.71396547168229 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| IT | central | 2030 | peak_load_gw | 74.55755817031027 | CH historical peak/average load ratio |
| IT | central | 2030 | winter_demand_twh | 172.22346141406982 | CH historical winter demand share |
| IT | central | 2030 | pv_twh | 104.56012221783101 | country PV capacity factor bridge |
| IT | central | 2030 | wind_twh | 54.219919672596006 | country wind capacity factor bridge |
| IT | central | 2030 | battery_power_gw | 23.69125 | 4h battery duration bridge |
| IT | central | 2030 | ev_twh | 31.708955344593186 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| IT | central | 2030 | heatpump_twh | 33.57568692108134 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| IT | fast | 2030 | peak_load_gw | 75.93050689387053 | CH historical peak/average load ratio |
| IT | fast | 2030 | winter_demand_twh | 175.39489013730469 | CH historical winter demand share |
| IT | fast | 2030 | pv_twh | 141.6695134718106 | country PV capacity factor bridge |
| IT | fast | 2030 | wind_twh | 56.93220360000001 | country wind capacity factor bridge |
| IT | fast | 2030 | battery_power_gw | 28.6683836098373 | 4h battery duration bridge |
| IT | fast | 2030 | ev_twh | 34.29085223529651 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| IT | fast | 2030 | heatpump_twh | 34.61699239451909 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
| IT | slow | 2030 | peak_load_gw | 73.18460944674999 | CH historical peak/average load ratio |
| IT | slow | 2030 | winter_demand_twh | 169.0520326908349 | CH historical winter demand share |
| IT | slow | 2030 | pv_twh | 94.03194239999999 | country PV capacity factor bridge |
| IT | slow | 2030 | wind_twh | 50.51305080000001 | country wind capacity factor bridge |
| IT | slow | 2030 | battery_power_gw | 23.69125 | 4h battery duration bridge |
| IT | slow | 2030 | ev_twh | 29.127058453889862 | TYNDP Demand transport electricity REF2019-to-2040 bridge |
| IT | slow | 2030 | heatpump_twh | 32.53438144764359 | TYNDP Demand electric space-heating REF2019-to-2040 bridge |
