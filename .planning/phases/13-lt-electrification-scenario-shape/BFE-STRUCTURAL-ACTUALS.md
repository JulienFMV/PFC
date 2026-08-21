# BFE Structural Actuals Import

## Outputs

| name | path |
| --- | --- |
| reservoir | data\bfe_hydro_reservoir_weekly.parquet |
| plants | data\bfe_ch_production_plants.parquet |
| capacity | data\bfe_ch_installed_capacity_actuals.parquet |
| wasta_plants | data\bfe_wasta_hydro_plants.parquet |
| wasta_summary | data\bfe_wasta_hydro_summary.parquet |
| structural | data\bfe_ch_structural_actuals.parquet |

## Reservoir Weekly Actuals

| region | variable | unit | rows | first_date | last_date |
| --- | --- | --- | --- | --- | --- |
| grisons | hydro_reservoir_capacity_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| grisons | hydro_reservoir_content_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| grisons | hydro_reservoir_fill_ratio | ratio | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| other_ch | hydro_reservoir_capacity_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| other_ch | hydro_reservoir_content_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| other_ch | hydro_reservoir_fill_ratio | ratio | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| ticino | hydro_reservoir_capacity_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| ticino | hydro_reservoir_content_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| ticino | hydro_reservoir_fill_ratio | ratio | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| total_ch | hydro_reservoir_capacity_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| total_ch | hydro_reservoir_content_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| total_ch | hydro_reservoir_fill_ratio | ratio | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| valais | hydro_reservoir_capacity_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| valais | hydro_reservoir_content_gwh | GWh | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |
| valais | hydro_reservoir_fill_ratio | ratio | 1380 | 2000-01-03 00:00:00+00:00 | 2026-06-08 00:00:00+00:00 |

## Installed Capacity Actuals

| technology | variable | value | unit | plant_count |
| --- | --- | --- | --- | --- |
| hydro | hydro_capacity_gw | 12.99724294 | GW_ac_equivalent | 1326 |
| pv | pv_gw | 8.45111628 | GW_ac_equivalent | 325184 |
| nuclear | nuclear_gw | 3.0145999999999997 | GW_ac_equivalent | 4 |
| waste | waste_capacity_gw | 0.3538196 | GW_ac_equivalent | 28 |
| thermal | thermal_capacity_gw | 0.2987442 | GW_ac_equivalent | 189 |
| biomass | biomass_capacity_gw | 0.243143 | GW_ac_equivalent | 429 |
| wind | wind_gw | 0.1087769 | GW_ac_equivalent | 63 |

## WASTA Hydro Summary

| type_code | type_de | hydro_capacity_gw | hydro_twh | pumped_storage_power_gw | plant_count |
| --- | --- | --- | --- | --- | --- |
| t1 | Laufkraftwerk | 4.96269 | 21.29311 | 0.00116 | 605 |
| t2 | reines Umwälzwerk | 0.529 | 0.0 | 0.49429999999999996 | 3 |
| t3 | Speicherkraftwerk | 8.32692 | 18.24914 | 0.27682 | 99 |
| t4 | Pumpspeicherkraftwerk | 3.6786499999999998 | 1.81427 | 3.29459 | 21 |

## Model-Facing Structural Actuals

| variable | value | unit | measurement_date | source_id |
| --- | --- | --- | --- | --- |
| biomass_capacity_gw | 0.243143 | GW | 2026-06-12 00:00:00+00:00 | bfe_electricity_production_plants |
| hydro_capacity_gw | 17.49726 | GW | 2025-12-31 00:00:00+00:00 | bfe_wasta_hydropower_plants |
| hydro_reservoir_capacity_twh | 8.895 | TWh | 2026-06-08 00:00:00+00:00 | bfe_speicherseen_weekly |
| hydro_reservoir_fill_ratio | 0.2505902192242833 | ratio | 2026-06-08 00:00:00+00:00 | bfe_speicherseen_weekly |
| hydro_reservoir_twh | 2.229 | TWh | 2026-06-08 00:00:00+00:00 | bfe_speicherseen_weekly |
| hydro_twh | 41.356519999999996 | TWh | 2025-12-31 00:00:00+00:00 | bfe_wasta_hydropower_plants |
| nuclear_gw | 3.0145999999999997 | GW | 2026-06-12 00:00:00+00:00 | bfe_electricity_production_plants |
| pumped_storage_power_gw | 4.06687 | GW | 2025-12-31 00:00:00+00:00 | bfe_wasta_hydropower_plants |
| pv_gw | 8.45111628 | GW | 2026-06-12 00:00:00+00:00 | bfe_electricity_production_plants |
| thermal_capacity_gw | 0.2987442 | GW | 2026-06-12 00:00:00+00:00 | bfe_electricity_production_plants |
| waste_capacity_gw | 0.3538196 | GW | 2026-06-12 00:00:00+00:00 | bfe_electricity_production_plants |
| wind_gw | 0.1087769 | GW | 2026-06-12 00:00:00+00:00 | bfe_electricity_production_plants |

## Production Use

These are official actual/baseline datasets for CH calibration and scenario actualization.
They do not create a governed 2030 scenario path and must not flip Phase 13 production gates to GO.
Use them only behind explicit actualization logic with publication/measurement date checks.

## Source URLs

| source_id | dataset_id | url |
| --- | --- | --- |
| bfe_speicherseen_weekly | ogd17-bfe | https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv |
| bfe_electricity_production_plants | ch.bfe.elektrizitaetsproduktionsanlagen | https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/csv/2056/ch.bfe.elektrizitaetsproduktionsanlagen.zip |
| bfe_wasta_hydropower_plants | ch.bfe.statistik-wasserkraftanlagen | https://data.geo.admin.ch/ch.bfe.statistik-wasserkraftanlagen/statistik-wasserkraftanlagen/statistik-wasserkraftanlagen_2056.csv.zip |
