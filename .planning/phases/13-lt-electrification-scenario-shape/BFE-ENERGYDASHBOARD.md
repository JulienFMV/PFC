# BFE Energiedashboard Import

* output: `data\bfe_energiedashboard_daily.parquet`
* raw cache: `C:\Users\jbattaglia\pfc_local_data\scenarios\bfe_energiedashboard`
* rows: `70610`

## Source Summary

| dataset_id | source | track | rows | variables | first_date | last_date | quality_flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BFE-DS-0082 | BFE Energiedashboard SDSC power consumption forecast | forecast | 4152 | 3 | 2022-09-01 00:00:00+00:00 | 2026-06-15 00:00:00+00:00 | official_forecast_snapshot_ingest_time_only |
| BFE-DS-0087 | BFE Energiedashboard day-ahead base spot price | actual | 3448 | 1 | 2017-01-01 00:00:00+00:00 | 2026-06-10 00:00:00+00:00 | official_daily_actual_no_row_publication_time |
| BFE-DS-0093 | BFE Energiedashboard electricity production Swissgrid | actual | 25140 | 6 | 2015-01-01 00:00:00+00:00 | 2026-06-23 00:00:00+00:00 | official_daily_actual_no_row_publication_time |
| BFE-DS-0094 | BFE Energiedashboard daily electricity import/export flows | actual | 31032 | 9 | 2017-01-01 00:00:00+00:00 | 2026-06-10 00:00:00+00:00 | official_daily_actual_no_row_publication_time |
| BFE-DS-0095 | BFE Energiedashboard model-based estimated national consumption | actual | 24 | 1 | 2026-06-01 00:00:00+00:00 | 2026-06-24 00:00:00+00:00 | official_daily_actual_no_row_publication_time |
| BFE-DS-0096 | BFE Energiedashboard national and final consumption | actual | 6814 | 2 | 2017-01-01 00:00:00+00:00 | 2026-04-30 00:00:00+00:00 | official_daily_actual_no_row_publication_time |

## Variables

| dataset_id | variable | unit | rows |
| --- | --- | --- | --- |
| BFE-DS-0082 | final_consumption_forecast_mean_gwh | GWh | 1384 |
| BFE-DS-0082 | final_consumption_forecast_p025_gwh | GWh | 1384 |
| BFE-DS-0082 | final_consumption_forecast_p975_gwh | GWh | 1384 |
| BFE-DS-0087 | spot_baseload_eur_mwh | EUR/MWh | 3448 |
| BFE-DS-0093 | production_hydro_ror_gwh | GWh | 4192 |
| BFE-DS-0093 | production_hydro_storage_gwh | GWh | 4180 |
| BFE-DS-0093 | production_nuclear_gwh | GWh | 4192 |
| BFE-DS-0093 | production_solar_gwh | GWh | 4192 |
| BFE-DS-0093 | production_thermal_gwh | GWh | 4192 |
| BFE-DS-0093 | production_wind_gwh | GWh | 4192 |
| BFE-DS-0094 | flow_at_to_ch_gwh | GWh | 3448 |
| BFE-DS-0094 | flow_ch_to_at_gwh | GWh | 3448 |
| BFE-DS-0094 | flow_ch_to_de_gwh | GWh | 3448 |
| BFE-DS-0094 | flow_ch_to_fr_gwh | GWh | 3448 |
| BFE-DS-0094 | flow_ch_to_it_gwh | GWh | 3448 |
| BFE-DS-0094 | flow_de_to_ch_gwh | GWh | 3448 |
| BFE-DS-0094 | flow_fr_to_ch_gwh | GWh | 3448 |
| BFE-DS-0094 | flow_it_to_ch_gwh | GWh | 3448 |
| BFE-DS-0094 | net_import_gwh | GWh | 3448 |
| BFE-DS-0095 | national_consumption_estimated_gwh | GWh | 24 |
| BFE-DS-0096 | final_consumption_gwh | GWh | 3407 |
| BFE-DS-0096 | national_consumption_gwh | GWh | 3407 |

## Production Use

These feeds are official daily actuals useful for CH historical calibration and diagnostics.
They are not long-term 2030 scenario paths. The SDSC forecast feed is only vintage-safe
from the ingestion timestamp unless this importer is scheduled daily without overwriting snapshots.
