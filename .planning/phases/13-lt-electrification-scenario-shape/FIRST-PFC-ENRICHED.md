# First EP2050 PFC

* market: `CH`
* scenario: `ZERO_Basis`
* delivery start: `2030-01-01`
* horizon days: `365`
* EEX forward date: `2026-06-05`
* forward keys: `20`
* source scenario path: `data/electrification_scenarios_ep2050_enriched.parquet`
* expanded scenario path: `output\first_ep2050_pfc_2030_zero_basis_enriched_scenario_expanded.parquet`
* electrification shape: `ON`
* intraday amplitude shrinkage: `ON`
* missing scenario behavior: `fail-fast`
* output parquet: `output/first_ep2050_pfc_2030_zero_basis_enriched.parquet`

## Price Summary

| metric | value |
|---|---:|
| `rows` | 35040 |
| `start` | 2030-01-01 00:00:00+00:00 |
| `end` | 2030-12-31 23:45:00+00:00 |
| `mean` | 68.7398 |
| `min` | 19.2875 |
| `max` | 124.4428 |
| `p05` | 32.5818 |
| `p95` | 105.9458 |
| `midday_mean` | 63.1455 |
| `evening_mean` | 70.7542 |
| `night_mean` | 69.2224 |

## Forward Keys

2026-06, 2026-07, 2026-08, 2026-09, 2026-10, 2026-11, 2026-12, 2026-Q3, 2026-Q4, 2027, 2027-Q1, 2027-Q2, 2027-Q3, 2027-Q4, 2028, 2028-Q1, 2029, 2030, 2031, 2032

## Expanded Scenario Used

| scenario | delivery_year | quality_flag | demand_twh | pv_twh | wind_twh | battery_power_gw | battery_energy_gwh | managed_charging_share | winter_demand_twh | hydro_twh | hydro_reservoir_twh | net_import_twh | ntc_ch_de_gw | ntc_ch_fr_gw | ntc_ch_it_gw | scenario_weight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZERO_Basis | 2030 | official_proxy_enriched | 70.4407 | 8.6778 | 0.5780 | 2.0000 | 5.0000 | 0.3500 | 19.8906 | 41.6655 | 3.9781 | 7.4853 | 4.0000 | 4.0000 | 3.0000 | 0.5000 |
| ZERO_Basis | 2031 | official_proxy_enriched_interpolated | 71.0584 | 9.8256 | 0.7052 | 2.3000 | 6.0000 | 0.3800 | 20.0730 | 41.7177 | 4.0146 | 8.5194 | 4.0000 | 4.0000 | 3.0000 | 0.5000 |
