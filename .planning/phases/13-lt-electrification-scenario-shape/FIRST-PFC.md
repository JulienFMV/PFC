# First EP2050 PFC

* market: `CH`
* scenario: `ZERO_Basis`
* delivery start: `2030-01-01`
* horizon days: `365`
* EEX forward date: `2026-06-05`
* forward keys: `20`
* source scenario path: `data/electrification_scenarios_ep2050.parquet`
* expanded scenario path: `output\first_ep2050_pfc_2030_zero_basis_scenario_expanded.parquet`
* electrification shape: `ON`
* intraday amplitude shrinkage: `ON`
* missing scenario behavior: `fail-fast`
* output parquet: `output/first_ep2050_pfc_2030_zero_basis.parquet`

## Price Summary

| metric | value |
|---|---:|
| `rows` | 35040 |
| `start` | 2030-01-01 00:00:00+00:00 |
| `end` | 2030-12-31 23:45:00+00:00 |
| `mean` | 68.7398 |
| `min` | 19.2718 |
| `max` | 122.7220 |
| `p05` | 31.9446 |
| `p95` | 105.5806 |
| `midday_mean` | 61.4799 |
| `evening_mean` | 73.4628 |
| `night_mean` | 69.2726 |

## Forward Keys

2026-06, 2026-07, 2026-08, 2026-09, 2026-10, 2026-11, 2026-12, 2026-Q3, 2026-Q4, 2027, 2027-Q1, 2027-Q2, 2027-Q3, 2027-Q4, 2028, 2028-Q1, 2029, 2030, 2031, 2032

## Expanded Scenario Used

| scenario | delivery_year | quality_flag | demand_twh | pv_twh | wind_twh | hydro_twh | net_import_twh |
|---|---|---|---|---|---|---|---|
| ZERO_Basis | 2030 | official | 70.4407 | 8.6778 | 0.5780 | 41.6655 | 7.4853 |
| ZERO_Basis | 2031 | official_interpolated | 71.0584 | 9.8256 | 0.7052 | 41.7177 | 8.5194 |
