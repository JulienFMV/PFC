# Electrification Scenario Inventory Validation

* vintage: `2026-06-05`
* country: `CH`
* scenarios: `slow, central, fast`
* delivery years: `2027-2031`
* coverage: `OK`
* production gate: `FAILED`

## Summary

| rows_total | rows_asof | countries | scenarios | publication_min | publication_max | delivery_years_country |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 15 | CH | central, fast, slow | 2026-06-05 00:00:00+00:00 | 2026-06-05 00:00:00+00:00 | 2027, 2028, 2029, 2030, 2031 |

## Missing Recommended Columns

_none_

## Production Gate Issues

### missing_production_columns

| column |
| --- |
| co2_eur_t |
| coal_eur_mwh |
| dsm_gw |
| electrolysis_twh |
| gas_eur_mwh |
| ingested_at_utc |
| measurement_date |
| ntc_ch_at_gw |
| p2x_gw |
| scenario_edition |
| track |

### missing_required_countries

| country |
| --- |
| DE |
| FR |
| IT |
| AT |

### missing_required_country_scenario_years

| country | scenario | delivery_year |
| --- | --- | --- |
| DE | slow | 2027 |
| DE | slow | 2028 |
| DE | slow | 2029 |
| DE | slow | 2030 |
| DE | slow | 2031 |
| DE | central | 2027 |
| DE | central | 2028 |
| DE | central | 2029 |
| DE | central | 2030 |
| DE | central | 2031 |
| DE | fast | 2027 |
| DE | fast | 2028 |
| DE | fast | 2029 |
| DE | fast | 2030 |
| DE | fast | 2031 |
| FR | slow | 2027 |
| FR | slow | 2028 |
| FR | slow | 2029 |
| FR | slow | 2030 |
| FR | slow | 2031 |
| FR | central | 2027 |
| FR | central | 2028 |
| FR | central | 2029 |
| FR | central | 2030 |
| FR | central | 2031 |
| FR | fast | 2027 |
| FR | fast | 2028 |
| FR | fast | 2029 |
| FR | fast | 2030 |
| FR | fast | 2031 |
| IT | slow | 2027 |
| IT | slow | 2028 |
| IT | slow | 2029 |
| IT | slow | 2030 |
| IT | slow | 2031 |
| IT | central | 2027 |
| IT | central | 2028 |
| IT | central | 2029 |
| IT | central | 2030 |
| IT | central | 2031 |
| IT | fast | 2027 |
| IT | fast | 2028 |
| IT | fast | 2029 |
| IT | fast | 2030 |
| IT | fast | 2031 |
| AT | slow | 2027 |
| AT | slow | 2028 |
| AT | slow | 2029 |
| AT | slow | 2030 |
| AT | slow | 2031 |
| AT | central | 2027 |
| AT | central | 2028 |
| AT | central | 2029 |
| AT | central | 2030 |
| AT | central | 2031 |
| AT | fast | 2027 |
| AT | fast | 2028 |
| AT | fast | 2029 |
| AT | fast | 2030 |
| AT | fast | 2031 |

### unacceptable_quality_flags

| country | scenario | delivery_year | quality_flag | rows |
| --- | --- | --- | --- | --- |
| CH | central | 2027 | internal_midpoint_proxy_enriched_interpolated | 1 |
| CH | central | 2028 | internal_midpoint_proxy_enriched_interpolated | 1 |
| CH | central | 2029 | internal_midpoint_proxy_enriched_interpolated | 1 |
| CH | central | 2030 | internal_midpoint_proxy_enriched | 1 |
| CH | central | 2031 | internal_midpoint_proxy_enriched_interpolated | 1 |
| CH | fast | 2027 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
| CH | fast | 2028 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
| CH | fast | 2029 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
| CH | fast | 2030 | official_proxy_enriched_scenario_mapped | 1 |
| CH | fast | 2031 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
| CH | slow | 2027 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
| CH | slow | 2028 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
| CH | slow | 2029 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
| CH | slow | 2030 | official_proxy_enriched_scenario_mapped | 1 |
| CH | slow | 2031 | official_proxy_enriched_scenario_mapped_interpolated | 1 |
