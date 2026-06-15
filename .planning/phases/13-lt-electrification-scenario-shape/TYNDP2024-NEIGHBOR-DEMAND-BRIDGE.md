# TYNDP 2024 Neighbour Demand Bridge

* output: `data\electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet`
* status: `PARTIAL - production gate must fail`
* method: interpolate REF 2019 to official DE/GA 2040, then map lower/midpoint/higher to slow/central/fast
* peak and winter demand: scaled with latest complete historical ENTSO-E load year available before vintage
* rule: this is an internal bridge, not a governed production scenario

## Output Preview

| country | scenario | delivery_year | demand_twh | peak_load_gw | winter_demand_twh | quality_flag |
| --- | --- | --- | --- | --- | --- | --- |
| AT | central | 2030 | 91.1289 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| AT | fast | 2030 | 94.2423 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| AT | slow | 2030 | 88.0155 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| DE | central | 2030 | 698.3242 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| DE | fast | 2030 | 741.5245 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| DE | slow | 2030 | 655.1239 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| FR | central | 2030 | 495.7005 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| FR | fast | 2030 | 499.5305 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| FR | slow | 2030 | 491.8704 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| IT | central | 2030 | 359.8417 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| IT | fast | 2030 | 366.4681 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |
| IT | slow | 2030 | 353.2154 | nan | nan | internal_tyndp_demand_bridge_partial_proxy |

## Diagnostics

| country | ref_2019_twh | de_2040_twh | ga_2040_twh | historical_year | historical_annual_twh | peak_to_annual_gw_per_twh | winter_share | ratio_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT | 69.2024 | 117.0059 | 105.1184 | <NA> | nan | nan | nan | missing_complete_historical_load_year |
| DE | 556.2141 | 909.9885 | 745.0418 | <NA> | nan | nan | nan | missing_complete_historical_load_year |
| FR | 464.1197 | 531.7222 | 517.0984 | <NA> | nan | nan | nan | missing_complete_historical_load_year |
| IT | 310.5972 | 417.2598 | 391.9592 | <NA> | nan | nan | nan | missing_complete_historical_load_year |
