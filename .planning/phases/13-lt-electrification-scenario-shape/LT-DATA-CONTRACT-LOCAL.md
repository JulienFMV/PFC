# LT Data Contract Local Materialization

* local root: `C:\Users\jbattaglia\pfc_local_data`
* vintage: `2026-06-05`
* scenarios: `slow, central, fast`
* delivery years: `2027-2031`
* canonical scenario output: `data\electrification_scenarios.parquet`
* canonical feature output: `data\hpfc_scenario_features.parquet`
* production complete: `NO - local proxy only`

## Local Cache Directories

| path |
|---|
| C:\Users\jbattaglia\pfc_local_data\entsoe\raw_xml |
| C:\Users\jbattaglia\pfc_local_data\entsoe\bronze_timeseries |
| C:\Users\jbattaglia\pfc_local_data\entsoe\bronze_exchanges |
| C:\Users\jbattaglia\pfc_local_data\entsoe\bronze_forecasts |
| C:\Users\jbattaglia\pfc_local_data\scenarios\ofen_ep2050 |
| C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024 |
| C:\Users\jbattaglia\pfc_local_data\scenarios\pronovo |
| C:\Users\jbattaglia\pfc_local_data\scenarios\mastr |
| C:\Users\jbattaglia\pfc_local_data\scenarios\swissgrid_ntc |
| C:\Users\jbattaglia\pfc_local_data\market\eex |
| C:\Users\jbattaglia\pfc_local_data\market\epex |
| C:\Users\jbattaglia\pfc_local_data\market\commodities |

## Model Asset Audit

| asset | path | exists | rows | columns | purpose | error |
|---|---|---|---|---|---|---|
| EPEX CH spot | pfc_shaping\data\epex_15min.parquet | yes | 112016 | 2 | historical price shape calibration |  |
| EPEX DE spot | pfc_shaping\data\epex_de_15min.parquet | yes | 112316 | 2 | cross-border/CT diagnostics |  |
| EPEX CH hourly | data\epex_hourly.parquet | yes | 28079 | 1 | LT perfect-foresight and local PFC runner |  |
| ENTSO-E wide physicals | pfc_shaping\data\entso_15min.parquet | yes | 186093 | 64 | LT/CT physical features |  |
| Hydro reservoir | pfc_shaping\data\hydro_reservoir.parquet | yes | 1367 | 8 | water-value diagnostics |  |
| Generation outages | pfc_shaping\data\outages_15min.parquet | yes | 86016 | 5 | availability diagnostics |  |
| EEX forwards | data\eex_forwards_history.parquet | yes | 148748 | 7 | market anchors |  |
| Commodities | data\commodities_cache.parquet | yes | 503 | 3 | thermal/basis diagnostics |  |

## Canonical Scenario Contract

| file | rows | status |
|---|---|---|
| data\electrification_scenarios.parquet | 15 | written and vintage-gated |
| data\hpfc_scenario_features.parquet | 15 | written from canonical scenario file |

## External Data Still Required For 10/10 Production

| source | gap |
|---|---|
| TYNDP 2024 multi-country scenarios | missing official DE/FR/IT/AT backbone in canonical local table |
| Pronovo CH PV actuals | missing vintage actualization of CH PV trajectory |
| MaStR DE PV/battery actuals | missing DE PV/battery commissioning actualization |
| ENTSO-E forecast snapshots | current DE renewable forecast lacks as_of_utc and is not vintage-safe |
| Swissgrid/TYNDP/JAO long-term NTC | CH-AT and governed NTC scenario feed not loaded |
| Fuel/CO2/electrolysis/DSM scenarios | thermal-set and belly-refill structural drivers not loaded |

## Decision

The local project is now wired to the canonical Phase 13 model paths using the best available governed local EP2050 proxy inventory. This is sufficient for local smoke/prod-readiness runs with `require_electrification_scenarios=True`, but it is not sufficient to enable Phase 13 as final production signal until the external gaps above are loaded and validated.
