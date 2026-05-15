# Swiss CT Data Implementation Plan

## Objective
Prepare a clean implementation path for Swiss CT data imports so the local parquet layout can evolve into a governed Databricks-first architecture, with a possible operational serving layer later.

This document is the execution bridge between:
- the conceptual contract in [swiss_ct_dataset_contract.md](./swiss_ct_dataset_contract.md)
- the actual import code that must be written or refactored next

## Guiding Choice
Use **Databricks as system of record** for historical CT training datasets.

Use **local parquet only as a developer cache**.

Use **Influx only later if needed for operational serving / dashboards / latest values**, not for the authoritative historical training store.

## Build Order

### Wave 1: Stabilize Current Critical Inputs
These are the first imports to make production-grade because the current CT model already depends on them.

1. `ct_price_15min_ch`
2. `ct_price_15min_de`
3. `ct_entso_fundamentals_15min`
4. `ct_forecast_de_renewables_15min`
5. `ct_hydro_daily`
6. `ct_outages_15min`
7. `ct_commodities_daily`

### Wave 2: Complete Swiss Cross-Border Coverage
These inputs are necessary to close the “missing fundamentals” gap.

1. `ct_price_15min_fr`
2. `ct_price_15min_at`
3. `ct_price_15min_it`
4. `ct_entso_border_15min`
5. `ct_forecast_multi_country_15min`
6. `ct_weather_forecast_hourly`

### Wave 3: Gold Feature Views
After silver is trustworthy:

1. `ct_features_j1_hourly`
2. `ct_features_j1_15min`
3. `ct_model_input_health`
4. `ct_backtest_dataset_snapshot`

## Dataset-by-Dataset Plan

### 1. CH price
- Target silver table: `market_data.ct_price_15min_ch`
- Local cache: `pfc_shaping/data/epex_15min.parquet`
- Source now: `energy-charts`
- Grain: `15min`
- Time key: `timestamp_utc`
- Primary columns:
  - `timestamp_utc`
  - `price_eur_mwh`
  - `source_system`
  - `ingestion_ts_utc`
- Freshness SLA: `T+0 daily after market publication`
- Backfill status: already strong
- Work needed:
  - unify schema
  - add source metadata and ingestion timestamp

### 2. DE price
- Target silver table: `market_data.ct_price_15min_de`
- Local cache: `pfc_shaping/data/epex_de_15min.parquet`
- Source now: `energy-charts`
- Grain: `15min`
- Time key: `timestamp_utc`
- Work needed:
  - same governance as CH

### 3. FR/AT/IT prices
- Target silver tables:
  - `market_data.ct_price_15min_fr`
  - `market_data.ct_price_15min_at`
  - `market_data.ct_price_15min_it`
- Local caches:
  - `pfc_shaping/data/epex_fr_15min.parquet`
  - `pfc_shaping/data/epex_at_15min.parquet`
  - `pfc_shaping/data/epex_it_15min.parquet`
- Current role:
  - used as neighbor prices
- Work needed:
  - same schema contract as CH/DE
  - explicit freshness monitoring

### 4. ENTSO fundamentals
- Target silver table: `market_data.ct_entso_fundamentals_15min`
- Local cache: `pfc_shaping/data/entso_15min.parquet`
- Source now: ENTSO-E / energy-charts mixed
- Grain: `15min`
- Time key: `timestamp_utc`
- Logical sections:
  - CH realized load / solar / wind / cross-border
  - DE realized load / solar / wind / residual
  - FR / AT / IT realized load / solar / wind
- Work needed:
  - separate `realized fundamentals` from `border metrics`
  - add `coverage_start`, `coverage_end`, and ingestion metadata
  - explicitly record source provenance by sub-family

### 5. ENTSO border metrics
- Target silver table: `market_data.ct_entso_border_15min`
- Source now: contained in `entso_15min.parquet`
- Grain: `15min`
- Time key: `timestamp_utc`
- Required fields:
  - `scheduled_net_export_ch_de_mw`
  - `scheduled_net_export_ch_fr_mw`
  - `scheduled_net_export_ch_at_mw`
  - `scheduled_net_export_ch_it_mw`
  - `ntc_export_*`
  - `ntc_import_*`
  - `ntc_total_*`
  - `ntc_balance_*`
  - z-scored variants only in gold, not necessarily in silver
- Work needed:
  - split raw border metrics from engineered metrics
  - track causality and availability timing

### 6. DE renewable forecast
- Target silver table: `market_data.ct_forecast_de_renewables_15min`
- Local cache: `pfc_shaping/data/de_renewable_forecast.parquet`
- Source now: ENTSO-E
- Grain: `15min`
- Time keys:
  - `forecast_for_ts_utc`
  - `available_at_ts_utc` if available from source
- Required fields:
  - `forecast_wind_de_mw`
  - `forecast_solar_de_mw`
- Work needed:
  - preserve forecast issuance semantics
  - not just overwrite with “latest known”

### 7. Multi-country forecasts
- Target silver table: `market_data.ct_forecast_multi_country_15min`
- Countries:
  - `CH`
  - `DE`
  - `FR`
  - `AT`
  - `IT`
- Required forecast families:
  - `load`
  - `solar`
  - `wind`
- This is currently the biggest missing block.
- Work needed:
  - source selection first
  - then unified schema by `country_code`, `variable_name`

### 8. Weather forecasts
- Target silver table: `market_data.ct_weather_forecast_hourly`
- Required variables:
  - `temperature`
  - `cloud_cover`
  - `solar_irradiance`
  - `wind_speed`
  - `wind_direction`
- Geography:
  - at least one CH point
  - one or more DE renewable-heavy points
  - likely FR and North Italy points
- Work needed:
  - choose provider
  - define issuance/validity keys cleanly

### 9. Hydro
- Target silver table: `market_data.ct_hydro_daily`
- Local cache: `pfc_shaping/data/hydro_reservoir.parquet`
- Current problem:
  - stale local data
- Required fields:
  - `fill_pct`
  - `fill_gwh`
  - regional sub-basins if available
  - derived `water_value_proxy`
- Work needed:
  - refresh reliability first
  - then extend scope

### 10. Outages
- Target silver table: `market_data.ct_outages_15min`
- Local cache: `pfc_shaping/data/outages_15min.parquet`
- Work needed:
  - keep as governed structural signal
  - optionally split by technology / country in future

### 11. Commodities
- Target silver table: `market_data.ct_commodities_daily`
- Local cache: `data/commodities_cache.parquet`
- Current role:
  - secondary structural context
- Work needed:
  - decide if public proxy remains acceptable
  - else migrate to governed FMV market-data source

## Coding Strategy

### Phase A: Registry-driven imports
All new import code should be written against a registry, not hardcoded paths.

Needed components:
1. dataset registry file
2. generic loader helpers
3. freshness / coverage audit
4. source-specific importer modules

### Phase B: Separate raw and engineered data
Do not keep mixing:
- raw source values
- engineered z-scores
- model-facing compact features

Recommended:
- raw source -> silver raw-normalized table
- engineered metrics -> gold feature table

### Phase C: Snapshot-aware backtests
Every benchmark run should bind to:
- a dataset snapshot date
- a coverage audit
- a feature manifest

## Immediate Coding Backlog

### First coding block
1. introduce registry-driven dataset metadata
2. refactor current local loaders to reference registry names
3. add a generic freshness audit command

### Second coding block
1. create proper silver split between:
   - `ct_entso_fundamentals_15min`
   - `ct_entso_border_15min`
   - `ct_forecast_de_renewables_15min`
2. make local parquet caches conform to those schemas

### Third coding block
1. add `ct_forecast_multi_country_15min`
2. add `ct_weather_forecast_hourly`

## Success Criteria
The dataset layer is good enough when:
- all critical inputs are `ok` in the audit
- production and evaluation use the same logical inputs
- dataset snapshots can be reproduced
- migration to Databricks is a storage change, not a semantic rewrite
