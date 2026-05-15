# Swiss CT Dataset Contract

## Goal
Build a governed Swiss CT dataset that is:
- causally available at auction time
- historically complete enough for robust backtests
- portable from local parquet files to Databricks and later to an operational serving layer

The current local folder layout is acceptable for research, but it is not the target operating model.

## Design Principles
- One canonical dataset contract for Swiss CT. Model work must depend on this contract, not on ad hoc local files.
- Separate `realized`, `forecast`, and `structural` inputs.
- Keep auction-time causality explicit. A variable is valid only if it is known before the forecast is issued.
- Version dataset freshness and coverage, not just model code.
- Prefer Databricks as the historical system of record; use Influx only for operational time-series serving if needed, not as the canonical backfill store.

## Target Logical Layers

### Bronze
Raw source-aligned tables with minimal transformation.
- `epex_price_ch_raw`
- `epex_price_de_raw`
- `entso_load_generation_raw`
- `entso_border_raw`
- `entso_forecast_raw`
- `outages_raw`
- `hydro_raw`
- `commodities_raw`
- `weather_forecast_raw`

### Silver
Harmonized, time-aligned, quality-checked tables.
- `ct_price_15min_ch`
- `ct_price_15min_de`
- `ct_price_15min_fr`
- `ct_price_15min_at`
- `ct_price_15min_it`
- `ct_entso_fundamentals_15min`
- `ct_entso_border_15min`
- `ct_forecast_de_renewables_15min`
- `ct_forecast_multi_country_15min`
- `ct_hydro_daily`
- `ct_outages_15min`
- `ct_commodities_daily`
- `ct_weather_forecast_hourly`

### Gold
Auction-ready feature views for model training and inference.
- `ct_features_j1_hourly`
- `ct_features_j1_15min`
- `ct_model_input_health`
- `ct_backtest_dataset_snapshot`

## Minimum Variable Groups

### Critical
- CH day-ahead price
- DE day-ahead price
- CH realized load, solar, wind, cross-border
- DE realized load, solar, wind, residual load
- DE day-ahead renewable forecast
- Multi-country `J+1` forecasts for load, solar, wind
- Weather forecasts relevant to CH and neighboring systems

### High
- FR, AT, IT day-ahead prices
- CH border capacities, schedules, net transfer balances
- FR nuclear stress / availability
- Hydro reservoir state and derived water value proxies
- Outages / generation unavailability

### Medium
- Fuel and CO2 curves
- Refined calendar / holiday mismatch / bridge-day features

## Data That Must Exist Historically

For a variable to be considered production-grade for Swiss CT:
- it must have sufficient historical depth for backtesting
- it must have controlled freshness
- it must be available under the same causal rules in both training and inference

If a feature exists only in the recent tail of history, it should not be injected directly into the main model without a governance decision.

## Current Gaps To Close

### Missing or incomplete major layers
- Multi-country `J+1` forecasts for load/solar/wind beyond DE renewables
- Weather forecast layer
- Fresh hydro state beyond the stale local cache
- Governed historical border / flow / forecast views prepared for direct model use

### Modeling implication
Do not compensate missing causal inputs by endlessly adding local heuristics. Fix the data contract first.

## Storage Recommendation

### Databricks
Use Databricks as the target canonical historical store:
- best for bronze/silver/gold separation
- easier batch backfills
- stronger governance and reproducibility
- better fit for feature views and offline backtests

Recommended table families:
- `market_data.ct_*` for silver
- `forecasting.ct_*` for gold

### Influx
Use Influx only if needed for:
- operational dashboards
- latest ticks / recent windows
- low-latency monitoring

Do not make Influx the only source of truth for historical CT training data.

## Governance Rules
- Every critical dataset needs:
  - owner
  - source system
  - update cadence
  - freshness SLA
  - backfill procedure
  - schema contract
- Every model benchmark must record the exact dataset snapshot used.
- Evaluation harnesses must use the same logical input set as production.

## Immediate Roadmap
1. Audit the current local dataset and classify each variable group as `ok`, `partial`, or `missing`.
2. Refresh and stabilize all critical local sources.
3. Add missing forecast layers before further model-architecture work.
4. Promote the governed local contract to Databricks silver/gold tables.
