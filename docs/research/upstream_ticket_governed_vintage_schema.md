## Upstream Ticket: Governed Forecast Vintage Schema

Affected datasets:
- `pfc_shaping/data/multi_country_forecast_15min.parquet`
- `pfc_shaping/data/weather_forecast_hourly.parquet`

Problem:
- Current governed forecast caches are indexed only by delivery timestamp.
- They do not expose the forecast publication timestamp.
- As a result, CT A/B results using governed features are only an upper-bound until causality is proven.

Required schema:
- `publication_ts` (UTC)
- `delivery_ts` (UTC)
- one value column per signal

Required constraint:
- `publication_ts <= delivery_ts - 13h`

Reason:
- This approximates the SDAC D-1 11:00 CET cutoff for Swiss CT day-ahead usage.
- Any row violating this rule can create look-ahead bias in backtests.

Required upstream action:
1. Persist governed forecast caches with both timestamps.
2. Preserve vintage history when backfilling.
3. Document whether each source is true D-1 day-ahead forecast, nowcast, or later revision.

Current local mitigation:
- CT code now flags `vintage_schema_verified = false` when the schema is absent.
- If the schema is present, rows violating the cutoff are excluded from governed pivots.
