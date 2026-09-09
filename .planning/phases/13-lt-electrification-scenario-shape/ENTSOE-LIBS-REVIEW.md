# ENTSO-E Client Libraries Review

Reviewed repositories:

* `EnergieID/entsoe-py`
* `krose/entsoeapi`

Date: 2026-06-12

## Decision

Keep `entsoe-py` as the production Python dependency. Do not introduce an R
runtime dependency through `entsoeapi`.

Use `entsoeapi` as a design reference for a better ENTSO-E bronze layer:
endpoint URL traceability, up-front argument validation, request pagination,
compressed response handling, EIC/code definition enrichment and explicit point
timestamps.

## Why This Helps PFC LT

`entsoe-py` already covers most data families needed by the LT stack:

| LT data family | `entsoe-py` fit | use in PFC LT |
|---|---|---|
| Actual load/generation | strong | existing `entso_15min` physical calibration and neighbour residual-load features |
| Load / RES / generation forecasts | strong | CT known-future and vintage-safe forecast snapshot archive |
| Installed generation capacity | strong | historical capacity calibration and actualization checks |
| Cross-border flows and schedules | strong | CH basis diagnostics and border pressure regimes |
| NTC/offered capacity | partial | useful where ENTSO-E publishes data; CH still needs Swissgrid/JAO/TYNDP fallback |
| Hydro storage | useful | water-value history and hydro state diagnostics, not a 2030 reservoir-capacity scenario |
| Outages | useful | FR nuclear and neighbouring availability regimes |

The main missing layer is not the API wrapper. It is our own governed storage
contract:

```text
raw XML/ZIP response
request URL / endpoint parameters
created_date_time / revision_number when available
ingested_at_utc
as_of_utc / snapshot date
target interval timestamps in UTC
quality flag and gap diagnostics
```

## Implementation Recommendation

1. Add `pfc_shaping/data/entsoe_bronze.py`.
   It should wrap `EntsoeRawClient` and persist raw XML/ZIP plus a manifest row
   per request. This is the audit layer.

2. Keep `EntsoePandasClient` only for convenience transforms.
   The Pandas output is useful for model features, but raw responses are needed
   for publication/revision audit and vintage-safe replay.

3. Add a normalized long table writer.
   Target schemas:
   `entso_timeseries`, `entso_exchanges`, `entso_forecasts`,
   `entso_outages`, `entso_installed_capacity`.

4. Add daily forecast snapshots.
   Forecast rows must have `target_ts_utc`, `as_of_utc`, `source_endpoint`,
   `doc_type`, `process_type`, `value`, `unit`. Never overwrite prior snapshots.

5. Keep Swissgrid for CH NTC production baseline.
   `entsoe-py` exposes NTC/offered-capacity methods, but the current local
   ENTSO-E cache has zero usable CH NTC observations. Swissgrid remains the
   better CH-specific source until a governed TYNDP/JAO LT path is wired.

## Version Note

The repo currently pins `entsoe-py==0.7.11`. GitHub shows a newer `v0.8.0`
release in April 2026. Upgrade only behind a focused regression pass because
parser output shapes can change and our ingestion code relies on column names
and MultiIndex conventions.

## No-Go Items

* Do not use ENTSO-E forecasts without `as_of_utc`.
* Do not zero-fill missing generation, NTC, outage or capacity values in the
  bronze/silver production path.
* Do not treat ENTSO-E historical capacity or hydro storage as a governed 2030
  scenario. They are calibration/actualization inputs, not LT assumptions.
* Do not add `entsoeapi` as a runtime dependency unless an R-based validation
  notebook is explicitly requested.
