# Session handoff — Databricks spot/weather control-plane inventory

Date: 2026-08-06  
Status: metadata presence established; data/PIT/model authority false

## Outcome

A bounded Unity Catalog control-plane inventory covered every visible table in
the accessible `dev` and `prd` catalogs without SQL compute. It found 534
objects in `dev` and 406 in `prd`.

Spot evidence is development-only:

- `dev.silver.ge_market_euler_spot` exposes interval-level decimal prices,
  native frequency, product/curve and quotation timestamps;
- `dev.gold.factspotpricemonthly` is monthly aggregation only;
- no equivalent spot/Euler table is visible in `prd`;
- `prd.gold.facteexpricedaily` remains EEX derivatives settlement evidence,
  not spot truth.

Weather evidence exists in `prd`:

- `prd.gold.factweather`;
- `prd.gold.factweatherforecasthistms`;
- `prd.gold.factweatherforecasthistom`;
- corresponding MeteoSwiss measurement/forecast and Open-Meteo forecast
  tables in `prd.silver`.

Schemas prove rich weather variables but not national spatial coverage or
forecast point-in-time availability. `MinLeadTime`/`lead_time` exists, while an
explicit forecast issue timestamp is not visible.

## Files

- `docs/research/DATABRICKS-SPOT-WEATHER-INVENTORY-20260806.md`
- `docs/data/sql/databricks_dev_spot_profile.sql`
- `docs/data/sql/databricks_prd_weather_profile.sql`
- `tests/test_databricks_spot_weather_profile_sql.py`
- `build/databricks-eex-daily/capture_catalog_surface.py`
- `build/databricks-eex-daily/capture_table_schemas.py`
- `build/databricks-eex-daily/2026-08-06/catalog-surface-prd-dev.json`
- `build/databricks-eex-daily/2026-08-06/selected-spot-weather-table-schemas.json`
- this handoff

The `build/` helpers and captures are ignored local evidence.

## Execution boundary

- control-plane GETs: 28;
- SQL statements: 0;
- business rows opened: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- both visible classic `2X-Small` Warehouses were `STOPPED` with a 45-minute
  auto-stop setting.

No CT, Power BI, AFRY, OMPEX, T057, `H:` or heavy desk-data file was opened or
changed.

## Next safe batch

Do not start a Warehouse merely for this inventory. When one is already active
under an authorized workload, run one bounded read-only profiling batch for:

1. spot source identity/licence, curves, countries, frequency, history,
   completeness and duplicates;
2. weather locations/coordinates, sources, algorithms, frequency, history,
   gaps and revisions;
3. authoritative forecast issue-time semantics and PIT eligibility.

Repatriate the bounded result under `build/` and perform every subsequent check
offline. Until then, source, PIT, predictive, model-input, selection, candidate,
promotion and production authorities remain false.

The two profiling SQL files are prepared but were not executed. A static local
roast forbids mutation tokens, unexpected source tables, invented forecast issue
time and unbounded output.
