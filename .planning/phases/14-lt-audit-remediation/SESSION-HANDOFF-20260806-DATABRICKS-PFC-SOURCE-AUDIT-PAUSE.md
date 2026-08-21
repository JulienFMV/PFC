# Session handoff — Databricks PFC source audit pause

Superseded for current Gold/NTC status by D291 and
`SESSION-HANDOFF-20260806-DATABRICKS-GOLD-NTC-SOURCE-CONTRACT.md`.

Date: 2026-08-06  
Decision: D-20260806-287  
Status: modelling paused; metadata discovery PASS; content admission pending

## Outcome

The user paused modelling until Databricks spot, weather, Swissgrid and ENTSO-E
tables are analysed. No AFRY, OMPEX or model batch may resume before this data
gate closes.

The complete visible control-plane inventory covers 534 `dev` and 406 `prd`
tables/views. It establishes:

- EEX derivatives are present in `prd` and already have one immutable local
  capture;
- granular spot exists as a candidate only in
  `dev.silver.ge_market_euler_spot`; its EPEX identity/licence and coverage are
  unproven;
- rich MeteoSwiss/Open-Meteo measurements and forecast histories exist in
  `prd`;
- Swissgrid control-area balance, balancing energy/prices and reserve tenders
  exist in `prd`;
- dedicated Swissgrid NTC, cross-border flow and redispatch tables were not
  observed among the accessible `prd` objects;
- ENTSO-E remains in `dev` and still requires the governed data-engineer
  delivery.

## Swissgrid findings

`prd.silver.ge_power_swissgrid_cab` exposes quarter-hour targets and aFRR,
scheduled/direct mFRR, RR, FRCE, system imbalance and long/short/one-price
fields. The source is explicitly described as indicative real-time data whose
billing may differ. Units, signs, revisions, coverage and final-settlement
reconciliation are therefore mandatory before modelling.

`prd.silver.ge_market_swissgrid_sdl_tenders` retains tender/product text,
countries, offered/awarded volumes, capacity prices and their units, source
snapshots and DQ flags. Exact FCR/aFRR/mFRR family mapping requires row content.

Reactive-power and node-billing tables are excluded from the baseline PFC
pending a specific economic hypothesis and independent predictive gain.

## Files

- `docs/research/DATABRICKS-PFC-SOURCE-AUDIT-20260806.md`
- `docs/research/DATABRICKS-SPOT-WEATHER-INVENTORY-20260806.md`
- `.planning/phases/14-lt-audit-remediation/DATABRICKS-SPOT-WEATHER-SWISSGRID-DATA-ENGINEER-REQUEST-20260806.md`
- `docs/data/sql/databricks_dev_spot_profile.sql`
- `docs/data/sql/databricks_prd_weather_profile.sql`
- `docs/data/sql/databricks_prd_swissgrid_balancing_profile.sql`
- `docs/data/sql/databricks_prd_swissgrid_tender_profile.sql`
- `tests/test_databricks_spot_weather_profile_sql.py`
- `build/databricks-eex-daily/2026-08-06/catalog-surface-prd-dev.json`
- `build/databricks-eex-daily/2026-08-06/selected-spot-weather-table-schemas.json`
- `build/databricks-eex-daily/2026-08-06/selected-swissgrid-table-schemas.json`
- this handoff

The three captures and their helper scripts are ignored under `build/`.

## Verification and cost boundary

- control-plane discovery GETs: 44 cumulative;
- SQL statements: 0;
- business rows opened: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- both visible classic `2X-Small` Warehouses were stopped with 45-minute
  auto-stop;
- combined request and static SQL roast final: `11 passed in 0.65s`, including
  the documented Databricks `TIMESTAMPADD` expression, explicit invalid spot
  frequencies, UTC-epoch quarter-hour alignment and empty-source duplicate
  handling;
- all four profiles are explicitly `NON AUTORISÉ EN L'ÉTAT`: their `LIMIT`
  bounds returned rows, not scanned bytes. Delta size, partitions and estimated
  plan must be cost-gated first;
- technical report/source-matrix SHA-256:
  `1ed339a20b044dbc6a899fcb92bae6513f61fc1c51ea198996b687ced651c244` /
  `c6482c32b704ff85e987588e19f6c77420cad897b0e12a2336ea1a27c4fb48f4`;
- focused test SHA-256:
  `d710e1e35a23075ff64135a59ff566f555f9097ec30d6c5ce2baef3da859d71a`;
- profile-first data-engineer request SHA-256:
  `5c761405b566c010cfef93f9f8ed362c7da53b11abce87dee662882206c1bdc6`;
- spot/weather/Swissgrid-balancing/Swissgrid-tender SQL SHA-256:
  `2852484c01335a348475230d5cd9c91c1d6f6ad70edb4dddba93e9c8647656c0` /
  `8183f34dfd9ce322b0ec4011145d62d4984e3250807c4aa107aec644ebae049c` /
  `12bfcd9dbd12ac337c0e701116a50d612a8737507f75e487e9e5626101a32be3` /
  `acfe34cbf60c96ad221be1ec51dde4349164dccdef3d0cf843fa0723fdb814d2`;
- `sqlglot` was not installed, so no offline dialect parse was claimed.

No CT, Power BI, AFRY, OMPEX, T057, `H:` or heavy desk-data file was opened or
changed.

## Next authorized work

Keep modelling paused. The next content step needs either:

1. a governed data-engineer export produced under its own cost authority, or
2. a separately authorized moment when a SQL Warehouse is already running,
   after the scan-cost gate is satisfied.

Do not start a stopped Warehouse under the current no-cost policy. Once results
are copied below `build/`, run exact local checks for grain, duplicates,
coverage, gaps, freshness, revisions, units/signs, source reconciliation and
PIT before considering any feature experiment.

The data-engineer request now enforces a profile-first sequence: four small
results plus source semantics and a hash manifest, with no backfill or fact dump
before an explicit GO.
