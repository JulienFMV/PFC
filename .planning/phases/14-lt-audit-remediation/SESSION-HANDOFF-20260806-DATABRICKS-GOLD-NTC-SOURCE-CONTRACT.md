# Session handoff — Databricks Gold and NTC source contract

Date: 2026-08-06  
Decision: D-20260806-291  
Status: modelling paused; Gold contract frozen; content export pending

## Outcome

The Databricks inventory has been converted into decisions rather than
questions for the data engineer. Silver remains an audit/reconciliation layer.
The PFC consumption boundary is governed Gold followed by one immutable
Parquet export under repo-local
`build/databricks-exports/<snapshot_id>` on `C:`. Iterative profiling, feature
engineering and backtests must then run locally without repeated Databricks
scans.

## Current Gold gaps

- Spot: no interval fact exists in `prd.gold`; the only observed points are in
  `dev.silver.ge_market_euler_spot`, while
  `dev.gold.factspotpricemonthly` is too aggregated for shape.
- Weather: Gold facts exist in `prd`, but forecast issue/vintage semantics are
  not explicit.
- Swissgrid balancing: Gold keeps only system imbalance and one-price; Silver
  contains the relevant aFRR/mFRR/RR/NRV/FRCE and long/short components plus
  file lineage.
- ENTSO-E: dimension/latest/vintages exist only in `dev.gold`; no equivalent
  surfaces were observed in `prd.gold`.
- No dedicated NTC, cross-border flow, scheduled exchange, redispatch or
  outage object was found among the 406 visible `prd` objects.

## NTC verdict

The 2026-08-03 ENTSO-E audit observed day/month/year-ahead NTC, physical flows
and scheduled exchanges as macro-families in `dev.gold`. Exact current series,
four-border/two-direction coverage, revisions and point-in-time availability
remain unproved. Intraday NTC is not proved.

The user-provided `NTC-202609.pdf` was copied byte-for-byte from Downloads to
ignored `build/external-inputs/swissgrid-ntc/`. It is a four-page Swissgrid
month-ahead forecast for September 2026, version 1, dated 2026-07-28. It covers
CH-DE and CH-AT in both directions at hourly grain, but not CH-FR or CH-IT.
Swissgrid states that monthly NTC is indicative, D-2 and intraday are separate,
and CH-IT/IT-CH is published on ENTSO-E.

Required Gold metric types are therefore distinct:

- `NTC_MONTH_AHEAD`;
- `NTC_DAY_AHEAD_D2`;
- `NTC_INTRADAY`;
- `SCHEDULED_EXCHANGE`;
- `PHYSICAL_FLOW`.

They must retain source, border, direction, target interval, native resolution,
value/unit, publication/first-observation/ingestion timestamps, document ID and
revision. Duplicate Swissgrid/ENTSO-E observations are reconciled, never
silently coalesced.

## Expected Gold surfaces

- `prd.gold.dimspotproduct` and new `prd.gold.factspotpriceinterval`;
- the existing three `prd.gold` weather facts, enriched with governed forecast
  origin and vintages;
- enriched `prd.gold.factswissgridbalancingquarterhourly` and traceable
  `prd.gold.factswissgridtenderofferresult`;
- promoted/enriched `prd.gold.dimentsoeseries`,
  `prd.gold.factentsoetimeserieslatest` and
  `prd.gold.factentsoetimeseriesvintages`.

## Cost boundary and validator binding

D287 modelling pause and D288 Phase 0 remain active. The D290 value-blind
cost-preflight validator was rebound to the current D291 request hash. It still
cannot authorize Phase A or start a Warehouse. The previous D290 request
binding and synthetic proof are historical only.

## Files and hashes

- Gold/NTC request:
  `43a838a159267e59eae165eb77a551666b31f0ba7c5b9d5554aa9b9dd7a3f67f`;
- ENTSO-E request:
  `288995d5d4abf6b6e6a3fb2e5d055ccd76ba5993dce87ef0a8ffa43238a5a96b`;
- source audit:
  `7e2499a51a37b274e9d4277a08f3bc0bd055c15c49d69991bd0a9e88bf0fb799`;
- cost contract raw/content:
  `96051c3843ddea6d9f09a26241767be7453c33c449c04bfd6e34547a7a7efdc9` /
  `3428e12004eb7ec69f8b600afd557eba9e94923d6bbcedca7c38664e53e03d6c`;
- cost validator:
  `88b3a6dea8d4d99af32c1aa1dcd06df2aff29fa9eaab5b1b8bf731cdbc849a0f`;
- local Swissgrid PDF:
  `3b690c5f281321dd16609e2db168fc67f986d05432b864a82886dd6439b6de36`.

Verification: request, SQL and cost-preflight roast
`47 passed in 0.81s`. Databricks SQL/business rows/Warehouse starts/writes in
D291: `0/0/0/0`. No `H:` access, CT, Power BI, AFRY, OMPEX, T057 or heavy desk
data was touched.

## Next action

Send the corrected Gold/NTC request to the data engineer. Accept only the D291
Phase 0 cost receipt or a governed platform export under the data team's cost
authority. Do not run Phase A or resume modelling until the actual Gold
profiles and source semantics have been inspected locally.
