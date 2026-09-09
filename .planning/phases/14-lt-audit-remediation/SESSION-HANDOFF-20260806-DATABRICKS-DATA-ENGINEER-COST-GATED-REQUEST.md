# Session handoff — Databricks data-engineer cost-gated request

Date: 2026-08-06  
Decision: D-20260806-288  
Status: modelling paused; request frozen; no remote execution authorized

## Outcome

The existing spot/weather/Swissgrid data-engineer request now has three
strictly separated phases:

1. Phase 0 returns only cost/planning metadata: Delta size and file count,
   partitions, estimated bytes read, already-active Warehouse identity, timeout
   and cost ceiling. It may not open business rows or start a Warehouse.
2. Phase A remains blocked until a separate written GO after Phase 0. If
   authorized later, it allows at most four read-only profiling statements,
   one execution each and no automatic retry.
3. Phase B, the immutable fact export/backfill, requires another GO after the
   profiles are reviewed.

The key correction is explicit: a final SQL `LIMIT` bounds rows returned, not
bytes scanned. None of the prepared SQL files is an execution authorization.

## Delivery expected after a future Phase A GO

- `spot_profile.parquet`;
- `weather_profile.parquet`;
- `swissgrid_balancing_profile.parquet`;
- `swissgrid_tender_profile.parquet`;
- `source_semantics.md`;
- `cost_receipt.json`;
- `manifest.json` binding each file's name, count, size, SHA-256 and creation
  timestamp.

The semantics request covers spot source/licence and curve mapping, weather
locations and forecast issue/vintages, Swissgrid units/signs/final-versus-
indicative status, tender product mapping and the still-unobserved NTC, flows,
scheduled exchange, redispatch and outage families.

## Files and hashes

- `.planning/phases/14-lt-audit-remediation/DATABRICKS-SPOT-WEATHER-SWISSGRID-DATA-ENGINEER-REQUEST-20260806.md`
  - SHA-256:
    `5c761405b566c010cfef93f9f8ed362c7da53b11abce87dee662882206c1bdc6`.
- `tests/test_databricks_source_data_engineer_request.py`
  - SHA-256:
    `1a1b623a65b7445e91f9bdc5f6a10fb2df232298250414fa0c9d370dd93a6826`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`.
- `.planning/HANDOFF.md`.
- this handoff.

The four prepared SQL files remain unchanged by D288 and explicitly
non-authorized under D287.

## Verification

- Ruff format/check on the new request test: pass.
- Request contract plus four SQL safety profiles:
  `11 passed in 0.20s`.
- D288 Databricks SQL statements: 0.
- business rows opened: 0.
- Warehouse starts: 0.
- Databricks writes: 0.
- network calls and `H:` accesses: 0.

## Authorities and next action

The request proves only that a future delivery can be bounded and audited.
It grants no source, data-quality, PIT, predictive, model-input, selection,
candidate, OMPEX-superiority, promotion or production authority.

Keep modelling paused. The next action belongs to the data engineer: return
Phase 0 only, or produce a governed export under the platform team's own cost
authority. Do not run Phase A merely because a Warehouse becomes visible; the
cost receipt and an explicit GO are both mandatory.
