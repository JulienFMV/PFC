# Session handoff - Databricks zero-query acquisition plan

Date: 2026-08-05  
Decision: D-20260805-231  
Status: `PASS_STATIC_PLAN_ONLY_ZERO_DATABRICKS_EXECUTION`

## Outcome

D231 performs no Databricks, Warehouse or network action. It reuses the exact
existing local EEX `prd.gold` capture and freezes one future metadata-only
ENTSO-E schema statement without executing it. Current statement, Warehouse
start, network-call and remote-write budgets are zero.

The future metadata statement is not authorized by this batch. If the user
later authorizes it explicitly, the budget is one statement, 60 seconds and
1,024 metadata rows. Its execution must not be described as zero cost. The
ENTSO-E data-statement budget remains zero until exact physical-column mapping
and schema fingerprint admission.

## Changed files

- `.planning/phases/14-lt-audit-remediation/CH-LT-DATABRICKS-ZERO-QUERY-ACQUISITION-PLAN-V1.json`
- `docs/data/sql/entsoe_dev_gold_schema_inventory.sql`
- `pfc_shaping/validation/databricks_zero_query_acquisition_plan.py`
- `tests/test_databricks_zero_query_acquisition_plan.py`
- `build/databricks-eex-daily/materialize_zero_query_acquisition_plan.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057 or heavy desk-data file was opened or changed
by D231.

## Canonical identities

- plan raw SHA-256:
  `f62bd9e0a9ffe6f0daa2b02917b47763b64a340f28ab59cdd664cbdd8ec58999`
- plan canonical content ID:
  `3395f45bd1d22663386aa7cd4e93cfe2bc02079fa7cb8b103825b9c5650dc1af`
- ENTSO-E metadata SQL SHA-256:
  `dec7e207603e3a8b69f5808b42454575f1b4a985a25fa8aca3e5c6a2c95b72fa`
- validator SHA-256:
  `cd407a7209fbaf4f8ecdb35ef485676cd3ebab6ff6ff8b992a04e13336cc83fb`
- tests SHA-256:
  `4190774d9beb6124589698e5c0582081a7d378125cc1161e567d7d5c25d98ac2`
- materializer SHA-256:
  `a2c19771760191499b399de5dcd07e7893900875335a97bd4aa17a203dbbf237`
- research note SHA-256 after D231:
  `cd2653dcd425ff14614da497c7a635f5fab0f01144950de5f5b97729f4034c5e`

Reproducible proof:

- content ID:
  `127506f29101c98738d4fc876fb428295722a8253c4ee290e820b55ef67d3a83`
- manifest SHA-256:
  `40a13e9a4931d028c8ec296136b8395c5dd342adcd19b043658b5c45913c620a`
- assessment SHA-256:
  `3d6195e3763abf7b1c80189375eb6efa32f049a70bf74a2b10ee5e95d8d211df`
- path:
  `build/databricks-eex-daily/2026-08-05/zero-query-acquisition-plan-proofs/127506f29101c98738d4fc876fb428295722a8253c4ee290e820b55ef67d3a83/`

## Verification

- Ruff on validator, tests and materializer: passed.
- focused test: `25 passed in 0.16s`.
- adjacent acquisition/publication suite:
  `127 passed, 1 skipped, 1 warning in 3.80s`.
- materializer executed twice locally: identical proof content ID.
- AST, JSON, maximum-line and secret-pattern scan: passed.
- proof counters: zero Databricks request, Warehouse start, network call, `H:`
  access and remote write.

The adjacent warning is the pre-existing timezone-to-period warning from
`ingest_energy_charts.py`; it is unrelated to D231.

## Data-quality gates frozen for the future intake

Following the structured-data quality review, admission must cover:

- completeness and required-field mapping;
- uniqueness at dimension, latest and vintage grains;
- validity of timestamps, numeric values, units and native resolution;
- cross-table consistency and 100% fact-to-dimension referential integrity;
- freshness, gaps, backfills and revision behaviour;
- expected volume and shape without many-to-many expansion;
- point-in-time leakage, using `as_of_utc` at every rolling origin.

## Risks and next permitted step

Governed ENTSO-E values, exact physical column mapping, same-snapshot EEX/ENTSO-E
PIT evidence and a new independently frozen future holdout remain missing.
Training, selection, model input, candidate assembly, promotion and production
remain false. T057 stays sealed; the monthly solver remains sole level authority.

The next permitted remote action is only the prepared metadata query, and only
after a new explicit user authorization acknowledging that it may wake the SQL
Warehouse and incur cost. Until then, continue with local evidence only.
