# ENTSO-E series inventory preflight

Date: 2026-08-06  
Decision: D-20260806-243

## Outcome

One bounded, dimension-only SQL statement and its offline admission validator
are ready. Neither was executed against Databricks in this batch.

The query reads only `dev.gold.dimentsoeseries`; no fact table is referenced.
It groups the exact ENTSO-E semantic descriptors, returns series counts and
dimension load-timestamp bounds, and includes window totals so truncation is
detectable. It returns at most 10,001 rows while the validator admits at most
10,000 signatures.

## Cost controls

Execution is permitted by the contract only when:

- the selected SQL Warehouse is already `RUNNING`;
- no Warehouse start is attributed to the capture;
- the Europe/Zurich daily capture reservation has been consumed;
- this is the only capture for that Europe/Zurich day;
- the statement remains read-only and produces no Databricks write.

The static contract itself grants no query-execution authority. Current batch
counters are zero Databricks requests, zero SQL statements, zero Warehouse
starts and zero Databricks writes.

## Interpretation boundary

The validator reports literal normalized-name matches against the 13 required
families and 13 additional high-value families. A literal match is diagnostic,
not coverage authority. An owner-reviewed mapping of the raw `GroupName`,
`FieldName`, ENTSO-E types, zones and units remains required before a family can
be declared present or absent.

The inventory also exposes signature and series totals, null-bearing semantic
fields weighted by series count, observed units including bare `EUR` versus
`EUR/MWh`, and dimension load-timestamp bounds.

It does not open market/fact values and does not prove freshness, temporal
coverage, native cadence, sign convention, point-in-time history, model value
or production readiness.

## Evidence

- contract raw SHA-256:
  `ba8f6945b4a43b54762fa475228edabea4018bfea5871f2581dc53536c0743a1`;
- contract canonical content ID:
  `1e183fc51f2673cfc3fed0035dc4a5e3d84664f51ff8447a355264e5e819ddc6`;
- query SHA-256:
  `16a989d2b1528f79b3ecb7a2d9f8f221a6be67cc1c4d3c622516c8bd0dde95e7`;
- validator SHA-256:
  `b9ef41e6e54b032b79fc7451770ac8d4eb3d0712e2eee4e76f43aa554c198311`;
- focused roast: `15 passed`;
- adjacent ENTSO-E matrix: `143 passed`;
- Ruff: pass.

## Files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-SERIES-INVENTORY-CONTRACT-V1.json`
- `docs/data/sql/entsoe_dev_gold_series_inventory.sql`
- `pfc_shaping/validation/entsoe_series_inventory.py`
- `tests/test_entsoe_series_inventory.py`

