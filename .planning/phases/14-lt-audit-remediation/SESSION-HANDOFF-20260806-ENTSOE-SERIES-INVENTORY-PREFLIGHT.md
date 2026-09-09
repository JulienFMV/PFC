# Session handoff - ENTSO-E series inventory preflight

Date: 2026-08-06  
Decision: D-20260806-243

## Outcome

Prepared but did not execute the single bounded query needed to inventory the
real `dev.gold.dimentsoeseries` business taxonomy. The query reads no fact
table and returns only semantic signatures, series counts, dimension load
timestamp bounds and truncation-detection totals.

The offline validator binds the exact query and immutable contract, rejects
more than one statement, truncation, more than 10,000 admissible signatures,
bad totals, duplicate signatures, malformed strings/timestamps, daily-date
drift, Warehouse start, Databricks writes and any authority escalation.

Literal normalized `GroupName` matches are reported for the 13 required
families and 13 additional high-value families, but are explicitly not family
coverage authority. Raw labels still require an owner-reviewed mapping.

## Cost and execution

- Databricks requests in D243: 0;
- SQL statements in D243: 0;
- Warehouse starts in D243: 0;
- Databricks writes in D243: 0;
- market/fact values opened in D243: 0.

The future capture contract requires a Warehouse already in `RUNNING` state,
zero Warehouse starts, an already consumed Europe/Zurich daily reservation and
at most one capture that day.

## Evidence

- contract raw SHA-256:
  `ba8f6945b4a43b54762fa475228edabea4018bfea5871f2581dc53536c0743a1`;
- contract content ID:
  `1e183fc51f2673cfc3fed0035dc4a5e3d84664f51ff8447a355264e5e819ddc6`;
- query SHA-256:
  `16a989d2b1528f79b3ecb7a2d9f8f221a6be67cc1c4d3c622516c8bd0dde95e7`;
- validator SHA-256:
  `b9ef41e6e54b032b79fc7451770ac8d4eb3d0712e2eee4e76f43aa554c198311`;
- tests SHA-256:
  `a22531918c7ed51a7f278c4e0aa02a1bcd1d85f6df04c7fdb0a1ae2a67568611`;
- focused tests: `15 passed in 0.10s`;
- adjacent D232/D234/D239/D240/D241/D243 slice: `143 passed in 6.25s`;
- Ruff: pass.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-SERIES-INVENTORY-CONTRACT-V1.json`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-SERIES-INVENTORY-PREFLIGHT-20260806.md`
- `docs/data/sql/entsoe_dev_gold_series_inventory.sql`
- `pfc_shaping/validation/entsoe_series_inventory.py`
- `tests/test_entsoe_series_inventory.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

## Next safe step

Do not execute while the Warehouse is stopped. When an appropriate Warehouse
is already running, consume the daily reservation and execute exactly the
bound query once. Persist only the sanitized inventory capture locally, then
run `assess_series_inventory` offline. After owner-reviewed raw-to-logical
mapping, classify each required and additional family as present, absent or
ambiguous. Fact-value quality and PIT admission remain later, separate gates.

Predecessor handoffs:

- `SESSION-HANDOFF-20260806-CH-LT-PROBABILISTIC-STATUS-FAIL-CLOSED.md`;
- `SESSION-HANDOFF-20260806-ENTSOE-DEV-CONTROL-PLANE-INVENTORY.md`.

