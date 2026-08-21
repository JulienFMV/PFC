# Session handoff - EEX causal Gold materialization and LT regression

Date: 2026-08-21

## Outcome

The existing Databricks EEX daily normalizer now has an explicit offline LT
materialization boundary. It consumes the exact 12-column joined projection of
`prd.gold.facteexpricedaily`, `prd.gold.dimeexproduct` and
`prd.gold.dimeexdeliveryperiod`, then:

- excludes rows whose `FactLoadTimestampUtc` is after the requested origin;
- excludes quotation dates after the Swiss market date of that origin;
- reuses the existing delivery-boundary and settlement-price normalizer;
- emits only Month/Quarter/Year history to the monthly-solver candidate frame;
- preserves all source and production authorities as false.

This is not a second EEX vintage system. The three-table query provenance,
external source-time evidence and conversion to the existing signed EEX
vintage catalogue remain required before model admission.

No Databricks connector, SQL statement, Warehouse start, network call or
remote write was used.

## Changed files

- `pfc_shaping/data/databricks_lt_materialization.py`
- `tests/test_databricks_lt_materialization.py`
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
- `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Verification

- baseline Databricks/solver/continuity matrix: `151 passed`;
- EEX materializer and normalizer matrix: `36 passed`;
- broad LT constraints/cascading/water-value/assembler/PIT/quality/Databricks
  regression: `341 passed, 3 skipped in 131.05s`;
- targeted Ruff: pass;
- `git diff --check`: pass;
- Databricks connections/statements/Warehouse starts: `0/0/0`.

## Counter-audit

- No call path grants the new output model, calibration, publication or
  production authority.
- `FactLoadTimestampUtc`, not quotation date alone, is the causal observation
  bound. This prevents a recent FTP backfill from being presented as if FMV
  had observed it historically.
- Source row order does not change the semantic source hash or either output
  frame.
- The joined Parquet is not yet accepted as proof of its three physical source
  tables; exact join SQL and table identities remain an export-manifest duty.
- Weather forecast activation remains deferred until source issue-time and
  station/coverage semantics are proven. Weather cannot set monthly levels.

## Next real-data steps

1. After the data engineer promotes the sources, create one bounded PRD export
   under the v4 manifest with exact query, predicates, watermarks and cost
   counters.
2. Generate the ENTSO-E `SeriesKey` mapping from the admitted Gold dimension
   and replay Silver vintages at several origins.
3. Bind the EEX three-table join and convert causal daily snapshots to the
   existing signed vintage catalogue.
4. Qualify observed weather and hydro as optional shape candidates. Keep
   forecast weather out of PIT backtests until its issuance timestamp is
   explicit.
5. Run rolling-origin LT comparison and a new independently frozen holdout.
   T057 remains sealed.

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.

Durable decision: D-20260821-255.
