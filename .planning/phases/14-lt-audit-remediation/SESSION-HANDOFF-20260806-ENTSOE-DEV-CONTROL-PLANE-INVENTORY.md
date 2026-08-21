# Session handoff - ENTSO-E dev control-plane inventory

Date: 2026-08-06

## Outcome

Read-only Databricks control-plane calls confirmed the current schema of the
three ENTSO-E tables in `dev.gold` without running SQL or starting a Warehouse.

- `dimentsoeseries`: `SeriesID`, `FieldName`, `GroupName`, `DocumentType`,
  `BusinessType`, `ProcessType`, `PsrType`, `FromZone`, `ToZone`, `Unit`,
  `Meta_Load_Timestamp`.
- `factentsoetimeserieslatest`: `SeriesID`, `DateTimeUtc`, `DateUtc`,
  `FieldValue`, `Epoch`, `PublicationTimestampUtc`, `IsHistorical`,
  `Meta_Load_Timestamp`.
- `factentsoetimeseriesvintages`: `VintageID`, `SeriesID`, `DateTimeUtc`,
  `DateUtc`, `FieldValue`, `Epoch`, `PullTimestampUtc`,
  `PublicationTimestampUtc`, `IsHistorical`, `Meta_Load_Timestamp`.

All three are managed Delta tables. `DateTimeUtc` is documented as the UTC
right edge of the ENTSO-E interval.

## Findings

The real schema does not satisfy the current normalized intake contract:

- no native resolution per series;
- no source endpoint or source document identifier;
- no quality flag or revision number in either fact;
- no explicit sign convention for directional series;
- key descriptive and timestamp fields are nullable;
- PIT semantics do not yet select and document a canonical `as_of_utc` from
  `PublicationTimestampUtc` and `PullTimestampUtc`.

The exact distinct `GroupName`/`FieldName` inventory remains unproven. The
2026-08-03 observation that useful CH groups existed is informative but not a
fresh, hash-bound exhaustive inventory.

Additional high-value ENTSO-E families to look for are production/network
unavailability, installed capacity, balancing energy prices and volumes,
imbalance/system-balance measures, reserve-capacity procurement, redispatch and
countertrading, net positions and evolving intraday cross-zonal capacity.

## Cost and execution evidence

- `.env` exists, exposes only the expected variable names and is Git-ignored;
  no secret value was printed or persisted.
- Configured Warehouse: classic `2X-Small`, stopped, 45-minute auto-stop.
- Other visible DEV Warehouse: classic `2X-Small`, stopped, 45-minute auto-stop.
- successful control-plane GETs: 5;
- SQL statements: 0;
- Warehouse starts: 0;
- Databricks data writes: 0;
- market-value or ENTSO-E fact rows opened: 0.

The D241 purpose-built capture then performed exactly three additional Unity
Catalog table-metadata GETs. It performed zero SQL statements, opened zero
table rows, started zero Warehouses and wrote nothing to Databricks. Therefore
the full session/day total is eight successful control-plane GETs and still
zero SQL, Warehouse starts, table rows and Databricks writes.

## D241 reproducible schema-admission evidence

- proof ID:
  `d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544`;
- capture content ID:
  `d69fdab73ba1d9c55f70f77925f2253d583564d06d922f2b41e035763bca176f`;
- local path:
  `build/databricks-control-plane/2026-08-06/entsoe-unity-catalog-schema-captures/d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544`;
- status:
  `FAIL_REAL_CONTROL_PLANE_SCHEMA_INCOMPATIBLE_NO_MODEL_AUTHORITY`;
- exact column counts: dimension 11, latest 8, vintages 10;
- no mapped-field type failure;
- missing logical fields: dimension `native_resolution`, `source_endpoint`,
  `document_id`; latest `quality_flag`, `revision_number`; vintages
  `as_of_utc`, `quality_flag`, `revision_number`;
- both `PublicationTimestampUtc` and `PullTimestampUtc` exist as possible
  vintage availability timestamps, but neither has canonical authority.

The capture is sanitized: it persists a workspace-host SHA-256 only and never
persists the token or host. The validator requires exact tables, managed Delta,
columns, positions, metadata types, execution counters and all-false authority.
It refuses authority escalation and ambiguous or structurally altered input.

Verification:

- Ruff: pass on the validator, capture script and focused test;
- focused pytest: `11 passed in 0.11s`;
- no table-value or SQL test was run.

## Changed implementation

- `pfc_shaping/validation/entsoe_unity_catalog_schema_compatibility.py`
- `build/databricks-eex-daily/capture_entsoe_unity_catalog_schema.py`
- `tests/test_entsoe_unity_catalog_schema_compatibility.py`

The initial standard-library HTTP attempt failed locally while loading the
Windows certificate store and did not reach Databricks. The retry used the
installed HTTP library CA bundle.

## Changed documentation

- `.planning/phases/14-lt-audit-remediation/ENTSOE-DEV-CONTROL-PLANE-INVENTORY-20260806.md`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
- `pfc_shaping/validation/entsoe_unity_catalog_schema_compatibility.py`
- `build/databricks-eex-daily/capture_entsoe_unity_catalog_schema.py`
- `tests/test_entsoe_unity_catalog_schema_compatibility.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

## Next safe step

Run one dimension-only grouped inventory query after either:

1. an appropriate Warehouse is already running; or
2. the user explicitly approves a monetary ceiling that accounts for the
   classic Warehouse's 45-minute idle auto-stop.

The query should return no fact values, only distinct group/field/type/zone/unit
combinations, series counts and dimension load-timestamp bounds. Until then,
the conclusion is structural: the tables exist, but complete family coverage
is not proven. Training, selection, model input and production remain false.

Predecessor handoff:
`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260806-ENTSOE-ARTIFACT-PACKAGE-INTEGRITY-PREFLIGHT.md`.
