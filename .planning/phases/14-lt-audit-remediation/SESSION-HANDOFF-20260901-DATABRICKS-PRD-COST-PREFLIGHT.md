# Session handoff - Databricks PRD cost preflight

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

A value-blind PRD cost preflight was completed through Databricks control-plane
and Unity Catalog metadata only. The SQL Warehouse was not started and no SQL,
business row or write was issued.

Verdict:
`STOP_NO_ACTIVE_WAREHOUSE_AND_INCOMPLETE_PRD_SURFACE`.

The local metadata receipt is:

`build/databricks-cost-preflight/2026-09-01/prd-metadata-preflight.json`.

## Evidence

- Warehouse: stopped Classic `2X-Small`, one fixed cluster, serverless false,
  45-minute auto-stop.
- Control-plane metadata GETs: `52`.
- SQL statements / Warehouse starts / business rows / writes: `0/0/0/0`.
- Accessible source tables: `17`; expected Gold spot tables not found or not
  visible under their contracted FQNs: `2`.
- Catalog statistics across inspected Gold/Silver tables: `925,686,040` rows
  and `38,583,179,923` bytes. These are storage metadata, not scan or billing
  measurements.
- Unity Catalog did not expose file counts or query-plan scan estimates.
- A complete paginated listing exposed `212` tables across `prd.gold` and
  `prd.silver`, including `31` whose names match the audited EEX, ENTSO-E,
  weather or Swissgrid domains. No spot-named table was visible.

## Cost and data-shape findings

- EEX Gold facts and dimensions are about 49.7 MB in total. They are
  unpartitioned but small enough for a separately authorized bounded scan.
- ENTSO-E Gold plus Silver history is about 1.79 GB. The 1.54 GB Silver vintage
  fact is partitioned by `_year`, `_month`, enabling a bounded first export.
- Gold weather is about 103 MB. Its forecast tables still expose target time,
  minimum lead time and load time but no explicit governed issue timestamp or
  vintage key; they remain unsafe for PIT backtests.
- Silver weather is about 35.87 GB. Open-Meteo alone is about 33.41 GB and no
  partition column is declared. It is excluded from the first export.
- Swissgrid Gold/Silver is kept as a separate optional candidate and
  reconciliation batch, not part of the first core extraction.
- `prd.gold.dimspotproduct` and `prd.gold.factspotpriceinterval` both returned
  HTTP 404, and no spot-named table appeared in the complete visible PRD
  Gold/Silver inventory. They may be absent or hidden from the current
  identity; in either case the serving surface is not proven for spot shape
  truth.

## Next admissible batch

The first real export should contain only:

1. EEX Gold fact plus its two dimensions;
2. ENTSO-E Gold dimension/latest plus Silver vintages pruned by `_year` and
   `_month`.

Before SQL, the platform owner must provide file counts and a scan upper bound
or an equivalent export quote, identify the missing spot Gold surface, and
confirm either an already-running separately authorized Warehouse or a
platform-owned export. The batch also needs a statement timeout, a maximum
statement count and an approved DBU plus Azure VM cost ceiling.

## Residual governance

No data, PIT, model, selection or production authority was granted. Model
admission remains `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; T057
remains sealed. The CH monthly BASE solver remains the sole level authority.

Durable decision: D-20260901-257.
