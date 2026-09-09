# Session handoff — Databricks LSEG LT benchmark inventory

Date: 2026-08-07  
Decision: D-20260807-295  
Status: structural candidate found; CH/HPFC identity and content pending

## Outcome

Unity Catalog metadata establishes a strong LSEG LT-curve benchmark candidate
in `dev`: a 17-row curve dimension, a latest fact and a versioned vintage fact.
The facts extend to 2028-12-31 and preserve forecast, pull and known-at times.
No row was read because the configured Classic `2X-Small` Warehouse was
stopped and the user forbids avoidable Databricks cost.

The candidate is not yet admitted as a Swiss PFC/HPFC. At the next already
scheduled Warehouse run, read only the 17-row dimension first. Confirm CH,
power price, EUR/MWh, native frequency, timezone and provider semantics before
touching the fact tables.

## Material limits

- delivery horizon ends 2028-12-31, short of a complete rolling N+3 from
  August 2026;
- `KnownAtTimestampUtc` begins 2026-06-15 although forecast dates begin in
  2022, so historical PIT authority is not proved;
- Gold vintages contain 164,269,954 rows / about 11.94 GB and have no partition
  or effective clustering;
- Bronze points are partitioned by ingestion year/month and retain source
  lineage, but their minimum source timestamp is the suspicious sentinel
  2000-01-01;
- LSEG local-export/licence rights remain to be confirmed.

## Files

- `docs/research/DATABRICKS-LSEG-LT-BENCHMARK-INVENTORY-20260807.md`
- `build/databricks-lseg/2026-08-07/lseg-selected-table-schemas.json`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Execution

The batch used 20 Unity Catalog/warehouse control-plane GETs, zero SQL,
zero business-row reads, zero Warehouse starts and zero Databricks writes.
The local schema capture SHA-256 is
`55383924184e2ca3bb3f94bf29151f67589fb5f7709acd0c23d55a3ad1271701`.

No CT, Power BI, AFRY, OMPEX, T057, `H:` or heavy desk-data file was opened or
changed.
