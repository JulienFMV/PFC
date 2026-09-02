# Session handoff - ENTSO-E day-ahead self-service evidence

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

The current token and repository evidence answer most of the remaining
day-ahead spot questions without assistance from the data engineer and without
starting the SQL Warehouse.

## Answers established locally

### Series identity contract

The source pipeline defines:

- group `day_ahead_prices`;
- fields `ch_price`, `at_price`, `de_lu_price`, `fr_price` and
  `it_nord_price`;
- A44, `value_kind=price`, `unit=EUR_MWh`, two-day horizon;
- base key `day_ahead_prices||<field_name>`;
- optional suffix `||<classification_sequence>` when ENTSO-E supplies one.

Exact PRD classification suffixes are row values and were not visible through
Unity Catalog metadata.

### Rebuild evidence

PRD Silver vintages and Gold dimension were both newly created on 2026-08-24,
about 2 hours 22 minutes apart. The Silver table has 16,903,903 catalog-
statistics rows. This is strong evidence of a fresh rebuild rather than an
in-place coexistence of the pre-migration Gold table, but it is not an
independent job-run receipt.

The token sees one unrelated Databricks job and one unrelated Lakeflow
pipeline. It cannot inspect the ENTSO-E production run, so the exact full-
rebuild execution remains operational evidence to obtain from the platform.

### Coverage and availability

The DEV managed five-zone projection derived from ENTSO-E Silver vintages has
14,351 catalog-statistics rows from 2025-01-01 00:00 UTC through 2026-08-21
22:00 UTC. Catalog column statistics report zero nulls for all five price
columns.

The complete PRD Silver vintage table has non-null availability timestamps
from 2025-09-30 through 2026-08-27 and eight distinct resolutions across all
families. These global statistics do not prove family-specific PRD coverage or
the native resolution of each price series.

### Scan and cost proxy

Historical read-only DEV queries visible in SQL history provide bounded
proxies:

- CH-relevant dimension inventory: 9,433 bytes;
- typical latest-fact profile: 17,469,891 bytes;
- vintage family/CH profile: 656,297,497 bytes in 3.87 seconds.

The PRD Silver vintage table is 1,538,686,371 bytes and partitioned by year and
month. That table size is a conservative full-table storage upper bound, not a
guarantee of actual bytes scanned. Query history does not expose a billed DBU
or Azure VM amount for the proposed PRD export.

## What still requires external evidence or one authorized tiny query

1. The ENTSO-E job/run receipt proving the requested full rebuild and its
   validation verdict.
2. The exact current PRD SeriesKeys, including any classification suffixes,
   plus family-specific start/end, resolution and gap profile.
3. Actual file count and billed DBU/VM cost for the proposed PRD extraction.

Items 2 and the data-profile part of 3 can be answered by one small bounded
SQL batch once compute is already active or separately authorized. The
workstation did not start it.

## Execution evidence

Across the extended 2026-09-01 self-service investigation:

- control-plane, Unity Catalog, lineage and query-history GETs: `144`;
- SQL statements / Warehouse starts / business-row reads / writes: `0/0/0/0`;
- Warehouse remains stopped.

No data, model, monthly-level or production authority was granted. T057
remains sealed.

Durable decision: D-20260901-259.
