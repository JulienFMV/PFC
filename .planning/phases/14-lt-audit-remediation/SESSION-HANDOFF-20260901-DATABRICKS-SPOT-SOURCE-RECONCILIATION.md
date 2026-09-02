# Session handoff - Databricks spot source reconciliation

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

An expanded value-blind Unity Catalog and lineage search reconciled the spot
source question without SQL or Warehouse start.

The direct Euler spot chain exists only in DEV:

- `dev.bronze.euler_spot`;
- `dev.silver.ge_market_euler_spot`;
- `dev.gold.dimspotproduct`;
- `dev.gold.factspotpricemonthly`.

The Silver Euler fact has 31,608 catalog-statistics rows, is about 2.25 MB,
partitions on `quotation_date`, and exposes interval start/end, frequency,
price and unit. No corresponding direct PRD Euler spot table is visible.

The production-capable alternative is already part of the governed ENTSO-E
source contract: `prd.silver.ge_power_entsoe_time_series_vintages` carries the
candidate `day_ahead_prices` family. The pipeline contract requests A44 prices
for CH, AT, DE-LU, FR and IT-North with `value_kind=price`, `unit=EUR_MWh` and
a two-day future horizon. Local PFC acceptance requires the same family and
EUR/MWh units.

Databricks lineage confirms that the managed DEV table
`dev.data_science.ds_ge_market_entsoe_spot_prices` is built from DEV ENTSO-E
Silver vintages. It exposes hourly CH, DE-LU, FR, AT and IT-North prices. This
proves the transformation route, not the PRD rows or their model admission.

The PRD Likron tables contain FMV orders, executions, VWAPs and cost-versus-
spot diagnostics. They are execution evidence, not an independent market
fixing authority.

## Corrected decision

Missing PRD Gold EPEX interval tables are not an absolute blocker for the
first PFC source export. The first export may source coupled day-ahead spot
truth from PRD ENTSO-E Silver vintages, provided it binds exact Gold dimension
SeriesKeys, the five required zones, EUR/MWh, availability timestamps,
classification sequence, native resolution and coverage.

Direct Euler/EPEX spot remains a desirable independent cross-check. Its DEV
chain cannot be presented as PRD evidence and its monthly Gold aggregate
cannot replace interval truth.

## Execution evidence

Across the extended 2026-09-01 preflight and source search:

- Databricks control-plane/Unity Catalog GETs: `137`;
- SQL statements / Warehouse starts / business rows / writes: `0/0/0/0`;
- expanded inventory: `1,222` objects across accessible DEV, PRD and staging
  schemas;
- exact lineage checks: ENTSO-E DEV spot table to Silver vintages; Euler Silver
  spot to Bronze and monthly Gold.

## Residual gate

The Warehouse remains stopped. A bounded PRD export is still needed to select
the actual `day_ahead_prices` SeriesKeys and prove PIT coverage. No data,
model, monthly-level or production authority was granted; T057 remains sealed.

Durable decision: D-20260901-258.
