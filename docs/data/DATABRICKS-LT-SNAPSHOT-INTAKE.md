# LT Databricks snapshot intake

## Outcome

The LT workstation consumes immutable, manifested local exports. It does not
run model training or PFC generation against live Databricks tables.

This contract supersedes the Gold-only D293 draft. ENTSO-E Gold remains the
current serving layer, while ENTSO-E Silver vintages are the canonical
point-in-time layer. Silver is therefore intentional for this one role and is
not a fallback used to excuse an incomplete Gold table.

## Required source roles

| Local role | Databricks source | Layer | Required use |
|---|---|---|---|
| `eex_product_dimension` | `prd.gold.dimeexproduct` | Gold | product identity |
| `eex_delivery_period_dimension` | `prd.gold.dimeexdeliveryperiod` | Gold | delivery-period identity |
| `eex_price_daily` | `prd.gold.facteexpricedaily` | Gold | EEX forward constraints and vintages available in that fact |
| `spot_product_dimension` | `prd.gold.dimspotproduct` | Gold | spot product identity |
| `spot_price_interval` | `prd.gold.factspotpriceinterval` | Gold | realized interval truth |
| `entsoe_series_dimension` | `prd.gold.dimentsoeseries` | Gold | semantic series dictionary |
| `entsoe_latest` | `prd.gold.factentsoetimeserieslatest` | Gold | current values |
| `entsoe_series_resources_current` | `prd.gold.bridgeentsoeseriesresources` | Gold | current equipment enrichment; optional for the core PFC |
| `entsoe_vintages` | `prd.silver.ge_power_entsoe_time_series_vintages` | Silver | PIT, revisions, availability and vintage resource lineage |

Weather and Swissgrid exports are separate optional domains. Their presence
does not authorize them as model inputs; each keeps its own freshness,
availability and feature-admission checks.

The legacy `prd.gold.factentsoetimeseriesvintages` is deliberately excluded.
It duplicates Silver history and is no longer maintained by the audited
ENTSO-E Gold pipeline.

## ENTSO-E semantics that must survive export

The Gold dimension must retain `SeriesID`, `SeriesKey`, family, document,
business/process/PSR types, directions, zones, unit and source identifiers.
Gold Latest must retain the native interval, resolution, value, publication
and availability fields.

Silver vintages must retain at least:

- `SK_ge_power_entsoe_time_series_vintages` and `series_key`;
- `IntervalStartUtc`, `Date_Time_UTC`, `IntervalEndUtc`, `resolution` and
  `field_value`;
- publication, first/last-seen pull and ingest timestamps;
- `availability_basis`, `availability_known` and
  `availability_timestamp_utc`;
- source document identity/revision, snapshot/point lineage and DQ flags;
- `resource_details` at the vintage grain.

`UNKNOWN_BACKFILL` is valid historical content but cannot be presented as a
known-at vintage. A backtest must filter or quarantine rows whose availability
is unknown at the simulated origin.

## LSEG benchmark intake

The selected Swiss curve is the `continuous_forward/CHE` HPFC identified in
DEV as curve/measure `110181967`. Export only this selected curve and the
required horizon, not the complete multi-gigabyte vendor vintage fact.

The benchmark package contains:

- the selected row from `DimLsegCurves`;
- bounded rows from `FactLsegCurveValuesLatest`;
- bounded rows from `FactLsegCurveValueVintages` when a vintage comparison is
  required.

`VendorLastUpdatedAtUtc` is vendor metadata and remains diagnostic.
`PipelineFirstSeenAtUtc` records FMV observation. `KnownAtTimestampUtc` is a
deprecated compatibility alias of pipeline first-seen and must not be
misrepresented as a vendor-known-at timestamp. Rows marked with unknown
availability are not PIT evidence.

LSEG cannot set Swiss monthly levels, train the production curve or select a
model until a distinct benchmark-admission decision exists.

## Local package and incrementality

Each export is written once under an immutable generation in `FMV_DATA_ROOT`
and contains Parquet payloads plus one manifest. The manifest binds:

- catalog, schema and full source table for every artifact;
- extraction predicate and lower/upper watermark;
- row count, byte size, Parquet schema hash and SHA-256;
- source publication/vendor timestamps and FMV observation timestamps;
- code revision, export time, cost receipt and previous generation ID;
- explicit authorities, all false until downstream validation succeeds.

Incremental pulls use a durable high-water mark with overlap and deduplication.
A retry is idempotent. A missing watermark fails closed instead of silently
reading full history. Full backfills are explicit, chunked and separately
costed.

The model reads only the immutable generation selected by
`views/pfc_lt/current.json`. It never mutates an admitted snapshot and never
uses directory discovery as promotion authority.

## Admission sequence

1. Verify inventory, hashes, sizes, schemas and source-layer identities.
2. Validate units, grains, native intervals, DST, directions and series joins.
3. Validate availability semantics and construct point-in-time views.
4. Reconcile Gold Latest with Silver latest winners and current bridge rows.
5. Profile coverage, gaps and revisions by family, zone and direction.
6. Freeze a local manifest-backed model-input snapshot.
7. Run rolling-origin qualification and the independent future holdout.

No step grants monthly-level or production authority implicitly.
