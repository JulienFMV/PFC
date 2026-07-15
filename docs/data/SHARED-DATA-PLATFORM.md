# FMV Shared Data Platform Contract

## Purpose

`FMV_DATA_ROOT` identifies one consumer-neutral data platform shared by PFC
LT, PFC CT, trader dashboards, sales tooling and other internal analytics. A
model repository consumes governed views of this platform; it does not own the
source data and must not silently copy it into its repository.

The current workstation root remains:

```text
C:\Users\jbattaglia\pfc_local_data
```

The path can later be mounted in Docker or replaced by an IT-managed volume
without changing model code:

```text
FMV_DATA_ROOT=/data/fmv
```

## Logical Layout

```text
FMV_DATA_ROOT/
  catalog/                         # datasets, schemas, owners, vintages
  entsoe/
    raw_xml/
    bronze_timeseries/
    bronze_forecasts/
    bronze_exchanges/
    curated/
    imports/<immutable_import_id>/
    quarantine/
  market/
    eex/
    epex/
    commodities/
  hydro/
  scenarios/
  ep2050/
  snapshots/<immutable_generation_id>/
  views/
    pfc_lt/current.json
    pfc_ct/current.json
    trader_dashboard/current.json
    sales/current.json
```

Top-level directories represent data domains, not applications. Processing
layers remain inside each domain. `views/<consumer>` contains only contracts or
pointers selecting exact immutable vintages; it must not duplicate payloads.

Only `views/pfc_lt` is implemented today. The other view names reserve the
contract boundary; they must be created only when their own freshness,
entitlement and quality rules are defined.

## Location Contract

New deployments define only:

```text
FMV_DATA_ROOT=C:\Users\jbattaglia\pfc_local_data
```

The following names are deprecated compatibility aliases:

- `PFC_SHARED_DATA_ROOT`
- `PFC_LT_DATA_ROOT`
- `PFC_ENTSOE_DATA_ROOT`

When a canonical variable and an alias are both set, they must resolve to the
same root or execution fails. A direct domain alias must identify the expected
child, for example `<FMV_DATA_ROOT>/entsoe`.

## Governance Rules

- Raw/import payloads are append-only and receive an immutable ID and receipt.
- `curated` data is not automatically model-eligible.
- A consumer reads only the generation selected by its own view contract.
- Cross-domain snapshots bind paths, timestamps, byte hashes and acquisition
  authority. Discovery of a file never activates it.
- LT and CT may share source bytes but keep independent view and promotion
  contracts.
- EEX workbooks and ENTSO-E imports require source-specific provenance and
  point-in-time controls before calibration use.
- OMPEX HFC is benchmark-only: no OMPEX value, residual or derived feature may
  enter training, calibration, priors, constraints or model selection.
- Repository-local heavy Parquets and DuckDB files are not canonical storage.
- Workstation ACLs are not enterprise immutability. IT production requires
  managed identity, least-privilege read/write roles, retention, backup,
  monitoring and signed promotion evidence.

## Migration

`scripts/materialize_shared_data_views.py` copies the existing LT pointer bytes
to `views/pfc_lt/current.json`. It does not move or delete data. It refuses to
overwrite a different existing view. Once present, the consumer view is the
only write target and the root-level `current.json` is a frozen, read-only
fallback for roots that have not yet been migrated.

After every deployed consumer reads `views/pfc_lt`, removal of the legacy
fallback is a separate, explicit IT migration. New publishers must never
dual-write both pointers because two filesystem replacements are not one
transaction.

## Storage Efficiency Roadmap

The current `lt_input_snapshot.v1` bootstrap copies the exact consumed bytes
into each cross-domain generation. This is intentionally conservative: a
snapshot remains immutable even when a domain-level curated file changes. It
is not the long-term archive design and must not be used to multiply arbitrary
research copies.

The IT target is a content-addressed object layer with one immutable object per
SHA-256 and lightweight consumer manifests referencing those objects. That
requires a versioned snapshot schema, object-level retention and independently
enforced write protection. Plain hardlinks are not an acceptable shortcut:
mutating one hardlink mutates the supposedly immutable object for every
consumer.

Until that schema is implemented and replay-audited, snapshot byte copies are
an explicit storage cost of fail-closed provenance. Import IDs and generation
IDs must not be recreated merely to obtain a different consumer-facing name.
