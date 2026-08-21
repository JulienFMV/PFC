# Databricks LT local materialization

## Status

This is an offline, deterministic development boundary. It opens only local
DataFrames already exported from Databricks. It contains no connector, SQL,
Warehouse start, remote write, snapshot signing or production authority.

Implementation:
`pfc_shaping/data/databricks_lt_materialization.py`.

Exact source-to-frame replay and the self-contained unsigned pre-publication
package are implemented in `pfc_shaping/data/databricks_lt_replay.py`.
The signed snapshot binding and isolated-publisher admission are implemented
in `pfc_shaping/data/databricks_lt_snapshot.py` as
`lt_input_snapshot.v4`.

## Source-to-model paths

| Source export | Local materialization | Model-facing frame |
|---|---|---|
| Gold spot interval fact | exact market/product filter, known-and-delivered cutoff, atomic interval expansion | `epex_<market>` raw `price_eur_mwh`, then governed `clean_epex` features |
| Gold ENTSO-E dimension + Latest | explicit SeriesKey mapping and known current-state selection | current-serving `entso` raw and derived features |
| Gold ENTSO-E dimension + Silver vintages | availability-known, `availability_timestamp_utc <= origin`, DQ exclusion and latest eligible revision | point-in-time `entso` raw and derived features |
| Gold EEX daily fact/dimensions | `materialize_eex_forward_history`: exact joined projection, `FactLoadTimestampUtc <= as_of_utc`, Swiss quotation-date cutoff, then the existing EEX normalizer | causal `eex_forwards_history` candidate content |

The materialization result always declares all scientific and production
authorities false. Layer acceptance, immutable publication, signing and model
admission are separate steps.

## ENTSO-E mapping contract

No `SeriesID`, `SeriesKey`, `GroupName` or `FieldName` is guessed. The caller
must provide an exact `fmv_entsoe_feature_mapping.v1` contract containing:

- `market = CH`;
- the semantic SHA-256 of the consumed Gold dimension projection;
- one or more weighted `SeriesKey` values for `load_mw`, `solar_mw` and
  `wind_mw`;
- optional signed physical-flow series for `cross_border_mw`.

The mapping validator enforces:

- `actual_load` A65/A16 and `generation_actual` A75/A16 families;
- production `BusinessType=A01` for solar and wind, excluding consumption
  series;
- ENTSO-E PSR types B16 for solar and B18/B19 for wind;
- `GenerationDirection=GENERATION` for solar and wind;
- MW units and Swiss zone participation;
- unique use of each SeriesKey and finite, non-zero weights.

Bidirectional flows use one enforced convention: `NET_EXPORT_FROM_CH`.
CH-to-neighbor series must use `+1` and neighbor-to-CH series must use `-1`.
The validator derives the expected sign from `FromZone`/`ToZone` and rejects a
mapping with the opposite weight; it never infers direction from row order.

The mapping object itself carries the complete Gold-dimension semantic hash.
Calling the materializer directly cannot bypass the contract-to-dimension
binding.

## Point-in-time policy

Silver vintages are eligible only when:

- `availability_known = true`;
- `availability_timestamp_utc <= as_of_utc`;
- `dq_failed = false`;
- availability basis and publication/first-seen timestamps are coherent.

`UNKNOWN_BACKFILL` rows remain valid stored history but cannot fill a PIT
feature grid. Later revisions are excluded at an earlier origin. Ambiguous
ties at the same availability/revision/last-seen ordering fail closed.

The Gold Latest path is current-serving only. It also requires known
availability, but it does not become historical PIT evidence merely because
the same transformation succeeds.

## Time and interval policy

- Input interval start/end/right edge and ISO-8601 resolution must agree.
- Atomic 15-, 30- and 60-minute values may be expanded to a 15-minute UTC
  transport grid inside their declared intervals.
- Overlaps, gaps, non-finite values and inconsistent component coverage fail.
- Europe/Zurich DST fall-back keeps two distinct UTC hours; no local timestamp
  is deduplicated.
- Fractional cadence declarations and sub-minute interval drift are rejected;
  durations are compared exactly rather than truncated to integer minutes.
- The production climatology now fails when any Swiss-local month/hour/quarter
  slot lacks admitted history. It no longer silently inserts neutral ENTSO-E
  feature values.

## Exact local replay package

The replay adapter builds and verifies an in-memory, self-contained package
containing:

- the exact bounded source Parquet bytes and their full table identities;
- canonical raw and derived model frames;
- the exact installed materializer source;
- a canonical replay config and materialization audit;
- a content-derived build ID and hash/size inventory.

Verification replays from the source bytes and rejects byte mutation, code or
runtime drift, audit/config divergence, an unexpected source inventory and a
mixed DEV/PRD package. The package declares zero Databricks connections,
statements, Warehouse starts, network calls and remote writes. It is unsigned,
authority-negative and cannot be published without the independent v4
acquisition, time, journal and publication authorities.

The EEX materialization is deliberately one step short of this generic replay
package. Its joined Gold projection is now deterministic and causal, but the
three physical source tables and their join SQL still have to be bound by the
future export manifest before v4 publication. This avoids treating a joined
Parquet file as proof of dimension provenance.

## Governed snapshot v4

`lt_input_snapshot.v4` keeps the API-specific provider replay and the
Databricks replay as explicit, different evidence families. Each replayed role
declares exactly one `upstream_replay.kind`:

- `PROVIDER_API` retains the v3 raw-envelope proof, currently used for hydro
  and any role not yet exported from Databricks;
- `DATABRICKS_EXPORT` binds the exact replay manifest, every package member, a
  canonical export manifest and the downstream raw-to-feature derivation.

For a calibration-eligible Databricks role, the validator requires PRD tables,
an exact read-only query and selected-column inventory, a PIT upper watermark,
actual Parquet row/hash/size reconciliation, explicit cost counters, ordered
export/receipt/replay timestamps and quality report `lt_source_quality.v3`.
The acquisition receipt is bound to the replay manifest bytes, and the signed
source-journal root covers the complete role declaration.

The initial v4 deliberately admits only `FULL_SNAPSHOT` with no predecessor.
It rejects a claimed incremental generation until predecessor + delta merge,
overlap, deduplication and cumulative-output replay are implemented as one
deterministic proof. This prevents a cheap delta from being mistaken for a
complete model history.

## Remaining integration work

1. Generate the real SeriesKey mapping contract from the admitted Gold
   dimension inventory.
2. Implement deterministic incremental composition before enabling a
   predecessor-bearing mode; v4 currently accepts full snapshots only.
3. Bind the EEX joined projection to the three Gold tables and exact join SQL,
   then convert its causal normalizer output into the signed historical vintage
   catalog required by the monthly solver.
4. Keep hydro on its separately governed SFOE/FMV source until an exact Gold
   table and transformation are approved.
5. Treat weather, Swissgrid and LSEG as separately admitted candidates or
   benchmarks; they are not core authority merely because they exist in Gold.
6. Run the v4 chain on a user-authorized, bounded PRD export and retain the
   actual Databricks query-history/cost observation.

Until those steps and real-data acceptance pass, model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.
