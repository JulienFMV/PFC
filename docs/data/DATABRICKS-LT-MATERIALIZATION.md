# Databricks LT local materialization

## Status

This is an offline, deterministic development boundary. It opens only local
DataFrames already exported from Databricks. It contains no connector, SQL,
Warehouse start, remote write, snapshot signing or production authority.

Implementation:
`pfc_shaping/data/databricks_lt_materialization.py`.

## Source-to-model paths

| Source export | Local materialization | Model-facing frame |
|---|---|---|
| Gold spot interval fact | exact market/product filter, known-and-delivered cutoff, atomic interval expansion | `epex_<market>` raw `price_eur_mwh`, then governed `clean_epex` features |
| Gold ENTSO-E dimension + Latest | explicit SeriesKey mapping and known current-state selection | current-serving `entso` raw and derived features |
| Gold ENTSO-E dimension + Silver vintages | availability-known, `availability_timestamp_utc <= origin`, DQ exclusion and latest eligible revision | point-in-time `entso` raw and derived features |
| Gold EEX daily fact/dimensions | existing `databricks_eex_daily_snapshot` normalizer | `eex_forwards_history` candidate content |

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
- MW units and Swiss zone participation;
- unique use of each SeriesKey and finite, non-zero weights.

Bidirectional flows therefore require an explicit sign convention. For a net
Swiss export feature, for example, CH-to-neighbor series use `+1` and
neighbor-to-CH series use `-1`. The chosen convention is visible in the audit;
it is never inferred from row order.

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
- The production climatology now fails when any Swiss-local month/hour/quarter
  slot lacks admitted history. It no longer silently inserts neutral ENTSO-E
  feature values.

## Remaining integration work

1. Generate the real SeriesKey mapping contract from the admitted Gold
   dimension inventory.
2. Embed this materializer in a governed Databricks-export publisher with raw
   artifact, code/config, query, watermark and source-manifest bindings.
3. Extend or replace the API-specific `lt_input_snapshot.v3` provider envelope
   so Databricks exports can satisfy exact replay without pretending to be API
   responses.
4. Convert the existing EEX normalizer output into the signed historical
   vintage catalog required by the monthly solver.
5. Keep hydro on its separately governed SFOE/FMV source until an exact Gold
   table and transformation are approved.
6. Treat weather, Swissgrid and LSEG as separately admitted candidates or
   benchmarks; they are not core authority merely because they exist in Gold.

Until those steps and real-data acceptance pass, model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.
