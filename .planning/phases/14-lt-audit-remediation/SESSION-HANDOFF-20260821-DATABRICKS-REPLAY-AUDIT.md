# Session handoff - Databricks materializer roast and exact replay

Date: 2026-08-21

Starting commit:
`f7d23e0548`

Branch:
`fix/lt-audit-remediation`

## Outcome

The offline Databricks materializer was reviewed adversarially before further
integration. Material defects were corrected, and a self-contained exact-byte
Databricks replay package now bridges bounded local exports to the existing LT
raw/derived replay layer. It is unsigned, authority-negative and not accepted
by the isolated snapshot publisher yet.

No Databricks connector, SQL statement or Warehouse was used. GitHub was read
only to compare the local contract with ENTSO-E DEV commit
`dadbad2c793ee133bd0ccd8d053a5e591b44cb7f`.

## Audit findings corrected

- Direct construction of `EntsoeFeatureMapping` could bypass its dimension
  contract hash. The mapping object now carries and verifies that hash.
- `GenerationDirection` was absent from the signature. Solar and wind now
  require both A01 production and `GENERATION`.
- Physical-flow weights could be reversed. `cross_border_mw` now enforces
  `NET_EXPORT_FROM_CH` from exact `FromZone`/`ToZone` values.
- String booleans could be coerced incorrectly. Availability/DQ flags now
  require real non-null boolean values.
- A valid unknown historical backfill with no publication timestamp was
  rejected. It is now accepted as stored history but still excluded from PIT.
- Fractional cadence or sub-minute interval drift could pass integer
  truncation. Duration comparison is now exact.
- Gold availability was omitted from its source semantic hash, and input row
  ordering could destabilize hashes. Gold, Silver, dimension and spot source
  projections are now complete and canonically ordered.
- Per-row `date_range` expansion was replaced with vectorized quarter-hour
  expansion.
- Gold/Silver layer acceptance now requires and reconciles
  `GenerationDirection` for actual generation.
- The two new data modules are in the governed LT wheel positive inventory.

## Exact replay package

`pfc_shaping/data/databricks_lt_replay.py` implements:

- `fmv_databricks_lt_replay_config.v1`;
- `fmv_databricks_lt_replay_build.v1`;
- Gold spot, Gold ENTSO-E current and Silver ENTSO-E PIT source modes;
- exact DEV/PRD table identities with a no-mixing rule;
- source-byte, materializer-code, runtime, selection, audit and raw/derived
  frame bindings;
- a self-contained in-memory artifact package with content-derived build ID;
- complete replay verification and tamper rejection.

The package declares Databricks connections/statements/Warehouse starts,
network calls and remote writes as `0/0/0/0/0`. All scientific, model,
calibration, publication and production authorities remain false.

## Changed files

- `pfc_shaping/data/databricks_lt_materialization.py`
- `pfc_shaping/data/databricks_lt_replay.py`
- `pfc_shaping/validation/databricks_pfc_layer_acceptance.py`
- `pfc_shaping/package_contract.py`
- `tests/test_databricks_lt_materialization.py`
- `tests/test_databricks_pfc_layer_acceptance.py`
- `tests/test_lt_package_contract.py`
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
- `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`
- `docs/data/README.md`
- `README.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Verification

All mutable test/cache/temp paths were below `build/`.

- original targeted audit matrix: `53 passed`;
- first corrected matrix: `61 passed`;
- replay/package plus wheel matrix: `49 passed`;
- expanded materializer/layer/replay/quality/package matrix: `95 passed`;
- final bounded CI data/PIT/package matrix:
  `178 passed, 2 skipped in 14.40s`;
- targeted Ruff: pass;
- `git diff --check`: pass before the final handoff update;
- Databricks connections/statements/Warehouse starts: `0/0/0`;
- remote writes: `0`.

The recurring pytest warning is pre-existing: the active pytest version does
not recognize the configured `cache_dir` option. Test collection and results
are unaffected.

## Residual work

1. Receive a bounded local Gold dimension export and create the real mapping
   contract; no SeriesKey is guessed in Git.
2. Add a Databricks-aware governed snapshot schema that binds the replay
   package to export query/predicate, watermarks, predecessor generation,
   source receipt and incremental-cost evidence.
3. Admit that schema in the isolated publisher. Do not relabel the package as
   the API-response-specific `lt_input_snapshot.v3`.
4. Convert the EEX Gold normalizer output into the signed forward vintage
   catalog required by the monthly solver.
5. Keep hydro separate until its exact governed source is approved. Weather,
   Swissgrid and LSEG remain separate candidates/benchmarks.
6. Run real-data acceptance only from a user-authorized bounded export. That
   future extraction is the only step that may incur Databricks cost.

Durable decision: D-20260821-253.

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; T057 remains sealed.
