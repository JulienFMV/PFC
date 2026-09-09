# Session handoff - Databricks governed snapshot v4

Date: 2026-08-21

## Outcome

The offline Databricks replay from D-253 is now connected to the existing
authenticated acquisition, source-time journal, quality and isolated snapshot
publisher chain through `lt_input_snapshot.v4`.

No Databricks connector, SQL statement, Warehouse start, network call or remote
write was used. All validation used synthetic repo-local Parquet payloads below
`build/`.

## Architecture

- v3 remains the exact provider-API raw-envelope schema.
- v4 is hybrid: every replayed role declares `PROVIDER_API` or
  `DATABRICKS_EXPORT` explicitly.
- Databricks calibration evidence requires PRD. DEV packages remain unsigned
  engineering evidence only.
- The source receipt binds the exact Databricks replay manifest. The signed
  journal root then binds the complete role declaration.
- The export manifest binds one bounded read-only query per source payload,
  selected columns, PIT watermark, rows, bytes, hashes, export time and cost
  counters.
- `lt_source_quality.v3` binds replay/export manifests, code/config/audit and
  raw/derived model frames.
- The publisher positive inventory now includes the three offline Databricks
  modules and no signer or connector.
- v4 currently admits only `FULL_SNAPSHOT`. Incremental composition remains
  fail-closed until predecessor + delta merge can be replayed exactly.

## Changed files

- `pfc_shaping/data/acquisition_contract.py`
- `pfc_shaping/data/databricks_lt_snapshot.py` (new)
- `pfc_shaping/data/lt_input_sources.py`
- `pfc_shaping/data/snapshot_publisher.py`
- `pfc_shaping/package_contract.py`
- `pfc_shaping/publisher_package_contract.py`
- `.dockerignore`
- `deploy/publisher/runtime-contract.json`
- `tests/test_databricks_lt_snapshot.py` (new)
- `tests/test_governed_lt_input_snapshot_v2.py`
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
- `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`
- `docs/data/README.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Audit corrections made during implementation

1. Removed a circular identity design in which cost evidence repeated the
   parent export ID while also participating in that ID.
2. Added actual Parquet row-count and selected-column reconciliation rather
   than trusting signed counters alone.
3. Added explicit `exported_at <= received_at <= replayed_at` ordering.
4. Bound the source-receipt locator to the canonical sorted PRD table set and
   added an explicit test rejecting a v3 bundle merely relabelled as v4.
5. Preserved the existing v3 error-contract wording for downstream tests.
6. Added the new runtime files to the Docker positive inventory.
7. Updated the publisher runtime contract to the current unchanged `uv.lock`
   SHA-256 `ad9077398246618202b7bb3ae0c6621d4d2b2b5b6186fd0f827a4ac77b0b30a6`;
   the prior contract hash was already stale before this change.

## Verification

All mutable test/cache/temp paths were below the canonical workspace.

- D-253 counter-audit: `59 passed`.
- Direct v4 + materializer matrix: `31 passed`.
- Signed hybrid v4 end-to-end path: `6 passed`.
- Broad replay/input/publisher/package matrix: `209 passed, 15 skipped, 3
  failed` on first run; all three findings were corrected.
- Corrected-finding/runtime-closure rerun: `11 passed`.
- Final replay/input/publisher/package matrix after the independent audit
  corrections: `213 passed, 15 skipped in 134.25s`.
- LT solver, LT/CT import separation and candidate-evidence matrix: `113
  passed, 1 skipped in 204.33s`.
- Targeted Ruff: pass after formatting.
- Databricks connections/statements/Warehouse starts: `0/0/0`.
- Network calls/remote writes: `0/0`.

## Residual work

1. Produce the real SeriesKey mapping from a bounded Gold dimension export.
2. Run a user-authorized, bounded PRD v4 export and retain real query-history
   and cost evidence. This is the only future step here that may incur
   Databricks cost.
3. Implement deterministic incremental predecessor + delta composition before
   enabling incremental publication.
4. Convert Gold EEX forwards into the signed vintage catalog required by the
   monthly solver.
5. Run rolling-origin model qualification and a new independently frozen
   future holdout. T057 remains sealed.

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.

Durable decision: D-20260821-254.
