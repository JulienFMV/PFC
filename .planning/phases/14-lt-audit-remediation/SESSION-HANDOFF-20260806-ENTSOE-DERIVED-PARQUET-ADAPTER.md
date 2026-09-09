# Session handoff - ENTSO-E derived Parquet adapter

Date: 2026-08-06  
Decision: D-20260806-284  
Status: synthetic bounded-Parquet adapter PASS; real and model authority remain false

## Outcome

D284 adds a read-only adapter from one normalized synthetic Parquet package to
the unchanged D283 canonical-decimal executor. It is not a raw ENTSO-E reader
and it does not touch Databricks.

The adapter:

- requires a strict manifest and one fixed sibling file named
  `derived_operand_values.parquet`;
- reads both as stable single-link files and re-reads them after execution;
- checks declared SHA-256, byte size, row count and row-group count;
- checks footer allocation budgets before decoding: at most 64 MiB, 200,000
  rows, exactly nine columns, 1.8 million cells and 1,024 row groups;
- permits only flat `BYTE_ARRAY` and `INT64` physical columns with ZSTD;
- requires an exact ordered Arrow schema, including `timestamp[us, tz=UTC]`
  and a nullable string `value_decimal_mw`; binary floats and coercion fail;
- binds each row at grain `role + feature_record_id + target interval` to the
  exact D282 assessment;
- splits only explicit `REALIZED_ACTUAL` and `OPERATIONAL_FORECAST` rows into
  D283 artifacts, then delegates all decimal, lineage and physical checks;
- rejects duplicates, missing/orphan/substituted rows, unknown roles, future
  manifests, D282 detachment, hardlinks and mid-execution mutation.

The D284 assessment and proof persist hashes and counts only. The temporary
synthetic Parquet and all operand/output values are removed before publication.

## Files changed

- `.planning/phases/14-lt-audit-remediation/ENTSOE-LT-DERIVED-PARQUET-ADAPTER-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_lt_derived_parquet_adapter.py`
- `tests/test_entsoe_lt_derived_parquet_adapter.py`
- `build/d284_materialize.py` (local-only materializer)
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `docs/research/forwards_sources.md`
- `.planning/HANDOFF.md`
- this handoff

Generated local-only proof:

- `build/databricks-eex-daily/2026-08-06/entsoe-lt-derived-parquet-adapter-proofs/860b652a495089234fbc546fb5743a63f0d6bcc2b8cba45f6a7dee2c974921fd/manifest.json`

## Exact identities

- contract raw SHA-256:
  `58133559b8079b35e244fa31bf7e32dc85d35acb7c16698103715f60feb8a3ae`
- contract canonical content ID:
  `1ebcd75defc4e7641927117a56f19c718fa4278ef9fae2dfc16c735ee8b0635c`
- Arrow schema content ID:
  `ccda708fbdbd7187ca2e8f860f172ce85101d9424ba63840844f79414e9e6d90`
- validator SHA-256:
  `0ea365f84d60ea6874f75b04c2194c971c659cd47438aa22f5237e8b55e612bf`
- tests SHA-256:
  `41e4e34a1fdb0f69f4a80316f61bb98a339e2b14cbb1dc980c63e13b03fc5acb`
- materializer SHA-256:
  `11b7d9224c17f370bb899db128d4c2eeda4739437ca2050a36c5d9900557674a`
- proof ID:
  `860b652a495089234fbc546fb5743a63f0d6bcc2b8cba45f6a7dee2c974921fd`
- proof raw SHA-256:
  `2343063f72b14a9af3675cf58a215c413fa75ca1f06052e1cc5e95dda5dbf25b`
- assessment content ID:
  `59800e3cbb00f2d0938d1e64dde6278a898e7d387e031efa7145279a3130e0b5`
- D283 output artifact content ID:
  `4b65a9ca95180dea3d1bcf3b6dbf5c7b8008a057a7d2847d7f14b13db079861d`

## Verification

- focused D284 mutation roast: `25 passed`;
- D280-D284 predecessor chain: `113 passed`;
- adjacent D244/D245/D243/D253/D254/D255/D261/D270/D272/D273/D280/D281/
  D282/D283/D284 matrix: `342 passed`;
- all current `tests/test_entsoe_*.py`: `529 passed`;
- Ruff check and format check: pass;
- independent materializer executions produced the same proof ID twice;
- proof canonical hash equals its content-addressed directory name;
- all proof authorities are false and all execution counters are zero;
- clear synthetic operands, output values and `value_decimal_mw` are absent
  from the proof.

Commands used with repo-local `TEMP`, `TMP` and pytest basetemps:

- `python -m ruff format ...`
- `python -m ruff check ...`
- `python -m ruff format --check ...`
- `python -m pytest -q tests/test_entsoe_lt_derived_parquet_adapter.py ...`
- `python -m pytest -q` over D280-D284
- `python -m pytest -q` over the adjacent D244-D284 matrix
- `python -m pytest -q` over every current `tests/test_entsoe_*.py`
- `python -m build.d284_materialize` twice

Non-product failures during the roast:

- the first Ruff format check correctly reported the new test file required
  formatting; it was formatted, then check and format-check passed;
- the first predecessor command named a nonexistent guessed D280 test file and
  collected zero tests. It was corrected to
  `tests/test_entsoe_lt_model_input_admission.py`; the real chain passed 113.

## Execution and authority boundary

- Databricks connections/statements/writes: `0/0/0`.
- Warehouse starts: `0`.
- Network calls: `0`.
- `H:` accesses: `0`.
- Real values opened/persisted: `0/0`.
- Synthetic Parquet or operand/output values persisted in proof: `0/0`.
- No CT module was imported or changed.

Real source, real arithmetic, PIT, predictive value, model input, model
selection, candidate, promotion and production authorities remain false. The
monthly solver remains the sole CH monthly-level authority. Any later admitted
forecast-error contribution must be zero-mean within the solver month.

## Remaining blockers

- No governed real data-engineer package has passed D244-D284.
- The delivered Parquet schema in D284 is a normalized target interface, not
  proof that current `dev.gold` contains every required family or vintage.
- Exact family, zone/EIC history, effective cadence, unit/sign, revision,
  provenance and PIT evidence remain required from the data engineer.
- Predictive usefulness remains unknown until a new independently frozen
  rolling-origin holdout exists. T057 remains sealed.

## Next safe batch

Do not add synthetic feature authority. The next empirical batch starts only
when the immutable data-engineer delivery exists locally: admit it through
D244-D284, first in metadata/value-free mode where possible, and stop at the
first failed gate. Do not start or query a Databricks Warehouse merely to
manufacture that delivery.

OMPEX remains a post-freeze benchmark only and `H:` must not be accessed from
this workstation workflow.
