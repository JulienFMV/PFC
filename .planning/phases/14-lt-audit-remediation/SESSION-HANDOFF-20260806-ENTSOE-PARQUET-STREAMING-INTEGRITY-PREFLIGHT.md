# Session handoff - ENTSO-E Parquet streaming integrity preflight

Date: 2026-08-06  
Decision: `D-20260806-244`  
Status: `PASS_SYNTHETIC_PARQUET_STREAM_INTEGRITY_MODEL_NO_GO`

## Outcome

D244 implements a bounded streaming verifier for the future normalized ENTSO-E
Parquet interface. It is entirely offline and synthetic. It binds D240's exact
artifact-integrity proof and D241's real Unity Catalog schema proof.

The D241 finding is preserved rather than hidden: the real raw tables contain
11/8/10 columns and still lack required normalized cadence, source-document
lineage, quality, revision, sign and canonical PIT fields. D244 therefore does
not admit current raw exports and never fabricates those fields.

For each synthetic normalized role, the verifier requires an exact
content-addressed path, inventory, SHA-256, size, row count, row-group count,
normalized schema, Arrow schema, codec set, Parquet format version and
`created_by`. It hashes bounded chunks before and after the scan on the same
descriptor, rejects links/reparse points/hardlinks and detects descriptor or
path identity changes.

The Parquet scan checks `PAR1` magic, footer size/equality, Thrift limits,
page checksums, exact schemas, codecs, row-group rows/bytes, package-wide
compressed/decompressed caps and decompression ratio. Every bounded Arrow
`RecordBatch` is visited. Batch rows/bytes and string bytes are bounded; nulls,
non-finite values and negative revisions fail closed.

No pandas import, DataFrame conversion, whole-Parquet byte materialization or
full D239 profile occurs. The data-quality method materially shaped the result:
byte integrity, batch-level validity, incremental analytical quality and real
source authority remain four separate verdicts. Only the first two synthetic
mechanics are green.

## Authority and cost boundary

Only `synthetic_parquet_stream_integrity_verified` is true. Incremental quality,
real receipt/hash/value, source authenticity, external time, PIT, seasonal
diagnostics, training, selection, model input, candidate assembly, promotion
and production are false.

D244 made zero Databricks connections or statements, zero Warehouse starts,
zero network calls, zero `H:` accesses and zero remote writes. The bound D241
source separately records its prior three control-plane GETs and zero SQL,
opened table rows, Warehouse starts or writes. D244 generated no Databricks
cost.

## Decision-number reconciliation

D241 and D242 were consumed by concurrent schema and probabilistic-output
batches while this batch was active; D243 was also reserved by the offline
series-inventory preflight. This batch was therefore finalized as D244 and all
contract, implementation, fixture and proof identities were regenerated. The
earlier local proof directories are superseded, not selected.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-PARQUET-STREAMING-INTEGRITY-PREFLIGHT-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_parquet_streaming_integrity.py`
- `tests/test_entsoe_parquet_streaming_integrity.py`
- `build/databricks-eex-daily/materialize_entsoe_parquet_streaming_integrity.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057, `H:` or heavy desk-data file was opened or
changed by D244. Existing unrelated worktree changes were preserved.

## Canonical identities

- contract raw SHA-256 / canonical content ID:
  `f5c7e34d383ec476fb8893348b91ac3aab011d42d36d7321b172d081461e8cca` /
  `9a79779d7ebe01b0e4dbfc65c40eaa23ead4f1f984685bc5055cc16f7a23969b`
- validator SHA-256:
  `784f900307542dcd68c3e6a476a078c56361ef626294e2106f7b20128404147f`
- tests SHA-256:
  `a8f44f673316ad98d1cf834021a9698764c87f9fc0811bb634f4463fa31b6e58`
- materializer SHA-256:
  `7e27a3c36c509491fb5db51d0f80c582031df5491bdbe1699b6e1a4b4b66b922`

Selected reproducible proof:

- content ID:
  `42c1065bf66117a2be2c792424f08d08511056d0df5f3b4b706ac57af1fcf564`
- manifest SHA-256:
  `53e4fe281d405fea64e6f6084d0ae129f608c4f6ff88e944d60be3bf904177d4`
- assessment SHA-256:
  `b65b01c19edf662a57c77f9764d14e2db77fbc6ce6afa4cd7257445233e62563`
- streaming-receipt SHA-256 / canonical content ID:
  `9f45ac53fefea7a28cddcc69e2594ac7d6f70f274d6279f9c56c6a9b2a7855e9` /
  `e9f6b8e93488fe9da66a44fb7aebfbc38a477e84707ab235c5d318f2c7c83986`
- synthetic package manifest raw SHA-256 / canonical content ID:
  `e7f91403eab06e897048ac94efc9225ed94bb77757e886ef0aed26e27e99583a` /
  `306339758b20fefd8f72143f13a6cc1b6e7b82af4f441da642baec5a4adc1bb2`
- path:
  `build/databricks-eex-daily/2026-08-06/entsoe-parquet-streaming-integrity-proofs/42c1065bf66117a2be2c792424f08d08511056d0df5f3b4b706ac57af1fcf564/`

Bound predecessor evidence:

- D240 proof content ID / manifest SHA-256:
  `5595a20c5b997485bbaa0e3aa41f90b131190e3e89d812b1acfdfaaebc88536b` /
  `97fc381771033c31da4f79789750638c70676bf32d1d3e844edc2609b3448ec3`
- D241 real-schema proof content ID / manifest SHA-256 / capture content ID:
  `d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544` /
  `1835f93a517e9c6769079984a376fb9879b22d7c8ce2922aa42b0dd646627ada` /
  `d69fdab73ba1d9c55f70f77925f2253d583564d06d922f2b41e035763bca176f`
- bound D241 status:
  `FAIL_REAL_CONTROL_PLANE_SCHEMA_INCOMPATIBLE_NO_MODEL_AUTHORITY`

## Synthetic fixture

The mechanical fixture contains three dimension rows, 8,193 latest rows and
8,193 vintage rows. Seven record batches are visited and the row-group metadata
declares 485,578 uncompressed bytes. These values test streaming mechanics only;
they are not an ENTSO-E quality sample or empirical model evidence. No raw value
is persisted in the proof bundle.

## Verification

Every shell action verified cwd and Git top-level as the canonical
`C:\Users\jbattaglia\PFC_LT`. Mutable outputs stayed below `build/`.

- Ruff on validator, tests and materializer: passed.
- focused D244: `16 passed in 4.66s`.
- D231-D244 acquisition chain: `374 passed in 11.58s`.
- materializer executed twice after final D244 reconciliation: identical proof
  content ID.
- proof counters: zero Databricks connection/statement, Warehouse start,
  network call, `H:` access and remote write.
- one failed producer attempt left an exact repo-local staging directory locked
  by a PyArrow metadata handle; explicit handle closure was added and that exact
  non-material staging directory was safely removed before the final reruns.

Focused test command:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe -m pytest tests\test_entsoe_parquet_streaming_integrity.py -q --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-d244-final
```

Materialization command, executed twice:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe build\databricks-eex-daily\materialize_entsoe_parquet_streaming_integrity.py --output-root C:\Users\jbattaglia\PFC_LT\build\databricks-eex-daily\2026-08-06\entsoe-parquet-streaming-integrity-proofs
```

## Remaining gaps and next permitted step

D244 is not an incremental quality profiler and cannot admit the current raw
Unity Catalog schemas. The next offline batch should specify and roast a
streaming quality accumulator for independently admitted normalized batches:
grain/key uniqueness, dimension joins, required family/zone/unit coverage,
native-grid completeness, freshness, revision chronology, latest/vintage
agreement and rolling-origin availability. It must preserve bounded memory and
remain synthetic until a governed real normalization and receipt exist.

Separately, D243's dimension-only query remains unexecuted. It may run only on a
future Europe/Zurich day when the Warehouse is already running and the daily
reservation is consumed. D244 does not change or grant that authority.

T057 remains sealed. The monthly solver remains sole level authority, LT
remains independent from CT, OMPEX remains benchmark-only and AFRY descriptive.
