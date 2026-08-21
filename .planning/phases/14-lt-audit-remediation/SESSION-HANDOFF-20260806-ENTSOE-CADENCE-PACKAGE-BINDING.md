# Session handoff - ENTSO-E cadence package binding

Date: 2026-08-06  
Decision: D-20260806-270  
Status: `PASS_SYNTHETIC_CADENCE_BOUND_TO_QUALITY_PACKAGE_NO_GO`

## Outcome

D270 composes the stable D244/D245 Parquet package with the D261 explicit
cadence-window contract without weakening either. The base package remains
exactly three roles. Effective-dated regimes are delivered in a separate
content-addressed `series_resolution_regimes.parquet` sidecar whose manifest
binds the exact snapshot, base manifest, quality context, assessment window
and metadata cut-off.

D245 full incremental quality must pass first. The additional cadence scan
then opens only the dimension identity/group/resolution and latest target
timestamp columns. It does not decode numeric values. Ephemeral SQLite below
`build/` stores keys, timestamps and regimes; expected and observed native
slots are compared as sorted streams with a global slot cap.

## Exact changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-CADENCE-PACKAGE-BINDING-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_cadence_package_binding.py`
- `tests/test_entsoe_cadence_package_binding.py`
- `build/databricks-eex-daily/materialize_entsoe_cadence_package_binding_proof.py`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

The same session completed the reserved D261 v2 proof materializer and added
`SESSION-HANDOFF-20260806-ENTSOE-RESOLUTION-REGIMES-EXPLICIT-WINDOW.md`.
Concurrent D255/D271 work was preserved; no CT, AFRY numeric, Power BI or heavy
desk-data file was touched.

## Canonical D270 evidence

- contract SHA-256 / content ID:
  `b390cdb3eb19791bd540a07833769fa18f0db3a529c56eed6c05b6567ae508d5` /
  `db1ffcd92bb3ae720b44da02a6e6f151a630b9e3887bd84902e91cddc3cb717f`;
- validator / tests / materializer SHA-256:
  `7f0f34596f040895d9462d41472eae8abcf28cc41d45e7bbed353899fd2ee809` /
  `05963d99057557821d51eb021795f93e4ee053cb1afa946aa68dff8c92ae12a1` /
  `f7bf50cc3a2f1a094299c545264e5551227e4a484705f18ef00206a07edb9d8b`;
- deterministic proof ID:
  `e05afce560c93a7d897c67cd8caa36b04026a950dc1ac03d149e1dc32e4a9018`;
- proof manifest / assessment SHA-256:
  `63132ee50ef270582cebb64b80eeed84719c81e25e2b38698865b116bbe25c7a` /
  `68db19407319edb7716e0a9186fa05adccfb2fd08165c176b105e1d2f382bfe4`;
- proof path:
  `build/databricks-eex-daily/2026-08-06/entsoe-cadence-package-binding-proofs/e05afce560c93a7d897c67cd8caa36b04026a950dc1ac03d149e1dc32e4a9018/`.

The selected proof contains 33 hashed synthetic series profiles, 33 regimes,
792 expected and observed timestamps, zero missing slots and a 69,632-byte
SQLite peak. It contains no real values or clear series identifiers. Two
materializations plus a final replay returned the same proof ID.

## Verification and failures handled

- focused D270 roast: `12 passed`;
- final adjacent D244/D245/D261/D270 matrix: `70 passed`;
- Ruff passes on validator, tests and materializer;
- the expanded-window mutation reports exactly 66 leading/trailing missing
  slots (two one-hour edges across 33 series).

The first adjacent launch reported `61 passed, 9 setup errors` because a D245
fixture imported as a Pytest plugin disappeared when its source test module
was also collected. The fixture was wrapped locally; targeted `12 passed` and
the exact matrix then passed. The first materializer invocation stopped before
construction because the repo root was absent from `sys.path`; the entry point
was fixed and subsequent deterministic runs passed. Neither failed launch is
cited as green evidence.

## Cost and authority

Zero Databricks connections/statements/writes, zero Warehouse starts, zero
network calls, zero `H:` accesses and zero opened real rows. No real-source,
owner, PIT, value, model, candidate, promotion or production authority is
granted. D247's same-day Databricks reservation remains consumed.

## What this changes for the first ambitious PFC

The local admission shape is now materially stronger: native cadence can no
longer be detached from the exact quality-checked export, and missing edges
cannot hide behind observed bounds. This removes a source-data fabrication
risk but does not yet make ENTSO-E empirical evidence admissible.

Next safe batch: compose the already independent family, directional,
effective-zone, cadence, quality, revision and PIT receipts into one real
package admission gate. Only after the governed local EEX/ENTSO-E export
passes that gate should a new future holdout be frozen and rolling-origin
model selection resume. OMPEX remains post-freeze benchmark only.

