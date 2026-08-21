# Session handoff - ENTSO-E native-cadence legacy ingest

Date: 2026-08-06
Decision: D-20260806-271
Scope: offline code hardening only; no data or model admission

## Outcome

D271 removes the last explicit source-grid fabrication from
`pfc_shaping/data/ingest_entso.py`. Hourly source series remain hourly, mixed
native grids join with nulls, and missing generation is no longer changed to
zero. Load selection rejects ambiguous response columns. Border schedules,
physical flows and NTC retain both raw directions; net/total values exist only
where both directions overlap.

The local cache updater now rejects any existing file before network access.
The historical `entso_15min.parquet` filename does not prove native cadence,
and older bytes may contain forward-filled quarter hours. A future capture
must therefore use a new immutable path and still pass the governed D243-D270
evidence chain before model use.

## Files changed

- `pfc_shaping/data/ingest_entso.py`
  - native timestamp preservation;
  - no raw zero-fill;
  - explicit actual-load selection;
  - duplicate timestamp rejection;
  - separate directional border series;
  - fail-before-network legacy-cache refusal;
- `tests/test_entsoe_native_cadence_ingest.py`
  - seven mutation tests plus the cache/network ordering test;
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
  - immutable snapshot and directional-null instruction;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/HANDOFF.md`.

No heavy data file, CT module, Power BI artifact or Databricks object was
opened or modified.

## Verification

All commands ran from `C:\Users\jbattaglia\PFC_LT` through the governed
repo-local wrapper.

- `entnative271a`: initial roast, `20 passed`, two test-fixture failures:
  missing fake-client method handles and an incorrect arithmetic expectation;
  no product-code failure was observed;
- `entnative271b`: corrected fixtures, `22 passed`;
- `entnat271c`: final focused native-cadence, replay and LT-contract matrix,
  `23 passed in 1.34s`;
- `entall271a`: intermediate complete ENTSO-E matrix, `539 passed`;
- `entall271b`: final complete ENTSO-E matrix, `540 passed in 27.29s`;
- `entrf271a` and `entrf271b`: Ruff passes.

An attempted Ruff wrapper call used a run ID longer than the permitted 16
characters and was rejected before Ruff. The conforming rerun passed and the
rejected launch has no test authority.

One explicit `py_compile` diagnostic created the ignored, non-authoritative
`pfc_shaping/data/__pycache__/ingest_entso.cpython-311.pyc`. An exact-path
PowerShell cleanup attempt was rejected by the workstation tool policy before
execution; no broader or alternate deletion was attempted. The bytecode is
not Git evidence and is not used by the governed test wrapper.

SHA-256:

- `pfc_shaping/data/ingest_entso.py`:
  `52da22e3e5b6833f3e7ae17084016c8878f647ebae0d7a211d98d0b17cf460e5`;
- `tests/test_entsoe_native_cadence_ingest.py`:
  `e97e459210bbd713ab1be3dc8406286784fe6df4f6d99ffc443f08d6d9ceeb8f`.

## Execution and cost receipt

- ENTSO-E API/network calls: 0;
- Databricks/control-plane calls: 0;
- SQL statements: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- local heavy-data writes: 0;
- real ENTSO-E value rows opened: 0;
- `H:` access: 0.

## Authority and residual risks

D271 makes the legacy code less capable of fabricating data; it does not make
legacy local data admissible. Effective resolution metadata, exact family and
zone/border mappings, units, signs, lineage, quality, revisions, PIT and real
coverage remain unproven. The versioned replay transform still uses neutral
defaults for some derived features; that separate behavior was not changed in
D271 and must not be interpreted as raw-data completeness.

D261 is complete. D270 remains reserved/concurrent and was not edited. Its
cadence-package binding must finish independently. The next safe model-facing
batch is a PIT-safe ENTSO-E feature contract, not empirical selection: actual
same-target values and forecast errors must never leak into an earlier
forecast origin, and all derived features must remain zero-mean shape
candidates until rolling-origin evidence and a new holdout exist.

The AFRY context and its source, semantic and diagnostic contracts were read
before this LT shaping review. No restricted AFRY numeric value was opened,
copied or used. AFRY remains descriptive only; T057 remains sealed; the
monthly solver remains the sole level authority.
