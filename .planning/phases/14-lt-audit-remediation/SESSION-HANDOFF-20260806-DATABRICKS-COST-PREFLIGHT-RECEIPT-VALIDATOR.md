# Session handoff — Databricks cost-preflight receipt validator

Superseded for current request binding by D291 and
`SESSION-HANDOFF-20260806-DATABRICKS-GOLD-NTC-SOURCE-CONTRACT.md`. The D290
proof below remains historical and must not be used for the edited request.

Date: 2026-08-06  
Decision: D-20260806-290  
Status: modelling paused; local receipt validator complete; no remote authority

## Outcome

D290 validates the future D288 Phase 0 receipt before any human cost decision.
It binds the current enriched data-engineer request and the four D289-qualified
SQL files. No Databricks connection is made.

The exact receipt grain is one snapshot with four profile roles. It declares
Delta bytes/files, partitions, scan lower/upper bounds, estimation method,
pruning evidence, Warehouse state, metadata-only execution counters and
canonical DBU/runtime proposals. The snapshot ID hashes the canonical receipt
without its own ID.

The validator returns only:

- `READY_FOR_HUMAN_COST_REVIEW_NO_EXECUTION_AUTHORITY`;
- `STOP_NO_ACTIVE_WAREHOUSE`;
- `STOP_UNCAPPED_SCAN`.

`READY` is deliberately not Phase A GO. All source, PIT, predictive, model,
selection, candidate, OMPEX-superiority, promotion and production authorities
remain false.

## Fail-closed controls

- exact current request and query hashes;
- strict JSON with duplicate-key rejection;
- receipt age at most 24 hours and no future timestamp;
- exact four-role inventory with no substitution or duplicate;
- nonnegative Delta size/file and scan-bound consistency;
- canonical unavailable-estimate representation;
- pruning only with declared partitions and an explicit predicate;
- at most four metadata statements, zero business statements/rows, starts,
  writes and retries;
- no request-triggered or automatic Warehouse start;
- canonical decimal DBU estimates and runtime at most 900 seconds;
- all GO flags false;
- stable single-link receipt below repo-local `build/` plus final TOCTOU reread.

## Files and hashes

- `.planning/phases/14-lt-audit-remediation/DATABRICKS-PFC-SOURCE-COST-PREFLIGHT-CONTRACT-V1.json`
  - raw SHA-256:
    `74111f4e1e3cf3d3e9843b2b3300517fb8647eef908b57d5b22c41b065af160f`;
  - canonical content ID:
    `950ebfaa02a5ebbf81efb7ecfa3b7697684440e9c88c69590cc5bde14e404f14`.
- `pfc_shaping/validation/databricks_pfc_source_cost_preflight.py`
  - SHA-256:
    `21d8b1f7eb6c7e1946da6271acdadec81fa29fe6202534a8331f187ce417aafc`.
- `tests/test_databricks_pfc_source_cost_preflight.py`
  - SHA-256:
    `a1283d3dd92532fbe037f5a6cff17eb4377bdf6429cebab8af41dba69b35041b`.
- `build/d290_materialize.py`
  - SHA-256:
    `0ca51160cf1fa06048d182519f11d4c1f2018a4f1e83d7c81ae6b84bb124816b`.
- `docs/research/DATABRICKS-PFC-SOURCE-AUDIT-20260806.md`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`.
- `.planning/HANDOFF.md`.
- this handoff.

## Proof and verification

Proof:

`build/databricks-eex-daily/2026-08-06/databricks-pfc-source-cost-preflight-proofs/e3d3d7643bce35137cd8f9dd1c537e4d640021ba7fe86da59ed9f577a8f50cfc/manifest.json`

- proof content ID:
  `e3d3d7643bce35137cd8f9dd1c537e4d640021ba7fe86da59ed9f577a8f50cfc`;
- proof raw SHA-256:
  `75966436443bbd1091a32e3e065502e1367c7b6ce5ae8cf6253c1b25685b535c`;
- ready-assessment content ID:
  `8082c5f62d456cb3979f3f8e27e0162d4e0d4dc707db89cb70c5b67ac01cfd8a`;
- deterministic replay count: 2;
- receipt, Warehouse ID and business values persisted: false.

Tests:

- focused D290 mutation roast: `35 passed`;
- D288/D289/D290 plus LT import boundary: `63 passed, 1 skipped`;
- LT package boundary: `26 passed`;
- Ruff format/check and Python compilation: pass.

One adjacent command named a nonexistent `tests/test_path_safety.py`, then a
larger regrouping hit its 120-second runner timeout. No assertion ran in the
first command and no failure was emitted by the second. The split matrices
above completed explicitly green.

A preliminary synthetic proof was briefly materialized with the already-used
D289 decision number before the concurrent D289 SQL decision was observed. It
was removed after exact path and decision verification and replaced by the D290
proof above; it must not be used or reconstructed.

## Execution and next action

D290 performed zero Databricks connections/statements, business-row reads,
Warehouse starts, writes, network calls or `H:` accesses. No CT, Power BI,
AFRY, OMPEX, T057 or heavy desk-data file was changed.

Keep modelling paused. Wait for a real Phase 0 receipt or a governed platform
export. A real receipt must be copied below `build/` and pass D290; after that,
the user still decides whether any Phase A cost is acceptable.
