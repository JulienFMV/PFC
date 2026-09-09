# Session handoff - ENTSO-E export bridge and hedging scope - 2026-09-02

## Outcome

Two consumer-complete, hash-bound ENTSO-E day-ahead export contracts now bridge
Silver vintages to the normalized interval consumer without Databricks access.
`causal_asof` and `realized_final` are separate; neither availability nor
finality is fabricated. A self-contained unsigned replay proves deterministic
adaptation and detects value/evidence tampering.

The user-supplied hedging description was checked against the code. CH/DE is
recorded as the target valuation/hedging scope, but current implementation
authority is not symmetric: the hard monthly solver targets CH only, while the
DE branch remains on the legacy monthly path. FR/AT/IT-North remain
observation/risk or future-market candidates.

## Files added

- `pfc_shaping/validation/entsoe_day_ahead_export.py`
  - exact raw Silver export schema;
  - explicit market subset and market-purpose validation;
  - causal and realized parameter builders/adapters;
  - exact external-evidence scope and covered-frame hash binding;
  - offline cost preflight;
  - in-memory self-contained replay package and verifier.
- `tests/test_entsoe_day_ahead_export.py`
  - 14 synthetic contract, integration, cost and tamper tests.
- `docs/data/sql/databricks_prd_entsoe_day_ahead_causal_export_v2.sql`
  - SHA-256
    `f86cfeeec6bb2dc6d9c579426df091d7d45b01f0bca848b47aba62500cedbc6d`.
- `docs/data/sql/databricks_prd_entsoe_day_ahead_realized_export_v2.sql`
  - SHA-256
    `9f4104e4db8b16bfa21e0fce26afb47d364f983b95209424293356489fe6d81a`.
- `docs/data/ENTSOE-DAY-AHEAD-EXPORT-V2.md`
  - lineage, finality, availability, market scope and cost fence.
- `docs/model/PFC-HEDGING-SCOPE-AND-MODEL-ROADMAP.md`
  - verified model inventory, CH/DE authority distinction and execution order.
- this handoff.

## Files updated

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - added D-20260902-268.
- `.planning/HANDOFF.md`
  - current-state and next-reading update.

All pre-existing tracked and untracked user/session work was preserved. The
historical PIT/profile SQL, capture schemas, receipt bytes and SHA constants
were not modified.

## Producer and model evidence inspected

- Governed read-only clone:
  `build/data-engineer-repos/opendata-lakehouse`.
- The clone contains deployed commit
  `a7e920d95b94b2db59180412f31213f917e8d8a3` and parser correction
  `8549319ff944bfcf2e8123b05907ac025a8f23b8`; ancestry check returned success.
- Deployed Silver DDL/projection confirms direct availability, provenance,
  normalized interval and source revision fields, but no finality column.
- `pfc_shaping/config.yaml` selects `shape_hourly_mode: mlp` and a CH-only
  monthly solver target.
- `production_phases.py` calls the MLP on CH history, Huber/Ridge on native
  DE-LU quarter-hours post-2025-10, aggregate hydro correction and ENTSO-E
  climatology; it builds CH and legacy-level DE branches.
- `shape_hourly_mlp.py` computes 180-day weights but does not pass sample
  weights to the final MLP fit. The weights mostly cancel inside each hourly
  aggregation, so effective recency weighting is not proven.

## Export contract

### Common direct Silver fields

Series identity/classification, normalized start/right-edge/end, resolution,
value, publication/first-seen, availability, history/DQ, source TimeSeries,
document/revision, Bronze snapshot/file and vintage ID.

### Deterministic local fields

Market zone/timezone from the closed field mapping and `quality_status=PASSED`
only after exact `dq_failed is False` validation.

### External-only claims

- `original_publication_proven`: scoped platform receipt, timestamp before
  delivery, exact covered-frame semantic hash.
- `is_final`: scoped finality/settlement evidence, assertion after delivery and
  before assessment cutoff, exact covered-frame semantic hash.

Latest-revision rank alone never grants finality. `curve_type` is neither
projected nor inferred.

## Time, market and cost boundaries

- explicit non-empty subset of CH/DE-LU/AT/FR/IT-North;
- no default AT or DE-LU auction sequence;
- at most 32 days and two explicit UTC year/month partitions, sufficient for a
  Europe/Zurich market month across DST;
- SQL is read-only, has no `SELECT *`, and uses a 20,001-row rejection sentinel;
- `LIMIT` is not treated as a scan-byte cap;
- offline preflight returns `STOP_NO_ACTIVE_WAREHOUSE` for the observed stopped
  Warehouse and always declares execution authority false.

## Verification

```text
specialized export/evidence/replay
14 passed

adjacent consumer/PRD/capture/reconciliation/Databricks replay-snapshot
153 passed

complete non-slow ENTSO-E/LSEG/spot matrix (39 files)
854 passed in 125.65s

LT minimum + LT package contract
84 passed, 1 skipped in 97.71s

targeted Ruff check
pass

targeted Ruff format check
pass
```

The skip is the pre-existing optional TensorFlow import boundary. One attempted
test selector included two XML fixtures and ran zero tests; a second selector
matched none. Both were corrected before the reported 39-file/854-test run and
are not counted as verification.

## Execution and cost

- Databricks connections/statements/business rows: `0/0/0`.
- Warehouse starts/resizes/creates: `0/0/0`.
- writes/Databricks network calls/remote writes: `0/0/0`.
- one read-only GitHub connector lookup returned `404`; it made no mutation,
  and producer verification used the governed local clone instead.
- incremental DBU/Azure exposure: zero.
- no real export, finality receipt, original-publication receipt or v2 replay
  package was claimed.

## Residual blockers and next admissible actions

1. Obtain a platform-owned or explicitly approved, hard-capped real export;
   the observed stopped Warehouse must not be started for this work.
2. Obtain exact effective-dated AT/DE-LU series-selection evidence and real
   LSEG reconciliations/finality receipts bound to exported values.
3. Keep historical returned-document `createdDateTime` blocked for causal use
   until original publication is independently proven.
4. Freeze the rolling-origin protocol, challenger set, metrics and a new
   future holdout before any MLP weighting fix or retraining.
5. Replay the unchanged current CH candidate as baseline, then compare the
   preregistered weighted-MLP/Ridge/GAM/LightGBM challengers.
6. Treat a governed DE monthly level authority as a separate project; do not
   infer it from the existing legacy DE branch.
7. Keep market PFC, physical P scenarios and the FMV hydro optimizer as three
   distinct products.

Global status remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; T057 remains sealed and the
monthly CH solver remains the sole current level authority.
