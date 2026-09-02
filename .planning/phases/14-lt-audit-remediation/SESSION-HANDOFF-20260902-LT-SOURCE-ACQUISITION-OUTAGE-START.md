# Session handoff - outage-aware LT source acquisition start - 2026-09-02

## Outcome

Point 1 of the post-interface-audit sequence has started without Databricks or
model execution. The public ENTSO-E incident is isolated from the governed
admission state, the existing EEX capture is rehashed, and an exact
zero-execution acquisition plan now separates EEX, ENTSO-E and LSEG.

SMARD reported on 1 September 2026 that an ENTSO-E Transparency Platform
technical failure was causing partial time-series gaps and strong delays. The
Transparency Platform itself timed out during the read-only web check on 2
September. These observations corroborate a service incident but grant no
authority over FMV Silver availability, source freshness, original
publication, finality or execution.

## Files added

- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`
- `tests/test_lt_source_acquisition_outage_plan.py`
- this handoff

## Files updated

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - added D-20260902-271;
- `.planning/HANDOFF.md`
  - updated the current reading order and status.

No CT, heavy desk data, model weight, solver logic, production flag or T057
file was opened or modified.

## EEX lane

The existing local bytes match their recorded identities:

- `build/databricks-eex-daily/2026-08-05/eex_ch_power.ndjson`
  - size: `29,763,661` bytes;
  - SHA-256:
    `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`;
- `build/databricks-eex-daily/2026-08-05/manifest.json`
  - size: `3,810` bytes;
  - SHA-256:
    `f8ec096be43851d85b16ec2b678d4a695fb0521c2c651e8bcf7c2491a29b50c1`;
  - declared rows: `82,552`;
  - declared source tables: the existing EEX fact/product/delivery-period
    three-table join.

Status is `LOCAL_BYTES_HASH_VALID_ADMISSION_EVIDENCE_PENDING`. No price row
was parsed in this session. The remaining work is exact query/predicate
provenance, independent source time, signed envelopes/external time and
conversion into the existing signed EEX vintage catalogue. The direct EEX API
readiness remains `NO_GO`; no new query is justified.

## ENTSO-E and LSEG lane

The laptop-side state remains `STOP_NO_ACTIVE_WAREHOUSE`. The plan authorizes
no connection, statement, compute start, resize or creation.

The first external delivery request is a platform-owned export from rows that
were already materialized before the incident:

- use: `realized_final` candidate, adapter/reconciliation smoke only;
- market month: July 2026, Europe/Zurich;
- UTC window: `[2026-06-30T22:00:00Z, 2026-07-31T22:00:00Z)`;
- partitions: `2026-06`, `2026-07`;
- markets: CH, DE-LU, AT, FR and IT-North;
- exact realized v2 SQL SHA-256:
  `9f4104e4db8b16bfa21e0fce26afb47d364f983b95209424293356489fe6d81a`.

Before values, exact effective-dated AT/DE-LU series-selection evidence is
required. Passing the adapter also requires value-bound finality evidence.
LSEG waits for the exact validated ENTSO-E frame, then covers CH/AT/DE-LU/FR;
IT-North remains ENTSO-E-only.

This July export is not causal history, a holdout or a model input. The August
rebuild is backfill and cannot become July point-in-time truth. Prospective
`causal_asof` remains blocked until producer recovery, frozen origins and a new
future holdout.

## Verification

Executed from the canonical workspace through the repo-local supervisor:

```text
python -m scripts.run_workspace_local --run-id p1outage -- python -m pytest \
  tests/test_lt_source_acquisition_outage_plan.py \
  tests/test_eex_entsoe_governed_acquisition_package.py \
  tests/test_entsoe_day_ahead_export.py \
  tests/test_spot_source_reconciliation.py -q

91 passed in 1.63s
```

`git diff --check`, targeted Ruff check and targeted Ruff format check passed.
The first Ruff check found only one import-block blank-line formatting issue;
it was corrected before the final results above.

## Execution and authority counters

- Databricks connections/statements/business rows: `0/0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- model training/retraining: `0`;
- remote writes: `0`;
- CT changes: `0`;
- monthly CH solver-authority changes: `0`.

All model-input, selection, monthly-level, publication, production and trading
authorities remain false. Global admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.

## Next action

1. Send or hand off the exact plan to the EEX evidence owner and the ENTSO-E
   platform/data owner; this repo work does not itself authorize execution.
2. Accept EEX evidence only around the exact bound bytes.
3. Accept ENTSO-E only if an existing materialized snapshot and exact
   effective-dated series selection are independently delivered. If not,
   wait for producer recovery without substitution.
4. Validate the unsigned v2 replay locally, then request the matched LSEG and
   finality evidence.
5. Freeze prospective origins and the new future holdout before any causal
   capture, challenger comparison or retraining.

Durable decision: D-20260902-271.
