# Session handoff - ENTSO-E series selection resolved - 2026-09-03

## Outcome

The July 2026 construction reference is now resolved:

- AT: `day_ahead_prices||at_price||1`;
- DE-LU: `day_ahead_prices||de_lu_price||1`.

Both sequence-1 series equal their independently configured LSEG EPEX
day-ahead curves on every one of the 2,976 quarter-hours in the exact July
Swiss-local window. Sequence 2 differs materially in both markets. No
in-window change, missing matched quarter-hour or overlap was observed.

This closes the construction-selection ambiguity. It does not grant model,
monthly-level, publication, production or trading authority and does not turn
the July backfill into causal truth.

## Why the prior owner wait was removed

The user clarified that the local `JulienFMV` GitHub identity can read Jérôme's
producer repository and explicitly authorized using the available Databricks
access to answer the question. The GitHub app connector did not expose the
private `FMVSA` organization, but the existing local `gh` authentication did.

The deployed producer code at
`a7e920d95b94b2db59180412f31213f917e8d8a3` confirms that A44
classification positions 1 and 2 are distinct source auction identities. It
intentionally preserves both and defines no default. Code inspection alone
therefore could not select one, but it established that neither sequence was a
duplicate or stale rebuild residue.

An exact full-window comparison to the already-bound LSEG EPEX actual-price
curves supplied the missing construction criterion. A further human answer is
not needed for the July construction smoke export.

## Databricks execution and result

The `PBI SQL Warehouse - Analytics` was observed `RUNNING` before submission.
It was not started, resized or created. One aggregate-only statement ran:

- statement ID: `01f1a79c-0849-1281-af0c-ee155e8346ce`;
- execution: `2026-09-03T13:33:22.103000Z` to
  `2026-09-03T13:33:49.508000Z`;
- output: four aggregate rows, one chunk, not truncated;
- raw price rows returned: `0`;
- bytes/files read: `9,364,142,086 / 102`;
- bytes/files pruned: `1,533,406,893 / 155`;
- remote-write bytes and disk spill: `0/0`.

The statement expanded valid aligned ENTSO-E intervals to 15-minute points,
selected the latest revisions observable at execution time and matched the
exact AT/DE-LU LSEG curves at interval start. Query history exposed only a
redacted or unavailable ten-byte query-text placeholder, so exact executed SQL
provenance is not claimed. The comparison must not be repeated; its 9.36 GB
read is material and the conclusion is already decisive.

## Aggregate comparison

| Market | Sequence | Matched / expected | MAE EUR/MWh | RMSE EUR/MWh | Correlation | Equal within half-cent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AT | 1 | 2,976 / 2,976 | 0.000000 | 0.000000 | 1.000000000 | 2,976 |
| AT | 2 | 2,976 / 2,976 | 11.299412 | 16.209279 | 0.960061613 | 4 |
| DE-LU | 1 | 2,976 / 2,976 | 0.000000 | 0.000000 | 1.000000000 | 2,976 |
| DE-LU | 2 | 2,976 / 2,976 | 9.549943 | 14.263226 | 0.973996024 | 2 |

Every row has zero missing LSEG quarter-hours and zero overlapping expanded
ENTSO-E quarter-hours.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-EVIDENCE-V1-20260903.json`:
  aggregate evidence, source/code identities, exact selection, execution cost,
  limitations and negative authorities.
- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`:
  records the one statement and six Databricks API requests, freezes the
  construction selection and advances the remaining execution order.
- `tests/test_lt_source_acquisition_outage_plan.py`: canonical evidence
  binding, exact metrics/selection, cost accounting and authority-negative
  assertions.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`: records the resolved
  selection and prohibits repeating the aggregate comparison.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`: durable decision
  D-20260903-282.
- `.planning/HANDOFF.md` and this handoff.

The previously committed request artifact remains byte-identical and truthful:
it was not transmitted and no owner response was received. No LT runtime,
model, solver, CT, scenario, stochastic-path, T057, Power BI or heavy desk-data
file changed.

## Frozen evidence identity

- schema:
  `fmv_entsoe_day_ahead_effective_series_selection_evidence.v1`;
- canonical JSON byte length: `3,767`;
- canonical JSON SHA-256:
  `968c3e0d5336f4ff38a130271456f7889c7a885e2e119ef62e5df596f7f8c0ab`.

## Verification

```text
focused outage-plan/evidence matrix: 12 passed in 0.16s
adjacent ENTSO-E export/LSEG reconciliation matrix: 52 passed in 13.83s
required LT minimum: 58 passed, 1 skipped in 68.66s
Ruff, JSON syntax and git diff --check: pass
```

The skip is the pre-existing optional TensorFlow import boundary. Every pytest
basetemp and mutable test output remained below `build/`.

## Authority and residual boundary

- Databricks API requests/statements: `6/1`;
- Warehouse starts/resizes/creates: `0/0/0`;
- aggregate comparison/raw returned rows: `1/0`;
- remote writes: `0`;
- model training/retraining: `0/0`;
- CT changes, T057 access and CH solver-authority changes: `0/0/0`.

Next, request the platform-owned bounded July `realized_final` export using the
two frozen sequence-1 keys. That export still requires value-bound finality
evidence and offline replay before LSEG reconciliation. Do not infer
`causal_asof`, retrain or start scenario/path work. The EEX source-time,
signature and signed-vintage conversion gaps remain independent.

Durable decision: D-20260903-282.
