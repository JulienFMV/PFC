# Session handoff - ENTSO-E effective series selection request - 2026-09-03

## Outcome

The second local preparation item in the source-outage plan is complete. A
machine-readable, metadata-only request now asks the ENTSO-E data owner for the
missing effective-dated AT and DE-LU classification-sequence choices for the
July 2026 `realized_final` smoke export.

The request does not choose a SeriesKey. It has not been transmitted, no owner
response has been received and no source-selection or model authority follows.
No Databricks, Warehouse, ENTSO-E, LSEG, business-value, model, CT, T057 or CH
solver operation occurred.

## Audit and scope decision

The existing external-time and signature stack was reviewed first. It already
prepares the relevant DSSE roles, Merkle commitments and RFC 3161 request
surface, but real EEX completion still requires externally supplied signer,
trust-root, policy and timestamp evidence. Creating local signatures or a
synthetic timestamp would not close that gap and would falsely suggest
authority, so no code was added there.

The next explicit outage-plan task was ENTSO-E effective-dated selection. The
existing runtime already admits exactly seven candidate keys and rejects
unknown keys without a default or averaging rule. The smallest correct change
was therefore to freeze the owner request and test it against that inventory,
not to create another runtime validator or transport abstraction.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-REQUEST-V1-20260903.json`:
  exact metadata-only owner request, response requirements and zero-authority
  counters.
- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`:
  binds the request, records response/selection false, removes the completed
  EEX query-provenance gap and advances the remaining execution order.
- `tests/test_lt_source_acquisition_outage_plan.py`: canonical request binding,
  authority-negative assertions, runtime-inventory parity and all four
  synthetic AT/DE-LU candidate-pair parameter builds.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`: records the prepared but
  untransmitted request and the still-missing owner response.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`: durable decision
  D-20260903-281.
- `.planning/HANDOFF.md` and this handoff.

No LT runtime, model, solver, CT, production, scenario, stochastic-path, T057,
Power BI or heavy data file changed.

## Frozen request identity and scope

- schema: `fmv_entsoe_day_ahead_effective_series_selection_request.v1`;
- canonical JSON byte length: `2,301`;
- canonical JSON SHA-256:
  `6794566035e40b69eb5d104d09e317896b69502ca800513ef094ca46673d6cfb`;
- usage: `realized_final` only;
- Swiss market month: July 2026;
- UTC window: `[2026-06-30T22:00:00Z, 2026-07-31T22:00:00Z)`;
- UTC partitions: `2026-06`, `2026-07`;
- fixed keys: CH, FR and IT-North unique admitted keys;
- owner decisions: one of two admitted sequence keys for AT and one of two for
  DE-LU.

The request is explicitly metadata-only and asks for no business values.

## Contract guarantees

- The plan binds the request by canonical semantic JSON hash, independent of
  line-ending or formatting changes.
- Request candidates are tested directly against the existing runtime
  inventory.
- All four AT/DE-LU candidate pairs build the exact July UTC window, partition
  pruning parameters and fixed CH/FR/IT-North keys.
- A future response must contain exactly one rule per field covering the
  complete half-open request window and bind owner identity, selection basis,
  source reference, classification sequence, evidence hash and assertion time.
- An in-window key change is not representable by the current single-key export
  parameters. It blocks this request and requires a separately reviewed
  segmented-export plan.
- Defaults, averaging and consumer inference are forbidden.
- The request, synthetic qualification and any unauthenticated future response
  grant no selection, model, monthly-level, publication, production or trading
  authority.

## Verification

The first focused run produced `3 passed, 1 failed`. The failure exposed that
the test expected only window bounds and SeriesKeys while the production
builder also correctly emits four UTC partition-pruning parameters. The exact
June/July parameters were added to the assertion; no production behavior or
contract was relaxed.

Final focused matrix:

```text
4 passed, 7 deselected in 0.11s
```

Adjacent ENTSO-E export and outage-plan matrix:

```text
24 passed, 1 deselected in 0.48s
```

The one deselected historical EEX test rehashes the 29.8 MB opaque artifact.
It was unnecessary for this metadata-only increment; the new D280 semantic
binding assertion did run.

Required LT minimum:

```text
58 passed, 1 skipped in 16.88s
```

The skip is the pre-existing optional TensorFlow import boundary. Targeted
Ruff checks and both JSON syntax checks pass. Every pytest basetemp and mutable
test output remained below `build/`.

## Authority and cost

- business or price rows opened: `0`;
- Databricks connections/statements: `0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- network calls or remote writes: `0/0`;
- owner responses, real signatures or trusted timestamps created: `0/0/0`;
- model training/retraining or artifacts: `0/0`;
- CT changes, T057 access and CH solver-authority changes: `0/0/0`.

## Residual risks and next action

1. An external owner must receive the frozen request and return authenticated,
   effective-dated evidence. Local code must not infer or manufacture the AT or
   DE-LU choice.
2. Only after that evidence is admitted may the platform owner prepare the
   bounded July `realized_final` export from already-materialized Silver rows.
   This laptop must not query Databricks or start the stopped Warehouse.
3. The realized export still needs exact value-bound finality evidence and an
   offline replay before matched LSEG reconciliation.
4. July backfill remains non-causal. Future `causal_asof` work and model
   comparison remain blocked until a genuinely prospective frozen origin and
   independent holdout exist.
5. EEX still needs independent source time, signed envelopes/external time and
   conversion into the existing signed vintage catalogue.

Durable decision: D-20260903-281.
