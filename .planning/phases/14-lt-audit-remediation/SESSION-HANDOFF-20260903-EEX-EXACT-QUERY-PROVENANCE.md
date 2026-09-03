# Session handoff - EEX exact query provenance - 2026-09-03

## Outcome

The first local EEX evidence-completion item in the outage plan is complete.
The existing D231 zero-query validator now freezes the exact historical
three-table SQL bytes and binds their SHA-256 to the exact 5 August capture
manifest, ordered PRD physical tables, CH/POWER predicates, 12-column result
schema and opaque NDJSON artifact declaration.

The validator reads only the 3,810-byte manifest. It does not open, parse or
rehash the 29.8 MB market-value artifact. It makes no Databricks, Warehouse or
network call. Independent source time, signed envelopes and conversion into
the existing signed EEX vintage catalogue remain unverified; model-input,
training, selection, production and promotion authority remain false.

No newer EEX capture was requested. The existing capture remains the exact
reused evidence object even if other local datasets contain observations
through 31 August.

## Assumption and scope decision

The Phase 14 execution order still blocks model replay/comparison, stochastic
paths and scenario generation until governed EEX/ENTSO-E inputs and the future
holdout exist. The next safe local task was therefore interpreted as closing
the explicitly listed EEX query-provenance gap, not advancing model work.

The repository already had the correct boundary in
`databricks_zero_query_acquisition_plan.py`. Extending it was smaller and safer
than adding another intake or vintage system. The workbook-specific signed EEX
intake remains unchanged and is not made to accept the Databricks export.

## Changed files

- `pfc_shaping/validation/databricks_zero_query_acquisition_plan.py`:
  freezes the exact query bytes/hash, table/filter/schema inventory and fully
  validates the exact historical capture metadata; reports the three remaining
  authority gaps explicitly.
- `tests/test_databricks_zero_query_acquisition_plan.py`: exact-query,
  metadata-only positive and adversarial mutation tests, including JSON scalar
  type-confusion rejection.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`: marks only EEX local
  item 1 complete and leaves items 2-3 open.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`: durable decision
  D-20260903-280.
- `.planning/HANDOFF.md` and this handoff.

No CT, model, solver, training, scenario, stochastic-path, T057, Power BI,
heavy data or production file changed.

## Frozen evidence identities

- exact query byte length: `625`;
- exact query SHA-256:
  `54a2e7e1752af4506673d2b5cbc2666f0deea45ec96e6d82561e4b265c78797a`;
- exact historical manifest byte length: `3,810`;
- exact historical manifest SHA-256:
  `f8ec096be43851d85b16ec2b678d4a695fb0521c2c651e8bcf7c2491a29b50c1`;
- opaque artifact declaration: `82,552` rows, `29,763,661` bytes;
- opaque artifact SHA-256:
  `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`;
- validator normalized-LF source SHA-256:
  `892faa176f68f8fdd4dd72af39cc99205827c5c34e71e31bf933dcbbf235d50d`;
- focused test normalized-LF SHA-256:
  `4c53135fcf26aa8549384cc9a6e417dd86af1d9f5ef73e028677d1ccbea534cb`.

## Contract guarantees

- Any byte drift in the frozen SQL fails closed.
- The manifest must have the exact field inventory, fixed capture identity,
  ordered table list, CH/POWER filter, exact output schema, read-only/no-write
  flags, statement metadata, counts and opaque artifact declaration.
- Boolean and integer scalar type confusion is rejected at the new boundary.
- The combined path verifier first checks the exact manifest byte hash, then
  validates its semantics; it never discovers a capture by directory scan.
- A successful assessment claims only
  `exact_query_and_predicate_provenance_verified=True`.
- Source time, signed-envelope and vintage-conversion evidence remain false,
  as do model, training, selection and production authorities.

## Verification

Initial focused run:

```text
27 passed, 10 failed
```

All ten failures were the same fixture defect: the historical PowerShell JSON
contains a UTF-8 BOM and the new direct test helper decoded it as plain UTF-8.
The production path already used `utf-8-sig`. The fixture was corrected to
reproduce the historical encoding; no validator condition was relaxed.

Final focused D231 matrix:

```text
39 passed in 0.16s
```

Final adjacent EEX/Databricks/governed-acquisition matrix:

```text
119 passed in 1.50s
```

Required LT minimum:

```text
58 passed, 1 skipped in 6.37s
```

The skip is the pre-existing optional TensorFlow import boundary.

Targeted quality checks:

```text
python -m ruff check <2 changed Python/test files>
All checks passed!
git diff --check
pass (expected Windows CRLF notices only)
```

A whole-file formatter run briefly changed pre-existing adjacent lines. The
counter-audit identified and removed that unrelated style drift; the final
diff retains only request-traceable changes.

Every pytest basetemp and mutable temporary path was below `build/`.

## Authority and cost

- price/business-value rows opened by this implementation: `0`;
- Databricks requests/statements: `0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- network calls or remote writes: `0/0`;
- independent timestamps, signatures or CAS objects created: `0/0/0`;
- model training/retraining or artifacts: `0/0`;
- CT changes, T057 access and CH solver-authority changes: `0/0/0`.

## Residual risks and next action

1. Exact query provenance proves what the historical statement selected; it
   does not prove source publication time, independent observation time,
   source authenticity, signature custody or PIT admissibility.
2. The next EEX item requires independently supplied source-time and signed
   envelope evidence over the already frozen batch/artifact identities. Local
   code must not manufacture those facts or self-sign them into authority.
3. Only after that evidence exists should a separate governed adapter convert
   the causal daily observations into the existing signed EEX vintage
   catalogue. Do not create a parallel catalogue.
4. ENTSO-E remains on the independent platform-owned delivery lane. Do not
   query Databricks, start a Warehouse, substitute legacy/synthetic history or
   treat the public incident status as completeness evidence.
5. Model replay, challenger comparison and scenario/path generation remain
   blocked pending governed inputs and the independently frozen future
   holdout. T057 remains sealed.

Durable decision: D-20260903-280.
