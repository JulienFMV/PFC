# Session handoff - ENTSO-E bounded execution proposal and receipt

Date: 2026-08-05  
Decision: `D-20260805-236`

## Outcome

D236 adds the final local, zero-query guard before any separately authorized
ENTSO-E extraction. It validates a cost-bounded proposal and a synthetic
receipt structure only. It does not authorize or prove a Databricks execution.

The proposal must bind the exact D233 governed package and D234 physical
compiler evidence, then freeze:

- exact compiler template hashes and parameters for the three normalized roles;
- one identical half-open UTC window for latest and vintage values, from 1 to
  31 whole days, with the vintage cutoff between target start and the frozen
  snapshot reference;
- a fixed-decimal estimated cost and hard cap, currency, estimation method,
  Warehouse size, five-minute auto-stop, one future start maximum, exactly
  three reads, zero retry, 300-second statement and 900-second batch ceilings;
- exact per-role row and byte caps, with mandatory rejection on truncation,
  reached limit or cap breach.

The synthetic receipt validator binds statement/template/parameter hashes,
requires ordered read-only `SELECT` statements, reconciles durations and local
export bytes, rejects writes and unsafe artifact paths, and never opens the
artifact bytes or ENTSO-E values. A real proposal remains unauthorized and a
real receipt cannot be admitted by this contract.

## Canonical files and hashes

- `.planning/phases/14-lt-audit-remediation/ENTSOE-BOUNDED-EXECUTION-PARAMETER-RECEIPT-CONTRACT-V1.json`
  - raw SHA-256: `9ae529cb242ea79d97a8deae32317c8d0fc2522c7afd216863c9f7c885a6c7b2`
  - canonical content ID:
    `834cbba541395072886e1ba80012a76cc55a0a8990b6d354f6d1bfe110a12156`
- `pfc_shaping/validation/entsoe_bounded_execution_receipt.py`
  - SHA-256: `0f41a0147e06128d5912c40e0d8c14e0b99aa1f932f4263bb61cae1c895a18d8`
- `tests/test_entsoe_bounded_execution_receipt.py`
  - SHA-256: `2827c16ac120d5e179b4d672658e8461650c766a4335e2056350e66c5fb923c1`
- `build/databricks-eex-daily/materialize_entsoe_bounded_execution_receipt.py`
  - SHA-256: `224f71bcb40d15045259abf0e398d1f49676fb2b46b030ac6911c76daa65e19f`

Durable documentation was updated in:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `docs/research/forwards_sources.md`;
- `.planning/HANDOFF.md`.

The inclusive D231-D236 test count was corrected from `187` to `237`; the
previous number omitted the 50 D235 external-time tests while labelling the
matrix D231-D236.

## Proof

- content ID:
  `772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247`
- path:
  `build/databricks-eex-daily/2026-08-05/entsoe-bounded-execution-receipt-proofs/772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247/`
- manifest SHA-256:
  `9d76631395176718b600ca0de8f7303ad8faea931c24298f514a54354da3d6e0`
- assessment SHA-256:
  `8fbe37824d333902a9c2eb87bbb11eb65d82efa62a4cee1d33cc4dfecd67e724`

Two final materializations returned the same content ID.

## Verification

- Ruff on validator, tests and materializer: passed.
- focused D236: `31 passed`.
- inclusive D231-D236 matrix: `237 passed in 0.80s`.
- adjacent D229/D231-D236 matrix: `296 passed in 6.00s`.
- proof counters: zero Databricks connection or statement, Warehouse start,
  network call, `H:` access and remote write.

All mutable pytest and proof outputs stayed below repository `build/`. No
connector, network call, secret, Databricks query, Warehouse start, remote
write, `H:` access, market value, ENTSO-E value or AFRY value was used.

## Concurrency reconciliation

A duplicate local draft was detected while the canonical bounded-execution
implementation was being written concurrently. The draft source, contract,
tests, materializer and its generated proof JSON files were removed before
handoff. They contained no real data and no unique evidence; the canonical
D236 files above are the sole retained implementation.

## Remaining gaps and next permitted step

D236 is preparation, not execution authority. Before a real capture:

1. a real owner-asserted D232 metadata capture and D234 physical mapping are
   still required;
2. the operator must supply the exact bounded proposal, explicit monetary cap
   and cost estimate;
3. the user must separately authorize that exact proposal and maximum cost;
4. only then may a future runner perform at most one governed local capture for
   the Europe/Zurich day, with no retries or Databricks writes;
5. a separate real-receipt and artifact-admission contract must verify hashes,
   actual values, units, grain, nulls, duplicates, joins and PIT availability.

Until those steps exist, real values, artifact hashes, data quality, PIT and
the independent future holdout remain unverified. Training, selection, model
input, candidate assembly, promotion and production remain false; T057 stays
sealed. The monthly solver remains sole level authority, LT stays independent
from CT, OMPEX remains benchmark-only and AFRY remains descriptive.
