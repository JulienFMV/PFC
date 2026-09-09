# Session handoff - ENTSO-E derived arithmetic execution

Date: 2026-08-06  
Decision: D-20260806-283  
Status: synthetic decimal arithmetic PASS; real and model authority remain false

## Outcome

D283 adds the first executable derived ENTSO-E transform after D281 lineage and
D282 physical compatibility. It consumes separate synthetic realized-load and
operational-forecast artifacts bound to the exact D282 assessment and exact
feature-record hashes.

The executor:

- accepts canonical base-10 decimal text or `NULL`, never binary floats;
- computes exactly `REALIZED_ACTUAL_MINUS_OPERATIONAL_FORECAST` with decimal
  precision 34 and `ROUND_HALF_EVEN`;
- preserves `NULL` when either operand is missing;
- canonicalizes negative zero to `0`;
- rejects non-finite/non-canonical/exponent values, excessive scale or
  precision, implausible magnitude, duplicates, orphans and substitutions;
- rechecks series, zone, `MW`, native resolution, target interval and duration
  against D282;
- detects future evidence and mutation during execution;
- sorts output rows deterministically and content-addresses the full output.

The assessment and proof contain no operand or output value. They persist only
content hashes, counts and false authority flags. The output artifact exists
only in memory during the synthetic execution.

## Files changed

- `.planning/phases/14-lt-audit-remediation/ENTSOE-LT-DERIVED-ARITHMETIC-EXECUTION-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_lt_derived_arithmetic_execution.py`
- `tests/test_entsoe_lt_derived_arithmetic_execution.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `docs/research/forwards_sources.md`
- `.planning/HANDOFF.md`
- this handoff

Generated local-only materializer and proof:

- `build/d283_materialize.py`
- `build/databricks-eex-daily/2026-08-06/entsoe-lt-derived-arithmetic-execution-proofs/a2304c18e44f785a4f9e66bc200b0c50decb57e9f1646473ee0e25e218b10631/manifest.json`

## Exact identities

- contract raw SHA-256:
  `d0c2ce35c10c30d772f57cccc06c24377161bf8353b65eeadb751bec064370b8`
- contract canonical content ID:
  `92946ffd261fd2724c165e530387a13b12ed5d7b6f24a821f0bcf1c5cc3b5f08`
- algorithm content ID:
  `372ddf109142fc490210c0d09c319a6ecd46383efc348c34fdb760f416907de4`
- validator SHA-256:
  `4228c242073c3254c884f210acabd0d47a2d641643e1bccc877cd94fe7a4f247`
- tests SHA-256:
  `b8cd7dd9b6a9c466269593eee6a2ef01082e6bb886c27cdcdd976b843b30709c`
- materializer SHA-256:
  `16023554b6e9a6d075863f37a33739aaed50dcbfa8838153a78f5abca4775cc6`
- proof ID:
  `a2304c18e44f785a4f9e66bc200b0c50decb57e9f1646473ee0e25e218b10631`
- assessment content ID:
  `217732da8aa26ece2a578985256e3e7020474282b932200c5bac3c1836f32b1f`
- output artifact content ID:
  `6c715e1c01cf4f7e122694ff8d793391565f99c84f168df55f294c1cefdc7232`

## Verification

- focused D283 mutation roast: `30 passed`;
- D280-D283 predecessor chain: `88 passed`;
- adjacent D244/D245/D243/D253/D254/D255/D261/D270/D272/D273/D280/D281/
  D282/D283 matrix: `269 passed`;
- all current `tests/test_entsoe_*.py`: `504 passed`;
- Ruff check and format check: pass;
- deterministic independent materializer replay count: `2`.

Mutation coverage includes operand sign/order, null propagation, negative zero,
binary floats, exponent/non-canonical decimals, scale, significant digits,
magnitude, series/zone/unit/cadence drift, target mismatch, missing/duplicate/
orphan rows, D282 detachment, D282 value/authority escalation, future evidence,
mid-execution mutation, contract mutation and duplicate JSON keys.

## Execution and authority boundary

- Databricks connections/statements/writes: `0/0/0`.
- Warehouse starts: `0`.
- Network calls: `0`.
- `H:` accesses: `0`.
- Real values opened/persisted: `0/0`.
- Synthetic operand/output values persisted in proof: `0/0`.
- No CT module was imported or changed.

Real source, real arithmetic, PIT, predictive value, model input, model
selection, candidate, promotion and production authorities remain false. The
monthly solver remains sole CH monthly-level authority. Any later admitted
forecast-error contribution must be zero-mean within the solver month.

## Remaining blockers

- No governed real D244/D245/D270 package has passed D280-D283.
- The data engineer still must provide exact family, zone/EIC, cadence, unit,
  sign, revision, provenance and PIT evidence.
- D283 is an in-memory synthetic executor, not yet a governed Parquet adapter
  for real operands.
- Predictive usefulness remains unknown until a new independently frozen
  rolling-origin holdout exists. T057 remains sealed.

## Next safe batch

Do not add more synthetic feature authority. Either:

1. receive and locally admit the immutable data-engineer package through
   D244-D283; or
2. if the package is not yet available, define a D284 read-only Parquet adapter
   preflight that binds exact columns and row grain to D283 but opens only
   synthetic fixtures and grants no real/model authority.

OMPEX remains a post-freeze benchmark only and `H:` must not be accessed from
this workstation workflow.
