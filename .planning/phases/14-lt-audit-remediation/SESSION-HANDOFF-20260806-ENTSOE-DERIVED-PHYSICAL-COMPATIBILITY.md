# Session handoff - ENTSO-E derived physical compatibility

Date: 2026-08-06  
Decision: D-20260806-282  
Status: synthetic physical-compatibility PASS; model-input authority remains false

## Outcome

D282 composes exact D280 physical selections with exact D281 lagged load
forecast-error lineage. Each realized-load and operational-forecast primitive
must join exactly once to a distinct physical series, its D272 semantic
signature, D253 canonical unit and D255 temporal-zone interval.

The synthetic proof establishes one compatible transform:

- exact formula remains `REALIZED_ACTUAL_MINUS_OPERATIONAL_FORECAST`;
- actual and forecast both resolve to one common logical-zone hash;
- both canonical units are `MW`;
- both effective native resolutions are `PT1H` for the roasted interval;
- the target duration is 3,600 seconds and no resampling is performed;
- all clear series/feature identifiers and all values remain absent.

D280 enters composition with exactly
`DERIVED_TRANSFORM_LINEAGE_REQUIRED`. D281 supplies the exact lineage. D282
discharges that blocker structurally only; derived model-input authority stays
false.

## Files changed

- `.planning/phases/14-lt-audit-remediation/ENTSOE-LT-DERIVED-PHYSICAL-COMPATIBILITY-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_lt_derived_physical_compatibility.py`
- `tests/test_entsoe_lt_derived_physical_compatibility.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `docs/research/forwards_sources.md`
- `.planning/HANDOFF.md`
- this handoff

Generated local-only materializer and proof:

- `build/d282_materialize.py`
- `build/databricks-eex-daily/2026-08-06/entsoe-lt-derived-physical-compatibility-proofs/5961fba69241f0de74d14d3d0db062c03182eeb9b4748a2540f6553c0b9d4092/manifest.json`

## Exact identities

- contract raw SHA-256:
  `81158dedb6aaefcfe895590b7a0ac92d94a68c5fa7f4bda5f89f2ae7a163a04d`
- contract canonical content ID:
  `2976562eb360f93da5135d3f542e57aebd0848a9ee692654bab0477b67d9985e`
- validator SHA-256:
  `7f167bd536b08667f7a63dc798d561ab2790a4a711360ee704bc58b8ff01862e`
- tests SHA-256:
  `dcfe7bafbd7b6952c12eee4c168dca9724ca5b2d295d43e3c14f5c64fd03b611`
- materializer SHA-256:
  `a4b6d760b16065202af890a3f62cb30ff35225d54827c6b06ae417680e2504ba`
- proof ID:
  `5961fba69241f0de74d14d3d0db062c03182eeb9b4748a2540f6553c0b9d4092`
- assessment content ID:
  `93968ee86e2351d592cc221341478593e6bd023a181a2a55035371c92cdc5842`

## Verification

- focused D282 roast: `20 passed`.
- adjacent D244/D245/D243/D253/D254/D255/D261/D270/D272/D273/D280/D281/D282
  matrix: `239 passed`.
- Exact ENTSO-E contractual regression matrix through D282: `646 passed` in
  `54.54s`; stable status `TARGET_EXIT_ZERO_NOT_AUTHORITY`, target and runner
  exit codes `0`.
- Ruff check and format check: pass.
- deterministic replay count: `2`.

Mutation coverage includes logical-zone mismatch, non-MW unit, missing temporal
binding, unsupported cadence, orphan primitive, non-lineage D280 blocker,
formula drift, every composite content binding, predecessor authority
escalation, duplicate JSON keys and mutation during verification.

The first direct materializer attempt failed before import because the repo
root was absent from `sys.path`; no proof was created. Module execution then
created draft proofs before fixture cleanup and replay idempotence were fully
hardened. The local `b976...` and `7339...` drafts are superseded and must not
be cited. The final materializer uses deterministic UUIDs, exhausts fixture
generators so they clean normally, and two separate runs both adopt the exact
`5961...` proof.

## Execution and authority boundary

- Databricks connections/statements/writes: `0/0/0`.
- Warehouse starts: `0`.
- Network calls: `0`.
- `H:` accesses: `0`.
- Real or feature values opened/recomputed/persisted: `0/0/0`.
- No CT module was imported or changed.

Real source, official zone registry, PIT, arithmetic correctness, predictive
value, model input, model selection, candidate, promotion and production
authorities remain false. The data-quality method shaped the exact grain,
cardinality, temporal-window, join-integrity, unit-consistency and leakage
checks.

## Remaining blockers

- No governed real D244/D245/D270 package has passed D280-D282.
- D282 does not recompute `actual - forecast`; arithmetic evidence requires
  governed values and a separate value-integrity gate.
- Predictive usefulness remains unknown until an independently frozen
  rolling-origin holdout exists. T057 remains sealed.

## Next safe batch

Define D283 as a value-free arithmetic-execution contract: exact operand
artifacts, null propagation, sign convention, deterministic implementation and
output hash. It must remain synthetic-only and must not query Databricks. It
prepares later real arithmetic validation without admitting the feature or
using the sealed holdout.
