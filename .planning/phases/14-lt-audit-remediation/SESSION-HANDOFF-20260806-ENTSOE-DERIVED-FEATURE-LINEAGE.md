# Session handoff - ENTSO-E derived feature lineage

Date: 2026-08-06  
Decision: D-20260806-281  
Status: synthetic lineage PASS; D280 derived model-input admission remains blocked

## Outcome

D281 adds a value-free lineage gate for lagged load forecast error. The exact
formula is `REALIZED_ACTUAL_MINUS_OPERATIONAL_FORECAST`, with positive output
when actual load exceeds the operational forecast.

The gate content-binds one output D273 `FORECAST_ERROR` record to the exact
`actual_load` and `load_forecast` primitive records. All three must share the
same target interval and origin. Output dependency timestamps must identify
those exact primitives, output availability must equal their latest
availability, and calculation cannot precede availability or follow lineage
declaration.

Missing, duplicate, reversed, self-referential or unbound dependencies fail
closed. Nulls remain null. Stable JSON reads reject duplicate keys.

## Deliberate authority boundary

D281 does not open values or recompute the subtraction. It does not prove that
the two physical series have the same unit, zone or effective cadence. Those
facts require a later composite with D280. Therefore D280 remains unchanged
and continues to report `DERIVED_TRANSFORM_LINEAGE_REQUIRED` for derived model
inputs.

Real source, PIT, physical semantics, arithmetic correctness, predictive value,
model input, candidate, promotion and production authorities remain false.

## Files changed

- `.planning/phases/14-lt-audit-remediation/ENTSOE-LT-DERIVED-FEATURE-LINEAGE-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_lt_derived_feature_lineage.py`
- `tests/test_entsoe_lt_derived_feature_lineage.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `docs/research/forwards_sources.md`
- `.planning/HANDOFF.md`
- this handoff

Generated local-only materializer:

- `build/d281_materialize.py`

## Exact identities

- contract raw SHA-256:
  `d9157ebce9aaf02950231c2bac21bb28a10ee8577b0450bbbbd6eb8b33447919`
- contract canonical content ID:
  `7f3fdde5c5d5f192cee06b570063173eb4a55344fe2f871a5441bf140cb8c2ac`
- validator SHA-256:
  `f7e77b8cb5809631ac6d59241f0ba0a06a3cfdafd6ced5cfcb6b389a3fea4613`
- tests SHA-256:
  `e1b9882bb1e73e647d31d3a0dc76074bf3b0b7caba2e0af7af3693c7bc0ea281`
- proof ID:
  `7bbeab342f8d687068ec7591c69ad5690f63196e1916dd0d8c68fcb2794b0495`
- assessment content ID:
  `39cc9af0e22ed25bcfa770e50a53ab17ec7960038f51d5446cbf903a273a6033`
- proof path:
  `build/databricks-eex-daily/2026-08-06/entsoe-lt-derived-feature-lineage-proofs/7bbeab342f8d687068ec7591c69ad5690f63196e1916dd0d8c68fcb2794b0495/manifest.json`

## Verification

- focused D281 roast: `19 passed`.
- adjacent D244/D245/D243/D253/D254/D255/D261/D270/D272/D273/D280/D281
  matrix: `219 passed`.
- Ruff check and format check: pass.
- deterministic replay count: `2`.

Mutation coverage includes operand order, source family, target and origin,
dependency timestamp identity, calculation/declaration chronology,
missing/duplicate/self dependencies, missing output lineage, formula,
missingness, sign convention, physical-authority escalation, content
detachment, rejected D273 inputs and duplicate JSON keys.

## Execution and safety

- Databricks connections/statements/writes: `0/0/0`.
- Warehouse starts: `0`.
- Network calls: `0`.
- `H:` accesses: `0`.
- Real feature values opened/recomputed/persisted: `0/0/0`.
- No CT module was imported or changed.

## Next safe batch

Define D282 as a synthetic-only composition between D280 physical evidence and
D281 transform lineage. It must prove that actual, forecast and error refer to
the same logical zone, unit and effective cadence before it can remove the
lineage blocker structurally. It must still grant no real-data, predictive or
model authority and must not query Databricks.
