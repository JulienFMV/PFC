# Session handoff - ENTSO-E physical-series semantic binding

Date: 2026-08-06  
Decision: D-20260806-272  
Scope: LT ENTSO-E integrity only; offline and value-free

## Outcome

D272 closes the missing exact join between the physical `SeriesID` carried by
the normalized ENTSO-E package and the aggregate raw semantic signatures
inventoried by D243 and mapped to logical families by D253. Every base series
must be bound exactly once. Per-signature physical cardinality must equal D243
`series_count`, and normalized family/field/unit/current-resolution/directional
border semantics must reconcile with D253.

This is structural evidence only. Temporal zone binding, effective cadence,
quality, PIT availability, model input, candidate assembly and production
authorities all remain false.

## Changed files

- `pfc_shaping/validation/entsoe_series_semantic_binding.py`
  - exact D239 dimension validation and content identity;
  - D243/D253 validation and immutable input binding;
  - exact raw-signature recomputation;
  - complete physical-series coverage and cardinality reconciliation;
  - logical family, field, unit, current-resolution and CH-border checks;
  - zero Databricks/network/write/value-row counters and zero authorities.
- `tests/test_entsoe_series_semantic_binding.py`
  - success, input drift, missing/duplicate series, signature drift,
    cardinality, disposition, family, direction, unit, resolution, evidence,
    time ordering, authority escalation and template tests.
- `docs/data/templates/entsoe_series_semantic_binding.v1.json.template`
  - value-free engineer handoff template; placeholders are non-admissible.
- `docs/data/templates/README-ENTSOE-ZONE-BINDING.md`
  - documents semantic binding before temporal zone binding.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
  - requests the exact raw tuple per `SeriesID` in the existing bounded
    dimension stream, without extra SQL.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - final D-20260806-272 decision.
- `.planning/HANDOFF.md`
  - latest-handoff pointer and D272 summary.

## Verification

Commands were run only from the canonical workspace through the governed
workspace-local runner:

- `python -B -m scripts.run_workspace_local --run-id entsem272b python -B -m pytest tests/test_entsoe_series_semantic_binding.py tests/test_entsoe_series_family_mapping.py tests/test_entsoe_series_inventory.py tests/test_entsoe_series_zone_binding.py -q`
  - `61 passed in 0.47s`.
- `python -B -m scripts.run_workspace_local --run-id entsrf272b python -B -m ruff check pfc_shaping/validation/entsoe_series_semantic_binding.py tests/test_entsoe_series_semantic_binding.py`
  - `All checks passed!`.
- full `test_*entsoe*.py` pass under run ID `entall272`:
  - pytest output: `555 passed in 23.89s`;
  - supervisor receipt: `TARGET_EVIDENCE_INVALID`, target exit `0`, because
    repository execution identity changed during the run; this functional
    result is not stable execution authority.
- repeat under `entall272b` after concurrent D273 files appeared:
  - `574 passed`, one failure in the new concurrent
    `test_entsoe_lt_feature_availability.py`; the failure is outside D272 and
    must be closed by D273 before citing a complete current ENTSO-E matrix.

## Hashes

- validator:
  `638e454f2fba355d66b043841c505e9b3f76f54a4e800367fd370ec3b941f096`
- tests:
  `ffc1c7bc1f0b06bfefd0622fd99a21e07f511ceb789d6aa40beccc954b04a52c`
- binding template:
  `9cce2bbc02612797bb46bdfe1664f104ccfe485fe104fe299df9c327aaaefd94`
- template README:
  `a371010abbac4b7b9ffba162980f748baa42661fad185062d23e2e860322925a`
- engineer request:
  `abf76d603c645ca89636e88ca92a7d0cdba2aadb9755d6e13bb66c6062d6cfe4`

## Cost and access receipt

- Databricks requests / SQL statements / Warehouse starts / writes: `0`;
- ENTSO-E real value rows opened: `0`;
- network requests: `0`;
- `H:` access: `0`;
- same-day Databricks retry: not attempted and not authorized.

## Incident to retain

A local diagnostic intended to locate the workspace-run receipt used an
over-broad `rg` over `build/`. It surfaced restricted AFRY-derived lines in
the private tool output before timing out. No AFRY value was copied into Git,
documentation, a user response, network traffic, RAG or model code, and no
restricted file was changed. Do not repeat this command; inspect only the
exact `build/workspace-local-runs/<run-id>/execution-receipt.json` path.

## Next safe action

Close D273's concurrent feature-availability failure, then compose the
point-in-time feature roles only from a D272-bound physical series and the
independent D254/D255, D260/D261/D270 and quality evidence. Do not use real
values until a governed local package and independent future holdout exist.

