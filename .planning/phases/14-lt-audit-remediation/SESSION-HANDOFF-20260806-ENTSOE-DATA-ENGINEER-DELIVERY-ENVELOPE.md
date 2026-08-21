# Session handoff - ENTSO-E data-engineer delivery envelope

Date: 2026-08-06  
Decision: D-20260806-285  
Status: local value-blind envelope integrity PASS; data and model authority remain false

## Outcome

D285 closes the inventory gap between the data-engineer request and D244-D284.
The existing gates cover the normalized dimension/latest/vintage core and
derived operands, but the request also requires cadence history, zone/EIC
history, gap evidence, family inventory and exact source reconciliation.

The delivery envelope now requires a flat directory named by its derived
`snapshot_id` with ten mandatory artifacts:

1. `series_dimension.parquet`
2. `series_resolution_history.parquet`
3. `zone_history.parquet`
4. `latest_values.parquet`
5. `vintage_values.parquet`
6. `quality_summary.json`
7. `series_quality.parquet`
8. `family_inventory.json`
9. `gap_report.parquet`
10. `source_reconciliation.parquet`

`excluded_series.parquet` is the eleventh artifact exactly when the declared
excluded-series count is positive. A zero count requires the file to be absent;
a positive count requires its logical record count to match exactly.

The manifest binds exact artifact order, fixed sibling names, media types,
SHA-256, byte sizes, logical record counts, schema content IDs and exact
timestamp-bound fields by role. Its `snapshot_id` is the canonical SHA-256 of
the manifest without `snapshot_id`.

The verifier:

- accepts packages only below repo-local `build/`;
- rejects links/reparse points, hardlinks, traversal and unexpected inventory;
- hashes all artifacts in bounded chunks, then repeats the full-package pass;
- rechecks the manifest and inventory after both passes;
- never decodes a Parquet row or JSON artifact;
- grants only `local_delivery_envelope_integrity`.

## Files changed

- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-DELIVERY-ENVELOPE-CONTRACT-V1.json`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
- `pfc_shaping/validation/entsoe_data_engineer_delivery_envelope.py`
- `tests/test_entsoe_data_engineer_delivery_envelope.py`
- `build/d285_materialize.py` (local-only materializer)
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `docs/research/forwards_sources.md`
- `.planning/HANDOFF.md`
- this handoff

Generated local-only proof:

- `build/databricks-eex-daily/2026-08-06/entsoe-data-engineer-delivery-envelope-proofs/9b0f83b111eab55eb65e77fd340a38b248c882090ab5bbbbe129f58309839f84/manifest.json`

## Exact identities

- contract raw SHA-256:
  `231fcda0a3bf0a8257571275ea5a5debfadfb5651edc94920b6a9ddc7bb9fcb4`
- contract canonical content ID:
  `fdad6cecd95b3714521949b0dd2663e616b03f968f9cf6e5f033ec5237e00ea6`
- data-engineer request SHA-256:
  `70d836da1bfe4f830c168be55da3137fb348c7d6c02800a497f955413f19996e`
- validator SHA-256:
  `010f69a5ce3934898d16bf9b4c4dbeaf0e07ff4339a73a88bf1b02f403d4f554`
- tests SHA-256:
  `4d5e2b515c6877df2eafbe84803d0bc7a91d9b8115146cd5cf3bb99443f4117e`
- materializer SHA-256:
  `9566dee78748bfa8366ebfa91e51f827c85bf8c578b705286d16d2b000883f7e`
- proof ID:
  `9b0f83b111eab55eb65e77fd340a38b248c882090ab5bbbbe129f58309839f84`
- proof raw SHA-256:
  `7671995e2d31803460a47d8777fa0ce816ba627064a6398d5ea1ae3cf0b7532f`
- assessment content ID:
  `f1de7175487c9c6787fa830b55f311d1eca5a76e9b1bd89bba681c0b1640e32d`

## Verification

- focused D285 mutation roast: `34 passed`;
- adjacent D244-D285 matrix: `376 passed`;
- all current `tests/test_entsoe_*.py`: `563 passed`;
- Ruff check and format check: pass;
- two independent materializer executions produced the same proof ID;
- proof canonical hash equals its content-addressed directory name;
- only local envelope integrity is true; all execution counters are zero;
- fixture bytes and temporary paths are absent from the proof.

Mutation coverage includes missing/extra files, wrong directory, relative path,
hardlink, duplicate JSON keys, source-table/catalog drift, future manifest,
authority escalation, artifact order, traversal, missing/extra/reordered
timestamp bounds, one-sided/inverted/noncanonical timestamps, exclusion
bijection/count mismatch, byte tampering, cross-pass change and final manifest
or inventory mutation.

Non-product issues during the batch:

- two initially guessed D244/D245 handoff filenames did not exist; the actual
  `*-PREFLIGHT.md` and `*-PROFILE.md` files were located and read;
- one broad D285 search timed out and was replaced with a bounded search over
  the decision log and relevant directories;
- one UTF-8 documentation patch used a mojibake context and did not apply; the
  file was reread explicitly as UTF-8 and patched with the exact text.

## Execution and authority boundary

- Databricks connections/statements/writes: `0/0/0`.
- Warehouse starts: `0`.
- Network calls: `0`.
- `H:` accesses: `0`.
- Decoded Parquet rows/JSON artifacts/real value rows: `0/0/0`.
- Remote writes: `0`.
- No CT, Power BI, AFRY, OMPEX or heavy desk-data file was opened or changed.

Local envelope integrity is not source authenticity. Schema correctness, data
quality, real PIT, predictive value, model input, selection, candidate,
promotion and production remain false until the actual package passes its
independent downstream gates.

## Remaining blockers

- The real data-engineer package has not been delivered locally.
- D285 does not decode or validate artifact schemas/rows; D244-D284 remain
  mandatory and must stop at the first failed gate.
- Exact family, zone/EIC, effective cadence, sign/unit, revision, provenance,
  reconciliation and PIT evidence remain empirical requirements.
- Predictive usefulness still requires a new independently frozen holdout.
  T057 remains sealed.

## Next safe batch

When the package arrives, copy it below a governed repo-local `build/` intake
root and run D285 first. Only a D285 PASS permits schema/streaming validation;
it does not permit model input. Do not start or query a Databricks Warehouse to
create synthetic evidence or bypass a missing delivery.

OMPEX remains a post-freeze benchmark only and `H:` remains outside this
workstation workflow.
