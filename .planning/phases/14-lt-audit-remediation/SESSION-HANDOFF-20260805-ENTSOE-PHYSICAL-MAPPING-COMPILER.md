# Session handoff - ENTSO-E physical mapping compiler

Date: 2026-08-05  
Decision: D-20260805-234  
Status: `PASS_LOCAL_DETERMINISTIC_PHYSICAL_MAPPING_FIXTURE_ONLY_NO_REAL_METADATA_NO_EXECUTION`

## Outcome

D234 implements the offline compiler between a real future D232 metadata
admission and the three normalized ENTSO-E deliverables specified by D233. It
binds the exact D233 governed acquisition package rather than superseding it.
No Databricks connector, SQL Warehouse, workspace request, remote write, market
value or restricted benchmark was opened.

The compiler accepts an exact owner-asserted mapping bound to metadata payload
hash, schema fingerprint and statement ID. It checks all dimension/latest/
vintage normalized fields, injective physical mappings, safe exact identifiers,
compatible type classes and exact type equality for shared cross-table fields.
UTC remains an owner assertion. Implicit casts, upsampling, forward-fill and
latest-as-PIT are rejected.

It compiles three SQL templates in memory with explicit projections and aliases,
fixed qualified tables, delimited identifiers, deterministic grain ordering,
fixed row limits, half-open target windows and a vintage `as_of` cutoff. The
proof persists only template hashes, parameter names and limits; it does not
persist the synthetic SQL text. Execution remains explicitly unauthorized.

## Decision-number reconciliation

The current worktree already contained canonical D233 for the governed
EEX/ENTSO-E package. The compiler was therefore assigned D234. D233 files and
evidence were preserved and bound into D234. No historical decision was
overwritten.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-PHYSICAL-MAPPING-COMPILER-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_physical_mapping_compiler.py`
- `tests/test_entsoe_physical_mapping_compiler.py`
- `build/databricks-eex-daily/materialize_entsoe_physical_mapping_compiler.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057 or heavy desk-data file was opened or changed
by D234.

## Canonical identities

- compiler contract raw SHA-256 / canonical content ID:
  `f33bc696d8e7a535f7700b40c13dc90d73e2526073400edcd2588e57432a9b3a` /
  `5dc43e18dcf107020e24638307a707d8c7afbf894ca1131e62907f8680ad0951`
- validator SHA-256:
  `2100c9d126edf063b663790993b75bb9a190dbdde1281ab8057e44133ed796a5`
- tests SHA-256:
  `8670c1e883356308454893aad5be7b2788702f63adb606901994acb8967becf8`
- materializer SHA-256:
  `bade17d6246805e3d5cd85f02ab4ff45a58aa1d2910e18d3bd10197bb0bd753a`
- research note SHA-256 after D234:
  `b8959263c41fe8527003d925822078fcea060801fd284638d1c28ff323407b47`

Reproducible proof:

- content ID:
  `bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766`
- manifest SHA-256:
  `7d86330e2143e2009947b486eb1e8428dd6022735ebc33c9aa4cfc4b871c5268`
- assessment SHA-256:
  `47a65e11c12e848b2c0cbc8d568e9f1b97e84996bcb92d52c9d242e507ffde9b`
- path:
  `build/databricks-eex-daily/2026-08-05/entsoe-physical-mapping-compiler-proofs/bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766/`

D234 source bindings include:

- D232 proof content ID / manifest SHA-256:
  `9a270975187e9ff334d80afba308ad0021f0df97a4269348943a62d655bc4147` /
  `197b5c89598b0328ce5bc3bdcb103e5a58adba089fe36d51df2921b0c0063420`
- D233 proof content ID / manifest SHA-256:
  `314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53` /
  `d7f6ad3af60d2efd087718e0515c8b73a6534387ab12c7c7a5f88f42aeadf2b4`
- ENTSO-E intake contract SHA-256:
  `7ede1698099390babfa1d130bfecae61fd1e090888a3d6e6e4f892119db52b87`

## Verification

- Ruff on compiler, tests and materializer: passed.
- focused D234: `47 passed in 0.22s`.
- combined D231-D234: `156 passed in 0.56s`.
- expanded acquisition/receipt/trust/PIT/ENTSO-E suite:
  `465 passed, 3 skipped, 1 warning in 130.46s`.
- materializer executed twice on reconciled D233-bound sources: identical D234
  proof content ID.
- proof counters: zero Databricks request, Warehouse start, network call, `H:`
  access and remote write.

The warning is the pre-existing timezone-to-period warning from
`ingest_energy_charts.py`; it is unrelated to D234.

## Public technical references

Databricks named parameter markers are supported by the Statement Execution API
and separate supplied values from SQL structure:

- `https://docs.databricks.com/gcp/en/sql/language-manual/sql-ref-parameter-marker`

`LIMIT` constrains returned rows and should be paired with deterministic order;
it is not treated here as a cost guarantee:

- `https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-qry-select-limit`

## Risks and next permitted step

The fixture proves compiler behaviour only. A real mapping owner, real metadata,
timestamp semantics, actual values, joins, null/duplicate rates, units,
resolution, freshness, gaps, revisions and PIT availability remain unverified.
Named parameters protect literal values but not arbitrary identifiers; D234
therefore accepts only exact admitted regex-safe identifiers and always delimits
them.

The next safe local batch can validate future execution parameters and capture
receipts (31-day maximum window, positive limits, hit-limit rejection and query
hash binding) without executing anything. Any actual metadata or data statement
still requires new explicit user authorization acknowledging possible Warehouse
cost. Training, selection, model input, candidate assembly, promotion and
production remain false; T057 stays sealed and the monthly solver remains sole
level authority.
