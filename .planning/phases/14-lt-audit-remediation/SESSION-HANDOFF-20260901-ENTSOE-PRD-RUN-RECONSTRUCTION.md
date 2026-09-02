# Session handoff - ENTSO-E PRD run reconstruction

Date: 2026-09-01

## Outcome

The ENTSO-E PROD rebuild relevant to `classification_sequence` was
reconstructed independently from GitHub deployment evidence, the PROD
`dq_ops_pipeline_runs` table and the existing value-blind July profile. No
message to the data engineer is required for the LT day-ahead profiling scope.

The evidence is sufficient to establish that the required code was deployed
before a successful full Bronze -> Silver -> Gold rebuild, and that the seven
current day-ahead series show zero legacy/new interval overlap. It is not a
platform-signed top-level Databricks job receipt and does not prove a separate
post-backfill validation notebook run across every ENTSO-E family.

## Deployment and ancestry

- GitHub PROD workflow run `32699692915` completed successfully on
  2026-08-24T07:01:45Z.
- The deployed SHA was
  `a7e920d95b94b2db59180412f31213f917e8d8a3`.
- `git merge-base --is-ancestor` returned zero for required commit
  `db3a93316cd431a95b4e096d8482e482fda3491e` against the deployed SHA.
- The workflow's bundle validation and bundle deployment steps both passed.
- Canonical workflow URL:
  `https://github.com/FMVSA/opendata-lakehouse/actions/runs/32699692915`.

## PROD full rebuild rows

The PROD audit table records the following successful `mode=full` sequence on
2026-08-24, after the deployment above:

- Bronze: run `a849d312-9909-4ef6-8720-9778d2178d16`,
  07:11:18Z -> 10:36:12Z, `SUCCESS`.
- Silver: run `e778fc04-79eb-4e98-8b47-fbd7863253e2`,
  11:43:46Z -> 12:13:45Z, `SUCCESS`, `dq_overall_status=PASSED`,
  `dq_failed_count=0`, 16,865,137 vintage rows and 16,824,887 latest rows.
- Gold: run `d591c620-4209-4d4c-b666-0c09eb89b5d5`,
  14:06:01Z -> 14:08:52Z, `SUCCESS`, `dq_overall_status=PASSED`,
  `dq_failed_count=0`.

## Day-ahead key-state evidence

The independently captured July profile contains exactly seven current series:
base keys for CH, FR and IT-North and classification sequences 1 and 2 for AT
and DE-LU. For every series it reports:

- `canonical_series_key_mismatch_count=0`;
- `legacy_new_overlap_interval_count=0`;
- `duplicate_vintage_key_count=0`;
- `gold_series_key_duplicate_count=0`;
- `latest_grain_duplicate_count=0`;
- `orphan_series_key_count=0`.

Sequences 1 and 2 are legitimate A44 auction identities, not old/new rebuild
duplicates. This proves the current July day-ahead row state needed by the LT
profile; it does not extend the claim to unrelated ENTSO-E families.

## Euler promotion state

A current paginated Unity Catalog inventory and exact table lookups found no
visible Euler- or spot-named table in PROD Bronze, Silver or Gold. In contrast,
`dev.bronze.euler_spot` and `dev.silver.ge_market_euler_spot` are both visible
managed tables. Direct Euler spot therefore remains DEV-only. The governed
PROD day-ahead candidate remains ENTSO-E Silver vintages, with LSEG as the
independent four-zone cross-check.

## Access boundary and cost

- The current token returns zero visible Databricks jobs/runs, so it cannot
  retrieve the top-level service-principal job receipt directly.
- It can read the PROD data-quality audit table and Unity Catalog metadata.
- Two read-only SQL statements ran on the already-running PBI SQL Warehouse:
  - broad audit-table pass: 200 rows, 1,688 ms, 4,142,455 remote bytes;
  - ENTSO-E-only pass: 63 rows, 586 ms, 4,142,455 cache bytes and zero remote
    bytes.
- Both statements finished successfully, wrote zero bytes and caused zero
  Warehouse starts.
- Thirteen current control-plane GETs established the Euler catalog state.

## Residual boundary

A separate platform-signed post-backfill validation receipt remains unavailable.
Require it only for formal producer-wide promotion evidence, not to continue
the bounded LT day-ahead work. Historical `createdDateTime` remains unsuitable
as original publication/PIT evidence, and A03 wide blocks must be interpreted
or expanded according to their source interval semantics.
