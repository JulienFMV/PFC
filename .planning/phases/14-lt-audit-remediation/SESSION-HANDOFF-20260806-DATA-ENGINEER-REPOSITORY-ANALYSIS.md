# Session handoff — data-engineer repository analysis

Date: 2026-08-06  
Decision: D-20260806-294  
Status: repository analysis complete; Databricks content validation pending

## Outcome

The data-engineer request is now grounded in the actual lakehouse repositories
instead of an abstract export package. The data engineer publishes governed
Databricks tables; the PFC team later creates its own selective, read-only,
immutable local Parquet snapshot. Do not ask the engineer to write to `C:` and
do not rebuild every data source locally while Gold publication is pending.

The concise analysis and Jira text are in
`.planning/phases/14-lt-audit-remediation/DATA-ENGINEER-REPOSITORY-GAP-ANALYSIS-20260806.md`.

## Repositories inspected

All clones are below `build/data-engineer-repos/`, detached at the current
remote `dev` commit and clean:

- `epi-lakehouse`: `a91b2e17c903454d1d1ec420f0a1dbd1b49243cc`;
- `sdl-lakehouse`: `905a8abbff06caab2f9d9da1d22f2265616af2e9`;
- `opendata-lakehouse`: `10b25187f0b56473a70adcbdd675932c0838d27a`.

GitHub traffic only was used. Databricks connections, SQL statements, business
rows read, Warehouse starts and writes were all zero.

## Material findings

- EPI Euler spot is already hourly in Silver
  `ge_market_euler_spot`; current Gold `FactSpotPriceMonthly` destroys the
  interval shape. Add an interval Gold rather than creating a second source
  pipeline. PROD Euler file-arrival triggers are configured paused.
- SDL covers Swissgrid tender results only. It does not own Swissgrid balancing,
  NTC or cross-border flows. The Gold tender fact also drops source-snapshot
  time and original typed price components.
- OpenData `dev` implements 22 ENTSO-E request groups and the complete
  Landing/Bronze/Silver/Gold path. PROD remains paused. Resolution exists in
  Bronze but is not propagated to Silver/Gold. Gold also omits first-observed
  lineage, source series identity and the exact selected vintage link.
- The generic ENTSO-E grain `field_name x Date_Time_UTC` is unsafe for
  production-unit outages and installed-capacity-per-unit if several source
  series coexist at the same timestamp. Preserve the source series/unit or
  publish an explicitly governed aggregation.
- No weather or Swissgrid-balancing implementation exists in the three supplied
  repositories; obtain the actual repository/job links before specifying work.

## Verification

- 21 notebook files parse as valid JSON.
- `src/entsoe_pipeline.py` and `src/entsoe_validation.py` compile.
- All three cloned worktrees report zero dirty files.
- The ENTSO-E test file contains 7 test methods, but local discovery fails at
  import because its PySpark stub lacks `StructType`.
- CI in all three repositories runs YAML lint and bundle validation, not
  notebook/pipeline logic tests.

## Changed PFC files

- `.planning/phases/14-lt-audit-remediation/DATA-ENGINEER-REPOSITORY-GAP-ANALYSIS-20260806.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D294 prepended;
  older decisions untouched)
- `.planning/HANDOFF.md`
- this handoff

## Next action

Send the concise Jira comment from the repository analysis. Ask for the actual
weather and Swissgrid-balancing repository/job links. When the data engineer
responds or pushes changes, review the repository delta first; do not query
Databricks until a bounded cost preflight and explicit user GO exist.

