# Session handoff — Databricks DEV promotion audit (2026-08-10)

## Outcome

Deep read-only audit completed for the ENTSO-E and LSEG DEV lakehouses.
Decision D-20260810-298 keeps both sources at `NO_GO_PROD`.

The configured SQL Warehouse was `STOPPED`. No SQL statement was submitted,
no Warehouse was started and no Databricks write occurred. Content-level
freshness/gap validation therefore remains an external receipt requirement.

## Exact audited revisions

- `build/data-engineer-repos/opendata-lakehouse`:
  `55c3c436271d320b771d8a126b284249a878abdc` (`origin/dev`), clean detached
  checkout. `origin/main...origin/dev = 1 / 6` commits.
- `build/data-engineer-repos/lseg-lakehouse`:
  `ebc3f23ff0a7e62e65471d861e4be993f35fdde1` (`origin/dev`), clean detached
  checkout. `origin/main...origin/dev = 0 / 44` commits.
- No open PR and no GitHub release was found in either repository.

## Primary findings

1. ENTSO-E Bronze preserves `source_time_series_id`, `resolution`, document
   mRID/revision and raw lineage, but Silver/Gold collapse to
   `field_name x timestamp`; Gold `SeriesID` is derived only from
   `field_name`. This can silently overwrite multi-series unit/outage and
   parallel-category responses.
2. ENTSO-E Silver/Gold drop native resolution and interval start, retaining
   only a right-edge timestamp. The curated contract is insufficient for
   unambiguous hourly/15-minute and DST alignment.
3. ENTSO-E Gold vintages expose latest-observed publication/pull timestamps,
   not first-known time. `createdDateTime` is documented by ENTSO-E as the
   issuing system's document-generation time; its use as historical source
   availability remains unproved.
4. LSEG still parses only `forecastDate`. Vendor `Updated`/`Corrected`
   metadata is absent; `KnownAtTimestampUtc` derives from FMV ingest/pull.
5. ENTSO-E default dense validation now checks only actual load and hydro
   storage; it does not prove the internal completeness of the Swiss NTC and
   physical-flow series.
6. Both CI workflows lint YAML only. LSEG has no unit tests. The current
   ENTSO-E test module fails on import because its PySpark stub lacks
   `StructType`.
7. LSEG PROD is configured `UNPAUSED`; first release must be deployed paused
   until initial load/backfill validation is accepted.

Positive evidence is also material: ENTSO-E config contains 22 request groups
and all eight directed Swiss borders; LSEG contains Swiss continuous-forward
curve `110181967`; all 14 expected DEV Bronze/Silver/Gold tables are visible
and their columns align with the reviewed repo generation.

## Canonical local artifacts

- `build/databricks-dev-promotion-audit/2026-08-10/report.html`
- `build/databricks-dev-promotion-audit/2026-08-10/report-verification.json`
- `build/databricks-dev-promotion-audit/2026-08-10/artifact.json`
- `build/databricks-dev-promotion-audit/2026-08-10/findings.json`
- `build/databricks-dev-promotion-audit/2026-08-10/catalog-surface-dev.json`
- `build/databricks-dev-promotion-audit/2026-08-10/selected-table-schemas.json`
- `build/databricks-dev-promotion-audit/2026-08-10/control-plane-metadata.json`
- `build/databricks-dev-promotion-audit/2026-08-10/job-runs.json`

Key SHA-256 values:

- report HTML: `3f47c0ccad6623591258dbe528228c197207a0cb300a54188957eb89489bbd6f`;
- findings JSON: `b13a25e5fcef5fb7a4072504d922f6102f35075430e1ad006db450cbe45ea15f`;
- selected schema receipt: `d55666267d4fd55ccc4981d5d682392251121700bbfb6b2c1fff652892734715`;
- control-plane receipt: `c5eed84f1233b670d0d35895c50f3991134cfdf41a894052a6702006ce69f7db`.

The portable report build passed. Structural verification returned
`PASS_STRUCTURAL_ONLY` with 11 blocks, one chart, one metric strip and four
tables. Browser QA and static chart SVG extraction were intentionally not run
because the workstation contract prohibits Playwright/browser runtimes.

## Files created or changed in the canonical repo

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` — prepended D298.
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260810-DATABRICKS-DEV-PROMOTION-AUDIT.md`
- `build/databricks-dev-promotion-audit/capture_control_plane_metadata.py`
- `build/databricks-dev-promotion-audit/capture_job_runs.py`
- `build/databricks-dev-promotion-audit/verify_report_structure.mjs`
- all canonical artifacts listed above under the dated build directory.

No ENTSO-E/LSEG repository source was edited. Existing unrelated dirty-root
worktree changes were preserved.

## Commands and results

- Unity Catalog schema capture over 14 explicit tables: PASS, 14 control-plane
  GETs, zero SQL/write/start.
- Sanitized Warehouse/table metadata capture: PASS, Warehouse `STOPPED`, 15
  control-plane GETs, zero SQL/write/start.
- Sanitized Jobs API capture: current token sees zero jobs; this is an access
  limitation, not evidence that the jobs do not exist.
- GitHub branch, PR, release and workflow inspection: PASS. Recent ENTSO-E DEV
  deploy at `55c3c43` succeeded; LSEG latest DEV deploy at `ebc3f23` succeeded.
  These deploy workflows do not execute row-level validation.
- Python source and five notebooks per repo parsed/compiled: PASS
  (`opendata-lakehouse`: five Python files/five notebooks;
  `lseg-lakehouse`: one Python file/five notebooks).
- ENTSO-E unit test: FAIL during import with
  `AttributeError: module 'pyspark.sql.types' has no attribute 'StructType'`.
- Local DuckDB provenance queries over `findings.json`: PASS; generated exact
  metric, severity, finding, repository, positive-evidence and gate datasets.
- Portable report build: PASS. Structural embedded-artifact verification:
  PASS.

One attempted cleanup of audit-only `pycache`, pytest basetemp and `tmp`
directories was rejected by the tool policy before execution. They are below
`build/databricks-dev-promotion-audit/`, are non-authoritative and may be
removed later through an approved workspace-cleanup path.

## Required data-engineer evidence before promotion

1. Correct the ENTSO-E curated grain and interval contract.
2. Establish source-aware `KnownAtTimestampUtc` rules for ENTSO-E and LSEG;
   mark backfilled availability unknown where the source cannot prove it.
3. Run both `90_post_backfill_validation` notebooks with
   `fail_on_error=true`; provide the JSON outputs, exact commit, run IDs and
   Delta versions with zero failed checks.
4. Add explicit gap checks for NTC and physical flows on the eight directed CH
   borders at native resolution.
5. Repair/add CI tests, then open PRs to main, validate staging and tag the
   accepted commits.
6. Keep the first LSEG PROD schedules paused until initial production load and
   validation are explicitly accepted.

## Invariants

- CH EEX-constrained monthly solver remains the sole level authority.
- LSEG remains an independent benchmark candidate; it cannot rewrite monthly
  means.
- No modelling gate is lifted until governed real exports and an independent
  future holdout exist. T057 remains sealed.
- Local C: export is a later PFC-owned incremental immutable step, separate
  from data-engineer publication.
- Never start a stopped Databricks Warehouse for inspection without explicit
  authorization.
