# Session handoff - ENTSO-E and LSEG cost-optimization audit

Date: 2026-08-21

## Scope

Deep, read-only audit of the data engineer's current DEV heads before PROD:

- `FMVSA/opendata-lakehouse`
  `db3a93316cd431a95b4e096d8482e482fda3491e`
- `FMVSA/lseg-lakehouse`
  `c81a41237d27101564b13959eede05ee3460b9fa`

No Databricks SQL, Warehouse, job or schedule was started. The review itself
generated no Databricks compute cost.

## Local evidence inspected

- ENTSO-E checkout:
  `build/data-engineer-repos/opendata-lakehouse`
- LSEG checkout:
  `build/data-engineer-repos/lseg-lakehouse`
- Both checkouts were clean and detached at the exact audited commits.
- GitHub branch heads were checked: the current `dev` heads matched the two
  hashes above.
- `git diff --check` was clean for both reviewed commit ranges.

## ENTSO-E findings

The active Gold design intentionally removes the duplicate
`FactEntsoeTimeSeriesVintages` serving fact while preserving the full canonical
history in `silver.ge_power_entsoe_time_series_vintages`. Active Gold is now:

- `DimEntsoeSeries`;
- `FactEntsoeTimeSeriesLatest`;
- `BridgeEntsoeSeriesResources`.

The bridge now represents the current `SeriesID x resource` mapping. It no
longer represents `VintageID x resource`. It remains useful for asset-level
outage enrichment, equipment identity, nominal power, lineage and QA, but it
is not the primary PFC fact and cannot answer historical as-of resource-state
questions. Those questions must use Silver `resource_details`.

Commit `db3a933` correctly drops the legacy vintage-to-dimension foreign key
before rebuilding the Gold dimension constraint. The check is conditional and
the operation is idempotent. No functional blocker was found.

The broader incremental Gold implementation:

- scopes normal Gold work from the upstream Silver run-start timestamp;
- avoids writes when only the load timestamp changed;
- incrementally refreshes bridge rows for affected series;
- performs one full bridge rebuild when it detects the legacy `VintageID`
  schema;
- keeps historical cutoff checks in backfill validation, not daily
  operational validation.

Residual ENTSO-E cautions:

- A standalone incremental Gold run without an upstream marker falls back to
  an unbounded Silver read; pass `since_date` for manual bounded execution.
- Once migration and consumer checks are complete, the stale legacy Gold
  vintage table should be removed to avoid ambiguity and storage cost. Silver
  vintages must remain.
- Repository CI does not replace a real Spark/Delta migration observation.

CI observed at the audited head:

- deploy workflow: success, run `32388695945`;
- unit tests and YAML lint: success, run `32388695946`;
- unit test result: `315 tests` passed.

## LSEG findings

Commit `c81a412` is repository hygiene only: it ignores all local
`.databricks/` bundle state. No tracked file was hidden and no data semantics
changed.

The preceding DEV chain contains the material cost and correctness changes:

- fail-closed incremental Gold bootstrap with an explicit or previous-success
  watermark instead of a `1900-01-01` fallback;
- collision-safe temporary view names;
- corrected Gold validation aliases/contracts;
- Silver Latest updates from Delta Change Data Feed rather than rereading all
  historical vintages for every touched key;
- targeted full-history fallback only for keys whose winner ordering regresses;
- final DEV runtime on DBR 18 Standard, not Photon.

No blocking correctness defect was found in the CDF/current-winner algorithm.
The first full run must establish a successful Gold watermark before ordinary
incremental execution.

CI observed at the audited head:

- deploy workflow: success, run `32406145004`;
- lint/unit workflow: success, run `32406144967`;
- pytest result: `59 passed`.

## PROD-readiness verdict

Code verdict: `GO_FOR_CONTROLLED_DEV_OBSERVATION`, followed by conditional
PROD if the following real-run evidence is green.

ENTSO-E evidence:

- first migration run drops the legacy FK and reports one bridge full refresh;
- next run is incremental, uses the upstream Silver marker and does not repeat
  the full bridge refresh;
- daily validation is green and row counts/duration/DBUs are bounded;
- Silver vintage history and `resource_details` remain intact;
- no consumer still depends on the legacy Gold vintage fact before it is
  dropped.

LSEG evidence:

- CDF is enabled and normal candidate scope is
  `durable_vintage_cdf_plus_current_latest`;
- regression fallback is normally false and, if triggered, is restricted to
  the affected keys;
- a successful Gold watermark exists and the next run uses
  `previous_success`;
- a replay does not rewrite unchanged rows;
- observed DBUs/runtime are lower or at least bounded relative to the former
  implementation.

Operational asymmetry to make explicit:

- ENTSO-E PROD schedules are currently paused.
- LSEG PROD schedules are currently unpaused. Deploying its PROD bundle will
  activate the scheduled jobs unless this is deliberately changed or accepted.

## Durable decision

Recorded as `D-20260821-248` in `DECISION-LOG.md`.

## Model-governance status

This source-code audit does not admit Databricks data, does not authorize the
local export, does not unseal T057 and does not clear
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.
