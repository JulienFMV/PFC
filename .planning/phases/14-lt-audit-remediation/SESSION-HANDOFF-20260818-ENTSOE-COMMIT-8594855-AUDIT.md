# Session handoff — ENTSO-E commit `8594855` audit

Date: 2026-08-18  
Scope: read-only code, configuration and GitHub-check audit. No Databricks SQL,
warehouse start, notebook execution or schedule mutation was performed by Codex.

## Exact source reviewed

- Repository: `FMVSA/opendata-lakehouse`
- Governed local clone: `build/data-engineer-repos/opendata-lakehouse`
- Requested commit:
  `8594855b3bd962530cabfc954f69dd8e22a6b6da`
- Direct parent:
  `b5f10392ea76a674ed912b42c4a771042ba57167`
- Merge base with the previously audited `71c963a`:
  `71c963a06e76770362710aaddf8fc74432ade44d`
- Commits newly activated since `71c963a`:
  - `52f9a85 chore: pause dev jobs`
  - `b5f1039 fix(entsoe): tighten operational validation scopes`
  - `8594855 chore: resume dev schedules`
- GitHub API confirmed the protected `dev` branch points to `8594855`. The
  local partial clone's `origin/dev` ref was stale at `71c963a`, so it was not
  used as branch-authority evidence.
- `git diff --check 71c963a..8594855`: clean.

## Verdict

`CODE_GO_FOR_ACTIVE_DEV_SCHEDULES_PENDING_RUNTIME_RECEIPTS`.

No blocking functional defect was found. The requested commit itself changes
only DEV `schedule_pause_status` from `PAUSED` to `UNPAUSED`; staging and PROD
remain paused. The important functional code is the intermediate `b5f1039`,
which was therefore audited as part of the activated state.

## Functional improvements activated by `b5f1039`

- Operational validation now uses an explicit contract rather than deriving
  expected groups from whichever groups happened to be observed.
- It requires 16 daily groups after the evening pipeline:
  8 morning, 1 intraday and 7 evening groups.
- It adds the 6 weekly reference groups to the Saturday-evening contract.
- Expected-group checks are restricted to the current Zurich business day, so
  yesterday's rows cannot mask a missed current schedule.
- Operational scope is built from successful Landing-to-Bronze runs and is
  restricted to `mode=incremental` using exact `run_id` linkage in both
  `dq_ingestion_runs` and `dq_ops_pipeline_runs`.
- Metadata lookup failure, absence of an incremental run, or an empty bounded
  Bronze scope fails closed through `operational_scope_integrity`.
- Silver latest, Silver vintages and Gold facts are scoped by their full
  production semantic keys, not by a lossy SeriesID-only shortcut.
- Historical hydro gaps are informational; the most recent 12 weekly periods
  are blocking in backfill validation using a fixed expected window.
- DST conversion uses `Europe/Zurich` and has explicit summer, winter and DST
  transition tests.

## GitHub evidence

- Commit `8594855`: unit-tests success, YAML lint success, DEV deploy success.
- Unit test log: `Ran 296 tests in 0.490s` / `OK`.
- Deployment run: `32068604721`, completed successfully.
- YAML annotations are non-blocking line-length warnings in existing workflow
  and jobs files. GitHub also reports the Node 20 action-runtime deprecation.
- Tests cover pure semantics and static wiring but do not execute the complete
  Spark/Delta validation notebook on rebuilt DEV tables.

## Runtime acceptance still required

Collect the real DEV run IDs and validation receipt proving:

- morning, intraday and evening wrapper jobs complete successfully;
- the dependency-driven validation runs only after evening Gold succeeds;
- `operational_scope_integrity` passes with `scope_method=ingestion_run_ids`,
  metadata lookup true, at least one incremental run and a non-empty Bronze
  scope;
- all 16 daily groups appear through Bronze, Silver and Gold within the current
  Zurich business day;
- on the next Saturday, all 6 weekly groups are additionally present;
- all blocking checks are zero and the result exits successfully with
  `fail_on_error=true`.

## Cost and monitoring notes

`UNPAUSED` has a real recurring DEV cost. The reviewed schedule declares about
233 base API requests per day (155 morning + 15 intraday + 63 evening), plus
129 weekly requests. Compute normally means four job-cluster activations per
day (three core runs plus the separate validation job), plus the weekly core
run. Monitor actual duration/DBU after the first complete day and week; do not
run additional manual copies merely for audit evidence.

The weekly required-group contract is intentionally enforced only on Saturday
after 07:00 local. From Sunday through Friday it relies on the weekly wrapper's
failure notification rather than reasserting weekly freshness every evening.
This is acceptable for DEV but is a documented monitoring choice to revisit
before PROD if persistent weekly-staleness detection is required.

## Commands and mutations

All shell commands first validated the canonical workspace root
`C:\Users\jbattaglia\PFC_LT`. Evidence used `git show`, `git diff`, `git log`,
`rg`, `gh api` and `gh run view`. The governed clone was detached at the exact
commit for inspection. No Databricks or ENTSO-E request was launched.
