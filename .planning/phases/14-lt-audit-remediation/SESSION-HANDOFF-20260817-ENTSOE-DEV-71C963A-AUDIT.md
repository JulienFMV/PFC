# Session handoff — ENTSO-E DEV `71c963a` audit — 2026-08-17

## Scope and immutable evidence

- Repository reviewed: `FMVSA/opendata-lakehouse` in
  `build/data-engineer-repos/opendata-lakehouse`.
- Remote DEV head fetched and checked out detached at
  `71c963a06e76770362710aaddf8fc74432ade44d`
  (`feat(entsoe): harden operational validation for prod readiness`).
- Baseline: `d7dcaed788507196adaa6f9002439edc2d382e85`.
- Diff: 10 files, 1,660 insertions and 247 deletions.
- `git diff --check d7dcaed..71c963a` returned clean.
- The user reported that a Databricks DEV rebuild was already running. Codex
  did not start, stop, poll or query any Databricks job, SQL Warehouse or table.

## Confirmed corrections

- `dev_local` schedules are now `PAUSED`; only the governed DEV target remains
  scheduled.
- The independent 19:15 validation schedule is removed. The evening wrapper
  invokes validation only after `run_core` succeeds, eliminating the clock race.
- Operational validation uses a 26-hour bounded scope. It first selects
  successful landing-to-Bronze run IDs and otherwise applies a bounded
  `_ingest_ts` fallback.
- A03 dense coverage now merges `[interval_start_utc, interval_end_utc)` islands
  rather than treating one compressed block as one missing native slot.
- A78 is canonically named `transmission_new_ntc_mw`; historical old-suffix
  Bronze rows are normalized before Silver semantic identity is built.
- Gold Dim validation reuses the production dimension contract constants and
  includes the relevant semantic identity fields.
- Regression tests were added for interval merging, operational scope,
  scheduling, dimension semantics and A78 canonicalization.

## Material findings before PROD

### 1. Operational scope can include a rebuild/backfill run — high cost and semantic risk

`_resolve_operational_scope_from_ingestion()` selects every successful
`00_landing_to_bronze_entsoe` run whose `start_ts` is in the last 26 hours.
`dq_ingestion_runs` stores neither `mode`, `request_groups`, nor a parent job/run
identity. Consequently a successful rebuild/backfill can be selected by the
evening operational validation. This is directly relevant while the reported
DEV rebuild is in progress.

The selected `_run_id` then scopes Bronze, but the physical Bronze tables are
partitioned by value year/month, not by run ID. The validator contains 57
`assert_zero_df()` call sites, each triggering a Spark action, and does not cache
the scoped frames. Including a rebuild run can therefore make the supposedly
cheap operational check scan a very large scope repeatedly.

Required correction: persist execution context in the ingestion audit record
(`mode`, requested groups and preferably parent Databricks job/run identity),
and select only the intended incremental daily runs. At minimum, exclude
backfill/full/rebuild runs from `validation_mode=operational`. Validate the
actual query runtime and scanned bytes after the change.

### 2. A fully absent daily family can pass — high functional risk

In operational mode, `recent_expected_groups` is computed as the union of group
names that are already observed in scoped Bronze, Silver and Gold. If a family
is absent from every recent layer, it is absent from that union and all six
expected-group checks pass. The global Gold Dim check proves only that the
family existed historically, not that today's refresh delivered it.

Required correction: define the expected operational family set independently
of observed data. For the current daily wrappers it is the explicit union of
the 8 morning groups, the intraday-renewables group and the 7 evening groups
(16 groups). Weekly-only groups need a separate Saturday/readiness rule. Each
daily expected group must have a successful recent run and non-empty/reconciled
Bronze→Silver→Gold delivery, subject only to explicit publication-aware
`not_applicable` rules.

## Non-blocking test gaps

- The A03 parser fixture asserts curve type and row count but still does not
  assert the exact position-3/position-13 interval boundaries. The production
  parser implementation itself is correct: next position starts the next A03
  block and the final block ends at the Period end.
- Several A78 regression tests inspect source strings rather than executing a
  Spark transformation. The DEV rebuild is therefore important empirical
  evidence that old/new field names converge to one Silver/Gold identity.

## CI limitation

- The local review did not run Python tests because this managed workspace
  allows project tests only through its governed local runner and the reviewed
  external repo is not allowlisted.
- `gh run list` for the exact SHA returned GitHub API 404 despite an authenticated
  account with `repo` scope. Therefore no CI result is claimed for `71c963a`.
  The ongoing DEV rebuild proves deployment/runtime progress, but its final
  result and validation output remain required evidence.

## Verdict

The commit fixes the previously reported A03 coverage, A78 naming, validation
race, duplicate schedule and Gold Dim contract issues. It is a strong DEV
candidate, but not yet a PROD go: exclude rebuild/backfill runs from operational
scope and make daily expected families independent of the observed data. After
that, require one successful dependency-driven operational validation with its
scope metadata, group list, duration and scanned-byte/cost evidence.
