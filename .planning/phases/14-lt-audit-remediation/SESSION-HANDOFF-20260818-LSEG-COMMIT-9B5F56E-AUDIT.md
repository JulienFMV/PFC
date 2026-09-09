# Session handoff — LSEG commit `9b5f56e` audit

Date: 2026-08-18  
Scope: read-only audit of `FMVSA/lseg-lakehouse` DEV; no Databricks SQL,
notebook run, schedule start or table mutation.

## Exact source reviewed

- Local governed clone:
  `build/data-engineer-repos/lseg-lakehouse`
- Requested and checked-out commit:
  `9b5f56ed3b031d503597d0120470a6fda2451f91`
- Parent:
  `9e55899822235edb89c9aea4b7cdbc51f528a281`
- History also reviewed since the previous LSEG audit at `5bb89a6`:
  `34ad183`, `1ae3ca0`, `9e55899`, `9b5f56e`.
- Worktree state during review: clean, detached at the requested commit.
- `git diff --check 9e55899..9b5f56e`: clean.

## Verdict

`CODE_GO_FOR_DEV_REBUILD_AND_RUNTIME_VALIDATION`.

No blocking functional defect was found in the requested commit. It fixes the
two concrete Silver bootstrap/replay failures:

1. the vintage staging projection no longer emits
   `known_at_timestamp` twice;
2. fresh Silver tables declare all semantic timestamp columns, while existing
   tables are upgraded before either `MERGE`;
3. the Silver latest-value `MERGE` uses a null-safe payload-change predicate,
   excluding audit timestamps, so an identical replay is a no-op.

The commit is not by itself production evidence. Promotion still requires a
successful real DEV rebuild plus the fail-closed post-backfill validation
receipt for the rebuilt tables.

## CI and deployment evidence

- GitHub Actions run `32040183811`, commit `9b5f56e`: success.
- `yamllint`: success.
- `pytest`: `44 passed in 0.31s`.
- DEV deployment job `32040183996`: success.
- These tests are primarily Python/static notebook guardrails; they do not
  execute Spark/Delta `ALTER TABLE`, `MERGE`, or the full Databricks rebuild.
- Non-functional CI warning only: `actions/checkout@v4` is forced from the
  deprecated Node.js 20 runtime to Node.js 24 by GitHub.

## Runtime acceptance evidence still required

Retain the actual DEV job/run identifiers and the validation output proving:

- Bronze, Silver latest, Silver vintages and Gold complete successfully;
- no duplicate vintage ID or latest business key;
- forecast issue timestamps are present only for forecast curves;
- vendor last-update metadata is non-null for configured actual and forecast
  families;
- `pipeline_first_seen_at_utc <= pull_ts_utc` and deprecated aliases agree;
- Bronze-to-Silver and Silver-to-Gold row counts and contract hashes reconcile;
- a second identical incremental replay reports no matched-row update caused
  only by `_silver_updated_ts`.

## Operational cost/restart risk

The one-time historical notebook defaults to 2022-01-01 through 2026-08-11
and plans approximately 1,515 sequential pull/landing chunks:

- continuous forwards: 5 measures x 241 seven-day windows = 1,205;
- PMT spot forecasts: 5 measures x 55 31-day windows = 275;
- EPEX actuals: 7 measures x 5 365-day windows = 35.

It prints the plan but has no explicit `dry_run`, checkpoint, or automatic
resume cursor. Do not restart the full default backfill blindly after a partial
failure. Resume with bounded dates/groups based on the last successful
run/landing receipt, or add a governed resume control before another expensive
full replay. DEV/staging schedules are paused in the reviewed bundle; PROD is
configured unpaused.

## Durable semantic decision

- `forecast_issued_at_utc`: vendor forecast issue/vintage axis; null for actual
  curves.
- `vendor_last_updated_at_utc`: vendor metadata and diagnostic revision
  provenance, not proof that FMV possessed the value at that timestamp.
- `pipeline_first_seen_at_utc`: earliest platform receipt retained for the
  deterministic vintage; this is the usable local as-of boundary.
- `known_at_timestamp`: deprecated compatibility alias only.

Do not give `vendor_last_updated_at_utc` point-in-time model authority.

## Commands/evidence collection

All shell commands first validated the canonical root
`C:\Users\jbattaglia\PFC_LT`. Read-only commands included `git show`, `git
diff`, `git log`, `rg`, `gh api` for commit checks, and `gh run view` for the
successful CI log. No project interpreter, Databricks warehouse, vendor pull,
or notebook was launched locally.
