# Session handoff - ENTSO-E July candidate capture - 2026-09-04

## Outcome

July can advance independently of the current September CH source gap. After
the user explicitly authorized using the PBI Warehouse and proceeding with the
historical month, one normalized value-blind check proved exact coverage of
the frozen Swiss-local July scope:

- window: `[2026-06-30T22:00:00Z, 2026-07-31T22:00:00Z)`;
- CH, AT sequence 1, DE-LU sequence 1, FR and IT-North each expand to exactly
  2,976 quarter-hours;
- missing, overlapping and invalid native intervals: zero for every series;
- coverage statement read 65,771,005 bytes and returned five metadata rows.

The exact v2 statement then captured one local latest-revision candidate:

- 12,083 native rows and 21 contract columns;
- one 6,653,296-byte Arrow result download;
- 564,727-byte local Parquet;
- Parquet SHA-256:
  `046ae86ab84a72c61ea44547cc386f85e49d37d325352f4bfca208a08e5a9baa`;
- semantic SHA-256:
  `5a6d72b5531e843dc4e7920c519cc3d93189e70d91277662a763f6575967374e`.

The complete local validator accepts the SQL hash, selected keys, window,
quality and interval contract, then stops exactly at
`evidence does not exactly cover selected SeriesKeys`. The bytes are therefore
a quarantined latest-revision candidate, not yet `realized_final`. No price
was printed, committed or transmitted.

## Frozen non-value evidence

- path:
  `.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-JULY-CANDIDATE-CAPTURE-V1-20260904.json`;
- file bytes/SHA-256:
  `10015` / `6ca8665ff7a378d95c6657444e66b8e1bc95e96cc82f7c54345dd5a3990e9b51`;
- canonical JSON bytes/SHA-256:
  `8396` / `ec937b823551e301dc5b0c553301b2b21b1584c6b2c6a166d7803dab4206c56d`;
- status:
  `PASS_QUARANTINED_LATEST_REVISION_CANDIDATE_PENDING_VALUE_BOUND_FINALITY`.

The local value-bearing artifacts remain ignored below `build/`:

- `build/entsoe-july-candidate-20260904/latest-revision-candidate.parquet`;
- `build/entsoe-july-candidate-20260904/manifest.json`, SHA-256
  `5f02aa423fcdb27d8b60b6541f5024bf48fb326dbb6047205a6c6f29a6ea571e`,
  content ID
  `7f877d0c32ad77406fd8b50fd1e98e8554603ae34b1deaea7ca69d8880117639`;
- `build/entsoe-july-candidate-20260904/local-validation.json`, SHA-256
  `9df64223c610ebe0c4bed35131b9662d63c0eace12c01af2fecd732bbc9fd250`,
  content ID
  `3940c49b4331753fda82ebc26fb8dbb5cbe2797cfe1abb15f14c34770a2ebfcc`;
- `build/entsoe-july-candidate-20260904/query-history-metrics.json`, SHA-256
  `de279dd77664ad9b84a6174b2a521fa887d3eae0cb8b39401f6b0ea5fafa3302`.

Do not move the Parquet into Git or a publication surface.

## Statements and cost

### Historical v1 profile

- statement: `01f1a839-2d94-17af-a3d5-034530bd0fbc`;
- SQL SHA-256:
  `e48bc8b09d6f3676616ed42966d50a3f44d9eaf3649f9ce9c9543a0bc024259e`;
- seven metadata rows, no prices;
- 48,299,260 bytes read, 189,423,656 file bytes, 27 files, 8.124 seconds;
- status: `BLOCKED_ENTSOE_DAY_AHEAD_PRD_PROFILE`.

Its availability-order and exact-duration findings are not the current v2
interval admission gate. D267 already superseded exact-duration equality with
positive aligned integer-multiple normalized blocks. The normalized check
below found zero invalid blocks. The v1 result remains recorded, not promoted.

### Normalized exact coverage

- statement: `01f1a839-d542-1737-b4ff-0f1db847ad4f`;
- assessed at: `2026-09-04T08:22:56.608Z`;
- local SQL SHA-256:
  `e8b8f974677341f0275a85ff7f0c5edfc02e808203b4d7f5f165db07d866e806`;
- 65,771,005 bytes read, 104,344,392 file bytes, four files, 3.364 seconds;
- 1,140,185 rows scanned, 2,992,087,820 bytes and 270 files pruned;
- five metadata rows, zero writes/spill/failed tasks, no value column opened.

### Quarantined candidate

- statement: `01f1a83b-585c-1557-9635-6a5640331f9a`;
- assessed at: `2026-09-04T08:33:46.008Z`;
- exact v2 SQL SHA-256:
  `9f4104e4db8b16bfa21e0fce26afb47d364f983b95209424293356489fe6d81a`;
- 31,967,426 bytes read, 52,172,196 file bytes, two files, 6.955 seconds;
- 570,093 rows scanned, 1,496,043,910 bytes and 135 files pruned;
- 12,083 rows returned, zero writes/spill/failed tasks.

All successful statements observed the Warehouse already `RUNNING`. This
client issued no start, resize or create. The previously rejected stop was not
retried; rely on the configured 45-minute auto-stop or a platform owner.

## Failed attempts retained

- `scripts.run_workspace_local` rejected the capture module because it is not
  allowlisted. The direct single-purpose module invocation then succeeded.
- Three Databricks SQL Connector attempts failed before statement submission
  on the managed Windows certificate-store ASN.1 error.
- A fourth connector transport attempt was interrupted after 180 seconds.
  Query history showed no connector query and no output existed.
- The candidate was then submitted exactly once through the Statement
  Execution API and downloaded through one short-lived Arrow link. The URL was
  neither logged nor persisted. No candidate statement retry occurred.

## Platform coordination

Issue 4 now contains a second non-value comment requesting a platform-signed
finality receipt bound to the exact semantic hash, window and five SeriesKeys:

`https://github.com/FMVSA/opendata-lakehouse/issues/4#issuecomment-5537931901`

IT-North requires platform authority; LSEG reconciliation cannot substitute
for it. The current September CH backfill remains a separate issue.

## Cumulative external-operation accounting

- Databricks REST API requests: `86`;
- Databricks statements: `6`;
- raw business rows returned and quarantined locally: `12,083`;
- Warehouse starts/resizes/creates: `0/0/0`;
- Warehouse stop requests/successes: `1/0`, historical and not retried;
- governance remote writes: `3` (issue creation and two comments);
- Databricks data writes, model training/retraining, CT changes, T057 access
  and solver changes: `0/0/0/0/0/0`.

This increment added 13 Databricks REST calls, three successful statements,
four non-statement connector transport attempts and one external result
download. The prohibited 9.36 GB comparison was not repeated.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-JULY-CANDIDATE-CAPTURE-V1-20260904.json`
  - freezes coverage, candidate identity, costs, failures and authorities.
- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`
  - records the captured candidate and revised execution order.
- `tests/test_lt_source_acquisition_outage_plan.py`
  - binds the evidence hash, exact coverage and finality-negative state.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`
  - routes the next operator to validate existing bytes after receipt.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds durable decision D-20260904-286.
- `.planning/HANDOFF.md`
  - advances the current state and handoff pointer.
- this handoff.

No LT runtime/model, monthly solver, CT, scenario, stochastic-path, Power BI,
T057 or heavy desk-data file changed. The new value-bearing artifacts are all
ignored below `build/`.

## Verification

```text
python -B -m scripts.run_workspace_local --run-id entjulct1 --wall-timeout-seconds 600 -- python -B -m pytest tests/test_lt_source_acquisition_outage_plan.py tests/test_entsoe_day_ahead_export.py -q -p no:cacheprovider
1 failed, 29 passed, 1 warning in 10.11s

Reason: one stale LSEG status assertion still expected the pre-capture state.
The assertion was updated to the new finality wait state.

python -B -m scripts.run_workspace_local --run-id entjulct2 --wall-timeout-seconds 600 -- python -B -m pytest tests/test_lt_source_acquisition_outage_plan.py tests/test_entsoe_day_ahead_export.py -q -p no:cacheprovider
30 passed, 1 warning in 7.32s

python -B -m scripts.run_workspace_local --run-id ruff286d --wall-timeout-seconds 300 -- python -B -m ruff check tests/test_lt_source_acquisition_outage_plan.py
All checks passed!

python -B -m scripts.run_workspace_local --run-id ruff286e --wall-timeout-seconds 300 -- python -B -m ruff format --check tests/test_lt_source_acquisition_outage_plan.py
1 file already formatted
```

The warning is the existing unknown pytest `cache_dir` option. The required LT
minimum from the preceding increment remains `58 passed, 1 skipped`; this
increment changed only governance evidence, documentation and its contract
test. All governed and local receipt JSON files parse successfully and
`git diff --check` passes; its only output is the existing Windows
LF-to-CRLF working-copy warning. `git status` confirms that no `build/` file,
including the value-bearing Parquet, is tracked or offered for commit.

## Next admissible action

Do not rerun the July query. Wait for the platform-signed finality receipt on
issue 4. It must bind semantic SHA-256
`5a6d72b5531e843dc4e7920c519cc3d93189e70d91277662a763f6575967374e`,
the exact delivery window and all five SeriesKeys. If it passes, validate the
existing local Parquet as `realized_final`, then build and independently replay
the unsigned v2 package. Continue to treat September CH recovery separately.

All causal, model-input, model-selection, monthly-level, publication,
production and trading authorities remain false. Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; the CH monthly BASE solver
remains sole level authority and T057 remains sealed.

Durable decision: D-20260904-286.
