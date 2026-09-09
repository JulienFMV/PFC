# Session handoff - ENTSO-E July realized export preflight - 2026-09-04

## Outcome

The July 2026 `realized_final` construction export is prepared but was not
executed. Its exact scope is now frozen:

- Swiss-local delivery window:
  `[2026-06-30T22:00:00Z, 2026-07-31T22:00:00Z)`;
- UTC partitions: `2026-06` and `2026-07`;
- CH: `day_ahead_prices||ch_price`;
- AT: `day_ahead_prices||at_price||1`;
- DE-LU: `day_ahead_prices||de_lu_price||1`;
- FR: `day_ahead_prices||fr_price`;
- IT-North: `day_ahead_prices||it_nord_price`;
- realized-export SQL SHA-256:
  `9f4104e4db8b16bfa21e0fce26afb47d364f983b95209424293356489fe6d81a`.

A live metadata-only cost preflight observed the configured PBI SQL Warehouse
`STOPPED`. Unity Catalog reconfirmed that the managed Delta source is
partitioned by `_year` and `_month`, but exposed neither `sizeInBytes` nor
`numFiles`. The exact query has a hash-bound two-partition predicate, but a
current physical scan upper bound is not proven. The verdict is therefore:

`STOP_NO_ACTIVE_WAREHOUSE_AND_SCAN_BOUND_UNPROVEN`.

No SQL, business-row read, Warehouse start/resize/create or remote write
occurred. All model, monthly-level, publication, production and trading
authorities remain false.

## Contract audit

The existing v2 contract remains internally consistent:

- the query is read-only and uses the 20,001-row rejection sentinel;
- the complete Swiss month correctly spans the June and July UTC partitions;
- AT and DE-LU use only the construction-frozen sequence-1 keys from D282;
- latest-revision ranking remains a candidate, not finality proof;
- the query cutoff `assessed_at_utc` is deliberately unset until the platform
  can bind it to the applicable finality assertion and statement submission;
- `realized_final` cannot be relabelled `causal_asof`;
- the offline replay and value-bound finality gates remain downstream and
  unchanged.

The preflight also exposes two independent execution blockers. A stopped
Warehouse violates the explicit no-start policy, and no current hard scan
upper bound or human-approved maximum exists. The 1 September value-blind
profile read 47,915,292 bytes from 188,573,842 bytes of files, but covered one
UTC July partition and a different projection; it is not a hard bound for this
two-partition export. The stale 1,538,686,371-byte whole-table storage figure
is likewise not a current query cap.

## Databricks evidence and cost

Three read-only control-plane metadata GETs were made in this session:

1. an initial Warehouse-state GET;
2. a Unity Catalog GET for the Silver vintage table;
3. a final Warehouse-state GET fixed the observation time at
   `2026-09-04T06:56:56.217Z`.

Observed Warehouse metadata:

- name: `PBI SQL Warehouse - Analytics`;
- state: `STOPPED`;
- type/size: Classic `2X-Small`;
- clusters: fixed `1..1`;
- serverless: false;
- auto-stop: 45 minutes;
- Warehouse ID SHA-256:
  `3f1905a33ac0c15557f450880efcdd01320ad36c116d5a80f914df1903e9fb2a`.

Execution totals added by this session:

- Databricks API GETs/statements/business rows: `3/0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- remote writes: `0`;
- model training/retraining: `0/0`;
- CT changes, T057 access and solver changes: `0/0/0`.

The previous 9,364,142,086-byte disambiguation statement was not repeated and
must never be repeated. Its exact selection result remains frozen under D282.

## New governed evidence

`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-REALIZED-FINAL-EXPORT-PREFLIGHT-V1-20260904.json`

- file byte length: `5,081`;
- file SHA-256:
  `05bad86a52f62a07e767d413fd0d15d463b827ea9fb2acb505af274dfde1b4bc`;
- canonical JSON byte length: `4,277`;
- canonical JSON SHA-256:
  `b520d9c44215341ddd6b09c68180d22dde45135c0b70627ba6c0f3d7a029fd1f`.

The outage plan binds that canonical identity and now records nine cumulative
Databricks API requests, one historical aggregate statement, zero Warehouse
starts and zero returned raw business rows.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-REALIZED-FINAL-EXPORT-PREFLIGHT-V1-20260904.json`
  - exact prepared export, live metadata preflight, cost blockers,
    prohibitions and negative authorities.
- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`
  - binds the new preflight, advances cumulative GETs from six to nine and
    updates the next execution order.
- `tests/test_lt_source_acquisition_outage_plan.py`
  - validates the exact scope, sequence-1 selection, hash binding, stopped
    state, zero execution counters, absent scan cap and negative authorities.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`
  - records the stopped preflight and the next platform requirements.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds durable decision D-20260904-283.
- `.planning/HANDOFF.md`
  - advances the current-state summary and handoff pointer.
- this handoff.

No LT runtime/model, monthly solver, CT, scenario, stochastic-path, Power BI,
T057 or heavy desk-data file changed.

## Verification

Focused contract matrix:

```text
python -B -m scripts.run_workspace_local --run-id erfocus3 --wall-timeout-seconds 600 -- python -B -m pytest tests/test_lt_source_acquisition_outage_plan.py tests/test_entsoe_day_ahead_export.py -q -p no:cacheprovider
27 passed, 1 warning in 9.93s
```

Required LT minimum:

```text
python -B -m scripts.run_workspace_local --run-id ltmin283 --wall-timeout-seconds 1200 -- python -B -m pytest tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
58 passed, 1 skipped, 1 warning in 45.79s
```

Ruff and static checks:

```text
python -B -m scripts.run_workspace_local --run-id ruff283a --wall-timeout-seconds 300 -- python -B -m ruff check tests/test_lt_source_acquisition_outage_plan.py
All checks passed!

python -B -m scripts.run_workspace_local --run-id ruff283d --wall-timeout-seconds 300 -- python -B -m ruff format --check tests/test_lt_source_acquisition_outage_plan.py
1 file already formatted

JSON parse: 2 files passed
git diff --check: passed with existing Windows LF-to-CRLF notices only
```

The pytest warning is the existing environment's unknown `cache_dir` option.
The skip is the pre-existing optional TensorFlow import boundary.

Three local precheck attempts did not run tests: one run ID exceeded the
16-character limit, one supplied a forbidden caller pytest basetemp and the
next reused the precreated supervisor namespace. `erfocus2` then passed before
formatting; `erfocus3` is the final focused result above. Ruff's formatting
command changed one file and the subsequent format check passed.

## Next admissible action

Do not run SQL yet. Obtain from the platform either a current hard scan upper
bound for the exact hash-bound query or a platform export quote. Then observe
the same Warehouse already `RUNNING` for another authorized workload and get
explicit human acceptance of the exact scan/cost ceiling. The platform-owned
delivery must set the exact assessment cutoff, preserve the five selected
keys, return a bounded raw export and provide value-bound finality evidence for
all five series, including platform evidence for IT-North.

After delivery, validate the raw frame, build and replay the unsigned v2
package locally, then request the matched LSEG reconciliation. Do not infer
causal history, retrain or begin scenario/path work.

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; the CH monthly BASE solver
remains the sole level authority and T057 remains sealed.

Durable decision: D-20260904-283.
