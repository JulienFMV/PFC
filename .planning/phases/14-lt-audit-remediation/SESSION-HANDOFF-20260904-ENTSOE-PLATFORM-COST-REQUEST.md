# Session handoff - ENTSO-E platform cost request - 2026-09-04

## Outcome

The next admissible external step is complete. The bounded July 2026
`realized_final` export remains unexecuted, and a platform cost/export-contract
request is now open at:

`https://github.com/FMVSA/opendata-lakehouse/issues/4`

The issue was created by `JulienFMV` at `2026-09-04T07:38:22Z`. At immediate
verification it was `OPEN`, unassigned and had no comments. It requests only:

- a current hard scan upper bound or platform export quote;
- expected files/bytes, runtime, DBU and cloud-cost ceiling with currency;
- confirmation that any later execution uses a Warehouse already running for
  a separately authorized workload;
- exact bounded output terms;
- the `assessed_at_utc` rule;
- value-bound finality evidence for all five SeriesKeys, including IT-North.

Issue 4 is not SQL, Warehouse-start or export authorization. Explicit human
acceptance of the returned ceiling remains required.

## Latest Warehouse check

Before opening the issue, one new read-only Databricks control-plane GET
observed the same PBI SQL Warehouse still `STOPPED` at
`2026-09-04T07:36:57.614Z`. No SQL or business-row read was attempted and no
Warehouse was started, resized or created.

The frozen D283 preflight remains byte-identical:

- path:
  `.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-REALIZED-FINAL-EXPORT-PREFLIGHT-V1-20260904.json`;
- canonical JSON SHA-256:
  `b520d9c44215341ddd6b09c68180d22dde45135c0b70627ba6c0f3d7a029fd1f`;
- status: `STOP_NO_ACTIVE_WAREHOUSE_AND_SCAN_BOUND_UNPROVEN`.

The plan now records ten cumulative Databricks API requests, one historical
aggregate statement, zero raw rows returned, zero Warehouse starts and one
governance-coordination remote write.

## Exact requested export scope

- usage: `realized_final` construction/reconciliation smoke only;
- window: `[2026-06-30T22:00:00Z, 2026-07-31T22:00:00Z)`;
- partitions: `2026-06`, `2026-07`;
- SQL SHA-256:
  `9f4104e4db8b16bfa21e0fce26afb47d364f983b95209424293356489fe6d81a`;
- row limit: 20,001 rejection sentinel;
- CH: `day_ahead_prices||ch_price`;
- AT: `day_ahead_prices||at_price||1`;
- DE-LU: `day_ahead_prices||de_lu_price||1`;
- FR: `day_ahead_prices||fr_price`;
- IT-North: `day_ahead_prices||it_nord_price`.

The issue explicitly prohibits repeating the prior 9.36 GB series-selection
comparison and prohibits representing the July backfill as `causal_asof`.

## External operations

- GitHub repository search for a duplicate request: one read-only call, no
  matching issue;
- GitHub issue creation: one governance remote write;
- GitHub issue verification: one read-only call;
- Databricks control-plane GETs/statements/business rows in this continuation:
  `1/0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- data writes or business-value remote writes: `0`;
- model training/retraining, CT changes, T057 access and solver changes:
  `0/0/0/0/0`.

## Changed files

- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`
  - records issue 4, one coordination remote write, ten cumulative Databricks
    API calls and the external wait state.
- `tests/test_lt_source_acquisition_outage_plan.py`
  - verifies the exact issue receipt and preserves negative authorities.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`
  - routes the next operator to issue 4 and keeps SQL stopped.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds durable decision D-20260904-284.
- `.planning/HANDOFF.md`
  - advances the current state and handoff pointer.
- this handoff.

The preceding uncommitted D283 preflight artifact and documentation remain in
the same working tree. No LT runtime/model, monthly solver, CT, scenario,
stochastic-path, Power BI, T057 or heavy desk-data file changed.

## Verification

```text
python -B -m scripts.run_workspace_local --run-id erfocus5 --wall-timeout-seconds 600 -- python -B -m pytest tests/test_lt_source_acquisition_outage_plan.py tests/test_entsoe_day_ahead_export.py -q -p no:cacheprovider
28 passed, 1 warning in 0.57s

python -B -m scripts.run_workspace_local --run-id ruff284a --wall-timeout-seconds 300 -- python -B -m ruff check tests/test_lt_source_acquisition_outage_plan.py
All checks passed!

python -B -m scripts.run_workspace_local --run-id ruff284b --wall-timeout-seconds 300 -- python -B -m ruff format --check tests/test_lt_source_acquisition_outage_plan.py
1 file already formatted
```

The required LT minimum from the immediately preceding D283 increment remains
`58 passed, 1 skipped`; this continuation changed only governance evidence,
documentation and its contract test. The warning is the existing unknown
pytest `cache_dir` option, and the skip is the pre-existing optional TensorFlow
import boundary.

## Next admissible action

Wait for a complete response on issue 4. Do not poll by running SQL and do not
start the Warehouse. Once the platform returns a current hard scan bound or
quote and exact delivery/finality terms:

1. validate the response against the frozen scope and reject any sequence,
   window, query-hash or authority drift;
2. obtain explicit human acceptance of the exact scan/cost ceiling;
3. recheck that the configured Warehouse is already running for a separately
   authorized workload;
4. only then permit the single bounded platform-owned export;
5. validate and replay the unsigned v2 package locally before any LSEG
   reconciliation request.

All model-input, model-selection, monthly-level, publication, production and
trading authorities remain false. Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; the CH monthly BASE solver
remains sole level authority and T057 remains sealed.

Durable decision: D-20260904-284.
