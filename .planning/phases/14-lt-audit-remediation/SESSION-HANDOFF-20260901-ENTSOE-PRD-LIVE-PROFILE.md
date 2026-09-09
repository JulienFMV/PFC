# Session handoff - ENTSO-E PRD live value-blind profile

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

The next Databricks step is complete and audited. One real, parameterized,
value-blind profile was executed for `[2026-07-01, 2026-08-01)` on the existing
`PBI SQL Warehouse - Analytics`, observed `RUNNING` before execution.

The statement succeeded and returned the exact seven expected source series.
The gate nevertheless remains blocked for two independent reasons:

1. every one of the 17,925 profiled Silver rows fails at least one combined
   `publication_timestamp_utc` / `first_seen_pull_ts_utc` /
   `last_seen_pull_ts_utc` order predicate;
2. 404 rows fail the interval timestamp/duration predicate, and the formal PRD
   rebuild manifest is still absent.

No price value was selected or persisted. No Warehouse start, resize, create,
Databricks write or SQL retry occurred. Bounded PIT extraction, model input,
model selection and production remain unauthorized.

## Real profile evidence

Window: `[2026-07-01T00:00:00Z, 2026-08-01T00:00:00Z)`.

| Series | Resolution | Rows / distinct intervals | Invalid availability order | Invalid interval |
|---|---:|---:|---:|---:|
| `day_ahead_prices||at_price||1` | PT15M | 2,935 / 2,935 | 2,935 | 24 |
| `day_ahead_prices||at_price||2` | PT15M | 2,914 / 2,914 | 2,914 | 49 |
| `day_ahead_prices||ch_price` | PT60M | 737 / 737 | 737 | 6 |
| `day_ahead_prices||de_lu_price||1` | PT15M | 2,922 / 2,922 | 2,922 | 36 |
| `day_ahead_prices||de_lu_price||2` | PT15M | 2,933 / 2,933 | 2,933 | 36 |
| `day_ahead_prices||fr_price` | PT15M | 2,713 / 2,713 | 2,713 | 112 |
| `day_ahead_prices||it_nord_price` | PT15M | 2,771 / 2,771 | 2,771 | 141 |

All seven rows report zero null-value, `dq_failed` and unknown-availability
counts. Global duplicate vintage, orphan SeriesKey, duplicate Gold SeriesKey,
duplicate latest-grain and legacy/new-overlap counts are zero. Those clean
checks do not override the blocking timestamp and interval findings.

## Cost and execution evidence

Databricks query history reports:

- final state `FINISHED` / API state `SUCCEEDED`;
- duration 10,196 ms;
- 47,915,292 bytes read from 17 files;
- 5,052,298 rows read;
- 7,776,297,639 bytes and 687 files pruned;
- seven result rows, no truncation;
- zero remote-write bytes, zero spill and zero failed tasks.

The exact SQL text in query history hashes to the bound template SHA-256
`e48bc8b09d6f3676616ed42966d50a3f44d9eaf3649f9ce9c9543a0bc024259e`.

Artifacts:

- `build/entsoe-day-ahead-prd-profile/20260901-july-profile-api/capture.json`
  - file SHA-256
    `661f0efaaaab19a4654a526d7f7c71c8782a3c7ebf20c7af5d740fdf2e5383d0`;
  - content ID
    `abef7f875e43358cdd663f1f17fd45a862907a79ddedb6aef9dd42da121bad2c`;
  - offline deterministic replay: `PASS_CAPTURE_REPLAY`.
- `build/entsoe-day-ahead-prd-profile/20260901-july-profile-api/query_history.json`
  - file SHA-256
    `8972050fadcfb9ba33357d41ce3747ebcfe03768b5920db87194536113da57ca`.

## Changed files

- `docs/data/sql/databricks_prd_entsoe_day_ahead_profile.sql`
  - exact `_year = delivery_year` and `_month = delivery_month` pruning;
  - SHA-256
    `e48bc8b09d6f3676616ed42966d50a3f44d9eaf3649f9ce9c9543a0bc024259e`.
- `pfc_shaping/validation/entsoe_day_ahead_prd.py`
  - shared single-month parameter fence for profile and PIT queries;
  - public typed profile-parameter builder;
  - SHA-256
    `2c1dd5a89138a43d80456218cb5481f3dd9ae6717fdd6d560d3bf4ef00a89d40`.
- `scripts/capture_entsoe_day_ahead_prd_profile.py`
  - one-shot hash-bound Statement Execution API capture;
  - existing-running-Warehouse guard, native parameters, cancel-on-wait,
    zero retry, value/row fences and atomic repo-local receipt;
  - deterministic offline capture replay and tamper detection;
  - SHA-256
    `591e1e6fb11748548b5ca38c083f81ed173ced352e66a352688182641ee62646`.
- `tests/test_entsoe_day_ahead_prd.py`
  - monthly profile-parameter and SQL partition-fence tests;
  - SHA-256
    `c1d6b822c94a0cbbdd814fba24035f390b24eb75df440c6470864a797e94521a`.
- `tests/test_capture_entsoe_day_ahead_prd_profile.py`
  - result schema/type/truncation, output path, replay and tamper tests;
  - SHA-256
    `3aadf85b67aa662baee4d95d088440f203cb9b2c522a39a3ba7b5aa7108f6186`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - durable decision D-20260901-263.
- `.planning/HANDOFF.md`
  - current handoff pointer and live profile state.

## Audit evidence

```text
dapreplay3
pytest tests/test_entsoe_day_ahead_prd.py tests/test_capture_entsoe_day_ahead_prd_profile.py
43 passed

dapallent1
pytest all tests/test_*entsoe*.py -m "not slow"
778 passed

dapdbmatrix1
pytest Databricks + ENTSO-E profile/capture + spot reconciliation
258 passed

dapltbound2
pytest tests/test_lt_ct_imports.py tests/test_lt_package_contract.py
43 passed, 1 skipped

dapfinalruff1 / dapfinalfmt1
Ruff check and Ruff format check
pass
```

The offline real-capture replay returned `PASS_CAPTURE_REPLAY` with zero
Databricks statements.

## Operational notes and next gate

Two failed transport attempts occurred before the successful statement:

- a PowerShell 5.1 serialization loop was stopped locally after query history
  proved that no profile statement had been submitted;
- the first Python attempt failed during local Windows CA loading, before the
  control-plane GET and before SQL. The final client uses the installed
  read-only `certifi` CA bundle.

An empty failed-staging directory may remain at
`build/entsoe-day-ahead-prd-profile/20260901-july-profile`; the host rejected
its cleanup command. It contains no files and has no authority.

Before asking Jerome anything, the smallest next step is one separately
reviewed value-blind aggregate diagnostic that splits the five
availability-order predicates and the interval-failure predicates per
SeriesKey. It must keep the same July partition fence, one-statement budget,
no prices and no extraction authority. Only the resulting irreducible source
contract question should be sent to him.

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; T057 remains sealed.

Durable decision: D-20260901-263.
