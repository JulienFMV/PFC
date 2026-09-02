# Session handoff - ENTSO-E day-ahead PRD gate

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

A focused, offline and fail-closed gate now prepares the first governed PRD
day-ahead pass without starting Databricks compute. It profiles only the five
required A44 price fields and validates a one-month point-in-time extract. A
pass never authorizes model input, selection or production.

Durable decision: D-20260901-260.

## Changed files

- `pfc_shaping/validation/entsoe_day_ahead_prd.py`
  - verifies both SQL hashes;
  - assesses the value-blind PRD profile;
  - requires exact five-SeriesKey binding and a complete rebuild receipt;
  - builds one-month partition-pruned PIT parameters;
  - validates the local long-form PIT extract and rejects leakage, duplicate
    grain, invalid intervals, non-finite prices and the row-limit sentinel.
- `docs/data/sql/databricks_prd_entsoe_day_ahead_profile.sql`
  - read-only aggregate profile;
  - `_year` range pruning;
  - no raw price returned;
  - 101-row rejection sentinel;
  - SHA-256
    `d89bc5f42b1ec6cefcfb1cfb5ef044f22a647941c598f9f4530c86c5d1295603`.
- `docs/data/sql/databricks_prd_entsoe_day_ahead_pit_extract.sql`
  - exact five-SeriesKey bindings;
  - one `_year`/`_month` partition;
  - `availability_timestamp_utc <= as_of_utc`;
  - deterministic latest eligible vintage;
  - 20,001-row rejection sentinel;
  - SHA-256
    `7b444430db6b62a7bcd9f0bc6e5f05858a56b9756be68f70016c107a4b271ebf`.
- `tests/test_entsoe_day_ahead_prd.py`
  - 28 tests covering the pass path and every fail-closed family.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - decision D-20260901-260.
- `.planning/HANDOFF.md`
  - current handoff pointer and state updated.

## Public API

`pfc_shaping.validation.entsoe_day_ahead_prd` exposes three operational
operations plus SQL verification:

- `assess_day_ahead_prd_profile(...)`;
- `build_day_ahead_pit_parameters(...)`;
- `validate_day_ahead_pit_extract(...)`;
- `verify_sql_bindings()`.

The gate requires fields `ch_price`, `at_price`, `de_lu_price`, `fr_price` and
`it_nord_price`, units `EUR/MWh`, document type A44 and canonical keys with an
optional non-empty classification suffix.

## Rebuild evidence contract

The real receipt must identify PRD, run ID, manifest SHA-256, deployed Git
commit, required-change commit, ancestry proof, full mode, rebuilt groups,
successful run and post-backfill validation, zero old/new coexistence and a
UTC completion timestamp. The required change is commit
`db3a93316cd431a95b4e096d8482e482fda3491e`.

The rebuilt groups must include:

- `day_ahead_prices`;
- the three outage families;
- `installed_capacity_per_unit`;
- `generation_forecast`.

## Verification

Commands ran through `scripts.run_workspace_local`; mutable state stayed below
`build/`.

- dedicated: `28 passed`;
- relevant compatibility matrix: `160 passed`;
- LT/CT and package boundary: `43 passed, 1 skipped`;
- Ruff check: pass;
- Ruff format check: pass.

The broad command `pytest tests -q -m "not slow"` reached the supervisor wall
timeout at 1,800 seconds before pytest wrote a result. This is not recorded as
a pass or a test failure. The terminal receipt is
`build/workspace-local-supervisors/dapfull1/supervisor-receipt.json`; it
confirms timeout, complete tree termination and zero authority.

The recurring `PytestConfigWarning` for unknown `cache_dir` remains an existing
environment/configuration warning and did not affect the passing matrices.

Exact successful commands:

```text
python -B -m scripts.run_workspace_local --run-id daptest3 -- python -B -m pytest tests\test_entsoe_day_ahead_prd.py -q -p no:cacheprovider
python -B -m scripts.run_workspace_local --run-id dapmatrix1 --wall-timeout-seconds 1800 -- python -B -m pytest tests\test_entsoe_day_ahead_prd.py tests\test_databricks_pfc_layer_acceptance.py tests\test_databricks_pfc_source_cost_preflight.py tests\test_databricks_lt_materialization.py tests\test_entsoe_resolution_regimes.py tests\test_entsoe_bounded_execution_receipt.py -q -p no:cacheprovider -m "not slow"
python -B -m scripts.run_workspace_local --run-id dapbound1 --wall-timeout-seconds 600 -- python -B -m pytest tests\test_lt_ct_imports.py tests\test_lt_package_contract.py -q -p no:cacheprovider -m "not slow"
python -B -m scripts.run_workspace_local --run-id dapruff3 -- python -B -m ruff check pfc_shaping\validation\entsoe_day_ahead_prd.py tests\test_entsoe_day_ahead_prd.py
python -B -m scripts.run_workspace_local --run-id dapfmt3 -- python -B -m ruff format --check pfc_shaping\validation\entsoe_day_ahead_prd.py tests\test_entsoe_day_ahead_prd.py
```

Their execution receipts are below
`build/workspace-local-runs/{daptest3,dapmatrix1,dapbound1,dapruff3,dapfmt3}/execution-receipt.json`.

Corrected intermediate failures:

- `daptest1`: `16 failed, 12 passed`; the gate incorrectly validated the
  40-character Git commit as a 64-character SHA-256. Git SHA-1 and artifact
  SHA-256 validation are now separate, and `daptest2`/`daptest3` pass 28/28.
- `dapfmt1`: format check requested two files; `dapfmt2` formatted them, and
  terminal `dapfmt3` confirms both are already formatted.
- `dapfull1`: terminal supervisor timeout, not a pytest verdict; exact command
  was `python -B -m scripts.run_workspace_local --run-id dapfull1
  --wall-timeout-seconds 1800 -- python -B -m pytest tests -q -p
  no:cacheprovider -m "not slow"`.

## Current blockers and next action

No real profile was executed. Databricks SQL statements, Warehouse starts and
writes remain `0/0/0`; the configured Warehouse remains stopped.

Wait for the independently supplied PRD rebuild receipt. Once it is present
and compute is already active or separately authorized:

1. execute the hash-bound profile query with explicit start/end years;
2. bind exactly one admitted SeriesKey per field;
3. admit the profile locally;
4. execute the monthly PIT query in bounded batches;
5. run the existing effective-dated cadence and downstream snapshot gates.

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`. A new independently frozen
future holdout is still required before empirical selection. T057 remains
sealed.
