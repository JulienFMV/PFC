# Session handoff - ENTSO-E exact series-family mapping

Date: 2026-08-06  
Decision: D-20260806-251  
Status: offline mapping gate ready; real current inventory still not captured

## Outcome

The 2026-08-03 dev audit supports actual load, day-ahead prices, actual and
forecast generation, solar/wind forecasts, hydro reservoir storage, physical
flows, scheduled exchanges and day/month/year-ahead NTC. It still does not
prove the complete current family inventory, load forecast, renewable forecast
horizons, technology coverage, all CH borders/directions, native cadence,
canonical unit metadata, signs, source lineage, revision/quality or PIT.

D251 adds an offline exact owner-mapping gate for the future bounded D243
dimension inventory. It hashes the nine raw semantic fields, rejects fuzzy or
duplicate mappings, and reports core family, technology, renewable horizon,
border, unit and unmapped coverage. A raw day-ahead price unit `EUR` cannot be
accepted as the canonical business unit; an accountable mapping must state
`EUR/MWh`.

The second exploration tier now includes per-unit actual generation, available
generation capacity, curtailment, load forecast margin, flow-based capacity
parameters and allocated cross-zonal balancing capacity. These are useful
scarcity/congestion candidates but are not blockers for the first governed CH
LT baseline.

## Exact changed files

- `pfc_shaping/validation/entsoe_series_inventory.py`
  - added stable `series_signature_id` over the nine raw semantic fields;
  - SHA-256
    `563da7dce3d4b9c6307c9a9e432219ee458359dee122f2f71f14bb49dcc94d1b`.
- `pfc_shaping/validation/entsoe_series_family_mapping.py`
  - new offline exact mapping and value-free coverage assessment;
  - SHA-256
    `a2a2fec240780eba9dbed7e347d380ff2ec6964eba0bf2f6ddb591ba03f08418`.
- `tests/test_entsoe_series_family_mapping.py`
  - synthetic-only mapping, ambiguity, unit, border, binding and authority
    tests;
  - SHA-256
    `1bd2d8d2c3e73143e5db9338fe863f6d4f4b45d21c28be93365ce31bf6109065`.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DEV-FAMILY-COVERAGE-STATUS-20260806.md`
  - added the non-blocking second exploration tier.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
  - added the same short engineering request.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - added D-20260806-251.
- `.planning/HANDOFF.md`
  - routed new sessions to this handoff.

Concurrent pre-existing/untracked work, including D250 real physical-mapping
remediation, was preserved and not rewritten.

## Commands and outputs

Every command verified exact cwd `C:\Users\jbattaglia\PFC_LT` and Git root
`C:/Users/jbattaglia/PFC_LT`. All mutable test state was routed through
`scripts.run_workspace_local` below `build/`.

- focused mapping plus inventory:
  `python -B -m scripts.run_workspace_local --run-id entmap251a python -B -m pytest tests/test_entsoe_series_family_mapping.py tests/test_entsoe_series_inventory.py -q`
  -> `26 passed in 0.25s`;
- adjacent ENTSO-E matrix:
  `python -B -m scripts.run_workspace_local --run-id entadj251 python -B -m pytest <all tests/*entsoe*.py> -q`
  -> `443 passed in 28.48s`; final rerun `entadj251b` ->
  `443 passed in 21.21s`;
- Ruff:
  `python -B -m scripts.run_workspace_local --run-id entruff251a python -B -m ruff check <D251 files>`
  -> `All checks passed!`.

## Cost and authority

- D251 Databricks control-plane requests: 0;
- SQL statements: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- real ENTSO-E rows opened: 0.

The 2026-08-06 reservation remains consumed by D247. Do not retry today. On a
future Europe/Zurich day, run D243 at most once and only if the selected
Warehouse is already running. Then obtain the owner-reviewed exact mapping and
run D251 offline. Classification alone never authorizes model use.

## Next safe action

Wait for a future allowed capture day and an already-running Warehouse. Capture
only the bounded D243 dimension inventory, never facts for discovery. Ask the
data engineer to map exact signature IDs and close D250 cadence/PIT/lineage/
quality/revision/sign gaps before any governed local value export.

Monthly solver authority, LT/CT separation, T057 sealing, OMPEX benchmark-only
status, AFRY descriptive-only status and strict production `NO_GO` remain
unchanged.
