# Session handoff - ENTSO-E directional family mapping v2

Date: 2026-08-06  
Decision: D-20260806-253  
Status: offline mapping semantics hardened; real current inventory still absent

## Outcome

The D251 exact series-family mapping was roasted against the D239/D245 quality
requirements and D250 real-schema remediation. A material false-completeness
path was found: border presence did not prove both relevant directions. A
single CH-DE mapping could therefore satisfy the border check while reverse
flow, exchange or capacity remained absent.

Mapping schema v2 closes this path. Every directional entry now declares one
of:

- `CH_TO_NEIGHBOR_ONLY`;
- `NEIGHBOR_TO_CH_ONLY`;
- `BOTH_DIRECTIONS_IN_ONE_SIGNED_SERIES`.

The last option is admissible only for physical flow and only with an explicit
positive sign convention. It is forbidden for NTC, scheduled exchange and
allocated cross-zonal balancing capacity. Coverage reports missing direction
for every required family and each CH-DE/FR/IT/AT border.

The roast also removed a semantic authority overclaim. A caller-declared real
owner review is now reported only as `owner_asserted_*`; external owner
identity and `family_mapping_verified` remain false. A mapping may not predate
its bound inventory capture.

## Changed files

- `pfc_shaping/validation/entsoe_series_family_mapping.py`
  - mapping and assessment schema v2;
  - exact directional coverage and sign consistency;
  - owner-assertion versus verification separation;
  - SHA-256
    `243e494475950f729cf302113630bfe34d22d696577f5e1035c64f6970c2d8ae`.
- `tests/test_entsoe_series_family_mapping.py`
  - direction/sign mismatch, signed bidirectional physical flow, forbidden NTC
    shortcut, temporal binding and owner-authority regression tests;
  - SHA-256
    `ba6b34ca9485f0fb9fdc30ba5ff3043fa8bd7e13ff636efd7b4db1bda6398536`.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DEV-FAMILY-COVERAGE-STATUS-20260806.md`
  - documents exact direction requirements.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
  - adds the short directional request for the data engineer.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D253; the independently reserved D252 entry was preserved.
- `.planning/HANDOFF.md`
  - routes current work to this handoff.

## Verification

All shell commands first verified exact cwd
`C:\Users\jbattaglia\PFC_LT` and Git root
`C:/Users/jbattaglia/PFC_LT`. Mutable test state stayed below `build/` through
`scripts.run_workspace_local`.

- focused:
  `python -B -m scripts.run_workspace_local --run-id entmap252b python -B -m pytest tests/test_entsoe_series_family_mapping.py tests/test_entsoe_series_inventory.py -q`
  -> `32 passed in 0.18s`; final rerun `entmap253f` ->
  `32 passed in 0.21s`;
- adjacent ENTSO-E:
  `python -B -m scripts.run_workspace_local --run-id entadj252a python -B -m pytest <all tests/*entsoe*.py> -q`
  -> `449 passed in 19.67s`; final matrix including the concurrent D252
  coverage-audit tests, `entadj253f` -> `459 passed in 18.67s`;
- Ruff:
  `python -B -m scripts.run_workspace_local --run-id entruff252a python -B -m ruff check <mapping files>`
  -> `All checks passed!`.
- final D252/D253 integration:
  `python -B -m scripts.run_workspace_local --run-id entcov253f python -B -m pytest tests/test_entsoe_series_family_mapping.py tests/test_entsoe_series_inventory.py tests/test_entsoe_pfc_data_coverage.py -q`
  -> `42 passed in 0.29s`.

## Cost and authority

D253 used zero Databricks requests, SQL statements, Warehouse starts, remote
writes and real ENTSO-E rows. The 2026-08-06 reservation remains consumed; no
same-day retry is allowed.

Direction mapping proves neither value completeness nor PIT. A future D243
dimension capture must occur at most once on another Europe/Zurich day and
only when the Warehouse is already running. The data owner must then map the
exact signatures and close D250 cadence, interval-edge, lineage, quality,
revision and PIT decisions before any local governed fact export.

The monthly solver remains sole level authority. LT/CT separation, hourly CH
historical truth, T057 sealing, OMPEX benchmark-only status, AFRY
descriptive-only status and production `NO_GO` remain unchanged.
