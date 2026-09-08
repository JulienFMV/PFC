# Session handoff - hourly feature inventory

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decisions: D-20260904-289 through D-20260904-291

## Outcome

The exact common feature inventory for the five hourly benchmark candidates is
now frozen without opening real data or changing any model implementation.

The ordered matrix is:

1. `hour_sin`;
2. `hour_cos`;
3. `month_sin`;
4. `month_cos`;
5. `dow_sin`;
6. `dow_cos`;
7. `is_holiday`;
8. `hydro_fill`;
9. `years_ahead`.

Calendar and maturity entries are deterministic. Historical `hydro_fill`
requires a realized value available before the origin; prediction uses only a
week-of-year climatology fitted strictly before that origin. Nulls remain null
until the single common complete-case mask. The final matrix unit is explicitly
a normalized fraction `[0,1]`, not a raw percentage.

The incumbent source still has three outage positions. They are not candidate
data columns: the hash-bound governed config disables outages, so the incumbent
replay keeps fixed internal compatibility constants. This is not permission to
zero-fill missing outage observations or forecasts. No complete origin-available
M01-M36 outage forecast has been established.

Raw PRD `load_mw`, `solar_mw`, `wind_mw` and `cross_border_mw` are realized
history, not future same-target features. Their causal `solar_regime`,
`load_deviation` and `flow_deviation` climatologies remain separate common
quarter-hour shaping context. They cannot be added only to challengers.

The PRD input adapter now rejects every noncanonical feature inventory or order.
No Databricks request, Warehouse action, data materialization, model fit, truth
opening, GPU execution or remote write occurred. Every model and production
authority remains false.

## Changed files

- `pfc_shaping/lt/evaluation_feature_inventory.py`
  - new immutable inventory and authority-negative semantic manifest;
  - semantic SHA-256:
    `83fabacc804de4201560877400fefa6974076f64d14f9738306a86bcf815a6fd`;
  - file SHA-256:
    `fb46b71208bf820603e4c4d81552c5539e605908139f439e4b42ae1118de0a70`.
- `pfc_shaping/lt/evaluation_inputs.py`
  - now binds and enforces the canonical inventory;
  - file SHA-256:
    `e020a3a393dc5b5efe9377124538ec08dfadb5c32cabb6d1b902d035f49f4331`.
- `tests/test_lt_evaluation_feature_inventory.py`
  - new exact-inventory, exclusions, hash, authority and no-I/O tests;
  - file SHA-256:
    `9939bf9e1074016564da0057f3af4c65a4447104490d12206e30940f918dfa87`.
- `tests/test_lt_evaluation_inputs.py`
  - uses the nine-column inventory and proves alternate order rejection;
  - file SHA-256:
    `2ecfc205606140f2fadd008031373a87036759807d30f33ee1d5276ee7f06dcf`.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - records the exact hourly matrix and intraday-context separation.
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
  - states that PRD presence does not imply future feature admission.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D291.
- `.planning/HANDOFF.md`
  - points to D291 and this handoff.

All preceding uncommitted 4 September ENTSO-E/LSEG work remains preserved.

## Verification

Every shell action first verified that both the current directory and Git root
were exactly `C:\Users\jbattaglia\PFC_LT`. Mutable test receipts stayed below
`build/` through `scripts.run_workspace_local`.

- first focused run: `15 passed, 1 warning`;
- post-format focused run: `15 passed, 1 warning`;
- adjacent evaluation/feature-availability/materialization/LT-import matrix:
  `132 passed, 1 skipped, 1 warning`;
- final matrix including the governed package contract:
  `158 passed, 1 skipped, 1 warning`;
- targeted Ruff check: pass;
- the first Ruff format check reported three files requiring formatting; the
  governed formatter changed only those three files and the post-format tests
  passed.

The warning is the existing unknown pytest `cache_dir` option. The skipped case
is an existing optional dependency/runtime test.

Read-only audit slips before implementation had no repository side effect: one
PowerShell command had an unmatched quote, one search used the wrong
`validation/` path for the materializer, and one `rg` invocation used Windows
wildcards as literal paths. Each was corrected immediately. The first combined
patch also failed on stale context and changed no file; it was split into exact
small patches.

## Next safe step

This step is now complete under D292 and
`SESSION-HANDOFF-20260904-HOURLY-FEATURE-BUILDER.md`.

After that, add the adapter from prepared arrays to the incumbent and synthetic
challenger interfaces. Real-data fitting still requires the independently
registered prospective origin and governed snapshot evidence.

## Session/window status

No new window is required yet. This handoff is sufficient if a new window is
preferred before implementing the feature constructor or the later real-data
runner.

## Invariants

- PRD is the enterprise-validated source boundary; upstream ENTSO-E/API
  recovery belongs to Data Engineering. Consumer checks address causal fitness,
  not source legitimacy.
- Every candidate receives the same ordered feature matrix and eligible rows.
- Disabled outage constants cannot become observed or forecast feature values.
- Causal ENTSO-E climatologies remain common downstream quarter-hour context.
- The CH monthly BASE solver is the sole monthly-level authority.
- LT remains independent from CT, T057 remains sealed, and every acquisition,
  training, truth, selection, publication and production authority is false.
