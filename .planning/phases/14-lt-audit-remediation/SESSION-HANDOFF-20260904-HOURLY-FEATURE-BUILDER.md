# Session handoff - hourly feature builder

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decisions: D-20260904-289 through D-20260904-292

## Outcome

The frozen nine-column hourly feature inventory now has a pure in-memory value
constructor. It consumes only an already materialized exact input:

- `delivery_at_utc`;
- `hydro_available_at_utc`;
- `hydro_fill` as a normalized fraction `[0,1]`.

The constructor verifies one exact origin from the frozen protocol, the
training/prediction delivery side, the split-specific hydro information role,
availability at the origin and the strict pre-origin cutoff for prediction
climatology. It computes Swiss-local hour/month/weekday and Valais/German
holiday semantics through the shared calendar, computes maturity with the
incumbent 365.25-day convention and calls the incumbent encoder directly.

Only its first nine admitted positions are returned. Extra raw ENTSO-E or
outage columns fail closed. Missing hydro stays null for the later one-common
mask. Percentages, infinities and non-numeric hydro values fail instead of
triggering a guessed normalization. Returned values are detached and read-only
with deterministic hashes.

No Databricks request, Warehouse action, real data, feature forecast, model fit,
truth opening, GPU execution or remote write occurred. Every authority remains
false.

## Changed files

- `pfc_shaping/lt/evaluation_feature_builder.py`
  - new pure constructor;
  - SHA-256:
    `dc506bb85456f215ec588598c44dd757a8134e73e2498d5597fefe6440058504`.
- `tests/test_lt_evaluation_feature_builder.py`
  - parity, Swiss time/DST, holiday, role/timing, unit, missingness,
    immutability and prohibited-column tests;
  - SHA-256:
    `9fe4a5422edf675f58d576686ca0d96d5abf19fdc3856bc294d02ed5a3091b91`.
- `pfc_shaping/lt/evaluation_feature_inventory.py`
  - explicitly binds hydro fraction and 365.25-day maturity units;
  - semantic SHA-256:
    `83fabacc804de4201560877400fefa6974076f64d14f9738306a86bcf815a6fd`;
  - SHA-256:
    `fb46b71208bf820603e4c4d81552c5539e605908139f439e4b42ae1118de0a70`.
- `tests/test_lt_evaluation_feature_inventory.py`
  - verifies the two explicit units;
  - SHA-256:
    `9939bf9e1074016564da0057f3af4c65a4447104490d12206e30940f918dfa87`.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - documents constructor responsibility and failure semantics.
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
  - assigns unit conversion and climatology generation explicitly upstream.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D292 and updates D291's unit/hash evidence.
- `.planning/HANDOFF.md`
  - points to D292 and this handoff.

All preceding uncommitted ENTSO-E/LSEG and D289-D291 work remains preserved.

## Verification

Every shell action first verified both current directory and Git root as
`C:\Users\jbattaglia\PFC_LT`. Test receipts stayed below `build/` through
`scripts.run_workspace_local`.

- initial builder suite: `14 passed, 2 warnings`;
- after removing the test-only pandas assignment warning and formatting,
  builder/inventory/input suite: `29 passed, 1 warning`;
- expanded evaluation, PIT availability, materialization, LT-import and package
  matrix: `172 passed, 1 skipped, 1 warning`.

The remaining warning is the pre-existing unknown pytest `cache_dir` option.
The skipped case is the existing optional dependency/runtime test.

One initial read-only discovery command used the obsolete root
`pfc_shaping/calendar_ch.py` path and Windows wildcard literals. It produced no
write and was immediately corrected to `pfc_shaping/data/calendar_ch.py` with
`rg --glob`. The first Ruff format check reported two files; the governed
formatter changed only those files and all post-format tests passed.

## Next safe step

This step is now complete under D293 and
`SESSION-HANDOFF-20260904-CONSTRUCTED-INPUT-ASSEMBLY.md`.

Before implementing candidate execution, audit the interface mismatch between
the native incumbent pipeline and the generic challenger matrix; do not replace
the source-bound incumbent with an unregistered surrogate.

## Session/window status

No new window is required yet. Start a new one before the real-data runner if
the next adapter expands into execution orchestration; this handoff is complete
for exact resume.

## Invariants

- PRD is the enterprise-validated source boundary; upstream source recovery is
  Data Engineering responsibility.
- This constructor does not generate hydro climatology or convert its units.
- Missing hydro remains visible until the one common mask.
- Every candidate receives the same exact feature values and eligible rows.
- The CH monthly BASE solver remains the sole monthly-level authority.
- LT remains independent from CT, T057 remains sealed, and every model and
  production authority remains false.
