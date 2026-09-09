# Session handoff - common full-price assembly

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decision: D-20260904-297

## Outcome

The read-only assembly audit is closed with one small pure in-memory adapter.

`PFCAssembler.build` is the smallest safe common boundary. The source-bound
incumbent still calls native `ShapeHourlyMLP.apply`. Each challenger is already
in postprocessed `f_H`; a minimal proxy changes only that return value on a
shallow assembler copy and carries immutable copies of the incumbent's `f_W`
maps. No assembly formula, final projection or production orchestration was
duplicated.

All five candidates receive exactly the same solved monthly price mapping,
quoted products, frozen origin, UTC quarter-hour grid, ENTSO-E context and
hydro context. The adapter verifies bit-identical `B`, `f_S`, `f_W`, `f_Q`,
`f_WV`, `delta_wv` and `f_bridge`, requires the final solver projection and
checks each full EUR/MWh curve preserves monthly `B` means within `1e-6`.
Outputs are hash-bound, detached, read-only and accepted directly by
`SyntheticEvaluationSet`.

The frozen comparison excludes outage positions and rejects solar modulation,
electrification shaping, amplitude shrinkage, uncertainty output and legacy
level paths. Those are not authorized common evaluation layers. The permitted
common path is incumbent `f_W`, context-aware `f_Q`, water value, horizon
damping, near-term bridge, solver-month recentering and hard final product
projection.

No Databricks statement, Warehouse action, business-data read, monthly BASE
solver run, model fit, truth opening, scoring, ranking, GPU use or remote write
occurred. The synthetic tests did exercise the assembler's final hard-product
projection. Every model, selection, monthly-level-change, publication and
production authority remains false.

## Changed files

- `pfc_shaping/lt/evaluation_curve_assembly.py`
  - pure five-candidate common assembly adapter;
  - SHA-256:
    `a34b0324785b036a9b0735335c10392bf9225bd92795c22f033458ed403278f3`.
- `tests/test_lt_evaluation_curve_assembly.py`
  - native/challenger parity, common-layer, BASE/PEAK, scorer-shape,
    immutability and fail-closed tests;
  - SHA-256:
    `469c7c9868343831bdea382fb02650da98366ef28b65926a8106afd52f0c3738`.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - documents the completed common EUR/MWh seam.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D297.
- `.planning/HANDOFF.md`
  - points to D297 and this handoff.

The pre-existing dirty worktree and all preceding D267-D296 work remain
preserved.

## Protected-file evidence

- `pfc_shaping/lt/model/shape_hourly_mlp.py` was not modified:
  - raw SHA-256:
    `e11e991f9d1585f214870bcdbc32934485dc7b8cf1b8bce39127b0486f5130d8`;
  - normalized-LF SHA-256 remains the protocol-bound
    `8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef`.
- `pfc_shaping/pipeline/production_phases.py` was not modified:
  - SHA-256:
    `6dcdba561946747dbb8023ff799f72d1188c2898f56a7c47b568b6ac9016d712`.
- `git diff --` for both protected files was empty.

## Verification

Every shell action first checked that current directory and Git root both
resolve exactly to `C:\Users\jbattaglia\PFC_LT`. Python tests and formatting
used `scripts.run_workspace_local` with repo-local mutable paths.

- focused suite after final implementation: `36 passed`;
  receipt:
  `build/workspace-local-runs/d297test3/execution-receipt.json`;
- expanded 16-file evaluation/feature/materialization/monthly-solver/import/
  package matrix: `248 passed, 1 skipped` in 72.88 seconds;
  receipt:
  `build/workspace-local-runs/d297matrix/execution-receipt.json`;
- required minimum LT matrix: `58 passed, 1 skipped` in 6.89 seconds;
  receipt:
  `build/workspace-local-runs/d297ltmin/execution-receipt.json`;
- targeted Ruff check: pass;
  receipt:
  `build/workspace-local-runs/d297lint3/execution-receipt.json`;
- targeted Ruff format: pass after formatting;
- `git diff --check`: pass.

The skipped case is the existing optional dependency/runtime boundary.

## Residual boundary

No additional local assembly abstraction is justified. A real candidate runner
still requires separately governed origin registration, admitted PRD inputs,
authorized fits and later future-truth opening. Until those external gates are
closed, do not add a production hook, run the solver for evaluation, fit any of
the five candidates or call the scoring engine on real outcomes.

## Invariants

- The source-bound incumbent remains native and unmodified.
- Challenger normalization occurs exactly once before this adapter.
- Only `f_H` varies; monthly levels, `f_W`, quarter-hour context and admitted
  downstream layers remain common.
- Full EUR/MWh assembly occurs before monthly-neutralized scoring.
- The CH monthly BASE solver remains sole level authority.
- LT remains independent from CT, T057 remains sealed and every operational
  authority remains false.
