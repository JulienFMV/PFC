# Session handoff - challenger factor post-processing

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decisions: D-20260904-294 through D-20260904-296

## Outcome

The candidate learning and factor-output spaces are now aligned locally
without adding a model runner.

The source-bound incumbent remains native. The four challengers learn the
incumbent-equivalent `target_f_h` from the D295 pure constructor. Their raw
predictions now pass through one challenger-only pure postprocessor that
matches `ShapeHourlyMLP.apply`: floor at `0.1`, arithmetic mean normalization
within each Swiss-local day, then clip to `[0.4, 2.0]`.

The postprocessor resolves the candidate and origin from the frozen protocol,
requires a timezone-aware quarter-hour grid and rejects missing/nonfinite or
misaligned values. It rejects the incumbent ID explicitly, so the native
factor cannot be normalized twice. Raw values, delivery timestamps and final
factor values receive separate hashes; outputs are detached and read-only.

A synthetic fixed-predictor test compares the result directly with the
hash-bound native incumbent `apply` and passes exactly. No estimator is fitted.
The earlier D295 integration test already proves target/feature/common-input
population identity, so no duplicate parity-bundle abstraction was added.

No Databricks statement, Warehouse action, business-data read, real fit, truth
opening, GPU execution or remote write occurred. Every model, selection,
monthly-level, publication and production authority remains false.

## Changed files

- `pfc_shaping/lt/evaluation_factor_postprocess.py`
  - pure challenger-only factor normalization and immutable evidence;
  - SHA-256:
    `9a3976ff88559281cdc30f869b8b4994736840dd456a91099dade2589020558e`.
- `tests/test_lt_evaluation_factor_postprocess.py`
  - native parity, double-application prevention, fail-closed and authority
    tests;
  - SHA-256:
    `70870396ea6fd61c8f4846021076ddb6f79b6e61a2dc6d7db7790a85ff8d8fde`.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - documents the completed challenger factor seam.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D296.
- `.planning/HANDOFF.md`
  - points to D296 and this handoff.

All preceding uncommitted ENTSO-E/LSEG and D289-D295 work remains preserved.

## Verification

Every shell action first checked that current directory and Git root both
resolve exactly to `C:\Users\jbattaglia\PFC_LT`. Python tests and formatting
used `scripts.run_workspace_local` with repo-local mutable paths.

- initial focused postprocessor suite: `13 passed`;
- post-format expanded evaluation, feature-availability, materialization,
  LT/CT-import and package matrix: `211 passed, 1 skipped`;
- targeted Ruff check: pass;
- targeted Ruff format: pass after formatting;
- `git diff --check`: pass; only inherited line-ending notices were emitted.

The skipped case is the existing optional dependency/runtime boundary.

## Next safe step and window boundary

The next task is no longer another small transform. It is an audit of common
full-price assembly: determine the exact minimal seam through which native
incumbent `f_H` and postprocessed challenger `f_H` can use identical solver
monthly levels, `f_W`, quarter-hour context and permitted zero-mean layers
before producing EUR/MWh predictions for `evaluation_engine`.

Start a new window for that audit. Read this handoff after `AGENTS.md` and
`.planning/HANDOFF.md`. Begin with read-only interface tracing; do not alter
`production_phases.py` or the source-bound incumbent. Prefer a small evaluation
adapter over reusing the production entry point if repository evidence shows
that the latter would import unrelated fitting or authority paths.

Do not run real data, fit candidates, open future truth, start a Warehouse or
use the GPU during the audit. GPU remains available for a later explicitly
authorized parity-qualified training run, not for contract construction.

## Invariants

- The source-bound incumbent is replayed natively and never replaced by a
  surrogate.
- Challenger normalization occurs exactly once; incumbent normalization stays
  inside native `apply`.
- `f_H` remains a factor until identical full-price assembly is complete.
- PRD Databricks remains the enterprise-validated source boundary; upstream
  ENTSO-E recovery and freshness are Data Engineering responsibilities.
- The CH monthly BASE solver remains sole level authority.
- LT remains independent from CT, T057 remains sealed and every operational
  authority remains false.
