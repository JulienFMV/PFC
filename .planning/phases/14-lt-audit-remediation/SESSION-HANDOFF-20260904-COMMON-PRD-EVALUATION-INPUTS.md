# Session handoff - common PRD evaluation inputs

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decisions: D-20260904-289 and D-20260904-290

## Outcome

The benchmark roadmap now has two small, separate authority-negative
boundaries:

1. the transparent market-constrained seasonal model is the permanent primary
   promotion reference outside the five-model selection inventory;
2. one pure common-input adapter validates already materialized enterprise PRD
   frames before any real-data model runner exists.

For one exact frozen origin, the input adapter requires:

- exact and distinct training/prediction row identities;
- training delivery and maximum dependency availability strictly before the
  origin;
- prediction delivery at or after the origin and feature availability no later
  than the origin;
- no target column in the prediction frame;
- one explicit ordered feature inventory shared by the seasonal reference,
  incumbent and all four challengers;
- one deterministic complete-case mask per split before model execution;
- stable delivery/row-ID ordering, detached read-only arrays and hash-only
  bindings for source snapshots, identities and values.

The adapter does not choose the feature inventory. The current MLP has 12
historical variables, while newer PRD materializations also expose load, solar,
wind and flows. No union, substitution, neutral fill or model-specific mask was
invented. That feature decision is the next explicit modeling step.

No Databricks request, Warehouse action, source read, model fit, truth opening,
GPU execution or remote write occurred. Every model and production authority
remains false.

## Changed files

Seasonal-reference milestone:

- `pfc_shaping/lt/evaluation_reference.py`;
- `tests/test_lt_evaluation_reference.py`;
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`;
- `.planning/phases/14-lt-audit-remediation/PFC-FMV-PRODUCT-QUALITY-CHARTER-20260713.md`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` D289.

Common-input milestone:

- `pfc_shaping/lt/evaluation_inputs.py`
  - implementation SHA-256:
    `e020a3a393dc5b5efe9377124538ec08dfadb5c32cabb6d1b902d035f49f4331`;
- `tests/test_lt_evaluation_inputs.py`
  - test SHA-256:
    `2ecfc205606140f2fadd008031373a87036759807d30f33ee1d5276ee7f06dcf`;
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - documents the common PRD input boundary and unresolved feature inventory;
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`
  - links offline materialization to the downstream pure adapter;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D290;
- `.planning/HANDOFF.md`
  - points to this result and handoff.

All preceding uncommitted 4 September ENTSO-E/LSEG files remain in the same
working tree and must be preserved.

## Verification

Workspace guards verified both current directory and Git top-level before each
command.

- seasonal-reference test first failed with the expected missing-module import;
- seasonal-reference focused suite: `6 passed, 1 warning`;
- seasonal-reference adjacent matrix: `55 passed, 1 skipped, 1 warning`;
- common-input test first failed with the expected missing-module import;
- initial post-format test wrapper could not atomically rename its repo-local
  execution receipt under `build/workspace-local-runs/ltinput2` (`WinError 5`);
  no external path or elevation was used, and a fresh run ID succeeded;
- common-input focused suite: `9 passed, 1 warning`;
- combined input/reference/protocol/engine/challenger/materialization/LT-import
  matrix: `93 passed, 1 skipped, 1 warning`;
- targeted Ruff check and format check: pass;
- `git diff --check`: pass.

The warning is the pre-existing unknown pytest `cache_dir` option. The skipped
test is the existing optional dependency/runtime case.

## GPU boundary

GPU hardware is available and recorded for later qualified nonlinear work. It
is intentionally unused here because validation/masking has no material GPU
benefit. The current runtime v1 still provides:

- CPU float64 oracle for monthly solver, EEX repricing and every hard gate;
- deterministic CPU identity for LightGBM and the seasonal reference;
- GPU eligibility for frozen inference, scenario transform and repeated
  scoring after parity qualification;
- no GPU fit or model selection until a separate deterministic CPU/GPU parity
  receipt exists.

## Next safe step

Freeze the exact common feature inventory before real-data execution. The
decision must compare, feature by feature:

- the incumbent MLP's 12 inputs;
- the causal PRD features actually materializable at each origin;
- availability, coverage and forecast-horizon semantics;
- whether a feature is shared, candidate-specific (normally forbidden) or
  excluded.

The result should be a small metadata contract plus tests. It must not access
real truth, fit candidates or tune on the future holdout.

## Session/window status

No new window is required yet. The context and changed-file inventory remain
tractable. Start a new window before the real-data runner or GPU parity work if
the next feature-inventory milestone materially expands the diff; this handoff
is sufficient to resume exactly.

## Invariants

- Treat PRD as the enterprise source boundary; upstream ENTSO-E/API recovery is
  Data Engineering responsibility.
- Do not guess the feature inventory or permit model-specific complete-case
  populations.
- Keep the seasonal reference outside selection and the five candidates exact.
- Do not run real-data training, truth opening or GPU fitting yet.
- The CH monthly BASE solver remains the sole monthly-level authority.
- LT remains independent from CT, T057 remains sealed, and every model and
  production authority remains false.
