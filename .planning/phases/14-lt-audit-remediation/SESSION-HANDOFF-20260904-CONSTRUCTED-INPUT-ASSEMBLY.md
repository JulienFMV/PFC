# Session handoff - constructed input assembly

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decisions: D-20260904-289 through D-20260904-293

## Outcome

Constructed hourly feature batches can now be assembled safely with PRD
observation metadata and delegated to the existing common-input validator.

Training metadata is exactly:

- `row_id`;
- `delivery_at_utc`;
- `target_available_at_utc`;
- `target_eur_mwh`.

Prediction metadata is exactly `row_id` and `delivery_at_utc`; prediction truth
cannot enter. Each delivery population must match the corresponding constructed
batch exactly and uniquely. Feature rows are aligned by canonical UTC timestamp,
not by caller order.

Training `available_at_utc` is computed as the elementwise maximum of target and
hydro availability. Prediction uses the admitted aligned hydro-climatology
availability. Callers cannot inject an unexplained aggregate timestamp.

Before assembly, each constructed batch is revalidated for origin, split,
feature inventory, shape, deterministic finiteness, hydro fraction domain,
read-only state, negative authority and exact value/delivery hashes. The result
then passes through the single existing complete-case mask and deterministic
ordering in `prepare_prd_origin_inputs`; no second population rule exists.

No Databricks request, Warehouse action, real data, model fit, truth opening,
GPU execution or remote write occurred. Every model and production authority
remains false.

## Changed files

- `pfc_shaping/lt/evaluation_inputs.py`
  - adds `prepare_constructed_prd_origin_inputs` and internal integrity/alignment
    checks;
  - SHA-256:
    `e020a3a393dc5b5efe9377124538ec08dfadb5c32cabb6d1b902d035f49f4331`.
- `tests/test_lt_evaluation_inputs.py`
  - adds reordered alignment, exact-population, aggregate-availability,
    split/origin, hash-tamper and authority tests;
  - SHA-256:
    `2ecfc205606140f2fadd008031373a87036759807d30f33ee1d5276ee7f06dcf`.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - documents the exact assembly contract.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D293 and refreshes the extended input-module hashes in D290/D291.
- `.planning/HANDOFF.md`
  - points to D293 and this handoff.
- preceding D290-D292 handoffs
  - refresh current extended file hashes or point to the completed successor
    step; no historical execution result was changed.

All preceding uncommitted ENTSO-E/LSEG and D289-D292 work remains preserved.

## Verification

Every shell action first verified both current directory and Git root as
`C:\Users\jbattaglia\PFC_LT`. Test receipts stayed below `build/` through
`scripts.run_workspace_local`.

- initial assembly-focused input suite: `17 passed, 1 warning`;
- after batch-integrity revalidation: `18 passed, 1 warning`;
- post-format builder/inventory/input suite: `37 passed, 1 warning`;
- expanded evaluation, PIT availability, materialization, LT-import and package
  matrix: `180 passed, 1 skipped, 1 warning`.

The remaining warning is the pre-existing unknown pytest `cache_dir` option.
The skipped case is the existing optional dependency/runtime test.

The first expanded-matrix wrapper call used run ID `ltassemblematrix1`, longer
than the wrapper's 16-character portable-ID limit. It failed in the repo-local
precheck before pytest or any project execution. The corrected `ltasmatrix1`
run passed. Ruff formatting changed only `evaluation_inputs.py`; all post-format
tests passed.

## Next safe step

Audit the candidate-execution interface before coding it. The native incumbent
does not consume a prebuilt generic nine-column matrix: it reconstructs 12
positions internally, aggregates quarter-hours and fits its own target path.
The four synthetic challengers consume a generic matrix. Replacing the native
incumbent with a convenient generic MLP would break its source-bound identity;
feeding real prepared values into the synthetic-only challenger API would
misstate provenance.

The next decision must therefore choose and prove one of two materially
different paths from repository evidence:

1. a native incumbent replay plus rigorously equivalent challenger observation
   construction; or
2. a newly registered common estimator interface, explicitly treated as a new
   candidate rather than the existing incumbent.

Do not guess this choice and do not fit real data. Independently registered
prospective origins and governed snapshot evidence remain required for later
real execution.

## Session/window status

No new window is required for the interface audit. Start a new window before a
real-data runner or broad execution orchestration; this handoff supports exact
resume.

## Invariants

- PRD is the enterprise-validated source boundary; upstream source recovery is
  Data Engineering responsibility.
- Timestamp alignment is exact; position-only joins are forbidden.
- Only the D290 validator forms the complete-case population.
- Synthetic-only interfaces cannot be relabelled as real-data execution paths.
- The source-bound incumbent cannot be replaced silently by a surrogate.
- The CH monthly BASE solver remains the sole monthly-level authority.
- LT remains independent from CT, T057 remains sealed, and every model and
  production authority remains false.
