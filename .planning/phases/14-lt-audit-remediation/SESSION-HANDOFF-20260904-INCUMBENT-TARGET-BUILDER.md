# Session handoff - incumbent-equivalent target builder

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decisions: D-20260904-294 and D-20260904-295

## Outcome

The first executable step behind the candidate interface is complete without
opening a model runner. A pure constructor now transforms already-materialized
direct CH quarter-hour prices into the exact hourly `target_f_h` learned by the
source-bound incumbent.

The constructor reproduces Swiss-local daily means, the strict `> 5 EUR/MWh`
day rule, price/day ratios, clipping to `[0.2, 3.0]`, 180-day within-hour
recency weighting and grouping by local date and clock hour. It deliberately
merges the repeated autumn hour because the frozen incumbent does. Each output
row uses the earliest contributing UTC timestamp and the latest contributing
price availability.

Output row IDs and identity/delivery/availability/value hashes are
deterministic. `to_metadata_frame()` produces the exact `target_f_h` input
accepted by the common D293/D294 assembly. The output arrays are detached and
read-only; callers cannot mutate the source or returned metadata and affect the
recorded values.

A synthetic parity test replaces only the sklearn estimator with a no-training
capture double, invokes the hash-bound native incumbent preprocessing and
compares the captured target and first nine feature positions group by group.
The comparison passes at `1e-15`, including the autumn repeated hour. No
estimator was trained.

No Databricks statement, Warehouse action, business-data read, real fit, truth
opening, GPU execution or remote write occurred. Every model, selection,
monthly-level, publication and production authority remains false.

## Changed files

- `pfc_shaping/lt/evaluation_target_builder.py`
  - pure target construction, immutable output and exact hashes;
  - SHA-256:
    `2e73bda5aaae8dc27c3b345cddae158e5e22f0f9f53dc16d21de29f13dfe85db`.
- `tests/test_lt_evaluation_target_builder.py`
  - native-capture parity, DST, threshold/clip/weighting, fail-closed,
    immutability and common-assembly integration tests;
  - SHA-256:
    `e97f2b85ecbc8a95453648a0ec25ffcfa941390fa43bf538eb1e4742706cd981`.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - records the implemented target boundary and parity method.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D295.
- `.planning/HANDOFF.md`
  - points to D295 and this handoff.
- `SESSION-HANDOFF-20260904-CANDIDATE-EXECUTION-CONTRACT.md`
  - clarifies the provisional missing-price rule now resolved from native code.

The D294 execution contract and corrected D290/D293 input files retain their
reported hashes. All preceding uncommitted ENTSO-E/LSEG and D289-D294 work
remains preserved.

## Verification

Every shell action first checked that current directory and Git root both
resolve exactly to `C:\Users\jbattaglia\PFC_LT`. Python tests and formatting
used `scripts.run_workspace_local` with repo-local mutable paths.

- initial target-builder suite: `11 passed`;
- target-builder plus common-input integration suite: `30 passed`;
- post-format expanded evaluation, feature-availability, materialization,
  LT/CT-import and package matrix: `198 passed, 1 skipped`;
- targeted Ruff check: pass;
- targeted Ruff format: pass after formatting;
- `git diff --check`: pass; only inherited line-ending notices were emitted.

The skipped case is the existing optional dependency/runtime boundary.

## Next safe step

Implement the second D294 item only: a pure parity bundle that combines the
constructed target rows with the D292 training feature rows and proves exact
feature/target population identity before the D290 common mask. Reuse the
existing builders and adapter; do not create another feature or masking path.

After that, audit the factor post-processing seam. The incumbent applies floor,
local-day normalization and final clipping internally, while challengers
currently expose raw predictions. A shared challenger-only postprocessor must
match native behaviour without double-applying it to the incumbent.

Do not yet add a real-data runner, fit candidates, open future truth, start a
Warehouse or use the GPU. No new window is required for the small parity
bundle. Start a new window before broad real-data execution or common
full-curve orchestration.

## Invariants

- The source-bound incumbent is replayed natively and never replaced by a
  surrogate.
- Missing source prices follow native exclusion; missing candidate features
  remain null until the one common input mask.
- Generic challengers learn `f_H`, not raw price, and are scored only after
  common full-price assembly.
- PRD Databricks remains the enterprise-validated source boundary; upstream
  ENTSO-E recovery and freshness are Data Engineering responsibilities.
- The CH monthly BASE solver remains sole level authority.
- LT remains independent from CT, T057 remains sealed and every operational
  authority remains false.
