# Session handoff - seasonal reference and GPU boundary

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decision: D-20260904-289

## Outcome

The deterministic benchmark ambiguity is closed without changing the frozen
five-model candidate inventory. The transparent market-constrained seasonal
model is now the permanent primary promotion reference, scored separately
before candidate ranking on the same origins, rows, masks, weights, metrics
and lead buckets. It is not a sixth candidate, receives no tuning grid and
cannot participate in model selection.

Only an authority-negative semantic companion contract was added. It performs
no data access, fit, scoring or ranking. The real-data implementation and
separate reference scorer remain pending the common PRD observation/feature
adapter.

The user-confirmed GPU capability is explicitly retained. The seasonal
reference and canonical LightGBM remain deterministic CPU identities. The
current compute contract still requires separate CPU/GPU parity qualification
before GPU fit or model selection; GPU availability does not replace the CPU
float64 hard-gate oracle.

No Databricks request, Warehouse action, data read, model fit, GPU execution or
remote write occurred.

## Changed files

- `pfc_shaping/lt/evaluation_reference.py`
  - adds the immutable seasonal-reference placement and scoring contract;
  - companion semantic SHA-256:
    `a0b1dd1f8add11086b1dba8e1b447377388b748a2221e9edd6029092001d0418`;
  - file SHA-256:
    `f869da02b5807cf2b6cc6d0fa136307c995c6f95d9991c6975295ff560cc26ae`.
- `tests/test_lt_evaluation_reference.py`
  - verifies placement outside the candidate inventory, exact shared metrics
    and buckets, solver-level invariants, GPU boundary, immutable negative
    authority, stable semantic hash and absence of data/CT access.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - documents the separate primary-reference lane and current GPU limits.
- `.planning/phases/14-lt-audit-remediation/PFC-FMV-PRODUCT-QUALITY-CHARTER-20260713.md`
  - makes the primary reference's non-candidate role explicit.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D-20260904-289.
- `.planning/HANDOFF.md`
  - points to this result and handoff.

All preceding uncommitted 4 September ENTSO-E/LSEG files remain in the same
working tree and must be preserved.

## Verification

Workspace guards verified both current directory and Git top-level before each
command.

- the new test initially failed during collection with the expected
  `ModuleNotFoundError` before implementation;
- focused reference tests: `6 passed, 1 warning`;
- adjacent reference/protocol/engine/challenger/LT-import matrix:
  `55 passed, 1 skipped, 1 warning`;
- targeted Ruff check and format check: pass;
- `git diff --check`: pass.

The warning is the pre-existing unknown pytest `cache_dir` option. The skipped
test is the existing optional dependency/runtime case.

## Next safe step

Specify the common PRD observation/feature adapter consumed identically by the
seasonal reference, incumbent and four challengers. Start with its metadata,
causal-cutoff, row-identity and complete-case contract plus synthetic tests;
do not open real training truth or fit models yet.

The current session still has sufficient context. A new window is not required
before that adapter-contract milestone; create a fresh handoff/window before
real-data fitting or GPU qualification if the accumulated context becomes
materially larger.

## Invariants

- Keep the exact five-model candidate inventory unchanged.
- The seasonal reference is primary but never selectable or tunable.
- GPU use requires deterministic runtime evidence and CPU/GPU parity; CPU
  remains the hard-gate oracle.
- The CH monthly BASE solver remains the sole monthly-level authority.
- Keep all data-acquisition, training, selection, publication and production
  authorities false.
- LT remains independent from CT and T057 remains sealed.
