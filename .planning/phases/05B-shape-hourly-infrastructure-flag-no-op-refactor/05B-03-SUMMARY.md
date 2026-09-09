---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: "03"
subsystem: pfc-lt-shape-hourly
tags: [feature-flag, env-var, freeze-at-init, sidecar, no-op-refactor, shape-hourly, tdd]
dependency_graph:
  requires:
    - 05B-02 (shape_hourly.meta.parquet sidecar @ 3598646)
  provides:
    - pfc_shaping/lt/model/shape_hourly.py (feature flag _use_seasonal_hourly accepted, frozen, persisted, restored — gates ZERO behavior in 5bis-A)
    - tests/test_shape_hourly_infra.py (26 new TDD tests for flag mechanics)
  affects:
    - plans 05B-04..05B-05 (test infrastructure + regression tests)
    - plan 05B-B (bowl-deepening — will add behavioral branches gated by this flag)
tech_stack:
  added:
    - os (stdlib — single read of os.getenv at __init__ time)
  patterns:
    - "freeze-at-init: env var read once in __init__, stored in self._use_seasonal_hourly (bool)"
    - "constructor wins over env (D-06): explicit kwarg takes precedence over PFC_LT_USE_SEASONAL_HOURLY_SHAPE"
    - "parquet wins over env (D-07): load() restores from sidecar, preventing train/serve skew"
    - "_resolve_flag() private helper centralizes precedence logic — single callsite"
    - "cross-plan compat: missing use_seasonal_hourly key in old sidecar → _resolve_flag(None), no crash"
key_files:
  created: []
  modified:
    - pfc_shaping/lt/model/shape_hourly.py
    - tests/test_shape_hourly_infra.py
decisions:
  - "import os added at module level (consistent with D-06: single read site in _resolve_flag)"
  - "_FLAG_ENV_VAR = 'PFC_LT_USE_SEASONAL_HOURLY_SHAPE' exported constant (importable for tests)"
  - "Invalid env value emits logger.warning and defaults to False (D-06: default off)"
  - "use_seasonal_hourly persisted in hyperparams JSON with sort_keys=True (consistent with 05B-02 convention)"
  - "Existing 05B-02 tests updated to expect new use_seasonal_hourly key in hyperparams JSON — intentional schema extension"
metrics:
  duration: "~12 minutes"
  completed: "2026-05-18T21:25:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 0
  files_modified: 2
---

# Phase 05B Plan 03: Feature flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE (no-op)

## One-liner

Feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` introduced with correct mechanics: constructor arg + env-default, frozen at `__init__`, persisted into `shape_hourly.meta.parquet` sidecar, restored on `load()` with parquet winning over env — gates ZERO behavior in Phase 5bis-A.

## What Was Built

### Task 1 — Constructor arg + env-default + freeze-at-init

Added to `pfc_shaping/lt/model/shape_hourly.py`:

1. `import os` at module header (alongside existing `import json`, `import logging`)
2. `_FLAG_ENV_VAR = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"` — exported module-level constant (importable in tests)
3. `_resolve_flag(explicit: bool | None) -> bool` — private helper centralizing flag precedence:
   - `explicit is not None` → `bool(explicit)` (constructor wins, D-06)
   - Otherwise: `raw = os.getenv(_FLAG_ENV_VAR, "0")` — single read
   - `"1"` → True, `"0"` → False, anything else → `logger.warning(...)` + False
4. `__init__` extended with `use_seasonal_hourly: bool | None = None` (last kwarg, additive — all existing call sites unaffected)
5. `self._use_seasonal_hourly: bool = _resolve_flag(use_seasonal_hourly)` — frozen at construction time
6. Inline docstring comment: flag is resolved once and frozen, persisted by `save()`, overwritten by `load()`, gates NO behavior in 5bis-A

Env var is read exactly once (in `_resolve_flag`) and never again in `fit()`, `apply()`, `save()`, or anywhere else.

### Task 2 — Persist flag in meta sidecar + restore on load

Modified `save()`: hyperparams JSON dict now includes `"use_seasonal_hourly": bool(self._use_seasonal_hourly)` with `sort_keys=True`, extending the schema from 05B-02.

Modified `load()`: after restoring other hyperparams, if `"use_seasonal_hourly"` is present in the parsed dict: `obj._use_seasonal_hourly = bool(hp["use_seasonal_hourly"])` with inline comment `# parquet wins over env — prevents train/serve skew (D-07)`. If key is absent (cross-plan compat — parquet written by 05B-02 before 05B-03 was merged): leave `obj._use_seasonal_hourly` at the value set by `cls()` which already called `_resolve_flag(None)` — no crash, no KeyError.

### New tests in test_shape_hourly_infra.py (26 added)

- `TestFlagEnvVarConstant`: `_FLAG_ENV_VAR == "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"`
- `TestFlagSignature`: kwarg present, default is None
- `TestFlagEnvDefault`: unset→False, '1'→True, '0'→False, invalid→False+warning
- `TestFlagConstructorWins`: constructor arg beats env in both directions (D-06)
- `TestFlagFreezeAtInit`: env mutation after construction has no effect (4 tests)
- `TestFlagAttributeIsBool`: `type(sh._use_seasonal_hourly) is bool` for all paths
- `TestFlagPersistenceInSidecar`: JSON cell updated, all 4 expected keys present
- `TestFlagRestoredOnLoad`: parquet wins over env (D-07), cross-plan compat fallback
- `TestFlagNoOpContract`: flag ON/OFF produce same attribute structure

### Updated tests (05B-02 compatibility)

Two tests in `TestSaveSidecarSchema` and `TestSaveUnfitted` that asserted exact hyperparams JSON content were updated to expect the new `use_seasonal_hourly` key — intentional schema extension.

## Verification Results

All criteria satisfied:

- Task 1 verify script (flag mechanics): PASS (prints `OK`)
- Task 2 verify script (save/load roundtrip with parquet-wins): PASS (prints `OK`)
- `_FLAG_ENV_VAR == 'PFC_LT_USE_SEASONAL_HOURLY_SHAPE'`: PASS
- `grep -q "use_seasonal_hourly" shape_hourly.py`: PASS
- `grep -q "parquet wins over env" shape_hourly.py`: PASS
- `pytest tests/ -x`: **201 passed, 3 skipped** (was 175 before — 26 new tests)
- No-op contract: flag gates ZERO behavioral branches in 5bis-A

### No-Op Contract Verification

`_use_seasonal_hourly` attribute is set and frozen, persisted and restored — but no code path in `fit()`, `apply()`, `get()`, `get_for_horizon()`, `_fit_trends()`, or any other method reads `self._use_seasonal_hourly` to branch on behavior. Flag ON and flag OFF produce numerically identical outputs (verified conceptually — the behavioral branches belong to Phase 5bis-B).

## Commits

| Task | Phase | Commit | Files |
|------|-------|--------|-------|
| 1+2 RED | TDD failing tests | 29a4dfb | tests/test_shape_hourly_infra.py |
| 1+2 GREEN | Implementation | e4ab9ee | pfc_shaping/lt/model/shape_hourly.py, tests/test_shape_hourly_infra.py |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Two existing 05B-02 tests asserted exact hyperparams JSON without `use_seasonal_hourly`**

- **Found during:** Task 2 GREEN run (`test_hyperparams_row`, `test_save_unfitted_hyperparams_correct`)
- **Issue:** Tests written in 05B-02 asserted `{"halflife_days": ..., "hydro_weight_sigma": ..., "sigma": ...}` with no `use_seasonal_hourly` key. After adding the key in 05B-03, these tests failed with `AssertionError: left contains 1 more item: {'use_seasonal_hourly': False}`.
- **Fix:** Updated both tests to include the new key in the expected dict with appropriate comment explaining the 05B-03 extension.
- **Files modified:** `tests/test_shape_hourly_infra.py`
- **Commit:** e4ab9ee (included in GREEN commit)

**2. [Note] `grep -c "os.getenv\|os.environ"` counts 3 (plan specified "exactly 1")**

- **Issue:** The acceptance criterion `grep -c "os.getenv\|os.environ" shape_hourly.py` reports 3 because 2 of the matches are in docstring/comments (lines 52 and 62) and 1 is the actual code site (line 72 in `_resolve_flag`).
- **Resolution:** The intent of the criterion is satisfied: there is exactly ONE executable `os.getenv()` call. The two comment occurrences are documentation explaining the convention. This is not a code quality issue.

## TDD Gate Compliance

- RED gate: commit `29a4dfb` (`test(05B-03)`) ✓
- GREEN gate: commit `e4ab9ee` (`feat(05B-03)`) ✓
- REFACTOR: no cleanup required — implementation was clean on first pass

## Known Stubs

None — the flag attribute is fully wired (set, persisted, restored). The behavioral branches that the flag will eventually gate belong to Phase 5bis-B and are intentionally absent in this plan.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary changes. The feature flag reads one environment variable at process startup (frozen thereafter) and writes one extra JSON key to a local filesystem file. This is standard configuration hygiene with no security surface.

## Self-Check: PASS

- [x] `pfc_shaping/lt/model/shape_hourly.py` modified and committed (e4ab9ee)
- [x] `tests/test_shape_hourly_infra.py` updated and committed (29a4dfb RED, e4ab9ee GREEN)
- [x] Commits `29a4dfb` and `e4ab9ee` exist in git log
- [x] No file deletions in either commit
- [x] 201 passed, 3 skipped (full suite — up from 175)
- [x] Flag mechanics verified: env-default, constructor wins, freeze-at-init, parquet wins
- [x] No behavioral branches: flag is declarative only in 5bis-A
- [x] `05B-03-SUMMARY.md` created at correct path
