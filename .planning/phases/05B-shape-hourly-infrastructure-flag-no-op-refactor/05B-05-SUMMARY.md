---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: "05"
subsystem: pfc-lt-test-infrastructure
tags: [tests, conftest, baseline-regression, legacy-compat, no-op-proof, shape-hourly, ast-guard, SHP-01, SHP-04]
dependency_graph:
  requires:
    - 05B-01 (baseline_pfc_seed42.parquet @ 9cc959b)
    - 05B-02 (save/load sidecar @ 3598646)
    - 05B-03 (feature flag _use_seasonal_hourly @ e4ab9ee)
    - 05B-04 (factors_3d_ view + assembler capability check @ 43e5eb7)
  provides:
    - tests/conftest.py (autouse PFC_LT_* env-var hygiene)
    - tests/test_shape_hourly_infra.py (7 new standalone proof tests D-14..D-19 + §8)
    - tests/fixtures/shape_hourly_legacy.parquet (format-compatible mock, no sidecar)
    - tests/fixtures/f_W_legacy.parquet (legacy f_W companion)
    - tests/fixtures/_generate_legacy_fixture.py (deterministic generator, seed=42)
    - tests/fixtures/__init__.py (enables import from tests.fixtures)
  affects:
    - All future plans that add behavior to shape_hourly.py (test_no_hidden_behavior_branch guards)
    - Phase 5bis-B (will extend ALLOWED_FUNCTIONS when behavior gating is added)
tech_stack:
  added:
    - ast (stdlib) — AST scan in test_no_hidden_behavior_branch
    - shutil (stdlib) — temp dir copy in test_save_load_legacy_compat
  patterns:
    - "Autouse pytest fixture for env-var isolation (conftest.py autouse=True)"
    - "Parametrized baseline regression: @pytest.mark.parametrize('flag', [False, True])"
    - "AST guard pattern: walk FunctionDef nodes, deny Attribute references to _use_seasonal_hourly outside ALLOWED_FUNCTIONS"
    - "Format-compatible mock legacy fixture (labeled, not verbatim from main@28dfd65)"
    - "Reusable build_pfc(seed, flag) function extracted from main() in _generate_baseline.py"
key_files:
  created:
    - tests/conftest.py
    - tests/fixtures/shape_hourly_legacy.parquet
    - tests/fixtures/f_W_legacy.parquet
    - tests/fixtures/_generate_legacy_fixture.py
    - tests/fixtures/__init__.py
  modified:
    - tests/test_shape_hourly_infra.py (7 new tests + imports + module docstring)
    - tests/fixtures/_generate_baseline.py (build_pfc() extracted, main() delegates)
    - tests/fixtures/README.md (sections for legacy fixtures + build_pfc entry point)
decisions:
  - "build_pfc(seed, flag) extracted from main() in _generate_baseline.py for test reuse — no-op on parquet content (verified at atol=1e-12)"
  - "Legacy fixture produced as format-compatible mock (not verbatim main@28dfd65 artifact) — labeled per Codex review §6"
  - "atol=1e-12, rtol=0 is the default contract; check_freq=False + index.freq=None reset before comparison because parquet does not preserve DatetimeIndex.freq"
  - "ALLOWED_FUNCTIONS = {__init__, save, load, _resolve_flag} defined as module-level constant in test file for deliberate extension by future plans"
  - "tests/fixtures/__init__.py added (empty) to enable `from tests.fixtures._generate_baseline import build_pfc` from test files"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-18T20:38:28Z"
  tasks_completed: 3
  tasks_total: 3
  files_created: 5
  files_modified: 3
---

# Phase 05B Plan 05: Test Infrastructure for Phase 5bis-A (no-op proof)

## One-liner

Autouse env-var hygiene conftest, seven standalone proof tests D-14..D-19 + Codex §8 AST guard, and format-compatible legacy fixtures establishing that Phase 5bis-A is a numerical no-op (`assert_frame_equal(atol=1e-12)` for both flag=False and flag=True).

## What Was Built

### Task 1: `tests/conftest.py` — autouse PFC_LT_* env-var hygiene (D-12)

Created `tests/conftest.py` with a single `@pytest.fixture(autouse=True)` named `_pfc_lt_env_hygiene`:
- Snapshots all `PFC_LT_*` env-var keys before each test
- Yields to the test
- Restores snapshot: deletes keys added during test, restores keys that were present

38 lines total. Suite unchanged after addition: 239 passed, 3 skipped.

### Task 2: Legacy fixture parquets (format-compatible mock)

Created two files representing pre-5bis-A `ShapeHourly.save()` output (NO `.meta.parquet` sidecar):

- `tests/fixtures/shape_hourly_legacy.parquet`: 48 rows (2 cells × 24h), columns: `saison`, `type_jour`, `heure`, `f_H`, `n_obs` — matches save() schema at shape_hourly.py:443-455
- `tests/fixtures/f_W_legacy.parquet`: 5 rows (all TYPES_JOUR), columns: `type_jour`, `f_W`

Generator script `tests/fixtures/_generate_legacy_fixture.py`:
- Uses `np.random.seed(42)` for determinism
- Labeled as "FORMAT-COMPATIBLE MOCK" per Codex review §6 (not verbatim main@28dfd65 artifact)
- Does NOT import from pfc_shaping (byte-stable across future refactors)

Also added `tests/fixtures/__init__.py` (empty) to enable `from tests.fixtures._generate_baseline import build_pfc` imports from test files.

Suite still: 239 passed, 3 skipped.

### Task 3: Seven new proof tests + `build_pfc()` refactor

**`_generate_baseline.py` refactored:** `main()` body extracted into `build_pfc(seed: int = 42, flag: bool = False) -> pd.DataFrame`. `main()` now delegates to `build_pfc(42, False)`. Verified: re-running produces parquet identical to committed `baseline_pfc_seed42.parquet` at `atol=1e-12, rtol=0`.

**Seven new standalone test functions added to `tests/test_shape_hourly_infra.py`:**

| Test | D-XX | What it proves |
|------|------|----------------|
| `test_factors_3d_view_consistency` | D-14 | `factors_3d_[(s,tj,h)] == factors_[(s,tj)][h]` over all cells × 24h; TypeError on write |
| `test_save_load_full_roundtrip` | D-15 | All 10 attributes roundtrip at atol=1e-12: factors_, factors_by_year_, trend_per_hour_, f_W_seasonal_, f_W_, _climatological_fill, sigma, halflife_days, hydro_weight_sigma, _use_seasonal_hourly, global_factors_ |
| `test_save_load_legacy_compat` | D-16 | Legacy fixtures (no sidecar) load without crash, 1 warning emitted, defaults used |
| `test_flag_freeze_at_init` | D-17 | 4 combinatorial freeze sub-assertions; post-init env mutation has no effect |
| `test_flag_persisted_in_parquet` | D-18 | Parquet value wins over env at load time, both directions (True→env=0, False→env=1) |
| `test_baseline_regression[False]` | D-19 | build_pfc(flag=False) == baseline at atol=1e-12 (THE no-op proof) |
| `test_baseline_regression[True]` | D-19 | build_pfc(flag=True) == baseline at atol=1e-12 (confirms flag is plumbing only) |
| `test_no_hidden_behavior_branch` | Codex §8 | AST scan: _use_seasonal_hourly NOT in fit/apply/etc. — only in ALLOWED_FUNCTIONS |

**`ALLOWED_FUNCTIONS`** constant defined at module level: `{"__init__", "save", "load", "_resolve_flag"}` — deliberate extension point for Phase 5bis-B.

**Suite result: 247 passed, 3 skipped (250 tests collected).**

## Verification Results

All acceptance criteria satisfied:

- `test -f tests/conftest.py`: PASS
- `grep -q "PFC_LT_" tests/conftest.py`: PASS
- `grep -q "autouse=True" tests/conftest.py`: PASS
- `wc -l tests/conftest.py`: 38 ≤ 50: PASS
- `test -f tests/fixtures/shape_hourly_legacy.parquet`: PASS
- `test -f tests/fixtures/f_W_legacy.parquet`: PASS
- `test ! -e tests/fixtures/shape_hourly_legacy.meta.parquet`: PASS (no sidecar)
- `grep -q "FORMAT-COMPATIBLE MOCK" tests/fixtures/_generate_legacy_fixture.py`: PASS
- All 7 individual test assertions: PASS
- `pytest tests/ -x`: **247 passed, 3 skipped** (>= 149 required)
- `pytest tests/ --co -q | tail -1`: **250 tests collected** (>= 153 required)
- Source files `shape_hourly.py` and `assembler.py` NOT modified by this plan: PASS (last commit is `43e5eb7` from 05B-04)
- `python tests/fixtures/_generate_baseline.py` still runnable post-refactor: PASS
- Reproducibility at `atol=1e-12, rtol=0`: PASS

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | autouse PFC_LT_* env-var hygiene fixture | 00d7eb2 | tests/conftest.py |
| 2 | legacy fixture parquets (format-compatible mock) | d4ac0ef | tests/fixtures/_generate_legacy_fixture.py, shape_hourly_legacy.parquet, f_W_legacy.parquet |
| 3 | seven D-14..D-19 tests + AST guard + build_pfc() refactor | bfc147b | tests/test_shape_hourly_infra.py, tests/fixtures/_generate_baseline.py, tests/fixtures/README.md, tests/fixtures/__init__.py |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Parquet does not preserve DatetimeIndex.freq**

- **Found during:** Task 3 first run of `test_baseline_regression`
- **Issue:** `pd.testing.assert_frame_equal` failed with `AssertionError: (<15 * Minutes>, None)` — the freshly-built DataFrame has `freq=<15 * Minutes>` while the loaded baseline parquet has `freq=None` (parquet format does not preserve the `freq` attribute of `DatetimeIndex`)
- **Fix:** Reset `df_cmp.index.freq = None` before comparison; added inline comment explaining the known pandas/parquet behavior
- **Files modified:** `tests/test_shape_hourly_infra.py`
- **Commit:** bfc147b (part of Task 3)

**2. [Rule 1 - Bug] `pd.testing.assert_index_equal` has no `check_freq` kwarg in this pandas version**

- **Found during:** Task 3 development
- **Issue:** `pd.testing.assert_index_equal(..., check_freq=False)` raised `TypeError: got an unexpected keyword argument 'check_freq'` (parameter added in a later pandas version)
- **Fix:** Replaced `assert_index_equal` call with direct index equality checks: `len`, `all()`, `str(tz)`. The `assert_frame_equal` comparison was fixed via the `freq=None` reset approach above
- **Commit:** bfc147b

**3. [Rule 1 - Bug] `pd.testing.assert_series_equal` failed on `_climatological_fill` `name` attribute**

- **Found during:** Task 3 first run of `test_save_load_full_roundtrip`
- **Issue:** `_climatological_fill` Series saved in the sidecar is reloaded with `name="fill_pct"` (set by `pd.Series(..., name="fill_pct")` in load()), but the manually-set test Series has `name=None`
- **Fix:** Replaced `assert_series_equal` with direct `np.testing.assert_allclose` on values + explicit index comparison
- **Commit:** bfc147b

## Known Stubs

None — all tests use real computation paths (no hardcoded mock return values). The legacy fixtures are labeled as format-compatible mocks but the test code exercises the real `ShapeHourly.load()` code path.

## Threat Flags

None — this plan adds test infrastructure only (conftest, test module, fixture parquets, generator script). No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries.

## Self-Check: PASS

- [x] `tests/conftest.py` exists and contains autouse fixture (00d7eb2)
- [x] `tests/fixtures/shape_hourly_legacy.parquet` exists (d4ac0ef)
- [x] `tests/fixtures/f_W_legacy.parquet` exists (d4ac0ef)
- [x] `tests/fixtures/_generate_legacy_fixture.py` contains "FORMAT-COMPATIBLE MOCK" (d4ac0ef)
- [x] `tests/test_shape_hourly_infra.py` contains all 7 test functions (bfc147b)
- [x] `tests/fixtures/_generate_baseline.py` exposes `build_pfc()` (bfc147b)
- [x] Commits `00d7eb2`, `d4ac0ef`, `bfc147b` exist in git log
- [x] No file deletions in any of the three commits
- [x] Suite: 247 passed, 3 skipped (250 collected)
- [x] Source files `shape_hourly.py` and `assembler.py` NOT modified by this plan
- [x] `test_baseline_regression[False]` and `[True]` both pass at atol=1e-12 (no-op proven)
