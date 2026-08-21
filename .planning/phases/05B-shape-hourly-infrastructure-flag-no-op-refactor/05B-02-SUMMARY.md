---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: "02"
subsystem: pfc-lt-shape-hourly
tags: [save-load, sidecar, roundtrip, no-op-refactor, shape-hourly, bug-fix]
dependency_graph:
  requires:
    - 05B-01 (baseline_pfc_seed42.parquet frozen @ 9cc959b)
  provides:
    - pfc_shaping/lt/model/shape_hourly.py (complete save/load roundtrip via meta sidecar)
    - tests/test_shape_hourly_infra.py (32 TDD tests for sidecar behavior)
  affects:
    - plans 05B-03..05B-05 (flag + test infrastructure + regression test)
tech_stack:
  added:
    - json (stdlib — for hyperparams serialization in sidecar)
  patterns:
    - "${stem}.meta.parquet" sidecar convention (model-specific stem avoids collision with sibling components)
    - Long-format parquet with `attr` discriminator column for heterogeneous attributes
    - All `value` cells stored as repr(float)/JSON strings to allow mixed types in single parquet file
    - TDD RED/GREEN cycle with separate commits per phase
key_files:
  created:
    - tests/test_shape_hourly_infra.py
  modified:
    - pfc_shaping/lt/model/shape_hourly.py
decisions:
  - "All value cells in meta sidecar stored as strings (repr(float) for numerics, JSON for hyperparams) to avoid pyarrow ArrowInvalid on mixed-type columns — cast back at load time"
  - "global_factors_ intentionally NOT persisted to meta sidecar — reconstructed deterministically from factors_ via _compute_global_fallback() at load time"
  - "Legacy parquet (no sidecar) loads without crash, emits exactly one logger.warning naming all missing attributes"
  - "No-op contract confirmed: freq metadata mismatch between fresh result and baseline is pre-existing (parquet does not preserve DatetimeIndex.freq), numerical values identical at atol=1e-12"
metrics:
  duration: "~18 minutes"
  completed: "2026-05-18T20:06:40Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 1
---

# Phase 05B Plan 02: Fix ShapeHourly save/load bug via meta sidecar

## One-liner

Pre-existing ShapeHourly save/load bug fixed: `shape_hourly.meta.parquet` sidecar now persists `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, and scalar hyperparams; `global_factors_` reconstructed deterministically; legacy parquets load with warning.

## What Was Built

### Task 1: Extend `save()` to write `${stem}.meta.parquet` sidecar

Added to `pfc_shaping/lt/model/shape_hourly.py`:

1. `import json` at module top (stdlib, no new dependency)
2. `_META_SIDECAR_SUFFIX = ".meta.parquet"` — module-level constant with rationale comment (cross-AI review consensus to avoid name collision with future `ShapeIntraday` sidecar)
3. `_meta_path(main_path) -> Path` — private helper that derives `${stem}.meta.parquet` from the main artifact path, centralizing the naming convention so `save()` and `load()` cannot drift
4. Extended `save()` to write the sidecar after the existing `f_W.parquet` block:
   - `factors_by_year_` → long-format rows (saison, type_jour, year, heure, value)
   - `trend_per_hour_` → long-format rows (saison, type_jour, heure, value)
   - `f_W_seasonal_` → 1 row per (saison, type_jour)
   - `_climatological_fill` → rows with (week, value), absent if None
   - `_hydro_fill_weekly` → rows with (timestamp ISO-8601, value), absent if None
   - `hyperparams` → single JSON-string row with sigma, halflife_days, hydro_weight_sigma (sorted keys)
   - **`global_factors_` NOT written** — reconstructed at load via `_compute_global_fallback()`
5. All `value` cells stored as `repr(float)` / JSON strings to resolve pyarrow `ArrowInvalid` on mixed-type columns (floats + JSON strings in one column)

### Task 2: Extend `load()` to restore from sidecar

Extended `load()` classmethod to:

1. Compute `meta_path = _meta_path(path)` after loading the main factors + f_W
2. **If meta sidecar exists:**
   - Parse `hyperparams` JSON row → assign `sigma`, `halflife_days`, `hydro_weight_sigma`
   - Reconstruct `factors_by_year_` by groupby (saison, type_jour, year), sort by heure, apply float cast
   - Reconstruct `trend_per_hour_` similarly
   - Reconstruct `f_W_seasonal_` dict (float cast per row)
   - Restore `_climatological_fill` as `pd.Series(index=week_int)` if rows present, else None
   - Restore `_hydro_fill_weekly` as UTC-tz `pd.Series` indexed by `pd.to_datetime(..., utc=True)`
3. **If meta sidecar absent (legacy):** emit exactly one `logger.warning` naming all 7 missing attributes, use constructor defaults — no exception raised
4. Inline comment at `_compute_global_fallback()` call: `# global_factors_ is intentionally NOT persisted to the meta sidecar — reconstructed deterministically from factors_ (see Plan 05B-02 review consensus).`
5. Signature unchanged: `load(cls, path)` only

## Verification Results

All criteria satisfied:

- `_META_SIDECAR_SUFFIX == '.meta.parquet'`: PASS
- `_meta_path('shape_hourly.parquet').name == 'shape_hourly.meta.parquet'`: PASS
- Task 1 verify script (`save()` + meta content): PASS (prints `OK`)
- Task 2 verify script (`load()` roundtrip + legacy compat): PASS (prints `OK`)
- `load()` signature unchanged (`['path']`): PASS
- `grep -c "logger.warning" shape_hourly.py` = 5 (was 4 before — one new legacy warning): PASS
- `grep -q "global_factors_ is intentionally NOT persisted"`: PASS
- `pytest tests/ -x` = **175 passed, 3 skipped** (143 original + 32 new infra tests): PASS

### No-Op Contract Verification

Fresh fit+predict (no save/load) produces numerically identical output to `baseline_pfc_seed42.parquet` at `atol=1e-12, rtol=0`. Confirmed that the freq metadata mismatch (`<15 * Minutes>` vs `None`) is **pre-existing** and present in the original code before Plan 05B-02 — it is caused by parquet not preserving DatetimeIndex.freq and is not related to our changes. The numerical values of `price_shape` are bit-for-bit identical.

### Save→Load→Predict Roundtrip

Manually verified: `sh1.fit(...).save(p)` → `sh2 = ShapeHourly.load(p)` → `sh2.sigma`, `sh2.factors_by_year_`, `sh2.trend_per_hour_`, `sh2.f_W_seasonal_`, `sh2._climatological_fill`, `sh2.global_factors_` all numerically equal at `atol=1e-12, rtol=0`.

Double roundtrip (fit → save → load → save → load) also verified identical.

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1+2 RED | TDD failing tests for save/load meta sidecar | 238d23b | tests/test_shape_hourly_infra.py |
| 1+2 GREEN | Implement save/load sidecar fix | 3598646 | pfc_shaping/lt/model/shape_hourly.py |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mixed-type column causes pyarrow ArrowInvalid on to_parquet()**

- **Found during:** Task 1 implementation (first GREEN run)
- **Issue:** The `value` column contained float values (for numerical attrs) AND a JSON string (for hyperparams). PyArrow cannot unify these into a single type: `ArrowInvalid: Could not convert '{"halflife_days": ...}' with type str: tried to convert to double`.
- **Fix:** Store ALL `value` cells as strings — `repr(float(v))` for numerical values, `json.dumps(...)` for hyperparams. Parse back to appropriate type at load time via `.apply(float)` / `json.loads()`. This avoids any new dependency and is transparent to callers.
- **Files modified:** `pfc_shaping/lt/model/shape_hourly.py` (save + load blocks)
- **Commit:** 3598646 (included in GREEN commit)

### Out-of-Scope Discovery

The `hash()` of `price_shape.values.tobytes()` in `_generate_baseline.py` changes between Python process invocations due to Python's PYTHONHASHSEED randomization (since Python 3.3). This is a **pre-existing non-determinism in the hash display only** — the actual parquet numerical content is reproducible. Documented in `deferred-items.md` below for future attention, but NOT fixed in this plan (out-of-scope of 05B-02).

## TDD Gate Compliance

- RED gate: commit `238d23b` (`test(05B-02)`) ✓
- GREEN gate: commit `3598646` (`feat(05B-02)`) ✓
- REFACTOR: no cleanup required — implementation was clean on first pass (post bug fix)

## Known Stubs

None — all attributes are fully wired. The `_hydro_fill_weekly` path is exercised only when hydro_df is provided; its None sentinel is correctly handled in both save and load.

## Threat Flags

None — this plan only adds persistence to existing in-memory attributes. No new network endpoints, auth paths, file access patterns outside the model save/load directory, or schema changes at trust boundaries.

## Self-Check: PASS

- [x] `pfc_shaping/lt/model/shape_hourly.py` modified and committed (3598646)
- [x] `tests/test_shape_hourly_infra.py` created and committed (238d23b)
- [x] Commits `238d23b` and `3598646` exist in git log
- [x] No file deletions in either commit
- [x] 175 passed, 3 skipped (full suite)
- [x] No-op contract: numerical values identical at atol=1e-12 (freq metadata pre-existing issue noted)
- [x] Legacy compat: load without sidecar emits exactly 1 warning, no crash
