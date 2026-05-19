---
phase: 05C-shape-hourly-bowl-deepening
plan: "01"
subsystem: model
tags: [shape_hourly, hydro_kernel, gaussian, flag_gating, fixture, sidecar, parquet]

requires:
  - phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
    provides: "flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE persisted in sidecar, freeze-at-init, baseline_pfc_seed42.parquet frozen, conftest env hygiene, 247 tests green"

provides:
  - "Lever 1 kernel reformulation: _apply_hydro_analogue_weights uses per-timestamp get_climatological_fill(woy(t)) target under flag=ON (D-A1-1)"
  - "Backward-compat flag=OFF path preserved bit-pour-bit (D-A1-2, atol=1e-12 SC #4 contract)"
  - "hydro_weight_sigma_off=0.25 / _on=0.08 flag-aware ctor args with resolution precedence (D-A1-4, D-A3-2)"
  - "Sidecar schema extended: hydro_weight_sigma_off/_on/_resolved keys + cross-plan load fallback (D-A3-3)"
  - "Deterministic bowl fixture: _generate_bowl_fixture.py + bowl_seed42.parquet (~167KB, seed=42, duck curve)"
  - "M2 cross-AI review fix: committed scripts/calibrate_bowl_thresholds.py + tests/fixtures/_bowl_calibration_report.json (sha256-linked, threshold loaded by test via json.load)"
  - "New test module tests/test_shape_hourly_bowl.py with D-A4-3 (kernel) and D-A4-8 (SC #4 baseline)"
  - "_last_clim_target_ debug attr on ShapeHourly for test verification (D-A4-3 Option A)"

affects:
  - "05C-02 — Lever 2 split level/anomaly depends on bowl_data fixture and _bowl_calibration_report.json"
  - "05C-03 — Lever 3 sigma ctor depends on extended sidecar schema; will re-run calibration script for all-3-lever threshold"

tech-stack:
  added: []
  patterns:
    - "per-timestamp climatological kernel target via vectorized woy lookup (≤52 dict calls, O(N) array fill)"
    - "flag-aware hydro_weight_sigma resolution with legacy-wins precedence (D-A3-2)"
    - "M2 auditability: committed calibration script + JSON report with fixture_sha256 tamper detection"
    - "SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER: placeholder value for Plan 05C-02 to overwrite"

key-files:
  created:
    - tests/fixtures/_generate_bowl_fixture.py
    - tests/fixtures/bowl_seed42.parquet
    - scripts/calibrate_bowl_thresholds.py
    - tests/fixtures/_bowl_calibration_report.json
    - tests/test_shape_hourly_bowl.py
  modified:
    - pfc_shaping/lt/model/shape_hourly.py
    - tests/test_shape_hourly_infra.py

key-decisions:
  - "hydro_weight_sigma_on=0.08: calibrated for ±10pp anomaly scale (vs legacy 0.25 on ±30pp); CV=0.393 equivalent to legacy 0.384 on dry-run N(0,0.10)"
  - "Option A for D-A4-3 test: _last_clim_target_ private debug attr set at end of kernel computation (avoids caplog dependency)"
  - "bowl_seed42.parquet covers 3 months (Jan-Mar 2022) — (Ete, Ouvrable) cell has 0 obs and falls back to global Ouvrable; SC1_PTP_THRESHOLD = 1.05 (plancher) as a result"
  - "M2 cross-AI review fix: calibration is committed script + JSON artifact, not interactive one-shot; fixture_sha256 enables tamper detection by Plan 05C-03 test"
  - "freq=None workaround applied in test_flag_off_bit_for_bit_baseline (parquet drops DatetimeIndex.freq, same pattern as test_baseline_regression in infra suite)"

patterns-established:
  - "Lever-gated behavior: flag=ON / flag=OFF conditional in _apply_hydro_analogue_weights with shared floor+NaN path after branching"
  - "Cross-plan sidecar compat: load() checks 'if key in hp' then reads off/on, else applies legacy single-value as both off and on"
  - "Wave 0 calibration as committed artifact: scripts/*.py + tests/fixtures/_*.json co-committed"

requirements-completed:
  - SHP-04
  - D-A1-1
  - D-A1-2
  - D-A1-3
  - D-A1-4
  - D-A1-5
  - D-A3-2
  - D-A3-3
  - D-A4-1
  - D-A4-2
  - D-A4-3
  - D-A4-8

duration: ~70min
completed: 2026-05-19
---

# Phase 05C Plan 01: Shape Hourly Bowl-Deepening Lever 1 Summary

**Gaussian kernel reformulation to per-timestamp climatological fill target under flag=ON, with M2-compliant committed calibration script + sha256-linked JSON threshold artifact, adding 2 tests (249 total).**

## Performance

- **Duration:** ~70 min
- **Started:** 2026-05-19T15:10:00Z
- **Completed:** 2026-05-19T16:21:00Z
- **Tasks:** 5 / 5
- **Files modified:** 7 (1 production, 1 test infra, 5 new)

## Accomplishments

- Lever 1 shipped: `_apply_hydro_analogue_weights` now uses `get_climatological_fill(woy(t))` per-timestamp as kernel target under `flag=ON`, replacing the legacy global scalar `current_fill`. flag=OFF path is bit-pour-bit identical to 5bis-A baseline (atol=1e-12, SC #4 verified).
- M2 cross-AI review fix (REVIEWS.md consensus #3) fully implemented: `scripts/calibrate_bowl_thresholds.py` is a committed reproducible script; `tests/fixtures/_bowl_calibration_report.json` carries `fixture_sha256` for tamper detection; test loads `SC1_PTP_THRESHOLD` via `json.load`, not from a free-floating comment.
- `hydro_weight_sigma_off=0.25` / `hydro_weight_sigma_on=0.08` flag-aware ctor args with legacy-wins resolution precedence (D-A3-2); sidecar extended with 3 new hydro_weight_sigma keys + cross-plan fallback at `load()`.
- Deterministic bowl fixture (`bowl_seed42.parquet`, seed=42, 8928 rows, sha256-stable) with analytically-controlled duck curve (solar -18/-25 EUR/MWh summer/WE, evening peak +22 EUR/MWh, night -8 EUR/MWh).

## Task Commits

1. **Task 1: Bowl fixture generator + bowl_seed42.parquet** - `70034ee` (feat)
2. **Task 2: Lever 1 kernel + ctor/save/load extensions** - `9ec5f18` (feat)
3. **Task 3: test_shape_hourly_infra.py Pitfall 4 update** - `43d9d81` (fix)
4. **Task 4: Calibration script + JSON report (M2)** - `34d9ba2` (feat)
5. **Task 5: test_shape_hourly_bowl.py D-A4-3 + D-A4-8** - `9db1f53` (feat)

## Files Created/Modified

- `pfc_shaping/lt/model/shape_hourly.py` — Lever 1 kernel refactor (4 surgical edits: __init__, _apply_hydro_analogue_weights, save, load); +127/-12 lines
- `tests/fixtures/_generate_bowl_fixture.py` — Deterministic bowl fixture generator (seed=42, duck curve)
- `tests/fixtures/bowl_seed42.parquet` — Committed synthetic fixture (8928 rows, 167KB, sha256-stable)
- `scripts/calibrate_bowl_thresholds.py` — M2-compliant committed calibration script
- `tests/fixtures/_bowl_calibration_report.json` — Immutable calibration artifact (sc1_ptp_ratio=1.0041, SC1_PTP_THRESHOLD=1.05)
- `tests/test_shape_hourly_bowl.py` — New isolated test module, 2 tests: D-A4-3 (kernel) + D-A4-8 (SC #4 baseline)
- `tests/test_shape_hourly_infra.py` — 3 test updates for extended sidecar schema + ALLOWED_FUNCTIONS extension

## Decisions Made

- **hydro_weight_sigma_on=0.08**: dry-run calibration on N(0,0.10) anomaly distribution; CV=0.393 matches legacy 0.384 on ±30pp. Deferred validation on real Swiss hydro data to Phase 10 (T1 deferred-research item acknowledged in plan).
- **_last_clim_target_ debug attr** (Option A): set at end of both kernel branches; not persisted, not public API. Enables direct test verification of D-A4-3 without caplog dependency.
- **SC1_PTP_THRESHOLD = 1.05 (plancher)**: the 3-month fixture covers only Jan-Mar (Hiver) so `(Ete, Ouvrable)` falls back to global Ouvrable cell. The Lever-1-only gain on this cell is 1.0041, which pushes threshold to the plancher. Plan 05C-03 Task 3 will re-run calibration with all 3 levers active and overwrite this value.
- **freq=None workaround**: applied in `test_flag_off_bit_for_bit_baseline` (same pattern as the existing `test_baseline_regression` in `test_shape_hourly_infra.py`). This is expected behavior — parquet drops `DatetimeIndex.freq`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] ALLOWED_FUNCTIONS extended in test_no_hidden_behavior_branch (AST guard)**
- **Found during:** Task 3 (`test_shape_hourly_infra.py` update)
- **Issue:** The 5bis-A AST guard (`test_no_hidden_behavior_branch`) checks that `_use_seasonal_hourly` is only accessed in `{"__init__", "save", "load", "_resolve_flag"}`. After Task 2 added the conditional branch in `_apply_hydro_analogue_weights`, the test failed with "Phase 5bis-A violation".
- **Fix:** Added `"_apply_hydro_analogue_weights"` to `ALLOWED_FUNCTIONS` with a traceability comment citing D-A1-2 and 05C-CONTEXT.md. This is the exactly the extension mechanism the test itself documents ("To allow a new function to reference the flag ... extend ALLOWED_FUNCTIONS deliberately").
- **Files modified:** `tests/test_shape_hourly_infra.py`
- **Verification:** `pytest tests/test_shape_hourly_infra.py` exits 0, 104 passed.
- **Committed in:** `43d9d81` (Task 3 commit)

**2. [Rule 1 - Bug] freq=None workaround in test_flag_off_bit_for_bit_baseline**
- **Found during:** Task 5 (new bowl test module)
- **Issue:** `assert_frame_equal(df_off, baseline)` failed with `(<15 * Minutes>, None)` — parquet drops DatetimeIndex.freq, causing a freq mismatch between the freshly-built DataFrame and the loaded fixture.
- **Fix:** Added `df_cmp = df_off.copy(); df_cmp.index.freq = None` before the assertion (exact same workaround as `test_baseline_regression` in `test_shape_hourly_infra.py`, documented in that test's comments).
- **Files modified:** `tests/test_shape_hourly_bowl.py`
- **Verification:** `pytest tests/test_shape_hourly_bowl.py` exits 0, 2 passed.
- **Committed in:** `9db1f53` (Task 5 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking (AST guard), 1 bug (freq mismatch))
**Impact on plan:** Both fixes were necessary and expected from plan context. No scope creep.

## Known Stubs

- `SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER = 0.50` in `tests/fixtures/_bowl_calibration_report.json` and loaded as `SC3_M30_AMPLITUDE_THRESHOLD` in `tests/test_shape_hourly_bowl.py`. This is an **intentional placeholder** — Plan 05C-02 Task 3 will update this key after the Lever 2 `_split_level_anomaly` implementation is calibrated. The stub does NOT prevent Plan 05C-01's goal from being achieved (the two tests in this plan do not use `SC3_M30_AMPLITUDE_THRESHOLD`).

## Issues Encountered

- SC1_PTP_THRESHOLD landed at the plancher (1.05) because the 3-month bowl_seed42 fixture covers only January-March (Hiver), so the `(Ete, Ouvrable)` cell has 0 observations and falls back to the global Ouvrable profile. Lever 1 gain on the Ouvrable fallback profile is modest (1.0041). Plan 05C-03 Task 3 will re-run calibration after all 3 levers are shipped — by then the `(Ete, Ouvrable)` cell gain should reflect the full summer duck curve amplification.

## Deferred Research

- **T1** (tracked in `<deferred_research>` of the plan): `hydro_weight_sigma_on=0.08` calibrated analytically from simulated N(0,0.10) anomaly distribution. Validation on real Swiss BFE/OFEN weekly reservoir data (one-off offline artifact, quantiles + floor-hit rate + recommendation) should be done before D-FLIP-1 production flag flip. Does NOT block 5bis-B ship — `flag=OFF` default + Phase 10 MAE gate provide the safety net.

## Next Phase Readiness

- Plan 05C-02 (Lever 2: `_split_level_anomaly` + assembler integration) can proceed: `bowl_data` fixture available via `build_bowl_fixture()`, `_bowl_calibration_report.json` scaffold in place for SC3 threshold update.
- Plan 05C-03 (Lever 3: `sigma_off/_on` ctor + new baseline `baseline_pfc_seed42_bowl.parquet`) depends on 05C-01 + 05C-02 both complete.

---
*Phase: 05C-shape-hourly-bowl-deepening*
*Plan: 01*
*Completed: 2026-05-19*

## Self-Check: PASSED

Files verified:
- `pfc_shaping/lt/model/shape_hourly.py` — FOUND
- `tests/fixtures/_generate_bowl_fixture.py` — FOUND
- `tests/fixtures/bowl_seed42.parquet` — FOUND
- `scripts/calibrate_bowl_thresholds.py` — FOUND
- `tests/fixtures/_bowl_calibration_report.json` — FOUND
- `tests/test_shape_hourly_bowl.py` — FOUND
- `tests/test_shape_hourly_infra.py` — FOUND (modified)

Commits verified:
- `70034ee` — FOUND (feat: bowl fixture)
- `9ec5f18` — FOUND (feat: kernel refactor)
- `43d9d81` — FOUND (fix: infra test update)
- `34d9ba2` — FOUND (feat: calibration script + JSON)
- `9db1f53` — FOUND (feat: test_shape_hourly_bowl.py)
