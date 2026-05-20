---
phase: 05-msfc-log-prix-retire-silent-floors
plan: "01"
subsystem: pfc-lt-negative-prices
tags:
  - msfc
  - arbitrage-free
  - negative-prices
  - enforce-positivity
  - signed-clamp
  - ctor-args
dependency_graph:
  requires:
    - "Phase 5bis-A (baseline frozen, conftest autouse, shape_hourly.meta.parquet sidecar)"
    - "Phase 5bis-B (bowl deepening — gates Plan 05-03 SC #2 acceptance test)"
  provides:
    - "enforce_positivity kwarg on smooth_base_prices + _enforce_mean_constraints (NEG-01)"
    - "signed-aware extrapolation clamp with degenerate-knot margin floor (D-A1-2 + codex action #3)"
    - "enforce_m_factor_floor kwarg on ArbitrageFreeCalibrator (NEG-02)"
    - "converged=False + reason-tagged INFO logs for floor-induced non-convergence (codex action #6)"
    - "NEG-05 wording reformulated in REQUIREMENTS.md (D-A4-7)"
    - "Phase 5 test scaffold: 5 passing + 9 stubbed-skipped (Plans 02+03)"
    - "conftest.py: PFC_LT_ALLOW_NEGATIVE_PRICES documented inline"
  affects:
    - "Plan 05-02 (WaterValueCorrection delta-additif — depends on API stable from this plan)"
    - "Plan 05-03 (master flag + baselines + acceptance test — depends on full ctor-args API)"
tech_stack:
  added: []
  patterns:
    - "ctor arg default False (negative-ready) — D-A2-1 convention introduced for Phase 5"
    - "kwarg propagation to helpers (enforce_positivity to _enforce_mean_constraints) to condition BOTH floors"
    - "signed-aware numpy clamp: margin = max(0.5 * np.ptp(y_knots), 1.0)"
    - "reason-tagged INFO log via extra={'reason': '...'} on LogRecord (codex action #6)"
key_files:
  created:
    - path: "tests/test_phase05_negative_prices.py"
      description: "Phase 5 test scaffold: 14 tests total, 5 passing, 9 stubbed-skipped"
  modified:
    - path: ".planning/REQUIREMENTS.md"
      description: "NEG-05 wording reformulated per D-A4-7 (monthly forward négatif, not Cal annuel)"
    - path: "pfc_shaping/lt/model/msfc_spline.py"
      description: "enforce_positivity kwarg + signed clamp + helper propagation"
    - path: "pfc_shaping/calibration/arbitrage_free.py"
      description: "enforce_m_factor_floor kwarg + converged=False + reason-tagged logs"
    - path: "tests/conftest.py"
      description: "Inline doc comment listing PFC_LT_ALLOW_NEGATIVE_PRICES alongside PFC_LT_USE_SEASONAL_HOURLY_SHAPE"
decisions:
  - "D-A2-1: All 4 floor ctor args default False (negative-ready). This plan ships the first two: enforce_positivity + enforce_m_factor_floor."
  - "D-A1-2 + codex action #3: Signed-aware clamp margin = max(0.5 * np.ptp(y_knots), 1.0) prevents inverted bounds for all-negative knots AND prevents clamp collapse for degenerate equal knots."
  - "NEG-02 + codex action #6: converged=False propagated when floor mutates m_factor; reason-tagged INFO logs distinguish 'm_factor_floor_hit' from 'iteration_limit' for operator triage."
  - "D-A4-7: NEG-05 reformulated from 'Cal annuel negatif' (non-realiste) to 'monthly forward negatif July M-07 = -2 EUR/MWh' (test realiste confirme par D-A4-7/CONTEXT.md)."
  - "Tasks 5+6 were implemented inline within Task 4 (scaffold created all 5 populated tests directly rather than stub-then-flip) — valid per plan note 'If Tasks 2+3 ship before Task 4, scaffold can include populated bodies directly'."
metrics:
  duration_minutes: 25
  completed_date: "2026-05-20"
  tasks_completed: 6
  tasks_total: 6
  files_modified: 4
  files_created: 1
---

# Phase 5 Plan 01: MSFC enforce_positivity + ArbitrageFreeCalibrator enforce_m_factor_floor Summary

**One-liner:** MSFC and ArbitrageFreeCalibrator gain `enforce_positivity=False` and `enforce_m_factor_floor=False` ctor args (defaults OFF = negative-ready) with signed-aware extrapolation clamp, NEG-02 converged=False propagation, reason-tagged INFO logs, and Phase 5 test scaffold (5 passing, 9 stubbed-skipped).

## Commits

| Hash | Type | Description |
|------|------|-------------|
| f10c524 | docs | Reformulate NEG-05 wording per D-A4-7 in REQUIREMENTS.md |
| 3911f3f | feat | enforce_positivity kwarg + signed-aware clamp in msfc_spline.py |
| fda761e | feat | enforce_m_factor_floor kwarg + converged=False + reason logs in arbitrage_free.py |
| 93895d0 | test | Phase 5 test scaffold: 5 passing + 8 stubbed-skipped tests |
| 9da519e | test | 14th stub test (codex action #7) + conftest.py Phase 5 doc comment |

## What Was Built

### Task 1: NEG-05 wording reformulation (.planning/REQUIREMENTS.md)

Replaced "Un Cal'27 forward coté -10 €/MWh" (non-réaliste — un Cal annuel n'est jamais négatif en pratique) par "Un monthly forward négatif (e.g., July M-07'27 = -2 €/MWh, autres months positifs typiques EEX)" per D-A4-7. Added provenance note (2026-05-19) and D-A4-7 decision-ID inline for traceability.

### Task 2: MSFC signed-aware (pfc_shaping/lt/model/msfc_spline.py)

Three surgical edits:

**A. `smooth_base_prices` signature:** Added `enforce_positivity: bool = False` kwarg. Docstring updated with full rationale (D-A2-1, NEG-01, RESEARCH Pitfall 1 warning about TWO floors).

**B. Signed-aware extrapolation clamp (line 120):** Replaced broken formula `np.clip(..., y_knots.min()*0.5, y_knots.max()*2.0)` (inverted bounds for all-negative knots — e.g. `[-30,-20,-25]` → `[-15,-40]`, lo > hi) with:
```python
margin = max(0.5 * float(np.ptp(y_knots)), 1.0)
B_smooth_raw = np.clip(B_smooth_raw, y_knots.min() - margin, y_knots.max() + margin)
```
The `max(..., 1.0)` floor (codex action #3) prevents clamp collapse when all knots are equal (np.ptp == 0 → margin would be 0 without the floor, pinning B_smooth_raw to a single value).

**C. Dual floor propagation (lines 131 + 203):** Both floor #1 (`np.maximum(B_smooth, 1.0)` after `_enforce_mean_constraints` call) and floor #2 (`return np.maximum(result, 1.0)` inside `_enforce_mean_constraints`) are now conditional on `enforce_positivity`. The kwarg is propagated to `_enforce_mean_constraints` — critical RESEARCH Pitfall 1 fix: if only floor #1 is conditioned, floor #2 remains active silently and `test_msfc_signed_monthly_repricing` would fail with `mean ≈ 1.0` instead of `-2.0`.

### Task 3: ArbitrageFreeCalibrator floor (pfc_shaping/calibration/arbitrage_free.py)

**A. `__init__` signature:** Added `enforce_m_factor_floor: bool = False` kwarg stored as `self.enforce_m_factor_floor`. Docstring updated with Phase 5 rationale (D-A2-1, NEG-02, codex action #6).

**B. Conditional clip at m_factor (line 517):** Added `floor_applied = False` before the `if self.mode == "multiplicative":` block. Within the multiplicative path, the `np.maximum(m_factor, 0.1)` is now wrapped in `if self.enforce_m_factor_floor:` with `floor_applied` tracking actual mutation (n_clipped > 0).

**C. Reason-tagged INFO logs (codex action #6):** Two independent INFO log records:
- When `max_abs_residual > self.tol`: `extra={"reason": "iteration_limit"}`
- When `floor_applied=True`: `extra={"reason": "m_factor_floor_hit"}`, `converged = False` forced

Pipeline callsite audit (RESEARCH Open Question #1): `pfc_shaping/pipeline/production_phases.py:550` uses `ArbitrageFreeCalibrator(smoothness_weight=1.0, tol=0.01)` — mode defaults to "multiplicative". No other callsite found. The enforce_m_factor_floor kwarg has no effect in additive mode (no m_factor there) but is stored unconditionally for traceability.

### Task 4: Phase 5 test scaffold (tests/test_phase05_negative_prices.py)

Created with 14 tests total:

**5 populated and passing (Plan 05-01 scope):**
1. `test_msfc_signed_monthly_repricing`: July 2027 = -2 EUR/MWh forward → mean(B_smooth[July]) ≈ -2.0 (atol=0.01). Verifies BOTH floors disabled (Pitfall 1 critical).
2. `test_arbitrage_free_signed_target`: Negative target (-10 EUR/MWh) with enforce_m_factor_floor=False → converged=True, max_abs_residual < tol.
3. `test_msfc_clamp_all_equal_knots`: All-equal knots (np.ptp==0) → margin=1.0 floor prevents clamp collapse → result is finite and within ±1.5 of knot value.
4. `test_msfc_clamp_all_negative_knots_no_inverted_bounds`: All-negative knots → no inversion, no NaN, result is negative and finite.
5. `test_arbitrage_free_converged_reason_floor_induced`: curve=50, target=-10, enforce_m_factor_floor=True → m_factor=-0.2 < 0.1, floor fires → converged=False, log record with reason='m_factor_floor_hit'.

**9 stubbed-skipped with stable cross-plan messages:**
- 2 for Plan 05-02: NEG-03 (water value delta + assembler integration)
- 6 for Plan 05-03: NEG-04 cascading spread + master flag + acceptance + baselines
- 1 codex action #7: empty spot history fallback

### Tasks 5+6: conftest.py + 14th stub test

Added inline doc comment in `_pfc_lt_env_hygiene` listing both `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` and `PFC_LT_ALLOW_NEGATIVE_PRICES` (prefix match covers both automatically; comment is informational). Added `test_fit_peak_spreads_empty_spot_history` stub (codex action #7, Plan 05-03 scope).

## Test Results

| Suite | Before | After |
|-------|--------|-------|
| Full suite (pytest tests/) | 258 passed, 3 skipped | 263 passed, 12 skipped |
| Phase 5 file only | N/A | 5 passed, 9 skipped |

The 5 success criteria tests from the plan frontmatter pass:
- `test_msfc_signed_monthly_repricing` ✓
- `test_arbitrage_free_signed_target` ✓
- `test_msfc_clamp_all_equal_knots` ✓
- `test_msfc_clamp_all_negative_knots_no_inverted_bounds` ✓
- `test_arbitrage_free_converged_reason_floor_induced` ✓

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] floor_applied NameError in additive mode**
- **Found during:** Task 3 implementation
- **Issue:** `floor_applied` was declared inside `if self.mode == "multiplicative":` but referenced after the block in both modes. Would cause `NameError` in additive mode.
- **Fix:** Added `floor_applied = False` before the `if self.mode == "multiplicative":` block (one line sentinel).
- **Files modified:** `pfc_shaping/calibration/arbitrage_free.py`
- **Commit:** fda761e

**2. [Rule 2 - Scope deviation] Tasks 5+6 populated in Task 4 scaffold**
- **Found during:** Task 4
- **Issue:** The plan stages Tasks 5+6 as separate "flip stub to populated" micro-tasks. Since Tasks 2+3 were completed before Task 4, the plan explicitly allows direct population: "If Tasks 2+3 ship before Task 4, scaffold can include the populated bodies directly."
- **Fix:** The three codex-action tests (test_msfc_clamp_all_equal_knots, test_msfc_clamp_all_negative_knots_no_inverted_bounds, test_arbitrage_free_converged_reason_floor_induced) were directly populated in Task 4 scaffold commit. Tasks 5+6 commits cover the 14th stub and conftest.py doc comment.
- **Impact:** Final counts differ slightly from the intermediate counts predicted by the plan (plan predicted 2 passed, 12 skipped at end of Task 4; actual was 5 passed, 9 skipped). Final totals are correct.

### Out-of-scope pre-existing warnings

The `RuntimeWarning: divide by zero encountered in matmul` warnings in `arbitrage_free.py:613` (smoothness_cost computation) appear for tests with strongly negative m_factor (near-zero product). These are pre-existing behavior unrelated to Phase 5 changes — the `if not np.isfinite(smoothness_cost): smoothness_cost = 0.0` guard handles them. Not fixed (out of scope, RESEARCH scope boundary).

## Known Stubs

The following tests are intentionally stub-skipped, documented in their docstrings with clear cross-plan ownership:

| Test | Owner Plan | Contract |
|------|------------|----------|
| test_water_value_delta_sign_invariant | 05-02 | compute_delta_wv(B=-10, f_wv=1.20) → delta=+2.0 |
| test_assembler_delta_additive | 05-02 | P = B×f_H×f_W + delta_wv (not multiplicative) |
| test_cascading_spread_signed_base | 05-03 | spread additif: -10 + 5 = -5 (not -10.5 ratio legacy) |
| test_fit_peak_ratios_deprecated | 05-03 | DeprecationWarning + shim to fit_peak_spreads |
| test_master_flag_audit_log | 05-03 | PFC_LT_ALLOW_NEGATIVE_PRICES INFO log at init |
| test_phase05_summer_bowl_negative_acceptance | 05-03 | SC #2 gated by 5bis-B bowl marker |
| test_phase05_baseline_regression | 05-03 | baseline_pfc_seed42_phase05 atol=1e-12 |
| test_phase05_baseline_5bisA_via_enforce_true | 05-03 | legacy baseline via enforce_*=True |
| test_fit_peak_spreads_empty_spot_history | 05-03 | codex action #7 fallback spread + WARN |

All stubs import cleanly (`pytest --collect-only` passes without error) and include stable cross-plan owner messages.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Phase 5 Plan 01 is a pure math refactor (ctor args + conditional floor + clamp formula change). No threat flags.

## Self-Check: PASSED

Files exist:
- `.planning/REQUIREMENTS.md` — FOUND (NEG-05 reformulated)
- `pfc_shaping/lt/model/msfc_spline.py` — FOUND (enforce_positivity present)
- `pfc_shaping/calibration/arbitrage_free.py` — FOUND (enforce_m_factor_floor present)
- `tests/test_phase05_negative_prices.py` — FOUND (14 tests collectible)
- `tests/conftest.py` — FOUND (PFC_LT_ALLOW_NEGATIVE_PRICES documented)

Commits exist:
- f10c524 — FOUND
- 3911f3f — FOUND
- fda761e — FOUND
- 93895d0 — FOUND
- 9da519e — FOUND

Test counts verified: 263 passed, 12 skipped (full suite), 5 passed, 9 skipped (Phase 5 file).
