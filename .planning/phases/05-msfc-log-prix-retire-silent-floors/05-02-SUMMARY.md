---
phase: 05-msfc-log-prix-retire-silent-floors
plan: "02"
subsystem: pfc-lt-negative-prices
tags:
  - water-value
  - delta-additive
  - assembler
  - negative-prices
  - codex-action-1
dependency_graph:
  requires:
    - "Plan 05-01 (enforce_positivity + enforce_m_factor_floor ctor args, Phase 5 test scaffold)"
  provides:
    - "WaterValueCorrection(enforce_floor=False) ctor kwarg (NEG-03, D-A2-1)"
    - "compute_delta_wv(B_smooth, *, fill_df, calendar_df) public method (D-A3-1, codex action #1)"
    - "assembler.build() delta-additive path: price_raw = B*f_S*f_W*f_H*f_Q*f_bridge + delta_wv (D-A3-2)"
    - "INFO telemetry 'WV delta_wv: min=… max=… mean=… €/MWh, sign(B) flips: N' (D-A3-5)"
    - "delta_wv column in build() returned DataFrame (plan contract B3)"
    - "codex action #1 closed: *, assert guard, KEYWORD-ONLY call site"
    - "3 newly-passing tests: test_water_value_delta_sign_invariant, test_assembler_delta_additive, test_compute_delta_wv_index_alignment"
  affects:
    - "Plan 05-03 (BlockCascading spread additif + master flag + baseline — depends on stable API from this plan)"
tech_stack:
  added: []
  patterns:
    - "delta-additive WV: delta_wv = (f_wv - 1) * |B_smooth| (sign-invariant by construction)"
    - "KEYWORD-ONLY args via * separator (codex action #1 ergonomics)"
    - "assembler branching on wv.enforce_floor (delta-additive vs legacy multiplicative)"
    - "assert delta_wv.index.equals(B.index) precondition guard (codex action #1)"
    - "unittest.mock.patch.object for f_wv fixture in unit tests"
key_files:
  modified:
    - path: "pfc_shaping/lt/model/water_value.py"
      description: "enforce_floor ctor kwarg + conditional clips + compute_delta_wv public method"
    - path: "pfc_shaping/lt/model/assembler.py"
      description: "delta-additive path branch + codex action #1 guards + telemetry + delta_wv column"
    - path: "tests/test_phase05_negative_prices.py"
      description: "2 stubs flipped + 1 new test (total 15 tests, 8 passed, 7 skipped)"
    - path: "tests/test_shape_hourly_bowl.py"
      description: "[Rule 1 fix] Baseline schema assertions relaxed to baseline_cols subset"
    - path: "tests/test_shape_hourly_infra.py"
      description: "[Rule 1 fix] Baseline schema assertions relaxed to baseline_cols subset"
decisions:
  - "D-A3-1: compute_delta_wv = (f_wv - 1) * |B_smooth| shipped — sign-invariant by construction"
  - "D-A3-2: assembler.build() now applies price_raw = B*f_S*f_W*f_H*f_Q*f_bridge + delta_wv on delta-additive path"
  - "D-A3-4: compute_delta_wv raises ValueError when enforce_floor=True (incompatible semantics)"
  - "RESEARCH Open Question #2 resolved: shape_freedom['f_WV'] damping bypassed on delta-additive path; horizon_decay inside WaterValueCorrection.apply() is sole source of far-horizon shrinkage"
  - "codex action #1 fully closed: * separator on compute_delta_wv, KEYWORD-ONLY call at assembler site, assert guard"
  - "5bis-A/B baseline schema assertion fix: compare against baseline_cols subset (new delta_wv column legitimate schema extension)"
metrics:
  duration_minutes: 20
  completed_date: "2026-05-20"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 5
  files_created: 0
---

# Phase 5 Plan 02: WaterValueCorrection delta-additif + assembler integration Summary

**One-liner:** WaterValueCorrection gains `enforce_floor=False` ctor arg and `compute_delta_wv(B_smooth, *, fill_df, calendar_df)` (KEYWORD-ONLY, codex action #1), assembler.build() branches on delta-additive path with precondition guard and INFO telemetry, 3 NEG-03 tests populated — suite advances from 263/12 to 266/10.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 5418576 | feat | WaterValueCorrection enforce_floor + compute_delta_wv (NEG-03) |
| 283d27f | feat | assembler.build() delta-additive WV path + codex action #1 guards |
| f0d2dfc | test | populate NEG-03 tests + new test_compute_delta_wv_index_alignment |

## What Was Built

### Task 1: WaterValueCorrection modifications (pfc_shaping/lt/model/water_value.py)

Three surgical edits (+91 lines / −3 lines):

**Edit A — `__init__` signature extended:**
```python
def __init__(self, enforce_floor: bool = False) -> None:
    self.enforce_floor: bool = bool(enforce_floor)
    ...
```
`self.enforce_floor` is the FIRST init-body line. Class docstring updated with Phase 5 D-A2-1 / D-A3-1 / NEG-03 rationale.

**Edit B — Conditional clips at lines 394 and 407:**
Both `raw_f_wv.clip(lower=F_WV_FLOOR, upper=F_WV_CAP)` and the post-renormalization `f_wv.clip(...)` are now wrapped in `if self.enforce_floor:`. When False (default), f_wv flows unclipped from `beta_wv_ × season_sensitivity_ × horizon_decay`. The `F_WV_FLOOR = 0.80` and `F_WV_CAP = 1.20` module constants remain defined (used by guarded clips and future legacy callers).

**Edit C — New public method `compute_delta_wv`:**
```python
def compute_delta_wv(
    self,
    B_smooth: pd.Series,
    *,                              # codex action #1: keyword-only after B_smooth
    fill_df: pd.DataFrame | None,
    calendar_df: pd.DataFrame,
) -> pd.Series:
```
Returns `delta = (f_wv - 1.0) * B_smooth.abs()` with `delta.name = 'delta_wv'`. Raises `ValueError` when `self.enforce_floor is True` (D-A3-4). Internal call: `f_wv = self.apply(B_smooth.index, calendar_df, fill_df)` (reuses calibrated `beta_wv_`, `season_sensitivity_`, `horizon_decay` — RESEARCH §Don't Hand-Roll).

### Task 2: PFCAssembler.build() refactoring (pfc_shaping/lt/model/assembler.py)

Four surgical edits (+66 lines / −6 lines):

**Edit A — Branch f_WV computation block:**
Replaced the simple `if self.wv is not None:` with a tri-branch on `use_delta_additive_wv = (self.wv is not None) and (not self.wv.enforce_floor)`. Sets `delta_wv_pending = True/False` flag to control subsequent branches.

**Edit B — Gate shape_freedom['f_WV'] damping:**
```python
if not delta_wv_pending:
    f_WV = 1.0 + (f_WV - 1.0) * shape_freedom["f_WV"]
```
On the delta-additive path, f_WV is pass-through 1.0 and the damping is explicitly skipped (RESEARCH Pitfall 2 guard — prevents double-damping with `horizon_decay` inside `WaterValueCorrection.apply()`).

**Edit C — Branch price_raw formula:**
```python
if delta_wv_pending:
    delta_wv = self.wv.compute_delta_wv(B, fill_df=hydro_forecast, calendar_df=cal)  # codex action #1
    assert delta_wv.index.equals(B.index), ...  # codex action #1 precondition guard
    price_raw = B * f_S * f_W * f_H * f_Q * f_bridge + delta_wv
    sign_flips = int((np.sign(B) != np.sign(B.shift(1))).fillna(False).sum())
    logger.info("WV delta_wv: min=%.2f, max=%.2f, mean=%.2f €/MWh, sign(B) flips: %d", ...)
else:
    delta_wv = pd.Series(0.0, index=idx, name="delta_wv")
    price_raw = B * f_S * f_W * f_H * f_Q * f_WV * f_bridge
```

**Edit D — delta_wv column in returned DataFrame:**
```python
"delta_wv": delta_wv,  # inserted after "f_WV": f_WV, before "f_bridge"
```
On legacy path: all-zeros Series. On delta-additive path: actual correction in €/MWh.

**Baseline schema fix [Rule 1 - Bug]:**
Tests `test_flag_off_bit_for_bit_baseline`, `test_flag_on_bowl_baseline`, `test_baseline_regression[False]` compared `list(df.columns) == list(baseline.columns)` strictly. The new `delta_wv` column (not present in 5bis-A/B parquet baselines) caused these tests to fail. Fix: compute `baseline_cols = list(baseline.columns)` and compare only those columns in both the column list check and `assert_frame_equal`. The new column is a legitimate schema extension.

### Task 3: Test population (tests/test_phase05_negative_prices.py)

File grew from 14 → 15 tests (+309 lines / −8 lines):

**test_water_value_delta_sign_invariant (5-02-01, NEG-03):**
Four cases using `patch.object(WaterValueCorrection, 'apply', return_value=...)`:
1. Scarcity f_wv=1.20, B=-10 → delta ≈ +2.0 (correct: less negative)
2. Abundance f_wv=0.80, B=-10 → delta ≈ -2.0 (correct: more negative)
3. Sign-invariance: mixed B=±10 with f_wv=1.20 → all delta ≈ +2.0 (depends on |B|)
4. enforce_floor=True → ValueError(match='enforce_floor') via pytest.raises

**test_assembler_delta_additive (5-02-02, NEG-03):**
Builds minimal PFCAssembler with WaterValueCorrection (default enforce_floor=False) + mock f_wv=1.10. Captures INFO logs during build(). Asserts:
- At least 1 record starts with "WV delta_wv:" with "min=", "max=", "mean=", "sign(B) flips:"
- result["delta_wv"] is pd.Series, float64, non-zero (WV contribution flows through)
- Negative control: enforce_floor=True → no "WV delta_wv:" log, delta_wv all-zeros

**test_compute_delta_wv_index_alignment (5-02-03, NEW — codex action #1):**
- Case A: keyword-only call works, delta.index.equals(B_smooth.index), name='delta_wv', shape matches
- Case B: positional fill_df/calendar_df → pytest.raises(TypeError)

## RESEARCH Open Question #2 Resolution

**Resolved:** `shape_freedom['f_WV']` damping in assembler.py is BYPASSED on the delta-additive path. The `horizon_decay` inside `WaterValueCorrection.apply()` (exponential decay with `horizon_halflife_days_ = 270` days) is the single source of truth for far-horizon shrinkage of the WV correction. Adding the `shape_freedom["f_WV"]` knot-schedule damping on top would be double-damping (RESEARCH Pitfall 2). The explicit `if not delta_wv_pending:` guard in assembler.py documents this design intent inline.

## Codex Review Action #1 Closure

| Change point | File | Detail |
|---|---|---|
| `*` separator in signature | `water_value.py:compute_delta_wv` | `def compute_delta_wv(self, B_smooth, *, fill_df, calendar_df)` |
| KEYWORD-ONLY call site | `assembler.py:build()` | `self.wv.compute_delta_wv(B, fill_df=hydro_forecast, calendar_df=cal)` |
| Precondition guard | `assembler.py:build()` | `assert delta_wv.index.equals(B.index), ...` |
| Test enforcement | `test_phase05_negative_prices.py::test_compute_delta_wv_index_alignment` | `pytest.raises(TypeError)` on positional call |

## Test Results

| Suite | Before | After |
|-------|--------|-------|
| Full suite (pytest tests/) | 263 passed, 12 skipped | 266 passed, 10 skipped |
| Phase 5 file only | 5 passed, 9 skipped | 8 passed, 7 skipped |

Test counts validated:
- `test_water_value_delta_sign_invariant` — PASSED (NEG-03, 5-02-01)
- `test_assembler_delta_additive` — PASSED (NEG-03, 5-02-02)
- `test_compute_delta_wv_index_alignment` — PASSED (codex action #1, 5-02-03)

## 5bis-A Baseline test_baseline_regression[False] Outcome

**PASS** — the fix applied in Task 2 (subset `baseline_cols` comparison) ensures this test passes. The divergence predicted by RESEARCH §Dry-Run D-A3-3 (0.2–1.4 €/MWh ≫ 1e-12) is NOT tested against the 5bis-A baseline here because the baseline parquet is compared only on its original columns (which exclude `delta_wv`). The numerical values of `price_shape` in the 5bis-A test do NOT change (the test invokes `build_pfc(seed=42, flag=False)` which uses `water_value=None` — no WV in that fixture, so the delta-additive path is not triggered).

The canonical Phase-5 baseline with the delta-additive math will be generated in Plan 05-03.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Schema mismatch in 5bis-A/B baseline tests after adding delta_wv column**
- **Found during:** Task 2 verification (full test suite run)
- **Issue:** `tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline`, `test_flag_on_bowl_baseline`, and `tests/test_shape_hourly_infra.py::test_baseline_regression[False]` assert `list(df.columns) == list(baseline.columns)`. The new `delta_wv` column in build()'s returned DataFrame is not present in the pre-Phase-5 parquet baselines.
- **Fix:** Compute `baseline_cols = list(baseline.columns)` in each test and compare only those columns (both in the `list(df.columns)` check and in `assert_frame_equal`). This correctly tests that the historical contract is preserved on the historical columns, while allowing the schema to grow.
- **Files modified:** `tests/test_shape_hourly_bowl.py` (2 tests), `tests/test_shape_hourly_infra.py` (1 test)
- **Commit:** 283d27f

## Known Stubs

The following tests remain intentionally stub-skipped with Plan 05-03 ownership:

| Test | Owner Plan | Contract |
|------|------------|----------|
| test_cascading_spread_signed_base | 05-03 | spread additif: -10 + 5 = -5 |
| test_fit_peak_ratios_deprecated | 05-03 | DeprecationWarning + shim to fit_peak_spreads |
| test_master_flag_audit_log | 05-03 | PFC_LT_ALLOW_NEGATIVE_PRICES INFO log at init |
| test_phase05_summer_bowl_negative_acceptance | 05-03 | SC #2 gated by 5bis-B bowl marker |
| test_phase05_baseline_regression | 05-03 | baseline_pfc_seed42_phase05 atol=1e-12 |
| test_phase05_baseline_5bisA_via_enforce_true | 05-03 | legacy baseline via enforce_*=True |
| test_fit_peak_spreads_empty_spot_history | 05-03 | codex action #7 fallback spread + WARN |

## Cross-plan Handoff (Plan 05-03)

Plan 05-03 owns:
1. `BlockCascading.fit_peak_spreads(spot_history)` + `allow_negative_peak=True` default
2. `fit_peak_ratios` DeprecationWarning shim (codex action #2)
3. Master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` audit-trail INFO log in `PFCAssembler.__init__`
4. Fixture generator `tests/fixtures/_generate_phase05_fixture.py` + forwards_phase05_seed42.parquet
5. Canonical Phase-5 baseline `baseline_pfc_seed42_phase05.parquet` (defaults OFF + delta-additive WV)
6. All 7 remaining stub-skipped tests (test IDs 5-03-01..5-03-07)
7. Production callsite migration: `production_phases.py:344,644` from `fit_peak_ratios` to `fit_peak_spreads`

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries introduced. Plan 05-02 is a pure math refactor (ctor arg + conditional floors + new method + assembler path branching). No threat flags.

## Self-Check: PASSED

Files exist:
- `pfc_shaping/lt/model/water_value.py` — FOUND (enforce_floor + compute_delta_wv present)
- `pfc_shaping/lt/model/assembler.py` — FOUND (delta_wv_pending + WV delta_wv: telemetry present)
- `tests/test_phase05_negative_prices.py` — FOUND (15 tests collectible)
- `tests/test_shape_hourly_bowl.py` — FOUND (baseline schema fix applied)
- `tests/test_shape_hourly_infra.py` — FOUND (baseline schema fix applied)

Commits exist:
- 5418576 — FOUND
- 283d27f — FOUND
- f0d2dfc — FOUND

Test counts verified: 266 passed, 10 skipped (full suite), 8 passed, 7 skipped (Phase 5 file).
