---
phase: 5
slug: msfc-log-prix-retire-silent-floors
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-19
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Derived from `05-RESEARCH.md §Validation Architecture` + `05-CONTEXT.md §D-A4-4/D-A4-5`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 |
| **Config file** | `pyproject.toml` / `pytest.ini` (existing) |
| **Quick run command** | `pytest tests/test_phase05_negative_prices.py -x -q` |
| **Full suite command** | `pytest tests/ -q --tb=short` |
| **Estimated runtime** | ~30 s quick, ~3 min full suite |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_phase05_negative_prices.py -x -q` (the phase-specific file when it exists from wave 1 onward; before that, run the targeted file for the module touched).
- **After every plan wave:** Run `pytest tests/ -q --tb=short` (full suite).
- **Before `/gsd:verify-work`:** Full suite must be green, including baseline regression `test_phase05_baseline_regression` and legacy baseline `test_phase05_baseline_5bisA_via_enforce_true`.
- **Max feedback latency:** ~30 seconds (quick), ~3 minutes (full suite).

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 1 | NEG-05 (reformulation) | — | N/A | docs | `grep -F 'monthly forward négatif' .planning/REQUIREMENTS.md` | ✅ | ⬜ pending |
| 5-01-02 | 01 | 1 | NEG-01 | — | `smooth_base_prices` exposes `enforce_positivity=False`, removes both `np.maximum(B, 1.0)` (l.131, l.203) and rewrites the l.120 extrapolation clamp to `[min - margin, max + margin]` with `margin = 0.5 * np.ptp(y_knots)` | unit | `pytest tests/test_phase05_negative_prices.py::test_msfc_signed_monthly_repricing -x` | ❌ W0 | ⬜ pending |
| 5-01-03 | 01 | 1 | NEG-02 | — | `ArbitrageFreeCalibrator(enforce_m_factor_floor=False)` removes l.517 `np.maximum(m_factor, 0.1)`; propagates `converged=False` when floor would otherwise have hit (with `enforce_m_factor_floor=True`); logs WARN per clip | unit | `pytest tests/test_phase05_negative_prices.py::test_arbitrage_free_signed_target -x` | ❌ W0 | ⬜ pending |
| 5-02-01 | 02 | 2 | NEG-03 | — | `compute_delta_wv(B_smooth, fill_df, calendar_df) → pd.Series` returns `delta_wv = (f_wv - 1) × |B_smooth|`. `WaterValueCorrection(enforce_floor=False)` retires the F_WV_FLOOR clip at l.394 and l.407 | unit | `pytest tests/test_phase05_negative_prices.py::test_water_value_delta_sign_invariant -x` | ❌ W0 | ⬜ pending |
| 5-02-02 | 02 | 2 | NEG-03 | — | `assembler.build()` consumes `compute_delta_wv` additively: `P = B × f_H × f_W + delta_wv`. INFO log `"WV delta_wv: min=… max=… mean=… €/MWh, sign(B) flips=…"` per build | unit | `pytest tests/test_phase05_negative_prices.py::test_assembler_delta_additive -x` | ❌ W0 | ⬜ pending |
| 5-03-01 | 03 | 3 | NEG-04 | — | `ContractCascader.fit_peak_spreads(spot_history)` persists `peak_base_spreads_: dict[int, float]`. `synthesize_peak_prices` uses `result[peak_key] = base + peak_base_spreads_[month]` when `allow_negative_peak=True` (default) | unit | `pytest tests/test_phase05_negative_prices.py::test_cascading_spread_signed_base -x` | ❌ W0 | ⬜ pending |
| 5-03-02 | 03 | 3 | NEG-04 | — | `fit_peak_ratios` raises `NotImplementedError` (with migration message pointing to `fit_peak_spreads`) on call | unit | `pytest tests/test_phase05_negative_prices.py::test_fit_peak_ratios_deprecated -x` | ❌ W0 | ⬜ pending |
| 5-03-03 | 03 | 3 | NEG-01..05 | — | `PFCAssembler.__init__` reads `PFC_LT_ALLOW_NEGATIVE_PRICES` once, emits INFO audit log `"PFC_LT_ALLOW_NEGATIVE_PRICES={state}, floors_disabled={msfc:enforce_positivity, af:m_factor_floor, wv:floor, cascading:allow_neg_peak}"` | unit | `pytest tests/test_phase05_negative_prices.py::test_master_flag_audit_log -x` | ❌ W0 | ⬜ pending |
| 5-03-04 | 03 | 3 | SC #2 ROADMAP | — | Fixture `forwards_phase05_seed42.parquet` (Cal'27=30, July M-07'27=20 dépressé, autres months positifs typiques EEX, seed=42) built deterministically. `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1 + PFC_LT_ALLOW_NEGATIVE_PRICES=1` → `pfc[Sunday, h13, July 2027].mean() < -20 €/MWh` | acceptance | `pytest tests/test_phase05_negative_prices.py::test_phase05_summer_bowl_negative_acceptance -x` | ❌ W0 | ⬜ pending |
| 5-03-05 | 03 | 3 | SC #5 ROADMAP | — | `baseline_pfc_seed42_phase05.parquet` frozen; `assert_frame_equal(build(forwards_phase05_seed42), baseline_pfc_seed42_phase05, atol=1e-12, rtol=0)` | regression | `pytest tests/test_phase05_negative_prices.py::test_phase05_baseline_regression -x` | ❌ W0 | ⬜ pending |
| 5-03-06 | 03 | 3 | SC #5 (legacy backward-compat) | — | `assembler.build(forwards_5bisA, enforce_positivity=True, enforce_m_factor_floor=True, enforce_floor=True, allow_negative_peak=False)` matches `baseline_pfc_seed42.parquet` (5bis-A baseline) `atol=1e-12, rtol=0` | regression | `pytest tests/test_phase05_negative_prices.py::test_phase05_baseline_5bisA_via_enforce_true -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase05_negative_prices.py` — file created in Plan 05-01 with stubs for the 4 unit math tests (`test_msfc_signed_monthly_repricing`, `test_arbitrage_free_signed_target`, `test_water_value_delta_sign_invariant`, `test_cascading_spread_signed_base`), the 1 system acceptance (`test_phase05_summer_bowl_negative_acceptance`, gated by 5bis-B bowl marker), and the 2 baseline regressions (`test_phase05_baseline_regression`, `test_phase05_baseline_5bisA_via_enforce_true`). Stubs at Plan 05-01 wave-start; assertions populated as each ctor arg lands.
- [ ] `tests/fixtures/_generate_phase05_fixture.py` — script generating `tests/fixtures/forwards_phase05_seed42.parquet` (Cal'27=30 €/MWh, July M-07'27=20 €/MWh dépressé, autres months positifs typiques EEX, seed=42, deterministic). Mirrors `_generate_baseline.py` (5bis-A) / `_generate_bowl_fixture.py` (5bis-B) conventions.
- [ ] `tests/fixtures/forwards_phase05_seed42.parquet` — fixture committed; identical bytes on regeneration (`md5sum` stable).
- [ ] `tests/fixtures/baseline_pfc_seed42_phase05.parquet` — frozen Phase-5-canonical baseline output (post-floors-off, post-delta-additive WV, post-spread-additive Peak, `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1`, `PFC_LT_ALLOW_NEGATIVE_PRICES=1`). Committed in Plan 05-03 once the math is stable.
- [ ] `tests/conftest.py` — extend `PFC_LT_*` env-var autouse snapshot/restore list with `PFC_LT_ALLOW_NEGATIVE_PRICES` (~1 line). Required so test isolation holds when SC #2 / regression tests set the flag.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `assembler.build()` INFO log line shape (telemetry readability) | D-A2-2, D-A3-5 | Format-quality judgement, not a behavior assertion. Automated test asserts the substring `"PFC_LT_ALLOW_NEGATIVE_PRICES="` and `"WV delta_wv"` are present; full readability checked by hand. | After Plan 05-03 lands, run `python -c "from pfc_shaping.pipeline.autoresearch import build_pfc; build_pfc(seed=42)"` with `PYTHONLOGGING=INFO`, eyeball the two log lines for telemetry usefulness in incident triage. |
| 2020-Q2 covid negative-spot historical slice | (out of scope, deferred) | Real-data slice not in CI ; possible validation post-merge. | If desired post-merge: load EEX historical mai-juin 2020, rerun assembler, log mid-market hourly negatives; record in VERIFICATION.md `## Real-data spot-check`. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (each plan has at least one new automated test asserting its math change)
- [ ] Wave 0 covers all MISSING references (`tests/test_phase05_negative_prices.py`, fixture script, fixture parquet, baseline parquet, conftest extension)
- [ ] No watch-mode flags (pytest is one-shot per command)
- [ ] Feedback latency < 30 s (quick command) ; full suite < 3 min
- [ ] Tolerance contract enforced: `atol=1e-12, rtol=0`, `check_exact=False`, identical columns/dtypes/index/sort order (5bis-A REVIEWS addendum)
- [ ] `nyquist_compliant: true` set in frontmatter after execute-phase fills statuses

**Approval:** pending
