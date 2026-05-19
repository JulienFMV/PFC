---
status: passed
phase: 05C-shape-hourly-bowl-deepening
phase_label: "Phase 5bis-B — Shape Hourly Bowl-Deepening (math change)"
verified_at: 2026-05-19T17:32:00Z
verifier: claude-opus-4-7 (inline — gsd-verifier agent unregistered in this runtime)
plans_verified:
  - 05C-01
  - 05C-02
  - 05C-03
success_criteria_met: 5
success_criteria_total: 5
cross_ai_review_fixes_landed: 4
cross_ai_review_fixes_total: 4
tests_total: 261
tests_passed: 258
tests_skipped: 3
tests_failed: 0
baseline_5bisA_preserved: true
baseline_bowl_frozen: true
flag_off_bit_for_bit: true
---

# 05C VERIFICATION — Phase 5bis-B (Shape Hourly Bowl-Deepening)

## Verdict

**PASSED.** All 5 success criteria from ROADMAP, all 4 cross-AI consensus fixes (M1-M4), and all 17 plan-level success_criteria items across 05C-01/02/03 are honored in the merged code. Test suite shows **258 passed, 3 skipped** (was 247 at end of 5bis-A — net **+11 tests** all green: 2 in 05C-01, 3 in 05C-02, 3 in 05C-03 = 8 new bowl tests + 3 new sidecar-compat parametrize cases).

The phase delivers the business value behind SHP-01/SHP-02/SHP-03/SHP-04 (already structurally satisfied by 5bis-A): the duck-curve bowl now actually deepens under `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1`, and the assembler propagates that signature through `f_H` damping to the far horizon (~M+30) instead of flattening it.

## Goal-vs-Result

| ROADMAP Success Criterion | Test | Status |
|---|---|---|
| **SC #1**: `np.ptp(factors_[("Ete","Ouvrable")])` strictly > baseline 5bis-A on bowl fixture (bowl amplifies) | `test_factors_ptp_deepens_under_flag` (D-A4-5) | PASS |
| **SC #2**: `assembler.build` produces `|Δ price_shape[Dim,Été,h10-15] − Dim,Hiver,h10-15]| > 5 €/MWh` | `test_seasonal_solar_winter_evening_delta` (D-A4-7) | PASS |
| **SC #3**: `df["f_H"]` post-damping à M+30 garde amplitude > calibrated threshold (0.2846 = 1.87× legacy 0.1902) | `test_f_H_amplitude_preserved_at_M30` (D-A4-6) | PASS |
| **SC #4**: Flag OFF reproduit baseline bit-pour-bit (`atol=1e-12, rtol=0`) | `test_flag_off_bit_for_bit_baseline` (D-A4-8) + `test_baseline_regression[False]` (5bis-A) | PASS |
| **SC #5**: Suite full reste verte (was 247 at end of 5bis-A) | Full pytest run | PASS (258 passed, 3 skipped — no regressions) |

## Cross-AI Review Consensus (REVIEWS.md — Gemini + Codex) — Landing Audit

| Fix | Where shipped | Test/Artifact | Status |
|---|---|---|---|
| **M1** — caplog-based telemetry test for D-A2-5 drift warning (silent SHP-03 degradation guard) | Plan 05C-02 | `tests/test_shape_hourly_bowl.py::test_split_level_anomaly_drift_warning` + extracted `_emit_level_drift_telemetry()` helper in `assembler.py:125` (testable without full PFC build) | LANDED |
| **M2** — committed `scripts/calibrate_bowl_thresholds.py` + immutable `tests/fixtures/_bowl_calibration_report.json` with `fixture_sha256` binding | Plans 05C-01, 05C-02 (extension), 05C-03 (sha256 binding test) | `tests/test_shape_hourly_bowl.py::test_calibration_report_matches_fixture` asserts sha256 match (tamper-fail-loud) | LANDED |
| **M3** — `## Window-dependence` docstring on `_split_level_anomaly` AND `PFCAssembler.build()` (inspect.getdoc-verifiable) | Plan 05C-02 | docstrings present in `shape_hourly.py` and `assembler.py` | LANDED |
| **M4** — `tests/test_sidecar_compat.py::test_sidecar_load_matrix` parametrized across `pre_5bisA / 5bisA / 5bisB` sidecar schemas (fixture-factory, not committed binaries) | Plan 05C-03 | 3 parametrize cases all PASS (deterministic, < 100ms each) | LANDED |

## Requirements Traceability

| Req ID | Source | How satisfied | Plan |
|---|---|---|---|
| SHP-01 | REQUIREMENTS.md (structural, 5bis-A) | Business value behind shipped in 5bis-B (per-timestamp hydro fill drives true seasonality) | 05C-01 |
| SHP-02 | REQUIREMENTS.md | f_H split preserves seasonal × type_jour signature through damping | 05C-02 |
| SHP-03 | REQUIREMENTS.md | energy-normalization invariant preserved (level≈1.0 + zero-mean anomaly construction); telemetry warns at 1e-6 drift | 05C-02 |
| SHP-04 | REQUIREMENTS.md | σ_on=0.25 sharpens the bowl when flag=ON; σ_off=0.5 preserves legacy smoothing | 05C-03 |
| D-A1-1..D-A1-5 | CONTEXT.md | hydro kernel target + sigma_off/_on resolution + sidecar persistence | 05C-01 |
| D-A2-1..D-A2-6 | CONTEXT.md | `_split_level_anomaly` + flag-gated assembler branch + telemetry drift detection | 05C-02 |
| D-A3-1..D-A3-6 | CONTEXT.md | final `__init__` signature + resolution precedence (Pitfall-3 guarded) + sidecar 10-key schema + cross-plan fallback + EPFL init log | 05C-03 |
| D-A4-1..D-A4-9 | CONTEXT.md | bowl fixture + 8 new tests + frozen flag=ON baseline (D-A4-9 convention) | 05C-01/02/03 |
| D-FLIP-1 | CONTEXT.md | row in `.planning/PROJECT.md::Key Decisions` (line 87) — flag flips default ON only after Phase 10 success gate (Δ MAE bloc ≤ -1.5 €/MWh vs HFC OMPEX 2024-2025) | 05C-03 |

## Code-Level Spot Checks

- `pfc_shaping/lt/model/shape_hourly.py:202-203` — final ctor signature has `sigma_off: float = _SIGMA_OFF_DEFAULT, sigma_on: float = _SIGMA_ON_DEFAULT` (D-A3-1 ✓)
- `pfc_shaping/lt/model/shape_hourly.py:206-207` — same for `hydro_weight_sigma_off/_on` (D-A1-4 ✓)
- `pfc_shaping/lt/model/shape_hourly.py:239` — Pitfall-3 guarded comparison against canonical default constants (`_SIGMA_OFF_DEFAULT`/`_SIGMA_ON_DEFAULT`), NOT received params → no spurious warnings on `ShapeHourly()` ✓
- `pfc_shaping/lt/model/shape_hourly.__all__` — exports `[ShapeHourly, GAUSSIAN_SIGMA, _FLAG_ENV_VAR, _resolve_flag, _meta_path, _split_level_anomaly]` (Pitfall C ✓)
- `pfc_shaping/lt/model/assembler.py:125` — `_emit_level_drift_telemetry(level, logger_)` standalone helper (M1 testability fix ✓)
- `pfc_shaping/lt/model/assembler.py:147` — D-A2-5 canonical warning message `"f_H split: level drift %.2e > 1e-6 — SHP-03 invariant may be degraded"` ✓
- `pfc_shaping/lt/model/assembler.py:383` — `_emit_level_drift_telemetry(level, logger)` called from `build()` under flag=ON ✓
- `.planning/PROJECT.md:87` — D-FLIP-1 entry present ✓
- `.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` — SUPERSEDED header added (5bis pre-doc retired) ✓
- `tests/fixtures/baseline_pfc_seed42_bowl.parquet` — 119,475 bytes, generated AFTER all 3 levers ship (Pitfall B ✓)
- `tests/fixtures/_bowl_calibration_report.json` — schema includes `fixture_sha256` + `thresholds_emitted.{SC1_PTP_THRESHOLD, SC3_M30_AMPLITUDE_THRESHOLD}` (M2 ✓)
- `tests/test_sidecar_compat.py` — 9811 bytes, 3 parametrize cases `[pre_5bisA-False-0.5-0.25-0.5-0.25]`, `[5bisA-False-0.5-0.25-0.5-0.25]`, `[5bisB-False-0.5-0.25-0.25-0.08]` all PASS (M4 ✓)

## Test Suite

```
Focused probe (115 tests):
  tests/test_shape_hourly_bowl.py    : 9 tests   (SC#1/#2/#3, M1, M2, baselines)
  tests/test_shape_hourly_infra.py   : 103 tests (5bis-A regression + Pitfall 4 surgical 10-key updates)
  tests/test_sidecar_compat.py       : 3 tests   (M4 matrix)
  → 115 passed in 15.82s

Full suite:
  → 258 passed, 3 skipped in 20.72s (no regressions vs 5bis-A baseline 247)
```

## Deviations from Plans (transparent, all auto-resolved per execute-plan.md Rule 1)

- **05C-01**: ALLOWED_FUNCTIONS extension for 5bis-B AST guard (Rule 3) + `freq=None` workaround in baseline test (Rule 1 — same pattern as existing infra suite).
- **05C-02**: 4 Rule-1 auto-fixes — threshold formula (multiplicative vs absolute), groupby pattern, caplog setup normalization, baseline-regression parametrize.
- **05C-03**: All 7 tasks executed as planned; no spec deviations beyond the standard fixture-cleanup edits.

All deviations documented in each plan's SUMMARY.md and accepted at execution time (Rule 1 — keep moving when the deviation is local and provably safe).

## Phase Goal Achievement

The phase goal as stated in ROADMAP — *"creuser la duck curve réelle de la PFC pour que les profile deals GRD soient pricés au juste prix (bloc nuit 18-9 + solaire WE OP1/OP2)"* — is **achieved**:

- Lever 1 fixes the long-standing bug where `_apply_hydro_analogue_weights` collapsed the historical hydro context to a single scalar `fill.iloc[-1]`. The kernel now uses the per-timestamp climatological fill, so the analogue search sees the correct seasonal context (week-of-year) at each sample.
- Lever 2 protects the duck-curve signature in `f_H` from `shape_freedom['f_H']` damping by splitting `f_H = level + anomaly` and only damping `level`. The empirically-measured outcome on the bowl fixture: M+30 amplitude jumps from `ptp_off=0.1902` to `ptp_on=0.3558` — a **1.87×** amplification of the bowl that legacy damping was flattening to a flat line by M+12.
- Lever 3 sharpens the bowl by lowering the Gaussian smoothing σ from 0.5 → 0.25 hours FWHM (RESEARCH §Lever 3 confirmed) under flag=ON.

The combined effect — verified by SC #1 + SC #2 + SC #3 tests on the bowl fixture — is that the flag=ON build produces a PFC with a true seasonal × diurnal duck curve where flag=OFF (the legacy default) preserves byte-identical 5bis-A semantics. The phase ships **gated by `PFC_LT_USE_SEASONAL_HOURLY_SHAPE`** with default OFF; flip to default ON is gated by Phase 10 success (D-FLIP-1).

## Human Verification

Not required for this phase — all success criteria are testable and pass. Phase 10 (backtest par bloc vs HFC OMPEX) will validate the business value empirically on real OMPEX data.

## Next

`/gsd:progress` — see updated roadmap.

Recommended next step: `/gsd:discuss-phase 5` (MSFC log-prix + retire silent floors) before planning.
