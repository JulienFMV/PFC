---
phase: 5bis-B
padded_phase: 05C
slug: shape-hourly-bowl-deepening
reviewers: [gemini, codex]
reviewers_skipped: [claude]
skip_reason: "claude skipped — review.md runtime rule SELF_CLI=claude when CLAUDE_CODE_ENTRYPOINT is set (executing AI inside Claude Code CLI)."
reviewed_at: 2026-05-19T06:48:20Z
plans_reviewed:
  - 05C-01-PLAN.md
  - 05C-02-PLAN.md
  - 05C-03-PLAN.md
---

# Cross-AI Plan Review — Phase 5bis-B (Shape Hourly Bowl-Deepening)

> Independent reviews from external AI CLIs (Gemini, Codex via OpenAI). Claude
> skipped per `review.md` self-identification rule (we are executing inside
> Claude Code). Use this file to inform `/gsd:plan-phase 5bis-B --reviews`.

---

## Gemini Review

This review evaluates implementation plans **05C-01**, **05C-02**, and **05C-03** for Phase 5bis-B of the PFC Long-Term project.

### 1. Summary
The plan sequence is technically sophisticated and demonstrates high-caliber "quant" engineering discipline. It correctly separates infrastructure preparation (fixture generation) from mathematical implementation, maintaining a rigorous `atol=1e-12` no-op contract for the legacy path. The solution for Lever 2 (level/anomaly split) is particularly elegant, as it solves the "bowl flattening" problem at the far horizon without sacrificing the energy-neutrality invariant (SHP-03). The delegation of numerical calls to analytic research (e.g., $\sigma=0.08$ for hydro) is sound and well-documented.

### 2. Strengths
*   **Mathematical Rigor**: The use of `_split_level_anomaly` to selectively dampen only the mean level while preserving the seasonal signature (anomaly) is a SOTA approach that avoids the over-smoothing seen in legacy utility-grade models.
*   **Resolution Precedence (D-A3-2)**: The implementation of the triple-choice constructor (`sigma`, `sigma_off`, `sigma_on`) correctly handles legacy callsites while allowing for the necessary breaking change of making `sigma` default to `None`.
*   **Defensive Engineering**: Leveraging `inspect.signature` in the assembler and AST scanning in tests (`test_no_hidden_behavior_branch`) provides best-in-class protection against silent regressions.
*   **Innovation Gating**: Distinguishing between math-validation (synthetic fixture) and data-validation (Phase 10 real data) ensures that "broken math" is caught in CI even when "real data" is inaccessible.

### 3. Concerns
*   **MEDIUM | Window-Dependent Level Drift**: In `_split_level_anomaly` (Plan 02, T1), the `level` is computed via `groupby().transform("mean")` over the timestamps provided in the current `apply()` call. If a user builds a very short PFC (e.g., 1 day), the `level` becomes the mean of that day. If that day's profile (with trends) doesn't average to 1.0, the selective damping will shrink it towards 1.0. This is technically correct, but might introduce small level discontinuities if consecutive `apply()` calls have different window lengths.
*   **LOW | Telemetry Warning Silence**: Plan 02, D-A2-5 introduces a warning for `max |level - 1.0| > 1e-6`. However, the test suite does not verify that this warning actually fires under drift. A silent bug in the fit's normalization could degrade SHP-03 without CI failing.
*   **LOW | Audit Trail of Calibration**: The Wave 0 calibration scripts (Plan 01 T4, Plan 03 T3) are run interactively and not committed. While the *results* are recorded in test comments, the *process* is less auditable for a secondary quant reviewer than a committed `scripts/recalibrate_bowl_thresholds.py`.
*   **LOW | Performance of Hydro Kernel**: The vectorized map for `clim_target` (RESEARCH §Common Pitfalls A) is efficient, but the initialization of the dict `clim_map` for each `_apply_hydro_analogue_weights` call might be redundant if the model is fitted multiple times in a loop.

### 4. Suggestions
*   **For Level Drift**: Ensure that `PFCAssembler.build()` documentation (or a docstring in `_split_level_anomaly`) notes that `f_H` normalization is most stable when the build window covers full seasonal cycles or at least full weeks.
*   **For Telemetry**: Add a small unit test in `test_shape_hourly_bowl.py` that mocks a misaligned `f_H` (mean != 1.0) and uses `pytest.warns()` or `caplog` to verify the D-A2-5 telemetry triggers.
*   **For Auditability**: Consider wrapping the Wave 0 calibration logic into a hidden test helper within `test_shape_hourly_bowl.py` (e.g. `_run_calibration_dry_run()`) so it can be re-executed easily by other developers if the fixture changes.

### 5. Risk Assessment
**Risk Level: LOW**

The phase is extremely well-guarded. The risk of a "trading-grade bug" (pricing error) is mitigated by:
1.  The `flag=OFF` regression test ensuring 5bis-A parity.
2.  The "plancher 1.05" adversarial check on SC #1 gain.
3.  The explicit `D-FLIP-1` decision which gates the production flip behind Phase 10 real-data validation.

The most likely risk is a minor "fixture-real gap" (SC #2 passing on synth but failing on HFC OMPEX), which the plan already classifies as a research task rather than a ship-blocker.

**Verdict: Approved for execution.** Proceed with Waves 1, 2, and 3.

---

## Codex Review

1. **Summary**
Plan set is strong on scope control, traceability, and rollback safety, but several validation choices are too self-referential for trading-grade math changes. The core architecture (flag gating, staged waves, OFF-baseline invariance) is good; the main weakness is that key thresholds and one major hyperparameter (`hydro_weight_sigma_on=0.08`) are calibrated on the same synthetic fixture used to pass tests, which weakens falsifiability and leaves real-data failure risk high.

2. **Strengths**
- Clear LT-only boundary, no CT contamination, and good phase decomposition (05C-01/02/03).
- Excellent backward-compat intent (D-A3-2), with explicit legacy precedence and cross-plan sidecar fallback.
- Strong OFF-path regression contract (`atol=1e-12, rtol=0`) and explicit baseline fixtures.
- Lever 1 design direction is correct: per-timestamp climatological target is structurally better than scalar `current_fill`.
- Lever 2 implementation is conceptually coherent: anomaly pass-through + level damping is a sensible way to preserve bowl at far horizon.
- Good operational hygiene: sequential waves, explicit artifacts, explicit test mapping, and decision IDs.

3. **Concerns**
- **HIGH**: SC#1 and SC#3 use "measure-then-assert" on the same fixture (`05C-01 Task 4`, `05C-02 Task 3`, `05C-03 Task 3`). This can devolve into tautological tests that ratchet to whatever code outputs.
- **HIGH**: `hydro_weight_sigma_on=0.08` is derived from simulated ±10pp assumptions, not Swiss historical anomaly distribution from production data (`D-A1-4`, RESEARCH Lever 1). Risk: over-selective weighting, floor saturation, regime brittleness.
- **HIGH**: Lever 2 sum-preservation claim is overstated in docs. `level+anomaly=f_H` is exact pre-damping, but post-damping modifies level unless `level==1`. Since level is recomputed on the *call window* groupby, horizon-window composition can induce drift and inconsistent behavior (`D-A2-2`, `D-A2-3`).
- **MEDIUM**: `_split_level_anomaly` grouping on call timestamps (not fit-stable cell anchors) introduces horizon-dependent dynamics; behavior can change with date window length/composition even with identical model params.
- **MEDIUM**: Telemetry warning path (`max |level-1| > 1e-6`) is not asserted by tests; silent logger config changes could hide invariant degradation.
- **MEDIUM**: Interactive Wave-0 calibration scripts are not committed artifacts; weak auditability/reproducibility for quant governance.
- **MEDIUM**: Changing `sigma` default to `None` is logically right, but API-behavior change risk remains for reflection/signature-based callers and any non-covered external scripts.
- **LOW**: Test-count expectations are brittle; parametrized test counting can differ by collection context, so hard numeric targets may create noise.

4. **Suggestions**
- Replace dynamic threshold fitting with fixed, pre-registered thresholds from an immutable calibration report; keep calibration script committed and versioned.
- Add an out-of-sample synthetic family (multiple seeds/regimes) and require SC#1/SC#3 pass across all, not one fixture.
- Add a real historical hydro anomaly diagnostic (even offline artifact) to justify `sigma_on=0.08`; include quantiles and floor-hit rate targets.
- Add a direct post-damping SHP-03 test: per-cell mean drift bounds across multiple horizons/windows, not only telemetry logging.
- Add explicit tests for warning behavior (`caplog`) for `_split_level_anomaly` NaN/missing-cal paths and level-drift threshold.
- Add compatibility tests for `ShapeHourly()` / `ShapeHourly(sigma=...)` / load old sidecars from pre-5bis-B fixtures in one matrix test.
- Commit Wave-0 calibration as a reproducible script output file (JSON/markdown with observed ratios, thresholds, git SHA, fixture hash).

5. **Risk Assessment**
**Overall risk: MEDIUM-HIGH.**
Implementation structure is disciplined and reversible, but the validation strategy for the core business claims is partially circular (calibrate and test on same fixture) and the hydro sigma change is not yet empirically grounded on real Swiss anomaly distributions. For a trading-grade pricing engine, that leaves meaningful model-risk and false-confidence risk despite strong software-engineering controls.

---

## Consensus Summary

Two independent reviewers (Gemini, Codex) read the same artifacts and converged on the *same* engineering strengths and the *same* validation weaknesses. They diverge sharply on **severity assessment** — Gemini reads the safeguards (flag-gated default OFF, Phase 10 gate, bit-pour-bit OFF baseline) as sufficient to land at LOW risk; Codex reads the test-fixture self-reference and the un-grounded `sigma_on=0.08` as MEDIUM-HIGH risk for a trading-grade engine. This divergence is itself the most important signal: the *process* is sound, the *empirical grounding* is not yet there.

### Agreed Strengths (2/2 reviewers)
- **OFF-path bit-pour-bit invariance contract** (`atol=1e-12, rtol=0`) is a strong safety net against silent regression while landing a math change.
- **Backward-compat resolution precedence** (D-A3-2) is correctly designed; the legacy-wins-when-explicit pattern preserves all four legacy callsites (autoresearch, rolling_update, infra tests) without migration churn.
- **Lever 2 `_split_level_anomaly`** is conceptually the right SOTA move (anomaly pass-through to far horizon, level shrink to neutral) — it solves the M+30 bowl-flattening problem cleanly.
- **Phase decomposition into 3 sequential plans** with explicit decision IDs and per-task traceability gives clean bisection on regression.
- **`D-FLIP-1`** (gating the production flag flip behind Phase 10 real-data MAE validation) is correctly modelled as an empirical gate, not a calendar event.

### Agreed Concerns — highest priority (2/2 reviewers)

1. **Telemetry warning (D-A2-5) is not asserted by tests.** Both reviewers flag that `max |level - 1.0| > 1e-6 → logger.warning` is fire-and-forget — if SHP-03 silently degrades, no CI signal fires. Codex: MEDIUM. Gemini: LOW + concrete fix proposal (`pytest.warns()` / `caplog` test).
2. **`_split_level_anomaly` grouping is window-dependent.** `groupby().transform("mean")` on the *call window* (not fit-stable cell anchors) means the decomposition depends on the build horizon length and composition. Codex: MEDIUM (HIGH for sum-preservation overclaim). Gemini: MEDIUM (mainly stylistic / docstring fix).
3. **Wave 0 calibration scripts are not committed.** Both reviewers want auditability — Codex wants an immutable calibration report (JSON + git SHA + fixture hash); Gemini wants at minimum a hidden test helper `_run_calibration_dry_run()`. Currently the script lives only in the executor's terminal and the *result* lives in a test-file comment.

### Agreed Concerns — secondary (2/2 reviewers)
4. **API-shape breaking change** (`sigma: float = GAUSSIAN_SIGMA` → `sigma: float | None = None`) is logically required for D-A3-2 resolution but exposes any external code that uses `inspect.signature(ShapeHourly.__init__)` or duck-types the `sigma` default. Plan 05C-03 Task 1 has a backward-compat audit covering the four known callsites, but external/uncovered scripts are unverified.

### Divergent Views — worth investigating

| Topic | Gemini | Codex | What to investigate |
|---|---|---|---|
| **Overall risk** | LOW (Approved) | MEDIUM-HIGH | Reflects different priors on "is test-on-synth-fixture sufficient for a math change?" — the right answer probably depends on whether Phase 10 lands before/after the flag flip. D-FLIP-1 already says "after". |
| **SC#1 / SC#3 measure-then-assert pattern** | Not flagged | HIGH (tautological) | Codex's framing is stronger: if the threshold is derived from the same fit that the test checks, the test only fails if implementation breaks *between calibration and execution*, not if the math is wrong. Mitigation: snapshot the calibration into an immutable JSON sidecar alongside the test, OR cross-validate on a second seed/fixture (Codex's "out-of-sample synthetic family"). |
| **`hydro_weight_sigma_on = 0.08`** | Not flagged ("sound, well-documented") | HIGH (simulated ±10pp ≠ real Swiss anomalies) | The dry-run in RESEARCH §Lever 1 simulates `N(0, 0.10)`; real Swiss `fill[t] - climatological[woy(t)]` may have heavier tails (2018/2022 droughts) or different std. Codex's suggestion of a "real historical hydro anomaly diagnostic" is the right falsifier — can be a one-time offline artifact, not blocking. |

### Recommended actions before / during execution

**Low-cost, high-leverage (incorporate via `/gsd:plan-phase 5bis-B --reviews`):**
1. Add a `test_split_level_anomaly_drift_warning` (caplog-based) to Plan 05C-02 Task 4 — closes the telemetry-silence concern at the cost of ~20 lines.
2. Commit `scripts/calibrate_bowl_thresholds.py` AND a `tests/fixtures/_bowl_calibration_report.json` with observed ratios, fixture sha256, git SHA — closes the auditability concern. Replace the in-comment "observed ratio = X" in `tests/test_shape_hourly_bowl.py` with a `json.load` from this artifact.
3. Add a docstring note in `_split_level_anomaly` and `PFCAssembler.build()` flagging the window-dependence of level / explaining why this is correct under the SHP-03 contract (or, equivalently, document the minimum recommended build horizon).
4. Add a sidecar-load matrix test: `load(pre-5bis-A sidecar)`, `load(5bis-A sidecar)`, `load(5bis-B sidecar)` must all produce identical `sh.sigma` / `sh.hydro_weight_sigma` for the same legacy single-σ caller.

**Higher-cost, deferrable (do not block 5bis-B ship, but track):**
5. Generate one real-data anomaly histogram on Swiss historical reservoirs (one-off, Phase 10 prep) to either confirm or replace `hydro_weight_sigma_on = 0.08`. Tracked as A3 in RESEARCH §Assumptions.
6. Add a second synthetic fixture (`bowl_seed99.parquet`) and parametrize SC#1/SC#3 across both seeds — falsifies "threshold fitted to one trace".
