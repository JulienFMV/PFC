---
phase: 05C-shape-hourly-bowl-deepening
plan: 03
type: execute
wave: 3
depends_on:
  - 05C-01
  - 05C-02
files_modified:
  - pfc_shaping/lt/model/shape_hourly.py
  - tests/test_shape_hourly_bowl.py
  - tests/test_shape_hourly_infra.py
  - tests/fixtures/baseline_pfc_seed42_bowl.parquet
  - .planning/PROJECT.md
  - .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md
autonomous: true
requirements:
  - SHP-04
  - SHP-01
  - SHP-02
  - SHP-03
  - D-A3-1
  - D-A3-2
  - D-A3-3
  - D-A3-4
  - D-A3-5
  - D-A3-6
  - D-A4-5
  - D-A4-7
  - D-A4-9
  - D-FLIP-1
must_haves:
  truths:
    - "`ShapeHourly.__init__` signature is fully extended to its 5bis-B final form: `(self, sigma: float | None = None, sigma_off: float = 0.5, sigma_on: float = 0.25, halflife_days: float = 180.0, hydro_weight_sigma: float | None = None, hydro_weight_sigma_off: float = 0.25, hydro_weight_sigma_on: float = 0.08, use_seasonal_hourly: bool | None = None)` (D-A3-1). The `sigma` default changes from `GAUSSIAN_SIGMA = 0.5` to `None` (RESEARCH Pitfall 2 — breaking change required for D-A3-2 resolution precedence)."
    - "Module-level constants added: `_SIGMA_OFF_DEFAULT = 0.5` (preserves legacy `GAUSSIAN_SIGMA = 0.5`), `_SIGMA_ON_DEFAULT = 0.25` (RESEARCH §Lever 3 confirmed FWHM=0.5887h). `GAUSSIAN_SIGMA = 0.5` constant remains in the module for any external code that imports it directly."
    - "Resolution precedence (D-A3-2) — sigma: if `sigma is not None`, set `self._sigma_off = self._sigma_on = float(sigma)` (legacy single-σ wins for both flag states). Else `self._sigma_off = float(sigma_off)` and `self._sigma_on = float(sigma_on)`. Conflict detection (RESEARCH §Lever 3 Pitfall 3): if `sigma is not None` AND (`sigma_off != _SIGMA_OFF_DEFAULT` OR `sigma_on != _SIGMA_ON_DEFAULT`), emit `logger.warning('ShapeHourly: sigma=%r (legacy) AND sigma_off=%r/sigma_on=%r both passed; legacy sigma wins for both flag states (D-A3-2)', sigma, sigma_off, sigma_on)`. The comparison MUST use the canonical default constants, not the received params (Pitfall 3 — prevents spurious warnings on `ShapeHourly()`)."
    - "Active-value resolution: `self.sigma = self._sigma_on if self._use_seasonal_hourly else self._sigma_off`. This single attribute is what `fit()` reads via `_gaussian_smooth_circular(..., sigma=self.sigma)` at line 279."
    - "Persisted sidecar `shape_hourly.meta.parquet` hyperparams JSON gains three new keys: `sigma_off`, `sigma_on`, `sigma_resolved` (D-A3-3). The legacy `sigma` key remains present (carrying the resolved/active value identical to `sigma_resolved`) for 5bis-A reader compat. Total hyperparams JSON keys after this plan: `{halflife_days, hydro_weight_sigma, hydro_weight_sigma_off, hydro_weight_sigma_on, hydro_weight_sigma_resolved, sigma, sigma_off, sigma_on, sigma_resolved, use_seasonal_hourly}` (10 keys, sort_keys=True preserved)."
    - "`ShapeHourly.load()` cross-plan fallback for sigma (D-A3-3): if sidecar pre-5bis-B lacks `sigma_off` key, set `obj._sigma_off = obj._sigma_on = float(hp.get('sigma', _SIGMA_OFF_DEFAULT))`. Active-value restore: `obj.sigma = float(hp.get('sigma_resolved', hp.get('sigma', _SIGMA_OFF_DEFAULT)))`. Mirrors the pattern from Plan 05C-01 for hydro_weight_sigma."
    - "Telemetry init log (D-A3-6): at the END of `__init__`, log `logger.info('ShapeHourly init: σ_resolved=%.4f, σ_off=%.4f, σ_on=%.4f, flag=%s, hydro_σ_resolved=%.4f, hydro_σ_off=%.4f, hydro_σ_on=%.4f', self.sigma, self._sigma_off, self._sigma_on, self._use_seasonal_hourly, self.hydro_weight_sigma, self._hydro_weight_sigma_off, self._hydro_weight_sigma_on)`. EPFL traceability — every instance creation logs its resolved hyperparams."
    - "Backward-compat preserved for all four legacy callsites (D-A3-5): `ShapeHourly()` (default no-arg, flag=OFF) -> `self.sigma == 0.5` identical legacy; `ShapeHourly(sigma=0.5)` -> legacy wins -> `self.sigma == 0.5` AND `self._sigma_off = self._sigma_on = 0.5`; `ShapeHourly(sigma=0.3)` (autoresearch.py:234) -> legacy wins -> both = 0.3; `ShapeHourly(sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25)` (test_shape_hourly_infra.py:239,250) -> legacy wins for both σ and hydro_weight_σ -> identical legacy semantics."
    - "`tests/fixtures/baseline_pfc_seed42_bowl.parquet` exists, generated AFTER all three levers ship (RESEARCH Pitfall B — MUST be in Plan 05C-03, not Plan 05C-01 or 05C-02). Generated via `build_pfc(seed=42, flag=True)` (5bis-A `tests/fixtures/_generate_baseline.py` reusable entry point). This is the NEW frozen baseline for flag=ON, following the convention D-A4-9 (`baseline_pfc_seed42_{feature_name}.parquet`)."
    - "Three new tests added to `tests/test_shape_hourly_bowl.py`: `test_factors_ptp_deepens_under_flag` (D-A4-5 / SC #1) verifies `np.ptp(sh_on.factors_[('Ete','Ouvrable')]) > np.ptp(sh_off.factors_[('Ete','Ouvrable')]) * SC1_PTP_THRESHOLD` on bowl_seed42; `test_seasonal_solar_winter_evening_delta` (D-A4-7 / SC #2) verifies `abs(mean(price_shape[Dim, Ete, h10-15]) - mean(price_shape[Dim, Hiver, h10-15])) > 5.0` EUR/MWh on bowl_seed42; `test_flag_on_bowl_baseline` (D-A4-9 / new convention) verifies bit-pour-bit regression against `tests/fixtures/baseline_pfc_seed42_bowl.parquet` at `atol=1e-12, rtol=0` + identical columns/dtypes/index/sort order."
    - "Wave 0 task in this plan re-calibrates SC1_PTP_THRESHOLD now that all three levers are active (Plan 05C-01's threshold measured Lever-1-only; Plan 05C-03 measures the combined Lever-1+2+3 gain on the bowl fixture). The plancher 1.05 (RESEARCH §Lever 1) is preserved as floor."
    - "`tests/test_shape_hourly_infra.py` — second authorized surgical update (D-A3-3): the same two tests updated in Plan 05C-01 (`test_hyperparams_row`, `test_save_unfitted_hyperparams_correct`) are extended to also accept `sigma_off`, `sigma_on`, `sigma_resolved` keys. After this plan, the full hyperparams JSON dict for `_minimal_fitted_sh(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7)` equals `{halflife_days: 90.0, hydro_weight_sigma: 0.3, hydro_weight_sigma_off: 0.7, hydro_weight_sigma_on: 0.7, hydro_weight_sigma_resolved: 0.7, sigma: 0.3, sigma_off: 0.3, sigma_on: 0.3, sigma_resolved: 0.3, use_seasonal_hourly: False}` (10 keys). For `ShapeHourly(sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25)` the dict equals `{halflife_days: 180.0, hydro_weight_sigma: 0.25, hydro_weight_sigma_off: 0.25, hydro_weight_sigma_on: 0.25, hydro_weight_sigma_resolved: 0.25, sigma: 0.5, sigma_off: 0.5, sigma_on: 0.5, sigma_resolved: 0.5, use_seasonal_hourly: False}`. RESEARCH Pitfall 4."
    - "`.planning/PROJECT.md` gains a `Key Decisions` entry D-FLIP-1: `2026-05-19 | Flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE livré default OFF | Flip default ON gated par Phase 10 success (Δ MAE bloc ≤ -1.5 EUR/MWh vs HFC OMPEX). EPFL/SOTA principle: no production change without empirical validation gate.` (D-FLIP-1)."
    - "`.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` (pre-doc, deferred ROADMAP backlog cleanup item from CONTEXT.md `## Deferred Ideas`) gains a header note `**STATUS: SUPERSEDED.** This pre-doc was the original 5bis context; post-2026-05-18 adversarial panel review, 5bis was split into 5bis-A (no-op infrastructure, see .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/) + 5bis-B (math change, see .planning/phases/05C-shape-hourly-bowl-deepening/). Retained for historical reference only.`"
    - "Cross-cutting truth (appears in all 3 plans): `flag=OFF baseline 5bis-A preserved at atol=1e-12 rtol=0`. After this plan, `test_flag_off_bit_for_bit_baseline` (Plan 05C-01) + `test_baseline_regression[False]` (5bis-A `test_shape_hourly_infra.py`) both continue to pass at the tightest tolerance. The σ resolution under flag=OFF uses `self._sigma_off = 0.5` (or legacy override), giving byte-identical math to 5bis-A."
  artifacts:
    - path: "pfc_shaping/lt/model/shape_hourly.py"
      provides: "Final 5bis-B `__init__` signature with sigma_off/_on + resolution precedence + conflict detection + telemetry init. Save/load sidecar carries 10 hyperparams JSON keys. Cross-plan fallback for sigma_off/_on."
      contains: "sigma_off: float = 0.5"
    - path: "tests/fixtures/baseline_pfc_seed42_bowl.parquet"
      provides: "Frozen flag=ON baseline (new convention D-A4-9, `baseline_pfc_seed42_{feature}` pattern). Generated AFTER all three levers ship (Pitfall B)."
    - path: "tests/test_shape_hourly_bowl.py"
      provides: "Three new tests: D-A4-5 (SC #1 ptp deepening), D-A4-7 (SC #2 seasonal delta), D-A4-9 (flag=ON baseline). Re-calibrated SC1_PTP_THRESHOLD."
      contains: "test_factors_ptp_deepens_under_flag"
    - path: "tests/test_shape_hourly_infra.py"
      provides: "Second authorized update to hyperparams JSON key-set tests: extended to include sigma_off/_on/_resolved (10 keys total)."
      contains: "sigma_off"
    - path: ".planning/PROJECT.md"
      provides: "D-FLIP-1 entry in Key Decisions table — flag flip gated by Phase 10 success (no auto-flip post-merge)."
      contains: "D-FLIP-1"
    - path: ".planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md"
      provides: "SUPERSEDED note at the top of the pre-doc (deferred ROADMAP backlog cleanup item)."
      contains: "SUPERSEDED"
  key_links:
    - from: "pfc_shaping/lt/model/shape_hourly.py::__init__"
      to: "self.sigma (active value resolved from sigma_off/_on)"
      via: "self.sigma = self._sigma_on if self._use_seasonal_hourly else self._sigma_off"
      pattern: "_sigma_on"
    - from: "tests/test_shape_hourly_bowl.py::test_flag_on_bowl_baseline"
      to: "tests/fixtures/baseline_pfc_seed42_bowl.parquet"
      via: "assert_frame_equal(build_baseline_pfc(seed=42, flag=True), baseline_bowl, check_exact=False, atol=1e-12, rtol=0) + columns/dtypes/index match"
      pattern: "baseline_pfc_seed42_bowl"
    - from: ".planning/PROJECT.md::Key Decisions"
      to: "D-FLIP-1 row referencing Phase 10 success gate"
      via: "table row entry"
      pattern: "D-FLIP-1"
---

<objective>
Implement Lever 3 of Phase 5bis-B: σ smoothing paramétrisation (`sigma_off=0.5` / `sigma_on=0.25`) with full backward-compat for legacy `ShapeHourly(sigma=X)` callsites (D-A3-1, D-A3-2). Extend the sidecar `shape_hourly.meta.parquet` to persist the new `sigma_off` / `sigma_on` / `sigma_resolved` keys (D-A3-3) with cross-plan fallback at load. Add the EPFL traceability telemetry init log (D-A3-6).

Generate the NEW frozen flag=ON baseline `tests/fixtures/baseline_pfc_seed42_bowl.parquet` (RESEARCH Pitfall B — must happen ONLY in this plan, after all three levers ship). This establishes the convention D-A4-9 (`baseline_pfc_seed42_{feature_name}.parquet`) that Phase 5 / 5ter / future shape phases will follow.

Append three new tests to `tests/test_shape_hourly_bowl.py`:
- `test_factors_ptp_deepens_under_flag` (D-A4-5 / SC #1) — the ptp-deepening proof of the bowl. Re-calibrate `SC1_PTP_THRESHOLD` (Wave 0) now that all three levers contribute jointly (Plan 05C-01's threshold was Lever-1-only).
- `test_seasonal_solar_winter_evening_delta` (D-A4-7 / SC #2) — the EUR/MWh delta proof on synthetic data (RESEARCH §SC #2 delta analytique vérifié shows expected delta ~11.5 EUR/MWh, well above the 5 EUR/MWh threshold).
- `test_flag_on_bowl_baseline` (D-A4-9 / NEW convention) — bit-pour-bit regression against the new baseline at `atol=1e-12, rtol=0`.

Update `tests/test_shape_hourly_infra.py` for the second and final time (D-A3-3): the same two tests modified in Plan 05C-01 are extended to also assert `sigma_off` / `sigma_on` / `sigma_resolved` keys in the hyperparams JSON (RESEARCH Pitfall 4 — only authorized infra touch in 5bis-B, second wave).

Update `.planning/PROJECT.md` to record the D-FLIP-1 decision: flag flip default OFF → ON is GATED by Phase 10 success (Δ MAE bloc ≤ -1.5 EUR/MWh vs HFC OMPEX). No auto-flip post-merge. This is the EPFL/SOTA principle: "no production change without empirical validation."

Add a SUPERSEDED note to `.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` (deferred ROADMAP backlog item — pre-doc was the original 5bis context, replaced by 5bis-A + 5bis-B post-adversarial-panel split).

Purpose: This plan ships the third and final lever, establishes the new baseline convention, validates ALL FIVE Success Criteria from ROADMAP.md (SC #1 ptp, SC #2 EUR/MWh delta, SC #3 M+30 amplitude, SC #4 flag=OFF bit-pour-bit, SC #5 247+ green), records the flag flip strategy as a permanent project decision, and cleans up the deferred pre-doc.

Output: 1 modified production file (`shape_hourly.py`, final 5bis-B form), 1 new frozen baseline parquet, 3 new tests in the bowl module, 1 second surgical update to the infra test file, 1 PROJECT.md decision entry, 1 SUPERSEDED note on the pre-doc. Test suite goes from 251 to 254 passing (4 skipped preserved). All five SC validated.
</objective>

<execution_context>
@.claude/get-shit-done/workflows/execute-plan.md
@.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/STATE.md
@.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md
@.planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md
@.planning/phases/05C-shape-hourly-bowl-deepening/05C-VALIDATION.md
@.planning/phases/05C-shape-hourly-bowl-deepening/05C-01-PLAN.md
@.planning/phases/05C-shape-hourly-bowl-deepening/05C-02-PLAN.md
@.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md
@.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md
@pfc_shaping/lt/model/shape_hourly.py
@pfc_shaping/lt/model/assembler.py
@tests/test_shape_hourly_bowl.py
@tests/test_shape_hourly_infra.py
@tests/fixtures/_generate_baseline.py
@tests/fixtures/_generate_bowl_fixture.py
@tests/fixtures/baseline_pfc_seed42.parquet

<interfaces>
Key contracts to consume (Plans 05C-01 + 05C-02 outputs):

From `pfc_shaping/lt/model/shape_hourly.py` (Plans 05C-01 + 05C-02 final state):
- Module constants: `GAUSSIAN_SIGMA = 0.5`, `_HYDRO_WEIGHT_SIGMA_OFF_DEFAULT = 0.25`, `_HYDRO_WEIGHT_SIGMA_ON_DEFAULT = 0.08`, `_FLAG_ENV_VAR`, `_META_SIDECAR_SUFFIX`. THIS PLAN ADDS: `_SIGMA_OFF_DEFAULT = 0.5`, `_SIGMA_ON_DEFAULT = 0.25`.
- `__all__` exists (Plan 05C-02) listing `["ShapeHourly", "GAUSSIAN_SIGMA", "_FLAG_ENV_VAR", "_resolve_flag", "_meta_path", "_split_level_anomaly"]`. No update needed here.
- `class ShapeHourly`: signature currently `__init__(sigma=GAUSSIAN_SIGMA, halflife_days=180.0, hydro_weight_sigma=None, hydro_weight_sigma_off=0.25, hydro_weight_sigma_on=0.08, use_seasonal_hourly=None)`. THIS PLAN MUTATES `sigma=GAUSSIAN_SIGMA` to `sigma=None` and ADDS `sigma_off=0.5, sigma_on=0.25` kwargs.
- `_split_level_anomaly` (Plan 05C-02) module-level helper. No update needed here.
- Existing private attrs: `self._use_seasonal_hourly`, `self._hydro_weight_sigma_off`, `self._hydro_weight_sigma_on`. THIS PLAN ADDS: `self._sigma_off`, `self._sigma_on`.

From `pfc_shaping/lt/model/assembler.py` (Plans 05C-01 + 05C-02 final state):
- Line ~333 has the flag-gated branch (legacy single-line under flag=OFF, split-based level damping under flag=ON). No edits needed in this plan.

From `tests/test_shape_hourly_bowl.py` (Plans 05C-01 + 05C-02 final state):
- 4 tests already passing: `test_hydro_kernel_uses_per_timestamp_climatological_target`, `test_flag_off_bit_for_bit_baseline`, `test_split_level_anomaly_invariant`, `test_f_H_amplitude_preserved_at_M30`.
- Constants: `SC1_PTP_THRESHOLD` (Plan 05C-01 Wave 0, Lever-1-only), `SC3_M30_AMPLITUDE_THRESHOLD` (Plan 05C-02 Wave 0, with Lever 2 active).
- Module-level fixture: `_bowl_pfc_setup` (Plan 05C-02 Task 4 if implemented as scope="module").

From `tests/fixtures/_generate_baseline.py`:
- `build_pfc(seed: int = 42, flag: bool = False) -> pd.DataFrame` — reusable entry point. THIS PLAN consumes it with `flag=True` to generate `baseline_pfc_seed42_bowl.parquet`.

Test count at start of this plan: `251 passed, 4 skipped`. Target after this plan: `254 passed, 4 skipped`.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Final ShapeHourly.__init__ extension — sigma_off/sigma_on + resolution + telemetry + sidecar 10-key schema</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py (current state — Plans 05C-01 + 05C-02 final form; in particular the `__init__` body where `_use_seasonal_hourly` and `_hydro_weight_sigma_*` are already resolved)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A3-1..D-A3-6 (full signature spec, resolution precedence, telemetry)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Lever 3 (final signature, conflict-detection pattern, save/load JSON spec)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Pitfalls 2 (sigma=None breaking change), §Pitfalls 3 (conflict detection must use canonical defaults)
    - pfc_shaping/pipeline/autoresearch.py:234 and pfc_shaping/pipeline/rolling_update.py:365 (legacy callsites that must keep working)
  </read_first>
  <action>
    Make four surgical edits to `pfc_shaping/lt/model/shape_hourly.py`:

    Edit A — Add module-level constants near the existing `_HYDRO_WEIGHT_SIGMA_OFF_DEFAULT`/`_HYDRO_WEIGHT_SIGMA_ON_DEFAULT` (Plan 05C-01) definitions:
    - `_SIGMA_OFF_DEFAULT: float = 0.5  # Plan 05C-03 D-A3-4: preserves legacy GAUSSIAN_SIGMA = 0.5`
    - `_SIGMA_ON_DEFAULT: float = 0.25  # Plan 05C-03 D-A3-4 / RESEARCH §Lever 3 CONFIRMED FWHM=0.5887h on hourly grid`

    Keep the existing `GAUSSIAN_SIGMA = 0.5` constant — external code may import it directly. The new `_SIGMA_OFF_DEFAULT` and `GAUSSIAN_SIGMA` reference the same numeric value but the canonical default for the new resolution logic is `_SIGMA_OFF_DEFAULT`.

    Edit B — Mutate the `__init__` signature (currently `sigma: float = GAUSSIAN_SIGMA`):
    Change `sigma: float = GAUSSIAN_SIGMA` to `sigma: float | None = None` (RESEARCH Pitfall 2 — required for D-A3-2 resolution to distinguish "user passed sigma" vs "user did not"). Add two new keyword arguments AFTER `sigma` and BEFORE `halflife_days`: `sigma_off: float = _SIGMA_OFF_DEFAULT` and `sigma_on: float = _SIGMA_ON_DEFAULT`. The full final signature is:

    `def __init__(self, sigma: float | None = None, sigma_off: float = _SIGMA_OFF_DEFAULT, sigma_on: float = _SIGMA_ON_DEFAULT, halflife_days: float = 180.0, hydro_weight_sigma: float | None = None, hydro_weight_sigma_off: float = _HYDRO_WEIGHT_SIGMA_OFF_DEFAULT, hydro_weight_sigma_on: float = _HYDRO_WEIGHT_SIGMA_ON_DEFAULT, use_seasonal_hourly: bool | None = None) -> None:`

    Note: also switch the hydro_weight_sigma_off/_on defaults from the hardcoded `0.25`/`0.08` (Plan 05C-01) to the module constants `_HYDRO_WEIGHT_SIGMA_OFF_DEFAULT` / `_HYDRO_WEIGHT_SIGMA_ON_DEFAULT` for consistency. This is a pure refactor — values unchanged.

    Inside the body, AFTER `self._use_seasonal_hourly = _resolve_flag(use_seasonal_hourly)` and BEFORE the hydro_weight_sigma resolution block (Plan 05C-01), insert the sigma resolution block (mirroring the hydro_weight_sigma pattern from Plan 05C-01):
    - If `sigma is not None`: detect conflict — if `sigma_off != _SIGMA_OFF_DEFAULT` or `sigma_on != _SIGMA_ON_DEFAULT`, emit `logger.warning("ShapeHourly: sigma=%r (legacy) AND sigma_off=%r/sigma_on=%r both passed; legacy sigma wins for both flag states (D-A3-2)", sigma, sigma_off, sigma_on)`. Then `self._sigma_off = self._sigma_on = float(sigma)`.
    - Else: `self._sigma_off = float(sigma_off)`, `self._sigma_on = float(sigma_on)`.
    - Active value: `self.sigma = self._sigma_on if self._use_seasonal_hourly else self._sigma_off`.

    The conflict-detection comparison MUST use the canonical default CONSTANTS (`_SIGMA_OFF_DEFAULT`, `_SIGMA_ON_DEFAULT`), NOT the received params. This is critical per RESEARCH Pitfall 3 — comparing `sigma_off != sigma_off` against the parameter would always be False and silently never warn.

    Edit C — At the END of `__init__` (after both resolutions complete), insert the telemetry init log (D-A3-6):
    `logger.info("ShapeHourly init: σ_resolved=%.4f, σ_off=%.4f, σ_on=%.4f, flag=%s, hydro_σ_resolved=%.4f, hydro_σ_off=%.4f, hydro_σ_on=%.4f", self.sigma, self._sigma_off, self._sigma_on, self._use_seasonal_hourly, self.hydro_weight_sigma, self._hydro_weight_sigma_off, self._hydro_weight_sigma_on)`

    This single log line is emitted once per `ShapeHourly()` construction. It does NOT pollute test output excessively because most tests construct ShapeHourly < 10 times. If specific tests want to suppress it, they can use `caplog.set_level(logging.WARNING)` — but this is not a refactor request.

    Edit D — Extend `save()` hyperparams JSON (around shape_hourly.py:518-529, Plan 05C-01 already extended hydro keys):
    The current JSON dict written by save() after Plan 05C-01 contains 7 keys: `halflife_days, hydro_weight_sigma, hydro_weight_sigma_off, hydro_weight_sigma_on, hydro_weight_sigma_resolved, sigma, use_seasonal_hourly`. Add THREE more keys: `sigma_off: self._sigma_off, sigma_on: self._sigma_on, sigma_resolved: self.sigma`. Total = 10 keys, sort_keys=True preserved. The legacy `sigma` key remains (carrying the resolved/active value identical to `sigma_resolved`) for 5bis-A reader compat.

    Edit E — Extend `load()` cross-plan fallback for sigma (around shape_hourly.py:562-575, Plan 05C-01 already extended hydro keys):
    Currently the load block has the hydro_weight_sigma cross-plan fallback (Plan 05C-01) and the legacy `obj.sigma = hp.get("sigma", obj.sigma)` line. Replace the latter with:
    - If `"sigma_off" in hp`: `obj._sigma_off = float(hp["sigma_off"])`, `obj._sigma_on = float(hp["sigma_on"])`.
    - Else (sidecar 5bis-A or pre-5bis-B, only `sigma` legacy key present): `legacy_sigma = float(hp.get("sigma", _SIGMA_OFF_DEFAULT))`, `obj._sigma_off = obj._sigma_on = legacy_sigma`.
    - Active value: `obj.sigma = float(hp.get("sigma_resolved", hp.get("sigma", _SIGMA_OFF_DEFAULT)))`.

    The existing `obj._use_seasonal_hourly` restore from `hp["use_seasonal_hourly"]` (5bis-A D-07) is preserved unchanged.

    Backward-compat audit (run as a sanity check after all edits):
    - `ShapeHourly()` — flag=OFF, no sigma arg -> `_sigma_off=0.5, _sigma_on=0.25, sigma=0.5` (legacy default semantics preserved); `_hydro_weight_sigma_off=0.25, _hydro_weight_sigma_on=0.08, hydro_weight_sigma=0.25`.
    - `ShapeHourly(sigma=0.5)` — legacy callsite -> legacy wins -> `_sigma_off=_sigma_on=0.5, sigma=0.5`. No warning emitted (sigma_off/sigma_on at canonical defaults).
    - `ShapeHourly(sigma=0.3)` (autoresearch.py:234 pattern) -> legacy wins -> `_sigma_off=_sigma_on=0.3, sigma=0.3`. No warning.
    - `ShapeHourly(use_seasonal_hourly=True)` — flag=ON, no sigma arg -> `_sigma_off=0.5, _sigma_on=0.25, sigma=0.25` (new flag=ON default).
    - `ShapeHourly(use_seasonal_hourly=True, sigma_on=0.30)` — explicit sigma_on override under flag=ON -> `_sigma_off=0.5, _sigma_on=0.30, sigma=0.30`. No warning.
    - `ShapeHourly(sigma=0.5, sigma_on=0.30)` — conflict (legacy + explicit on) -> WARNING emitted, legacy wins -> `_sigma_off=_sigma_on=0.5, sigma=0.5`.
  </action>
  <verify>
    <automated>python -c "import logging; logging.basicConfig(level=logging.DEBUG); from pfc_shaping.lt.model.shape_hourly import ShapeHourly, _SIGMA_OFF_DEFAULT, _SIGMA_ON_DEFAULT; assert _SIGMA_OFF_DEFAULT == 0.5; assert _SIGMA_ON_DEFAULT == 0.25; sh = ShapeHourly(); assert sh._sigma_off == 0.5 and sh._sigma_on == 0.25 and sh.sigma == 0.5, f'default: {sh._sigma_off},{sh._sigma_on},{sh.sigma}'; sh = ShapeHourly(sigma=0.5); assert sh._sigma_off == 0.5 and sh._sigma_on == 0.5 and sh.sigma == 0.5, 'legacy sigma=0.5'; sh = ShapeHourly(sigma=0.3); assert sh._sigma_off == 0.3 and sh._sigma_on == 0.3 and sh.sigma == 0.3, 'legacy sigma=0.3'; sh = ShapeHourly(use_seasonal_hourly=True); assert sh._sigma_off == 0.5 and sh._sigma_on == 0.25 and sh.sigma == 0.25, 'flag ON default'; sh = ShapeHourly(use_seasonal_hourly=True, sigma_on=0.30); assert sh._sigma_off == 0.5 and sh._sigma_on == 0.30 and sh.sigma == 0.30, 'flag ON override'; print('OK init backward-compat 5/5')" && pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "_SIGMA_OFF_DEFAULT = 0.5" pfc_shaping/lt/model/shape_hourly.py` matches a module-level definition.
    - `grep -n "_SIGMA_ON_DEFAULT = 0.25" pfc_shaping/lt/model/shape_hourly.py` matches a module-level definition.
    - `grep -n "sigma: float | None = None" pfc_shaping/lt/model/shape_hourly.py` matches `__init__` signature (Pitfall 2 — sigma=None breaking change required).
    - `grep -n "sigma_off: float =" pfc_shaping/lt/model/shape_hourly.py` matches `__init__` signature.
    - `grep -n "sigma_on: float =" pfc_shaping/lt/model/shape_hourly.py` matches `__init__` signature.
    - `grep -c "sigma_resolved" pfc_shaping/lt/model/shape_hourly.py` reports ≥ 2 (one in save, one in load).
    - `grep -q "ShapeHourly init:" pfc_shaping/lt/model/shape_hourly.py` exits 0 (telemetry init log present).
    - The python init-sanity command above prints `OK init backward-compat 5/5`.
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` exits 0 — the bit-pour-bit no-op contract is preserved across Lever 3.
    - `pytest tests/test_shape_hourly_infra.py::TestBaselineRegression -x` or equivalent class exits 0 — 5bis-A regression preserved.
    - `pytest tests/test_shape_hourly_infra.py -x -q` may show 1-2 failures on `test_hyperparams_row` / `test_save_unfitted_hyperparams_correct` due to the now-extended 10-key JSON schema — those are updated in Task 3.
  </acceptance_criteria>
  <done>ShapeHourly.__init__ at its final 5bis-B form. All five backward-compat scenarios verified. flag=OFF baseline still bit-pour-bit identical to 5bis-A.</done>
</task>

<task type="auto">
  <name>Task 2: Update `tests/test_shape_hourly_infra.py` hyperparams JSON tests for the FINAL 10-key schema (second authorized infra surgical update)</name>
  <files>tests/test_shape_hourly_infra.py</files>
  <read_first>
    - tests/test_shape_hourly_infra.py:197-263 (the two tests `test_hyperparams_row` and `test_save_unfitted_hyperparams_correct` — already updated by Plan 05C-01 for hydro keys; this plan extends them for sigma keys)
    - pfc_shaping/lt/model/shape_hourly.py (Task 1 final state — confirms the 10-key JSON schema)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Pitfall 4 (explicit authorization for this second surgical update)
  </read_first>
  <action>
    Update the SAME two tests in `tests/test_shape_hourly_infra.py` that Plan 05C-01 Task 3 modified (the hydro_weight_sigma extension). This second update adds the `sigma_off`, `sigma_on`, `sigma_resolved` keys.

    Update 1 — `test_hyperparams_row` (around line 197):
    After Plan 05C-01, the test asserts equality against the 7-key dict `{halflife_days: 90.0, hydro_weight_sigma: 0.7, hydro_weight_sigma_off: 0.7, hydro_weight_sigma_on: 0.7, hydro_weight_sigma_resolved: 0.7, sigma: 0.3, use_seasonal_hourly: False}`. Extend to the 10-key dict: `{halflife_days: 90.0, hydro_weight_sigma: 0.7, hydro_weight_sigma_off: 0.7, hydro_weight_sigma_on: 0.7, hydro_weight_sigma_resolved: 0.7, sigma: 0.3, sigma_off: 0.3, sigma_on: 0.3, sigma_resolved: 0.3, use_seasonal_hourly: False}`. The `0.3` for sigma_off/_on/_resolved derives from the legacy callsite `ShapeHourly(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7)` in `_minimal_fitted_sh` (D-A3-2 backward-compat: legacy wins -> off=on=0.3).

    Update 2 — `test_save_unfitted_hyperparams_correct` (around line 249):
    After Plan 05C-01, the test asserts the 7-key dict for `ShapeHourly(sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25)`. Extend to: `{halflife_days: 180.0, hydro_weight_sigma: 0.25, hydro_weight_sigma_off: 0.25, hydro_weight_sigma_on: 0.25, hydro_weight_sigma_resolved: 0.25, sigma: 0.5, sigma_off: 0.5, sigma_on: 0.5, sigma_resolved: 0.5, use_seasonal_hourly: False}`. The `0.5` for sigma_off/_on/_resolved derives from legacy `sigma=0.5` (D-A3-2 backward-compat).

    For BOTH updates, augment the test docstring with: "Updated by Plan 05C-03 (D-A3-3 / RESEARCH Pitfall 4 second wave): hyperparams JSON gains the sigma_off/_on/_resolved triplet alongside the hydro_weight_sigma_off/_on/_resolved triplet added by Plan 05C-01. Total schema = 10 keys."

    DO NOT touch any OTHER test in `test_shape_hourly_infra.py`. The 5bis-A test_baseline_regression must continue to pass at `atol=1e-12, rtol=0` because the legacy sigma path is preserved bit-pour-bit (Task 1 backward-compat audit scenario 2).

    Verify the roundtrip test `test_save_load_full_roundtrip` (5bis-A Plan 05B-05) still passes: the test asserts equality on `sh.sigma` after save->load. With Task 1's load() setting `obj.sigma = hp.get("sigma_resolved", ...)` and save() writing `sigma_resolved = self.sigma`, the roundtrip preserves the resolved value identically. No update needed.
  </action>
  <verify>
    <automated>pytest tests/test_shape_hourly_infra.py -x -q 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_shape_hourly_infra.py -x -q` exits 0 reporting `247 passed, 4 skipped` (5bis-A baseline preserved including the parametrized `test_baseline_regression[False|True]`).
    - `grep -q "sigma_off.*: 0.3" tests/test_shape_hourly_infra.py` exits 0 (new key in `test_hyperparams_row`).
    - `grep -q "sigma_off.*: 0.5" tests/test_shape_hourly_infra.py` exits 0 (new key in `test_save_unfitted_hyperparams_correct`).
    - `grep -q "Plan 05C-03" tests/test_shape_hourly_infra.py` exits 0 (traceability comment present).
    - `pytest tests/test_shape_hourly_infra.py::TestSaveBasic::test_hyperparams_row -x` exits 0.
    - `pytest tests/test_shape_hourly_infra.py::TestSaveUnfitted::test_save_unfitted_hyperparams_correct -x` exits 0 (class name may differ — match actual containing class).
  </acceptance_criteria>
  <done>Two infra-suite tests updated to the FINAL 10-key hyperparams JSON schema. 5bis-A regression still bit-pour-bit at atol=1e-12 rtol=0.</done>
</task>

<task type="auto">
  <name>Task 3 (Wave 0): Generate and commit `tests/fixtures/baseline_pfc_seed42_bowl.parquet` + re-calibrate SC1_PTP_THRESHOLD</name>
  <files>tests/fixtures/baseline_pfc_seed42_bowl.parquet, tests/test_shape_hourly_bowl.py</files>
  <read_first>
    - tests/fixtures/_generate_baseline.py (`build_pfc(seed=42, flag=True)` is the reusable entry point — already exists from 5bis-A)
    - tests/test_shape_hourly_bowl.py (current `SC1_PTP_THRESHOLD` constant — Plan 05C-01 measured Lever-1-only)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Pitfall B (`baseline_pfc_seed42_bowl.parquet` MUST be generated AFTER all three levers ship, i.e. in THIS plan)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A4-5 (SC #1 threshold measure-then-assert), D-A4-9 (new baseline convention)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-01-PLAN.md (5bis-A baseline freeze pattern — same workflow applied here for flag=ON)
  </read_first>
  <action>
    This task has two parts: (a) generate and commit the new frozen flag=ON baseline; (b) re-calibrate SC1_PTP_THRESHOLD now that all three levers are active.

    Part A — Generate baseline_pfc_seed42_bowl.parquet:
    Run the existing entry point from `tests/fixtures/_generate_baseline.py` with `flag=True`:
    ```
    python -c "from tests.fixtures._generate_baseline import build_pfc; df = build_pfc(seed=42, flag=True); df.to_parquet('tests/fixtures/baseline_pfc_seed42_bowl.parquet', index=True); print(f'rows={len(df)} cols={list(df.columns)} hash={hash(df[\"price_shape\"].values.tobytes())}')"
    ```

    The output parquet `tests/fixtures/baseline_pfc_seed42_bowl.parquet` is the NEW frozen baseline per D-A4-9 convention. It must be committed to git. Verify it differs from the legacy `baseline_pfc_seed42.parquet` in the relevant columns (`price_shape`, `f_H` at minimum) — the difference is the math change that 5bis-B delivers. Use this snippet to confirm:
    ```
    python -c "import pandas as pd, numpy as np; off = pd.read_parquet('tests/fixtures/baseline_pfc_seed42.parquet'); on = pd.read_parquet('tests/fixtures/baseline_pfc_seed42_bowl.parquet'); diff = (off['price_shape'] - on['price_shape']).abs(); print(f'price_shape max abs diff = {diff.max():.4f} EUR/MWh, mean = {diff.mean():.4f}, n_nonzero(>1e-12) = {int((diff > 1e-12).sum())}/{len(diff)}'); assert diff.max() > 0.01, f'baselines too similar: max diff {diff.max()} — flag=ON math change is not actually active'"
    ```
    Expected: max abs diff > 0.01 EUR/MWh (the bowl deepening + per-timestamp hydro target). If the diff is near-zero, something is broken — STOP and investigate.

    Document the diff in a one-line comment at the top of the parquet's companion README entry (update `tests/fixtures/README.md` if it exists; otherwise note the diff in this plan's SUMMARY).

    Part B — Re-calibrate SC1_PTP_THRESHOLD on the bowl fixture:
    Plan 05C-01's Wave 0 measured the threshold with Lever 1 only (Plan 05C-02 / 05C-03 not yet shipped). Now that all three levers are active, re-measure:

    ```
    python -c "
    import numpy as np
    from tests.fixtures._generate_bowl_fixture import build_bowl_fixture
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    from pfc_shaping.lt.model.shape_hourly import ShapeHourly
    epex_df, hydro_df = build_bowl_fixture(seed=42)
    cal = enrich_15min_index(epex_df.index, country='CH')
    sh_off = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, cal, hydro_df)
    sh_on  = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal, hydro_df)
    key = ('Ete', 'Ouvrable')
    if key not in sh_off.factors_ or key not in sh_on.factors_:
        key = list(sh_off.factors_.keys())[0]
        print(f'(Ete,Ouvrable) absent — falling back to {key}')
    ratio = float(np.ptp(sh_on.factors_[key]) / np.ptp(sh_off.factors_[key]))
    threshold = max(ratio - 0.15, 1.05)
    print(f'SC#1 observed ratio (all 3 levers) = {ratio:.4f}')
    print(f'Committed SC1_PTP_THRESHOLD = {threshold:.4f}')
    "
    ```

    Update `tests/test_shape_hourly_bowl.py`:
    - Replace the existing `SC1_PTP_THRESHOLD = <Plan 05C-01 value>` with `SC1_PTP_THRESHOLD = <Plan 05C-03 value>`.
    - Update the comment to: `# Wave 0 re-calibrated by Plan 05C-03 on bowl_seed42 with all three levers active. Plan 05C-01 measured Lever-1-only = X.XXXX, Plan 05C-03 measures combined = Y.YYYY. Threshold = max(ratio - 0.15, 1.05).`

    Sanity bounds: the combined ratio should be ≥ the Lever-1-only ratio measured in Plan 05C-01 (the additional Levers 2 and 3 only add bowl gain, never subtract). If `ratio_combined < ratio_lever1_only`, STOP and investigate. Per RESEARCH §Lever 1 + §Lever 3, the analytic estimate is `ratio ≈ 1.13-1.18 × 1.025 ≈ 1.16-1.21`. Plancher 1.05.

    Commit the new baseline parquet AND the updated test file in this task. DO NOT yet add the consuming tests (Task 4).
  </action>
  <verify>
    <automated>test -f tests/fixtures/baseline_pfc_seed42_bowl.parquet && python -c "import pandas as pd, numpy as np; off = pd.read_parquet('tests/fixtures/baseline_pfc_seed42.parquet'); on = pd.read_parquet('tests/fixtures/baseline_pfc_seed42_bowl.parquet'); assert list(off.columns) == list(on.columns), 'columns must match between baselines'; assert off.index.equals(on.index), 'indexes must match'; diff = (off['price_shape'] - on['price_shape']).abs(); assert diff.max() > 0.01, f'baselines too similar: {diff.max()}'; print(f'OK new_baseline rows={len(on)} max_diff={diff.max():.4f}')" && python -c "import re; src = open('tests/test_shape_hourly_bowl.py').read(); m = re.search(r'SC1_PTP_THRESHOLD\s*=\s*([0-9.]+)', src); assert m, 'SC1_PTP_THRESHOLD constant not found'; val = float(m.group(1)); assert 1.05 <= val <= 1.50, f'threshold {val} out of plausible bounds'; assert 'Plan 05C-03' in src, 'Plan 05C-03 traceability comment missing'; print(f'OK SC1_PTP_THRESHOLD={val:.4f}')"</automated>
  </verify>
  <acceptance_criteria>
    - `test -f tests/fixtures/baseline_pfc_seed42_bowl.parquet` exits 0.
    - The diff sanity verify above prints `OK new_baseline rows=...` and `max_diff > 0.01`.
    - `SC1_PTP_THRESHOLD` in `tests/test_shape_hourly_bowl.py` is in the bounds `[1.05, 1.50]`.
    - `grep -q "Plan 05C-03" tests/test_shape_hourly_bowl.py` exits 0 (re-calibration traceability).
    - `git status tests/fixtures/baseline_pfc_seed42_bowl.parquet` shows the file is staged/untracked (new commit candidate).
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` continues to exit 0 (Plan 05C-01's baseline regression unchanged — that test consumes `baseline_pfc_seed42.parquet`, not the new `_bowl` baseline).
  </acceptance_criteria>
  <done>New frozen flag=ON baseline committed. SC1_PTP_THRESHOLD re-calibrated for the joint 3-lever gain. Pitfall B respected: baseline generated AFTER all three levers ship.</done>
</task>

<task type="auto">
  <name>Task 4: Append three tests to `tests/test_shape_hourly_bowl.py`: D-A4-5 (SC #1), D-A4-7 (SC #2), D-A4-9 (new baseline)</name>
  <files>tests/test_shape_hourly_bowl.py</files>
  <read_first>
    - tests/test_shape_hourly_bowl.py (Plans 05C-01/02 + Task 3 state — 4 existing tests + updated thresholds + new baseline available)
    - tests/fixtures/baseline_pfc_seed42_bowl.parquet (Task 3 output — frozen flag=ON baseline)
    - tests/fixtures/_generate_bowl_fixture.py (`build_bowl_fixture()` reusable)
    - tests/fixtures/_generate_baseline.py (`build_pfc(seed, flag)` reusable for the D-A4-9 baseline regression)
    - pfc_shaping/lt/model/shape_hourly.py (Task 1 final state)
    - pfc_shaping/lt/model/assembler.py (Plan 05C-02 final state — flag-gated branch active)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A4-5, D-A4-7, D-A4-9 (test specs)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Validation Architecture rows 3, 5, 7 (SC #1, SC #2, D-A4-9 verification details)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md §1 (tolerance contract atol=1e-12 rtol=0 + identical columns/dtypes/index — applies to D-A4-9 baseline regression)
  </read_first>
  <action>
    Append three tests to `tests/test_shape_hourly_bowl.py`. Do NOT modify existing tests, fixture, threshold constants, or imports unless a new import is strictly required.

    Test 5 — `test_factors_ptp_deepens_under_flag` (D-A4-5 / SC #1):

    Docstring: cite D-A4-5 + SC #1 + RESEARCH §Lever 1 (the SC #1 threshold rationale and Wave 0 re-calibration in Task 3). Reference `SC1_PTP_THRESHOLD` constant (now committed at the Plan 05C-03 re-calibrated value).

    Behavior:
    1. `epex_df, hydro_df = build_bowl_fixture(seed=42)`.
    2. `cal = enrich_15min_index(epex_df.index, country="CH")`.
    3. Fit BOTH `sh_off = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, cal, hydro_df)` and `sh_on = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal, hydro_df)`.
    4. Locate the test cell: prefer `("Ete", "Ouvrable")`; if absent in either fit (e.g. fixture too short), fall back to `("Hiver", "Ouvrable")` then to the first available cell common to both `sh_off.factors_` and `sh_on.factors_`. Document the fallback choice in the test docstring.
    5. Compute `ptp_off = float(np.ptp(sh_off.factors_[key]))` and `ptp_on = float(np.ptp(sh_on.factors_[key]))`.
    6. Assert `ptp_on > ptp_off * SC1_PTP_THRESHOLD` with a diagnostic message including `ptp_off`, `ptp_on`, the ratio, the threshold, and an instruction to re-run Wave 0 calibration if the threshold is stale.

    Test 6 — `test_seasonal_solar_winter_evening_delta` (D-A4-7 / SC #2 on synth):

    Docstring: cite D-A4-7 + SC #2 + RESEARCH §SC #2 delta analytique vérifié (expected delta ~11.5 EUR/MWh, well above 5.0). Document the fixture-real gap per RESEARCH Pitfall 5: "Pass = math correcte. Phase 10 valide sur HFC OMPEX réel (condition suffisante). Failure ici = math broken (ship-blocker immédiat). Pass ici + failure Phase 10 = fixture-real gap (informe future fixture design, PAS un rollback 5bis-B)."

    Behavior:
    1. `epex_df, hydro_df = build_bowl_fixture(seed=42)`.
    2. Build full PFC pipeline with flag=ON: fit `sh = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal_3yr, hydro_df)` and `si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=cal_3yr)`. Construct `PFCAssembler(shape_hourly=sh, shape_intraday=si, ...)` (all other components None). Build a PFC sufficiently far ahead so the seasonal signature is fully expressed: `start_date="2027-01-01"`, `horizon_days=365`, `reference_date=pd.Timestamp("2026-01-01", tz="UTC")`.
    3. Join the resulting `df_pfc` with its calendar enrichment to identify (saison, type_jour, heure_hce) per timestamp.
    4. Filter to `(type_jour == "Dimanche") & (saison == "Ete") & (heure_hce >= 10) & (heure_hce < 15)`. Compute `mean_ete = df_pfc.loc[mask_ete, "price_shape"].mean()`.
    5. Filter to `(type_jour == "Dimanche") & (saison == "Hiver") & (heure_hce >= 10) & (heure_hce < 15)`. Compute `mean_hiver = df_pfc.loc[mask_hiver, "price_shape"].mean()`.
    6. Assert `abs(mean_ete - mean_hiver) > 5.0` with a diagnostic message including both means, the delta, the cell counts (`mask_ete.sum()`, `mask_hiver.sum()`), and the RESEARCH-expected delta ~11.5 EUR/MWh for sanity.

    Edge case: if either mask has zero rows (calendar gap), skip the test with `pytest.skip("Insufficient calendar coverage in fixture")` and log a warning. This should NOT happen with the year-long horizon, but is a defensive guard.

    Test 7 — `test_flag_on_bowl_baseline` (D-A4-9 / new convention):

    Docstring: cite D-A4-9 + RESEARCH §Validation Architecture row 7 + 5bis-A REVIEWS.md §1 tolerance contract. Document the new baseline convention: "Convention établie par Plan 05C-03: chaque flag transition / math change atomique = nouvelle baseline frozen séparée. Pattern: `baseline_pfc_seed42_{feature_name}.parquet`. 5bis-B feature_name = `bowl` (duck-curve deepening). Phase 5 / 5ter / phases shape futures suivront ce pattern."

    Behavior (mirrors `test_flag_off_bit_for_bit_baseline` from Plan 05C-01 but with flag=True):
    1. `df_on = build_baseline_pfc(seed=42, flag=True)` (the 5bis-A reusable entry point).
    2. `baseline_bowl = pd.read_parquet("tests/fixtures/baseline_pfc_seed42_bowl.parquet")`.
    3. Strict column/dtype/index identity:
       - `assert list(df_on.columns) == list(baseline_bowl.columns)`
       - `assert df_on.dtypes.to_dict() == baseline_bowl.dtypes.to_dict()`
       - `assert df_on.index.equals(baseline_bowl.index)`
    4. Numerical equality at the tightest contract:
       - `assert_frame_equal(df_on, baseline_bowl, check_exact=False, atol=1e-12, rtol=0)`
    5. CI-drift fallback policy: same as `test_flag_off_bit_for_bit_baseline` — start at `atol=1e-12, rtol=0`. Fallback to `atol=1e-10` only with documented inline `# CI-drift fallback: ...` annotation per 5bis-A REVIEWS contract.

    All three tests must reference their decision IDs in the docstring (D-A4-5, D-A4-7, D-A4-9), use the autouse env-var hygiene from `tests/conftest.py`, and be ≤ 80 lines each. Reuse the module-scope fixture from Plan 05C-02 (`_bowl_pfc_setup` or equivalent) if it covers their needs; otherwise add the necessary setup inline.
  </action>
  <verify>
    <automated>pytest tests/test_shape_hourly_bowl.py -v 2>&1 | tail -15 && pytest tests/ -x -q 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_shape_hourly_bowl.py::test_factors_ptp_deepens_under_flag -x` exits 0.
    - `pytest tests/test_shape_hourly_bowl.py::test_seasonal_solar_winter_evening_delta -x` exits 0 (OR skips with the calendar-coverage guard — explicit skip is acceptable but failure is not).
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_on_bowl_baseline -x` exits 0.
    - `pytest tests/test_shape_hourly_bowl.py -v 2>&1 | grep -c "PASSED\|SKIPPED"` reports ≥ 7 (4 existing + 3 new).
    - `pytest tests/ -x -q` exits 0 reporting `254 passed, 4 skipped` (251 baseline + 3 new). Tolerance: if `test_seasonal_solar_winter_evening_delta` legitimately skips on the synthetic fixture, the count is `253 passed, 5 skipped` — document the actual count in SUMMARY.
    - `grep -q "D-A4-5\|D-A4-7\|D-A4-9" tests/test_shape_hourly_bowl.py` exits 0 (all three decision IDs cited).
    - `grep -q "baseline_pfc_seed42_bowl" tests/test_shape_hourly_bowl.py` exits 0.
    - `grep -q "fixture-real gap" tests/test_shape_hourly_bowl.py` exits 0 (RESEARCH Pitfall 5 docstring).
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` exits 0 (5bis-A no-op contract still preserved end-to-end).
  </acceptance_criteria>
  <done>Seven tests passing in `tests/test_shape_hourly_bowl.py` covering all 5 ROADMAP Success Criteria + the new baseline convention. Suite at 254 passed, 4 skipped.</done>
</task>

<task type="auto">
  <name>Task 5: Record flag flip strategy in `.planning/PROJECT.md` (D-FLIP-1) and add SUPERSEDED note to 05bis pre-doc</name>
  <files>.planning/PROJECT.md, .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md</files>
  <read_first>
    - .planning/PROJECT.md (current Key Decisions table — last entry is from 2026-05-18, format reference)
    - .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md (pre-doc to mark SUPERSEDED — at minimum the first 30 lines for context preservation)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-FLIP-1 (full text of the decision to record) and `## Deferred Ideas` (the pre-doc cleanup is listed there)
  </read_first>
  <action>
    Make two surgical updates to project documentation.

    Update A — `.planning/PROJECT.md`:
    Append a new row to the `## Key Decisions` table (the table ends around line 86-87 in the current file). The new row format follows the existing convention `| Date | Décision | Rationale |`:

    `| 2026-05-19 | **Flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE livré default OFF en Phase 5bis-B (commit XYZ). Flip default ON gated par Phase 10 success (Δ MAE bloc ≤ -1.5 EUR/MWh vs HFC OMPEX 2024-2025).** (D-FLIP-1) | EPFL/SOTA principle: no production change without empirical validation gate. Pas T+30j auto post-merge. Voir .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-FLIP-1. |`

    The `commit XYZ` placeholder is replaced by the SHA of THIS plan's final commit (or left as `commit TBD` if the row is added before the commit is created — `/gsd:execute-plan` will resolve it later).

    Do NOT modify any other line in `PROJECT.md`. The Constraints section, the Out of Scope section, the Context section, etc. all stay untouched.

    Update B — `.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md`:
    Add a header note at the VERY TOP of the file (above the existing title or first heading). The note format:

    ```
    > **STATUS: SUPERSEDED.** This pre-doc was the original Phase 5bis context (gathered 2026-05-17). Post-2026-05-18 adversarial panel review (3 reviewers, verdict unanime "disagree" on monolithic 5bis), Phase 5bis was split into two atomic phases:
    > - **5bis-A** (no-op infrastructure refactor) — see `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/` (DELIVERED 2026-05-18).
    > - **5bis-B** (math change: bowl-deepening via 3 levers) — see `.planning/phases/05C-shape-hourly-bowl-deepening/` (IN PLANNING 2026-05-19).
    >
    > This pre-doc is retained for historical reference only. The current authoritative context for 5bis is:
    > - `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md` (5bis-A)
    > - `.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md` (5bis-B)
    ```

    Add a blank line before and after the note. Preserve all the rest of the file's content unchanged — the SUPERSEDED note is additive, not destructive (the historical record is valuable for the audit trail).
  </action>
  <verify>
    <automated>grep -q "D-FLIP-1" .planning/PROJECT.md && grep -q "Phase 10 success" .planning/PROJECT.md && grep -q "STATUS: SUPERSEDED" .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md && grep -q "5bis-A.*5bis-B" .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md && echo "OK doc updates"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q "D-FLIP-1" .planning/PROJECT.md` exits 0.
    - `grep -q "PFC_LT_USE_SEASONAL_HOURLY_SHAPE" .planning/PROJECT.md` exits 0 (flag name preserved in the decision text).
    - `grep -q "Phase 10 success" .planning/PROJECT.md` exits 0 (gating condition preserved).
    - `grep -q "Δ MAE bloc" .planning/PROJECT.md` exits 0 (KPI quantification preserved).
    - `grep -q "EPFL/SOTA principle" .planning/PROJECT.md` exits 0 (rationale preserved).
    - `grep -q "STATUS: SUPERSEDED" .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` exits 0.
    - `grep -q "5bis-A" .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` exits 0 (pointer to split-phase context).
    - `grep -q "5bis-B" .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` exits 0.
    - `grep -q "DELIVERED 2026-05-18" .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` exits 0.
    - The verify command above prints `OK doc updates`.
    - `git diff --stat .planning/PROJECT.md` shows the change is exactly one row addition to the Key Decisions table (no other modifications).
    - `wc -l .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` shows the original line count + ~10 lines for the SUPERSEDED note (file not destroyed).
  </acceptance_criteria>
  <done>D-FLIP-1 recorded permanently in PROJECT.md Key Decisions. 5bis pre-doc marked SUPERSEDED with pointers to the split-phase replacements. Audit trail intact.</done>
</task>

</tasks>

<verification>
- `pytest tests/ -x -q` exits 0 reporting `254 passed, 4 skipped` (tolerance: `253 passed, 5 skipped` if SC #2 fixture-coverage skip path triggers).
- `pytest tests/ --co -q | tail -1` reports `>= 261 tests collected`.
- All FIVE ROADMAP Success Criteria validated by automated tests:
  - SC #1 (ptp deepening) — `test_factors_ptp_deepens_under_flag`
  - SC #2 (€5/MWh delta) — `test_seasonal_solar_winter_evening_delta`
  - SC #3 (M+30 amplitude) — `test_f_H_amplitude_preserved_at_M30` (Plan 05C-02)
  - SC #4 (flag=OFF bit-pour-bit) — `test_flag_off_bit_for_bit_baseline` (Plan 05C-01) + `test_baseline_regression[False]` (5bis-A)
  - SC #5 (247+5bis-B tests verts) — full suite green
- `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; sh = ShapeHourly(); assert sh._sigma_off == 0.5 and sh._sigma_on == 0.25 and sh.sigma == 0.5; sh = ShapeHourly(sigma=0.5); assert sh._sigma_off == 0.5 and sh._sigma_on == 0.5; sh = ShapeHourly(use_seasonal_hourly=True); assert sh.sigma == 0.25; print('OK')"` prints `OK`.
- `test -f tests/fixtures/baseline_pfc_seed42_bowl.parquet` exits 0; file is committed to git.
- `grep -q "D-FLIP-1" .planning/PROJECT.md` exits 0.
- `grep -q "STATUS: SUPERSEDED" .planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` exits 0.
- `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline tests/test_shape_hourly_infra.py -x -q` exits 0 — the 5bis-A no-op contract is preserved end-to-end across all three levers.
</verification>

<success_criteria>
- **All three levers of Phase 5bis-B shipped:** Lever 1 (hydro kernel, Plan 05C-01), Lever 2 (split f_H, Plan 05C-02), Lever 3 (σ paramétrisation, this plan).
- **All five ROADMAP Success Criteria validated by automated tests on synthetic fixture** (SC #1, SC #2, SC #3 on synth; SC #4 + SC #5 universally).
- **New baseline convention established:** `tests/fixtures/baseline_pfc_seed42_bowl.parquet` frozen + tested at `atol=1e-12, rtol=0`. Pattern `baseline_pfc_seed42_{feature_name}.parquet` documented for Phase 5 / 5ter / future shape phases.
- **Backward-compat preserved across the 3-plan series:** 4 legacy callsites (autoresearch.py:234, rolling_update.py:365, test_shape_hourly_infra.py:239, test_shape_hourly_infra.py:250) continue to work without modification. `ShapeHourly(sigma=X)` resolves to legacy single-σ semantics; `ShapeHourly()` defaults to flag=OFF + σ=0.5; explicit `sigma_off`/`sigma_on` available for new callsites.
- **Sidecar schema final form:** 10 keys in `shape_hourly.meta.parquet` hyperparams JSON. Cross-plan compat: pre-5bis-B sidecars load via fallback path.
- **EPFL traceability live:** every `ShapeHourly()` construction logs the 7 resolved hyperparams at INFO; `assembler.build()` under flag=ON logs `max |level - 1.0|` and warns above 1e-6.
- **Flag flip strategy recorded permanently:** D-FLIP-1 in PROJECT.md Key Decisions. Default OFF until Phase 10 real-data validation gate passes.
- **Project docs cleaned up:** 5bis pre-doc marked SUPERSEDED with audit trail intact.
- **Test count: 251 → 254** (4 skipped preserved, tolerance ±1 on skip count).
- **Cross-cutting truth (final wave):** `flag=OFF baseline 5bis-A preserved at atol=1e-12 rtol=0` holds after Lever 3 ships.
</success_criteria>

<output>
Create `.planning/phases/05C-shape-hourly-bowl-deepening/05C-03-SUMMARY.md` when done.
</output>
