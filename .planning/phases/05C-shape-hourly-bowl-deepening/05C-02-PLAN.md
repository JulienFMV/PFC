---
phase: 05C-shape-hourly-bowl-deepening
plan: 02
type: execute
wave: 2
depends_on:
  - 05C-01
files_modified:
  - pfc_shaping/lt/model/shape_hourly.py
  - pfc_shaping/lt/model/assembler.py
  - tests/test_shape_hourly_bowl.py
  - scripts/calibrate_bowl_thresholds.py
  - tests/fixtures/_bowl_calibration_report.json
autonomous: true
requirements:
  - SHP-03
  - SHP-04
  - D-A2-1
  - D-A2-2
  - D-A2-3
  - D-A2-4
  - D-A2-5
  - D-A2-6
  - D-A4-4
  - D-A4-6
must_haves:
  truths:
    - "Module-level helper `_split_level_anomaly(f_H_series: pd.Series, cal_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]` exists in `pfc_shaping/lt/model/shape_hourly.py` (D-A2-1). Its body computes `level[t] = mean_h(f_H | saison(t), type_jour(t))` via groupby-transform-mean on `(saison, type_jour)` of `cal_df`, and `anomaly[t] = f_H[t] - level[t]` (D-A2-2)."
    - "`__all__` is created in `pfc_shaping/lt/model/shape_hourly.py` (RESEARCH Pitfall C — currently absent) listing `[\"ShapeHourly\", \"GAUSSIAN_SIGMA\", \"_FLAG_ENV_VAR\", \"_resolve_flag\", \"_meta_path\", \"_split_level_anomaly\"]`. The helper is publicly accessible via `from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly`."
    - "Invariants of `_split_level_anomaly` (D-A2-2): `level + anomaly` equals `f_H` at ulp exactness (`numpy.allclose(level + anomaly, f_H, atol=1e-15, rtol=0)`); per-cell mean of anomaly is zero (`abs(anomaly.groupby([saison, type_jour]).mean()).max() < 1e-12`) for every cell present in `cal_df`."
    - "`pfc_shaping/lt/model/assembler.py:333` line `f_H = 1.0 + (f_H - 1.0) * shape_freedom['f_H']` is replaced by a conditional branch gated by `self.sh._use_seasonal_hourly` (D-A2-3 / D-A2-4). Under flag=ON: `level, anomaly = _split_level_anomaly(f_H, cal); level_damped = 1.0 + (level - 1.0) * shape_freedom['f_H']; f_H = level_damped + anomaly`. Under flag=OFF: legacy `f_H = 1.0 + (f_H - 1.0) * shape_freedom['f_H']` UNCHANGED."
    - "Import `from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly` is added to `pfc_shaping/lt/model/assembler.py` at the top of the file alongside the other `pfc_shaping.lt.model` imports."
    - "Telemetry drift detection (D-A2-5): under flag=ON, after computing `level`, log `logger.info('f_H split: max |level - 1.0| = %.2e', max_level_drift)` where `max_level_drift = float(abs(level - 1.0).max())`. If `max_level_drift > 1e-6`, also log `logger.warning('f_H split: level drift %.2e > 1e-6 — SHP-03 invariant may be degraded', max_level_drift)`. The telemetry runs every `assembler.build()` call when flag=ON, never under flag=OFF."
    - "Knot schedule for `level_damped` (D-A2-6) is `shape_freedom['f_H']` already returned by `assembler._shape_freedom()` at lines 803-843 — `[(0,1.00), (6,0.98), (12,0.88), (24,0.62), (36,0.42)]`. NO changes to `_shape_freedom()`. The anomaly is damping-free (effective damping 1.0 — pass-through to far horizon)."
    - "`f_H` energy-normalization invariant (SHP-03) is preserved under flag=ON: because `_split_level_anomaly` is sum-preserving by construction (D-A2-2), `mean_h(level + anomaly) == mean_h(f_H)` and `level ~= 1.0` exactly (the fit normalization at `shape_hourly.py:281` guarantees `mean(smoothed) = 1.0` per cell), so `level_damped + anomaly ~= 1.0 + (1.0 - 1.0) * sf + anomaly = 1.0 + anomaly`. Mean-preservation holds at the telemetry-monitored 1e-6 tolerance."
    - "Two new tests added to `tests/test_shape_hourly_bowl.py`: `test_split_level_anomaly_invariant` (D-A4-4) verifies the two ulp-exact invariants directly on synthetic `f_H` arrays; `test_f_H_amplitude_preserved_at_M30` (D-A4-6 / SC #3) verifies that under flag=ON the `f_H` amplitude at horizon ~M+30 exceeds `SC3_M30_AMPLITUDE_THRESHOLD` (calibrated in this plan's Wave 0 via measure-then-assert; plancher 0.50)."
    - "M3 cross-AI review fix (05C-REVIEWS.md consensus #2): `_split_level_anomaly` has a docstring section `## Window-dependence` stating (a) level is computed over the call-window timestamps, NOT fit-stable cell anchors; (b) recommended minimum build window is one full year (≥ 52 ISO weeks × 8 cells = 416 covered); (c) `level + anomaly == f_H` is exact pre-damping only — post-damping invariance holds only when level == 1.0 (the SHP-03 contract). `PFCAssembler.build()` docstring has a complementary `Notes` block with the same three points. Both are verified by `inspect.getdoc()`-based acceptance criteria."
    - "Wave 0 task in this plan measures `np.ptp(f_H_at_M30)` under flag=ON and flag=OFF on the bowl fixture, commits the actual `SC3_M30_AMPLITUDE_THRESHOLD` (replacing the placeholder 0.50 introduced in Plan 05C-01) using the formula `threshold = max(observed_split_amplitude - 0.20, observed_legacy_amplitude * 1.50, 0.50)`. The threshold is calibrated to be (a) well above the legacy M+30 amplitude (~0.52 per RESEARCH §Lever 2 dry-run), (b) well below the observed split amplitude (~0.99 expected), (c) at least the plancher 0.50."
    - "M1 cross-AI review fix (05C-REVIEWS.md consensus #1): `tests/test_shape_hourly_bowl.py::test_split_level_anomaly_drift_warning` exists and uses pytest's `caplog` fixture. It injects 1e-4 of per-cell mean drift into a synthetic `level` series, invokes the D-A2-5 telemetry path (preferably via a small `_emit_level_drift_telemetry` helper extracted from `assembler.py::build` Edit B), and asserts EXACTLY ONE WARNING-level record is captured matching the canonical message substring `f_H split: level drift`. This closes the cross-AI reviewers' joint silence concern: silent SHP-03 invariant degradation will fail CI loudly."
    - "Cross-cutting truth (appears in all 3 plans): `flag=OFF baseline 5bis-A preserved at atol=1e-12 rtol=0`. After this plan, `test_flag_off_bit_for_bit_baseline` (Plan 05C-01) continues to pass — the assembler `if self.sh._use_seasonal_hourly:` branch executes the legacy single-line damping under flag=OFF, byte-identical math path to 5bis-A."
  artifacts:
    - path: "pfc_shaping/lt/model/shape_hourly.py"
      provides: "Module-level `_split_level_anomaly` helper + `__all__` list including the new symbol (D-A2-1, RESEARCH Pitfall C)."
      contains: "_split_level_anomaly"
    - path: "pfc_shaping/lt/model/assembler.py"
      provides: "Flag-gated branch at line ~333 selecting between legacy single-line damping (flag=OFF) and split-based level-only damping (flag=ON) (D-A2-3, D-A2-4). Telemetry drift logging (D-A2-5)."
      contains: "_split_level_anomaly"
    - path: "tests/test_shape_hourly_bowl.py"
      provides: "Three new tests: `test_split_level_anomaly_invariant` (D-A4-4), `test_f_H_amplitude_preserved_at_M30` (D-A4-6), AND `test_split_level_anomaly_drift_warning` (M1 cross-AI review fix — caplog-based D-A2-5 telemetry assertion). Updated `SC3_M30_AMPLITUDE_THRESHOLD` constant (calibrated in this plan's Wave 0 — written into `tests/fixtures/_bowl_calibration_report.json` per M2)."
      contains: "test_split_level_anomaly_drift_warning"
  key_links:
    - from: "pfc_shaping/lt/model/assembler.py:333 (post-edit, conditional branch)"
      to: "pfc_shaping/lt/model/shape_hourly.py::_split_level_anomaly"
      via: "import + function call when self.sh._use_seasonal_hourly is True"
      pattern: "_split_level_anomaly"
    - from: "pfc_shaping/lt/model/shape_hourly.py::__all__"
      to: "_split_level_anomaly"
      via: "explicit public export (Pitfall C)"
      pattern: "__all__"
    - from: "tests/test_shape_hourly_bowl.py::test_f_H_amplitude_preserved_at_M30"
      to: "pfc_shaping/lt/model/assembler.py post-split f_H series at months_ahead ~30"
      via: "Build PFC with start_date='2029-06-01', horizon_days=31, reference_date='2027-01-01' (~M+30); assert np.ptp(df['f_H']) > SC3_M30_AMPLITUDE_THRESHOLD"
      pattern: "f_H"
---

<objective>
Implement Lever 2 of Phase 5bis-B: split `f_H = level + anomaly` where `level = mean_h(f_H | saison, type_jour)` is the per-cell average and `anomaly = f_H - level` is the zero-mean residual carrying the duck-curve signature.

At `pfc_shaping/lt/model/assembler.py:333` (the current single-line damping `f_H = 1.0 + (f_H - 1.0) * shape_freedom['f_H']`), introduce a flag-gated branch:
- flag=ON: dampen ONLY the `level` component (`level_damped = 1.0 + (level - 1.0) * shape_freedom['f_H']`), let the `anomaly` survive at 100% to far horizon (`f_H = level_damped + anomaly`). This is the math change that lets the duck-curve bowl persist at M+30 / Y+2 / Y+3 (RESEARCH §Lever 2 dry-run: legacy M+30 ptp 0.516 vs split M+30 ptp 0.992, gain ratio 1.92).
- flag=OFF: legacy single-line damping UNCHANGED.

Create the module-level helper `_split_level_anomaly(f_H_series, cal_df)` in `pfc_shaping/lt/model/shape_hourly.py` exposed via a new `__all__` list (RESEARCH Pitfall C — `__all__` currently absent in the file). The helper is unit-testable independently of `assembler.build()`.

Add telemetry drift detection at every `assembler.build()` call under flag=ON: log `max |level - 1.0|` at INFO, warn if it exceeds 1e-6 (D-A2-5, future-proof against silent SHP-03 invariant degradation if Phase 5 MSFC log-prix or future fits change the normalization).

Append two new tests to `tests/test_shape_hourly_bowl.py`: `test_split_level_anomaly_invariant` (D-A4-4, direct verification of the math invariants on synthetic `f_H` arrays) and `test_f_H_amplitude_preserved_at_M30` (D-A4-6 / SC #3, the end-to-end proof that the split preserves bowl amplitude at far horizon). Wave 0 calibrates the M+30 amplitude threshold via measure-then-assert and overwrites the placeholder constant introduced in Plan 05C-01.

Purpose: This plan delivers the structural change that lets the duck-curve survive M+30 — the test 5bis-A `test_baseline_regression` cannot detect this (M+1 horizon). Lever 2 is the SC #3 driver per RESEARCH §Lever 2.

Output: `_split_level_anomaly` helper + `__all__` in `shape_hourly.py`, conditional branch + import in `assembler.py`, telemetry log lines, two new tests in `test_shape_hourly_bowl.py`. Test suite goes from 249 to 251 passing (4 skipped preserved).
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
@.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md
@pfc_shaping/lt/model/shape_hourly.py
@pfc_shaping/lt/model/assembler.py
@tests/test_shape_hourly_bowl.py
@tests/fixtures/_generate_bowl_fixture.py

<interfaces>
Key contracts to consume (Plan 05C-01 outputs):

From `pfc_shaping/lt/model/shape_hourly.py` (Plan 05C-01 final state):
- Module constants: `GAUSSIAN_SIGMA = 0.5`, `_HYDRO_WEIGHT_SIGMA_OFF_DEFAULT = 0.25`, `_HYDRO_WEIGHT_SIGMA_ON_DEFAULT = 0.08`, `_FLAG_ENV_VAR`, `_META_SIDECAR_SUFFIX`.
- NO `__all__` yet — this plan creates it per RESEARCH Pitfall C.
- `class ShapeHourly`: `__init__(sigma=GAUSSIAN_SIGMA, halflife_days=180.0, hydro_weight_sigma=None, hydro_weight_sigma_off=0.25, hydro_weight_sigma_on=0.08, use_seasonal_hourly=None)`. (Plan 05C-03 will switch `sigma` to `float | None = None`.)
- Public attributes available from `self.sh` in assembler: `self.sh._use_seasonal_hourly: bool`, `self.sh._hydro_weight_sigma_off`, `self.sh._hydro_weight_sigma_on`.

From `pfc_shaping/lt/model/assembler.py`:
- Line 333 (current): `f_H = 1.0 + (f_H - 1.0) * shape_freedom["f_H"]` — single damping line, the integration point.
- Lines 307-316: f_H sourcing via `self.sh.apply(...)` with capability-check branch (5bis-A D-13).
- Lines 803-843: `_shape_freedom()` returns dict with knot `'f_H': [(0,1.00),(6,0.98),(12,0.88),(24,0.62),(36,0.42)]` — DO NOT modify.
- The `calendar_df` available at the integration point is `cal` (line 280, built via `enrich_15min_index(idx, country=country)`), which already has `saison` and `type_jour` columns required by `_split_level_anomaly`.

From `tests/test_shape_hourly_bowl.py` (Plan 05C-01 final state):
- `SC1_PTP_THRESHOLD: float = <measured>` (set by Plan 05C-01 Wave 0)
- `SC3_M30_AMPLITUDE_THRESHOLD: float = 0.50` (PLACEHOLDER — this plan's Wave 0 replaces)
- Two existing tests: `test_hydro_kernel_uses_per_timestamp_climatological_target`, `test_flag_off_bit_for_bit_baseline`

Test count at start of this plan: `249 passed, 4 skipped`. Target after this plan: `251 passed, 4 skipped`.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create `_split_level_anomaly` helper + `__all__` in `pfc_shaping/lt/model/shape_hourly.py`</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py:28-100 (current module top — imports, constants, helpers `_resolve_flag` and `_meta_path` already at module level; pattern for the new `_split_level_anomaly`)
    - pfc_shaping/lt/model/shape_hourly.py:930-938 (current `_gaussian_smooth_circular` — module-level utility at the bottom, sister-helper location)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Lever 2 (full helper signature + body verbatim, RESEARCH Pitfall 6 fallback handling)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Open Questions item 2 (decision: add `__all__` here, no downstream `import *` consumers exist)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A2-1, D-A2-2 (helper contract and invariants)
  </read_first>
  <behavior>
    The helper must satisfy these properties on any well-formed input (`f_H_series` is a pd.Series indexed by a DatetimeIndex, `cal_df` has at least the columns `saison` and `type_jour` indexed compatibly with `f_H_series`):

    - Test 1 (ulp-exact sum): For random `f_H_series` with values ~ N(1, 0.2), `numpy.allclose(level + anomaly, f_H, atol=1e-15, rtol=0)` is True.
    - Test 2 (zero-mean per cell): For each unique `(saison, type_jour)` cell present in `cal_df`, `abs(anomaly.groupby([saison, type_jour]).mean()).max() < 1e-12`.
    - Test 3 (single-cell degenerate): When `cal_df` has only one cell (e.g. all timestamps are `("Hiver","Ouvrable")`), `level == f_H.mean()` constant array, `anomaly == f_H - f_H.mean()`.
    - Test 4 (index alignment): `level.index.equals(f_H_series.index)` and `anomaly.index.equals(f_H_series.index)` — outputs preserve the input index exactly.
    - Test 5 (missing cal robustness — RESEARCH Pitfall 6): When `cal_df` has NaN in `saison` or `type_jour` for some timestamps, the helper does NOT crash. It logs a warning `"_split_level_anomaly: %d timestamps with missing cal — using f_H directly (level=1.0)"` and for those timestamps assigns `level=1.0` and `anomaly = f_H - 1.0`. Cells with valid cal compute level/anomaly normally.

    These behaviors are NOT all tested in the production code path (Task 4 only tests Tests 1-3 directly). Tests 4 and 5 are guaranteed by construction and documented in the docstring.
  </behavior>
  <action>
    Make three edits to `pfc_shaping/lt/model/shape_hourly.py`:

    Edit A — Add `__all__` after the imports (around line 41, after `logger = logging.getLogger(__name__)`):

    Insert a comment line `# Public API — explicit __all__ added by Plan 05C-02 (RESEARCH Pitfall C). Pre-5bis-B this module had no __all__; verified no downstream "import *" consumers exist.` followed by `__all__ = ["ShapeHourly", "GAUSSIAN_SIGMA", "_FLAG_ENV_VAR", "_resolve_flag", "_meta_path", "_split_level_anomaly"]`.

    The `_HYDRO_WEIGHT_SIGMA_*_DEFAULT` constants are private and intentionally NOT exported. Plan 05C-03 may extend `__all__` with `_split_level_anomaly` already present, so future additions append without churn.

    Edit B — Add `_split_level_anomaly` helper at module level, immediately AFTER `_gaussian_smooth_circular` (i.e. after the existing bottom utility at line 938):

    The helper is a pure function (no class state). Signature `def _split_level_anomaly(f_H_series: pd.Series, cal_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:`.

    Body specification (the executor writes the implementation following these constraints — NOT inlined code):
    1. Module docstring with sections: Purpose (one line), Math (the two equations from D-A2-2: `level[t] = mean_h(f_H | saison(t), type_jour(t))`, `anomaly[t] = f_H[t] - level[t]`), Invariants (ulp-exact sum, zero-mean per cell), Args (types + index expectations), Returns (`(level, anomaly)` both pd.Series with same index as `f_H_series` and names `"level"` / `"anomaly"`), AND a MANDATORY "## Window-dependence" section (M3 cross-AI review fix — `05C-REVIEWS.md` consensus #2 / Codex MEDIUM concern). The Window-dependence section MUST state verbatim three points: (a) "`level` is computed via `groupby().transform("mean")` over the timestamps present in the CURRENT call window — it is NOT a fit-stable cell anchor; the decomposition depends on the build horizon length and composition." (b) "For stable decomposition the recommended MINIMUM build window is one full year (or equivalently 52 ISO weeks × all 8 (saison, type_jour) cells = 416 cells covered)." (c) "The invariant `level + anomaly == f_H` is EXACT pre-damping only; post-damping invariance (`level_damped + anomaly == f_H` modulo level shrinkage) holds ONLY when `level == 1.0` per cell, which is the SHP-03 contract. The D-A2-5 telemetry (`max |level - 1.0| > 1e-6` → `logger.warning`) exists precisely to detect violations of this contract; the `test_split_level_anomaly_drift_warning` test (Task 5) asserts the telemetry fires under drift."
    2. Build `df = pd.DataFrame({"f_H": f_H_series}).join(cal_df[["saison", "type_jour"]])`. The `join` aligns on index; mismatched indices yield NaN in `saison`/`type_jour` for those timestamps.
    3. NaN-cell fallback (RESEARCH Pitfall 6): detect rows where `saison` or `type_jour` is NaN. If any, log `logger.warning("_split_level_anomaly: %d timestamps with missing cal — using f_H directly (level=1.0)", n_missing)`. For those timestamps, the fallback sets `level=1.0` (no normalization), `anomaly = f_H - 1.0`.
    4. For non-NaN rows: `cell_means = df.groupby(["saison", "type_jour"])["f_H"].transform("mean")`. This produces a Series aligned with `df.index` where each row holds its cell's mean.
    5. Construct outputs:
       - `level = cell_means.where(non_nan_mask, 1.0).rename("level")` (cells with NaN cal get 1.0)
       - `anomaly = (f_H_series - level).rename("anomaly")` (ulp-exact subtraction; sum invariant trivially holds)
    6. Return `(level, anomaly)`.

    Numerical exactness: the subtraction `anomaly = f_H_series - level` is IEEE-754 exact when `level` is finite (no over/underflow with values near 1.0), giving ulp-exact `level + anomaly == f_H` regardless of float roundoff in the mean. This invariant is what makes the helper "sum-preserving by construction" per D-A2-2.

    Edit C — Verify backward compat: the new symbol `_split_level_anomaly` is not yet consumed anywhere (Task 2 wires it in `assembler.py`). After Edit B, `python -c "from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly; print(_split_level_anomaly.__doc__)"` must print a non-empty docstring.
  </action>
  <verify>
    <automated>python -c "import numpy as np; import pandas as pd; from pfc_shaping.lt.model import shape_hourly; assert hasattr(shape_hourly, '__all__'); assert '_split_level_anomaly' in shape_hourly.__all__; assert 'ShapeHourly' in shape_hourly.__all__; from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly; idx = pd.date_range('2027-01-01', periods=96, freq='15min', tz='UTC'); f_H = pd.Series(np.random.default_rng(0).normal(1.0, 0.1, 96), index=idx, name='f_H'); cal = pd.DataFrame({'saison': ['Hiver']*48 + ['Ete']*48, 'type_jour': ['Ouvrable']*96}, index=idx); level, anomaly = _split_level_anomaly(f_H, cal); assert np.allclose(level + anomaly, f_H, atol=1e-15, rtol=0); ca = anomaly.groupby([cal['saison'], cal['type_jour']]).mean(); assert abs(ca).max() < 1e-12, f'max={abs(ca).max()}'; assert level.index.equals(f_H.index); assert anomaly.index.equals(f_H.index); assert level.name == 'level' and anomaly.name == 'anomaly'; print('OK helper invariants verified')" && pytest tests/ -x -q 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "^__all__" pfc_shaping/lt/model/shape_hourly.py` matches a single module-level definition line.
    - `grep -q "_split_level_anomaly" pfc_shaping/lt/model/shape_hourly.py` finds at least 2 hits (one in `__all__`, one in the def).
    - `grep -q "def _split_level_anomaly" pfc_shaping/lt/model/shape_hourly.py` exits 0.
    - The python invariant-verification command above prints `OK helper invariants verified`.
    - `pytest tests/ -x -q` exits 0 reporting `249 passed, 4 skipped` (no behavior change to consumers yet — Task 2 wires the helper into `assembler.py`).
    - `grep -q "Pitfall C" pfc_shaping/lt/model/shape_hourly.py` exits 0 (traceability comment near `__all__`).
    - M3 docstring fix (cross-AI review consensus #2): `python -c "import inspect; from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly; d = inspect.getdoc(_split_level_anomaly) or ''; assert 'Window-dependence' in d, 'missing section'; assert 'level + anomaly == f_H' in d, 'missing invariant statement'; assert '52 ISO weeks' in d or 'one full year' in d, 'missing min-window guidance'; print('OK M3 docstring')"` exits 0.
  </acceptance_criteria>
  <done>`_split_level_anomaly` helper exists at module level, exposed via `__all__`, passes the two D-A2-2 invariants on the python-line sanity check, AND its docstring contains the mandated M3 "Window-dependence" section. No downstream consumer yet.</done>
</task>

<task type="auto">
  <name>Task 2: Wire `_split_level_anomaly` into `assembler.build()` at line 333 with flag-gated branch + telemetry (D-A2-3..D-A2-5)</name>
  <files>pfc_shaping/lt/model/assembler.py</files>
  <read_first>
    - pfc_shaping/lt/model/assembler.py:1-40 (existing imports — find the section to add `from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly`)
    - pfc_shaping/lt/model/assembler.py:275-345 (the `build()` method around the f_H integration point — line 280 builds `cal`, line 307-316 sources `f_H`, line 333 dampens it)
    - pfc_shaping/lt/model/assembler.py:803-843 (`_shape_freedom()` — DO NOT modify; the knot table is preserved for level damping per D-A2-6)
    - pfc_shaping/lt/model/shape_hourly.py (Task 1 final state — confirms `_split_level_anomaly` is importable)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A2-3, D-A2-4, D-A2-5 (integration contract)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Lever 2 "Integration surface dans assembler.py" (verbatim integration code block + telemetry block)
  </read_first>
  <action>
    Make two surgical edits to `pfc_shaping/lt/model/assembler.py`:

    Edit A — Add import at the top of the file, grouped with other `pfc_shaping.lt.model` imports:
    Add `from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly` to the import block at the top of `assembler.py` (current imports include `from pfc_shaping.lt.model.shape_hourly import ShapeHourly` or similar — co-locate). If `ShapeHourly` is imported via `from . import shape_hourly` pattern, prefer the explicit symbol import for clarity.

    Edit B — Replace the single-line damping at line 333 with a flag-gated branch.

    Current line 333 (verbatim): `f_H = 1.0 + (f_H - 1.0) * shape_freedom["f_H"]`.

    Replace with a conditional block. Pseudocode (the executor writes the actual code following this contract):
    - Comment: `# Lever 2 (Plan 05C-02, D-A2-3..D-A2-5): split-based damping under flag=ON, legacy single-line under flag=OFF.`
    - `if self.sh._use_seasonal_hourly:` block:
      - `level, anomaly = _split_level_anomaly(f_H, cal)`  (`cal` is the calendar_df already built at line 280)
      - Telemetry: compute `max_level_drift = float(abs(level - 1.0).max())`; `logger.info("f_H split: max |level - 1.0| = %.2e", max_level_drift)`; if `max_level_drift > 1e-6`: `logger.warning("f_H split: level drift %.2e > 1e-6 — SHP-03 invariant may be degraded", max_level_drift)`.
      - `level_damped = 1.0 + (level - 1.0) * shape_freedom["f_H"]`
      - `f_H = level_damped + anomaly`
    - `else:` block (LEGACY — preserve byte-identical to 5bis-A):
      - `f_H = 1.0 + (f_H - 1.0) * shape_freedom["f_H"]`

    Verify the `f_H` Series produced is identical in type/index to before the edit (pd.Series indexed by `idx`, dtype float, name="f_H" if it had a name). The legacy line preserved the name; the split path must preserve it via `level + anomaly = (named "level") + (named "anomaly")` which yields a Series with `name=None` — to preserve telemetry-friendliness, add `f_H = f_H.rename("f_H")` at the end of the if-block. (NOT needed in the else-block; the existing line 349 `df["f_H"] = f_H` consumes regardless of name.)

    DO NOT modify `_shape_freedom()` at lines 803-843. The knot table for `f_H` stays at `[(0,1.00),(6,0.98),(12,0.88),(24,0.62),(36,0.42)]` per D-A2-6.

    DO NOT modify any other line in `build()`. The lines 331 (f_S damping), 332 (f_W damping), 334 (f_Q damping), 335 (f_WV damping) stay untouched — only the f_H damping at line 333 changes.

    Verify the import does NOT create a circular dependency: `python -c "import pfc_shaping.lt.model.assembler"` must succeed.

    **Edit C — Extend `PFCAssembler.build()` docstring with a "Notes" block (M3 cross-AI review fix — `05C-REVIEWS.md` consensus #2 / Codex MEDIUM concern on window-dependence; complement to the helper-side docstring extension in Task 1).**
    Locate the existing docstring of `PFCAssembler.build` (or `build()`; method name in current code base). Append a "Notes" section (preserve the existing parameter / returns sections). The Notes section MUST contain three points: (a) "When `self.sh._use_seasonal_hourly is True`, the f_H damping uses `_split_level_anomaly(f_H, cal)` (Plan 05C-02 / D-A2-3..D-A2-5). The `level` component is computed via `groupby().transform("mean")` over the timestamps in the CURRENT build window — it is window-dependent, NOT a fit-stable cell anchor (M3 cross-AI review documentation)." (b) "For stable bowl shape across calls, the recommended MINIMUM build horizon is one full year (≥ 52 ISO weeks × all 8 (saison, type_jour) cells = 416 cells covered). Shorter horizons may exhibit small level discontinuities between consecutive `build()` calls with different windows." (c) "The `max |level - 1.0|` telemetry (logged at INFO every flag=ON build, warning if > 1e-6) is the runtime detection of SHP-03 invariant degradation. See `tests/test_shape_hourly_bowl.py::test_split_level_anomaly_drift_warning` (Plan 05C-02 Task 5) for the CI signal that the warning actually fires."

    Do NOT modify the behavior of `build()` — Edit C is documentation-only.
  </action>
  <verify>
    <automated>python -c "from pfc_shaping.lt.model.assembler import PFCAssembler; from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly; print('OK imports')" && pytest tests/ -x -q 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q "_split_level_anomaly" pfc_shaping/lt/model/assembler.py` exits 0 (import + usage present).
    - `grep -c "level_damped" pfc_shaping/lt/model/assembler.py` reports ≥ 1.
    - `grep -q "max |level - 1.0|" pfc_shaping/lt/model/assembler.py` exits 0 (telemetry log message D-A2-5).
    - `grep -q "self.sh._use_seasonal_hourly" pfc_shaping/lt/model/assembler.py` exits 0 (flag-gated branch).
    - The python import-sanity command prints `OK imports`.
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` exits 0 — this is the no-op proof: flag=OFF preserves 5bis-A baseline bit-pour-bit (the else-branch runs the legacy line UNCHANGED).
    - `pytest tests/test_shape_hourly_infra.py::TestBaselineRegression -x` or equivalent class containing parametrized `test_baseline_regression[False]` exits 0 (5bis-A regression preserved).
    - `pytest tests/ -x -q` exits 0 reporting `249 passed, 4 skipped` (no new test count yet — Task 4 adds the 2 new tests).
    - M3 docstring fix (cross-AI review consensus #2): `python -c "import inspect; from pfc_shaping.lt.model.assembler import PFCAssembler; d = inspect.getdoc(PFCAssembler.build) or ''; assert 'Notes' in d, 'missing Notes section'; assert 'window-dependent' in d.lower() or 'window-dependence' in d.lower(), 'missing window-dependence note'; assert '52 ISO weeks' in d or 'one full year' in d, 'missing min-horizon guidance'; print('OK M3 build docstring')"` exits 0.
  </acceptance_criteria>
  <done>Assembler integration complete. Flag=OFF still produces byte-identical math; flag=ON now executes the split-based damping with telemetry. `PFCAssembler.build()` docstring carries the M3 Notes block on window-dependence. Full suite green.</done>
</task>

<task type="auto">
  <name>Task 3 (Wave 0): Calibrate SC3_M30_AMPLITUDE_THRESHOLD on bowl fixture, commit measured value</name>
  <files>tests/fixtures/_bowl_calibration_report.json, scripts/calibrate_bowl_thresholds.py</files>
  <read_first>
    - tests/test_shape_hourly_bowl.py (Plans 05C-01 + 05C-02 Task 1-2 state — `SC3_M30_AMPLITUDE_THRESHOLD` is loaded from the JSON sidecar via `json.load`, NOT a free-floating constant per M2 cross-AI review fix)
    - tests/fixtures/_bowl_calibration_report.json (Plan 05C-01 Task 4 output — schema includes `thresholds_emitted.SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER = 0.50`; this task overwrites it with the calibrated value)
    - scripts/calibrate_bowl_thresholds.py (Plan 05C-01 Task 4 output — this task EXTENDS the script with the SC #3 calibration, then re-runs it to refresh the JSON)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-REVIEWS.md (M2 consensus #3 — Codex framing: all thresholds flow through the immutable JSON artifact, not in-comment values)
    - tests/fixtures/_generate_bowl_fixture.py (reusable `build_bowl_fixture()`)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Lever 2 (dry-run expected: legacy 0.516, split 0.992)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §M+30 amplitude threshold SC #3 (D-A4-6) — formula
  </read_first>
  <action>
    EXTEND `scripts/calibrate_bowl_thresholds.py` (Plan 05C-01 Task 4 committed script) with the SC #3 M+30 amplitude calibration, then RE-RUN it to refresh `tests/fixtures/_bowl_calibration_report.json` with the actual `SC3_M30_AMPLITUDE_THRESHOLD` (replacing the `SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER = 0.50` placeholder). This implements M2 cross-AI review fix (`05C-REVIEWS.md` consensus #3) for the SC #3 threshold path — auditability through the same committed-script + immutable-JSON-sidecar pattern Plan 05C-01 established.

    **Part A — Extend the script with `_calibrate_sc3_m30`:**

    Modify `scripts/calibrate_bowl_thresholds.py`:
    1. Import `ShapeIntraday` from its canonical module path (mirror `tests/fixtures/_generate_baseline.py:128-133` pattern). Also import `PFCAssembler` from `pfc_shaping.lt.model.assembler`.
    2. Add a new helper `_calibrate_sc3_m30(epex_df, hydro_df, cal) -> tuple[float, float, float]`:
       - Fit `sh_off = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, cal, hydro_df)` and `sh_on = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal, hydro_df)`.
       - Fit a minimal `si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=cal)`.
       - Build two PFCs at horizon ~M+30: `start_date='2029-06-01'`, `horizon_days=31`, `reference_date=pd.Timestamp('2027-01-01', tz='UTC')`. Use `PFCAssembler(shape_hourly=sh_X, shape_intraday=si, ...)` with other components `None`.
       - Compute `ptp_off = float(np.ptp(df_off['f_H']))`, `ptp_on = float(np.ptp(df_on['f_H']))`.
       - Compute `threshold = max(ptp_on - 0.20, ptp_off * 1.50, 0.50)`.
       - Return `(ptp_off, ptp_on, threshold)`.
    3. In `main()`, AFTER the SC #1 calibration block and BEFORE writing the report dict, call `(sc3_ptp_off, sc3_ptp_on, sc3_threshold) = _calibrate_sc3_m30(epex_df, hydro_df, cal)`.
    4. EXTEND the `report["ratios"]` sub-dict with three new keys: `"sc3_ptp_off_m30": sc3_ptp_off`, `"sc3_ptp_on_m30": sc3_ptp_on`, `"sc3_amplitude_formula": "max(ptp_on - 0.20, ptp_off * 1.50, 0.50)"`.
    5. EXTEND the `report["thresholds_emitted"]` sub-dict: REMOVE the placeholder key `"SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER"` and ADD the calibrated key `"SC3_M30_AMPLITUDE_THRESHOLD": sc3_threshold`. The downstream test in `tests/test_shape_hourly_bowl.py` (Plan 05C-01 Task 5) reads this key via `json.load(...)["thresholds_emitted"]["SC3_M30_AMPLITUDE_THRESHOLD"]` — update the `tests/test_shape_hourly_bowl.py` module-level `SC3_M30_AMPLITUDE_THRESHOLD` constant assignment to use the new (non-placeholder) key.
    6. Update the script's `report["notes"]` field to: `"Plan 05C-02 extended this report with SC #3 M+30 amplitude calibration. Plan 05C-03 will re-run this script with all 3 levers active (Task 3 of 05C-03); the updated artifact MUST overwrite this one and be re-committed."`.

    **Part B — Run the updated script and commit the refreshed JSON:**

    Execute `python scripts/calibrate_bowl_thresholds.py`. Verify the new `tests/fixtures/_bowl_calibration_report.json` contains `thresholds_emitted.SC3_M30_AMPLITUDE_THRESHOLD` (NOT `..._PLACEHOLDER`) and that the value is in plausible bounds.

    Sanity bounds (if outside, STOP and request investigation):
    - If `sc3_ptp_on < 0.70`: WARN — Lever 2 is failing to preserve amplitude at M+30. Possible bug in `_split_level_anomaly` or `assembler` integration. STOP the plan.
    - If `sc3_ptp_on > 1.10`: WARN — anomaly is producing un-bounded growth. Possible level computation error. STOP.
    - If `sc3_ptp_off > 0.70`: WARN — legacy M+30 amplitude is much higher than RESEARCH analytic estimate 0.52. Could indicate the bowl fixture is too aggressive; verify against fixture sanity. Continue but document in SUMMARY.

    **Part C — Update `tests/test_shape_hourly_bowl.py` (one-line edit):**

    The placeholder load `SC3_M30_AMPLITUDE_THRESHOLD: float = _calibration_report["thresholds_emitted"]["SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER"]` set by Plan 05C-01 Task 5 must be updated to read the new (non-placeholder) key: `SC3_M30_AMPLITUDE_THRESHOLD: float = _calibration_report["thresholds_emitted"]["SC3_M30_AMPLITUDE_THRESHOLD"]`. Update the inline comment to reference Plan 05C-02 Task 3 as the source of the actual calibrated value.

    Commit the JSON (refreshed), the script (extended with the SC #3 helper), and the one-line test edit. DO NOT touch the test bodies in this task — Task 4 adds the consumer test `test_f_H_amplitude_preserved_at_M30`.
  </action>
  <verify>
    <automated>python scripts/calibrate_bowl_thresholds.py &amp;&amp; python -c "
import json
r = json.load(open('tests/fixtures/_bowl_calibration_report.json'))
assert 'SC3_M30_AMPLITUDE_THRESHOLD' in r['thresholds_emitted'], f'SC3 key missing, got: {list(r["thresholds_emitted"])}'
assert 'SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER' not in r['thresholds_emitted'], 'placeholder still present — task did not overwrite'
val = r['thresholds_emitted']['SC3_M30_AMPLITUDE_THRESHOLD']
assert 0.50 &lt;= val &lt;= 1.00, f'threshold {val} out of plausible bounds [0.50, 1.00]'
assert 'sc3_ptp_on_m30' in r['ratios'] and 'sc3_ptp_off_m30' in r['ratios'], 'SC3 ratios missing'
print(f'OK SC3 threshold={val:.4f} ptp_off={r["ratios"]["sc3_ptp_off_m30"]:.4f} ptp_on={r["ratios"]["sc3_ptp_on_m30"]:.4f}')
" &amp;&amp; python -c "import json, re; src = open('tests/test_shape_hourly_bowl.py').read(); assert 'SC3_M30_AMPLITUDE_THRESHOLD\"\]' in src or "SC3_M30_AMPLITUDE_THRESHOLD']" in src, 'test file does not load SC3 key from JSON'; assert 'SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER' not in src, 'test still loads placeholder key'; print('OK test wired to new key')"</automated>
  </verify>
  <acceptance_criteria>
    - `python -c "import json; r = json.load(open('tests/fixtures/_bowl_calibration_report.json')); assert 'SC3_M30_AMPLITUDE_THRESHOLD' in r['thresholds_emitted']; assert 'SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER' not in r['thresholds_emitted']"` exits 0.
    - `python -c "import json; r = json.load(open('tests/fixtures/_bowl_calibration_report.json')); v = r['thresholds_emitted']['SC3_M30_AMPLITUDE_THRESHOLD']; assert 0.50 &lt;= v &lt;= 1.00, v"` exits 0.
    - `grep -q "_calibrate_sc3_m30\|SC3" scripts/calibrate_bowl_thresholds.py` exits 0 (extended script committed).
    - `grep -q "SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER" tests/test_shape_hourly_bowl.py` exits 1 (test file no longer references the placeholder key after Part C edit).
    - `grep -q "SC3_M30_AMPLITUDE_THRESHOLD" tests/test_shape_hourly_bowl.py` exits 0 (test file still references the calibrated key via JSON load).
    - `grep -q "Plan 05C-02" tests/test_shape_hourly_bowl.py` exits 0 (Part C traceability comment present).
    - The verify command above prints `OK SC3 threshold=...` and `OK test wired to new key`, both exit 0.
    - `pytest tests/ -x -q` exits 0 reporting `249 passed, 4 skipped` (no test references the threshold yet — Task 4 adds the consumer).
  </acceptance_criteria>
  <done>SC #3 threshold flows through the M2 immutable JSON artifact (not an in-file constant). Calibration script extended with `_calibrate_sc3_m30` helper. Test file's `SC3_M30_AMPLITUDE_THRESHOLD` constant now points at the calibrated value via `json.load`. Audit trail: re-running the script regenerates the JSON; the matching `test_calibration_report_matches_fixture` in Plan 05C-03 enforces fixture_sha256 immutability.</done>
</task>

<task type="auto">
  <name>Task 4: Append `test_split_level_anomaly_invariant` (D-A4-4) and `test_f_H_amplitude_preserved_at_M30` (D-A4-6) to `tests/test_shape_hourly_bowl.py`</name>
  <files>tests/test_shape_hourly_bowl.py</files>
  <read_first>
    - tests/test_shape_hourly_bowl.py (Plan 05C-01 + Task 3 state — calibrated thresholds + 2 existing tests + imports)
    - pfc_shaping/lt/model/shape_hourly.py (Task 1 state — `_split_level_anomaly` available)
    - pfc_shaping/lt/model/assembler.py (Task 2 state — flag-gated branch + telemetry)
    - tests/fixtures/_generate_bowl_fixture.py (`build_bowl_fixture()` reusable)
    - tests/fixtures/_generate_baseline.py (`build_pfc(seed, flag)` — useful pattern for the M+30 test)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A4-4, D-A4-6 (test specs)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Validation Architecture (test row 2 for D-A2-2, test row 4 for SC #3)
  </read_first>
  <action>
    Append two tests to `tests/test_shape_hourly_bowl.py`. Do NOT modify the existing tests, fixture, threshold constants, or imports unless a new import is strictly required.

    Test 3 — `test_split_level_anomaly_invariant` (D-A4-4):

    Docstring: cite D-A4-4 + D-A2-2. Test scope: direct verification of the two helper invariants on a synthetic `f_H` array, independent of `assembler.build()`.

    Behavior:
    1. Construct a synthetic `f_H_series` of length 96 (one day, 15-min freq, tz="UTC", starting 2027-01-01) with values drawn from a fixed-seed RNG (e.g. `np.random.default_rng(123).normal(1.0, 0.15, 96)`).
    2. Construct a `cal_df` with at least two cells: e.g. first 48 timestamps `("Hiver","Ouvrable")`, next 48 `("Ete","Samedi")`. Use a DataFrame indexed by the same DatetimeIndex.
    3. Call `level, anomaly = _split_level_anomaly(f_H_series, cal_df)`.
    4. Assert ulp-exact sum: `numpy.testing.assert_allclose(level.values + anomaly.values, f_H_series.values, atol=1e-15, rtol=0)`.
    5. Assert zero-mean per cell: build `cell_keys = list(zip(cal_df['saison'], cal_df['type_jour']))`; `cell_anom_means = anomaly.groupby(cell_keys).mean()`; assert `float(abs(cell_anom_means).max()) < 1e-12`.
    6. Assert index alignment: `assert level.index.equals(f_H_series.index)`, `assert anomaly.index.equals(f_H_series.index)`.
    7. Assert names: `assert level.name == "level"`, `assert anomaly.name == "anomaly"`.

    Imports needed: `from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly` (add to existing imports if not present).

    Test 4 — `test_f_H_amplitude_preserved_at_M30` (D-A4-6 / SC #3):

    Docstring: cite D-A4-6 + SC #3 + RESEARCH §Lever 2 dry-run (expected ptp_split ~0.99 vs ptp_legacy ~0.52). Reference the Wave-0-calibrated threshold via the module constant.

    Behavior:
    1. Use `build_bowl_fixture(seed=42)` to obtain `epex_df, hydro_df`.
    2. Build `cal_3yr = enrich_15min_index(epex_df.index, country="CH")`.
    3. Fit `sh_on = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal_3yr, hydro_df)`.
    4. Fit a minimal ShapeIntraday on the same `epex_df` (mirror `_generate_baseline.py:128-133` pattern — `ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=cal_3yr)`).
    5. Construct a PFCAssembler with `shape_hourly=sh_on`, `shape_intraday=si`, all other components None.
    6. Build the PFC at far horizon: `df_pfc = assembler.build(base_prices={"2029": 80.0}, start_date="2029-06-01", horizon_days=31, reference_date=pd.Timestamp("2027-01-01", tz="UTC"), country="CH")`. This gives `months_ahead ~= 29-30` across the window.
    7. Assert `float(np.ptp(df_pfc["f_H"])) > SC3_M30_AMPLITUDE_THRESHOLD`.
    8. Diagnostic info on failure (include in the assertion message): the observed ptp value, the threshold, and the suggestion to re-run Wave 0 calibration if the threshold is stale.

    The test uses `pytest.fixture(scope="module")` for the expensive ShapeHourly+ShapeIntraday fit to share across this test (and future tests in Plan 05C-03 that may need the same setup). If a fixture already exists from Plan 05C-01, extend it; otherwise create one named `_bowl_pfc_setup` returning the assembler + cal_3yr.

    Both tests must be ≤ 80 lines, use the autouse env-var hygiene fixture from `tests/conftest.py` (inherited automatically), and reference their decision IDs in the docstring.

    Telemetry verification: under flag=ON the assembler emits `logger.info("f_H split: max |level - 1.0| = ...")`. The M+30 test does NOT need to verify telemetry directly (it's already validated by the math invariant test that `level - 1.0` is bounded by `f_H.std()` per cell, which for typical fits is well under 1e-6 after re-normalization at `shape_hourly.py:281`). If implementers want a belt-and-suspenders check, an OPTIONAL `caplog`-based assertion `assert any("f_H split:" in r.message for r in caplog.records)` may be added, but it is not in the must-have set.
  </action>
  <verify>
    <automated>pytest tests/test_shape_hourly_bowl.py -v 2>&1 | tail -15 && pytest tests/ -x -q 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_shape_hourly_bowl.py::test_split_level_anomaly_invariant -x` exits 0.
    - `pytest tests/test_shape_hourly_bowl.py::test_f_H_amplitude_preserved_at_M30 -x` exits 0.
    - `pytest tests/test_shape_hourly_bowl.py -v 2>&1 | grep -c PASSED` reports ≥ 4 (2 existing + 2 new).
    - `pytest tests/ -x -q` exits 0 reporting `251 passed, 4 skipped` (249 baseline + 2 new). If parametrization differs slightly, document the actual count in SUMMARY.
    - `grep -q "test_split_level_anomaly_invariant" tests/test_shape_hourly_bowl.py` exits 0.
    - `grep -q "test_f_H_amplitude_preserved_at_M30" tests/test_shape_hourly_bowl.py` exits 0.
    - `grep -q "D-A4-4\|D-A4-6" tests/test_shape_hourly_bowl.py` exits 0 (both decision IDs cited in docstrings).
    - `grep -q "SC3_M30_AMPLITUDE_THRESHOLD" tests/test_shape_hourly_bowl.py` reports the constant is REFERENCED in test body (not just defined).
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` exits 0 (5bis-A baseline still preserved after Lever 2 ships).
  </acceptance_criteria>
  <done>Four tests passing in `tests/test_shape_hourly_bowl.py` (D-A4-3, D-A4-4, D-A4-6, D-A4-8). Suite at 251 passed, 4 skipped. Lever 2 math change validated end-to-end on synthetic fixture.</done>
</task>

<task type="auto">
  <name>Task 5 (M1 cross-AI review fix): Add `test_split_level_anomaly_drift_warning` caplog-based assertion to close the D-A2-5 telemetry-silence concern</name>
  <files>tests/test_shape_hourly_bowl.py</files>
  <read_first>
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-REVIEWS.md (consensus #1 — both Gemini LOW + Codex MEDIUM flag the D-A2-5 `max |level - 1.0| > 1e-6 → logger.warning` path as fire-and-forget; required fix is a caplog test that mocks misaligned f_H and asserts the warning fires)
    - tests/test_shape_hourly_bowl.py (Plans 05C-01 + 05C-02 Tasks 1-4 final state — exposes the `_split_level_anomaly` import, the bowl fixture, and the autouse conftest)
    - pfc_shaping/lt/model/shape_hourly.py (Task 1 final state — `_split_level_anomaly` and its D-A2-5 warning string; confirm the canonical warning message format used by the implementation, which is `logger.warning("_split_level_anomaly: %d timestamps with missing cal — using f_H directly (level=1.0)", n_missing)` for the NaN-cell path; the LEVEL-DRIFT warning is emitted from `pfc_shaping/lt/model/assembler.py` per Task 2 with message `"f_H split: level drift %.2e > 1e-6 — SHP-03 invariant may be degraded"`)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A2-5 (telemetry contract — the warning surface lives in assembler at the build-time check, not in the helper itself)
  </read_first>
  <action>
    Append a single new test `test_split_level_anomaly_drift_warning` to `tests/test_shape_hourly_bowl.py`. This is the M1 fix from `05C-REVIEWS.md` consensus #1: both reviewers (Gemini LOW, Codex MEDIUM) flag that the D-A2-5 `logger.warning("f_H split: level drift %.2e > 1e-6 ...")` path in `pfc_shaping/lt/model/assembler.py` (Task 2) is fire-and-forget. If SHP-03 silently degrades, no CI signal fires. This test closes the loop by asserting (via pytest's `caplog` fixture) that the warning ACTUALLY emits when level drift exceeds 1e-6.

    **Where the warning lives:** the D-A2-5 warning is emitted from `pfc_shaping/lt/model/assembler.py` inside the `if self.sh._use_seasonal_hourly:` branch of `build()` (Task 2 Edit B). The helper `_split_level_anomaly` itself produces `level + anomaly == f_H` ulp-exact by construction; what triggers drift is when the input `f_H` (already passed through `ShapeHourly.fit().apply()`) does NOT have per-cell mean == 1.0 — e.g. because a hypothetical Phase 5 MSFC log-prix re-normalization breaks the SHP-03 invariant. The test SIMULATES this regression by directly calling the assembler path with a synthetic mis-normalized f_H input.

    **Test signature:** `def test_split_level_anomaly_drift_warning(caplog):` — `caplog` is pytest's built-in logging-capture fixture (no import needed; standard pytest plumbing).

    **Test docstring (mandatory references):** cite M1 from `05C-REVIEWS.md` consensus #1, cite D-A2-5, and explain that the test directly invokes `_split_level_anomaly` to extract a clean `(level, anomaly)` decomposition, then mocks the drift scenario by INJECTING `1e-4` of per-cell mean drift into the level series, then re-runs the D-A2-5 telemetry check (the inline assembler logic `max_level_drift = float(abs(level - 1.0).max())` + the conditional `logger.warning(...)` call) and asserts the warning record is captured.

    **Test body — concrete behavior:**
    1. Build a synthetic `f_H_series` of length 96 (one day, 15-min freq, tz="UTC", starting 2027-01-01) with values drawn from a fixed-seed RNG (`np.random.default_rng(456).normal(1.0, 0.05, 96)`) — small spread so the natural per-cell mean is near 1.0.
    2. Build a `cal_df` with at least two cells: first 48 timestamps `("Hiver", "Ouvrable")`, next 48 `("Ete", "Samedi")`, indexed by the same DatetimeIndex.
    3. Call `level, anomaly = _split_level_anomaly(f_H_series, cal_df)`. At this point `max |level - 1.0|` is governed by float roundoff of the per-cell mean — typically well under 1e-12. Confirm via an assertion: `assert float(abs(level - 1.0).max()) < 1e-6, "test setup invalid: level already drifted"`.
    4. INJECT drift: set `level_drifted = level + 1e-4` (uniform 1e-4 shift, simulating a downstream re-normalization bug that breaks the SHP-03 invariant). Confirm `float(abs(level_drifted - 1.0).max()) > 1e-6`.
    5. Trigger the D-A2-5 telemetry path. Two options for how to do this without copy-pasting the entire `build()` body:
       - **Option A (preferred): factor the D-A2-5 check into a tiny helper in `pfc_shaping/lt/model/assembler.py` (or in `shape_hourly.py`) so the test can call it directly.** Pattern: add `def _emit_level_drift_telemetry(level: pd.Series, logger_) -> None:` to `pfc_shaping/lt/model/assembler.py` near the import block. Body: `max_level_drift = float(abs(level - 1.0).max()); logger_.info("f_H split: max |level - 1.0| = %.2e", max_level_drift); if max_level_drift > 1e-6: logger_.warning("f_H split: level drift %.2e > 1e-6 — SHP-03 invariant may be degraded", max_level_drift)`. Then INSIDE the `build()` if-block (Task 2 Edit B), replace the inline 3-line telemetry block with a single call `_emit_level_drift_telemetry(level, logger)`. The test imports and calls `_emit_level_drift_telemetry(level_drifted, logger)` directly. This is a refactor of Task 2's Edit B for testability; backward-compat unchanged (same log messages, same threshold).
       - **Option B (fallback if refactor is undesired): the test reproduces the inline 3-line telemetry logic locally inside the test body, using the SAME logger** (`logger = logging.getLogger("pfc_shaping.lt.model.assembler")` — i.e. the module logger of the assembler) — and asserts that pytest's `caplog` captures the warning. The risk is that the test asserts on a code path it duplicates rather than the production path; Option A is structurally stronger.

       Pick Option A. If Option A requires re-editing Task 2's `<action>`, document that explicitly in the SUMMARY for Plan 05C-02 ("Task 5 required a small refactor of Task 2's Edit B: extracted the D-A2-5 telemetry to `_emit_level_drift_telemetry()` for testability per M1 / `05C-REVIEWS.md`").
    6. Use `caplog.at_level(logging.WARNING, logger="pfc_shaping.lt.model.assembler")` as a context manager around the telemetry call (this is the canonical pytest pattern; it captures records emitted at WARNING level or above from the named logger).
    7. After the telemetry call, assertions:
       - `warning_records = [r for r in caplog.records if r.levelname == "WARNING"]`
       - `assert len(warning_records) == 1, f"expected exactly 1 WARNING record, got {len(warning_records)}: {[r.message for r in warning_records]}"` (exactly one; not zero — the silence concern — and not many — defensive against accidental spamming refactors).
       - `assert "f_H split: level drift" in warning_records[0].message, f"unexpected message: {warning_records[0].message}"` (canonical message substring; matches the format string in `pfc_shaping/lt/model/assembler.py` per Task 2 Edit B).
       - `import re; assert re.search(r"level drift.*1e-?6|level drift.*1\.[0-9]+e-04", warning_records[0].message) or "1e-04" in warning_records[0].message, f"expected drift magnitude in message"` (verify the formatted drift value reflects the injected 1e-4 scale, not the natural sub-1e-12 baseline; defensive against the warning firing with stale data).
    8. Negative-case assertion (defensive): run the same telemetry path on the ORIGINAL `level` (un-drifted). Assert `len([r for r in caplog.records if r.levelname == "WARNING" and "level drift" in r.message]) == 1` still — i.e. the un-drifted call does NOT add a second warning record. Use `caplog.clear()` between calls if needed to keep the assertion clean.

    **Test scope:** ≤ 50 lines. Imports needed: `import logging`, `import re` (if regex assertion used), and either `from pfc_shaping.lt.model.assembler import _emit_level_drift_telemetry` (Option A) OR no new imports (Option B reproduces the logic locally).

    **Update Plan 05C-02 test count expectation:** this task brings the total from 251 → 252 (4 skipped preserved). Update the `<verification>` and `<success_criteria>` blocks accordingly.

    DO NOT modify existing tests, the bowl fixture, the threshold constants, or the existing imports unless needed for Option A's `_emit_level_drift_telemetry` import.
  </action>
  <verify>
    <automated>pytest tests/test_shape_hourly_bowl.py::test_split_level_anomaly_drift_warning -v 2>&amp;1 | tail -8 &amp;&amp; pytest tests/ -x -q 2>&amp;1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `test -f tests/test_shape_hourly_bowl.py` exits 0 (file still exists).
    - `grep -q "def test_split_level_anomaly_drift_warning" tests/test_shape_hourly_bowl.py` exits 0.
    - `grep -q "caplog" tests/test_shape_hourly_bowl.py` exits 0 (M1 fix uses the pytest caplog fixture).
    - `grep -q "M1\|REVIEWS.md\|05C-REVIEWS" tests/test_shape_hourly_bowl.py` exits 0 (cross-AI review traceability cited in docstring or comment).
    - `pytest tests/test_shape_hourly_bowl.py::test_split_level_anomaly_drift_warning -x` exits 0.
    - `pytest tests/test_shape_hourly_bowl.py -v 2>&amp;1 | grep -c PASSED` reports ≥ 5 (4 existing + 1 new).
    - `pytest tests/ -x -q` exits 0 reporting `252 passed, 4 skipped` (251 + 1 new from M1). If `test_seasonal_solar_winter_evening_delta` (Plan 05C-03) tolerance affects count, retain ±1 elasticity and document in SUMMARY.
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` exits 0 (5bis-A baseline still preserved).
    - If Option A taken (Task 2 Edit B refactor for testability): `grep -q "_emit_level_drift_telemetry" pfc_shaping/lt/model/assembler.py` exits 0 AND `grep -q "_emit_level_drift_telemetry" tests/test_shape_hourly_bowl.py` exits 0.
  </acceptance_criteria>
  <done>M1 cross-AI review fix landed: the D-A2-5 telemetry warning is now asserted by `test_split_level_anomaly_drift_warning`. Silent SHP-03 invariant degradation (e.g. from a future Phase 5 MSFC log-prix re-normalization) will fail CI loudly. Suite at 252 passed, 4 skipped.</done>
</task>

</tasks>

<verification>
- `pytest tests/ -x -q` exits 0 reporting `252 passed, 4 skipped` (251 from Tasks 1-4 + 1 new from M1 / Task 5).
- `pytest tests/ --co -q | tail -1` reports `>= 258 tests collected` (251 + 1 M1 test + parametrization).
- `python -c "from pfc_shaping.lt.model.shape_hourly import _split_level_anomaly; from pfc_shaping.lt.model import shape_hourly; assert '_split_level_anomaly' in shape_hourly.__all__; print('OK')"` prints `OK`.
- `grep -n "self.sh._use_seasonal_hourly" pfc_shaping/lt/model/assembler.py | wc -l` reports `>= 1` (the new flag-gated branch).
- `git diff --stat pfc_shaping/lt/model/assembler.py` shows changes ONLY around line 333 plus the top-of-file import (no other modifications to assembler.py).
- `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` exits 0 — the no-op contract is preserved end-to-end across Lever 2 (Wave 2).
- `test_baseline_regression[False]` from 5bis-A in `tests/test_shape_hourly_infra.py` continues to pass at `atol=1e-12, rtol=0`.
- `pytest tests/ -k "shape" -v` shows all shape-related tests green.
</verification>

<success_criteria>
- Lever 2 math change shipped: `assembler.build()` under flag=ON dampens only the `level` component of the f_H split, anomaly survives to far horizon.
- Helper `_split_level_anomaly` exists at module level, exposed via `__all__`, with two D-A2-2 invariants verified by `test_split_level_anomaly_invariant` (D-A4-4).
- SC #3 validated: `np.ptp(f_H)` at ~M+30 under flag=ON exceeds the Wave-0-calibrated `SC3_M30_AMPLITUDE_THRESHOLD` (well above legacy ~0.52, at the plancher 0.50 minimum).
- Backward-compat preserved bit-pour-bit: `test_flag_off_bit_for_bit_baseline` (Plan 05C-01) AND `test_baseline_regression[False]` (5bis-A) both still pass at `atol=1e-12, rtol=0` — the else-branch in `assembler.py` executes the legacy single-line damping unchanged.
- Telemetry drift detection live: `assembler.build()` under flag=ON logs `max |level - 1.0|` at INFO; warns above 1e-6.
- Test count: 249 → 252 (4 skipped preserved). +2 from D-A4-4/D-A4-6 (Task 4) + 1 from M1 telemetry caplog test (Task 5 cross-AI review fix).
- **M1 cross-AI review fix shipped:** `test_split_level_anomaly_drift_warning` asserts via `caplog` that the D-A2-5 `logger.warning` fires when level drift > 1e-6. Silent SHP-03 invariant degradation will fail CI loudly.
- Cross-cutting truth: `flag=OFF baseline 5bis-A preserved at atol=1e-12 rtol=0` holds after this plan.
</success_criteria>

<output>
Create `.planning/phases/05C-shape-hourly-bowl-deepening/05C-02-SUMMARY.md` when done.
</output>
