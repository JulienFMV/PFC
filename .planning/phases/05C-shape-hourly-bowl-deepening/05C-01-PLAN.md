---
phase: 05C-shape-hourly-bowl-deepening
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pfc_shaping/lt/model/shape_hourly.py
  - tests/fixtures/_generate_bowl_fixture.py
  - tests/fixtures/bowl_seed42.parquet
  - tests/test_shape_hourly_bowl.py
  - tests/test_shape_hourly_infra.py
  - scripts/calibrate_bowl_thresholds.py
  - tests/fixtures/_bowl_calibration_report.json
autonomous: true
requirements:
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
must_haves:
  truths:
    - "`_apply_hydro_analogue_weights` Gaussian kernel uses `clim_target[i] = get_climatological_fill(woy(df.index[i]))` per historical sample (instead of scalar `current_fill = float(fill.iloc[-1])`) when `self._use_seasonal_hourly is True` (D-A1-1)."
    - "When `self._use_seasonal_hourly is False`, the kernel target stays `current_fill` (the legacy scalar) and the numerical output of `_apply_hydro_analogue_weights` is bit-pour-bit identical to the 5bis-A baseline at `atol=1e-12, rtol=0` (D-A1-2)."
    - "The floor `np.maximum(hydro_weight, 0.3)` is preserved under both flag states (D-A1-3) — it is applied AFTER the kernel computation in both branches."
    - "`ShapeHourly.__init__` accepts two new kwargs `hydro_weight_sigma_off: float = 0.25` and `hydro_weight_sigma_on: float = 0.08` keeping legacy `hydro_weight_sigma: float | None = None` for backward compat (D-A1-4)."
    - "Legacy callsite `ShapeHourly(hydro_weight_sigma=X)` (where X is non-None) resolves to `self._hydro_weight_sigma_off = self._hydro_weight_sigma_on = X` (D-A1-5 / D-A3-2 resolution precedence)."
    - "The persisted sidecar `shape_hourly.meta.parquet` hyperparams JSON gains three new keys: `hydro_weight_sigma_off`, `hydro_weight_sigma_on`, `hydro_weight_sigma_resolved`. Legacy key `hydro_weight_sigma` remains present (carrying the resolved/active value) for 5bis-A reader compat (D-A3-3)."
    - "`ShapeHourly.load()` cross-plan fallback: if sidecar pre-5bis-B lacks `hydro_weight_sigma_off` key, set `obj._hydro_weight_sigma_off = obj._hydro_weight_sigma_on = hp.get('hydro_weight_sigma', 0.25)` (D-A3-3 cross-plan compat — mirrors the 5bis-A D-07 pattern for `use_seasonal_hourly`)."
    - "Wave 0 calibration is executed via a COMMITTED script `scripts/calibrate_bowl_thresholds.py` (not an interactive terminal one-shot) and produces a COMMITTED immutable artifact `tests/fixtures/_bowl_calibration_report.json` with schema `{calibrated_at: ISO8601, git_sha: str, fixture_sha256: str, ratios: {sc1_ptp_ratio: float, sc1_floor_multiplier: float}, thresholds_emitted: {SC1_PTP_THRESHOLD: float, SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER: float}}`. The threshold constant `SC1_PTP_THRESHOLD` in `tests/test_shape_hourly_bowl.py` is loaded from this report via `json.load(...)['thresholds_emitted']['SC1_PTP_THRESHOLD']` — NOT a free-floating `# observed ratio = X` comment (M2 cross-AI review fix; Codex framing wins)."
    - "`tests/fixtures/_generate_bowl_fixture.py` (seed=42) generates a deterministic synthetic EPEX 15-min DataFrame + a Swiss-like weekly hydro DataFrame covering ≥52 weeks, with an analytically-controlled duck curve (solar depression h10-15 strongest in summer, evening peak h17-20, weekend solar bowl deeper, night discount h22-6). Output `tests/fixtures/bowl_seed42.parquet` (~50KB) is committed (D-A4-1)."
    - "`tests/test_shape_hourly_bowl.py` is created in this plan with: (a) module docstring referencing 5bis-B and D-A4-2, (b) imports + threshold constants loaded from `_bowl_calibration_report.json` (M2), (c) two passing tests `test_hydro_kernel_uses_per_timestamp_climatological_target` (D-A4-3) and `test_flag_off_bit_for_bit_baseline` (D-A4-8). The file is the durable scaffold; plans 02 and 03 will append the remaining 5 tests (D-A4-4, D-A4-5, D-A4-6, D-A4-7, D-A4-9)."
    - "Cross-cutting truth (appears in all 3 plans): `flag=OFF baseline 5bis-A preserved at atol=1e-12 rtol=0`. After this plan, the full suite stays green and `test_baseline_regression[False]` (from 5bis-A `tests/test_shape_hourly_infra.py`) continues to pass against `tests/fixtures/baseline_pfc_seed42.parquet`."
    - "`test_hyperparams_json_has_all_keys` (or any test in `tests/test_shape_hourly_infra.py` asserting the exact set of hyperparams JSON keys) is updated in THIS plan to accept the new key set: at minimum `{halflife_days, hydro_weight_sigma, hydro_weight_sigma_off, hydro_weight_sigma_on, hydro_weight_sigma_resolved, sigma, use_seasonal_hourly}` (sigma_off/sigma_on/sigma_resolved added in Plan 05C-03). RESEARCH Pitfall 4 — only authorized modification to `test_shape_hourly_infra.py` in 5bis-B."
  artifacts:
    - path: "pfc_shaping/lt/model/shape_hourly.py"
      provides: "Refactored `_apply_hydro_analogue_weights` (kernel target gated by flag), extended `__init__` signature (hydro_weight_sigma_off/_on + backward-compat resolution), extended `save()` hyperparams JSON (3 new keys), extended `load()` cross-plan fallback for hydro_weight_sigma."
      contains: "hydro_weight_sigma_off"
    - path: "tests/fixtures/_generate_bowl_fixture.py"
      provides: "Deterministic synthetic EPEX + hydro fixture generator (seed=42)."
      contains: "bowl_seed42"
    - path: "tests/fixtures/bowl_seed42.parquet"
      provides: "Long-format 15-min EPEX parquet with analytically-controlled duck curve (~50KB)."
    - path: "scripts/calibrate_bowl_thresholds.py"
      provides: "Reproducible Wave 0 calibration script (M2 cross-AI review fix). Reads bowl fixture, fits sh_off + sh_on, computes ptp ratios, writes JSON report to tests/fixtures/_bowl_calibration_report.json."
      contains: "calibrate_bowl_thresholds"
    - path: "tests/fixtures/_bowl_calibration_report.json"
      provides: "Immutable calibration artifact (M2 cross-AI review fix). Schema: {calibrated_at, git_sha, fixture_sha256, ratios, thresholds_emitted}. Consumed by tests via json.load; mismatched fixture_sha256 → CI fails loudly (test added in Plan 05C-03)."
      contains: "thresholds_emitted"
    - path: "tests/test_shape_hourly_bowl.py"
      provides: "New isolated test module for 5bis-B math-change tests. Two passing tests after this plan: D-A4-3 (kernel) and D-A4-8 (flag=OFF baseline). Threshold constants loaded from _bowl_calibration_report.json (M2)."
      contains: "test_hydro_kernel_uses_per_timestamp_climatological_target"
    - path: "tests/test_shape_hourly_infra.py"
      provides: "Updated `test_hyperparams_json_has_all_keys` (or equivalent) to accept the extended sidecar schema (RESEARCH Pitfall 4 exception)."
      contains: "hydro_weight_sigma_off"
  key_links:
    - from: "pfc_shaping/lt/model/shape_hourly.py::_apply_hydro_analogue_weights"
      to: "pfc_shaping/lt/model/shape_hourly.py::get_climatological_fill"
      via: "vectorized per-timestamp lookup (nearest-neighbor safe per RESEARCH Pitfall 1)"
      pattern: "get_climatological_fill"
    - from: "pfc_shaping/lt/model/shape_hourly.py::__init__"
      to: "self._use_seasonal_hourly (set by 5bis-A `_resolve_flag`)"
      via: "active-value resolution `self.hydro_weight_sigma = self._hydro_weight_sigma_on if self._use_seasonal_hourly else self._hydro_weight_sigma_off`"
      pattern: "_use_seasonal_hourly"
    - from: "pfc_shaping/lt/model/shape_hourly.py::load"
      to: "shape_hourly.meta.parquet hyperparams JSON"
      via: "cross-plan fallback `if 'hydro_weight_sigma_off' in hp: ... else: legacy_hws = hp.get('hydro_weight_sigma', 0.25); off = on = legacy_hws`"
      pattern: "hydro_weight_sigma_off"
    - from: "tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline"
      to: "tests/fixtures/baseline_pfc_seed42.parquet"
      via: "assert_frame_equal(check_exact=False, atol=1e-12, rtol=0) + identical columns/dtypes/index/sort order"
      pattern: "baseline_pfc_seed42"
    - from: "tests/test_shape_hourly_bowl.py (SC1_PTP_THRESHOLD constant)"
      to: "tests/fixtures/_bowl_calibration_report.json"
      via: "json.load(open(...))['thresholds_emitted']['SC1_PTP_THRESHOLD'] at module load time"
      pattern: "_bowl_calibration_report"
---

<objective>
Implement Lever 1 of Phase 5bis-B: refactor `_apply_hydro_analogue_weights` so that under `self._use_seasonal_hourly == True` the Gaussian kernel target becomes the per-timestamp climatological fill `get_climatological_fill(week_of_year(t))` (replacing the scalar `current_fill = float(fill.iloc[-1])`). Under `self._use_seasonal_hourly == False` the legacy scalar `current_fill` is preserved, yielding numerical bit-pour-bit equality with the 5bis-A baseline (`atol=1e-12, rtol=0`).

Extend `ShapeHourly.__init__` with two new flag-aware kwargs `hydro_weight_sigma_off=0.25` and `hydro_weight_sigma_on=0.08` (calibrated value from `05C-RESEARCH.md` §Lever 1, dry-run on ±10pp anomaly distribution), preserving full backward-compat for the legacy callsite `ShapeHourly(hydro_weight_sigma=X)` (legacy wins → `off = on = X`).

Extend the `${stem}.meta.parquet` sidecar to persist three new keys (`hydro_weight_sigma_off`, `hydro_weight_sigma_on`, `hydro_weight_sigma_resolved`) while preserving the legacy `hydro_weight_sigma` key for 5bis-A readers. Implement cross-plan fallback at `load()` for sidecars pre-5bis-B that lack the new keys.

Create the deterministic synthetic duck-curve fixture (`tests/fixtures/_generate_bowl_fixture.py` + `tests/fixtures/bowl_seed42.parquet`) and the new isolated test module `tests/test_shape_hourly_bowl.py` scaffolded with the two tests that fit in this plan's scope (D-A4-3 kernel test, D-A4-8 flag=OFF baseline). Wave 0 calibrates `SC1_PTP_THRESHOLD` on the new fixture via the **committed reproducible script** `scripts/calibrate_bowl_thresholds.py` (M2 cross-AI review fix — Codex framing wins over the original interactive-only approach), producing the **committed immutable artifact** `tests/fixtures/_bowl_calibration_report.json` (schema: calibrated_at, git_sha, fixture_sha256, ratios, thresholds_emitted). The test file loads the threshold from this JSON via `json.load`, not from a free-floating in-comment value.

Update the single 5bis-A test that asserts the exact hyperparams JSON key set (`test_hyperparams_json_has_all_keys` or equivalent in `tests/test_shape_hourly_infra.py`) to accept the extended schema. RESEARCH Pitfall 4 — only authorized modification to `test_shape_hourly_infra.py` across 5bis-B.

Purpose: This plan delivers the "hydro kernel reformulation" lever (the biggest expected contribution to SC #1 ptp deepening per RESEARCH §Lever 1) and establishes the bowl-fixture + new-test-module scaffold that plans 02 and 03 will extend. The flag=OFF baseline test proves the regression contract continues to hold after the math change. The committed calibration script + JSON sidecar (M2) closes the cross-AI review's auditability concern: a secondary quant reviewer can re-run `python scripts/calibrate_bowl_thresholds.py` and observe whether the artifact changes — if it does, the test asserts a fixture mismatch and CI fails loudly.

Output: 1 modified production file (`shape_hourly.py`), 1 new fixture generator script, 1 new fixture parquet, 1 new committed calibration script (`scripts/calibrate_bowl_thresholds.py`), 1 new committed calibration report (`tests/fixtures/_bowl_calibration_report.json`), 1 new test module (with 2 passing tests + threshold constants loaded from JSON), 1 surgical update to an existing infra test. Test suite goes from 247 → 249 passing (4 skipped preserved).
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
@.planning/phases/05C-shape-hourly-bowl-deepening/05C-REVIEWS.md
@.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md
@.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md
@pfc_shaping/lt/model/shape_hourly.py
@pfc_shaping/lt/model/assembler.py
@tests/fixtures/_generate_baseline.py
@tests/fixtures/baseline_pfc_seed42.parquet
@tests/conftest.py
@tests/test_shape_hourly_infra.py

<interfaces>
Key contracts already established by 5bis-A that this plan extends (NOT re-explores):

From `pfc_shaping/lt/model/shape_hourly.py` (5bis-A frozen state):
```
GAUSSIAN_SIGMA: float = 0.5
_META_SIDECAR_SUFFIX: str = ".meta.parquet"
_FLAG_ENV_VAR: str = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"

def _resolve_flag(explicit: bool | None) -> bool: ...
def _meta_path(main_path) -> Path: ...

class ShapeHourly:
    def __init__(
        self,
        sigma: float = GAUSSIAN_SIGMA,           # → becomes `sigma: float | None = None` in Plan 05C-03 (NOT in this plan)
        halflife_days: float = 180.0,
        hydro_weight_sigma: float = 0.25,        # → THIS plan changes to `hydro_weight_sigma: float | None = None` + adds hydro_weight_sigma_off/_on
        use_seasonal_hourly: bool | None = None,
    ) -> None: ...

    def fit(self, epex_df, calendar_df, hydro_df=None) -> "ShapeHourly": ...
    def apply(self, timestamps, calendar_df, reference_date=None) -> pd.Series: ...
    def get_climatological_fill(self, week: int) -> float: ...  # nearest-neighbor safe (shape_hourly.py:362-371)
    def save(self, path) -> None: ...                            # writes ${stem}.meta.parquet via _meta_path()
    @classmethod
    def load(cls, path) -> "ShapeHourly": ...                    # reads ${stem}.meta.parquet, with legacy-compat warning if absent

    # Private state (set in __init__, persisted, frozen):
    self._use_seasonal_hourly: bool                              # 5bis-A
    self._climatological_fill: pd.Series | None                  # set in _apply_hydro_analogue_weights
    self._hydro_fill_weekly: pd.Series | None                    # set in _apply_hydro_analogue_weights
```

Current `_apply_hydro_analogue_weights` body (shape_hourly.py:839-911) — kernel computation lives at lines 895-902:
```
sigma = self.hydro_weight_sigma
hydro_weight = np.exp(-0.5 * ((fill_values - current_fill) / sigma) ** 2)
hydro_weight = np.where(np.isnan(hydro_weight), 1.0, hydro_weight)
hydro_weight = np.maximum(hydro_weight, 0.3)  # floor
```
where `current_fill = float(fill.iloc[-1])` is set at line 877.

Current `save()` hyperparams JSON (shape_hourly.py:518-529):
```
meta_records.append({
    "attr": "hyperparams",
    "value": json.dumps(
        {
            "halflife_days": self.halflife_days,
            "hydro_weight_sigma": self.hydro_weight_sigma,
            "sigma": self.sigma,
            "use_seasonal_hourly": bool(self._use_seasonal_hourly),
        },
        sort_keys=True,
    ),
})
```

Current `load()` hyperparams restore (shape_hourly.py:563-575):
```
hp_rows = meta_df[meta_df["attr"] == "hyperparams"]
if len(hp_rows) > 0:
    hp = json.loads(hp_rows["value"].iloc[0])
    obj.sigma = hp.get("sigma", obj.sigma)
    obj.halflife_days = hp.get("halflife_days", obj.halflife_days)
    obj.hydro_weight_sigma = hp.get("hydro_weight_sigma", obj.hydro_weight_sigma)
    if "use_seasonal_hourly" in hp:
        obj._use_seasonal_hourly = bool(hp["use_seasonal_hourly"])
```

From `tests/fixtures/_generate_baseline.py` (5bis-A pattern — model for `_generate_bowl_fixture.py`):
```
def build_pfc(seed: int = 42, flag: bool = False) -> pd.DataFrame: ...
def main() -> None: ...
```

Test counts at start of this plan (verified state from 5bis-A): `247 passed, 4 skipped`. This plan must end at `249 passed, 4 skipped` (adds 2 new tests + updates 1 existing test; total count delta = +2).
</interfaces>
</context>

<deferred_research>
**T1 (cross-AI review consensus — Codex HIGH, Gemini not flagged):** `hydro_weight_sigma_on = 0.08` is calibrated against simulated N(0, 0.10) anomalies from RESEARCH §Lever 1 dry-run, NOT against Swiss historical reservoir anomaly distribution. A real Swiss historical reservoir anomaly diagnostic (one-off offline artifact, quantiles + floor-hit rate, source: BFE / OFEN weekly reservoir data) should validate or revise this value before D-FLIP-1 production flip. Tracked as A3 in RESEARCH.md §Assumptions. **Does not block 5bis-B ship** — `flag=OFF` default + D-FLIP-1 gate (Phase 10 Δ MAE bloc ≤ -1.5 EUR/MWh vs HFC OMPEX) provide the safety net. Follow-up phase scope: ~1 day offline notebook, output a markdown diagnostic with histogram + quantile table + recommendation (keep 0.08, or revise to data-driven value).
</deferred_research>

<tasks>

<task type="auto">
  <name>Task 1 (Wave 0): Create deterministic bowl fixture generator + bowl_seed42.parquet</name>
  <files>tests/fixtures/_generate_bowl_fixture.py, tests/fixtures/bowl_seed42.parquet</files>
  <read_first>
    - tests/fixtures/_generate_baseline.py (5bis-A pattern — function signatures, seed=42 convention, repo-root sys.path bootstrap, docstring discipline)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Synthetic Fixture Design (canonical formulas for `_build_bowl_epex` and `_build_hydro_df` — copy these verbatim into the generator, adapting only docstring wording)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A4-1 (fixture contract: ~50KB, 3 months 15-min, analytically-controlled duck curve, seed=42)
  </read_first>
  <action>
    Create `tests/fixtures/_generate_bowl_fixture.py` modeled on `tests/fixtures/_generate_baseline.py`. Required structure:

    1. Module docstring naming purpose: "Deterministic bowl-fixture generator for Phase 5bis-B. Produces `tests/fixtures/bowl_seed42.parquet` (synthetic 15-min EPEX with analytically-controlled duck curve) used by `tests/test_shape_hourly_bowl.py` for SC #1, SC #2, SC #3 validation on synthetic data (RESEARCH §Innovation gating: math validation gate, NOT real-data validation gate which is Phase 10)." Reference D-A4-1 in the docstring.
    2. Imports identical to `_generate_baseline.py` (numpy, pandas, sys, Path bootstrap to repo root). Do NOT import anything from `pfc_shaping` — the fixture writes parquets directly via `pandas.DataFrame.to_parquet` so it stays byte-stable across future refactors of save/apply.
    3. Function `_build_bowl_epex(n_15min: int, rng: np.random.Generator) -> pd.DataFrame` — implement verbatim per RESEARCH §Synthetic Fixture Design code block. Key parameters: base 80 EUR/MWh, annual seasonal cycle ±8 EUR/MWh, summer mask `doy ∈ [152, 244]`, solar depression -18 EUR/MWh (summer) / -2 EUR/MWh (winter) on h10-15, weekend summer solar deeper -25 EUR/MWh, evening peak +22 EUR/MWh on h17-20, night discount -8 EUR/MWh on h22-6, noise N(0, 2). Clip to [-50, 200]. Index: 3 months × 96 slots/day = ~8640 timestamps starting 2022-01-01 UTC freq="15min". Return long-format `DataFrame({"price_eur_mwh": price}, index=idx)`.
    4. Function `_build_hydro_df(rng: np.random.Generator) -> pd.DataFrame` — implement per RESEARCH: 104 weeks (2 years) starting 2022-01-01 UTC freq="W", `seasonal = 0.55 + 0.25 * sin(2π * (doy - 30) / 365)`, noise N(0, 0.05), clip [0.05, 0.98], scale to percentage (0-100). RESEARCH Pitfall 1: hydro must cover ≥52 weeks so `_climatological_fill` covers all WOY; 104 weeks is the chosen safe upper bound. Return `DataFrame({"fill_pct": fill * 100}, index=weeks)`.
    5. Function `build_bowl_fixture(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(epex_df, hydro_df)`. This is the reusable entry point for `tests/test_shape_hourly_bowl.py` tests (they import and call this).
    6. `main()`: instantiate `rng = np.random.default_rng(42)`, build both DataFrames via `build_bowl_fixture()`, write ONLY the EPEX DataFrame to `tests/fixtures/bowl_seed42.parquet` via `to_parquet(_OUT_PATH, index=True)`. The hydro DataFrame is built in-process by tests (it is deterministic from the same seed), so it does NOT need a separate committed parquet. Print confirmation line with row count + columns + hash of `price_eur_mwh` for grep-based acceptance.
    7. `if __name__ == "__main__": main()` block.
    8. Run the generator once via `python tests/fixtures/_generate_bowl_fixture.py` so the committed parquet is identical to what tests will re-derive. The parquet must be ≤ 100KB (target ~50KB per D-A4-1).

    Do NOT modify any other file in this task.
  </action>
  <verify>
    <automated>python tests/fixtures/_generate_bowl_fixture.py && python -c "
import pandas as pd, os
p = 'tests/fixtures/bowl_seed42.parquet'
df = pd.read_parquet(p)
assert 'price_eur_mwh' in df.columns, df.columns.tolist()
assert isinstance(df.index, pd.DatetimeIndex), type(df.index)
assert df.index.tz is not None, 'index must be tz-aware UTC'
assert str(df.index.tz) == 'UTC', f'tz must be UTC, got {df.index.tz}'
assert len(df) >= 8000 and len(df) <= 10000, f'expected ~8640 rows, got {len(df)}'
size_kb = os.path.getsize(p) / 1024
assert size_kb &lt; 200, f'fixture too large: {size_kb:.1f} KB (target ~50KB, ceiling 200)'
print(f'OK rows={len(df)} size={size_kb:.1f}KB price_range=[{df.price_eur_mwh.min():.1f}, {df.price_eur_mwh.max():.1f}]')
"</automated>
  </verify>
  <acceptance_criteria>
    - `test -f tests/fixtures/_generate_bowl_fixture.py` exits 0.
    - `test -f tests/fixtures/bowl_seed42.parquet` exits 0.
    - `grep -q "build_bowl_fixture" tests/fixtures/_generate_bowl_fixture.py` exits 0 (reusable entry point present).
    - `grep -q "D-A4-1" tests/fixtures/_generate_bowl_fixture.py` exits 0 (decision ID traceable).
    - `grep -q "np.random.default_rng(42)\|seed=42\|seed: int = 42" tests/fixtures/_generate_bowl_fixture.py` exits 0 (seed convention).
    - Verify command above prints `OK rows=...` and exits 0.
    - Re-running `python tests/fixtures/_generate_bowl_fixture.py` produces a byte-identical parquet (determinism check): `sha256sum tests/fixtures/bowl_seed42.parquet` before and after must match.
    - `pytest tests/ -x -q` exits 0 reporting `247 passed, 4 skipped` (no new test consumes the fixture yet — added in Task 5).
  </acceptance_criteria>
  <done>Fixture generator + parquet committed; reusable entry point `build_bowl_fixture()` is available for downstream tests.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor `_apply_hydro_analogue_weights` (Lever 1 kernel reformulation) + extend `__init__` with hydro_weight_sigma_off/_on + extend save/load sidecar</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py:44-95 (module constants + `_resolve_flag` + `_meta_path` 5bis-A helpers — pattern to clone for `_resolve_sigma_pair`)
    - pfc_shaping/lt/model/shape_hourly.py:166-195 (current `__init__` signature)
    - pfc_shaping/lt/model/shape_hourly.py:516-530 (current `save()` hyperparams JSON block)
    - pfc_shaping/lt/model/shape_hourly.py:562-575 (current `load()` hyperparams restore)
    - pfc_shaping/lt/model/shape_hourly.py:839-911 (`_apply_hydro_analogue_weights` full body — refactor target)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Lever 1 (sigma_on=0.08 calibration), §Lever 3 (D-A3-2 resolution precedence, conflict-detection logic), §Code Surface Map Plan 05C-01
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Implementation Pitfalls 1 (nearest-neighbor `get_climatological_fill` access), §Common Pitfalls A (vectorized lookup)
    - pfc_shaping/pipeline/autoresearch.py:234, pfc_shaping/pipeline/rolling_update.py:365 (legacy callsites that must continue to work)
  </read_first>
  <action>
    Modify `pfc_shaping/lt/model/shape_hourly.py` in four surgical edits:

    **Edit A — Extend `__init__` signature (lines 166-192):**
    Change `hydro_weight_sigma: float = 0.25` to `hydro_weight_sigma: float | None = None`. Add two new keyword arguments AFTER `hydro_weight_sigma`: `hydro_weight_sigma_off: float = 0.25` and `hydro_weight_sigma_on: float = 0.08`. Place them BEFORE the existing `use_seasonal_hourly` kwarg so the resolution can run after `self._use_seasonal_hourly` is set. The constants `0.25` and `0.08` are calibrated values from RESEARCH §Lever 1 (legacy 0.25 preserved as off-default; on-default 0.08 dry-run-derived for ±10pp anomaly scale).

    Inside the body, AFTER `self._use_seasonal_hourly = _resolve_flag(use_seasonal_hourly)` is computed, add the resolution block (RESEARCH §Lever 3 D-A3-2 pattern, adapted for hydro_weight_sigma only — sigma resolution lives in Plan 05C-03):
    - If `hydro_weight_sigma is not None`: detect conflict — if `hydro_weight_sigma_off != 0.25` or `hydro_weight_sigma_on != 0.08`, emit `logger.warning("ShapeHourly: hydro_weight_sigma=%r (legacy) AND hydro_weight_sigma_off=%r/hydro_weight_sigma_on=%r both passed; legacy hydro_weight_sigma wins for both flag states (D-A3-2)", hydro_weight_sigma, hydro_weight_sigma_off, hydro_weight_sigma_on)`. Then assign `self._hydro_weight_sigma_off = self._hydro_weight_sigma_on = float(hydro_weight_sigma)`.
    - Else: `self._hydro_weight_sigma_off = float(hydro_weight_sigma_off)`, `self._hydro_weight_sigma_on = float(hydro_weight_sigma_on)`.
    - Active-value resolution: `self.hydro_weight_sigma = self._hydro_weight_sigma_on if self._use_seasonal_hourly else self._hydro_weight_sigma_off`. This single attribute is the runtime kernel bandwidth read by `_apply_hydro_analogue_weights`.

    Defaults `0.25` and `0.08` MUST be defined at module level above `_FLAG_ENV_VAR` (near `GAUSSIAN_SIGMA = 0.5`) as constants `_HYDRO_WEIGHT_SIGMA_OFF_DEFAULT = 0.25` and `_HYDRO_WEIGHT_SIGMA_ON_DEFAULT = 0.08` so the conflict-detection comparison uses the canonical defaults (RESEARCH Pitfall 3 — never compare against received params).

    Backward-compat audit (verify by reading the four callsites):
    - `ShapeHourly()` → `hydro_weight_sigma=None`, `off=0.25`, `on=0.08` → `self.hydro_weight_sigma = 0.25` (when flag=OFF) identical to legacy 0.25 — OK.
    - `ShapeHourly(hydro_weight_sigma=0.25)` (test_shape_hourly_infra.py:239,250) → legacy wins, both off and on become 0.25, no warning (off/on are at defaults) — OK.
    - `ShapeHourly(hydro_weight_sigma=0.7)` (test_shape_hourly_infra.py:51,274,375,430) → legacy wins, both become 0.7, no warning — OK.
    - `ShapeHourly(sigma=sigma)` (autoresearch.py:234) — `hydro_weight_sigma` not passed → defaults active — OK.

    **Edit B — Refactor `_apply_hydro_analogue_weights` body (lines 839-911):**

    The current line 877 `current_fill = float(fill.iloc[-1])` is preserved (still used under flag=OFF). The current line 900 kernel `np.exp(-0.5 * ((fill_values - current_fill) / sigma) ** 2)` becomes a conditional branch gated by `self._use_seasonal_hourly`.

    Concretely, AFTER `fill_values = fill_at_date.values.astype(float)` (around line 895) and BEFORE the `sigma = self.hydro_weight_sigma` line, insert a branch:
    - When `self._use_seasonal_hourly` is True: compute `clim_target` array per timestamp. Use the vectorized pattern from RESEARCH §Common Pitfalls A: extract `woy_arr` from `df.index` via `df.index.isocalendar().week.values` (or the fallback `df.index.to_series().dt.isocalendar().week.values` when `isocalendar` is unavailable on the index — match the existing fallback at lines 868-871 for consistency). Build a dict `clim_map = {int(w): self.get_climatological_fill(int(w)) for w in np.unique(woy_arr)}` (≤52 lookups, all nearest-neighbor safe per RESEARCH Pitfall 1). Build `clim_target = np.array([clim_map[int(w)] for w in woy_arr], dtype=float)`. **Normalize scale:** `get_climatological_fill` returns the fill in the same units as `_climatological_fill` was constructed from `fill.values` AFTER the existing percentage-normalization at lines 859-860 (`if fill.max() > 1.5: fill = fill / 100.0`). Since `_climatological_fill` is built from the already-normalized `fill` (line 873 uses `fill.values`), `clim_target` is in the [0, 1] range — same scale as `fill_values`. No additional unit conversion is required. Add an inline comment naming this invariant.
    - When `self._use_seasonal_hourly` is False: use `clim_target = current_fill` (scalar broadcast). This preserves legacy bit-pour-bit equality.

    The kernel computation becomes:
    `hydro_weight = np.exp(-0.5 * ((fill_values - clim_target) / sigma) ** 2)`
    (`current_fill` is replaced by `clim_target` which is either the scalar `current_fill` under flag=OFF, or the vector of per-timestamp climato targets under flag=ON.)

    Floor and NaN handling at lines 901-902 (`np.where(np.isnan(...), 1.0, ...)` and `np.maximum(hydro_weight, 0.3)`) are PRESERVED unchanged — both branches share them (D-A1-3).

    Update the `logger.info("Hydro analogue: current fill=...")` line 878 to include the flag state for traceability: when flag=ON, log `"Hydro analogue: flag=ON, per-timestamp clim target (mean=%.1f%%), σ=%.2f, ..."` using `np.mean(clim_target) * 100` instead of `current_fill * 100`. When flag=OFF, keep the legacy log message verbatim.

    **Edit C — Extend `save()` hyperparams JSON (lines 518-529):**
    Add three new keys to the JSON dict (after `hydro_weight_sigma`, before `sigma`, preserving sort_keys=True): `"hydro_weight_sigma_off": self._hydro_weight_sigma_off`, `"hydro_weight_sigma_on": self._hydro_weight_sigma_on`, `"hydro_weight_sigma_resolved": self.hydro_weight_sigma`. Keep the existing `"hydro_weight_sigma": self.hydro_weight_sigma` key — it carries the resolved/active value identical to `hydro_weight_sigma_resolved` and is preserved for 5bis-A reader compat (no breakage of code that reads the legacy single-σ key).

    **Edit D — Extend `load()` hyperparams restore (lines 562-575):**
    Replace the `obj.hydro_weight_sigma = hp.get("hydro_weight_sigma", obj.hydro_weight_sigma)` line with the cross-plan fallback block (D-A3-3 / RESEARCH §Lever 3 Cross-plan fallback):
    - If `"hydro_weight_sigma_off" in hp`: `obj._hydro_weight_sigma_off = float(hp["hydro_weight_sigma_off"])`, `obj._hydro_weight_sigma_on = float(hp["hydro_weight_sigma_on"])`.
    - Else (sidecar 5bis-A or earlier — only `hydro_weight_sigma` legacy key present): `legacy_hws = float(hp.get("hydro_weight_sigma", _HYDRO_WEIGHT_SIGMA_OFF_DEFAULT))`, `obj._hydro_weight_sigma_off = obj._hydro_weight_sigma_on = legacy_hws`.
    - Active-value: `obj.hydro_weight_sigma = float(hp.get("hydro_weight_sigma_resolved", hp.get("hydro_weight_sigma", obj.hydro_weight_sigma)))`.

    Do NOT touch the sigma / use_seasonal_hourly restore lines — those are 5bis-A's contract and Plan 05C-03's extension surface.

    After all four edits, verify with a manual sanity run: `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; sh = ShapeHourly(); print(sh.hydro_weight_sigma, sh._hydro_weight_sigma_off, sh._hydro_weight_sigma_on)"` must print `0.25 0.25 0.08` (default flag=OFF).
  </action>
  <verify>
    <automated>python -c "
from pfc_shaping.lt.model.shape_hourly import ShapeHourly, _HYDRO_WEIGHT_SIGMA_OFF_DEFAULT, _HYDRO_WEIGHT_SIGMA_ON_DEFAULT
assert _HYDRO_WEIGHT_SIGMA_OFF_DEFAULT == 0.25
assert _HYDRO_WEIGHT_SIGMA_ON_DEFAULT == 0.08
# Default (no arg, flag OFF) — backward compat with 5bis-A
sh = ShapeHourly()
assert sh._use_seasonal_hourly is False
assert sh._hydro_weight_sigma_off == 0.25
assert sh._hydro_weight_sigma_on == 0.08
assert sh.hydro_weight_sigma == 0.25
# Legacy single-σ callsite (autoresearch / rolling_update / infra tests)
sh = ShapeHourly(hydro_weight_sigma=0.7)
assert sh._hydro_weight_sigma_off == 0.7
assert sh._hydro_weight_sigma_on == 0.7
assert sh.hydro_weight_sigma == 0.7
# Flag ON, no legacy override
sh = ShapeHourly(use_seasonal_hourly=True)
assert sh._use_seasonal_hourly is True
assert sh.hydro_weight_sigma == 0.08
print('OK init')
" &amp;&amp; pytest tests/test_shape_hourly_infra.py -x -q 2>&amp;1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "_HYDRO_WEIGHT_SIGMA_OFF_DEFAULT = 0.25" pfc_shaping/lt/model/shape_hourly.py` matches a module-level definition.
    - `grep -n "_HYDRO_WEIGHT_SIGMA_ON_DEFAULT = 0.08" pfc_shaping/lt/model/shape_hourly.py` matches a module-level definition.
    - `grep -n "hydro_weight_sigma_off: float = 0.25" pfc_shaping/lt/model/shape_hourly.py` matches `__init__` signature.
    - `grep -n "hydro_weight_sigma_on: float = 0.08" pfc_shaping/lt/model/shape_hourly.py` matches `__init__` signature.
    - `grep -n "clim_target" pfc_shaping/lt/model/shape_hourly.py` finds the new branch in `_apply_hydro_analogue_weights`.
    - `grep -v '^[[:space:]]*#' pfc_shaping/lt/model/shape_hourly.py | grep -c "hydro_weight_sigma_resolved"` reports ≥ 2 (one in save, one in load).
    - The python init-sanity command above prints `OK init`.
    - `pytest tests/test_shape_hourly_infra.py -x -q` exits 0 with all 247 baseline tests except the hyperparams-key-set tests passing (those are updated in Task 3 below; before Task 3 runs, expect 1 or 2 failures on `test_hyperparams_row` / `test_save_unfitted_hyperparams_correct` for the now-extended JSON schema — record the failure list in the SUMMARY and move to Task 3).
    - `pytest tests/test_country_tz_plumbing.py tests/test_long_term_branch.py -x -q` exits 0 (unrelated suite still green, ensuring no cross-cutting regression).
  </acceptance_criteria>
  <done>`shape_hourly.py` refactored with the four surgical edits; legacy single-σ callsites preserved; sidecar schema extended; per-timestamp climato kernel gated by flag.</done>
</task>

<task type="auto">
  <name>Task 3: Update `test_shape_hourly_infra.py` hyperparams-key-set tests for the extended sidecar schema (RESEARCH Pitfall 4 surgical exception)</name>
  <files>tests/test_shape_hourly_infra.py</files>
  <read_first>
    - tests/test_shape_hourly_infra.py:197-263 (`test_hyperparams_row`, `test_save_unfitted_hyperparams_correct` — exact assertion sets that must be updated)
    - pfc_shaping/lt/model/shape_hourly.py (post-Task-2 state — confirms the new keys are present)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Pitfall 4 (explicit authorization for this surgical update — only allowed touch to test_shape_hourly_infra.py in 5bis-B)
  </read_first>
  <action>
    Update two tests in `tests/test_shape_hourly_infra.py` to reflect the extended hyperparams JSON schema produced by `save()` after Task 2. Do NOT touch any other test in this file.

    **Update 1 — `test_hyperparams_row` (around line 197):** Currently asserts the dict equals `{"halflife_days": 90.0, "hydro_weight_sigma": 0.7, "sigma": 0.3, "use_seasonal_hourly": False}`. Update to assert equality against `{"halflife_days": 90.0, "hydro_weight_sigma": 0.7, "hydro_weight_sigma_off": 0.7, "hydro_weight_sigma_on": 0.7, "hydro_weight_sigma_resolved": 0.7, "sigma": 0.3, "use_seasonal_hourly": False}`. The `0.7` for off/on/resolved derives from the legacy single-σ callsite `ShapeHourly(hydro_weight_sigma=0.7)` in `_minimal_fitted_sh` (D-A1-5 backward-compat: legacy wins → off=on=0.7).

    **Update 2 — `test_save_unfitted_hyperparams_correct` (around line 249):** Currently asserts the dict equals `{"halflife_days": 180.0, "hydro_weight_sigma": 0.25, "sigma": 0.5, "use_seasonal_hourly": False}`. Update to assert equality against `{"halflife_days": 180.0, "hydro_weight_sigma": 0.25, "hydro_weight_sigma_off": 0.25, "hydro_weight_sigma_on": 0.25, "hydro_weight_sigma_resolved": 0.25, "sigma": 0.5, "use_seasonal_hourly": False}`. The constructor `ShapeHourly(sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25)` passes `hydro_weight_sigma=0.25` (legacy wins → off=on=0.25).

    For BOTH updates, the test docstring must gain an inline reference to the source of the schema change:
    > "Updated by Plan 05C-01 (D-A3-3 / RESEARCH Pitfall 4): hyperparams JSON gains hydro_weight_sigma_off/_on/_resolved keys when 5bis-B Lever 1 ships. Plan 05C-03 will add the sigma_off/_on/_resolved triplet; that follow-up update is the responsibility of Plan 05C-03 Task 3."

    DO NOT pre-add the `sigma_off`/`sigma_on`/`sigma_resolved` keys to these tests in this plan — they are introduced by Plan 05C-03 Lever 3 and adding them prematurely would cause Plan 05C-03 to need a re-update. Strict scope discipline: this plan covers ONLY the hydro_weight_sigma triplet.

    DO NOT touch any other test in `test_shape_hourly_infra.py` (test_factors_3d_view_consistency, test_save_load_full_roundtrip, test_flag_freeze_at_init, test_baseline_regression, test_no_hidden_behavior_branch, etc.). Those tests assert behaviors that are PRESERVED across Lever 1 (the flag is still freeze-at-init, save/load full roundtrip still works including the new keys via the new `_hydro_weight_sigma_off/_on` private attrs which are also covered by the existing roundtrip equality check on `hydro_weight_sigma` resolved).

    Verify the roundtrip test still passes: the `test_save_load_full_roundtrip` test (5bis-A Plan 05B-05 Task 3) asserts equality on `sh.hydro_weight_sigma` after save→load. With Task 2's `load()` setting `obj.hydro_weight_sigma = hp.get("hydro_weight_sigma_resolved", ...)` and Task 2's `save()` writing `hydro_weight_sigma_resolved = self.hydro_weight_sigma`, the roundtrip preserves the resolved value identically. No update needed.
  </action>
  <verify>
    <automated>pytest tests/test_shape_hourly_infra.py -x -q 2>&amp;1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_shape_hourly_infra.py::TestSaveBasic::test_hyperparams_row -x` exits 0.
    - `pytest tests/test_shape_hourly_infra.py::TestSaveUnfitted::test_save_unfitted_hyperparams_correct -x` exits 0 (class name may differ slightly — match the actual containing class).
    - `pytest tests/test_shape_hourly_infra.py -x -q` exits 0 reporting `247 passed, 4 skipped` (5bis-A baseline preserved).
    - `grep -q "hydro_weight_sigma_off" tests/test_shape_hourly_infra.py` exits 0 (new key referenced in updated tests).
    - `grep -q "Plan 05C-01" tests/test_shape_hourly_infra.py` exits 0 (traceability comment present).
    - `grep -c "sigma_off" tests/test_shape_hourly_infra.py | head -1 | xargs test 5 -ge` — confirm `sigma_off` (without `hydro_weight_` prefix) is NOT yet referenced in this file (scope discipline; Plan 05C-03 will add it).
  </acceptance_criteria>
  <done>Two infra-suite tests updated to accept the extended schema; full 5bis-A baseline regression remains green at `atol=1e-12, rtol=0`.</done>
</task>

<task type="auto">
  <name>Task 4 (Wave 0 — REVISED per M2 cross-AI review): Create committed reproducible calibration script `scripts/calibrate_bowl_thresholds.py` + immutable JSON sidecar `tests/fixtures/_bowl_calibration_report.json`</name>
  <files>scripts/calibrate_bowl_thresholds.py, tests/fixtures/_bowl_calibration_report.json</files>
  <read_first>
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-REVIEWS.md (consensus item 3 + recommended-action #2 — Codex framing of the auditability fix: committed script + immutable JSON sidecar with calibrated_at, git_sha, fixture_sha256, ratios, thresholds_emitted)
    - tests/fixtures/_generate_bowl_fixture.py (Task 1 output — exposes `build_bowl_fixture()`)
    - pfc_shaping/lt/model/shape_hourly.py (post-Task-2 state — confirms ShapeHourly `use_seasonal_hourly` kwarg accepts True/False)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Lever 1 (np.ptp threshold formula — `threshold = max(observed_ratio - 0.15, 1.05)` with plancher 1.05)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A4-3, D-A4-5 (the SC #1 measure-then-assert protocol)
  </read_first>
  <action>
    This task REPLACES the original "interactive terminal Wave 0 calibration" with a committed reproducible script + immutable JSON artifact. This implements **M2 from `05C-REVIEWS.md` consensus** (cross-AI review fix; Codex framing wins over Gemini's lighter "hidden helper" suggestion — Codex's auditability framing is stronger for trading-grade math changes).

    **Part A — Create `scripts/calibrate_bowl_thresholds.py` (a committed, executable, version-controlled script):**

    File path: `scripts/calibrate_bowl_thresholds.py` (create `scripts/` directory if it does not exist; verify first with `test -d scripts && ls scripts/ | head -5`; if scripts/ already contains a `_generate_*.py` or `*.py` artifact, co-locate; do NOT replace existing files).

    Required structure:
    1. Module docstring: "Wave 0 calibration script for Phase 5bis-B SC #1 (ptp ratio threshold). Reads tests/fixtures/bowl_seed42.parquet, fits sh_off + sh_on, computes ptp ratio on (Ete, Ouvrable) cell, writes immutable JSON artifact tests/fixtures/_bowl_calibration_report.json. Commit BOTH this script AND the JSON output to git after every re-run. M2 cross-AI review fix (REVIEWS.md consensus #3). Reproduce via: python scripts/calibrate_bowl_thresholds.py."
    2. Imports: `from __future__ import annotations`, `argparse`, `hashlib`, `json`, `subprocess`, `datetime`, `pathlib.Path`, `numpy as np`. Plus the repo bootstrap (sys.path manipulation if needed) followed by `from tests.fixtures._generate_bowl_fixture import build_bowl_fixture`, `from pfc_shaping.data.calendar_ch import enrich_15min_index`, `from pfc_shaping.lt.model.shape_hourly import ShapeHourly`.
    3. Module constants:
       - `REPO_ROOT = Path(__file__).resolve().parent.parent`
       - `FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "bowl_seed42.parquet"`
       - `REPORT_PATH = REPO_ROOT / "tests" / "fixtures" / "_bowl_calibration_report.json"`
       - `SC1_FLOOR_MULTIPLIER = 1.05` (RESEARCH §Lever 1 plancher)
       - `SC1_RATIO_MARGIN = 0.15` (RESEARCH §Lever 1 `max(ratio - 0.15, plancher)` formula)
    4. Function `_compute_fixture_sha256(path: Path) -> str`: read the parquet file in binary mode (`path.read_bytes()`), return `hashlib.sha256(...).hexdigest()`. This is the M2-mandated immutability link between the JSON report and the fixture binary.
    5. Function `_get_git_sha() -> str`: invoke `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=False)`. Return `result.stdout.strip()` if `result.returncode == 0` else `"unknown-not-in-git"`. This handles CI environments where git may not be available.
    6. Function `_calibrate_sc1(epex_df, hydro_df, cal) -> tuple[float, float, float]`: fit `sh_off = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, cal, hydro_df)` and `sh_on = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal, hydro_df)`. Locate key `("Ete", "Ouvrable")`; fall back to first common key if absent. Compute `ptp_off = float(np.ptp(sh_off.factors_[key]))`, `ptp_on = float(np.ptp(sh_on.factors_[key]))`, `ratio = ptp_on / ptp_off`. Return `(ptp_off, ptp_on, ratio)`.
    7. Function `main() -> None`:
       - Build fixture: `epex_df, hydro_df = build_bowl_fixture(seed=42)`.
       - Build calendar: `cal = enrich_15min_index(epex_df.index, country="CH")`.
       - Run SC #1 calibration: `(ptp_off, ptp_on, ratio) = _calibrate_sc1(epex_df, hydro_df, cal)`.
       - Compute threshold: `sc1_threshold = max(ratio - SC1_RATIO_MARGIN, SC1_FLOOR_MULTIPLIER)`.
       - Build report dict matching the M2-mandated schema:
         ```
         report = {
             "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "git_sha": _get_git_sha(),
             "fixture_sha256": _compute_fixture_sha256(FIXTURE_PATH),
             "fixture_path": "tests/fixtures/bowl_seed42.parquet",
             "ratios": {
                 "sc1_ptp_off": ptp_off,
                 "sc1_ptp_on": ptp_on,
                 "sc1_ptp_ratio": ratio,
                 "sc1_floor_multiplier": SC1_FLOOR_MULTIPLIER,
                 "sc1_ratio_margin": SC1_RATIO_MARGIN,
             },
             "thresholds_emitted": {
                 "SC1_PTP_THRESHOLD": sc1_threshold,
                 "SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER": 0.50,
             },
             "notes": "Plan 05C-01 ships Lever 1 only — sc1_ptp_ratio is the Lever-1-only gain. Plan 05C-03 will re-run this script with all 3 levers active (Plan 05C-03 Task 3); the updated artifact MUST overwrite this one and be re-committed.",
         }
         ```
       - Write to `REPORT_PATH` via `REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")`. The trailing newline + `sort_keys=True` ensures git-friendly stable diffs across re-runs.
       - Print a one-line confirmation: `print(f"Wave 0 calibrated: ratio={ratio:.4f}, threshold={sc1_threshold:.4f}, report={REPORT_PATH.relative_to(REPO_ROOT)}")`.
    8. `if __name__ == "__main__": main()` block.

    **Part B — Run the script ONCE and commit BOTH the script AND the report JSON:**

    Execute `python scripts/calibrate_bowl_thresholds.py`. Verify `tests/fixtures/_bowl_calibration_report.json` is created with valid JSON. Sanity-check by `python -c "import json; r = json.load(open('tests/fixtures/_bowl_calibration_report.json')); assert set(r) == {'calibrated_at', 'git_sha', 'fixture_sha256', 'fixture_path', 'ratios', 'thresholds_emitted', 'notes'}; assert 0.50 < r['thresholds_emitted']['SC1_PTP_THRESHOLD'] < 2.00; assert len(r['fixture_sha256']) == 64; print('OK report schema')"`.

    Sanity bounds (if outside, STOP and request investigation):
    - If `sc1_ptp_ratio < 1.00`: WARN — Lever 1 is REGRESSING amplitude vs flag=OFF. Possible bug in `_apply_hydro_analogue_weights`. STOP.
    - If `sc1_ptp_ratio > 3.00`: WARN — gain is implausibly large vs RESEARCH §Lever 1 analytic estimate 1.13-1.18. Verify fixture or kernel logic. STOP.
    - If `sc1_ratio` is below 1.20: threshold falls back to plancher 1.05 (acceptable per RESEARCH §Lever 1). Continue.

    DO NOT create `tests/test_shape_hourly_bowl.py` in this task — that is Task 5. This task ONLY produces the script + JSON artifact.

    DO NOT inline the script body inside a heredoc — use Write tool (per CRITICAL rules in planner).
  </action>
  <verify>
    <automated>test -f scripts/calibrate_bowl_thresholds.py &amp;&amp; python scripts/calibrate_bowl_thresholds.py &amp;&amp; python -c "
import json, hashlib
r = json.load(open('tests/fixtures/_bowl_calibration_report.json'))
expected_keys = {'calibrated_at', 'git_sha', 'fixture_sha256', 'fixture_path', 'ratios', 'thresholds_emitted', 'notes'}
assert set(r) == expected_keys, f'schema drift: {set(r) ^ expected_keys}'
assert len(r['fixture_sha256']) == 64, f'sha256 len {len(r[\"fixture_sha256\"])}'
fixture_bytes = open('tests/fixtures/bowl_seed42.parquet', 'rb').read()
actual_sha = hashlib.sha256(fixture_bytes).hexdigest()
assert r['fixture_sha256'] == actual_sha, f'fixture sha mismatch: report={r[\"fixture_sha256\"][:16]}... actual={actual_sha[:16]}...'
thr = r['thresholds_emitted']['SC1_PTP_THRESHOLD']
assert 0.50 &lt; thr &lt; 2.00, f'threshold {thr} out of plausible bounds'
print(f'OK report: ratio={r[\"ratios\"][\"sc1_ptp_ratio\"]:.4f} thr={thr:.4f} sha256={r[\"fixture_sha256\"][:8]}...')
"</automated>
  </verify>
  <acceptance_criteria>
    - `test -f scripts/calibrate_bowl_thresholds.py` exits 0.
    - `test -f tests/fixtures/_bowl_calibration_report.json` exits 0.
    - `grep -q "M2" scripts/calibrate_bowl_thresholds.py` exits 0 (traceability to REVIEWS.md fix).
    - `grep -q "REVIEWS.md" scripts/calibrate_bowl_thresholds.py` exits 0.
    - `grep -q "_compute_fixture_sha256\|_get_git_sha\|_calibrate_sc1" scripts/calibrate_bowl_thresholds.py` exits 0 (three required helper functions present).
    - `python -c "import json; r = json.load(open('tests/fixtures/_bowl_calibration_report.json')); assert {'calibrated_at', 'git_sha', 'fixture_sha256', 'fixture_path', 'ratios', 'thresholds_emitted', 'notes'} == set(r), set(r)"` exits 0 (M2-mandated schema present and complete).
    - The verify command above prints `OK report: ratio=... thr=... sha256=...` and exits 0.
    - `python -c "import json; r = json.load(open('tests/fixtures/_bowl_calibration_report.json')); t = r['thresholds_emitted']['SC1_PTP_THRESHOLD']; assert 0.50 < t < 2.00, t"` exits 0 (threshold in plausible bounds).
    - Re-running `python scripts/calibrate_bowl_thresholds.py` produces a JSON whose `fixture_sha256` and `ratios.sc1_ptp_ratio` match the previous run (only `calibrated_at` and possibly `git_sha` may change between runs); confirmed via `python -c "import json; old = json.load(open('tests/fixtures/_bowl_calibration_report.json')); ..."` before and after.
    - `pytest tests/ -x -q` exits 0 reporting `247 passed, 4 skipped` (still no new test consumer — Task 5 wires it in).
  </acceptance_criteria>
  <done>Reproducible calibration script + immutable JSON sidecar committed. M2 cross-AI review fix landed: a secondary quant reviewer can re-run the script and observe whether the artifact changes; the JSON carries `fixture_sha256` for tamper detection (the matching test_calibration_report_matches_fixture in Plan 05C-03 closes the loop).</done>
</task>

<task type="auto">
  <name>Task 5: Create `tests/test_shape_hourly_bowl.py` with thresholds loaded from _bowl_calibration_report.json, D-A4-3 (kernel test), D-A4-8 (flag=OFF baseline)</name>
  <files>tests/test_shape_hourly_bowl.py</files>
  <read_first>
    - tests/fixtures/_generate_bowl_fixture.py (Task 1 output — exposes `build_bowl_fixture()`)
    - tests/fixtures/_bowl_calibration_report.json (Task 4 output — source of SC1_PTP_THRESHOLD via json.load)
    - scripts/calibrate_bowl_thresholds.py (Task 4 output — re-run instructions if threshold needs refresh)
    - tests/fixtures/_generate_baseline.py (5bis-A — `build_pfc(seed, flag)` reusable entry point used by D-A4-8 to regenerate the OFF baseline)
    - tests/fixtures/baseline_pfc_seed42.parquet (5bis-A frozen reference for D-A4-8)
    - pfc_shaping/lt/model/shape_hourly.py (post-Task-2 state — confirms `_apply_hydro_analogue_weights` branch and ctor signature)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-A4-2, D-A4-3, D-A4-8 (test specs)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-RESEARCH.md §Validation Architecture, §Pitfall 5 (fixture-real gap docstring)
    - .planning/phases/05C-shape-hourly-bowl-deepening/05C-REVIEWS.md (M2 — threshold MUST flow from JSON artifact, not in-comment value)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md §1-2 (tolerance contract addendum atol=1e-12 rtol=0 + identical columns/dtypes/index/sort order)
  </read_first>
  <action>
    Create `tests/test_shape_hourly_bowl.py` — a new test module isolated from `test_shape_hourly_infra.py` per D-A4-2.

    **Module docstring:** Reference Phase 5bis-B, the 7 tests planned (this plan delivers 2 of 7; Plans 05C-02 and 05C-03 deliver the other 5), the no-op contract from 5bis-A REVIEWS.md (`atol=1e-12, rtol=0` + identical columns/dtypes/index/sort order), and the convention that flag=OFF preserves 5bis-A baseline bit-pour-bit per D-A1-2 / SC #4. Document the fixture-real gap explicitly per RESEARCH Pitfall 5. State explicitly that threshold constants are loaded from `tests/fixtures/_bowl_calibration_report.json` (committed by Plan 05C-01 Task 4, re-calibrated by Plan 05C-03 Task 3) — to refresh, re-run `python scripts/calibrate_bowl_thresholds.py`.

    **Threshold loading block (M2 cross-AI review fix — REPLACES the original in-comment SC1_PTP_THRESHOLD = X pattern):**

    At the top of the module, AFTER imports and BEFORE the test definitions, load the calibration report via `json.load`:

    ```
    _CALIBRATION_REPORT_PATH = Path(__file__).parent / "fixtures" / "_bowl_calibration_report.json"
    _calibration_report = json.loads(_CALIBRATION_REPORT_PATH.read_text())

    # Thresholds derived from committed _bowl_calibration_report.json (M2 cross-AI review fix —
    # REVIEWS.md consensus #3). To refresh: `python scripts/calibrate_bowl_thresholds.py`.
    # The matching test_calibration_report_matches_fixture in Plan 05C-03 enforces that
    # the report's fixture_sha256 matches the actual fixture bytes — if the fixture changes
    # without re-running calibration, CI fails loudly.
    SC1_PTP_THRESHOLD: float = _calibration_report["thresholds_emitted"]["SC1_PTP_THRESHOLD"]
    SC3_M30_AMPLITUDE_THRESHOLD: float = _calibration_report["thresholds_emitted"]["SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER"]  # PLACEHOLDER 0.50 — Plan 05C-02 Task 3 overwrites this key in the report
    ```

    The constants `SC1_PTP_THRESHOLD` and `SC3_M30_AMPLITUDE_THRESHOLD` are now module-level Python floats sourced from the JSON. Downstream tests in Plans 05C-02 and 05C-03 can import them without changes. Plan 05C-02 Task 3 updates the JSON's `SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER` key (renaming and overwriting); Plan 05C-03 Task 3 re-runs the calibration script after all 3 levers ship and overwrites `SC1_PTP_THRESHOLD`.

    **Required imports at file top:**
    `from __future__ import annotations`, `json`, `pytest`, `numpy as np`, `pandas as pd`, `from pathlib import Path`, `from pandas.testing import assert_frame_equal`, and the relevant symbols from `pfc_shaping`: `ShapeHourly`, `enrich_15min_index`. Also import the reusable entry points: `from tests.fixtures._generate_bowl_fixture import build_bowl_fixture` and `from tests.fixtures._generate_baseline import build_pfc as build_baseline_pfc`.

    **Module-level constants (in addition to the JSON-loaded thresholds):** `_FIXTURE_DIR = Path(__file__).parent / "fixtures"`, `_BASELINE_5BISA = _FIXTURE_DIR / "baseline_pfc_seed42.parquet"`, `_BASELINE_BOWL = _FIXTURE_DIR / "baseline_pfc_seed42_bowl.parquet"  # generated by Plan 05C-03`.

    **Test 1 — `test_hydro_kernel_uses_per_timestamp_climatological_target` (D-A4-3):**

    Docstring: cite D-A4-3 + D-A1-1. Reference RESEARCH §Implementation Pitfalls 1 (nearest-neighbor `get_climatological_fill` must be used, not direct dict access).

    Behavior to assert (direct verification of D-A1-1):
    1. Build `epex_df, hydro_df = build_bowl_fixture(seed=42)`.
    2. Build calendar via `cal = enrich_15min_index(epex_df.index, country="CH")`.
    3. Fit `sh_on = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal, hydro_df)` and `sh_off = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, cal, hydro_df)`.
    4. Reach into the implementation to verify the kernel target. Two approaches; pick whichever survives Task 2's implementation choice:
       - **Option A (preferred): refactor `_apply_hydro_analogue_weights` in Task 2 to set `self._last_clim_target_` (a private debug attribute, dict `{"flag_on": clim_target_array, "flag_off": current_fill_scalar}`) at the end of the method.** Then the test asserts `np.allclose(sh_on._last_clim_target_, [sh_on.get_climatological_fill(w) for w in expected_woy_arr])` where `expected_woy_arr` is computed from a known subset of `epex_df.index` (e.g. first 96 timestamps). And asserts `sh_off._last_clim_target_ == float(hydro_df["fill_pct"].iloc[-1] / 100)` (legacy scalar).
       - **Option B (fallback if private attr feels intrusive): instrument via `caplog` — Task 2 logs the kernel mean target at INFO level, the test asserts the logged `mean(clim_target) * 100` value matches the expected weighted mean of `_climatological_fill`.**
       Pick Option A. If implementing Option A, AMEND Task 2's `<action>` mentally by adding the single-line `self._last_clim_target_ = clim_target` at the end of the kernel computation branch (both flag=ON sets the array, flag=OFF sets the scalar). Document this private attribute with a comment `# Private debug attribute — populated by _apply_hydro_analogue_weights for test verification (D-A4-3). NOT part of the public API, NOT persisted.`
    5. Assert that for flag=ON, `sh_on._last_clim_target_` is a NumPy array of length equal to the EPEX timestamps consumed by the kernel (after the date-range alignment step at lines 883-889), and that each entry `clim_target[i]` equals `sh_on.get_climatological_fill(woy(epex_df.index[i_kernel]))` where `i_kernel` is the index into the kernel-consumed slice. Use `numpy.testing.assert_allclose(clim_target, expected, atol=1e-12, rtol=0)`.
    6. Assert that for flag=OFF, `sh_off._last_clim_target_` is a scalar float equal to `float(hydro_df["fill_pct"].iloc[-1] / 100.0)` (the legacy `current_fill` after the normalize-to-[0,1] step at lines 859-860). Tolerance `atol=1e-12, rtol=0`.

    **Test 2 — `test_flag_off_bit_for_bit_baseline` (D-A4-8 / SC #4):**

    Docstring: cite D-A4-8 + SC #4 + 5bis-A REVIEWS.md §1 tolerance contract. Note this test extends 5bis-A's `test_baseline_regression[False]` to the 5bis-B refactored kernel surface — the no-op contract is preserved by the flag=OFF branch in `_apply_hydro_analogue_weights`.

    Behavior:
    1. `df_off = build_baseline_pfc(seed=42, flag=False)` — uses the 5bis-A reusable entry point from `tests/fixtures/_generate_baseline.py`.
    2. `baseline = pd.read_parquet(_BASELINE_5BISA)`.
    3. Strict column/dtype/index identity:
       - `assert list(df_off.columns) == list(baseline.columns)`
       - `assert df_off.dtypes.to_dict() == baseline.dtypes.to_dict()`
       - `assert df_off.index.equals(baseline.index)`
    4. Numerical equality at the 5bis-A REVIEWS contract:
       - `assert_frame_equal(df_off, baseline, check_exact=False, atol=1e-12, rtol=0)`
    5. CI-drift fallback policy: the test starts at `atol=1e-12, rtol=0`. If pandas/pyarrow patch-level drift breaks the assertion in CI, the fallback to `atol=1e-10` is permitted with an inline `# CI-drift fallback: ...` comment per 5bis-A Plan 05B-05 Task 3 contract. Default remains `atol=1e-12, rtol=0`.

    All tests must:
    - Use `pytest.fixture` scope="module" for the expensive `build_bowl_fixture()` call so it runs once per module.
    - Inherit the autouse env-var hygiene fixture from `tests/conftest.py` (D-12, 5bis-A) — no explicit import needed.
    - Be ≤ 80 lines each.
    - Reference their decision IDs in the docstring (D-A4-3, D-A4-8) for traceability.
  </action>
  <verify>
    <automated>pytest tests/test_shape_hourly_bowl.py -v 2>&amp;1 | tail -10 &amp;&amp; pytest tests/ 2>&amp;1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `test -f tests/test_shape_hourly_bowl.py` exits 0.
    - `grep -q "SC1_PTP_THRESHOLD" tests/test_shape_hourly_bowl.py` exits 0.
    - `grep -q "SC3_M30_AMPLITUDE_THRESHOLD" tests/test_shape_hourly_bowl.py` exits 0 (placeholder loaded from JSON; Plan 05C-02 Task 3 updates the JSON value).
    - `grep -q "_bowl_calibration_report" tests/test_shape_hourly_bowl.py` exits 0 (M2 fix — threshold sourced from JSON, NOT in-comment).
    - `grep -q "json.loads\|json.load" tests/test_shape_hourly_bowl.py` exits 0 (M2 fix — explicit load call).
    - `python -c "import json, ast; src = open('tests/test_shape_hourly_bowl.py').read(); assert '_calibration_report' in src and 'thresholds_emitted' in src, 'M2 wiring incomplete'; print('OK M2 wiring')"` exits 0.
    - `grep -q "D-A4-3\|D-A4-8" tests/test_shape_hourly_bowl.py` exits 0.
    - `grep -q "atol=1e-12" tests/test_shape_hourly_bowl.py` exits 0 (tolerance contract).
    - `grep -q "build_bowl_fixture\|build_baseline_pfc" tests/test_shape_hourly_bowl.py` exits 0 (reusable entry points imported).
    - `grep -q "Plan 05C-02\|Plan 05C-03" tests/test_shape_hourly_bowl.py` exits 0 (cross-plan reference for the threshold-refresh expectation).
    - `pytest tests/test_shape_hourly_bowl.py::test_hydro_kernel_uses_per_timestamp_climatological_target -x` exits 0.
    - `pytest tests/test_shape_hourly_bowl.py::test_flag_off_bit_for_bit_baseline -x` exits 0.
    - `pytest tests/ -x -q` exits 0 reporting `249 passed, 4 skipped` (247 baseline + 2 new tests). If the count is 248 (one test parametrized differently), document the actual count in the SUMMARY.
    - `pytest tests/test_shape_hourly_bowl.py -v 2>&amp;1 | grep -c PASSED` reports ≥ 2.
  </acceptance_criteria>
  <done>New isolated test module created. Threshold constants loaded from committed _bowl_calibration_report.json (M2 fix). Two passing tests cover D-A4-3 (kernel) and D-A4-8 (flag=OFF baseline). Full suite green at 249 passed.</done>
</task>

</tasks>

<verification>
- `pytest tests/ -x -q` exits 0 reporting `249 passed, 4 skipped`.
- `pytest tests/ --co -q | tail -1` reports `>= 255 tests collected` (the parametrized 5bis-A `test_baseline_regression[False|True]` counts as 2).
- `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; sh = ShapeHourly(); assert sh._hydro_weight_sigma_off == 0.25; assert sh._hydro_weight_sigma_on == 0.08; assert sh.hydro_weight_sigma == 0.25; print('OK')"` prints `OK`.
- `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; sh = ShapeHourly(hydro_weight_sigma=0.7); assert sh._hydro_weight_sigma_off == 0.7 and sh._hydro_weight_sigma_on == 0.7; print('OK legacy')"` prints `OK legacy`.
- `python tests/fixtures/_generate_bowl_fixture.py` is idempotent: re-running produces a byte-identical `bowl_seed42.parquet` (sha256 stable).
- `python scripts/calibrate_bowl_thresholds.py` re-runs cleanly; the resulting JSON's `fixture_sha256` equals `sha256(tests/fixtures/bowl_seed42.parquet)` (M2 invariant — closed by the test added in Plan 05C-03).
- `test_baseline_regression[False]` from 5bis-A `tests/test_shape_hourly_infra.py` continues to pass against `tests/fixtures/baseline_pfc_seed42.parquet` at `atol=1e-12, rtol=0`.
- `git diff --stat tests/test_shape_hourly_infra.py` shows ONLY the two surgical edits to `test_hyperparams_row` + `test_save_unfitted_hyperparams_correct` (and docstring updates). No other test modified.
- `git log --oneline tests/test_shape_hourly_bowl.py | wc -l` reports 1 (single commit introducing the new file).
- `git ls-files scripts/calibrate_bowl_thresholds.py tests/fixtures/_bowl_calibration_report.json | wc -l` reports 2 (both M2 artifacts committed).
</verification>

<success_criteria>
- **Lever 1 math change shipped:** `_apply_hydro_analogue_weights` kernel target switches to per-timestamp `get_climatological_fill(woy(t))` when `self._use_seasonal_hourly is True`, preserves legacy `current_fill` scalar when False.
- **Backward-compat preserved bit-pour-bit:** `test_baseline_regression[False]` from 5bis-A still passes at `atol=1e-12, rtol=0`. `test_flag_off_bit_for_bit_baseline` (D-A4-8, this plan) passes at the same contract.
- **Ctor extension safe:** All four legacy callsites (`autoresearch.py:234`, `rolling_update.py:365`, `test_shape_hourly_infra.py:239,250,628`) continue to work with no warning. New `hydro_weight_sigma_off/_on` kwargs available for explicit usage.
- **Sidecar schema extended:** `shape_hourly.meta.parquet` hyperparams JSON includes the 3 new keys (`hydro_weight_sigma_off`, `hydro_weight_sigma_on`, `hydro_weight_sigma_resolved`) + legacy `hydro_weight_sigma` preserved for 5bis-A reader compat. `load()` cross-plan fallback handles pre-5bis-B sidecars.
- **Bowl fixture committed:** `tests/fixtures/bowl_seed42.parquet` (~50KB, seed=42, analytically-controlled duck curve) + reusable `build_bowl_fixture()` entry point.
- **Auditable calibration (M2 cross-AI review fix):** `scripts/calibrate_bowl_thresholds.py` is a committed reproducible script; `tests/fixtures/_bowl_calibration_report.json` is the committed immutable artifact (schema: calibrated_at, git_sha, fixture_sha256, ratios, thresholds_emitted). Threshold flows from JSON to test via `json.load`, not from a free-floating in-comment value.
- **New test module scaffolded:** `tests/test_shape_hourly_bowl.py` with 2 passing tests (D-A4-3 kernel, D-A4-8 baseline) + Wave 0 calibrated `SC1_PTP_THRESHOLD` loaded from the JSON sidecar.
- **Single authorized infra touch:** `test_hyperparams_row` and `test_save_unfitted_hyperparams_correct` in `tests/test_shape_hourly_infra.py` updated for the extended JSON schema. All other 5bis-A infra tests untouched and green.
- **Test count: 247 → 249** (4 skipped preserved).
- **T1 deferred-research item acknowledged:** real Swiss hydro anomaly diagnostic justifying `hydro_weight_sigma_on=0.08` is tracked in `<deferred_research>`. Does not block 5bis-B ship.
</success_criteria>

<output>
Create `.planning/phases/05C-shape-hourly-bowl-deepening/05C-01-SUMMARY.md` when done.
</output>
</content>
</invoke>