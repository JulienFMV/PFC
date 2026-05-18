---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/fixtures/_generate_baseline.py
  - tests/fixtures/baseline_pfc_seed42.parquet
  - tests/fixtures/README.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "A reproducible script produces tests/fixtures/baseline_pfc_seed42.parquet from the current code state (== main@28dfd65 for pfc_shaping/* code per `git diff --stat 28dfd65..HEAD -- pfc_shaping/ tests/` = empty)."
    - "Re-running tests/fixtures/_generate_baseline.py on the same git SHA produces a parquet with `pandas.testing.assert_frame_equal(reread, baseline, atol=0)` (byte-equivalent on numeric columns at default tolerance)."
    - "The baseline parquet is committed in a SEPARATE commit AHEAD of any 5bis-A logic commits (Plans 02-05)."
    - "The script uses only synthetic fixtures (no dependency on data/*.xlsx, no dependency on H:\\ HFC OMPEX dir)."
  artifacts:
    - path: "tests/fixtures/_generate_baseline.py"
      provides: "Deterministic baseline generator (seed=42, synthetic forwards, Cal'27 1-month horizon)"
      min_lines: 60
    - path: "tests/fixtures/baseline_pfc_seed42.parquet"
      provides: "Frozen PFC build reference for bit-for-bit regression testing"
      contains: "columns price_shape, f_S, f_W, f_H, f_Q, f_WV"
    - path: "tests/fixtures/README.md"
      provides: "Reproducibility instructions + git SHA pin"
      contains: "SHA"
  key_links:
    - from: "tests/fixtures/_generate_baseline.py"
      to: "pfc_shaping.lt.model.assembler.PFCAssembler.build"
      via: "import + call with seed=42"
      pattern: "PFCAssembler.*build"
    - from: "tests/fixtures/_generate_baseline.py"
      to: "pfc_shaping.lt.model.shape_hourly.ShapeHourly"
      via: "fit on synthetic EPEX df (deterministic via numpy seed=42)"
      pattern: "ShapeHourly\\(\\).fit"
---

<objective>
Generate and commit the frozen baseline snapshot `tests/fixtures/baseline_pfc_seed42.parquet` that serves as the bit-for-bit regression reference for the entire Phase 5bis-A no-op refactor and all subsequent shape phases (5bis-B, 5, etc.).

Purpose: Without a frozen baseline produced from the CURRENT code state (HEAD ≡ main@28dfd65 for pfc_shaping/* per `git diff --stat 28dfd65..HEAD -- pfc_shaping/ tests/` = empty), the "no-op refactor" claim of Phase 5bis-A is unfalsifiable. This plan is intentionally a SEPARATE commit AHEAD of any logic change so the baseline is unambiguously sourced from pre-refactor code.

Output: `tests/fixtures/_generate_baseline.py` (reproducible generator), `tests/fixtures/baseline_pfc_seed42.parquet` (frozen reference), and `tests/fixtures/README.md` (reproducibility instructions + pinned git SHA).
</objective>

<execution_context>
@.claude/get-shit-done/workflows/execute-plan.md
@.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md
@pfc_shaping/lt/model/assembler.py
@pfc_shaping/lt/model/shape_hourly.py
@pfc_shaping/data/calendar_ch.py

<interfaces>
Key signatures the generator MUST call (extracted from current HEAD == main@28dfd65 for code):

From pfc_shaping/lt/model/shape_hourly.py:
```
class ShapeHourly:
    def __init__(self, sigma: float = 0.5, halflife_days: float = 180.0, hydro_weight_sigma: float = 0.25) -> None: ...
    def fit(self, epex_df: pd.DataFrame, calendar_df: pd.DataFrame, hydro_df: pd.DataFrame | None = None) -> "ShapeHourly": ...
    def apply(self, timestamps: pd.DatetimeIndex, calendar_df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.Series: ...
```

From pfc_shaping/lt/model/assembler.py:
```
class PFCAssembler:
    def __init__(self, shape_hourly: ShapeHourly, ...) -> None: ...
    def build(self, base_prices: dict, quoted_keys: set[str] | None = None, start_date: str | None = None, horizon_days: int = HORIZON_DAYS, entso_forecast: pd.DataFrame | None = None, hydro_forecast: pd.DataFrame | None = None, outages_forecast: pd.DataFrame | None = None, reference_date: pd.Timestamp | None = None, country: str = "CH") -> pd.DataFrame: ...
# Returns DataFrame columns ['price_shape', 'f_S', 'f_W', 'f_H', 'f_Q', 'f_WV', 'profile_type', 'confidence', 'p10', 'p90', 'calibrated']
```

From pfc_shaping/data/calendar_ch.py:
```
def enrich_15min_index(idx: pd.DatetimeIndex, country: str = "CH") -> pd.DataFrame: ...
```

Existing synthetic test fabrication pattern: see tests/test_country_tz_plumbing.py (does NOT need real EPEX parquet; builds calendar_df + epex_df from synthetic timestamps).
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write deterministic baseline generator script</name>
  <files>tests/fixtures/_generate_baseline.py</files>
  <read_first>
    - pfc_shaping/lt/model/assembler.py (lines 195-260 for `build()` signature + dependencies — what `base_prices` dict shape and `entso_forecast`/`hydro_forecast` shapes are accepted, what `country="CH"` triggers)
    - pfc_shaping/lt/model/shape_hourly.py (lines 55-165 for `__init__` defaults sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25, and `fit()` requirements on `epex_df`/`calendar_df` columns)
    - pfc_shaping/data/calendar_ch.py (the `enrich_15min_index(idx, country)` helper used to build the calendar DataFrame the synthetic EPEX must merge with)
    - tests/test_country_tz_plumbing.py (full file — exemplifies the synthetic fixture-building pattern; do NOT touch real data/*.xlsx)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-10, D-11 — exact specification of seed, horizon, deterministic fixture)
  </read_first>
  <action>
    Create `tests/fixtures/_generate_baseline.py` as a standalone runnable Python script (CLI: `python tests/fixtures/_generate_baseline.py`).

    The script MUST be 100% deterministic and rely ONLY on synthetic inputs.

    Concrete spec (per D-10):
    - Set `numpy.random.seed(42)` and `random.seed(42)` at top of `main()`.
    - Build a synthetic 15min EPEX DataFrame covering 3 calendar years (e.g. 2022-01-01..2024-12-31 UTC) with column `price_eur_mwh`: use a deterministic structural model — e.g. `30 + 10*sin(2π·hour/24) + 5*sin(2π·dayofyear/365) + np.random.normal(0, 2, n)` clipped to [-50, 200]. Index UTC, freq='15min'.
    - Build `calendar_df = enrich_15min_index(epex_df.index, country="CH")`.
    - Fit `sh = ShapeHourly().fit(epex_df, calendar_df)` (no hydro_df — keeps `_climatological_fill = None`).
    - Build `PFCAssembler(shape_hourly=sh, shape_intraday=None, water_value=None, calibrator=None, cascader=None)` — pass `None` for any optional component the assembler accepts. If the assembler constructor REQUIRES an instance (not None), instantiate minimal stubs from existing concrete classes with their default `__init__()`.
    - Call `df_pfc = assembler.build(base_prices={"2027": 80.0}, start_date="2027-01-01", horizon_days=31, reference_date=pd.Timestamp("2026-05-18", tz="UTC"), country="CH")`.
    - Write to parquet: `df_pfc.to_parquet("tests/fixtures/baseline_pfc_seed42.parquet", index=True)`.
    - Emit a final `print(f"Wrote baseline: rows={len(df_pfc)} cols={list(df_pfc.columns)} hash_price_shape={hash(df_pfc['price_shape'].values.tobytes())}")` so re-runs can be eyeballed for stability.

    Module docstring MUST include:
    - The git SHA the script is intended to be run against (placeholder `HEAD == 28dfd65 for pfc_shaping/* per `git diff --stat 28dfd65..HEAD -- pfc_shaping/ tests/` = empty`).
    - The exact CLI to regenerate: `python tests/fixtures/_generate_baseline.py`.
    - A warning that this file is INPUT-ONLY for the regression test in plan 05B-05; modifying behavior of pfc_shaping requires explicitly regenerating + committing both files in a single PR with justification.

    If `PFCAssembler.__init__` requires components that cannot be safely instantiated as stubs (e.g. needs a fitted `ShapeIntraday`), fit minimal versions on the same synthetic EPEX using their default ctor. Document any such workaround in the script's module docstring. NEVER call into real data files. NEVER reach for `H:\` or `data/*.xlsx`.
  </action>
  <verify>
    <automated>python tests/fixtures/_generate_baseline.py 2>&1 | grep -q "Wrote baseline" && test -f tests/fixtures/baseline_pfc_seed42.parquet</automated>
  </verify>
  <acceptance_criteria>
    - `test -f tests/fixtures/_generate_baseline.py` exits 0.
    - `python tests/fixtures/_generate_baseline.py` exits 0 (no exception).
    - `python tests/fixtures/_generate_baseline.py` printed line contains substring `Wrote baseline: rows=`.
    - `python -c "import pandas as pd; df=pd.read_parquet('tests/fixtures/baseline_pfc_seed42.parquet'); assert {'price_shape','f_S','f_W','f_H','f_Q','f_WV'}.issubset(df.columns), df.columns.tolist()"` exits 0.
    - Re-running the script twice in a row produces identical parquet contents: `python tests/fixtures/_generate_baseline.py && cp tests/fixtures/baseline_pfc_seed42.parquet /tmp/baseline_a.parquet && python tests/fixtures/_generate_baseline.py && python -c "import pandas as pd; a=pd.read_parquet('/tmp/baseline_a.parquet'); b=pd.read_parquet('tests/fixtures/baseline_pfc_seed42.parquet'); pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)"` exits 0.
    - `grep -L "data/.*\\.xlsx\|H:\\\\" tests/fixtures/_generate_baseline.py` exits 0 (no real-data path references).
  </acceptance_criteria>
  <done>Script committed, runs deterministically, no real-data dependency.</done>
</task>

<task type="auto">
  <name>Task 2: Generate baseline parquet + write README with pinned SHA</name>
  <files>tests/fixtures/baseline_pfc_seed42.parquet, tests/fixtures/README.md</files>
  <read_first>
    - tests/fixtures/_generate_baseline.py (the script just produced in Task 1)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-11 — test contract `assert_frame_equal(build(flag=OFF), baseline, atol=1e-10)`)
  </read_first>
  <action>
    Execute `python tests/fixtures/_generate_baseline.py` once to produce `tests/fixtures/baseline_pfc_seed42.parquet`. Then write `tests/fixtures/README.md` documenting:

    1. **Purpose**: frozen reference for `test_baseline_regression` (added in plan 05B-05) — `assert_frame_equal(build(flag=OFF), baseline, atol=1e-10)`.
    2. **Source SHA**: the git SHA of HEAD at the time of generation (insert via `git rev-parse HEAD` output — must equal `e8a3012` or descendant on branch `claude/clean-lt-ct-integration`; `git diff --stat 28dfd65..HEAD -- pfc_shaping/ tests/` MUST be empty, confirming code equivalence to `main@28dfd65`).
    3. **Regeneration policy**: this fixture MUST NOT be regenerated lightly. Any PR that modifies its contents must include a justification block in the PR description and bump a corresponding annotation in this README.
    4. **Regeneration command**: `python tests/fixtures/_generate_baseline.py`.
    5. **Schema**: list the parquet columns observed (output of `pd.read_parquet(...).columns.tolist()` and `.dtypes`).

    Do NOT use the `Bash(cat << EOF)` heredoc pattern — use the Write tool only.

    Stage and commit these THREE files (`_generate_baseline.py`, `baseline_pfc_seed42.parquet`, `README.md`) as a SINGLE commit BEFORE any code changes from plans 05B-02 through 05B-05. Commit message: `test(05B-01): freeze baseline_pfc_seed42 from main@28dfd65 for no-op refactor regression`. This commit must precede all subsequent 5bis-A commits in `git log`.
  </action>
  <verify>
    <automated>test -f tests/fixtures/baseline_pfc_seed42.parquet && test -f tests/fixtures/README.md && grep -q "SHA" tests/fixtures/README.md && grep -q "_generate_baseline.py" tests/fixtures/README.md</automated>
  </verify>
  <acceptance_criteria>
    - `git log --oneline -1 tests/fixtures/baseline_pfc_seed42.parquet | grep -q "05B-01"` exits 0 (file is in a 05B-01 commit).
    - `git log --oneline tests/fixtures/baseline_pfc_seed42.parquet | wc -l` reports exactly 1 (file is introduced in a single commit, never amended).
    - `python -c "import pandas as pd; df=pd.read_parquet('tests/fixtures/baseline_pfc_seed42.parquet'); print(df.shape, df.columns.tolist())"` exits 0 and prints a row count > 2000 (≥ 31 days × 96 quarters/day).
    - `grep -q "main@28dfd65\|28dfd65" tests/fixtures/README.md` exits 0.
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped` (or unchanged from baseline — Plan 01 adds NO test code, only fixtures).
  </acceptance_criteria>
  <done>Baseline parquet + README committed as a single isolated commit ahead of the rest of 5bis-A.</done>
</task>

</tasks>

<verification>
After completion:
- `git log --oneline tests/fixtures/baseline_pfc_seed42.parquet` shows exactly one entry, before any commit touching `pfc_shaping/lt/model/shape_hourly.py` for plans 05B-02..05B-05.
- `pytest tests/ -x` exits 0 with `142 passed, 4 skipped`.
- The script can be re-run to verify byte-equivalence (idempotency check).
</verification>

<success_criteria>
- `tests/fixtures/_generate_baseline.py` is deterministic, synthetic-only, < 200 lines.
- `tests/fixtures/baseline_pfc_seed42.parquet` exists with PFC columns (`price_shape`, `f_S`, `f_W`, `f_H`, `f_Q`, `f_WV` at minimum).
- `tests/fixtures/README.md` documents source SHA + regeneration policy.
- Three files committed in one atomic commit; subsequent plan commits build ON this commit.
- Suite remains green (142 passed, 4 skipped — no test code added yet).
</success_criteria>

<output>
Create `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-01-SUMMARY.md` when done.
</output>
