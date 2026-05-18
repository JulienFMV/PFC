---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: 05
type: execute
wave: 5
depends_on:
  - 05B-01
  - 05B-02
  - 05B-03
  - 05B-04
files_modified:
  - tests/conftest.py
  - tests/test_shape_hourly_infra.py
  - tests/fixtures/shape_hourly_legacy.parquet
  - tests/fixtures/f_W_legacy.parquet
autonomous: true
requirements:
  - SHP-01
  - SHP-04
must_haves:
  truths:
    - "`tests/conftest.py` autouse fixture snapshots all `PFC_LT_*` environment variable keys before each test and restores them after, preventing test→test env-var leak (per D-12)."
    - "`test_factors_3d_view_consistency` asserts `factors_3d_[(s,tj,h)] == factors_[(s,tj)][h]` over all populated cells × 24 hours, and that mutation raises `TypeError`."
    - "`test_save_load_full_roundtrip` asserts that `factors_`, `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, `sigma`, `halflife_days`, `hydro_weight_sigma`, `_use_seasonal_hourly` all roundtrip identically through `save → load`."
    - "`test_save_load_legacy_compat` loads a fixture parquet committed in this plan (`tests/fixtures/shape_hourly_legacy.parquet` + `tests/fixtures/f_W_legacy.parquet` without a `_meta.parquet` sidecar) without crash, asserts one `logger.warning` was emitted, and asserts `factors_`/`f_W_` are populated."
    - "`test_flag_freeze_at_init` asserts post-construction mutations of `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']` do not change `sh._use_seasonal_hourly`, in both directions (`True→False`, `False→True`)."
    - "`test_flag_persisted_in_parquet` asserts a fit-save with `use_seasonal_hourly=True`, then load with env var set to `'0'` yields `sh._use_seasonal_hourly is True` (parquet wins)."
    - "`test_baseline_regression` is parametrized over `flag in (False, True)`. For each flag value, it builds `assembler.build(...)` with the same synthetic inputs used to generate `tests/fixtures/baseline_pfc_seed42.parquet` and asserts `pandas.testing.assert_frame_equal(df, baseline, check_exact=False, atol=1e-10)`. This is THE proof that Phase 5bis-A is a numerical no-op."
    - "All 142 pre-existing tests + the new tests added in this plan pass; total test count rises to `>= 142 + N` where N is the new test count, with 4 skipped CT-only baseline preserved."
  artifacts:
    - path: "tests/conftest.py"
      provides: "Autouse env-var hygiene fixture for `PFC_LT_*` keys"
      contains: "PFC_LT_"
    - path: "tests/test_shape_hourly_infra.py"
      provides: "Six new tests covering view, save/load, legacy compat, flag freeze, flag persistence, baseline regression"
      contains: "test_baseline_regression"
    - path: "tests/fixtures/shape_hourly_legacy.parquet"
      provides: "Binary fixture: a legacy parquet emitted by pre-5bis-A code (no _meta.parquet sidecar)"
    - path: "tests/fixtures/f_W_legacy.parquet"
      provides: "Binary fixture: legacy f_W sidecar paired with the legacy shape_hourly parquet"
  key_links:
    - from: "tests/test_shape_hourly_infra.py::test_baseline_regression"
      to: "tests/fixtures/baseline_pfc_seed42.parquet"
      via: "pandas.read_parquet + assert_frame_equal(atol=1e-10)"
      pattern: "baseline_pfc_seed42"
    - from: "tests/conftest.py::_pfc_lt_env_hygiene"
      to: "os.environ"
      via: "autouse fixture, save/restore PFC_LT_* keys"
      pattern: "PFC_LT_"
---

<objective>
Add the missing test infrastructure for Phase 5bis-A:

1. **`tests/conftest.py`** with an autouse fixture that snapshots and restores all `PFC_LT_*` env vars per test (per D-12).
2. **`tests/test_shape_hourly_infra.py`** with the six tests specified in D-14..D-19, plus parametrization over both flag states for the baseline regression test.
3. **Legacy binary fixtures** (`tests/fixtures/shape_hourly_legacy.parquet` + `tests/fixtures/f_W_legacy.parquet`) emitted by pre-5bis-A code shape so `test_save_load_legacy_compat` can validate the warning-fallback path on a real legacy artefact.

Purpose: This plan is THE assertion layer that proves 5bis-A is a numerical no-op vs the baseline (frozen in plan 05B-01) and that the new infrastructure (save/load, flag, view) works as specified. Without these tests, plans 05B-02..05B-04 cannot ship — the no-op claim is unfalsifiable.

Output: 1 new `conftest.py` + 1 new test module + 2 binary fixture parquets. Suite goes from `142 passed, 4 skipped` to `>= 148 passed, 4 skipped` (exact count depends on parametrization expansion).
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
@.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md
@tests/fixtures/_generate_baseline.py
@tests/fixtures/baseline_pfc_seed42.parquet
@pfc_shaping/lt/model/shape_hourly.py
@pfc_shaping/lt/model/assembler.py

<interfaces>
Test patterns and fixtures expected (consumed by the new test module):

From `tests/fixtures/_generate_baseline.py` (plan 05B-01):
```
def main() -> pd.DataFrame:
    # Builds synthetic epex_df, calendar_df, fits ShapeHourly, builds PFC
    # Returns the resulting PFC DataFrame
```
The test must REUSE this generator to build a fresh PFC inside the test, then compare to the fixture parquet.

From `pfc_shaping/lt/model/shape_hourly.py` (plans 05B-02/03/04):
```
class ShapeHourly:
    def __init__(self, sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25,
                 use_seasonal_hourly: bool | None = None) -> None: ...
    @property
    def factors_3d_(self) -> Mapping[tuple[str, str, int], float]: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "ShapeHourly": ...

_FLAG_ENV_VAR: str = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"
_META_SIDECAR_FILENAME: str = "_meta.parquet"
```

Existing test pattern for synthetic calendar_df: see `tests/test_country_tz_plumbing.py` and `tests/test_long_term_branch.py`.

Existing test count baseline (verified via `pytest tests/ --co -q`): `146 tests collected` (142 passing + 4 skipped CT-only).
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create `tests/conftest.py` with autouse env-var hygiene fixture</name>
  <files>tests/conftest.py</files>
  <read_first>
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-12 — autouse fixture spec)
    - tests/test_country_tz_plumbing.py (existing pytest patterns in this project — fixture style, imports)
  </read_first>
  <action>
    Create `tests/conftest.py` (a new file — `cat tests/conftest.py` currently exits with "No such file or directory").

    Content requirements:
    1. Module docstring naming the purpose: "Project-wide pytest fixtures. The `_pfc_lt_env_hygiene` autouse fixture snapshots all `PFC_LT_*` env-var keys before each test and restores them after, preventing leak between tests that set the `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` flag (or future PFC_LT_* keys)."
    2. Imports: `from __future__ import annotations`, `import os`, `import pytest`.
    3. Fixture:
       ```
       @pytest.fixture(autouse=True)
       def _pfc_lt_env_hygiene(monkeypatch):
           # Snapshot all PFC_LT_* keys present at test entry
           snapshot = {k: v for k, v in os.environ.items() if k.startswith("PFC_LT_")}
           # Yield to test — test may freely set/del env vars
           yield
           # Restore: delete keys that were absent, restore keys that were present
           current = {k for k in os.environ if k.startswith("PFC_LT_")}
           for k in current - set(snapshot):
               os.environ.pop(k, None)
           for k, v in snapshot.items():
               os.environ[k] = v
       ```
       (The `monkeypatch` parameter is unused but accepted to allow future tests to depend on it; if pytest emits a warning for the unused parameter, you may drop it — the fixture must keep `autouse=True` and remain importable from `conftest.py`.)
    4. Do NOT add any other fixture or import side-effect. Keep the file < 40 lines.

    Sanity: running `pytest tests/ -x` after this task MUST still report `142 passed, 4 skipped`. Adding the conftest must not break any existing test.
  </action>
  <verify>
    <automated>test -f tests/conftest.py && python -c "
import os
os.environ['PFC_LT_TEST_KEY'] = 'leak'
" && pytest tests/test_country_tz_plumbing.py -x -q 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `test -f tests/conftest.py` exits 0.
    - `grep -q "PFC_LT_" tests/conftest.py` exits 0.
    - `grep -q "autouse=True" tests/conftest.py` exits 0.
    - `wc -l tests/conftest.py` reports a line count ≤ 50.
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped`.
  </acceptance_criteria>
  <done>conftest.py is the autouse env-var hygiene boundary; suite is still green.</done>
</task>

<task type="auto">
  <name>Task 2: Produce legacy fixture parquets (pre-5bis-A shape) for `test_save_load_legacy_compat`</name>
  <files>tests/fixtures/shape_hourly_legacy.parquet, tests/fixtures/f_W_legacy.parquet</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py (current save layout: long-format parquet with columns `saison, type_jour, heure, f_H, n_obs` per the existing save block; `f_W.parquet` sidecar with columns `type_jour, f_W`)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-05, D-16 — legacy parquet fixture used in legacy-compat test)
  </read_first>
  <action>
    Produce two small binary fixture files that emulate what `ShapeHourly.save()` produced BEFORE plan 05B-02 added the `_meta.parquet` sidecar. The legacy schemas (frozen by the existing code at `shape_hourly.py:308-325`) are:
    - `shape_hourly_legacy.parquet`: long-format with columns `saison`, `type_jour`, `heure`, `f_H`, `n_obs`. Populate with at least 2 cells × 24 hours (e.g. `("Hiver","Ouvrable")` and `("Ete","Samedi")`) using deterministic `f_H` values whose mean ≈ 1.0 per cell (use `np.linspace(0.8, 1.2, 24); arr /= arr.mean()` then store `arr`).
    - `f_W_legacy.parquet`: columns `type_jour`, `f_W`, with rows for all 5 `TYPES_JOUR` keys.

    Do NOT include a `_meta.parquet` next to these two files in the fixtures directory. The whole point of this fixture is to be `_meta.parquet`-absent so the legacy-compat warning path in `load()` is exercised.

    Implementation: write a small helper script `tests/fixtures/_generate_legacy_fixture.py` (mirroring the pattern from plan 05B-01) that builds the two DataFrames in-process and writes both parquets. The script does NOT use any code from `pfc_shaping.lt.model.shape_hourly` (it writes the parquets via `pandas.DataFrame.to_parquet` directly), to guarantee the fixture is byte-stable across future refactors of `save()`. Commit BOTH the script and the two parquets.

    Run the script once: `python tests/fixtures/_generate_legacy_fixture.py`. Verify the two parquets exist and contain the expected columns.
  </action>
  <verify>
    <automated>python tests/fixtures/_generate_legacy_fixture.py && python -c "
import pandas as pd
a = pd.read_parquet('tests/fixtures/shape_hourly_legacy.parquet')
b = pd.read_parquet('tests/fixtures/f_W_legacy.parquet')
assert {'saison','type_jour','heure','f_H','n_obs'}.issubset(a.columns), a.columns.tolist()
assert {'type_jour','f_W'}.issubset(b.columns), b.columns.tolist()
assert len(a) >= 48
import os
assert not os.path.exists('tests/fixtures/_meta.parquet'), 'legacy fixture must NOT have _meta sidecar'
print('OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - `test -f tests/fixtures/shape_hourly_legacy.parquet` exits 0.
    - `test -f tests/fixtures/f_W_legacy.parquet` exits 0.
    - `test ! -e tests/fixtures/_meta.parquet` exits 0 (legacy fixture has NO meta sidecar — that is the whole point).
    - `test -f tests/fixtures/_generate_legacy_fixture.py` exits 0.
    - Verify command above exits 0 and prints `OK`.
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped` (no test consuming these files yet — those are added in Task 3).
  </acceptance_criteria>
  <done>Legacy fixtures committed and read by Task 3's tests.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Write the six new tests in `tests/test_shape_hourly_infra.py`</name>
  <files>tests/test_shape_hourly_infra.py</files>
  <read_first>
    - tests/fixtures/_generate_baseline.py (from plan 05B-01 — the test_baseline_regression test will REUSE this code to build a fresh PFC inside the test)
    - tests/fixtures/baseline_pfc_seed42.parquet (the frozen reference)
    - tests/fixtures/shape_hourly_legacy.parquet (Task 2)
    - tests/fixtures/f_W_legacy.parquet (Task 2)
    - pfc_shaping/lt/model/shape_hourly.py (final state — `_FLAG_ENV_VAR`, `_META_SIDECAR_FILENAME`, `factors_3d_`, save/load, flag mechanics from plans 05B-02..05B-04)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-14 through D-20 — exact assertion specs)
  </read_first>
  <behavior>
    Each test is mandated by a specific D-XX decision:

    - `test_factors_3d_view_consistency` (D-14): fit on small synthetic data, assert `factors_3d_[(s,tj,h)] == factors_[(s,tj)][h]` for ALL `(s,tj)` in `sh.factors_` × `h in range(24)`. Assert assignment raises `TypeError`.
    - `test_save_load_full_roundtrip` (D-15): fit synthetic, save, load, compare all 9 attributes: `factors_` (numpy.allclose per cell), `factors_by_year_` (numpy.allclose per cell), `trend_per_hour_` (numpy.allclose per cell), `f_W_seasonal_` (dict equality on float values), `f_W_` (dict equality), `_climatological_fill` (pd.Series equality if not None), `sigma`/`halflife_days`/`hydro_weight_sigma` (`==` on floats), `_use_seasonal_hourly` (`==` on bool).
    - `test_save_load_legacy_compat` (D-16): copy `tests/fixtures/shape_hourly_legacy.parquet` + `tests/fixtures/f_W_legacy.parquet` to a temp dir (rename both to `shape_hourly.parquet` and `f_W.parquet` respectively to match `Path(path).with_name(...)` lookup in `load()`), then call `ShapeHourly.load(tmp/"shape_hourly.parquet")`. Assert no exception, assert one `logger.warning` emitted with message containing `"legacy"` (use `caplog.at_level(logging.WARNING)`), assert `sh.factors_` is non-empty, assert `sh.factors_by_year_ == {}` (legacy: lost attribute → empty), assert `sh.sigma == 0.5` (default after legacy load).
    - `test_flag_freeze_at_init` (D-17): four sub-assertions, all using `monkeypatch.setenv` (so the conftest hygiene fixture cleans up).
      - `monkeypatch.setenv(...,"0"); sh = ShapeHourly(use_seasonal_hourly=True)`; assert `True`. Then `monkeypatch.setenv(...,"1")`; assert `sh._use_seasonal_hourly is True` (unchanged — frozen).
      - `monkeypatch.delenv(..., raising=False); sh = ShapeHourly(use_seasonal_hourly=None)`; assert `False`. Set env to `"1"`; assert `sh._use_seasonal_hourly is False` (unchanged — frozen on first read).
      - `monkeypatch.setenv(...,"1"); sh = ShapeHourly()`; assert `True`. Then `delenv`; assert `True` (unchanged — frozen).
      - `monkeypatch.setenv(...,"0"); sh = ShapeHourly()`; assert `False`. Then `setenv "1"`; assert `False` (unchanged — frozen).
    - `test_flag_persisted_in_parquet` (D-18): fit synthetic `ShapeHourly(use_seasonal_hourly=True)`, save to tmpdir, then with `monkeypatch.setenv(...,"0")`, call `load(p)`, assert `sh2._use_seasonal_hourly is True` (parquet wins). Then reverse: fit with `use_seasonal_hourly=False`, save, `monkeypatch.setenv(...,"1")`, load → assert `False`.
    - `test_baseline_regression` (D-19, parametrized): use `@pytest.mark.parametrize("flag", [False, True])`. Re-build the PFC using the SAME generator function from `tests/fixtures/_generate_baseline.py` (import it: `from tests.fixtures._generate_baseline import main as build_baseline_pfc` — adjust if the function isn't named `main`, OR refactor the generator script in Task 4 to expose a `build_pfc(seed: int, flag: bool) -> pd.DataFrame` function). Inside the test, build with `flag` parametrized, then `assert_frame_equal(df, pd.read_parquet("tests/fixtures/baseline_pfc_seed42.parquet"), check_exact=False, atol=1e-10)`. Both flag values MUST pass — this is THE proof of no-op.

    All tests must be < 50 lines each, use `tmp_path` / `monkeypatch` / `caplog` pytest fixtures, depend ONLY on synthetic inputs and committed fixtures.
  </behavior>
  <action>
    Create `tests/test_shape_hourly_infra.py` with the six tests above. Module docstring must name the requirements covered (SHP-01, SHP-04) and reference `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md` D-14..D-20.

    Required imports: `pytest`, `numpy as np`, `pandas as pd`, `tempfile`, `pathlib.Path`, `logging`, `os`, and the relevant symbols from `pfc_shaping.lt.model.shape_hourly`.

    For `test_baseline_regression`, if `tests/fixtures/_generate_baseline.py` does NOT already expose a reusable `build_pfc(...)` function (only a `main()` script), perform a minimal refactor: factor out the body of `main()` into a reusable function `def build_pfc(seed: int = 42, flag: bool = False) -> pd.DataFrame:` and call it from both the test AND a thin `if __name__ == "__main__"` block in `_generate_baseline.py`. Update `tests/fixtures/README.md` to mention the new entry point. This refactor MUST NOT change the bytes of the committed `tests/fixtures/baseline_pfc_seed42.parquet` (verify by re-running the generator and `assert_frame_equal(reread, baseline, atol=1e-12)`).

    For the parametrized `test_baseline_regression`, use the `_pfc_lt_env_hygiene` autouse fixture from `conftest.py` for cleanup; explicitly set the flag via `ShapeHourly(use_seasonal_hourly=flag)` in `build_pfc`'s body so the parametrization works regardless of the ambient env.

    Document inside each test docstring which D-XX it implements.
  </action>
  <verify>
    <automated>pytest tests/test_shape_hourly_infra.py -v 2>&1 | tail -20 && pytest tests/ 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `pytest tests/test_shape_hourly_infra.py::test_factors_3d_view_consistency -x` exits 0.
    - `pytest tests/test_shape_hourly_infra.py::test_save_load_full_roundtrip -x` exits 0.
    - `pytest tests/test_shape_hourly_infra.py::test_save_load_legacy_compat -x` exits 0.
    - `pytest tests/test_shape_hourly_infra.py::test_flag_freeze_at_init -x` exits 0.
    - `pytest tests/test_shape_hourly_infra.py::test_flag_persisted_in_parquet -x` exits 0.
    - `pytest "tests/test_shape_hourly_infra.py::test_baseline_regression[False]" -x` exits 0.
    - `pytest "tests/test_shape_hourly_infra.py::test_baseline_regression[True]" -x` exits 0.
    - `pytest tests/ -x` exits 0 reporting `>= 148 passed, 4 skipped` (142 baseline + ≥6 new; parametrized adds 1 each, so exact count is `148 passed, 4 skipped` if no other expansion).
    - `grep -q "D-14\|D-19\|baseline_pfc_seed42" tests/test_shape_hourly_infra.py` exits 0.
  </acceptance_criteria>
  <done>All six tests pass; suite goes from 142 → ≥148 passed; 4 skipped preserved.</done>
</task>

</tasks>

<verification>
- `pytest tests/ -x` exits 0 reporting `>= 148 passed, 4 skipped`.
- `pytest tests/ --co -q | tail -1` reports `>= 152 tests collected` (146 existing + ≥6 new; parametrization counts as separate).
- `python tests/fixtures/_generate_baseline.py` is still runnable post-refactor and produces a byte-equivalent parquet.
- `git log --oneline tests/conftest.py` shows a single commit introducing the file.
</verification>

<success_criteria>
- All six new tests pass, including parametrized baseline regression for both flag states.
- Conftest autouse fixture prevents `PFC_LT_*` leak.
- Suite goes from 142 → ≥ 148 passed (4 skipped preserved).
- Phase 5bis-A is now a proven numerical no-op (assert_frame_equal(atol=1e-10) holds for flag OFF and flag ON).
</success_criteria>

<output>
Create `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-05-SUMMARY.md` when done.
</output>
