---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: 02
type: execute
wave: 2
depends_on:
  - 05B-01
files_modified:
  - pfc_shaping/lt/model/shape_hourly.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "`ShapeHourly.save(path)` writes a sidecar `_meta.parquet` next to `path` containing all trained attributes that were silently lost at reload pre-5bis-A: `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, and the scalar hyperparams (`sigma`, `halflife_days`, `hydro_weight_sigma`)."
    - "`ShapeHourly.load(path)` detects the presence of `_meta.parquet` and restores every persisted attribute identically."
    - "If `_meta.parquet` is absent (legacy parquet from pre-5bis-A), `load()` emits ONE `logger.warning` naming the missing attributes and proceeds with constructor defaults — no exception raised."
    - "`fit → save → load → save → load` roundtrip on synthetic data yields identical `factors_`, `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `f_W_`, `_climatological_fill`, `sigma`, `halflife_days`, `hydro_weight_sigma` between rounds (numpy.allclose atol=1e-12 on arrays, == on scalars)."
    - "This plan changes NO numerical behavior — `assembler.build` output is unchanged from baseline (verified in plan 05B-05)."
  artifacts:
    - path: "pfc_shaping/lt/model/shape_hourly.py"
      provides: "Completed save/load roundtrip via `_meta.parquet` sidecar + legacy compat warning"
      contains: "_meta.parquet"
  key_links:
    - from: "pfc_shaping/lt/model/shape_hourly.py::ShapeHourly.save"
      to: "Path(path).with_name('_meta.parquet')"
      via: "pandas.DataFrame.to_parquet"
      pattern: "_meta.parquet"
    - from: "pfc_shaping/lt/model/shape_hourly.py::ShapeHourly.load"
      to: "Path(path).with_name('_meta.parquet')"
      via: "pandas.read_parquet + logger.warning fallback"
      pattern: "_meta.parquet"
---

<objective>
Fix the pre-existing save/load bug in `ShapeHourly`: today `save()` persists only `factors_` and `f_W_`, while `load()` silently discards `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, and all scalar hyperparams (`sigma`, `halflife_days`, `hydro_weight_sigma`). This means models fitted with non-default hyperparams or with hydro_df reload as different objects, producing different predictions vs the fitted-in-memory model.

Purpose: Restore round-trip fidelity on ALL trained attributes via a `_meta.parquet` sidecar. This is the prerequisite for plan 05B-03 (feature flag), which will piggyback on the same sidecar to persist `_use_seasonal_hourly`. It is also a prerequisite for any train→serve workflow.

Output: `pfc_shaping/lt/model/shape_hourly.py` with completed `save()`/`load()` writing/reading `_meta.parquet`, plus a one-shot `logger.warning` on legacy-load path.
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
@pfc_shaping/lt/model/shape_hourly.py

<interfaces>
Attributes that MUST roundtrip after this plan (extracted from `ShapeHourly.__init__` and `fit()` body in shape_hourly.py:55-165):

```
# Already persisted today (in shape_hourly.parquet + f_W.parquet):
self.factors_: dict[(str, str), np.ndarray]               # (saison, type_jour) -> array[24]
self.n_obs_: dict[(str, str), int]                        # (saison, type_jour) -> int
self.f_W_: dict[str, float]                               # type_jour -> ratio

# Currently lost at reload — must be persisted to _meta.parquet:
self.f_W_seasonal_: dict[(str, str), float]              # (saison, type_jour) -> ratio
self.factors_by_year_: dict[(str, str, int), np.ndarray]  # (saison, type_jour, year) -> array[24]
self.trend_per_hour_: dict[(str, str), np.ndarray]       # (saison, type_jour) -> slope[24]
self._hydro_fill_weekly: pd.Series | None                 # weekly fill_pct series
self._climatological_fill: pd.Series | None              # week_of_year -> mean fill
self.global_factors_: np.ndarray | None                   # mean profile, length 24

# Scalar hyperparams (set in __init__, currently lost at reload because load() does `cls()`):
self.sigma: float
self.halflife_days: float
self.hydro_weight_sigma: float
```

Existing save/load implementation lives at `shape_hourly.py:308-343`.
Existing convention: long-format parquet — see `factors_` save loop at lines 310-317 and `f_W.parquet` sidecar at lines 319-323.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Extend `ShapeHourly.save()` to write `_meta.parquet` sidecar with all currently-lost attributes</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py (FULL file — class structure, existing save/load at lines 308-343, attribute initializations at lines 55-75)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-03 — exact sidecar schema specification)
  </read_first>
  <behavior>
    - Test 1: After `sh.save("/tmp/sh.parquet")` on a fitted instance, `Path("/tmp/_meta.parquet").exists()` is True. (File naming uses `Path(path).with_name("_meta.parquet")` per existing `f_W.parquet` convention at shape_hourly.py:320.)
    - Test 2: `_meta.parquet` is a single parquet file containing a long-format schema with a discriminator column `attr` ∈ {`factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, `_hydro_fill_weekly`, `hyperparams`}.
    - Test 3: For each `(saison, type_jour, year)` in `self.factors_by_year_`, the saved rows allow exact reconstruction of the array[24] (24 rows per cell with columns `saison`, `type_jour`, `year`, `heure`, `value`, `attr="factors_by_year_"`).
    - Test 4: For each `(saison, type_jour)` in `self.trend_per_hour_`, 24 rows are saved with `attr="trend_per_hour_"`, columns `saison`, `type_jour`, `heure`, `value`.
    - Test 5: For each `(saison, type_jour)` in `self.f_W_seasonal_`, 1 row is saved with `attr="f_W_seasonal_"`, columns `saison`, `type_jour`, `value`.
    - Test 6: `self._climatological_fill` (pd.Series indexed by week-of-year) is saved as `attr="_climatological_fill"` with columns `week`, `value`. If `None`, NO rows for this attr (sentinel = absence).
    - Test 7: `self._hydro_fill_weekly` is saved analogously with `attr="_hydro_fill_weekly"`, columns `timestamp` (str ISO-8601), `value`.
    - Test 8: A single row with `attr="hyperparams"` and JSON-string column `value` = `'{"sigma": 0.5, "halflife_days": 180.0, "hydro_weight_sigma": 0.25}'` (use `json.dumps` with sorted keys for determinism).
    - Test 9: Calling `save()` on an UNFITTED `ShapeHourly()` (empty `factors_`) does NOT crash and writes an `_meta.parquet` containing only the `hyperparams` row.
  </behavior>
  <action>
    In `pfc_shaping/lt/model/shape_hourly.py`, extend the existing `save()` method (currently at lines 308-325) to write `_meta.parquet` as a sidecar next to `path`, alongside the existing `f_W.parquet`.

    Concrete steps:
    1. Add `import json` to the module header (top of file, alongside existing `import logging`).
    2. In `save()`, after the existing `pd.DataFrame(records).to_parquet(path, index=False)` and the existing `f_W.parquet` write block, build a list `meta_records: list[dict]` containing one Python dict per row, with the discriminator column `attr` as documented in `<behavior>`.
    3. Use `Path(path).with_name("_meta.parquet")` to compute the sidecar path (mirrors the existing `f_W.parquet` convention at the current line 320).
    4. Write `pd.DataFrame(meta_records).to_parquet(meta_path, index=False)`. If `meta_records` ends up empty (extreme edge case: even hyperparams should make it non-empty), still write a 1-row DataFrame with the `hyperparams` row only so `_meta.parquet` always exists on disk.
    5. Use `json.dumps({"sigma": self.sigma, "halflife_days": self.halflife_days, "hydro_weight_sigma": self.hydro_weight_sigma}, sort_keys=True)` for the hyperparams value cell.
    6. Update the existing `logger.info("ShapeHourly sauvegardé : %s", path)` to also mention the sidecar: `logger.info("ShapeHourly sauvegardé : %s (+ _meta.parquet sidecar)", path)`.

    Do NOT touch the existing `f_H` parquet schema, the existing `f_W.parquet` sidecar, or the existing `n_obs_` persistence — only ADD the new sidecar. Keep all column names in snake_case ASCII.

    Do NOT introduce a new dependency or any non-stdlib import beyond `json`.

    Add a one-line module-level constant `_META_SIDECAR_FILENAME = "_meta.parquet"` near the top of the file, reused by both `save()` and `load()` (avoids string drift).
  </action>
  <verify>
    <automated>python -c "
from pfc_shaping.lt.model.shape_hourly import ShapeHourly
import numpy as np, pandas as pd, tempfile, os
sh = ShapeHourly(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.5)
sh.factors_[('Hiver','Ouvrable')] = np.arange(24, dtype=float)
sh.factors_[('Hiver','Ouvrable')] /= sh.factors_[('Hiver','Ouvrable')].mean()
sh.n_obs_[('Hiver','Ouvrable')] = 100
sh.f_W_['Ouvrable'] = 1.05
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'shape_hourly.parquet')
    sh.save(p)
    assert os.path.exists(os.path.join(d, '_meta.parquet')), 'no _meta.parquet sidecar'
    meta = pd.read_parquet(os.path.join(d, '_meta.parquet'))
    assert 'attr' in meta.columns
    hp = meta[meta['attr'] == 'hyperparams']
    assert len(hp) == 1
    import json as J
    obj = J.loads(hp['value'].iloc[0])
    assert obj == {'sigma': 0.3, 'halflife_days': 90.0, 'hydro_weight_sigma': 0.5}, obj
print('OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - `python -c "from pfc_shaping.lt.model.shape_hourly import _META_SIDECAR_FILENAME; assert _META_SIDECAR_FILENAME == '_meta.parquet'"` exits 0.
    - The verify command above exits 0 and prints `OK`.
    - `grep -c "_meta.parquet" pfc_shaping/lt/model/shape_hourly.py` ≥ 2 (used in both save and load — load wired in Task 2).
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped`.
  </acceptance_criteria>
  <done>`save()` writes `_meta.parquet` with all currently-lost attributes; existing parquet outputs unchanged.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Extend `ShapeHourly.load()` to restore from `_meta.parquet` + emit warning on legacy parquet</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py (UPDATED file from Task 1 — note the `_META_SIDECAR_FILENAME` constant and the meta row schema)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-04, D-05 — load contract + legacy warning)
  </read_first>
  <behavior>
    - Test 1: After `sh1.fit(...).save(p); sh2 = ShapeHourly.load(p)`, `sh2.factors_by_year_` is a non-empty dict equal to `sh1.factors_by_year_` (numpy.allclose on each cell's array[24]).
    - Test 2: Same for `sh2.trend_per_hour_` (deep-equal to `sh1.trend_per_hour_` via numpy.allclose).
    - Test 3: Same for `sh2.f_W_seasonal_` (`dict == dict` exact comparison after rounding to default float repr).
    - Test 4: `sh2.sigma == sh1.sigma`, `sh2.halflife_days == sh1.halflife_days`, `sh2.hydro_weight_sigma == sh1.hydro_weight_sigma`, even when `sh1` was constructed with non-default values.
    - Test 5: If `_climatological_fill` was non-None at save time, `sh2._climatological_fill` is a `pd.Series` numerically equal (index + values) — when None, `sh2._climatological_fill is None`.
    - Test 6: If `_hydro_fill_weekly` was non-None at save time, restored as a `pd.Series` of equal length and values (UTC-aware DatetimeIndex preserved via ISO-8601 column).
    - Test 7: Loading a legacy parquet (one without `_meta.parquet` next to it — simulate by saving via Task 1 logic, then deleting `_meta.parquet`) succeeds, returns a `ShapeHourly` instance with `factors_` and `f_W_` populated, `factors_by_year_`/`trend_per_hour_`/`f_W_seasonal_` as empty dicts, `_climatological_fill is None`, scalar hyperparams at constructor defaults (`sigma=0.5`, `halflife_days=180.0`, `hydro_weight_sigma=0.25`). A `logger.warning(...)` is emitted EXACTLY ONCE during the call.
  </behavior>
  <action>
    In `pfc_shaping/lt/model/shape_hourly.py`, extend the existing `load()` classmethod (currently at lines 327-343) to:

    1. After the existing block that loads `factors_` and `n_obs_` and the `f_W.parquet` sidecar, compute `meta_path = Path(path).with_name(_META_SIDECAR_FILENAME)`.
    2. If `meta_path.exists()`:
       - Read `meta_df = pd.read_parquet(meta_path)`.
       - Parse the `attr == "hyperparams"` row's `value` column with `json.loads(...)`; assign `obj.sigma`, `obj.halflife_days`, `obj.hydro_weight_sigma` from the parsed dict, falling back to existing attribute (set by `cls()` already, equal to constructor defaults) if a key is missing.
       - For `attr == "factors_by_year_"`: groupby `["saison", "type_jour", "year"]`, sort by `heure`, populate `obj.factors_by_year_[(s, tj, int(y))] = grp.sort_values("heure")["value"].to_numpy()`.
       - For `attr == "trend_per_hour_"`: same pattern, populate `obj.trend_per_hour_[(s, tj)] = grp.sort_values("heure")["value"].to_numpy()`.
       - For `attr == "f_W_seasonal_"`: iterate rows, populate `obj.f_W_seasonal_[(s, tj)] = float(value)`.
       - For `attr == "_climatological_fill"`: build `pd.Series(rows["value"].to_numpy(), index=rows["week"].astype(int).to_numpy(), name="fill_pct")` if any rows present; else leave `obj._climatological_fill = None`.
       - For `attr == "_hydro_fill_weekly"`: build a UTC-tz `pd.Series` indexed by `pd.to_datetime(rows["timestamp"], utc=True)` with values from `rows["value"]`. If no rows, leave `obj._hydro_fill_weekly = None`.
    3. If `not meta_path.exists()`: emit exactly one `logger.warning("Loading legacy %s without _meta sidecar — factors_by_year_, trend_per_hour_, f_W_seasonal_, _climatological_fill, _hydro_fill_weekly, and scalar hyperparams (sigma, halflife_days, hydro_weight_sigma) unavailable; falling back to constructor defaults", path)` and proceed. Do NOT raise.
    4. Keep the existing `obj.global_factors_ = obj._compute_global_fallback()` call at the end unchanged.
    5. Return `obj` as before.

    Be defensive about types coming back from parquet: `year` may roundtrip as `np.int64` — cast to `int(...)` when keying the dict. `value` may roundtrip as `np.float64` — cast to `float(...)` for the seasonal dict entries.

    Do NOT change the signature of `load(cls, path)`. Do NOT change the existing `factors_` / `f_W_` load logic. Do NOT introduce any new public method.
  </action>
  <verify>
    <automated>python -c "
from pfc_shaping.lt.model.shape_hourly import ShapeHourly
import numpy as np, pandas as pd, tempfile, os, logging
logging.basicConfig(level=logging.INFO)

sh1 = ShapeHourly(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7)
sh1.factors_[('Hiver','Ouvrable')] = np.linspace(0.8, 1.2, 24); sh1.factors_[('Hiver','Ouvrable')] /= sh1.factors_[('Hiver','Ouvrable')].mean()
sh1.n_obs_[('Hiver','Ouvrable')] = 100
sh1.f_W_['Ouvrable'] = 1.05
sh1.f_W_seasonal_[('Hiver','Ouvrable')] = 1.08
sh1.factors_by_year_[('Hiver','Ouvrable',2023)] = np.linspace(0.7, 1.3, 24); sh1.factors_by_year_[('Hiver','Ouvrable',2023)] /= sh1.factors_by_year_[('Hiver','Ouvrable',2023)].mean()
sh1.trend_per_hour_[('Hiver','Ouvrable')] = np.linspace(-0.01, 0.01, 24)
sh1._climatological_fill = pd.Series([0.5,0.6,0.7], index=[1,2,3])
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'shape_hourly.parquet')
    sh1.save(p)
    sh2 = ShapeHourly.load(p)
    assert sh2.sigma == 0.3, sh2.sigma
    assert sh2.halflife_days == 90.0, sh2.halflife_days
    assert sh2.hydro_weight_sigma == 0.7, sh2.hydro_weight_sigma
    assert np.allclose(sh2.factors_by_year_[('Hiver','Ouvrable',2023)], sh1.factors_by_year_[('Hiver','Ouvrable',2023)])
    assert np.allclose(sh2.trend_per_hour_[('Hiver','Ouvrable')], sh1.trend_per_hour_[('Hiver','Ouvrable')])
    assert sh2.f_W_seasonal_[('Hiver','Ouvrable')] == 1.08
    assert sh2._climatological_fill is not None
    assert list(sh2._climatological_fill.index) == [1,2,3]
    # Legacy compat
    os.remove(os.path.join(d, '_meta.parquet'))
    sh3 = ShapeHourly.load(p)
    assert sh3.sigma == 0.5, sh3.sigma  # default
    assert sh3.factors_by_year_ == {}
    assert sh3._climatological_fill is None
print('OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - Verify command above exits 0 and prints `OK`.
    - `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; import inspect; sig=inspect.signature(ShapeHourly.load); assert list(sig.parameters) == ['path'], list(sig.parameters)"` exits 0 (signature unchanged).
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped` (or unchanged baseline — no new tests added in this plan; those live in plan 05B-05).
    - `grep -c "logger.warning" pfc_shaping/lt/model/shape_hourly.py` ≥ existing count + 1 (one new legacy-compat warning added).
  </acceptance_criteria>
  <done>`load()` restores every attribute persisted by Task 1; legacy parquets reload without crash and emit exactly one warning.</done>
</task>

</tasks>

<verification>
- `pytest tests/ -x` exits 0 with `142 passed, 4 skipped`.
- `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; sh = ShapeHourly(); sh.save('/tmp/sh.parquet'); sh2 = ShapeHourly.load('/tmp/sh.parquet'); assert sh2.sigma == sh.sigma"` exits 0.
- Numerical behavior of `apply()` is UNCHANGED (will be verified by the baseline regression test in plan 05B-05).
</verification>

<success_criteria>
- All trained attributes survive `save → load` roundtrip.
- Legacy parquets continue to load (warning, no crash).
- 142 tests remain green.
- Numerical output of `assembler.build` is unchanged from baseline (verified later).
</success_criteria>

<output>
Create `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-02-SUMMARY.md` when done.
</output>
