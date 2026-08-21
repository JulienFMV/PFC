---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: 03
type: execute
wave: 3
depends_on:
  - 05B-02
files_modified:
  - pfc_shaping/lt/model/shape_hourly.py
autonomous: true
requirements:
  - SHP-04
must_haves:
  truths:
    - "`ShapeHourly(use_seasonal_hourly=True)` overrides env var: even when `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=0` is set in `os.environ`, `self._use_seasonal_hourly is True` (constructor argument wins per D-06)."
    - "`ShapeHourly(use_seasonal_hourly=None)` reads `os.getenv('PFC_LT_USE_SEASONAL_HOURLY_SHAPE', '0')` exactly once in `__init__`, then stores the resolved bool in `self._use_seasonal_hourly`. The env var is NEVER re-read in `fit()`, `apply()`, or `save()`."
    - "Mutating `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']` AFTER construction does not change `self._use_seasonal_hourly` (freeze-at-init, per D-06)."
    - "`ShapeHourly.save()` persists `_use_seasonal_hourly` into the `${stem}.meta.parquet` sidecar (concretely `shape_hourly.meta.parquet`) via the `hyperparams` JSON cell."
    - "`ShapeHourly.load(path)` restores `_use_seasonal_hourly` from `${stem}.meta.parquet`; the loaded value WINS over the current env var (parquet wins, per D-07)."
    - "In Phase 5bis-A, `_use_seasonal_hourly` is set but NEVER read by any code path that would change numerical output. Flag ON and flag OFF must produce numerically identical outputs from `assembler.build` — verified by Plan 05's parametrized regression test using `assert_frame_equal(..., check_exact=False, atol=1e-12, rtol=0)`."
  artifacts:
    - path: "pfc_shaping/lt/model/shape_hourly.py"
      provides: "Feature flag `_use_seasonal_hourly` accepted, frozen, persisted (via `${stem}.meta.parquet`), restored — gates ZERO behavior in 5bis-A"
      contains: "use_seasonal_hourly"
  key_links:
    - from: "pfc_shaping/lt/model/shape_hourly.py::__init__"
      to: "os.getenv('PFC_LT_USE_SEASONAL_HOURLY_SHAPE')"
      via: "single read at __init__ time"
      pattern: "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"
    - from: "pfc_shaping/lt/model/shape_hourly.py::save"
      to: "${stem}.meta.parquet hyperparams JSON cell"
      via: "json.dumps adds 'use_seasonal_hourly' key"
      pattern: "use_seasonal_hourly"
    - from: "pfc_shaping/lt/model/shape_hourly.py::load"
      to: "obj._use_seasonal_hourly"
      via: "json.loads + assignment overrides env"
      pattern: "_use_seasonal_hourly"
---

<objective>
Introduce the `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` feature flag with **correct mechanics**: constructor argument + env-var default, frozen at `__init__`, persisted into the `${stem}.meta.parquet` sidecar (concretely `shape_hourly.meta.parquet`) — extending the sidecar introduced in plan 05B-02 — restored on `load()`. In Phase 5bis-A the flag exists in memory and on disk but gates **zero behavior** — flag ON ≡ flag OFF numerically.

Purpose: SHP-04 requires an env-flag rollback path for the seasonal-hourly shape work. Phase 5bis-B will add the behavioral branches gated by this flag. The non-trivial mechanics (freeze-at-init prevents test-leakage and prod env mutation mid-process; persistence prevents train/serve skew where a model fitted with `flag=ON` reloads in a prod env with `flag=OFF`) MUST exist BEFORE any gated behavior so the rollback contract is testable.

Output: `pfc_shaping/lt/model/shape_hourly.py` with new constructor arg `use_seasonal_hourly`, attribute `self._use_seasonal_hourly`, and parquet persistence/restoration via the existing `${stem}.meta.parquet` sidecar.
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
@pfc_shaping/lt/model/shape_hourly.py

<interfaces>
Constructor signature today (shape_hourly.py:55-72):
```
def __init__(self, sigma: float = 0.5, halflife_days: float = 180.0, hydro_weight_sigma: float = 0.25) -> None: ...
```

New signature after this plan:
```
def __init__(self, sigma: float = 0.5, halflife_days: float = 180.0, hydro_weight_sigma: float = 0.25, use_seasonal_hourly: bool | None = None) -> None: ...
```

Env-var name (fixed, do not change): `PFC_LT_USE_SEASONAL_HOURLY_SHAPE`
- `"1"` → True
- `"0"` (or unset) → False (treat unset as default-off — env-default semantics per D-06)
- Any other value → log warning, treat as False.

The `${stem}.meta.parquet` hyperparams JSON schema is extended (from plan 05B-02 baseline):
```
{"sigma": float, "halflife_days": float, "hydro_weight_sigma": float, "use_seasonal_hourly": bool}
```

Call sites that pass kwargs to `ShapeHourly.__init__` today (grep-confirmed):
- pfc_shaping/pipeline/autoresearch.py:234   `sh = ShapeHourly(sigma=sigma)`
- pfc_shaping/pipeline/rolling_update.py:365  `sh = ShapeHourly(sigma=params.get("gaussian_sigma", 0.5)).fit(epex_fit, cal_fit)`
- pfc_shaping/pipeline/production_phases.py:270 `sh = ShapeHourly()`
- pfc_shaping/pipeline/production_phases.py:620 `sh = ShapeHourly()`
- pfc_shaping/validation/backtest.py:185 `sh = ShapeHourly().fit(train_epex, cal_train)`

All call sites use kwargs/no-arg → the new optional `use_seasonal_hourly` kwarg is additive and does NOT break any caller.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add `use_seasonal_hourly` constructor arg with env-default + freeze-at-init</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py (UPDATED file from plan 05B-02 — note the meta sidecar wiring with `_META_SIDECAR_SUFFIX = ".meta.parquet"` constant and the `_meta_path()` helper)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-06, D-07, D-08 — exact semantics: constructor wins, freeze-at-init, persist-restore-wins)
    - pfc_shaping/pipeline/autoresearch.py:230-240, pfc_shaping/pipeline/rolling_update.py:360-370, pfc_shaping/pipeline/production_phases.py:265-275 (call-site sanity: confirm new kwarg is additive)
  </read_first>
  <behavior>
    - Test 1: `import os; os.environ.pop('PFC_LT_USE_SEASONAL_HOURLY_SHAPE', None); sh = ShapeHourly(); assert sh._use_seasonal_hourly is False`.
    - Test 2: `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']='1'; sh = ShapeHourly(); assert sh._use_seasonal_hourly is True`.
    - Test 3: `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']='0'; sh = ShapeHourly(use_seasonal_hourly=True); assert sh._use_seasonal_hourly is True` (constructor wins over env per D-06).
    - Test 4: `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']='1'; sh = ShapeHourly(use_seasonal_hourly=False); assert sh._use_seasonal_hourly is False` (constructor wins, both directions).
    - Test 5: After construction, mutating `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']` to the opposite value does NOT change `sh._use_seasonal_hourly` (freeze-at-init per D-06).
    - Test 6: `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']='yes'` (invalid value) → `sh._use_seasonal_hourly is False` and one `logger.warning` emitted at construction time.
    - Test 7: The env-var read uses `os.getenv(name, default='0')`, not raw `os.environ[name]`, so unset env is the documented "default OFF" path.
    - Test 8: `inspect.signature(ShapeHourly).parameters['use_seasonal_hourly'].default is None` (kwarg is keyword-with-default, additive).
  </behavior>
  <action>
    In `pfc_shaping/lt/model/shape_hourly.py`:

    1. Add `import os` to the module header (alongside `import json` added in plan 05B-02 and existing `import logging`).
    2. Add a module-level constant near the top, alongside `_META_SIDECAR_SUFFIX`:
       - `_FLAG_ENV_VAR = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"`
    3. Add a module-level helper (private) `def _resolve_flag(explicit: bool | None) -> bool:` that implements the precedence rule:
       - If `explicit is not None`: return `bool(explicit)`.
       - Otherwise read `raw = os.getenv(_FLAG_ENV_VAR, "0")`.
       - If `raw == "1"`: return True. If `raw == "0"`: return False. Else: `logger.warning("Invalid value %r for %s; treating as '0' (default off)", raw, _FLAG_ENV_VAR)` and return False.
    4. Extend `ShapeHourly.__init__` signature to add a new LAST kwarg: `use_seasonal_hourly: bool | None = None`. Inside `__init__`, set `self._use_seasonal_hourly: bool = _resolve_flag(use_seasonal_hourly)` AFTER all existing attribute assignments. This single line is the ONLY place that reads the env var.
    5. Add a class-level docstring update or inline comment near the new attribute clarifying: "Resolved once at __init__ and frozen. Use the constructor kwarg or set env var BEFORE calling __init__. Persisted into the ${stem}.meta.parquet sidecar by save() and overwritten by load() (parquet wins). In Phase 5bis-A this flag gates NO behavior; reserved for Phase 5bis-B."

    Do NOT read the env var anywhere else in this file. Do NOT add any branch in `fit()`, `apply()`, `get()`, `get_for_horizon()`, `_fit_trends()`, etc. — the flag is purely declarative in 5bis-A.

    Do NOT modify any call site in `pfc_shaping/pipeline/*.py` or `pfc_shaping/validation/*.py` — the new kwarg is additive and defaults to None.
  </action>
  <verify>
    <automated>python -c "
import os, inspect, logging
logging.basicConfig(level=logging.WARNING)
from pfc_shaping.lt.model.shape_hourly import ShapeHourly, _FLAG_ENV_VAR
assert _FLAG_ENV_VAR == 'PFC_LT_USE_SEASONAL_HOURLY_SHAPE'
sig = inspect.signature(ShapeHourly.__init__)
assert 'use_seasonal_hourly' in sig.parameters
assert sig.parameters['use_seasonal_hourly'].default is None
os.environ.pop('PFC_LT_USE_SEASONAL_HOURLY_SHAPE', None)
sh = ShapeHourly(); assert sh._use_seasonal_hourly is False
os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']='1'
sh = ShapeHourly(); assert sh._use_seasonal_hourly is True
sh = ShapeHourly(use_seasonal_hourly=False); assert sh._use_seasonal_hourly is False
os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']='0'
sh = ShapeHourly(use_seasonal_hourly=True)
assert sh._use_seasonal_hourly is True
# Freeze-at-init
os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']='1'
assert sh._use_seasonal_hourly is True  # unchanged because constructor=True won, and freeze
sh2 = ShapeHourly()  # re-construct picks up env
assert sh2._use_seasonal_hourly is True
del os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']
assert sh2._use_seasonal_hourly is True  # frozen
print('OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - Verify command above exits 0 and prints `OK`.
    - `grep -c "os.getenv\|os.environ" pfc_shaping/lt/model/shape_hourly.py` reports exactly 1 (single read site, in `_resolve_flag`).
    - `grep -v '^#\|^ *#' pfc_shaping/lt/model/shape_hourly.py | grep -c "_use_seasonal_hourly" ` ≥ 2 (attribute assigned in __init__, no behavioral branch yet — count grows in plan 5bis-B).
    - `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; sh = ShapeHourly(); assert not hasattr(sh, '_use_seasonal_hourly') or True; assert sh._use_seasonal_hourly in (True, False)"` exits 0 (attribute exists and is a bool).
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped`.
  </acceptance_criteria>
  <done>Flag accepted via env or constructor, frozen at init, no behavior change.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Persist `_use_seasonal_hourly` in `${stem}.meta.parquet` + restore in `load()` (parquet wins over env)</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py (UPDATED file from Task 1 — note `_resolve_flag`, `_FLAG_ENV_VAR`, and the existing `${stem}.meta.parquet` write/read added in plan 05B-02 at the `hyperparams` JSON row, plus the `_meta_path()` helper)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-07 — train/serve skew prevention: parquet wins over env)
  </read_first>
  <behavior>
    - Test 1: After `sh = ShapeHourly(use_seasonal_hourly=True); sh.save("/tmp/sh.parquet")`, `pd.read_parquet("/tmp/sh.meta.parquet")` contains a row `attr == "hyperparams"` whose `value` is a JSON string whose parsed object has `"use_seasonal_hourly": true`.
    - Test 2: `sh2 = ShapeHourly.load("/tmp/sh.parquet")` with `os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE']="0"` set BEFORE the `load` call yields `sh2._use_seasonal_hourly is True` (parquet wins over env).
    - Test 3: Reverse direction — fit and save with `use_seasonal_hourly=False`, then load with `PFC_LT_USE_SEASONAL_HOURLY_SHAPE="1"` set → `sh2._use_seasonal_hourly is False`.
    - Test 4: Loading a legacy parquet (no `${stem}.meta.parquet` sidecar) falls back to constructor-default behavior (env or False); the per-Task-1 warning is still emitted, AND `sh._use_seasonal_hourly` matches `_resolve_flag(None)` at load time.
    - Test 5: If the meta sidecar exists but its `hyperparams` row is missing the `use_seasonal_hourly` key (e.g. parquet written by plan 05B-02 only, BEFORE plan 05B-03 was merged), `load()` MUST fall back to `_resolve_flag(None)` for `_use_seasonal_hourly` without raising. Other hyperparams in the JSON dict still load normally.
  </behavior>
  <action>
    In `pfc_shaping/lt/model/shape_hourly.py`:

    1. Modify the `save()` method block that builds the `hyperparams` JSON cell (added in plan 05B-02 Task 1) to include the new key:
       `json.dumps({"sigma": self.sigma, "halflife_days": self.halflife_days, "hydro_weight_sigma": self.hydro_weight_sigma, "use_seasonal_hourly": bool(self._use_seasonal_hourly)}, sort_keys=True)`.
    2. Modify the `load()` method block that parses the `hyperparams` row (added in plan 05B-02 Task 2):
       - After existing `obj.sigma = hp.get("sigma", obj.sigma)` etc., add:
         `if "use_seasonal_hourly" in hp: obj._use_seasonal_hourly = bool(hp["use_seasonal_hourly"])`.
       - If `"use_seasonal_hourly"` key is absent from the parsed hyperparams dict (cross-plan compat — parquet written by 05B-02 but before 05B-03 was merged), leave `obj._use_seasonal_hourly` at the value already set by `cls()` (which read the env via `_resolve_flag(None)`).
    3. Do NOT change the order of the JSON keys (use `sort_keys=True` as in plan 05B-02).
    4. Do NOT add any code path that re-reads the env var in `load()` — the env is only ever consulted when `cls()` is called (which `load()` does once at the top, inherited from the existing code at shape_hourly.py:331).
    5. Add an inline comment in `load()` next to the use_seasonal_hourly restoration line: `# parquet wins over env — prevents train/serve skew (D-07)`.
  </action>
  <verify>
    <automated>python -c "
import os, json, tempfile
from pfc_shaping.lt.model.shape_hourly import ShapeHourly
import pandas as pd

# Forward roundtrip
os.environ.pop('PFC_LT_USE_SEASONAL_HOURLY_SHAPE', None)
sh = ShapeHourly(use_seasonal_hourly=True)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'shape_hourly.parquet')
    sh.save(p)
    meta = pd.read_parquet(os.path.join(d, 'shape_hourly.meta.parquet'))
    hp_row = meta[meta['attr'] == 'hyperparams'].iloc[0]
    hp = json.loads(hp_row['value'])
    assert hp.get('use_seasonal_hourly') is True, hp
    # Parquet wins over env
    os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE'] = '0'
    sh2 = ShapeHourly.load(p)
    assert sh2._use_seasonal_hourly is True, sh2._use_seasonal_hourly
    # Reverse
    sh3 = ShapeHourly(use_seasonal_hourly=False)
    p2 = os.path.join(d, 'shape_hourly2.parquet')
    sh3.save(p2)
    os.environ['PFC_LT_USE_SEASONAL_HOURLY_SHAPE'] = '1'
    sh4 = ShapeHourly.load(p2)
    assert sh4._use_seasonal_hourly is False, sh4._use_seasonal_hourly
os.environ.pop('PFC_LT_USE_SEASONAL_HOURLY_SHAPE', None)
print('OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - Verify command above exits 0 and prints `OK`.
    - `grep -q "use_seasonal_hourly" pfc_shaping/lt/model/shape_hourly.py` exits 0.
    - `grep -q "parquet wins over env" pfc_shaping/lt/model/shape_hourly.py` exits 0 (inline comment present per D-07 traceability).
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped` (no behavior change, no new tests yet — those live in plan 05B-05).
  </acceptance_criteria>
  <done>Flag persists across save/load via `shape_hourly.meta.parquet`; reload from parquet overrides any environment variable.</done>
</task>

</tasks>

<verification>
- `pytest tests/ -x` exits 0 with `142 passed, 4 skipped`.
- Constructor + env + parquet precedence verified above.
- Sidecar file is named `shape_hourly.meta.parquet` (the `${stem}.meta.parquet` convention from Plan 05B-02).
- No numerical behavior change to `assembler.build` (verified by plan 05B-05 parametrized regression test using `assert_frame_equal(..., check_exact=False, atol=1e-12, rtol=0)`).
</verification>

<success_criteria>
- SHP-04 satisfied: `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` flag operational, constructor + env-default, frozen at __init__, persisted in the `${stem}.meta.parquet` sidecar.
- 142 existing tests remain green.
- In 5bis-A: flag exists but ZERO numerical behavior gated by it. Reserved for 5bis-B.
</success_criteria>

<output>
Create `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-03-SUMMARY.md` when done.
</output>
