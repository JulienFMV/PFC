---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: 04
type: execute
wave: 4
depends_on:
  - 05B-03
files_modified:
  - pfc_shaping/lt/model/shape_hourly.py
  - pfc_shaping/lt/model/assembler.py
autonomous: true
requirements:
  - SHP-01
must_haves:
  truths:
    - "`ShapeHourly.factors_3d_` exposes a read-only mapping keyed by 3-tuple `(saison, type_jour, hour)` returning a `float`."
    - "For every `(s, tj, h)` reachable from `factors_`, `factors_3d_[(s, tj, h)] == factors_[(s, tj)][h]` exactly (no copy, no smoothing, no normalization — it is a pure view)."
    - "Mutating `factors_3d_` raises a `TypeError` (read-only). The underlying `factors_` dict is unchanged."
    - "Iterating `factors_3d_` yields exactly `len(factors_) * 24` keys, each a 3-tuple."
    - "`assembler.py:284` no longer uses `try/except TypeError` around `self.sh.apply(...)`. The decision of whether to pass `outages_forecast` is made via an explicit signature inspection on `type(self.sh).apply` (per D-13)."
    - "The capability check produces IDENTICAL routing decisions vs the previous `try/except TypeError`: when `self.sh` is a `ShapeHourly`, `outages_forecast` is NOT passed; when `self.sh` is a `ShapeHourlyMLP`, it IS passed."
    - "On the first `build()` call (or at `__init__`, depending on implementation), the assembler emits exactly one `logger.info` line naming the detected `self.sh` class and whether `outages_forecast` is being passed — useful for production audits."
    - "This plan changes NO numerical output of `assembler.build` (verified by plan 05B-05 baseline regression using `assert_frame_equal(..., check_exact=False, atol=1e-12, rtol=0)`)."
  artifacts:
    - path: "pfc_shaping/lt/model/shape_hourly.py"
      provides: "Read-only 3D view `factors_3d_` on top of nested `factors_` dict"
      contains: "factors_3d_"
    - path: "pfc_shaping/lt/model/assembler.py"
      provides: "Explicit capability check on `self.sh.apply` signature at line ~284 (was try/except TypeError) + one-shot logger.info naming the detected implementation"
      contains: "inspect.signature"
  key_links:
    - from: "pfc_shaping/lt/model/shape_hourly.py::ShapeHourly.factors_3d_"
      to: "self.factors_[(s, tj)][h]"
      via: "property returning a lazy Mapping view"
      pattern: "factors_3d_"
    - from: "pfc_shaping/lt/model/assembler.py:~284"
      to: "inspect.signature(type(self.sh).apply).parameters"
      via: "capability check on `outages_forecast` parameter"
      pattern: "outages_forecast"
---

<objective>
Two surgical changes:

1. **SHP-01 literal satisfaction**: Add a read-only 3D view `factors_3d_` to `ShapeHourly` so the requirement `factors_ indexed by (saison, type_jour, hour)` is satisfied without changing on-disk storage or the smoothing pipeline. Per D-01..D-02, the internal `dict[(saison, type_jour)] → array[24]` representation is preserved (smoothing is intra-cell, normalization is intra-cell — the array IS the natural unit). The 3D view is purely a Mapping facade.

2. **Replace `try/except TypeError` with explicit capability check** at `assembler.py:284` (per D-13). The current code masks a real `TypeError` if it ever leaks from a bug inside `ShapeHourly.apply()` — a brittle pattern. Replace with `inspect.signature(...)` introspection on `type(self.sh).apply` to decide whether to forward `outages_forecast`. Emit a one-shot `logger.info` naming the detected implementation for production audit visibility.

Purpose: SHP-01 must be literally checkable in tests. The capability check removes a known bug-masking pattern flagged in audit. Both changes are zero-behavior at the numerical level.

Output: `pfc_shaping/lt/model/shape_hourly.py` with a `factors_3d_` property (or Mapping subclass) and `pfc_shaping/lt/model/assembler.py` with the `try/except TypeError` block replaced by an explicit signature inspection + one-shot operator log.
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
@pfc_shaping/lt/model/assembler.py
@pfc_shaping/lt/model/shape_hourly_mlp.py

<interfaces>
Existing structure in `shape_hourly.py` after plans 05B-02/05B-03:
```
self.factors_: dict[tuple[str, str], np.ndarray]  # (saison, type_jour) -> array[24]
```

Two `apply` signatures the assembler must dispatch over (extracted from current HEAD):

From `pfc_shaping/lt/model/shape_hourly.py:241-242`:
```
def apply(self, timestamps: pd.DatetimeIndex, calendar_df: pd.DataFrame,
          reference_date: pd.Timestamp | None = None) -> pd.Series: ...
```

From `pfc_shaping/lt/model/shape_hourly_mlp.py:207-213`:
```
def apply(self, timestamps: pd.DatetimeIndex, calendar_df: pd.DataFrame,
          reference_date: pd.Timestamp | None = None,
          outages_forecast: pd.DataFrame | None = None) -> pd.Series: ...
```

Discriminator: presence of the `outages_forecast` parameter in the signature.

The current `try/except TypeError` to be replaced (`assembler.py:280-286`):
```
# Pass outages_forecast + reference_date to ShapeHourly
try:
    f_H = self.sh.apply(idx, cal, reference_date=reference_date,
                        outages_forecast=outages_forecast)
except TypeError:
    # ShapeHourly (table) doesn't accept outages_forecast
    f_H = self.sh.apply(idx, cal, reference_date=reference_date)
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add read-only `factors_3d_` view to `ShapeHourly` (SHP-01 literal)</name>
  <files>pfc_shaping/lt/model/shape_hourly.py</files>
  <read_first>
    - pfc_shaping/lt/model/shape_hourly.py (UPDATED file from plan 05B-03 — note attribute structure, fit() pipeline, smoothing at line 147 and renormalisation at line 150)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-01..D-02 — literal SHP-01 satisfaction via view, status quo on internal storage)
    - .planning/REQUIREMENTS.md (SHP-01..SHP-03 — exact wording)
  </read_first>
  <behavior>
    - Test 1: `sh.factors_3d_[("Hiver","Ouvrable",12)] == sh.factors_[("Hiver","Ouvrable")][12]` exactly (no rounding, same float).
    - Test 2: `sh.factors_3d_` supports `len(...)` and equals `len(sh.factors_) * 24`.
    - Test 3: `iter(sh.factors_3d_)` yields all 3-tuples `(s, tj, h)` for `(s, tj) in sh.factors_` and `h in range(24)`.
    - Test 4: `sh.factors_3d_[("nope","nope",0)]` raises `KeyError`.
    - Test 5: `sh.factors_3d_[("Hiver","Ouvrable",24)]` raises `KeyError` (hour must be in `[0, 24)`).
    - Test 6: `sh.factors_3d_[("Hiver","Ouvrable",12)] = 99.0` raises `TypeError` (read-only).
    - Test 7: After populating `sh.factors_[("X","Y")] = np.zeros(24)` AFTER first access of `factors_3d_`, the view reflects the new cell on next access (the view is a live facade, not a snapshot — must be reflected so partial fits behave intuitively).
    - Test 8: `sh.factors_3d_` is accessible BEFORE `fit()` is called (returns empty mapping length 0 if `factors_` is empty).
  </behavior>
  <action>
    In `pfc_shaping/lt/model/shape_hourly.py`, define a small read-only `Mapping` subclass at module level (above the `ShapeHourly` class) and expose it as a `@property` on `ShapeHourly`.

    Concrete steps:

    1. Add to the module imports: `from collections.abc import Mapping` (alongside existing imports).
    2. Define a private class `_Factors3DView(Mapping):` near the top of the module, after constants. It accepts a reference to the parent `factors_: dict[(str, str), np.ndarray]` in `__init__` (store as `self._factors`).
       - `__getitem__(self, key)`:
         - Expect `key` to be a 3-tuple `(saison, type_jour, hour)`. If not a 3-tuple, raise `KeyError(key)`.
         - Validate `0 <= hour < 24` else raise `KeyError(key)`.
         - Look up `arr = self._factors[(saison, type_jour)]`; raise `KeyError(key)` if absent.
         - Return `float(arr[hour])`.
       - `__iter__(self)`: yield `(s, tj, h)` for `(s, tj) in self._factors for h in range(24)`.
       - `__len__(self)`: return `len(self._factors) * 24`.
       - Do NOT override `__setitem__` — `Mapping` is already read-only, but ALSO explicitly add `__setitem__(self, key, value)` raising `TypeError(f"{type(self).__name__} is read-only")` for clarity.
       - Add a `__contains__` that returns False for invalid keys (gracefully) so `(s, tj, h) in view` works without raising.
    3. In `ShapeHourly`, add:
       ```
       @property
       def factors_3d_(self) -> Mapping[tuple[str, str, int], float]:
           """Read-only 3D view on factors_ keyed by (saison, type_jour, hour). SHP-01 literal."""
           return _Factors3DView(self.factors_)
       ```
    4. Do NOT change the internal `self.factors_` structure. Do NOT touch the smoothing/normalisation pipeline.

    Note on liveness (per behavior Test 7): the view stores a reference to the SAME dict, so cells added later are visible automatically (no snapshot). This is the desired behavior.

    Do NOT change save/load — the view is computed on demand and is not persisted.
  </action>
  <verify>
    <automated>python -c "
import numpy as np
from collections.abc import Mapping
from pfc_shaping.lt.model.shape_hourly import ShapeHourly
sh = ShapeHourly()
arr = np.linspace(0.8, 1.2, 24); arr = arr / arr.mean()
sh.factors_[('Hiver','Ouvrable')] = arr
v = sh.factors_3d_
assert isinstance(v, Mapping)
assert v[('Hiver','Ouvrable',12)] == float(arr[12])
assert len(v) == 24
assert sum(1 for _ in iter(v)) == 24
try:
    _ = v[('nope','nope',0)]; assert False, 'expected KeyError'
except KeyError: pass
try:
    _ = v[('Hiver','Ouvrable',24)]; assert False, 'expected KeyError'
except KeyError: pass
try:
    v[('Hiver','Ouvrable',12)] = 99.0; assert False, 'expected TypeError'
except TypeError: pass
# Liveness
sh.factors_[('Ete','Samedi')] = arr
assert len(v) == 48
assert v[('Ete','Samedi',5)] == float(arr[5])
print('OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - Verify command above exits 0 and prints `OK`.
    - `grep -q "class _Factors3DView" pfc_shaping/lt/model/shape_hourly.py` exits 0.
    - `grep -q "def factors_3d_" pfc_shaping/lt/model/shape_hourly.py` exits 0 (property defined).
    - `grep -q "from collections.abc import Mapping" pfc_shaping/lt/model/shape_hourly.py` exits 0.
    - `python -c "from pfc_shaping.lt.model.shape_hourly import ShapeHourly; sh = ShapeHourly(); assert len(sh.factors_3d_) == 0"` exits 0.
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped`.
  </acceptance_criteria>
  <done>SHP-01 literal: `factors_3d_[(s, tj, h)]` returns the underlying float without copy; read-only enforced.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Replace `try/except TypeError` at assembler.py:280-286 with explicit signature-based capability check + operator log</name>
  <files>pfc_shaping/lt/model/assembler.py</files>
  <read_first>
    - pfc_shaping/lt/model/assembler.py (lines 1-60 for imports; lines 120-150 for `__init__`; lines 200-320 for the `build()` method context, focusing on the existing try/except at lines 280-286)
    - pfc_shaping/lt/model/shape_hourly.py (lines 241-258: `apply()` signature — does NOT have `outages_forecast`)
    - pfc_shaping/lt/model/shape_hourly_mlp.py (lines 207-220: `apply()` signature — DOES have `outages_forecast`)
    - .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md (D-13 — explicit capability check, no bug masking)
  </read_first>
  <behavior>
    - Test 1: When `self.sh` is an instance of `ShapeHourly`, the capability check decides NOT to pass `outages_forecast`. The call is `self.sh.apply(idx, cal, reference_date=reference_date)`.
    - Test 2: When `self.sh` is an instance of `ShapeHourlyMLP`, the capability check decides to PASS `outages_forecast`. The call is `self.sh.apply(idx, cal, reference_date=reference_date, outages_forecast=outages_forecast)`.
    - Test 3: A `TypeError` raised from INSIDE `self.sh.apply(...)` (e.g. a bug in some helper) is no longer silently swallowed and retried — it propagates to the caller of `assembler.build`. (Test by monkeypatching `self.sh.apply` to raise `TypeError("bug")` and asserting the call site re-raises.)
    - Test 4: The cached signature lookup does NOT recompute on every `build()` call when called repeatedly on the same assembler instance (use `functools.lru_cache` on a module-level helper, or compute in `__init__`, or cache as instance attribute on first build).
    - Test 5: For a third-party `self.sh` whose `apply` signature has neither `reference_date` nor `outages_forecast` (a hypothetical legacy stub), the check raises a clear `TypeError(f"self.sh.apply must accept reference_date; got signature {sig}")` — explicit, not a masked one.
    - Test 6: A single `logger.info` line is emitted at first dispatch (or `__init__`) naming the detected implementation, e.g. `"Detected sh=ShapeHourly — outages_forecast skipped"` for `ShapeHourly` or `"Detected sh=ShapeHourlyMLP — outages_forecast passed"` for the MLP variant. Repeated `build()` calls on the same assembler do NOT re-emit the log line.
  </behavior>
  <action>
    In `pfc_shaping/lt/model/assembler.py`:

    1. Add `import inspect` to the module imports if not already imported (check existing imports first).
    2. Add a module-level helper function near the other private helpers (e.g. near `_country_local_tz`):
       ```
       def _sh_apply_accepts_outages(sh_class: type) -> bool:
           """Return True iff sh_class.apply has an `outages_forecast` parameter."""
           sig = inspect.signature(sh_class.apply)
           if "reference_date" not in sig.parameters:
               raise TypeError(
                   f"{sh_class.__name__}.apply must accept reference_date; got signature {sig}"
               )
           return "outages_forecast" in sig.parameters
       ```
    3. In `PFCAssembler.__init__` (line ~124), AFTER the existing assignments, add an instance attribute that caches the decision per assembler instance:
       `self._sh_accepts_outages: bool = _sh_apply_accepts_outages(type(shape_hourly))`.
       Immediately after caching, emit a one-shot operator log line:
       `logger.info("Detected sh=%s — outages_forecast %s", type(shape_hourly).__name__, "passed" if self._sh_accepts_outages else "skipped")`.
       This single emit at __init__ guarantees no per-build spam.
    4. Replace the existing block at `assembler.py:280-286` (the `try: ... except TypeError: ...` around `self.sh.apply(...)`) with:
       ```
       # Capability check (replaces former try/except TypeError — see D-13).
       if self._sh_accepts_outages:
           f_H = self.sh.apply(idx, cal, reference_date=reference_date,
                               outages_forecast=outages_forecast)
       else:
           f_H = self.sh.apply(idx, cal, reference_date=reference_date)
       ```
    5. Keep the surrounding comments / cosmetic separators intact. Do NOT widen the diff.

    Do NOT modify the `ShapeHourly` or `ShapeHourlyMLP` apply signatures. Do NOT introduce a new public constructor argument. Do NOT introduce a new dependency beyond `inspect` (stdlib).

    Verify locally that no other callsite passes `outages_forecast=` to `self.sh.apply` in the file (`grep -n "sh.apply\|sh\.apply" pfc_shaping/lt/model/assembler.py` should show only the one replaced site).
  </action>
  <verify>
    <automated>python -c "
import inspect, logging
from pfc_shaping.lt.model.assembler import _sh_apply_accepts_outages
from pfc_shaping.lt.model.shape_hourly import ShapeHourly
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
assert _sh_apply_accepts_outages(ShapeHourly) is False
assert _sh_apply_accepts_outages(ShapeHourlyMLP) is True
class Stub:
    def apply(self, timestamps, calendar_df): pass
try:
    _sh_apply_accepts_outages(Stub); assert False, 'expected TypeError'
except TypeError: pass
print('OK')
"</automated>
  </verify>
  <acceptance_criteria>
    - Verify command above exits 0 and prints `OK`.
    - `grep -c "except TypeError" pfc_shaping/lt/model/assembler.py` decreases by ≥ 1 vs `git show HEAD:pfc_shaping/lt/model/assembler.py | grep -c "except TypeError"`. (No silent swallowing of TypeErrors around `self.sh.apply`.)
    - `grep -q "_sh_apply_accepts_outages" pfc_shaping/lt/model/assembler.py` exits 0.
    - `grep -q "_sh_accepts_outages" pfc_shaping/lt/model/assembler.py` exits 0 (cached on instance).
    - `grep -q "Detected sh=" pfc_shaping/lt/model/assembler.py` exits 0 (one-shot operator log line present).
    - `pytest tests/ -x` exits 0 reporting `142 passed, 4 skipped`.
  </acceptance_criteria>
  <done>The try/except TypeError at line 284 is replaced by an explicit signature check that produces identical routing decisions, emits a one-shot operator log naming the detected implementation, and unmasks any real TypeError.</done>
</task>

</tasks>

<verification>
- `pytest tests/ -x` exits 0 with `142 passed, 4 skipped`.
- SHP-01 literal accessible via `factors_3d_`.
- No `try/except TypeError` left around `self.sh.apply()` in assembler.
- One-shot `logger.info("Detected sh=... — outages_forecast ...")` line present for operator audits.
- Numerical output of `assembler.build` is UNCHANGED from baseline (final verification in plan 05B-05 via `assert_frame_equal(..., check_exact=False, atol=1e-12, rtol=0)`).
</verification>

<success_criteria>
- SHP-01 satisfied literally via `factors_3d_` read-only view.
- D-13 satisfied: brittle try/except replaced by explicit capability check.
- Operator-visible logging in place for production audits.
- 142 existing tests remain green.
- Zero numerical change vs baseline.
</success_criteria>

<output>
Create `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-04-SUMMARY.md` when done.
</output>
