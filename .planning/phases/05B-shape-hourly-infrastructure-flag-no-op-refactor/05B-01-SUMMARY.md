---
phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
plan: "01"
subsystem: pfc-lt-fixtures
tags: [fixtures, baseline, regression, no-op-refactor, shape-hourly]
dependency_graph:
  requires: []
  provides:
    - tests/fixtures/baseline_pfc_seed42.parquet
    - tests/fixtures/_generate_baseline.py
    - tests/fixtures/README.md
  affects:
    - plans 05B-02..05B-05 (must be transitive ancestors of this commit)
tech_stack:
  added: []
  patterns:
    - synthetic 15-min EPEX fixture via numpy seed=42
    - numerical-equality contract (atol=1e-12, not byte-level) for parquet regression
key_files:
  created:
    - tests/fixtures/_generate_baseline.py
    - tests/fixtures/baseline_pfc_seed42.parquet
    - tests/fixtures/README.md
  modified: []
decisions:
  - "Numerical-equality contract (atol=1e-12, rtol=0) chosen over byte-level parquet equivalence (not guaranteed across pandas/pyarrow/Python versions)"
  - "ShapeIntraday fitted on same synthetic EPEX (seed=42) to satisfy PFCAssembler.__init__ requirement; documented as workaround in script docstring"
  - "All three files committed in one atomic commit BEFORE any 5bis-A logic changes"
metrics:
  duration: "~7 minutes"
  completed: "2026-05-18T19:55:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 0
---

# Phase 05B Plan 01: Freeze baseline_pfc_seed42 for no-op refactor regression

## One-liner

Frozen PFC baseline snapshot (seed=42, Cal'27 1-month, synthetic EPEX) committed ahead of all 5bis-A code changes to serve as numerical-equality regression reference.

## What Was Built

### Task 1: Deterministic baseline generator script

Created `tests/fixtures/_generate_baseline.py` — a standalone runnable Python script (CLI: `python tests/fixtures/_generate_baseline.py`) that:
- Sets `numpy.random.seed(42)` and `random.seed(42)` at the top of `main()`
- Builds a synthetic 15-min EPEX DataFrame covering 3 calendar years (2022-01-01..2024-12-31 UTC) with structural model: `30 + 10*sin(2pi*hour/24) + 5*sin(2pi*doy/365) + N(0,2)` clipped to [-50, 200]
- Builds `calendar_df = enrich_15min_index(epex_df.index, country="CH")`
- Fits `ShapeHourly().fit(epex_df, calendar_df)` (no hydro_df)
- Fits `ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=calendar_df)` to satisfy PFCAssembler requirement
- Calls `assembler.build(base_prices={"2027": 80.0}, start_date="2027-01-01", horizon_days=31, reference_date=pd.Timestamp("2026-05-18", tz="UTC"), country="CH")`
- Writes to `tests/fixtures/baseline_pfc_seed42.parquet`

Verified reproducible at `atol=1e-12, rtol=0` with identical columns/dtypes/index.

### Task 2: Generate baseline parquet + write README

- Generated `tests/fixtures/baseline_pfc_seed42.parquet`: 2976 rows (31 days x 96 quarters), 13 columns including `price_shape`, `f_S`, `f_W`, `f_H`, `f_Q`, `f_WV`
- Created `tests/fixtures/README.md` documenting: purpose, source SHA (3dc8552 + reference main@28dfd65), regeneration policy, schema, and numerical-equality contract

Both committed in a single atomic commit `9cc959b` BEFORE any 5bis-A code changes.

## Verification Results

- `python tests/fixtures/_generate_baseline.py` exits 0, prints "Wrote baseline: rows=2976 ..."
- `assert {'price_shape','f_S','f_W','f_H','f_Q','f_WV'}.issubset(df.columns)` passes
- Reproducibility at `atol=1e-12, rtol=0`: PASS
- `grep -L "data/.*\.xlsx\|H:\\" tests/fixtures/_generate_baseline.py`: PASS (no real-data paths)
- `grep -q "main@28dfd65\|28dfd65" tests/fixtures/README.md`: PASS
- `grep -q "numerically identical" tests/fixtures/README.md`: PASS
- Row count: 2976 > 2000 minimum
- pytest suite: 143 passed, 3 skipped (unchanged from pre-plan baseline)

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1+2 | Write generator + generate parquet + write README | 9cc959b | tests/fixtures/_generate_baseline.py, tests/fixtures/baseline_pfc_seed42.parquet, tests/fixtures/README.md |

## Deviations from Plan

### Auto-fixed Issues

None.

### Workaround: ShapeIntraday stub

The plan spec says: "Build `PFCAssembler(shape_hourly=sh, shape_intraday=None, ...)` — pass `None` for any optional component the assembler accepts."

However, `PFCAssembler.__init__` stores `self.si = shape_intraday` and `build()` calls `self.si.apply(...)` directly with no None guard. Passing `None` would crash at runtime.

**Fix applied (Rule 1 — Bug):** Fit a minimal `ShapeIntraday` on the same synthetic EPEX using `ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=calendar_df)`. This uses only the Layer 1 deterministic path (no exogenous features) and is fully synthetic + deterministic under seed=42. The workaround is documented in the script's module docstring.

This does not affect the "no real-data dependency" constraint or the numerical reproducibility.

## Parent-of-Commit Invariant

Commit `9cc959b` (baseline parquet introduction) is verified as an ancestor that Plans 02-05 MUST build upon. The baseline was committed before any pfc_shaping/* modifications on this branch.

```bash
git log --diff-filter=A --format=%H -- tests/fixtures/baseline_pfc_seed42.parquet | head -1
# => 9cc959bb57c515e32096ad840af94631b9908067
```

## Known Stubs

None — the baseline parquet contains real (synthetic) computed PFC values, not hardcoded placeholders.

## Threat Flags

None — this plan adds test fixtures only (read-only parquet + generator script + README). No new network endpoints, auth paths, or trust boundary changes.

## Self-Check: PASS

- [x] `tests/fixtures/_generate_baseline.py` exists
- [x] `tests/fixtures/baseline_pfc_seed42.parquet` exists
- [x] `tests/fixtures/README.md` exists
- [x] Commit `9cc959b` exists: `git log --oneline --all | grep 9cc959b` returns a match
- [x] No file deletions in the commit
- [x] Test suite unchanged: 143 passed, 3 skipped
