# tests/fixtures — Frozen Regression Fixtures

## baseline_pfc_seed42.parquet

### Purpose

Frozen reference for `test_baseline_regression` (added in plan 05B-05).

The regression contract is **numerically identical** — NOT parquet byte-equivalence:

```python
pandas.testing.assert_frame_equal(
    build(flag=OFF, seed=42),
    pd.read_parquet("tests/fixtures/baseline_pfc_seed42.parquet"),
    check_exact=False,
    atol=1e-12,
    rtol=0,
)
assert list(a.columns) == list(b.columns)
assert a.dtypes.to_dict() == b.dtypes.to_dict()
assert a.index.equals(b.index)
```

Parquet byte equivalence is NOT guaranteed across pandas/pyarrow/Python versions.
Only the numerical content and schema (columns, dtypes, index, sort order) must
be stable.

The test is parametrized: both `build(flag=OFF)` and `build(flag=ON)` must equal
this baseline (5bis-A no-op contract: flag ON == flag OFF numerically).

If the tighter `atol=1e-12, rtol=0` proves unreliable in CI due to
pandas/pyarrow patch-version drift, the test may fall back to `atol=1e-10`,
in which case the fallback MUST be documented inline in the test.

### Source SHA

This fixture was generated from commit:

    HEAD == 3dc8552c7fc6bd45dbebf27f3438f053aa717974 (branch: claude/clean-lt-ct-integration)

Verified: `git diff --stat 28dfd65..HEAD -- pfc_shaping/ tests/` is empty,
confirming code equivalence to `main@28dfd65` for all pfc_shaping/* and tests/*.

Reference SHA for pfc_shaping/* code state: **28dfd65**

### Regeneration Policy

THIS FIXTURE MUST NOT BE REGENERATED LIGHTLY.

Any PR that modifies this fixture's contents MUST:
1. Include a justification block in the PR description explaining why the
   baseline changed (behavioral intent, not accident).
2. Bump the "Regeneration history" table below.
3. Re-run the generator from clean state and verify reproducibility:
   ```bash
   python tests/fixtures/_generate_baseline.py
   cp tests/fixtures/baseline_pfc_sed42.parquet /tmp/baseline_a.parquet
   python tests/fixtures/_generate_baseline.py
   python -c "
   import pandas as pd
   a = pd.read_parquet('/tmp/baseline_a.parquet')
   b = pd.read_parquet('tests/fixtures/baseline_pfc_seed42.parquet')
   pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=0)
   assert list(a.columns) == list(b.columns)
   assert a.dtypes.to_dict() == b.dtypes.to_dict()
   assert a.index.equals(b.index)
   print('OK')
   "
   ```

### Regeneration Command

```bash
python tests/fixtures/_generate_baseline.py
```

### Schema

Generated from: `assembler.build(base_prices={"2027": 80.0}, start_date="2027-01-01", horizon_days=31, reference_date=pd.Timestamp("2026-05-18", tz="UTC"), country="CH")`

Shape: (2976, 13)  — 31 days × 96 quarters/day = 2 976 rows

| Column        | dtype   | Description                                    |
|---------------|---------|------------------------------------------------|
| price_shape   | float64 | Final PFC price (€/MWh)                        |
| B             | float64 | Base price level from forwards                 |
| f_S           | float64 | Seasonal monthly factor (mean=1)               |
| f_W           | float64 | Day-of-week factor (mean=1)                    |
| f_H           | float64 | Hourly shape factor (mean=1)                   |
| f_Q           | float64 | 15-min intraday factor (mean=1)                |
| f_WV          | float64 | Water value correction (1.0 = neutral)         |
| f_bridge      | float64 | Near-term bridge factor                        |
| profile_type  | object  | Horizon bucket ("Y+2/Y+3", "M+7..M+12", etc.) |
| confidence    | float64 | Model confidence score [0, 1]                  |
| calibrated    | bool    | Whether arbitrage-free calibration was applied |
| p10           | float64 | 10th percentile (bootstrap IC)                 |
| p90           | float64 | 90th percentile (bootstrap IC)                 |

Index: DatetimeIndex, UTC, freq≈15min

### Equivalence Contract

The contract is **numerical** (not byte-level):

- `assert_frame_equal(check_exact=False, atol=1e-12, rtol=0)` on all float columns
- Identical columns (name and order)
- Identical dtypes
- Identical index (values, tz, freq)
- Identical sort order

Parquet byte output varies across pandas/pyarrow/Python version combinations
and is NOT a valid regression criterion.

### Regeneration History

| Date       | SHA     | Reason                                              |
|------------|---------|-----------------------------------------------------|
| 2026-05-18 | 3dc8552 | Initial freeze from main@28dfd65 for 5bis-A baseline |

## _generate_baseline.py — Reusable entry point `build_pfc(seed, flag)`

Added in plan 05B-05: `_generate_baseline.py` now exposes a reusable function:

```python
from tests.fixtures._generate_baseline import build_pfc

df = build_pfc(seed=42, flag=False)  # → same as committed baseline
df = build_pfc(seed=42, flag=True)   # → must equal baseline (5bis-A no-op)
```

The `flag` parameter sets `ShapeHourly(use_seasonal_hourly=flag)` but has zero
numerical effect in Phase 5bis-A (plumbing only — see `test_no_hidden_behavior_branch`).

## shape_hourly_legacy.parquet + f_W_legacy.parquet

Generated by `_generate_legacy_fixture.py` (plan 05B-05).

These are **FORMAT-COMPATIBLE MOCK** artifacts representing pre-5bis-A save() output —
they do NOT have a `.meta.parquet` sidecar, which is the whole point: they exercise
the legacy-compat warning path in `ShapeHourly.load()`.

Schema matches `ShapeHourly.save()` at shape_hourly.py lines 443-455 (verified at HEAD).

**NOT verbatim artifacts from `main@28dfd65`** — hand-crafted to match schema.
See generator docstring for caveat details and instructions for producing a verbatim
legacy artifact if ever needed.

## Parent-hour-v1 successor baselines

The Phase 14 P1 intrahour-neutrality correction intentionally supersedes the
historical numerical baseline without overwriting it. Horizon-dependent `f_H`
is now evaluated at the UTC parent-hour boundary, so `f_H` is exactly constant
across the four quarter-hours and `f_Q` remains the sole intrahour shape layer.

Frozen successors generated on 2026-07-31:

| Fixture | flag | SHA-256 | Size |
|---|---:|---|---:|
| `baseline_pfc_seed42_parent_hour_v1.parquet` | OFF | `852a64aa3d9b278c0e949ba276b58690f5fd05d201664f0b7bc1f9b866967afa` | 99,992 bytes |
| `baseline_pfc_seed42_bowl_parent_hour_v1.parquet` | ON | `3dc1894d6de5d26e4e87a7be2df174ae008e8ac22d9f8c2a63c458df6b260d7f` | 99,992 bytes |

The transition generator rejects index/schema drift, rejects changes outside
`price_shape`, `f_H` and floating-point-ULP noise in `f_Q`, and requires the
maximum within-parent-hour range of `f_H` to be exactly zero.

```powershell
build\pytest-runtime-v1\python.exe -I -B -m tests.fixtures._generate_parent_hour_baselines
```

The historical `baseline_pfc_seed42*.parquet` files remain unchanged for audit
history and Phase 5 marker semantics.
