---
phase: 10-pfc-fmv-quality-scorecard
reviewed: 2026-05-21T00:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - pfc_shaping/requirements.txt
  - pfc_shaping/validation/__init__.py
  - pfc_shaping/validation/block_masks.py
  - pfc_shaping/validation/christoffersen.py
  - pfc_shaping/validation/dm_test.py
  - pfc_shaping/validation/scorecard.py
  - pfc_shaping/validation/structural_tests.py
  - tests/test_phase10_dm.py
  - tests/test_phase10_empirical.py
  - tests/test_phase10_hildmann.py
  - tests/test_phase10_infra.py
  - tests/test_phase10_probabilistic.py
findings:
  critical: 1
  warning: 9
  info: 6
  total: 16
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-05-21
**Depth:** standard
**Files Reviewed:** 12 (1 deps manifest, 6 source modules, 5 test modules)
**Status:** issues_found

## Summary

Phase 10 adds the `pfc_shaping/validation/` package implementing the 5-pillar
PFC FMV Quality Scorecard (Hildmann structural tests, KYOS empirical KPIs,
Christoffersen IC80 coverage, Diebold-Mariano vs baselines, peer-review SOTA
table). The code is generally well-documented with extensive docstrings,
explicit degenerate-case handling, and ex-ante frozen thresholds traceable to
NOTES/RESEARCH artifacts. Test coverage is broad: unit tests for each math
primitive plus integration tests with mock fixtures.

The most serious defect found is a **timezone-boundary inconsistency in the
Pillar 1 arbitrage-freeness gate** (`_period_mask` uses UTC year/month/quarter
to bucket a UTC index, while forwards are quoted against local-calendar
delivery periods). With the SC#1 gate tolerance of 0.01 €/MWh, the
~1-hour-per-boundary misclassification can plausibly produce a false-positive
gate failure on real data. Several secondary defects involve dead/wasted
compute paths in `run_scorecard_full` and `run_scorecard_pillar_1`,
docstring/implementation mismatches in `derive_forwards_from_epex_hist`, a
weak test that fails to assert the documented behavior under `var_d <= 0`,
and Mock-CI infrastructure that builds 4 real PFCs only to discard them.

No security vulnerabilities, hardcoded secrets, injection vectors, or unsafe
deserialization were detected. The IC95 explicit-rejection guard is a model
of defensive coding worth preserving as a pattern.

---

## Critical Issues

### CR-01: `_period_mask` uses UTC year/month/quarter on a UTC index, but forwards reference local-calendar delivery periods

**File:** `pfc_shaping/validation/structural_tests.py:104-141`
**Issue:**
`_period_mask` is invoked from `test_arb_free` with the PFC's UTC index
(`pfc.index`) and groups using `idx_utc.year`, `idx_utc.month`,
`idx_utc.quarter`. The PFC is canonically UTC-indexed, but the EEX/EPEX
forward contracts (Cal/Q/M base) reference **calendar periods in local time**
(Europe/Zurich), not UTC. This shifts roughly one hour per Cal/Q/M boundary
into the wrong bucket (e.g. 2024-12-31 23:00 UTC is in calendar 2024 in UTC
but in calendar 2025 in Europe/Zurich, i.e. CET → 00:00 2025-01-01).

For a monthly forward (≈720 hours), one misbucketed hour at the boundary
contributes `Δ_price / 720` €/MWh of bias to `|mean(PFC | month) − forward|`.
With month-to-month jumps of even 5-10 €/MWh in 2024-2025 real EPEX, this
yields ≈0.007-0.014 €/MWh of deviation purely from the bucketing bug —
already at or above the 0.01 €/MWh `tol` of `test_arb_free`. The error
compounds at quarter and Cal boundaries.

`block_masks.py` correctly uses `tz_convert("Europe/Zurich")` for the
trader-semantic hour/weekday/month buckets and even calls out DST-safety
explicitly. The same convention is missing here and silently breaks the
SC#1 gate criterion most sensitive to it (arb-free at tol=0.01).

`test_continuity` already adopts the correct convention (line 456:
`idx_local = pfc.index.tz_convert("Europe/Zurich")` then `.to_period("M")`)
— this proves the team knows the right pattern; `_period_mask` is an
oversight.

**Fix:**
```python
def _period_mask(
    idx_utc: pd.DatetimeIndex,
    product_type: str,
    year: int,
    sub: int | None,
) -> np.ndarray:
    if idx_utc.tz is None:
        raise ValueError("_period_mask: idx must be tz-aware")
    # Forwards bucket by LOCAL calendar (Europe/Zurich trader convention).
    idx_local = idx_utc.tz_convert("Europe/Zurich")
    if product_type == "Cal":
        return np.asarray(idx_local.year == year, dtype=bool)
    if product_type == "Quarter":
        return np.asarray(
            (idx_local.year == year) & (idx_local.quarter == sub), dtype=bool
        )
    if product_type == "Month":
        return np.asarray(
            (idx_local.year == year) & (idx_local.month == sub), dtype=bool
        )
    return np.zeros(len(idx_utc), dtype=bool)
```
Add a unit test that constructs a PFC with a +10 €/MWh step at the local
2024→2025 boundary and verifies that the Cal-2024 mean falls within tol
of a forward set to its local-bucket true mean.

---

## Warnings

### WR-01: `run_scorecard_pillar_1` mock branch builds 4 PFCs via `build_one` and then discards every value

**File:** `pfc_shaping/validation/scorecard.py:509-528, 544-555`
**Issue:**
In the mock branch (`epex_source == "mock"`), the per-vintage loop
(lines 509-528) invokes `build_one(...)` four times and writes each result
to a parquet cache file. Immediately afterwards (line 544-550) the function
discards `pfc_per_vintage` entirely and evaluates the 4 Hildmann tests
against `_synth_pfc_for_mock(...)`. The real-assembler outputs never feed
into `results`.

Consequences:
1. The 4 `build_one` invocations represent the dominant runtime cost of
   the mock fixture (the docstring at line 415 even budgets ≤5 min for
   exactly this work) — yet none of the output is exercised by the
   assertions.
2. The CI's mock SC#1 tests in `tests/test_phase10_hildmann.py` therefore
   provide *zero* end-to-end coverage of the `ShapeHourly` →
   `ShapeIntraday` → `PFCAssembler` chain when `epex_source='mock'`. The
   tests at most prove that `_synth_pfc_for_mock` passes the 4 tests by
   construction, which is tautological.
3. If the goal is to keep no-crash coverage of `build_one`, that intent
   should be expressed by asserting non-empty / no-NaN on `pfc_v`. As
   written, a silent `build_one` regression (e.g. all NaN) is invisible.

**Fix:**
Either remove the wasted loop entirely (lines 509-528 in the mock branch
and the cache directory creation) or assert post-conditions on each
`pfc_v` (e.g. `assert pfc_v["price_shape"].notna().all()` and
`assert pfc_v["price_shape"].abs().max() < 1000`). Document explicitly in
the docstring that mock-mode does not exercise the assembler when the
synth-PFC short-circuit is taken.

---

### WR-02: `run_scorecard_full` Pillar 4 has dead variables and a no-op inner loop computing `fwds_v` then `pass`-ing

**File:** `pfc_shaping/validation/scorecard.py:1567-1587`
**Issue:**
Two defects in the Pillar 4 build loop:

1. Line 1568 initialises `baseline_pieces: dict[str, list[pd.Series]] = {b: [] for b in BASELINES}` but the variable is never read or written again. The actual aggregation uses `baseline_pieces_b` at line 1595 inside a different scope.

2. Lines 1579-1585 contain a triple-nested loop body whose only statement is
   `pass`:
   ```python
   fwds_v = _forwards_for_vintage(forwards_df, vintage, epex_hist)
   for bname in BASELINES:
       for bloc in ALL_BLOCKS:
           # Pre-build per bloc happens below ; here we collect
           # full-window broadcast (scalar over realised_window).
           # We will mask by bloc inside compute_pillar4_dm.
           pass
   ```
   `fwds_v` is computed (an O(hist) operation) and then thrown away, then
   recomputed identically inside the inner bloc/baseline loop at line 1598.
   Per (horizon × vintage) iteration this wastes one `_forwards_for_vintage`
   call (which does a parquet lookup + dict-build) plus 15 useless inner
   iterations (3 baselines × 5 blocs).

For a full real run (4 configs × 5 horizons × 24 vintages = 480
iterations of the outer block), this is ≈480 wasted forwards lookups and
7200 dead inner iterations. Doesn't change correctness but obscures intent
and is exactly the kind of cruft that breaks during refactor.

**Fix:**
Delete both blocks:
```python
# Remove line 1568 entirely (baseline_pieces dict, unused)
# Replace lines 1578-1587 with a comment explaining that fwds_v is
# computed inside the bloc loop below for per-bloc context.
```
After deletion, verify `pytest tests/test_phase10_*.py` still passes
(should — the variables are dead).

---

### WR-03: `derive_forwards_from_epex_hist` quarter loop iterates over `vintage.year + 4` instead of stopping at horizon

**File:** `pfc_shaping/validation/scorecard.py:326-335`
**Issue:**
The yearly loop (lines 318-323) iterates `offset in (1, 2, 3)` correctly
producing keys for years +1/+2/+3. The quarter loop, however, uses
`range(vintage.year + 1, vintage.year + 1 + (horizon_days // 365) + 1)`.

With `horizon_days = 3*365`, `horizon_days // 365 == 3`, so the range is
`[vintage.year + 1, vintage.year + 5)` — i.e. years +1, +2, +3, **+4**.
The `+4` quarters are filtered by `q_start > end` at line 330, so for
Q1/Q2/Q3 of vintage.year+4 the condition `q_start > vintage + 1095 days`
fails for at least Q1 (typically `vintage.year+3` end + ~3 months still
within end), and the loop emits Q-keys for years beyond what `Y+1/Y+2`
consumers expect.

Concretely, for `vintage = 2024-06-28`, `end = 2027-06-27`. The loop emits
keys for years 2025, 2026, 2027, 2028. For 2028, `q_start = 2028-01-01` >
`end = 2027-06-27` → skipped. For 2027, Q1 (2027-01-01) and Q2 (2027-04-01)
pass the guard and produce keys `2027-Q1`, `2027-Q2`. Comparable Cal-2027
key is missing because the yearly loop only emits Y+1, Y+2, Y+3 (so the
yearly loop already covers 2027 actually, so this is OK at Y level). But
the off-by-one comment + the unnecessary `+1` make the intent murky.

In addition, the monthly loop uses `pd.date_range(vintage, end, freq="MS")`,
which generates months from the **next** "MS" anchor after `vintage`
through `end`. For `vintage = 2024-06-28`, the first MS is 2024-07-01 →
months 2024-07 through 2027-07, i.e. **37 months** rather than the 36
implied by horizon. Minor over-generation; harmless because consumers
look up by key.

**Fix:**
Tighten the quarter loop to the same set of years as the yearly loop:
```python
for y in (vintage.year + 1, vintage.year + 2, vintage.year + 3):
    for q in (1, 2, 3, 4):
        q_start = pd.Timestamp(f"{y}-{(q-1)*3 + 1:02d}-01", tz="UTC")
        if q_start > end:
            continue
        mask = hist_q == q
        if mask.sum() == 0:
            continue
        out[f"{y:04d}-Q{q}"] = float(hist[mask].mean())
```
Or parametrize the year range explicitly off `horizon_days // 365` and
document the off-by-one rationale.

---

### WR-04: `test_var_d_negative_fallback` cannot trigger the documented `var_d < 0` fallback because the constructed input collapses to constant `d`

**File:** `tests/test_phase10_dm.py:116-141`
**Issue:**
The test claims to "force potentiellement var_d négative en HAC". The
construction:
```python
d_oscillating = np.tile([-1.0, 1.0], n // 2)  # n=30
errors_a = d_oscillating * 0.5     # |a| = 0.5 constant
errors_b = d_oscillating * 0.5 + d_oscillating  # = d_oscillating*1.5
                                              # |b| = 1.5 constant
```
yields `|a| = 0.5` everywhere and `|b| = 1.5` everywhere, so the loss
differential `d = |a| - |b| = -1.0` is a *constant* sequence with zero
autocovariance at every lag (`gammas[0] = var_d = 0`). This drives the
`if var_d <= 0` branch into the second fallback (`gammas[0]` is still 0)
and returns `degenerate=True` — but not because of negative HAC variance.
The test acknowledges this with the `if not res["degenerate"]: ... else:
...` split, so the test passes but verifies **neither** the documented
fallback path (var_d ≤ 0 from autocovariance summation) nor the success
path.

**Fix:**
Construct a `d` with negative dominant lag-1 autocovariance and positive
`gammas[0]`. For example:
```python
n = 30
rng = np.random.RandomState(7)
# AR(-0.8) process induces strong negative lag-1 autocov ; HAC sum
# at moderate lags can go negative before the Bartlett rolloff.
d = np.zeros(n)
d[0] = rng.normal()
for i in range(1, n):
    d[i] = -0.8 * d[i-1] + rng.normal(0, 0.1)
# Map back to (errors_a, errors_b) such that |a|-|b| = d
errors_b = np.full(n, 2.0)
errors_a = errors_b + d  # |a| - |b| ≈ d when errors_b > 0
res = diebold_mariano(errors_a, errors_b, h=10, loss="mae")
assert "degenerate" in res
```
Then add an explicit assertion that the fallback path triggers (e.g. by
patching `logger.warning` and asserting it was called, or by checking
`res["var_d"]` equals `gammas[0]`).

---

### WR-05: `_synth_pfc_for_mock` ignores its `forwards_asof` argument's prices entirely, defeating the contract

**File:** `pfc_shaping/validation/scorecard.py:567-670`
**Issue:**
The docstring (line 584-586) acknowledges "Forwards `forwards_asof` IGNORÉS
en mock", but this means the function's signature lies: it takes
`forwards_asof` and returns synth forwards constructed from its own PFC.
Several issues compound:

1. The caller (`run_scorecard_pillar_1` line 548-550) cannot pass meaningful
   forwards through the mock pipeline — any test that wants to exercise
   the mock SC#1 gate with non-trivial forwards is impossible without
   bypassing the helper.
2. The `seed` parameter (line 595) is declared but never used. The
   docstring admits this ("réservé pour future extension, non utilisé ici
   car déterministe"), but a parameter that's documented-as-unused but
   silently accepted is a refactor trap.
3. The function reads `forwards_asof.keys()` only to discover the window
   bounds (first/last month) — coupling the *time scope* of the synth PFC
   to whatever month keys happen to be in `forwards_asof`. If the caller
   builds forwards with a different schema (e.g. only Y+N keys), the
   helper falls back to `range(1, 13)` for 2024 silently (line 609). No
   error, no warning — and the synth PFC will not match any real-data
   window.

**Fix:**
Either (a) honor the input forwards by *adjusting the synth PFC level per
period* to match them while preserving the structural properties, or (b)
remove `forwards_asof` from the signature and accept explicit
`(start, end)` plus a list of keys to synthesize. Drop the unused `seed`
parameter or wire it into the `np.convolve` kernel choice.

If the design intent is irreversible ("mock CI tests are tautological
proof of code-path coverage only"), make that contract explicit by
renaming the helper `_synth_passing_fixture(...)` and removing the
misleading argument.

---

### WR-06: `test_arb_free` silently skips keys whose `_period_mask` returns empty, which can mask Cal/Q/M coverage gaps

**File:** `pfc_shaping/validation/structural_tests.py:188-219`
**Issue:**
The loop (lines 188-209) catches three "skip" scenarios:
- `parse_key(key)` raises ValueError (line 192) → skipped.
- `_period_mask` returns all-False (line 198) → skipped.
- Mask is non-empty → computed.

Skipped keys are stored in `details["_skipped"]` but the function returns
`passed = max_dev < tol` regardless of how many keys were skipped. This
means a PFC that covers only 6 months of a 24-month forwards horizon will
have 18 keys "silently skipped" while the 6 reproducible months pass — and
SC#1 reports a green tick despite 75% of the contract universe being
untested.

In particular `_period_mask` returning empty for a Cal year that's
*entirely outside* the PFC window is treated identically to a Cal year
*partially missing data within the window*. The former is benign; the
latter is a real coverage gap.

**Fix:**
Add a coverage-aware threshold:
```python
n_keys = len(forwards)
n_evaluated = len(details) - (1 if "_skipped" in details else 0)
if n_keys > 0 and n_evaluated == 0:
    return TestResult(
        passed=False,
        observed=float("nan"),
        threshold=tol,
        details={**details, "reason": "no_keys_evaluable", "degenerate": True},
    )
```
Optionally emit a warning when `len(skipped) / n_keys > 0.5`.

---

### WR-07: `lr_unconditional_coverage` degenerate-bucket flags `x==0 OR x==n` as degenerate, which throws away the *most informative* coverage signal

**File:** `pfc_shaping/validation/christoffersen.py:91-102`
**Issue:**
The guard correctly avoids `log(0)`. However the degenerate flag is too
strong: `x == 0` with large `n` is an *extremely* strong signal of
overcoverage (perfect IC, no violations observed) — testably significant
at the 5% level for any `n >= 14` at `p = 0.20` (since `(1-p)^n =
0.8^14 ≈ 0.044`). Similarly `x == n` is a strong signal of total
miscalibration. Returning `degenerate=True` and `p_value=NaN` silently
discards this evidence, and downstream `compute_pillar3_coverage` will
report nothing actionable.

The RESEARCH §Pattern 3 reference is correct that LR_uc itself blows up
at the boundary, but the standard remediation is a continuity correction
(replace `x=0` with `x=0.5`, `x=n` with `x=n-0.5`) rather than skipping.
Alternatively, when `x == 0`, use the exact binomial p-value directly:
```python
from scipy.stats import binom
if x == 0:
    p_exact = (1 - p) ** n  # P(X=0 | nominal)
    return {..., "p_value": float(p_exact),
            "degenerate": False, "method": "binomial_exact"}
```

As coded, a perfectly conservative PFC (zero IC violations) and a
catastrophically wrong PFC (every observation outside IC) both produce
identical NaN output. SC#1 gate evaluators reading the parquet cannot
distinguish them.

**Fix:**
Replace the `x in {0, n}` branch with a continuity correction (Haldane-
Anscombe: `x' = x + 0.5`, `n' = n + 1`) or compute the exact binomial
p-value for the boundary case. Update the test
`test_degenerate_x_in_extremes` to reflect the new contract (either
explicit NaN-via-correction documentation, or finite p_value from exact
binomial).

---

### WR-08: `BlockSummerSolarBowl` and `BlockWinterEveningPeak` use month boundaries that disagree with the docstrings on `≥`/`≤` semantics for the season test

**File:** `pfc_shaping/validation/block_masks.py:96-116`
**Issue:**
The `BlockSummerSolarBowl` docstring says "mai-août" (May-August). The
implementation `(idx_local.month >= 5) & (idx_local.month <= 8)` is
correct (months 5,6,7,8 → 4 months).

But `BlockWinterEveningPeak` docstring says "nov-fév" (Nov-Feb). The
implementation `(idx_local.month >= 11) | (idx_local.month <= 2)` produces
months 11, 12, 1, 2 — also correct (4 months).

These are correct. However the test
`test_block_winter_evening_peak_seasonality`
(`tests/test_phase10_infra.py:139-152`) only asserts that "January has
some TRUE" and "June has none TRUE". It does NOT verify that the bloc
correctly handles the wrap-around (e.g. that February still triggers but
March does not, or that October does not trigger but November does). A
regression that drops the wrap-around (e.g. someone "simplifies" to
`(month >= 11) & (month <= 2)` which is `False` for every month) would
not be caught by the existing test.

Similarly `BlockSummerSolarBowl` is tested for April/September=False and
June h=12=True, but the boundary months May, August are not asserted.

**Fix:**
Strengthen the seasonal tests:
```python
def test_block_winter_evening_peak_wraparound():
    from pfc_shaping.validation.block_masks import BlockWinterEveningPeak
    bloc = BlockWinterEveningPeak()
    for month, expected_any in [(11, True), (12, True), (1, True),
                                 (2, True), (3, False), (10, False)]:
        idx = pd.date_range(
            f"2024-{month:02d}-01", periods=24*7, freq="1h", tz="UTC"
        )
        mask = bloc.apply(idx)
        assert mask.any() == expected_any, (
            f"month {month}: expected any={expected_any}, got {mask.any()}"
        )
```

---

### WR-09: `compute_pillar4_dm` does not return `dm["dm_stat"]` value when the dict literal is built with `**dm` followed by a key collision check

**File:** `pfc_shaping/validation/scorecard.py:1130-1138`
**Issue:**
The return statement spreads `**dm` first, then adds `bloc`, `h_months`,
`mae_pfc`, `mae_baseline`, `delta_mae`, `better_than_baseline`. There is
**no key collision** between `dm` and the added keys (`dm` returns
`dm_stat`, `p_value`, `n`, `mean_d`, `var_d`, `n_lags_hac`, `degenerate`).

However the *empty-aligned* and *empty-mask* code paths
(lines 1074-1088 and 1095-1109) hard-code 12 keys (the union of `dm`'s 7
keys minus `degenerate` plus the 6 added keys + `degenerate` = 13). The
non-degenerate path returns `**dm + 6 extras = 7 + 6 = 13` keys. Both are
13 keys — consistent.

But: the degenerate paths set `"n_lags_hac": int(max(h_months - 1, 0))`
locally (which is right), while `diebold_mariano`'s degenerate return
(line 137) uses `int(n_lags)` where `n_lags = max(h - 1, 0)`. These agree
**only because the wrapper also computes `h - 1` correctly**. If anyone
changes the lag convention in one place but not the other, the two
degenerate dicts diverge silently — downstream consumers reading the
parquet cannot tell whether the dict came from the wrapper-degenerate
path or the `diebold_mariano`-degenerate path.

Also: `better_than_baseline` is `"N"` in both degenerate paths — but a
caller filtering Pillar 4 rows with `df["better_than_baseline"] == "Y"`
cannot distinguish "PFC failed to beat baseline" from "data degenerate, no
test possible". This biases summary statistics in
`render_markdown_report:1910-1918` ("strictly better in N/total cells")
toward looking worse than reality (degenerate cells counted in
denominator).

**Fix:**
Centralize the degenerate dict construction via a helper:
```python
def _dm_degenerate_dict(bloc_name: str, h_months: int) -> dict:
    return {
        "bloc": bloc_name,
        "h_months": int(h_months),
        "dm_stat": float("nan"), "p_value": float("nan"),
        "n": 0, "mean_d": float("nan"), "var_d": float("nan"),
        "n_lags_hac": int(max(h_months - 1, 0)),
        "degenerate": True,
        "mae_pfc": float("nan"), "mae_baseline": float("nan"),
        "delta_mae": float("nan"),
        "better_than_baseline": "DEGEN",  # explicit, distinct from "N"
    }
```
Update `render_markdown_report` to filter out `"DEGEN"` before computing
the better-than-baseline ratio.

---

## Info

### IN-01: Emojis (`✓`, `⚠`) in `render_markdown_report` output

**File:** `pfc_shaping/validation/scorecard.py:1862, 1866, 1951`
**Issue:**
The markdown report uses Unicode tick/cross/warning emojis in the gate
banner and per-test row. Per the project's writing conventions
(documented in user memory `feedback_permissions.md` style), emojis
should be avoided unless explicitly requested. This is rendered into the
canonical `10-VERIFICATION.md` artifact that ships in the planning
directory.
**Fix:** Replace `✓`/`✗`/`⚠` with ASCII (`[PASS]` / `[FAIL]` / `[WARN]`)
for consistency with the rest of the planning corpus.

---

### IN-02: `baseline_persistence_y1` will crash on Feb 29 vintage dates

**File:** `pfc_shaping/validation/dm_test.py:298`
**Issue:**
`target = vintage_date - pd.DateOffset(years=1)`. If `vintage_date` is
Feb 29 of a leap year, pandas raises (or rolls to Feb 28 depending on
version). The 24 vintages in `list_vintages_2024_2025` use
`BMonthEnd(0)`, which for Feb 2024 yields Feb 29 (Thursday, last business
day). Subtracting one year hits Feb 29 2023 which doesn't exist.
**Fix:** Add a guard:
```python
try:
    target = vintage_date - pd.DateOffset(years=1)
except (ValueError, OutOfBoundsDatetime):
    target = vintage_date - pd.Timedelta(days=365)
```
Verify behavior under pandas 2.3.3 (current pin); likely the DateOffset
silently rolls to Feb 28 but document the contract explicitly.

---

### IN-03: `mz_test` silently swallows `f_result.pvalue is None` without distinguishing it from `NaN`

**File:** `pfc_shaping/validation/scorecard.py:734-741`
**Issue:**
The branching at line 735 converts both `None` and "uncoercible-to-float"
to `NaN`, but downstream consumers cannot tell whether the model failed to
produce a p-value (None — likely API contract violation) versus produced
a NaN (perfect fit edge case — expected). The test
`TestStatsmodelsFTestSignature.test_statsmodels_f_test_api_signature`
even acknowledges both states with `pval is None or np.isnan(pval) or
isinstance(float(pval), float)`.
**Fix:** Add a separate flag `pval_source: str` in {`"finite"`, `"nan"`,
`"none"`, `"uncoercible"`} or log a warning at the `None` branch since
that's a statsmodels API contract violation worth flagging.

---

### IN-04: `_synth_epex_hist_for_mock` modulation creates a tiny `cos(2π × (hour − 14)/24)` discontinuity at hour 23 → 0

**File:** `pfc_shaping/validation/scorecard.py:389-393`
**Issue:**
The hour cycle `cos(2π × (hour − 14) / 24)` is mathematically periodic
with period 24, so hour=23 and hour=0 should connect smoothly. But because
`hour` is a discrete integer (0..23) the consecutive samples evaluate to
`cos(2π × 9/24)` and `cos(2π × (-14)/24) = cos(2π × 10/24)` — a jump in
phase of `1/24` cycle, not a discontinuity but a small step. For Pillar 1
continuity (max 2 €/MWh threshold) this is dwarfed by other modulations.
Informational only — preserved as documentation of a subtle numerical
detail.
**Fix:** None required; if synth-PFC fidelity matters in the future, use
sub-hour resolution and trigonometric continuity directly.

---

### IN-05: `_period_mask` returns silently-empty mask for `Peak`/`Offpeak` keys

**File:** `pfc_shaping/validation/structural_tests.py:140-141`
**Issue:**
The fallthrough `return np.zeros(len(idx_utc), dtype=bool)` for any
non-Base product type is silent. The comment says "Peak/Offpeak ou type
inconnu → mask vide (skip silencieux)". Combined with `test_arb_free`'s
silent-skip behavior (WR-06), a forwards dict that includes
`2024-01-Peak` keys will be silently ignored.
**Fix:** Add explicit logging at WARNING level when a Peak/Offpeak key is
skipped, or document explicitly in `test_arb_free` that the test only
covers Base products and that callers must filter their forwards dict.

---

### IN-06: `render_markdown_report` does not escape `|` in cell values; a `forwards_source` containing `|` would corrupt the markdown table

**File:** `pfc_shaping/validation/scorecard.py:1956-1958, 2003-2006, 2042-2045, 2079-2083`
**Issue:**
Markdown table cells delimit on `|`. Any string interpolated into a cell
that contains a literal pipe character will break the table layout.
Current `forwards_source` values (`"real_eex_xlsx"`, `"fallback_diagnostic"`,
`"mixed"`) don't contain pipes, but the `details` dict structure for
arb-free etc. could in principle, and any future source string is
unconstrained.
**Fix:** Add an `.replace("|", "\\|")` filter when interpolating
free-form string values into markdown table cells.

---

_Reviewed: 2026-05-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
