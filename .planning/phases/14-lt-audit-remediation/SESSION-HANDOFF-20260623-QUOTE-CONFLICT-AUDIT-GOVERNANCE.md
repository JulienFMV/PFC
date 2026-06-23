# Session Handoff - 2026-06-23 - Quote Conflict Audit Governance

## Scope

Follow-up after `SESSION-HANDOFF-20260623-CANDIDATE-AUDIT-BLOCKERS.md`.
The prior candidate showed all quote-aware delivered BASE/PEAK buckets passing,
while direct redundant parent product rows failed because the 2026-06-17 EEX
snapshot contains internally inconsistent finer-vs-parent quotes.

## Pushed Baseline

Before this patch, pushed:

```text
3d3c85409 fix(lt): align local solver months with delivered window
```

Branch after push was aligned with `origin/fix/lt-audit-remediation`.

## Roaster Results

Two read-only subagents reviewed the audit semantics:

- Feynman: recommended a separate `QUOTE_CONFLICT` data-quality class, not
  ordinary delivered-curve `CRITICAL` and not generic `UNSUPPORTED`. It must
  remain blocking by default.
- Godel: recommended reclassifying only direct parent BASE/PEAK failures that
  are fully covered by finer quote-aware buckets, with all those finer buckets
  passing. If any fine bucket fails, the parent remains `CRITICAL`.

## Code Changes

- `scripts/audit_ch_product_normalization.py`
  - Schema bumped to `ch_product_normalization_audit.v2`.
  - Added `QUOTE_CONFLICT` status and `quote_conflict` severity.
  - Reclassifies redundant parent direct BASE/PEAK rows from `CRITICAL` to
    `QUOTE_CONFLICT` only when:
    - product is a Cal/Quarter direct parent;
    - quote-aware finer buckets fully cover the parent;
    - no parent residual bucket is needed;
    - every covering quote-aware bucket passes;
    - delivered parent mean is explained by the finer bucket targets.
  - Reclassifies implied OFFPEAK rows for the same parent as
    `QUOTE_CONFLICT` when the parent BASE/PEAK conflict explains them.
  - Adds output columns when applicable:
    - `quote_conflict_basis`
    - `covered_by_quote_aware_products`
  - Summary now includes:
    - `quote_conflict_count`
    - `delivered_curve_drift_count`
  - `all_gates_pass` and `covered_hard_gates_pass` remain false when any
    `QUOTE_CONFLICT` exists.
  - CLI remains fail-closed by default; `--allow-failed-gates` is still the
    diagnostic-only escape.
- `tests/test_audit_ch_product_normalization_script.py`
  - Added synthetic redundant parent quote conflict fixture.
  - Added tests for:
    - quote conflict reclassification;
    - parent remaining `CRITICAL` when a fine quote-aware bucket fails;
    - CLI default failing on quote conflict;
    - `--allow-failed-gates` returning 0 with quote conflict.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added D-20260623-11.

## Verification

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py -q
```

Result:

```text
12 passed in 2.75s
```

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py -q
```

Result:

```text
17 passed in 7.18s
```

Candidate audit rerun:

```powershell
python scripts/audit_ch_product_normalization.py `
  --csv output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --forwards data/eex_forwards_history.parquet `
  --required-forward-date 2026-06-17 `
  --price-column price_weighted_mean_eur_mwh `
  --output-csv output/phase14/20260623_solver_peak_candidate/delivered_product_normalization_gates_quoteconflict.csv `
  --summary-json output/phase14/20260623_solver_peak_candidate/delivered_product_normalization_summary_quoteconflict.json
```

Result: exit 1, still fail-closed.

```text
all_gates_pass=false
critical_count=0
delivered_curve_drift_count=0
quote_conflict_count=9
unsupported_count=9
status_counts={PASS: 70, QUOTE_CONFLICT: 9, UNSUPPORTED: 9}
supported_hard_gate_max_abs_residual_eur_mwh=0.10292307435898351
audit_script_sha256=2fe81aff431f8c7e456279b68fa356d0cfa08930c5143570fd8dbb0323e28161
input_csv_sha256=4d79737ae985a227e5f81498a512b54259a7090cc4a77cbb9abe6cfb7e3c32fe
forwards_sha256=c4bedaeb4cf7a04324bcf667be35ef9f92eeb2118c431109220076b114f9a3c5
```

Strict Power BI rerun:

```powershell
python scripts/build_powerbi_exports.py `
  --csv output/phase14/20260623_solver_peak_candidate/ch_hfc_hourly_solver_peak_20260613_20301231.csv `
  --forwards data/eex_forwards_history.parquet `
  --spot data/epex_hourly.parquet `
  --output-dir output/phase14/20260623_solver_peak_candidate/powerbi_strict_after_quoteconflict
```

Result: exit 1, still blocked on non-product gates:

```text
Power BI export blocked by quality gates.
- shape_score_10=6.75 < 8.50
- monthly_split_critical_flags=1
```

## Current Verdict

Production remains NO-GO.

The product audit no longer mislabels source quote hierarchy conflicts as
delivered curve drift. It still blocks promotion because:

- `QUOTE_CONFLICT`: 9 rows for redundant parent products from the
  2026-06-17 snapshot;
- `UNSUPPORTED`: 9 rows for partial/out-of-window products (`2026-06`, `2031`,
  `2032`);
- Power BI strict still blocks on `shape_score_10=6.75` and
  `monthly_split_critical_flags=1`.

## Next Work

- Decide whether promotion requires a cleaned EEX snapshot or a future
  manifest-backed hierarchy acceptance policy for `QUOTE_CONFLICT`.
- Fix remaining model-quality gates through priors/objective/shape/fan-chart
  calibration, not individual month patches:
  - `2027 Q2 BASE` monthly split vs DE;
  - structural width too narrow;
  - ramp/boundary metrics lowering shape score.
- Regenerate candidate and rerun strict gates after the model-quality fixes.
