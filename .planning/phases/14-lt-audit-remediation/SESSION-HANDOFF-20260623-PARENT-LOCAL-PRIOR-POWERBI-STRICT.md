# Session Handoff - 2026-06-23 - Parent-Local Prior And Power BI Strict Pass

## Scope

Continued Phase 14 LT audit remediation on branch
`fix/lt-audit-remediation`.

The session started from the post-roaster state where:

- delivered-product audit v2 was available and fail-closed;
- local solver delivery months followed the delivered artifact window;
- strict Power BI gates were hardened;
- promotion remained NO-GO without fresh real artifacts.

## Code Changes

- `pfc_shaping/calibration/monthly_curve_priors.py`
  - Added direct-month parent bucket diagnostics for neighbor panel priors.
  - Changed fused prior weighting so `STRUCTURAL_TEMPLATE` is suppressed only
    inside parent buckets fully covered by direct monthly panel evidence.
  - Preserves zero-mean parent-space prior contract.
- `scripts/audit_ch_hfc_seasonal_coherence.py`
  - Added `reference_spread_eur_mwh` and `reference_spread_basis`.
  - Residual/calendar cross-year same-month checks now use comparable block
    spread for severity while retaining full parent spread columns.
- `tests/test_monthly_forward_curve_priors.py`
  - Added regression for `2027-Q2` direct monthly panel dominance over
    structural template.
- `tests/test_audit_ch_hfc_seasonal_coherence_script.py`
  - Added/updated residual-calendar comparable-block cross-year tests.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added D-20260623-12 and D-20260623-13.

No `pfc_shaping/ct/*`, `powerbi/data/*`, `powerbi/PFC_QA.*`, or heavy data
files were edited.

## Candidate Artifacts

Final diagnostic candidate:

`output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/`

Key files:

- CSV:
  `ch_hfc_hourly_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.csv`
- Fan chart:
  `ch_hfc_fan_chart_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.parquet`
- Monthly manifest:
  `ch_hfc_fan_chart_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.monthly_curve_manifest.json`
- Export report:
  `export_report.md`
- Shape audit:
  `hourly_shape_report.md`
- Seasonal audit:
  `seasonal_coherence_report.md`
- Delivered-product audit:
  `delivered_product_normalization_gates.csv`
  `delivered_product_normalization_summary.json`
- Strict Power BI sidecars:
  `powerbi_strict/`

Candidate command:

```powershell
$dir = 'output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126'
python scripts/export_local_test_ch_hourly_csv.py `
  --enable-monthly-forward-curve-solver `
  --enable-eex-peak-calibration `
  --monthly-solver-lambda-shape 25 `
  --monthly-solver-lambda-smooth-month 0.1 `
  --monthly-solver-lambda-smooth-yoy 10 `
  --enable-structural-shape-upgrade `
  --structural-shape-upgrade-intensity 0.5 `
  --structural-scenario-spread-intensity 1.26 `
  --required-forward-date 2026-06-17 `
  --output "$dir/ch_hfc_hourly_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.csv" `
  --report "$dir/export_report.md" `
  --fan-chart-output "$dir/ch_hfc_fan_chart_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.parquet" `
  --skip-powerbi-refresh
```

Generated Phase 13 markdown files were restored after local exports and are not
part of the curated code commit.

## Manifest Values

From monthly manifest:

- `monthly_level_authority`: `solver`
- `forward_snapshot_date`: `2026-06-17`
- `active_config_hash`:
  `145b123177061c9d2cd64ec831b83ea4ac84ff500356adb03623c4a9d1f86fc0`
- `solver_config_hash`:
  `9ecbbe7c9dadeb964689e43f363b5bde5545fbcf43ca7adae522ca24305c8414`
- `monthly_solution_hash`:
  `f1e352245102106e0df1e9c3fe0334cd2d1df032250d62a93586182198ecdde2`
- `source_hashes.forwards_path`:
  `c4bedaeb4cf7a04324bcf667be35ef9f92eeb2118c431109220076b114f9a3c5`
- solver config:
  - `lambda_shape`: `25.0`
  - `lambda_smooth_month`: `0.1`
  - `lambda_smooth_yoy`: `10.0`
  - `neighbor_shrinkage`: `0.5`
  - `panel_weight`: `1.0`
  - `history_weight`: `0.5`
  - `structural_weight`: `1.0`
  - `structural_amplitude_eur_mwh`: `110.0`
  - `allow_template_structural_fallback`: `true`

## Audit Results

Shape audit:

```text
[shape-audit] score=9.00/10
```

Power BI strict export:

```text
[powerbi] source csv -> output\phase14\20260623_parent_local_prior_lshape25_yoy10_structural_s126\ch_hfc_hourly_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.csv
[powerbi] exports -> output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/powerbi_strict
```

`powerbi_strict/summary_metrics.csv`:

- `shape_score_10`: `9`
- `max_eex_base_error_eur_mwh`: `0.000000`
- `max_eex_peak_error_eur_mwh`: `0.000000`
- `negative_gate_status`: `PASS`
- `seasonal_critical_flags`: `0`
- `monthly_split_critical_flags`: `0`
- `monthly_path_critical_flags`: `0`
- `cross_year_month_shape_critical_flags`: `0`
- `calendar_critical_flags`: `0`
- `powerbi_quality_gate_status`: `PASS`
- `powerbi_quality_gate_issues`: empty

Seasonal audit:

```text
[seasonal-audit] critical=0 warning=2
```

Remaining warnings:

- annual-only 2029 amplitude warning;
- one cross-year comparable-block warning for 2028->2029 June.

Delivered-product normalization audit:

```json
{
  "all_gates_pass": false,
  "covered_hard_gates_pass": false,
  "critical_count": 0,
  "delivered_curve_drift_count": 0,
  "quote_conflict_count": 9,
  "unsupported_count": 9,
  "status_counts": {
    "PASS": 70,
    "QUOTE_CONFLICT": 9,
    "UNSUPPORTED": 9
  },
  "input_csv_sha256": "464fe44b623ffbd15142eac68ea9ab01c7532b86d42b4611ea7d226f0af6665b",
  "audit_script_sha256": "9455196c0155c4018e2dbd93900ab6b9c4417f634462482cf083281b92488c2d",
  "forwards_sha256": "c4bedaeb4cf7a04324bcf667be35ef9f92eeb2118c431109220076b114f9a3c5"
}
```

Interpretation: no delivered curve drift remains on covered hard gates, but
promotion remains blocked by source quote conflicts and unsupported product
windows under the current fail-closed audit policy.

## Verification Commands

```powershell
python -m pytest tests/test_audit_ch_hfc_seasonal_coherence_script.py -q
```

Result:

```text
11 passed in 3.66s
```

```powershell
python -m pytest tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_solver.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q
```

Result:

```text
76 passed, 1 skipped in 126.88s (0:02:06)
```

## Remaining NO-GO Items

- Production promotion remains NO-GO.
- Delivered-product audit still exits non-zero because:
  - `QUOTE_CONFLICT=9`
  - `UNSUPPORTED=9`
- Candidate uses diagnostic CLI knobs:
  - `lambda_shape=25`
  - `lambda_smooth_month=0.1`
  - `lambda_smooth_yoy=10`
  - structural shape upgrade `ON`
  - spread intensity `1.26`
- These knobs need selected-lambda / governance artifact treatment before any
  production promotion.
- Real production/local export/selected-lambda manifest triad still needs to
  be regenerated and checked together.

## Recommended Next Steps

1. Run roaster read-only review on:
   - direct monthly panel dominance policy;
   - comparable-reference cross-year audit policy;
   - `lambda_shape=25`, `lambda_smooth_month=0.1`, `lambda_smooth_yoy=10`
     candidate evidence.
2. Decide source hierarchy policy for redundant EEX parent quote conflicts.
   Until then, keep `QUOTE_CONFLICT` blocking.
3. If policy accepts the lambda/shape knobs, promote them through the selected
   lambda artifact path rather than ad hoc CLI flags.
4. Regenerate production/local/selected-lambda manifests and rerun:
   - delivered-product audit;
   - seasonal audit;
   - strict Power BI export;
   - promotion manifest checker.
