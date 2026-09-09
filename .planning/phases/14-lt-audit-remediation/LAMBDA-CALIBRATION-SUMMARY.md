# Lambda Calibration Summary - Monthly Curve D0

Date: 2026-06-19
Scope: research/offline only
Production approved: false

## Status

The Phase D0 harness exists and is fail-closed. It builds rolling-origin
withholding tests from historical EEX snapshots, masks the withheld product and
directly revealing own/neighbor products, uses only history strictly before the
origin date, and scores against the same-snapshot withheld quote.

Current smoke run status:

```text
final_status=UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA
production_approved=false
```

This is an acceptable research result. It means the smoke sample did not
identify a lambda configuration that clearly beats the baseline under the
configured identifiability guard. No production default is changed.

## Train/Deploy Horizon Gap

The calibration evidence is limited by the available traded products. Withheld
monthly/quarterly targets are available mostly at near tenors (`h+0/h+1`), while
the observed sparse-year defect lives in `h+2/h+3`.

The harness now exposes this explicitly in:

- `scoring.csv`: `withheld_horizon_years`, `withheld_horizon_bucket`
- `calibration_summary.json`: `by_tenor_horizon`
- `calibration_summary.json`: `train_deploy_gap`
- `calibration_manifest.json`: `withheld_products_by_tenor_horizon`
- `candidate_config.yaml`: `baseline_comparison.by_tenor_horizon`

If no `h+2/h+3` withheld products exist in a run, the summary reports:

```text
UNSUPPORTED_NO_FAR_HORIZON_MONTHLY_TRUTH
```

That status is not a script failure. It is the expected disclosure that the
far-horizon monthly shape is not directly validated by same-snapshot monthly
truth.

## Selection Method

The current implementation uses:

- baseline comparison;
- per-config withheld-product MAE;
- stability gates through hard constraint residuals and critical gate counts;
- an identifiability guard:
  - absolute MAE improvement >= `0.05` EUR/MWh, or
  - relative MAE improvement >= `0.01`.

If the guard is not met, the final status is:

```text
UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA
```

This is conservative, but it is not yet the full L-curve/Pareto knee selection
described in the reform plan. The missing L-curve layer should compare
withheld error, curvature, historical outlier score, and neighbor disagreement
before any production promotion.

## Required Next Evidence

Before production approval, run a broader calibration and report:

- error by `withheld_tenor`;
- error by `withheld_horizon_bucket`;
- baseline vs candidate by horizon;
- count of `h+2/h+3` withheld products, expected to be zero or sparse;
- explicit statement that sparse 2028-2030 deployment is not directly validated
  by traded monthly truth if no far-horizon withheld products exist;
- L-curve/Pareto diagnostics or a documented conservative alternative.

No artifact from this phase may set `production_approved=true`.
