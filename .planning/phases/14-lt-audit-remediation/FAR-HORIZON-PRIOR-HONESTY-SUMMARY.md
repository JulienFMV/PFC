# Far-Horizon Prior Honesty Summary

Date: 2026-06-19
Scope: monthly curve priors, research/offline path
Production approved: false

## Objective

Address the audit finding that the neighbor panel prior could look stronger
than it really is for sparse far-horizon delivery years. The key requirement is
to distinguish true monthly evidence from block-level fallback, especially for
`h+2/h+3+`.

## Changes

- `build_neighbor_panel_shape_prior(...)` now accepts optional
  `run_timestamp`.
- Panel diagnostics now include horizon-specific counts:
  - `direct_month_quotes_h+0`
  - `direct_month_quotes_h+1`
  - `direct_month_quotes_h+2`
  - `direct_month_quotes_h+3+`
  - `block_shape_months_h+*`
  - `covered_months_h+*`
- Panel diagnostics distinguish:
  - `prior_far_horizon_monthly_evidence`
  - `market_far_horizon_monthly_evidence`
- Ambiguous prior statuses were tightened:
  - `PARTIAL_PANEL_MULTI_MARKET` -> `PARTIAL_MONTHLY_PANEL`
  - `DE_SINGLE_MARKET` -> `DE_SINGLE_MARKET_MONTHLY`
  - `SINGLE_MARKET` -> `SINGLE_MARKET_MONTHLY`
- The sparse-year proof script passes the snapshot timestamp into the panel
  prior builder, so proof artifacts expose real horizon buckets.

## Real Data Proof

Command:

```powershell
python scripts/run_monthly_curve_sparse_year_proof.py --forwards data/eex_forwards_history.parquet --output-dir output/monthly_curve_sparse_year_proof --no-plot
```

Result:

```text
max_abs_constraint_residual=2.132e-13
neighbor_level_leakage_max_abs=1.421e-13
gate_summary={'PASS': 19, 'WARNING': 1}
panel_status=PARTIAL_MONTHLY_PANEL
history_status=PARTIAL_HISTORY_FORWARD
structural_status=UNSUPPORTED
fused_status=PARTIAL_MONTHLY_PANEL
```

Key panel evidence from `panel_prior_diagnostics.csv`:

```text
DE: direct_month_quotes_h+2 = 6, market_far_horizon_monthly_evidence = DE_FAR_HORIZON_MONTHLY_EVIDENCE
FR: direct_month_quotes_h+2 = 0, market_far_horizon_monthly_evidence = NO_FAR_HORIZON_MONTHLY_EVIDENCE
AT: direct_month_quotes_h+2 = 0, market_far_horizon_monthly_evidence = NO_FAR_HORIZON_MONTHLY_EVIDENCE
IT: direct_month_quotes_h+2 = 0, market_far_horizon_monthly_evidence = NO_FAR_HORIZON_MONTHLY_EVIDENCE
```

This means the far-horizon monthly panel is not represented as a generic
multi-market monthly panel. It is explicitly DE-supported and partial.

## Tests

Validated:

```text
pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_solver.py -q
pytest tests/test_monthly_curve_lambda_calibration.py -q
pytest tests/test_monthly_forward_curve_*.py tests/test_monthly_curve_lambda_calibration.py -q
python scripts/run_monthly_curve_lambda_calibration.py --forwards data/eex_forwards_history.parquet --grid .planning/phases/14-lt-audit-remediation/lambda_grid.yaml --output-dir output/monthly_curve_calibration --smoke
```

Observed:

```text
monthly suite: 46 passed
lambda smoke: final_status=UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA, production_approved=False
```

## Remaining Limits

- CH still has no direct monthly truth at `h+2/h+3+` in the current parquet.
- DE h+2 monthly evidence is partial, not a full far-horizon panel.
- Structural prior remains `UNSUPPORTED` in the proof run unless data coverage
  supports it or a template fallback is explicitly enabled.
- These changes improve prior honesty; they do not by themselves approve the
  solver for production.
