# Phase F Promotion Criteria Summary - 2026-06-19

## Scope

This patch makes the monthly-curve promotion criteria executable. It does not
change solver behavior, does not mutate any curve after solve, and does not
toggle the monthly solver production flag.

## Delivered

- Added `pfc_shaping/calibration/monthly_curve_promotion.py`.
- Added `scripts/check_monthly_curve_promotion.py`.
- Added unit tests in `tests/test_monthly_curve_promotion.py`.

The promotion evaluator consumes:

```text
audit_gates.csv
historical_thresholds.csv
manifest.json optional
```

It emits:

```text
promotion_decision.json
promotion_decision_details.csv
```

## Rules

- `CRITICAL` audit rows always block.
- Required hard gates must be present and `PASS`:
  - `hard_monthly_curve_repricing`
  - `neighbor_level_leakage`
- `UNSUPPORTED` rows on near-horizon or calibrable populations block.
- Far-horizon `UNSUPPORTED` shape rows may be accepted only when the matching
  `historical_thresholds.csv` row proves:
  - threshold generation was attempted;
  - threshold status is `UNSUPPORTED`;
  - `n_snapshots < min_required_n`;
  - P90/P97.5 are not populated.
- `monthly_shape_regression_2028_2030=UNSUPPORTED` is accepted only when all
  child unsupported targeted rows are accepted far-horizon risks.

This encodes the desk rule: far-horizon `UNSUPPORTED` can document residual
model risk, but it cannot hide a `CRITICAL`, a missing hard gate, or a
near-horizon unsupported gate.

## Verified

Commands run:

```powershell
pytest tests/test_monthly_curve_promotion.py -q
python scripts/check_monthly_curve_promotion.py --audit-gates output/monthly_curve_sparse_year_proof/audit_gates.csv --historical-thresholds output/monthly_curve_sparse_year_proof/historical_thresholds.csv --manifest output/monthly_curve_sparse_year_proof/manifest.json --run-timestamp 2026-06-17 --output output/monthly_curve_sparse_year_proof/promotion_decision.json --details-output output/monthly_curve_sparse_year_proof/promotion_decision_details.csv
```

Results:

- Promotion unit tests: `5 passed`.
- Sparse proof promotion evidence:

```text
status=PROMOTION_EVIDENCE_PASS
approved=true
audit_gate_status_counts={'PASS': 21, 'UNSUPPORTED': 10}
threshold_status_counts={'PASS': 13, 'UNSUPPORTED': 13}
blocking_count=0
```

## Remaining Limits

- This is an evidence gate, not a flag flip. Production default activation
  still requires the desk to accept the named far-horizon residual risk.
- The evaluator currently enforces the first production-critical hard gates and
  far-horizon unsupported policy. Future Phase F gates such as
  `lambda_calibration_artifact_present`, `point_in_time_data_contract` and
  `production_export_path_parity` should be wired as first-class rows once they
  are emitted by the audit package.
