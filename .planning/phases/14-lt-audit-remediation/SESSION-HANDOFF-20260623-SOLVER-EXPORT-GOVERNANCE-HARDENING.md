# Session Handoff - 2026-06-23 - Solver Export Governance Hardening

## Scope

Curated Phase 14 hardening after roaster review. This is a code/test commit
scope only; generated data, Power BI layout/model churn and local outputs remain
excluded.

## Changes

- Monthly solver comparable-block logic:
  - Month smoothness no longer crosses incompatible parent buckets.
  - Residual-vs-calendar YoY shape rows compare the same month-number block,
    not a residual block against a full calendar.
  - Sparse parent-flat baseline fills unrepresented months from represented
    bucket levels instead of zero.
- Structural prior governance:
  - Template fallback defaults are explicit and aligned at `110.0` EUR/MWh.
  - Fallback diagnostics record source, fallback reason, parent zero-mean
    residuals and history counts.
  - Manifest records `structural_prior_summary`.
- Lambda/promotion hash contract:
  - Selected lambda config hashes the same active payload as production/export
    manifests, including prior weights, structural fallback flag, structural
    amplitude and structural sample settings.
  - Lambda calibration defaults now match active production prior weights.
  - Manifest fallback hash reconstruction uses the widened payload.
- Export/Power BI gates:
  - Final EEX PEAK calibration reruns after final mutators before CSV write.
  - Structural fan columns prefer ordered low/central/high columns and fall
    back to row-wise ordered scenario prices.
  - Power BI sidecar export is fail-closed unless `--allow-failed-gates`.
  - `finite_ok`, structural quantile ordering and PEAK repricing evidence are
    explicit hard quality gates.

## Verification

- `python -m pytest tests/test_monthly_curve_lambda_calibration.py tests/test_check_monthly_curve_promotion_from_manifests.py -q`
  - `21 passed, 1 warning`
- `python -m pytest tests/test_monthly_forward_curve_constraints.py tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_forward_curve_integration.py -q`
  - `57 passed`
- `python -m pytest tests/test_build_powerbi_exports_script.py tests/test_audit_ch_pfc_hourly_shape_script.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py -q`
  - `48 passed`
- Combined:
  - `python -m pytest tests/test_monthly_curve_lambda_calibration.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_monthly_forward_curve_constraints.py tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_forward_curve_integration.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_pfc_hourly_shape_script.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_build_ep2050_multi_scenario_pfc_script.py tests/test_build_local_test_ch_pfc_script.py -q`
  - `131 passed, 1 warning`

Warning observed:

- `RuntimeWarning: All-NaN slice encountered` in
  `monthly_curve_priors.py` during an intentionally insufficient-history
  lambda calibration test.

## Roaster Results

- Hegel initially found P0: production manifest active config hash included
  prior-stack knobs while lambda selected artifacts did not. Fixed.
- Hegel also found P1: lambda calibration defaults diverged from production
  prior weights. Fixed.
- Hilbert initially found P1: Power BI gates did not hard-block finite/order
  invariants or missing PEAK evidence. Fixed.
- Averroes re-roast: GO on hash parity and Power BI strict gates; no P0/P1
  remaining in the audited correction.

## Curated Commit Include

- `pfc_shaping/calibration/monthly_curve_lambda_calibration.py`
- `pfc_shaping/calibration/monthly_curve_priors.py`
- `pfc_shaping/calibration/monthly_forward_curve.py`
- `pfc_shaping/config.yaml`
- `pfc_shaping/pipeline/monthly_curve_authority.py`
- `pfc_shaping/pipeline/production_phases.py`
- `scripts/audit_ch_hfc_seasonal_coherence.py`
- `scripts/audit_ch_pfc_hourly_shape.py`
- `scripts/build_ep2050_multi_scenario_pfc.py`
- `scripts/build_local_test_ch_pfc.py`
- `scripts/build_powerbi_exports.py`
- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `scripts/export_local_test_ch_hourly_csv.py`
- `scripts/run_monthly_curve_sparse_year_proof.py`
- `tests/fixtures/monthly_curve_phase_e_parity_baseline.json`
- `tests/test_audit_ch_hfc_seasonal_coherence_script.py`
- `tests/test_audit_ch_pfc_hourly_shape_script.py`
- `tests/test_build_powerbi_exports_script.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `tests/test_export_local_test_ch_hourly_csv_script.py`
- `tests/test_monthly_curve_lambda_calibration.py`
- `tests/test_monthly_forward_curve_constraints.py`
- `tests/test_monthly_forward_curve_integration.py`
- `tests/test_monthly_forward_curve_priors.py`
- `tests/test_monthly_forward_curve_solver.py`

## Excluded

- `data/eex_forwards_history.parquet`
- `pfc_shaping/data/pfc_local.duckdb`
- `pfc_shaping/output/*`
- `powerbi/PFC_QA.*`
- unrelated local docs and generated reports

## Next

This still is not production promotion. Next phase should regenerate a fresh
candidate, run the delivered-product audit and strict Power BI export on real
artifacts, then continue structural width calibration if gates still fail.
