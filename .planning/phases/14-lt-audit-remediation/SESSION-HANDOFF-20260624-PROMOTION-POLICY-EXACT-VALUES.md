# Session Handoff - 2026-06-24 - Promotion Policy Exact Values

## Scope

Follow-up to the MIT roaster re-audit of commit `4057152463`.

The previous P0 selected-config promotion path was closed, but roasters found
two remaining P1 exactness issues. This session fixes those two issues only.

## Changed Files

- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `scripts/audit_ch_product_normalization.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-PROMOTION-POLICY-EXACT-VALUES.md`
- `.planning/HANDOFF.md`

## Roaster Findings Addressed

- P1: `selection_status` used substring matching and could accept
  `NOT_PRODUCTION_APPROVED` if `production_approved=true`. Fixed by requiring
  exact `selection_status.upper() == "PRODUCTION_APPROVED"`.
- P1: `expected_quote_conflict_count` used `int(...)`, accepting strings,
  floats and truncation. Fixed by requiring a real integer and rejecting
  booleans, strings and floats.

## Verification

Targeted tests:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py tests/test_check_monthly_curve_promotion_from_manifests.py -q
```

Result: `27 passed in 13.48s`

Broader Phase 14 targeted tests:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_audit_ch_product_normalization_script.py tests/test_monthly_forward_curve_priors.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_monthly_curve_lambda_calibration.py tests/test_lt_ct_imports.py -q
```

Result: `104 passed, 1 skipped, 1 warning in 75.56s`

Warning observed:

`RuntimeWarning: All-NaN slice encountered` in
`monthly_curve_priors.py` during the intentional insufficient-history lambda
calibration test.

Delivered-product audit with the current draft source hierarchy policy:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/ch_hfc_hourly_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.csv --forwards data/eex_forwards_history.parquet --required-forward-date 2026-06-17 --price-column price_weighted_mean_eur_mwh --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_yoy10_structural_s126.json --output-csv output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/delivered_product_normalization_gates_with_policy.csv --summary-json output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/delivered_product_normalization_summary_with_policy.json
```

Result: exit code `1`, expected NO-GO.

Summary:

- `critical_count`: `0`
- `delivered_curve_drift_count`: `0`
- `quote_conflict_count`: `9`
- `accepted_quote_conflict_count`: `0`
- `blocking_quote_conflict_count`: `9`
- `unsupported_count`: `9`
- `source_hierarchy_policy.status`: `VALID_NOT_PRODUCTION_APPROVED`
- `source_hierarchy_policy.production_approved`: `false`
- `audit_script_sha256`: `eff74fe881de4e78ddf4ece29f54286ec21f80635f4034835b003a8bf870728c`

## Current Verdict

Production remains NO-GO.

Known remaining blockers:

- Current source hierarchy policy is draft and `production_approved=false`.
- Delivered-product audit still has `QUOTE_CONFLICT=9` and `UNSUPPORTED=9`.
- Strict Power BI still blocks on `cross_year_near_clone_warnings=1`.
- Current selected config artifact is diagnostic and `production_approved=false`.

## Remaining Roaster P2

A production-approved source hierarchy policy should eventually bind to the
exact conflict identities or input CSV hash, not only market/snapshot/count.
This was left as P2 because it needs an artifact schema decision before
implementation.
