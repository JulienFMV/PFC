# Session Handoff - 2026-06-24 - Source Hierarchy Identity Binding

## Scope

Implementation of the remaining roaster P2 for source hierarchy policy reuse.

A production-approved source hierarchy policy can no longer accept
`QUOTE_CONFLICT` from market/snapshot/count alone. It must bind to the audited
CSV and/or exact conflict identities.

## Changed Files

- `scripts/audit_ch_product_normalization.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_yoy10_structural_s126.json`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-SOURCE-HIERARCHY-IDENTITY-BINDING.md`
- `.planning/HANDOFF.md`

## Implementation

The audit now emits:

- `quote_conflict_identities`: canonical sorted rows with `gate_id`,
  `load_type`, `product`, `quote_conflict_basis`, and
  `covered_by_quote_aware_products`.
- `quote_conflict_identity_hash`: stable SHA-256 of those identities.

Policy validation now supports and validates:

- `input_csv_sha256`
- `quote_conflict_identity_hash`
- `expected_quote_conflicts`

For `production_approved=true`, at least one binding must be present and match.
Any provided binding field must match exactly. Missing or mismatched bindings
make the policy `INVALID` and accept zero quote conflicts.

The current draft policy was enriched with:

- `input_csv_sha256`: `464fe44b623ffbd15142eac68ea9ab01c7532b86d42b4611ea7d226f0af6665b`
- `quote_conflict_identity_hash`: `b13d9c87813f9cbf9c43d8cbbe0bb533b029e0845b80e363aee7aaa2946a66f9`
- the full 9-row `expected_quote_conflicts` list

It remains `production_approved=false`.

## Verification

Targeted tests:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py -q
```

Result: `24 passed in 17.80s`

Targeted promotion/product tests:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py tests/test_check_monthly_curve_promotion_from_manifests.py -q
```

Result: `33 passed in 11.42s`

Broader Phase 14 targeted tests:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_audit_ch_product_normalization_script.py tests/test_monthly_forward_curve_priors.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_monthly_curve_lambda_calibration.py tests/test_lt_ct_imports.py -q
```

Result: `110 passed, 1 skipped, 1 warning in 100.18s`

Warning observed:

`RuntimeWarning: All-NaN slice encountered` in
`monthly_curve_priors.py` during the intentional insufficient-history lambda
calibration test.

Delivered-product audit with enriched draft policy:

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
- `quote_conflict_identity_hash`: `b13d9c87813f9cbf9c43d8cbbe0bb533b029e0845b80e363aee7aaa2946a66f9`
- `audit_script_sha256`: `f0f680c2c0731f4abf9fb48793a42cd4037b46fe01582fbbac2282a7d2adf8c1`

## Current Verdict

GO on closing the source hierarchy policy identity-binding P2.

Production remains NO-GO:

- Current source hierarchy policy is still draft and `production_approved=false`.
- Delivered-product audit still has `QUOTE_CONFLICT=9` and `UNSUPPORTED=9`.
- Strict Power BI still blocks on `cross_year_near_clone_warnings=1`.
- Current selected config artifact is diagnostic and `production_approved=false`.

## Next

1. Run MIT roasters on the identity-binding patch.
2. Decide whether to production-approve the bound source hierarchy policy or
   use a cleaned source snapshot.
3. Resolve `UNSUPPORTED=9`.
4. Resolve strict Power BI `cross_year_near_clone_warnings=1`.
