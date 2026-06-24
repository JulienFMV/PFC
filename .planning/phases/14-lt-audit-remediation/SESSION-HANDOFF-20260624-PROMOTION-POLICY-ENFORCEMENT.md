# Session Handoff - 2026-06-24 - Promotion Policy Enforcement

## Scope

Follow-up implementation after MIT roaster audits on commit `389510cbaa`.

This session fixes the identified P0/P1 fail-open paths in promotion
governance. It does not promote the current candidate.

## Changed Files

- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `scripts/audit_ch_product_normalization.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-PROMOTION-POLICY-ENFORCEMENT.md`
- `.planning/HANDOFF.md`

## Roaster Findings Addressed

- Hubble P0: selected config with `production_approved=false` could still pass
  capstone promotion when `config_hash` matched. Fixed with required
  `selected_config_production_approval` gate.
- Hubble P1: selected/prod/export triad compared only `config_hash`. Fixed with
  required `selected_config_manifest_parity` gate that compares active config,
  monthly solution and active constraints hashes.
- Socrates/Copernicus P1: source hierarchy policy used Python truthiness for
  booleans. Fixed by requiring JSON/YAML boolean `is True`.
- Socrates/Copernicus P1: source hierarchy policy could omit
  `forward_snapshot_date`. Fixed as required and exact.
- Socrates/Copernicus P1/P2: source hierarchy policy did not validate
  `expected_quote_conflict_count`. Fixed as required and equal to the observed
  count.

## Verification

Targeted tests:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_audit_ch_product_normalization_script.py -q
```

Result: `22 passed in 5.00s`

Broader Phase 14 targeted tests:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_audit_ch_product_normalization_script.py tests/test_monthly_forward_curve_priors.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_monthly_curve_lambda_calibration.py tests/test_lt_ct_imports.py -q
```

Result: `99 passed, 1 skipped, 1 warning in 104.75s`

Warning observed:

`RuntimeWarning: All-NaN slice encountered` in
`monthly_curve_priors.py` during the intentional insufficient-history lambda
calibration test.

Delivered-product audit with the draft source hierarchy policy:

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
- `audit_script_sha256`: `7381382ac51b9b5180b08920087b1c063fb933394a7a38cb0c1fb9a5f47c8cea`

## Current Verdict

Production remains NO-GO.

The known P0/P1 governance fail-open paths are fixed in code/tests, but the
candidate still blocks on:

- `QUOTE_CONFLICT=9` with draft non-production-approved policy.
- `UNSUPPORTED=9` in delivered-product audit.
- Strict Power BI `cross_year_near_clone_warnings=1`.
- Selected config artifact remains diagnostic and `production_approved=false`.

## Next

1. Re-run MIT roasters on this enforcement patch.
2. Decide cleaned source snapshot vs production-approved hierarchy policy for
   the 2026-06-17 redundant CH quotes.
3. Fix the remaining cross-year near-clone through priors/objective/evidence,
   then regenerate; do not patch months individually.
4. Resolve or explicitly document the `UNSUPPORTED=9` rows without hiding
   critical evidence.
