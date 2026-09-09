# Session Handoff - 2026-06-24 - Roaster Follow-up Strict Gates Policy

## Scope

Follow-up to MIT roaster audit on the Phase 14 candidate
`parent_local_prior_lshape25_yoy10_structural_s126`.

This is still NO-GO for production. The work hardens the gates and records the
draft governance artifacts needed for future promotion review; it does not
approve the candidate.

## Changed Files

- `pfc_shaping/calibration/monthly_curve_priors.py`
- `scripts/audit_ch_product_normalization.py`
- `scripts/build_powerbi_exports.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `tests/test_build_powerbi_exports_script.py`
- `tests/test_monthly_forward_curve_priors.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_lshape25_yoy10_structural_s126.json`
- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_yoy10_structural_s126.json`
- `.planning/HANDOFF.md`

## Roaster Findings Addressed

- Lovelace P2: structural template suppression was triggered by direct monthly
  panel buckets even when `panel_weight=0`. Fixed by suppressing structural
  only when the panel source is active in the fused prior.
- Lovelace P3: fused-prior diagnostics now expose effective weight min/max and
  suppressed parent buckets by source.
- Aquinas P1: strict Power BI export now blocks on one cross-year near-clone
  warning.
- Plato governance caveat: production remains NO-GO until quote-conflict source
  hierarchy and selected-lambda/config artifacts are production-approved and
  tied into the manifest triad.

## Candidate Evidence

Candidate folder:

`output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/`

Candidate CSV:

`output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/ch_hfc_hourly_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.csv`

Candidate monthly manifest:

`output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/ch_hfc_fan_chart_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.monthly_curve_manifest.json`

Selected config diagnostic artifact:

`.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_lshape25_yoy10_structural_s126.json`

- `config_hash`: `145b123177061c9d2cd64ec831b83ea4ac84ff500356adb03623c4a9d1f86fc0`
- `active_config_hash_from_candidate_manifest`: `145b123177061c9d2cd64ec831b83ea4ac84ff500356adb03623c4a9d1f86fc0`
- `monthly_solution_hash`: `f1e352245102106e0df1e9c3fe0334cd2d1df032250d62a93586182198ecdde2`
- `active_constraints_hash`: `efb01468d31e43f9c6cd66102ad5c573a9aacc8913e26b1a20139358174144cf`
- `production_approved`: `false`

Draft source hierarchy policy:

`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_yoy10_structural_s126.json`

- `schema_version`: `ch_quote_conflict_source_hierarchy_policy.v1`
- `market`: `CH`
- `forward_snapshot_date`: `2026-06-17`
- `source_hierarchy`: `quote_aware_finer_buckets_over_redundant_parent`
- `expected_quote_conflict_count`: `9`
- `production_approved`: `false`

Policy hash from audit summary:

`bf1463ba35d4908311aaaac0940176c70a53d30e07c42a18e5d62f667e80ea29`

## Commands And Results

Targeted tests:

```powershell
python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_product_normalization_script.py -q
```

Result: `44 passed in 14.30s`

Broader targeted tests:

```powershell
python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_product_normalization_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_monthly_curve_lambda_calibration.py tests/test_lt_ct_imports.py -q
```

Result: `93 passed, 1 skipped, 1 warning in 77.68s`

Warning observed:

`RuntimeWarning: All-NaN slice encountered` in
`monthly_curve_priors.py` during the insufficient-history lambda calibration
test.

Whitespace check:

```powershell
git diff --check
```

Result: no whitespace errors. Git reported LF/CRLF normalization warnings only.

Strict Power BI export:

```powershell
python scripts/build_powerbi_exports.py --csv output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/ch_hfc_hourly_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.csv --forwards data/eex_forwards_history.parquet --spot data/epex_spot_history.parquet --output-dir output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/powerbi_strict_after_roast
```

Result: exit code `1`, expected strict block:

`Power BI export blocked by quality gates ... cross_year_near_clone_warnings=1`

Delivered-product audit with draft source hierarchy policy:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/ch_hfc_hourly_parent_local_prior_lshape25_yoy10_structural_s126_20260613_20301231.csv --forwards data/eex_forwards_history.parquet --required-forward-date 2026-06-17 --price-column price_weighted_mean_eur_mwh --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_yoy10_structural_s126.json --output-csv output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/delivered_product_normalization_gates_with_policy.csv --summary-json output/phase14/20260623_parent_local_prior_lshape25_yoy10_structural_s126/delivered_product_normalization_summary_with_policy.json
```

Result: exit code `1`, expected NO-GO because policy is not
production-approved.

Summary:

- `all_gates_pass`: `false`
- `critical_count`: `0`
- `delivered_curve_drift_count`: `0`
- `quote_conflict_count`: `9`
- `accepted_quote_conflict_count`: `0`
- `blocking_quote_conflict_count`: `9`
- `unsupported_count`: `9`
- `status_counts`: `PASS=70`, `QUOTE_CONFLICT=9`, `UNSUPPORTED=9`
- `source_hierarchy_policy.status`: `VALID_NOT_PRODUCTION_APPROVED`
- `source_hierarchy_policy.production_approved`: `false`
- `input_csv_sha256`: `464fe44b623ffbd15142eac68ea9ab01c7532b86d42b4611ea7d226f0af6665b`
- `forwards_sha256`: `c4bedaeb4cf7a04324bcf667be35ef9f92eeb2118c431109220076b114f9a3c5`
- `audit_script_sha256`: `681e39bee3f2a51215cde387b2ce452626fb9750cff414854658771d6fc96756`

## Current Verdict

Production remains NO-GO.

Blocking items:

- `QUOTE_CONFLICT=9` remains blocking because the source hierarchy policy is
  draft and `production_approved=false`.
- `UNSUPPORTED=9` remains blocking in the delivered-product audit.
- Strict Power BI export blocks on `cross_year_near_clone_warnings=1`.
- The selected config artifact is diagnostic only and
  `production_approved=false`; it is not yet part of a production/local
  export/selected artifact triad.

## Next

1. Decide whether the 2026-06-17 CH redundant quote conflicts should be handled
   by a production-approved source hierarchy policy or by a cleaned source
   snapshot.
2. Fix the remaining cross-year near-clone warning through priors/objective/gate
   evidence and regenerate; do not patch months individually.
3. Resolve or explicitly document the `UNSUPPORTED=9` rows without hiding any
   critical gate.
4. Re-run delivered-product audit and strict Power BI export on the regenerated
   candidate.
5. Only then assemble production/local-export/selected-lambda manifest triad
   and request promotion review.
