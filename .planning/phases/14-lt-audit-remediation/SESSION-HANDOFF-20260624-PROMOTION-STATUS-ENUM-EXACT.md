# Session Handoff - 2026-06-24 - Promotion Status Enum Exact

## Scope

Mini-roast follow-up on commit `feedbd56d9`.

Roasters confirmed the source-hierarchy exact-value P1 is closed and found one
remaining selected-config exactness issue: `selection_status.upper()` accepted
case variants of `PRODUCTION_APPROVED`.

## Changed Files

- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-PROMOTION-STATUS-ENUM-EXACT.md`
- `.planning/HANDOFF.md`

## Fix

`selection_status` is now compared as a case-sensitive enum:

`selection_status == "PRODUCTION_APPROVED"`

Tests now block:

- `NOT_PRODUCTION_APPROVED`
- `DIAGNOSTIC_SELECTED_NOT_PRODUCTION_APPROVED`
- `production_approved`
- `Production_Approved`

## Verification

Targeted tests:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_audit_ch_product_normalization_script.py -q
```

Result: `27 passed in 8.94s`

Broader Phase 14 targeted tests:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_audit_ch_product_normalization_script.py tests/test_monthly_forward_curve_priors.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_monthly_curve_lambda_calibration.py tests/test_lt_ct_imports.py -q
```

Result: `104 passed, 1 skipped, 1 warning in 100.93s`

Warning observed:

`RuntimeWarning: All-NaN slice encountered` in
`monthly_curve_priors.py` during the intentional insufficient-history lambda
calibration test.

## Current Verdict

GO on closing the exact-value P1 issues from the mini-roast.

Production remains NO-GO:

- Current selected config artifact is diagnostic and `production_approved=false`.
- Current source hierarchy policy is draft and `production_approved=false`.
- Delivered-product audit still has `QUOTE_CONFLICT=9` and `UNSUPPORTED=9`.
- Strict Power BI still blocks on `cross_year_near_clone_warnings=1`.

Remaining P2 before any production source hierarchy policy:

- Bind accepted quote conflicts to exact conflict identities or input CSV hash,
  not only market/snapshot/count.
