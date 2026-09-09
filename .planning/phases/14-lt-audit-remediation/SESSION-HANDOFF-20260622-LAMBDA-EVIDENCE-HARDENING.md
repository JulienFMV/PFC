# Session Handoff - 2026-06-22 - Lambda Evidence Hardening

## Scope

Micro-phase after external cloud audit of pushed commit `c7e8ab6`.

Goal: make the structural `Lambda(t)` activation auditable and reproducible,
not merely locally green.

This session did not touch CT intentionally and did not write to
`powerbi/data/*`. Existing dirty Power BI/report/data changes remain outside
this micro-phase.

## Starting Point

External audit verdict accepted:

- `c7e8ab6` plumbing was useful but did not really activate structural
  `Lambda(t)` in pushed code.
- Local `64 passed` count was from a dirty worktree, because
  `tests/test_build_powerbi_exports_script.py` was not pushed.
- Pushed code/test mismatch: test expected
  `allow_template_structural_fallback=True`, code kept fallback disabled.
- A simple default flip was rejected unless diagnostics expose
  `STRUCTURAL_TEMPLATE`, amplitude, source, parent zero-mean residuals and
  far-horizon status.

Expert read-only roasts were run before coding:

- Quant/prior: no-go for promotion until structural fallback evidence,
  fallback reasons, history support and full prior knobs enter diagnostics and
  manifest/hash contracts.
- Repro/cloud: cloud cannot reproduce local counts unless the Power BI export
  test is tracked; use a curated commit, not `git commit -a`.
- PFC quality: monthly BASE chain is fixed, but delivered graph remains
  non-promotion-ready; next quality work is PEAK residuals, structural
  width/quantiles and cross-year Q4 allocation.

## Changed Files In This Micro-Phase

Intentional code/test changes:

- `pfc_shaping/calibration/monthly_curve_priors.py`
- `pfc_shaping/calibration/monthly_forward_curve.py`
- `pfc_shaping/pipeline/monthly_curve_authority.py`
- `tests/test_monthly_forward_curve_priors.py`
- `tests/test_monthly_forward_curve_integration.py`
- `tests/test_build_powerbi_exports_script.py` is now `git add -N` so audit
  diffs include the cloud-reproducibility test.

Planning changes:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-LAMBDA-EVIDENCE-HARDENING.md`
- `.planning/HANDOFF.md`

Pre-existing or unrelated dirty files are still present in the worktree,
including Power BI layout changes, generated data/output files and other
scripts/tests from earlier phases. Do not revert them without explicit
instruction.

## Implementation Details

Structural prior diagnostics:

- `build_structural_monthly_shape_prior(...)` now accepts and records
  `fallback_reason` and `history_counts`.
- `build_structural_monthly_shape_prior_from_history(...)` records:
  `empty_history`, `no_monthly_cal_history` or
  `insufficient_monthly_history`.
- Diagnostics now include `fallback_reason` and `n_history` per month.

Monthly authority manifest:

- Manifest now includes `structural_prior_summary`.
- Summary records status, diagnostic row count, sources, fallback reasons,
  amplitude min/max, parent residual min/max, history count min/max and
  `zero_mean_parent_space_all`.
- `active_config_hash` now hashes the full active prior-stack payload rather
  than the narrow lambda config subset. It includes:
  `markets`, `history_lookback_years`, `min_structural_snapshots`,
  `allow_template_structural_fallback`, `structural_amplitude_eur_mwh`,
  `panel_weight`, `history_weight`, `structural_weight` and monthly curve
  settings.

Lambda calibration baseline:

- `_parent_flat_baseline(...)` no longer leaves months without an active
  bucket at zero during withheld-product scoring.
- Missing masked months are filled with the duration-weighted average of
  represented bucket values when possible.
- This prevents point-in-time lambda calibration from manufacturing an 80
  EUR/MWh error in a flat synthetic fixture solely because the product was
  masked.

## Diagnostic Manifest Smoke Check

Manual smoke solve with empty history and structural fallback enabled produced:

- `active_config_hash`:
  `459e85c0ba35108f9c57c5789e1c6043e0e8c500bc3ebd72df6daddbd09bd847`
- `structural_status`: `STRUCTURAL_TEMPLATE`
- `structural_prior_summary`:
  `{'status': 'STRUCTURAL_TEMPLATE', 'diagnostic_rows': 12, 'sources': ['template_structural_monthly_ratios'], 'fallback_reasons': ['empty_history'], 'amplitude_eur_mwh_min': 110.0, 'amplitude_eur_mwh_max': 110.0, 'max_abs_parent_mean_residual_min': 2.0831303293012557e-15, 'max_abs_parent_mean_residual_max': 2.0831303293012557e-15, 'n_history_min': 0.0, 'n_history_max': 0.0, 'zero_mean_parent_space_all': True}`

## Commands And Results

Initial verification before this micro-phase:

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_build_powerbi_exports_script.py tests/test_export_local_test_ch_hourly_csv_script.py -q
```

Result: `64 passed in 26.77s`.

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_monthly_forward_curve_solver.py tests/test_audit_ch_hfc_seasonal_coherence_script.py -q
```

Result: `18 passed in 9.83s`.

After adding diagnostics/hash tests:

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py -q
```

Result: `39 passed in 3.38s`.

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_monthly_forward_curve_solver.py tests/test_audit_ch_hfc_seasonal_coherence_script.py -q
```

Result: `18 passed in 2.66s`.

```powershell
$env:PYTHONPATH='.'; python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_build_powerbi_exports_script.py tests/test_export_local_test_ch_hourly_csv_script.py -q
```

Result: `67 passed in 23.73s`.

Broad guardrail:

```powershell
$env:PYTHONPATH='.'; $files = Get-ChildItem tests -Filter 'test_monthly_forward_curve_*.py' | ForEach-Object { $_.FullName }; python -m pytest $files tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_long_term_branch.py tests/test_lt_ct_imports.py -q
```

Result: `129 passed, 1 skipped, 1 warning in 15.50s`.

Warning:

- `RuntimeWarning: All-NaN slice encountered` in
  `monthly_curve_priors.py::test_fail_closed_if_history_is_insufficient`.
  This occurs in an insufficient-history path and did not fail tests.

## Current Truth State

The structural default and diagnostics are now internally consistent in the
local worktree, and the cloud-reproducibility test file is visible to audit
diffs.

This is not a production or Power BI promotion. The Phase 2 delivered-curve
diagnostic still stands:

- solver monthly mean -> assembler `B` -> `price_shape` -> hourly CSV preserves
  monthly BASE means.
- delivered curve remains not promotion-ready with prior Phase 2 score
  `3.25/10`.
- open defects are PEAK residuals, quantile/width sidecars and cross-year Q4
  allocation.

## Next Recommended Phase

1. Curate the commit scope so cloud sees the exact code and tests used locally.
   Do not include unrelated generated data, Power BI layout churn or large desk
   files unless explicitly requested.
2. Regenerate a fresh diagnostic candidate only after the curated lambda
   evidence patch is committed/pushed.
3. Attack delivered-curve quality in this order:
   PEAK calibration, structural quantile/width sidecars, then cross-year Q4
   allocation.
4. Continue each phase with read-only expert roast before coding and an
   auditor handoff before closeout.
