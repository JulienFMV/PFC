# Session Handoff - 2026-06-24 - YoY50 Candidate Passes Strict Gates

## Scope

Continued Phase 14 toward an auditable CH LT candidate after source hierarchy
full-binding governance. Work stayed in LT/audit scope; no CT files, Power BI
data files, or heavy data files were edited.

## Changed Files

- `scripts/audit_ch_product_normalization.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `tests/fixtures/monthly_curve_phase_e_parity_baseline.json`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_lshape25_yoy50_structural_s126.json`
- `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_lshape25_yoy50_structural_s126.json`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260624-YOY50-CANDIDATE-PASSES-STRICT.md`
- `.planning/HANDOFF.md`

Generated local artifacts remain under `output/phase14/...` and are not commit
targets unless explicitly requested.

## Audit Logic Change

Delivered-product audit now distinguishes:

- `OUT_OF_SCOPE`: the full EEX product window is outside the delivered artifact
  window. It is reported with info severity and does not block when in-scope
  evidence exists.
- `UNSUPPORTED`: missing in-scope delivered rows or required quote evidence.
  It remains blocking.

The audit fails closed with `audit_evidence_present=CRITICAL` if it emits only
out-of-scope rows.

## Candidate Generation

Selected candidate:

`output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/`

Generation command:

```powershell
python scripts/export_local_test_ch_hourly_csv.py --enable-monthly-forward-curve-solver --enable-eex-peak-calibration --monthly-solver-lambda-shape 25 --monthly-solver-lambda-smooth-month 0.1 --monthly-solver-lambda-smooth-yoy 50 --enable-structural-shape-upgrade --structural-shape-upgrade-intensity 0.5 --structural-scenario-spread-intensity 1.26 --required-forward-date 2026-06-17 --output output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/ch_hfc_hourly_parent_local_prior_lshape25_yoy50_structural_s126_20260613_20301231.csv --report output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/export_report.md --fan-chart-output output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/ch_hfc_fan_chart_parent_local_prior_lshape25_yoy50_structural_s126_20260613_20301231.parquet --skip-powerbi-refresh
```

Relevant manifest hashes:

- `active_config_hash`: `9a29207f20efd39be80a33b3fa1ffc4c02b28daa2fd550e1db20837d1f8966db`
- `active_constraints_hash`: `efb01468d31e43f9c6cd66102ad5c573a9aacc8913e26b1a20139358174144cf`
- `monthly_solution_hash`: `7a4e09fb58f0699ce022f0ccc7d7ec47245f660b011781a9b4ad5f5a56d81bd5`
- `solver_config_hash`: `5c2bf87b6a5299e75f0d274e91ac286a1c878cb2bec67e2e48f052b0524d2aaf`

Probe outcomes:

- `lshape25_yoy10`: strict Power BI failed with
  `cross_year_near_clone_warnings=1`.
- `lshape25_yoy2`: strict Power BI failed with
  `cross_year_near_clone_warnings=2`.
- `lshape40_yoy10`: strict Power BI failed with
  `cross_year_near_clone_warnings=1`.
- `lshape25_yoy50`: strict Power BI passed.

## Delivered-Product Audit

Command:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/ch_hfc_hourly_parent_local_prior_lshape25_yoy50_structural_s126_20260613_20301231.csv --forwards data/eex_forwards_history.parquet --required-forward-date 2026-06-17 --price-column price_weighted_mean_eur_mwh --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_lshape25_yoy50_structural_s126.json --output-csv output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/delivered_product_normalization_gates_with_policy.csv --summary-json output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/delivered_product_normalization_summary_with_policy.json
```

Result: exit `0`, `all_gates_pass=true`.

Summary:

- `PASS=70`
- `QUOTE_CONFLICT=9`
- `accepted_quote_conflict_count=9`
- `blocking_quote_conflict_count=0`
- `UNSUPPORTED=0`
- `OUT_OF_SCOPE=9`
- `critical_count=0`
- `delivered_curve_drift_count=0`
- `input_csv_sha256`: `6bc75c9aac0628210d301bbd7bc2bee0e4e86246f94cad53696a4bee2dfda8b1`
- `forwards_sha256`: `c4bedaeb4cf7a04324bcf667be35ef9f92eeb2118c431109220076b114f9a3c5`
- `quote_conflict_identity_hash`: `b13d9c87813f9cbf9c43d8cbbe0bb533b029e0845b80e363aee7aaa2946a66f9`
- source hierarchy policy sha:
  `c529e1da3f76e3158a089f9834eb42fc90e9c787de776b61014b5023ee38b00a`

The new policy is production-approved for this exact bound artifact only.

## Strict Power BI Export

Command:

```powershell
python scripts/build_powerbi_exports.py --csv output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/ch_hfc_hourly_parent_local_prior_lshape25_yoy50_structural_s126_20260613_20301231.csv --forwards data/eex_forwards_history.parquet --spot data/epex_hourly.parquet --output-dir output/phase14/20260624_parent_local_prior_lshape25_yoy50_structural_s126/powerbi_strict
```

Result: exit `0`.

Key `summary_metrics.csv` values:

- `powerbi_quality_gate_status=PASS`
- `powerbi_quality_gate_issues=` empty
- `shape_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `negative_gate_status=PASS`
- `seasonal_critical_flags=0`
- `monthly_split_critical_flags=0`
- `monthly_path_critical_flags=0`
- `cross_year_month_shape_critical_flags=0`
- `cross_year_month_shape_warning_flags=0`

## Selected Config Artifact

Created:

`.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_lshape25_yoy50_structural_s126.json`

It is marked:

- `production_approved=true`
- `selection_status=PRODUCTION_APPROVED`
- `promotion_scope=LOCAL_CANDIDATE_SELECTION_ONLY`
- `production_manifest_triad_validated=false`
- `production_promotion_approved=false`
- `config_hash=9a29207f20efd39be80a33b3fa1ffc4c02b28daa2fd550e1db20837d1f8966db`
- `monthly_solution_hash=7a4e09fb58f0699ce022f0ccc7d7ec47245f660b011781a9b4ad5f5a56d81bd5`
- `active_constraints_hash=efb01468d31e43f9c6cd66102ad5c573a9aacc8913e26b1a20139358174144cf`

Governance parity was simulated using the candidate manifest as both the
production and export manifest; all selected-config gates passed in that
simulation. This is not real production promotion evidence.

## Tests

Targeted product audit:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py -q
```

Result: `36 passed in 15.02s`.

After roaster feedback, two additional regression tests were added for:

- a partial boundary month classified as `OUT_OF_SCOPE` while the next full
  month passes in scope;
- an accepted `QUOTE_CONFLICT` policy preserving `OUT_OF_SCOPE` counters.

Rerun:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py -q -p no:cacheprovider
```

Result: `38 passed in 17.52s`.

Governance/monthly parity targeted rerun:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_monthly_forward_curve_integration.py -q -p no:cacheprovider
```

Result: `24 passed in 5.75s`.

Targeted LT/audit regression suite:

```powershell
python -m pytest tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_forward_curve_integration.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `126 passed, 1 skipped in 63.80s`.

The monthly parity fixture hash was updated to the current deterministic hash
after verifying direct/history equality, unchanged active constraints hash, and
unchanged quoted/synthetic key lists. The prior fixture was stale after earlier
monthly-authority changes.

## Roaster Follow-Up

Read-only MIT roasters were launched after the green regression suite:

- Kepler: GO on `OUT_OF_SCOPE` / `UNSUPPORTED` fail-closed logic; requested
  two non-blocking test gaps, both added and green.
- Mencius: GO for local candidate governance, NO-GO production; warned that
  selected config could be overread if isolated from the handoff. The artifact
  now carries `promotion_scope=LOCAL_CANDIDATE_SELECTION_ONLY`,
  `production_manifest_triad_validated=false`, and
  `production_promotion_approved=false`.
- Sagan: GO for candidate evidence and tests, NO-GO production until the real
  production/export/selected triad is validated.

All roasters agree: commit is acceptable for a local auditable candidate, but
not for production promotion.

## Current Verdict

Local Phase 14 candidate is auditable:

- delivered-product audit passes with exact artifact-bound source hierarchy
  policy;
- strict Power BI export passes without `--allow-failed-gates`;
- selected config artifact exists and is production-approved for the selected
  hash set.

Production remains NO-GO until the real production manifest, local export
manifest, and selected config artifact are regenerated/checked as one manifest
triad. The current repository production manifest is still from an older
configuration, so simulated parity must not be treated as promotion evidence.

## Next

1. Run the targeted regression suite after this handoff update.
2. Commit only code/tests/docs/policy artifacts, not generated `output/phase14`
   data.
3. Run the real production/local-export/selected manifest promotion check when
   ready to promote.
