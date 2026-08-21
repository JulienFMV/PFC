# Session Handoff - 2026-07-09 - T056 Selection Governance

## Scope

Phase 14 LT only. No CT code touched. OMPEX remains advisory-only and was not
used for model, calibration, selection, or gates.

User ask: launch expert auditors for the next steps and continue toward the
best expert flow. Three read-only auditors reviewed:

- T056 promotion readiness / manifest chain;
- QUOTE_CONFLICT source hierarchy policy;
- quant/model selection methodology.

## Changed Files

Code/tests/docs:

- `scripts/summarize_epex_shape_lab_spot_backtests.py`
- `tests/test_summarize_epex_shape_lab_spot_backtests_script.py`
- `scripts/check_epex_lab_promotion_readiness.py`
- `scripts/build_epex_lab_adjusted_production_manifest.py`
- `scripts/build_epex_lab_adjusted_production_chain.py`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t056_asof20260707_postval_final_micro.json`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260709-T056-SELECTION-GOVERNANCE.md`

Generated local evidence, not commit targets unless explicitly requested:

- `output/phase14/t056_postval_final_micro/t005_diagnostics/product_normalization_initial/`
- `output/phase14/t056_postval_final_micro/t005_diagnostics/product_normalization_with_policy/`
- `output/phase14/t056_postval_final_micro/t005_diagnostics/powerbi_strict/`
- `output/phase14/t056_postval_final_micro/t005_diagnostics/promotion_readiness/`
- `output/phase14/t056_postval_final_micro/t005_diagnostics/staged_adjusted_candidate_selection_guard/`

## Decision Updates

Added to `DECISION-LOG.md`:

- D-20260709-52: T056 replacement selection requires explicit selected
  artifact.
- D-20260709-53: T056 diagnostics pass but production promotion remains NO-GO.

## Selection Fix

Problem found: the EPEX lab spot-backtest summarizer could select the top weak
bucket even when it degraded a core incumbent metric. T056 exposed this:

- top weak bucket: `t001_w075_l025_p088_e005_n055_r00`
  - best overall score but degrades post-valuation;
- true replacement candidate:
  `t005_w075_l025_p089_e005_n055_r00`
  - beats T046 on every declared core metric.

Fix:

- selection summary now writes `best_replacement_trial`;
- `selected_trial` prefers replacement candidate over top weak bucket;
- `selected_adjusted_csv_sha256` binds the exact selected CSV;
- ranking CSV includes `replacement_candidate` and `degraded_vs_incumbent`.

Selected T056 artifact:

- trial: `t005_w075_l025_p089_e005_n055_r00`
- adjusted CSV:
  `output/phase14/t056_postval_final_micro/t005_w075_l025_p089_e005_n055_r00/candidate_epex_shape_lab_adjusted.csv`
- adjusted CSV sha256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- selection summary:
  `output/phase14/t056_postval_final_micro_selection_summary/spot_backtest_selection_summary.json`
- selection summary sha256:
  `b2a319ac91eff51947387bc2a1dcc4784b2f5bf5536ea861f2e63ab9fc5cf10d`
- `replacement_candidate_count=1`
- `replacement_verdict.replace_incumbent=true`
- OMPEX flags false for model, selection, and backtest.

T056 t005 vs T046 incumbent:

- overall `0.4506842423821014` vs `0.40548354103189205`
- evening `0.4688940576897349` vs `0.45338812791781463`
- solar-tail `0.46091530831501754` vs `0.4372953091304925`
- weekend `0.3283653976588017` vs `0.2889611347370835`
- post-valuation `0.3049947368951571` vs `0.3048038417338681`
- night `0.16252506955713483` vs `0.03190894115068499`
- ramp `0.053194830053255315` vs `0.035478178105887714`

Important caveat: post-valuation improvement is tiny and based on a short
24-hour window. Do not keep micro-tuning against post-valuation; treat it as a
non-degradation veto.

## CLI Hardening

Direct `python scripts/check_epex_lab_promotion_readiness.py --help` initially
failed with:

`ModuleNotFoundError: No module named 'scripts.epex_lab_selection_policy'`

The three promotion CLIs now keep the package import for pytest/module use and
fall back to local import for direct script execution:

- `scripts/check_epex_lab_promotion_readiness.py`
- `scripts/build_epex_lab_adjusted_production_manifest.py`
- `scripts/build_epex_lab_adjusted_production_chain.py`

Verified direct `--help` for all three scripts.

## Product Normalization Diagnostics

Initial audit without policy:

```powershell
python scripts/audit_ch_product_normalization.py --csv output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\candidate_epex_shape_lab_adjusted.csv --forwards output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\diagnostic_forwards_history_rebuilt_20260708.parquet --required-forward-date 2026-07-07 --output-csv output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_initial\gates.csv --summary-json output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_initial\summary.json --allow-failed-gates
```

Result:

- `all_gates_pass=false`
- `critical_count=0`
- `unsupported_count=0`
- `blocking_quote_conflict_count=6`
- `quote_conflict_identity_hash=a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`
- `input_csv_sha256=5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- `forwards_sha256=a6244638c2234781853284ce2ad58d55d01265568cca6c85d4461f21446e8d76`

Created hash-bound policy:

`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t056_asof20260707_postval_final_micro.json`

Policy sha256:

`71abb1151bf4f46728baffbdb6e6398c4a9a70e7273c4cc22fdb6a4fdfa73962`

Strict audit with policy:

```powershell
python scripts/audit_ch_product_normalization.py --csv output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\candidate_epex_shape_lab_adjusted.csv --forwards output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\diagnostic_forwards_history_rebuilt_20260708.parquet --required-forward-date 2026-07-07 --source-hierarchy-policy .planning\phases\14-lt-audit-remediation\quote_conflict_source_hierarchy_policy_t056_asof20260707_postval_final_micro.json --output-csv output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_with_policy\gates.csv --summary-json output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_with_policy\summary.json
```

Result:

- `all_gates_pass=true`
- `critical_count=0`
- `unsupported_count=0`
- `accepted_quote_conflict_count=6`
- `blocking_quote_conflict_count=0`
- source hierarchy policy status:
  `ACCEPTED_PRODUCTION_APPROVED`

The policy approves only the QUOTE_CONFLICT source hierarchy for the exact
bound CSV/forwards/conflict identity. It is not production approval for the
curve.

## Power BI Strict Diagnostics

Command:

```powershell
python scripts/build_powerbi_exports.py --csv output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\candidate_epex_shape_lab_adjusted.csv --forwards output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\diagnostic_forwards_history_rebuilt_20260708.parquet --spot output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t056_postval_final_micro\t005_diagnostics\powerbi_strict
```

Summary:

`output/phase14/t056_postval_final_micro/t005_diagnostics/powerbi_strict/summary_metrics.csv`

Result:

- `powerbi_quality_gate_status=PASS`
- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `weighted_negative_hours=0`
- seasonal/monthly/calendar/cross-year critical flags `0`

## Promotion Readiness

Readiness with diagnostics only:

```powershell
python scripts/check_epex_lab_promotion_readiness.py --lab-manifest output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\ab_lab_manifest.json --governance-audit output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\governance_audit\epex_shape_lab_governance_audit.json --independent-summary output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\independent_ab_comparison\ab_comparison_summary.json --product-summary output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_with_policy\summary.json --powerbi-summary output\phase14\t056_postval_final_micro\t005_diagnostics\powerbi_strict\summary_metrics.csv --output output\phase14\t056_postval_final_micro\t005_diagnostics\promotion_readiness\decision.json
```

Result:

- `strict_diagnostics_pass=true`
- `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
- missing evidence:
  `adjusted_production_manifest`, `adjusted_export_manifest`,
  `adjusted_selected_config`, `adjusted_capstone`

Staged reproducibility command:

```powershell
python scripts/stage_epex_lab_adjusted_lt_candidate.py --candidate-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --source-export-manifest output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_stage_t046_source_export_manifest\source_export_manifest.json --output-dir output\phase14\t056_postval_final_micro\t005_diagnostics\staged_adjusted_candidate_selection_guard --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --valuation-timestamp 2026-07-07T00:00:00Z --weekend-intensity 0.75 --low-tail-intensity 0.25 --peak-subshape-intensity 0.89 --evening-recovery-intensity 0.05 --night-intensity 0.55 --ramp-intensity 0.0 --max-abs-delta-eur-mwh 2.75 --negative-price-floor -30.0 --max-weighted-negative-hours 0 --lookback-years 5 --baseline-monthly-manifest output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\fan_asof20260707_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json --product-summary output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_with_policy\summary.json --powerbi-summary output\phase14\t056_postval_final_micro\t005_diagnostics\powerbi_strict\summary_metrics.csv --source-hierarchy-policy .planning\phases\14-lt-audit-remediation\quote_conflict_source_hierarchy_policy_t056_asof20260707_postval_final_micro.json --independent-summary output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\independent_ab_comparison\ab_comparison_summary.json --governance-audit output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\governance_audit\epex_shape_lab_governance_audit.json --selection-summary output\phase14\t056_postval_final_micro_selection_summary\spot_backtest_selection_summary.json
```

Result:

- staged adjusted CSV sha256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- source CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- source provenance manifest sha256:
  `347d617a23cddd35e3f3a791d42b205e2989c04885fa03ebb23942d9d5c5d2e6`
- adjusted production manifest NO-GO sha256:
  `052fdd1c3bc82cfe41f8e3600c9f577a1b571be99a2ba20123ad6118b7747c8d`
- `adjusted_production_contract_pass=true`
- `production_approved=false`
- `production_promotion_approved=false`

Readiness with staged manifest:

```powershell
python scripts/check_epex_lab_promotion_readiness.py --lab-manifest output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\ab_lab_manifest.json --governance-audit output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\governance_audit\epex_shape_lab_governance_audit.json --independent-summary output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\independent_ab_comparison\ab_comparison_summary.json --product-summary output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_with_policy\summary.json --powerbi-summary output\phase14\t056_postval_final_micro\t005_diagnostics\powerbi_strict\summary_metrics.csv --adjusted-production-manifest output\phase14\t056_postval_final_micro\t005_diagnostics\staged_adjusted_candidate_selection_guard\adjusted_production_manifest_no_go.json --output output\phase14\t056_postval_final_micro\t005_diagnostics\promotion_readiness\decision_with_staged_manifest.json
```

Result:

- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
- failing checks are intentional:
  - `adjusted_production_manifest_approved`
  - `adjusted_production_manifest_run_identity_valid` because git commit is
    null in the CLI/staging NO-GO manifest.
- missing evidence:
  `adjusted_export_manifest`, `adjusted_selected_config`,
  `adjusted_capstone`

## Expert Auditor Conclusions

Promotion-readiness auditor:

- T056 selection evidence is good, but promotion was missing product audit,
  Power BI strict, source/provenance chain, adjusted production manifest, and
  production chain. Product/Power BI/source provenance have now been produced
  locally; production approval and export/selected/capstone remain missing.

QUOTE_CONFLICT/source hierarchy auditor:

- T056 policy is acceptable only as a waiver for the six source hierarchy
  conflicts.
- It must be hash-bound to T056 CSV, forwards snapshot, and conflict identity.
- It must not be treated as model approval or production promotion.

Quant/model auditor:

- T056 t005 is defensible as the current no-OMPEX lab replacement candidate.
- The post-valuation edge is tiny and should not be tuned further.
- Next scientific step should be a locked future no-OMPEX holdout, not another
  post-val micro-sweep.

## Current NO-GO

T056 is not production promoted.

Still required for promotion:

- a real adjusted production run with run identity:
  `production_run_id`, `production_entrypoint`, 40-hex `git_commit`;
- an approved adjusted production manifest with approval flags true and
  selection-policy pass;
- adjusted export manifest;
- adjusted selected artifact;
- adjusted capstone;
- preferably a locked future no-OMPEX holdout before treating T056 as final.

Do not use OMPEX for any of these gates.
