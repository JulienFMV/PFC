# Session Handoff - 2026-07-08 - Daily Generation As-Of 2026-07-07

## Scope

Generated and audited the CH LT PFC for Wednesday 2026-07-08, bound to the real
EEX quote snapshot `2026-07-07`. Work stayed in LT/Phase 14 scope. No
`pfc_shaping/ct/*`, `powerbi/data/*`, or `powerbi/PFC_QA.*` files were edited.

Important date convention: the workbook was available on 2026-07-08, but the
latest usable CH/DE/FR quote rows inside it are dated 2026-07-07. All current
evidence is therefore bound to `forward_snapshot_date=2026-07-07`.

## Data Refresh

Workbook used:

`H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx`

Observed workbook timestamp: `2026-07-08 05:03:28` local filesystem time.

Local forward history was refreshed with
`pfc_shaping.data.ingest_forwards.update_forwards_parquet(...)`.

Coverage after refresh:

- rows: `147540`
- CH latest BASE date: `2026-07-07`
- DE latest BASE date: `2026-07-07`
- FR latest BASE date: `2026-07-07`
- AT latest BASE date: `2026-06-17`
- IT latest BASE date: `2026-06-17`

AT/IT were not present in the daily workbook; exact-as-of neighbor evidence for
the 2026-07-07 solve used DE/FR only.

## Final Candidate

Candidate:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/`

Local export result:

- rows: `57025`
- weighted mean: `79.52`
- scenario spread mean: `0.5205`
- CSV sha256: `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- export manifest sha256: `cb52a502e8e95af2e5f3fabc3b2b34ca8f365999214cfd7c53718ed7f5ef456a`

Manifest values:

- `active_config_hash=f95e81bf8987174eb8b553406de296fc8cfb67a3dfde35f006b88cc006a66469`
- `active_constraints_hash=fd95393bd94c2ce5d6ff02ba5c57a0633d00cbc9f6acc540877802fc81a2a7ab`
- `monthly_solution_hash=3882baa358bb2479d4b25aec464b45d74c15713f36ee34d0389790e848430c9e`
- `forward_snapshot_date=2026-07-07`

## Strict Gates

Power BI strict:

- `powerbi_quality_gate_status=PASS`
- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `seasonal_critical_flags=0`
- `seasonal_warning_flags=0`
- `monthly_path_critical_flags=0`
- `monthly_path_warning_flags=4`
- `cross_year_month_shape_critical_flags=0`
- `cross_year_month_shape_warning_flags=0`
- `latest_hfc_winter_summer_spread_eur_mwh=24.91`

Delivered-product audit strict result:

- `all_gates_pass=true`
- `PASS=90`
- `QUOTE_CONFLICT=6`
- `accepted_quote_conflict_count=6`
- `blocking_quote_conflict_count=0`
- `UNSUPPORTED=0`
- `critical_count=0`
- `delivered_curve_drift_count=0`
- `quote_conflict_identity_hash=a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`
- source hierarchy policy sha256: `7f9db8f436b175d95496ad299bde9c370ed8b4ccebd4692db4acc1a740632806`

## Production Generation

Production LT save result:

- `LT_PRODUCTION_SAVE_OK`
- today: `2026-07-08`
- markets: `CH`, `DE`
- CH rows: `227328`, source `EEX XLSX CH (40 keys)`
- DE rows: `227328`, source `EEX XLSX DE (69 keys)`

Production outputs:

- `pfc_shaping/output/pfc_15min_2026-07-08.csv`
- `pfc_shaping/output/pfc_15min_2026-07-08.parquet`
- `pfc_shaping/output/pfc_de_15min_2026-07-08.csv`
- `pfc_shaping/output/pfc_de_15min_2026-07-08.parquet`
- `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`

Production manifest sha256:

`a7589cf7b52ed36b8dc993cb8f1bbf8cdf402458a3a178e7e1cce1b775331987`

Production manifest values:

- `active_config_hash=f95e81bf8987174eb8b553406de296fc8cfb67a3dfde35f006b88cc006a66469`
- `active_constraints_hash=fd95393bd94c2ce5d6ff02ba5c57a0633d00cbc9f6acc540877802fc81a2a7ab`
- `monthly_solution_hash=3882baa358bb2479d4b25aec464b45d74c15713f36ee34d0389790e848430c9e`

Follow-up source-hash hardening regenerated the production manifest with:

- `source_hashes.forwards_path=159680087cb2f2de6322863660fb481fa531ebc9239e40de4f3735ecdc382ea1`
- `source_hashes.eex_report_path=dedae2a6d66ce59b9e3d4a0ab7c85e6800d8eb7d3e911d37651711a393fd4005`
- regenerated production manifest sha256:
  `9b6b238bcbce72bb485f29ce1c6142ebce15b696d8df0a36fc5c673a2dbd4598`

## Promotion Evidence

Selected config artifact:

`.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260707_lshape100_yoy150_amp150_2032.json`

Selected config sha256:

`5ca8b3dc3c1dfadf6b2153bc22f6d69cfb2ad767ae8dbada9615f753760e1f34`

Quote conflict policy artifact:

`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260707_lshape100_yoy150_amp150_2032.json`

Sparse proof:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/monthly_curve_sparse_year_proof_full_export_window_strict/`

Sparse proof status counts:

- `PASS=33`
- `WARNING=2`
- `UNSUPPORTED=7`
- `CRITICAL=0`

Capstone:

- `approved=true`
- `status=PROMOTION_EVIDENCE_PASS`
- `blocking_count=0`
- `audit_gate_status_counts={"PASS": 35, "UNSUPPORTED": 7, "WARNING": 2}`

## PNG Diagnostics

PNG diagnostics were generated in:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/png_diagnostics/`

Key PNGs:

- `01_monthly_means_by_year.png`
- `05_heatmap_month_hour_2030.png`
- `09_executive_qa_summary.png`

## OMPEX Benchmark

OMPEX benchmark is read-only and was not used in model inputs, priors,
objectives, calibration, or candidate selection.

Latest benchmark file observed:

`H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260707_101700.xlsx`

No `2026-07-08` OMPEX file was observed at the time of the baseline
production-ready benchmark. A later `HFC_Ompex_20260708_101700.xlsx` file was
used only for post-selection advisory checks on frozen EPEX lab artifacts, not
for model input or parameter selection.

Output:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ompex_benchmark_read_only/benchmark_metrics.json`

Metrics:

- overlap: `2026-07-01 00:00:00` -> `2031-01-01 00:00:00`
- points: `39473`
- MAE: `14.0197` EUR/MWh
- RMSE: `18.4823` EUR/MWh
- bias: `1.3495` EUR/MWh
- p95 absolute error: `38.0138` EUR/MWh
- correlation: `0.8331`
- `ompex_used_in_model=false`

## Tests

Command:

```powershell
python -m pytest tests/test_monthly_forward_curve_integration.py tests/test_check_monthly_curve_promotion_from_manifests.py -q -p no:cacheprovider
```

Result: `26 passed`.

Follow-up hardening test command:

```powershell
python -m pytest tests/test_long_term_branch.py tests/test_monthly_forward_curve_integration.py tests/test_check_monthly_curve_promotion_from_manifests.py -q -p no:cacheprovider
```

Result: `41 passed`.

## Roasters / MIT Audit

Three read-only Roasters/MIT agents audited the 2026-07-08 package after the
capstone.

Verdict: GO for Phase 14 promotion evidence.

P0 findings: none.

P1 findings:

- Quant/shaping and contamination agents found no P1 blocker.
- Governance agent flagged two packaging traceability issues, accepted as
  non-blocking because the selected config plus capstone are the promotion
  authority:
  - generated `export_report.md` remains a local-test report and still says
    production approval from the local report is `NO`; follow-up wording now
    points auditors to the manifest-backed capstone.
  - generated `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`
    initially had empty `source_hashes`; follow-up hardening now records both
    `forwards_path` and `eex_report_path`. Production/export/selected parity
    still holds on `active_config_hash`, `active_constraints_hash`, and
    `monthly_solution_hash`.

Accepted P2 findings:

- sparse proof has `WARNING=2`, `UNSUPPORTED=7`, `CRITICAL=0`; capstone
  accepts these with `blocking_count=0`.
- Power BI monthly path warnings remain `4`, with no monthly path critical,
  seasonal critical, cross-year critical, or EEX repricing residual.
- sparse-proof standalone `manifest.json` is less clear than the capstone
  because it records its own internal proof hash and `production_approved=false`;
  use the capstone decision as the authoritative promotion evidence.
- `data/eex_forwards_history.parquet` is modified locally from the refresh and
  must stay excluded from the commit.

Contamination check:

- no sign of OMPEX/HFC benchmark data entering `pfc_shaping/lt`.
- `rolling_update.py` runs HFC benchmark after build/export/persistence.
- config keeps benchmark policy advisory: `benchmark_policy=advisory`,
  `fail_on_benchmark=false`.
- `ompex_benchmark_read_only/benchmark_metrics.json` records
  `read_only=true` and `ompex_used_in_model=false`.

OMPEX quality caveat:

- OMPEX is an imperfect external benchmark, not ground truth.
- OMPEX must not be used as a model input, optimization target, calibration
  target, or production promotion authority.
- Improvements should be accepted only when they also improve independent
  physics/market diagnostics and preserve EEX BASE/PEAK gates.

Repeatable benchmark tooling:

- `scripts/compare_hpfc_ompex_benchmark.py`
- output from the 2026-07-08 run:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ompex_benchmark_read_only_20260708_scripted/`
- selected alignment: `ompex_minus_1h_hourending`
- metrics: `MAE=12.5271`, `RMSE=16.4805`, `bias=0.7010`,
  `correlation=0.8741`
- output JSON states `benchmark_policy=advisory`, `read_only=true`,
  `ompex_used_in_model=false`, and records the OMPEX quality caveat.

## Follow-up Hardening

Code changes:

- `pfc_shaping/pipeline/production_phases.py` now hashes the monthly solver
  history parquet and EEX workbook, then passes those hashes into the
  production monthly curve manifest.
- `scripts/export_local_test_ch_hourly_csv.py` now states that the generated
  report is not the production promotion authority and points to the
  selected-config plus capstone evidence.
- `tests/test_long_term_branch.py` covers the new source hash helper.

Validation:

- refreshed `data/eex_forwards_history.parquet` locally from the 2026-07-08 EEX
  workbook to `max_date=2026-07-07`, sha256
  `159680087cb2f2de6322863660fb481fa531ebc9239e40de4f3735ecdc382ea1`.
- regenerated production LT output for `today=2026-07-08`; CH manifest keeps
  `monthly_solution_hash=3882baa358bb2479d4b25aec464b45d74c15713f36ee34d0389790e848430c9e`.
- regenerated local export; CSV sha remains
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895` and
  export manifest sha remains
  `cb52a502e8e95af2e5f3fabc3b2b34ca8f365999214cfd7c53718ed7f5ef456a`.
- reran capstone with the source-hashed production manifest:
  `approved=true`, `status=PROMOTION_EVIDENCE_PASS`, `blocking_count=0`.

## Experimental EPEX A/B Shape Lab

After the OMPEX advisory benchmark was formalized, a new LT-only experimental
shape lab scaffold was added for the next model-improvement cycle.

Files:

- `pfc_shaping/lt/model/epex_shape_lab.py`
- `tests/test_epex_ab_shape_lab.py`

Scope:

- off by default; not wired into production/export
- no OMPEX/HFC input, target, loss, calibration, or promotion gate
- fits shape-only templates from CH EPEX spot residuals before the configured
  valuation timestamp; fitting fails without an explicit valuation timestamp
- supports weekend, low-tail, and peak-subshape additive deltas
- projects candidate deltas into the nullspace of quote-aware
  BASE/PEAK/OFFPEAK constraints before applying them
- requires monthly BASE constraints for every delivered month by default
- applies the same projected delta to slow/central/fast scenarios, shifts the
  existing weighted mean/fan by that delta, and preserves structural width

Validation:

```powershell
python -m pytest tests/test_epex_ab_shape_lab.py -q -p no:cacheprovider
```

Result: `4 passed`.

```powershell
python -m pytest tests/test_epex_ab_shape_lab.py tests/test_seam_nullspace_smoothing.py tests/test_lt_quant_contract_matrix.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `34 passed, 1 skipped`.

Later validation after lab hardening:

- `python -m pytest tests/test_epex_ab_shape_lab.py -q -p no:cacheprovider`
  returned `9 passed`.
- `python -m pytest tests/test_epex_ab_shape_lab.py tests/test_seam_nullspace_smoothing.py tests/test_lt_quant_contract_matrix.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `39 passed, 1 skipped`.

## EPEX A/B Lab Runner

A local-only runner was added after the lab scaffold:

- `scripts/run_epex_shape_lab_ab.py`
- `tests/test_run_epex_shape_lab_ab_script.py`

The runner derives monthly BASE/PEAK constraints from the input hourly
candidate, fits EPEX residual templates with an explicit valuation timestamp,
applies the lab delta, and writes a lab-only manifest plus before/after
constraint residuals. It does not read OMPEX/HFC and is not wired into
production/export.

2026-07-08 local trial:

```powershell
python scripts/run_epex_shape_lab_ab.py --candidate-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --spot-parquet data/epex_hourly.parquet --output-dir output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial --valuation-timestamp 2026-07-07T00:00:00Z --weekend-intensity 0.5 --low-tail-intensity 0.5 --peak-subshape-intensity 0.5 --max-abs-delta-eur-mwh 6.0
```

Result:

- runtime after projection optimization: about `31` seconds
- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/`
- manifest: `activation_status=lab_only`, `production_approved=false`,
  `ompex_used_in_selection=false`
- source hashes:
  - candidate CSV:
    `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
  - EPEX spot parquet:
    `5718d243ef681476cabeabac7e866c0c7a63f686750283a2ff50a7d70c216a3d`
- constraints preserved:
  `base_monthly_constraints=78`, `peak_monthly_constraints=78`,
  `max_after_abs_error_eur_mwh=1.666666804567285e-07`
- `weighted_negative_hours=0`

Validation:

```powershell
python -m pytest tests/test_run_epex_shape_lab_ab_script.py tests/test_epex_ab_shape_lab.py -q -p no:cacheprovider
```

Result: `10 passed`.

Decision log entry: `D-20260708-06`.

## Independent A/B Comparison

A local-only independent comparison script was added after the lab runner:

- `scripts/compare_epex_shape_lab_ab.py`
- `tests/test_compare_epex_shape_lab_ab_script.py`

It compares baseline vs adjusted candidates without OMPEX and writes timestamp
alignment, monthly drift, annual shape, calendar-bucket deltas, fan-width
preservation, quantile-order, negative-hour, and ramp metrics.

2026-07-08 local comparison:

```powershell
python scripts/compare_epex_shape_lab_ab.py --baseline-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/candidate_epex_shape_lab_adjusted.csv --output-dir output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/independent_ab_comparison
```

Result:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/independent_ab_comparison/`
- `benchmark_policy=independent_no_ompex`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`
- `n_hours=57025`
- `finite_adjusted_ok=true`
- `quantile_order_adjusted_ok=true`
- `weighted_negative_hours_adjusted=0`
- `max_abs_monthly_mean_delta_eur_mwh=9.722222239124298e-08`
- `max_abs_width_delta_eur_mwh=0.0`
- `max_abs_delta_eur_mwh=6.000000000000002`
- solar-tail mean delta `-2.0652588766029956`
- midday mean delta `-1.8175837490740738`
- evening-ramp mean delta `0.9288002084175085`
- weekend mean delta `-0.6855023410557364`
- annual evening-minus-midday change is about `+2.75` EUR/MWh for 2027-2032

Validation:

```powershell
python -m pytest tests/test_compare_epex_shape_lab_ab_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

Decision log entry: `D-20260708-07`.

Next diagnostic step: run OMPEX comparison separately on the adjusted candidate
as advisory-only evidence. Do not use the OMPEX result to select A/B
parameters retroactively.

## OMPEX Advisory Post-Check On Adjusted A/B

After the independent no-OMPEX A/B comparison was recorded, OMPEX was run as a
post-check on the adjusted candidate:

```powershell
python scripts/compare_hpfc_ompex_benchmark.py --hpfc-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/candidate_epex_shape_lab_adjusted.csv --ompex-xlsx "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260708_101700.xlsx" --output-dir output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ompex_advisory_adjusted_20260708
```

Output:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ompex_advisory_adjusted_20260708/`

Adjusted metrics:

- alignment: `ompex_minus_1h_hourending`
- points: `39481`
- MAE: `12.328552488842737`
- RMSE: `16.247141314210175`
- bias: `0.7010425073326411`
- correlation: `0.8775633169206011`
- p95 absolute error: `32.776404`
- OMPEX inside p10/p90 rate: `0.15807603657455485`
- max absolute error: `101.939482`

Advisory delta vs baseline OMPEX benchmark:

- MAE `-0.1985248878447834`
- RMSE `-0.2333863878943987`
- correlation `+0.0035026506205321217`
- p95 absolute error `-0.5472940000000008`
- OMPEX inside p10/p90 rate `+0.0043058686456776685`
- max absolute error `+1.553652999999997`

Interpretation:

- Aggregate OMPEX advisory metrics moved in the right direction, but max
  absolute error worsened.
- This is not production approval and was not used to choose A/B parameters.
- Any further parameter change must be pre-registered and rerun through the
  independent no-OMPEX comparison first.

Decision log entry: `D-20260708-08`.

## EPEX A/B Governance Audit

A local-only governance audit script was added:

- `scripts/audit_epex_shape_lab_governance.py`
- `tests/test_audit_epex_shape_lab_governance_script.py`

It verifies that the lab artifacts remain lab-only, OMPEX is not used for
model or selection, independent comparison is no-OMPEX, monthly drift/fan drift
are below thresholds, and optional OMPEX metrics are advisory/read-only.

2026-07-08 command:

```powershell
python scripts/audit_epex_shape_lab_governance.py --lab-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ab_lab_manifest.json --independent-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/independent_ab_comparison/ab_comparison_summary.json --ompex-metrics output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ompex_advisory_adjusted_20260708/benchmark_metrics.json --output-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/governance_audit/epex_shape_lab_governance_audit.json
```

Result:

- status: `PASS`
- failed count: `0`
- production approval: `NO`
- promotion gate: `false`
- OMPEX role: `advisory_post_check_only`
- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/governance_audit/epex_shape_lab_governance_audit.json`

Validation:

```powershell
python -m pytest tests/test_audit_epex_shape_lab_governance_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

```powershell
python -m pytest tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `31 passed, 1 skipped`.

Decision log entry: `D-20260708-09`.

## Adjusted A/B Promotion-Style Diagnostics

Existing diagnostics were run on the adjusted A/B candidate as lab-only
evidence.

Important data note:

- the committed/local `data/eex_forwards_history.parquet` currently observed
  in this session had `max_date=2026-06-17`, too stale for `asof20260707`.
- a local Yearly-only diagnostic forwards parquet was therefore built under the
  trial folder:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/diagnostic_forwards_yearly_only.parquet`
- source workbook:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx`
- coverage: CH `2024-07-01 -> 2026-07-07`
- sha256:
  `63a40871677a0a82356de762d5a9ceb944a6b431f145d23598b8fb91e6966ce3`

Shape audit:

- adjusted:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/adjusted_shape_audit/shape_audit_report.md`
- baseline:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/baseline_shape_audit/shape_audit_report.md`
- both report `score=7.00/10`; no adjusted-vs-baseline score degradation
  under this local audit.

Power BI strict diagnostic on adjusted:

```powershell
python scripts/build_powerbi_exports.py --csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/candidate_epex_shape_lab_adjusted.csv --forwards output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/diagnostic_forwards_yearly_only.parquet --spot data/epex_hourly.parquet --output-dir output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/adjusted_powerbi_strict
```

Result:

- `powerbi_quality_gate_status=PASS`
- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `weighted_negative_hours=0`
- `negative_gate_status=PASS`
- `monthly_path_warning_flags=4`
- all critical flag counts are `0`

Product normalization diagnostic:

- adjusted:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/adjusted_product_normalization/`
- baseline:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/baseline_product_normalization/`
- both baseline and adjusted have `critical_count=0`, `unsupported_count=0`,
  `delivered_curve_drift_count=0`, `quote_conflict_count=6`, and
  `status_counts={"PASS": 90, "QUOTE_CONFLICT": 6}`.
- no source hierarchy policy was supplied, so `all_gates_pass=false`; this is
  expected and correct for lab-only evidence.

Decision log entry: `D-20260708-10`.

## Pre-Registered Next EPEX Sweep

A sweep plan generator was added:

- `scripts/plan_epex_shape_lab_sweep.py`
- `tests/test_plan_epex_shape_lab_sweep_script.py`

It writes a lab-only, no-OMPEX pre-registration plan with hashes, parameters,
and commands for each trial. It does not execute trials and does not read
OMPEX/HFC files.

2026-07-08 local plan:

```powershell
python scripts/plan_epex_shape_lab_sweep.py --candidate-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --spot-parquet data/epex_hourly.parquet --output-root output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1 --valuation-timestamp 2026-07-07T00:00:00Z --max-abs-delta-eur-mwh 6.0 --plan-id epex_shape_lab_sweep_v1_asof20260707 --output-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/pre_registered_sweep_plan.json
```

Result:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/pre_registered_sweep_plan.json`
- plan id: `epex_shape_lab_sweep_v1_asof20260707`
- trial count: `27`
- `benchmark_policy=pre_registered_independent_no_ompex`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`
- candidate CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- EPEX spot parquet sha256:
  `5718d243ef681476cabeabac7e866c0c7a63f686750283a2ff50a7d70c216a3d`

Validation:

```powershell
python -m pytest tests/test_plan_epex_shape_lab_sweep_script.py -q -p no:cacheprovider
```

Result: `1 passed`.

Decision log entry: `D-20260708-11`.

Next execution rule: run the planned trials and select using independent
no-OMPEX comparison plus governance PASS only. OMPEX advisory comparison may be
run after a trial is selected/frozen, not during parameter selection.

## Executed Pre-Registered EPEX Sweep

The pre-registered sweep was executed with a dedicated no-OMPEX executor:

- `scripts/execute_epex_shape_lab_sweep.py`
- `tests/test_execute_epex_shape_lab_sweep_script.py`

The executor was hardened after read-only audit feedback. It now rejects
malformed plans, duplicate trials, trial output directories outside the sweep
root, negative `--max-trials`, and stale resume artifacts. It also returns
`best_trial=null` when no trial is eligible, instead of promoting the top
ineligible row.

Local command:

```powershell
python scripts/execute_epex_shape_lab_sweep.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/pre_registered_sweep_plan.json --output-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/sweep_execution_summary.json
```

Local outputs:

- summary:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/sweep_execution_summary.json`
- ranking:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/sweep_execution_summary.csv`

Execution result:

- `benchmark_policy=executed_independent_no_ompex`
- `trial_count_planned=27`
- `trial_count_executed=27`
- `eligible_count=27`
- `production_approved=false`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`

Best selected no-OMPEX trial:

- `trial_002_w0.25_l0.25_p0.50`
- weekend intensity: `0.25`
- low-tail intensity: `0.25`
- peak-subshape intensity: `0.50`
- independent shape score: `6.350975764045719`
- duck-change mean: `3.6754139784914535` EUR/MWh
- solar-tail mean delta: `-2.535581627746391` EUR/MWh
- weekend mean delta: `-0.6477966303078719` EUR/MWh
- ramp p99 increase: `2.0312658899999896` EUR/MWh
- max monthly mean drift: `1.1155913942688404e-07` EUR/MWh
- max fan-width drift: `0.0`
- weighted negative hours: `0`
- governance status: `PASS`

Validation:

```powershell
python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `33 passed, 1 skipped`.

After executor hardening, the targeted suite was rerun:

```powershell
python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `37 passed, 1 skipped`.

Resume check on the existing sweep:

```powershell
python scripts/execute_epex_shape_lab_sweep.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/pre_registered_sweep_plan.json --output-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/sweep_execution_summary.json
```

Result: `{"eligible_count": 27, "trial_count_executed": 27}`.

Next evidence rule: the selected trial is frozen for any OMPEX advisory
post-check. Do not use OMPEX to re-rank or re-tune. This lab artifact is still
NO-GO production until regenerated through production/export/capstone evidence
with artifact-bound source hierarchy and selected-lambda/config manifests.

## Hardened Next-Sweep Selection Policy

Read-only expert audits accepted the baseline as promotion-ready but rejected
production adoption of any EPEX lab artifact. They also identified research
risks that must be hard gates before the next sweep: stale EPEX spot data,
cap saturation, weak ramp penalty, and insufficient negative-price controls.

The planner and executor were therefore hardened:

- `scripts/plan_epex_shape_lab_sweep.py` now writes explicit
  `selection_thresholds`, `scoring_policy`, and `max_abs_delta_grid`.
- `scripts/execute_epex_shape_lab_sweep.py` now applies those thresholds when
  deciding trial eligibility.
- The executor records per-trial EPEX spot age and fit coverage.
- Future plans can sweep cap values with `--max-abs-delta-grid-json`, for
  example `[2.0, 3.0, 4.0, 6.0]`.

Default new-plan thresholds:

- `max_epex_spot_age_days=14.0`
- `min_epex_fit_coverage_days=730.0`
- `max_ramp_p99_increase_eur_mwh=1.0`
- `min_adjusted_price_eur_mwh=-10.0`

Default new-plan scoring:

- `duck_weight=1.0`
- `solar_tail_weight=1.0`
- `weekend_weight=1.0`
- `ramp_penalty_weight=1.0`

Validation:

```powershell
python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `39 passed, 1 skipped`.

Decision log entry: `D-20260708-13`.

Next research rule: refresh EPEX spot first, then generate a new
pre-registered sweep plan with a delta-cap grid. The existing `trial_002`
remains frozen lab evidence, not a production candidate.

## Fresh-Spot EPEX Sweep V2

The stale `data/epex_hourly.parquet` and `pfc_shaping/data/epex_15min.parquet`
both ended at `2026-03-15 22:00 UTC`. A local generated refresh was built under
the Phase 14 output folder, without committing data caches:

- `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_15min_ch_energy_charts_20260708.parquet`
- `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- hourly coverage: `2023-01-01 00:00 UTC -> 2026-07-08 23:00 UTC`
- source: energy-charts CH prices for `2026-03-15 -> 2026-07-09`, merged with
  the local CH 15min cache.

The V2 sweep was pre-registered and executed:

```powershell
python scripts/plan_epex_shape_lab_sweep.py --candidate-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --spot-parquet output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet --output-root output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2 --valuation-timestamp 2026-07-07T00:00:00Z --max-abs-delta-eur-mwh 6.0 --max-abs-delta-grid-json "[2.0, 3.0, 4.0, 6.0]" --plan-id epex_sweep_v2_fresh_spot_asof20260707 --output-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/pre_registered_sweep_plan.json
```

```powershell
python scripts/execute_epex_shape_lab_sweep.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/pre_registered_sweep_plan.json --output-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/sweep_execution_summary.json
```

Results:

- plan id: `epex_sweep_v2_fresh_spot_asof20260707`
- trial count: `108`
- eligible count: `39`
- cap grid: `[2.0, 3.0, 4.0, 6.0]`
- `production_approved=false`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`

Best frozen no-OMPEX trial:

- `t046_w05_l025_p075_d03`
- weekend intensity: `0.5`
- low-tail intensity: `0.25`
- peak-subshape intensity: `0.75`
- cap: `3.0` EUR/MWh
- independent shape score: `2.2242277207731145`
- duck-change mean: `1.6614858713007632` EUR/MWh
- solar-tail mean delta: `-1.172939635010313` EUR/MWh
- weekend mean delta: `-0.3774464344619922` EUR/MWh
- ramp p99 increase: `0.9876442199999538` EUR/MWh
- min adjusted price: `-3.825623` EUR/MWh
- EPEX spot age: `0.041666666666666664` days
- EPEX fit coverage: `1282.9583333333333` days
- monthly mean drift: `8.602150532151586e-08` EUR/MWh
- fan-width drift: `0.0`
- weighted negative hours: `0`
- governance: `PASS`

OMPEX was then run only as advisory post-check against
`HFC_Ompex_20260708_101700.xlsx`. Advisory deltas selected minus baseline:

- MAE: `-0.13162060282161114`
- RMSE: `-0.1631453735600843`
- correlation: `+0.0024745976229031408`
- p95 absolute error: `-0.40473999999999677`
- inside p10/p90 rate: `+0.002482206631037709`
- max absolute error: `+0.987836999999999`

Validation:

```powershell
python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `40 passed, 1 skipped`.

Decision log entry: `D-20260708-14`.

Next rule: `t046_w05_l025_p075_d03` is better lab evidence than V1, but still
NO-GO production. Promotion would require regenerating it through the real
production/export/capstone path with artifact-bound source hierarchy and strict
gates.

## T047 V3 Night/Ramp EPEX Sweep Pre-Registration

Read-only expert audits after T046 recommended not promoting T046 immediately:
T046 strict diagnostics are strong, but no-OMPEX spot evidence remains weak on
night and hourly-ramp buckets, and T046 is close to the ramp p99 selection cap.
The next model step is therefore T047 v3, still lab-only and no-OMPEX.

Implementation changes:

- `pfc_shaping/lt/model/epex_shape_lab.py` now fits and applies
  `night_delta_eur_mwh` and `ramp_delta_eur_mwh` templates via new
  `ABShapeLabConfig.night_intensity` and `ABShapeLabConfig.ramp_intensity`.
- `scripts/run_epex_shape_lab_ab.py` accepts `--night-intensity` and
  `--ramp-intensity` and records them in manifests.
- `scripts/plan_epex_shape_lab_sweep.py` can pre-register the new dimensions,
  includes `night_weight` in scoring, and supports `@file.json` CLI arguments
  for robust PowerShell JSON passing.
- `scripts/execute_epex_shape_lab_sweep.py` passes/validates the new
  intensities and records them in ranking rows.
- `scripts/compare_epex_shape_lab_ab.py` adds `night_00_05` to independent
  calendar diagnostics.

Pre-registered T047 v3 plan:

- folder:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/`
- plan:
  `pre_registered_sweep_plan.json`
- trial count: `18`
- candidate CSV:
  `ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- spot parquet:
  `epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- grid:
  - weekend `0.5`
  - low-tail `0.25`
  - peak-subshape `0.75`
  - night `[0.0, 0.25, 0.5]`
  - ramp `[0.0, 0.25, 0.5]`
  - cap `[2.0, 3.0]`
- thresholds:
  - max EPEX spot age `14.0` days
  - min EPEX fit coverage `730.0` days
  - max ramp p99 increase `0.9` EUR/MWh
  - min adjusted price `-10.0` EUR/MWh
- scoring:
  - duck `1.0`
  - solar tail `1.0`
  - weekend `1.0`
  - night `0.75`
  - ramp penalty `1.5`

Smoke execution:

```powershell
python scripts/execute_epex_shape_lab_sweep.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/pre_registered_sweep_plan.json --output-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_smoke_summary.json --max-trials 2 --no-resume
```

Result:

- `trial_count_executed=2`
- `eligible_count=1`
- best smoke trial: `t001_w05_l025_p075_n00_r00_d02`
- ramp p99 increase: `0.6714740399999428` EUR/MWh
- max monthly drift: `1.0119047646367243e-07`
- width drift: `0`
- weighted negative hours: `0`
- governance: `PASS`
- the cap `3.0` smoke trial is not eligible under the tighter ramp threshold.

Validation:

```powershell
python -m pytest tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `40 passed, 1 skipped`.

Full T047 v3 sweep was then executed:

```powershell
python scripts/execute_epex_shape_lab_sweep.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/pre_registered_sweep_plan.json --output-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_summary.json
```

Result:

- `trial_count_executed=18`
- `eligible_count=9`
- `production_approved=false`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`

Best internal ranking trial:

- `t005_w05_l025_p075_n00_r05_d02`
- cap `2.0`, night `0.0`, ramp `0.5`
- independent shape score `1.315185990064843`
- ramp p99 increase `0.7062480299999798`
- max monthly drift `1.1111111105262712e-07`
- width drift `0`
- weighted negative hours `0`
- governance `PASS`

Because the internal score is not the final weak-bucket selection criterion,
all 9 eligible trials were then run through the no-OMPEX spot bucket backtest.
Outputs:

- `output/phase14/t047_spot_backtest_by_trial/`
- `output/phase14/t047_spot_backtest_by_trial/eligible_spot_backtest_summary.csv`

Best weak-bucket compromise:

- `t013_w05_l025_p075_n05_r00_d02`
- adjusted CSV SHA-256:
  `d7b93c7caf4c38ec51cd94d37f0f5308feef9df50bb1ca263705627ac8d7b1fb`
- overall profile MAE improvement: `0.29295542439021466`
- night MAE improvement: `0.11792184918005748`, positive folds `10/12`
- hourly-ramp MAE improvement: `0.034116846702457994`, positive folds `10/12`
- evening recovery MAE improvement: `0.32206130775585989`
- solar-tail MAE improvement: `0.29033105604920667`
- weekend MAE improvement: `0.1990459761545178`
- post-valuation MAE improvement: `0.22709564079301003`

T046 reference still dominates overall:

- overall profile MAE improvement: `0.4054835410318921`
- night MAE improvement: `0.031908941150684988`, positive folds `5/12`
- hourly-ramp MAE improvement: `0.035478178105887714`, positive folds `8/12`
- evening recovery MAE improvement: `0.45338812791781463`
- solar-tail MAE improvement: `0.43729530913049253`
- weekend MAE improvement: `0.28896113473708351`
- post-valuation MAE improvement: `0.3048038417338681`

Decision: T047 v3 is useful diagnostic evidence but is not frozen as a T046
replacement. It materially improves night evidence, but it does not beat T046
on overall, solar/evening/weekend, post-valuation or mean ramp MAE. Next model
step should refine night/ramp selection and component design, not promote T047.
OMPEX remains advisory-only post-selection with locked
`ompex_minus_1h_hourending` alignment.

The weak-bucket selection step is now reproducible rather than manual.

New script:

- `scripts/summarize_epex_shape_lab_spot_backtests.py`

New test:

- `tests/test_summarize_epex_shape_lab_spot_backtests_script.py`

The script reads an executed no-OMPEX sweep summary, per-trial spot-backtest
summaries, and optional incumbent evidence, then writes a fail-closed lab-only
selection summary. It rejects any OMPEX use or production-approved/promotion
gate evidence.

Real T047 v3 selection summary command:

```powershell
python scripts/summarize_epex_shape_lab_spot_backtests.py --sweep-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_summary.json --backtest-root output/phase14/t047_spot_backtest_by_trial --incumbent-backtest output/phase14/t046_spot_backtest_v2_buckets/spot_backtest_summary.json --output-dir output/phase14/t047_spot_backtest_selection_summary
```

Outputs:

- `output/phase14/t047_spot_backtest_selection_summary/spot_backtest_selection_summary.json`
- `output/phase14/t047_spot_backtest_selection_summary/spot_backtest_trial_ranking.csv`

Result:

- `trial_count_from_sweep=9`
- `trial_count_summarized=9`
- `strict_pass_count=9`
- `weak_bucket_candidate_count=1`
- best weak-bucket trial: `t013_w05_l025_p075_n05_r00_d02`
- verdict:
  `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- `replace_incumbent=false`

Validation including the new summarizer:

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `45 passed, 1 skipped`.

The eligible-trial backtest execution is now scripted as well.

New script:

- `scripts/run_epex_shape_lab_sweep_spot_backtests.py`

New test:

- `tests/test_run_epex_shape_lab_sweep_spot_backtests_script.py`

Real T047 resume command:

```powershell
python scripts/run_epex_shape_lab_sweep_spot_backtests.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/pre_registered_sweep_plan.json --sweep-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_summary.json --output-root output/phase14/t047_spot_backtest_by_trial --output-summary output/phase14/t047_spot_backtest_by_trial/run_summary_from_runner.json --incumbent-backtest output/phase14/t046_spot_backtest_v2_buckets/spot_backtest_summary.json --selection-output-dir output/phase14/t047_spot_backtest_selection_summary_from_runner
```

Result:

- `trial_count_backtested=9`
- `reused_existing_count=9`
- no OMPEX model/selection/backtest flags
- chained selection verdict remains
  `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- `replace_incumbent=false`

Validation including the runner:

```powershell
python -m pytest tests/test_run_epex_shape_lab_sweep_spot_backtests_script.py tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `48 passed, 1 skipped`.

## T046 Strict Product and Power BI Diagnostics

T046 was then run through strict product-normalization and Power BI diagnostics
using local rebuilt forwards and refreshed spot evidence.

Forwards note:

- `data/eex_forwards_history.parquet` was stale locally (`max_date=2026-06-17`)
  and had no CH snapshot on `2026-07-07`.
- A local diagnostic forwards parquet was rebuilt from the desk workbooks:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet`
- Source workbooks:
  - `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx`
  - `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Hist.xlsx`
- rebuilt CH coverage: `2020-05-04 -> 2026-07-07`
- rebuilt forwards sha256:
  `a6244638c2234781853284ce2ad58d55d01265568cca6c85d4461f21446e8d76`

Source hierarchy policy:

- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t046_asof20260707_fresh_epex_sweep_v2.json`
- policy sha256 from audit:
  `b79aec178312816e7d9554065a2e2acc0d0b419c43d3b85b4373639e22dc64df`
- input CSV sha256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- quote conflict identity hash:
  `a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`
- expected quote conflicts: `6`
- caveat: this policy accepts only redundant source hierarchy conflicts for
  the exact bound diagnostic artifact; it is not production approval for the
  adjusted curve.

Product normalization strict:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/candidate_epex_shape_lab_adjusted.csv --forwards output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet --required-forward-date 2026-07-07 --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t046_asof20260707_fresh_epex_sweep_v2.json --output-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_product_normalization_with_policy/gates.csv --summary-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_product_normalization_with_policy/summary.json
```

Result:

- `all_gates_pass=true`
- `critical_count=0`
- `unsupported_count=0`
- `quote_conflict_count=6`
- `accepted_quote_conflict_count=6`
- `blocking_quote_conflict_count=0`
- `delivered_curve_drift_count=0`
- `status_counts={"PASS": 90, "QUOTE_CONFLICT": 6}`

Power BI strict:

```powershell
python scripts/build_powerbi_exports.py --csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/candidate_epex_shape_lab_adjusted.csv --forwards output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet --spot output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet --output-dir output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_powerbi_strict
```

Result:

- `powerbi_quality_gate_status=PASS`
- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `weighted_negative_hours=0`
- `negative_gate_status=PASS`
- `min_weighted_eur_mwh=4.84`
- `min_price_eur_mwh=-3.83`
- `p10_negative_hours=118`
- `monthly_path_warning_flags=4`
- all critical flag counts: `0`

Decision log entry: `D-20260708-15`.

Remaining promotion blockers for t046:

- no production manifest for the adjusted curve
- no local export manifest for the adjusted curve
- no selected config artifact for the adjusted curve
- no capstone triad decision for the adjusted curve
- generated rebuilt forwards / spot / audit outputs are local evidence only
  and were not committed

## T046 Lab Promotion Readiness Decision

A dedicated checker was added because the monthly solver capstone proves the
baseline monthly triad, not the adjusted hourly lab CSV:

- `scripts/check_epex_lab_promotion_readiness.py`
- `tests/test_check_epex_lab_promotion_readiness_script.py`

Command run on t046:

```powershell
python scripts/check_epex_lab_promotion_readiness.py --lab-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/ab_lab_manifest.json --governance-audit output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/governance_audit/epex_shape_lab_governance_audit.json --independent-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/independent_ab_comparison/ab_comparison_summary.json --product-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_product_normalization_with_policy/summary.json --powerbi-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_powerbi_strict/summary_metrics.csv --ompex-advisory-delta output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/ompex_advisory_delta_selected_t046_20260708.json --output output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision.json
```

The checker intentionally returned non-zero because production approval is not
complete. It still wrote the local decision:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision.json`
- `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
- `approved=false`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- missing:
  - `adjusted_production_manifest`
  - `adjusted_export_manifest`
  - `adjusted_selected_config`
  - `adjusted_capstone`

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `86 passed, 1 skipped`.

Decision log entry: `D-20260708-16`.

## T046 Local Adjusted Evidence Bundle

A local non-production bundle was added so t046 has explicit adjusted
export/selected/local-capstone evidence without pretending to be production:

- `scripts/build_epex_lab_promotion_bundle.py`
- `tests/test_build_epex_lab_promotion_bundle_script.py`

Command run:

```powershell
python scripts/build_epex_lab_promotion_bundle.py --lab-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/ab_lab_manifest.json --baseline-monthly-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_asof20260707_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json --product-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_product_normalization_with_policy/summary.json --powerbi-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_powerbi_strict/summary_metrics.csv --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t046_asof20260707_fresh_epex_sweep_v2.json --independent-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/independent_ab_comparison/ab_comparison_summary.json --governance-audit output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_w05_l025_p075_d03/governance_audit/epex_shape_lab_governance_audit.json --ompex-advisory-delta output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/ompex_advisory_delta_selected_t046_20260708.json --output-dir output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_local_promotion_bundle
```

Generated local bundle:

- `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_local_promotion_bundle/adjusted_export_manifest.json`
- `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_local_promotion_bundle/adjusted_selected_artifact.json`
- `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_local_promotion_bundle/adjusted_local_capstone_no_go.json`

The readiness checker was rerun with these local artifacts:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision_with_local_bundle.json`
- `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
- `approved=false`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- missing evidence now only:
  - `adjusted_production_manifest`

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_promotion_bundle_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `87 passed, 1 skipped`.

Decision log entry: `D-20260708-17`.

## T046 Enriched No-OMPEX Shape Diagnostics

After read-only expert audit, the independent A/B comparison was enriched to
make the T046 shape deformation reviewable before any production wiring:

- `scripts/compare_epex_shape_lab_ab.py`
- `tests/test_compare_epex_shape_lab_ab_script.py`

New outputs from the comparison:

- `load_type_delta_summary.csv`
- `month_hour_delta_summary.csv`
- `peak_offpeak_monthly_summary.csv`
- `boundary_delta_jumps.csv`
- `delta_heatmap_month_hour_<year>.png`
- `peak_offpeak_spread_delta_by_month.png`
- `boundary_delta_jumps.png`

Command run on the 20260708 baseline vs T046:

```powershell
python scripts\compare_epex_shape_lab_ab.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\t046_w05_l025_p075_d03\candidate_epex_shape_lab_adjusted.csv --output-dir output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\t046_enriched_ab_diagnostics
```

Generated local evidence:

- `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_enriched_ab_diagnostics/`
- `benchmark_policy=independent_no_ompex`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`
- `n_hours=57025`
- `max_abs_delta_eur_mwh=3.0`
- `max_abs_monthly_mean_delta_eur_mwh=8.602150532151586e-08`
- `max_abs_width_delta_eur_mwh=0.0`
- `quantile_order_adjusted_ok=true`
- `weighted_negative_hours_adjusted=0`
- `ramp_abs_p99_baseline_eur_mwh=24.818882960000007`
- `ramp_abs_p99_adjusted_eur_mwh=25.80652717999996`
- largest mean month-hour deltas concentrate in April around h13-h14 negative
  and h19 positive
- largest month-boundary delta jump is about `0.279 EUR/MWh`

This is diagnostic evidence only. It does not change the production status:
baseline 20260708 remains the only promotion-ready candidate and T046 remains
NO-GO production until a real adjusted production/export/selected/capstone
chain exists and passes.

Validation:

```powershell
python -m pytest tests/test_compare_epex_shape_lab_ab_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

```powershell
python -m pytest tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_run_epex_shape_lab_ab_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `34 passed, 1 skipped`.

Decision log entry: `D-20260708-22`.

## Fan-to-Hourly Parity Diagnostic

After expert audit, the fan-parquet staging blocker was made explicit with a
read-only diagnostic:

- `scripts/diagnose_fan_to_hourly_parity.py`
- `tests/test_diagnose_fan_to_hourly_parity_script.py`

The diagnostic converts the fan parquet with the same lightweight helper used
by staging (`to_hourly_csv_frame`), aligns it to the audited hourly CSV, writes
column/month/load-type/boundary deltas, and can run product-normalization
audits on both artifacts. It is deliberately not a promotion gate and writes
`promotion_gate=false`.

Command run:

```powershell
python scripts\diagnose_fan_to_hourly_parity.py --fan-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\fan_asof20260707_lshape100_yoy150_amp150_2032.parquet --reference-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --local-start-date 2026-07-01 --local-end-date 2032-12-31 --forwards output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\diagnostic_forwards_history_rebuilt_20260708.parquet --required-forward-date 2026-07-07 --output-dir output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\fan_to_hourly_parity_diagnostic
```

Generated local evidence:

- `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_to_hourly_parity_diagnostic/`
- `fan_rows_15min=228192`
- `fan_hourly_rows=57025`
- `reference_rows=57025`
- `aligned_rows=57025`
- `missing_price_columns=[]`
- `max_abs_weighted_delta_eur_mwh=24.290847000000014`
- `mean_abs_weighted_delta_eur_mwh=2.797042955633494`
- `max_abs_monthly_weighted_delta_eur_mwh=0.5222635513888889`
- load-type deltas:
  - `PEAK` mean delta about `+1.2408 EUR/MWh`
  - `OFFPEAK` mean delta about `-0.6773 EUR/MWh`

Product audit within the diagnostic confirms why fan-derived hourly cannot be
promotion-facing yet:

- fan-derived hourly:
  - `all_gates_pass=false`
  - `critical_count=56`
  - `delivered_curve_drift_count=38`
  - max supported hard-gate residual about `21.915846 EUR/MWh`
- audited reference CSV under the same diagnostic forwards:
  - `critical_count=0`
  - `delivered_curve_drift_count=0`

The baseline source hierarchy policy is hash-bound to its exact production
forwards (`forwards_sha256=159680...`), while the T046 diagnostic forwards are
`a624...`; therefore this diagnostic must not be used as a replacement for the
baseline capstone. It only explains the fan-to-hourly blocker.

Validation:

```powershell
python -m pytest tests/test_diagnose_fan_to_hourly_parity_script.py -q -p no:cacheprovider
```

Result: `1 passed`.

```powershell
python -m pytest tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `62 passed, 1 skipped`.

Decision log entry: `D-20260708-23`.

## Fan Staging Production-Contract Guard

The staging runner now enforces the fan-to-hourly finding directly:

- `scripts/stage_epex_lab_adjusted_lt_candidate.py`
- `tests/test_stage_epex_lab_adjusted_lt_candidate_script.py`

Behavior:

- `source_kind=fan_parquet`:
  - `source_promotion_eligible=false`
  - `production_contract_blockers` includes
    `source_kind_fan_parquet_requires_audited_hourly_export`
  - no `adjusted_production_manifest_no_go.json` is written, even when strict
    evidence inputs are supplied
- `source_kind=candidate_csv`:
  - `source_promotion_eligible=true`
  - can still package an adjusted production contract NO-GO when all strict
    evidence inputs are supplied

This is a guard against evidence confusion only. It does not promote T046.
Promotion-facing EPEX adjusted work must begin from an audited hourly export or
from a production path that emits and gates the same artifact.

Validation:

```powershell
python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py -q -p no:cacheprovider
```

Result: `5 passed`.

```powershell
python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `70 passed, 1 skipped`.

Decision log entry: `D-20260708-24`.

## Adjusted Production Manifest Source Provenance Guard

The adjusted production-manifest contract now also enforces source provenance:

- `scripts/build_epex_lab_adjusted_production_manifest.py`
- `scripts/check_epex_lab_promotion_readiness.py`
- `tests/test_build_epex_lab_adjusted_production_manifest_script.py`
- `tests/test_check_epex_lab_promotion_readiness_script.py`

Behavior:

- CLI-built adjusted production manifests remain NO-GO by default.
- Any approved adjusted production manifest now requires:
  - `production_run_id`;
  - `production_entrypoint`;
  - `git_commit`;
  - `source_provenance_manifest`.
- Source provenance must prove:
  - schema `epex_lab_adjusted_lt_candidate_stage.v1`;
  - `source_kind=candidate_csv`;
  - `source_promotion_eligible=true`;
  - empty `production_contract_blockers`;
  - adjusted CSV path or SHA bound to the lab adjusted CSV.
- Readiness now requires adjusted production manifest
  `contract_pass=true` and `source_provenance_pass=true` in addition to
  production approval flags and adjusted CSV binding.

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result: `9 passed`.

```powershell
python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `63 passed, 1 skipped`.

Decision log entry: `D-20260708-25`.

## Candidate-CSV Staging With Source Provenance

The candidate-CSV staging path now writes a stable source provenance artifact
and passes it into the adjusted production contract builder:

- `source_provenance_manifest.json`
- `adjusted_production_manifest_no_go.json`

Real command run on the audited 20260708 hourly baseline wrote:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_with_provenance/`

Key outputs:

- adjusted CSV SHA-256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- source provenance SHA-256:
  `8d3cacb36637ea6e57446d840458d85d6219da72a94c200f1ac8c559a3d2a6b9`
- adjusted production contract NO-GO SHA-256:
  `5600b737482e0db537059f36fe997f3bbe9e15c8435874c98b1ca8b59e0e2f09`
- contract fields:
  - `contract_pass=true`
  - `source_kind=candidate_csv`
  - `source_promotion_eligible=true`
  - `source_provenance_pass=true`
  - `production_approved=false`
  - `production_promotion_approved=false`

Readiness was rerun with this provenance-aware contract and local bundle
artifacts:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_with_provenance/readiness_no_go.json`
- `approved=false`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `missing_production_evidence=[]`
- provenance checks pass
- remaining failures are expected NO-GO production approvals:
  - adjusted capstone approved;
  - adjusted production manifest approved;
  - adjusted export manifest production-ready;
  - adjusted selected artifact production-ready.

Validation:

```powershell
python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py -q -p no:cacheprovider
```

Result: `5 passed`.

```powershell
python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `72 passed, 1 skipped`.

Decision log entry: `D-20260708-26`.

## T046 Multi-Date Stability Summary

`scripts/summarize_epex_shape_lab_stability.py` now summarizes frozen
independent no-OMPEX A/B comparisons across valuation dates. It reads each
case's A/B summary plus governance audit and gates on no OMPEX model/selection
usage, governance `PASS`, finite and ordered quantiles, zero weighted negative
hours, min adjusted price floor, monthly mean drift, fan width drift, and ramp
p99 increase.

Real stability output:

`output/phase14/t046_stability_summary_v1/`

Cases:

- `asof20260706`
- `asof20260707`

Result:

- `status=PASS`
- `case_count=2`
- `passed_case_count=2`
- `failed_case_count=0`
- `benchmark_policy=multi_date_independent_no_ompex`
- `promotion_gate=false`

Case values:

- `asof20260706`: monthly drift `8.602150568442747e-08`, width drift `0`,
  weighted negative hours `0`, min adjusted price `-3.887768`, ramp p99
  increase `0.9378621600000336`, governance `PASS`.
- `asof20260707`: monthly drift `8.602150532151586e-08`, width drift `0`,
  weighted negative hours `0`, min adjusted price `-3.825623`, ramp p99
  increase `0.9876442199999538`, governance `PASS`.

Validation:

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `38 passed, 1 skipped`.

Decision log entry: `D-20260708-27`.

## Post-Audit Promotion Provenance Hardening

Read-only expert audit found one future-path P0: the readiness checker could
previously trust self-attested fields inside an adjusted production manifest.
The current checked T046 artifacts were still NO-GO, but a fabricated future
manifest could have claimed `source_provenance_pass=true`.

Corrections:

- `scripts/check_epex_lab_promotion_readiness.py` now reloads the
  `source_provenance_manifest`, validates its SHA-256 against the production
  manifest, and verifies `source_kind=candidate_csv`,
  `source_promotion_eligible=true`, empty contract blockers, adjusted CSV
  binding, source CSV SHA binding, staged candidate SHA binding, lab manifest
  SHA binding, and source export manifest SHA/binding.
- Readiness now rejects minimal handwritten export, selected, and capstone
  artifacts by requiring known schemas and production-chain fields.
- `scripts/build_epex_lab_adjusted_production_manifest.py` now includes source
  provenance presence in `contract_pass`; `contract_pass=true` can no longer
  mean diagnostics passed while provenance is absent.
- `scripts/stage_epex_lab_adjusted_lt_candidate.py` now requires a
  source-export manifest bound to the candidate CSV before setting
  `source_promotion_eligible=true` or writing an adjusted production-contract
  package. A candidate CSV without this manifest remains stageable but gets
  blocker `candidate_csv_requires_source_export_manifest`.

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py -q -p no:cacheprovider
```

Result: `17 passed`.

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `41 passed, 1 skipped`.

Decision log entry: `D-20260708-28`.

## Post-Audit Diagnostic Hardening

Expert audit also identified two non-blocking diagnostic gaps. These are now
closed as read-only lab evidence:

- `scripts/compare_epex_shape_lab_ab.py` recomputes implied structural width as
  `structural_p90 - structural_p10`, reports implied-width delta, and reports
  reported-minus-implied width for baseline and adjusted. This catches stale
  `structural_width_eur_mwh` columns.
- `scripts/diagnose_fan_to_hourly_parity.py` reports missing timestamps in the
  fan-derived hourly series and reference CSV, fan/reference coverage ratios,
  and `coverage_status=PASS` or `PARTIAL_OVERLAP`.

Validation:

```powershell
python -m pytest tests/test_compare_epex_shape_lab_ab_script.py tests/test_diagnose_fan_to_hourly_parity_script.py -q -p no:cacheprovider
```

Result: `5 passed`.

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `43 passed, 1 skipped`.

Decision log entry: `D-20260708-29`.

## T046 Stability V2 Local-Shape Gates

`scripts/compare_epex_shape_lab_ab.py` now promotes the local-shape diagnostics
into `ab_comparison_summary.json`, and
`scripts/summarize_epex_shape_lab_stability.py` requires them for PASS.

New gated fields:

- implied structural width drift: `structural_p90 - structural_p10`;
- reported-minus-implied width drift;
- max PEAK/OFFPEAK spread delta;
- max month-hour mean delta;
- max month-boundary delta jump;
- p10 negative hours;
- p10 negative cluster length.

Real v2 evidence:

- `asof20260706` A/B v2:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/epex_stage_t046_stability_probe/independent_ab_comparison_v2/`
- `asof20260707` A/B v2:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_with_provenance/independent_ab_comparison_v2/`
- stability v2:
  `output/phase14/t046_stability_summary_v2/`

Result:

- `status=PASS`;
- `case_count=2`;
- `passed_case_count=2`;
- `promotion_gate=false`;
- max month-hour mean delta about `2.2225` EUR/MWh;
- max boundary delta jump about `0.2786` EUR/MWh;
- max PEAK/OFFPEAK spread delta about `2.5e-07` EUR/MWh;
- implied width drift about `2.84e-14`;
- p10 negative hours `125` and `118`;
- p10 negative cluster max `6`;
- weighted negative hours `0`.

Validation:

```powershell
python -m pytest tests/test_compare_epex_shape_lab_ab_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_summarize_epex_shape_lab_stability_script.py -q -p no:cacheprovider
```

Result: `9 passed`.

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `45 passed, 1 skipped`.

Decision log entry: `D-20260708-30`.

## T046 Delta-Field Stability

`scripts/summarize_epex_shape_lab_delta_stability.py` compares the actual
hourly delta field across dates, not just aggregate gates. It reads each case
as `LABEL|ALIGNED_AB_CSV|AB_SUMMARY_JSON|LAB_MANIFEST_JSON`, validates
no-OMPEX/lab-only status, checks T046 config hash consistency, discloses
missing timestamps, and compares timestamp-level, month-hour, and
month-boundary delta differences.

Real output:

`output/phase14/t046_delta_stability_summary_v1/`

Cases:

- reference `asof20260706`;
- comparison `asof20260707`.

Result:

- `status=PASS`;
- `comparison_count=1`;
- `passed_comparison_count=1`;
- `promotion_gate=false`;
- `config_consistent=true`;
- config hash:
  `e9c1f0831cb896f03987eeefcbb92dfbf900a53eaad4c4d45ab58e761e163b51`;
- timestamp delta correlation `0.9999797676440942`;
- timestamp delta MAE `0.0011066597457297395` EUR/MWh;
- timestamp delta RMSE `0.005290483624245774` EUR/MWh;
- timestamp delta max abs `0.05641400000001795` EUR/MWh;
- month-hour delta correlation `0.9999818326388252`;
- month-hour delta MAE `0.0009053573348698669` EUR/MWh;
- month-hour delta max abs `0.04788046543778779` EUR/MWh;
- boundary jump diff MAE `0.0013372207792282818` EUR/MWh;
- boundary jump diff max abs `0.015835000000009813` EUR/MWh;
- missing timestamps `0`.

Validation:

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_delta_stability_script.py -q -p no:cacheprovider
```

Result: `3 passed`.

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `48 passed, 1 skipped`.

Decision log entry: `D-20260708-31`.

## Source-Export Provenance For T046 Staging

`scripts/build_epex_lab_source_export_manifest.py` now builds a hash-bound
source-export manifest for candidate-CSV EPEX lab staging. This closes the D28
source-manifest requirement without approving the adjusted T046 curve.

Real source-export manifest:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_source_export_manifest/source_export_manifest.json`

Key values:

- source CSV SHA-256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`;
- source-export manifest SHA-256:
  `d662548e2e7605ba2b59e024afd3040f2724fe84c5f3c7d3491fbaa0e1909f1d`;
- baseline monthly manifest SHA-256:
  `cb52a502e8e95af2e5f3fabc3b2b34ca8f365999214cfd7c53718ed7f5ef456a`;
- selected config SHA-256:
  `5ca8b3dc3c1dfadf6b2153bc22f6d69cfb2ad767ae8dbada9615f753760e1f34`;
- baseline capstone SHA-256:
  `091105ba9bc313b36364a75a9dd88ab9e3eaa9e740151c44c8a40c42cce1048c`;
- `production_approved=false`;
- `production_promotion_approved=false`;
- `promotion_scope=SOURCE_CSV_PROVENANCE_ONLY`.

Rerun staging:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/`

Key values:

- adjusted CSV SHA-256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`;
- source provenance SHA-256:
  `eefe822b24a876a176b78afd9ccc21552d4c5248d8833a7c8ee1bbd368789d1f`;
- adjusted production manifest NO-GO SHA-256:
  `7824522ca68f64da20bd7871cba0beed246f27bfb888beef2d1cc65ffdbd17a9`;
- `source_kind=candidate_csv`;
- `source_promotion_eligible=true`;
- `source_export_manifest_bound=true`;
- `production_contract_blockers=[]`;
- `adjusted_production_contract_pass=true`;
- `production_approved=false`;
- `production_promotion_approved=false`.

Readiness:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/readiness_no_go.json`

Result:

- `approved=false`;
- `strict_diagnostics_pass=true`;
- `production_chain_pass=false`;
- `missing_production_evidence=[]`;
- adjusted production contract PASS;
- source provenance reload/SHA/binding checks PASS;
- expected FAIL checks remain adjusted capstone approval, adjusted production
  manifest approval, adjusted export manifest production-ready, and adjusted
  selected artifact production-ready.

Operational note: the first A/B rerun under the very long staging directory
failed while writing PNGs because of Windows path length. The successful A/B
evidence is under the shorter directory:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/t046_srcprov_ab_v2/`.

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_source_export_manifest_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result: `19 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `50 passed, 1 skipped`.

Decision log entry: `D-20260708-32`.

## D33 No-OMPEX Spot Backtest For T046

Added a realized EPEX spot diagnostic for T046 that remains explicitly
lab-only and no-OMPEX.

Files:

- `scripts/backtest_epex_shape_lab_against_spot.py`
- `tests/test_backtest_epex_shape_lab_against_spot_script.py`

Real output:

`output/phase14/t046_spot_backtest_v1/spot_backtest_summary.json`

Command:

```powershell
python scripts/backtest_epex_shape_lab_against_spot.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_stage_t046_from_hourly_baseline_source_export_provenance\epex_lab\candidate_epex_shape_lab_adjusted.csv --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t046_spot_backtest_v1 --valuation-timestamp 2026-07-07T00:00:00Z --lookback-years 2 --eval-days 30 --embargo-days 1 --max-auto-folds 12 --min-eval-hours 24
```

Result:

- `status=DIAGNOSTIC_PASS`
- `strict_lab_gate_pass=true`
- `promotion_gate=false`
- `production_approved=false`
- `independent_production_evidence=false`
- `benchmark_policy=rolling_origin_epex_spot_no_ompex_lab_only`
- OMPEX flags all false

Hashes:

- baseline CSV:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- adjusted CSV:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- spot parquet:
  `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`

Metrics:

- rolling folds `12/12` eligible;
- all rolling folds pass no-temporal-leak checks;
- positive MAE improvement folds `12/12`;
- mean baseline profile MAE `14.153227063777985` EUR/MWh;
- mean adjusted profile MAE `13.747743522746092` EUR/MWh;
- mean improvement `0.40548354103189205` EUR/MWh;
- mean baseline correlation `0.8771034667706381`;
- mean adjusted correlation `0.881938557794624`;
- post-valuation overlap only `24` hours;
- post-valuation residual MAE improvement `0.3048038417338681` EUR/MWh.

Important limitation: all 12 historical rolling folds are recorded as not
independent of the current candidate fit because T046 was selected after those
historical spot rows were known. D33 is therefore useful shape evidence and an
anti-overfit diagnostic, but it is not promotion evidence.

Validation:

```powershell
python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

```powershell
python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `52 passed, 1 skipped`.

Decision log entry: `D-20260708-33`.

## D34 Economic Bucket And Ramp Spot Diagnostics

Extended the D33 no-OMPEX spot backtest with economic buckets and hourly ramp
metrics.

Changed files:

- `scripts/backtest_epex_shape_lab_against_spot.py`
- `tests/test_backtest_epex_shape_lab_against_spot_script.py`

New output:

- `rolling_spot_bucket_metrics.csv`
- `rolling_bucket_metrics` in `spot_backtest_summary.json`

Real output:

`output/phase14/t046_spot_backtest_v2_buckets/spot_backtest_summary.json`

Command:

```powershell
python scripts/backtest_epex_shape_lab_against_spot.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_stage_t046_from_hourly_baseline_source_export_provenance\epex_lab\candidate_epex_shape_lab_adjusted.csv --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t046_spot_backtest_v2_buckets --valuation-timestamp 2026-07-07T00:00:00Z --lookback-years 2 --eval-days 30 --embargo-days 1 --max-auto-folds 12 --min-eval-hours 24
```

Result:

- `status=DIAGNOSTIC_PASS`
- `strict_lab_gate_pass=true`
- `promotion_gate=false`
- `production_approved=false`
- `independent_production_evidence=false`
- OMPEX flags all false

Selected bucket metrics, mean MAE improvement in EUR/MWh:

- residual level all: `0.24513954474101998`, positive folds `11/12`
- weekend: `0.2889611347370835`, positive folds `12/12`
- weekday: `0.22708125671275944`, positive folds `10/12`
- PEAK-like weekday 08-19: `0.32096908439747596`, positive folds `10/12`
- OFFPEAK-like: `0.20198153831529964`, positive folds `12/12`
- solar tail Mar-Oct 10-16: `0.4372953091304925`, positive folds `8/12`
- midday 11-15: `0.35776460522648684`, positive folds `9/12`
- evening ramp 17-21: `0.45338812791781463`, positive folds `12/12`
- night 00-05: `0.03190894115068499`, positive folds `5/12`
- hourly ramp all: `0.035478178105887714`, positive folds `8/12`

Interpretation:

- T046 helps most on evening recovery, solar/midday, PEAK-like hours, and
  weekend buckets.
- Night and hourly ramp gains are weak; do not overclaim ramp quality from
  this experiment.
- D34 remains shape research evidence only and does not alter T046 NO-GO
  production status.

Validation:

```powershell
python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `52 passed, 1 skipped`.

Decision log entry: `D-20260708-34`.

## D35 Future Approval Path Audit

Added a compact read-only audit that summarizes the exact production blockers
for T046 from readiness evidence and optional spot-backtest policy evidence.

Files:

- `scripts/audit_epex_lab_future_approval_path.py`
- `tests/test_audit_epex_lab_future_approval_path_script.py`

Real output:

`output/phase14/t046_future_approval_path_audit_v1/future_approval_path_audit.json`

Command:

```powershell
python scripts/audit_epex_lab_future_approval_path.py --readiness-json output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_stage_t046_from_hourly_baseline_source_export_provenance\readiness_no_go.json --spot-backtest-summary output\phase14\t046_spot_backtest_v2_buckets\spot_backtest_summary.json --output output\phase14\t046_future_approval_path_audit_v1\future_approval_path_audit.json
```

Result:

- `status=NO_GO_PRODUCTION_CHAIN_INCOMPLETE`
- `approved=false`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `spot_backtest_policy.pass=true`
- `missing_production_evidence=[]`

Remaining blockers:

- `adjusted_capstone_approved`
- `adjusted_export_manifest_production_ready`
- `adjusted_production_manifest_approved`
- `adjusted_selected_artifact_production_ready`

Interpretation:

- The needed files are present in staging/local form, but their production
  approval booleans are false.
- The next action is to replace local diagnostic approval flags with real
  production-approved adjusted artifacts, not to build another local bundle.
- D35 does not change the NO-GO production verdict.

Validation:

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result: `3 passed`.

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `55 passed, 1 skipped`.

Decision log entry: `D-20260708-35`.

## D36 Adjusted Production Approval Identity Hardening

Hardened the API-only path that can set adjusted EPEX lab production approval
flags.

Changed files:

- `scripts/build_epex_lab_adjusted_production_manifest.py`
- `tests/test_build_epex_lab_adjusted_production_manifest_script.py`

Behavior:

- CLI-built adjusted production manifests remain NO-GO by default.
- If Python API callers request either `production_approved=True` or
  `production_promotion_approved=True`, the builder now requires:
  - non-empty `production_run_id`;
  - non-empty `production_entrypoint`;
  - `git_commit` matching `[0-9a-f]{40}`;
  - existing `source_provenance_manifest`.
- Invalid identity raises `ValueError` before writing an approved manifest.

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py -q -p no:cacheprovider
```

Result: `6 passed`.

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `56 passed, 1 skipped`.

Decision log entry: `D-20260708-36`.

## D37 Production Chain Binding In Readiness

Hardened T046 readiness so a future `PROMOTION_READY` cannot be assembled from
approved-looking but unbound adjusted artifacts.

Changed files:

- `scripts/check_epex_lab_promotion_readiness.py`
- `scripts/audit_epex_lab_future_approval_path.py`
- `tests/test_check_epex_lab_promotion_readiness_script.py`

New readiness checks:

- `adjusted_production_manifest_run_identity_valid`
- `adjusted_export_manifest_production_chain_bound`
- `adjusted_selected_artifact_production_chain_bound`
- `adjusted_capstone_production_chain_bound`

Required future contract:

- adjusted production manifest has valid run identity;
- adjusted export manifest binds to that production manifest path or SHA and
  matches run identity;
- adjusted selected artifact binds to that production manifest path or SHA and
  matches run identity;
- adjusted capstone binds to the adjusted production manifest, adjusted export
  manifest, adjusted selected artifact, and run identity.

Real readiness v2:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/readiness_no_go_v2_chain_bound.json`

Result:

- `approved=false`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`

New explicit real blockers:

- `adjusted_production_manifest_run_identity_valid=FAIL`
- `adjusted_export_manifest_production_chain_bound=FAIL`
- `adjusted_selected_artifact_production_chain_bound=FAIL`
- `adjusted_capstone_production_chain_bound=FAIL`

Future approval path audit v2:

`output/phase14/t046_future_approval_path_audit_v2_chain_bound/future_approval_path_audit.json`

Result:

- `status=NO_GO_PRODUCTION_CHAIN_INCOMPLETE`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `spot_backtest_policy.pass=true`
- failed production checks `8`

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result: `7 passed`.

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `57 passed, 1 skipped`.

Decision log entry: `D-20260708-37`.

## D38 Strict Adjusted Production Chain Builder

Added a guarded builder for the three remaining adjusted production-chain
artifacts after an adjusted production manifest is already approved.

Files:

- `scripts/build_epex_lab_adjusted_production_chain.py`
- `tests/test_build_epex_lab_adjusted_production_chain_script.py`

Behavior:

- Refuses NO-GO or local diagnostic adjusted production manifests.
- Requires the input adjusted production manifest to be:
  - `production_approved=true`;
  - `production_promotion_approved=true`;
  - `contract_pass=true`;
  - `source_provenance_pass=true`;
  - valid `production_run_id`;
  - valid `production_entrypoint`;
  - `git_commit` matching `[0-9a-f]{40}`;
  - no-OMPEX;
  - adjusted CSV path/SHA bound.
- Writes:
  - `adjusted_export_manifest.json`;
  - `adjusted_selected_artifact.json`;
  - `adjusted_production_capstone.json`.
- Output artifacts bind to the adjusted production manifest path/SHA and run
  identity. The capstone also binds to the generated export and selected
  artifacts.

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_chain_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `59 passed, 1 skipped`.

Decision log entry: `D-20260708-38`.

Governance:

- Decision log entry: `D-20260708-05`.
- This lab does not change the accepted 2026-07-08 promotion candidate.
- Before any production/export wiring, run a pre-registered A/B with identical
  forwards/snapshot inputs, strict Power BI/product/source-hierarchy/capstone
  gates, and OMPEX retained only as advisory benchmark evidence.
- The lab manifest marks artifacts `activation_status=lab_only`,
  `production_approved=false`, `ompex_used_in_selection=false`, and
  `source_hashes_required_for_promotion=true`.
- Research callers can disable the monthly-constraint guard, but such outputs
  are not valid promotion evidence under `monthly_level_authority="solver"`.

## Worktree / Commit Hygiene

## Continuation - T048 / T049 EPEX Lab

After the scripted T047 runner, a T048 night/core-recovery sweep was run as
local no-OMPEX lab evidence. A first long output path failed on Windows path
length while writing comparison artifacts; the executed evidence is the short
path only and should be cited as authoritative:

- executed plan:
  `output/phase14/t048_ncr/pre_registered_sweep_plan.json`
- sweep summary:
  `output/phase14/t048_ncr/sweep_execution_summary.json`
- spot backtest run:
  `output/phase14/t048_ncr_spot_backtests/run_summary.json`
- selection summary:
  `output/phase14/t048_ncr_selection_summary/spot_backtest_selection_summary.json`

T048 result:

- planned trials: `32`
- executed trials: `32`
- eligible trials: `27`
- spot-backtested eligible trials: `27`
- strict pass count: `27`
- weak-bucket candidate count: `16`
- best weak-bucket trial:
  `t004_w05_l025_p075_n05_r00_d275`
- best overall trial:
  `t020_w075_l025_p075_n05_r00_d275`
- strongest compromise:
  `t024_w075_l025_p01_n05_r00_d275`
- official verdict:
  `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- `replace_incumbent=false`
- `production_approved=false`
- `promotion_gate=false`
- OMPEX flags remain false.

Read-only MIT/Roaster audits concluded:

- NO-GO to replace T046 with T048.
- NO-GO to promote T046/T047/T048 to production because they remain lab-only
  adjusted artifacts without a chain-bound adjusted production manifest,
  adjusted export manifest, adjusted selected artifact, and adjusted capstone.
- Baseline 2026-07-08 promotion evidence remains coherent, but current local
  `data/eex_forwards_history.parquet` hash no longer matches the source hash
  recorded in the 2026-07-08 manifests. Restore or regenerate source-bound
  evidence before claiming reproducible local promotion.
- OMPEX remains advisory-only and must not be used for model input, selection,
  or gates.

Next planned sweep is T049 core-balance. It should be pre-registered and run
under short paths:

- output root:
  `output/phase14/t049_core_balance`
- spot backtest root:
  `output/phase14/t049_core_balance_spot_backtests`
- selection root:
  `output/phase14/t049_core_balance_selection_summary`

T049 design:

- weekend intensity `[0.65, 0.75]`
- low-tail intensity `[0.25]`
- peak-subshape intensity `[0.75, 0.875, 1.0]`
- night intensity `[0.4, 0.5, 0.6]`
- ramp intensity `[0.0, 0.125]`
- max absolute delta `[2.5, 2.75]`
- `max_ramp_p99_increase_eur_mwh=0.90`
- `ramp_penalty_weight=2.0`

T049 replacement bar: beat T046 on night and ramp without material regression
on overall, evening, solar-tail, weekend, post-valuation, monthly/fan drift,
negative-price stress, or BASE/PEAK/OFFPEAK normalization. If a candidate
passes this no-OMPEX bar, OMPEX can be run only afterward as advisory evidence.

Decision log entry: `D-20260708-43`.

T049 was then executed:

- plan:
  `output/phase14/t049_core_balance/pre_registered_sweep_plan.json`
- sweep summary:
  `output/phase14/t049_core_balance/sweep_execution_summary.json`
- spot backtest run:
  `output/phase14/t049_core_balance_spot_backtests/run_summary.json`
- selection summary:
  `output/phase14/t049_core_balance_selection_summary/spot_backtest_selection_summary.json`

T049 result:

- `trial_count_executed=72`
- `eligible_count=52`
- `trial_count_backtested=52`
- `strict_pass_count=52`
- `weak_bucket_candidate_count=52`
- `replace_incumbent=false`

The automatic best weak-bucket trial improved night/ramp/overall/solar/weekend
but still degraded evening and post-valuation versus T046. A better frontier
candidate was found:

- trial id: `t070_w075_l025_p01_n06_r00_d275`
- adjusted CSV sha256:
  `f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`
- parameters: weekend `0.75`, low-tail `0.25`, peak-subshape `1.0`, night
  `0.6`, ramp `0.0`, cap `2.75`
- ramp p99 increase: `0.8886024799999568`, below the T049 threshold `0.90`
- strict lab checks pass; no OMPEX flags; production flags remain false.

T049 `t070` versus T046 spot-backtest improvements:

- overall `0.42709956252228376` vs `0.40548354103189205`
- night `0.15957030400928707` vs `0.03190894115068499`
- ramp `0.05043407090595627` vs `0.035478178105887714`
- evening `0.45756877823583286` vs `0.45338812791781463`
- solar-tail `0.4312351115165488` vs `0.4372953091304925`
- weekend `0.2990265337481961` vs `0.2889611347370835`
- post-valuation `0.3053769058019675` vs `0.3048038417338681`

Conclusion: no full bucket dominator exists in T049, but `t070` is the current
no-OMPEX frontier. It misses only solar-tail by about `0.00606` EUR/MWh.

T050 micro-balance was run around `t070`:

- plan:
  `output/phase14/t050_t070_micro_balance/pre_registered_sweep_plan.json`
- sweep summary:
  `output/phase14/t050_t070_micro_balance/sweep_execution_summary.json`
- spot backtest run:
  `output/phase14/t050_t070_micro_balance_spot_backtests/run_summary.json`
- selection summary:
  `output/phase14/t050_t070_micro_balance_selection_summary/spot_backtest_selection_summary.json`

T050 result:

- `trial_count_executed=12`
- `eligible_count=4`
- `trial_count_backtested=4`
- `strict_pass_count=4`
- best trial `t007_w075_l025_p01_n06_r00_d275`
- adjusted CSV sha256 matches T049 `t070`
- replacement verdict remains
  `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- only degraded metric versus T046 is solar-tail.

Decision log entry: `D-20260708-44`.

Recommended next action: stop broad parameter sweeps for now. Treat `t070` as
the lab frontier and run stricter delivered-curve diagnostics against it:
delivered-product normalization, Power BI strict export into an isolated
`output/phase14/...` directory, source-hierarchy policy binding, and optional
OMPEX advisory post-check only after the no-OMPEX diagnostic package is frozen.

Generated or refreshed local evidence, not default commit targets:

- `data/eex_forwards_history.parquet`
- `pfc_shaping/output/*2026-07-08*`
- `pfc_shaping/model/artifacts/*`
- `output/phase14/20260708_*`
- local benchmark outputs under `ompex_benchmark_read_only/`

The export script modified a tracked Phase 13 generated report; it was restored
from HEAD to avoid cross-phase pollution.

Commit candidates for this follow-up:

- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260708-DAILY-GENERATION-ASOF20260707.md`
- `scripts/compare_epex_shape_lab_ab.py`
- `scripts/audit_epex_lab_future_approval_path.py`
- `scripts/build_epex_lab_adjusted_production_chain.py`
- `scripts/backtest_epex_shape_lab_against_spot.py`
- `scripts/build_epex_lab_source_export_manifest.py`
- `scripts/diagnose_fan_to_hourly_parity.py`
- `scripts/summarize_epex_shape_lab_stability.py`
- `scripts/summarize_epex_shape_lab_delta_stability.py`
- `scripts/stage_epex_lab_adjusted_lt_candidate.py`
- `scripts/build_epex_lab_adjusted_production_manifest.py`
- `scripts/check_epex_lab_promotion_readiness.py`
- `tests/test_compare_epex_shape_lab_ab_script.py`
- `tests/test_audit_epex_lab_future_approval_path_script.py`
- `tests/test_build_epex_lab_adjusted_production_chain_script.py`
- `tests/test_backtest_epex_shape_lab_against_spot_script.py`
- `tests/test_build_epex_lab_source_export_manifest_script.py`
- `tests/test_diagnose_fan_to_hourly_parity_script.py`
- `tests/test_summarize_epex_shape_lab_stability_script.py`
- `tests/test_summarize_epex_shape_lab_delta_stability_script.py`
- `tests/test_stage_epex_lab_adjusted_lt_candidate_script.py`
- `tests/test_build_epex_lab_adjusted_production_manifest_script.py`
- `tests/test_check_epex_lab_promotion_readiness_script.py`
