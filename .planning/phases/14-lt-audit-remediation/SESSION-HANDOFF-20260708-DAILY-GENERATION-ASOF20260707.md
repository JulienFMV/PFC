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

No `2026-07-08` OMPEX file was observed during this run.

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
    production approval `NO`; do not ship it as standalone external promotion
    evidence.
  - generated `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`
    has empty `source_hashes`, while the export manifest binds the forwards
    hash. Production/export/selected parity still holds on `active_config_hash`,
    `active_constraints_hash`, and `monthly_solution_hash`.

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

## Worktree / Commit Hygiene

Generated or refreshed local evidence, not default commit targets:

- `data/eex_forwards_history.parquet`
- `pfc_shaping/output/*2026-07-08*`
- `pfc_shaping/model/artifacts/*`
- `output/phase14/20260708_*`
- local benchmark outputs under `ompex_benchmark_read_only/`

The export script modified a tracked Phase 13 generated report; it was restored
from HEAD to avoid cross-phase pollution.

Commit candidates if audit GO:

- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260708-DAILY-GENERATION-ASOF20260707.md`
- `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260707_lshape100_yoy150_amp150_2032.json`
- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260707_lshape100_yoy150_amp150_2032.json`
