# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260708-DAILY-GENERATION-ASOF20260707.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current daily generation: Wednesday 2026-07-08 was regenerated from the EEX
workbook available on 2026-07-08. The latest usable CH/DE/FR quote row in that
workbook is `2026-07-07`, so all new 2026-07-08 evidence is bound to
`forward_snapshot_date=2026-07-07`.

Current promotion-ready 2026-07-08 candidate:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/`

Power BI strict passes without `--allow-failed-gates`:
`powerbi_quality_gate_status=PASS`, `shape_score_10=9`, BASE/PEAK EEX residuals
`0.000000`, `monthly_path_critical_flags=0`, and
`cross_year_month_shape_warning_flags=0`. PNG diagnostics are in
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/png_diagnostics/`.

Production LT dry-run/save also completed and wrote:

- `pfc_shaping/output/pfc_15min_2026-07-08.csv`
- `pfc_shaping/output/pfc_15min_2026-07-08.parquet`
- `pfc_shaping/output/pfc_de_15min_2026-07-08.csv`
- `pfc_shaping/output/pfc_de_15min_2026-07-08.parquet`
- `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`

Promotion evidence is complete for this 2026-07-08 candidate:

- source hierarchy policy:
  `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260707_lshape100_yoy150_amp150_2032.json`
- selected config:
  `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260707_lshape100_yoy150_amp150_2032.json`
- capstone:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad.json`

Capstone reports `approved=true`, `status=PROMOTION_EVIDENCE_PASS`, and
`blocking_count=0`. Delivered-product audit passes strictly with
`accepted_quote_conflict_count=6`, `UNSUPPORTED=0`, `critical_count=0`, and
`delivered_curve_drift_count=0`.

OMPEX benchmark policy: OMPEX is useful but imperfect external evidence. It is
read-only, advisory, not ground truth, not an optimizer target, and not a
promotion authority. Use `scripts/compare_hpfc_ompex_benchmark.py` for
repeatable comparisons and retain alignment sensitivity, especially
`ompex_minus_1h_hourending` for files timestamped as hour-ending.

Previous 2026-07-07 promotion-ready daily candidate:
`output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/`.

Read-only Roasters/MIT audits after capstone all returned GO with no P0/P1
blocker. Follow-up hardening resolved the production manifest `source_hashes`
gap and clarified generated `export_report.md` wording: the local report is not
the promotion authority; selected config plus manifest-backed capstone remain
authoritative. Accepted residual P2s are sparse/far-horizon warnings without
any hidden CRITICAL. Do not commit `data/eex_forwards_history.parquet` or
generated output artifacts.

Previous promotion-ready Phase 14 CH candidate:
`output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/` supersedes
the earlier `asof20260623_yoy50_2032` candidate. The older candidate passed
manifest/audit gates, but PNG diagnostics showed an unacceptable far-horizon
monthly shape: annual-only years were too flat. The current candidate is bound
to the latest usable EEX quote row `2026-06-23` from the workbook available on
`2026-06-24`. Do not describe this as a 2026-06-24 forward snapshot.

Production/export/selected triad now passes:

- production manifest:
  `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`
- local export manifest:
  `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/fan_asof20260623_lshape100_yoy10_amp200_2032.monthly_curve_manifest.json`
- selected config artifact:
  `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260623_lshape100_yoy10_amp200_2032.json`
- `active_config_hash`:
  `f4b64f88919149a42a85693135c047b442ffa099011ce17e41c1cfe8782db88e`
- `active_constraints_hash`:
  `a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262`
- `monthly_solution_hash`:
  `d717a426f5fee7fe62abf294a0e44311040115fd4edb6a3a118f06bf7243832e`

Capstone:
`output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad.json`
reports `approved=true`, `status=PROMOTION_EVIDENCE_PASS`, and
`blocking_count=0`.

Delivered-product audit passes with the exact artifact-bound source hierarchy
policy
`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260623_lshape100_yoy10_amp200_2032.json`
(`accepted_quote_conflict_count=9`, `UNSUPPORTED=0`, `OUT_OF_SCOPE=3`), and
strict Power BI export passes without `--allow-failed-gates`
(`powerbi_quality_gate_status=PASS`, base/peak EEX error `0`, cross-year
warnings `0`, `seasonal_warning_flags=0`). PNG diagnostics are in
`output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/png_diagnostics/`.
Local generated output and refreshed `data/eex_forwards_history.parquet` are
evidence artifacts, not commit targets unless explicitly requested.

