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

Experimental next-step model work: `pfc_shaping/lt/model/epex_shape_lab.py`
and `tests/test_epex_ab_shape_lab.py` add an LT-only, off-by-default EPEX
shape lab scaffold. It fits point-in-time CH EPEX residual templates, projects
hourly deltas into the BASE/PEAK/OFFPEAK nullspace, requires monthly BASE
constraints by default, shifts the existing fan rather than rebuilding it, and
explicitly forbids OMPEX/HFC as input, target, loss, or gate. It is not wired
into production or export and does not change the promotion-ready 2026-07-08
candidate.

Local A/B runner: `scripts/run_epex_shape_lab_ab.py` applies the lab to an
hourly candidate while deriving monthly BASE/PEAK constraints from that same
candidate. The 2026-07-08 trial wrote local lab-only evidence to
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/`
with `production_approved=false`, `ompex_used_in_selection=false`, 78 BASE and
78 PEAK monthly constraints, max after-constraint error
`1.666666804567285e-07`, and weighted negative hours `0`. Treat this as local
research evidence only, not promotion evidence.

Independent A/B comparison: `scripts/compare_epex_shape_lab_ab.py` compares the
baseline and adjusted lab candidates without OMPEX. The 2026-07-08 comparison
under `epex_shape_lab_ab_trial/independent_ab_comparison/` reports
`benchmark_policy=independent_no_ompex`, `max_abs_monthly_mean_delta_eur_mwh`
`9.722222239124298e-08`, fan width drift `0`, quantile order OK, weighted
negative hours `0`, solar-tail delta about `-2.07`, evening-ramp delta about
`+0.93`, and annual duck change about `+2.75` EUR/MWh. OMPEX should only be run
after this as advisory evidence, not as parameter-selection evidence.

OMPEX advisory post-check on the adjusted A/B candidate was run only after the
independent comparison. Output:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ompex_advisory_adjusted_20260708/`.
Adjusted vs baseline advisory deltas: MAE `-0.1985`, RMSE `-0.2334`,
correlation `+0.0035`, p95 absolute error `-0.5473`, inside p10/p90 rate
`+0.0043`, but max absolute error worsened by `+1.5537`. Treat this as
external advisory evidence only, not production approval and not parameter
selection evidence.

EPEX A/B governance audit: `scripts/audit_epex_shape_lab_governance.py` checks
lab-only status, OMPEX non-selection, independent no-OMPEX comparison,
monthly/fan drift thresholds, and optional advisory OMPEX role. The 2026-07-08
trial audit output is
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/governance_audit/epex_shape_lab_governance_audit.json`
with `status=PASS`, `failed_count=0`, `production_approval=NO`, and
`promotion_gate=false`.

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

