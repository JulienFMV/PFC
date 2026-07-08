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

Adjusted A/B promotion-style diagnostics are local lab evidence only. A
Yearly-only diagnostic forwards parquet was built under the trial folder
because the observed `data/eex_forwards_history.parquet` was stale
(`max_date=2026-06-17`). With that diagnostic source, adjusted Power BI strict
passes in
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/adjusted_powerbi_strict/`
(`powerbi_quality_gate_status=PASS`, shape score `9`, BASE/PEAK errors `0`,
critical flags `0`). Product normalization for adjusted and baseline both have
`critical_count=0`, `unsupported_count=0`, `quote_conflict_count=6`, and no
source hierarchy policy, so `all_gates_pass=false` as expected for lab-only
evidence.

Next EPEX sweep is pre-registered in
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/pre_registered_sweep_plan.json`
using `scripts/plan_epex_shape_lab_sweep.py`. It contains 27 trials over
weekend/low-tail/peak-subshape intensities `[0.25, 0.5, 0.75]`, records
`benchmark_policy=pre_registered_independent_no_ompex`, and forbids OMPEX/HFC
as selection input. Run/select these trials only via independent no-OMPEX
comparison and governance PASS; OMPEX can be advisory post-check only after a
trial is frozen.

The pre-registered sweep has now been executed with
`scripts/execute_epex_shape_lab_sweep.py` and summarized in
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/sweep_execution_summary.json`.
Result: 27/27 trials executed and eligible under independent no-OMPEX
governance. The frozen best no-OMPEX trial is
`trial_002_w0.25_l0.25_p0.50` with independent shape score
`6.350975764045719`, duck-change mean `3.6754139784914535` EUR/MWh,
solar-tail mean delta `-2.535581627746391` EUR/MWh, weekend mean delta
`-0.6477966303078719` EUR/MWh, ramp p99 increase
`2.0312658899999896` EUR/MWh, max monthly drift
`1.1155913942688404e-07`, fan-width drift `0`, weighted negative hours `0`,
and governance `PASS`. This remains lab-only and `production_approved=false`;
OMPEX can be run only now as advisory post-check, not as a re-ranking signal.
After read-only audit feedback, the executor was hardened against stale resume
artifacts, malformed/tampered plans, output directories outside the sweep root,
negative `--max-trials`, and ineligible `best_trial` reporting. Targeted
validation now reports `37 passed, 1 skipped`, and a resume check on the
existing sweep still reports `{"eligible_count": 27, "trial_count_executed":
27}`.

Next-sweep policy hardening is committed in the local work after expert audit
feedback. New EPEX sweep plans now include `selection_thresholds`,
`scoring_policy`, and optional `max_abs_delta_grid`; the executor records EPEX
spot age and fit coverage, applies freshness/coverage/ramp/min-price
thresholds, and ranks with the pre-registered scoring weights. Defaults for
new plans are `max_epex_spot_age_days=14.0`,
`min_epex_fit_coverage_days=730.0`, `max_ramp_p99_increase_eur_mwh=1.0`,
`min_adjusted_price_eur_mwh=-10.0`, and `ramp_penalty_weight=1.0`. Validation:
`39 passed, 1 skipped`. Next research action is to refresh EPEX spot, generate
a new no-OMPEX plan with a cap grid such as `[2.0, 3.0, 4.0, 6.0]`, then run
that sweep. The existing `trial_002` remains frozen lab evidence only.

Fresh-spot EPEX sweep V2 has now been run. Local generated spot refresh lives
under
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/`
with hourly coverage `2023-01-01 00:00 UTC -> 2026-07-08 23:00 UTC`; no
repository data cache was committed. V2 plan/summary are under
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/`.
It ran 108 no-OMPEX trials with cap grid `[2.0, 3.0, 4.0, 6.0]`; 39 were
eligible. Best frozen no-OMPEX trial is `t046_w05_l025_p075_d03`:
weekend `0.5`, low-tail `0.25`, peak-subshape `0.75`, cap `3.0`, score
`2.2242277207731145`, ramp p99 increase `0.9876442199999538`, min adjusted
price `-3.825623`, EPEX spot age `0.041666666666666664` days, fit coverage
`1282.9583333333333` days, monthly drift `8.602150532151586e-08`, width drift
`0`, weighted negative hours `0`, governance `PASS`. OMPEX 2026-07-08 was run
only after selection as advisory: selected minus baseline MAE `-0.1316206`,
RMSE `-0.1631454`, correlation `+0.0024746`, p95 abs `-0.40474`, inside p10/p90
`+0.0024822`, max abs `+0.987837`. Validation: `40 passed, 1 skipped`.
`t046` is still lab-only and NO-GO production until a real production/export/
capstone chain and artifact-bound source hierarchy policy exist.

T046 strict diagnostics now exist. A local diagnostic forwards parquet was
rebuilt from the EEX desk workbooks because `data/eex_forwards_history.parquet`
was stale at `2026-06-17`:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet`
with CH coverage `2020-05-04 -> 2026-07-07` and sha256
`a6244638c2234781853284ce2ad58d55d01265568cca6c85d4461f21446e8d76`.
Committed source hierarchy policy:
`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t046_asof20260707_fresh_epex_sweep_v2.json`
binds the exact t046 CSV sha
`8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`, forwards
sha above, and quote conflict identity hash
`a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`.
Product normalization strict for t046 passes with `all_gates_pass=true`,
`critical_count=0`, `unsupported_count=0`, `accepted_quote_conflict_count=6`,
`blocking_quote_conflict_count=0`. Power BI strict for t046 passes with
`powerbi_quality_gate_status=PASS`, shape score `9`, HFC-vs-spot score `9`,
BASE/PEAK EEX residuals `0`, weighted negative hours `0`, min weighted price
`4.84`, min price `-3.83`, and all critical flag counts `0` (`monthly_path`
warnings remain `4`). T046 still remains NO-GO production: no adjusted
production manifest, export manifest, selected config artifact, or capstone
triad exists.

`scripts/check_epex_lab_promotion_readiness.py` now makes that NO-GO explicit
instead of reusing the baseline monthly solver capstone for the adjusted hourly
lab CSV. T046 readiness output:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision.json`.
Status is `STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING` with
`approved=false`, `strict_diagnostics_pass=true`, `production_chain_pass=false`,
and missing `adjusted_production_manifest`, `adjusted_export_manifest`,
`adjusted_selected_config`, `adjusted_capstone`. Validation including the new
checker: `86 passed, 1 skipped`.

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

