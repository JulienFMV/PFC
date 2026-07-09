# Current Handoff

Latest active handoff:

`.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260709-T057-LOCKED-HOLDOUT.md`

Read order for new agents:

1. `AGENTS.md`
2. `CLAUDE.md` if running Claude Code
3. `.planning/HANDOFF.md`
4. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
5. Latest session handoff linked above

Do not treat older Phase 14 generated reports as accepted production evidence
unless the latest handoff or decision log names them explicitly.

Current Phase 14 EPEX lab status: T056 t005 is the best no-OMPEX replacement
candidate and now has strict diagnostic evidence, but it is not production
promoted. Selected trial:
`t005_w075_l025_p089_e005_n055_r00`; adjusted CSV sha256:
`5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`.

T056 selection evidence:

- selection summary:
  `output/phase14/t056_postval_final_micro_selection_summary/spot_backtest_selection_summary.json`
- `replacement_verdict.replace_incumbent=true`
- `selected_adjusted_csv_sha256=5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- OMPEX flags false for model, selection, and backtest.

T056 strict diagnostics now pass:

- source hierarchy policy:
  `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t056_asof20260707_postval_final_micro.json`
- product audit:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/product_normalization_with_policy/summary.json`
  reports `all_gates_pass=true`, `critical_count=0`, `unsupported_count=0`,
  `accepted_quote_conflict_count=6`, `blocking_quote_conflict_count=0`.
- Power BI strict:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/powerbi_strict/summary_metrics.csv`
  reports `powerbi_quality_gate_status=PASS`, `weighted_negative_hours=0`, and
  no critical flags.
- staged reproducibility:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/staged_adjusted_candidate_selection_guard/staged_lt_epex_lab_candidate_manifest.json`
  regenerates the exact adjusted CSV sha256 and records source provenance.
- readiness:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/promotion_readiness/decision_with_staged_manifest.json`
  reports `strict_diagnostics_pass=true`,
  `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`, and
  `production_chain_pass=false`.

Do not promote T056 until a real approved adjusted production manifest with run
identity exists and the adjusted export/selected/capstone chain is built from
that approved manifest. The source hierarchy policy approval is only a
QUOTE_CONFLICT waiver for the bound CSV/forwards/identity hash, not curve
approval.

T057 future holdout is now pre-registered and locked:

- plan:
  `.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t057_t056_asof20260709.json`
- plan sha256:
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`
- frozen at:
  `2026-07-09T00:00:00Z`
- holdout window:
  `2026-07-10T00:00:00Z` to `2026-07-24T00:00:00Z`
- baseline CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- adjusted CSV sha256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- pass criteria:
  at least `300` holdout hours and residual MAE improvement `>= 0.0 EUR/MWh`
  on the locked future no-OMPEX window.

When future EPEX spot data covers the window, run the plan's backtest command
with a refreshed future spot parquet, then run
`scripts/audit_epex_lab_locked_holdout.py` against the plan and the generated
`spot_backtest_summary.json`. Do not edit the plan after the holdout window
starts; create a new plan if the window or criteria must change.

Use `scripts/check_epex_lab_locked_holdout_coverage.py` before running the
backtest. Current coverage check against the 2026-07-08 spot parquet wrote
`output/phase14/t057_locked_t056_future_holdout/coverage_status_current_spot.json`
and reports `WAITING_FOR_FULL_SPOT_COVERAGE`: spot max
`2026-07-08T23:00:00Z`, observed holdout hours `0`, expected `336`.
For routine execution, prefer `scripts/run_epex_lab_locked_holdout.py`: it
writes coverage first and refuses to run backtest/audit until coverage is
complete. Current run summary:
`output/phase14/t057_locked_t056_future_holdout/current_spot_runner/locked_holdout_run_summary.json`
with `backtest_ran=false` and `audit_ran=false`.
The consolidated future approval audit with this run summary is
`output/phase14/t057_locked_t056_future_holdout/future_approval_path_with_holdout_current.json`;
it reports `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING` and includes
`locked_holdout_pass` in `remaining_blockers`.
The readiness checker also accepts `--locked-holdout-summary`; current enriched
readiness output is
`output/phase14/t057_locked_t056_future_holdout/promotion_readiness_with_locked_holdout_current.json`
and includes `locked_holdout_pass=FAIL` with
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

Follow-up hardening on 2026-07-09 binds T057 into the adjusted production
artifact chain. `scripts/build_epex_lab_adjusted_production_manifest.py` now
accepts `--locked-holdout-summary` and refuses requested production approval
unless the holdout run/audit is passing, read-only, non-promotional, and
no-OMPEX. NO-GO diagnostic manifests remain allowed without T057. The adjusted
chain builder now rejects approved manifests without a bound passing holdout
and propagates `locked_holdout_summary_sha256` into export, selected artifact,
and capstone. Readiness verifies the same holdout hash across
production/export/selected/capstone, and future approval audit refuses
`PROMOTION_READY` without a passing locked holdout.

Validation for this hardening:
`python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `53 passed, 1 skipped`. CLI help was checked for
`scripts/build_epex_lab_adjusted_production_manifest.py` and
`scripts/build_epex_lab_adjusted_production_chain.py`.

Second expert-audit hardening on 2026-07-09 made T057 binding SHA-strict:
readiness no longer accepts a same-path holdout replacement when the file hash
differs, future approval audits require all production readiness checks to be
present and passing, and the provided locked-holdout sidecar SHA must match the
SHA reported by readiness binding checks. `scripts/audit_epex_lab_locked_holdout.py`
now writes top-level no-OMPEX flags so audit-schema holdout artifacts are
directly consumable. Validation:
`python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `59 passed, 1 skipped`; after adding explicit tests for divergent
selected/capstone holdout SHA, false holdout policy on the production manifest,
and `production_chain_pass=true` without a holdout, it reported
`63 passed, 1 skipped`.
The future approval audit now also emits `required_production_checks` and a
specific next action when readiness was generated without the full production
check set. Targeted validation:
`python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `37 passed, 1 skipped`; regenerated current approval audit remains
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.
The local promotion bundle path is also covered: local bundle + readiness is
audited as `NO_GO_PRODUCTION_CHAIN_INCOMPLETE` and still lists locked-holdout
production checks as required. Validation:
`python -m pytest tests/test_build_epex_lab_promotion_bundle_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `38 passed, 1 skipped`.
Readiness now publishes `required_production_checks` directly, and future
approval takes the union of that declared set and its internal minimum to avoid
allowing synthetic readiness payloads to remove required checks. Validation:
`python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `39 passed, 1 skipped`. Regenerated current readiness and future
approval audit remain `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

Third expert-audit hardening on 2026-07-09 made the locked-holdout evidence
plan-identity-bound, not just pass-flag-bound. `scripts/run_epex_lab_locked_holdout.py`
and `scripts/audit_epex_lab_locked_holdout.py` now write a
`locked_plan_identity` containing the plan id, plan JSON SHA, holdout window,
baseline/adjusted CSV hashes, lab manifest hash, and selection summary hash.
`scripts/epex_lab_locked_holdout_policy.py` centralizes the policy used by the
production manifest builder, chain builder, readiness checker, and future
approval audit. A future passing holdout is rejected if its plan JSON is
missing, hash-mismatched, or no longer matches the recorded locked identity.
The current regenerated runner/audit still remain fail-closed:
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, `blocking_stage=locked_holdout_coverage`,
and the recommended command still points to the locked T057 plan.

Validation:
`python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `73 passed, 1 skipped`.

Coverage report hardening on 2026-07-09 extends the same plan identity to
`scripts/check_epex_lab_locked_holdout_coverage.py`. The coverage JSON now
contains `locked_plan_identity` with the locked T057 plan SHA and candidate
hashes, so the coverage preflight, runner summary, audit, readiness, and future
approval chain all refer back to the same frozen plan identity. The current
regenerated coverage remains `WAITING_FOR_FULL_SPOT_COVERAGE`, with plan SHA
`f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.
Validation:
`python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `42 passed, 1 skipped`.

T057 holdout audit schema hardening on 2026-07-09 requires the backtest summary
to be the expected lab-only schema before a locked holdout can pass. The audit
now checks `schema_version=epex_shape_lab_spot_backtest.v1`, `read_only=true`,
and `independent_production_evidence=false` in addition to the existing
no-OMPEX, lab-only, source-hash, valuation timestamp, and holdout metric
checks. Validation:
`python -m pytest tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `45 passed, 1 skipped`.

T057 backtest-output binding on 2026-07-09 makes the realized spot backtest
summary hash-bind its generated CSV outputs. `scripts/backtest_epex_shape_lab_against_spot.py`
now writes `output_hashes` for rolling folds, bucket metrics, candidate
profiles, and post-valuation residuals. `scripts/audit_epex_lab_locked_holdout.py`
now rejects a locked holdout if the post-valuation residual CSV hash does not
match the summary. Validation:
`python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `46 passed, 1 skipped`.

T057 inter-artifact binding on 2026-07-09 also makes passable run/audit
summaries hash-bind their referenced evidence files. A passing
`epex_lab_locked_holdout_run.v1` must now carry valid hashes for
`spot_backtest_summary` and `locked_holdout_audit`; a passing
`epex_lab_locked_holdout_audit.v1` must carry valid hashes for
`spot_backtest_summary` and the post-valuation residual CSV. The shared policy
rejects tampered linked files. Validation:
`python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `82 passed, 1 skipped`.
Future approval audit CLI now exits `0` only when `approved=true`; all NO-GO
states exit `1`. Validation:
`python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `41 passed, 1 skipped`. The current command against T057 evidence
returns exit `1` as expected because status remains
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

T057 coverage preflight now also verifies spot value usability before
unlocking the backtest. `scripts/check_epex_lab_locked_holdout_coverage.py`
requires the plan benchmark policy to be `locked_future_no_ompex_holdout`,
requires `price_eur_mwh` to be present in the refreshed EPEX spot parquet, and
requires all observed holdout-window prices to be finite. A complete hourly
index without the locked policy or usable prices remains
`WAITING_FOR_FULL_SPOT_COVERAGE` and the CLI exits `1`.

Expert-audit hardening also tightened downstream holdout evidence: passable
promotion policy now accepts only `epex_lab_locked_holdout_run.v1` summaries
from `scripts/run_epex_lab_locked_holdout.py`, not audit-only JSON sidecars.
The run summary must hash-bind `coverage_status.json`, the backtest summary,
and the locked-holdout audit; the shared policy opens those linked JSON files
and verifies schema/status/no-OMPEX/lab-only flags plus the same locked plan
identity. `scripts/audit_epex_lab_locked_holdout.py` now requires
`DIAGNOSTIC_PASS` and `strict_lab_gate_pass=true`.

The T057 runner now also requires explicit frozen-plan hash anchoring:
`scripts/run_epex_lab_locked_holdout.py` must be called with
`--expected-plan-sha256`. It computes the actual plan JSON hash before
coverage/backtest and writes
`NO_GO_LOCKED_HOLDOUT_PLAN_HASH_MISMATCH` without running coverage, backtest,
or audit if the hash differs. Passable run summaries must carry matching
`expected_plan_json_sha256`, `actual_plan_json_sha256`, and locked plan
identity hash. The current T057 expected plan hash remains
`f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.

Validation:
`python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `86 passed, 1 skipped`.
Follow-up validation after explicit frozen-plan hash anchoring:
`python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `90 passed, 1 skipped`.
Current regenerated T057 runner still exits `1` with
`WAITING_FOR_FULL_SPOT_COVERAGE`, spot max `2026-07-08T23:00:00Z`, observed
holdout hours `0`, expected `336`, and `coverage_status_sha256`,
`expected_plan_json_sha256`, and `actual_plan_json_sha256` present.
Additional preflight source-integrity hardening now verifies the locked
baseline and adjusted CSV paths before any T057 backtest can run. Coverage JSON
reports both paths and hashes, and `ready_to_run_backtest` requires both files
to exist and match the plan hashes. Source mismatch is reported as
`NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH`, not as a spot-coverage
wait. Current regenerated T057 evidence shows both source checks passing:
`baseline_csv_sha256_bound=true` and `adjusted_csv_sha256_bound=true`.
The same preflight now also verifies candidate CSV schema and holdout timestamp
coverage before backtest: required hourly export columns, parseable CH
timestamps, no duplicate timestamps, finite price/quantile columns, and full
coverage of the locked UTC holdout window. Current regenerated T057 evidence
shows these candidate checks passing, with
`baseline_candidate_missing_holdout_hours=0` and
`adjusted_candidate_missing_holdout_hours=0`.
Validation after this hardening:
`python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `94 passed, 1 skipped`.

Expert-audit follow-up on 2026-07-09 made the T057 holdout CLIs fail-closed:

- `scripts/check_epex_lab_locked_holdout_coverage.py` exits `0` only when
  `ready_to_run_backtest=true`; incomplete coverage exits `1`.
- `scripts/run_epex_lab_locked_holdout.py` exits `0` only for
  `LOCKED_HOLDOUT_PASS` with `holdout_pass=true`; coverage-pending and failed
  holdouts exit `1`.
- `scripts/audit_epex_lab_locked_holdout.py` exits `0` only when
  `holdout_pass=true`; `NO_GO_LOCKED_HOLDOUT_FAIL` exits `1`.

Validation:

```powershell
python -m pytest tests/test_run_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py -q -p no:cacheprovider
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Results: `12 passed` and `62 passed, 1 skipped`. Current real coverage and
runner commands against the 2026-07-08 spot parquet both exit `1` as expected.
The already locked T057 plan JSON was not rewritten. For future locked plans,
`scripts/plan_epex_lab_locked_holdout.py` now also emits
`commands.run_locked_holdout_template`, pointing to the fail-closed wrapper
`scripts/run_epex_lab_locked_holdout.py`, while keeping the separate backtest
and audit templates for traceability. Validation:
`python -m pytest tests/test_plan_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py -q -p no:cacheprovider`
reported `14 passed`.

Context hygiene: `.planning/CONTEXT.md` is historical Phase 5bis context and
now has a top-level notice pointing agents back to this handoff, the Phase 14
decision log, and the latest session handoff. The 2026-06-18 residual-anchor
external audit prompt is historical diagnostic context only; the active Phase
14 target architecture is the 2026-06-19 monthly solver reform. T057 outcome
policy is explicit: PASS permits only the next packaging step, not automatic
promotion; FAIL requires a new pre-registered lineage rather than retuning T056
against the locked window. The current T057 `>= 0.0 EUR/MWh` MAE criterion is a
non-degradation gate, not a full economic superiority proof.

Operational recheck after the context/plan-template changes:

- locked T057 plan hash remains
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`;
- `git diff` shows no changes to
  `.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t057_t056_asof20260709.json`;
- current coverage command against the 2026-07-08 spot parquet still exits
  `1`, as expected, with `WAITING_FOR_FULL_SPOT_COVERAGE`;
- source/staging/adjusted production-chain tests:
  `python -m pytest tests/test_build_epex_lab_source_export_manifest_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  reported `60 passed, 1 skipped`.

Future approval audit now emits machine-readable next-step routing:
`blocking_stage` and `next_required_step`. The current T057 audit remains
NO-GO and now reports `blocking_stage=locked_holdout_coverage` with
`next_required_step=wait_for_full_spot_coverage_then_run_locked_holdout`.
It also includes `recommended_commands.run_locked_holdout` with the fail-closed
wrapper command, the expected T057 plan hash when known, and placeholders for
`<FRESH_FUTURE_SPOT_PARQUET>` and `<T057_HOLDOUT_OUTPUT_DIR>`.
Validation:
`python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `41 passed, 1 skipped`; regenerating the current T057 future-approval
audit exits `1` as expected.

Current daily generation: Wednesday 2026-07-08 was regenerated from the EEX
workbook available on 2026-07-08. The latest usable CH/DE/FR quote row in that
workbook is `2026-07-07`, so all new 2026-07-08 evidence is bound to
`forward_snapshot_date=2026-07-07`.

Separate baseline daily production-ready 2026-07-08 candidate
(not the T056 EPEX lab adjusted candidate):

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

Promotion evidence is complete for this separate baseline daily candidate:

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

T047 v3 is the current next model step after expert audit of T046. The lab now
supports `night_intensity` and `ramp_intensity`, and a no-OMPEX 18-trial plan
has been pre-registered under
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/pre_registered_sweep_plan.json`.
The full sweep now reports `trial_count_executed=18`, `eligible_count=9` in
`sweep_execution_summary.json`. All 9 eligible trials were then run through
no-OMPEX spot bucket backtests under `output/phase14/t047_spot_backtest_by_trial/`.
Best internal ranking trial is `t005_w05_l025_p075_n00_r05_d02`; best
weak-bucket compromise is `t013_w05_l025_p075_n05_r00_d02`, with night MAE
improvement `0.1179` and ramp MAE improvement `0.0341`. T047 v3 is not frozen
as a T046 replacement because T046 still dominates overall profile, solar-tail,
evening recovery, weekend and post-valuation evidence. Validation for the T047
code path: `40 passed, 1 skipped`. Next action: refine night/ramp component
design and selection; OMPEX remains advisory-only after a candidate is frozen.
The weak-bucket selection is now reproducible through
`scripts/summarize_epex_shape_lab_spot_backtests.py`; real output is
`output/phase14/t047_spot_backtest_selection_summary/spot_backtest_selection_summary.json`
with `weak_bucket_candidate_count=1`,
`replacement_verdict.status=WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`,
and `replace_incumbent=false`. Validation including this summarizer:
`45 passed, 1 skipped`.
The eligible-trial spot-backtest step is now also scripted by
`scripts/run_epex_shape_lab_sweep_spot_backtests.py`. Real T047 resume run:
`output/phase14/t047_spot_backtest_by_trial/run_summary_from_runner.json`,
with `trial_count_backtested=9`, `reused_existing_count=9`, no OMPEX flags, and
the same chained selection verdict under
`output/phase14/t047_spot_backtest_selection_summary_from_runner/`. Validation
including the runner: `48 passed, 1 skipped`.

T048 night/core-recovery was run as local no-OMPEX lab evidence under short
paths to avoid Windows path-length failures:

- plan: `output/phase14/t048_ncr/pre_registered_sweep_plan.json`
- sweep summary: `output/phase14/t048_ncr/sweep_execution_summary.json`
- backtest run: `output/phase14/t048_ncr_spot_backtests/run_summary.json`
- selection summary:
  `output/phase14/t048_ncr_selection_summary/spot_backtest_selection_summary.json`

Result: 32 trials executed, 27 eligible, 27 backtested, 16 weak-bucket
candidates, all OMPEX flags false. Official verdict remains
`WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS` and
`replace_incumbent=false`. Best weak-bucket trial `t004` improves night/ramp
but degrades core buckets; best overall `t020` degrades evening/post-valuation;
best compromise `t024` is close but still slightly behind T046 on
solar-tail/post-valuation. Read-only MIT/Roaster audits therefore returned
NO-GO for replacing T046 with T048 and NO-GO for promoting any T046/T047/T048
adjusted lab artifact without a real chain-bound adjusted production manifest,
export manifest, selected artifact, and capstone.

Next model action is T049 core-balance under short paths:
`output/phase14/t049_core_balance`,
`output/phase14/t049_core_balance_spot_backtests`, and
`output/phase14/t049_core_balance_selection_summary`. It is centered on the
T048 `t020`/`t024` neighborhood with weekend `[0.65, 0.75]`, low-tail `[0.25]`,
peak-subshape `[0.75, 0.875, 1.0]`, night `[0.4, 0.5, 0.6]`, ramp
`[0.0, 0.125]`, cap `[2.5, 2.75]`, ramp p99 threshold `0.90`, and
`ramp_penalty_weight=2.0`. Replacement requires beating T046 on night/ramp
without material regression on overall, evening, solar-tail, weekend,
post-valuation, monthly/fan drift, negative-price stress, or delivered-product
normalization. Decision log entry: `D-20260708-43`.

T049 core-balance was executed after that handoff update. It ran 72 trials; 52
were eligible and spot-backtested. Result remains `replace_incumbent=false`.
The automatic best weak-bucket trial still degrades evening/post-valuation
versus T046, but T049 identified a stronger no-OMPEX frontier:
`t070_w075_l025_p01_n06_r00_d275`, adjusted CSV sha256
`f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`.
It beats T046 on overall, night, ramp, evening, weekend, and post-valuation,
but misses solar-tail by about `0.00606` EUR/MWh. Ramp p99 increase is
`0.8886024799999568`, below the pre-registered `0.90` threshold, and all
strict lab/no-OMPEX flags pass.

T050 micro-balance around `t070` was also run under
`output/phase14/t050_t070_micro_balance`. It executed 12 trials, 4 eligible,
and reproduced the same best adjusted CSV under trial
`t007_w075_l025_p01_n06_r00_d275`. Verdict remains
`WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`; only solar-tail
is degraded versus T046. T046 remains the incumbent lab candidate. Next action
is stricter delivered-curve diagnostics on the `t070` frontier, not a broad
sweep: delivered-product normalization, strict Power BI export in isolated
output, source-hierarchy binding, and optional OMPEX advisory only after the
no-OMPEX package is frozen. Decision log entry: `D-20260708-44`.

Those strict `t070` frontier diagnostics now pass, while production remains
NO-GO. New hash-bound source hierarchy policy:
`.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t070_asof20260707_t049_core_balance.json`
with policy sha256 `6c2f1b1f8bcf3bd732858a7e0b593c6e678d1e2758b5fc3c11f1bd5a4bbb462e`.
Product normalization strict output:
`output/phase14/t049_core_balance/t070_diagnostics/product_normalization_with_policy/summary.json`;
result `all_gates_pass=true`, `critical_count=0`, `unsupported_count=0`,
`accepted_quote_conflict_count=6`, `blocking_quote_conflict_count=0`. Power BI
strict output:
`output/phase14/t049_core_balance/t070_diagnostics/powerbi_strict/summary_metrics.csv`;
result `powerbi_quality_gate_status=PASS`, shape score `9`, BASE/PEAK residuals
`0`, weighted negative hours `0`, all critical flags `0`. Promotion readiness
output:
`output/phase14/t049_core_balance/t070_diagnostics/promotion_readiness/decision.json`;
result `approved=false`, `strict_diagnostics_pass=true`,
`production_chain_pass=false`,
`status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`. Missing adjusted
production evidence remains `adjusted_production_manifest`,
`adjusted_export_manifest`, `adjusted_selected_config`, and
`adjusted_capstone`. Decision log entry: `D-20260708-45`.

OMPEX 20260708 advisory was run only after T070 no-OMPEX selection and strict
diagnostics were frozen. Outputs:
`output/phase14/t049_core_balance/t070_diagnostics/ompex_advisory_baseline_20260708/benchmark_metrics.json`
and
`output/phase14/t049_core_balance/t070_diagnostics/ompex_advisory_t070_20260708/benchmark_metrics.json`.
Alignment is `ompex_minus_1h_hourending`, overlap `39481` points. T070 minus
baseline advisory deltas: MAE `-0.141892598541069`, RMSE
`-0.171529868510628`, correlation `+0.0026155586617813`, p95 absolute error
`-0.398625999999986`, inside p10/p90 `+0.00268483574377548`, max absolute
error `+0.937722`. This is favorable advisory evidence on average errors, but
it is not a model input, ranking signal, gate, or promotion authority. Decision
log entry: `D-20260708-46`.

T070 local non-production bundle was built under
`output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle/`.
It contains `adjusted_export_manifest.json`, `adjusted_selected_artifact.json`,
and `adjusted_local_capstone_no_go.json`, all local diagnostic evidence only.
Readiness with this bundle wrote
`output/phase14/t049_core_balance/t070_diagnostics/promotion_readiness/decision_with_local_bundle.json`;
exit code `1` is expected. Result:
`strict_diagnostics_pass=true`, `production_chain_pass=false`,
`approved=false`, `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`,
and `missing_production_evidence=["adjusted_production_manifest"]`. The local
export/selected/capstone artifacts are hash-bound but correctly fail
production-ready checks because they are not tied to a real adjusted production
manifest/run identity. Remaining blocker: governance/production packaging, not
another no-OMPEX shape sweep. Decision log entry: `D-20260708-47`.

The production-staging path was then fixed for T070. New code in
`scripts/stage_epex_lab_adjusted_lt_candidate.py` accepts `night_intensity` and
`ramp_intensity`, passes them to the EPEX lab runner, and records them in
`epex_lab_config`; tests in
`tests/test_stage_epex_lab_adjusted_lt_candidate_script.py` cover API and CLI
propagation. Validation: `23 passed, 1 skipped` for the stageer plus LT/CT
import guard. T070 staging now reproduces the frontier CSV exactly:
`output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate/`
has adjusted CSV sha256
`f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`, source
provenance manifest sha256
`dbc3bb810dffba948e6eadfc237890a6ebea3887e57a85e4236fda5e60473d51`, and
adjusted production manifest NO-GO sha256
`0e09ea55a130bce73de8bf9ba6a163cbc124f656fdb7b00375c8b2ac4d249048`.
Readiness with that NO-GO production manifest now has
`missing_production_evidence=[]`, source provenance PASS, and
`adjusted_production_contract_pass=true`, but remains
`production_chain_pass=false` because the manifest is not approved, run identity
is intentionally invalid/NO-GO, and local export/selected/capstone are not
production-chain-bound. Remaining blocker: create a real approved adjusted
production manifest from a production path, then use the strict adjusted
production-chain builder. Decision log entry: `D-20260708-48`.

The strict adjusted production-chain builder was also hardened before any real
approval attempt. `scripts/build_epex_lab_adjusted_production_chain.py` now
revalidates the underlying source provenance manifest instead of trusting
`source_provenance_pass=true`: provenance path/SHA, schema/role,
candidate-CSV source kind, promotion eligibility, no blockers, adjusted/source/
staged/lab/source-export hashes, source-export binding, and no-OMPEX flags.
Tests now reject missing/self-attested provenance and tampered source hashes.
Validation: `34 passed, 1 skipped` for production-chain/readiness/production-
manifest tests plus LT/CT import guard. Running the builder on the real T070
NO-GO manifest still fails as expected with
`approved adjusted production manifest required: production_approved,
production_promotion_approved, git_commit`. Decision log entry:
`D-20260708-49`.

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
warnings remain `4`). T046 still remains NO-GO production: strict diagnostics
pass, but there is no production-approved adjusted production/export/selected/
capstone chain for the adjusted hourly lab curve.

T046 independent no-OMPEX A/B diagnostics have been enriched in
`scripts/compare_epex_shape_lab_ab.py`. The 2026-07-08 run under
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_enriched_ab_diagnostics/`
adds load-type deltas, month-hour deltas, PEAK/OFFPEAK monthly spread deltas,
boundary delta jumps, and PNG heatmaps. It remains advisory diagnostic evidence
only: `benchmark_policy=independent_no_ompex`, OMPEX not used in model or
selection, max monthly mean drift about `8.6e-08`, width drift `0`, weighted
negative hours `0`, ramp p99 increases from about `24.82` to `25.81`, and the
largest month-boundary delta jump is about `0.279 EUR/MWh`.

The fan-parquet staging blocker is now diagnosed by
`scripts/diagnose_fan_to_hourly_parity.py`. The 2026-07-08 fan vs audited
hourly CSV run under
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_to_hourly_parity_diagnostic/`
shows identical aligned hourly row count (`57025`) but max absolute weighted
delta about `24.29 EUR/MWh`, mean absolute weighted delta about `2.80 EUR/MWh`,
PEAK mean delta about `+1.24 EUR/MWh`, OFFPEAK mean delta about
`-0.68 EUR/MWh`, and raw fan-derived product audit failures
(`critical_count=56`, `delivered_curve_drift_count=38`). This confirms raw
`to_hourly_csv_frame` output is not equivalent to the audited export chain and
must not become promotion-facing until the full calibration/export path is
reconciled.
`scripts/stage_epex_lab_adjusted_lt_candidate.py` now enforces that finding:
fan-parquet staging records `source_promotion_eligible=false` and blocks
adjusted production-contract packaging with
`source_kind_fan_parquet_requires_audited_hourly_export`; candidate CSV staging
remains eligible to package a NO-GO contract when strict evidence is supplied.
`scripts/build_epex_lab_adjusted_production_manifest.py` and
`scripts/check_epex_lab_promotion_readiness.py` also require source provenance
for any future adjusted production approval: complete run identity plus a
`source_provenance_manifest` proving `source_kind=candidate_csv`,
`source_promotion_eligible=true`, empty contract blockers, and adjusted CSV
path/SHA binding. Readiness requires both `contract_pass=true` and
`source_provenance_pass=true`.
The audited-hourly T046 staging path has been rerun under
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_with_provenance/`;
its adjusted CSV hash is still `8b50...`, the NO-GO contract now has
`contract_pass=true` and `source_provenance_pass=true`, and readiness remains
`approved=false` because production/export/selected/capstone approval flags are
still deliberately false.

T046 stability v1 has been summarized across the two available current-data
baselines without OMPEX:
`output/phase14/t046_stability_summary_v1/`. The cases are `asof20260706` and
`asof20260707`; both pass read-only stability checks with
`benchmark_policy=multi_date_independent_no_ompex`, zero weighted negative
hours, fan width drift `0`, monthly drift about `8.6e-08`, and ramp p99
increases below the `1.0` EUR/MWh threshold (`0.9379` and `0.9876`). This is
lab evidence only: `promotion_gate=false`, T046 remains NO-GO production, and
OMPEX remains advisory benchmark evidence only.

Post-audit hardening D28 supersedes the earlier D26 wording about a "complete"
T046 contract. A future adjusted production approval now requires the readiness
checker to reload and hash-validate `source_provenance_manifest`, prove
`source_kind=candidate_csv`, prove `source_promotion_eligible=true`, prove an
export manifest is bound to the source CSV, and reject self-attested production,
export, selected, or capstone manifests. Under this tightened contract, a
candidate CSV without a source export manifest remains stageable but not
promotion-contract eligible.

Post-audit diagnostic hardening D29 adds implied-width and coverage signals.
`scripts/compare_epex_shape_lab_ab.py` now reports implied width
`structural_p90 - structural_p10`, reported-minus-implied width, and monthly
implied-width drift. `scripts/diagnose_fan_to_hourly_parity.py` now reports
missing timestamps, coverage ratios, and `coverage_status`. These are read-only
lab diagnostics and do not promote T046.

Stability v2 D30 now gates local-shape risk in the no-OMPEX stability summary:
PEAK/OFFPEAK spread delta, month-hour mean delta, month-boundary delta jump,
implied width drift, reported-minus-implied width, p10 negative hours, and p10
negative cluster length. Real T046 v2 outputs:
`output/phase14/t046_stability_summary_v2/`; status `PASS`, two cases passed,
`promotion_gate=false`. Key observed maxima: month-hour about `2.2225`
EUR/MWh, boundary jump about `0.2786` EUR/MWh, PEAK/OFFPEAK spread delta about
`2.5e-07` EUR/MWh, p10 negative hours `125`/`118`, p10 cluster max `6`,
weighted negatives `0`.

Delta-field stability D31 compares the actual frozen T046 delta across
`asof20260706` and `asof20260707`:
`output/phase14/t046_delta_stability_summary_v1/`. Status `PASS`,
`promotion_gate=false`, config hash stable
`e9c1f0831cb896f03987eeefcbb92dfbf900a53eaad4c4d45ab58e761e163b51`,
timestamp delta correlation `0.9999797676440942`, timestamp MAE
`0.0011066597457297395` EUR/MWh, timestamp max abs `0.05641400000001795`
EUR/MWh, month-hour MAE `0.0009053573348698669`, boundary jump max abs
`0.015835000000009813`, and no missing timestamps.

Source-export provenance D32 now closes the D28 source-manifest gap without
promoting T046. Source manifest:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_source_export_manifest/source_export_manifest.json`
binds the baseline hourly CSV SHA
`12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895` and
has SHA `d662548e2e7605ba2b59e024afd3040f2724fe84c5f3c7d3491fbaa0e1909f1d`.
Rerun staging:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/`
has source provenance SHA
`eefe822b24a876a176b78afd9ccc21552d4c5248d8833a7c8ee1bbd368789d1f`,
NO-GO adjusted production manifest SHA
`7824522ca68f64da20bd7871cba0beed246f27bfb888beef2d1cc65ffdbd17a9`,
`source_promotion_eligible=true`, `source_export_manifest_bound=true`, and
`adjusted_production_contract_pass=true`. Readiness still reports
`approved=false`, `strict_diagnostics_pass=true`, `production_chain_pass=false`;
the remaining failures are only the expected non-production approval flags.

No-OMPEX spot backtest D33 adds a lab-only realized EPEX spot diagnostic without
changing the T046 production verdict. New script:
`scripts/backtest_epex_shape_lab_against_spot.py`; test:
`tests/test_backtest_epex_shape_lab_against_spot_script.py`. Real output:
`output/phase14/t046_spot_backtest_v1/spot_backtest_summary.json`.
Result `status=DIAGNOSTIC_PASS`, `strict_lab_gate_pass=true`,
`promotion_gate=false`, `production_approved=false`,
`independent_production_evidence=false`, and
`benchmark_policy=rolling_origin_epex_spot_no_ompex_lab_only`. Bound hashes:
baseline CSV `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`,
adjusted CSV `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`,
spot parquet `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`.
Rolling folds: `12/12` eligible, all no-temporal-leak, positive MAE
improvement `12/12`, mean baseline profile MAE `14.153227063777985`, mean
adjusted profile MAE `13.747743522746092`, mean improvement
`0.40548354103189205` EUR/MWh. The diagnostic explicitly records that all 12
historical folds are not independent of the current candidate fit; the only
true post-valuation overlap is 24 hours, with residual MAE improvement
`0.3048038417338681` EUR/MWh. This is useful shape evidence, not promotion
evidence.

D34 extends the same no-OMPEX spot backtest with economic bucket and hourly
ramp diagnostics. Real output:
`output/phase14/t046_spot_backtest_v2_buckets/spot_backtest_summary.json`;
bucket CSV:
`output/phase14/t046_spot_backtest_v2_buckets/rolling_spot_bucket_metrics.csv`.
Result remains `DIAGNOSTIC_PASS`, `promotion_gate=false`,
`production_approved=false`, and all OMPEX flags false. Selected mean MAE
improvements in EUR/MWh: all residual level `0.24513954474101998`, weekend
`0.2889611347370835`, weekday `0.22708125671275944`, PEAK-like weekday 08-19
`0.32096908439747596`, OFFPEAK-like `0.20198153831529964`, solar tail
Mar-Oct 10-16 `0.4372953091304925`, midday 11-15 `0.35776460522648684`,
evening ramp 17-21 `0.45338812791781463`, night 00-05
`0.03190894115068499`, and hourly ramp all `0.035478178105887714`. This says
T046 helps most on evening recovery, solar/midday, peak-like, and weekend
buckets; night and ramp gains are weak and should guide future research rather
than promotion claims.

Future approval path audit D35 now summarizes the production blockers in a
compact review artifact. New script:
`scripts/audit_epex_lab_future_approval_path.py`; test:
`tests/test_audit_epex_lab_future_approval_path_script.py`. Real output:
`output/phase14/t046_future_approval_path_audit_v1/future_approval_path_audit.json`.
Result `status=NO_GO_PRODUCTION_CHAIN_INCOMPLETE`, `approved=false`,
`strict_diagnostics_pass=true`, `production_chain_pass=false`,
`spot_backtest_policy.pass=true`, and `missing_production_evidence=[]`. The
remaining blockers are present-but-not-approved production approvals:
`adjusted_capstone_approved`, `adjusted_export_manifest_production_ready`,
`adjusted_production_manifest_approved`, and
`adjusted_selected_artifact_production_ready`. Next action is not another local
bundle; it is replacing local diagnostic approval flags with real
production-approved adjusted artifacts.

D36 hardens the only API path that can set adjusted production-manifest
approval flags. `scripts/build_epex_lab_adjusted_production_manifest.py` now
requires valid run identity before accepting approval flags: non-empty
`production_run_id`, non-empty `production_entrypoint`, `git_commit` matching
`[0-9a-f]{40}`, and an existing `source_provenance_manifest`. The CLI remains
NO-GO by default and exposes no approval flags. Validation including this
hardening: `56 passed, 1 skipped`.

D37 hardens readiness so production-ready adjusted export, selected, and
capstone artifacts must bind to the same adjusted production manifest and run
identity, not only to the adjusted CSV. New readiness checks include
`adjusted_production_manifest_run_identity_valid`,
`adjusted_export_manifest_production_chain_bound`,
`adjusted_selected_artifact_production_chain_bound`, and
`adjusted_capstone_production_chain_bound`. Real T046 readiness v2:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/readiness_no_go_v2_chain_bound.json`.
Future approval audit v2:
`output/phase14/t046_future_approval_path_audit_v2_chain_bound/future_approval_path_audit.json`.
Result remains NO-GO: strict diagnostics pass, spot backtest policy pass, but
8 production checks fail because local/staging artifacts are not approved and
not bound to a real production run identity. Validation including D37:
`57 passed, 1 skipped`.

D38 adds a strict builder for the remaining adjusted production-chain artifacts:
`scripts/build_epex_lab_adjusted_production_chain.py` with tests in
`tests/test_build_epex_lab_adjusted_production_chain_script.py`. It can write
`adjusted_export_manifest.json`, `adjusted_selected_artifact.json`, and
`adjusted_production_capstone.json`, but only if the input adjusted production
manifest is already approved, promotion-approved, contract-pass,
source-provenance-pass, no-OMPEX, adjusted-CSV hash-bound, and run-identity
valid. This is the safe future path after a real adjusted production manifest
exists; it cannot be used to promote current T046 local NO-GO staging.
Validation including D38: `59 passed, 1 skipped`.

`scripts/check_epex_lab_promotion_readiness.py` now makes that NO-GO explicit
instead of reusing the baseline monthly solver capstone for the adjusted hourly
lab CSV. T046 readiness output:
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision.json`.
Status is `STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING` with
`approved=false`, `strict_diagnostics_pass=true`, `production_chain_pass=false`,
and missing `adjusted_production_manifest`, `adjusted_export_manifest`,
`adjusted_selected_config`, `adjusted_capstone`. Validation including the new
checker: `86 passed, 1 skipped`.

`scripts/build_epex_lab_promotion_bundle.py` now packages local non-production
adjusted evidence for t046. Generated bundle under
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_local_promotion_bundle/`
contains `adjusted_export_manifest.json`, `adjusted_selected_artifact.json`,
and `adjusted_local_capstone_no_go.json`, all explicitly
`production_approved=false` / local diagnostic scope. Rerunning readiness with
this bundle writes
`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision_with_local_bundle.json`;
status remains `STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`, but missing
evidence is now only `adjusted_production_manifest`. Validation: `87 passed,
1 skipped`.

Follow-up Phase 14 D50 hardening: adjusted production approval now requires an
explicit no-OMPEX selection pass in addition to diagnostics, source provenance,
and production manifest checks. `scripts/build_epex_lab_adjusted_production_manifest.py`
records `selection_summary`, `selection_summary_sha256`, and
`selection_policy_pass`; production approval requires
`selection_policy_pass=true`. `scripts/epex_lab_selection_policy.py` provides a
shared validator that requires explicit no-OMPEX flags,
`replacement_verdict.replace_incumbent=true`, and exact selected-artifact hash
binding. `scripts/check_epex_lab_promotion_readiness.py` and
`scripts/build_epex_lab_adjusted_production_chain.py` reload the bound
`selection_summary`, verify its sha256, and recalculate that policy instead of
trusting the manifest boolean. They fail closed when the selection file is
absent, tampered, not no-OMPEX, not selected, or not replacement-approved, and
`scripts/stage_epex_lab_adjusted_lt_candidate.py` forwards
`--selection-summary`.

Current T070 selection-guard staging remains NO-GO:

- adjusted CSV sha256:
  `f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`
- selection summary:
  `output/phase14/t049_core_balance_selection_summary/spot_backtest_selection_summary.json`
- selection summary sha256:
  `0822379db522fadedbb12ae0ab327763fc2cbf28dac4443905ca2f010fb62183`
- NO-GO adjusted production manifest:
  `output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate_selection_guard/adjusted_production_manifest_no_go.json`
- NO-GO adjusted production manifest sha256:
  `a042a9b22ac8144e00b62f46c879d46921f4fd9686e94698f54348d4271c12e1`
- `selection_policy_pass=false`
- `replacement_verdict.replace_incumbent=false`
- no OMPEX flags.

Validation for D50:
`python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `47 passed, 1 skipped`.

Post-D50 search status: T070 remains NO-GO because it misses the incumbent
solar-tail metric. T051/T052 showed that tuning `peak_subshape_intensity` and
cap alone cannot beat both solar-tail and evening/post-valuation. A new
lab-only EPEX parameter, `evening_recovery_intensity`, was added to split the
h17-h21 recovery from the broader peak-subshape lever. It is fitted only from
EPEX residuals, projected through the existing BASE/PEAK nullspace, recorded in
lab/staging manifests, and remains no-OMPEX/off-production.

Validation for the evening-recovery component:
`python -m pytest tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
reported `34 passed, 1 skipped`.

Historical pre-T056 local frontier was T053:

- selection summary:
  `output/phase14/t053_evening_recovery_bridge_selection_summary/spot_backtest_selection_summary.json`
- best trial:
  `t003_w075_l025_p082_e025_n055_r00_d27`
- adjusted CSV sha256:
  `8b1c7f43bdaf3513d417fb6f436470847270af4b83ad5e5053eab08c16b94762`
- beats T046 on overall, evening, solar-tail, weekend, night, and ramp;
- still misses post-valuation:
  `0.292289994623653` vs incumbent `0.3048038417338681`;
- verdict remains `replace_incumbent=false`.

T054 high-peak/low-tail reproduced the T070 family and remains NO-GO on
solar-tail. Do not promote T053/T054. This block is retained as historical
pre-T056 search context only; the active adjusted-candidate line is T056/T057,
and any new search must be pre-registered without weakening D50 selection
policy or using OMPEX as model/selection/gate input.

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

## 2026-07-09 T057 Candidate Timestamp-Set Identity Follow-Up

- `scripts/check_epex_lab_locked_holdout_coverage.py` now records each locked
  candidate CSV timestamp count, UTC min/max, and sorted timestamp-set SHA.
- `ready_to_run_backtest` requires
  `checks.candidate_timestamp_sets_identical=true`.
- `scripts/epex_lab_locked_holdout_policy.py` now requires explicit
  baseline/adjusted candidate preflight checks, source CSV hash binding, and
  identical candidate timestamp-set SHA in the embedded coverage payload.
- Regenerated current T057 runner remains
  `WAITING_FOR_FULL_SPOT_COVERAGE`, exit `1`, because the spot parquet ends at
  `2026-07-08T23:00:00Z` and covers `0/336` locked holdout hours.
- Candidate preflight is clean: baseline and adjusted CSVs both have
  `57025` unique timestamps from `2026-06-30T22:00:00Z` through
  `2032-12-31T22:00:00Z`, with identical timestamp-set SHA
  `c1ac9c621b1293e296f5789c342da5ecfee8444dc8fa0ad1030686079245020e`.
- Regenerated future approval audit remains
  `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`; the new candidate preflight checks
  are true, and the holdout pass remains blocked only by future spot coverage.

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `96 passed, 1 skipped`.

## 2026-07-09 T057 Self-Attested Coverage Policy Follow-Up

Read-only expert audits after D70 returned GO overall and NO-GO promotion until
future spot coverage exists. One P1 was accepted: the downstream policy was
still too dependent on boolean `coverage.checks` fields.

`scripts/epex_lab_locked_holdout_policy.py` now rejects passable run summaries
unless the embedded coverage payload also carries
`schema_version=epex_lab_locked_holdout_coverage.v1`, read-only/non-promotional
flags, locked-plan identity matching the run summary, source CSV SHA fields
matching identity, non-empty equal candidate timestamp-set SHA fields, positive
equal timestamp counts, and equal non-empty timestamp min/max bounds.

Passing holdout fixtures in manifest/chain/readiness/future-approval tests now
include this raw coverage evidence. New policy tests reject coverage without raw
candidate timestamp evidence and coverage whose embedded locked-plan identity
does not match the run.

Validation:

```powershell
python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py -q -p no:cacheprovider
```

Result: `22 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `98 passed, 1 skipped`.

Regenerated future approval audit remains
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, exit `1`. The new raw-coverage checks
are true on the current T057 runner; the remaining holdout blocker is still
future spot coverage.

## 2026-07-09 T057 Explicit UTC Offset Follow-Up

The accepted P2 from the read-only data audit has been addressed:
`utc_offset_ch` is now required for locked baseline and adjusted candidate CSVs.
This closes the DST ambiguity fallback for promotion evidence.

Implementation summary:

- `scripts/check_epex_lab_locked_holdout_coverage.py` requires
  `utc_offset_ch` and reports `baseline_candidate_utc_offset_present` /
  `adjusted_candidate_utc_offset_present`.
- `scripts/epex_lab_locked_holdout_policy.py` requires the new offset checks in
  downstream candidate coverage policy.
- Tests now cover missing offset, duplicate parsed candidate timestamps,
  non-finite candidate prices/quantiles, and a DST fall-back case with repeated
  local `02:00` rows distinguished by explicit offsets.

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `26 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `102 passed, 1 skipped`.

Current regenerated T057 runner remains
`WAITING_FOR_FULL_SPOT_COVERAGE`, exit `1`, with
`baseline_candidate_utc_offset_present=true` and
`adjusted_candidate_utc_offset_present=true`. Regenerated future approval audit
remains `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`; the remaining holdout blocker
is still future spot coverage.

## 2026-07-09 T057 Resolved Evidence Paths Follow-Up

The remaining path-sensitivity P2 from the read-only audit has been addressed
for newly generated T057 evidence.

- `build_locked_plan_identity()` now resolves `plan_json` before storing and
  hashing it.
- Coverage, runner, and locked-holdout audit writers now resolve their main
  input/output paths before writing summaries.
- Future-approval audit now quotes real CLI arguments in the recommended
  `run_locked_holdout` command, so absolute paths with spaces remain runnable.
- A policy unit test covers relative `plan_json` input being stored as a
  resolved path with the correct SHA.

Validation:

```powershell
python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result: `49 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `103 passed, 1 skipped`.

Regenerated current T057 runner remains `WAITING_FOR_FULL_SPOT_COVERAGE`,
exit `1`, but writes resolved UNC paths for `plan_json`, `spot_parquet`,
`output_dir`, `coverage_status`, and `run_summary`. Regenerated future approval
audit remains `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, exit `1`, and its
recommended command quotes the resolved locked plan path.

## 2026-07-09 T057 Future Plan Timestamp Identity Follow-Up

New locked EPEX lab holdout plans now freeze candidate timestamp identity at
plan creation time.

- `scripts/plan_epex_lab_locked_holdout.py` parses `timestamp_ch` plus
  `utc_offset_ch` from baseline and adjusted CSVs.
- Plan build fails if either candidate timestamp set is missing, unparseable,
  duplicated, or different between baseline and adjusted.
- New plans include `candidate_timestamp_identity` and copy
  `candidate_timestamp_count` / `candidate_timestamp_set_sha256` into
  `pass_criteria`.
- `scripts/check_epex_lab_locked_holdout_coverage.py` enforces those optional
  criteria via `candidate_timestamp_set_matches_plan` and
  `candidate_timestamp_count_matches_plan`.
- `scripts/epex_lab_locked_holdout_policy.py` requires those checks downstream.

Validation:

```powershell
python -m pytest tests/test_plan_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `31 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `105 passed, 1 skipped`.

Current frozen T057 plan remains unchanged and backward-compatible:
regenerated runner reports `expected_candidate_timestamp_set_sha256=null`,
both plan-match checks true, and still `WAITING_FOR_FULL_SPOT_COVERAGE`.
Regenerated future approval audit remains
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

## 2026-07-09 Future Locked Plan Resolved Paths Follow-Up

New locked EPEX lab holdout plans now resolve source/evidence paths at plan
creation time.

- `scripts/plan_epex_lab_locked_holdout.py` resolves `baseline_csv`,
  `adjusted_csv`, optional `selection_summary`, optional `lab_manifest`, and
  optional `output` before hashing, reading, writing, or building command
  templates.
- Command templates now carry resolved paths, with existing quoting protecting
  spaces.
- Relative-path plan-builder test covers a working directory with a space and
  verifies resolved stored paths plus quoted command arguments.

Validation:

```powershell
python -m pytest tests/test_plan_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `32 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `106 passed, 1 skipped`.

Current frozen T057 plan remains unchanged and backward-compatible; regenerated
runner remains `WAITING_FOR_FULL_SPOT_COVERAGE`, exit `1`.

## 2026-07-09 Future Locked Plan Selection/Lab Manifest Follow-Up

New locked EPEX lab holdout plans must now include both candidate-selection
evidence and lab-config provenance.

- `scripts/plan_epex_lab_locked_holdout.py` rejects missing
  `selection_summary`.
- It also rejects missing `lab_manifest`.
- The CLI now requires `--selection-summary` and `--lab-manifest`.
- Tests cover both missing-artifact failures, plus existing unbound-selection
  and timestamp-set mismatch failures.

Validation:

```powershell
python -m pytest tests/test_plan_epex_lab_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `6 passed`.

```powershell
python -m pytest tests/test_plan_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_run_epex_lab_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `39 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `108 passed, 1 skipped`.

Current frozen T057 plan remains unchanged and already includes both artifacts.
Regenerated runner remains `WAITING_FOR_FULL_SPOT_COVERAGE`, exit `1`.

## 2026-07-09 Locked Holdout Coverage Status Routing Follow-Up

Locked EPEX lab holdout coverage preflight now separates invalid inputs from
the normal future-spot waiting state.

- `scripts/check_epex_lab_locked_holdout_coverage.py` emits
  `blocking_checks`.
- Source/candidate path/hash/timestamp failures remain
  `NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH`.
- Invalid plan or spot inputs now report
  `NO_GO_LOCKED_HOLDOUT_INPUT_INVALID`.
- True future spot incompleteness still reports
  `WAITING_FOR_FULL_SPOT_COVERAGE`.
- Tests cover missing spot price column, non-finite prices, wrong locked
  benchmark policy, and duplicate holdout spot rows.

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py -q -p no:cacheprovider
```

Result: `18 passed`.

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `68 passed, 1 skipped`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `109 passed, 1 skipped`.

Current frozen T057 plan remains unchanged. Regenerated current T057 runner
still exits `1` with `WAITING_FOR_FULL_SPOT_COVERAGE`; coverage
`blocking_checks` are only `full_window_covered` and `min_holdout_hours_met`.

## 2026-07-09 Locked Holdout Policy Blocking-Checks Follow-Up

Passable locked EPEX lab holdout run summaries must now embed coverage evidence
with `blocking_checks=[]`.

- `scripts/epex_lab_locked_holdout_policy.py` requires
  `coverage_blocking_checks_clear`.
- Passing fixtures now include `blocking_checks=[]`.
- A policy test rejects a run summary whose coverage lists
  `full_window_covered` as a blocking check despite otherwise pass-like flags.

Validation:

```powershell
python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result: `22 passed`.

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `69 passed, 1 skipped`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result: `57 passed`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `110 passed, 1 skipped`.

Current frozen T057 plan remains unchanged. Regenerated current T057 runner
still exits `1` with `WAITING_FOR_FULL_SPOT_COVERAGE`; coverage
`blocking_checks` are `full_window_covered` and `min_holdout_hours_met`, so a
future PASS must be generated after full spot coverage with empty blockers.

## 2026-07-09 Locked Holdout Input-Invalid Routing Follow-Up

Locked holdout policy and future approval audit now preserve
`NO_GO_LOCKED_HOLDOUT_INPUT_INVALID`.

- `scripts/epex_lab_locked_holdout_policy.py` preserves the input-invalid
  status instead of collapsing it into generic holdout failure.
- `scripts/audit_epex_lab_future_approval_path.py` routes it to
  `blocking_stage=locked_holdout_input_invalid`.
- The audit emits
  `next_required_step=fix_locked_holdout_plan_or_spot_inputs_then_rerun_preflight`.

Validation:

```powershell
python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result: `24 passed`.

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `71 passed, 1 skipped`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `112 passed, 1 skipped`.

Current frozen T057 plan remains unchanged. Regenerated current T057 runner
still exits `1` with `WAITING_FOR_FULL_SPOT_COVERAGE`.

## 2026-07-09 T061 Separate Future Holdout For T060

Created a separate future holdout line for the T060 EPEX-only cap
decompression challenger. This does not modify T057 and does not approve
production.

Code change:

- `scripts/plan_epex_lab_locked_holdout.py` now emits generic locked-holdout
  placeholders for newly generated plans:
  `<LOCKED_HOLDOUT_PLAN_JSON>`,
  `<LOCKED_HOLDOUT_PLAN_JSON_SHA256>`, and
  `<LOCKED_HOLDOUT_OUTPUT_DIR>`.
- `tests/test_plan_epex_lab_locked_holdout_script.py` validates those generic
  placeholders and checks that new plans no longer carry a T056-specific note.

Generated tracked plan:

- `.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t061_t060_asof20260709.json`
- Plan SHA256:
  `29a633cf56279eae817cd6c63872a476cc2c10b187f08c3952f73cdad76db135`
- `plan_id=t061_locked_t060_future_holdout`
- frozen at `2026-07-09T00:00:00Z`
- holdout window `2026-07-24T00:00:00Z` to `2026-08-07T00:00:00Z`
- valuation timestamp `2026-07-07T00:00:00Z`
- baseline CSV SHA256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- adjusted CSV SHA256:
  `0a0fe8ce8c12bfeb64ac517ef60ac4d2850fbd1d13255c823c213c94c98391a6`
- `selection_policy.pass=true`
- `production_approved=false`, `promotion_gate=false`

Validation so far:

```powershell
python -m pytest tests\test_plan_epex_lab_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `6 passed`.

```powershell
python -m pytest tests\test_plan_epex_lab_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_audit_epex_lab_locked_holdout_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `62 passed, 1 skipped`.

Operational conclusion:

- T057 remains frozen for T056/t005 and must not be retuned or replaced.
- T061 is frozen separately for T060/t007 and cannot run until future EPEX spot
  coverage exists for `2026-07-24` to `2026-08-07` UTC.
- T060/T061 remains lab/future-holdout evidence only. Promotion remains NO-GO
  until future holdout and production/export/selected/capstone evidence pass.

## 2026-07-09 Locked Holdout Queue Audit

Added read-only queue audit:

`scripts/audit_epex_lab_locked_holdout_queue.py`

Purpose:

- summarize multiple locked holdout plans in one operator-facing JSON;
- compute and expose exact plan SHA values;
- verify bound baseline, adjusted, lab-manifest, and selection-summary
  artifacts exist locally and match the plan hashes;
- discover locked holdout plans with `--plan-glob` so T057/T061 cannot be
  accidentally omitted from the queue audit;
- classify each plan as waiting for start, in-window, or ready for spot refresh;
- emit exact Energy Charts locked-holdout commands with plan SHA binding;
- never fetch spot, run a holdout, tune a candidate, or approve production.

Validation:

```powershell
python -m pytest tests\test_audit_epex_lab_locked_holdout_queue_script.py -q -p no:cacheprovider
```

Result after glob support and artifact-mismatch CLI exit checks: `7 passed`.

```powershell
python -m pytest tests\test_audit_epex_lab_locked_holdout_queue_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_plan_epex_lab_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `67 passed, 1 skipped`.

Latest focused validation after adding locked artifact hash checks:

```powershell
python -m pytest tests\test_audit_epex_lab_locked_holdout_queue_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_plan_epex_lab_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `85 passed, 1 skipped`.

Latest validation after adding `--plan-glob` and artifact-invalid CLI exit
checks:

```powershell
python -m pytest tests\test_audit_epex_lab_locked_holdout_queue_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_plan_epex_lab_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `87 passed, 1 skipped`.

Current local queue audit command:

```powershell
python scripts\audit_epex_lab_locked_holdout_queue.py --plan-glob ".planning\phases\14-lt-audit-remediation\locked_holdout_plan_*.json" --as-of-utc 2026-07-09T00:00:00Z --search-root output\phase14 --output output\phase14\locked_holdout_queue_audit_20260709.json
```

Result:

- `status=WAITING_FOR_FUTURE_HOLDOUT_WINDOWS`
- `plan_count=2`
- `future_window_count=2`
- `active_window_count=0`
- `spot_refresh_due_count=0`
- `invalid_plan_count=0`
- `policy_invalid_plan_count=0`
- `artifact_invalid_plan_count=0`
- T057 artifact checks all true for baseline CSV, adjusted CSV, lab manifest,
  and selection summary hash binding.
- T061 artifact checks all true for baseline CSV, adjusted CSV, lab manifest,
  and selection summary hash binding.
- T057 plan SHA:
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`
- T061 plan SHA:
  `29a633cf56279eae817cd6c63872a476cc2c10b187f08c3952f73cdad76db135`

Operational conclusion:

- On `2026-07-09T00:00:00Z`, neither T057 nor T061 should be run as a
  completed holdout.
- The next action for both plans is `wait_without_retuning_candidate`.
- Promotion remains NO-GO.

## 2026-07-09 Energy Charts Pre-Window Guard

Hardened the Energy Charts locked-holdout wrapper so it does not fetch spot data
before the locked holdout window is complete.

Changed:

- `scripts/run_energy_charts_epex_locked_holdout.py`
  - added optional `--as-of-utc` for deterministic operator/pre-window checks;
  - computes `latest_required_holdout_utc = holdout_end_utc - 1h`;
  - returns `LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE` before fetching spot when
    `as_of_utc <= latest_required_holdout_utc`;
  - writes a run summary with `spot_fetch_ran=false`,
    `locked_holdout_ran=false`, and `holdout_pass=false`.
- `scripts/epex_lab_locked_holdout_policy.py`
  - routes `LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE` as
    `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, not as a model failure.
- Tests updated:
  - `tests/test_run_energy_charts_epex_locked_holdout_script.py`
  - `tests/test_epex_lab_locked_holdout_policy.py`

Validation:

```powershell
python -m pytest tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result: `48 passed`.

```powershell
python -m pytest tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_audit_epex_lab_locked_holdout_queue_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `78 passed, 1 skipped`.

Current local pre-window guard command:

```powershell
python scripts\run_energy_charts_epex_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --output-dir output\phase14\t057_locked_t056_future_holdout\energy_charts_pre_window_guard_20260709 --as-of-utc 2026-07-09T00:00:00Z
```

Expected exit: `1`.

Observed:

- `status=LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE`
- `spot_fetch_ran=false`
- `locked_holdout_ran=false`
- `holdout_pass=false`
- `latest_required_holdout_utc=2026-07-23T23:00:00Z`
- next action:
  `Wait until the locked holdout window is complete, then refresh Energy Charts spot.`

Operational conclusion:

- Before the locked window is complete, the wrapper now produces only a
  read-only status artifact and avoids unnecessary spot fetch artifacts.
- Promotion remains NO-GO.

## 2026-07-09 EPEX Shape Explainability Diagnostic

Added lab-only no-OMPEX diagnostic:

`scripts/explain_epex_shape_lab_adjustment.py`

Purpose:

- reconstruct raw EPEX shape-lab component contributions from
  `epex_shape_templates.csv` and the lab manifest config;
- compare raw component totals with the final adjusted-minus-baseline delta;
- summarize deltas by bucket, component, month, and hour;
- verify monthly BASE and EEX PEAK delta conservation;
- keep the output explicitly non-promotional.

Real T056/t005 diagnostic generated under ignored output:

`output/phase14/t056_postval_final_micro/t005_diagnostics/shape_explainability_20260709/shape_explainability_summary.json`

Result:

- `status=DIAGNOSTIC_PASS`
- `monthly_base_delta_conserved=true`
- `monthly_peak_delta_conserved=true`
- max monthly BASE mean absolute delta:
  `9.555854647901084e-08`
- max monthly PEAK mean absolute delta:
  `8.333333436638669e-08`
- no OMPEX usage flags are all false.

Interpretation:

- final actual deltas are small and constraint-preserving;
- raw template contributions are materially larger than final projected
  deltas, so projection/capping/floor mechanics are compressing the raw
  component signal and should be reviewed before further tuning.
- enriched compression fields now report:
  - mean absolute raw delta `7.2446193711283255` versus final actual delta
    `0.606012689469531`;
  - raw-to-actual absolute ratio `11.954567118833522`;
  - projection-residual-to-raw absolute ratio `0.9252234998202383`;
  - most compressed bucket `night_00_05` with ratio
    `0.9563567714561076`;
  - all 9 diagnostic buckets have compression ratio at or above `0.75`.
- stage decomposition now separates projection, max-delta cap, and floor guard:
  - projection constraint residual `1.6153745008296028e-14`;
  - max raw delta `35.81471201388889`;
  - max projected delta `31.6464827773043`;
  - max capped/final reconstructed delta `2.75`;
  - cap scale `0.08689749250656693`;
  - floor guard scale `1.0`;
  - mean absolute projection loss `3.6112390551929443`;
  - mean absolute cap loss `6.367867353601689`;
  - mean absolute floor-guard loss `0.0`;
  - mean absolute unexplained delta `2.480501238010863e-07`.

Validation:

```powershell
pytest tests\test_explain_epex_shape_lab_adjustment_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

```powershell
pytest tests\test_backtest_epex_shape_lab_against_spot_script.py tests\test_audit_epex_shape_lab_governance_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `21 passed, 1 skipped`.

Latest focused validation after adding compression ratios:

```powershell
pytest tests\test_explain_epex_shape_lab_adjustment_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

Latest focused validation after adding stage decomposition:

```powershell
pytest tests\test_explain_epex_shape_lab_adjustment_script.py tests\test_backtest_epex_shape_lab_against_spot_script.py tests\test_audit_epex_shape_lab_governance_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `23 passed, 1 skipped`.

Promotion remains NO-GO pending T057 full coverage and production-chain
evidence.

## 2026-07-09 T060 Cap Decompression Pre-Registration

Added pre-registered no-OMPEX T060 planning files:

- `.planning/phases/14-lt-audit-remediation/t060_epex_only_cap_decompression_grid.json`
- `.planning/phases/14-lt-audit-remediation/t060_epex_only_cap_decompression_delta_grid.json`
- `.planning/phases/14-lt-audit-remediation/t060_epex_only_cap_decompression_thresholds.json`
- `.planning/phases/14-lt-audit-remediation/t060_epex_only_cap_decompression_scoring.json`

Purpose:

- test whether slight cap decompression improves useful weak-bucket shape after
  the explainability diagnostic showed the `2.75` cap is the main compression
  stage;
- keep intensities near T056/t005 instead of broad retuning;
- require per-trial no-OMPEX shape explainability.

Planner change:

- `scripts/plan_epex_shape_lab_sweep.py` now writes an
  `explain_adjustment_no_ompex` command for every trial, pointing to
  `scripts/explain_epex_shape_lab_adjustment.py`.
- `selection_basis` includes shape explainability with
  projection/cap/floor decomposition.

Generated local plan under ignored output:

`output/phase14/t060_epex_only_cap_decompression_plan.json`

Plan facts:

- `plan_id=t060_epex_only_cap_decompression`
- `trial_count=16`
- `benchmark_policy=pre_registered_independent_no_ompex`
- baseline candidate SHA:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- EPEX spot parquet SHA:
  `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`
- cap grid: `[2.75, 3.0, 3.25, 3.5]`
- low-tail grid: `[0.2, 0.25]`
- night grid: `[0.5, 0.55]`
- fixed weekend `0.75`, peak-subshape `0.89`, evening recovery `0.05`,
  ramp `0.0`.

T060 is a separate lab line. It does not change frozen T057 and is not
promotion evidence.

T060 execution results:

- `scripts/execute_epex_shape_lab_sweep.py` now runs
  `explain_epex_shape_lab_adjustment.py` for every executed trial and writes
  `explainability_count` plus cap/projection/floor metrics into the ranking.
- Full T060 sweep:
  `output/phase14/t060_epex_only_cap_decompression_summary_full.json`
- `trial_count_executed=16`
- `eligible_count=10`
- `explainability_count=16`
- best independent-shape trial:
  `t003_w075_l02_p089_e005_n05_r00_d325`
- no-OMPEX spot-backtest selection:
  `output/phase14/t060_epex_only_cap_decompression_selection_full/spot_backtest_selection_summary.json`
- `replacement_verdict.status=WEAK_BUCKET_AND_CORE_METRICS_BEAT_INCUMBENT`
- `replace_incumbent=true` in lab diagnostics only.
- selected spot-backtest trial:
  `t007_w075_l02_p089_e005_n055_r00_d325`
- selected adjusted CSV SHA:
  `0a0fe8ce8c12bfeb64ac517ef60ac4d2850fbd1d13255c823c213c94c98391a6`
- selected metrics:
  - overall `0.5362165721168545`
  - post-valuation `0.3526155364023289`
  - evening `0.5598005946763284`
  - solar-tail `0.5456001492329747`
  - weekend `0.3879656335001635`
  - night `0.19742555594807984`
  - ramp `0.06516701458031711`
- incumbent T056/t005 post-valuation remains `0.3049947368951571`, so T060
  is a genuine lab challenger, but it still requires future holdout and full
  production-chain evidence before promotion discussion.

## 2026-07-09 T057 Energy Charts Fail-Closed Spot Refresh

Added helper:

`scripts/fetch_energy_charts_epex_spot_hourly.py`

Purpose:

- fetch Energy Charts EPEX price data for a requested UTC window;
- aggregate only observed returned timestamps to hourly `price_eur_mwh`;
- write a T057-compatible parquet only when the requested window is fully
  covered, unless `--allow-partial` is explicitly used for diagnostics;
- avoid the generic 15-minute loader's forward-fill behavior for future spot
  holdout windows.

Operational finding:

- A manual refresh on 2026-07-09 had appeared to create spot through
  `2026-07-10T23:00:00Z`.
- The new fail-closed raw API fetch for `2026-07-09` to `2026-07-11` found only
  `22/48` observed hours, with `spot_max_utc=2026-07-09T21:00:00Z`.
- The script exited `1` by design with `status=PARTIAL_COVERAGE` and wrote no
  parquet.
- The manually generated local 2026-07-09 parquets were removed from ignored
  `output/phase14/t057_locked_t056_future_holdout/epex_spot_refresh_20260709`
  to avoid selecting a potentially forward-filled candidate.

Current fail-closed discovery:

```powershell
python scripts\discover_epex_spot_parquet_candidates.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --search-root output\phase14 --output-json output\phase14\t057_locked_t056_future_holdout\spot_parquet_discovery_20260709_failclosed.json --max-candidates 10
```

Result:

- `candidate_count=1`
- best candidate is still
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- `observed_holdout_hours=0`
- `missing_holdout_hours=336`
- `spot_max_utc=2026-07-08T23:00:00Z`

Validation:

```powershell
pytest tests\test_fetch_energy_charts_epex_spot_hourly_script.py
```

Result: `4 passed`.

```powershell
pytest tests\test_fetch_energy_charts_epex_spot_hourly_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_check_epex_lab_locked_holdout_coverage_script.py
```

Result: `28 passed`.

Promotion status remains NO-GO. Next correct action is to refresh spot with the
fail-closed helper after Energy Charts publishes the full T057 window, then run
the locked T057 runner with the frozen plan SHA.

## 2026-07-09 T057 Energy Charts Locked Runner Wrapper

Added operator wrapper:

`scripts/run_energy_charts_epex_locked_holdout.py`

Purpose:

- verify the frozen T057 plan SHA before any network fetch;
- fetch the plan's full `holdout_start_utc` to `holdout_end_utc` from Energy
  Charts with the fail-closed observed-hour helper;
- write no spot parquet unless every expected hour is covered;
- run `scripts/run_epex_lab_locked_holdout.py` only after full spot coverage;
- persist `energy_charts_locked_holdout_run_summary.json` for WAITING and PASS
  states.

The spot fetch helper was also hardened to convert UTC bounds into Energy
Charts `YYYY-MM-DD` API parameters and to persist `SPOT_FETCH_ERROR` instead
of raising a traceback when the future full window is not yet published.

Current operator command:

```powershell
python scripts\run_energy_charts_epex_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --output-dir output\phase14\t057_locked_t056_future_holdout\energy_charts_locked_runner_20260709 --bzn CH
```

Current result:

- exit `1` by design;
- `status=LOCKED_HOLDOUT_SPOT_WAITING`;
- `spot_fetch.status=SPOT_FETCH_ERROR`;
- Energy Charts request used `start=2026-07-10`, `end=2026-07-24`;
- API returned 404 because the full future window is not published;
- `expected_hour_count=336`, `observed_hour_count=0`,
  `missing_hour_count=336`;
- `locked_holdout_ran=false`;
- no spot parquet written.

Validation:

```powershell
pytest tests\test_run_energy_charts_epex_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `4 passed`.

```powershell
pytest tests\test_fetch_energy_charts_epex_spot_hourly_script.py tests\test_run_energy_charts_epex_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `10 passed`.

```powershell
pytest tests\test_run_epex_lab_locked_holdout_script.py tests\test_check_epex_lab_locked_holdout_coverage_script.py -q -p no:cacheprovider
```

Result: `25 passed`.

Operational next action remains unchanged: rerun this wrapper after the full
T057 spot window is published. Promotion remains NO-GO.

## 2026-07-09 T057 Wrapper Evidence Policy Routing

Promotion/future-approval checks now understand the one-command Energy Charts
wrapper summary.

Changes:

- `scripts/epex_lab_locked_holdout_policy.py` recognizes
  `energy_charts_epex_locked_holdout_run.v1`.
- `scripts/run_energy_charts_epex_locked_holdout.py` emits
  `locked_plan_identity`, `benchmark_policy`, and no-OMPEX flags.
- Wrapper `LOCKED_HOLDOUT_SPOT_WAITING` routes to
  `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.
- Wrapper PASS is accepted only if it links to a hash-bound inner
  `epex_lab_locked_holdout_run.v1` summary that passes the existing locked
  holdout policy.

Current real wrapper policy check:

- input:
  `output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260709/energy_charts_locked_holdout_run_summary.json`
- `status=NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`
- `pass=false`
- `operator_wrapper_status=LOCKED_HOLDOUT_SPOT_WAITING`
- `spot_fetch_summary_matches_embedded=true`
- `plan_identity_matches_plan_json=true`

Validation:

```powershell
pytest tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `15 passed`.

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result: `26 passed`.

Promotion status remains NO-GO.

## 2026-07-09 Adjusted Production Chain Holdout Policy Rebinding

The adjusted production-chain builder now rejects stale embedded locked-holdout
policy metadata.

Change:

- `scripts/build_epex_lab_adjusted_production_chain.py` recomputes
  `locked_holdout_policy` from the manifest's hash-bound
  `locked_holdout_summary`.
- It raises `locked_holdout_policy_bound` if the embedded policy object differs
  from the recomputed policy.
- This prevents export/selected/capstone artifacts from carrying stale or
  hand-edited holdout policy metadata.

Validation:

```powershell
pytest tests\test_build_epex_lab_adjusted_production_chain_script.py tests\test_build_epex_lab_adjusted_production_manifest_script.py -q -p no:cacheprovider
```

Result: `25 passed`.

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `30 passed`.

Promotion status remains NO-GO until T057 and the full production chain pass.

## 2026-07-09 Expert Read-Only Audits

Two read-only expert audits were launched after the T057 Energy Charts wrapper
and production-chain hardening work.

Consensus:

- Production remains NO-GO.
- Primary blocker is still full T057 observed EPEX spot coverage for
  `2026-07-10T00:00:00Z` through `2026-07-24T00:00:00Z`.
- Once coverage is complete, rerun the Energy Charts locked-holdout wrapper,
  then rebuild adjusted production/export/selected/capstone evidence only if
  T057 passes.
- OMPEX remains advisory only and must not enter model inputs, lambda
  selection, holdout scoring, or promotion gates.
- Shape improvement work should focus on explainability diagnostics and
  EPEX-only hypotheses, especially solar-tail/weekend/night/ramp contributions
  and post-valuation tradeoffs.

## 2026-07-09 Promotion Readiness Recommended Commands

Promotion readiness now mirrors future approval routing for T057 coverage
waiting states.

Change:

- `scripts/check_epex_lab_promotion_readiness.py` emits
  `recommended_commands`.
- For `production_blocking_stage=locked_holdout_coverage`, it recommends
  `run_energy_charts_locked_holdout` first and keeps `run_locked_holdout` as a
  manual fallback.
- Other blocking stages keep an empty command map.

Validation:

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `42 passed`.

```powershell
pytest tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `17 passed, 1 skipped`.

Promotion status remains NO-GO.

## 2026-07-09 Future Approval Recommended Commands

Future approval routing now recommends the fail-closed Energy Charts wrapper
when T057 is blocked on spot coverage.

Change:

- `scripts/audit_epex_lab_future_approval_path.py` emits
  `recommended_commands.run_energy_charts_locked_holdout` for
  `blocking_stage=locked_holdout_coverage`.
- The existing `recommended_commands.run_locked_holdout` remains as a fallback
  for a separately approved fresh future spot parquet.
- `scripts/epex_lab_locked_holdout_policy.py` exposes wrapper `bzn` in policy
  output so the recommended command can preserve the source zone.

Validation:

```powershell
pytest tests\test_audit_epex_lab_future_approval_path_script.py tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `27 passed`.

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result: `14 passed`.

Promotion status remains NO-GO.

## 2026-07-09 Expert Audit + Discovery Coverage Follow-Up

Read-only expert agents were launched for the next Phase 14 steps:

- Tesla audited promotion readiness and T057 evidence routing.
- Cicero audited model-quality sequencing and OMPEX benchmark discipline.

Consensus:

- T056/t005 remains frozen for T057.
- Promotion remains NO-GO.
- The immediate blocker is still future EPEX spot coverage for the locked
  T057 window plus the approved production/export/selected/capstone evidence
  chain.
- T058 is research-only and does not replace T056/t005.
- OMPEX is useful for desk review but must remain outside model input,
  selection, backtest truth, and promotion gates.

Discovery helper follow-up:

- `scripts/discover_epex_spot_parquet_candidates.py` now reports per-candidate
  holdout coverage metrics:
  `expected_holdout_hours`, `observed_holdout_hours`,
  `missing_holdout_hours`, `first_missing_holdout_utc`,
  `last_missing_holdout_utc`, and `full_window_covered`.
- The current local discovery still finds one hourly candidate:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`.
- That candidate has `spot_max_utc=2026-07-08T23:00:00Z` and covers
  `0/336` locked holdout hours:
  `missing_holdout_hours=336`, `first_missing_holdout_utc=2026-07-10T00:00:00Z`,
  `last_missing_holdout_utc=2026-07-23T23:00:00Z`,
  `full_window_covered=false`.

Validation:

```powershell
python -m pytest tests/test_discover_epex_spot_parquet_candidates_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `42 passed`.

Next command when fresh spot data is available:

```powershell
python scripts\discover_epex_spot_parquet_candidates.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --search-root output\phase14 --output-json output\phase14\t057_locked_t056_future_holdout\spot_parquet_discovery_<YYYYMMDD>.json --max-candidates 5
```

Only after a candidate reports full coverage through
`2026-07-23T23:00:00Z`, run the locked holdout runner with the unchanged T057
plan SHA.

## 2026-07-09 T059 EPEX-Only Low-Tail/Cap/Night Interaction Sweep

T059 was run as a separate no-OMPEX research line. It does not alter frozen
T056/t005 or the locked T057 promotion path.

Durable parameter files:

- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_grid.json`
- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_delta_grid.json`
- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_thresholds.json`
- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_scoring.json`

Generated ignored artifacts:

- plan:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_plan.json`
  SHA256 `56405a821f4c3bd91afce975c47beb0bf810736929fd6ccb27fd44dd852fb545`
- execution summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_summary_full.json`
  SHA256 `79a17516deea1be9362a9a6e56497b1a9ec69715d900bfafd0fc53bb046900e3`
- spot-backtest orchestration summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_spot_backtests_summary_full.json`
  SHA256 `73aea816694ab8823e8523d6de74399bfcfbfda2d353db1e8d9c1bb7dc80de55`
- selection summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_selection_full/spot_backtest_selection_summary.json`
  SHA256 `748b8c1103565e0fe6615cff6c1dbb8d82c9fce93f1975b7ed48dbf67e199fb9`

Verdict:

- `trial_count=36`, `trial_count_executed=36`, `eligible_count=36`,
  `strict_pass_count=36`.
- `replacement_candidate_count=0`.
- `replacement_verdict.replace_incumbent=false`.
- Best weak-bucket trial is
  `t009_w075_l01_p089_e005_n055_r00_d275`.
- It improves weak buckets versus incumbent
  (`overall=0.4651499923241654`, `solar_tail=0.47194371091304294`,
  `night=0.17918873802407917`, `ramp=0.06397807990619993`) but loses
  post-valuation (`0.29827066207436914` versus T056/t005
  `0.3049947368951571`).

Operational conclusion:

- T059 confirms the tradeoff: lowering low-tail/cap improves weak historical
  buckets but does not beat T056/t005 on the core post-valuation metric.
- T056/t005 remains frozen for T057.
- Next model-quality work should not continue broad low-tail lowering unless a
  new no-OMPEX hypothesis specifically protects post-valuation performance.

## 2026-07-09 EPEX Sweep Spot Backtest Path Hardening

During the T059 full spot-backtest run, the runner failed once with relative
`output/...` paths on Windows/UNC path handling and then succeeded when the
same output paths were passed as absolute paths. This was orchestration
fragility only; it did not change T059 inputs or verdict.

Follow-up:

- `scripts/run_epex_shape_lab_sweep_spot_backtests.py` now resolves CLI paths
  to absolute paths before running trials.
- Relative paths recorded inside plans and sweep summaries are resolved
  against the repo root.
- A regression test verifies that relative output paths from a changed cwd are
  converted to absolute backtest output paths.

Validation:

```powershell
python -m pytest tests/test_run_epex_shape_lab_sweep_spot_backtests_script.py tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `26 passed, 1 skipped`.

## 2026-07-09 T059 Parameter Sensitivity Diagnostic

Added no-OMPEX lab diagnostic:

`scripts/analyze_epex_shape_lab_sweep_sensitivity.py`

Purpose:

- join a pre-registered sweep plan with the spot-backtest ranking;
- quantify parameter response for weak-bucket and post-valuation metrics;
- keep the result read-only, no-OMPEX, non-promotional.

T059 command:

```powershell
python scripts\analyze_epex_shape_lab_sweep_sensitivity.py --plan-json output\phase14\t059_epex_only_lowtail_cap_night_interactions_plan.json --selection-summary output\phase14\t059_epex_only_lowtail_cap_night_interactions_selection_full\spot_backtest_selection_summary.json --output-dir output\phase14\t059_epex_only_lowtail_cap_night_interactions_sensitivity
```

Generated ignored summary:

`output/phase14/t059_epex_only_lowtail_cap_night_interactions_sensitivity/sweep_sensitivity_summary.json`

SHA256:

`ea7207b0601dd4f41aa4856122673fa15ed51f82f8435b972e83f796e011b28f`

Key result:

- `trial_count=36`
- `strict_pass_count=36`
- `weak_bucket_candidate_count=5`
- `replacement_candidate_count=0`
- `next_hypothesis_hint=protect_post_valuation_before_expanding_weak_bucket_gains`
- best overall/weak-bucket trial:
  `t009_w075_l01_p089_e005_n055_r00_d275`
- best post-valuation trial:
  `t036_w075_l025_p089_e005_n055_r00_d275`, the incumbent-like parameter
  neighborhood.

Interpretation:

- Lower `low_tail_intensity` improves weak historical buckets but hurts the
  post-valuation metric.
- `low_tail=0.10` max overall is `0.4651499923241654`, but max
  post-valuation is only `0.2982706620743691`.
- `low_tail=0.25` max overall is `0.4506842423821014`, but max
  post-valuation is `0.3049947368951571`, matching the incumbent.
- Next EPEX-only research should target post-valuation preservation first,
  not broader low-tail lowering.

Validation:

```powershell
python -m pytest tests/test_analyze_epex_shape_lab_sweep_sensitivity_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

## 2026-07-09 Explicit Post-Valuation Replacement Guard

The spot-backtest selection summary now exposes the replacement guard directly:

- per trial: `post_valuation_gate_pass` and `core_metric_gate_pass`
- summary-level: `replacement_guard`

For T059, the enriched regenerated selection summary is:

`output/phase14/t059_epex_only_lowtail_cap_night_interactions_selection_full/spot_backtest_selection_summary.json`

SHA256:

`dcf218c6f58f853aa02674f580748da24923bbba8a9f2da1c8c0f7d7f7e94f9b`

Verdict remains unchanged:

- `replacement_guard.status=CORE_METRIC_DEGRADATION`
- `replacement_guard.pass=false`
- degraded metric:
  `post_valuation_mae_improvement_eur_mwh`
- selected weak-bucket trial
  `t009_w075_l01_p089_e005_n055_r00_d275` remains non-replacement.

The T059 sensitivity summary was regenerated after this enrichment:

`output/phase14/t059_epex_only_lowtail_cap_night_interactions_sensitivity/sweep_sensitivity_summary.json`

SHA256:

`ab442633f4029d35a21973695055fe5de2ebcf1604f259783537332f98d64e42`

Validation:

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_spot_backtests_script.py -q -p no:cacheprovider
```

Result: `5 passed`.

## 2026-07-09 Sweep Selection Path Hardening

The EPEX spot-backtest selection summarizer now resolves paths before summary
generation:

- CLI paths are resolved against cwd.
- Relative `ranking_csv` recorded in sweep summaries is resolved against the
  repo root.
- Generated summaries record resolved paths.

T059 was regenerated after this path hardening. These supersede the prior
guard-only SHA values:

- selection summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_selection_full/spot_backtest_selection_summary.json`
  SHA256 `ea5c0401f04798d3fa665eda8cf0b9831f819811fd205b0c3cdb7625e203b073`
- sensitivity summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_sensitivity/sweep_sensitivity_summary.json`
  SHA256 `9e0e57999c7f03c29f8eb98585f656c0a5419bd7f438ed9739d42ceff82cf89b`

Verdict unchanged:

- `replacement_guard.status=CORE_METRIC_DEGRADATION`
- degraded metric:
  `post_valuation_mae_improvement_eur_mwh`
- T056/t005 remains frozen for T057.

Validation:

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_spot_backtests_script.py -q -p no:cacheprovider
```

Result: `6 passed`.

## 2026-07-09 T057 Spot Discovery Rejection Counters

T057 spot discovery now reports scan/rejection diagnostics:

- `scanned_file_count`
- `rejected_file_count`
- `spot_like_rejected_file_count`
- `rejection_reason_counts`

Current refreshed discovery:

`output/phase14/t057_locked_t056_future_holdout/spot_parquet_discovery_20260709_latest.json`

Result:

- `scanned_file_count=22`
- `candidate_count=1`
- `rejected_file_count=21`
- `spot_like_rejected_file_count=1`
- `rejection_reason_counts={"index_not_datetime":2,"missing_price_column_or_empty":18,"non_hourly_grid":1}`
- best candidate remains
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- `spot_max_utc=2026-07-08T23:00:00Z`
- T057 coverage remains `0/336h`, `full_window_covered=false`

Validation:

```powershell
python -m pytest tests/test_discover_epex_spot_parquet_candidates_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `42 passed`.

## 2026-07-09 Expert Audit and T058 Lab Plan

Expert read-only audits confirmed the current split of responsibilities:

- T056/t005 stays frozen for T057.
- Promotion remains NO-GO until T057 has full future spot coverage and a
  locked holdout PASS.
- OMPEX remains advisory only and must not feed model input, selection,
  backtest, or production gates.
- The next model-improvement work should be a separate EPEX-only lab line
  focused on solar-tail, midday, weekend, night, and ramp behavior.

Canonical T056/t005 diagnostics were generated locally under ignored output:

`output/phase14/t056_postval_final_micro/t005_diagnostics/canonical_ch_hfc_png_20260709`

Command:

```powershell
python scripts\plot_ch_hfc_diagnostics.py --csv output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\candidate_epex_shape_lab_adjusted.csv --forwards output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\diagnostic_forwards_history_rebuilt_20260708.parquet --output-dir output\phase14\t056_postval_final_micro\t005_diagnostics\canonical_ch_hfc_png_20260709 --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv
```

Result:

- Exit `0`.
- EEX residual max absolute error:
  `2.717391254236645e-07` EUR/MWh.
- Worst month-to-month mean moves include 2028-04 at `-38.914134` EUR/MWh and
  2027-04 at `-36.146815` EUR/MWh.
- Applied-delta boundary jumps remain small, about `0.879` EUR/MWh max
  absolute jump versus the no-smoothing baseline.

A separate lab-only T058 plan was pre-registered under ignored output:

`output/phase14/t058_epex_only_shape_micro_plan.json`

Plan SHA256:

`7818437211dc1b66c1645ffaf943ecbdfe1fe334ae0a51ac8910f94a5426e7d0`

Plan facts:

- `plan_id=t058_epex_only_shape_micro`
- `trial_count=162`
- `activation_status=lab_only`
- `production_approved=false`
- `benchmark_policy=pre_registered_independent_no_ompex`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`
- baseline candidate SHA:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- EPEX spot SHA:
  `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`
- scoring policy includes `midday_weight=1.25`, `solar_tail_weight=1.5`,
  and `weekend_weight=1.25`.
- plan selection thresholds are restricted to executor-enforceable checks:
  spot age, spot coverage, ramp p99 increase, and minimum adjusted price.
  Realized MAE/fold-count decisions remain in the spot-backtest summarizer,
  not in the sweep pre-filter.

The first 10 T058 trials were executed as a controlled partial sweep:

- sweep summary:
  `output/phase14/t058_epex_only_shape_micro_summary_first10.json`
- benchmark policy: `executed_independent_no_ompex`
- `trial_count_executed=10`
- `eligible_count=10`
- best independent-shape trial:
  `t002_w065_l015_p087_e005_n045_r00_d275`
- best independent-shape trial metrics include shape score
  `4.089226396580266`, midday mean delta `-0.9821475722222222`,
  solar-tail mean delta `-1.0645568976773383`, weekend mean delta
  `-0.3540212641799299`, ramp p99 increase `0.8039680399999583`, and min
  adjusted price `-3.668281`.

The top three first10 trials were spot-backtested under no-OMPEX lab-only
evidence and summarized against the frozen T056/t005 incumbent:

- backtest root:
  `output/phase14/t058_epex_only_shape_micro_spot_backtests_first10`
- selection summary:
  `output/phase14/t058_epex_only_shape_micro_selection_first10/spot_backtest_selection_summary.json`
- replacement verdict:
  `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- `replace_incumbent=false`
- best weak-bucket trial:
  `t004_w065_l015_p087_e005_n055_r00_d275`
- best weak-bucket metrics: overall improvement `0.428104291871567`,
  night improvement `0.17009783026880573`, ramp improvement
  `0.05713664911100517`, solar-tail improvement `0.4270889440372494`,
  weekend improvement `0.2945464419537373`, post-valuation improvement
  `0.3033020021281363`.
- incumbent T056/t005 remains stronger on overall, evening, solar-tail,
  weekend, and post-valuation improvement.

A targeted12 T058 subset was then executed around the T056/t005 neighborhood:

- subset plan:
  `output/phase14/t058_epex_only_shape_micro_targeted12_plan.json`
- subset plan SHA256:
  `9d38fc46232d7f669ffbbb8ddf8576a01c58281ecd3d228c009208213bc8e00c`
- parent T058 plan SHA256:
  `7818437211dc1b66c1645ffaf943ecbdfe1fe334ae0a51ac8910f94a5426e7d0`
- subset summary:
  `output/phase14/t058_epex_only_shape_micro_targeted12_summary.json`
- `trial_count_executed=12`
- `eligible_count=12`
- best independent-shape trial:
  `t064_w075_l015_p089_e005_n055_r00_d275`
- best independent-shape score: `4.308883116813839`

The top targeted12 signals were spot-backtested under no-OMPEX lab-only
evidence:

- backtest root:
  `output/phase14/t058_epex_only_shape_micro_targeted12_spot_backtests`
- selection summary:
  `output/phase14/t058_epex_only_shape_micro_targeted12_selection/spot_backtest_selection_summary.json`
- strict pass count: `5`
- best weak-bucket trial:
  `t064_w075_l015_p089_e005_n055_r00_d275`
- best weak-bucket adjusted CSV SHA:
  `9255a81e770184a4192f7ede1d3051c5283b802ade1f1b58d06d0eca3c485e34`
- best weak-bucket metrics: overall improvement `0.4599653156253434`,
  evening `0.48163132451829344`, night `0.17317636817961332`, ramp
  `0.06009215161278314`, solar-tail `0.46799710365738634`, weekend
  `0.329903771232736`, post-valuation `0.3007021210797465`.
- incumbent T056/t005 post-valuation remains stronger at
  `0.3049947368951571`, so the replacement verdict is
  `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS` and
  `replace_incumbent=false`.
- `t082_w075_l025_p089_e005_n055_r00_d275` reproduces the T056/t005 adjusted
  CSV SHA `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`,
  confirming the targeted subset includes the incumbent-equivalent point.

Operational next steps:

1. Do not retune T056/t005 before T057.
2. After full future spot coverage exists for `2026-07-10T00:00:00Z` to
   `2026-07-24T00:00:00Z`, rerun T057 with the locked plan SHA
   `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.
3. If T057 passes, build the real production/export/selected/capstone chain.
4. Execute T058 only as a separate EPEX-only lab branch; it is not part of the
   frozen T057 approval path.

## 2026-07-09 T057 Coverage Lag Diagnostics

T057 coverage preflight now includes informational spot-lag fields to make the
WAITING state easier to interpret.

- `scripts/check_epex_lab_locked_holdout_coverage.py` emits
  `latest_required_holdout_utc`, `spot_hours_until_holdout_start`, and
  `spot_hours_until_latest_required_holdout`.
- These fields are informational only and do not alter
  `ready_to_run_backtest` or `blocking_checks`.

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `38 passed`.

Current T057 recheck command:

```powershell
python scripts\check_epex_lab_locked_holdout_coverage.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output output\phase14\t057_locked_t056_future_holdout\coverage_status_20260709_lag_recheck.json
```

Result remains expected NO-GO/waiting:

- `status=WAITING_FOR_FULL_SPOT_COVERAGE`
- `observed_holdout_hours=0`
- `expected_holdout_hours=336`
- `missing_holdout_hours=336`
- `spot_max_utc=2026-07-08T23:00:00Z`
- `holdout_start_utc=2026-07-10T00:00:00Z`
- `latest_required_holdout_utc=2026-07-23T23:00:00Z`
- `spot_hours_until_holdout_start=25.0`
- `spot_hours_until_latest_required_holdout=360.0`
- `blocking_checks=["full_window_covered", "min_holdout_hours_met"]`

The locked T057 plan and candidate hashes remain unchanged.

## 2026-07-09 T057 Spot Parquet Discovery Helper

Added read-only helper:

`scripts/discover_epex_spot_parquet_candidates.py`

Purpose:

- scan one or more roots for EPEX spot parquet candidates;
- require exact hourly timestamps by default;
- rank candidates by `spot_max_utc`;
- write operator-ready coverage and locked-holdout runner commands bound to
  the locked plan SHA;
- never run the holdout and never approve production.

Validation:

```powershell
python -m pytest tests/test_discover_epex_spot_parquet_candidates_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `42 passed`.

Current local discovery command:

```powershell
python scripts\discover_epex_spot_parquet_candidates.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --search-root output\phase14 --output-json output\phase14\t057_locked_t056_future_holdout\spot_parquet_discovery_20260709.json --max-candidates 5
```

Result:

- `candidate_count=1`
- `require_hourly_grid=true`
- best candidate:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- best candidate `spot_max_utc=2026-07-08T23:00:00Z`
- best candidate `spot_hours_until_latest_required_holdout=360.0`
- 15-minute EPEX parquet is intentionally not returned by default; it requires
  explicit `--include-non-hourly-spot`.

## 2026-07-09 Promotion Readiness Routing Follow-Up

EPEX lab promotion readiness now emits machine-readable production blocking
route fields.

- `scripts/check_epex_lab_promotion_readiness.py` emits
  `missing_production_checks`, `failed_production_checks`,
  `production_blocking_stage`, and `next_required_step`.
- Routing distinguishes strict diagnostics, production evidence, missing T057,
  T057 coverage waiting, T057 input-invalid, T057 failure, production-check
  failures, and promotion-ready review.
- Tests cover production-evidence missing, missing T057, input-invalid T057,
  and promotion-ready routing.

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result: `14 passed`.

```powershell
python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `85 passed, 1 skipped`.

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `113 passed, 1 skipped`.

Current frozen T057 plan remains unchanged. Regenerated current T057 runner
still exits `1` with `WAITING_FOR_FULL_SPOT_COVERAGE`.

