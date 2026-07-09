# Session Handoff - 2026-07-09 - T057 Locked Holdout

## Scope

Phase 14 LT only. No CT code touched. No Power BI data files touched. OMPEX
remains benchmark/advisory only and is explicitly forbidden as model,
calibration, selection, backtest, or gate input.

Objective advanced: after T056 t005 became the current no-OMPEX lab replacement
candidate, freeze it before further tuning and pre-register a future holdout.

## Changed Files

Code/tests/docs:

- `scripts/plan_epex_lab_locked_holdout.py`
- `scripts/audit_epex_lab_locked_holdout.py`
- `tests/test_plan_epex_lab_locked_holdout_script.py`
- `tests/test_audit_epex_lab_locked_holdout_script.py`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t057_t056_asof20260709.json`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260709-T057-LOCKED-HOLDOUT.md`

## New Tools

`scripts/plan_epex_lab_locked_holdout.py`

- Builds a read-only, lab-only, no-OMPEX plan.
- Binds baseline CSV hash, adjusted CSV hash, selection summary hash, and lab
  manifest hash.
- Rejects a selection summary that is not replacement-approved, no-OMPEX, and
  bound to the exact adjusted CSV hash.
- Emits command templates for future backtest and holdout audit.

`scripts/audit_epex_lab_locked_holdout.py`

- Audits a completed spot backtest against the locked plan.
- Recomputes holdout metrics from
  `post_valuation_timestamp_residuals.csv` inside the pre-registered future
  window.
- Requires exact baseline/adjusted CSV hash binding, no-OMPEX flags,
  lab-only status, strict lab gate pass, minimum holdout hours, and
  non-degraded residual MAE.
- Does not approve production promotion.

`scripts/check_epex_lab_locked_holdout_coverage.py`

- Reads the locked holdout plan and a candidate future EPEX spot parquet.
- Verifies full hourly coverage for the pre-registered window before running
  the backtest.
- Reports missing hours, spot min/max, duplicate holdout rows, and whether it
  is ready to run the holdout backtest.

`scripts/run_epex_lab_locked_holdout.py`

- Preferred execution wrapper once future spot data is available.
- Writes `coverage_status.json` first.
- If coverage is incomplete, writes `locked_holdout_run_summary.json` and
  stops with `backtest_ran=false`, `audit_ran=false`.
- If coverage is complete, runs
  `scripts/backtest_epex_shape_lab_against_spot.py` through the Python API,
  then runs `scripts/audit_epex_lab_locked_holdout.py`.
- Never approves production promotion.

`scripts/audit_epex_lab_future_approval_path.py`

- Now accepts `--locked-holdout-summary`.
- The summary can be either a locked holdout runner summary or final holdout
  audit.
- A provided locked holdout blocks promotion unless it passed
  `LOCKED_HOLDOUT_PASS`.

`scripts/check_epex_lab_promotion_readiness.py`

- Now accepts `--locked-holdout-summary`.
- A complete adjusted production/export/selected/capstone bundle cannot be
  `PROMOTION_READY` unless the locked holdout has passed.
- Diagnostic-only readiness can still report strict diagnostics separately
  from production readiness.

## T057 Locked Plan

Plan file:

`.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t057_t056_asof20260709.json`

Plan sha256:

`f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`

Frozen candidate:

- plan id: `t057_locked_t056_future_holdout`
- frozen at: `2026-07-09T00:00:00Z`
- holdout window: `2026-07-10T00:00:00Z` to `2026-07-24T00:00:00Z`
- baseline CSV:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- baseline CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- adjusted CSV:
  `output/phase14/t056_postval_final_micro/t005_w075_l025_p089_e005_n055_r00/candidate_epex_shape_lab_adjusted.csv`
- adjusted CSV sha256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- selection summary:
  `output/phase14/t056_postval_final_micro_selection_summary/spot_backtest_selection_summary.json`
- selection summary sha256:
  `b2a319ac91eff51947387bc2a1dcc4784b2f5bf5536ea861f2e63ab9fc5cf10d`
- lab manifest sha256:
  `013a11ba0e6a0a2f32eeb78493e154731ab736542710bd5b31e148c37e7716bc`

Pass criteria:

- future backtest `benchmark_policy=rolling_origin_epex_spot_no_ompex_lab_only`;
- all OMPEX flags false;
- `strict_lab_gate_pass=true`;
- at least `300` hours in the planned holdout window;
- residual MAE improvement `>= 0.0 EUR/MWh`;
- exact baseline and adjusted CSV hashes.

## Commands Already Run

Plan generation:

```powershell
python scripts/plan_epex_lab_locked_holdout.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\candidate_epex_shape_lab_adjusted.csv --selection-summary output\phase14\t056_postval_final_micro_selection_summary\spot_backtest_selection_summary.json --lab-manifest output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\ab_lab_manifest.json --plan-id t057_locked_t056_future_holdout --frozen-at-utc 2026-07-09T00:00:00Z --holdout-start-utc 2026-07-10T00:00:00Z --holdout-end-utc 2026-07-24T00:00:00Z --valuation-timestamp-utc 2026-07-09T00:00:00Z --embargo-days 1 --eval-days 14 --min-holdout-hours 300 --min-residual-mae-improvement-eur-mwh 0.0 --output .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json
```

Initial targeted tests:

```powershell
python -m pytest tests/test_plan_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `34 passed, 1 skipped`.

Coverage check with the currently available 2026-07-08 spot parquet:

```powershell
python scripts/check_epex_lab_locked_holdout_coverage.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output output\phase14\t057_locked_t056_future_holdout\coverage_status_current_spot.json
```

Result:

- `status=WAITING_FOR_FULL_SPOT_COVERAGE`
- spot max `2026-07-08T23:00:00Z`
- observed holdout hours `0`
- expected holdout hours `336`
- first missing holdout hour `2026-07-10T00:00:00Z`

Runner check with the same incomplete spot parquet:

```powershell
python scripts/run_epex_lab_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t057_locked_t056_future_holdout\current_spot_runner
```

Result:

- `status=WAITING_FOR_FULL_SPOT_COVERAGE`
- `backtest_ran=false`
- `audit_ran=false`
- run summary:
  `output/phase14/t057_locked_t056_future_holdout/current_spot_runner/locked_holdout_run_summary.json`

Consolidated approval-path audit using the current locked holdout run summary:

```powershell
python scripts/audit_epex_lab_future_approval_path.py --readiness-json output\phase14\t056_postval_final_micro\t005_diagnostics\promotion_readiness\decision_with_staged_manifest.json --locked-holdout-summary output\phase14\t057_locked_t056_future_holdout\current_spot_runner\locked_holdout_run_summary.json --output output\phase14\t057_locked_t056_future_holdout\future_approval_path_with_holdout_current.json
```

Result:

- `status=NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `locked_holdout_policy.pass=false`
- `remaining_blockers` includes `locked_holdout_pass`,
  `adjusted_export_manifest`, `adjusted_selected_config`,
  `adjusted_capstone`, `adjusted_production_manifest_approved`, and
  `adjusted_production_manifest_run_identity_valid`.

Promotion readiness with the current locked holdout run summary:

```powershell
python scripts/check_epex_lab_promotion_readiness.py --lab-manifest output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\ab_lab_manifest.json --governance-audit output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\governance_audit\epex_shape_lab_governance_audit.json --independent-summary output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\independent_ab_comparison\ab_comparison_summary.json --product-summary output\phase14\t056_postval_final_micro\t005_diagnostics\product_normalization_with_policy\summary.json --powerbi-summary output\phase14\t056_postval_final_micro\t005_diagnostics\powerbi_strict\summary_metrics.csv --adjusted-production-manifest output\phase14\t056_postval_final_micro\t005_diagnostics\staged_adjusted_candidate_selection_guard\adjusted_production_manifest_no_go.json --locked-holdout-summary output\phase14\t057_locked_t056_future_holdout\current_spot_runner\locked_holdout_run_summary.json --output output\phase14\t057_locked_t056_future_holdout\promotion_readiness_with_locked_holdout_current.json
```

Result:

- command exits `1`, as expected for not approved;
- `strict_diagnostics_pass=true`;
- `production_chain_pass=false`;
- `locked_holdout_pass=FAIL`;
- `locked_holdout_policy.status=NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

## Next Execution When Future Spot Exists

After the refreshed EPEX spot parquet covers `2026-07-10T00:00:00Z` through
`2026-07-24T00:00:00Z`, run `scripts/run_epex_lab_locked_holdout.py` with a
fresh output dir. It will enforce coverage before backtest/audit. If manual
execution is needed, first run `scripts/check_epex_lab_locked_holdout_coverage.py`.
Only if it reports `READY_TO_RUN_HOLDOUT_BACKTEST`, use the plan's
`commands.run_future_backtest_template`, replacing:

- `<FUTURE_SPOT_PARQUET>`
- `<T057_HOLDOUT_OUTPUT_DIR>`

Then run the plan's `commands.audit_future_holdout_template`.

Do not edit the locked plan after the holdout window starts. If the future
window or criteria are wrong, create a new plan with a new id and document why.

## Current Status

T057 is pre-registered but not executed because the future spot window is not
complete yet. T056 remains diagnostic-pass but NO-GO production until both:

- locked holdout evidence is available and passes;
- real adjusted production/export/selected/capstone chain is approved.

## 2026-07-09 Follow-Up Hardening

The T057 holdout is now part of the adjusted production artifact contract, not
only a readiness sidecar.

Changed files:

- `scripts/build_epex_lab_adjusted_production_manifest.py`
  - added `--locked-holdout-summary`;
  - production approval requests now require a passing locked holdout run or
    audit summary;
  - the holdout policy requires read-only/non-promotional status, no OMPEX
    flags, and `LOCKED_HOLDOUT_PASS`;
  - diagnostic NO-GO manifests can still be built without T057.
- `scripts/build_epex_lab_adjusted_production_chain.py`
  - rejects approved adjusted production manifests without a bound passing
    holdout;
  - revalidates the holdout summary hash and policy;
  - propagates holdout path/hash/policy fields into export, selected artifact,
    and capstone.
- `scripts/check_epex_lab_promotion_readiness.py`
  - verifies that production/export/selected/capstone carry the same locked
    holdout hash.
- `scripts/audit_epex_lab_future_approval_path.py`
  - refuses a `PROMOTION_READY` readiness payload unless a passing locked
    holdout is provided.
- Tests updated:
  - `tests/test_build_epex_lab_adjusted_production_manifest_script.py`
  - `tests/test_build_epex_lab_adjusted_production_chain_script.py`
  - `tests/test_check_epex_lab_promotion_readiness_script.py`
  - `tests/test_audit_epex_lab_future_approval_path_script.py`

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `53 passed, 1 skipped`.

CLI help was also checked:

```powershell
python scripts/build_epex_lab_adjusted_production_manifest.py --help
python scripts/build_epex_lab_adjusted_production_chain.py --help
```

Operational impact:

- T057 remains NO-GO until the future spot window is complete.
- Once T057 passes, the passing run/audit summary must be supplied when
  building any approved adjusted production manifest.
- The production/export/selected/capstone chain must preserve the exact
  `locked_holdout_summary_sha256`.

Additional expert-audit hardening:

- `scripts/check_epex_lab_promotion_readiness.py` now requires SHA-strict
  holdout binding; the same holdout path with changed content fails.
- `scripts/audit_epex_lab_future_approval_path.py` now requires all expected
  production readiness checks to be present and passing, including
  `adjusted_production_manifest_locked_holdout_bound` and
  `locked_holdout_pass`.
- Future approval also compares the provided locked holdout sidecar SHA against
  the SHA reported by readiness binding checks and returns
  `NO_GO_LOCKED_HOLDOUT_HASH_MISMATCH` for unbound sidecars.
- `scripts/audit_epex_lab_locked_holdout.py` now emits top-level no-OMPEX flags
  so `epex_lab_locked_holdout_audit.v1` can be consumed by the same policy as
  runner summaries.

Validation after this hardening:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `59 passed, 1 skipped`.

Follow-up proof coverage added explicit tests that readiness rejects divergent
selected-artifact holdout SHA, divergent capstone holdout SHA, and
`locked_holdout_policy_pass=false` on an otherwise hash-bound production
manifest. Future approval also now has an explicit test for
`production_chain_pass=true` without a holdout even when `approved=false`.
The same targeted command reported `63 passed, 1 skipped`.

Future approval audit reporting now emits `required_production_checks` and adds
the next action `Regenerate readiness with the full required production-check
set before promotion review.` when checks are missing. Targeted validation:

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `37 passed, 1 skipped`. The current regenerated approval audit remains
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

The local promotion bundle path was also covered end-to-end: local bundle plus
readiness is audited as `NO_GO_PRODUCTION_CHAIN_INCOMPLETE` and still exposes
locked-holdout production checks as required. Validation:

```powershell
python -m pytest tests/test_build_epex_lab_promotion_bundle_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `38 passed, 1 skipped`.

Readiness now publishes `required_production_checks` directly. Future approval
uses the union of the readiness-declared checks and its own internal minimum
set, so a synthetic readiness payload cannot remove mandatory production
checks. Validation:

```powershell
python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `39 passed, 1 skipped`. Current regenerated readiness and future
approval audit remain `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

Future approval audit CLI now exits `0` only for `approved=true`; all NO-GO
states exit `1`. Validation:

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `41 passed, 1 skipped`. Running the current T057 command returns exit
`1` as expected because status remains
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

Expert-audit follow-up on 2026-07-09:

- `scripts/check_epex_lab_locked_holdout_coverage.py` now exits `1` unless the
  locked future window is ready to backtest.
- `scripts/run_epex_lab_locked_holdout.py` now exits `1` unless the final
  runner summary is `LOCKED_HOLDOUT_PASS` with `holdout_pass=true`.
- `scripts/audit_epex_lab_locked_holdout.py` now exits `1` unless the audit
  has `holdout_pass=true`.
- Tests now cover pass and non-pass CLI exit codes for all three T057 tools.

Validation:

```powershell
python -m pytest tests/test_run_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py -q -p no:cacheprovider
```

Result: `12 passed`.

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `62 passed, 1 skipped`.

Current real checks against the 2026-07-08 spot parquet:

- coverage CLI exits `1` with `WAITING_FOR_FULL_SPOT_COVERAGE`;
- runner CLI exits `1` with `WAITING_FOR_FULL_SPOT_COVERAGE`;
- both still write their JSON reports for auditability.

Planner follow-up:

- The locked T057 plan JSON was not rewritten; keep its registered hash intact.
- Future locked holdout plans generated by
  `scripts/plan_epex_lab_locked_holdout.py` now include
  `commands.run_locked_holdout_template`, pointing to the fail-closed wrapper
  `scripts/run_epex_lab_locked_holdout.py`.
- The separate backtest and audit templates remain in generated plans for
  traceability and manual inspection.

Validation:

```powershell
python -m pytest tests/test_plan_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py -q -p no:cacheprovider
```

Result: `14 passed`.

Context hygiene follow-up:

- `.planning/CONTEXT.md` now has a Phase 14 notice near the top: it is
  historical Phase 5bis context and not the active handoff.
- The 2026-06-18 residual-anchor external audit prompt is historical diagnostic
  context, not the current target architecture.
- The active Phase 14 target is the 2026-06-19 monthly solver reform plus the
  T056/T057 governance chain.
- T057 PASS permits only the next production packaging step; it is not
  automatic promotion.
- T057 FAIL or NO-GO requires a new pre-registered lineage instead of retuning
  T056 against the locked window.
- The `>= 0.0 EUR/MWh` T057 criterion is a non-degradation gate, not a material
  economic superiority proof.

Operational recheck after these changes:

- T057 plan SHA remains
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.
- `git diff -- .planning/phases/14-lt-audit-remediation/locked_holdout_plan_t057_t056_asof20260709.json`
  is empty.
- `scripts/check_epex_lab_locked_holdout_coverage.py` against the current
  2026-07-08 spot parquet exits `1`, expected while coverage is incomplete.
- Staging/source/export/readiness targeted validation:

```powershell
python -m pytest tests/test_build_epex_lab_source_export_manifest_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `60 passed, 1 skipped`.

Future approval next-step routing:

- `scripts/audit_epex_lab_future_approval_path.py` now emits
  `blocking_stage` and `next_required_step`.
- Current regenerated T057 audit remains
  `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, exits `1`, and reports:
  - `blocking_stage=locked_holdout_coverage`
  - `next_required_step=wait_for_full_spot_coverage_then_run_locked_holdout`
  - `recommended_commands.run_locked_holdout=python scripts/run_epex_lab_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --spot-parquet <FRESH_FUTURE_SPOT_PARQUET> --output-dir <T057_HOLDOUT_OUTPUT_DIR>`

Validation:

```powershell
python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `41 passed, 1 skipped`.

Additional expert-audit follow-up: locked holdout evidence is now bound to the
frozen plan identity, not just to pass/no-OMPEX flags.

Changed files:

- `scripts/epex_lab_locked_holdout_policy.py`
- `scripts/run_epex_lab_locked_holdout.py`
- `scripts/audit_epex_lab_locked_holdout.py`
- `scripts/build_epex_lab_adjusted_production_manifest.py`
- `scripts/build_epex_lab_adjusted_production_chain.py`
- `scripts/check_epex_lab_promotion_readiness.py`
- `scripts/audit_epex_lab_future_approval_path.py`
- `tests/test_epex_lab_locked_holdout_policy.py`
- `tests/test_run_epex_lab_locked_holdout_script.py`
- `tests/test_build_epex_lab_adjusted_production_manifest_script.py`
- `tests/test_build_epex_lab_adjusted_production_chain_script.py`
- `tests/test_check_epex_lab_promotion_readiness_script.py`
- `tests/test_audit_epex_lab_future_approval_path_script.py`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260709-T057-LOCKED-HOLDOUT.md`

Implementation notes:

- Runner and audit outputs now include `locked_plan_identity` with:
  `plan_id`, `plan_json_sha256`, window, baseline/adjusted CSV hashes, lab
  manifest hash, and selection summary hash.
- The shared policy recomputes the plan JSON SHA and compares the recorded
  identity to the plan contents.
- Production manifest, chain builder, readiness, and future approval all use
  this shared policy.
- The locked T057 plan JSON itself was not edited; SHA remains
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.

Regenerated current local NO-GO artifacts:

- `output/phase14/t057_locked_t056_future_holdout/current_spot_runner/locked_holdout_run_summary.json`
- `output/phase14/t057_locked_t056_future_holdout/future_approval_path_with_holdout_current.json`

Both remain local/generated output evidence. Current status remains
`NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, with
`blocking_stage=locked_holdout_coverage`.

Validation:

```powershell
python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `73 passed, 1 skipped`.

Context hygiene follow-up:

- `.planning/HANDOFF.md` now labels the 2026-07-08 promotion-ready candidate
  as the separate baseline daily production-ready candidate, not the T056 EPEX
  lab adjusted candidate.
- The T053/T054 search block is explicitly marked historical pre-T056 context.
  The active adjusted-candidate line remains T056/T057.

Coverage identity follow-up:

- `scripts/check_epex_lab_locked_holdout_coverage.py` now writes
  `locked_plan_identity` to the coverage report, matching the identity carried
  by runner/audit summaries.
- The ready-state coverage `next_action` now points to the fail-closed
  `scripts/run_epex_lab_locked_holdout.py` wrapper with a fresh output dir.
- Current regenerated local output:
  `output/phase14/t057_locked_t056_future_holdout/current_spot_runner/coverage_status.json`
  remains `WAITING_FOR_FULL_SPOT_COVERAGE` and records T057 plan SHA
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.
- The locked T057 plan JSON was not modified.

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `42 passed, 1 skipped`.

Holdout audit schema follow-up:

- `scripts/audit_epex_lab_locked_holdout.py` now requires the spot backtest
  summary to declare `schema_version=epex_shape_lab_spot_backtest.v1`,
  `read_only=true`, and `independent_production_evidence=false`.
- `tests/test_audit_epex_lab_locked_holdout_script.py` now covers rejection of
  an otherwise good summary with a wrong schema.
- `tests/test_run_epex_lab_locked_holdout_script.py` mocks were updated to
  match the real backtest summary contract.

Validation:

```powershell
python -m pytest tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `45 passed, 1 skipped`.

Backtest output hash follow-up:

- `scripts/backtest_epex_shape_lab_against_spot.py` now writes `output_hashes`
  for every generated diagnostic CSV, including
  `post_valuation_timestamp_residuals_csv`.
- `scripts/audit_epex_lab_locked_holdout.py` now requires the post-valuation
  residual CSV hash to match the summary via
  `post_valuation_csv_sha256_bound`.
- `tests/test_audit_epex_lab_locked_holdout_script.py` covers a tampered
  post-valuation CSV after summary creation.

Validation:

```powershell
python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `46 passed, 1 skipped`.

Run/audit linked-evidence hash follow-up:

- `scripts/run_epex_lab_locked_holdout.py` now writes
  `spot_backtest_summary_sha256` and `locked_holdout_audit_sha256`.
- `scripts/audit_epex_lab_locked_holdout.py` now writes
  `spot_backtest_summary_sha256` and
  `post_valuation_timestamp_residuals_csv_sha256`.
- `scripts/epex_lab_locked_holdout_policy.py` requires those linked evidence
  hashes for passable run/audit summaries.
- Test helpers for adjusted production manifest, chain, readiness, and future
  approval now create realistic linked evidence files for passing holdout
  summaries.

Validation:

```powershell
python -m pytest tests/test_epex_lab_locked_holdout_policy.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `82 passed, 1 skipped`.

Coverage value-usability follow-up:

- `scripts/check_epex_lab_locked_holdout_coverage.py` now requires the future
  EPEX spot parquet to contain `price_eur_mwh`.
- It also requires all observed holdout-window prices to be finite before
  setting `ready_to_run_backtest=true`.
- It requires `benchmark_policy=locked_future_no_ompex_holdout` in the locked
  plan before reporting `READY_TO_RUN_HOLDOUT_BACKTEST`.
- The coverage JSON now reports `spot_price_column`,
  `non_finite_holdout_price_rows`,
  `checks.plan_benchmark_policy_locked`,
  `checks.spot_price_column_present`, and
  `checks.holdout_prices_finite`.
- A full timestamp window without usable prices remains fail-closed with
  `WAITING_FOR_FULL_SPOT_COVERAGE` and CLI exit `1`.
- The locked T057 plan JSON was not modified.

Promotion evidence hardening:

- `scripts/run_epex_lab_locked_holdout.py` now writes
  `coverage_status_sha256`.
- It now requires `--expected-plan-sha256` and refuses to run coverage,
  backtest, or audit when the provided hash does not match the plan JSON.
  A mismatch writes
  `status=NO_GO_LOCKED_HOLDOUT_PLAN_HASH_MISMATCH`.
- `scripts/epex_lab_locked_holdout_policy.py` now accepts only
  `epex_lab_locked_holdout_run.v1` as passable promotion holdout evidence.
  Standalone `epex_lab_locked_holdout_audit.v1` remains diagnostic evidence
  but is rejected by production policy with
  `NO_GO_LOCKED_HOLDOUT_RUN_SUMMARY_REQUIRED`.
- Passable run summaries must hash-bind `coverage_status.json`,
  `spot_backtest_summary.json`, and `locked_holdout_audit.json`.
- Passable run summaries must carry matching `expected_plan_json_sha256`,
  `actual_plan_json_sha256`, and locked plan identity plan hash.
- The shared policy opens linked backtest/audit JSON and verifies schema,
  PASS statuses, no-OMPEX/lab-only flags, strict lab gate pass, and the same
  locked plan identity.
- `scripts/audit_epex_lab_locked_holdout.py` now explicitly requires
  `summary.status=DIAGNOSTIC_PASS` and `strict_lab_gate_pass=true`.
- Regenerated current local runner remains
  `WAITING_FOR_FULL_SPOT_COVERAGE`, exit `1`, with spot max
  `2026-07-08T23:00:00Z`, observed holdout hours `0`, expected `336`, and
  `coverage_status_sha256`, `expected_plan_json_sha256`, and
  `actual_plan_json_sha256` present.
- Coverage preflight now also verifies locked source CSV integrity before
  `READY_TO_RUN_HOLDOUT_BACKTEST`: baseline/adjusted paths must be present,
  files must exist, expected hashes must be present, and actual hashes must
  match the locked plan. Source failures report
  `NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH`.
- Current regenerated local runner shows `baseline_csv_sha256_bound=true` and
  `adjusted_csv_sha256_bound=true`; the remaining blocker is still only future
  spot coverage.
- Coverage preflight now also verifies that both locked candidate CSVs satisfy
  the backtest schema before `READY_TO_RUN_HOLDOUT_BACKTEST`: required hourly
  export columns, parseable CH timestamps, no duplicate timestamps, finite
  price/quantile values, and complete locked holdout timestamp coverage.
- Current regenerated local runner shows these candidate checks passing, with
  `baseline_candidate_missing_holdout_hours=0` and
  `adjusted_candidate_missing_holdout_hours=0`.
- Regenerated future approval audit remains
  `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, exit `1`, with
  `blocking_stage=locked_holdout_coverage` and a recommended runner command
  carrying the expected T057 plan hash.

Validation:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `86 passed, 1 skipped`.

Follow-up validation after explicit frozen-plan hash anchoring:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `90 passed, 1 skipped`.

Follow-up validation after source CSV preflight binding:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `92 passed, 1 skipped`.

Follow-up validation after candidate CSV schema/timestamp preflight:

```powershell
python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `94 passed, 1 skipped`.

Follow-up after candidate timestamp-set identity hardening:

- `scripts/check_epex_lab_locked_holdout_coverage.py` now records each locked
  candidate CSV timestamp count, UTC min/max, and sorted timestamp-set SHA.
- `ready_to_run_backtest` requires
  `checks.candidate_timestamp_sets_identical=true`.
- `scripts/epex_lab_locked_holdout_policy.py` now requires the downstream
  coverage payload to include explicit baseline/adjusted candidate preflight
  checks, source CSV hash binding, and identical candidate timestamp-set SHA.
- Current regenerated T057 runner remains
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

Follow-up after self-attested coverage policy hardening:

- Read-only expert audits after D70 returned GO overall and NO-GO promotion
  until future spot coverage exists.
- One P1 was accepted: the downstream policy was still too dependent on
  boolean `coverage.checks` fields.
- `scripts/epex_lab_locked_holdout_policy.py` now rejects passable run
  summaries unless the embedded coverage payload also carries
  `schema_version=epex_lab_locked_holdout_coverage.v1`, read-only/
  non-promotional flags, locked-plan identity matching the run summary, source
  CSV SHA fields matching identity, non-empty equal candidate timestamp-set SHA
  fields, positive equal timestamp counts, and equal non-empty timestamp
  min/max bounds.
- Passing holdout fixtures in manifest/chain/readiness/future-approval tests
  now include this raw coverage evidence.
- New policy tests reject coverage without raw candidate timestamp evidence and
  coverage whose embedded locked-plan identity does not match the run.

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

Follow-up after explicit UTC offset hardening:

- The accepted P2 from the read-only data audit has been addressed:
  `utc_offset_ch` is now required for locked baseline and adjusted candidate
  CSVs.
- `scripts/check_epex_lab_locked_holdout_coverage.py` reports
  `baseline_candidate_utc_offset_present` and
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

Follow-up after resolved evidence path hardening:

- The remaining path-sensitivity P2 from the read-only audit has been addressed
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

Follow-up after future plan timestamp identity hardening:

- New locked EPEX lab holdout plans now freeze candidate timestamp identity at
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

Follow-up after future locked plan path resolution:

- New locked EPEX lab holdout plans now resolve source/evidence paths at plan
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

Follow-up after requiring selection/lab manifest for future locked plans:

- New locked EPEX lab holdout plans must now include both candidate-selection
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

Follow-up after locked holdout coverage status routing:

- `scripts/check_epex_lab_locked_holdout_coverage.py` now emits
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

Follow-up after locked holdout policy blocking-checks hardening:

- `scripts/epex_lab_locked_holdout_policy.py` now requires
  `coverage_blocking_checks_clear`.
- Passing locked holdout fixtures now include `blocking_checks=[]`.
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

Follow-up after locked holdout input-invalid routing:

- `scripts/epex_lab_locked_holdout_policy.py` preserves
  `NO_GO_LOCKED_HOLDOUT_INPUT_INVALID` instead of collapsing it into generic
  holdout failure.
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

## 2026-07-09 Follow-Up - Production Chain Rebinds Holdout Policy

`scripts/build_epex_lab_adjusted_production_chain.py` now recomputes
`locked_holdout_policy` from the hash-bound locked holdout summary before
building export/selected/capstone artifacts.

It rejects an approved adjusted production manifest with
`locked_holdout_policy_bound` if the embedded policy object is stale or
hand-edited, even when the linked holdout summary itself still exists.

Validation:

```powershell
pytest tests\test_build_epex_lab_adjusted_production_chain_script.py tests\test_build_epex_lab_adjusted_production_manifest_script.py -q -p no:cacheprovider
```

Result: `25 passed`.

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `30 passed`.

## 2026-07-09 Follow-Up - Readiness Also Recommends The Wrapper

`scripts/check_epex_lab_promotion_readiness.py` now emits
`recommended_commands` when `production_blocking_stage` is
`locked_holdout_coverage`.

The recommended command mirrors future approval routing:

```powershell
python scripts/run_energy_charts_epex_locked_holdout.py --plan-json <T057_PLAN_JSON> --expected-plan-sha256 <T057_PLAN_JSON_SHA256> --output-dir <ENERGY_CHARTS_LOCKED_HOLDOUT_OUTPUT_DIR> --bzn CH
```

The manual `run_epex_lab_locked_holdout.py` command remains a fallback only for
a separately approved fresh future spot parquet.

Validation:

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `42 passed`.

```powershell
pytest tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `17 passed, 1 skipped`.

## 2026-07-09 Follow-Up - Future Approval Recommends The Wrapper

`scripts/audit_epex_lab_future_approval_path.py` now recommends the
fail-closed Energy Charts wrapper when locked holdout coverage is the blocking
stage.

Recommended command shape:

```powershell
python scripts/run_energy_charts_epex_locked_holdout.py --plan-json <T057_PLAN_JSON> --expected-plan-sha256 <T057_PLAN_JSON_SHA256> --output-dir <ENERGY_CHARTS_LOCKED_HOLDOUT_OUTPUT_DIR> --bzn CH
```

The manual runner remains available as a fallback only when a separately
approved fresh future spot parquet is supplied:

```powershell
python scripts/run_epex_lab_locked_holdout.py --plan-json <T057_PLAN_JSON> --expected-plan-sha256 <T057_PLAN_JSON_SHA256> --spot-parquet <FRESH_FUTURE_SPOT_PARQUET> --output-dir <T057_HOLDOUT_OUTPUT_DIR>
```

Validation:

```powershell
pytest tests\test_audit_epex_lab_future_approval_path_script.py tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `27 passed`.

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result: `14 passed`.

## 2026-07-09 Follow-Up - Expert Audit And Fail-Closed Spot Refresh

Read-only MIT/Roaster audit results:

- Quant/backtest auditor verdict: NO-GO, wait for full EPEX spot coverage.
  Do not run or validate T057 on partial coverage, do not modify the locked
  plan, and do not retune T056/t005 with the T057 holdout window.
- Governance auditor verdict: NO-GO production. T056/t005 remains the frozen
  lab candidate, but production promotion still requires a passing locked
  holdout plus real adjusted production/export/selected/capstone evidence.
- OMPEX remains desk benchmark/advisory only and must not enter model,
  calibration, selection, backtest, holdout, source hierarchy, or gates.

Additional tooling:

- Added `scripts/fetch_energy_charts_epex_spot_hourly.py`.
- Added `tests/test_fetch_energy_charts_epex_spot_hourly_script.py`.
- The helper fetches raw Energy Charts timestamps and aggregates only observed
  prices to hourly `price_eur_mwh`.
- It writes no parquet by default when the requested window is incomplete,
  preventing accidental forward-fill of future spot prices.

Important correction to local T057 diagnostics:

- A manual 2026-07-09 refresh had produced local parquets that appeared to
  cover `2026-07-10T00:00:00Z` to `2026-07-10T23:00:00Z`.
- The fail-closed helper showed the raw Energy Charts response for
  `2026-07-09` to `2026-07-11` only had observed CH prices through
  `2026-07-09T21:00:00Z`.
- The local manually generated 2026-07-09 parquets were removed from ignored
  output to avoid selecting potentially forward-filled future spot data.

Fail-closed refresh command:

```powershell
python scripts\fetch_energy_charts_epex_spot_hourly.py --start 2026-07-09 --end 2026-07-11 --bzn CH --output-parquet output\phase14\t057_locked_t056_future_holdout\epex_spot_refresh_20260709\epex_hourly_ch_energy_charts_20260709_script.parquet --summary-json output\phase14\t057_locked_t056_future_holdout\epex_spot_refresh_20260709\epex_hourly_ch_energy_charts_20260709_script_summary.json
```

Result:

- exit `1` as designed;
- `status=PARTIAL_COVERAGE`;
- `observed_hour_count=22`;
- `expected_hour_count=48`;
- `missing_hour_count=26`;
- `spot_max_utc=2026-07-09T21:00:00Z`;
- no parquet written.

Fail-closed T057 discovery:

```powershell
python scripts\discover_epex_spot_parquet_candidates.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --search-root output\phase14 --output-json output\phase14\t057_locked_t056_future_holdout\spot_parquet_discovery_20260709_failclosed.json --max-candidates 10
```

Result:

- `candidate_count=1`;
- best candidate:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`;
- `observed_holdout_hours=0`;
- `missing_holdout_hours=336`;
- `spot_max_utc=2026-07-08T23:00:00Z`.

Validation:

```powershell
pytest tests\test_fetch_energy_charts_epex_spot_hourly_script.py
```

Result: `4 passed`.

```powershell
pytest tests\test_fetch_energy_charts_epex_spot_hourly_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_check_epex_lab_locked_holdout_coverage_script.py
```

Result: `28 passed`.

Operational next step:

- Wait until Energy Charts or another approved EPEX spot parquet source covers
  the full locked window `2026-07-10T00:00:00Z` to
  `2026-07-23T23:00:00Z`.
- Refresh with the fail-closed helper.
- Run `scripts/run_epex_lab_locked_holdout.py` with the frozen plan SHA
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.

## 2026-07-09 Follow-Up - One-Command Energy Charts Locked Runner

Additional operator wrapper:

- Added `scripts/run_energy_charts_epex_locked_holdout.py`.
- Added `tests/test_run_energy_charts_epex_locked_holdout_script.py`.
- The wrapper verifies the locked plan SHA before fetching, then fetches the
  full T057 window with the fail-closed observed-hour Energy Charts helper.
- It writes no spot parquet and does not call the locked holdout runner unless
  the full pre-registered spot window is available.
- It writes `energy_charts_locked_holdout_run_summary.json` for both WAITING
  and future PASS/FAIL states.

Spot helper hardening:

- `scripts/fetch_energy_charts_epex_spot_hourly.py` now converts UTC bounds to
  Energy Charts date parameters before calling the API.
- API errors are persisted as `SPOT_FETCH_ERROR` summaries instead of
  surfacing as tracebacks.

Real operator command:

```powershell
python scripts\run_energy_charts_epex_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --output-dir output\phase14\t057_locked_t056_future_holdout\energy_charts_locked_runner_20260709 --bzn CH
```

Result:

- exit `1` by design;
- `status=LOCKED_HOLDOUT_SPOT_WAITING`;
- `spot_fetch.status=SPOT_FETCH_ERROR`;
- Energy Charts request uses `start=2026-07-10`, `end=2026-07-24`;
- API returned 404 because the full future window is not published yet;
- `expected_hour_count=336`;
- `observed_hour_count=0`;
- `missing_hour_count=336`;
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

Recommended future command is now the one-command wrapper above. It should be
rerun only after the full T057 window is expected to be available.

## 2026-07-09 Follow-Up - Wrapper Evidence Accepted By Policy

The shared locked holdout policy now recognizes the Energy Charts wrapper
summary:

- `scripts/epex_lab_locked_holdout_policy.py` supports
  `energy_charts_epex_locked_holdout_run.v1`.
- `LOCKED_HOLDOUT_SPOT_WAITING` maps to
  `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`, so promotion readiness and future
  approval path route it as a coverage wait rather than a generic schema
  failure.
- Future wrapper PASS can pass policy only if it links to a hash-bound inner
  `epex_lab_locked_holdout_run.v1` summary that passes the existing strict
  locked holdout policy.
- `scripts/run_energy_charts_epex_locked_holdout.py` now emits
  `locked_plan_identity`, `benchmark_policy`, and no-OMPEX flags.

Current real wrapper policy check:

```powershell
@'
import json
from pathlib import Path
from scripts.epex_lab_locked_holdout_policy import locked_holdout_policy
p = Path('output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260709/energy_charts_locked_holdout_run_summary.json')
policy = locked_holdout_policy(json.loads(p.read_text(encoding='utf-8')))
print(policy['status'], policy['pass'])
'@ | python -
```

Result:

- `status=NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`;
- `pass=false`;
- `operator_wrapper_status=LOCKED_HOLDOUT_SPOT_WAITING`;
- `spot_fetch_summary_matches_embedded=true`;
- `plan_identity_matches_plan_json=true`.

Validation:

```powershell
pytest tests\test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `15 passed`.

```powershell
pytest tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result: `26 passed`.

## 2026-07-09 Expert Audit + Discovery Coverage Follow-Up

Read-only expert agents were launched after the latest user request for the
next Phase 14 direction.

Agent verdicts:

- Tesla: promotion is still NO-GO. T056/t005 strict diagnostics pass, but the
  production chain is incomplete and T057 is blocked by future spot coverage.
  The locked plan SHA remains
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`;
  the adjusted CSV SHA remains
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`.
- Cicero: keep T056/t005 frozen for T057. Do not retune before the locked
  future holdout. T058 is research-only; any replacement must be a separate
  pre-registered EPEX-only lineage. OMPEX may explain shape differences after
  selection, but must not be used as model input, tuning target, selection
  metric, backtest truth, or promotion gate.

Code/test follow-up:

- `scripts/discover_epex_spot_parquet_candidates.py` now reports exact
  holdout coverage metrics for each discovered parquet candidate:
  `expected_holdout_hours`, `observed_holdout_hours`,
  `missing_holdout_hours`, `first_missing_holdout_utc`,
  `last_missing_holdout_utc`, and `full_window_covered`.
- `tests/test_discover_epex_spot_parquet_candidates_script.py` now verifies
  both full holdout coverage and the no-coverage lag case.

Validation:

```powershell
python -m pytest tests/test_discover_epex_spot_parquet_candidates_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `42 passed`.

Regenerated local discovery:

```powershell
python scripts\discover_epex_spot_parquet_candidates.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --search-root output\phase14 --output-json output\phase14\t057_locked_t056_future_holdout\spot_parquet_discovery_20260709.json --max-candidates 5
```

Result:

- `candidate_count=1`
- best candidate:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- `spot_max_utc=2026-07-08T23:00:00Z`
- `expected_holdout_hours=336`
- `observed_holdout_hours=0`
- `missing_holdout_hours=336`
- `first_missing_holdout_utc=2026-07-10T00:00:00Z`
- `last_missing_holdout_utc=2026-07-23T23:00:00Z`
- `full_window_covered=false`
- `spot_hours_until_latest_required_holdout=360.0`

Operational conclusion:

- The fresh helper makes the waiting state explicit per candidate, but does
  not change gates.
- Next promotion-readiness action remains: wait for a refreshed hourly EPEX
  spot parquet covering the full locked T057 window, then run the locked
  runner with the unchanged plan and expected plan SHA.
- Do not use T057 outcome for tuning. If T057 fails after complete coverage,
  document the failure and open a new pre-registered no-OMPEX lab lineage.

## 2026-07-09 T059 EPEX-Only Low-Tail/Cap/Night Interaction Sweep

Purpose:

- Follow the T058 signal without touching frozen T056/t005 or T057.
- Test whether lower low-tail intensity can preserve weak-bucket gains while
  recovering the post-valuation metric.
- Keep the line no-OMPEX, lab-only, non-promotional.

Durable T059 parameter files:

- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_grid.json`
- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_delta_grid.json`
- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_thresholds.json`
- `.planning/phases/14-lt-audit-remediation/t059_epex_only_lowtail_cap_night_interactions_scoring.json`

Plan command:

```powershell
python scripts\plan_epex_shape_lab_sweep.py --candidate-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-root output\phase14\t059_epex_only_lowtail_cap_night_interactions --valuation-timestamp 2026-07-07T00:00:00Z --grid-json '@.planning\phases\14-lt-audit-remediation\t059_epex_only_lowtail_cap_night_interactions_grid.json' --max-abs-delta-grid-json '@.planning\phases\14-lt-audit-remediation\t059_epex_only_lowtail_cap_night_interactions_delta_grid.json' --selection-thresholds-json '@.planning\phases\14-lt-audit-remediation\t059_epex_only_lowtail_cap_night_interactions_thresholds.json' --scoring-policy-json '@.planning\phases\14-lt-audit-remediation\t059_epex_only_lowtail_cap_night_interactions_scoring.json' --plan-id t059_epex_only_lowtail_cap_night_interactions --output-json output\phase14\t059_epex_only_lowtail_cap_night_interactions_plan.json
```

Plan result:

- `trial_count=36`
- plan SHA256:
  `56405a821f4c3bd91afce975c47beb0bf810736929fd6ccb27fd44dd852fb545`
- `benchmark_policy=pre_registered_independent_no_ompex`
- `activation_status=lab_only`
- `production_approved=false`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`

Execution:

```powershell
python scripts\execute_epex_shape_lab_sweep.py --plan-json output\phase14\t059_epex_only_lowtail_cap_night_interactions_plan.json --output-summary output\phase14\t059_epex_only_lowtail_cap_night_interactions_summary_full.json
```

Result:

- `trial_count_executed=36`
- `eligible_count=36`
- execution summary SHA256:
  `79a17516deea1be9362a9a6e56497b1a9ec69715d900bfafd0fc53bb046900e3`

Spot backtests:

Initial full backtest command with relative output paths hit a Windows
relative-path/UNC mkdir failure before writing the final summary. It was
rerun successfully with absolute output paths, without changing the plan or
inputs.

Successful command shape:

```powershell
$root = (Resolve-Path .).Path
$outRoot = Join-Path $root 'output\phase14\t059_epex_only_lowtail_cap_night_interactions_spot_backtests_full'
$outSummary = Join-Path $root 'output\phase14\t059_epex_only_lowtail_cap_night_interactions_spot_backtests_summary_full.json'
$selection = Join-Path $root 'output\phase14\t059_epex_only_lowtail_cap_night_interactions_selection_full'
python scripts\run_epex_shape_lab_sweep_spot_backtests.py --plan-json output\phase14\t059_epex_only_lowtail_cap_night_interactions_plan.json --sweep-summary output\phase14\t059_epex_only_lowtail_cap_night_interactions_summary_full.json --output-root $outRoot --output-summary $outSummary --incumbent-backtest output\phase14\t056_postval_final_micro_spot_backtests\t005_w075_l025_p089_e005_n055_r00\spot_backtest_summary.json --selection-output-dir $selection
```

Result:

- `trial_count_backtested=36`
- spot-backtest orchestration summary SHA256:
  `73aea816694ab8823e8523d6de74399bfcfbfda2d353db1e8d9c1bb7dc80de55`
- selection summary SHA256:
  `748b8c1103565e0fe6615cff6c1dbb8d82c9fce93f1975b7ed48dbf67e199fb9`

Selection verdict:

- `trial_count_summarized=36`
- `strict_pass_count=36`
- `replacement_candidate_count=0`
- `replacement_verdict.replace_incumbent=false`
- status:
  `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- degradation reason:
  `post_valuation_mae_improvement_eur_mwh`

Best weak-bucket trial:

- `t009_w075_l01_p089_e005_n055_r00_d275`
- adjusted CSV SHA256:
  `9ed04191719005f723259d61e0946047535002d10dc5384b480bbe5c30599e1c`
- overall improvement:
  `0.4651499923241654`
- solar-tail improvement:
  `0.47194371091304294`
- weekend improvement:
  `0.330769951895787`
- night improvement:
  `0.17918873802407917`
- ramp improvement:
  `0.06397807990619993`
- post-valuation improvement:
  `0.29827066207436914`

Incumbent T056/t005:

- adjusted CSV SHA256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- post-valuation improvement:
  `0.3049947368951571`

Conclusion:

- T059 confirms a monotonic-looking tradeoff: lowering low-tail/cap can improve
  weak historical buckets, but it does not protect post-valuation enough to
  replace T056/t005.
- T056/t005 remains frozen for T057.
- Do not continue broad low-tail lowering unless the next no-OMPEX hypothesis
  directly targets post-valuation preservation.

## 2026-07-09 EPEX Sweep Spot Backtest Path Hardening

During the T059 full backtest, the first command with relative output paths
failed before writing the final summary:

- symptom: Windows/UNC `mkdir` failure under
  `output\phase14\t059_epex_only_lowtail_cap_night_interactions_spot_backtests_full`
- successful workaround used during T059: pass absolute `output_root`,
  `output_summary`, and `selection_output_dir`

Follow-up implemented:

- `scripts/run_epex_shape_lab_sweep_spot_backtests.py` now resolves CLI paths
  to absolute paths on entry.
- Paths recorded in plans and sweep summaries are resolved against the repo
  root when relative.
- The run summary records resolved absolute output paths.
- `tests/test_run_epex_shape_lab_sweep_spot_backtests_script.py` now includes
  a regression for relative outputs from a changed cwd.

Validation:

```powershell
python -m pytest tests/test_run_epex_shape_lab_sweep_spot_backtests_script.py tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `26 passed, 1 skipped`.

This is orchestration hardening only. It does not change T059 plan parameters,
candidate hashes, no-OMPEX policy, selection metrics, or the verdict that T059
does not replace T056/t005.

## 2026-07-09 T059 Parameter Sensitivity Diagnostic

Added script:

`scripts/analyze_epex_shape_lab_sweep_sensitivity.py`

Function:

- read a pre-registered no-OMPEX sweep plan;
- read a no-OMPEX spot-backtest selection summary;
- join trial parameters with realized metrics;
- write parameter-response, correlation, merged trial, and summary artifacts;
- remain read-only, lab-only, non-promotional, and OMPEX-free.

Test:

`tests/test_analyze_epex_shape_lab_sweep_sensitivity_script.py`

Validation:

```powershell
python -m pytest tests/test_analyze_epex_shape_lab_sweep_sensitivity_script.py -q -p no:cacheprovider
```

Result: `2 passed`.

T059 analysis command:

```powershell
python scripts\analyze_epex_shape_lab_sweep_sensitivity.py --plan-json output\phase14\t059_epex_only_lowtail_cap_night_interactions_plan.json --selection-summary output\phase14\t059_epex_only_lowtail_cap_night_interactions_selection_full\spot_backtest_selection_summary.json --output-dir output\phase14\t059_epex_only_lowtail_cap_night_interactions_sensitivity
```

Result:

- `trial_count=36`
- sensitivity summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_sensitivity/sweep_sensitivity_summary.json`
- sensitivity summary SHA256:
  `ea7207b0601dd4f41aa4856122673fa15ed51f82f8435b972e83f796e011b28f`
- generated ignored files:
  `trial_parameter_metrics.csv`, `parameter_sensitivity.csv`,
  `parameter_metric_correlations.csv`, `sweep_sensitivity_summary.json`

Summary facts:

- `strict_pass_count=36`
- `weak_bucket_candidate_count=5`
- `replacement_candidate_count=0`
- `next_hypothesis_hint=protect_post_valuation_before_expanding_weak_bucket_gains`
- best overall/weak-bucket trial:
  `t009_w075_l01_p089_e005_n055_r00_d275`
- best post-valuation trial:
  `t036_w075_l025_p089_e005_n055_r00_d275`

Parameter reading:

- `low_tail=0.10`:
  - mean overall `0.4211366018533048`
  - max overall `0.4651499923241654`
  - mean post-valuation `0.2693469754455386`
  - max post-valuation `0.2982706620743691`
  - weak-bucket count `3`
- `low_tail=0.25`:
  - mean overall `0.4081703024032563`
  - max overall `0.4506842423821014`
  - mean post-valuation `0.2748380302045977`
  - max post-valuation `0.3049947368951571`
  - weak-bucket count `0`
- `max_abs_delta=2.75` is strongest in this grid:
  - mean overall `0.4551527817163272`
  - mean post-valuation `0.2988511662559723`
- Pearson correlations in this local grid include:
  - `low_tail_intensity` vs overall: `-0.14369962714720416`
  - `low_tail_intensity` vs post-valuation: `0.09254128382051807`
  - `max_abs_delta_eur_mwh` vs overall: `0.9878837595660808`
  - `max_abs_delta_eur_mwh` vs post-valuation: `0.9907428993390084`
  - `night_intensity` vs night: `0.5817850976590491`

Operational conclusion:

- T059 quantitatively confirms the tradeoff: lower low-tail improves
  weak-bucket score but does not preserve post-valuation.
- Next model-quality line should be pre-registered only if it has a specific
  no-OMPEX hypothesis for preserving post-valuation while keeping targeted
  weak-bucket gains.
- T056/t005 and locked T057 remain unchanged.

## 2026-07-09 Explicit Post-Valuation Replacement Guard

Follow-up implemented after T059 sensitivity:

- `scripts/summarize_epex_shape_lab_spot_backtests.py` now writes
  `post_valuation_gate_pass` and `core_metric_gate_pass` for each ranked
  trial.
- It also writes a summary-level `replacement_guard` object containing the
  replacement policy, required metrics, selected trial id, degraded metrics,
  and pass/status.
- This exposes the already-enforced rule that weak-bucket gains cannot replace
  the incumbent if any core replacement metric degrades, especially
  `post_valuation_mae_improvement_eur_mwh`.

Test update:

- `tests/test_summarize_epex_shape_lab_spot_backtests_script.py` now verifies
  that a weak-bucket trial with lower post-valuation fails both
  `post_valuation_gate_pass` and `core_metric_gate_pass`, while a true
  replacement candidate passes both.

Validation:

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_spot_backtests_script.py -q -p no:cacheprovider
```

Result: `5 passed`.

T059 was regenerated under ignored output with the explicit guard:

- enriched selection summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_selection_full/spot_backtest_selection_summary.json`
- selection summary SHA256:
  `dcf218c6f58f853aa02674f580748da24923bbba8a9f2da1c8c0f7d7f7e94f9b`
- enriched sensitivity summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_sensitivity/sweep_sensitivity_summary.json`
- sensitivity summary SHA256:
  `ab442633f4029d35a21973695055fe5de2ebcf1604f259783537332f98d64e42`

T059 enriched verdict:

- `replacement_guard.status=CORE_METRIC_DEGRADATION`
- `replacement_guard.pass=false`
- `replacement_guard.degraded_metrics=["post_valuation_mae_improvement_eur_mwh"]`
- selected trial remains
  `t009_w075_l01_p089_e005_n055_r00_d275`
- selected trial has `post_valuation_gate_pass=false` and
  `core_metric_gate_pass=false`
- no replacement; T056/t005 remains frozen for T057.

## 2026-07-09 Sweep Selection Path Hardening

Follow-up after explicit replacement guard:

- `scripts/summarize_epex_shape_lab_spot_backtests.py` now resolves CLI paths
  up front.
- Relative `ranking_csv` paths recorded in sweep summaries are resolved
  against the repo root.
- Generated selection summaries now record resolved paths, matching the
  path-hardening already applied to the sweep backtest runner.
- `tests/test_summarize_epex_shape_lab_spot_backtests_script.py` now covers a
  temp repo with a relative recorded `ranking_csv`, relative `backtest_root`,
  and relative `output_dir`.

Validation:

```powershell
python -m pytest tests/test_summarize_epex_shape_lab_spot_backtests_script.py -q -p no:cacheprovider
```

Result: `6 passed`.

T059 regenerated artifacts after path hardening:

- selection summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_selection_full/spot_backtest_selection_summary.json`
- selection summary SHA256:
  `ea5c0401f04798d3fa665eda8cf0b9831f819811fd205b0c3cdb7625e203b073`
- sensitivity summary:
  `output/phase14/t059_epex_only_lowtail_cap_night_interactions_sensitivity/sweep_sensitivity_summary.json`
- sensitivity summary SHA256:
  `9e0e57999c7f03c29f8eb98585f656c0a5419bd7f438ed9739d42ceff82cf89b`

The verdict is unchanged:

- `replacement_guard.status=CORE_METRIC_DEGRADATION`
- `replacement_guard.pass=false`
- degraded metric is still only
  `post_valuation_mae_improvement_eur_mwh`
- T056/t005 remains frozen for T057.

## 2026-07-09 T057 Spot Discovery Rejection Counters

Follow-up implemented while T057 still waits for future spot coverage:

- `scripts/discover_epex_spot_parquet_candidates.py` now emits
  `scanned_file_count`, `rejected_file_count`,
  `spot_like_rejected_file_count`, and `rejection_reason_counts`.
- Tests now assert these counters for:
  - a missing-price parquet;
  - a non-hourly spot-like parquet rejected by default;
  - inclusive non-hourly mode;
  - an empty search root.

Validation:

```powershell
python -m pytest tests/test_discover_epex_spot_parquet_candidates_script.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `42 passed`.

Refreshed T057 discovery command:

```powershell
python scripts\discover_epex_spot_parquet_candidates.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --search-root output\phase14 --output-json output\phase14\t057_locked_t056_future_holdout\spot_parquet_discovery_20260709_latest.json --max-candidates 10
```

Result:

- `scanned_file_count=22`
- `candidate_count=1`
- `rejected_file_count=21`
- `spot_like_rejected_file_count=1`
- `rejection_reason_counts={"index_not_datetime":2,"missing_price_column_or_empty":18,"non_hourly_grid":1}`
- best candidate remains:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- best candidate `spot_max_utc=2026-07-08T23:00:00Z`
- `observed_holdout_hours=0`
- `missing_holdout_hours=336`
- `full_window_covered=false`

Operational conclusion:

- T057 is still waiting for future hourly spot coverage.
- The one spot-like rejected parquet is non-hourly; default discovery
  correctly excludes it from the locked hourly holdout path.
- This is operator diagnostics only and does not change T057 gates.

## 2026-07-09 Expert Audit Follow-Up and T058 Lab Pre-Registration

Read-only expert audits were launched after the T056/t005 diagnostics and
OMPEX advisory review.

Model-quality audit conclusion:

- T056/t005 remains the best current no-OMPEX lab replacement candidate, but
  it is not promotable until T057 and the production evidence chain pass.
- Strict gates pass for the current lab candidate, and EEX BASE/PEAK level
  constraints are preserved at numerical zero.
- Remaining quality weaknesses are concentrated in solar-tail, midday,
  weekend, night, and ramp behavior. These must be improved with independent
  EPEX-only evidence, not by fitting to OMPEX.

Governance audit conclusion:

- Promotion remains NO-GO while T057 is
  `WAITING_FOR_FULL_SPOT_COVERAGE`.
- A future PASS must come from `epex_lab_locked_holdout_run.v1` bound to the
  unchanged locked plan and expected plan SHA
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.
- After T057 PASS, the production chain still needs a real adjusted
  production manifest, adjusted export manifest, selected artifact, and
  capstone all bound to the same production manifest SHA and locked holdout
  summary SHA.

Canonical T056/t005 PNG diagnostics were generated under ignored output:

`output/phase14/t056_postval_final_micro/t005_diagnostics/canonical_ch_hfc_png_20260709`

Command:

```powershell
python scripts\plot_ch_hfc_diagnostics.py --csv output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\candidate_epex_shape_lab_adjusted.csv --forwards output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_sweep_v2\diagnostic_forwards_history_rebuilt_20260708.parquet --output-dir output\phase14\t056_postval_final_micro\t005_diagnostics\canonical_ch_hfc_png_20260709 --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv
```

Result:

- Exit `0`.
- Generated `01_monthly_means_by_year.png`,
  `02_focus_2027_2028_eex_buckets.png`,
  `03_month_to_month_deltas.png`, `04_duck_curves_2027.png`,
  `04_duck_curves_2028.png`, `04_duck_curves_2030.png`,
  `05_heatmap_month_hour_2028.png`, `05_heatmap_month_hour_2030.png`,
  `06_negative_tail_fast_negative_hours.png`,
  `06_negative_tail_p10_negative_hours.png`,
  `07_eex_residuals_by_product.png`, `08_boundary_delta_jumps.png`,
  `09_executive_qa_summary.png`, `monthly_diagnostics.csv`, and
  `eex_residual_diagnostics.csv`.
- `eex_residual_diagnostics.csv` max absolute EEX residual is
  `2.717391254236645e-07` EUR/MWh, confirming that the level constraints are
  respected.
- Worst month-to-month mean moves include 2028-04 at `-38.914134` EUR/MWh and
  2027-04 at `-36.146815` EUR/MWh.
- Months with negative fast/P10 hours are concentrated in April-May from 2027
  onward. These are shape-quality review points, not production gates by
  themselves.
- Applied-delta boundary jumps versus the no-smoothing baseline remain small;
  the PNG summary reports max absolute boundary jump about `0.879` EUR/MWh.

T058 EPEX-only lab plan was pre-registered as a separate research line, not as
a replacement for frozen T056/T057:

`output/phase14/t058_epex_only_shape_micro_plan.json`

Plan SHA256:

`7818437211dc1b66c1645ffaf943ecbdfe1fe334ae0a51ac8910f94a5426e7d0`

Command path used: direct Python call into
`scripts.plan_epex_shape_lab_sweep.main()` to avoid PowerShell JSON quoting
issues. The script output was:

```json
{"plan_id": "t058_epex_only_shape_micro", "trial_count": 162}
```

Plan facts:

- `activation_status=lab_only`
- `production_approved=false`
- `benchmark_policy=pre_registered_independent_no_ompex`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`
- `ompex_postcheck_allowed_after_selection=true`
- baseline candidate SHA:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- EPEX spot parquet SHA:
  `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`
- valuation timestamp: `2026-07-07T00:00:00Z`
- grid:
  - `weekend_intensity`: `0.65`, `0.75`, `0.85`
  - `low_tail_intensity`: `0.15`, `0.25`, `0.35`
  - `peak_subshape_intensity`: `0.87`, `0.89`, `0.91`
  - `evening_recovery_intensity`: `0.05`
  - `night_intensity`: `0.45`, `0.55`, `0.65`
  - `ramp_intensity`: `0.0`
  - `max_abs_delta_eur_mwh`: `2.5`, `2.75`
- selection thresholds are restricted to executor-enforceable checks: spot age,
  spot coverage, ramp p99 increase, and minimum adjusted price. Realized
  MAE/fold-count decisions remain in the spot-backtest summarizer.
- scoring policy upweights solar-tail, weekend, and midday.

Executor contract hardening after T058 pre-registration:

- `scripts/plan_epex_shape_lab_sweep.py` now rejects unknown selection
  thresholds and scoring policy keys, preventing future pre-registered plans
  from carrying fields that `execute_epex_shape_lab_sweep.py` cannot apply.
- `scripts/execute_epex_shape_lab_sweep.py` now accepts `midday_weight`,
  emits `midday_mean_delta_eur_mwh`, and includes the midday bucket in the
  independent shape score.
- Tests added/updated in
  `tests/test_plan_epex_shape_lab_sweep_script.py` and
  `tests/test_execute_epex_shape_lab_sweep_script.py`.

Validation:

```powershell
python -m pytest tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `34 passed, 1 skipped`.

Controlled T058 first10 execution:

- A full sweep was started, confirmed slow, and stopped after 11 local trial
  directories. No Python process remained running afterward.
- A deterministic first10 summary was then generated with resume:

```powershell
python scripts\execute_epex_shape_lab_sweep.py --plan-json output\phase14\t058_epex_only_shape_micro_plan.json --output-summary output\phase14\t058_epex_only_shape_micro_summary_first10.json --max-trials 10
```

Result:

```json
{"eligible_count": 10, "trial_count_executed": 10}
```

First10 output facts:

- `benchmark_policy=executed_independent_no_ompex`
- `trial_count_executed=10`
- `eligible_count=10`
- best independent-shape trial:
  `t002_w065_l015_p087_e005_n045_r00_d275`
- best independent-shape score: `4.089226396580266`
- best trial independent deltas: midday `-0.9821475722222222`,
  solar-tail `-1.0645568976773383`, weekend `-0.3540212641799299`,
  ramp p99 increase `0.8039680399999583`, min adjusted price `-3.668281`.

Top-three first10 spot backtests:

```powershell
python scripts\backtest_epex_shape_lab_against_spot.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\t058_epex_only_shape_micro\t002_w065_l015_p087_e005_n045_r00_d275\candidate_epex_shape_lab_adjusted.csv --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t058_epex_only_shape_micro_spot_backtests_first10\t002_w065_l015_p087_e005_n045_r00_d275 --valuation-timestamp 2026-07-07T00:00:00Z
python scripts\backtest_epex_shape_lab_against_spot.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\t058_epex_only_shape_micro\t008_w065_l015_p089_e005_n045_r00_d275\candidate_epex_shape_lab_adjusted.csv --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t058_epex_only_shape_micro_spot_backtests_first10\t008_w065_l015_p089_e005_n045_r00_d275 --valuation-timestamp 2026-07-07T00:00:00Z
python scripts\backtest_epex_shape_lab_against_spot.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\t058_epex_only_shape_micro\t004_w065_l015_p087_e005_n055_r00_d275\candidate_epex_shape_lab_adjusted.csv --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t058_epex_only_shape_micro_spot_backtests_first10\t004_w065_l015_p087_e005_n055_r00_d275 --valuation-timestamp 2026-07-07T00:00:00Z
```

All three returned `status=DIAGNOSTIC_PASS`, `strict_lab_gate_pass=true`,
`benchmark_policy=rolling_origin_epex_spot_no_ompex_lab_only`, and all OMPEX
usage flags false.

T058 first10 selection summary against frozen T056/t005 incumbent:

```powershell
python scripts\summarize_epex_shape_lab_spot_backtests.py --sweep-summary output\phase14\t058_epex_only_shape_micro_summary_first10.json --backtest-root output\phase14\t058_epex_only_shape_micro_spot_backtests_first10 --output-dir output\phase14\t058_epex_only_shape_micro_selection_first10 --incumbent-backtest output\phase14\t056_postval_final_micro_spot_backtests\t005_w075_l025_p089_e005_n055_r00\spot_backtest_summary.json
```

Result:

- `replacement_verdict.status=WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- `replace_incumbent=false`
- `replacement_candidate_count=0`
- best weak-bucket trial:
  `t004_w065_l015_p087_e005_n055_r00_d275`
- best weak-bucket metrics: overall `0.428104291871567`, night
  `0.17009783026880573`, ramp `0.05713664911100517`, solar-tail
  `0.4270889440372494`, weekend `0.2945464419537373`, post-valuation
  `0.3033020021281363`.
- Frozen T056/t005 incumbent remains stronger on overall, evening, solar-tail,
  weekend, and post-valuation improvement.

Targeted12 T058 subset around the T056/t005 neighborhood:

- subset plan:
  `output/phase14/t058_epex_only_shape_micro_targeted12_plan.json`
- subset plan SHA256:
  `9d38fc46232d7f669ffbbb8ddf8576a01c58281ecd3d228c009208213bc8e00c`
- parent plan SHA256:
  `7818437211dc1b66c1645ffaf943ecbdfe1fe334ae0a51ac8910f94a5426e7d0`
- selected trial IDs:
  - `t074_w075_l025_p087_e005_n045_r00_d275`
  - `t076_w075_l025_p087_e005_n055_r00_d275`
  - `t078_w075_l025_p087_e005_n065_r00_d275`
  - `t080_w075_l025_p089_e005_n045_r00_d275`
  - `t082_w075_l025_p089_e005_n055_r00_d275`
  - `t084_w075_l025_p089_e005_n065_r00_d275`
  - `t086_w075_l025_p091_e005_n045_r00_d275`
  - `t088_w075_l025_p091_e005_n055_r00_d275`
  - `t090_w075_l025_p091_e005_n065_r00_d275`
  - `t064_w075_l015_p089_e005_n055_r00_d275`
  - `t136_w085_l025_p089_e005_n055_r00_d275`
  - `t028_w065_l025_p089_e005_n055_r00_d275`

Command:

```powershell
python scripts\execute_epex_shape_lab_sweep.py --plan-json output\phase14\t058_epex_only_shape_micro_targeted12_plan.json --output-summary output\phase14\t058_epex_only_shape_micro_targeted12_summary.json
```

Result:

```json
{"eligible_count": 12, "trial_count_executed": 12}
```

Targeted12 independent summary:

- `benchmark_policy=executed_independent_no_ompex`
- best independent-shape trial:
  `t064_w075_l015_p089_e005_n055_r00_d275`
- best independent-shape score: `4.308883116813839`
- best independent deltas: midday `-1.0216925744107743`, solar-tail
  `-1.117485511164918`, weekend `-0.3991375853253855`, ramp p99 increase
  `0.828056589999985`, min adjusted price `-3.941188`.

Top targeted12 spot backtests:

- Backtested:
  `t064_w075_l015_p089_e005_n055_r00_d275`,
  `t080_w075_l025_p089_e005_n045_r00_d275`,
  `t086_w075_l025_p091_e005_n045_r00_d275`,
  `t074_w075_l025_p087_e005_n045_r00_d275`, and
  `t082_w075_l025_p089_e005_n055_r00_d275`.
- All five returned `status=DIAGNOSTIC_PASS`,
  `strict_lab_gate_pass=true`,
  `benchmark_policy=rolling_origin_epex_spot_no_ompex_lab_only`, and all
  OMPEX usage flags false.
- `t082_w075_l025_p089_e005_n055_r00_d275` reproduces the frozen T056/t005
  adjusted CSV SHA:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`.

Targeted12 selection summary against frozen T056/t005 incumbent:

```powershell
python scripts\summarize_epex_shape_lab_spot_backtests.py --sweep-summary output\phase14\t058_epex_only_shape_micro_targeted12_summary.json --backtest-root output\phase14\t058_epex_only_shape_micro_targeted12_spot_backtests --output-dir output\phase14\t058_epex_only_shape_micro_targeted12_selection --incumbent-backtest output\phase14\t056_postval_final_micro_spot_backtests\t005_w075_l025_p089_e005_n055_r00\spot_backtest_summary.json
```

Result:

- `replacement_verdict.status=WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
- `replace_incumbent=false`
- `replacement_candidate_count=0`
- `strict_pass_count=5`
- best weak-bucket trial:
  `t064_w075_l015_p089_e005_n055_r00_d275`
- best weak-bucket adjusted CSV SHA:
  `9255a81e770184a4192f7ede1d3051c5283b802ade1f1b58d06d0eca3c485e34`
- best weak-bucket metrics: overall `0.4599653156253434`, evening
  `0.48163132451829344`, night `0.17317636817961332`, ramp
  `0.06009215161278314`, solar-tail `0.46799710365738634`, weekend
  `0.329903771232736`, post-valuation `0.3007021210797465`.
- Frozen T056/t005 incumbent metrics: overall `0.4506842423821014`,
  evening `0.4688940576897349`, night `0.16252506955713483`, ramp
  `0.053194830053255315`, solar-tail `0.46091530831501754`, weekend
  `0.3283653976588017`, post-valuation `0.3049947368951571`.
- Interpretation: `t064` improves the historical rolling buckets versus
  T056/t005 but is worse on the available post-valuation 24h check. The
  conservative replacement policy therefore keeps T056/t005 frozen for T057.

Operational conclusion:

- T056/t005 and T057 remain frozen.
- T058 is a lab-only, no-OMPEX research branch. The first10 slice does not
  justify replacing T056/t005. The targeted12 subset finds a historically
  stronger candidate but still does not justify replacement because
  post-valuation is weaker. A full sweep can be resumed later, but it does not
  change the T057 promotion path.
- Git output artifacts remain ignored under `output/phase14/`.

## 2026-07-09 T057 Coverage Lag Diagnostic Hardening

`scripts/check_epex_lab_locked_holdout_coverage.py` now emits informational
spot-lag fields:

- `latest_required_holdout_utc`
- `spot_hours_until_holdout_start`
- `spot_hours_until_latest_required_holdout`

These fields are for operator diagnostics only. They do not change
`ready_to_run_backtest`, `blocking_checks`, or any promotion gate.

Code/test changes:

- `scripts/check_epex_lab_locked_holdout_coverage.py`
- `tests/test_check_epex_lab_locked_holdout_coverage_script.py`

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `38 passed`.

Current T057 recheck:

```powershell
python scripts\check_epex_lab_locked_holdout_coverage.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output output\phase14\t057_locked_t056_future_holdout\coverage_status_20260709_lag_recheck.json
```

Result:

- Exit `1`, as expected while coverage is incomplete.
- `status=WAITING_FOR_FULL_SPOT_COVERAGE`
- `observed_holdout_hours=0`
- `expected_holdout_hours=336`
- `missing_holdout_hours=336`
- `first_missing_holdout_utc=2026-07-10T00:00:00Z`
- `last_missing_holdout_utc=2026-07-23T23:00:00Z`
- `spot_max_utc=2026-07-08T23:00:00Z`
- `latest_required_holdout_utc=2026-07-23T23:00:00Z`
- `spot_hours_until_holdout_start=25.0`
- `spot_hours_until_latest_required_holdout=360.0`
- `blocking_checks=["full_window_covered", "min_holdout_hours_met"]`

Operational conclusion:

- T057 remains NO-GO/waiting until a refreshed spot parquet covers
  `2026-07-10T00:00:00Z` through `2026-07-23T23:00:00Z`.
- The locked plan SHA and candidate hashes are unchanged.

## 2026-07-09 T057 Spot Parquet Discovery Helper

Added read-only helper:

`scripts/discover_epex_spot_parquet_candidates.py`

Purpose:

- discover candidate EPEX spot parquets under one or more local roots;
- rank candidates by `spot_max_utc`;
- require exact hourly timestamps by default, so the 15-minute EPEX parquet is
  not accidentally recommended for the hourly locked holdout runner;
- emit recommended coverage and locked-holdout runner commands bound to the
  locked plan SHA;
- never run the holdout and never approve production.

Code/test changes:

- `scripts/discover_epex_spot_parquet_candidates.py`
- `tests/test_discover_epex_spot_parquet_candidates_script.py`

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

- Exit `0`.
- `candidate_count=1`
- `require_hourly_grid=true`
- best candidate:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- best candidate `spot_max_utc=2026-07-08T23:00:00Z`
- best candidate `spot_hours_until_latest_required_holdout=360.0`
- recommended commands point to the locked plan SHA
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`.

Follow-up after expert roasts on production evidence and OMPEX advisory
governance:

- Read-only governance audit found two P1 issues:
  - `build_epex_lab_adjusted_production_chain.py` trusted approved manifests
    without reopening strict evidence files.
  - the production manifest did not require the source hierarchy policy used
    by the product audit to be the same policy cited by the manifest.
- Read-only quant/model audit confirmed T056/t005 remains the best no-OMPEX
  candidate, but OMPEX must stay advisory only. It also asked for explicit
  advisory flags in OMPEX benchmark output before official desk comparison.
- `scripts/build_epex_lab_adjusted_production_manifest.py` now requires the
  product summary's `source_hierarchy_policy` block to match the exact policy
  path/SHA and to report `ACCEPTED_PRODUCTION_APPROVED`,
  `production_approved=true`, and zero blocking quote conflicts.
- `scripts/build_epex_lab_adjusted_production_chain.py` now reloads and
  hash-validates strict evidence from the approved manifest before building
  export/selected/capstone artifacts. It rechecks monthly solver authority,
  product gates, Power BI strict gates, governance PASS, independent no-OMPEX,
  and source hierarchy policy binding.
- `scripts/compare_hpfc_ompex_benchmark.py` now writes explicit
  `promotion_gate=false`, `production_approved=false`,
  `ompex_used_in_selection=false`, and `ompex_used_in_backtest=false`.
- `scripts/audit_epex_lab_future_approval_path.py` now rejects optional spot
  sidecars unless `benchmark_policy` is
  `rolling_origin_epex_spot_no_ompex_lab_only`.

Changed files in this follow-up:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260709-T057-LOCKED-HOLDOUT.md`
- `scripts/audit_epex_lab_future_approval_path.py`
- `scripts/build_epex_lab_adjusted_production_chain.py`
- `scripts/build_epex_lab_adjusted_production_manifest.py`
- `scripts/compare_hpfc_ompex_benchmark.py`
- `tests/test_audit_epex_lab_future_approval_path_script.py`
- `tests/test_build_epex_lab_adjusted_production_chain_script.py`
- `tests/test_build_epex_lab_adjusted_production_manifest_script.py`
- `tests/test_compare_hpfc_ompex_benchmark_script.py`

Validation:

```powershell
python -m pytest tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py -q -p no:cacheprovider
```

Result: `37 passed`.

```powershell
python -m pytest tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `117 passed, 1 skipped`.

```powershell
python scripts/run_epex_lab_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t057_locked_t056_future_holdout\current_spot_runner
```

Result: exit `1` as expected. Wrapper validation accepted this. Status remains
`WAITING_FOR_FULL_SPOT_COVERAGE`; observed holdout hours `0`, expected `336`;
blocking checks are `full_window_covered` and `min_holdout_hours_met`.

Operational status after this follow-up:

- Current frozen T057 plan JSON remains unchanged.
- Promotion remains NO-GO until future EPEX spot covers
  `2026-07-10T00:00:00Z` to `2026-07-24T00:00:00Z`, the locked holdout
  runner passes, and approved production/export/selected/capstone artifacts
  are rebuilt from revalidated strict evidence.
- Next useful desk-side optional step is an OMPEX advisory comparison against
  the frozen T056/t005 adjusted CSV only after the candidate is fixed by hash;
  it must not feed model tuning, lambda selection, backtest, or promotion.

Follow-up after the remaining path-sensitivity audit finding:

- `scripts/check_epex_lab_locked_holdout_coverage.py` now resolves relative
  `baseline_csv` and `adjusted_csv` paths against the repo root inferred from
  the resolved plan path before falling back to cwd or plan directory.
- This keeps the frozen T057 plan unchanged while preventing a false
  `NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH` when the runner is
  launched from outside the repo root.
- Coverage output now includes `baseline_csv_resolved` and
  `adjusted_csv_resolved` for operator diagnostics.
- `tests/test_check_epex_lab_locked_holdout_coverage_script.py` includes a
  regression where a plan under `.planning` stores `output/...` relative
  source paths and `check_coverage` is called from a different cwd.

Changed files in this follow-up:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260709-T057-LOCKED-HOLDOUT.md`
- `scripts/check_epex_lab_locked_holdout_coverage.py`
- `tests/test_check_epex_lab_locked_holdout_coverage_script.py`

Validation:

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py -q -p no:cacheprovider
```

Result: `37 passed`.

```powershell
python -m pytest tests/test_check_epex_lab_locked_holdout_coverage_script.py tests/test_run_epex_lab_locked_holdout_script.py tests/test_epex_lab_locked_holdout_policy.py tests/test_audit_epex_lab_locked_holdout_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `117 passed, 1 skipped`.

```powershell
python scripts/run_epex_lab_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t057_locked_t056_future_holdout\current_spot_runner
```

Result: exit `1` as expected. Status remains
`WAITING_FOR_FULL_SPOT_COVERAGE`; observed holdout hours `0`, expected `336`;
blocking checks are `full_window_covered` and `min_holdout_hours_met`. The
coverage report now also records resolved absolute source paths for both locked
candidate CSVs.

Operational status:

- Current frozen T057 plan JSON remains unchanged.
- Promotion remains NO-GO until the future spot window is complete and the
  locked holdout passes.

Follow-up OMPEX advisory comparison for the frozen T056/t005 candidate:

- Purpose: desk benchmark only. OMPEX remains an imperfect external benchmark
  and must not feed model tuning, lambda selection, backtest, promotion, or
  production gates.
- Candidate CSV:
  `output/phase14/t056_postval_final_micro/t005_w075_l025_p089_e005_n055_r00/candidate_epex_shape_lab_adjusted.csv`
- Candidate CSV sha256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- OMPEX workbook:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260707_101700.xlsx`
- Output directory, ignored by Git:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/ompex_advisory_20260707`

Command:

```powershell
python scripts\compare_hpfc_ompex_benchmark.py --hpfc-csv output\phase14\t056_postval_final_micro\t005_w075_l025_p089_e005_n055_r00\candidate_epex_shape_lab_adjusted.csv --ompex-xlsx "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260707_101700.xlsx" --output-dir output\phase14\t056_postval_final_micro\t005_diagnostics\ompex_advisory_20260707 --alignment auto
```

Result:

- Exit `0`.
- Alignment selected: `ompex_minus_1h_hourending`.
- `benchmark_policy=advisory`, `read_only=true`, `promotion_gate=false`,
  `production_approved=false`, `ompex_used_in_model=false`,
  `ompex_used_in_selection=false`, `ompex_used_in_backtest=false`.
- Overlap: `39481` hourly points from `2026-07-01 00:00:00` to
  `2030-12-31 23:00:00`.
- Summary metrics: HPFC mean `84.70786023920365`, OMPEX mean
  `83.353833742813`, bias `1.354026496390669`, MAE
  `12.309986329728224`, RMSE `16.244705381704232`, correlation
  `0.8736510114242135`, p95 absolute error `32.742035`, max absolute error
  `102.47320500000001`.
- Enriched advisory diagnostics after the latest script update:
  - ramp points `39480`, ramp MAE `4.762832828672745`, ramp p95 absolute
    error `16.592220749999992`;
  - month-boundary jump points `53`, boundary jump MAE
    `10.393110113207543`, boundary jump p95 absolute error
    `41.01724999999999`;
  - largest boundary discrepancies are mostly transitions into May, where
    OMPEX has much sharper month-start jumps than the HPFC candidate.
- Advisory observations:
  - overall level is close versus OMPEX, but OMPEX is not ground truth;
  - largest hourly shape differences are around hours 16-19 and selected
    2027-2028 months;
  - month-hour heatmap confirms recurring advisory differences in selected
    summer midday/late-afternoon blocks and some evening blocks;
  - the OMPEX-inside-P10/P90 rate is low (`0.15964641219827258`), so width
    calibration remains worth reviewing with independent no-OMPEX evidence,
    not by retuning to OMPEX.
- Generated advisory files include `benchmark_metrics.json`,
  `alignment_sensitivity.csv`, `by_year_month.csv`, `by_hour.csv`,
  `by_bucket.csv`, `by_weekend.csv`, `ramp_metrics.csv`,
  `boundary_jumps.csv`, `month_hour_bias_matrix.csv`,
  `month_hour_mae_matrix.csv`, `top_abs_differences.csv`,
  `01_monthly_mean_hpfc_vs_ompex.png`, `02_error_by_hour.png`, and
  `03_month_hour_bias_heatmap.png`.

Code/test validation for the enriched advisory script:

```powershell
python -m pytest tests/test_compare_hpfc_ompex_benchmark_script.py -q -p no:cacheprovider
```

Result: `1 passed`.

```powershell
python -m pytest tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_plan_epex_lab_locked_holdout_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider
```

Result: `56 passed, 1 skipped`.

Operational conclusion:

- This comparison supports desk review only.
- It does not change T057 status; promotion remains NO-GO until locked future
  spot coverage and production-chain evidence pass.

Follow-up after promotion readiness routing:

- `scripts/check_epex_lab_promotion_readiness.py` now emits
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
