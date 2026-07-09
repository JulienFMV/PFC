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

Operational conclusion:

- T056/t005 and T057 remain frozen.
- T058 is a lab-only, no-OMPEX research branch. The first10 slice does not
  justify replacing T056/t005. A full sweep can be resumed later, but it does
  not change the T057 promotion path.
- Git output artifacts remain ignored under `output/phase14/`.

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
