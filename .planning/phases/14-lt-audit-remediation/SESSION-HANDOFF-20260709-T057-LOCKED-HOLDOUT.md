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
python scripts/run_epex_lab_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --spot-parquet output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\epex_spot_refresh_20260708\epex_hourly_ch_energy_charts_20260708.parquet --output-dir output\phase14\t057_locked_t056_future_holdout\current_spot_runner
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
  - `recommended_commands.run_locked_holdout=python scripts/run_epex_lab_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --spot-parquet <FRESH_FUTURE_SPOT_PARQUET> --output-dir <T057_HOLDOUT_OUTPUT_DIR>`

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
