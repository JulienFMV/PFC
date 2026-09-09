# Session Handoff - 2026-07-09 - T061 T060 Future Holdout

## Scope

Phase 14 LT audit remediation. This handoff records the separate future
holdout path created for the T060 EPEX-only cap decompression challenger.

Production remains NO-GO.

## Context

- Branch: `fix/lt-audit-remediation`
- T057/T056 locked holdout remains unchanged.
- T060/t007 is a lab challenger only. It must not be substituted into T057.
- OMPEX remains advisory only and forbidden in model, selection, backtest,
  holdout, and promotion gates.

## Files Changed

- `scripts/plan_epex_lab_locked_holdout.py`
- `tests/test_plan_epex_lab_locked_holdout_script.py`
- `.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t061_t060_asof20260709.json`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`

## Code Change

`scripts/plan_epex_lab_locked_holdout.py` now uses generic locked-holdout
placeholders for newly generated plans:

- `<LOCKED_HOLDOUT_PLAN_JSON>`
- `<LOCKED_HOLDOUT_PLAN_JSON_SHA256>`
- `<LOCKED_HOLDOUT_OUTPUT_DIR>`

This avoids generating T057-specific command templates for non-T057 future
holdout lines. The existing tracked T057 JSON was not regenerated or edited.

## Generated Plan

Plan path:

`.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t061_t060_asof20260709.json`

Plan SHA256:

`29a633cf56279eae817cd6c63872a476cc2c10b187f08c3952f73cdad76db135`

Key fields:

- `plan_id=t061_locked_t060_future_holdout`
- `frozen_at_utc=2026-07-09T00:00:00Z`
- `holdout_start_utc=2026-07-24T00:00:00Z`
- `holdout_end_utc=2026-08-07T00:00:00Z`
- `valuation_timestamp_utc=2026-07-07T00:00:00Z`
- `production_approved=false`
- `promotion_gate=false`
- `selection_policy.pass=true`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`
- `ompex_used_in_backtest=false`

Bound candidate hashes:

- baseline CSV SHA256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- adjusted CSV SHA256:
  `0a0fe8ce8c12bfeb64ac517ef60ac4d2850fbd1d13255c823c213c94c98391a6`
- candidate timestamp set SHA256:
  `c1ac9c621b1293e296f5789c342da5ecfee8444dc8fa0ad1030686079245020e`

T060/t007 locked lab config:

- `weekend_intensity=0.75`
- `low_tail_intensity=0.2`
- `peak_subshape_intensity=0.89`
- `evening_recovery_intensity=0.05`
- `night_intensity=0.55`
- `ramp_intensity=0.0`
- `max_abs_delta_eur_mwh=3.25`
- `max_weighted_negative_hours=0`
- `require_monthly_base_constraints=true`

## Commands Run

Unit validation before generating the plan:

```powershell
python -m pytest tests\test_plan_epex_lab_locked_holdout_script.py -q -p no:cacheprovider
```

Result:

`6 passed`

Focused validation after documentation and plan generation:

```powershell
python -m pytest tests\test_plan_epex_lab_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_audit_epex_lab_locked_holdout_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result:

`62 passed, 1 skipped`

Plan generation:

```powershell
python scripts\plan_epex_lab_locked_holdout.py --baseline-csv output\phase14\20260708_asof20260707_lshape100_yoy150_amp150_2032\ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --adjusted-csv output\phase14\t060_epex_only_cap_decompression\t007_w075_l02_p089_e005_n055_r00_d325\candidate_epex_shape_lab_adjusted.csv --selection-summary output\phase14\t060_epex_only_cap_decompression_selection_full\spot_backtest_selection_summary.json --lab-manifest output\phase14\t060_epex_only_cap_decompression\t007_w075_l02_p089_e005_n055_r00_d325\ab_lab_manifest.json --output .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t061_t060_asof20260709.json --plan-id t061_locked_t060_future_holdout --frozen-at-utc 2026-07-09T00:00:00Z --holdout-start-utc 2026-07-24T00:00:00Z --holdout-end-utc 2026-08-07T00:00:00Z --valuation-timestamp-utc 2026-07-07T00:00:00Z --min-holdout-hours 300
```

Plan verification:

```powershell
python - <<'PY'
import hashlib, json
from pathlib import Path
p=Path('.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t061_t060_asof20260709.json')
print(hashlib.sha256(p.read_bytes()).hexdigest())
plan=json.loads(p.read_text(encoding='utf-8'))
print(plan['plan_id'])
print(plan['holdout_start_utc'], plan['holdout_end_utc'])
print(plan['selection_policy']['pass'], plan['production_approved'], plan['promotion_gate'])
print('T057' in p.read_text(encoding='utf-8'), 'selected T056 candidate' in p.read_text(encoding='utf-8'))
PY
```

Observed output:

```text
29a633cf56279eae817cd6c63872a476cc2c10b187f08c3952f73cdad76db135
t061_locked_t060_future_holdout
2026-07-24T00:00:00Z 2026-08-07T00:00:00Z
True False False
False False
```

## Current Status

- T057: unchanged, still the frozen future holdout for T056/t005.
- T061: created as a separate future holdout line for T060/t007.
- T061 cannot run until future EPEX spot data covers
  `2026-07-24T00:00:00Z` to `2026-08-07T00:00:00Z`.
- T060/T061 is not production evidence yet.

## Queue Audit Follow-Up

Added:

`scripts/audit_epex_lab_locked_holdout_queue.py`

This read-only helper summarizes multiple locked holdout plans, computes exact
plan SHA values, classifies each window by `as_of_utc`, and emits exact
operator commands. It also verifies that the bound baseline CSV, adjusted CSV,
lab manifest, and selection summary artifacts exist locally and match the plan
hashes. It does not fetch spot data, run backtests, tune candidates, or approve
production.

The CLI supports `--plan-glob` and deduplicates resolved paths, so Phase 14
operators can audit every tracked `locked_holdout_plan_*.json` without
hand-listing T057 and T061.

The queue audit also blocks duplicate `plan_id` values and overlapping holdout
windows before recommending wait or run actions.

Test:

```powershell
python -m pytest tests\test_audit_epex_lab_locked_holdout_queue_script.py -q -p no:cacheprovider
```

Result:

`10 passed`

Current local queue audit:

```powershell
python scripts\audit_epex_lab_locked_holdout_queue.py --plan-glob ".planning\phases\14-lt-audit-remediation\locked_holdout_plan_*.json" --as-of-utc 2026-07-09T00:00:00Z --search-root output\phase14 --output output\phase14\locked_holdout_queue_audit_20260709.json
```

Observed:

- `status=WAITING_FOR_FUTURE_HOLDOUT_WINDOWS`
- `plan_count=2`
- `future_window_count=2`
- `active_window_count=0`
- `spot_refresh_due_count=0`
- `invalid_plan_count=0`
- `policy_invalid_plan_count=0`
- `artifact_invalid_plan_count=0`
- `duplicate_plan_id_count=0`
- `overlapping_window_count=0`
- `queue_issues=[]`
- T057 source artifacts are all present and hash-bound.
- T061 source artifacts are all present and hash-bound.
- T057 next step: `wait_without_retuning_candidate`
- T061 next step: `wait_without_retuning_candidate`

## Energy Charts Pre-Window Guard Follow-Up

Added a pre-window guard to:

`scripts/run_energy_charts_epex_locked_holdout.py`

Behavior:

- verifies schema and exact plan SHA first;
- computes `latest_required_holdout_utc = holdout_end_utc - 1h`;
- if `as_of_utc <= latest_required_holdout_utc`, exits before Energy Charts
  fetch with `LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE`;
- writes `spot_fetch_ran=false`, `locked_holdout_ran=false`, and
  `holdout_pass=false`.

Policy routing:

- `scripts/epex_lab_locked_holdout_policy.py` maps
  `LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE` to
  `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`.

Validation:

```powershell
python -m pytest tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result:

`48 passed`

Broader validation:

```powershell
python -m pytest tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_audit_epex_lab_locked_holdout_queue_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result:

`78 passed, 1 skipped`

Observed T057 pre-window command:

```powershell
python scripts\run_energy_charts_epex_locked_holdout.py --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd --output-dir output\phase14\t057_locked_t056_future_holdout\energy_charts_pre_window_guard_20260709 --as-of-utc 2026-07-09T00:00:00Z
```

Observed status:

- `LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE`
- `spot_fetch_ran=false`
- `locked_holdout_ran=false`
- `latest_required_holdout_utc=2026-07-23T23:00:00Z`

## Readiness Queue Guard Follow-Up

Added a promotion-readiness guard to:

`scripts/check_epex_lab_promotion_readiness.py`

Behavior:

- CLI accepts `--locked-holdout-queue-summary`.
- `required_production_checks` now includes `locked_holdout_queue_pass`.
- Once production/export/selected/capstone evidence is present, readiness
  requires a queue summary.
- A missing queue summary, invalid queue, duplicate plan id, overlapping
  window, artifact-invalid plan, or queue that does not include the bound
  locked-holdout plan SHA keeps readiness NO-GO.
- Queue failures route to `production_blocking_stage=locked_holdout_queue` and
  `next_required_step=fix_locked_holdout_queue_before_promotion_review`.

Current implication:

- The current real queue audit status is `WAITING_FOR_FUTURE_HOLDOUT_WINDOWS`.
- This global queue status does not block T057 by itself when the queue is
  otherwise valid and contains the bound T057 plan SHA.
- This is separate from the queue audit being read-only: the audit does not
  approve production, but readiness now requires it as a consistency gate
  before any complete production bundle can be accepted.

Focused validation:

```powershell
python -m pytest tests\test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider
```

Result:

`21 passed`

Broader validation:

```powershell
python -m pytest tests\test_audit_epex_lab_locked_holdout_queue_script.py tests\test_discover_epex_spot_parquet_candidates_script.py tests\test_plan_epex_lab_locked_holdout_script.py tests\test_run_epex_lab_locked_holdout_script.py tests\test_run_energy_charts_epex_locked_holdout_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result:

`96 passed, 1 skipped`

Future approval follow-up:

- `scripts/audit_epex_lab_future_approval_path.py` now includes
  `locked_holdout_queue_pass` in its internal minimum production-check set, so
  old or synthetic readiness JSON cannot omit the queue requirement.
- It propagates readiness queue details into `locked_holdout_queue_policy` and
  routes a queue-only failure to `blocking_stage=locked_holdout_queue`.
- It also rejects a synthetic scalar `locked_holdout_queue_pass=true` without
  the structured queue policy payload as
  `NO_GO_LOCKED_HOLDOUT_QUEUE_UNBOUND`.
- Current local queue/readiness/future-approval outputs were regenerated.
- Queue audit:
  `output/phase14/locked_holdout_queue_audit_20260709.json` reports
  `WAITING_FOR_FUTURE_HOLDOUT_WINDOWS`, `plan_count=2`,
  `future_window_count=2`.
- Readiness:
  `output/phase14/t057_locked_t056_future_holdout/promotion_readiness_with_locked_holdout_current.json`
  reports `STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`,
  `production_blocking_stage=locked_holdout_coverage`, with failed checks
  including `locked_holdout_pass`; `locked_holdout_queue_pass` is PASS.
- Future approval:
  `output/phase14/t057_locked_t056_future_holdout/future_approval_path_with_holdout_current.json`
  reports `NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING`,
  `blocking_stage=locked_holdout_coverage`, with remaining blockers including
  `locked_holdout_pass`, and
  `locked_holdout_queue_policy.status=LOCKED_HOLDOUT_QUEUE_PASS`.

Validation:

```powershell
pytest tests\test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider
```

Result:

`13 passed`

Broader validation:

```powershell
pytest tests\test_audit_epex_lab_future_approval_path_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_locked_holdout_queue_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_build_epex_lab_adjusted_production_manifest_script.py tests\test_build_epex_lab_adjusted_production_chain_script.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result:

`102 passed, 1 skipped`

## Production Chain Builder Follow-Up

Fixed a latent bug in:

`scripts/build_epex_lab_adjusted_production_chain.py`

Details:

- `_same_path()` now performs raw string equality followed by resolved
  filesystem path equality.
- Removed unreachable path-comparison code that had been left after
  `_powerbi_critical_count()`.
- `tests/test_build_epex_lab_adjusted_production_chain_script.py` now covers a
  resolved `adjusted_production_manifest` self reference and supplies the
  required locked-holdout queue summary when asserting that chain artifacts can
  unlock readiness.

Validation:

```powershell
pytest tests\test_build_epex_lab_adjusted_production_chain_script.py -q -p no:cacheprovider
```

Result:

`10 passed`

Broader validation:

```powershell
pytest tests\test_build_epex_lab_adjusted_production_manifest_script.py tests\test_build_epex_lab_adjusted_production_chain_script.py tests\test_check_epex_lab_promotion_readiness_script.py tests\test_audit_epex_lab_future_approval_path_script.py tests\test_epex_lab_locked_holdout_policy.py tests\test_lt_ct_imports.py -q -p no:cacheprovider
```

Result:

`91 passed, 1 skipped`

## Next Steps

1. Keep T057 frozen and wait for full T057 spot coverage.
2. When T057 coverage exists, run the locked T057 wrapper with the exact T057
   plan SHA.
3. Keep T061 frozen separately; do not retune T060/t007 against the future
   T061 window.
4. When T061 coverage exists, run its locked holdout with exact plan SHA
   `29a633cf56279eae817cd6c63872a476cc2c10b187f08c3952f73cdad76db135`.
5. Only after passing future holdout evidence should production/export/selected
   artifact/capstone promotion evidence be built.

## Risks

- Promoting T060 from lab metrics alone would be an audit failure.
- Reusing T057 evidence for T060 would contaminate both lines.
- OMPEX comparison remains useful for desk review, but must not enter model
  selection, backtesting, or promotion.
