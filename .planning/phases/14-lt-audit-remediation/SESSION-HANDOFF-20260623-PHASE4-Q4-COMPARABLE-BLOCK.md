# Session Handoff - 2026-06-23 - Phase 4 Q4 Comparable Block

## Scope

Worktree:
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase4_q4`

Canonical UNC worktree:
`\\fmvfs2\Data\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase4_q4`

Branch: `clean/phase4-cross-year-q4`, tracking
`origin/fix/lt-audit-remediation`.

Purpose: continue Phase 14 after solver structural-prior audit by hardening the
cross-year comparable-block audit for Q4-vs-calendar cases. No commit was made.

## Files Changed

- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/MONTHLY-FORWARD-CURVE-REFORM-PLAN-20260619.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-PHASE4-Q4-COMPARABLE-BLOCK.md`
- `pfc_shaping/calibration/monthly_curve_audit.py`
- `pfc_shaping/calibration/monthly_curve_promotion.py`
- `tests/test_monthly_forward_curve_audit.py`

## Implementation

- Extended `residual_vs_implied_comparable_block` audit rows to cover
  `quarter|calendar` parent comparisons in addition to existing
  `residual|calendar` comparisons.
- Updated the comparable-block gate evidence text to expose
  `comparable_parent_types`.
- Generalized the remediation hint from Apr-Dec residual-only wording to
  quoted seasonal sub-blocks versus full calendars.
- Extended historical threshold observation generation so Q4/quarter-vs-CAL
  comparable-block metrics can be calibrated separately from residual-vs-CAL
  metrics instead of always becoming `UNSUPPORTED`.
- Added `parent_type_pair` threshold provenance for comparable-block rows.
  Runtime Q4 rows now request `quarter|calendar` thresholds specifically; if
  the threshold artifact has only `residual|calendar` support, the Q4 row is
  fail-closed as `UNSUPPORTED`.
- Aligned promotion threshold lookup with `parent_type_pair`.
- Updated the Phase 14 gate specification table to name Q4-vs-CAL evidence
  explicitly.
- Added regression coverage:
  - a repriced Q4 December inversion against a next-year calendar produces a
    `CRITICAL` comparable-block row;
  - the historical threshold builder emits usable `month_12` threshold rows for
    Q4-vs-calendar history;
  - Q4-vs-CAL rows become `UNSUPPORTED` when only residual-vs-CAL threshold
    evidence exists;
  - reversed `calendar|quarter` runtime orientation normalizes to
    `quarter|calendar` threshold provenance.

## Expert Roast Results

Three read-only agent roasts were run after the initial Phase 4 patch:

- Quant/model roast: no blockers/highs; medium concern that residual/CAL and
  quarter/CAL thresholds were pooled without a parent-type dimension.
- Governance roast: high concern that Q4 runtime rows could silently reuse
  residual/CAL thresholds, violating fail-closed promotion evidence. Medium
  concern that the Phase 14 acceptance matrix still described the gate as
  residual-only. Low concern that handoff path used only a mapped drive.
- Implementation/test roast: no blockers/highs; same threshold-pooling medium,
  plus low issues around duplicated predicate, reversed orientation coverage,
  and test fixture duplication.

Corrections made after roasts:

- Added `parent_type_pair` to threshold rows and runtime/promotion lookup.
- Added one helper for calendar-vs-seasonal-subblock comparability.
- Added fail-closed and reversed-orientation tests.
- Updated the Phase 14 plan table and handoff path.

## Commands Run

```powershell
python -m pytest tests/test_monthly_forward_curve_audit.py -q
```

Initial result before expert-roast corrections: `14 passed in 2.48s`.

Final result after `parent_type_pair` provenance and fail-closed tests:
`17 passed in 10.73s`.

```powershell
python -m pytest tests/test_monthly_forward_curve_audit.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py -q
```

Initial result after threshold-builder fix: `26 passed in 3.85s`.

Final result after expert-roast corrections: `28 passed in 11.19s`.

```powershell
python -m pytest tests/test_monthly_forward_curve_audit.py tests/test_monthly_forward_curve_constraints.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_lt_ct_imports.py -q
```

Initial result before expert-roast corrections: `74 passed, 1 skipped in 85.46s`.

Final result after expert-roast corrections: `76 passed, 1 skipped in 47.97s`.

```powershell
git diff --check
```

Result: no whitespace errors; Git reported CRLF working-copy warnings only.

```powershell
rg -n "pfc_shaping\.ct|from pfc_shaping\.model|import pfc_shaping\.model|powerbi/PFC_QA|powerbi/data|pfc_shaping/data/.*\.(parquet|duckdb)|data/epex_hourly\.parquet" pfc_shaping/calibration/monthly_curve_audit.py tests/test_monthly_forward_curve_audit.py
```

Result: no code matches. The only matches in a broader check were text in the
Phase 14 plan documenting Power BI outputs and LT/CT import invariants.

## Decisions Recorded

Added `D-20260623-02 - Q4 Is A Comparable Seasonal Sub-Block` to
`.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`.

## Current Status

```text
## clean/phase4-cross-year-q4...origin/fix/lt-audit-remediation
 M .planning/HANDOFF.md
 M .planning/phases/14-lt-audit-remediation/DECISION-LOG.md
 M .planning/phases/14-lt-audit-remediation/MONTHLY-FORWARD-CURVE-REFORM-PLAN-20260619.md
 M pfc_shaping/calibration/monthly_curve_audit.py
 M pfc_shaping/calibration/monthly_curve_promotion.py
 M tests/test_monthly_forward_curve_audit.py
?? .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-PHASE4-Q4-COMPARABLE-BLOCK.md
```

## Residual Risks

- This branch hardens audit evidence only; it does not regenerate a candidate
  PFC or modify solver objective weights.
- The branch does not include the separate solver structural-prior branch
  changes. Merge order must preserve both decision-log additions.
- Structural fan-chart width and delivered hourly PEAK gates remain separate
  phases.
- Branch tracking points at `origin/fix/lt-audit-remediation`; confirm push
  target before any commit/push.

## Next Steps

1. Review/commit Phase 4 Q4 comparable-block audit as a separate patch if
   accepted.
2. Rebase or merge carefully with `clean/solver-structural-prior` so decision
   log entries and tests are both retained.
3. Continue with structural fan-chart width or delivered hourly promotion
   evidence only after the audit branches are committed/merged.
