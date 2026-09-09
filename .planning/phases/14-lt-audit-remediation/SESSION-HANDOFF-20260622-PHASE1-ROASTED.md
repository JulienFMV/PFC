# Session Handoff - Phase 14 Monthly Reform - 2026-06-22 Phase 1 Roasted

## Session Scope

Branch: `fix/lt-audit-remediation`

HEAD at handoff time: `ea78b55`

Goal: harden Phase 1 after adversarial review. The first Phase 1 patch added
`Lambda(t)` diagnostics and a solver monthly-mean preservation hook, but expert
review found two real gaps:

- structural cap was applied in raw space before parent recentering;
- solver monthly means were preserved before calibration but could still drift
  after near-term rebalance or partial-product final calibration.

This handoff records the amended state, not the superseded first attempt.

## Files Changed In This Session

Permanent / handoff hygiene:

- `AGENTS.md`
  - Added root-level handoff/context hygiene rules.
  - Added permanent Phase 14 invariants and "do not touch" list.
- `CLAUDE.md`
  - New minimal pointer: Claude Code should read `AGENTS.md`; rules are not
    duplicated.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - New append-only decision log.
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-PHASE1-ROASTED.md`
  - This file.
- `.planning/HANDOFF.md`
  - Pointer to the latest session handoff.

Code:

- `pfc_shaping/calibration/monthly_curve_priors.py`
  - Added `_parent_mean_diagnostics(...)`.
  - Added `_cap_zero_mean_by_parent(...)`.
  - `build_structural_monthly_shape_prior(...)` now:
    - validates finite amplitude;
    - rejects non-positive/non-finite cap;
    - rejects shrinkage outside `[0, 1]`;
    - rejects non-finite monthly ratios;
    - recenters raw structural deviations by parent before shrink/cap;
    - applies cap in zero-mean parent space;
    - emits parent hours, pre-recenter parent mean, final parent mean, and
      max parent-mean residual.

- `pfc_shaping/lt/model/assembler.py`
  - In solver mode, `f_S` is forced to 1.0.
  - Solver mode fails fast unless both legacy cascade and legacy BASE smoothing
    are skipped.
  - `_stabilize_raw_curve` is bypassed in solver mode.
  - `_preserve_monthly_base_means(...)` recenters hourly shape to preserve
    monthly `B` means.
  - The monthly mean preservation is applied both before calibration input and
    again after near-term rebalance / optional intraday amplitude shrink.
  - Monthly-solver final calibration rows are skipped unless the whole product
    bucket is covered by the build index.

Tests:

- `tests/test_monthly_forward_curve_priors.py`
  - Added diagnostics test for structural `Lambda(t)`.
  - Added invariance to constant ratio offset with active cap.
  - Added invalid parameter rejection tests.
- `tests/test_monthly_forward_curve_integration.py`
  - Added far-horizon monthly mean preservation test.
  - Added near-term rebalance monthly mean preservation test.
  - Added solver misconfiguration fail-fast test.
  - Added partial-product final calibration skip test.
  - Adjusted solver contract tests to use full local product coverage.

## Commands Run

```powershell
pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_build_powerbi_exports_script.py tests/test_export_local_test_ch_hourly_csv_script.py -q
```

Result:

```text
64 passed in 74.32s (0:01:14)
```

```powershell
pytest tests/test_monthly_forward_curve_solver.py tests/test_audit_ch_hfc_seasonal_coherence_script.py -q
```

Result:

```text
18 passed in 3.32s
```

## Decisions Recorded

See `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`.

Key decisions:

- `D-20260622-01`: solver monthly curve is the level authority.
- `D-20260622-02`: no naive month-by-month `CAL_y > CAL_y+1` hard rule.
- `D-20260622-03`: structural `Lambda(t)` is a soft zero-mean parent-space
  prior; cap applies in zero-mean space.
- `D-20260622-04`: `AGENTS.md` is canonical; `CLAUDE.md` is only a pointer.

## Audit Handoff

Cloud auditor should inspect:

```powershell
git diff -- AGENTS.md CLAUDE.md `
  .planning/phases/14-lt-audit-remediation/DECISION-LOG.md `
  .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-PHASE1-ROASTED.md `
  .planning/HANDOFF.md `
  pfc_shaping/calibration/monthly_curve_priors.py `
  pfc_shaping/lt/model/assembler.py `
  tests/test_monthly_forward_curve_priors.py `
  tests/test_monthly_forward_curve_integration.py
```

Primary questions for audit:

1. Does `Lambda(t)` remain zero-mean inside each parent block after cap/shrink?
2. Is the cap invariant to a constant offset in template ratios?
3. Do diagnostics alone expose enough parent-level evidence?
4. Does solver mode preserve monthly `B` means after near-term rebalance?
5. Are partial-product calibration rows correctly skipped rather than applied
   to incomplete windows?
6. Does solver mode fail fast when legacy cascade/smoothing are still active?

## Known Risks / Next Phase

- This phase did not regenerate the real PFC or Power BI sidecars.
- The current Power BI view may still point to an older bad diagnostic CSV.
- Next phase must generate a fresh candidate and compare:
  - monthly solver curve;
  - assembler `B`;
  - pre-calibration `price_raw`;
  - post-calibration `price_shape`;
  - Power BI sidecars.
- If the final generated curve still fails, do not patch months manually.
  Revisit solver objective / priors / final calibration contract construction.

