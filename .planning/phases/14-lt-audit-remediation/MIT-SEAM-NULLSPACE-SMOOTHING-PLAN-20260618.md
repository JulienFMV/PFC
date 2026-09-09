# MIT Seam Nullspace Smoothing Plan - 2026-06-18

## Objective

Reduce artificial hourly jumps at month boundaries in the CH LT HFC/PFC while preserving quoted EEX BASE and PEAK products exactly.

Current blocker:

`boundary_jump_abs_p95_eur_mwh = 28.37`, which caps the local shape score at `7.75/10`.

Hard rule:

Do not solve this by relaxing quoted EEX calibration. Calibration error may only be numerical tolerance, not a modelling knob.

## Scientific Position

The implementation must be aligned with the smooth-forward literature:

- Fleten and Lemming construct high-resolution electricity forward curves by combining sparse market quotes with model/prior shape information.
- Benth, Benth and Koekebakker's smooth-forward framing treats market contracts as hard constraints and optimizes curve smoothness around them.
- Intraday/duck literature implies we must not smooth away structural renewable-driven intraday shape; the rejected `lambda_smooth_h=30` candidate demonstrated this failure.

Therefore the correct formulation is:

```text
minimize    ||W(p - p0)||^2 + lambda_D1 ||D1 p||^2 + lambda_D2 ||D2 p||^2 + lambda_seam ||S p||^2
subject to  A p = q
```

Or equivalently:

```text
p = p_exact + N z,  where A N = 0
minimize smoothness over z
```

The second form is preferable for seam-specific post-processing because EEX exactness is guaranteed by construction.

## Rejected Shortcuts

1. Relax BASE/PEAK calibration.

Rejected. This breaks arbitrage-free governance.

2. Apply global quote-aware annual smoothing with overlapping BASE/PEAK constraints.

Rejected from direct probe:

- `boundary_p95` improved to `18.79`
- but `max_eex_peak_error_eur_mwh = 0.262`
- seasonal audit returned `1 critical / 3 warning`
- 2027/2028 monthly paths were economically distorted

3. Increase hourly smoothness lambda.

Rejected. `lambda_smooth_h=30` collapsed the 2030 evening-midday duck from about `26.2` to `12.6`.

## Recommended Patch

Implement a local seam nullspace smoother.

### Scope

Add a new local/test OFF-by-default pass after current final monthly/annual smoothing:

`--enable-final-seam-nullspace-smoothing`

Suggested default diagnostic parameters:

| parameter | value |
|---|---:|
| seam window hours | 120 |
| max abs delta | 12 EUR/MWh |
| target boundary p95 | <= 20 EUR/MWh |
| min EEX BASE/PEAK tolerance | <= 1e-6 EUR/MWh |

### Algorithm

For each month boundary where `abs(p_t - p_{t-1})` exceeds a threshold:

1. Select a local window around the seam, e.g. `[t - 120h, t + 120h]`.
2. Build active constraints affecting that window:
   - BASE bucket means for touched EEX buckets.
   - PEAK bucket means where CH PEAK quotes exist.
3. Construct a desired seam correction `r` that reduces the local step:
   - Use a smooth ramp/taper across the window.
   - Correction must be local and bounded.
4. Project `r` into the nullspace of constraints:
   - `r0 = r - A.T (A A.T)^+ A r`
   - equivalently solve least squares so `A r0 = 0`.
5. Apply `p <- p + alpha r0`, with line search:
   - enforce negative floor
   - enforce max weighted negative hours
   - enforce max abs delta
   - recheck all EEX BASE/PEAK residuals
6. Recompute weighted fan columns.

### Why Local Nullspace, Not Global Annual Solve

The global annual solve allowed the optimizer to reshape 2027/2028 too much and exposed inconsistency between overlapping BASE/PEAK quote systems. A local nullspace smoother targets only the artifact measured by `boundary_jump_abs_p95` and leaves monthly/seasonal economics mostly intact.

## Acceptance Criteria

Required for merge of the seam patch:

| check | required |
|---|---|
| full CLI writes CSV | PASS under 180s target or documented timing |
| max EEX BASE error | `<= 1e-6` |
| max EEX PEAK error | `<= 1e-6` |
| shape score | `>= 8.0` |
| boundary jump p95 | `<= 20 EUR/MWh` |
| hfc-vs-spot score | `>= 8.0` |
| seasonal critical/warning | `0 / 0` |
| monthly path critical/warning | `0 / 0` |
| calendar critical/warning | `0 / 0` |
| negative gate | PASS |
| 2030 evening-midday duck | stay in `18-30 EUR/MWh` band |
| visual PNGs | no 2028-2030 market-incoherent monthly shape |

## Tests To Add

1. Synthetic two-bucket seam test:
   - Create two adjacent monthly buckets with a midnight cliff.
   - Apply seam nullspace smoothing.
   - Assert bucket means unchanged.
   - Assert boundary jump reduced.

2. BASE+PEAK preservation test:
   - Add a quoted PEAK bucket.
   - Assert both BASE and PEAK means unchanged.

3. Negative-floor test:
   - Candidate correction that would breach floor must be line-searched or rejected.

4. Partial-horizon test:
   - First/last partial years must be skipped or handled without false PEAK infeasibility.

5. Regression test on generated candidate:
   - `boundary_jump_abs_p95 <= 20`
   - `max_eex_base_error <= 1e-6`
   - `max_eex_peak_error <= 1e-6`

## Governance

This remains local/test until:

1. full PFC is regenerated;
2. shape, seasonal, spot, PNG and Power BI audits are rerun;
3. an external review accepts the seam smoothing as an EEX-preserving operation;
4. thresholds are approved as market-governance settings rather than hidden code constants.

## Expert Review Summary

Three independent review streams converged on the same conclusion:

1. Do not relax quoted EEX BASE/PEAK calibration.
2. Add an explicit month-boundary jump operator; curvature seam penalties alone are insufficient.
3. Work on an additive correction `delta`, with `A delta = 0`.
4. Preserve PEAK explicitly, because BASE-only preservation can still break PEAK means.
5. Keep this in a dedicated LT module, not as another large block in the export script.

## Implemented Diagnostic Patch

Implemented module:

`pfc_shaping/lt/model/seam_nullspace_smoothing.py`

CLI flag:

`--enable-final-seam-nullspace-smoothing`

The implementation:

- builds local seam ramp corrections for month-boundary jumps above the threshold;
- projects the correction into the nullspace of BASE and PEAK bucket means;
- applies the same correction to slow/central/fast scenarios;
- recomputes weighted and structural fan columns;
- enforces negative floor and max weighted negative hours;
- writes an audit table with before/after seam jumps and max constraint residual.

## Full Integrated Reaudit Result

Final integrated CSV:

`output/ch_hfc_hourly_20260618_20301231_mit_seam.csv`

Observed audit results:

| metric | value |
|---|---:|
| shape_score_10 | 8.25 |
| hfc_vs_spot_score_10 | 8.25 |
| max_eex_base_error_eur_mwh | 0.000000 |
| max_eex_peak_error_eur_mwh | 0.000000 |
| boundary_jump_abs_p95_eur_mwh | 18.202333 |
| duck_2030_evening_minus_midday_eur_mwh | 26.168567 |
| negative_gate_status | PASS |
| seasonal_critical_flags | 0 |
| seasonal_warning_flags | 0 |
| monthly_path_critical_flags | 0 |
| monthly_path_warning_flags | 0 |
| calendar_critical_flags | 0 |
| calendar_warning_flags | 0 |

Largest corrected seams:

| seam | before | after | reduction |
|---|---:|---:|---:|
| 2027-09-01 00:00 | -35.459570 | -18.000001 | 17.459569 |
| 2028-04-01 00:00 | 32.291840 | 18.263167 | 14.028673 |
| 2028-11-01 00:00 | -30.540292 | -18.000000 | 12.540292 |

The result reaches the requested local/test `>= 8/10` threshold without sacrificing quoted EEX calibration.
