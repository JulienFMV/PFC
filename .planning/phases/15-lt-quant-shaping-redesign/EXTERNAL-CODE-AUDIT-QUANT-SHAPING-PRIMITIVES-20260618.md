# External Code Audit - LT Quant Shaping Primitives - 2026-06-18

Auditor role: independent quantitative engineering / model-risk reviewer.
Scope: the two pushed commits implementing the redesign primitives.

```text
74156b8 feat(lt): add quant shaping contract primitives
689d713 fix(lt): store aligned physical risk premium
```

Reviewed against `QUANT-SHAPING-REDESIGN-PLAN-20260618.md` and the prior plan audit.
This is a code audit: the implementation was read line by line, the test suites were
executed in a clean venv, and the scalability claim was benchmarked. No PR was opened.

---

## 1. Verdict

**Scientifically directionally aligned; implementation partially aligned; production
behaviour NOT yet aligned.** (Gate label: CONDITIONAL PASS, foundation layer.)

- **Science / direction - aligned.** The chosen method - a smoothed optimization under
  hard EEX averaging constraints, cross-border-as-shape, foundation-model-as-benchmark,
  structural negatives - matches Fleten-Lemming, Kiesel-Paraschiv, Keles, PriceFM, and
  Eurelectric. The plan is faithful to the literature.
- **Implementation - partially aligned.** The constraint engine, sparse KKT solver, and
  leakage-safe primitives exist and are mathematically correct (30 new tests pass, the
  historical LT suite is green, the reported remediations are all verifiable). But the
  *core mechanism the literature is actually about* - the smoothness / seam / calendar
  penalties and the prior estimator - is not built or not active, and there is one
  scalability defect (dense cross-border projector).
- **Production behaviour - NOT yet aligned.** With `lambda_smooth_h = 0` by default and no
  month-mean / seam penalty implemented, the optimizer currently reduces to a *projection*:
  it applies a **flat, piecewise-constant level shift per quoted bucket** ("flat monthly
  deltas"). Such a solve can reprice every EEX average exactly and still produce
  artificial discontinuities - steps at quoted-bucket boundaries (i.e. at month-end
  midnights) and intra-day breaks inherited from a blocky prior. This is precisely the
  artifact that triggered the redesign, and it is exactly what the smooth-forward
  literature forbids. **No test in the suite can currently detect it.** See finding P1-0.

The internal "10/10 from two expert agents" is defensible **within the primitives
perimeter** but must not be read as the redesign being complete: on its central objective -
producing a smooth, economically defensible intra-curve - the delivered behaviour is not
yet aligned.

## 2. Score

- **Quality of implemented primitives: 8.5 / 10.** Clean, well-typed, correctly
  formulated, strong unit tests, good module separation, no `pfc_shaping.ct` coupling.
  Docked for the dense-projector defect and a manifest that is still a skeleton vs plan
  Section 9.
- **Completeness vs the plan: ~30%.** Constraint engine, deterministic optimizer core,
  cross-border projection, negative-price scenario primitives, and the data contract are
  in. The prior model, the shape-shaping penalties that actually fix bad month paths, the
  validation/backtest layer, governance manifest completeness, and Power BI are out.

## 3. What was verified (claims corroborated)

| Claim | Result |
|---|---|
| New quant tests: 30 passed | **Confirmed** - `30 passed in 0.72s` in a clean venv |
| Historical LT tests pass | **Confirmed** - core LT model suite `81 passed, 8 skipped`, no failures (exact 58/1 depends on file selection) |
| No `pfc_shaping.ct` import in new LT/data modules | **Confirmed** - grep clean; `test_lt_ct_imports.py` green |
| Sparse (not dense) QP solver | **Confirmed** for the optimizer (`scipy.sparse` KKT + `spsolve`) - but see P1-1 for cross-border |
| PEAK quotes no longer omitted | **Confirmed** - `missing_raw_quotes` guard raises if any PEAK quote is unrepresented |
| Negative runs on correct axis | **Confirmed** - `apply(_max_negative_run, axis=0)` = per scenario over time |
| Aligned physical risk premium (689d713) | **Confirmed** - stores reindexed `rp`, not the raw input series |
| Cross-border level-leakage fixed | **Confirmed** - global level row makes the projector annihilate constants; `+50` invariance holds even with partial constraints |

## 4. Findings

### P0 - none

Nothing is wired into production; there is no live curve to corrupt. The math that is
present is correct. No showstopper.

### P1 - must fix before the module is used on a real horizon

**P1-0. (HEADLINE) The deterministic solve currently produces flat per-bucket monthly
deltas, so EEX-consistent curves can still carry artificial midnight / seam breaks - the
exact failure the redesign targets, and the suite cannot detect it.**

With `lambda_smooth_h = 0` (default) and no `lambda_smooth_m` / `lambda_seam` /
`lambda_calendar` implemented, `QuantShapeOptimizer.solve` reduces to the W-weighted
projection of the prior onto the constraints:

```text
min ||p - p_prior||^2_W   s.t.   A p = q
=>  p = p_prior + W^-1 A^T lambda,   A W^-1 A^T lambda = q - A p_prior
```

For disjoint duration-weighted average buckets with `W = I`, this is a **piecewise-constant
level shift**: `p_t = p_prior,t + c_b` for every hour `t` in bucket `b`, with
`c_b = q_b - mean_b(p_prior)`. Consequences:

1. The correction is **flat inside each bucket** ("deltas mensuels plats") and
   **discontinuous across bucket boundaries**. The step at a boundary is
   `c_{b+1} - c_b = (quoted spread) - (prior spread)`. Because the prior never matches the
   quoted month spread exactly, **every quoted-month boundary - i.e. a month-end midnight -
   carries a step.** Matching EEX averages provides *zero* protection against this; it is
   structural to a projection-only solve.
2. Inside a bucket the curve is exactly the prior shape, so any intra-day / midnight steps
   in a blocky (peak/offpeak/hour-cell) prior survive untouched.

Hence a curve can reprice every EEX BASE/PEAK average to `1e-9` and show a clean monthly
path, yet still break at midnights - the reported symptom. The smooth-forward literature
explicitly rejects this: Fleten-Lemming (and Benth et al. maximum smoothness,
Kiesel-Paraschiv) define the curve as the **smoothest** function (minimize the integral of
squared curvature) consistent with the averaging constraints, precisely to turn these
boundary level-differences into smooth ramps and remove the "swell"/break. A projection-only
solve is the pre-literature "match averages, keep prior shape" approach the redesign was
created to replace (cf. plan Anti-Pattern: "fix the chart by manually anchoring").

The architecture already supports the fix (the KKT solve smooths in the null space of the
quotes), so the work is: (a) implement and **activate** `lambda_smooth_h > 0` plus a
month-mean second-difference term and/or a seam term; (b) **calibrate** the lambdas (the
open P1 from the plan audit - they are currently free and default to no smoothing); and
(c) add a **continuity / smoothness acceptance test**. Today no test asserts continuity:
`test_lt_quant_optimizer_kkt.py` checks only repricing and KKT residuals, so this artifact
is not merely unfixed, it is **undetected**. Add a test that bounds the curve's first/second
differences at and around quoted-bucket boundaries (above the prior's intrinsic step), and a
flag-level seam gate, before any pipeline wiring.

This is the top finding: it is the redesign's central objective, it outranks the
scalability defect below (which sits on a module not yet on the critical path), and it is
why the verdict's "production behaviour" axis is NOT aligned.

**P1-1. Cross-border projector builds dense `n x n` matrices - infeasible at production
scale.** In `pfc_shaping/lt/model/cross_border_shape.py::project_neighbor_deviations`:

```python
winv = np.diag(1.0 / w)                                   # n x n dense
projector = np.eye(len(neighbor_curves)) - winv @ a_projection.T @ np.linalg.pinv(middle) @ a_projection
projected = projector @ values                            # n x n dense
```

`np.diag(1/w)`, `np.eye(n)`, and the explicit `projector` are all dense `n x n`. Measured
peak memory scales as ~3x n^2 (n=4000 -> 384 MB). Extrapolated:

| Horizon | n | single dense `n x n` (float64) |
|---|---:|---:|
| 1 year hourly | 8,760 | 0.61 GB |
| 5 years hourly | 43,800 | 15.35 GB (peak ~3x -> ~46 GB) |
| 5 years 15-min | 175,200 | 245 GB |

This OOMs long before the plan's "<= 30 min on a desk machine" runtime gate. It passes
today only because every test uses `n = 4`. The math is already correct - the `pinv` is on
the small `k x k` matrix `middle` - so the fix is purely to apply the projector
matrix-free:

```text
projected = values - (1/w)[:,None] * (A_proj.T @ (pinv(middle) @ (A_proj @ values)))
```

using only `(k x n)` and `(k x k)` objects (k = #constraints << n), never an `n x n`.
Add a test at a realistic `n` (e.g. 8760) that asserts a bounded peak-memory / wall-clock
budget so this regression cannot return.

### P2 - fix during continued implementation

**P2-1. Run manifest is a skeleton, far below plan Section 9.** `validate_run_manifest_schema`
requires only 7 fields and checks `available_at <= valuation`. The plan's contractual
manifest has ~25 fields (config hash, package versions, seeds, curve_csv sha, quote
snapshots, rank/kkt report paths, optimizer status, Power BI ids, artifact hashes). Track
the gap explicitly so the skeleton is not mistaken for the governance artifact; add a
`backtest_evidence_run_id` per the plan audit.

**P2-2. Mixed-granularity BASE/PEAK yields overlapping, not disjoint, constraints.**
`build_base_peak_offpeak_constraint_system` applies the clean disjoint PEAK/OFFPEAK
transform only when BASE and PEAK share the *same* bucket label. When granularities differ
(e.g. BASE Cal + PEAK month) it keeps the whole BASE row and adds standalone PEAK rows that
overlap it. This is mathematically valid (consistent average constraints, handled by the
rank/feasibility path) but departs from the plan's Section 4.4 disjoint convention and
produces overlapping rows whose duals are harder to interpret. Document the behaviour and
add a dedicated `test_lt_quant_peak_offpeak.py` (named in the plan but absent) covering the
mixed-granularity case.

**P2-3. No explicit subset assertion that PEAK intervals lie inside the matching BASE
bucket.** The disjoint decomposition relies on `peak_idx subset all_idx` so that
`h_base = h_peak + h_offpeak`. It holds in practice (same `calibration_buckets` semantics)
but is unguarded; a defensive assertion would harden it.

**P2-4. Scenario reconciliation can breach the regulatory price floor.**
`reconcile_q_scenario_mean_to_quotes` shifts every path by a per-timestamp additive
constant to hit the quote mean; it does not clamp to the EPEX/EUROPEX clearing floor
(e.g. -500 EUR/MWh). Add a post-adjustment bound check / flag (consistent with the plan's
"market price limits" requirement).

**P2-5. `build_physical_paths` silently defaults to a zero risk premium.**
`allow_diagnostic_without_rp=True` sets `rp = 0` and labels the bundle `"diagnostic"`. The
label is the only guard. For a production entry point, default to requiring an explicit
risk premium (`allow_diagnostic_without_rp=False`) so P-measure paths cannot silently
collapse onto the Q curve.

**P2-6. Optimizer ridge regularizes toward zero, not toward the prior.** `H` gets
`2*epsilon*I` but `f` only contains the prior term, so the ridge biases the solution toward
`0` rather than `p_prior`. Harmless at `epsilon=1e-10` (and the strictly-positive prior
weight already guarantees a unique solution), but it is a (tiny) systematic pull; either
fold the ridge into the prior weight or document that it is a pure tie-breaker.

**P2-7. (PROMOTED to P1-0)** The deterministic optimizer's missing/inactive shape-shaping
penalties (month smoothness, seam, calendar) were originally filed here as a scope item.
They are in fact the production-behaviour blocker and are now the headline finding P1-0;
this entry is retained only as a cross-reference.

**P2-8. Missing test files named in the plan matrix.** `test_lt_quant_dst_time_index.py`,
`test_lt_quant_flag_off_identity.py`, `test_lt_quant_backtest_protocol.py`,
`test_powerbi_manifest_binding.py`, and the dedicated peak/offpeak file are absent. DST is
handled correctly in principle (UTC index rejects duplicates; local conversion only for
masks), but it is asserted nowhere; add an explicit 23h/25h DST construction test.

**P2-9. Import coupling.** Importing `pfc_shaping.data.lt_data_contract` pulls in the whole
`pfc_shaping.data.__init__`, which transitively imports `yaml`/databricks. For a module
billed as "small and explicit," consider keeping the contract importable without the heavy
package init.

## 5. Recommended next steps (in order)

1. **Close P1-0 first - it is the redesign's reason to exist.** Implement and activate the
   month-mean / seam smoothness penalties (and `lambda_smooth_h > 0`) in the null space of
   the quotes; calibrate the lambdas (plan-audit P1); and add a **continuity / seam
   acceptance test** that bounds first/second differences at quoted-bucket boundaries so a
   flat-delta staircase with midnight breaks fails CI. Build `shape_priors.py` alongside,
   ensuring the prior is itself continuous across day/midnight boundaries.
2. Make `project_neighbor_deviations` matrix-free (P1-1) and add a realistic-`n`
   memory/time budget test.
3. Add the remaining missing test files (P2-8), especially DST and flag-off identity,
   before any pipeline wiring.
4. Promote the manifest skeleton to the full Section 9 schema (P2-1) as governance work
   begins.
5. Keep the primitives unwired from production until the prior model + active smoothness
   penalties + a continuity gate + a backtest exist; the plan's Definition of Done and the
   prior plan audit's P1 set remain the gate.

---

### Appendix - audit method

- Read all five new modules (`shape_constraints`, `quant_shape_optimizer`,
  `cross_border_shape`, `negative_price_regime`, `lt_data_contract`) and all five new test
  files line by line.
- Ran the suites in a clean Python 3.11 venv (numpy 2.0.2, scipy 1.13.1, pandas 2.3.3,
  scikit-learn 1.6.1, holidays 0.99, pytest 9.1.0): new quant tests `30 passed`; core LT
  model suite `81 passed, 8 skipped`.
- Verified KKT math (objective <-> penalty mapping, RHS sign, full-vs-independent-row
  repricing, redundant-row reduction, infeasibility hard-fail) and the cross-border
  projector identity `P*1 = 0` that guarantees `+50 EUR/MWh` shift invariance.
- Benchmarked `project_neighbor_deviations` at n in {500,1000,2000,4000} to confirm the
  dense `n x n` scaling and extrapolate to LT horizons.
- Confirmed no `pfc_shaping.ct` imports and no production/script wiring of the new modules.

---

## Addendum (2026-06-18b) - graduate-level review of commit `2aeff68` (seam smoothness)

Context: follow-up commits on `fix/lt-audit-remediation`:
`a52eefb` (matrix-free cross-border projection), `6ee36b0` (legacy residual-anchor patch),
`2aeff68` (add `lambda_smooth_m` + `lambda_seam` and a continuity test).

### Status changes
- **P1-1 RESOLVED.** `a52eefb` rewrites the projector matrix-free
  (`projected = values - (1/w) * Aᵀ (A W⁻¹ Aᵀ)⁺ A values`); no `n x n` is formed. The
  `+50 EUR/MWh` invariance test still passes. Confirmed correct.
- **P1-0 mechanism DELIVERED but not yet trustworthy.** `lambda_smooth_m` (curvature of
  monthly means) and `lambda_seam` (jump at month boundaries) exist, default OFF, are PSD
  (`opᵀop`), preserve exact repricing, and the new `test_lt_quant_curve_continuity.py`
  proves they collapse a ~10 EUR/MWh flat-delta cliff to < 0.1 while holding month means.
  33 quant tests pass. However the seam term is theoretically mis-specified and tz-fragile
  (Q1, Q2 below), so the "production behaviour" axis remains NOT aligned.

### New findings (quant level)

**Q1 (important). `lambda_seam` penalizes the first difference (level gap), not curvature,
so it over-flattens legitimate cross-boundary trends.** Smooth-forward theory
(Fleten-Lemming; Benth-Koekebakker-Ollmar maximum smoothness) minimizes the integral of the
squared *second* derivative subject to averaging constraints - this removes spurious cliffs
*without* penalizing genuine slope. A first-difference penalty at the seam cannot distinguish
a flat-delta cliff from a real seasonal ramp and suppresses both. Empirical proof, linear
monthly targets {Jan 10, Feb 20, Mar 30}, exact repricing in both runs:

| | curvature-only (`lambda_smooth_h`) | full (+`lambda_seam`=1000) |
|---|---:|---:|
| Feb-end -> Mar-start values | 23.97 -> 26.04 (natural ramp) | 24.99 -> 25.03 (flattened) |
| max month-boundary step | 2.06 | 0.03 |

The seam penalty converts a true +10/month trend into an artificial month-end plateau, with
compensating curvature pushed inside the month. Recommendation: prefer curvature continuity
across seams (the global `lambda_smooth_h` second difference already spans boundaries); if
extra seam emphasis is wanted, up-weight the curvature stencil rows that straddle a boundary
rather than adding a level-gap term. If `lambda_seam` is kept, bound it and add a
trend-preservation test.

**Q2 (latent correctness). Month bucketing follows the display timezone of `prior.index`;
it is only correct for a CH-local index, and is off by the UTC offset for a UTC index - the
canonical internal form.** `_calendar_month_period` does `index.tz_localize(None)`, i.e.
it buckets by whatever wall-clock the index carries. EEX CH delivery months are local. The
CH February boundary is `2030-01-31 23:00 UTC`. A Zurich-indexed prior places the seam there
(correct); a **UTC-indexed prior places it at `2030-02-01 00:00 UTC`** - one hour late (two
in summer) - and the smoothed "monthly mean" then aggregates a UTC month that does not match
the hard constraint's local-month bucket. `shape_constraints.validate_utc_index` mandates UTC
as the canonical form, so the natural usage is exactly the broken one; the continuity test
passes only because it uses a Zurich index throughout. Fix: localize the penalty calendar to
the same market tz/`country` the constraint builder uses (`tz_convert`, not `tz_localize`),
and assert the convention.

**Q3 (calibration/design). `lambda_smooth_h` and `lambda_seam` overlap near seams and are
uncalibrated; the seam term dominates.** In the probe, `lambda_seam` drove the boundary step
from the natural 2.06 to 0.03 - i.e. it, not the curvature term, sets near-seam shape. With
relative weights unset, near-seam behaviour is determined by an arbitrary ratio. Calibration
(L-curve / backtest) must set `lambda_smooth_h : lambda_smooth_m : lambda_seam` jointly, not
independently.

**Q4 (minor). The monthly-mean second difference assumes equal month spacing.** The stencil
`[1,-2,1]` on consecutive month means treats 28- and 31-day months as evenly spaced
(~10% curvature distortion for February). A divided-difference (centroid-spacing) weighting
is the rigorous form. Second order; document or refine.

**Q5 (minor). Endpoint effect.** Second-difference penalties have a linear null space, so the
first and last months are less constrained by smoothing; with `lambda_prior << lambda_smooth`
the endpoints can drift. Pin them via the prior weight or an explicit endpoint condition.

**Q6 (test scope).** `test_lt_quant_curve_continuity.py` is a good existence proof but narrow:
a symmetric box, a flat prior, all-quoted months, and `lambda`s (1000) that swamp the prior.
It does not test (a) trend preservation (the Q1 failure), (b) residual/unquoted-month
interaction, (c) absence of new intra-month spikes, or (d) a UTC-indexed prior (the Q2 bug).
Add these before calibration.

Non-issue checked: arithmetic vs energy-weighted month mean is *not* a problem here because
`interval_hours` enforces a uniform cadence, under which the two coincide; it would only
matter if mixed cadence were ever allowed.

### Scope caution (outside the audited commit)
`6ee36b0` adds "preserve residual anchor after smoothing" to
`scripts/export_local_test_ch_hourly_csv.py` - continued investment in the legacy
residual-anchor script that plan WP0/Section 6 explicitly says to stop expanding. Not part of
the new quant path, but it is the anti-pattern the redesign was created to retire; keep new
shape logic in the `pfc_shaping/lt/model` modules.

### Verdict (unchanged axis, refined)
Science: aligned. Implementation: now has the right *class* of tool, but the seam term is
mis-specified vs theory (Q1) and tz-fragile (Q2). Production behaviour: still NOT aligned -
the mechanism is available and tested, not yet correct-by-construction or calibrated. Close
Q1/Q2, add the Q6 tests, then calibrate `lambda_*` on a backtest before any CLI wiring.
