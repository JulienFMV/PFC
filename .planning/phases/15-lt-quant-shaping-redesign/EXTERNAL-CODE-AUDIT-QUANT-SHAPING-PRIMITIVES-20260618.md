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

**CONDITIONAL PASS (foundation layer).**

The code that exists is genuinely high quality: correct convex-QP/KKT math, a clean
average-price constraint engine, a mathematically sound zero-mean cross-border projector,
a correct P-vs-Q separation in the scenario layer, and disciplined point-in-time data
contracts. All 30 new quant tests pass and the historical LT suite is green, so the work
does not regress existing behaviour. The remediations the internal expert agents reported
(sparse QP solver, PEAK-quote representation, negative-run axis, manifest validation,
cross-border level-leakage) are all present and verifiable in the code.

Two qualifications keep this from an unconditional pass:

1. **One material defect (P1):** the cross-border projector materializes dense `n x n`
   matrices. It is correct at the `n = 4` test sizes but is memory-infeasible at any real
   LT horizon - the *same* dense-matrix failure class the team already fixed in the QP
   optimizer, missed in `cross_border_shape.py`. It must be made matrix-free before the
   module is run on a production curve.
2. **Scope:** this is the primitives foundation, not the redesign. The prior model
   (`shape_priors.py`), the global optimizer's shape penalties (month-path smoothness,
   seam, calendar), the backtest, the CLI runners, the full run manifest, and the Power BI
   binding are not implemented. Nothing is wired into a production path yet, so current
   production risk is zero - and the plan's Definition of Done is far from met.

The internal "10/10 from two expert agents" is defensible **within the primitives
perimeter**; it should not be read as the redesign being complete or production-ready.

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

**P2-7. Deterministic optimizer lacks the shape-shaping penalties.** `QuantShapeOptimizer`
implements only the prior term and an optional hourly second-difference smoother. The
month-path smoothness, seam, and calendar penalties from plan Section 4.1 - the terms that
actually defend against the "economically wrong month path" that triggered the redesign -
are not yet present. This is expected for a primitives commit but should be the next
priority, not deferred behind cross-border/scenario work.

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

1. Make `project_neighbor_deviations` matrix-free (P1-1) and add a realistic-`n`
   memory/time budget test.
2. Implement `shape_priors.py` and the deterministic optimizer's month/seam/calendar
   penalties (P2-7) - this is the core of the redesign's stated purpose.
3. Add the missing test files (P2-8), especially DST and flag-off identity, before any
   pipeline wiring.
4. Promote the manifest skeleton to the full Section 9 schema (P2-1) as governance work
   begins.
5. Keep the primitives unwired from production until the prior model + optimizer penalties
   + a backtest exist; the plan's Definition of Done and the prior plan audit's P1 set
   remain the gate.

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
