# External Audit Report - LT Quant Shaping Redesign Plan - 2026-06-18

Auditor role: external quantitative engineering / model-risk reviewer (independent).
Mandate: pre-implementation model-risk audit of the redesign blueprint, not a code audit.
Document under review: `.planning/phases/15-lt-quant-shaping-redesign/QUANT-SHAPING-REDESIGN-PLAN-20260618.md`.
Branch: `fix/lt-audit-remediation` (audited at the same tree merged into the working branch).
Plan self-score under review: `10/10 candidate for external audit`.

Scope note: this audit evaluates whether the plan is a rigorous, externally-defensible
implementation and validation blueprint. Per the audit prompt, generated CSV/Power BI
artifacts were not audited. Codebase claims (reusable modules, data assets) were spot-verified
only to test the plan's factual feasibility assertions.

---

## 1. Verdict

**CONDITIONAL PASS.**

The plan is audit-ready: it is precise enough to criticize line by line, it is grounded in
the mainstream HPFC/PFC literature, and its architecture (hard no-arbitrage constraints in a
convex QP, null-space smoothing, point-in-time anti-leakage contract, separation of the
deterministic curve from the probabilistic layer, and a full governance/manifest regime) is
methodologically sound. **There are no P0 findings**: the approach is not invalidated and the
document is reviewable.

However, the self-assigned `10/10` is not externally defensible. The plan's *quantitative
specification* — as opposed to its architecture — has multiple material gaps that an external
quant could legitimately require closed **before implementation starts** (regularization-weight
calibration, reduced-Hessian uniqueness, cross-border level-leakage scope, the probabilistic
measure (P vs Q) treatment, the scenario-to-quote reconciliation operator, statistical-test
corrections, and several undefined objects used directly in acceptance gates).

Recommendation: address the P1 set, then re-submit. The architecture does not need to change.

## 2. Score

**7.5 / 10.**

Justification:

- Architecture, no-arbitrage discipline, anti-leakage contract, and governance are genuinely
  strong — well above a typical internal plan. On those axes alone the plan is ~9/10. The
  +50 EUR/MWh neighbor-shift invariance test, the partial-horizon energy reconciliation, the
  KKT diagnostic requirements, and the run-manifest schema are sophisticated and correct in
  spirit.
- The deduction is concentrated in *quantitative specification precision*, which is exactly
  what an external model-risk function signs against:
  - the free parameters that actually control curve shape (the `lambda` weights) have **no**
    calibration, normalization, or identifiability procedure — despite the redesign being
    triggered by a wrong month path, which is governed by those weights;
  - the uniqueness condition for the QP (`N'HN ≻ 0`) is only gestured at, and the "documented
    ridge" is undefined in magnitude and repricing impact;
  - the cross-border level-leakage control is internally inconsistent (the projection spec is
    weaker than the invariance test it must satisfy);
  - the probabilistic layer conflates physical (P) and risk-neutral (Q) expectations and gives
    no scenario-to-quote reconciliation mechanism;
  - the statistical tests (Diebold-Mariano, bootstrap CIs, "material degradation") are not
    corrected for overlapping multi-horizon dependence or multiple testing;
  - several objects used *directly in acceptance gates and the objective* are undefined
    (`s_market`, "level-neutralized block error", the calibration basis of the 12 EUR/MWh seam
    gate).

A plan that is "specific enough for an external quant auditor to criticize line by line"
(its own DoD criterion) has been met; a plan that an external quant auditor would *pass at
10/10* has not.

## 3. Top Findings (material only, ordered by severity)

### P0 - blocks external-audit readiness

None. The approach is sound and the document is reviewable. (Stated explicitly because it is a
material conclusion: the redesign direction does not need to change.)

### P1 - must be fixed before implementation starts

**P1-1. Regularization-weight (`lambda`) calibration is entirely unspecified.**
§4.1 introduces `lambda_prior`, `lambda_smooth_m`, `lambda_smooth_h`, `lambda_seam`,
`lambda_calendar`. These weights *are* the control surface for month-path shape — the exact
quantity whose failure ("economically wrong month paths", §1) triggered the redesign. The plan
gives no selection method (L-curve, cross-validation, desk elicitation, Morozov discrepancy),
no normalization across terms with **different units and scales** (price levels in EUR/MWh vs
second differences of monthly means vs hourly second differences), and no sensitivity /
identifiability analysis. §3 principle 7 demands every threshold be "mathematically implied,
historically calibrated, or explicitly approved" — these weights violate that principle.
Without a calibration protocol the plan cannot prevent the very artifact it exists to fix.

**P1-2. QP uniqueness condition (`N'HN ≻ 0`) is gestured at, not pinned; the ridge is
undefined.** §4.1 says only that `H` is "positive semidefinite plus a documented
ridge/tie-breaker sufficient for a unique solution in the feasible null space." The smoothness
operators `D2` have non-trivial kernels (constants/linear sequences), so the smoothness-only
Hessian is provably rank-deficient — confirmed by the existing `ArbitrageFreeCalibrator`, which
already carries explicit rank-deficiency handling for its `sum (delta'')^2` objective. Uniqueness
therefore rests *entirely* on the prior term `||p - p_prior||^2_W` (only if `W ≻ 0` on every
free interval) and/or the ridge. The plan must state the **sufficient condition** for the
reduced Hessian `N'HN` to be positive definite, the **ridge magnitude and selection rule**, and
a **bound on the ridge's repricing/shape perturbation**. As written, the KKT system
`[H A'; A 0]` nonsingularity (the actual WP4 solve) is not guaranteed.

**P1-3. Cross-border level-leakage control is inconsistent between specification and test.**
§5.2 requires neighbor deviations to have weighted mean zero "for each relevant CH constrained
bucket" (`A_CH_bucket dev_M = 0`). But the cross-border term enters `p_prior`, and the hard
constraints pin only the *constrained* buckets. On **unconstrained residual months** — the
precise locus of the original failure — the curve follows `p_prior`, so a constant added to a
neighbor *can* shift a residual-month level unless the deviation is demeaned on residual months
too. The WP3 test (`+50 EUR/MWh` shift leaves CH output unchanged to `atol=1e-12`) is correctly
stated on the *final output* and is therefore **stronger** than the §5.2 projection spec it is
supposed to verify. Reconcile: the projection must demean on **every free aggregation** (all
reported buckets including unconstrained residual months and the global mean), not only the
constrained buckets.

**P1-4. Probabilistic layer conflates physical (P) and risk-neutral (Q) expectations.**
§4.6/§5/WP5 set `X_s,t = p_t + epsilon_s,t` and require the **scenario mean** to reprice the
forwards: `E_s[A_b X_s] = q_b`. But `p_t` is a *forward* (risk-adjusted, Q-measure) level,
whereas the scenario fan is described and used as a *physical* distribution (negative-price
probabilities, run lengths, P&L proxy — all P-measure objects). Forcing the physical mean onto
the risk-neutral forward injects the (sign-varying, seasonal) power risk premium into the
physical distribution and **biases tail probabilities** such as `P(price < 0)`, `P(< -10)`,
`P(< -30)` — which are then used as acceptance gates (§8). The plan must either (a) declare the
scenarios risk-neutral and used only for valuation, or (b) introduce an explicit risk-premium
term so physical scenarios may differ from the forward while a separate risk-neutral object
reprices `q`.

**P1-5. Scenario-to-quote reconciliation operator is unspecified, and the "paths vs marginal
quantiles" options are not interchangeable.** Enforcing `E_s[A_b X_s] = q_b` exactly while
simultaneously preserving (i) non-crossing quantiles and (ii) the calibrated negative-price tail
is non-trivial: naive sampling will not satisfy it, and a mean-shift to hit `q` distorts the
lower tail (hence `P(price<0)`). The plan asserts the constraint but gives no construction and
no test that the reconciliation **preserves tail calibration**. Separately, §4.6 offers
path-based scenarios *or* direct marginal quantiles as alternatives — but only the path-based
route supports the block-coherence requirement (`A_block X_s`, "not by summing hourly marginal
quantiles") and the mean-repricing requirement. Mandate paths as primitive; derive
marginals/quantiles from paths.

**P1-6. Statistical-test rigor is insufficient for an external model-risk sign-off.**
§WP6 proposes Diebold-Mariano "where sample size permits" and bootstrap CIs "over valuation
dates and delivery periods", and §8 gates on "no statistically material degradation." Three
corrections are required and currently absent: (a) with monthly snapshots and N+1/N+2/N+3
horizons the forecast errors are **overlapping and serially correlated**, so DM needs a HAC
(Newey-West) long-run variance and the small-sample Harvey-Leybourne-Newbold correction (or use
Giacomini-White conditional predictive ability with its stated conditions); (b) the bootstrap
must be a **block** bootstrap, not iid, or CIs will be understated; (c) "statistically material
degradation" must specify a significance level, a **pre-registered primary endpoint set**, and a
**multiple-testing control** (e.g., Romano-Wolf) across the many block x horizon x metric cells,
otherwise spurious degradations/improvements are guaranteed and the gate is not auditable.

**P1-7. Objects used directly in the objective and in gates are undefined.**
- `s_market` (the seam target in `||S p - s_market||^2`, §4.1) is never defined. Is it from
  forward history, the cross-border prior, or quoted quarter/cal completion? It drives the seam
  penalty and the "seam excess vs cross-border prior" gate.
- "Level-neutralized block error" / "shape-demeaned" comparison is the **headline validation
  metric** (§WP6) yet is never defined (subtract realized block mean? quoted forward level?
  per-delivery-period demean?). The definition determines whether it measures pure shape or
  shape+spread.
- The single concrete shape gate, "seam excess `> 12 EUR/MWh`", is explicitly "recalibrated by
  history before production" — i.e., the plan's one numeric defense against the triggering
  failure is currently uncalibrated and unsupported by evidence.

### P2 - can be fixed during implementation planning

**P2-1. Partial-horizon formula: energy vs settled-value ambiguity (§4.3).**
`q_remaining = (q_full * H_full - E_elapsed) / H_remaining` is correct only if `E_elapsed` is
*settled value* (sum of realized fixing x hour-weight, units EUR/MWh·h), not physical energy
(MWh) as "realized/locked energy" suggests. State the units, state `H_full = H_elapsed +
H_remaining` under the canonical UTC weighting, and state the settlement index the elapsed part
is averaged against.

**P2-2. Feasibility / rank determination (§4.2, §8).** `rank(A) == rank([A|q])` and
`||(I - A A+) q||_inf <= 1e-9` are not equivalent in finite precision; integer-rank equality is
brittle. Specify SVD-based numerical rank with a tolerance tied to the scaling/condition report.

**P2-3. Constraint scaling/preconditioning (§4.5).** Mixed Cal (~8760h), Quarter (~2160h),
Month (~720h) and Peak rows have very different norms; the dual/shadow values are not comparable
without normalization. Require energy-normalized (average-price) constraint rows and a
preconditioning step, not only a "scaling report."

**P2-4. Cross-border weight parameterization and overfit control (§5.2).**
`w_M(r,m,h)` spans 4 neighbors x 12 months x 24 hours x multiple regimes — thousands of free
weights. The plan shrinks the CH shape cells but not the cross-border *weights*. Specify the
functional form, sign/simplex constraints (may a neighbor weight be negative?), shrinkage toward
CH history, and an **out-of-sample weight-stability gate**. "All weights sum to one where
required" (§WP3 acceptance) is too vague.

**P2-5. Historical-regime vs future-regime weighting (§5.2).** Weights are conditioned on
*observed* NTC/congestion/scarcity regimes, but the LT curve is built over a *future* horizon
where those are unobserved. State how the future regime is chosen (climatology, scenario) and
how regime thresholds (scarcity / high-renewable / winter-stress / low-demand) are numerically
defined.

**P2-6. Hydro->price mapping and a water-value sign regression test (§5.4).** Hydro is named a
first-order driver but enters only as unspecified "interpretable coefficients." The reused
`water_value` module had a **sign defect fixed recently** (commit `1a0e641`,
"correct water value sign and block drift"); the plan must pin a regression test that locks the
water-value sign and direction so the defect cannot silently return.

**P2-7. Benchmark set is incomplete (§WP6).** Add the standard hard-to-beat forward benchmark
("last available forward curve carried forward" / no-change) and a climatological *probabilistic*
benchmark so CRPS/pinball skill scores are interpretable. A seasonal-naive deterministic
benchmark is also cheap and informative.

**P2-8. Manifest and waiver traceability (§7, §9).** Add `backtest_evidence_run_id` (the release
must reference the last backtest run, but the schema has no field for it), a model-version
(semver) distinct from `git_sha`, and a waiver register linking any `CONDITIONAL` verdict to
specific waiver IDs with expiry.

**P2-9. DoD self-contradiction (§12 item 4 vs §8).** "Flag-OFF path is byte-identical" conflicts
with the gate "Reproducibility flag OFF: numeric identity `atol=1e-12` on price columns."
Byte-identity is broken by trivial float/library differences. Reconcile on the `atol=1e-12`
numeric-identity wording.

**P2-10. RACI independence and external-auditor placement (§7).** State that the independent
quant validator is *organizationally* independent of the model owner (SR 11-7 / ECB TRIM
principle), name who commissions/accepts the external audit referenced in DoD item 11, and
assign challenger-model ownership.

**P2-11. Regulatory price floor as a hard scenario bound (§WP5).** Pin the EPEX/EUROPEX
day-ahead clearing price limits (e.g., the -500 EUR/MWh floor and its documented extension
mechanism) as a hard bound on deterministic and scenario tails, with the historical extension
events cited.

## 4. Missing Evidence

The following data, equations, thresholds, or governance artifacts must accompany the plan for
an external reviewer to sign:

1. **Forward-history snapshot density audit.** `data/eex_forwards_history.parquet` exists and
   contains CH/DE/FR/AT/IT with depth inferred to ~2019 (two-file ingestion incl. a
   "Historique2019" source). But the backtest power claim (`>= 36 valuation dates`,
   N+1/N+2/N+3, `>= 24 obs per critical block`) depends on **per-valuation-date CH product
   coverage and snapshot frequency**, which are *unverified*. Provide a coverage matrix:
   snapshots per year, products per snapshot, and CH BASE/PEAK quote count per candidate
   valuation date. If CH forward snapshots are sparse/illiquid pre-2021, the protocol's start
   date and statistical-power gates must be revised.
2. **Cross-border regime dataset acquisition plan.** `data/cross_border_regime_dataset.parquet`
   is confirmed **greenfield** — only Swissgrid NTC *baseline* loaders exist
   (`apply_swissgrid_ntc_baseline.py`, `audit_ntc_baseline_inputs.py`); no neighbor spot/flow/
   spread dataset is present. The plan lists it as a data contract but provides no sourcing
   (e.g., ENTSO-E Transparency for NTC/flows; neighbor day-ahead histories), licensing,
   point-in-time/vintage strategy, or effort/risk assessment. Because the entire cross-border
   component is gated `NO GO` without it, this is the critical-path dependency and must have an
   explicit acquisition + governance plan.
3. **Lambda-calibration protocol and sensitivity table** (see P1-1): an L-curve / CV /
   discrepancy procedure plus a documented sensitivity of month paths and seams to each weight.
4. **Reduced-Hessian uniqueness proof / ridge specification** (see P1-2): the condition that
   guarantees `N'HN ≻ 0`, the ridge value, its selection rule, and a repricing-impact bound.
5. **Risk-premium treatment** (see P1-4): either an explicit measure declaration or an estimated
   forward-spot premium term structure for CH.
6. **Numeric regime definitions** (scarcity / high-renewable / winter-stress / low-demand) and
   the historical calibration basis of the 12 EUR/MWh seam gate.
7. **Peak-calendar desk/legal confirmation** for CH/DE-LU/FR/AT/IT_NORD (the plan already gates
   this as `NO GO`; the signed confirmation itself is the missing artifact).
8. **Definition of "level-neutralized block error"** and of `s_market`, as formulae (see P1-7).

## 5. Recommended Patch (precise edits to the plan before implementation)

1. **§4.1** - Add the explicit mapping from the penalty form to `H` and `f`
   (`H = 2(lambda_prior W + lambda_smooth_m M'D2'D2 M + lambda_smooth_h D2'D2 + lambda_seam S'S
   + lambda_calendar C'C)`, `f = -2(lambda_prior W p_prior + lambda_seam S' s_market +
   lambda_calendar C' c_prior)`), **define `s_market`**, and add a subsection
   "4.1a Regularization Weight Calibration" specifying term normalization (energy-weighted,
   per-MWh), the selection method (L-curve / time-series CV on backtest blocks), and a mandatory
   weight-sensitivity report as a run artifact.
2. **§4.1 / new 4.1b** - State the uniqueness condition `N'HN ≻ 0`, require `W ≻ 0` on all free
   intervals **or** a ridge `epsilon I`, define the ridge magnitude rule and a gate that the
   ridge changes no monthly mean by more than a stated tolerance; cross-reference the existing
   `ArbitrageFreeCalibrator` rank-deficiency handling.
3. **§4.3** - Re-label `E_elapsed` as `V_elapsed` (settled value, EUR/MWh·h), add
   `H_full = H_elapsed + H_remaining`, and name the settlement index.
4. **§4.2 / §8** - Replace integer-rank equality with SVD numerical rank at a tolerance tied to
   the condition report; keep the projector-residual test as the numerical gate.
5. **§4.5** - Require energy-normalized constraint rows and report the post-scaling condition
   number; make dual/shadow values comparable across bucket types.
6. **§4.6 / §5 / §WP5** - (a) Declare the scenario measure (P vs Q) and add a risk-premium term
   if scenarios are physical; (b) make scenario *paths* primitive and derive quantiles from them;
   (c) specify the scenario->quote reconciliation operator (e.g., affine moment-match) and add a
   test that it preserves non-crossing and `P(price<0)/P(<-10)/P(<-30)` calibration; (d) pin the
   EPEX/EUROPEX price floor as a hard tail bound.
7. **§5.2** - Strengthen the projection to "weighted mean zero on every free aggregation
   bucket, including unconstrained residual months and the global mean," so it matches the
   `+50 EUR/MWh` invariance test; add the weight parameterization, sign/shrinkage constraints,
   and an out-of-sample weight-stability gate. Add numeric regime definitions and the
   future-regime selection rule.
8. **§5.4** - Specify the hydro->price mapping form and add a water-value **sign regression
   test** referencing commit `1a0e641`.
9. **§WP6 / §8** - Define "level-neutralized block error" as a formula; add HAC + HLN-corrected
   DM (or Giacomini-White CPA), block bootstrap, a pre-registered primary-endpoint set, and a
   multiple-testing correction; quantify "statistically material degradation" (level + family
   error rate); add no-change and climatological benchmarks; replace the placeholder 12 EUR/MWh
   seam gate with a history-calibrated value (or label it provisional with the calibration job
   named).
10. **§7 / §9** - Add `backtest_evidence_run_id`, a model semver, validator-independence
    language, the external-auditor commissioner, and a waiver register; **§12 item 4** - replace
    "byte-identical" with the `atol=1e-12` numeric-identity wording to match §8.
11. **§2 / cover** - Add an "Open data dependencies / critical path" subsection naming the
    greenfield `cross_border_regime_dataset` acquisition plan and the forward-history coverage
    audit as prerequisites, and **lower the self-score** from `10/10` to reflect the open P1 set
    (the document's own DoD criterion is "criticizable line by line", which is met; "passes
    external audit at 10/10" is not yet met).

---

### Appendix A - Feasibility verification performed for this audit

The plan's reuse and data claims were spot-checked against the repository:

- `calibration_buckets`, `ArbitrageFreeCalibrator` (KKT/Schur + scipy `splu`, smoothness
  objective `sum (delta'')^2` with rank-deficiency handling), `quote_aware_monthly_smoothing`
  (null-space KKT monthly deltas), `water_value`, `solar_modulation`, `electrification_shape` -
  **all exist** as claimed; the proposed `pfc_shaping/lt/model/` package already exists.
- `pfc_shaping/ct/` exists; the "do not import ct.* into LT" rule is enforceable.
- `data/eex_forwards_history.parquet` **exists** (CH/DE/FR/AT/IT; depth inferred ~2019+), but
  snapshot *density*/per-date CH coverage is unverified -> Missing Evidence #1.
- `data/cross_border_regime_dataset.parquet` is **greenfield** (only Swissgrid NTC baseline
  loaders exist) -> Missing Evidence #2.
- `scripts/export_local_test_ch_hourly_csv.py` is ~2200 lines, corroborating the §6 rationale to
  stop expanding it.

These confirm the plan is *feasible and well-grounded in existing assets*; the findings above
concern specification precision and quantitative correctness, not buildability.
