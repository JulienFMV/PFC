# Phase 14 Decision Log - LT Monthly Forward Curve Reform

Append-only. New decisions must preserve the fields: decision, reason,
rejected alternatives, invariants not to break.

## D-20260622-01 - Solver Monthly Level Authority

Decision: when `monthly_level_authority="solver"`, the monthly BASE curve
returned by the monthly forward solver is the level authority for LT assembly.
Hourly, weekday, intraday, water-value and bridge layers may shape within each
month, but must not change the all-hour monthly mean before final evidence
generation.

Reason: the original defect was caused by level construction being rewritten by
independent cascade/smoothing/post-process steps. A global monthly constrained
solve only fixes the defect if downstream layers respect its monthly means.

Rejected alternatives:

- Let legacy `f_S` seasonal fallback apply after the solver.
- Let `_stabilize_raw_curve` shrink solver monthly levels toward annual-flat
  structure.
- Patch individual months in the generated curve or Power BI output.

Invariants not to break:

- CH EEX quoted products remain hard constraints in the monthly solve.
- Synthetic solver months must not become original traded quote keys.
- Solver mode must fail fast if legacy cascade or legacy BASE smoothing remains
  active.

## D-20260622-02 - No Naive Month-by-Month CAL Ordering Rule

Decision: do not impose a hard rule that if `CAL_y > CAL_{y+1}` then every
month in year `y` must be above the same month in `y+1`.

Reason: with active sub-products such as `CAL2028 + Q1-2028`, the economically
comparable block for Apr-Dec is an implied residual, not the full calendar. A
same-month inversion can be legitimate only if supported by active quotes,
comparable-block math, or calibrated history/panel evidence. It cannot be
accepted just because a visual chart looks smoother.

Rejected alternatives:

- Force monotonic same-month ordering from calendar spreads alone.
- Compare a residual Apr-Dec block directly to a full calendar year.
- Accept visual PASS without numerical comparable-block gates.

Invariants not to break:

- Same-month gates must report parent block, parent mean, month deviation and
  quote support.
- Far-horizon `UNSUPPORTED` must remain explicit where history cannot calibrate
  P90/P97.5 thresholds.

## D-20260622-03 - Lambda(t) Prior Contract

Decision: structural `Lambda(t)` is a soft zero-mean monthly shape prior:
raw template deviations are recentered inside CH parent blocks, shrunk if
configured, optionally capped in the recentered zero-mean space, then recentered
again to preserve parent means.

Reason: external or structural evidence must influence only allocation inside
the CH quoted parent block. Applying a cap before recentring can make the final
shape depend on a level offset in the template ratios and can exceed the
advertised cap after recentring.

Rejected alternatives:

- Apply cap in raw ratio space before parent recentering.
- Silently clamp invalid cap/shrinkage parameters.
- Emit diagnostics that merely state `zero_mean_parent_space=True` without
  parent weighted means.

Invariants not to break:

- `Lambda(t)` never enters hard constraints.
- Diagnostics must expose parent hours, pre-recenter parent mean, final parent
  mean and max parent mean residual.
- Defaults remain no cap and zero shrinkage unless config explicitly wires more.

## D-20260622-04 - Handoff Hygiene

Decision: `AGENTS.md` is the canonical cross-agent contract. `CLAUDE.md` only
points to it. Phase 14 durable decisions live in this file. Session handoffs use
`SESSION-HANDOFF-YYYYMMDD-*.md` under this phase folder.

Reason: the repo has many generated reports and superseded candidates. Without
a short canonical state and decision log, a new agent can resume from an
obsolete diagnostic artifact.

Rejected alternatives:

- Keep decisions only in chat.
- Duplicate permanent rules in both `AGENTS.md` and `CLAUDE.md`.
- Use `.planning/CONTEXT.md` as permanent memory while it contains older
  phase-specific context.

Invariants not to break:

- Every phase closeout must include exact commands, results, touched files,
  artifacts, config values and unresolved risks.
- Context should be compacted or handed off around 60%, not near exhaustion.

## D-20260622-05 - Candidate Evidence Must Be Layered

Decision: a sparse monthly solver proof is necessary but not sufficient to
declare the delivered CH LT graph corrected. Phase 2 acceptance must compare
the monthly solver curve, assembler `B`, delivered `price_shape`, hourly CSV
and dashboard sidecars. A clean KKT/repricing proof cannot override failed
delivered-curve audits.

Reason: on 2026-06-22 the solver proof repriced BASE constraints and passed
neighbor-leakage checks, while the delivered local-test candidate still failed
hourly/Power BI diagnostics with PEAK residuals and cross-year month-shape
critical flags. The defect can therefore live downstream of the monthly solve
or in non-BASE/hourly audit layers.

Rejected alternatives:

- Declare Phase 2 green from `audit_gates.csv` alone.
- Treat Power BI sidecars generated with `--allow-failed-gates` as promotion
  evidence.
- Ignore `price_shape` / CSV diagnostics because `B` equals the solver curve.

Invariants not to break:

- `--allow-failed-gates` remains diagnostic-only.
- Handoffs must report both solver proof status and delivered-curve audit
  status.
- `price_raw` is not currently exposed by standard artifacts; do not claim it
  was verified unless a diagnostic artifact records it.

## D-20260622-06 - Lambda And Prior Hashes Are Not Yet Promotion Proof

Decision: until the hash contract is widened, matching `active_config_hash` or
`monthly_solution_hash` is not sufficient proof that the selected lambda/prior
artifact governs the run.

Reason: Phase 2 read-only review found that the active config hash covers the
core `MonthlyCurveConfig` but omits material prior knobs such as
`panel_weight`, `history_weight`, `structural_weight`,
`allow_template_structural_fallback`, `structural_amplitude_eur_mwh`, and
`min_structural_snapshots`. The sparse proof also uses a different default
`history_weight` than the local export wrapper unless explicitly overridden.

Rejected alternatives:

- Accept hash equality as a lambda-calibration gate without checking the
  selected artifact status and prior-stack knobs.
- Treat structural template defaults as calibrated market evidence.

Invariants not to break:

- Promotion must cite manifest-backed production, export and selected-lambda
  artifacts.
- Far-horizon `UNSUPPORTED` remains explicit and cannot hide a known-bad
  fixture failure.
- Candidate diagnostics must record the exact prior weights used.

## D-20260622-07 - Structural Lambda Activation Requires Evidence, Not Just Defaults

Decision: activating `allow_template_structural_fallback=True` is acceptable
only when the diagnostics and monthly authority manifest expose the structural
source, fallback reason, amplitude, history counts, parent zero-mean residuals,
and full prior-stack config hash inputs. A default flip without this evidence
is not promotion proof.

Reason: the pushed commit `c7e8ab6` had a test expecting structural template
fallback while the pushed code still had the fallback disabled. Local green
tests were also not reproducible in cloud because
`tests/test_build_powerbi_exports_script.py` was not tracked. Expert review
identified that the structural template is a material far-horizon model change,
so hidden defaults or incomplete hashes would replace a flat curve defect with
an unauditable prior.

Rejected alternatives:

- Treat `allow_template_structural_fallback=True` as sufficient by itself.
- Keep structural fallback reason and history support only in in-memory prior
  diagnostics.
- Continue using a narrow active config hash that omits material prior knobs.
- Let masked lambda calibration months fall back to an implicit zero baseline.

Invariants not to break:

- Structural fallback remains zero-mean in parent space and must report max
  parent residuals.
- The monthly authority manifest must expose `structural_prior_summary`.
- `active_config_hash` must change when material structural prior knobs change.
- Point-in-time lambda calibration must not score withheld monthly products
  against synthetic zero levels created by masking.

## D-20260622-08 - Delivered Hourly Shaping Gates Are Separate From Monthly BASE Authority

Decision: PEAK repricing and structural fan-chart bridge correctness are
delivered-hourly invariants. They must be enforced at the final CSV/export
boundary and audited separately from the monthly BASE solver. Passing monthly
BASE constraints does not imply a promotable CH HFC.

Reason: Phase 3 expert roasts and local diagnostics showed that the monthly
BASE chain was not the source of the delivered-curve PEAK failure. The Phase 2
CSV missed quoted PEAK products because the existing BASE+PEAK calibration was
not enabled, and a final mutator could also run after an earlier PEAK
projection. The same investigation showed that inverted quantile rows came
from export/Power BI fallbacks treating `slow` and `fast` scenario labels as
ordered P10/P90 aliases, while the source fan chart already had ordered
`structural_scenario_low/high` bracket columns.

Rejected alternatives:

- Move PEAK hard constraints into the monthly BASE solver.
- Patch individual months or PEAK products after export.
- Treat `slow`, `central`, `fast` labels as ordered quantiles.
- Declare the curve green after PEAK residuals are fixed while structural
  width and cross-year allocation still fail.

Invariants not to break:

- If `--enable-eex-peak-calibration` is used, final CSV output must satisfy
  both quoted BASE and PEAK residuals within tolerance after all hourly
  mutators.
- Structural export must prefer ordered fan-chart bracket columns over
  scenario label aliases.
- If structural columns are missing, fallbacks must compute ordered row-wise
  low/median/high from scenario prices, not assign `slow -> low` and
  `fast -> high`.
- Power BI strict export remains blocked unless all quality gates pass without
  `--allow-failed-gates`.

## D-20260622-09 - P1 Product Normalization Audit Is Delivered-Artifact Repricing

Decision: add a read-only P1 audit that checks the exact delivered hourly CSV
against CH EEX product averages. The audit must cover hard BASE repricing over
all delivery hours, hard PEAK repricing over EEX peak hours, implied OFFPEAK
energy balance where BASE and PEAK are both quoted, and quote-aware
non-overlapping buckets. The CLI is fail-closed by default; exploratory runs
must pass `--allow-failed-gates`.

Reason: solver-level monthly diagnostics and Power BI screenshots do not prove
the exported HPFC/PFC. Benth/HPFC literature requires product-window
normalization after the complete shaping stack, not only at the intermediate
monthly solver layer. The 2026-06-22 phase3 peak-calibration probe still emits
hard delivered-product failures: 9 `CRITICAL`, 3 `UNSUPPORTED`, max residual
`0.10246153717949369` EUR/MWh.

Rejected alternatives:

- Treat quote-aware buckets alone as proof that every original parent quote
  reprices.
- Let a plain CLI run return success with `CRITICAL` or `UNSUPPORTED` gates.
- Treat missing or partial product windows as `PASS`.
- Use screenshots or Power BI aggregate visuals as the promotion gate.

Invariants not to break:

- The audit reads delivered CSV and EEX parquet only; it must not regenerate or
  mutate the curve.
- `UNSUPPORTED` is not `PASS`.
- Empty evidence is `CRITICAL`.
- Required load types default to `BASE,PEAK`; missing PEAK evidence for a fully
  covered quoted product is `UNSUPPORTED`, not silently ignored.
- Promotion evidence must include the gate CSV, summary JSON, input hashes,
  script hash, and command arguments.

## D-20260623-10 - Local Export Solver Months Follow The Delivered Artifact Window

Decision: for the local-test hourly CSV export, when the monthly solver is
enabled, the solver delivery months are derived from the intended local
artifact window, not from the rounded whole-UTC-day build horizon used by the
intermediate fan-chart builder.

Reason: `scripts/export_local_test_ch_hourly_csv.py` has to overbuild by whole
UTC days so the 15-minute fan chart fully covers the requested local CSV
window. For the 2026-06-13 to 2030-12-31 candidate this technical overbuild
included local 2031-01 rows, causing the monthly solver to see a partial
delivery grid for quoted `CAL 2031` and fail before generating the delivered
artifact. The intended delivered CSV does not include 2031, so the solver
months must follow the delivered artifact horizon. The low-level partial
product check remains fail-closed.

Rejected alternatives:

- Relax `_raise_on_partial_product_grid` in the monthly constraint builder.
- Silently filter partially overlapping quoted products inside the solver.
- Extend the candidate horizon merely to avoid the exception without declaring
  a full-horizon change across CSV, audits, manifests and Power BI sidecars.
- Patch 2031 months or quoted products after the solve.

Invariants not to break:

- A quoted product that overlaps the solver delivery grid only partially must
  still raise `partial delivery grid`.
- Quotes fully outside the delivered artifact window remain outside the solver
  delivery grid and must not become active hard constraints for that artifact.
- Intermediate overbuilt fan-chart rows are not promotion evidence unless the
  candidate horizon explicitly includes them and audits cover them.

## D-20260623-11 - Redundant Quote Conflicts Are Blocking Source-Quality Evidence

Decision: delivered-product normalization audit separates redundant parent
quote conflicts from delivered curve drift. If a direct parent BASE/PEAK quote
fails repricing only because finer quote-aware non-overlapping buckets fully
cover that parent and pass, the direct parent row is reported as
`QUOTE_CONFLICT`, not `CRITICAL` or `UNSUPPORTED`. `QUOTE_CONFLICT` remains a
non-pass, promotion-blocking status by default.

Reason: the 2026-06-17 CH EEX snapshot contains internally inconsistent
redundant quotes, e.g. monthly/quarterly finer quotes imply parent levels that
differ from quoted parents by small but non-zero amounts. A delivered curve
cannot satisfy both the quote-aware hierarchy and the redundant parent quote at
`1e-6` tolerance. Reporting that as ordinary delivered-curve `CRITICAL`
misdirects remediation toward solver/hourly shaping. Reporting it as
`UNSUPPORTED` is also wrong because evidence exists and is contradictory.

Rejected alternatives:

- Keep all redundant parent mismatches as ordinary `CRITICAL` curve drift.
- Convert source quote conflicts to `PASS` because quote-aware buckets pass.
- Hide quote conflicts under `UNSUPPORTED` or far-horizon insufficiency logic.
- Patch individual months or hourly values to satisfy contradictory quotes.

Invariants not to break:

- Quote-aware non-overlapping BASE and PEAK bucket repricing remains a hard
  delivered-product gate.
- A direct parent row may become `QUOTE_CONFLICT` only when finer quote-aware
  buckets fully cover it and all those buckets pass.
- Direct product failures that are not explained by passing finer buckets
  remain `CRITICAL`.
- `QUOTE_CONFLICT` blocks strict CLI/Power BI/promotion flows unless an
  explicit future manifest-backed policy accepts a source hierarchy.
- The audit must keep the parent target, delivered mean, residual, covered
  finer bucket names and summary counts visible.

