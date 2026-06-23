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

## D-20260623-02 - Final Delivered-Hourly PEAK Projection

Decision: when `--enable-eex-peak-calibration` is active, the delivered-hourly
export runs a final BASE+PEAK projection immediately before CSV write, after
all mutating hourly shape layers.

Reason: intermediate PEAK calibration is not stable if later seam/monthly
mutators alter PEAK hours. The final delivered CSV is the evidence surface, so
the last mutating step must restore quoted BASE and PEAK means before writing
the artifact.

Rejected alternatives:

- Calibrate PEAK only before final seam or path smoothing.
- Preserve the pre-existing BASE level after PEAK shifts instead of solving the
  OFFPEAK mean from quoted BASE energy and quoted PEAK energy.
- Accept PEAK residuals as a Power BI or downstream presentation issue.

Invariants not to break:

- Final projection must preserve quoted BASE and PEAK means on exact product
  windows.
- Hourly layers may shape inside product windows but must not rewrite final
  quoted product means.
- The projection remains opt-in via `--enable-eex-peak-calibration`.

## D-20260623-03 - Ordered Structural Bridge And Strict Diagnostic Export

Decision: delivered-hourly and Power BI export bridges must use ordered
structural brackets when available and row-wise ordered scenario fallback when
not. Strict Power BI sidecar export blocks on quality gates unless
`--allow-failed-gates` is explicitly requested.

Reason: `slow/central/fast` scenario labels are not guaranteed to be ordered
P10/P50/P90 quantiles. Treating them as ordered quantiles can invert fan-chart
width and hide shaping defects. Diagnostic Power BI exports should not be
silently generated from a curve that fails the audited quality gates.

Rejected alternatives:

- Continue assigning `slow/central/fast` directly to P10/P50/P90.
- Keep structural columns mandatory for Power BI input discovery.
- Let strict Power BI export write sidecars while only recording failed gates in
  summary metrics.

Invariants not to break:

- Ordered `structural_scenario_low/central/high/spread` columns take precedence
  over legacy structural columns when both exist.
- Row-wise fallback must produce non-negative width.
- Failed quality gates remain blocking by default; diagnostic sidecars require
  explicit `--allow-failed-gates`.
