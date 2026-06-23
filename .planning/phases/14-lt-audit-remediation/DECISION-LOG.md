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

## D-20260622-05 - UTC-Aware Hourly Export Timestamp

Decision: LT hourly CSV exports must write `timestamp_utc` with an explicit UTC
offset.

Reason: Phase P1 product-window gates require unambiguous UTC timestamps before
checking DST, leap-year and EEX PEAK/OFFPEAK masks. A text value such as
`21.06.2026 22:00` may be intended as UTC, but it is timezone-naive evidence
and must fail closed.

Rejected alternatives:

- Let the P1 audit infer UTC from the column name.
- Repair delivered CSV timestamps inside the audit before checking products.
- Use `timestamp_ch` plus `utc_offset_ch` as a substitute while leaving
  `timestamp_utc` naive.

Invariants not to break:

- Existing CH local timestamp and offset columns remain available for Power BI
  and analyst display.
- The P1 audit continues to reject timezone-naive `timestamp_utc` values.
- Old delivered artifacts are not retroactively converted to PASS evidence.

## D-20260622-06 - Source Quote Parent/Child Consistency Gate

Decision: P1 audit reports a `source_quote_parent_child_consistency` gate when
a selected CH EEX snapshot contains a quoted parent product and a complete set
of quoted child products whose hour-weighted mean differs beyond tolerance.

Reason: when the source snapshot itself has parent/child inconsistencies, no
single delivered curve can satisfy all overlapping direct quotes exactly. The
audit still fails closed, but it must identify the upstream quote conflict
instead of leaving the direct quote residuals ambiguous.

Rejected alternatives:

- Silently prefer Month > Quarter > Calendar without reporting the dropped
  parent inconsistency.
- Treat parent/child conflicts as curve-only residuals.
- Relax the direct quote tolerance to hide small parent/child source conflicts.

Invariants not to break:

- Quote-aware bucket checks still run and preserve Month > Quarter > Calendar
  priority.
- Conflicting source quotes are `CRITICAL` gates, not analyst warnings.
- Direct quote residuals remain visible; the source gate explains, but does not
  erase, the failed evidence.

## D-20260622-07 - P1 Active Quote Set Direct Checks

Decision: Phase P1 direct product mean checks are run against an explicit
active quote set using `Month > Quarter > Calendar`. A parent quote with a
complete finer child quote set is dropped from direct checks and recorded as
`active_quote_set_parent_dropped`.

Clarification: this supersedes the earlier provisional gate naming in
D-20260622-06. Parent/child source conflicts are still fail-closed, but the
final emitted gate is `active_quote_set_parent_dropped` with
`dropped_reason=parent_child_conflict`, so the audit records both the active
quote hierarchy and the source inconsistency in one row.

Reason: when a selected EEX snapshot contains both a parent and a complete
child set, the active market-evidence hierarchy must be explicit. Otherwise the
audit double-counts the same conflict: first as a source inconsistency and then
again as a curve residual that no single curve can remove without violating the
children.

Rejected alternatives:

- Continue checking all overlapping direct products as if they were jointly
  satisfiable.
- Drop parent quotes silently.
- Let bucket residual logic imply the hierarchy without exposing which quotes
  were active or dropped.

Invariants not to break:

- Dropped parents remain auditable evidence with `dropped_reason`,
  `child_products`, target, implied child-weighted value and residual.
- `parent_child_conflict` remains `CRITICAL` until governance explicitly
  approves a different promotion policy.
- Active direct quote checks and quote-aware bucket checks must both be run
  from the same active quote set.

## D-20260623-01 - Explicit CSV Window Scope For P1

Decision: Phase P1 may explicitly scope selected forwards to the delivered CSV
window. Only quoted products with no overlap with the CSV window are excluded,
and each exclusion is reported as `INFO out_of_scope_quote`.

Reason: a delivered export can legitimately start after the first product in
the selected EEX snapshot, for example a July 1 export against a snapshot that
also contains June quotes. Those no-overlap quotes are outside the evidence
population for that CSV. Without an explicit scope contract, they create
`UNSUPPORTED quoted_product_absent` rows that obscure the actual product
normalization result.

Rejected alternatives:

- Drop no-overlap quotes silently.
- Treat partial product windows as out of scope.
- Enable scoping by default.

Invariants not to break:

- The default audit remains fail-closed: absent quoted products are
  `UNSUPPORTED` unless explicit CSV-window scoping is requested.
- Partial product windows remain `UNSUPPORTED`; only zero-overlap quotes can be
  scoped out.
- Scoping does not change active quote hierarchy or `parent_child_conflict`
  severity.

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

## D-20260623-04 - Audited Structural Template Fallback Defaults

Decision: the monthly solver default/config enables
`allow_template_structural_fallback=true`, uses
`structural_amplitude_eur_mwh=110.0`, and sets `structural_weight=1.0`. The
selected structural prior is summarized in the solver manifest, and the active
config hash includes the structural fallback knobs and prior weights.

Reason: when CH monthly structural history is absent or below support, the
solver still needs a deterministic zero-mean structural prior rather than a
silent flat fallback. The fallback must be visible in promotion evidence so the
curve can be audited against the exact prior that shaped unquoted months.

Rejected alternatives:

- Keep config defaults at `allow_template_structural_fallback=false` while tests
  and phase planning assume the template prior is active.
- Let the template prior influence the solver without recording amplitude,
  fallback status or zero-mean diagnostics in the manifest.
- Hash only the numerical solver lambdas while excluding structural prior knobs.

Invariants not to break:

- The structural template remains a soft prior only; CH EEX constraints remain
  hard level authority.
- Structural deviations must remain zero-mean inside active CH parent blocks.
- Changing fallback amplitude or structural prior weight must change
  `active_config_hash`.
- No CT, Power BI, heavy data, or individual generated month patch is part of
  this decision.

## D-20260623-05 - Q4 Is A Comparable Seasonal Sub-Block

Decision: the `residual_vs_implied_comparable_block` audit population covers
quoted seasonal sub-blocks against full calendars, not only residual buckets
against calendars. In particular, a quoted Q4 parent block compared with a
next-year full CAL parent must emit comparable-block rows and use calibrated
historical thresholds for the same `parent_type_pair` where available.

Reason: the Phase 4 cross-year failure mode is not limited to Apr-Dec residual
buckets. A quoted Q4 can also make same-month visual comparisons misleading if
the audit treats the opposite year's full calendar as directly comparable.
The gate must therefore compare same-month deviations from parent blocks for
`quarter|calendar` as well as `residual|calendar`.

Rejected alternatives:

- Rely on `same_month_rank_consistency` alone for Q4-vs-CAL cases.
- Treat the gate name as limiting the population to residual buckets only.
- Let Q4-vs-CAL rows become permanently `UNSUPPORTED` because the historical
  threshold builder ignores quarter/calendar observations.

Invariants not to break:

- No naive rule forces every month to follow the calendar spread sign.
- Quote support can suppress a shape critical only when active hard constraints
  cover the exact month or parent block under test.
- Historical threshold rows remain point-in-time and fail closed as
  `UNSUPPORTED` when undersampled or when the required `parent_type_pair` is
  unsupported.
- The audit remains evidence-only and must not mutate monthly solver output.

## D-20260623-06 - Structural Scenario Bracket Is Not A Probability Fan

Decision: local/test structural scenario fan columns are canonical as
`structural_scenario_low_eur_mwh`,
`structural_scenario_central_eur_mwh`,
`structural_scenario_high_eur_mwh`, and
`structural_scenario_spread_eur_mwh`. Legacy
`structural_p10_eur_mwh`, `structural_p50_eur_mwh`,
`structural_p90_eur_mwh`, and `structural_width_eur_mwh` remain export aliases
only and are populated from ordered low, central scenario, ordered high, and
spread. They are not computed as probabilistic quantiles from three scenario
points.

Reason: slow/central/fast are governed structural scenarios, not a calibrated
distribution. Computing p10/p50/p90 from three points mislabeled a deterministic
scenario bracket as probabilistic risk and made downstream charts look more
statistically calibrated than the data supports.

Rejected alternatives:

- Keep weighted quantile calculations over slow/central/fast in each mutator.
- Rename the global LT `p10/p90` uncertainty contract, which is a separate
  probabilistic export surface.
- Drop legacy structural columns immediately and break current audit/reporting
  consumers.

Invariants not to break:

- `price_weighted_mean_eur_mwh` remains the weighted scenario mean.
- `structural_scenario_central_eur_mwh` is the central scenario curve, not the
  median if central is not between slow and fast.
- `structural_scenario_spread_eur_mwh` is high minus low and must be
  non-negative.
- When canonical and legacy structural columns conflict, canonical scenario
  columns win and legacy aliases are overwritten from them.
- Production LT uncertainty columns named `p10`/`p90` remain out of scope.

## D-20260623-07 - Solver Authority Export Is Manifest-Backed And Non-Mutating

Decision: local/test hourly export treats `monthly_level_authority=solver` as a
hard level authority. Fresh solver builds and prebuilt solver fan charts skip
export-time EEX BASE recalibration, block local/test post-processors that can
rewrite solver monthly means, and require a complete adjacent solver manifest.
Prebuilt skip-build exports without a manifest now fail closed unless explicitly
declared as `legacy`.

Reason: a solver-authority fan chart has already satisfied hard CH EEX monthly
constraints in the monthly solver. Re-running legacy export calibration or
post-calibration rebalancers after that point silently reassigns level authority
to the export script. A one-field manifest is also not enough evidence to
disable calibration; the export must validate market, snapshot date, solver
config hash, source hashes, constraint/solution hashes, KKT diagnostics, and
legacy-bypass flags.

Rejected alternatives:

- Trust `--fan-chart-monthly-level-authority=solver` without an adjacent
  manifest.
- Allow solver-authority exports to run structural shape upgrade,
  post-calibration rebalancers, EEX PEAK calibration, or final/monthly mutators.
- Preserve stale legacy `structural_p10/p50/p90` aliases in Power BI when
  canonical structural scenario columns or scenario price columns are present.

Invariants not to break:

- Solver-authority export may aggregate and alias columns but must not change
  solver monthly means.
- `active_config_hash` must match `config_hash(solver_config)`.
- `solver_config`, `source_hashes`, and `solver_kkt` must be non-empty, with
  required KKT fields and boolean flags validated.
- Solver manifest evidence must bind to the exported fan parquet through
  `fan_chart_sha256` and `fan_chart_weighted_monthly_hash`.
- `source_hashes.forwards_path` must equal the SHA-256 of the actual
  `--forwards` parquet used by the export.
- Solver-authority exports must validate every fully covered EEX BASE Month,
  Quarter, and Calendar quote independently, including parent quotes that are
  also covered by finer quotes. Solver authority is rejected when the export
  window contains no fully covered EEX BASE quote.
- Solver-authority KKT evidence must reject large residuals, non-positive or
  extreme condition numbers, and ridge/lstsq fallback solves.
- Skip-build without a manifest is allowed only when the caller explicitly
  declares the prebuilt fan chart as `legacy`.
- Power BI structural aliases are recomputed from canonical structural scenario
  columns when present, otherwise from ordered slow/central/fast scenario prices.
