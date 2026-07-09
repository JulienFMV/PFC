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

## D-20260623-12 - Direct Monthly Panel Evidence Dominates Template Structural Shape Locally

Decision: when a neighbor panel has direct monthly evidence for every month of
an active CH parent bucket, the fused monthly prior suppresses
`STRUCTURAL_TEMPLATE` contribution inside that parent bucket only. The panel
does not set CH level; it remains a zero-mean shape prior inside the CH parent
bucket.

Reason: the 2026-06-17 candidate showed `2027-Q2 BASE` still influenced by the
generic structural template even though DE had direct monthly Apr/May/Jun
quotes covering the full comparable parent bucket. The template encoded an
Apr > May > Jun pattern while the directly quoted neighbor panel encoded
Apr/Jun high and May low. A global average of panel and template diluted the
best local evidence and produced a monthly split gate failure.

Rejected alternatives:

- Patch individual Q2 2027 months after the solver.
- Downgrade the monthly split gate while contradictory local prior evidence
  remained.
- Remove the structural template globally whenever any panel monthly quote
  exists.

Invariants not to break:

- The fused prior remains zero-mean inside each CH parent bucket.
- Structural template fallback remains available for parent buckets without
  full direct monthly panel evidence.
- Diagnostics must expose the direct-month parent buckets and the
  evidence-aware policy.

## D-20260623-13 - Cross-Year Audit Uses Comparable Reference Spread

Decision: cross-year same-month diagnostics keep reporting full parent targets
and parent spreads, but residual/calendar mixed comparisons use a
`reference_spread_eur_mwh` based on the economically comparable block. For a
residual Apr-Dec bucket against a calendar year, the reference is the calendar
year's Apr-Dec implied monthly block, not the full calendar average.

Reason: Phase 14 already decided that a residual block must not be compared
directly to a full calendar year. The Power BI strict gate was still blocked by
near-clone warnings produced from full parent spread logic. This could either
hide a real allocation issue or falsely escalate a valid seasonal
decomposition. The audit now exposes both `parent_spread_eur_mwh` and
`reference_spread_eur_mwh`; severity is based on the comparable reference.

Rejected alternatives:

- Revert solver comparable-block YoY logic to full-calendar comparisons.
- Remove the near-clone blocker from Power BI strict gates.
- Allow cross-year post-processors to mutate a solver-authoritative curve.

Invariants not to break:

- Cross-year checks remain fail-closed for `CRITICAL` severity.
- Near-clone warnings still block strict Power BI export when they occur
  against a non-zero comparable reference spread.
- The report must keep parent targets, parent spread, reference spread and
  reference basis visible.

## D-20260624-14 - One Cross-Year Near-Clone Warning Blocks Strict Power BI

Decision: strict Power BI sidecar export blocks on any cross-year near-clone
warning, not only on two or more warnings.

Reason: the 2026-06-23 candidate
`parent_local_prior_lshape25_yoy10_structural_s126` had one same-month
near-clone warning against a non-zero comparable reference spread. Expert
review found that allowing a strict export in this state made the Power BI
artifact look promotion-ready while a structural allocation warning remained
unresolved.

Rejected alternatives:

- Keep the previous threshold `near_clone >= 2`.
- Generate strict sidecars and leave interpretation to manual chart review.
- Downgrade near-clone warnings while comparable reference spread is non-zero.

Invariants not to break:

- `--allow-failed-gates` remains diagnostic-only.
- Strict Power BI exports must fail closed on unresolved cross-year near-clone
  warnings.
- The cross-year audit remains the source of near-clone evidence; Power BI does
  not rewrite monthly solver levels.

## D-20260624-15 - Quote Conflict Source Hierarchy Requires Production Approval

Decision: `QUOTE_CONFLICT` may stop blocking delivered-product normalization
only when an explicit source-hierarchy policy artifact is present, valid for
the market and forward snapshot, and marked `production_approved=true`.
Draft or missing policies keep all quote conflicts blocking.

Reason: the CH 2026-06-17 quote snapshot contains redundant parent quotes that
conflict with passing finer quote-aware BASE/PEAK buckets. The solver cannot
simultaneously satisfy contradictory source quotes at hard tolerance. Accepting
the finer-bucket hierarchy is a source-quality policy decision, not a model
fix, so it must be manifest-backed and production-approved before promotion.

Rejected alternatives:

- Treat all explained quote conflicts as `PASS` by default.
- Hide quote conflicts under `UNSUPPORTED`.
- Patch individual delivered months to satisfy contradictory quotes.
- Promote from a draft policy artifact.

Invariants not to break:

- A missing or draft policy leaves `blocking_quote_conflict_count` equal to
  `quote_conflict_count`.
- Direct failures not explained by passing finer buckets remain `CRITICAL`.
- The audit summary must expose policy path, hash, status, production approval,
  accepted count and blocking count.

## D-20260624-16 - Selected Lambda Artifact Is Required But Not Sufficient

Decision: Phase 14 may keep a diagnostic selected-lambda/config artifact for
the current candidate, but production remains NO-GO while that artifact is
`production_approved=false` or while the production/local-export/selected
artifact triad is not manifest-backed and mutually consistent.

Reason: the candidate config hash
`145b123177061c9d2cd64ec831b83ea4ac84ff500356adb03623c4a9d1f86fc0`
matches the local candidate manifest, but this only proves diagnostic hash
alignment for that run. It does not prove production selection, source policy
acceptance, or a full promotion triad.

Rejected alternatives:

- Treat config-hash equality as production promotion proof.
- Store selected lambda/config only in chat or untracked local output.
- Mark the candidate production-approved while product and Power BI gates still
  block.

Invariants not to break:

- Selected config artifacts must record the canonical config, config hash,
  manifest path, solution hash when available and production approval state.
- Production promotion must cite independent real manifests for production,
  local export and selected lambda/config.
- `production_approved=false` selected artifacts are evidence for diagnosis,
  not authority for prod.

## D-20260624-17 - Selected Config Approval Is A Promotion Gate

Decision: manifest-backed promotion must include explicit governance gates for
selected config production approval and selected/prod/export manifest parity.
`config_hash` equality alone is not enough.

Reason: read-only roaster audit found the capstone promotion script could pass
`lambda_calibration_artifact_present` when `active_config_hash` matched the
selected artifact even if that selected artifact had
`production_approved=false`. It also did not compare selected
`monthly_solution_hash` or `active_constraints_hash` against the production
and local-export manifests.

Rejected alternatives:

- Rely on handoff text saying the selected artifact is diagnostic-only.
- Keep the selected artifact outside required governance gates.
- Compare selected config only on `config_hash`.

Invariants not to break:

- Required promotion governance gates include
  `selected_config_production_approval` and
  `selected_config_manifest_parity`.
- The selected config artifact must have `production_approved=true` and a
  production-approved selection status before promotion can pass.
- Selected config, production manifest and export manifest must agree on active
  config, monthly solution and active constraints hashes.

## D-20260624-18 - Source Hierarchy Policy Is Strictly Typed And Count-Bound

Decision: a source hierarchy policy may accept `QUOTE_CONFLICT` only when its
governance fields are strictly typed and bound to the audited evidence:
`accept_quote_conflict is True`, `production_approved is True`,
`forward_snapshot_date` is present and equal to the audited snapshot, and
`expected_quote_conflict_count` equals the observed quote conflict count.

Reason: roaster audit found a future approved policy could otherwise accept
new conflicts too broadly. Python truthiness also meant string values such as
`"false"` could be interpreted as approval if not rejected explicitly.

Rejected alternatives:

- Cast policy booleans with `bool(...)`.
- Let an approved policy omit `forward_snapshot_date`.
- Let a policy approved for 9 conflicts accept a different conflict count.

Invariants not to break:

- Draft policies with `production_approved=false` remain valid but blocking.
- Malformed or stale policies are `INVALID` and accept zero quote conflicts.
- `blocking_quote_conflict_count` remains equal to the observed count unless
  every strict policy condition passes.

## D-20260624-19 - Promotion Policy Status And Counts Use Exact Values

Decision: promotion governance must use exact values for approval state. A
selected config is production-approved only when
`production_approved is True` and `selection_status == "PRODUCTION_APPROVED"`.
A source hierarchy policy conflict count is valid only when
`expected_quote_conflict_count` is a JSON/YAML integer, not a boolean, string
or float, and equals the observed quote conflict count.

Reason: roaster re-audit found two remaining fail-open edges. Substring status
matching accepted negative labels such as `NOT_PRODUCTION_APPROVED`, and
`int(...)` conversion accepted strings/floats such as `"9"` or `9.1` for an
integer conflict-count policy.

Rejected alternatives:

- Match selected config status with a substring.
- Cast conflict counts with `int(...)`.
- Treat numerically equal floats or strings as equivalent governance values.

Invariants not to break:

- Approval labels are enums, not prose searched by substring.
- Approval labels are case-sensitive exact values; `production_approved` or
  `Production_Approved` are invalid.
- Governance counts are strict integers and remain count-bound.
- Malformed approval/count fields make the artifact blocking, not partially
  accepted.

## D-20260624-20 - Source Hierarchy Policy Binds Conflict Identity

Decision: a production-approved source hierarchy policy must bind accepted
`QUOTE_CONFLICT` rows to the audited evidence, not only to market, snapshot and
count. At least one binding must match: `input_csv_sha256`,
`quote_conflict_identity_hash`, or the full canonical
`expected_quote_conflicts` list. If any binding field is provided, it must
match exactly.

Reason: roaster review accepted the P0/P1 governance exact-value gates but
left a P2 risk: a policy approved for one candidate could be reused on another
CSV with the same market, snapshot and conflict count but different conflicting
products. Binding to the input CSV hash or canonical conflict identities closes
that reuse path.

Rejected alternatives:

- Accept quote conflicts from market/snapshot/count only.
- Treat `candidate_csv` path text as sufficient evidence.
- Require manual review of conflicts while the CLI reports promotion-ready.

Invariants not to break:

- Draft non-production policies remain blocking.
- Production-approved policies with no binding are `INVALID`.
- Binding mismatches accept zero quote conflicts.
- The summary must expose `quote_conflict_identity_hash` and
  `quote_conflict_identities` so a policy can be reproduced and audited.

## D-20260624-21 - Production Source Hierarchy Policy Requires Full Artifact Binding

Decision: a production-approved source hierarchy policy must bind all three
artifact dimensions before accepting `QUOTE_CONFLICT`: the delivered CSV hash
(`input_csv_sha256`), the forwards snapshot file hash (`forwards_sha256`), and
the conflict identity evidence (`quote_conflict_identity_hash` or exact
`expected_quote_conflicts`). Every provided binding must match exactly.

Reason: MIT roasters accepted D-20260624-20 as closing the reuse P2, but noted
that a policy using only conflict identities could be reused across another
CSV with the same conflict set, and a policy using only CSV hash would not bind
the forwards snapshot. Requiring CSV + forwards + conflict identity closes both
residual reuse paths before any production policy approval.

Rejected alternatives:

- Accept a policy with only one binding dimension.
- Let `input_csv_sha256` alone stand for the source snapshot.
- Let conflict identity alone stand for the delivered artifact.

Invariants not to break:

- Draft non-production policies remain blocking but may carry full bindings as
  evidence.
- Production-approved policies missing any required artifact dimension are
  `INVALID`.
- A correct binding in one field must not override a mismatch in another.

## D-20260624-22 - Boundary Products Are OUT_OF_SCOPE, Not UNSUPPORTED

Decision: delivered-product audit rows whose full EEX product window is outside
the delivered artifact window are classified as `OUT_OF_SCOPE` with info
severity. In-scope missing delivered rows or missing required quotes remain
`UNSUPPORTED` and blocking. If an audit emits only out-of-scope rows and no
in-scope evidence, it fails closed with `audit_evidence_present=CRITICAL`.

Reason: the CH local candidate is delivered for `2026-06-13` to
`2030-12-31`. Full product windows such as `2026-06`, `2031`, and `2032` are
outside that artifact horizon and should remain visible without being treated
as failed repricing evidence. This preserves the prior invariant that true
in-scope missing coverage remains blocking.

Rejected alternatives:

- Keep boundary products as `UNSUPPORTED` and require impossible repricing over
  hours the delivered artifact does not contain.
- Drop out-of-scope rows entirely.
- Let an audit with no in-scope product evidence pass.

Invariants not to break:

- `UNSUPPORTED` remains blocking for missing in-scope evidence.
- `OUT_OF_SCOPE` rows are counted and reported in the summary.
- At least one in-scope product gate must exist for a passing audit.

## D-20260624-23 - Phase 14 Local Candidate Selects lambda_smooth_yoy=50

Decision: select the local CH candidate
`20260624_parent_local_prior_lshape25_yoy50_structural_s126` as the current
auditable Phase 14 candidate, with `lambda_shape=25`,
`lambda_smooth_month=0.1`, `lambda_smooth_yoy=50`, PEAK calibration enabled,
and structural scenario spread intensity `1.26`.

Reason: the prior `yoy10` candidate still had a strict Power BI
cross-year near-clone warning. Lowering YoY smoothness to `2` worsened the
diagnostic, and increasing local shape to `40` did not remove it. Increasing
YoY smoothness to `50` keeps the monthly solver as level authority and passes
strict Power BI without post-solver month patching.

Rejected alternatives:

- Patch the problematic month after the monthly solver.
- Lower `lambda_smooth_yoy`, which increased cross-year warnings.
- Treat the strict Power BI warning as acceptable promotion evidence.

Invariants not to break:

- Monthly solver remains the monthly level authority.
- Promotion still requires real production, local export, and selected config
  manifest parity.
- Source hierarchy `QUOTE_CONFLICT` acceptance remains exact artifact-bound.

## D-20260624-24 - Production Promotion Requires Explicit Promotion Scope

Decision: the manifest-backed promotion capstone must block a selected config
artifact when it declares `production_promotion_approved=false`, even if
`production_approved=true` and `selection_status="PRODUCTION_APPROVED"`.
Legacy selected config artifacts without this field keep the previous strict
`production_approved` plus exact status behavior.

Reason: the Phase 14 `yoy50` selected config is approved as the current local
auditable candidate, but it is not a real production promotion artifact. The
real prod/export/selected triad check showed production still on older hashes.
The extra scope flag prevents an isolated local-candidate selected config from
being overread as production promotion approval.

Rejected alternatives:

- Let `production_approved=true` alone pass a production promotion capstone.
- Rely only on prose in handoffs to distinguish candidate selection from
  production promotion.
- Mark the local candidate selected config as production-promotion approved
  before the real manifest triad matches.

Invariants not to break:

- Real promotion remains blocked until production, export and selected config
  hashes match.
- Candidate selection can stay documented without implying production
  deployment.
- Older selected config fixtures remain supported unless they opt into the new
  explicit promotion-scope field.

## D-20260624-25 - Current Data Promotion Evidence Uses 2026-06-23 Snapshot

Decision: refresh Phase 14 promotion evidence from the EEX workbook available
on 2026-06-24, but bind all solver, audit, and promotion artifacts to
`forward_snapshot_date=2026-06-23`, the latest usable row in that workbook.
Do not claim a 2026-06-24 forward fixing.

Reason: the workbook file was current on 2026-06-24, while CH/DE/FR forward
history rows inside it stop at 2026-06-23. AT/IT remain available only up to
2026-06-17 and are therefore skipped for exact as-of neighbor evidence rather
than mixed into the 2026-06-23 CH solve.

Rejected alternatives:

- Treat the workbook file date as the forward snapshot date.
- Mix latest-neighbor dates independently when CH is solved at a fixed as-of
  date.
- Hold promotion evidence on the older 2026-06-17 candidate after newer CH/DE/FR
  forward rows were available.

Invariants not to break:

- Snapshot dates in manifests must mean quote-row dates, not file timestamps.
- Neighbor priors used for a fixed-as-of solve must come from the same snapshot
  date when available; missing exact-date neighbors are skipped.
- Heavy local forward history parquet changes remain generated data and are not
  commit targets.

## D-20260624-26 - Phase 14 As-Of 2026-06-23 Triad Is Promotion Evidence Pass

Decision: accept that the `asof20260623_yoy50_2032` Phase 14 evidence triad
passed manifest governance, but supersede it for production promotion after PNG
diagnostics showed unacceptable far-horizon monthly shape. The accepted
promotion candidate is now recorded in D-20260624-27.

Reason: after regenerating production and local export on the refreshed
2026-06-23 snapshot, the real manifest triad matches:

- `active_config_hash`: `9a29207f20efd39be80a33b3fa1ffc4c02b28daa2fd550e1db20837d1f8966db`
- `active_constraints_hash`: `a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262`
- `monthly_solution_hash`: `e42887237658655e8bf5881c667b881d4361a0571ebbb413ad755d7172087a31`

The selected artifact is
`.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260623_yoy50_2032.json`
with `production_promotion_approved=true` and
`selection_status="PRODUCTION_APPROVED"`. The capstone output
`output/phase14/20260624_asof20260623_yoy50_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad.json`
reports `approved=true`, `blocking_count=0`, and
`audit_gate_status_counts={"PASS": 27, "UNSUPPORTED": 10}`.

Rejected alternatives:

- Promote with the older `2026-06-17` local-candidate selected artifact.
- Treat a local candidate artifact with `production_promotion_approved=false`
  as production promotion evidence.
- Ignore the selected-lambda artifact and rely only on production/export parity.

Supersession note:

- `asof20260623_yoy50_2032` is not the current promotion candidate.
- Keep the artifact only as historical evidence that manifest parity was
  repaired before the shape remediation.

Invariants not to break:

- The source hierarchy policy remains exact-artifact-bound and production
  approved only for matching hashes.
- `UNSUPPORTED` is acceptable here only because it is documented far-horizon
  threshold insufficiency and hides no `CRITICAL` gate.
- Promotion evidence remains manifest-backed; generated output artifacts are
  evidence, not code commit targets by default.

## D-20260624-27 - PNG Diagnostics Supersede YoY50 With Shape-Restored Candidate

Decision: replace the visually poor `asof20260623_yoy50_2032` candidate with
`output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/` as the
current Phase 14 promotion-ready evidence. The monthly solver remains the level
authority; the remediation changes solver priors/objective settings, not
individual months after the solve:

- `lambda_shape=100.0`
- `lambda_smooth_month=0.1`
- `lambda_smooth_yoy=10.0`
- `structural_amplitude_eur_mwh=200.0`

Reason: user review of PNG diagnostics correctly flagged the previous monthly
shape as unsatisfactory. The old candidate passed hard gates but annual-only
years collapsed toward a flat template. The new candidate restores seasonal
amplitude while preserving hard BASE/PEAK constraints, source hierarchy policy,
and production/export/selected manifest parity.

Key evidence:

- PNG diagnostics:
  `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/png_diagnostics/`
- annual amplitude from `monthly_diagnostics.csv`:
  2029 `44.87`, 2030 `39.75`, 2031 `39.33`, 2032 `39.33`
- Q1-Q3 spread:
  2029 `29.27`, 2030 `24.23`, 2031 `23.81`, 2032 `23.76`
- strict Power BI:
  `powerbi_quality_gate_status=PASS`, `shape_score_10=9`,
  `max_eex_base_error_eur_mwh=0.000000`,
  `max_eex_peak_error_eur_mwh=0.000000`,
  `seasonal_warning_flags=0`,
  `cross_year_month_shape_warning_flags=0`,
  `latest_hfc_winter_summer_spread_eur_mwh=31.10`
- delivered-product audit:
  `all_gates_pass=true`, `PASS=80`, `QUOTE_CONFLICT=9`,
  `accepted_quote_conflict_count=9`, `blocking_quote_conflict_count=0`,
  `UNSUPPORTED=0`, `OUT_OF_SCOPE=3`
- promotion capstone:
  `approved=true`, `status=PROMOTION_EVIDENCE_PASS`, `blocking_count=0`,
  `audit_gate_status_counts={"PASS": 27, "UNSUPPORTED": 10}`

Manifest triad:

- production manifest:
  `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`
- local export manifest:
  `output/phase14/20260624_asof20260623_lshape100_yoy10_amp200_2032/fan_asof20260623_lshape100_yoy10_amp200_2032.monthly_curve_manifest.json`
- selected config:
  `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260623_lshape100_yoy10_amp200_2032.json`
- `active_config_hash`:
  `f4b64f88919149a42a85693135c047b442ffa099011ce17e41c1cfe8782db88e`
- `active_constraints_hash`:
  `a80d5e09d2b6eda2ca5f22fd83ed58116a96b91dd80e46f50b61eb7e54baa262`
- `monthly_solution_hash`:
  `d717a426f5fee7fe62abf294a0e44311040115fd4edb6a3a118f06bf7243832e`

Rejected alternatives:

- Keep `lambda_smooth_yoy=50` because it passed strict gates.
- Patch annual-only months manually after the solver.
- Treat PNG diagnostics as cosmetic when the dashboard shape is a promotion
  acceptance criterion.

Invariants not to break:

- Monthly solver remains the monthly level authority.
- Exact CH hard BASE/PEAK constraints and quote-aware delivered-product audit
  remain mandatory.
- Source hierarchy `QUOTE_CONFLICT` acceptance remains exact-artifact-bound.
- Promotion evidence must keep production, local export, and selected config
  manifest parity.

## D-20260707-01 - Daily Generation Uses 2026-07-06 Forward Snapshot

Decision: regenerate the Tuesday 2026-07-07 CH LT PFC using the latest usable
desk EEX quote row, `2026-07-06`, not a nonexistent `2026-07-07` quote row.
Append the daily workbook `Price_Report_EEX.xlsx` to the local forward history
and bind current generation artifacts to `forward_snapshot_date=2026-07-06`.

Reason: the EEX workbook file was available on 2026-07-07, but the latest
quoted CH/DE/FR rows inside it stop at 2026-07-06. AT/IT are not present in the
daily workbook and remain unavailable at this exact as-of date, so they are not
mixed into exact-date neighbor evidence.

Rejected alternatives:

- Claim a 2026-07-07 forward snapshot from a workbook whose quote rows stop on
  2026-07-06.
- Keep using the 2026-06-23 pinned snapshot after fresh CH/DE/FR daily quotes
  were available.
- Mix AT/IT stale dates into a fixed 2026-07-06 CH solve.

Invariants not to break:

- Manifest snapshot dates mean quote-row dates, not file timestamps.
- Refreshed `data/eex_forwards_history.parquet` remains a local generated data
  artifact and is not a commit target by default.
- OMPEX/HFC benchmark files remain read-only benchmark evidence and are not
  model inputs.

## D-20260707-02 - 2026-07-07 Strict-Pass Local Candidate Uses YoY150

Decision: for the 2026-07-07 daily generation, supersede the first
`lshape100/yoy10/amp200` and `lshape100/yoy50/amp200` local candidates with
`output/phase14/20260707_asof20260706_lshape100_yoy150_amp200_2032/`.

Reason: the initial `yoy10` candidate generated successfully and had good PNG
shape, but strict Power BI blocked it with
`monthly_path_critical_flags=1` and `cross_year_near_clone_warnings=3`.
`yoy50` removed the cross-year near-clone warnings but still had one residual
edge monthly-path critical. A targeted solver probe showed
`lambda_smooth_yoy=150` retained far-horizon seasonal amplitude while giving
enough margin on the 2028 Q2 -> residual edge. The final candidate passes
strict Power BI without `--allow-failed-gates`.

Key evidence:

- candidate CSV:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp200_2032/ch_hfc_hourly_asof20260706_lshape100_yoy150_amp200_2032.csv`
- local export manifest:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp200_2032/fan_asof20260706_lshape100_yoy150_amp200_2032.monthly_curve_manifest.json`
- PNG diagnostics:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp200_2032/png_diagnostics/`
- strict Power BI:
  `powerbi_quality_gate_status=PASS`, `shape_score_10=9`,
  `max_eex_base_error_eur_mwh=0.000000`,
  `max_eex_peak_error_eur_mwh=0.000000`,
  `monthly_path_critical_flags=0`,
  `cross_year_month_shape_warning_flags=0`,
  `seasonal_warning_flags=0`
- monthly shape:
  2029 amplitude `45.15`, 2030 amplitude `41.98`, 2031 amplitude `40.63`,
  2032 amplitude `40.13`
- manifest hashes:
  `active_config_hash=7bbcb4a63fb51013c5f0b80654167811ea1f86a649004640dc89a61c7b02d94c`,
  `active_constraints_hash=451f9a23a4388973addabc7b467b535df2352bec74e69af34dcf4c28cdbeb15d`,
  `monthly_solution_hash=0c42cda6a20fdcde282f6f4b401a347f84073d830da788bceca927f622f9dad0`

Delivered-product audit status:

- diagnostic audit only, no 2026-07-07 source-hierarchy policy artifact yet
- `PASS=90`, `QUOTE_CONFLICT=6`, `UNSUPPORTED=0`
- `critical_count=0`, `delivered_curve_drift_count=0`
- `quote_conflict_identity_hash=a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`

Production dry-run/save:

- wrote `pfc_shaping/output/pfc_15min_2026-07-07.*`
- wrote `pfc_shaping/output/pfc_de_15min_2026-07-07.*`
- wrote `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`
- production manifest has the same active config, active constraints, and
  monthly solution hashes as the strict-pass local candidate; its file sha256 is
  `003586a10204a7cc236669cc0e15ec8d48b5cd316f9861374b9ad3b05c14491e`

Rejected alternatives:

- Promote the first `yoy10` run despite strict Power BI critical gates.
- Stop at `yoy50` because PNGs were good while one monthly path critical
  remained.
- Patch July 2028 or any individual month after the solver.

Invariants not to break:

- Monthly solver remains the monthly level authority.
- Exact-source hierarchy policy is still required before treating
  `QUOTE_CONFLICT` rows as production accepted.
- Full promotion evidence still requires production, local export, and selected
  config manifest parity plus capstone; this daily run has not created the
  2026-07-07 selected config/policy artifacts.

## D-20260707-03 - 2026-07-07 Promotion Candidate Uses Amp150

Decision: supersede the `lshape100/yoy150/amp200` 2026-07-07 candidate with
`output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/` as the
promotion-ready candidate.

Reason: `amp200` passed strict Power BI, but the monthly sparse proof failed on
2028-06 vs 2029-06 and the aggregate 2028-2030 focus population gate. A solver
probe showed that lowering the structural monthly amplitude to `150` clears all
CRITICAL sparse-proof gates while preserving visible seasonal shape. This is an
objective/spec correction, not an individual month patch.

Key evidence:

- final CSV:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260706_lshape100_yoy150_amp150_2032.csv`
- CSV sha256:
  `4349f4774a4faace76f2022f2a6c39970eb40eb0916099a82801739d53381668`
- production manifest:
  `pfc_shaping/model/artifacts/production_monthly_curve_manifest.json`
- export manifest:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/fan_asof20260706_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json`
- selected config:
  `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260706_lshape100_yoy150_amp150_2032.json`
- source hierarchy policy:
  `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260706_lshape100_yoy150_amp150_2032.json`
- `active_config_hash`:
  `f95e81bf8987174eb8b553406de296fc8cfb67a3dfde35f006b88cc006a66469`
- `active_constraints_hash`:
  `451f9a23a4388973addabc7b467b535df2352bec74e69af34dcf4c28cdbeb15d`
- `monthly_solution_hash`:
  `a505fd3a07cb5573adc2a76061caa52f300ee90043766e10cfe4819107171a04`

Gate results:

- strict Power BI: `PASS`
- delivered-product audit: `all_gates_pass=true`,
  `accepted_quote_conflict_count=6`, `UNSUPPORTED=0`
- sparse proof: `PASS=33`, `WARNING=2`, `UNSUPPORTED=7`, `CRITICAL=0`
- capstone:
  `approved=true`, `status=PROMOTION_EVIDENCE_PASS`, `blocking_count=0`

OMPEX benchmark:

- read-only only; not used in model inputs, priors, objectives, or calibration
- file:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260707_101700.xlsx`
- overlap: `2026-07-01 00:00:00` -> `2031-01-01 00:00:00`
- points: `39473`, MAE `14.0163`, RMSE `18.4226`, bias `0.6167`,
  correlation `0.8330`

Rejected alternatives:

- Promote `amp200` because strict Power BI was already green.
- Relax or bypass sparse-proof CRITICAL gates.
- Patch June 2028 or June 2029 manually after the solver.
- Use OMPEX HFC benchmark data as a modeling input.

Invariants not to break:

- Monthly solver remains the monthly level authority.
- `QUOTE_CONFLICT` acceptance remains exact-artifact-bound.
- OMPEX remains read-only benchmark evidence only.
- Generated data/output artifacts remain non-commit targets unless explicitly
  requested.

## D-20260707-04 - Roasters Audit Confirms GO With Commit Hygiene Guardrails

Decision: accept the read-only Roasters/MIT audit verdict for the 2026-07-07
`amp150` candidate as GO for promotion governance, with no P0/P1 blocker.
Proceed only with curated staging; do not use broad `git add -A`.

Reason: three independent read-only audits found the production/export/selected
triad coherent, the capstone valid, strict Power BI and product-normalization
gates passing, no OMPEX model contamination, and no staged generated artifacts.

Accepted P2 findings:

- Generated `export_report.md` still says local/test and NO-GO, but it is not
  the selected governance artifact and is superseded by selected config plus
  capstone.
- Sparse-proof standalone manifest is less clear than the capstone because it
  records a diagnostic solution hash and `production_approved=false`; promotion
  must be read from the capstone using real production/export/selected
  manifests.
- Sparse-proof historical-threshold lineage is slightly messy but materially
  equivalent; this is traceability cleanup, not a promotion blocker.
- Residual warnings are accepted and documented: 2 sparse-proof warnings, 7
  far-horizon `UNSUPPORTED`, and 4 Power BI monthly path warnings, with no
  `CRITICAL`.

Commit guardrails:

- Include Phase 14 code/docs/governance artifacts only.
- Exclude `data/eex_forwards_history.parquet`, `output/**`,
  `pfc_shaping/output/**`, `pfc_shaping/model/artifacts/**`, and OMPEX
  benchmark outputs.
- Do not commit a fixed `forwards.eex_as_of_date` default. The `2026-07-06`
  as-of was a local run pin for this evidence package; durable config should
  default back to latest available quote row unless explicitly pinned.

Rejected alternatives:

- Treat the generated local export report as the promotion authority.
- Re-run tuning because of accepted P2 documentation findings.
- Stage all changed/untracked files.

Invariants not to break:

- Promotion evidence authority is the selected config plus capstone triad.
- OMPEX remains read-only benchmark evidence only.
- Monthly solver remains the level authority when enabled.

## D-20260708-01 - Daily Generation Uses 2026-07-07 Forward Snapshot

Decision: regenerate the Wednesday 2026-07-08 CH LT PFC using the latest usable
desk EEX quote row, `2026-07-07`, from the workbook available on 2026-07-08.
Bind all current artifacts to `forward_snapshot_date=2026-07-07`.

Reason: the EEX workbook file was refreshed on 2026-07-08, but CH/DE/FR quote
rows inside it stop at 2026-07-07. AT/IT remain unavailable at the exact
as-of date and are skipped rather than mixed into the fixed-as-of CH solve.

Key evidence:

- candidate:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/`
- CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- selected config:
  `.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260707_lshape100_yoy150_amp150_2032.json`
- source hierarchy policy:
  `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260707_lshape100_yoy150_amp150_2032.json`
- `active_config_hash`:
  `f95e81bf8987174eb8b553406de296fc8cfb67a3dfde35f006b88cc006a66469`
- `active_constraints_hash`:
  `fd95393bd94c2ce5d6ff02ba5c57a0633d00cbc9f6acc540877802fc81a2a7ab`
- `monthly_solution_hash`:
  `3882baa358bb2479d4b25aec464b45d74c15713f36ee34d0389790e848430c9e`

Gate results:

- strict Power BI: `PASS`
- delivered-product audit: `all_gates_pass=true`,
  `accepted_quote_conflict_count=6`, `UNSUPPORTED=0`
- sparse proof: `PASS=33`, `WARNING=2`, `UNSUPPORTED=7`, `CRITICAL=0`
- capstone:
  `approved=true`, `status=PROMOTION_EVIDENCE_PASS`, `blocking_count=0`
- targeted tests: `26 passed`

OMPEX benchmark:

- read-only only; not used in model inputs, priors, objectives, or calibration
- latest observed OMPEX file:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260707_101700.xlsx`
- no `2026-07-08` OMPEX file was observed
- points: `39473`, MAE `14.0197`, RMSE `18.4823`, bias `1.3495`,
  correlation `0.8331`

Rejected alternatives:

- Claim a 2026-07-08 forward snapshot from a workbook whose quote rows stop on
  2026-07-07.
- Mix stale AT/IT quotes into the exact 2026-07-07 CH solve.
- Use OMPEX benchmark data as a modeling input.

Invariants not to break:

- Snapshot dates in manifests mean quote-row dates, not file timestamps.
- Monthly solver remains the monthly level authority.
- Generated data/output artifacts remain non-commit targets unless explicitly
  requested.

## D-20260708-02 - Accept 2026-07-08 Roasters GO With Packaging Caveats

Decision: accept the read-only Roasters/MIT GO for the 2026-07-08 candidate
as promotion evidence, while treating `export_report.md` and standalone
sparse-proof manifest text as non-authoritative packaging artifacts.

Reason: all three agents found no P0 blocker. Quant/shaping and contamination
agents found no P1 blocker. The governance agent flagged packaging and
traceability caveats, but the authoritative capstone reports
`PROMOTION_EVIDENCE_PASS`, `approved=true`, and `blocking_count=0`, with
production/export/selected parity on:

- `active_config_hash=f95e81bf8987174eb8b553406de296fc8cfb67a3dfde35f006b88cc006a66469`
- `active_constraints_hash=fd95393bd94c2ce5d6ff02ba5c57a0633d00cbc9f6acc540877802fc81a2a7ab`
- `monthly_solution_hash=3882baa358bb2479d4b25aec464b45d74c15713f36ee34d0389790e848430c9e`

Accepted caveats:

- Generated `export_report.md` says production approval `NO`; it is local-test
  evidence only and must not be shipped as standalone promotion authority.
- Generated production manifest `source_hashes` is empty; export manifest still
  binds the forwards hash
  `159680087cb2f2de6322863660fb481fa531ebc9239e40de4f3735ecdc382ea1`.
- Sparse-proof standalone manifest has internal proof fields that are less
  clear than the capstone; capstone remains the authority.
- Sparse proof keeps `WARNING=2`, `UNSUPPORTED=7`, `CRITICAL=0`; these are
  non-blocking under the capstone.

Rejected alternatives:

- Override the capstone with the generated local export report.
- Re-tune the curve because of accepted P2 warnings without any CRITICAL gate.
- Commit refreshed forwards data or generated output artifacts as part of the
  audit documentation commit.

Invariants not to break:

- Promotion authority is selected config plus capstone triad.
- OMPEX remains read-only benchmark evidence only.
- Commit only curated docs/governance artifacts unless explicitly requested.

## D-20260708-03 - Resolve Production Manifest Source Hash Gap

Decision: harden production monthly curve manifest generation so production
manifests record hashes for the monthly solver forwards parquet and EEX
workbook when those files exist. Also clarify local export report wording so
it does not read as competing production-governance authority.

Reason: Roasters accepted the 2026-07-08 candidate, but flagged two packaging
caveats. Empty production `source_hashes` weakened audit traceability, and the
generated local report wording was easy to misread beside the capstone. The
solver hashes did not change; this is traceability hardening.

Evidence:

- regenerated production manifest sha256:
  `9b6b238bcbce72bb485f29ce1c6142ebce15b696d8df0a36fc5c673a2dbd4598`
- production manifest `source_hashes.forwards_path`:
  `159680087cb2f2de6322863660fb481fa531ebc9239e40de4f3735ecdc382ea1`
- production manifest `source_hashes.eex_report_path`:
  `dedae2a6d66ce59b9e3d4a0ab7c85e6800d8eb7d3e911d37651711a393fd4005`
- production manifest `monthly_solution_hash` unchanged:
  `3882baa358bb2479d4b25aec464b45d74c15713f36ee34d0389790e848430c9e`
- local export CSV sha unchanged:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- local export manifest sha unchanged:
  `cb52a502e8e95af2e5f3fabc3b2b34ca8f365999214cfd7c53718ed7f5ef456a`
- capstone rerun:
  `approved=true`, `status=PROMOTION_EVIDENCE_PASS`, `blocking_count=0`
- tests:
  `python -m pytest tests/test_long_term_branch.py tests/test_monthly_forward_curve_integration.py tests/test_check_monthly_curve_promotion_from_manifests.py -q -p no:cacheprovider`
  returned `41 passed`.

Rejected alternatives:

- Manually patch generated production manifests without changing code.
- Treat local `export_report.md` as production authority.
- Add generated data/output artifacts to Git.

Invariants not to break:

- Manifest source hashes are traceability metadata; monthly solver level and
  constraint hashes remain the promotion-critical parity fields.
- Local export reports stay local/test reports.
- Generated parquet/output artifacts remain out of curated commits.

## D-20260708-04 - Treat OMPEX as Imperfect Advisory Benchmark

Decision: formalize OMPEX/HFC as read-only advisory benchmark evidence. OMPEX
must not be treated as ground truth, model input, optimizer target, calibration
target, or production promotion authority.

Reason: OMPEX is useful for external comparison and can reveal shape issues,
but the desk view is that OMPEX also has errors and weaknesses. Optimizing the
HPFC to OMPEX directly would risk overfitting another vendor curve and could
degrade independent EEX/spot/physics consistency.

Implementation:

- Added `scripts/compare_hpfc_ompex_benchmark.py`.
- The script writes `benchmark_policy=advisory`, `read_only=true`,
  `ompex_used_in_model=false`, and an explicit OMPEX quality caveat.
- The script writes alignment sensitivity and supports auto-selection across:
  `direct`, `ompex_minus_1h_hourending`, and `ompex_plus_1h`.
- The 2026-07-08 comparison selects `ompex_minus_1h_hourending`, consistent
  with files timestamped as hour-ending.

2026-07-08 benchmark evidence:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ompex_benchmark_read_only_20260708_scripted/`
- OMPEX file:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260708_101700.xlsx`
- selected alignment: `ompex_minus_1h_hourending`
- points: `39481`
- MAE: `12.5271`
- RMSE: `16.4805`
- bias HPFC minus OMPEX: `0.7010`
- correlation: `0.8741`

Rejected alternatives:

- Fit or tune HPFC directly against OMPEX.
- Use an OMPEX benchmark improvement as sufficient proof of model improvement.
- Drop alignment sensitivity and assume all OMPEX files use one timestamp
  convention without evidence.

Invariants not to break:

- EEX BASE/PEAK residual gates and monthly solver authority dominate OMPEX
  benchmark evidence.
- OMPEX benchmark may guide diagnostics, but accepted model changes must be
  justified by independent spot/history/physics features.
- Generated OMPEX comparison outputs stay local evidence unless explicitly
  requested for packaging.

## D-20260708-05 - Add EPEX-Only A/B Shape Lab Scaffold

Decision: add an experimental LT-only EPEX shape lab scaffold, off by default,
to test mean-preserving hourly shape improvements without changing the
promotion-ready 2026-07-08 candidate or monthly solver authority.

Reason: OMPEX comparison is useful diagnostically but cannot be used as a
model target. The next admissible improvement path is to build candidate
hourly-shape deltas from independent CH EPEX spot residuals, then project those
deltas into the nullspace of active EEX BASE/PEAK/OFFPEAK average constraints.
This keeps the monthly solver as level authority while allowing controlled A/B
experiments on weekend, low-tail, and peak-subshape effects.

Implementation:

- Added `pfc_shaping/lt/model/epex_shape_lab.py`.
- Added `tests/test_epex_ab_shape_lab.py`.
- The lab fits templates only from rows strictly before the configured
  valuation timestamp, and fitting fails without an explicit
  `valuation_timestamp`.
- The apply path uses the same additive delta across slow/central/fast
  scenarios, shifts the existing weighted mean/fan by that delta, and preserves
  structural width.
- The delta is projected to zero residual against quote-aware
  BASE/PEAK/OFFPEAK constraints before application.
- By default, application requires monthly BASE constraints for every
  delivered month so post-solver use cannot preserve only annual/quarter means
  while rewriting solver monthly means.
- The manifest helper states that OMPEX/HFC is forbidden as input, target,
  loss, or gate, and marks artifacts `activation_status=lab_only` plus
  `production_approved=false`.

Validation:

- `python -m pytest tests/test_epex_ab_shape_lab.py -q -p no:cacheprovider`
  returned `9 passed`.
- `python -m pytest tests/test_epex_ab_shape_lab.py tests/test_seam_nullspace_smoothing.py tests/test_lt_quant_contract_matrix.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `39 passed, 1 skipped`.

Rejected alternatives:

- Tune directly against OMPEX or require OMPEX metric improvement as a gate.
- Patch individual months or post-solver levels.
- Wire the new lab into production/export by default before independent A/B
  evidence exists.
- Allow annual/quarter-only constraints in post-solver A/B application.

Invariants not to break:

- The 2026-07-08 candidate remains promotion-ready evidence; this lab is
  future experimental work only until explicitly wired and audited.
- Monthly solver authority and EEX BASE/PEAK residual gates remain dominant.
- The lab must remain LT-only and must not import `pfc_shaping.ct.*`.
- If a research caller disables the monthly-constraint guard, the output is
  not valid promotion evidence under `monthly_level_authority="solver"`.

## D-20260708-06 - Add Lab-Only EPEX A/B Runner

Decision: add a reproducible local runner for EPEX shape-lab A/B experiments,
without wiring the lab into production/export or changing promotion evidence.

Reason: after adding the off-production EPEX shape lab, the next admissible
step is a pre-registered A/B harness that applies the lab to a specific hourly
candidate while preserving the candidate's own monthly BASE/PEAK means. This
creates repeatable local evidence without using OMPEX as input, target, loss,
or gate.

Implementation:

- Added `scripts/run_epex_shape_lab_ab.py`.
- Added `tests/test_run_epex_shape_lab_ab_script.py`.
- The runner derives monthly BASE and PEAK constraints from the input
  candidate CSV itself, so the A/B delta cannot rewrite solver monthly levels.
- It writes:
  - `pre_registered_ab_plan.json`
  - `ab_lab_manifest.json`
  - `ab_lab_audit.csv`
  - `constraint_residuals_before_after.csv`
  - `epex_shape_templates.csv`
  - `candidate_epex_shape_lab_adjusted.csv`
- The manifest is explicitly `activation_status=lab_only`,
  `production_approved=false`, and `ompex_used_in_selection=false`.

2026-07-08 local trial:

- candidate:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- spot input:
  `data/epex_hourly.parquet`
- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/`
- valuation timestamp:
  `2026-07-07T00:00:00Z`
- intensities:
  `weekend=0.5`, `low_tail=0.5`, `peak_subshape=0.5`,
  `max_abs_delta=6.0`
- runtime after projection optimization:
  about `31` seconds on `57025` hourly rows
- monthly constraints:
  `base_monthly_constraints=78`, `peak_monthly_constraints=78`
- max after-constraint absolute error:
  `1.666666804567285e-07`
- weighted negative hours:
  `0`

Validation:

- `python -m pytest tests/test_run_epex_shape_lab_ab_script.py tests/test_epex_ab_shape_lab.py -q -p no:cacheprovider`
  returned `10 passed`.

Rejected alternatives:

- Add a production/export flag before independent A/B governance is complete.
- Compare or tune inside the runner against OMPEX.
- Preserve only annual/quarter products instead of candidate monthly solver
  levels.

Invariants not to break:

- Generated A/B output is local lab evidence, not a commit target and not
  promotion evidence.
- OMPEX can be run separately afterward as advisory comparison only.
- A future production wiring step requires strict Power BI/product/source
  hierarchy/capstone gates on the adjusted candidate.

## D-20260708-07 - Add Independent A/B Shape Comparison

Decision: add an independent baseline-vs-adjusted comparison script for EPEX
shape-lab trials, separate from OMPEX advisory benchmarking and from
production promotion gates.

Reason: the A/B runner can produce an adjusted candidate while preserving
monthly constraints, but the next governance step is to quantify the shape
effect without using OMPEX. The comparison must prove timestamp alignment,
monthly mean drift, fan-width preservation, quantile ordering, negative-hour
status, ramp changes, and calendar-bucket deltas before any external benchmark
is consulted.

Implementation:

- Added `scripts/compare_epex_shape_lab_ab.py`.
- Added `tests/test_compare_epex_shape_lab_ab_script.py`.
- The script writes:
  - `ab_comparison_summary.json`
  - `aligned_baseline_adjusted.csv`
  - `monthly_summary.csv`
  - `annual_summary.csv`
  - `calendar_delta_summary.csv`
- The summary records `benchmark_policy=independent_no_ompex`,
  `ompex_used_in_model=false`, and `ompex_used_in_selection=false`.

2026-07-08 independent A/B comparison:

- baseline:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- adjusted:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/candidate_epex_shape_lab_adjusted.csv`
- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/independent_ab_comparison/`
- `n_hours=57025`
- `finite_adjusted_ok=true`
- `quantile_order_adjusted_ok=true`
- `weighted_negative_hours_adjusted=0`
- `max_abs_monthly_mean_delta_eur_mwh=9.722222239124298e-08`
- `max_abs_width_delta_eur_mwh=0.0`
- `max_abs_delta_eur_mwh=6.000000000000002`
- calendar effects:
  - solar-tail mean delta `-2.0652588766029956`
  - midday mean delta `-1.8175837490740738`
  - evening-ramp mean delta `0.9288002084175085`
  - weekend mean delta `-0.6855023410557364`
- annual evening-minus-midday change is about `+2.75` EUR/MWh for 2027-2032.

Validation:

- `python -m pytest tests/test_compare_epex_shape_lab_ab_script.py -q -p no:cacheprovider`
  returned `2 passed`.

Rejected alternatives:

- Use OMPEX comparison as the primary A/B decision metric.
- Treat a lab-only A/B improvement as production promotion evidence.
- Skip timestamp alignment and monthly-drift checks.

Invariants not to break:

- Independent A/B comparison comes before advisory OMPEX comparison.
- OMPEX remains external read-only evidence only.
- Generated comparison outputs stay local artifacts unless explicitly
  requested for packaging.

## D-20260708-08 - Run OMPEX Advisory Post-Check After Independent A/B

Decision: run the OMPEX benchmark on the adjusted EPEX A/B lab candidate only
after the independent no-OMPEX A/B comparison was recorded. Treat the result as
external advisory evidence, not as parameter-selection evidence and not as a
promotion gate.

Reason: the A/B trial parameters were fixed before consulting OMPEX, and the
independent comparison already established monthly-level preservation,
fan-width preservation, quantile ordering, negative-hour status, and calendar
shape effects. OMPEX can now be used as an imperfect external sense-check
without contaminating model selection.

Command:

```powershell
python scripts/compare_hpfc_ompex_benchmark.py --hpfc-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/candidate_epex_shape_lab_adjusted.csv --ompex-xlsx "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260708_101700.xlsx" --output-dir output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ompex_advisory_adjusted_20260708
```

Output:

`output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ompex_advisory_adjusted_20260708/`

Adjusted advisory metrics:

- alignment: `ompex_minus_1h_hourending`
- points: `39481`
- MAE: `12.328552488842737`
- RMSE: `16.247141314210175`
- bias: `0.7010425073326411`
- correlation: `0.8775633169206011`
- p95 absolute error: `32.776404`
- OMPEX inside p10/p90 rate: `0.15807603657455485`
- max absolute error: `101.939482`

Baseline-vs-adjusted advisory deltas:

- MAE delta: `-0.1985248878447834`
- RMSE delta: `-0.2333863878943987`
- correlation delta: `0.0035026506205321217`
- p95 absolute error delta: `-0.5472940000000008`
- OMPEX inside p10/p90 rate delta: `0.0043058686456776685`
- max absolute error delta: `+1.553652999999997`

Rejected alternatives:

- Select or re-tune A/B parameters based on this OMPEX result.
- Treat the modest OMPEX improvement as production approval.
- Ignore the max-absolute-error deterioration because aggregate metrics
  improved.

Invariants not to break:

- OMPEX remains advisory and imperfect external evidence.
- Any next A/B parameter change must be pre-registered before looking at its
  OMPEX result.
- Production promotion still requires strict independent gates and capstone
  evidence, not OMPEX benchmark movement.

## D-20260708-09 - Add EPEX A/B Lab Governance Audit

Decision: add a lab-only governance audit for EPEX shape-lab artifacts. The
audit verifies contamination controls and preservation checks, but it is not a
production promotion gate.

Reason: after running independent no-OMPEX diagnostics and the separate OMPEX
advisory post-check, the evidence chain needed an automated check that future
readers cannot accidentally interpret OMPEX as parameter-selection evidence or
production approval.

Implementation:

- Added `scripts/audit_epex_shape_lab_governance.py`.
- Added `tests/test_audit_epex_shape_lab_governance_script.py`.
- The audit checks:
  - `activation_status=lab_only`
  - `production_approved=false`
  - OMPEX not used for model or selection
  - independent comparison policy is `independent_no_ompex`
  - monthly BASE/PEAK constraints are present
  - after-constraint error stays below threshold
  - monthly mean drift and fan-width drift stay below thresholds
  - adjusted candidate remains finite, quantile-ordered, and without weighted
    negative hours
  - OMPEX metrics, when supplied, are `advisory` and `read_only`

2026-07-08 governance audit:

Command:

```powershell
python scripts/audit_epex_shape_lab_governance.py --lab-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ab_lab_manifest.json --independent-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/independent_ab_comparison/ab_comparison_summary.json --ompex-metrics output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/ompex_advisory_adjusted_20260708/benchmark_metrics.json --output-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/governance_audit/epex_shape_lab_governance_audit.json
```

Result:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/governance_audit/epex_shape_lab_governance_audit.json`
- status: `PASS`
- failed count: `0`
- production approval: `NO`
- promotion gate: `false`
- OMPEX role: `advisory_post_check_only`
- monthly BASE constraints: `78`
- monthly PEAK constraints: `78`
- after-constraint error: `1.666666804567285e-07`
- independent monthly drift: `9.722222239124298e-08`
- independent fan-width drift: `0.0`

Validation:

- `python -m pytest tests/test_audit_epex_shape_lab_governance_script.py -q -p no:cacheprovider`
  returned `2 passed`.
- `python -m pytest tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `31 passed, 1 skipped`.

Rejected alternatives:

- Treat OMPEX post-check results as sufficient proof of model improvement.
- Promote adjusted A/B output without strict product/source/Power BI/capstone
  gates.
- Leave governance interpretation to prose only.

Invariants not to break:

- This audit is a lab-governance check, not production approval.
- Any future A/B parameter change requires a fresh pre-registration and
  independent no-OMPEX comparison before OMPEX advisory review.

## D-20260708-10 - Run Lab-Only Promotion-Style Diagnostics On Adjusted A/B

Decision: run existing shape, Power BI, and product-normalization diagnostics on
the adjusted EPEX A/B candidate as lab-only evidence. Do not interpret these
runs as production promotion because the adjusted artifact has no
production-approved source hierarchy policy and was not generated by the
production pipeline.

Reason: the A/B lab governance audit proved contamination controls, but we
also need to know whether the adjusted candidate would break existing
diagnostics. The current committed `data/eex_forwards_history.parquet` is stale
for this candidate (`max_date=2026-06-17`), so a local Yearly-only diagnostic
forwards parquet was built under the trial output folder from
`Price_Report_EEX_Yearly.xlsx` to cover snapshot `2026-07-07`.

Diagnostic forwards:

- path:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/diagnostic_forwards_yearly_only.parquet`
- source:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx`
- CH coverage:
  `2024-07-01 -> 2026-07-07`
- rows:
  `53620` all markets, `7912` CH
- sha256:
  `63a40871677a0a82356de762d5a9ceb944a6b431f145d23598b8fb91e6966ce3`

Shape audit:

- adjusted output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/adjusted_shape_audit/shape_audit_report.md`
- baseline output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/baseline_shape_audit/shape_audit_report.md`
- both baseline and adjusted report `score=7.00/10` under this local shape
  audit, so the adjusted A/B does not degrade that local score.

Power BI strict diagnostic on adjusted:

- command used no `--allow-failed-gates`
- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/adjusted_powerbi_strict/`
- `powerbi_quality_gate_status=PASS`
- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `weighted_negative_hours=0`
- `negative_gate_status=PASS`
- `monthly_path_warning_flags=4`
- all critical flag counts are `0`

Product normalization diagnostic:

- adjusted output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/adjusted_product_normalization/`
- baseline output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_ab_trial/baseline_product_normalization/`
- adjusted and baseline both have:
  - `critical_count=0`
  - `unsupported_count=0`
  - `delivered_curve_drift_count=0`
  - `quote_conflict_count=6`
  - `status_counts={"PASS": 90, "QUOTE_CONFLICT": 6}`
  - `source_hierarchy_policy.status=NOT_PROVIDED`
- adjusted supported hard-gate max residual:
  `0.0887692576922916`
- baseline supported hard-gate max residual:
  `0.0887692871795025`
- because no production-approved policy was supplied, quote conflicts remain
  blocking and `all_gates_pass=false`; this is expected for lab-only evidence.

Rejected alternatives:

- Reuse the production source hierarchy policy for the adjusted lab artifact.
  It is hash-bound to the baseline artifact and would be the wrong authority.
- Treat the Yearly-only diagnostic forwards parquet as a replacement for the
  canonical committed data cache.
- Promote the adjusted A/B because Power BI diagnostic passed.

Invariants not to break:

- Lab-only adjusted artifacts require a fresh production/export/capstone path
  before any promotion claim.
- Product-normalization quote conflicts can be accepted only by an explicit
  production-approved policy bound to the exact artifact.
- Generated diagnostic forwards and audit outputs stay local artifacts.

## D-20260708-11 - Pre-Register Next EPEX Shape-Lab Sweep

Decision: add a no-OMPEX sweep plan generator for the next EPEX shape-lab
parameter wave and generate a local pre-registered sweep plan for the
2026-07-08 candidate.

Reason: OMPEX has already been run as an advisory post-check on the first
adjusted A/B trial. Any subsequent parameter exploration must be
pre-registered before execution and selected only on independent no-OMPEX
diagnostics, otherwise OMPEX would become an implicit tuning target.

Implementation:

- Added `scripts/plan_epex_shape_lab_sweep.py`.
- Added `tests/test_plan_epex_shape_lab_sweep_script.py`.
- The plan generator writes candidate/spot hashes, trial parameters, output
  directories, and the commands to run:
  - `scripts/run_epex_shape_lab_ab.py`
  - `scripts/compare_epex_shape_lab_ab.py`
  - `scripts/audit_epex_shape_lab_governance.py`
- It does not read OMPEX/HFC and records:
  - `benchmark_policy=pre_registered_independent_no_ompex`
  - `ompex_used_in_model=false`
  - `ompex_used_in_selection=false`
  - `forbidden_selection_inputs=["OMPEX","HFC_OMPEX","external_HPFC_benchmark"]`

2026-07-08 local plan:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/pre_registered_sweep_plan.json`
- plan id:
  `epex_shape_lab_sweep_v1_asof20260707`
- trial count:
  `27`
- candidate CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- EPEX spot parquet sha256:
  `5718d243ef681476cabeabac7e866c0c7a63f686750283a2ff50a7d70c216a3d`
- grid:
  - weekend intensity: `[0.25, 0.5, 0.75]`
  - low-tail intensity: `[0.25, 0.5, 0.75]`
  - peak-subshape intensity: `[0.25, 0.5, 0.75]`
  - max absolute delta: `6.0`

Validation:

- `python -m pytest tests/test_plan_epex_shape_lab_sweep_script.py -q -p no:cacheprovider`
  returned `1 passed`.

Rejected alternatives:

- Continue tuning the already observed A/B parameters by reading OMPEX
  outcomes.
- Embed OMPEX commands in the pre-selection sweep plan.
- Treat the plan itself as model evidence; it is only a pre-registration
  artifact until trials are run and audited.

Invariants not to break:

- Selection for this sweep must use independent comparison and governance
  outputs only.
- OMPEX can be run only after a trial is selected/frozen, as advisory
  post-check evidence.
- Generated sweep plan and future trial outputs remain local artifacts unless
  explicitly requested for packaging.

## D-20260708-12 - Execute Pre-Registered No-OMPEX EPEX Sweep

Decision: add and run a sweep executor for the pre-registered EPEX shape-lab
plan, selecting the best trial only from independent no-OMPEX diagnostics and
governance PASS evidence.

Reason: after the initial A/B trial had already been compared to OMPEX as an
advisory post-check, any further parameter choice needed to be frozen by an
independent process that could not turn OMPEX into an implicit optimization
target.

Implementation:

- Added `scripts/execute_epex_shape_lab_sweep.py`.
- Added `tests/test_execute_epex_shape_lab_sweep_script.py`.
- The executor validates the plan policy, candidate hash, EPEX spot hash,
  lab-only status, and OMPEX exclusion before running trials.
- Post-audit hardening rejects malformed plans, duplicate trials, output
  directories outside the sweep root, negative `--max-trials`, stale resume
  manifests/comparisons/governance inputs, and reports `best_trial=null` when
  no trial is eligible.
- For each trial it runs the A/B adjustment, the independent baseline-vs-
  adjusted comparison, and the governance audit.
- It writes a JSON summary and CSV ranking without reading OMPEX/HFC inputs.

2026-07-08 local execution:

- plan:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/pre_registered_sweep_plan.json`
- summary:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/sweep_execution_summary.json`
- ranking:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_shape_lab_sweep_v1/sweep_execution_summary.csv`
- `benchmark_policy=executed_independent_no_ompex`
- `trial_count_planned=27`
- `trial_count_executed=27`
- `eligible_count=27`
- `production_approved=false`
- `ompex_used_in_model=false`
- `ompex_used_in_selection=false`

Best no-OMPEX trial:

- `trial_id=trial_002_w0.25_l0.25_p0.50`
- weekend intensity: `0.25`
- low-tail intensity: `0.25`
- peak-subshape intensity: `0.50`
- independent shape score: `6.350975764045719`
- duck-change mean: `3.6754139784914535` EUR/MWh
- solar-tail mean delta: `-2.535581627746391` EUR/MWh
- weekend mean delta: `-0.6477966303078719` EUR/MWh
- ramp p99 increase: `2.0312658899999896` EUR/MWh
- max monthly mean drift: `1.1155913942688404e-07` EUR/MWh
- max fan-width drift: `0.0`
- weighted negative hours: `0`
- governance status: `PASS`
- max after-constraint residual:
  `1.6666673730014736e-07` EUR/MWh

Validation:

- `python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py -q -p no:cacheprovider`
  returned `5 passed`.
- `python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `37 passed, 1 skipped`.
- Resume check on the existing 27-trial sweep returned:
  `{"eligible_count": 27, "trial_count_executed": 27}`.

Rejected alternatives:

- Select from the grid using OMPEX deltas.
- Promote the adjusted trial directly because governance passed.
- Commit generated sweep outputs as repo evidence.

Invariants not to break:

- The selected sweep trial remains lab-only until regenerated through a
  production/export/capstone path with artifact-bound policies.
- OMPEX may be run only after this trial selection is frozen, and only as an
  advisory benchmark.
- Generated trial outputs remain local artifacts unless explicitly packaged.

## D-20260708-13 - Pre-Register EPEX Sweep Selection Policy

Decision: harden the EPEX shape-lab sweep plan and executor so future sweeps
carry and enforce explicit no-OMPEX selection thresholds and scoring weights.

Reason: read-only expert audits found that the first sweep was correctly
no-OMPEX and governance-clean, but still not adoption-ready as a research
model-selection protocol. The EPEX spot history used by the best trial was too
stale for a 2026-07-07 valuation, all trials saturated the `6.0` EUR/MWh cap,
and the ranking over-weighted an internal duck-shape score relative to ramp and
negative-price risk.

Implementation:

- `scripts/plan_epex_shape_lab_sweep.py` now writes:
  - `selection_thresholds`
  - `scoring_policy`
  - `max_abs_delta_grid`
- Default new-plan thresholds:
  - `max_epex_spot_age_days=14.0`
  - `min_epex_fit_coverage_days=730.0`
  - `max_ramp_p99_increase_eur_mwh=1.0`
  - `min_adjusted_price_eur_mwh=-10.0`
- Default new-plan scoring:
  - `duck_weight=1.0`
  - `solar_tail_weight=1.0`
  - `weekend_weight=1.0`
  - `ramp_penalty_weight=1.0`
- The planner can now pre-register a cap grid via
  `--max-abs-delta-grid-json`, e.g. `[2.0, 3.0, 4.0, 6.0]`.
- `scripts/execute_epex_shape_lab_sweep.py` now records EPEX spot age and fit
  coverage per trial and applies pre-registered freshness, coverage, ramp, and
  minimum-price thresholds before marking a trial eligible.
- Existing historical plans without these policy fields remain readable as
  already-frozen lab evidence; new plans include the stricter policy.

Validation:

- `python -m pytest tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py -q -p no:cacheprovider`
  returned `8 passed`.
- `python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `39 passed, 1 skipped`.

Rejected alternatives:

- Adopt the already selected `trial_002_w0.25_l0.25_p0.50` as production
  evidence.
- Continue scoring future sweeps with the legacy weak ramp penalty.
- Let stale EPEX spot data remain an advisory note instead of a hard
  pre-registered eligibility criterion.

Invariants not to break:

- OMPEX/HFC remains forbidden as an input, target, loss, selection criterion,
  or governance gate.
- Baseline `20260708_asof20260707_lshape100_yoy150_amp150_2032` remains the
  only promotion-ready 2026-07-08 candidate until a future adjusted artifact
  passes a full production/export/capstone chain.
- Future EPEX lab promotion requires refreshed EPEX spot evidence, an
  artifact-bound source hierarchy policy, strict product normalization,
  strict Power BI gates, and capstone triad evidence.

## D-20260708-14 - Run Fresh-Spot EPEX Sweep V2

Decision: refresh a local EPEX spot copy, pre-register and execute a second
no-OMPEX EPEX shape-lab sweep with stricter selection gates and a delta-cap
grid, then run OMPEX only as an advisory post-check after the best trial was
frozen.

Reason: the first EPEX sweep was methodologically clean but fitted stale spot
history ending `2026-03-15 22:00 UTC`. The next expert step was to refresh
EPEX spot, enforce freshness/coverage/ramp/min-price thresholds, and reduce
cap saturation risk with a cap grid.

Local refreshed spot evidence:

- built under:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/`
- source baseline:
  `pfc_shaping/data/epex_15min.parquet`
- incremental source:
  energy-charts CH prices fetched for `2026-03-15 -> 2026-07-09`
- local 15min output:
  `epex_15min_ch_energy_charts_20260708.parquet`
- local hourly output:
  `epex_hourly_ch_energy_charts_20260708.parquet`
- hourly coverage:
  `2023-01-01 00:00 UTC -> 2026-07-08 23:00 UTC`
- no repository data cache was committed.

V2 sweep:

- plan:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/pre_registered_sweep_plan.json`
- summary:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/sweep_execution_summary.json`
- ranking:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/sweep_execution_summary.csv`
- plan id:
  `epex_sweep_v2_fresh_spot_asof20260707`
- trial count:
  `108`
- cap grid:
  `[2.0, 3.0, 4.0, 6.0]`
- selection thresholds:
  - `max_epex_spot_age_days=14.0`
  - `min_epex_fit_coverage_days=730.0`
  - `max_ramp_p99_increase_eur_mwh=1.0`
  - `min_adjusted_price_eur_mwh=-10.0`
- scoring:
  `duck + solar_tail + weekend - ramp_p99_increase`
- execution result:
  - `trial_count_executed=108`
  - `eligible_count=39`
  - `production_approved=false`
  - `ompex_used_in_model=false`
  - `ompex_used_in_selection=false`

Best no-OMPEX V2 trial:

- `trial_id=t046_w05_l025_p075_d03`
- weekend intensity: `0.5`
- low-tail intensity: `0.25`
- peak-subshape intensity: `0.75`
- max delta cap: `3.0` EUR/MWh
- independent shape score: `2.2242277207731145`
- duck-change mean: `1.6614858713007632` EUR/MWh
- solar-tail mean delta: `-1.172939635010313` EUR/MWh
- weekend mean delta: `-0.3774464344619922` EUR/MWh
- ramp p99 increase: `0.9876442199999538` EUR/MWh
- min adjusted price: `-3.825623` EUR/MWh
- EPEX spot age: `0.041666666666666664` days
- EPEX fit coverage: `1282.9583333333333` days
- max monthly mean drift: `8.602150532151586e-08` EUR/MWh
- max fan-width drift: `0.0`
- weighted negative hours: `0`
- governance status: `PASS`

OMPEX advisory post-check after no-OMPEX selection:

- OMPEX file:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260708_101700.xlsx`
- baseline output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/ompex_advisory_baseline_20260708/`
- selected output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/ompex_advisory_selected_t046_20260708/`
- delta summary:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/ompex_advisory_delta_selected_t046_20260708.json`
- advisory deltas selected minus baseline:
  - MAE: `-0.13162060282161114`
  - RMSE: `-0.1631453735600843`
  - correlation: `+0.0024745976229031408`
  - p95 absolute error: `-0.40473999999999677`
  - inside p10/p90 rate: `+0.002482206631037709`
  - max absolute error: `+0.987836999999999`

Validation:

- `python -m pytest tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `40 passed, 1 skipped`.

Rejected alternatives:

- Run V2 on stale `data/epex_hourly.parquet`.
- Select a high-cap trial with better internal shape score but ramp p99 above
  the pre-registered threshold.
- Use the OMPEX 2026-07-08 benchmark to select or re-rank the V2 grid.

Invariants not to break:

- V2 remains lab-only and does not replace the promotion-ready baseline.
- OMPEX advisory improvement is not promotion evidence and cannot be used to
  tune another V2/V3 grid.
- Any production adoption of `t046_w05_l025_p075_d03` requires a new
  production/export/capstone chain and artifact-bound source hierarchy policy.

## D-20260708-15 - Run Strict Diagnostics on Frozen T046 Lab Trial

Decision: run strict product-normalization and Power BI diagnostics on the
frozen V2 trial `t046_w05_l025_p075_d03`, and add a hash-bound source hierarchy
policy for its redundant quote conflicts. Keep the trial NO-GO production until
it is regenerated through a real production/export/capstone chain.

Reason: V2 selected `t046` using only pre-registered no-OMPEX criteria. The
next promotion-readiness question was whether the selected adjusted artifact
still respects delivered EEX BASE/PEAK/OFFPEAK product governance and strict
Power BI quality gates under fresh local forwards/spot evidence.

Forwards evidence:

- `data/eex_forwards_history.parquet` was stale locally (`max_date=2026-06-17`)
  and therefore rejected for a `2026-07-07` required snapshot.
- A local diagnostic forwards parquet was rebuilt from desk workbooks:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet`
- Source workbooks:
  - `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx`
  - `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Hist.xlsx`
- Rebuilt CH coverage:
  `2020-05-04 -> 2026-07-07`
- rebuilt forwards sha256:
  `a6244638c2234781853284ce2ad58d55d01265568cca6c85d4461f21446e8d76`

Source hierarchy policy:

- added:
  `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t046_asof20260707_fresh_epex_sweep_v2.json`
- policy sha256 from audit:
  `b79aec178312816e7d9554065a2e2acc0d0b419c43d3b85b4373639e22dc64df`
- bound input CSV sha256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- bound forwards sha256:
  `a6244638c2234781853284ce2ad58d55d01265568cca6c85d4461f21446e8d76`
- quote conflict identity hash:
  `a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`
- expected quote conflicts:
  `6`, same redundant parent/finer-bucket identities as the 2026-07-08
  baseline.

Product normalization strict:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_product_normalization_with_policy/`
- command used no `--allow-failed-gates`
- `all_gates_pass=true`
- `critical_count=0`
- `unsupported_count=0`
- `quote_conflict_count=6`
- `accepted_quote_conflict_count=6`
- `blocking_quote_conflict_count=0`
- `delivered_curve_drift_count=0`
- `status_counts={"PASS": 90, "QUOTE_CONFLICT": 6}`
- supported hard-gate max residual:
  `0.0887692589743665` EUR/MWh

Power BI strict:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_powerbi_strict/`
- command used no `--allow-failed-gates`
- `powerbi_quality_gate_status=PASS`
- `shape_score_10=9`
- `hfc_vs_spot_score_10=9`
- `max_eex_base_error_eur_mwh=0.000000`
- `max_eex_peak_error_eur_mwh=0.000000`
- `weighted_negative_hours=0`
- `negative_gate_status=PASS`
- `min_weighted_eur_mwh=4.84`
- `min_price_eur_mwh=-3.83`
- `p10_negative_hours=118`
- `monthly_path_warning_flags=4`
- all critical flag counts are `0`

Rejected alternatives:

- Treat the policy as production approval for the adjusted curve.
- Reuse the baseline policy despite the different adjusted CSV hash.
- Run product normalization against stale `data/eex_forwards_history.parquet`.
- Use OMPEX advisory deltas as a promotion gate.

Invariants not to break:

- The policy accepts only source hierarchy quote conflicts for the exact
  t046 CSV and rebuilt forwards hashes.
- T046 still lacks production manifest, export manifest, selected config
  artifact, and capstone evidence, so it remains NO-GO production.
- Generated forwards, Power BI outputs, and audit CSV/JSON artifacts stay
  local and are not committed.

## D-20260708-16 - Add T046 Lab Promotion Readiness Checker

Decision: add a dedicated readiness checker for selected EPEX lab artifacts
instead of reusing the monthly solver capstone as if it approved the adjusted
hourly CSV.

Reason: the existing capstone proves the baseline monthly solver triad. T046
preserves monthly constraints and passes strict diagnostics, but it is still a
lab-only hourly adjustment. A separate machine-readable decision is needed to
show exactly which evidence passes and which production-promotion evidence is
still missing.

Implementation:

- Added `scripts/check_epex_lab_promotion_readiness.py`.
- Added `tests/test_check_epex_lab_promotion_readiness_script.py`.
- The checker reads:
  - lab manifest
  - EPEX lab governance audit
  - independent no-OMPEX A/B summary
  - product-normalization summary
  - Power BI summary metrics
  - optional OMPEX advisory delta
  - optional adjusted production/export/selected/capstone artifacts
- It returns non-zero unless the adjusted artifact has both strict diagnostics
  and a complete adjusted production chain.

T046 readiness execution:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision.json`
- status:
  `STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
- `approved=false`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- missing production evidence:
  - `adjusted_production_manifest`
  - `adjusted_export_manifest`
  - `adjusted_selected_config`
  - `adjusted_capstone`
- all checker inputs pass:
  - lab-only / not production approved
  - OMPEX not used in selection
  - governance PASS
  - independent no-OMPEX comparison
  - product strict gates pass
  - Power BI strict gates pass
  - OMPEX advisory is read-only and not selection

Validation:

- `python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider`
  returned `2 passed`.
- `python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `86 passed, 1 skipped`.

Rejected alternatives:

- Reuse the baseline monthly capstone as production approval for t046.
- Treat strict product/Power BI diagnostics as equivalent to a production
  chain.
- Return success from the checker when production evidence is missing.

Invariants not to break:

- A selected lab trial cannot become promotion-ready without its own adjusted
  production manifest, export manifest, selected config artifact, and capstone.
- OMPEX remains advisory evidence only.
- Generated readiness decision JSON stays local evidence unless explicitly
  packaged.

## D-20260708-17 - Package T046 Local Adjusted Evidence Bundle

Decision: add a local bundle builder for selected EPEX lab artifacts and use it
to package T046 export/selected/local-capstone evidence without granting
production approval.

Reason: after strict diagnostics passed, the readiness checker still reported
four missing artifacts. Three of those can be truthfully represented as local
non-production evidence: adjusted export manifest, adjusted selected artifact,
and local capstone NO-GO decision. The remaining missing item must stay
`adjusted_production_manifest`, because no real production run has produced or
approved the adjusted hourly artifact.

Implementation:

- Added `scripts/build_epex_lab_promotion_bundle.py`.
- Added `tests/test_build_epex_lab_promotion_bundle_script.py`.
- Hardened `scripts/check_epex_lab_promotion_readiness.py` to validate
  provided adjusted export/selected artifacts are bound to the adjusted CSV and
  explicitly not production-approved.

T046 local bundle:

- output directory:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_local_promotion_bundle/`
- adjusted export manifest:
  `adjusted_export_manifest.json`
- adjusted selected artifact:
  `adjusted_selected_artifact.json`
- adjusted local capstone:
  `adjusted_local_capstone_no_go.json`
- all three declare local/lab diagnostic scope and no production approval.

Updated T046 readiness:

- output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_promotion_readiness/decision_with_local_bundle.json`
- `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
- `approved=false`
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- missing production evidence now only:
  - `adjusted_production_manifest`
- additional checker PASS rows:
  - `adjusted_export_manifest_bound`
  - `adjusted_export_manifest_not_production_approved`
  - `adjusted_selected_artifact_bound`
  - `adjusted_selected_artifact_not_production_approved`

Validation:

- `python -m pytest tests/test_build_epex_lab_promotion_bundle_script.py tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider`
  returned `3 passed`.
- `python -m pytest tests/test_build_epex_lab_promotion_bundle_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `87 passed, 1 skipped`.

Rejected alternatives:

- Create a fake adjusted production manifest.
- Mark the local capstone approved.
- Leave export/selected artifacts missing even though they can be represented
  accurately as local diagnostic evidence.

Invariants not to break:

- T046 remains NO-GO production until a real adjusted production manifest
  exists and the adjusted artifact is promoted by an approved production
  capstone.
- Local bundle JSONs are generated evidence and are not committed by default.
- The baseline 2026-07-08 candidate remains the production-ready artifact.

## D-20260708-18 - Harden T046 Promotion Readiness Against Lab Reclassification

Decision: the EPEX T046 lab manifest must remain `activation_status=lab_only`
and `production_approved=false`; promotion readiness can pass only through a
separate adjusted production/export/selected/capstone chain that is explicitly
production-approved and bound to the adjusted CSV by path or hash.

Reason: read-only governance, data, and quant reviewers converged that T046 is
strict-diagnostic PASS but still NO-GO production. The unsafe shortcut would be
to reclassify local lab evidence, or create an `adjusted_production_manifest`
post-hoc. The readiness checker now validates the contents of provided adjusted
production/export/selected artifacts instead of treating their existence as
sufficient evidence.

Implementation:

- Hardened `scripts/check_epex_lab_promotion_readiness.py`:
  - keeps strict diagnostic checks separate from production-chain checks;
  - requires `adjusted_production_manifest` schema
    `epex_lab_adjusted_production_manifest.v1`;
  - requires production/export/selected artifacts to be bound to the adjusted
    CSV by path or SHA-256;
  - requires `production_approved=true` and
    `production_promotion_approved=true` for adjusted production/export
    evidence;
  - requires selected artifact
    `selection_status="PRODUCTION_APPROVED"`;
  - requires capstone `approved=true`;
  - no longer requires the lab manifest itself to become
    `production_approved=true`.
- Extended `tests/test_check_epex_lab_promotion_readiness_script.py` with:
  - rejection of an unapproved adjusted production manifest;
  - acceptance of a fully separate approved production chain.

Validation:

- `python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_promotion_bundle_script.py -q -p no:cacheprovider`
  returned `5 passed`.
- `python -m pytest tests/test_build_epex_lab_promotion_bundle_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `89 passed, 1 skipped`.

Rejected alternatives:

- Set `lab.production_approved=true` to make the readiness checker pass.
- Treat the local T046 bundle as production evidence.
- Accept an adjusted production manifest without schema, approval flags, and
  adjusted CSV binding.

Invariants not to break:

- T046 remains NO-GO production until the adjusted curve is emitted by a real
  production path and promoted by a real capstone.
- OMPEX remains an advisory benchmark only and must not enter model training,
  selection, losses, or gates.
- The baseline 2026-07-08 candidate remains the only current production-ready
  artifact.

## D-20260708-19 - Add T046 Adjusted Production Manifest Contract

Decision: add a dedicated adjusted-production manifest builder for EPEX lab
artifacts, with CLI output deliberately NO-GO by default.

Reason: after hardening readiness, the remaining T046 blocker is a real
`adjusted_production_manifest`. Creating a JSON file by hand would be a false
proof. The next safe step is to define the exact schema and checks that a real
off-by-default LT production path must satisfy, while keeping CLI-generated
contract manifests non-promotional.

Implementation:

- Added `scripts/build_epex_lab_adjusted_production_manifest.py`.
- Added `tests/test_build_epex_lab_adjusted_production_manifest_script.py`.
- The builder emits schema
  `epex_lab_adjusted_production_manifest.v1`.
- The manifest records SHA-256 bindings for:
  - adjusted CSV;
  - lab manifest;
  - baseline monthly solver manifest;
  - source hierarchy policy;
  - product normalization summary;
  - Power BI summary;
  - independent EPEX summary;
  - governance audit.
- The contract requires:
  - lab artifact remains `activation_status=lab_only` and
    `production_approved=false`;
  - monthly authority is `solver`;
  - source hierarchy policy is production-approved;
  - product normalization and Power BI strict gates pass;
  - independent/governance evidence confirms no OMPEX in model or selection.
- CLI-built manifests keep:
  - `production_approved=false`;
  - `production_promotion_approved=false`;
  - `promotion_scope=LT_EPEX_LAB_PRODUCTION_CONTRACT_NO_GO`.
- Even through the Python API, setting production approval requires complete
  run identity: `production_run_id`, `production_entrypoint`, and `git_commit`.

Validation:

- `python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_promotion_bundle_script.py -q -p no:cacheprovider`
  returned `8 passed`.
- `python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `91 passed, 1 skipped`.

Rejected alternatives:

- Generate a local adjusted production manifest and mark it approved.
- Treat the existing lab CSV as having come from the production path.
- Wire the EPEX lab directly into the 15-minute production PFC before solving
  the format mismatch with the audited hourly export.

Invariants not to break:

- A CLI-built adjusted production manifest is contract evidence only, not a
  promotion approval.
- A future approval must come from a real off-by-default LT production path
  that emits the audited hourly adjusted artifact and then passes product,
  Power BI, selected-artifact, and capstone gates.
- OMPEX remains benchmark/advisory only.

## D-20260708-20 - Stage LT-Only EPEX Adjusted Candidate From Fan Evidence

Decision: add an off-by-default LT-only staging runner that starts from a
governed fan parquet or an already exported hourly CH CSV, applies the EPEX lab
with explicit parameters, and writes NO-GO hash-bound staging evidence.

Reason: T046 cannot be promoted from the ad hoc lab CSV alone, and the
production 15-minute PFC format is not the same artifact as the audited hourly
HFC export. The next safe production-integration step is a reproducible bridge
from LT fan evidence to the hourly EPEX-adjusted candidate, while preserving
the rule that staging evidence is not promotion evidence.

Implementation:

- Added `scripts/stage_epex_lab_adjusted_lt_candidate.py`.
- Added `tests/test_stage_epex_lab_adjusted_lt_candidate_script.py`.
- The runner accepts exactly one source:
  - `--fan-parquet` plus `--local-start-date` / `--local-end-date`; or
  - `--candidate-csv`.
- It reuses existing governed functions:
  - `to_hourly_csv_frame` for LT fan to hourly HFC conversion;
  - `run_ab` for the EPEX-only lab adjustment;
  - `build_epex_lab_adjusted_production_manifest.py` when all strict evidence
    paths are supplied.
- Defaults match the selected T046 lab settings:
  - `weekend_intensity=0.5`;
  - `low_tail_intensity=0.25`;
  - `peak_subshape_intensity=0.75`;
  - `max_abs_delta_eur_mwh=3.0`;
  - `negative_price_floor=-10.0`;
  - `max_weighted_negative_hours=0`.
- The staging manifest schema is
  `epex_lab_adjusted_lt_candidate_stage.v1` and keeps:
  - `activation_status=staged_lab_only`;
  - `production_approved=false`;
  - `production_promotion_approved=false`;
  - `promotion_scope=LT_EPEX_LAB_STAGING_NO_GO`;
  - `ompex_used_in_model=false`;
  - `ompex_used_in_selection=false`.

Validation:

- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_run_epex_shape_lab_ab_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider`
  returned `11 passed`.
- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_compare_hpfc_ompex_benchmark_script.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `124 passed, 1 skipped`.

Rejected alternatives:

- Apply T046 directly to the 15-minute production PFC and call it the same
  audited hourly artifact.
- Treat staging manifests as production approval.
- Read OMPEX in the staging path.

Invariants not to break:

- The staging runner remains LT-only and must not import CT modules.
- Staging output is local evidence; production GO still requires regenerated
  product normalization, Power BI strict, selected artifact, and capstone
  evidence on the staged adjusted CSV.
- OMPEX remains advisory only.

## D-20260708-21 - Execute T046 Staging on Real 20260708 Evidence

Decision: use the audited hourly baseline CSV, not the raw fan parquet, as the
promotion-facing staging source for T046 until the fan-to-hourly production
export path itself is proven by strict product gates.

Reason: executing the new staging runner on the 20260708 fan parquet produced
a reproducible adjusted CSV, but the resulting product-normalization diagnostic
failed hard repricing gates (`critical_count=56`,
`delivered_curve_drift_count=38`, max residual about `21.92 EUR/MWh`). The
same runner executed from the already audited hourly baseline reproduced the
known T046 adjusted artifact exactly and passed strict diagnostics.

Real staging evidence:

- Fan-source staging output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_fan/`
- Fan-source adjusted CSV SHA-256:
  `f85b868d08f4b6e43f810708d8d88973f5de98219f7ef20e12e056287cc3573c`
- Fan-source diagnostic product audit:
  `product_normalization_diagnostic/summary.json`
- Fan-source product result:
  - `all_gates_pass=false`
  - `critical_count=56`
  - `delivered_curve_drift_count=38`
  - `blocking_quote_conflict_count=4`
- Hourly-baseline staging output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline/`
- Hourly-baseline source CSV SHA-256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- Hourly-baseline staged adjusted CSV SHA-256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- EPEX spot parquet SHA-256:
  `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`

Strict evidence for hourly-baseline staging:

- independent comparison:
  `epex_stage_t046_from_hourly_baseline/independent_ab_comparison/ab_comparison_summary.json`
  - `benchmark_policy=independent_no_ompex`
  - `ompex_used_in_model=false`
  - `ompex_used_in_selection=false`
  - `max_abs_monthly_mean_delta_eur_mwh=8.602150532151586e-08`
  - `max_abs_width_delta_eur_mwh=0.0`
  - `weighted_negative_hours_adjusted=0`
- product normalization:
  `epex_stage_t046_from_hourly_baseline/product_normalization_with_policy/summary.json`
  - `all_gates_pass=true`
  - `critical_count=0`
  - `unsupported_count=0`
  - `accepted_quote_conflict_count=6`
  - `blocking_quote_conflict_count=0`
- governance:
  `epex_stage_t046_from_hourly_baseline/governance_audit/epex_shape_lab_governance_audit.json`
  - `status=PASS`
  - `failed_count=0`
- Power BI strict:
  `epex_stage_t046_from_hourly_baseline/powerbi_strict/summary_metrics.csv`
  - `powerbi_quality_gate_status=PASS`
  - `weighted_negative_hours=0`
  - `monthly_path_critical_flags=0`
  - `cross_year_month_shape_critical_flags=0`
- adjusted production contract:
  `epex_stage_t046_from_hourly_baseline/adjusted_production_manifest_no_go.json`
  - `schema_version=epex_lab_adjusted_production_manifest.v1`
  - `contract_pass=true`
  - `production_approved=false`
  - `production_promotion_approved=false`
- readiness:
  `epex_stage_t046_from_hourly_baseline/promotion_readiness/decision_with_contract_no_go.json`
  - `approved=false`
  - `strict_diagnostics_pass=true`
  - `missing_production_evidence=[]`
  - `production_chain_pass=false`

Implementation follow-up:

- Fixed `scripts/stage_epex_lab_adjusted_lt_candidate.py` so the CLI can import
  repo-local `scripts.*` modules when executed as `python scripts/...`.
- Extended `tests/test_stage_epex_lab_adjusted_lt_candidate_script.py` with a
  direct `main()` smoke test.

Validation:

- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_promotion_bundle_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_run_epex_shape_lab_ab_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_audit_ch_product_normalization_script.py tests/test_build_powerbi_exports_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `78 passed, 1 skipped`.

Rejected alternatives:

- Promote the fan-parquet staged result despite product gate failures.
- Treat the NO-GO contract manifest as production approval.
- Reuse the source hierarchy policy when the adjusted CSV hash does not match.

Invariants not to break:

- T046 remains NO-GO production until a real production-approved export,
  selected artifact, and capstone are generated for the adjusted curve.
- The raw fan parquet path needs its own strict product proof before it can be
  used as a promotion-facing source.
- OMPEX remains advisory only.

## D-20260708-22 - Enrich Independent No-OMPEX T046 Shape Diagnostics

Decision: extend the independent EPEX A/B comparison to emit gateable
diagnostics for PEAK/OFFPEAK effects, month-hour deltas, PEAK-OFFPEAK spread
changes, and month-boundary delta jumps before any further production wiring.

Reason: expert read-only audit found T046 promising but not production-ready.
The next safe step is better evidence on the shape deformation itself, not
promotion. The existing no-OMPEX comparison already proves level neutrality and
basic ramp/width checks; it now needs explicit diagnostics for the risk areas
seen in PNG reviews: PEAK/OFFPEAK shaping, localized solar-tail/evening-ramp
movement, and boundary jumps.

Implementation:

- Extended `scripts/compare_epex_shape_lab_ab.py` to write:
  - `load_type_delta_summary.csv`;
  - `month_hour_delta_summary.csv`;
  - `peak_offpeak_monthly_summary.csv`;
  - `boundary_delta_jumps.csv`;
  - `delta_heatmap_month_hour_<year>.png`;
  - `peak_offpeak_spread_delta_by_month.png`;
  - `boundary_delta_jumps.png`.
- Extended `tests/test_compare_epex_shape_lab_ab_script.py` to assert the new
  CSV/PNG evidence exists and preserves `benchmark_policy=independent_no_ompex`.

Real 20260708 T046 evidence:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/t046_enriched_ab_diagnostics/`
- Summary:
  - `benchmark_policy=independent_no_ompex`;
  - `ompex_used_in_model=false`;
  - `ompex_used_in_selection=false`;
  - `n_hours=57025`;
  - `max_abs_delta_eur_mwh=3.0`;
  - `max_abs_monthly_mean_delta_eur_mwh=8.602150532151586e-08`;
  - `max_abs_width_delta_eur_mwh=0.0`;
  - `quantile_order_adjusted_ok=true`;
  - `weighted_negative_hours_adjusted=0`;
  - `ramp_abs_p99_baseline_eur_mwh=24.818882960000007`;
  - `ramp_abs_p99_adjusted_eur_mwh=25.80652717999996`.
- The largest mean month-hour deltas are concentrated in April around
  h13-h14 negative and h19 positive, consistent with the intended solar-tail /
  evening-ramp reshaping.
- The largest month-boundary delta jump observed in the enriched diagnostic is
  about `0.279 EUR/MWh`, so T046's additive delta itself is not introducing a
  large new month-boundary discontinuity.

Validation:

- `python -m pytest tests/test_compare_epex_shape_lab_ab_script.py -q -p no:cacheprovider`
  returned `2 passed`.
- `python -m pytest tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_run_epex_shape_lab_ab_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `34 passed, 1 skipped`.

Rejected alternatives:

- Use OMPEX deltas to decide whether T046 is acceptable.
- Promote T046 because enriched diagnostics look coherent.
- Patch individual months where the delta looks visually strong.

Invariants not to break:

- The enriched diagnostics are independent no-OMPEX evidence only.
- T046 remains NO-GO production until a real adjusted production/export/
  selected/capstone chain exists and passes.
- The fan-parquet to hourly path remains a separate blocker for promotion-facing
  integration.

## D-20260708-23 - Diagnose Fan-to-Hourly Parity Before Promotion Wiring

Decision: add a read-only parity diagnostic for the LT fan parquet to hourly
CSV path and keep the raw fan-derived hourly artifact out of promotion-facing
staging until it is reconciled with the audited export chain.

Reason: read-only expert audit identified the fan-parquet staging path as the
main integration blocker. The staging runner used the lightweight
`to_hourly_csv_frame` conversion directly, while the promotion-ready hourly CSV
is the result of additional calibration, PEAK/OFFPEAK, smoothing, and strict
audit steps. Treating raw fan conversion as equivalent to the audited CSV
created hard product failures.

Implementation:

- Added `scripts/diagnose_fan_to_hourly_parity.py`.
- Added `tests/test_diagnose_fan_to_hourly_parity_script.py`.
- The diagnostic:
  - converts the fan parquet with `to_hourly_csv_frame`;
  - aligns it to a reference hourly CSV;
  - writes column, monthly, load-type, and boundary delta CSVs;
  - optionally runs product-normalization audits on both artifacts;
  - marks itself `promotion_gate=false`.

Real 20260708 evidence:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_to_hourly_parity_diagnostic/`
- Fan source:
  `fan_asof20260707_lshape100_yoy150_amp150_2032.parquet`
- Reference:
  `ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- Diagnostic forwards:
  `epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet`
- Summary:
  - `fan_rows_15min=228192`;
  - `fan_hourly_rows=57025`;
  - `reference_rows=57025`;
  - `aligned_rows=57025`;
  - `missing_price_columns=[]`;
  - `max_abs_weighted_delta_eur_mwh=24.290847000000014`;
  - `mean_abs_weighted_delta_eur_mwh=2.797042955633494`;
  - `max_abs_monthly_weighted_delta_eur_mwh=0.5222635513888889`.
- Load-type deltas show the raw fan-derived artifact is not PEAK/OFFPEAK
  calibrated relative to the audited CSV:
  - `PEAK` mean delta about `+1.2408 EUR/MWh`;
  - `OFFPEAK` mean delta about `-0.6773 EUR/MWh`.
- Product audit on the raw fan-derived hourly CSV:
  - `all_gates_pass=false`;
  - `critical_count=56`;
  - `delivered_curve_drift_count=38`;
  - max supported hard-gate residual about `21.915846 EUR/MWh`.
- Product audit on the reference CSV under the same diagnostic forwards:
  - `critical_count=0`;
  - `delivered_curve_drift_count=0`;
  - source hierarchy remains hash-bound to its exact production forwards, so
    this diagnostic is not a replacement for the baseline promotion capstone.

Validation:

- `python -m pytest tests/test_diagnose_fan_to_hourly_parity_script.py -q -p no:cacheprovider`
  returned `1 passed`.
- `python -m pytest tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `62 passed, 1 skipped`.

Rejected alternatives:

- Promote or continue using fan-parquet staging as promotion-facing evidence
  because row counts and monthly means are close.
- Recalibrate individual failed products after staging.
- Reuse the baseline source hierarchy policy for a fan-derived CSV with a
  different input hash.

Invariants not to break:

- Raw fan-to-hourly conversion is diagnostic/staging only until it passes the
  same product, Power BI, selected-artifact, source-hierarchy, and capstone
  chain as the audited CSV.
- Fixes must enter through the production/export specification, not through
  individual month patches.
- OMPEX remains advisory only and is irrelevant to this parity diagnostic.

## D-20260708-24 - Block Production Contracts From Raw Fan Staging

Decision: prevent the EPEX lab staging runner from building an adjusted
production-contract manifest when the source is a raw fan parquet. Only an
already exported hourly candidate CSV is eligible for adjusted production
contract packaging.

Reason: the fan-to-hourly parity diagnostic proved that raw
`to_hourly_csv_frame(fan)` output has the same hourly row count as the audited
CSV but is not the same artifact and does not pass product gates. Allowing a
fan-sourced staging run to package a production contract, even a NO-GO
contract, makes it too easy to confuse diagnostic staging with the audited
export path.

Implementation:

- Updated `scripts/stage_epex_lab_adjusted_lt_candidate.py`:
  - adds `source_promotion_eligible`;
  - records `production_contract_blockers`;
  - blocks contract building for `source_kind=fan_parquet` with
    `source_kind_fan_parquet_requires_audited_hourly_export`;
  - still allows contract packaging for `source_kind=candidate_csv` when all
    strict evidence inputs are present.
- Updated `tests/test_stage_epex_lab_adjusted_lt_candidate_script.py`:
  - fan-sourced staging remains hash-bound and NO-GO;
  - fan-sourced staging cannot write `adjusted_production_manifest_no_go.json`
    even when strict summaries are supplied;
  - candidate CSV staging can still write the contract NO-GO package.

Validation:

- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py -q -p no:cacheprovider`
  returned `5 passed`.
- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `70 passed, 1 skipped`.

Rejected alternatives:

- Keep allowing raw fan staging to package a NO-GO production contract.
- Make fan staging production-eligible based on row-count or monthly-mean
  parity only.
- Recalibrate individual failed PEAK products inside the staging runner.

Invariants not to break:

- Raw fan parquet staging is reproducibility evidence only.
- Promotion-facing EPEX adjusted work must begin from an audited hourly export
  or a production path that emits and gates the same artifact.
- This guard does not approve T046; T046 remains NO-GO production until the
  adjusted production/export/selected/capstone chain exists and passes.

## D-20260708-25 - Require Source Provenance For Adjusted Production Approval

Decision: require explicit source-provenance evidence before an adjusted EPEX
production manifest can be approved, and make readiness require both
`contract_pass=true` and `source_provenance_pass=true`.

Reason: blocking fan-sourced contract packaging in the staging runner is not
enough if the production-manifest builder can be called directly through its
Python API. Production approval must be impossible without a bound provenance
artifact proving that the adjusted CSV came from an audited hourly candidate
source, not raw fan conversion.

Implementation:

- Updated `scripts/build_epex_lab_adjusted_production_manifest.py`:
  - optional `source_provenance_manifest` input;
  - approval flags now require complete run identity plus a source provenance
    manifest;
  - source provenance checks require:
    - schema `epex_lab_adjusted_lt_candidate_stage.v1`;
    - `source_kind=candidate_csv`;
    - `source_promotion_eligible=true`;
    - empty `production_contract_blockers`;
    - adjusted CSV path or SHA bound to the lab adjusted CSV;
  - the manifest records `source_kind`, `source_promotion_eligible`, and
    `source_provenance_pass`.
- Updated `scripts/check_epex_lab_promotion_readiness.py`:
  - an adjusted production manifest is production-ready only when
    `contract_pass=true` and `source_provenance_pass=true`, in addition to
    approval flags and adjusted CSV binding.
- Updated tests:
  - `tests/test_build_epex_lab_adjusted_production_manifest_script.py`;
  - `tests/test_check_epex_lab_promotion_readiness_script.py`.

Validation:

- `python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider`
  returned `9 passed`.
- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `63 passed, 1 skipped`.

Rejected alternatives:

- Rely only on the staging runner to block fan-sourced contracts.
- Accept approval flags without a source provenance manifest.
- Treat any source provenance artifact as valid without checking source kind,
  blockers, and adjusted CSV binding.

Invariants not to break:

- CLI-built adjusted production manifests remain NO-GO by default.
- A future production approval must carry identity, strict diagnostics, source
  provenance, export manifest, selected artifact, and capstone approval.
- Fan-derived raw hourly artifacts remain ineligible for promotion-facing
  contract approval.

## D-20260708-26 - Emit Source Provenance From Candidate-CSV Staging

Decision: make the staging runner emit a dedicated source provenance manifest
and pass it into the adjusted production contract builder whenever the source
is an eligible hourly candidate CSV.

Reason: after adding the source-provenance requirement to the adjusted
production manifest, the candidate-CSV staging path still needed to produce a
real provenance artifact. The provenance must be separate from the staging
manifest because the staging manifest is later augmented with contract paths and
hashes; the contract should bind to a stable source-provenance artifact.

Implementation:

- Updated `scripts/stage_epex_lab_adjusted_lt_candidate.py`:
  - writes `source_provenance_manifest.json`;
  - includes source kind, source eligibility, source/staged/adjusted CSV
    hashes, lab manifest hash, and contract blockers;
  - passes `source_provenance_manifest` to
    `build_epex_lab_adjusted_production_manifest.py` for eligible
    `candidate_csv` staging.
- Updated `tests/test_stage_epex_lab_adjusted_lt_candidate_script.py`:
  - fan staging writes an ineligible provenance artifact but still cannot
    package a production contract;
  - candidate CSV staging writes a NO-GO contract with
    `source_provenance_pass=true`.

Real 20260708 evidence:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_with_provenance/`
- Source:
  `ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- Staged adjusted CSV SHA-256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- Source provenance SHA-256:
  `8d3cacb36637ea6e57446d840458d85d6219da72a94c200f1ac8c559a3d2a6b9`
- Adjusted production contract NO-GO SHA-256:
  `5600b737482e0db537059f36fe997f3bbe9e15c8435874c98b1ca8b59e0e2f09`
- Contract fields:
  - `contract_pass=true`;
  - `source_kind=candidate_csv`;
  - `source_promotion_eligible=true`;
  - `source_provenance_pass=true`;
  - `production_approved=false`;
  - `production_promotion_approved=false`.

Readiness with the provenance-aware contract:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_with_provenance/readiness_no_go.json`
- Result:
  - `approved=false`;
  - `strict_diagnostics_pass=true`;
  - `production_chain_pass=false`;
  - `missing_production_evidence=[]`;
  - provenance checks pass;
  - failures remain expected non-production approvals:
    adjusted capstone, adjusted production manifest approval, adjusted export
    manifest production readiness, and adjusted selected artifact production
    readiness.

Validation:

- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py -q -p no:cacheprovider`
  returned `5 passed`.
- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `72 passed, 1 skipped`.

Rejected alternatives:

- Use the mutable staged manifest itself as source provenance after adding
  contract fields.
- Leave candidate-CSV staging with `source_provenance_pass=false` even though
  its source is promotion-eligible.
- Promote the provenance-aware contract despite local NO-GO approval flags.

Invariants not to break:

- Source provenance can make a contract complete, but not approved.
- T046 remains NO-GO production until export, selected artifact, and capstone
  are real production-approved artifacts.
- Fan-sourced staging remains ineligible even though it also writes provenance.

## D-20260708-27 - Add No-OMPEX Multi-Date Stability Summary For T046

Decision: add a read-only multi-date stability summarizer for EPEX shape-lab
A/B cases and run it on the two available current-data baselines:
`asof20260706` and `asof20260707`.

Reason: T046 has promising single-date diagnostics, but expert audit called out
stability as a prerequisite before production wiring. The next evidence should
compare frozen no-OMPEX A/B outputs across dates, not use OMPEX or promote the
candidate.

Implementation:

- Added `scripts/summarize_epex_shape_lab_stability.py`.
- Added `tests/test_summarize_epex_shape_lab_stability_script.py`.
- The summarizer reads independent A/B summaries plus governance audits and
  gates each case on:
  - `benchmark_policy=independent_no_ompex`;
  - no OMPEX in model or selection;
  - governance `PASS`;
  - finite adjusted prices;
  - quantile order;
  - zero weighted negative hours;
  - adjusted min price above floor;
  - monthly drift and width drift thresholds;
  - ramp p99 increase threshold.

Real 20260707 stability probe:

- Source baseline:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260706_lshape100_yoy150_amp150_2032.csv`
- Output:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/epex_stage_t046_stability_probe/`
- Result:
  - adjusted CSV SHA-256:
    `9df080257f4a9df24314180be344a4c3618b9b21e23240d497403016e1d28e7e`;
  - `benchmark_policy=independent_no_ompex`;
  - `max_abs_delta_eur_mwh=3.0`;
  - `max_abs_monthly_mean_delta_eur_mwh=8.602150568442747e-08`;
  - `max_abs_width_delta_eur_mwh=0.0`;
  - `weighted_negative_hours_adjusted=0`;
  - `min_adjusted_price_eur_mwh=-3.887768`;
  - ramp p99 increase about `0.937862 EUR/MWh`;
  - governance `PASS`.

Multi-date summary:

- Output:
  `output/phase14/t046_stability_summary_v1/`
- Cases:
  - `asof20260706`;
  - `asof20260707`.
- Status:
  - `PASS`;
  - `case_count=2`;
  - `passed_case_count=2`;
  - `promotion_gate=false`;
  - `benchmark_policy=multi_date_independent_no_ompex`.
- Case values:
  - `asof20260706`: ramp p99 increase `0.9378621600000336`,
    min adjusted price `-3.887768`, monthly drift `8.602150568442747e-08`;
  - `asof20260707`: ramp p99 increase `0.9876442199999538`,
    min adjusted price `-3.825623`, monthly drift
    `8.602150532151586e-08`.

Validation:

- `python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py -q -p no:cacheprovider`
  returned `2 passed`.
- `python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `38 passed, 1 skipped`.

Rejected alternatives:

- Use OMPEX to judge cross-date stability.
- Treat two-date stability PASS as production approval.
- Promote T046 before production/export/selected/capstone artifacts exist.

Invariants not to break:

- Stability summary is lab evidence only and `promotion_gate=false`.
- T046 remains NO-GO production.
- More dates should be added before production promotion if historical
  comparable current-data baselines are available.

## D-20260708-28 - Require Reloaded Source Provenance For Adjusted Production Readiness

Decision: harden adjusted EPEX lab production readiness so that source
provenance and promotion manifests are validated from real files instead of
trusted as self-attested fields.

Reason: a read-only expert audit found a future-path P0. Existing T046 artifacts
were still NO-GO, but `scripts/check_epex_lab_promotion_readiness.py` could
approve a fabricated future production chain if a handwritten adjusted
production manifest claimed `contract_pass=true` and
`source_provenance_pass=true`. The checker must reload the referenced
provenance artifact, verify its hash, and validate the source/export chain.

Implementation:

- `scripts/check_epex_lab_promotion_readiness.py` now requires the adjusted
  production manifest to reference `source_provenance_manifest` and
  `source_provenance_manifest_sha256`.
- Readiness reloads that provenance file and validates:
  - schema and `schema_role=source_provenance`;
  - lab-only/non-production flags;
  - `source_kind=candidate_csv`;
  - `source_promotion_eligible=true`;
  - empty production contract blockers;
  - adjusted CSV path/SHA binding;
  - source CSV SHA binding;
  - staged candidate CSV SHA binding;
  - lab manifest SHA binding;
  - source export manifest SHA binding;
  - source export manifest binding back to the source CSV;
  - no OMPEX model/selection usage.
- Readiness now also rejects minimal handwritten export, selected, and capstone
  artifacts by requiring known schemas and production-chain fields.
- `scripts/build_epex_lab_adjusted_production_manifest.py` now includes
  source provenance presence in `contract_pass`; `contract_pass=true` can no
  longer mean diagnostics passed while provenance is absent.
- `scripts/stage_epex_lab_adjusted_lt_candidate.py` now requires a source
  export manifest bound to the candidate CSV before setting
  `source_promotion_eligible=true` or writing an adjusted production-contract
  package. A candidate CSV without this manifest remains stageable but receives
  blocker `candidate_csv_requires_source_export_manifest`.
- Tests now cover self-attested provenance rejection, fan-source provenance
  rejection, source-export-manifest requirement, and the positive path with
  full bound provenance.

Validation:

- `python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py -q -p no:cacheprovider`
  returned `17 passed`.
- `python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `41 passed, 1 skipped`.

Rejected alternatives:

- Trust `source_provenance_pass=true` copied into the adjusted production
  manifest.
- Allow `contract_pass=true` without source provenance, relying on readiness to
  check a second flag later.
- Treat any `--candidate-csv` as promotion-eligible without a source export
  manifest.
- Accept minimal handwritten export, selected, or capstone JSON as production
  evidence.

Invariants not to break:

- T046 remains NO-GO production until a real production-approved
  production/export/selected/capstone chain exists.
- Candidate CSV staging without a source export manifest is allowed for lab
  work but must not package an adjusted production contract.
- Fan-parquet staging remains ineligible for production contracts.
- OMPEX remains advisory evidence only and must not enter model, selection, or
  readiness gates.

## D-20260708-29 - Add Implied-Width And Fan-Coverage Diagnostics

Decision: enrich read-only EPEX lab diagnostics with two expert-audit signals:
implied structural width and fan-to-hourly timestamp coverage.

Reason: expert audit found that width conservation could be falsely reassuring
if only the reported `structural_width_eur_mwh` column was compared, and that
fan-to-hourly parity could hide partial timestamp overlap because it used an
inner join. These diagnostics should make those risks visible without changing
selection or promotion gates yet.

Implementation:

- `scripts/compare_epex_shape_lab_ab.py` now computes:
  - baseline and adjusted implied width as `structural_p90 - structural_p10`;
  - implied width delta;
  - reported-minus-implied width for baseline and adjusted;
  - summary fields for max absolute implied-width drift and stale reported
    width detection;
  - monthly implied-width drift diagnostics.
- `scripts/diagnose_fan_to_hourly_parity.py` now reports:
  - missing timestamps in the fan-derived hourly series;
  - missing timestamps in the reference audited CSV;
  - fan and reference coverage ratios;
  - `coverage_status=PASS` or `PARTIAL_OVERLAP`.
- Tests now cover a stale reported-width column and partial fan/reference
  timestamp overlap.

Validation:

- `python -m pytest tests/test_compare_epex_shape_lab_ab_script.py tests/test_diagnose_fan_to_hourly_parity_script.py -q -p no:cacheprovider`
  returned `5 passed`.
- `python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `43 passed, 1 skipped`.

Rejected alternatives:

- Trust the reported structural width column alone.
- Convert coverage mismatches into hard failures immediately.
- Use OMPEX to validate width or fan parity.

Invariants not to break:

- These diagnostics are read-only lab evidence and do not promote T046.
- Width governance should consider implied `p90-p10`, not only a stored width
  column.
- Fan-to-hourly inner joins must disclose coverage loss explicitly.

## D-20260708-30 - Gate T046 Stability On Local Shape And Negative-Risk Metrics

Decision: extend the no-OMPEX T046 stability summary from broad conservation
checks to local shape-risk gates: PEAK/OFFPEAK spread, month-hour extrema,
month-boundary delta jumps, implied width, and p10 negative clusters.

Reason: expert audit found that T046 could pass monthly mean, width, ramp p99,
and weighted-negative checks while still introducing a local deformation.
Stability evidence should prove that the frozen adjustment is not hiding a
large PEAK/OFFPEAK distortion, stale width column, month-boundary discontinuity,
or unacceptable p10 negative cluster.

Implementation:

- `scripts/compare_epex_shape_lab_ab.py` now writes the following fields to
  `ab_comparison_summary.json`:
  - `max_abs_month_hour_mean_delta_eur_mwh`;
  - `max_abs_peak_offpeak_spread_delta_eur_mwh`;
  - `max_abs_boundary_delta_jump_eur_mwh`;
  - implied width drift and reported-minus-implied width drift;
  - negative-hour counts for weighted, p10, p50, p90, slow, central, and fast;
  - max negative cluster length for weighted and p10.
- `scripts/summarize_epex_shape_lab_stability.py` now requires those fields
  for PASS and gates them with configurable thresholds:
  - `max_peak_offpeak_spread_delta=5.0`;
  - `max_month_hour_mean_delta=3.5`;
  - `max_boundary_delta_jump=1.0`;
  - `max_implied_width_delta=1e-9`;
  - `max_reported_minus_implied_width=1e-9`;
  - `max_p10_negative_hours=240`;
  - `max_p10_negative_cluster_hours=48`.

Real T046 stability v2:

- A/B v2 output for `asof20260706`:
  `output/phase14/20260707_asof20260706_lshape100_yoy150_amp150_2032/epex_stage_t046_stability_probe/independent_ab_comparison_v2/`
- A/B v2 output for `asof20260707`:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_with_provenance/independent_ab_comparison_v2/`
- Stability summary:
  `output/phase14/t046_stability_summary_v2/`
- Result:
  - `status=PASS`;
  - `case_count=2`;
  - `passed_case_count=2`;
  - `promotion_gate=false`;
  - `benchmark_policy=multi_date_independent_no_ompex`;
  - max month-hour mean delta about `2.2225` EUR/MWh on both dates;
  - max boundary delta jump about `0.2786` EUR/MWh on both dates;
  - max PEAK/OFFPEAK spread delta about `2.5e-07` EUR/MWh;
  - implied width drift about `2.84e-14`;
  - p10 negative hours `125` and `118`;
  - p10 negative cluster max `6` hours;
  - weighted negative hours `0`.

Validation:

- `python -m pytest tests/test_compare_epex_shape_lab_ab_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_summarize_epex_shape_lab_stability_script.py -q -p no:cacheprovider`
  returned `9 passed`.
- `python -m pytest tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `45 passed, 1 skipped`.

Rejected alternatives:

- Keep PEAK/OFFPEAK and boundary diagnostics only as CSV side outputs.
- Let old A/B summaries without local-shape fields pass stability.
- Use OMPEX as the arbiter for local-shape stability.

Invariants not to break:

- Stability v2 remains lab evidence only and `promotion_gate=false`.
- T046 remains NO-GO production.
- OMPEX remains advisory-only and cannot enter model or selection.
- Monthly solver level authority remains unchanged; these checks assess hourly
  shape only.

## D-20260708-31 - Add Cross-Date Delta-Field Stability Diagnostic For T046

Decision: add a read-only no-OMPEX diagnostic that compares the frozen EPEX
shape-lab delta field itself across dates.

Reason: stability v2 proves local-shape metrics are inside thresholds, but it
does not directly prove that the actual hourly adjustment field is stable
across current-data runs. Expert audit requested cross-date delta correlation,
L1/L2/L-infinity style differences, top month-hour drift, boundary drift, and
parameter consistency.

Implementation:

- Added `scripts/summarize_epex_shape_lab_delta_stability.py`.
- Added `tests/test_summarize_epex_shape_lab_delta_stability_script.py`.
- The diagnostic reads repeated cases as
  `LABEL|ALIGNED_AB_CSV|AB_SUMMARY_JSON|LAB_MANIFEST_JSON`.
- It validates no-OMPEX independent A/B summaries, lab-only manifests, no
  production approval, timestamp coverage, and a stable T046 config hash
  excluding only valuation timestamp.
- It compares every non-reference case to the first case using:
  - timestamp-level delta correlation;
  - timestamp-level mean, MAE, RMSE, and max absolute delta-field difference;
  - month-hour profile correlation, MAE, RMSE, and max absolute difference;
  - month-boundary jump MAE, RMSE, and max absolute difference.

Real T046 delta stability v1:

- Output:
  `output/phase14/t046_delta_stability_summary_v1/`
- Cases:
  - reference `asof20260706`;
  - comparison `asof20260707`.
- Result:
  - `status=PASS`;
  - `comparison_count=1`;
  - `passed_comparison_count=1`;
  - `config_consistent=true`;
  - config hash:
    `e9c1f0831cb896f03987eeefcbb92dfbf900a53eaad4c4d45ab58e761e163b51`;
  - timestamp delta correlation `0.9999797676440942`;
  - timestamp delta MAE `0.0011066597457297395` EUR/MWh;
  - timestamp delta RMSE `0.005290483624245774` EUR/MWh;
  - timestamp delta max abs `0.05641400000001795` EUR/MWh;
  - month-hour delta correlation `0.9999818326388252`;
  - month-hour delta MAE `0.0009053573348698669` EUR/MWh;
  - month-hour delta max abs `0.04788046543778779` EUR/MWh;
  - boundary jump diff MAE `0.0013372207792282818` EUR/MWh;
  - boundary jump diff max abs `0.015835000000009813` EUR/MWh;
  - missing timestamps `0`.

Validation:

- `python -m pytest tests/test_summarize_epex_shape_lab_delta_stability_script.py -q -p no:cacheprovider`
  returned `3 passed`.
- `python -m pytest tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `48 passed, 1 skipped`.

Rejected alternatives:

- Rely only on aggregated stability v2 metrics.
- Compare deltas against OMPEX.
- Ignore T046 parameter consistency across dates.

Invariants not to break:

- Delta stability is lab evidence only and `promotion_gate=false`.
- T046 remains NO-GO production.
- OMPEX remains advisory-only and cannot enter model or selection.
- Cross-date comparisons must disclose missing timestamps instead of silently
  using partial overlap.

## D-20260708-32 - Add Source-Export Manifest For T046 Candidate-CSV Provenance

Decision: add a source-export manifest builder and rerun T046 candidate-CSV
staging with a hash-bound source manifest, so D28 provenance checks are
complete without promoting T046.

Reason: D28 correctly blocks candidate-CSV staging from packaging a production
contract unless the source CSV is bound to a source-export manifest. The
previous T046 source-provenance staging proved the adjusted CSV but did not
carry that source-export manifest. The next evidence should close this
provenance gap while keeping all adjusted production approval flags false.

Implementation:

- Added `scripts/build_epex_lab_source_export_manifest.py`.
- Added `tests/test_build_epex_lab_source_export_manifest_script.py`.
- The manifest binds:
  - candidate/source hourly CSV path and SHA;
  - baseline monthly manifest and SHA;
  - selected config artifact and SHA;
  - source hierarchy policy and SHA;
  - baseline promotion capstone and SHA;
  - no OMPEX model/selection flags.
- The manifest is explicitly source provenance only:
  - `schema_version=epex_lab_source_export_manifest.v1`;
  - `production_approved=false`;
  - `production_promotion_approved=false`;
  - `promotion_scope=SOURCE_CSV_PROVENANCE_ONLY`.

Real source-export manifest:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_source_export_manifest/source_export_manifest.json`
- Source CSV:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- Source CSV SHA-256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- Manifest SHA-256:
  `d662548e2e7605ba2b59e024afd3040f2724fe84c5f3c7d3491fbaa0e1909f1d`
- Baseline monthly manifest SHA-256:
  `cb52a502e8e95af2e5f3fabc3b2b34ca8f365999214cfd7c53718ed7f5ef456a`
- Selected config SHA-256:
  `5ca8b3dc3c1dfadf6b2153bc22f6d69cfb2ad767ae8dbada9615f753760e1f34`
- Baseline capstone SHA-256:
  `091105ba9bc313b36364a75a9dd88ab9e3eaa9e740151c44c8a40c42cce1048c`

Real source-provenance staging:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/`
- Adjusted CSV SHA-256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- Source provenance SHA-256:
  `eefe822b24a876a176b78afd9ccc21552d4c5248d8833a7c8ee1bbd368789d1f`
- Adjusted production manifest NO-GO SHA-256:
  `7824522ca68f64da20bd7871cba0beed246f27bfb888beef2d1cc65ffdbd17a9`
- Staging result:
  - `source_kind=candidate_csv`;
  - `source_promotion_eligible=true`;
  - `source_export_manifest_bound=true`;
  - `production_contract_blockers=[]`;
  - `adjusted_production_contract_pass=true`;
  - `production_approved=false`;
  - `production_promotion_approved=false`.

Readiness:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/readiness_no_go.json`
- Result:
  - `approved=false`;
  - `strict_diagnostics_pass=true`;
  - `production_chain_pass=false`;
  - `missing_production_evidence=[]`;
  - `adjusted_production_manifest_contract_pass=PASS`;
  - `adjusted_production_manifest_source_provenance_pass=PASS`;
  - all source-provenance file/SHA/binding checks PASS;
  - remaining FAIL checks are the expected non-production approvals:
    adjusted capstone, adjusted production manifest approval, adjusted export
    manifest production-ready, and adjusted selected artifact production-ready.

Operational note:

- A first A/B comparison rerun under the very long staging output directory
  failed during PNG writing because of Windows path length. The comparison was
  rerun successfully under the shorter folder:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/t046_srcprov_ab_v2/`.

Validation:

- `python -m pytest tests/test_build_epex_lab_source_export_manifest_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider`
  returned `19 passed`.
- `python -m pytest tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `50 passed, 1 skipped`.

Rejected alternatives:

- Hand-write a one-off source-export manifest.
- Treat the previous source-provenance staging as complete under D28.
- Set adjusted production/export/selected/capstone approval flags true in a
  local staging path.

Invariants not to break:

- Source-export provenance can make T046 packaging complete, but not approved.
- T046 remains NO-GO production until a real production-approved adjusted
  production/export/selected/capstone chain exists.
- OMPEX remains advisory-only and cannot enter model, selection, provenance, or
  readiness gates.

## D-20260708-33 - Add No-OMPEX Spot Backtest For T046 Lab Shape

Decision: add a read-only, lab-only EPEX spot backtest for the T046 adjusted
candidate, with explicit anti-leakage metadata and no production promotion
authority.

Reason: expert review agreed that OMPEX must remain advisory-only and that the
next scientific check should compare the baseline and adjusted HPFC shapes
against realized EPEX spot profiles. The check must not be treated as
independent production evidence because the current T046 candidate was selected
after historical spot rows were known; it is useful as an anti-overfit and shape
reasonableness diagnostic only.

Implementation:

- Added `scripts/backtest_epex_shape_lab_against_spot.py`.
- Added `tests/test_backtest_epex_shape_lab_against_spot_script.py`.
- The diagnostic writes:
  - `spot_backtest_summary.json`;
  - `rolling_spot_profile_folds.csv`;
  - `candidate_month_hour_profiles.csv`;
  - `post_valuation_timestamp_residuals.csv`.
- It records `promotion_gate=false`, `production_approved=false`,
  `independent_production_evidence=false`,
  `benchmark_policy=rolling_origin_epex_spot_no_ompex_lab_only`, and all OMPEX
  usage flags false.
- It enforces lab guard checks for finite adjusted values, quantile order,
  monthly weighted mean drift, implied width drift, weighted negative hours,
  rolling folds, and no temporal overlap between each fold's train and
  evaluation spot windows.

Real T046 spot backtest v1:

- Output:
  `output/phase14/t046_spot_backtest_v1/`
- Baseline CSV SHA-256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- Adjusted CSV SHA-256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- Spot parquet SHA-256:
  `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`
- Valuation timestamp:
  `2026-07-07 00:00:00+00:00`
- Result:
  - `status=DIAGNOSTIC_PASS`;
  - `strict_lab_gate_pass=true`;
  - rolling folds `12/12` eligible;
  - all rolling folds pass no-temporal-leak checks;
  - historical folds not independent of current candidate fit: `12`;
  - post-valuation overlap: `24` hours.
- Rolling profile metrics:
  - mean baseline MAE `14.153227063777985` EUR/MWh;
  - mean adjusted MAE `13.747743522746092` EUR/MWh;
  - mean MAE improvement `0.40548354103189205` EUR/MWh;
  - median MAE improvement `0.39547100649685163` EUR/MWh;
  - positive improvement folds `12/12`;
  - mean baseline correlation `0.8771034667706381`;
  - mean adjusted correlation `0.881938557794624`.
- Post-valuation residual metrics:
  - baseline MAE `11.660298070900533` EUR/MWh;
  - adjusted MAE `11.355494229166665` EUR/MWh;
  - improvement `0.3048038417338681` EUR/MWh.

Validation:

- `python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py -q -p no:cacheprovider`
  returned `2 passed`.
- `python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `52 passed, 1 skipped`.

Rejected alternatives:

- Use OMPEX as target, loss, or promotion gate.
- Treat historical rolling folds as independent production evidence.
- Relax monthly-level or fan-width guard checks to make a shape diagnostic pass.

Invariants not to break:

- This backtest is lab evidence only and does not promote T046.
- OMPEX remains advisory-only and cannot enter model, selection, provenance, or
  readiness gates.
- T046 remains NO-GO production until real adjusted production/export/selected
  artifacts and capstone are approved.

## D-20260708-34 - Add Economic Bucket And Ramp Metrics To Spot Backtest

Decision: extend the no-OMPEX EPEX spot backtest with diagnostic economic
buckets and hourly ramp metrics.

Reason: the D33 global/profile MAE proved a small average shape improvement,
but expert review requested more interpretable evidence: weekend/weekday,
PEAK-like/OFFPEAK-like, solar tail, midday, evening recovery, night, and hourly
ramp behavior. These diagnostics identify where T046 helps or is weak without
using OMPEX and without changing production status.

Implementation:

- Updated `scripts/backtest_epex_shape_lab_against_spot.py`.
- Updated `tests/test_backtest_epex_shape_lab_against_spot_script.py`.
- Added output:
  `rolling_spot_bucket_metrics.csv`.
- Added JSON summary field:
  `rolling_bucket_metrics`.
- Buckets:
  - `all`;
  - `weekday`;
  - `weekend`;
  - `peak_like_weekday_08_19`;
  - `offpeak_like`;
  - `solar_tail_mar_oct_10_16`;
  - `midday_11_15`;
  - `evening_ramp_17_21`;
  - `night_00_05`;
  - `hourly_ramp:all`.

Real T046 spot backtest v2:

- Output:
  `output/phase14/t046_spot_backtest_v2_buckets/`
- Result:
  - `status=DIAGNOSTIC_PASS`;
  - `strict_lab_gate_pass=true`;
  - `promotion_gate=false`;
  - `production_approved=false`;
  - `independent_production_evidence=false`;
  - OMPEX flags all false.
- Source hashes match D33:
  - baseline CSV
    `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`;
  - adjusted CSV
    `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`;
  - spot parquet
    `008f552e0cd684d42dcb95f87a2681054b1af338c6511ae77c1ffa81b421e32f`.

Selected bucket diagnostics, mean MAE improvement in EUR/MWh:

- all residual level: `0.24513954474101998`, positive folds `11/12`;
- weekend: `0.2889611347370835`, positive folds `12/12`;
- weekday: `0.22708125671275944`, positive folds `10/12`;
- PEAK-like weekday 08-19: `0.32096908439747596`, positive folds `10/12`;
- OFFPEAK-like: `0.20198153831529964`, positive folds `12/12`;
- solar tail Mar-Oct 10-16: `0.4372953091304925`, positive folds `8/12`;
- midday 11-15: `0.35776460522648684`, positive folds `9/12`;
- evening ramp 17-21: `0.45338812791781463`, positive folds `12/12`;
- night 00-05: `0.03190894115068499`, positive folds `5/12`;
- hourly ramp all: `0.035478178105887714`, positive folds `8/12`.

Interpretation:

- T046's strongest no-OMPEX shape evidence is evening recovery, solar/midday,
  PEAK-like hours, and weekend buckets.
- Night and hourly ramp gains are weak; future model work should not overclaim
  ramp improvement from T046 v2.
- The diagnostic remains lab-only because the current candidate selection used
  historical spot information.

Validation:

- `python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py -q -p no:cacheprovider`
  returned `2 passed`.
- `python -m pytest tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `52 passed, 1 skipped`.

Rejected alternatives:

- Use only global MAE/correlation from D33.
- Tune T046 directly to OMPEX bucket differences.
- Treat weak ramp improvement as a promotion blocker; it is diagnostic evidence
  for next research work, not a production gate.

Invariants not to break:

- Bucket/ramp diagnostics are no-OMPEX and lab-only.
- T046 remains NO-GO production until the adjusted production/export/selected
  chain and capstone are real and approved.
- Do not use OMPEX to tune or select future T046-like parameters.

## D-20260708-35 - Add Future Approval Path Audit For T046

Decision: add a read-only future-approval audit that summarizes the exact T046
production blockers from readiness evidence and optional spot-backtest policy
evidence.

Reason: after D32-D34, T046 has strong lab diagnostics but still must not be
promoted. Reviewers need a compact artifact that says which evidence is already
strict-diagnostic PASS and which real production approvals are still missing or
false, without treating local bundles, spot diagnostics, or source provenance
as production approval.

Implementation:

- Added `scripts/audit_epex_lab_future_approval_path.py`.
- Added `tests/test_audit_epex_lab_future_approval_path_script.py`.
- The audit reads `check_epex_lab_promotion_readiness.py` output and optional
  no-OMPEX spot backtest output.
- It writes `future_approval_path_audit.json` with:
  - readiness status;
  - strict diagnostic and production-chain booleans;
  - failed production checks;
  - remaining blockers;
  - required production evidence;
  - spot-backtest policy checks;
  - next actions.

Real T046 future approval path audit v1:

- Output:
  `output/phase14/t046_future_approval_path_audit_v1/future_approval_path_audit.json`
- Inputs:
  - readiness:
    `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/readiness_no_go.json`;
  - spot backtest:
    `output/phase14/t046_spot_backtest_v2_buckets/spot_backtest_summary.json`.
- Result:
  - `status=NO_GO_PRODUCTION_CHAIN_INCOMPLETE`;
  - `approved=false`;
  - `strict_diagnostics_pass=true`;
  - `production_chain_pass=false`;
  - `spot_backtest_policy.pass=true`;
  - `missing_production_evidence=[]`.
- Remaining blockers:
  - `adjusted_capstone_approved`;
  - `adjusted_export_manifest_production_ready`;
  - `adjusted_production_manifest_approved`;
  - `adjusted_selected_artifact_production_ready`.
- Next action:
  replace local diagnostic approval flags with real production-approved adjusted
  artifacts.

Validation:

- `python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py -q -p no:cacheprovider`
  returned `3 passed`.
- `python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `55 passed, 1 skipped`.

Rejected alternatives:

- Rely on the long readiness checks list only.
- Treat missing evidence only as files absent from disk; D35 also flags present
  local artifacts whose production approval booleans are false.
- Include OMPEX benchmark deltas as promotion evidence.

Invariants not to break:

- Future-approval audit is read-only and `promotion_gate=false`.
- Local diagnostic bundles, spot diagnostics, and source provenance are not
  production approval.
- T046 remains NO-GO production until all four adjusted approval blockers are
  resolved by real production-approved artifacts.

## D-20260708-36 - Harden Adjusted Production Approval Run Identity

Decision: require a valid production run identity before the adjusted EPEX lab
production-manifest API can emit production approval flags.

Reason: D35 clarified that the remaining blockers are present-but-unapproved
adjusted production artifacts. The CLI is already NO-GO by default, but the
Python API path that can set `production_approved=True` should reject malformed
run identity up front, rather than accepting any non-empty `git_commit` string.

Implementation:

- Updated `scripts/build_epex_lab_adjusted_production_manifest.py`.
- Added `_production_identity_errors(...)`.
- When either approval flag is requested, the API now requires:
  - non-empty `production_run_id`;
  - non-empty `production_entrypoint`;
  - `git_commit` matching `[0-9a-f]{40}`;
  - existing `source_provenance_manifest`.
- Invalid identity raises `ValueError` before writing an approved manifest.
- Updated `tests/test_build_epex_lab_adjusted_production_manifest_script.py`
  with an invalid-git-commit rejection test.

Validation:

- `python -m pytest tests/test_build_epex_lab_adjusted_production_manifest_script.py -q -p no:cacheprovider`
  returned `6 passed`.
- `python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `56 passed, 1 skipped`.

Rejected alternatives:

- Accept any non-empty git commit string.
- Add CLI flags that can set production approval true from local staging.
- Treat source provenance alone as sufficient production run identity.

Invariants not to break:

- CLI-built adjusted production manifests remain NO-GO by default.
- Production approval remains API-only and requires manifest-backed strict
  diagnostics plus valid run identity.
- T046 remains NO-GO production until real adjusted production/export/selected
  artifacts and capstone are approved.

## D-20260708-37 - Require Adjusted Production Chain Binding In Readiness

Decision: require export, selected, and capstone artifacts to bind back to the
same adjusted production manifest and production run identity before T046 can
be `PROMOTION_READY`.

Reason: D35 identified that T046's remaining blockers are production approvals.
D36 hardened the adjusted production-manifest API identity, but readiness still
needed to prove that the export manifest, selected artifact, and capstone all
belong to that same approved production chain, not just to the same adjusted
CSV. This prevents mixing local or stale artifacts into an apparently complete
production bundle.

Implementation:

- Updated `scripts/check_epex_lab_promotion_readiness.py`.
- Updated `scripts/audit_epex_lab_future_approval_path.py`.
- Updated `tests/test_check_epex_lab_promotion_readiness_script.py`.
- New readiness checks:
  - `adjusted_production_manifest_run_identity_valid`;
  - `adjusted_export_manifest_production_chain_bound`;
  - `adjusted_selected_artifact_production_chain_bound`;
  - `adjusted_capstone_production_chain_bound`.
- A production-ready adjusted export/selected artifact must bind to the
  adjusted production manifest path or SHA and match its run identity.
- A production-ready capstone must bind to the adjusted production manifest,
  adjusted export manifest, adjusted selected artifact, and production run
  identity.

Real T046 readiness v2:

- Output:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_stage_t046_from_hourly_baseline_source_export_provenance/readiness_no_go_v2_chain_bound.json`
- Result:
  - `approved=false`;
  - `strict_diagnostics_pass=true`;
  - `production_chain_pass=false`;
  - `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`.
- New explicit blockers:
  - `adjusted_production_manifest_run_identity_valid=FAIL` because the local
    staging manifest has no real git commit;
  - `adjusted_export_manifest_production_chain_bound=FAIL`;
  - `adjusted_selected_artifact_production_chain_bound=FAIL`;
  - `adjusted_capstone_production_chain_bound=FAIL`.

Real future approval path audit v2:

- Output:
  `output/phase14/t046_future_approval_path_audit_v2_chain_bound/future_approval_path_audit.json`
- Result:
  - `status=NO_GO_PRODUCTION_CHAIN_INCOMPLETE`;
  - `strict_diagnostics_pass=true`;
  - `production_chain_pass=false`;
  - `spot_backtest_policy.pass=true`;
  - failed production checks `8`.
- Remaining blockers:
  - `adjusted_capstone_approved`;
  - `adjusted_capstone_production_chain_bound`;
  - `adjusted_export_manifest_production_chain_bound`;
  - `adjusted_export_manifest_production_ready`;
  - `adjusted_production_manifest_approved`;
  - `adjusted_production_manifest_run_identity_valid`;
  - `adjusted_selected_artifact_production_chain_bound`;
  - `adjusted_selected_artifact_production_ready`.

Validation:

- `python -m pytest tests/test_check_epex_lab_promotion_readiness_script.py -q -p no:cacheprovider`
  returned `7 passed`.
- `python -m pytest tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `57 passed, 1 skipped`.

Rejected alternatives:

- Keep readiness bound only to the adjusted CSV SHA.
- Allow a production capstone to approve without explicit export/selected
  artifact bindings.
- Treat local bundle links as enough for production-chain binding.

Invariants not to break:

- T046 remains NO-GO production until all approved adjusted artifacts are bound
  to the same production manifest/run identity.
- Local diagnostic bundles remain non-production even when all files exist.
- OMPEX remains advisory-only and cannot enter model, selection, readiness, or
  production-chain gates.

## D-20260708-38 - Add Strict Adjusted Production Chain Builder

Decision: add a strict builder for the remaining adjusted production-chain
artifacts: export manifest, selected artifact, and production capstone.

Reason: D37 made readiness require export/selected/capstone artifacts to bind
to the same approved adjusted production manifest and run identity. The repo
needed a safe way to produce those bound artifacts once, and only once, an
adjusted production manifest is already approved, contract-pass,
source-provenance-pass, and run-identity-valid. Local diagnostic bundles remain
non-production and are not a substitute.

Implementation:

- Added `scripts/build_epex_lab_adjusted_production_chain.py`.
- Added `tests/test_build_epex_lab_adjusted_production_chain_script.py`.
- The builder refuses to run unless the input adjusted production manifest has:
  - `schema_version=epex_lab_adjusted_production_manifest.v1`;
  - `production_approved=true`;
  - `production_promotion_approved=true`;
  - `contract_pass=true`;
  - `source_provenance_pass=true`;
  - valid `production_run_id`;
  - valid `production_entrypoint`;
  - `git_commit` matching `[0-9a-f]{40}`;
  - `ompex_used_in_model=false`;
  - `ompex_used_in_selection=false`;
  - adjusted CSV path/SHA binding.
- It writes:
  - `adjusted_export_manifest.json`;
  - `adjusted_selected_artifact.json`;
  - `adjusted_production_capstone.json`.
- All outputs bind back to the adjusted production manifest path/SHA and run
  identity. The capstone also binds to the generated export and selected
  artifacts.

Validation:

- `python -m pytest tests/test_build_epex_lab_adjusted_production_chain_script.py -q -p no:cacheprovider`
  returned `2 passed`.
- `python -m pytest tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_audit_epex_lab_future_approval_path_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_build_epex_lab_source_export_manifest_script.py tests/test_summarize_epex_shape_lab_delta_stability_script.py tests/test_summarize_epex_shape_lab_stability_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_diagnose_fan_to_hourly_parity_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_audit_epex_shape_lab_governance_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `59 passed, 1 skipped`.

Rejected alternatives:

- Extend the local diagnostic bundle builder to emit production-approved
  artifacts.
- Let export/selected/capstone artifacts self-approve without an already
  approved adjusted production manifest.
- Generate a real T046 production chain from the current local NO-GO staging
  manifest.

Invariants not to break:

- The strict chain builder does not approve T046 by itself; it requires an
  already approved adjusted production manifest.
- T046 remains NO-GO production in current real evidence.
- OMPEX remains advisory-only and cannot enter model, selection, readiness, or
  production-chain gates.

## D-20260708-39 - Pre-Register T047 Night/Ramp EPEX Lab Sweep

Decision: the next EPEX lab model step is a T047 v3 no-OMPEX sweep with
explicit `night_intensity` and `ramp_intensity` dimensions, while keeping the
existing weekend, low-tail, peak-subshape, nullspace projection, monthly BASE
authority, and OMPEX-forbidden selection contract.

Reason: read-only expert audit found T046 strict diagnostics strong but uneven
economically: evening recovery, solar/midday and weekend buckets improve, while
night and hourly ramp gains are weak. T046 is also close to the current ramp
p99 increase cap. Promoting T046 before addressing those weak buckets would
prematurely freeze a lab candidate that is not yet the best model.

Implementation:

- Updated `pfc_shaping/lt/model/epex_shape_lab.py`:
  - fitted templates now expose `night_delta_eur_mwh` and
    `ramp_delta_eur_mwh`;
  - `ABShapeLabConfig` includes `night_intensity` and `ramp_intensity`;
  - raw deltas include these terms before the existing BASE/PEAK nullspace
    projection and cap/floor enforcement.
- Updated `scripts/run_epex_shape_lab_ab.py`:
  - CLI/API accept `--night-intensity` and `--ramp-intensity`;
  - manifests and delta summaries record both values.
- Updated `scripts/plan_epex_shape_lab_sweep.py`:
  - plans can pre-register `night_intensity` and `ramp_intensity`;
  - scoring policy includes `night_weight`;
  - JSON CLI options accept `@file.json` to avoid shell quoting errors.
- Updated `scripts/execute_epex_shape_lab_sweep.py`:
  - executor passes and validates the new intensities;
  - ranking rows include `night_intensity`, `ramp_intensity`, and
    `night_mean_delta_eur_mwh`;
  - no-OMPEX selection basis explicitly records night/ramp components.
- Updated `scripts/compare_epex_shape_lab_ab.py`:
  - independent calendar diagnostics now include `night_00_05`.

Real T047 v3 pre-registration:

- Plan folder:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/`
- Plan:
  `pre_registered_sweep_plan.json`
- Candidate CSV:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv`
- Spot parquet:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet`
- Grid:
  - weekend `0.5`;
  - low-tail `0.25`;
  - peak-subshape `0.75`;
  - night `[0.0, 0.25, 0.5]`;
  - ramp `[0.0, 0.25, 0.5]`;
  - cap `[2.0, 3.0]`.
- Thresholds:
  - max EPEX spot age `14.0` days;
  - min EPEX fit coverage `730.0` days;
  - max ramp p99 increase `0.9` EUR/MWh;
  - min adjusted price `-10.0` EUR/MWh.
- Scoring:
  - duck `1.0`;
  - solar tail `1.0`;
  - weekend `1.0`;
  - night `0.75`;
  - ramp penalty `1.5`.
- Trial count: `18`.

Smoke execution:

- Command executed first two trials only:
  `python scripts/execute_epex_shape_lab_sweep.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/pre_registered_sweep_plan.json --output-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_smoke_summary.json --max-trials 2 --no-resume`
- Result:
  - `trial_count_executed=2`;
  - `eligible_count=1`;
  - best smoke trial `t001_w05_l025_p075_n00_r00_d02`;
  - ramp p99 increase `0.6714740399999428` EUR/MWh;
  - max monthly drift `1.0119047646367243e-07`;
  - width drift `0`;
  - weighted negative hours `0`;
  - governance `PASS`.
- The cap `3.0` smoke trial is not eligible under the tighter ramp threshold,
  which confirms the new threshold is active.

Validation:

- `python -m pytest tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `40 passed, 1 skipped`.

Rejected alternatives:

- Promote T046 immediately because strict diagnostics pass.
- Use OMPEX history to tune night/ramp parameters.
- Use fan-parquet staging for adjusted promotion artifacts while fan-to-hourly
  parity is still failed.

Invariants not to break:

- OMPEX remains advisory-only and cannot enter model, selection, scoring,
  readiness, or production-chain gates.
- EPEX lab deltas must remain projected into the BASE/PEAK nullspace and must
  preserve monthly solver authority.
- T047 is lab-only until a full strict diagnostic, stability, spot-backtest,
  product normalization, Power BI, governance, and real production-chain path
  passes.

## D-20260708-40 - T047 Full Sweep Is Diagnostic, Not A T046 Replacement

Decision: do not freeze or promote T047 v3 as a replacement for T046 based on
the first full sweep. Keep it as diagnostic model evidence and use the results
to design the next night/ramp refinement.

Reason: the full T047 v3 sweep produced eligible no-OMPEX candidates, but no
candidate cleanly dominates T046. The best ranking trial improves ramp p99
governance but does not improve the realized spot night/ramp buckets enough.
The best weak-bucket compromise improves night materially versus T046, but
does not beat T046 on overall profile MAE, evening recovery, solar-tail,
weekend, post-valuation MAE, or mean hourly-ramp MAE.

Real full sweep:

- Plan:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/pre_registered_sweep_plan.json`
- Execution summary:
  `output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_summary.json`
- Result:
  - `trial_count_executed=18`;
  - `eligible_count=9`;
  - `production_approved=false`;
  - `ompex_used_in_model=false`;
  - `ompex_used_in_selection=false`.

Best internal no-OMPEX ranking trial:

- `t005_w05_l025_p075_n00_r05_d02`
- cap `2.0`, night `0.0`, ramp `0.5`
- independent shape score `1.315185990064843`
- ramp p99 increase `0.7062480299999798`
- max monthly drift `1.1111111105262712e-07`
- width drift `0`
- weighted negative hours `0`
- governance `PASS`

No-OMPEX spot backtests were then run for the 9 eligible T047 trials:

- Output root:
  `output/phase14/t047_spot_backtest_by_trial/`
- Summary CSV:
  `output/phase14/t047_spot_backtest_by_trial/eligible_spot_backtest_summary.csv`

Best weak-bucket compromise:

- `t013_w05_l025_p075_n05_r00_d02`
- adjusted CSV SHA-256:
  `d7b93c7caf4c38ec51cd94d37f0f5308feef9df50bb1ca263705627ac8d7b1fb`
- overall profile MAE improvement: `0.29295542439021466`
- night MAE improvement: `0.11792184918005748`, positive folds `10/12`
- hourly-ramp MAE improvement: `0.034116846702457994`, positive folds `10/12`
- evening recovery MAE improvement: `0.32206130775585989`
- solar-tail MAE improvement: `0.29033105604920667`
- weekend MAE improvement: `0.1990459761545178`
- post-valuation MAE improvement: `0.22709564079301003`

Reference T046 no-OMPEX spot evidence:

- adjusted CSV SHA-256:
  `8b50a01af05dc152a5f95fbd85e36c4bbe0106f0e65c4dd118b3df42737378c8`
- overall profile MAE improvement: `0.4054835410318921`
- night MAE improvement: `0.031908941150684988`, positive folds `5/12`
- hourly-ramp MAE improvement: `0.035478178105887714`, positive folds `8/12`
- evening recovery MAE improvement: `0.45338812791781463`
- solar-tail MAE improvement: `0.43729530913049253`
- weekend MAE improvement: `0.28896113473708351`
- post-valuation MAE improvement: `0.3048038417338681`

Interpretation:

- T047 v3 proves the new night component can improve the night bucket
  materially and increase positive night folds.
- T047 v3 does not yet improve hourly-ramp mean MAE versus T046; it only
  improves ramp positive-fold consistency for the best weak-bucket compromise.
- T046 remains the stronger lab candidate overall, but still NO-GO production
  until a real adjusted production-chain path exists.

Rejected alternatives:

- Freeze `t005` from the internal ranking without spot bucket validation.
- Freeze `t013` because it improves night while ignoring degraded overall and
  solar/evening/weekend evidence.
- Use OMPEX to break the tie or tune the next grid.

Invariants not to break:

- T047 remains lab-only and no-OMPEX.
- A future replacement for T046 must be selected on pre-registered no-OMPEX
  evidence and must explicitly beat the incumbent on the targeted weak buckets
  without losing strict diagnostics.
- OMPEX remains advisory-only after a candidate is frozen.

## D-20260708-41 - Make T047 Spot-Bucket Selection Reproducible

Decision: add a read-only summarizer for no-OMPEX spot backtests of eligible
EPEX sweep trials. The summarizer is the canonical way to compare T047-like
trials against an incumbent on weak-bucket evidence before any freeze decision.

Reason: T047 v3 exposed a mismatch between the internal independent A/B score
and the realized spot weak-bucket objective. The internal sweep ranked
`t005_w05_l025_p075_n00_r05_d02` first, while manual spot-bucket comparison
showed `t013_w05_l025_p075_n05_r00_d02` as the only candidate that materially
improves night evidence while staying within the incumbent ramp tolerance. This
selection logic must be repeatable and fail-closed, not a one-off PowerShell
aggregation.

Implementation:

- Added `scripts/summarize_epex_shape_lab_spot_backtests.py`.
- Added `tests/test_summarize_epex_shape_lab_spot_backtests_script.py`.
- The script reads:
  - an executed no-OMPEX sweep summary;
  - per-trial no-OMPEX spot backtest summaries under a backtest root;
  - an optional incumbent spot-backtest summary.
- It rejects:
  - non-`executed_independent_no_ompex` sweep summaries;
  - any OMPEX usage in model, selection, or backtest;
  - production-approved or promotion-gate evidence.
- It writes:
  - `spot_backtest_trial_ranking.csv`;
  - `spot_backtest_selection_summary.json`.
- Weak-bucket candidate defaults:
  - night positive folds at least `8`;
  - ramp positive folds at least `8`;
  - night must beat incumbent;
  - ramp may regress by at most `0.002` EUR/MWh versus incumbent.

Real T047 v3 selection summary:

- Command:
  `python scripts/summarize_epex_shape_lab_spot_backtests.py --sweep-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_summary.json --backtest-root output/phase14/t047_spot_backtest_by_trial --incumbent-backtest output/phase14/t046_spot_backtest_v2_buckets/spot_backtest_summary.json --output-dir output/phase14/t047_spot_backtest_selection_summary`
- Output:
  `output/phase14/t047_spot_backtest_selection_summary/spot_backtest_selection_summary.json`
- Ranking CSV:
  `output/phase14/t047_spot_backtest_selection_summary/spot_backtest_trial_ranking.csv`
- Result:
  - `trial_count_from_sweep=9`;
  - `trial_count_summarized=9`;
  - `strict_pass_count=9`;
  - `weak_bucket_candidate_count=1`;
  - best weak-bucket trial `t013_w05_l025_p075_n05_r00_d02`;
  - `replacement_verdict.status=WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`;
  - `replace_incumbent=false`.

Validation:

- `python -m pytest tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `45 passed, 1 skipped`.

Rejected alternatives:

- Continue producing weak-bucket rankings with ad hoc shell aggregation.
- Freeze the internal best A/B ranking trial without spot-bucket selection.
- Use OMPEX to break the tie.

Invariants not to break:

- The summarizer is lab-only and cannot approve production.
- T046 remains the incumbent lab candidate until a future no-OMPEX selection
  beats weak buckets and core metrics without weakening strict diagnostics.
- OMPEX remains advisory-only after a candidate is frozen.

## D-20260708-42 - Orchestrate Eligible Trial Spot Backtests

Decision: add a read-only runner that executes no-OMPEX spot backtests for
eligible EPEX sweep trials directly from a pre-registered plan and executed
sweep summary, then optionally chains the weak-bucket summarizer.

Reason: D41 made weak-bucket selection reproducible once per-trial spot
backtests exist, but those backtests were still produced by an ad hoc
PowerShell loop. Future T048/T049 sweeps need a single fail-closed command for
the full post-sweep evaluation path: eligible trials only, no OMPEX, resume
safe, and optional incumbent comparison.

Implementation:

- Added `scripts/run_epex_shape_lab_sweep_spot_backtests.py`.
- Added `tests/test_run_epex_shape_lab_sweep_spot_backtests_script.py`.
- The runner reads:
  - `pre_registered_sweep_plan.json`;
  - executed sweep summary JSON;
  - sweep ranking CSV referenced by the summary.
- It validates:
  - plan is `lab_only` and `pre_registered_independent_no_ompex`;
  - sweep is `executed_independent_no_ompex`;
  - all production flags are false;
  - no OMPEX model/selection/backtest flags;
  - candidate and spot hashes match the plan;
  - only `eligible_for_selection=True` trials are processed.
- It supports resume by hash-validating existing `spot_backtest_summary.json`
  files against baseline, adjusted CSV and spot parquet hashes.
- It can call `scripts/summarize_epex_shape_lab_spot_backtests.py` after the
  backtest run when `--selection-output-dir` is provided.

Real T047 v3 runner validation:

- Command:
  `python scripts/run_epex_shape_lab_sweep_spot_backtests.py --plan-json output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/pre_registered_sweep_plan.json --sweep-summary output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_t047_v3/sweep_execution_summary.json --output-root output/phase14/t047_spot_backtest_by_trial --output-summary output/phase14/t047_spot_backtest_by_trial/run_summary_from_runner.json --incumbent-backtest output/phase14/t046_spot_backtest_v2_buckets/spot_backtest_summary.json --selection-output-dir output/phase14/t047_spot_backtest_selection_summary_from_runner`
- Result:
  - `trial_count_backtested=9`;
  - `reused_existing_count=9`;
  - `ompex_used_in_model=false`;
  - `ompex_used_in_selection=false`;
  - `ompex_used_in_backtest=false`;
  - chained selection verdict remains
    `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`;
  - `replace_incumbent=false`.

Validation:

- `python -m pytest tests/test_run_epex_shape_lab_sweep_spot_backtests_script.py tests/test_summarize_epex_shape_lab_spot_backtests_script.py tests/test_backtest_epex_shape_lab_against_spot_script.py tests/test_epex_ab_shape_lab.py tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_compare_epex_shape_lab_ab_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
  returned `48 passed, 1 skipped`.

Rejected alternatives:

- Keep using shell loops for eligible trial backtests.
- Backtest every planned trial instead of eligible trials only.
- Let a post-sweep runner read OMPEX or approve production.

Invariants not to break:

- The runner is lab-only and cannot promote production.
- It must process only no-OMPEX eligible trials from an executed no-OMPEX
  sweep.
- T046 remains incumbent until a future no-OMPEX candidate beats weak buckets
  and core metrics without weakening strict diagnostics.

## D-20260708-43 - T048 Is Evidence, Not Replacement; T049 Core-Balance Next

Decision: do not replace the current EPEX lab incumbent T046 with any T048
trial. Treat T048 as local no-OMPEX research evidence that identifies the next
search direction. The next sweep is T049 core-balance, centered around the
T048 `t020` / `t024` neighborhood, with a tighter ramp threshold and an
explicit no-regression replacement contract versus T046.

Reason: T048 materially improves the weak night and ramp buckets, but no T048
trial dominates T046 across the core business buckets and post-valuation
evidence. The official T048 selection output reports
`replacement_verdict.status=WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
and `replace_incumbent=false`. Read-only MIT/Roaster audits concurred:

- `t004_w05_l025_p075_n05_r00_d275` improves night, ramp, and post-valuation
  versus T046 but degrades overall, evening, solar-tail, and weekend.
- `t020_w075_l025_p075_n05_r00_d275` improves overall, night, ramp, solar-tail,
  and weekend, but degrades evening and post-valuation.
- `t024_w075_l025_p01_n05_r00_d275` is the strongest compromise, improving
  overall, night, ramp, evening, and weekend, but remains slightly behind T046
  on solar-tail and post-valuation.

T048 executed evidence:

- plan used for execution:
  `output/phase14/t048_ncr/pre_registered_sweep_plan.json`
- executed summary:
  `output/phase14/t048_ncr/sweep_execution_summary.json`
- spot backtest run:
  `output/phase14/t048_ncr_spot_backtests/run_summary.json`
- selection summary:
  `output/phase14/t048_ncr_selection_summary/spot_backtest_selection_summary.json`
- result:
  - `trial_count_executed=32`
  - `eligible_count=27`
  - `trial_count_backtested=27`
  - `strict_pass_count=27`
  - `weak_bucket_candidate_count=16`
  - all OMPEX flags remain false
  - `production_approved=false`
  - `promotion_gate=false`

T049 pre-registration contract:

- grid neighborhood:
  - weekend intensity `[0.65, 0.75]`
  - low-tail intensity `[0.25]`
  - peak-subshape intensity `[0.75, 0.875, 1.0]`
  - night intensity `[0.4, 0.5, 0.6]`
  - ramp intensity `[0.0, 0.125]`
  - max absolute delta grid `[2.5, 2.75]`
- selection thresholds:
  - `max_epex_spot_age_days=14.0`
  - `min_epex_fit_coverage_days=730.0`
  - `max_ramp_p99_increase_eur_mwh=0.90`
  - `min_adjusted_price_eur_mwh=-10.0`
- scoring policy:
  - `duck_weight=1.0`
  - `solar_tail_weight=1.25`
  - `weekend_weight=1.0`
  - `night_weight=1.0`
  - `ramp_penalty_weight=2.0`

Rejected alternatives:

- Freeze `t004` because it improves the weakest buckets while accepting core
  bucket degradation.
- Freeze `t020` because it improves the headline overall score while accepting
  evening and post-valuation degradation.
- Use OMPEX to break ties between T048 candidates.
- Treat local lab-only T048 artifacts as production promotion evidence.

Invariants not to break:

- T046 remains the incumbent lab candidate until a no-OMPEX candidate beats
  weak buckets and core metrics without weakening strict diagnostics.
- T049 remains lab-only and cannot promote production.
- OMPEX remains advisory-only after candidate freeze; it is not an input, loss,
  ranking signal, or gate.
- Any adjusted EPEX promotion still requires a real chain-bound adjusted
  production manifest, export manifest, selected artifact, and capstone.

## D-20260708-44 - T049/T050 Identify Frontier Candidate But No Full Replacement

Decision: keep T046 as the incumbent lab candidate. Treat T049/T050 best trial
`t070_w075_l025_p01_n06_r00_d275` as the current no-OMPEX frontier candidate,
not as a replacement and not as promotion evidence.

Reason: T049 core-balance improved materially over T048 and found a
near-dominating candidate, but still found no trial that beats T046 on every
replacement bucket. T050 micro-balance around that frontier did not recover the
remaining solar-tail gap; it reproduced the same best candidate under a
different trial id.

T049 executed evidence:

- plan:
  `output/phase14/t049_core_balance/pre_registered_sweep_plan.json`
- sweep summary:
  `output/phase14/t049_core_balance/sweep_execution_summary.json`
- spot backtest run:
  `output/phase14/t049_core_balance_spot_backtests/run_summary.json`
- selection summary:
  `output/phase14/t049_core_balance_selection_summary/spot_backtest_selection_summary.json`
- result:
  - `trial_count_executed=72`
  - `eligible_count=52`
  - `trial_count_backtested=52`
  - `strict_pass_count=52`
  - `weak_bucket_candidate_count=52`
  - `replace_incumbent=false`
  - degraded versus T046: evening and post-valuation for the automatic
    best-ranked weak-bucket trial.

T049 frontier candidate:

- trial id: `t070_w075_l025_p01_n06_r00_d275`
- adjusted CSV sha256:
  `f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`
- parameters:
  - weekend intensity `0.75`
  - low-tail intensity `0.25`
  - peak-subshape intensity `1.0`
  - night intensity `0.6`
  - ramp intensity `0.0`
  - max absolute delta `2.75`
- strict lab checks: PASS
- no-OMPEX flags: all false
- production flags: `production_approved=false`, `promotion_gate=false`
- ramp p99 increase: `0.8886024799999568`, below the T049 threshold `0.90`
- min adjusted price: `-3.740814`
- versus T046 incumbent spot-backtest improvements:
  - overall: `0.42709956252228376` versus `0.40548354103189205`
  - night: `0.15957030400928707` versus `0.03190894115068499`
  - ramp: `0.05043407090595627` versus `0.035478178105887714`
  - evening: `0.45756877823583286` versus `0.45338812791781463`
  - solar-tail: `0.4312351115165488` versus `0.4372953091304925`
  - weekend: `0.2990265337481961` versus `0.2889611347370835`
  - post-valuation: `0.3053769058019675` versus `0.3048038417338681`

T050 executed evidence:

- plan:
  `output/phase14/t050_t070_micro_balance/pre_registered_sweep_plan.json`
- sweep summary:
  `output/phase14/t050_t070_micro_balance/sweep_execution_summary.json`
- spot backtest run:
  `output/phase14/t050_t070_micro_balance_spot_backtests/run_summary.json`
- selection summary:
  `output/phase14/t050_t070_micro_balance_selection_summary/spot_backtest_selection_summary.json`
- result:
  - `trial_count_executed=12`
  - `eligible_count=4`
  - `trial_count_backtested=4`
  - `strict_pass_count=4`
  - `weak_bucket_candidate_count=4`
  - best trial `t007_w075_l025_p01_n06_r00_d275`
  - adjusted CSV sha256 matches T049 frontier:
    `f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`
  - replacement verdict remains
    `WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS`
  - only degraded metric versus T046 is solar-tail.

Rejected alternatives:

- Promote the near-dominating frontier candidate by treating a small solar-tail
  miss as immaterial without a pre-registered tolerance rule.
- Keep sweeping wider grids immediately despite the frontier now being a
  diagnostics/policy question rather than a coarse search question.
- Use OMPEX to decide whether the small solar-tail miss is acceptable.

Invariants not to break:

- `t070`/`t007` remains lab-only until strict delivered-product, Power BI,
  source hierarchy, selected artifact, and adjusted production-chain evidence
  exist and pass.
- T046 remains incumbent unless a future no-OMPEX policy explicitly permits a
  tolerance for the remaining solar-tail miss or another candidate dominates
  T046 on all required buckets.
- OMPEX may be run only as an advisory post-check after no-OMPEX candidate
  selection; it must not influence ranking, thresholds, or promotion.

## D-20260708-45 - T070 Strict Diagnostics Pass, Production Chain Still Missing

Decision: record `t070_w075_l025_p01_n06_r00_d275` as the current strict
diagnostic frontier, not a production candidate. It passes delivered-product
normalization with an exact source-hierarchy policy and passes Power BI strict
export, but production readiness remains NO-GO until the adjusted production
chain exists.

Reason: after T049/T050 identified `t070` as the no-OMPEX frontier, strict
delivered-curve checks were run in isolated `output/phase14` paths. These
checks prove the adjusted CSV is technically coherent at the delivered-product
and dashboard boundary, but the promotion checker correctly rejects it because
the adjusted production manifest, adjusted export manifest, adjusted selected
artifact, and adjusted capstone do not exist.

Source hierarchy policy:

- `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t070_asof20260707_t049_core_balance.json`
- policy sha256: `6c2f1b1f8bcf3bd732858a7e0b593c6e678d1e2758b5fc3c11f1bd5a4bbb462e`
- bound adjusted CSV sha256:
  `f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`
- bound forwards sha256:
  `a6244638c2234781853284ce2ad58d55d01265568cca6c85d4461f21446e8d76`
- quote conflict identity hash:
  `a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`

Delivered-product strict audit:

- command:
  `python scripts/audit_ch_product_normalization.py --csv output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/candidate_epex_shape_lab_adjusted.csv --forwards output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet --required-forward-date 2026-07-07 --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t070_asof20260707_t049_core_balance.json --output-csv output/phase14/t049_core_balance/t070_diagnostics/product_normalization_with_policy/gates.csv --summary-json output/phase14/t049_core_balance/t070_diagnostics/product_normalization_with_policy/summary.json`
- result:
  - `all_gates_pass=true`
  - `critical_count=0`
  - `unsupported_count=0`
  - `quote_conflict_count=6`
  - `accepted_quote_conflict_count=6`
  - `blocking_quote_conflict_count=0`

Power BI strict:

- command:
  `python scripts/build_powerbi_exports.py --csv output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/candidate_epex_shape_lab_adjusted.csv --forwards output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_sweep_v2/diagnostic_forwards_history_rebuilt_20260708.parquet --spot output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet --output-dir output/phase14/t049_core_balance/t070_diagnostics/powerbi_strict`
- result:
  - `powerbi_quality_gate_status=PASS`
  - `shape_score_10=9`
  - `hfc_vs_spot_score_10=9`
  - `max_eex_base_error_eur_mwh=0.000000`
  - `max_eex_peak_error_eur_mwh=0.000000`
  - `weighted_negative_hours=0.000000`
  - all critical flag counts `0`

Promotion readiness:

- output:
  `output/phase14/t049_core_balance/t070_diagnostics/promotion_readiness/decision.json`
- result:
  - `approved=false`
  - `strict_diagnostics_pass=true`
  - `production_chain_pass=false`
  - `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
  - missing evidence:
    - `adjusted_production_manifest`
    - `adjusted_export_manifest`
    - `adjusted_selected_config`
    - `adjusted_capstone`

Rejected alternatives:

- Treat strict diagnostics as sufficient promotion evidence.
- Reuse the T046 source-hierarchy policy for a different adjusted CSV hash.
- Reuse the baseline selected config or capstone as authority for the adjusted
  T070 hourly artifact.

Invariants not to break:

- The T070 source-hierarchy policy accepts only the exact bound CSV/forwards
  hashes and quote conflict identity.
- Strict diagnostics do not override the missing adjusted production chain.
- OMPEX remains advisory-only and was not used in these diagnostics.

## D-20260708-46 - T070 OMPEX Advisory Is Favorable But Non-Gating

Decision: record the OMPEX 2026-07-08 comparison for `t070` as advisory
post-freeze evidence only. It supports the no-OMPEX frontier direction on
average error metrics, but it does not change replacement, selection, or
production readiness status.

Reason: OMPEX is useful but imperfect external evidence. It must remain outside
model inputs, parameter selection, thresholds, ranking, and promotion gates.
The comparison was run only after T049/T050 no-OMPEX selection and after T070
strict diagnostics were frozen.

Inputs:

- OMPEX file:
  `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min\HFC_Ompex_20260708_101700.xlsx`
- baseline output:
  `output/phase14/t049_core_balance/t070_diagnostics/ompex_advisory_baseline_20260708/benchmark_metrics.json`
- T070 output:
  `output/phase14/t049_core_balance/t070_diagnostics/ompex_advisory_t070_20260708/benchmark_metrics.json`
- alignment: `ompex_minus_1h_hourending`
- overlap points: `39481`

T070 minus baseline advisory deltas:

- MAE: `-0.141892598541069`
- RMSE: `-0.171529868510628`
- correlation: `+0.0026155586617813`
- p95 absolute error: `-0.398625999999986`
- inside p10/p90 rate: `+0.00268483574377548`
- max absolute error: `+0.937722`

Rejected alternatives:

- Use OMPEX deltas to override the T046/T070 solar-tail no-OMPEX decision.
- Add OMPEX as a gate to replacement or production readiness.
- Tune another sweep against OMPEX.

Invariants not to break:

- OMPEX remains advisory-only and imperfect.
- No OMPEX/HFC artifact may enter the LT model, loss, ranking, or promotion
  authority.
- T070 remains NO-GO production until adjusted production-chain evidence
  exists and passes.

## D-20260708-47 - T070 Local Bundle Narrows Blocker To Adjusted Production Manifest

Decision: package T070 strict diagnostic evidence into a local non-production
bundle, then keep production readiness NO-GO. The local bundle is useful review
evidence, not a production chain substitute.

Reason: D45 proved T070 strict delivered-product and Power BI diagnostics pass,
but the first readiness check had four missing production-chain artifacts. The
existing bundle builder can package local adjusted export, selected artifact,
and local capstone evidence while preserving explicit non-production flags.
Rerunning readiness with that local bundle narrows missing evidence to the real
production blocker: `adjusted_production_manifest`.

Local bundle command:

`python scripts/build_epex_lab_promotion_bundle.py --lab-manifest output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/ab_lab_manifest.json --baseline-monthly-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_asof20260707_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json --product-summary output/phase14/t049_core_balance/t070_diagnostics/product_normalization_with_policy/summary.json --powerbi-summary output/phase14/t049_core_balance/t070_diagnostics/powerbi_strict/summary_metrics.csv --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t070_asof20260707_t049_core_balance.json --independent-summary output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/independent_ab_comparison/ab_comparison_summary.json --governance-audit output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/governance_audit/epex_shape_lab_governance_audit.json --output-dir output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle`

Local bundle outputs:

- `output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle/adjusted_export_manifest.json`
- `output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle/adjusted_selected_artifact.json`
- `output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle/adjusted_local_capstone_no_go.json`

Readiness command with local bundle:

`python scripts/check_epex_lab_promotion_readiness.py --lab-manifest output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/ab_lab_manifest.json --governance-audit output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/governance_audit/epex_shape_lab_governance_audit.json --independent-summary output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/independent_ab_comparison/ab_comparison_summary.json --product-summary output/phase14/t049_core_balance/t070_diagnostics/product_normalization_with_policy/summary.json --powerbi-summary output/phase14/t049_core_balance/t070_diagnostics/powerbi_strict/summary_metrics.csv --adjusted-export-manifest output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle/adjusted_export_manifest.json --adjusted-selected-config output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle/adjusted_selected_artifact.json --adjusted-capstone output/phase14/t049_core_balance/t070_diagnostics/local_promotion_bundle/adjusted_local_capstone_no_go.json --output output/phase14/t049_core_balance/t070_diagnostics/promotion_readiness/decision_with_local_bundle.json`

Readiness result:

- exit code `1`, expected for NO-GO production
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `approved=false`
- `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
- `missing_production_evidence=["adjusted_production_manifest"]`
- local adjusted export/selected/capstone artifacts are present and hash-bound,
  but their production-ready checks correctly fail because they are explicitly
  local diagnostic artifacts.

Rejected alternatives:

- Treat the local bundle as a production chain.
- Flip production approval flags without a valid production run identity,
  production entrypoint, 40-hex git commit, and source provenance manifest.
- Continue sweeping parameters before the production-chain decision is made.

Invariants not to break:

- Local bundle artifacts must remain `production_approved=false`.
- T070 remains NO-GO production until a real adjusted production manifest
  exists and the strict adjusted production-chain builder can bind export,
  selected artifact, and capstone to it.
- The remaining blocker is governance/production packaging, not another
  no-OMPEX shape sweep.

## D-20260708-48 - T070 Staging Reproducibility Requires Night/Ramp Parameters

Decision: extend `scripts/stage_epex_lab_adjusted_lt_candidate.py` to accept
and persist `night_intensity` and `ramp_intensity`, then stage T070 from the
audited 2026-07-08 baseline CSV with exact source provenance. This produces a
NO-GO adjusted production manifest with `contract_pass=true`, proving the
frontier can be reproduced through the staging path.

Reason: T070 depends on `night_intensity=0.6` and `ramp_intensity=0.0`. The
sweep executor and A/B runner already supported these parameters, but the
production-staging path did not. Without this, the adjusted production manifest
path could not reproduce the selected frontier candidate and would be
incomplete for any future production chain.

Implementation:

- `scripts/stage_epex_lab_adjusted_lt_candidate.py`
  - added `night_intensity` and `ramp_intensity` function parameters;
  - added CLI flags `--night-intensity` and `--ramp-intensity`;
  - passes both values to `run_ab`;
  - records both values in `epex_lab_config`.
- `tests/test_stage_epex_lab_adjusted_lt_candidate_script.py`
  - verifies API and CLI propagation into the staging manifest and lab
    manifest.

Validation:

- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
- result: `23 passed, 1 skipped`

T070 source export manifest:

- command:
  `python scripts/build_epex_lab_source_export_manifest.py --candidate-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --baseline-monthly-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_asof20260707_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json --selected-config .planning/phases/14-lt-audit-remediation/monthly_curve_selected_config_asof20260707_lshape100_yoy150_amp150_2032.json --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_asof20260707_lshape100_yoy150_amp150_2032.json --capstone output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/promotion_triad_real_prod_check/promotion_decision_real_prod_triad.json --output output/phase14/t049_core_balance/t070_diagnostics/source_export_manifest_baseline_20260708.json`
- source CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- source export manifest sha256:
  `d662548e2e7605ba2b59e024afd3040f2724fe84c5f3c7d3491fbaa0e1909f1d`

T070 staging command:

`python scripts/stage_epex_lab_adjusted_lt_candidate.py --candidate-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --source-export-manifest output/phase14/t049_core_balance/t070_diagnostics/source_export_manifest_baseline_20260708.json --output-dir output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate --spot-parquet output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet --valuation-timestamp 2026-07-07T00:00:00Z --weekend-intensity 0.75 --low-tail-intensity 0.25 --peak-subshape-intensity 1.0 --night-intensity 0.6 --ramp-intensity 0.0 --max-abs-delta-eur-mwh 2.75 --lookback-years 5 --baseline-monthly-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_asof20260707_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json --product-summary output/phase14/t049_core_balance/t070_diagnostics/product_normalization_with_policy/summary.json --powerbi-summary output/phase14/t049_core_balance/t070_diagnostics/powerbi_strict/summary_metrics.csv --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t070_asof20260707_t049_core_balance.json --independent-summary output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/independent_ab_comparison/ab_comparison_summary.json --governance-audit output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/governance_audit/epex_shape_lab_governance_audit.json`

T070 staging result:

- staged adjusted CSV sha256:
  `f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`
- source provenance manifest:
  `output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate/source_provenance_manifest.json`
- source provenance manifest sha256:
  `dbc3bb810dffba948e6eadfc237890a6ebea3887e57a85e4236fda5e60473d51`
- adjusted production manifest NO-GO:
  `output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate/adjusted_production_manifest_no_go.json`
- adjusted production manifest NO-GO sha256:
  `0e09ea55a130bce73de8bf9ba6a163cbc124f656fdb7b00375c8b2ac4d249048`
- `source_promotion_eligible=true`
- `production_contract_blockers=[]`
- `missing_production_contract_inputs=[]`
- `adjusted_production_contract_pass=true`

Readiness with NO-GO production manifest:

- output:
  `output/phase14/t049_core_balance/t070_diagnostics/promotion_readiness/decision_with_no_go_production_manifest.json`
- exit code `1`, expected for NO-GO production
- `strict_diagnostics_pass=true`
- `production_chain_pass=false`
- `missing_production_evidence=[]`
- key PASS checks:
  - adjusted production manifest bound;
  - adjusted production manifest contract pass;
  - adjusted production manifest source provenance pass;
  - source provenance source/export/staged/lab hashes bound.
- key FAIL checks:
  - adjusted production manifest approved;
  - adjusted production manifest run identity valid;
  - local adjusted export/selected/capstone are not production-ready or
    production-chain-bound.

Rejected alternatives:

- Treat T070 as unstaged because stage lacked night/ramp arguments.
- Manually edit a staged manifest to inject night/ramp values.
- Promote from local diagnostic bundle without a valid production run identity.

Invariants not to break:

- Staging remains lab-only and non-promotional.
- CLI-built adjusted production manifests remain NO-GO by default.
- Any real approval must come from a production path that supplies valid run
  identity, 40-hex git commit, source provenance, and then uses the strict
  adjusted production-chain builder.

## D-20260708-49 - Adjusted Production Chain Builder Validates Source Provenance

Decision: harden `scripts/build_epex_lab_adjusted_production_chain.py` so it
does not trust `source_provenance_pass=true` by assertion alone. Before writing
approved adjusted export, selected, or capstone artifacts, it now validates the
source provenance manifest path, sha256, schema, candidate-CSV source kind,
promotion eligibility, absence of blockers, adjusted CSV hash, source CSV
hash, staged candidate hash, lab manifest hash, source export manifest hash,
source export binding, and no-OMPEX flags.

Reason: D48 produced a T070 NO-GO adjusted production manifest with
`contract_pass=true` and a valid source provenance manifest. The next builder
in the chain was already strict on approval flags and run identity, but it
trusted the production manifest's `source_provenance_pass` field instead of
revalidating the underlying evidence. A future approved manifest must not be
able to generate chain-bound export/selected/capstone artifacts from
self-attested or tampered source provenance.

Implementation:

- `scripts/build_epex_lab_adjusted_production_chain.py`
  - added `_source_provenance_errors`;
  - added `_source_export_manifest_bound_to_csv`;
  - validates source provenance against actual files and hashes before chain
    artifacts are written.
- `tests/test_build_epex_lab_adjusted_production_chain_script.py`
  - added rejection coverage for missing/self-attested source provenance;
  - added rejection coverage for tampered source provenance source hash.

Validation:

- `python -m pytest tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
- result: `34 passed, 1 skipped`

Real T070 NO-GO chain-builder check:

- command:
  `python scripts/build_epex_lab_adjusted_production_chain.py --adjusted-production-manifest output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate/adjusted_production_manifest_no_go.json --output-dir output/phase14/t049_core_balance/t070_diagnostics/should_not_build_chain_from_no_go`
- result: exit code `1`, expected
- refusal:
  `approved adjusted production manifest required: production_approved, production_promotion_approved, git_commit`

Rejected alternatives:

- Rely on readiness alone to catch self-attested source provenance after chain
  artifacts have already been written.
- Trust `source_provenance_pass=true` without checking the bound provenance
  manifest.
- Allow fan-parquet or unbound source-export provenance into the production
  chain builder.

Invariants not to break:

- The chain builder writes production-approved artifacts only from an approved,
  contract-pass, source-provenance-pass, run-identity-valid manifest whose
  underlying source provenance is independently hash-validated.
- T070 remains NO-GO production until a real adjusted production manifest has
  valid approval flags and run identity.
- OMPEX remains outside model, selection, and production-chain validation.

## D-20260708-50 - Adjusted Production Approval Requires No-OMPEX Selection Pass

Decision: require `selection_policy_pass=true` on an adjusted production
manifest before it can be approved or used to build a production chain. The
selection policy must be no-OMPEX and must explicitly report
`replacement_verdict.replace_incumbent=true` for the bound adjusted CSV sha.

Reason: strict diagnostics and source provenance are not sufficient to promote
an adjusted EPEX lab candidate. T070 passes strict delivered-product and Power
BI diagnostics, and its staging provenance can reproduce the frontier CSV, but
the T049 selection summary still reports `replace_incumbent=false` because the
candidate misses the T046 solar-tail incumbent. Without a selection-policy
gate, a future approved production manifest could bypass the no-OMPEX
replacement decision.

Implementation:

- `scripts/build_epex_lab_adjusted_production_manifest.py`
  - added optional `selection_summary`;
  - added CLI flag `--selection-summary`;
  - records `selection_summary`, `selection_summary_sha256`, and
    `selection_policy_pass`;
  - `selection_policy_pass` requires explicit no-OMPEX flags
    (`ompex_used_in_model=false`, `ompex_used_in_selection=false`,
    `ompex_used_in_backtest=false`), a replacement verdict with
    `replace_incumbent=true`, and an adjusted CSV sha match to the canonical
    selected trial/artifact fields;
  - production approval now requires both `contract_pass=true` and
    `selection_policy_pass=true`.
- `scripts/epex_lab_selection_policy.py`
  - added the shared no-OMPEX selection policy validator.
- `scripts/check_epex_lab_promotion_readiness.py`
  - added `adjusted_production_manifest_selection_policy_pass`;
  - reloads `selection_summary`, validates `selection_summary_sha256`, and
    recalculates the policy instead of trusting the manifest boolean.
- `scripts/build_epex_lab_adjusted_production_chain.py`
  - refuses approved chain artifacts unless the same hash-bound
    `selection_summary` recalculation passes;
  - fixed the `_same_path` helper so source-export path binding is actually
    evaluated before falling back to sha binding.
- `scripts/stage_epex_lab_adjusted_lt_candidate.py`
  - accepts `--selection-summary` and forwards it to the adjusted production
    manifest builder.

Validation:

- `python -m pytest tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_build_epex_lab_adjusted_production_manifest_script.py tests/test_build_epex_lab_adjusted_production_chain_script.py tests/test_check_epex_lab_promotion_readiness_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
- result: `47 passed, 1 skipped`

Real T070 selection-guard staging:

- command:
  `python scripts/stage_epex_lab_adjusted_lt_candidate.py --candidate-csv output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/ch_hfc_hourly_asof20260707_lshape100_yoy150_amp150_2032.csv --source-export-manifest output/phase14/t049_core_balance/t070_diagnostics/source_export_manifest_baseline_20260708.json --output-dir output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate_selection_guard --spot-parquet output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/epex_spot_refresh_20260708/epex_hourly_ch_energy_charts_20260708.parquet --valuation-timestamp 2026-07-07T00:00:00Z --weekend-intensity 0.75 --low-tail-intensity 0.25 --peak-subshape-intensity 1.0 --night-intensity 0.6 --ramp-intensity 0.0 --max-abs-delta-eur-mwh 2.75 --lookback-years 5 --baseline-monthly-manifest output/phase14/20260708_asof20260707_lshape100_yoy150_amp150_2032/fan_asof20260707_lshape100_yoy150_amp150_2032.monthly_curve_manifest.json --product-summary output/phase14/t049_core_balance/t070_diagnostics/product_normalization_with_policy/summary.json --powerbi-summary output/phase14/t049_core_balance/t070_diagnostics/powerbi_strict/summary_metrics.csv --source-hierarchy-policy .planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t070_asof20260707_t049_core_balance.json --independent-summary output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/independent_ab_comparison/ab_comparison_summary.json --governance-audit output/phase14/t049_core_balance/t070_w075_l025_p01_n06_r00_d275/governance_audit/epex_shape_lab_governance_audit.json --selection-summary output/phase14/t049_core_balance_selection_summary/spot_backtest_selection_summary.json`

Result:

- adjusted CSV sha256 still matches T070:
  `f3d1f9d749823c9babd1104261670dcd115a63f797e6aed2e38ef480cbdf40cb`
- selection summary sha256:
  `0822379db522fadedbb12ae0ab327763fc2cbf28dac4443905ca2f010fb62183`
- `adjusted_production_contract_pass=false`
- adjusted production manifest NO-GO:
  `output/phase14/t049_core_balance/t070_diagnostics/staged_adjusted_candidate_selection_guard/adjusted_production_manifest_no_go.json`
- adjusted production manifest NO-GO sha256:
  `a042a9b22ac8144e00b62f46c879d46921f4fd9686e94698f54348d4271c12e1`
- manifest fields:
  - `contract_pass=false`
  - `selection_policy_pass=false`
  - `production_approved=false`
  - `production_promotion_approved=false`
- failed check:
  - `selection_policy_pass=FAIL`
  - `replace_incumbent=false`
  - no OMPEX flags

Rejected alternatives:

- Let production approval rely only on diagnostics/provenance.
- Use OMPEX advisory improvement to override `replace_incumbent=false`.
- Approve T070 with an implicit tolerance for the solar-tail miss.

Invariants not to break:

- A production-approved adjusted EPEX lab candidate must have no-OMPEX
  replacement evidence for the exact adjusted CSV sha.
- OMPEX advisory remains outside selection, thresholds, ranking, and promotion.
- T070 remains NO-GO production until either a future no-OMPEX candidate passes
  replacement or an explicit pre-approved no-OMPEX tolerance policy is added
  and bound to the selection artifact.

## D-20260708-51 - Split Evening Recovery From Peak Subshape in EPEX Lab

Decision: add a lab-only `evening_recovery_intensity` component to the EPEX
shape lab. The component is fitted only from historical EPEX residuals, is
projected through the same BASE/PEAK nullspace as the other EPEX lab deltas,
and remains off-production / no-OMPEX. It is pre-registerable in sweep plans and
recorded in lab/staging manifests.

Reason: T049/T050/T070 showed a single-parameter conflict:

- lower `peak_subshape_intensity` preserved solar-tail but degraded
  evening/post-valuation;
- higher `peak_subshape_intensity` recovered evening/post-valuation but lost
  the solar-tail incumbent;
- T070 therefore remained NO-GO after D50 because
  `replacement_verdict.replace_incumbent=false`.

T051 and T052 confirmed that tuning `peak_subshape_intensity` and cap alone
does not close all metrics. T053, using the new evening component, beats T046 on
overall, evening, solar-tail, weekend, night, and ramp, but still misses the
post-valuation metric. This is a material improvement of the search frontier
without changing selection policy.

Implementation:

- `pfc_shaping/lt/model/epex_shape_lab.py`
  - added `ABShapeLabConfig.evening_recovery_intensity`;
  - fits `evening_recovery_delta_eur_mwh` from EPEX residuals for h17-h21;
  - includes it in raw delta construction, validation, template validation,
    and audit output.
- `scripts/run_epex_shape_lab_ab.py`
  - added API/CLI propagation and pre-registration manifest recording.
- `scripts/plan_epex_shape_lab_sweep.py`
  - added grid dimension and command propagation.
- `scripts/execute_epex_shape_lab_sweep.py`
  - passes the parameter, validates resume manifests, records ranking rows.
- `scripts/stage_epex_lab_adjusted_lt_candidate.py`
  - stages and records the parameter.

Validation:

- `python -m pytest tests/test_run_epex_shape_lab_ab_script.py tests/test_plan_epex_shape_lab_sweep_script.py tests/test_execute_epex_shape_lab_sweep_script.py tests/test_stage_epex_lab_adjusted_lt_candidate_script.py tests/test_lt_ct_imports.py -q -p no:cacheprovider`
- result: `34 passed, 1 skipped`

Local no-OMPEX sweeps, generated artifacts only:

- T051 solar/evening bridge:
  - plan:
    `output/phase14/t051_solar_evening_bridge/pre_registered_sweep_plan.json`
  - sweep:
    `output/phase14/t051_solar_evening_bridge/sweep_execution_summary.json`
  - selection:
    `output/phase14/t051_solar_evening_bridge_selection_summary/spot_backtest_selection_summary.json`
  - result: `replace_incumbent=false`; degraded metrics:
    `evening_mae_improvement_eur_mwh`,
    `post_valuation_mae_improvement_eur_mwh`.
- T052 peak/cap bridge:
  - plan:
    `output/phase14/t052_peak_cap_bridge/pre_registered_sweep_plan.json`
  - selection:
    `output/phase14/t052_peak_cap_bridge_selection_summary/spot_backtest_selection_summary.json`
  - result: `replace_incumbent=false`; degraded metrics:
    `evening_mae_improvement_eur_mwh`,
    `post_valuation_mae_improvement_eur_mwh`.
- T053 evening-recovery bridge:
  - plan:
    `output/phase14/t053_evening_recovery_bridge/pre_registered_sweep_plan.json`
  - selection:
    `output/phase14/t053_evening_recovery_bridge_selection_summary/spot_backtest_selection_summary.json`
  - best trial:
    `t003_w075_l025_p082_e025_n055_r00_d27`
  - adjusted CSV sha256:
    `8b1c7f43bdaf3513d417fb6f436470847270af4b83ad5e5053eab08c16b94762`
  - metrics:
    - overall `0.4403741843600797` vs incumbent `0.40548354103189205`
    - evening `0.4675508496854987` vs incumbent `0.45338812791781463`
    - solar-tail `0.44832283438649295` vs incumbent `0.4372953091304925`
    - weekend `0.3277011767246539` vs incumbent `0.2889611347370835`
    - night `0.15638454130406818` vs incumbent `0.03190894115068499`
    - ramp `0.05921350078935572` vs incumbent `0.035478178105887714`
    - post-valuation `0.292289994623653` vs incumbent
      `0.3048038417338681`
  - result: `replace_incumbent=false`; only degraded metric:
    `post_valuation_mae_improvement_eur_mwh`.
- T054 high-peak/low-tail:
  - plan:
    `output/phase14/t054_high_peak_low_tail/pre_registered_sweep_plan.json`
  - selection:
    `output/phase14/t054_high_peak_low_tail_selection_summary/spot_backtest_selection_summary.json`
  - result: `replace_incumbent=false`; best trial reproduces T070 and still
    degrades `solar_tail_mae_improvement_eur_mwh`.

Rejected alternatives:

- Relax the replacement policy or accept T053 despite the post-valuation miss.
- Use OMPEX advisory performance to choose thresholds or override selection.
- Patch final months/hours after the solver or EPEX lab output.

Invariants not to break:

- `evening_recovery_intensity` remains lab-only and no-OMPEX.
- Selection still requires all incumbent core metrics to be non-degraded unless
  an explicit pre-approved no-OMPEX tolerance policy is added.
- Generated T051-T054 artifacts remain local evidence and are not commit
  targets.

## D-20260709-52 - T056 Replacement Selection Requires Explicit Selected Artifact

Decision: EPEX lab spot-backtest selection now distinguishes the best weak
bucket from the best replacement candidate. The selected artifact for promotion
evidence is the first weak-bucket candidate that does not degrade any incumbent
core metric, recorded explicitly as `selected_trial` and
`selected_adjusted_csv_sha256`.

Reason: T056 exposed a governance bug in the selection summary. The top weak
bucket trial improved the ranking score but slightly degraded the
post-valuation metric. A lower-ranked trial, T056 t005, beat the incumbent T046
on every declared core metric and was the actual replacement candidate. The
selection artifact must bind production evidence to that exact CSV hash instead
of relying on an implicit top-ranked weak-bucket row.

Validation:

- T056 selected trial:
  `t005_w075_l025_p089_e005_n055_r00`
- selected adjusted CSV sha256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- selection summary:
  `output/phase14/t056_postval_final_micro_selection_summary/spot_backtest_selection_summary.json`
- selection summary sha256:
  `b2a319ac91eff51947387bc2a1dcc4784b2f5bf5536ea861f2e63ab9fc5cf10d`
- `replacement_candidate_count=1`
- `replacement_verdict.replace_incumbent=true`
- OMPEX flags remain false for model, selection, and backtest.

T056 t005 improves the T046 incumbent on all declared core metrics:

- overall `0.4506842423821014` vs `0.40548354103189205`
- evening `0.4688940576897349` vs `0.45338812791781463`
- solar-tail `0.46091530831501754` vs `0.4372953091304925`
- weekend `0.3283653976588017` vs `0.2889611347370835`
- post-valuation `0.3049947368951571` vs `0.3048038417338681`
- night `0.16252506955713483` vs `0.03190894115068499`
- ramp `0.053194830053255315` vs `0.035478178105887714`

Rejected alternatives:

- Select the top weak-bucket row even when it degrades a core metric.
- Relax the post-valuation non-degradation requirement for T056.
- Use OMPEX advisory performance to choose or override the selected artifact.

Invariants not to break:

- `selected_adjusted_csv_sha256` must match the exact adjusted CSV being
  staged or promoted.
- Replacement selection remains no-OMPEX and hash-bound.
- Post-valuation remains a non-degradation veto, not a target for further
  micro-tuning.

## D-20260709-53 - T056 Diagnostics Pass But Production Promotion Remains NO-GO

Decision: T056 t005 is the current no-OMPEX lab replacement candidate and has
strict diagnostic evidence, but it remains NO-GO for production promotion until
a real adjusted production run identity and the export/selected/capstone chain
are approved and manifest-bound.

Reason: the exact T056 adjusted CSV now passes product normalization with a
hash-bound source hierarchy policy and strict Power BI gates. Reproducible
staging from the 2026-07-07 baseline CSV regenerates the same adjusted CSV
sha256 and records source provenance plus selection-policy pass. However, the
available adjusted production manifest is intentionally a CLI/staging NO-GO
artifact with no 40-hex production git commit and approval flags false.

Evidence:

- source hierarchy policy:
  `.planning/phases/14-lt-audit-remediation/quote_conflict_source_hierarchy_policy_t056_asof20260707_postval_final_micro.json`
- policy sha256:
  `71abb1151bf4f46728baffbdb6e6398c4a9a70e7273c4cc22fdb6a4fdfa73962`
- product audit with policy:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/product_normalization_with_policy/summary.json`
  - `all_gates_pass=true`
  - `critical_count=0`
  - `unsupported_count=0`
  - `accepted_quote_conflict_count=6`
  - `blocking_quote_conflict_count=0`
  - `quote_conflict_identity_hash=a28d7f15151e730dca2099335e1d7e75dcf52e3a77edb6871352f9942c882846`
- strict Power BI summary:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/powerbi_strict/summary_metrics.csv`
  - `powerbi_quality_gate_status=PASS`
  - `weighted_negative_hours=0`
  - critical flags `0`
- staged reproducibility manifest:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/staged_adjusted_candidate_selection_guard/staged_lt_epex_lab_candidate_manifest.json`
  - adjusted CSV sha256:
    `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
  - source CSV sha256:
    `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
  - source provenance sha256:
    `347d617a23cddd35e3f3a791d42b205e2989c04885fa03ebb23942d9d5c5d2e6`
  - adjusted production manifest NO-GO sha256:
    `052fdd1c3bc82cfe41f8e3600c9f577a1b571be99a2ba20123ad6118b7747c8d`
- readiness with staged manifest:
  `output/phase14/t056_postval_final_micro/t005_diagnostics/promotion_readiness/decision_with_staged_manifest.json`
  - `strict_diagnostics_pass=true`
  - `production_chain_pass=false`
  - `status=STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING`
  - missing evidence:
    `adjusted_export_manifest`, `adjusted_selected_config`,
    `adjusted_capstone`

Rejected alternatives:

- Treat the source hierarchy policy `production_approved=true` as approval of
  the curve itself.
- Promote a CLI/staging manifest with `production_approved=false`.
- Build export/selected/capstone artifacts before an approved adjusted
  production manifest exists.
- Use OMPEX as a promotion or selection gate.

Invariants not to break:

- Source hierarchy policy approval is only a QUOTE_CONFLICT/source hierarchy
  waiver for the bound CSV/forwards/identity hash.
- T056 promotion requires real production identity, source provenance,
  selection-policy pass, approved adjusted production manifest, adjusted export
  manifest, selected artifact, and capstone.
- OMPEX remains advisory-only after freeze and never a model, calibration,
  selection, or gate input.

## D-20260709-54 - Lock T057 Future Holdout Before Any Further T056 Tuning

Decision: freeze T056 t005 as the current no-OMPEX candidate and pre-register a
future locked holdout, T057, before any further EPEX lab tuning or production
promotion attempt. The T057 plan binds the exact baseline CSV, adjusted CSV,
selection summary, lab manifest, holdout window, and pass criteria.

Reason: T056 t005 is the first no-OMPEX replacement candidate, but the
post-valuation edge over T046 is very small and based on a short observed
window. Further micro-tuning against that window would overfit. A future
holdout must be frozen before the future spot rows are known, then audited
without OMPEX and without changing candidate parameters.

Pre-registered plan:

- plan:
  `.planning/phases/14-lt-audit-remediation/locked_holdout_plan_t057_t056_asof20260709.json`
- plan sha256:
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`
- plan id:
  `t057_locked_t056_future_holdout`
- frozen at:
  `2026-07-09T00:00:00Z`
- holdout window:
  `2026-07-10T00:00:00Z` to `2026-07-24T00:00:00Z`
- baseline CSV sha256:
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`
- adjusted CSV sha256:
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`
- selection summary sha256:
  `b2a319ac91eff51947387bc2a1dcc4784b2f5bf5536ea861f2e63ab9fc5cf10d`
- lab manifest sha256:
  `013a11ba0e6a0a2f32eeb78493e154731ab736542710bd5b31e148c37e7716bc`
- minimum holdout hours:
  `300`
- minimum residual MAE improvement:
  `0.0 EUR/MWh`

Implementation:

- Added `scripts/plan_epex_lab_locked_holdout.py`.
- Added `scripts/audit_epex_lab_locked_holdout.py`.
- Added `scripts/check_epex_lab_locked_holdout_coverage.py`.
- Added tests for no-OMPEX hash binding and holdout pass/fail audit.

Coverage status as of the 2026-07-08 spot parquet:

- command output:
  `output/phase14/t057_locked_t056_future_holdout/coverage_status_current_spot.json`
- status:
  `WAITING_FOR_FULL_SPOT_COVERAGE`
- spot max:
  `2026-07-08T23:00:00Z`
- observed holdout hours:
  `0`
- expected holdout hours:
  `336`

Rejected alternatives:

- Continue tuning T056 against the short post-valuation sample.
- Use OMPEX as a future holdout gate or target.
- Treat the locked holdout as production approval by itself.
- Edit the T057 plan after the holdout window starts; create a new plan if a
  different future window is needed.

Invariants not to break:

- T057 evaluates the exact selected T056 adjusted CSV hash only.
- T057 audit remains lab-only, read-only, no-OMPEX, and non-promotional.
- A passing T057 holdout can support scientific confidence but still does not
  replace the required adjusted production/export/selected/capstone chain.

