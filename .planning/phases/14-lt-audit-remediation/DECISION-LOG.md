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

