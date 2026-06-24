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

