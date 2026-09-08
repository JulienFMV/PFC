# Post-D312 quality roadmap - proposal only

Date: 2026-09-08. User asks for the next expert plan to reach a very high-quality
FMV PFC. This document proposes sequencing; it does not adopt a new model,
register a new scientific holdout, launch a daily collection, or promote a
release. No new numbered decision is created. All six authorities remain false.

## Read first and retained evidence

- AGENTS.md and .planning/HANDOFF.md.
- SESSION-HANDOFF-20260908-MATCHED-VINTAGE-PILOT.md (D312).
- PFC-FMV-PRODUCT-QUALITY-CHARTER-20260713.md (normative acceptance targets).
- CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-20260724.md (draft economic/truth design).
- docs/data/CH-DAY-AHEAD-RESOLUTION-20260908.md.
- build/lt-matched-vintages-20260908/future-truth-contract.json.

D304 remains the signed seasonal reference. D305-D310 results remain exposed
comparison evidence; failed fixed recency/maturity grids are not reopened to
search for a winner. D312 contains one actual daily observation out of 20;
19 future observations and automated collection remain pending. Its 159 passing
tests and four skips qualify the stated local first-day work, not production.

D312 preserves the raw D304 hourly shape and existing solver parameters while
refreshing eligible EEX quotes. Product gates: 80 PASS, 9 QUOTE_CONFLICT,
0 CRITICAL. Monthly residual: 1.0203393685515039e-11 EUR/MWh. On 27 common
months, OMPEX monthly-level MAD falls from D311 3.993154 to D312 2.893846
EUR/MWh; shape MAD remains about 11.48. These are distances between forecasts,
not accuracy or grounds for choosing a model. LSEG does not support 2029 in
the acquired snapshot. The supplied OMPEX workbook is native hourly despite
its directory name. Native Swiss quarter-hour truth remains unqualified.

Pinned D312 closure SHA256:
40038f832ee9deb934dc0829593a0902d46ae545f137f16f37917ffea7275921.
Pinned D312 recipe SHA256:
afeefbb98ca6f0349efe3ed7f275aae0e374d88140f687a07af4d21519c82c3f.
Pinned D312 candidate curve SHA256:
13bca031c6364d71ba32277d2db54459d6a0cc61551912b0af862f34abddc263.
These are previously recorded identities, not a fresh replay performed for
this proposal.

## Proposed sequence and exit criteria

1. Complete the market/data foundation first. Audit the nine quote conflicts
   against exact contracts, source/revision identities, publication calendars
   and eligible hierarchy. Prepare a reviewable hierarchy and evidence for
   any required product authority; no implicit waiver. Qualify source
   freshness, delivery timezone/DST, native interval, units, revisions and
   missing data by horizon. Keep solver levels and existing EEX projection.
   Exit: complete source/conflict register with explicit supported/unsupported
   horizons; no hidden critical failure. Product repricing <=1e-6 EUR/MWh;
   monthly conservation within the configured tolerance (local target 1e-9).

2. Make prospective evidence durable. Provide actual daily source capture,
   observation receipts, manifests and attributable revisions through a
   governed execution mechanism; the existing offline builder is not an
   installed scheduler. Continue the 20-day revision pilot without fabricated
   dates, stale replays or treating overlapping daily curves as independent
   validation folds. Independently register future candidate commitments,
   exact truth provider/series, availability/finalization policy and custodian.
   October 2026 is the first proposed delivery month; its Swiss monthly close
   is 2026-10-31T23:00:00Z and scoring must also wait for qualified final truth.
   Exit: replayable source-to-export lineage and an admitted future protocol;
   daily stability evidence remains separate from long-horizon accuracy.

3. Define economic acceptance alongside data work. Bind real FMV FIL/ACC
   volumes and BLOC13 contractual legs/settlement conventions; do not substitute
   a generic profile for a contract. Define generation/consumption signs and
   any CHF FX convention. Compare valuations, solar/hydro capture prices,
   hedge costs and dispatch decisions using one versioned nonanticipative
   optimizer and identical physical/information constraints. Any clairvoyant
   comparator must obey the same feasibility and terminal conditions.
   Exit: reproducible economic baseline and thresholds fixed before results.

4. Only then test one bounded structural hourly-shape improvement. Audit
   point-in-time load, solar/wind, hydro and neighbouring-market drivers and
   their genuinely available future inputs. Actual future fundamentals cannot
   be silently used as operational forecasts. Use an interpretable additive
   residual first, with a small tree comparator only if justified and frozen
   before evaluation. Components have zero monthly mean; assembler/projection
   remain unchanged. One global recipe, no delivery-month model selection.
   Separate monthly level, centred hourly shape and full-price diagnostics;
   score delivery years, seasons, horizons, negative/near-zero prices, ramps
   and shape extremes with pre-origin thresholds and minimum support.
   Exit: charter targets of >=2% weighted MAE AND RMSE improvement, MAE wins
   on >=70% eligible folds, and no critical regime worse by >2% without the
   specified economic justification and product-owner waiver. Prior exploratory
   5% checks are not production acceptance criteria. Insufficiently supported
   regimes stay unsupported. Exposed historical folds alone cannot promote.

5. Add coherent uncertainty and hydro decision evidence. Freeze a modest set
   of joint weather/load/renewable/hydro/neighbour scenarios and evaluate
   calibration, dependence and economic decisions. Forward levels constrain
   the energy-weighted ensemble expectation; level-risk scenarios may differ
   individually under the frozen design. A shape-only experiment that fixes
   every path's monthly means is reported separately. Targets: >=2% WIS/CRPS
   improvement and 80/90% coverage within 3 percentage points in supported
   buckets, plus economic validation. GPU is authorized when a measured batch
   workload warrants it, with reproducibility and suitable CPU parity checks.

6. Qualify operational release after the scientific and source gates. Use
   python -m pfc_shaping.cli.governed_release; require independent production,
   export and selected-configuration manifests, final-curve audit, repeatable
   hashes, explicit failure states, recovery/rollback and ownership. Route
   executable/browser/container E2E requirements to the governed standard-user
   CI runner, not this managed laptop. No release authority is inferred from
   successful local tests or this proposal.

Immediate recommended execution package: source/conflict/freshness audit,
durable daily capture and future-truth admission preparation, with FMV economic
contract inventory in parallel. The next model experiment depends on these
deliverables; no fresh broad hyperparameter sweep is recommended.

## Rationale, rejected alternatives and invariants

Forward market consistency and realized-spot forecast accuracy answer different
questions; a physical expectation and a traded forward can differ through risk
premia. Do not replace solver levels to minimize realized monthly spot error.
External OMPEX/LSEG differences are diagnostic evidence only: no fitting,
selection, priors, target blending or automatic authority. Product-specific
Swiss hourly/day-ahead and intraday quarter-hour markets remain distinct; no
calendar-only conversion or repeated-hour series is native quarter-hour truth.

Reject further unfrozen model sweeps, per-month winners/patches, invented
prospective vintages, treating 20 overlapping snapshots as 20 independent
origins, and a claim of validated three-year accuracy within a few weeks.
Near-term data/reliability work can finish before the future evaluation period;
long-horizon qualification depends on genuine historical vintages and future
outcomes. Freeze exposure-aware evaluation before observing new scores.

Preserve all existing artifacts, solver authority, LT/CT isolation and all six
false authorities. No AFRY, T057, protected desk data or Power BI work. User GPU
and Warehouse permissions persist. A future Warehouse workload must retain
bounded queries and documented termination behavior: prior manual stop HTTP403
is an external limitation; auto-stop is 45 minutes, not proof of shutdown.

Primary sources consulted for planning context (not model inputs):
- https://arxiv.org/abs/1308.3378 (power-market pricing measure/risk premium).
- https://arxiv.org/abs/2103.16918 (spot/futures models for risk management).
- https://www.eex.com/en/markets/power/power-futures
- https://www.epexspot.com/en/basicspowermarket
- https://www.swissgrid.ch/content/dam/swissgrid/about-us/newsroom/publications/balancing-roadmap-en.pdf
- https://www.jao.eu/news/tsos-survey-introduction-15-minutes-mtu
For the Swiss resolution status, the existing 8 September verification cites
the later June JAO delay and July EPEX specification; earlier roadmap targets
must not supersede that evidence or establish an unverified launch date.

## Files, commands, verification and completion boundary

Only new file in this planning response:
.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260908-POST-D312-QUALITY-ROADMAP.md.
Existing handoffs, frozen closure inputs and DECISION-LOG are preserved. No
code, source data, model, artifact export or authority was changed.

Read-only PowerShell inspection used the required exact cwd and Git top-level
guard at C:\Users\jbattaglia\PFC_LT. Commands read the contracts and D312
handoff above and confirmed this proposal path did not already exist.
The new file was added with apply_patch. A final read/hash check validates the
document only; no model tests or numerical replay are claimed for this plan.
All previous test results and hashes above are attributed to D312 evidence.
