# Current handoff

Read in this order:

1. `AGENTS.md`
2. `.planning/HANDOFF.md`
3. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
4. `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260903-LT-ORIGIN-REGISTRY-TRUST-CONFORMANCE-V1.md`

## Current state

- The repository baseline is aligned with the audited EEX, ENTSO-E and LSEG
  source layers.
- ENTSO-E Gold is the current-serving layer; Silver vintages are the PIT
  authority. The Gold resource bridge is optional enrichment.
- LSEG curve `110181967` is an independent benchmark only.
- Local datasets, model weights, copied research PDFs and runtime outputs are
  outside Git. Deterministic test fixtures and governed evidence remain in
  Git.
- Generated caches and output directories were cleaned on 2026-08-21.
- The root `AGENTS.md` now permanently requires explicit material assumptions,
  minimal designs, surgical diffs and outcome-driven verification. It asks for
  clarification only when repository evidence cannot resolve a consequential
  choice, and permits a single-use abstraction only when it materially
  isolates a security invariant, stable contract or necessary test seam.
- The external audit's confirmed front-edge, KKT, dependency, CI and legacy
  entrypoint defects are remediated locally. Its post-solver seam-patch and
  dead-contract claims were rejected after code-path verification.
- The offline Gold/Silver materializer was roast-hardened for mapping binding,
  generation/flow direction, boolean/PIT semantics, exact intervals and stable
  hashes. `lt_input_snapshot.v4` now binds its exact replay package, PRD export
  manifest, query/watermark/cost evidence and quality proof into the isolated
  publisher. Real-data admission remains pending.
- The Gold EEX joined projection now has a causal offline materializer. It
  filters by FMV fact-load time and Swiss quotation date, reuses the existing
  EEX normalizer and remains authority-negative pending three-table manifest
  binding and signed vintage conversion.
- The Silver/Gold role split is confirmed for the next real-data pass, and the
  local contract matrix passes. No PRD v4 export or fresh cost receipt exists.
  The configured SQL Warehouse was observed stopped through one read-only
  control-plane call, so no SQL or Warehouse start was attempted.
- The value-blind PRD metadata preflight is now complete. It found the two
  expected Gold spot tables not found or not visible under their contracted
  FQNs, a stopped Warehouse, no file-count/scan estimate and an unpartitioned
  33.41 GB Silver Open-Meteo forecast table. The
  first proposed export is therefore restricted to EEX Gold and year/month-
  pruned ENTSO-E Gold/Silver after a separate platform cost fence.
- Expanded inventory and lineage show that direct Euler spot remains DEV-only,
  while the governed PRD candidate for coupled day-ahead prices is already the
  ENTSO-E Silver vintage family `day_ahead_prices`. The missing EPEX Gold
  interval fact is therefore not an absolute blocker if exact PRD SeriesKeys,
  zones, availability and coverage are proven in the bounded export.
- Self-service control-plane and query-history evidence now establishes the
  SeriesKey formula, a high-confidence fresh 2026-08-24 PRD rebuild signal,
  DEV five-zone coverage and conservative scan proxies. Exact PRD
  classification suffixes, family-specific coverage and the production run
  receipt remain unobserved because the token cannot see the ENTSO-E job.
- A focused hash-bound PRD gate now validates the value-blind five-price
  profile and one-month PIT extracts. Its dedicated, compatibility and package
  matrices pass. It authorizes neither model input nor production; the real
  rebuild receipt, exact PRD keys, bounded profile, cadence evidence and future
  holdout remain pending.
- A separate hash-bound LSEG gate now selects the exact CH, AT, DE-LU and FR
  EPEX actual-price vintages by FMV pipeline first-seen time and reconciles
  them with ENTSO-E at complete UTC-hour grain. IT-North remains visibly
  ENTSO-E-only. Thresholds are explicit and hashed; discrepancies block with
  no silent source substitution. No real values or Databricks compute were
  opened for this offline increment.
- The ENTSO-E day-ahead gate now reflects the observed seven-key Gold
  inventory: base CH/FR/IT-North plus classification sequences 1 and 2 for AT
  and DE-LU. Inventory and PIT selection are separate; the latter requires an
  explicit one-key-per-field choice and has no multi-auction default. The
  complete ENTSO-E, Databricks and LT/CT audit matrices pass without SQL or
  Warehouse start.
- One real July 2026 value-blind profile has now run on the already-active PBI
  SQL Warehouse. It returned the seven expected series in 10.196 seconds and
  read 47.9 MB while pruning 7.78 GB. The capture replays exactly offline, but
  PIT extraction remains blocked: all 17,925 profiled rows fail the combined
  availability-order predicate, 404 fail interval consistency and the formal
  rebuild manifest is still missing. No price, Warehouse start or Databricks
  write occurred.
- A second and final value-blind root-cause statement exactly reconciled those
  failures. All 17,925 availability failures are `publication_timestamp` after
  first-seen and delivery. Producer-code inspection now proves that this field
  is the response XML `createdDateTime`, while first-seen pull time is the run
  start encoded in every landing filename. It is not original historical
  day-ahead publication time. The 404 duration mismatches follow the explicit
  A03 next-position/Period-end block contract, not a SQL `LEAD` defect.
- Independent evidence now reconstructs the 24 August PROD rebuild: successful
  deployment SHA `a7e920d` contains required commit `db3a933`, followed by
  successful full Bronze, Silver and Gold runs; Silver and Gold DQ passed with
  zero failed checks. The July profile reports zero canonical-key mismatch,
  legacy/new interval overlap, duplicate or orphan counts for all seven
  day-ahead series. A platform-signed top-level job/post-backfill-validation
  receipt remains unavailable but is no longer required for bounded LT
  day-ahead continuation.
- A fresh paginated Unity Catalog inventory confirms direct Euler spot is still
  DEV-only: the DEV Bronze and Silver tables are visible, while no Euler- or
  spot-named table is visible in PROD Bronze, Silver or Gold.
- A pure LT consumer contract now separates ENTSO-E day-ahead
  `realized_final` from `causal_asof`, expands producer-normalized Silver
  intervals at native cadence, requires hash-bound effective-dated series
  rules plus the applicable LSEG control, preserves
  UTC/market-time/provenance/quality semantics and derives only
  duration-weighted zero-mean monthly shape. It remains
  authority-negative and has not consumed real prices or Databricks compute.
- Producer code and Jerome's confirmation now close the downstream
  `curve_type` question: Bronze normalizes A01/A03 bounds and Silver vintages
  and latest expose those normalized bounds without `curve_type`. Consumer v2
  therefore validates and expands positive aligned cadence multiples without
  inferring or emitting XML representation metadata. PIT and synthetic
  ENTSO-E/LSEG reconciliation are versioned v2; the two hash-bound SQL files
  and real July profile/diagnostic receipts remain unchanged and replay
  exactly under v1. Direct PRD model use is still blocked by availability/PIT,
  governed-export, effective-dated AT/DE-LU selection and future-holdout gates.
- Two new consumer-complete Silver-vintage exports now separate
  `causal_asof` from a `realized_final` candidate without changing the
  historical PIT/profile SQL or receipts. They accept an explicit market
  subset, bind a Swiss local month across at most two UTC partitions, preserve
  complete lineage, require exact value-bound publication/finality evidence
  and replay offline from a three-artifact unsigned package. The observed
  stopped Warehouse yields `STOP_NO_ACTIVE_WAREHOUSE`; no Databricks access was
  used.
- The hedge-use clarification is now scoped precisely: CH/DE is the target
  valuation and hedge-risk perimeter, while FR/AT/IT-North begin as
  observation/risk markets. The hard monthly solver remains CH-only; the
  current DE branch still uses the legacy monthly path and is not promoted as
  CH-equivalent. Code audit also found that the configured MLP computes
  180-day weights without reweighting hourly samples in its final fit. That is
  a preregistered challenger/ablation issue, not an authorized baseline patch.
- The user has now validated the durable FMV target: a forward-consistent
  Swiss quarter-hour central curve, separately labelled physical/fundamental
  scenarios, stochastic future-spot paths, and downstream portfolio/hedge
  decisions for Swiss hydro/PV, Valais clients and large industry. CH is the
  production priority, DE the first hedge/spread market, and FR/AT/IT-North
  remain observation/risk markets until their independent execution gates.
  Shape-only scenarios preserve solver buckets; level-changing scenarios are
  separate fundamental curves and may not overwrite the central EEX curve.
- The existing LT interfaces now have a durable compatibility/gap map. A new
  standalone metadata-only contract types scenario definitions and the three
  CH curve products without changing `production_phases.py`. It forces all
  operational authorities false, keeps shape-only scenarios solver-month
  neutral, isolates level-changing scenarios behind a separate upstream solve
  requirement and binds stochastic paths to a typed fundamental parent. Its
  qualification is synthetic only.
- Point 1 of the next-session sequence has started with an outage-aware,
  zero-execution acquisition plan. SMARD reported partial gaps and strong
  delays from an ENTSO-E Transparency Platform incident on 2026-09-01; that
  public report grants no internal availability, freshness, publication or
  finality authority. The existing EEX capture rehashes exactly and can advance
  only through provenance/signature completion. ENTSO-E is restricted to a
  platform-owned July `realized_final` smoke export from already-materialized
  Silver rows; the laptop preflight remains `STOP_NO_ACTIVE_WAREHOUSE`, and
  historical backfill remains forbidden as causal truth.
- Point 2 now has a compact immutable LT evaluation contract. It binds the
  existing estimand, origin-registry v2 and dependence/power design; freezes
  the current unweighted MLP plus weighted-MLP, Ridge, spline-Ridge GAM and
  deterministic CPU LightGBM families; and schedules a separate 12-origin
  prospective cohort from October 2026 through September 2027. The semantic
  manifest is hash-closed, but external registration is pending, countable
  origins remain zero, truth opening/training/selection are unauthorized and
  T057 is neither referenced nor consumed.
- The evaluation protocol is now source-bound v5. Four challenger
  implementations and a scoring engine operate only on immutable synthetic
  fixtures: the weighted MLP uses a verified observation-level loss, the
  linear/GAM/tree challengers are deterministic, and all outputs retain
  negative authority. Scoring uses one complete-case intersection, local-month
  energy neutralization and all four lead buckets; unimplemented economic or
  uncovered horizon metrics remain visibly `UNSUPPORTED_NEVER_PASS`. No local
  data through 31 August was opened or used.
- The next offline registration-preparation increment is complete. LT now
  builds exact canonical bytes with domain-separated identities for independent
  Ed25519 schedule-entry signing, verifies a complete ordered 12-slot schedule,
  checks Swiss local delivery starts and binds one origin information set to
  the exact schedule,
  signer and artifact commitments. The runtime has no private-key capability,
  no real signed schedule was created, and trusted time/request/remote
  registry/HEAD evidence remains explicitly missing. Scheduled origins remain
  12, countable origins remain zero and future truth remains closed.
- Local construction now covers the exact request-v2 signing surface. The
  request reverifies the envelope and schedule, binds the HEAD expectation,
  complete commitments and exact opaque trusted-time receipt bytes, derives
  the protocol origin/request IDs and requires disjoint request/schedule
  signer keys. A valid synthetic signature grants no external authority.
  A new local hash-frozen receipt/HEAD wire draft now closes ID derivation,
  domain-separated signature bytes, strict receipt/request binding, exact
  nonce binding and a five-minute maximum HEAD TTL. Its public-key-only
  verifier reverifies the complete request/envelope/schedule chain and has no
  private-key, I/O, registry-write or clock capability. Cryptographic receipt
  and fresh-HEAD success remain separate from authority: registry-key trust,
  external CAS/WORM operation, trusted commit time and independent conformance
  evidence are missing, so externally registered/countable origins remain
  zero. The incompatible SQLite reference is not promoted.
- The next authority-negative registry layer is now locally frozen and
  synthetic-tested. A signed trust-bundle chain constrains registry public-key
  validity and irreversible lifecycle transitions under a distinct
  caller-supplied root key. A thread-safe in-memory harness qualifies
  transport-neutral HEAD, atomic compare-and-append, exact retry, uniqueness
  and immutable sanitized rejection semantics. It has no key custody, clock,
  nonce, persistence, network or external CAS/WORM capability; trust admission,
  registration and countable origins remain false. Evaluation protocol v6
  binds the exact contract and implementation hashes without changing its
  candidates, metrics, cohort or monthly solver boundary.

## Invariants

- The CH monthly BASE solver is the sole monthly-level authority.
- ENTSO-E, neighboring markets, history, weather, Swissgrid, AFRY and LSEG may
  shape or benchmark only; they cannot rewrite monthly solver means.
- LT code must not import `pfc_shaping.ct.*`.
- T057 remains sealed.
- Model admission remains
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS` until independently
  governed local exports and a new future holdout exist.

See durable decisions D-20260821-248 through D-20260903-278 and
`SESSION-HANDOFF-20260903-LT-ORIGIN-REGISTRY-TRUST-CONFORMANCE-V1.md` for the
latest exact implementation files, hashes, tests and residual risks. The PFC
target, source-acquisition, scenario-product, ENTSO-E export,
normalized-interval and preceding repo-hygiene handoffs retain their exact
historical evidence and cleanup counts.
