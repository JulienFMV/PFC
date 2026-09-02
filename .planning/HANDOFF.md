# Current handoff

Read in this order:

1. `AGENTS.md`
2. `.planning/HANDOFF.md`
3. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
4. `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260902-LT-SCENARIO-CURVE-AUTHORITY-CONTRACT.md`

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

## Invariants

- The CH monthly BASE solver is the sole monthly-level authority.
- ENTSO-E, neighboring markets, history, weather, Swissgrid, AFRY and LSEG may
  shape or benchmark only; they cannot rewrite monthly solver means.
- LT code must not import `pfc_shaping.ct.*`.
- T057 remains sealed.
- Model admission remains
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS` until independently
  governed local exports and a new future holdout exist.

See durable decisions D-20260821-248 through D-20260902-270 and
`SESSION-HANDOFF-20260902-LT-SCENARIO-CURVE-AUTHORITY-CONTRACT.md` for the
latest exact implementation files, hashes, tests and residual risks. The PFC
target, ENTSO-E export, normalized-interval and preceding repo-hygiene handoffs
retain their exact historical evidence and cleanup counts.
