# Current handoff

Read in this order:

1. `AGENTS.md`
2. `.planning/HANDOFF.md`
3. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
4. `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260907-LOCAL-PFC-SOURCE-INTEGRATION.md`
5. `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260907-LOCAL-PFC-PRD-INPUT-READINESS.md`
6. `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260904-COMMON-FULL-PRICE-ASSEMBLY.md`

## Current state

- D317 audit response implemented; read
  SESSION-HANDOFF-20260908-AUDIT-RESPONSE.md and
  docs/model/PFC-CH-AUDIT-RESPONSE-20260908.md first. Claude audit imported
  unchanged; all33findings dispositioned. Collector/PIT/time/grid/provenance
  fixes pass716tests/5skip; five independent old/current probes and75monthly
  solver levels reverified (maxerror0.0). D304 frozen bindings unchanged.
  New daily recordsv2 bind EEX quotation dates; oldv1bytes remain readable.
  Signed conflict policy, real PRD PIT/block qualification, physical cadence
  weighting, future holdout and lint debt remain open. Do not run a new model
  benchmark before F-01/F-05/F-06/F-10 safeguards. WatchlistV2adds4events;
  extended D312 horizon is2026-10through2032-12, core comparison2026–2029.
  Pilot1/20, no new SQL/Warehouse/GPU/fit; all6false. Six richer local private
  notes remain intentionally different from public Git. Publication/CI and
  preservation receipts are in build/lt-audit-response-20260908/.

- D316 national source readiness completed; read
  SESSION-HANDOFF-20260908-NATIONAL-READINESS.md and
  docs/data/CH-NATIONAL-INPUT-READINESS-20260908.md.21national families;
  four bounded SELECTs, one truncated/rejected and three independently
  reconciled. D300 national physical/hydro evidence reverified. Missing work
  is grids/revisions, outage semantics and future trajectories; no client
  file prerequisite. Structural event registry proposed with four sourced
  examples, no scheduler/model activation. D304/solver unchanged, all6false.
  GitHub code checkpoint dee652bc919d06345f71304d1f1eaacf0edf7bc7, branch
  fix/lt-audit-remediation,591tests passed/5skipped. Six richer local business
  notes intentionally differ from public Git; do not stage them wholesale.

- D315 user clarification: national CH PFC quality comes first; customer budgets
  are optional economic exposure profiles, not Swiss load/generation proxies
  or prerequisites. Read SESSION-HANDOFF-20260908-PUBLIC-AUDIT-CHECKPOINT.md and
  docs/model/PFC-CH-AUDIT-ENTRYPOINT-20260908.md. User authorized GitHub commit/
  push for a Claude audit. Public checkpoint preserves code/tests/protocols;
  granular portfolio details and artifacts stay local. D304/solver unchanged,
  all6false. Continue bounded national input qualification from D300 evidence.

- D314 separate client-volume source discovery is retained locally. It is not
  national Swiss load/generation evidence or a prerequisite for CH model work.
  Public summary: SESSION-HANDOFF-20260908-LT-VOLUME-DISCOVERY.md. D304 and
  solver remain unchanged, all6false; pilot1/20, no scheduler installed.

- D313 source-quality foundation completed:38EEXquotes,6direct/3derived
  conflicts remain unaccepted;134tests pass/4skip. D304/solver unchanged;
  daily collector and history-preservation regression verified. Granular
  client-source diagnostics remain local; national CH work is the priority.
  Read SESSION-HANDOFF-20260908-QUALITY-FOUNDATION.md. All6false.

- D312 first common-cutoff snapshot complete; read `SESSION-HANDOFF-20260908-MATCHED-VINTAGE-PILOT.md` first.
  Actual Sep8T09:45:22UTC, EEXSep7quotes, unchanged D304signedshape/solverrecipe.
  Pilot1/20days,19future captures pending; no background schedule. Commoncutoff
  local observation, not identical vendorissuetimes or independentPIT.159pass/
  4skip,independentreplay,exports,80PASS/9QUOTE_CONFLICT/0CRITICAL;all6false.
  Offline dailybuilder/registry reject duplicate days/stale observations;
  futuretruthcontracts/custody/finalization still pending. OneEEXSELECT on
  activeWarehouse,no start,manualstop403not retried,auto-stop45min,finalRUNNING.
  Report build/lt-matched-vintages-20260908/RAPPORT-PILOTE.md. NoCT/AFRY/T057.


- D311 external benchmark completed; read `SESSION-HANDOFF-20260908-EXTERNAL-PFC-BENCHMARK.md` first.
  OMPEX Sep8 workbook captured; LSEG PRD CHE110181967 single Sep8issue captured.
  D304 Sep7 unchanged. Descriptive distances only: no same-vintage accuracy,
  tuning or promotion. OMPEX39months/LSEG27months,2029LSEGunsupported; levels
  separate from shape.111tests pass/4skip; independent replay verified.
  User now authorizes Warehouse start when needed (GPU also authorized).
  One start/6SELECTs; stopHTTP403, auto-stop45min, user informed, no further SQL.
  Report build/lt-external-benchmark-20260908/RAPPORT-BENCHMARK-EXTERNE.md.
  All6authoritiesfalse, no model/product-code change, AFRY/T057 excluded.


- D310 completed; read `SESSION-HANDOFF-20260908-HOURLY-MATURITY.md` first.
  D304 retained: maturity residual worsens shape MAE5.447341%,25regime
  regressions; calendar residual also rejected.14fits/14assemblies, no tuning
  or month switches; solver unchanged, all authoritiesfalse. Independent
  review and6candidate/controlCSV exports verified;282pass/5skip/2exact known
  Phase5 failures. One Parquet freq metadata review failure preserved and
  corrected only in replay adapter. GPU explicitly authorized when useful;
  this small ridge ran onCPU. User-reported LSEG Databricks and daily10:18 OMPEX
  source location saved for external matched-vintage qualification; not read
  yet. No Warehouse/AFRY/T057. Report build/lt-maturity-20260908/RAPPORT-MATURITE.md.


- D309 revision/seam audit complete. Read
  `SESSION-HANDOFF-20260908-HOURLY-REVISION-AUDIT.md` first. D304 retained;
  seasonal seams do not justify smoothing:30seasonal/62other-month events,
  full/shape MAE ratios0.962204/0.797615, below fixed1.10. No new model.
  Four counterfactual assemblies isolate levels/history/EEX projection on72337
  common delivery hours; history effects1.21–1.92 EUR/MWh mean absolute, smaller
  than level changes8.04–106.13. Two early pairs have no overlap; real daily
  archived-vintage stability remains UNSUPPORTED, revised data not PIT.
  Independent1796inputs/104sources/173outputs,59120events/11136seam metrics,
  32revision series;733 D301 and746 D308 closure-bound files unchanged.
  Current counterfactual CSVs28513hourly/114052repeated-QH rows;80PASS/
  9QUOTE_CONFLICT/0CRITICAL, solver<=2.4102e-11.18/74future D304 seam scale
  exceedances remain descriptive, not established failures. Holdout Oct2026–
  Sep2027 draft not independently registered, truth/custodian requirements open.
  Tests90focused pass;272pass/5skip/2 exact known Phase5 failures.
  Report `build/lt-hourly-revision-audit-20260908/RAPPORT-AUDIT.md`.
  No product-code change, active process, Warehouse/GPU/AFRY/T057 or authority.

- D308 hourly CH stability benchmark complete. Read
  `SESSION-HANDOFF-20260908-HOURLY-STABILITY.md` first. D304 remains primary;
  D305–D307 comparisons preserved. Four global25%/50% blends against both
  recency half-lives,28 new assemblies,28 cached controls,0 fits. No month switch.
  Three blends show no>5% regime/origin regressions, but all miss2% shape MAE
  gain; no adoption. One-year50% loses delivery2024 and origin2023. New shape
  tails supported across4origins; absolute HIGH_PRICE0 remains UNSUPPORTED.
  Monthly level MAE47.260956 unchanged; solver residual<=2.4102e-11.
  Four candidate hourly/repeated-QH CSV pairs28513/114052rows;80PASS/
  9QUOTE_CONFLICT/0CRITICAL, retained7September valuation. Independent review
  verifies1173inputs/100sources/622outputs,8768metrics/1920month decompositions,
  49signed predictions and28 affine assembly checks;733 D301 files unchanged.
  Tests78focused pass;255pass/5skip/2 exact pre-existing Phase5 failures.
  Report: `build/lt-hourly-stability-20260908/RAPPORT-COMPARATIF.md`.
  No product code changes, Warehouse/GPU/AFRY/T057 or active process;
  all authorities false. The fixed grid is exhausted, not a tuning invitation.

- D307 hourly CH recency comparison is complete. Read
  `SESSION-HANDOFF-20260908-HOURLY-RECENCY.md` first. Web verification confirms
  EPEX July2026 CH day-ahead60min definition; JAO30June delays Swiss-border
  capacity15min MTU/block bids to2027, without an end2027 or domestic energy
  auction date. See `docs/data/CH-DAY-AHEAD-RESOLUTION-20260908.md`.
  Same six origins, frozen365.25/730.5day half-lives and D304/MLP controls.
  Two-year shape MAE19.991763/RMSE29.214678 versus D30420.345234/30.180010;
  gains1.737%/3.199%,3/4 origin wins but4 regime regressions: no adoption.
  One-year candidate also fails (2/4 wins,12 regressions). Delivery2024 and
  autumn/near-zero ramps expose weaknesses; absolute HIGH_PRICE assessment
  mask is empty/UNSUPPORTED, never a passed spike control. Monthly forward-spot
  error is identical across candidates and is distinct from solver consistency.
  Three signed current hourly CSVs28513 rows each plus repeated15min transport,
  retained7September valuation and80PASS/9QUOTE_CONFLICT/0CRITICAL.
  Independent review verifies799 inputs/95 sources/373 outputs,3808 metrics,
  960 month decompositions,21 raw reference replays and12096 support cells.
  Tests57focused pass; matrix234pass/5skip/2 unchanged Phase5 failures.
  Report: `build/lt-hourly-recency-20260908/RAPPORT-COMPARATIF.md`.
  All authorities false; no Warehouse/GPU/AFRY/T057, no active process.
  D304–D306 remain references; no model/month switching from exposed results.
- D306 price-conditioned intraday benchmark is complete. Read
  `SESSION-HANDOFF-20260907-PRICE-CONDITIONED-INTRADAY.md` first.
  Reused existing ShapeIntraday factors in signed price space; no new estimator,
  adapter, assembler change or default replacement. D304/D305 remain references.
  36 DE origin-month pairs separate observed-parent disaggregation from an
  actual pre-origin calendar-price forecast. Native conditional MAE improves
  5.87% versus flat but negative-parent MAE worsens12.12%; effective forecast
  MAE improves only0.366%. No robust global adoption; unconditional additive
  near-zero weakness persists. CH15min truth is still hourly repeated.
  Two current CSV candidates114052 QH each (Oct2026-Dec2029), full Parquet
 219268 QH through2032;80 PASS/9 QUOTE_CONFLICT/noCRITICAL. Hourly drift<=1.80e-12,
  monthly solver residual<=2.40e-11. All authorities false; no hydro addition,
  Warehouse/GPU/AFRY/T057. Four new CPU fits;16 D305 fits reused;14 CH assemblies.
  Independent review-v3 verifies619 input pins/90 sources/178 outputs,
 2880 metric rows and288 predictions;733 D301 files unchanged.
  Tests99 focused pass; matrix221pass/5skip/2 unchanged known Phase5 failures.
  Report: `build/lt-price-conditioned-20260907/RAPPORT-COMPARATIF.md`.
  No process remains. Next protocol should address price-regime/support
  conditioning explicitly and hourly forecast error; no score-driven month
  switching or claims of validated CH transfer/2030 accuracy.
- D305 signed composition is complete. Read
  `SESSION-HANDOFF-20260907-SIGNED-COMPOSITION.md` in Phase14 first.
  Existing assembler accepts explicit hourly-neutral additive intraday shape
  and the existing monthly-neutral water correction in the signed lane. No
  adapter, default replacement, Warehouse/GPU/AFRY/T057 or authority change.
  Local comparison: hydro shows no gain (signed MAE20.345234 versus20.352302
  with hydro). Additive DE intraday MAE6.225856 versus native6.571542, but
  negative-parent MAE4.707404 versus2.152440 (+118.70%); reject unconditional
  adoption despite aggregate gain. Eight DE months are conditional intrahour
  diagnostics using actual parent-hour prices, not price forecasts or CH proof.
  Five current PFC variants exported Oct2026-Dec2029,114052 QH each; all share
  solver levels and80 PASS/9 QUOTE_CONFLICT, noCRITICAL. Current native control
  exact; D304 signed replay error<=5.6844e-14; intraday hourly-mean drift<=1.94e-12.
  Final run-v4:6 hydro fits+16 intraday fits,29 curves;48 error stages independently
  recomputed,12 water+16 factor model replays exact.151 inputs/87 source pins/
  203 outputs verified. Canonical run repeats unchanged numerics after direct
  binding of two helper dependencies;156 CSV/Parquet files byte-identical to v3.
  Matrix196pass/5skip/2 unchanged existing Phase5 fixture failures;78 focused pass.
  Report: `build/lt-signed-composition-20260907/RAPPORT-COMPARATIF.md`.
  Next: qualify signed hourly composition with price-conditioned intraday,
  starting with the existing component and negative/near-zero regime checks.
  CH native15min truth is still missing for transfer accuracy. Existing hydro
  calibration UTC month edges remain a known limitation, not silently fixed.
  Future physical2030 assumptions remain separate; no auto model switch. No
  active process remains. A fresh window can resume from this handoff.
- D304 signed-shape integration and fixed CPU comparison are complete. Read
  `SESSION-HANDOFF-20260907-SIGNED-SHAPE-BENCHMARK.md` in Phase14 first.
  Existing PFCAssembler now accepts an explicit monthly-neutral signed hourly
  EUR/MWh input, with neutral ancillary layers and the shared final projection.
  No new adapter. The signed seasonal reference leads this local experiment:
  MAE20.345234/RMSE30.180010 versus current MLP22.817767/33.604667, gains10.836%/
  10.191%, four origin wins and improvement in all declared aggregate segments.
  Signed LightGBM worsens MAE4.594%; no automatic model replacement or promotion.
  Report: `build/lt-signed-benchmark-20260907/RAPPORT-COMPARATIF.md`.
  Twelve CPU fits,24 new candidate curves,6 cached native comparisons and6
  signed round-trip controls. Sixty prediction stages independently checked;
  all12 serialized models replay identically. Monthly residual max2.5182e-11.
  New tests44 pass; broader matrix161 pass/5skip/2pre-existing Phase5 golden
  fixture failures, reproduced with identical assertions on the pre-edit source.
  Source capture preserves75 D301 files; live assembler has a new version/hash,
  so old scientific/current-source pins must not be silently re-certified.
  Prior curve/benchmark artifacts, default numerical behavior and authorities
  remain unchanged. No Warehouse/GPU/AFRY/T057. No active process remains.
  Next: qualify the promising signed seasonal candidate with the components
  needed for the complete PFC; its hourly transport does not qualify intraday,
  water-value or uncertainty composition. Continue preparing the coherent2030
  assumptions from D303; no isolated battery kernel or complexity by default.
- D303 reviewed lot sequencing against the FMV PFC objective and executed a
  verified post-hoc diagnostic of saved D301 predictions. Read
  `SESSION-HANDOFF-20260907-PFC-PRIORITY-REVIEW.md` in Phase 14 first.
  D304 above executes its signed integration and bounded comparison, ahead of
  an isolated storage/flexibility kernel.
  Prepare public 2030 assumptions alongside that work; the physical milestone
  is a coherent chronology-to-prices-to-PFC demonstrator, not storage alone.
  Current MLP remains the global reference. Negative-truth origin-hours are
  3.209655% of the evaluated population and contribute25.091520% of its squared
  error; this identifies an experiment, not a proven clipping-causality gain.
  12 model/origin diagnostics,6 training origins and139 input hashes verified.
  Task root: `build/lt-priority-review-20260907/`; no product code change, fits,
  new forecasts, Warehouse or GPU. D300/D301/D302 bytes and authorities unchanged.
- D302 first structural-shaping lot implemented and executed. Read
  `SESSION-HANDOFF-20260907-STRUCTURAL-SHAPING-V1.md` in Phase 14.
  Five pinned public-source inventories audited separately for 2030–2035;
  partial/proxy/neutralized fields remain explicit. Signed hourly target
  arithmetic verified on 91 complete CH months, 66,455 native hours including
  1,032 negative-price hours; maximum monthly mean residual5.558e-14 EUR/MWh.
  135 tests pass/four optional CT skips. Report and matrices under
  `build/lt-structural-shaping-20260907/audit-v2/`; independent verification
  confirms unchanged D300 curve/prepared manifest and D301 plan/scores.
  No dispatch, new forecast, fit or signed-shape assembly integration yet.
  D303 above supersedes the immediate storage-kernel priority; actual future
  operation still needs reconciled trajectories and profiles.
  No AFRY values, Warehouse or GPU; all operational authorities remain false.
- Latest clarification: how to shape 2030+ with solar, batteries and flexible
  electrification. Read `SESSION-HANDOFF-20260907-DUCK-CURVE-2030-PROPOSAL.md`
  in Phase 14 after compact. Current local curve uses historical shapes and
  climatology; existing structural modulation is disabled and heuristic.
  Proposed priority: define structural trajectories and chronological energy
  operation before claiming a modern model solves far-horizon duck curves.
  Explanation/proposal only; no AFRY values consumed or model authority changed.
- Post-D301 recommendation, researched against publications dated through
  31 August 2026: read
  `SESSION-HANDOFF-20260907-SOTA-AUGUST-NEXT-STEP-PROPOSAL.md` in Phase 14.
  Proposed local v2 addresses signed shape, learned maturity and controlled
  weighting, then NBEATSx and bounded Chronos-2/TiRex-2 CPU qualification.
  This is a proposal, with no new execution or adopted model choice. D301
  assessment years are now development evidence, not a fresh independent test.
- D301 local CPU benchmark and comparative report are complete on retained D300 bytes.
  Read `SESSION-HANDOFF-20260907-LOCAL-CPU-BENCHMARK.md` in Phase 14 and
  `docs/model/LT-LOCAL-CPU-BENCHMARK.md`. No renewed permission is needed.
  Task root: `build/local-lt-benchmark-20260907/`. Frozen tuning years 2021–2022;
  assessment origins 2023–2026 with 36/32/20/8 supported months. Plan SHA-256
  `d0ea1b8fc21725e74da9cdf1234e4e473c3f877f0b9c39c8a1856028a74f56e1`.
  Report: task-root `rapport-comparatif.html` (plus Markdown/CSV/notebook evidence).
  Current hydro-corrected MLP leads overall: MAE22.817767 EUR/MWh; LightGBM
  22.915155 (+0.426809% error), reference23.113075, GAM25.024331, Ridge25.930839.
  Weighted MLP failed convergence on both tuning origins and is unrankable.
  LightGBM improves the common first12-month view on three origins by2.51%
  versus corrected MLP, but does not justify a global replacement. All models
  miss the combined2% MAE/RMSE policy target on the primary comparison.
  108 tests passed/four optional CT skips; 102 primary and12 secondary BASE-only
  predictions independently checked. Monthly residual max2.4102e-11 EUR/MWh.
  Revised-history/PIT, sparse PEAK, dependent origins and hourly-only scope
  remain explicit. Existing scientific v6 and D300 artifacts stay unchanged;
  all production/promotion/scientific/trading authorities remain false. No
  active benchmark process, new data acquisition, Warehouse or GPU execution.
- Authorization history: the user accepted proceeding with the proposed local
  CPU benchmark ("on fais la suite") and asked about session/compact choice.
  Resume implementation after `/compact`; no renewed calibration or local
  benchmark permission is needed. Read
  `SESSION-HANDOFF-20260907-BENCHMARK-NEXT-STEP-PROPOSAL.md` in Phase 14 for
  the proposed local development comparison, existing four challengers,
  corrected-versus-frozen incumbent distinction and remaining scientific
  confirmation boundary. D301 above implements that authorization; D300's
  source/model/curve artifacts remain unchanged.
- D300 completed the user-authorized local FMV PFC on CPU using fresh PRD
  exports. Main delivery: October 2026-December 2029, 114,052 quarter-hours,
  EEX quotes dated 4 September. Open
  `build/local-pfc-source-preflight-20260907/README-PFC-FMV.md` for CSV, Parquet,
  chart and review links. No new assembly adapter or production flag change.
  Solver monthly means are preserved within 8.072e-12 EUR/MWh; all 38 raw
  BASE/PEAK quotes reprice within 0.003224 EUR/MWh. The strict audit retains
  80 PASS and 9 QUOTE_CONFLICT, zero CRITICAL/UNSUPPORTED; all-gates-pass is
  false, with no signed hierarchy waiver. Production/promotion/scientific/
  trading authorities remain false. No independent predictive win is claimed.
- Complete source windows, native CPU fits, model/source hashes and exports
  are retained below the same task root. Fixes cover real ENTSO-E block replay,
  selected-window SFOE reconciliation, MLP Swiss hydro alignment and sparse
  intraday correction reload. The frozen incumbent MLP source remains unchanged;
  the explicit local hydro-corrected component has a separate identity.
  Final matrix: 217 passed, four optional CT dependency skips and one installed
  wheel qualification skip (exact audited wheel/root not supplied). Next work should
  target sparse intraday support, negative-price behaviour, age weighting and
  missing outages, with independent validation before claiming improvement.
- Acquisition is finished: one authorized Warehouse start, no SQL data writes,
  14 accepted statements (13 successful). One stop request returned 403;
  the final GET at 08:46:08.331Z confirms **STOPPED**. Do not restart or repeat
  extraction for this artifact. Reuse the retained source bytes and models.
- Read-only external-prerequisite audit on 7 September confirms no admission
  established for the real runner: zero countable origins, PRD EEX signed/time
  evidence pending, ENTSO-E July still an unsigned latest-candidate replay and
  the October 2026-September 2027 holdout locally frozen only. Existing capture
  and replay hashes match. Both read-only GitHub issue 4 calls returned 404,
  so later owner responses and current remote state remain unverified. Obtain
  the externally owned origin profile and hash-bound source attestations for
  read-only review for the scientific runner. D-20260907-298 records those
  external blockers; D299/D300 supersede its execution restriction only for
  the explicitly authorized local construction. T057 remains sealed.

## Earlier evidence and repository context

The checkpoints below retain their historical scope; D300 above is the latest
source, Warehouse and local-generation state.

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
- The remaining origin transport gap has been audited and deliberately stops
  at the operational authority boundary. Request/response signatures,
  operation replay/lookup, sanitized rejections and fresh HEAD challenges
  already cover the locally provable semantics. The unresolved identity,
  credential, endpoint, timeout/status, availability/SLO and remote durability
  properties require one externally selected service profile. No origin
  client, generic transport abstraction or duplicate error envelope was added;
  the separate snapshot-publication mTLS client is precedent only and must not
  be imported or copied into the pure LT origin surface.
- The first local EEX outage-plan evidence item is now closed without a query
  or value read. The existing D231 validator binds the exact historical
  three-table SQL bytes/hash to the exact 5 August manifest, ordered PRD
  tables, CH/POWER predicates, 12-column schema and opaque artifact
  declaration. Independent source time, signed envelopes and conversion to the
  existing signed EEX vintage catalogue remain missing, so the capture is not
  a model input and no later evaluation/scenario step is authorized.
- The next local ENTSO-E preparation item is now closed without a data query or
  invented selection. A canonical machine-readable request freezes the exact
  July 2026 `realized_final` window, fixed CH/FR/IT-North keys and the two
  admitted candidates for each of AT and DE-LU. It has not been transmitted,
  no owner response exists and all source-selection/model authorities remain
  false. The machine outage plan now reflects both this state and the completed
  D280 EEX query binding.
- After the user clarified the available `JulienFMV` GitHub and Databricks
  access, the July AT/DE-LU construction selection was resolved. Deployed
  producer code confirms sequences 1/2 are distinct A44 auction identities;
  one aggregate comparison on an already-running Warehouse shows sequence 1
  equals the independent LSEG EPEX actual curve on all 2,976 July quarter-hours
  for both markets, while sequence 2 differs materially. AT and DE-LU sequence
  1 are therefore frozen for the construction smoke export only. The statement
  returned no raw prices but read 9.36 GB, so it must not be repeated. Finality,
  causal, model and production authorities remain false.
- The exact July `realized_final` export is now prepared with CH/FR/IT-North
  fixed keys and AT/DE-LU sequence 1. A fresh metadata-only preflight used
  three control-plane GETs, opened no business rows and observed the PBI SQL
  Warehouse `STOPPED`. Unity Catalog reconfirmed `_year`/`_month` partitions
  but exposed no current byte/file statistics, so the hard scan bound remains
  unproven. No SQL ran and no Warehouse was started. The next platform action
  requires a hard scan bound or export quote, an already-running separately
  authorized Warehouse, explicit human acceptance of the ceiling and
  value-bound finality evidence; every model/production authority remains
  false.
- A continuation check found the Warehouse still `STOPPED`. The exact
  cost/export-contract request is now open as private producer issue
  `FMVSA/opendata-lakehouse#4`, created by `JulienFMV`. It asks only for a hard
  scan bound or platform quote, cost ceiling, output terms, assessment cutoff
  and value-bound finality coverage. It authorizes neither SQL nor Warehouse
  start. The issue is open with no assignee or response; export execution is
  waiting on that external evidence and explicit cost acceptance.
- After the user explicitly authorized use of the PBI Warehouse for the
  freshness/cost check, the Warehouse was already `STARTING`; this client
  issued no start. Two value-blind statements established that the Silver
  table was modified on 4 September and that the bounded scan read only
  7,798,287 bytes after partition pruning. AT, DE-LU, FR and IT-North extend
  through 4 September delivery, but CH stops at 2 September, consistent with
  the reported ENTSO-E outage. A stop request was rejected `403`; it was not
  retried, the Warehouse remained `RUNNING` with no active sessions and a
  45-minute auto-stop, and issue 4 now records the gap and lifecycle result.
  No prices or July export rows were returned and every model/production
  authority remains false.
- The user then authorized advancing July independently of the September
  outage recovery. A value-blind normalized check proved exact coverage of
  2,976 quarter-hours for each frozen SeriesKey with zero gaps, overlaps or
  invalid native intervals. The exact v2 statement produced one quarantined
  local candidate below `build/`: 12,083 native rows, 564,727 Parquet bytes,
  semantic SHA-256
  `5a6d72b5531e843dc4e7920c519cc3d93189e70d91277662a763f6575967374e`.
  Local validation passes every raw contract gate. A follow-up audit of the
  producer contract found latest-revision/correction semantics but no signed
  finality service, so missing finality now blocks only promotion to
  `realized_final`, not local work. The new authority-negative
  `realized_latest_candidate` lane replayed all 12,083 rows exactly from the
  existing Parquet with no Warehouse use. Replay build ID is
  `b00f2b725c66d91e7b8ec681b578fd080be77837a6e75f9b1de675caade20a26`.
  `is_final`, consumer, model and production authorities remain false. Issue 4
  was corrected accordingly; it still tracks optional finality promotion and
  the separate September CH recovery.
- The separately authorized July LSEG reconciliation is now complete under a
  pre-registered half-cent policy. A metadata preflight rejected the
  unpartitioned 11.54 GB vintage table without querying it and selected the
  16.8 MB latest Silver table under a 32 MiB ceiling. The successful bounded
  statement read 13,141,107 bytes and returned 9,672 native rows; PBI was
  already running, so no Warehouse start, resize or creation occurred. CH, AT
  sequence 1, DE-LU sequence 1 and FR each match ENTSO-E exactly for all 744
  July hours, with no gaps and zero difference. IT-North remains ENTSO-E-only.
  This validates latest-source consistency, not finality, causal availability
  or model authority; all model and production authorities remain false.
- The benchmark-roadmap ambiguity is now closed without changing the exact
  five-model selection inventory. The transparent market-constrained seasonal
  model is a separate permanent primary promotion reference, scored on the
  same frozen population before candidate ranking, with no tuning or selection
  eligibility. A small authority-negative companion contract freezes this
  placement and records the available GPU without changing runtime authority:
  seasonal reference and canonical LightGBM remain CPU deterministic, while
  GPU fit/model selection still requires explicit CPU/GPU parity qualification.
  No data, model fit, GPU or remote execution occurred.
- The next benchmark boundary is also closed without guessing the unresolved
  feature inventory. A pure in-memory adapter now validates already
  materialized PRD training/prediction frames against one exact frozen origin,
  requires the exact frozen feature order, applies one deterministic
  complete-case mask per split, rejects future availability and future truth,
  and emits detached read-only arrays plus hash-only source/value identities.
  It performs no I/O, fit, scoring or GPU work and grants no authority.
- That feature decision is now closed for the hourly candidate lane. The exact
  common matrix is the incumbent's nine non-disabled inputs: seven deterministic
  calendar encodings, origin-safe hydro fill and years ahead. The three outage
  positions remain explicit incumbent compatibility constants because outages
  are disabled; they cannot be populated by missing-value zero fill. Raw PRD
  ENTSO-E actuals are excluded from future inputs. Their causal climatology
  derivatives remain shared quarter-hour context, not challenger-only
  predictors. The PRD adapter rejects every alternate inventory. No data,
  model fit, truth, Warehouse or GPU was opened.
- The corresponding pure feature constructor is now implemented. It consumes
  only already-materialized delivery, hydro-availability and normalized
  hydro-fill columns; verifies the exact frozen origin, split, role and
  climatology cutoff; derives Swiss-local calendar encodings and maturity by
  calling the incumbent encoder; preserves missing hydro; and emits detached
  read-only values with hashes. Exact input columns reject raw ENTSO-E actuals
  and outage fields. Hydro percentages are not guessed: only fraction `[0,1]`
  is accepted. No connector, fit, truth, Warehouse or GPU path exists.
- Constructed feature batches can now be joined safely to observation metadata.
  The assembly requires exact one-to-one delivery timestamp populations,
  aligns by timestamp rather than row position, computes historical row
  availability as the maximum of target and hydro availability, and reverifies
  batch shape, units, hashes, immutability and negative authority before
  delegating to the existing common PRD validator. It does not duplicate
  feature construction or masking and still exposes no execution path.
- The candidate execution interface is now frozen without conflating numerical
  spaces. The source-bound incumbent keeps its native fit/apply path; the four
  challengers use the common matrix but must learn the same daily-normalized
  hourly `f_H`, never raw EUR/MWh. The input adapter now requires
  `target_f_h`. Candidate factors receive equivalent post-processing and common
  full-curve assembly before the existing EUR/MWh scorer. The contract is
  metadata-only; no model, Warehouse, GPU or truth was opened.
- The incumbent-equivalent learning target is now executable as a separate
  pure transform. It converts already-materialized direct CH quarter-hour
  prices into hash-bound hourly `target_f_h` rows using the exact native
  daily-mean, eligibility, clipping, within-hour weighting and DST grouping
  rules. Synthetic capture tests prove target and feature parity with the
  hash-bound incumbent preprocessing, and the output integrates directly with
  the one common input assembly. No estimator or real data was opened.
- Raw challenger predictions now have a separate pure factor postprocessor
  that exactly matches the incumbent's native floor, Swiss-local daily
  normalization and final clipping. It accepts only the four challenger IDs
  and rejects the incumbent to prevent double application. A fixed-predictor
  comparison with native `ShapeHourlyMLP.apply` passes exactly; outputs remain
  authority-negative `f_H`, not EUR/MWh scoring curves.
- The common full-price assembly seam is now implemented without duplicating
  the production formula. One pure in-memory adapter shallow-copies a
  solver-authority `PFCAssembler`: the incumbent copy calls the exact native
  `ShapeHourlyMLP.apply`, while challenger copies replace only its returned
  `f_H`. All five reuse the incumbent `f_W`, monthly solver levels, frozen
  quarter-hour context, intraday/water-value layers, horizon damping, bridge,
  monthly recentering and final product projection. The adapter rejects
  unfrozen optional layers and emits immutable EUR/MWh vectors accepted by the
  existing scorer input. Synthetic parity, BASE/PEAK and authority tests pass;
  no real data, fit, truth, Warehouse or GPU was opened.

## Invariants

- The CH monthly BASE solver is the sole monthly-level authority.
- ENTSO-E, neighboring markets, history, weather, Swissgrid, AFRY and LSEG may
  shape or benchmark only; they cannot rewrite monthly solver means.
- LT code must not import `pfc_shaping.ct.*`.
- T057 remains sealed.
- Model admission remains
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS` until independently
  governed local exports and a new future holdout exist.

See durable decisions D-20260821-248 through D-20260907-299 and
`SESSION-HANDOFF-20260907-LOCAL-PFC-PRD-INPUT-READINESS.md` for the
latest exact implementation files, hashes, tests and residual risks. The PFC
target, source-acquisition, scenario-product, ENTSO-E export,
normalized-interval and preceding repo-hygiene handoffs retain their exact
historical evidence and cleanup counts.
