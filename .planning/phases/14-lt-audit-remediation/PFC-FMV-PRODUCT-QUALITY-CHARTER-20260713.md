# PFC FMV Product Quality Charter

Date: 2026-07-13

Status: normative target for Phase 14 and subsequent industrialisation work.

## Product objective

Build the best defensible Swiss power forward curve available to FMV: a
point-in-time, arbitrage-consistent and operationally reproducible hourly / 15
minute curve whose market levels come from eligible CH EEX products and whose
shape is supported by out-of-sample evidence. OMPEX remains an external
advisory benchmark and is forbidden from model fitting, candidate selection,
backtest truth and promotion gates.

No statistical score can compensate for a failed market, provenance or
promotion invariant.

## Tier 0 - Non-negotiable promotion invariants

1. One production entrypoint executes the Phase 14 LT path. Diagnostics,
   challengers and healthchecks cannot publish production artifacts.
2. The final delivered curve, after every bridge and intraday layer, reprices
   every supported hard CH BASE and PEAK product within `1e-6 EUR/MWh` and
   satisfies the implied OFFPEAK identity within `1e-6 EUR/MWh`.
3. `QUOTE_CONFLICT` is accepted only through an explicit production-approved
   source hierarchy policy bound to the exact input CSV, forward snapshot and
   conflict identity hashes.
4. When `monthly_level_authority="solver"`, no downstream layer may alter the
   solver monthly BASE mean. No individual month patch is allowed.
5. Solver constraint and stationarity residuals must be finite and within the
   configured tolerances. Numerical fallback is never an implicit pass.
6. Every production input is point-in-time: `available_at <= as_of`, source and
   revision are known, and the immutable source snapshot is hash-bound to the
   run manifest. Spot-derived proxy prices can never become hard forward
   constraints.
7. Candidate outputs are written to an isolated run directory. Publication is
   an atomic promotion after all gates pass; failed candidates remain
   quarantined and cannot be selected by dashboards or downstream jobs.
8. Promotion evidence is independent and mutually consistent: production
   manifest, export manifest, selected-configuration artifact, delivered
   product audit and final artifact hashes.

## Tier 1 - Data quality service levels

Freshness is evaluated against the expected publication calendar, not file
modification time. A missing observation is distinct from an economic zero.

| Data family | Production requirement |
| --- | --- |
| EEX forwards | Expected latest business snapshot; finite CH BASE/PEAK coverage by required horizon; explicit source and conflict register |
| EPEX spot | Native hourly coverage; latest complete delivery day no older than 48 h; no unbounded forward fill |
| ENTSO-E fundamentals | Feature-level freshness and coverage; critical features no older than 72 h; revisions retained |
| Swiss hydro | Latest weekly publication no older than 10 days; publication and acquisition timestamps retained |
| Outages | Successful acquisition proven; latest expected horizon covered; API failure cannot be represented as zero outage |

Every canonical observation must be traceable to `event_time`, `published_at`,
`available_at`, `fetched_at`, `source`, `source_document_id`, `revision`,
`row_hash`, `ingestion_run_id` and `schema_version`, or carry a documented
source-specific reason why a field is unavailable.

## Tier 2 - Deterministic model quality

Evaluation uses frozen rolling-origin vintages and a pre-registered future
holdout. The primary benchmark is a transparent market-constrained seasonal
model; OMPEX is reported separately and remains advisory.

Promotion targets:

- at least 2% improvement in weighted MAE and RMSE versus the frozen seasonal
  baseline on the aggregate eligible population;
- positive MAE improvement in at least 70% of eligible folds;
- no critical regime (winter peak, summer solar, night, weekend, ramp,
  negative-price regime) degrades by more than 2% without an explicit economic
  justification and product-owner waiver;
- exact quote repricing and monthly conservation in every fold;
- day-over-day changes attributable to changed quotes, fundamentals or a
  versioned model/config change;
- simple seasonal/GAM/tree baselines retained as permanent challengers so a
  neural model is promoted only when it proves incremental value.

These thresholds are product policy, not claims that the literature provides
universal numerical cutoffs. They are deliberately demanding FMV acceptance
criteria built on rolling-origin and benchmark discipline.

## Tier 3 - Probabilistic and hydro decision quality

Marginal historical bands are not sufficient for production approval.
Uncertainty must be calibrated from genuine out-of-sample forecast errors and
must support temporally coherent scenarios.

- report pinball loss, CRPS or WIS, empirical coverage, interval width and PIT
  diagnostics by horizon and regime;
- target at least 2% improvement in WIS/CRPS versus the frozen probabilistic
  baseline;
- 80% and 90% interval coverage must be within 3 percentage points of nominal
  coverage on sufficiently populated evaluation buckets;
- scenarios must preserve market-product identities and model dependence
  across hours, days, hydro state, load, renewables and neighbouring markets;
- final model choice must include an economic hydro dispatch / hedge backtest,
  not only price-error metrics.

## Tier 4 - IT and operational acceptance

- one installable package and one documented CLI with explicit `as_of`,
  `run_id`, config and input snapshot references;
- deterministic rerun from the same image digest, config and snapshots yields
  identical final artifact hashes;
- locked dependencies, multi-stage non-root container, no embedded credentials
  or workstation paths, and read-only inputs;
- CI runs unit, integration, import-isolation, data-contract, repricing and
  end-to-end smoke tests;
- structured logs, durable failure manifest, heartbeat, runtime/data-freshness
  metrics and alerting;
- atomic promotion, immutable run history, retention policy and tested rollback;
- IT runbook covers scheduling, secrets, recovery, support ownership and
  promotion authority.

## Scientific basis and limitations

- Benth shaping: market-consistent forward averages plus a smooth correction
  around an explicit seasonal component; bid/ask information should be used
  when available.
- Caro, Swiss empirical methods: Swiss hydro, neighbouring DE/FR markets,
  calendar and weather are relevant shape drivers, subject to point-in-time
  validation.
- Stochastic hydro hedging literature: incomplete power markets require
  coherent scenarios for decision support; a deterministic curve with
  marginal bands is not enough.
- Energies 2024 forecasting review: benchmark diversity, rolling validation,
  ensembles and interpretability are expected, but this general review is not
  itself evidence that an HPFC is valid for Switzerland.

The four supplied papers guide architecture and evaluation. None excuses stale
data, look-ahead bias, weak provenance or a non-reproducible production path.

## Current release decision

`NO_GO_PRODUCTION_AND_IT_HANDOFF` until every Tier 0 invariant passes on a fresh
candidate and the locked future holdout supplies eligible evidence. Docker work
starts only after the canonical pipeline and data contracts are stable.
