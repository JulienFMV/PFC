# LT Quant Shaping Redesign Plan - CH Cross-Border PFC - 2026-06-18

Status: expert-audited redesign plan candidate  
Branch context: `fix/lt-audit-remediation` at local HEAD `559bf3f` before this planning phase  
Scope: long-term CH electricity HFC/PFC shaping with cross-border DE/FR/AT/IT_NORD priors, EEX calibration, monthly/hourly consistency, probabilistic tail diagnostics, Power BI QA  
Production status: `NO GO`

## 1. Executive Decision

The current residual-anchor candidate must not be promoted as a production PFC.

It is a diagnostic artifact that proved two things:

1. EEX bucket repricing can be preserved while changing the residual monthly shape.
2. Local rules and visual gates can still miss economically wrong month paths.

It did not prove that the generated CH curve is a robust forward curve. The next step is not another local correction. The next step is a quant redesign: formulate the LT curve construction as a single constrained optimization problem, with market quotes as hard constraints and shape priors learned from spot history, cross-border markets, structural drivers, and analyst-approved governance.

## 2. Scientific Anchors

The target design follows the mainstream HPFC/PFC literature rather than ad hoc month edits.

Key references:

1. Fleten and Lemming, 2003, "Constructing forward price curves in electricity markets"  
   Relevance: forward curves are high-resolution approximations inferred from sparse traded contracts and should combine quoted market information with bottom-up/structural shape information.

2. Kiesel, Paraschiv, Saethero, 2019, "On the construction of hourly price forward curves for electricity prices"  
   Relevance: HPFC construction should jointly account for observed futures prices and historical spot seasonality in one optimization procedure, not treat seasonality as an untouchable exogenous shape.

3. Caldana, Fusai, Roncoroni, 2017, "Electricity Forward Curves with Thin Granularity"  
   Relevance: hourly EPEX/EEX forward curves should be jointly consistent with BASE/PEAK futures and historical day-ahead patterns, with smoothness and stability constraints.

4. Keles et al., 2020, "Cross-border Effects in Interconnected Electricity Markets - An Analysis of the Swiss Electricity Prices"  
   Relevance: Switzerland is not DE-only. CH prices correlate strongly with DE in summer, tend to follow FR in winter, and are affected by French/Italian load and German renewable generation.

5. Nowotarski and Weron, 2018, "Recent advances in electricity price forecasting"  
   Relevance: probabilistic validation must use proper scoring and benchmark discipline, not only point-error comparisons.

6. Gneiting and Raftery, 2007, "Strictly Proper Scoring Rules, Prediction, and Estimation"  
   Relevance: probabilistic tails and scenario fans must be validated with proper scoring rules, calibration, and sharpness.

7. Ziel and Steinert, 2018, probabilistic mid/long-term electricity price forecasting literature  
   Relevance: medium/long horizon distributional EPF should separate market-consistent forward levels from physical spot uncertainty.

8. Yu et al., 2026, "PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting"  
   Relevance: modern European electricity price forecasting explicitly models spatial/cross-border structure, topology, and exogenous load/solar/wind drivers. This can inform benchmark design and feature checks. It must not directly produce production `p_t` because it is not an arbitrage-free LT PFC construction method.

9. Eurelectric, 2024, "Understanding ultra-low and negative power prices"  
   Relevance: negative prices are a structural market feature under high renewables, low demand, and limited flexibility. They should be modeled as a bounded structural tail, not suppressed by positivity floors or allowed by manual exceptions.

Source hygiene:

- Methodological references are binding for model design only where they map to explicit equations, tests, or acceptance gates below.
- Recent ML/foundation-model references are benchmark and feature-design references only.
- All external references used in an implementation PR must be recorded with URL/DOI, version where applicable, and access date.

## 3. Hard Principles

1. No-arbitrage comes first. Every quoted EEX BASE and PEAK product used in the run must reprice exactly within tolerance.
2. The curve is not a spot forecast. It is a forward curve constrained by traded delivery products, with spot history used as a shape prior.
3. CH shape cannot be copied from DE. Neighbor markets may inform shape only through deviations, spreads, covariates, or learned weights.
4. Monthly, weekly, weekend, holiday, peak/offpeak, and hourly shapes are one problem. They must not be patched independently after the fact.
5. Directly quoted EEX products are hard market signals. Smoothing may operate in the null space of quoted constraints only.
6. Negative prices are allowed only through a calibrated structural mechanism and distributional checks; bounds may come from market rules and approved stress cases, not only historical quantiles.
7. Every threshold must be either mathematically implied, historically calibrated, or explicitly approved by the desk.
8. A curve that fails visual or analyst sanity checks is not production-ready even if EEX residuals are zero.
9. Power BI QA must display the same curve and diagnostics that were generated by the audited run.
10. Local/test overlays remain disabled by default until they pass a full run, backtest, audit, and governance path.

## 4. Target Mathematical Formulation

### 4.1 Deterministic Market Curve

Let `t` index canonical UTC delivery intervals `[start_utc, end_utc)` with duration weight `Delta_t` in hours. Local CH/neighbor timezones are derived views used only for product masks, peak/offpeak calendars, holidays, and reporting. Duplicate local hours in October must never be collapsed.

Let `p_t` be the deterministic CH LT forward curve, interpreted as the market-consistent conditional mean/central curve used for valuation.

The v1 production optimizer must be a convex equality-constrained QP:

```text
min_p
    1/2 p' H p + f' p

equivalently:

    lambda_prior       * ||p - p_prior||^2_W
  + lambda_smooth_m    * ||D2_month mean(p)||^2
  + lambda_smooth_h    * ||D2_time p||^2
  + lambda_seam        * ||S p - s_market||^2
  + lambda_calendar    * ||C p - c_prior||^2

subject to
    A p = q
    quoted products are not overridden by synthetic smoothing
```

Where:

- `H` must be positive semidefinite plus a documented ridge/tie-breaker sufficient for a unique solution in the feasible null space.
- `A` is the canonical hard-constraint matrix for accepted EEX CH products after quote hierarchy, feasibility, and BASE/PEAK/OFFPEAK transformation.
- `q` is the hard target vector for accepted products.
- `p_prior` is an interpretable prior built from CH spot history, CH forward history, neighbor-market shape, and structural drivers.
- `S p` extracts cross-bucket month-to-month seams such as Q1 to residual.
- `C p` extracts calendar features: weekend/weekday, holiday/non-holiday, peak/offpeak, block prices, week-to-week deltas.

Implementation may solve directly in `p`, but the audit formulation must also be expressible as:

```text
p = p0 + N z
A p0 = q
A N = 0
min_z 1/2 (p0 + N z)' H (p0 + N z) + f' (p0 + N z)
```

This makes explicit that priors, smoothing, seams, and calendar penalties operate in the null space of quoted EEX constraints.

### 4.2 Hard Constraint Feasibility

Before any QP solve, the accepted hard quote set must pass:

```text
rank(A) == rank([A | q])
or
||(I - A A+) q||_inf <= 1e-9
```

Rules:

- consistent redundant quotes may be reduced to an independent basis for solving, but every original quote must be repriced and reported after the solve;
- inconsistent overlapping quotes must fail hard before optimization;
- any quote excluded by hierarchy, staleness, missing coverage, or inconsistency must be listed in `excluded_quote_set.csv` and the run manifest;
- pseudo-inverse least-squares fitting must never silently transform hard quotes into soft quotes.

### 4.3 Partial Horizon Policy

An EEX product may be accepted only if one of the following holds:

1. the optimization universe contains the full delivery interval of the product; or
2. the elapsed part has auditable realized/locked energy and the remaining target is computed as:

```text
q_remaining = (q_full * H_full - E_elapsed) / H_remaining
```

If neither condition holds, the product is excluded and the run fails if the remaining quote set no longer gives enough no-arbitrage coverage.

### 4.4 BASE/PEAK/OFFPEAK Convention

BASE and PEAK must use one canonical disjoint formulation.

For every product where both BASE and PEAK are accepted as hard quotes:

```text
PeakMean    = q_peak
OffPeakMean = (q_base * H_base - q_peak * H_peak) / H_offpeak
```

The solver constrains disjoint Peak and OffPeak rows. It then audits the original BASE and PEAK quotes. Synthetic PEAK priors may influence `p_prior` or soft penalties, but must never become hard quotes unless they are actual EEX/desk-approved hard inputs.

Peak calendar source of truth:

- before production, desk/legal must confirm whether EEX PEAK excludes national holidays for each market/product used;
- this convention must be versioned in `calendar_version` and tested for CH, DE-LU, FR, AT, and IT_NORD/IT proxy;
- any unresolved holiday/peak ambiguity is a production `NO GO`.

### 4.5 KKT Diagnostics

Each optimizer run must output:

- primal residuals for every hard EEX constraint;
- stationarity residual `||H p + f + A' lambda||_inf`;
- rank, condition number, scaling report, and solver status;
- dual/shadow values for constraints, especially if a quote forces a visually sharp seam;
- decomposition of each monthly mean into EEX level, prior shape, cross-border contribution, and smoothing correction;
- no hidden post-calibration patch that changes curve shape without audit rows.

### 4.6 Probabilistic Layer Is Separate

The deterministic curve `p_t` is not the full physical distribution of future spot prices.

If scenarios or fan quantiles are generated, define either:

```text
X_s,t = p_t + epsilon_s,t
```

or non-crossing marginal quantiles:

```text
Q_alpha,t, alpha in {0.05, 0.10, 0.50, 0.90, 0.95}
```

Rules:

- only the scenario mean or conditional expectation must be market-consistent with EEX constraints: `E_s[A_b X_s] = q_b`;
- individual quantiles do not have to reprice forwards;
- block distributions must be computed from coherent scenario paths `A_block X_s`, not by summing hourly marginal quantiles;
- non-crossing quantiles are required where quantiles are reported;
- probabilistic validation is separate from deterministic EEX repricing.

## 5. Prior Model

The prior is not a single DE shape. It is a governed ensemble:

```text
p_prior = level_CH + shape_CH_hist + basis_cross_border + structural_future + residual_regime
```

### 5.1 CH Historical Shape Prior

Use realized CH day-ahead/EPEX history to estimate robust additive deviations:

- month within quarter/year;
- hour within month;
- peak/offpeak;
- weekday/weekend/holiday;
- solar belly hours;
- evening ramp;
- winter morning/evening premium;
- hydro/water-value regime with vintage metadata;
- negative-price tail regimes.

Recommended estimators:

- robust M-estimation or Huber/quantile regression for shape cells;
- shrinkage across adjacent months and years;
- regime-aware weighting, with more weight on recent high-renewable years but not a single-year overfit.

### 5.2 Cross-Border Prior

Use DE, FR, AT, and IT_NORD as shape signals, not absolute levels. Use `IT` only as a documented proxy if no IT_NORD source exists; such proxy use requires a haircut/QA warning and cannot be silent.

Minimum model:

```text
shape_cross_border_CH(m,h)
  = w_DE(r,m,h)      * dev_DE(m,h)
  + w_FR(r,m,h)      * dev_FR(m,h)
  + w_AT(r,m,h)      * dev_AT(m,h)
  + w_IT_NORD(r,m,h) * dev_IT_NORD(m,h)
```

Neighbor deviations must be defined as projections orthogonal to CH hard level constraints:

```text
dev_M = P_null_CH_bucket(neighbor_shape_M)
A_CH_bucket dev_M = 0
```

For each relevant CH constrained bucket, load type, and time aggregation, the neighbor deviation must have weighted mean zero. Adding a constant `+50 EUR/MWh` to every DE/FR/AT/IT_NORD input curve must not change the CH PFC, unless a separate explicit spread-level component is approved, versioned, and reported.

Where weights depend on:

- season/month;
- observed historical CH-neighbor correlation;
- spread stability;
- CH-DE / CH-FR / CH-AT / CH-IT_NORD interconnector or NTC regime;
- import/export direction and congestion regime;
- load/temperature regime;
- renewable generation regime;
- product liquidity and freshness.

Minimum governance rule:

- summer CH may lean more on DE/AT renewable-driven shape;
- winter CH must explicitly test FR load-driven influence;
- IT_NORD must be a cap/spread sanity check for southern scarcity regimes;
- if a market lacks reliable month/hour shape data, it receives zero production weight and a QA warning.

Production requires a `cross_border_regime_dataset` covering every CH border through observed data or a signed fallback bridge:

- NTC import/export or available transfer capacity;
- realized/commercial flows, or explicit unavailable-source marker with fallback bridge;
- spot and forward spreads by hour/block;
- congestion indicator;
- scarcity, high-renewable, winter-stress, and low-demand regimes;
- data vintage and availability timestamp.

Without this dataset, or a desk-approved fallback bridge with expiry date, cross-border production approval is `NO GO`.

### 5.3 Forward-History Prior

Use historical CH/DE/FR/AT/IT or IT_NORD-proxy EEX forward snapshots to calibrate:

- month-vs-quarter deviations;
- residual-bucket completions after partial quote sets;
- seasonal spreads;
- Cal/Q/M consistency;
- stability of month-to-month seams over rolling snapshots.

The file `data/eex_forwards_history.parquet` and desk Excel histories are essential inputs, but must be treated as versioned data with snapshot dates and freshness checks.

### 5.4 Structural Future Prior

The LT horizon cannot rely only on historical spot shapes because solar, batteries, electrification, nuclear availability, hydro, and interconnectors evolve.

Use structural covariates already present or planned in the repo:

- CH/DE/FR/AT/IT_NORD or IT-proxy solar and wind capacity or generation trajectories;
- load/electrification scenarios;
- storage/flexibility indicators;
- NTC/interconnection assumptions;
- hydro/water-value proxy;
- carbon/fuel regime with source/fallback marker.

These should alter shape through interpretable coefficients, not opaque final price overrides.

Swiss hydro must be modeled explicitly because it is a first-order CH shape driver. The minimum hydro regime feature set is:

- reservoir level and anomaly versus climatology;
- run-of-river proxy;
- pumping/storage proxy with source/fallback marker;
- snowmelt/refill season;
- drought stress indicator;
- water-value seasonal pressure;
- publication lag and `available_at` timestamp.

Missing hydro data is a production `NO GO` unless the desk approves a dated fallback prior.

## 6. Implementation Architecture

Create a new LT module. Do not continue expanding `scripts/export_local_test_ch_hourly_csv.py`.

Proposed files:

```text
pfc_shaping/lt/model/quant_shape_optimizer.py
pfc_shaping/lt/model/shape_priors.py
pfc_shaping/lt/model/cross_border_shape.py
pfc_shaping/lt/model/negative_price_regime.py
pfc_shaping/lt/model/shape_constraints.py
pfc_shaping/validation/lt_quant_shape_backtest.py
scripts/run_lt_quant_shape_candidate.py
scripts/audit_lt_quant_shape_candidate.py
```

Existing reusable pieces:

- `pfc_shaping.calibration.eex_contract_selection.calibration_buckets`
- `pfc_shaping.calibration.arbitrage_free.ArbitrageFreeCalibrator`
- `pfc_shaping.lt.model.quote_aware_monthly_smoothing`
- `pfc_shaping.lt.model.water_value`
- `pfc_shaping.lt.model.solar_modulation`
- `pfc_shaping.lt.model.electrification_shape`
- existing Power BI refresh scripts, after semantic-model binding is fixed.

New data contracts:

```text
data/lt_quant_shape_training.parquet
data/cross_border_regime_dataset.parquet
data/lt_quant_data_coverage_matrix.csv
data/lt_quant_point_in_time_feature_catalog.csv
```

Non-goal:

- do not put production logic into local audit/export scripts;
- do not import `pfc_shaping.ct.*` into LT;
- do not make Power BI the source of truth.

## 7. Work Packages

### WP0 - Stop-the-Line Governance

Goal: prevent accidental promotion of the current candidate.

Actions:

- mark `output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv` as diagnostic only;
- record that the current seasonal audit with improved seam gate produces critical flags;
- do not refresh Power BI for production using this candidate;
- preserve current artifacts for reproducibility.

Exit criteria:

- planning note states `NO GO`;
- external auditor cannot misread the current candidate as production-approved.

### WP1 - Contract Matrix and Constraint Engine

Goal: one canonical contract aggregation engine for BASE/PEAK/month/quarter/cal.

Actions:

- build sparse aggregation matrices for all active CH EEX products;
- support partial horizons and DST exactly;
- support exact EEX peak definition after desk/legal confirmation on holidays;
- expose product residual table and rank diagnostics;
- fail on inconsistent or stale quote sets.

Tests:

- Cal + Q1 creates residual bucket with exact weighted residual target;
- quoted month overrides quarter/cal completion;
- DST March/October products preserve exact hour-weighted means;
- PEAK products preserve BASE means through the canonical disjoint Peak/OffPeak transformation;
- hard quote sets pass the rank/infeasibility test before solve.

Acceptance:

- max hard residual `< 1e-9` in synthetic tests;
- max production hard residual `< 1e-6 EUR/MWh` after CSV/report rounding and `< 1e-9 EUR/MWh` in internal solver state;
- no hidden fallback to raw curve on failed calibration.

### WP2 - Prior Dataset Builder

Goal: construct a reproducible training dataset for shape priors.

Inputs:

- CH spot/hourly or 15-minute history;
- EEX forward snapshot history for CH/DE/FR/AT/IT_NORD or IT proxy;
- neighboring spot histories for DE/FR/AT/IT_NORD, or explicit zero-weight justification;
- load, solar, wind, hydro, temperature, NTC/interconnector data with source/fallback markers;
- public scenario features from Phase 13.

Actions:

- create `data/lt_quant_shape_training.parquet`;
- include as-of timestamps and publication dates;
- enforce no leakage: no future realized spot data beyond valuation date;
- create feature completeness and freshness report.

Point-in-time contract:

Every feature row must carry:

```text
observation_start
observation_end
publication_timestamp
ingestion_timestamp
revision_timestamp
available_at
source_system
source_file_or_query
source_hash
market
timezone
```

The training and run builders must enforce:

```text
available_at <= valuation_timestamp
```

Revised public/statistical series are allowed only if historical vintages are available. Otherwise they must be labeled `latest_revision_only` and excluded from as-of backtests unless a validator approves their use.

Minimum production data coverage:

| Data block | Minimum requirement | Failure mode |
|---|---:|---|
| CH spot history | hourly or finer, >= 5 years, point-in-time safe | `NO GO` |
| CH EEX forwards | latest audited snapshot and historical snapshots | `NO GO` |
| DE/FR/AT/IT_NORD spot/spread | hourly or finer, >= 3 years, or explicit zero-weight by market | `NO GO` if unexplained |
| Multi-market EEX forwards | CH/DE/FR/AT and IT/IT_NORD proxy with freshness metadata | `NO GO` if CH/DE missing; zero-weight warning for missing secondary |
| CH hydro | reservoir/water-value proxy with vintage | `NO GO` unless fallback signed |
| NTC/interconnection | border-level NTC/flow/congestion or desk-approved bridge | `NO GO` |
| Structural covariates | solar/load/wind/storage scenario vintages | `CONDITIONAL` with scenario warning |

Tests:

- as-of split test;
- no duplicate timestamps;
- no future data leakage;
- adversarial future-feature injection must be detected;
- market/timezone boundary tests for CH/DE/FR/AT/IT_NORD.

Acceptance:

- dataset can be rebuilt from documented sources;
- every feature has source, vintage, and freshness metadata.

### WP3 - Interpretable Prior Model

Goal: produce `p_prior` with explainable components.

Initial model:

- robust additive month/hour/calendar prior from CH spot history;
- cross-border deviation prior from DE/FR/AT/IT_NORD or documented IT proxy;
- structural adjustment from solar/load/storage scenario features;
- optional regime clustering for high-renewable and winter-stress regimes.

Do not start with a black-box foundation model in production. Use recent foundation-model literature to justify spatial/covariate features and as a benchmark, not as the first governed production engine.

Tests:

- CH historical monthly residual shapes are reproduced out of sample;
- DE-only shape is rejected when FR/IT_NORD/AT evidence contradicts it;
- summer and winter weights differ in the expected direction;
- priors remain finite and bounded under missing neighbor inputs;
- adding a constant `+50 EUR/MWh` to every neighbor input curve leaves CH output unchanged to `atol=1e-12` unless an approved spread-level term is explicitly enabled.

Acceptance:

- prior decomposition table per month/year;
- no absolute neighbor level leakage;
- all weights sum to one where required or have documented shrinkage mass to CH history.

### WP4 - Global Constrained Optimizer

Goal: solve the deterministic final PFC in one pass.

Implementation:

- sparse quadratic programming or equality-constrained least squares;
- hard constraints for EEX quotes;
- soft penalties for prior distance, month path smoothness, seams, and calendar coherence;
- optional inequality constraints for regime-aware floor/ceiling, but no universal positivity floor.

Recommended first implementation:

- start with scipy sparse KKT/equality-constrained QP already similar to existing `ArbitrageFreeCalibrator`;
- keep all penalties convex in v1;
- log objective decomposition and constraint residuals;
- do not include non-convex tail/fan quantile penalties in v1 deterministic `p_t`; use the separate probabilistic layer in WP5.

Tests:

- flag-OFF byte identity against current production path;
- optimizer ON exact repricing;
- synthetic known-solution test;
- directly quoted products unchanged by smoothing;
- residual bucket mean preservation under unequal hour counts;
- no neighbor absolute level leakage;
- deterministic reproducibility at `atol=1e-12` for fixed inputs;
- inconsistent overlapping quotes fail before solve;
- redundant consistent quotes pass and all original quotes are audited;
- partial month/quarter/cal products are rejected or converted to auditable remaining targets;
- UTC duplicate local hour in October is not collapsed;
- KKT residual report is emitted and within threshold.

Acceptance:

- full CLI run completes within the production runtime gate (`<= 30 min` unless a batch SLA is documented);
- all hard constraints pass;
- objective/audit rows explain every material monthly correction.

### WP5 - Negative Price Regime

Goal: replace manual negative-price allowlists with a calibrated structural probabilistic regime.

Deterministic `p_t` and the physical distribution must remain separate. Negative-price diagnostics may influence soft priors and scenario generation, but hard EEX repricing applies to the deterministic expectation/scenario mean.

Scenario model:

- estimate probability and severity of negative hours conditional on month, hour, weekday, solar/load/wind, neighbor regimes, and historical volatility;
- generate non-crossing marginal quantiles plus coherent scenario paths `X_s,t`;
- impose `E_s[A_b X_s] = q_b` for every hard EEX product if scenarios are used for valuation;
- compute block distributions from scenario paths, not from summed hourly quantiles.

Tests:

- negative hours localize in historically and structurally plausible windows but are not hard-coded only to Apr-Sep 10-16;
- tail mass calibrated to historical high-renewable regimes;
- no quantile crossing;
- weighted mean negatives allowed only when structurally justified and bounded;
- negative run-length distribution calibrated by regime;
- negative-block VWAP distribution calibrated for business blocks;
- co-occurrence of CH/DE/FR negative episodes tested;
- tail dependence under congestion/NTC regimes tested;
- stress cases cover solar buildout, battery/flexibility uncertainty, low-demand holidays, hydro/flexibility constraints, and market price limits.

Acceptance:

- thresholds derived from historical quantiles and analyst-approved stress cases;
- Power BI shows negative-price regime diagnostics;
- no implicit positivity floor or historical-tail cap may hide a structurally plausible future negative regime.

### WP6 - Validation and Backtest

Goal: prove that the model improves economically relevant decisions.

Validation must separate two layers:

1. Market layer: EEX repricing, quote hierarchy, no-arbitrage, and run determinism.
2. Physical distribution layer: shape-demeaned or level-neutralized comparisons to realized spot, probabilistic calibration, and business-block risk.

Raw MAE/MAPE versus realized spot is not a production criterion for `p_t` because it mixes shape error, risk premium, and realized physical shocks.

Rolling backtest protocol:

- valuation dates: at least monthly snapshots from 2019-01-01 to the latest complete year, and weekly snapshots where forward history supports them;
- for every valuation date `T`, use only data with `available_at <= T`;
- build N+1, N+2, and N+3 delivery horizons where quote coverage exists;
- compare against fixed benchmarks listed below;
- report confidence intervals by block via bootstrap over valuation dates and delivery periods;
- use Diebold-Mariano or conditional predictive ability tests where sample size permits;
- require non-degradation on critical business blocks unless a signed desk waiver exists.

Benchmarks:

- current production/baseline LT model with flags OFF;
- residual-anchor diagnostic candidate, labeled diagnostic only;
- EEX flat-bucket naive curve;
- CH-history-only prior;
- desk curve if available and frozen before evaluation.

Business blocks:

- block pricing backtest for business-relevant blocks:
  - weekday summer 10-15;
  - weekday winter 18-9;
  - weekend solar hours;
  - peak/offpeak;
  - month strips and residual buckets.

Metrics:

- EEX residuals;
- level-neutralized block error vs realized spot and desk benchmark;
- month path error;
- seam excess vs multi-market prior;
- weekend/holiday premium error;
- negative-hour count and severity calibration;
- economic P&L proxy for profile deals;
- CH-DE, CH-FR, CH-AT, CH-IT_NORD spread MAE, spread sign accuracy, spread quantiles by regime, congestion-condition behavior, and neighbor-envelope breaches.

Probabilistic validation metrics:

- pinball loss by quantile;
- CRPS or WIS;
- coverage/reliability by regime and block;
- PIT or rank histograms where scenario paths exist;
- exceedance calibration for `P(price < 0)`, `P(price < -10)`, and `P(price < -30)`;
- negative run-length distribution;
- negative-block VWAP distribution;
- skill versus benchmarks and conditional tail calibration.

Acceptance:

- no degradation on core EEX repricing;
- no statistically material degradation versus current model on critical blocks, unless a signed desk waiver exists;
- 2027/2028 analyst objections addressed numerically;
- all material failures generate explicit findings, not hidden warnings.

### WP7 - Power BI and Governance

Goal: make QA visible and non-stale.

Actions:

- Power BI refresh must take an explicit CSV/run id, not infer stale files silently;
- bind new sidecars into the semantic model, not just write CSVs;
- display monthly path/seam diagnostics with severity;
- display prior decomposition and cross-border weights;
- display EEX residuals and calibration status;
- write one run manifest with source CSV, git SHA, data vintages, and QA verdict;
- add an automated binding check that compares `run_id`, `curve_sha256`, and `manifest_sha256` shown in Power BI against the audited disk artifacts.

Acceptance:

- Power BI source path equals audited run manifest;
- semantic model includes every sidecar used in summary metrics;
- stale output folder cannot show previous data as current;
- missing or mismatched Power BI run identifiers are production blocking.

Governance RACI:

| Role | Responsibility | Required sign-off |
|---|---|---|
| Model owner | implementation, model documentation, run evidence pack | before validation |
| Independent quant validator | mathematical audit, backtest review, no-leakage review | before external audit |
| Market desk approver | market plausibility, waiver approvals, thresholds requiring expert judgment | before production |
| Production owner | run orchestration, rollback, data freshness, scheduling | before production |
| Power BI owner | semantic-model binding, refresh traceability, dashboard QA | before production |

Waivers:

- every waiver must state scope, reason, evidence, approver, expiry date, and rollback condition;
- no waiver may override EEX hard residuals, future-data leakage, or stale Power BI binding;
- waivers expire after at most 90 calendar days or at the next material model/data change.

## 8. Production Acceptance Gates

These gates are deliberately numeric. They may be tightened after backtest calibration; loosening requires independent validation and desk sign-off.

| Gate | Threshold | Severity if failed |
|---|---:|---|
| Internal hard EEX residual | max `<= 1e-9 EUR/MWh` | P0 |
| Published/CSV residual after rounding | max `<= 1e-6 EUR/MWh` | P0 |
| Feasibility check | `rank(A)==rank([A|q])` or infeasibility `<= 1e-9` | P0 |
| KKT stationarity | `||H p + f + A'lambda||_inf <= 1e-7` | P0 |
| Future-data leakage | zero rows with `available_at > valuation_timestamp` | P0 |
| Quote freshness | latest required CH EEX snapshot no older than 2 business days unless valuation-date explicit | P0 |
| Cross-border data coverage | all required blocks pass or explicit zero-weight/fallback signed | P0 |
| Power BI binding | displayed `run_id`, `curve_sha256`, `manifest_sha256` equal manifest | P0 |
| Full CLI runtime | complete audited candidate run `<= 30 min` on desk machine, or documented batch SLA | P1 |
| Reproducibility flag OFF | numeric identity `atol=1e-12` on price columns and stable sorted index | P0 |
| Reproducibility optimizer ON | deterministic hash equality for fixed inputs excluding timestamped manifests | P1 |
| Seam excess vs cross-border prior | critical if absolute excess `> 12 EUR/MWh`; thresholds recalibrated by history before production | P1 |
| Negative price expected count | within regime-calibrated 5%-95% band or approved stress case | P1 |
| Negative price run length | within regime-calibrated 5%-95% band or approved stress case | P1 |
| Backtest coverage | at least 36 valuation dates and all critical blocks with >= 24 observations | P1 |
| Critical block non-degradation | no statistically material degradation vs current baseline; any degradation requires desk waiver | P1 |
| Probabilistic calibration | coverage error by main quantile <= 5 percentage points by major regime, or waiver | P1 |

## 9. Run Manifest Schema

`output/<run_id>/run_manifest.json` is contractual. Minimum fields:

```json
{
  "run_id": "string",
  "created_at_utc": "timestamp",
  "git_sha": "string",
  "git_branch": "string",
  "git_dirty": true,
  "command_line": "string",
  "config_hash_sha256": "string",
  "python_version": "string",
  "package_versions": {"package": "version"},
  "random_seeds": {"component": 0},
  "valuation_timestamp": "timestamp",
  "curve_csv": {"path": "string", "sha256": "string", "rows": 0},
  "input_artifacts": [{"path": "string", "sha256": "string", "available_at": "timestamp"}],
  "quote_snapshots": [{"market": "CH", "load_type": "BASE", "snapshot_date": "date", "source": "string", "sha256": "string"}],
  "data_vintages": [{"feature_group": "string", "max_available_at": "timestamp", "latest_revision_policy": "string"}],
  "accepted_quote_set_path": "string",
  "excluded_quote_set_path": "string",
  "rank_report_path": "string",
  "kkt_report_path": "string",
  "optimizer_status": {"status": "converged", "primal_inf": 0.0, "stationarity_inf": 0.0},
  "qa_verdict": "PASS|CONDITIONAL|NO_GO",
  "powerbi": {"dataset_id": "string", "report_id": "string", "refresh_timestamp": "timestamp"},
  "artifact_hashes": [{"path": "string", "sha256": "string"}]
}
```

## 10. Required Outputs

Each candidate run must produce:

```text
output/ch_hfc_hourly_<run_id>.csv
output/<run_id>/run_manifest.json
output/<run_id>/accepted_quote_set.csv
output/<run_id>/excluded_quote_set.csv
output/<run_id>/constraint_matrix_metadata.csv
output/<run_id>/rank_report.json
output/<run_id>/kkt_report.json
output/<run_id>/eex_constraint_residuals.csv
output/<run_id>/data_coverage_gaps.csv
output/<run_id>/cross_border_regime_checks.csv
output/<run_id>/monthly_prior_decomposition.csv
output/<run_id>/cross_border_weights.csv
output/<run_id>/monthly_path_and_seam_checks.csv
output/<run_id>/calendar_checks.csv
output/<run_id>/negative_price_regime_checks.csv
output/<run_id>/probabilistic_validation.csv
output/<run_id>/block_pricing_backtest.csv
output/<run_id>/qa_verdict.md
output/<run_id>/diagnostics/*.png
```

The CSV alone is insufficient evidence.

## 11. Test Matrix

| Test file | Purpose | Gate |
|---|---|---|
| `tests/test_lt_quant_contract_matrix.py` | feasibility, redundant quotes, inconsistent quotes, partial horizon policy | P0 |
| `tests/test_lt_quant_peak_offpeak.py` | disjoint BASE/PEAK/OFFPEAK convention and holiday calendar version | P0 |
| `tests/test_lt_quant_dst_time_index.py` | UTC interval index, 23h/25h DST, duplicate local hour not collapsed | P0 |
| `tests/test_lt_quant_no_leakage.py` | point-in-time feature catalog and adversarial future-feature injection | P0 |
| `tests/test_lt_quant_cross_border_shape.py` | zero-mean neighbor deviations and +50 EUR/MWh shift invariance | P0 |
| `tests/test_lt_quant_optimizer_kkt.py` | QP convexity, KKT residuals, deterministic solve | P0 |
| `tests/test_lt_quant_flag_off_identity.py` | flag-OFF numeric identity `atol=1e-12` on price columns/index | P0 |
| `tests/test_lt_quant_negative_regime.py` | non-crossing quantiles, scenario mean EEX consistency, negative event stats | P1 |
| `tests/test_lt_quant_backtest_protocol.py` | rolling split, benchmarks, metrics, minimum coverage | P1 |
| `tests/test_powerbi_manifest_binding.py` | Power BI run id/hash binding to audited manifest | P0 |

CI must run all P0 unit tests. Full rolling backtests may run in scheduled or pre-release jobs, but their last successful run id must be referenced in the release evidence pack.

## 12. Definition of Done

The redesign is done only when all items pass:

1. Full pipeline run, not direct post-processing, produces the candidate.
2. EEX BASE and PEAK hard constraints pass for every active product.
3. No future data leakage in priors or validation.
4. Flag-OFF path is byte-identical to pre-redesign production path.
5. Cross-border model uses DE/FR/AT/IT_NORD or documented IT proxy where data quality permits, with no absolute-level leakage.
6. Monthly paths and seams are explainable by quotes or priors.
7. Negative prices are calibrated as a structural distributional feature.
8. Rolling backtest beats or is desk-approved against current model on business blocks.
9. Power BI displays the same run manifest and diagnostics generated by the audited run.
10. KKT, rank, accepted/excluded quote set, and data coverage reports are emitted.
11. External audit verdict is at least `CONDITIONAL PASS`; production requires explicit desk approval.

## 13. Anti-Patterns Explicitly Rejected

- "Fix the chart" by manually anchoring a month sequence after calibration.
- Tune thresholds until the current curve passes.
- Copy DE monthly shape into CH and call it market-consistent.
- Allow negative prices by exception windows only.
- Use Power BI refresh as proof of quality.
- Hide direct-vs-CLI differences.
- Treat green tests as sufficient without quantitative backtest.
- Promote a curve whose critical diagnostics remain unresolved.

## 14. Immediate Next Actions

1. Freeze current candidate as diagnostic only.
2. Audit this plan with independent expert agents:
   - no-arbitrage and optimization;
   - cross-border market structure;
   - negative-price/probabilistic shape;
   - validation/governance and Power BI.
3. Update this document until it scores 10/10 against the self-audit checklist.
4. Only then create implementation tasks and commits.

## 15. Self-Audit Checklist

This plan is 10/10 only if it:

- is grounded in HPFC/PFC literature;
- rejects ad hoc curve patching as production methodology;
- provides a precise mathematical objective and constraints;
- covers BASE, PEAK, monthly, residual, hourly, weekend, and negative-price behavior;
- treats CH as a cross-border market influenced by DE/FR/AT/IT_NORD or documented IT proxy, not DE-only;
- preserves no-arbitrage and direct quote hierarchy;
- includes reproducible data-vintage and no-leakage controls;
- defines concrete implementation modules;
- defines tests and acceptance criteria;
- defines Power BI/run-manifest governance;
- states that the current candidate remains `NO GO`;
- is specific enough for an external quant auditor to criticize line by line.

## 16. Expert Audit Integration Log

This plan was revised after four independent expert-agent audits:

- no-arbitrage/optimization: added feasibility rank checks, null-space formulation, UTC interval universe, BASE/PEAK/OFFPEAK convention, KKT diagnostics;
- cross-border market structure: added zero-mean neighbor projections, +50 EUR/MWh shift invariance, IT_NORD, NTC/spread regime dataset, hard data coverage gates, Swiss hydro regime;
- validation/governance: added numeric production gates, rolling backtest protocol, manifest schema, RACI, waiver rules, test matrix, Power BI binding checks;
- probabilistic/negative prices: separated deterministic forward curve `p_t` from physical scenarios/quantiles, added non-crossing/scenario-path rules, proper scoring metrics, negative event/run/co-occurrence tests, and foundation-model limitations.

Current self-score: `10/10 candidate for external audit`, subject to the external auditor's independent review.

## 17. Source Links For Auditor

- Fleten and Lemming, 2003: https://orbit.dtu.dk/en/publications/constructing-forward-price-curves-in-electricity-markets/
- Kiesel, Paraschiv, Saethero, 2019: https://ideas.repec.org/a/spr/comgts/v16y2019i1d10.1007_s10287-018-0300-6.html
- Caldana, Fusai, Roncoroni, 2017: https://openaccess.city.ac.uk/id/eprint/17018/
- Keles et al., 2020: https://publikationen.bibliothek.kit.edu/1000125633/142347297
- PriceFM, 2026: https://arxiv.org/html/2508.04875v4
- EPF deep learning review, 2026: https://arxiv.org/html/2602.10071v2
- Eurelectric negative prices explainer, 2024: https://www.eurelectric.org/wp-content/uploads/2024/11/Eurelectric-explainer-on-negative-prices-1.pdf
- Gneiting and Raftery, 2007: https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- Nowotarski and Weron, 2018: https://ideas.repec.org/a/eee/rensus/v81y2018ip1p1548-1568.html
- Probabilistic mid/long-term EPF, 2023: https://ideas.repec.org/a/eee/eneeco/v120y2023ics0140988323001007.html
