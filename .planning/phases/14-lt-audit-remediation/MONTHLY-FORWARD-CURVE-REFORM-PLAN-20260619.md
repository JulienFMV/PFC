# Monthly Forward Curve Reform Plan - CH LT PFC - 2026-06-19

## 0. Executive Decision

Verdict: the current CH LT pipeline is not production-ready for sparse long-term delivery years until the monthly level construction is reformulated.

The defect is not "April 2028 vs April 2029" in isolation. The defect is that the model builds sparse annual and residual delivery years through a sequence of independent cascade/smoothing/post-processing steps rather than through one global arbitrage-free monthly curve problem. Exact EEX calibration after the fact can preserve wrong monthly allocation perfectly.

The target architecture is a single monthly BASE curve solve under hard CH EEX constraints, informed by CH history and neighboring EEX markets only through recentered shape deviations. The hourly/15-minute stack must consume this monthly curve as the level input.

This plan is intentionally conservative:

- no relaxation of CH EEX quote calibration in v1;
- no manual month-level overrides;
- no neighbor absolute level leakage;
- no promotion of local-test post-processors into production as-is;
- no "score-only" approval without visual and quantitative audit for 2028-2030.
- no assumption that far-horizon monthly forward quotes exist when the history
  parquet shows they do not.

## 1. Problem Statement

### 1.1 Observed Defect

The monthly plot exposed market-incoherent shapes for sparse years, especially 2028-2030. Numerical repricing gates passed while the monthly shape looked wrong.

Examples of suspect behavior that must be captured by controls:

- monthly shapes for adjacent long-term years become near-clones despite materially different EEX calendar levels;
- a year with high Q1 and a residual Apr-Dec bucket can contaminate comparisons if the model compares a residual bucket against a full calendar year;
- month ranking across years can contradict calendar spreads without any explanatory seasonal-mix adjustment;
- December / winter values can invert across years without support from CH quotes or robust neighbor/history evidence.

### 1.2 Root Cause Hypothesis

Current production path:

```text
run_pfc_production.py
  -> pfc_shaping.pipeline.production_phases.run_long_term_phase
  -> ContractCascader.cascade(...)
  -> PFCAssembler.build(...)
  -> _resolve_base(...)
  -> msfc_spline.smooth_base_prices(...)
  -> multiplicative hourly/weekly/intraday/water-value shape
  -> ArbitrageFreeCalibrator
```

Current local-test export path:

```text
scripts/export_local_test_ch_hourly_csv.py
  -> PFCAssembler / CSV assembly
  -> quote-aware smoothing / neighbor anchors / final quant annual smoothing
  -> cross-year seasonal post-processor
  -> seam nullspace smoothing
  -> repeated EEX recalibrations
```

These are two different model paths. The local-test path contains valuable diagnostics and experiments, but it is not the canonical monthly level model. The production path can still construct monthly levels from sparse years using year-by-year cascade and smoothing choices that are not globally optimized across years.

### 1.3 Non-Goals

This plan does not:

- approve the existing cross-year seasonal shape optimizer for production;
- solve short-term CT overlay behavior;
- change CH EEX quote levels;
- replace hourly shape modules in one step;
- introduce a black-box ML model for monthly levels.

## 2. Literature-Aligned Modeling Principles

The plan follows three principles consistent with standard electricity forward curve construction:

1. **Average contracts as hard constraints.** Monthly, quarterly, annual and residual products constrain delivery-period averages.
2. **Smoothness / regularity in the unquoted degrees of freedom.** Missing points should be determined by a smooth or regularized optimization, not by ad hoc independent bucket shifts.
3. **External information enters as shape, not absolute level.** Neighbor markets and history can guide deviations from parent means, but CH levels are fixed by CH quotes.

Relevant conceptual anchors:

- Fleten-Lemming style construction: combine market quotes with model/shape information while respecting traded products.
- Benth/Koekebakker/Ollmar style maximum smoothness from average-based contracts: solve a constrained curve problem rather than fitting isolated delivery buckets.
- Hourly price forward curve literature such as Kiesel/Paraschiv/Saetheroe: seasonality and granularity are model objects, not residual plotting artifacts.

## 2.1 Data Coverage Reality Check

The latest local `data/eex_forwards_history.parquet` has dense calendar and
quarter quotes for long horizons, but almost no far-horizon monthly quotes.
Before implementation, reproduce and persist a coverage report:

```text
output/monthly_curve_calibration/monthly_quote_coverage_by_horizon.csv
```

Observed BASE monthly quote counts in the local parquet (`1569` snapshots,
2020-05-04 to 2026-06-17):

| market | h+0 | h+1 | h+2 | h+3+ |
|---|---:|---:|---:|---:|
| CH | 6343 | 1097 | 0 | 0 |
| DE | 9830 | 7257 | 637 | 0 |
| FR | 673 | 120 | 0 | 0 |
| AT | 609 | 120 | 0 | 0 |
| IT | 502 | 20 | 0 | 0 |

Implications for v1:

- CH far-horizon month-vs-parent forward history is unavailable at `h+2+`;
- a robust four-market monthly panel is not available at `h+2+`; in practice,
  DE is the only market with meaningful far-horizon monthly quotes, and even
  DE is partial by delivery year;
- lambda calibration trained only on `h+0/h+1` monthly-rich data has a
  train/deploy mismatch for the sparse `h+2+` years that triggered the defect;
- for far horizons, maximum smoothness under hard CH constraints plus
  reliability-weighted DE shape and CH structural/climatological seasonality is
  the core model, not a fallback.

This does not weaken the plan. It prevents over-engineering a panel/history
prior that the data cannot support.

## 3. Target Architecture

### 3.1 New Monthly Layer

Add:

```text
pfc_shaping/calibration/monthly_forward_curve.py
```

This module builds a monthly BASE curve for a target market, initially CH.

Primary function:

```python
solve_monthly_forward_curve(
    *,
    market: str,
    delivery_months: pd.PeriodIndex,
    own_quotes: Sequence[MarketQuote],
    eex_history: pd.DataFrame,
    neighbor_markets: Sequence[str] = ("DE", "FR", "AT", "IT"),
    neighbor_quotes: Sequence[MarketQuote] | None = None,
    run_timestamp: pd.Timestamp,
    config: MonthlyCurveConfig | None = None,
) -> MonthlyCurveResult
```

Production and local export must call the contract-based entry point:

```python
solve_monthly_forward_curve_from_inputs(inputs: MonthlyCurveInputs) -> MonthlyCurveResult
```

Data classes:

```python
@dataclass(frozen=True)
class MonthlyCurveConfig:
    lambda_prior: float
    lambda_smooth_month: float
    lambda_smooth_yoy: float
    lambda_shape: float
    neighbor_shrinkage: float
    robust_panel_quantile: float
    min_history_snapshots: int
    max_prior_residual_eur_mwh: float | None
    constraint_tolerance: float
    stationarity_tolerance: float

@dataclass(frozen=True)
class MonthlyCurveResult:
    monthly_curve_schema_version: str
    monthly_curve: pd.Series
    constraints: pd.DataFrame
    residuals: pd.DataFrame
    priors: pd.DataFrame
    diagnostics: pd.DataFrame
    kkt: dict[str, float | bool]
```

### 3.2 First-Class Data And Constraint Contracts

Add explicit contracts instead of passing loosely shaped dictionaries through
the stack:

```text
pfc_shaping/data/lt_data_contract.py
pfc_shaping/calibration/monthly_forward_curve.py
```

Core data objects:

```python
@dataclass(frozen=True)
class DeliveryGrid:
    months: pd.PeriodIndex
    timezone: str
    month_hours: pd.Series
    calendar: str

@dataclass(frozen=True)
class MarketQuote:
    market: str
    product: str
    load_type: str
    price: float
    snapshot_date: pd.Timestamp
    source: str
    available_at: pd.Timestamp

@dataclass(frozen=True)
class MonthlyCurveInputs:
    delivery_grid: DeliveryGrid
    own_quotes: tuple[MarketQuote, ...]
    neighbor_quotes: tuple[MarketQuote, ...]
    eex_history: pd.DataFrame
    run_timestamp: pd.Timestamp
    config: MonthlyCurveConfig
    source_hashes: Mapping[str, str]
```

Production and local export must both call the same
`build_monthly_curve_inputs(...)` helper before `solve_monthly_forward_curve(...)`.
No path-local quote filtering, date slicing, or neighbor selection is allowed.

Move shared constraint primitives to:

```text
pfc_shaping/calibration/constraints.py
```

The LT layer may import from calibration. The calibration layer must not import
from `pfc_shaping.lt.model.*` merely to build monthly constraints. Keep a
compatibility adapter in `pfc_shaping.lt.model.shape_constraints` if existing
LT modules need the old import path.

Monthly row metadata must include:

```text
source_quote_keys
is_residual
parent_product
active
dropped_reason
active_row_indices
```

The solver may only consume quotes and historical observations whose
`available_at <= run_timestamp`. This is mandatory for backtests and for any
lambda calibration; otherwise the history prior can leak future EEX shape into
past simulations.

Every run must produce a manifest:

```text
run_timestamp
git_commit
source_file_hashes
forward_snapshot_date
solver_config
lambda_config
active_constraints_hash
monthly_solution_hash
monthly_curve_schema_version
```

This manifest is part of the audit artifact, not optional metadata.

Hashing and serialization rules:

- sorted rows and columns;
- fixed float precision;
- UTC timestamps;
- explicit schema version;
- canonical JSON separators;
- hashes exclude volatile generation timestamps but include input hashes,
  config hashes, active constraints and monthly prices.

### 3.3 Position in the Pipeline

The monthly curve solve must occur before hourly assembly.

Target production sequence:

```text
load EEX quotes
  -> solve_monthly_forward_curve(CH)
  -> build assembler_base_prices from solved monthly BASE plus original traded non-BASE keys
  -> PFCAssembler.build(..., base_prices=assembler_base_prices, quoted_keys=original_ch_quote_keys)
  -> hourly shape
  -> one final EEX calibration / audit
```

Important: `quoted_keys` must retain the original CH market products, not the synthetic monthly products generated by the monthly solver. Otherwise the final calibrator would treat synthetic monthly levels as traded quotes and over-constrain the hourly curve.

Concrete integration points:

```text
pfc_shaping/pipeline/production_phases.py
  after EEX quote loading and before PFCAssembler.build(...)

scripts/export_local_test_ch_hourly_csv.py
  same solver call and same config object as production

scripts/build_powerbi_exports.py
  read monthly solver diagnostics and manifest
```

The export script must not have a private monthly construction path. If it
needs diagnostics, it can request more output from the same solver.

### 3.4 Relationship to Existing Modules

Keep:

- `pfc_shaping.lt.model.quant_shape_optimizer` as a possible solver backend if it cleanly supports the monthly problem;
- `pfc_shaping.lt.model.shape_constraints` primitives for duration-weighted average constraints;
- `pfc_shaping.lt.model.cross_border_shape` concepts for no-level-leakage projection;
- `ContractCascader` for fallback / legacy / non-CH paths during migration.

Do not promote as production core:

- `cross_year_seasonal_shape.py` post-processor;
- `quote_aware_monthly_smoothing.py` as a separate final correction layer;
- repeated calibrate-after-each-post-process sequences.

Eventually deprecate for sparse years:

- independent annual cascade in `ContractCascader.cascade(...)` for CH annual-only years;
- `msfc_spline.smooth_base_prices(...)` as the monthly level authority.

## 4. Mathematical Formulation

### 4.1 Variables

Let `M` be all delivery months in the LT horizon.

```text
x_m = CH BASE monthly price for month m
```

Use EUR/MWh variables for direct interpretability in v1. Consider log-relative variables only after proving negative/low-price handling is robust.

### 4.2 Hard Constraints

For each CH quoted product `p`:

```text
sum_{m in p} h_m x_m / sum_{m in p} h_m = q_p
```

where `h_m` is actual delivery hours in Europe/Zurich, including leap years and DST.

Products:

- monthly quotes: `YYYY-MM`;
- quarterly quotes: `YYYY-Qn`;
- calendar quotes: `YYYY`;
- residual buckets implied by the quote-aware non-overlap selection, e.g. if `2028` and `2028-Q1` are quoted, Apr-Dec is constrained by the residual energy budget.

Residual formula:

```text
R = Y \ S
q_R = (H_Y q_Y - sum_{p in S} H_p q_p) / H_R
```

where `Y` is a quoted calendar, `S` is the union of quoted sub-products,
`H_*` is actual Europe/Zurich delivery hours, and `R` is the remaining
delivery set.

The constraint builder must avoid rank-inconsistent duplicate constraints. It must report:

- original quote list;
- active independent constraints;
- implied residual constraints;
- dropped redundant constraints;
- rank and infeasibility.

Fail hard if the quote set is inconsistent beyond tolerance. Do not silently
choose one quote over another. The exception must include the smallest
conflicting subset if it can be identified, or at least the residual vector in
product space.

Quoted-product consistency rule:

- a CH quoted product may only be removed from the active matrix if its row is
  linearly dependent on already-active rows and its target is consistent within
  `constraint_tolerance`;
- if a quoted calendar, quarter, month, or implied residual conflicts with the
  active system, raise infeasible-quote diagnostics;
- the solver must never silently choose one traded quote over another to
  recover feasibility.

### 4.3 Objective

Solve:

```text
min_x  0.5 * [
    lambda_prior        * ||W_prior (x - x_prior)||^2
  + lambda_smooth_month * ||D2_month x||^2
  + lambda_smooth_yoy   * ||D_yoy shape(x)||^2
  + lambda_panel_shape  * ||W_panel (shape(x) - shape_panel)||^2
  + lambda_history_shape * ||W_hist  (shape(x) - shape_ch_history)||^2
]
subject to A x = q
```

v1 active objective must be identifiable under sparse far-horizon data. Use the
full expression above as the general form, but enable only a slim core unless
coverage diagnostics justify more terms:

```text
min_x  0.5 * [
    lambda_smooth_month * ||D2_month x||^2
  + lambda_smooth_yoy   * ||D_yoy shape(x)||^2
  + lambda_shape        * ||W_shape (shape(x) - shape_fused)||^2
]
subject to A x = q
```

where `shape_fused` is one reliability-weighted shape prior combining available
DE forward shape, CH near-tenor forward climatology and CH structural/spot
climatology in zero-mean parent-block space. The level prior term is a
regularization/ridge or parent-flat feasible baseline only; it must not become
a second shape prior that double-counts `shape_fused`.

The separate `lambda_panel_shape` and `lambda_history_shape` terms in the
general expression are reserved for later sensitivity runs. They are not v1
production defaults unless coverage diagnostics prove that both sources are
independently identifiable at the target horizon.

Definitions:

```text
H = diag(h_m)
(P_b x)_m = sum_{n in b} h_n x_n / sum_{n in b} h_n, for m in b
shape(x) = (I - P) x
```

`parent(m)` is the economically comparable block:

- quoted quarter if the month is inside a quoted quarter;
- residual block if the year has quoted sub-products and an implied residual;
- calendar if annual-only;
- never compare a residual Apr-Dec block directly against a full calendar year without converting to a comparable block.

`x_prior` is not allowed to be the legacy cascaded curve, a post-processed
diagnostic curve, or any curve carrying neighbor absolute levels. In v1 it is
one of:

- a CH-only feasible parent-flat baseline; or
- a feasible projection of CH historical and panel shape priors, recentered
  inside each CH parent block.

Any `x_prior` source must be recorded in diagnostics with an explicit
`prior_source` field and a no-neighbor-level-leakage test.

If `x_prior` is a feasible projection of CH historical or panel shape priors,
the solver must avoid double-counting the same evidence. Either use a
parent-flat CH-only feasible `x_prior` with one separate `lambda_shape` penalty
on `shape_fused`, or set/report evidence-specific lambda terms so each prior
source enters the objective exactly once.

Smoothness operators:

```text
D2_month x[y,m] = x[y,m-1] - 2 x[y,m] + x[y,m+1]
D_yoy shape(x)[y,m] = shape(x)[y,m] - shape(x)[y-1,m]
```

Both operators must be scaled by delivery duration and/or calibrated inverse
historical variance so penalties are comparable across months with different
hour counts. Smoothness rows should be global across adjacent delivery months
whenever both months exist on the delivery grid. Hard quote boundaries do not
by themselves remove smoothness rows; constraints already protect quoted
averages. Rows are removed only when the row would compare incompatible
parent-block definitions, missing months, or unsupported comparable-shape
states. Removed rows must be reported.

The implementation must solve the equality-constrained system directly or by
an explicit nullspace parameterization:

```text
x = x_feasible + N z
min_z objective(x_feasible + N z)
```

This is the preferred production formulation because it guarantees that
smoothness, neighbor and historical priors can only move the unquoted degrees
of freedom. The optimizer must report:

```text
max_abs_constraint_residual
stationarity_residual
objective_terms
condition_number_or_regularization
active_constraint_rank
nullspace_dimension
active_row_indices
dropped_rows
full_A_residual_max
active_A_residual_max
ridge_used
```

Equivalent KKT form:

```text
[ Q  A.T ] [ x  ] = [ -c ]
[ A   0  ] [ nu ]   [  q ]
```

`Q` must be positive definite on the nullspace of `A`; if a ridge is required,
the ridge size and effect on diagnostics must be reported.

The final hourly EEX calibrator remains a verification and small numerical
alignment step, not the place where the monthly level model is repaired.

### 4.4 Panel Shape Prior

For each neighbor market `k in {DE, FR, AT, IT}`:

1. Build a neighbor monthly no-arbitrage curve using that market's quotes and history.
2. Convert it to deviations from its own comparable parent blocks.
3. Recenter / project deviations so they have zero weighted mean inside the relevant CH parent block.
4. Combine markets using robust aggregation:

```text
shape_panel_m = weighted_median_k(shape_{k,m})
```

Weights can use:

- availability of current quotes for the delivery product;
- market liquidity proxy from quote coverage count;
- geographic/economic relevance;
- shrinkage to CH history when neighbor evidence is sparse.

Do not use neighbor absolute levels. The following invariance must hold:

```text
neighbor_prices + C  =>  same CH monthly solution
```

for any constant `C` applied to all neighbor prices within a market/curve.

Coverage-aware v1 rule:

- do not build recursive full monthly neighbor curves for markets without
  monthly evidence at the target horizon;
- use directly observed neighbor month-vs-quarter/calendar deviations where
  monthly products exist;
- use neighbor quarterly/calendar products to guide block-level seasonal mix,
  not intra-quarter monthly splits, when monthly products are absent;
- compute a robust multi-market panel only when at least two markets have
  comparable current monthly or block-shape evidence;
- if only DE has usable far-horizon monthly evidence, label the prior
  `DE_SINGLE_MARKET` and shrink it toward CH structural/climatological shape;
- if no current neighbor monthly evidence exists, the model falls back to
  maximum smoothness plus CH structural/climatological shape, with explicit
  `UNSUPPORTED` flags for forward-monthly evidence.

Amplitude rule:

- neighbor markets provide zero-mean shape only;
- after recentering inside each CH parent block, neighbor deviations are
  standardized or shrunk against CH historical forward-shape dispersion before
  entering the objective;
- a neighbor market may influence seasonal allocation, but not import a high or
  low absolute level regime;
- tests must cover constant shifts, block-specific constant shifts, missing
  markets and one-market outliers.

### 4.5 CH Historical Prior

Use `data/eex_forwards_history.parquet` to estimate historical distributions of:

- month vs calendar deviations;
- month vs quarter deviations;
- residual Apr-Dec profiles when a year has Q1 and calendar quotes;
- same-month year-on-year shape changes;
- winter/spring/summer/autumn slope distributions.

History must be computed by snapshot date and delivery product, not from realized spot. Spot can inform hourly shape, but the monthly forward layer should primarily learn from forward-market seasonal structure.

Coverage exception for far horizons:

- CH forward monthly history at `h+2+` is unavailable in the current parquet;
- near-tenor CH forward month-vs-parent patterns may be used only as
  climatological shape evidence with an explicit tenor-mismatch penalty;
- realized CH spot or climate/fundamental seasonality may be used for
  zero-mean structural monthly shape when forward monthly evidence is
  unsupported;
- realized/spot-derived inputs may never set absolute forward levels and must
  be recentered inside CH parent blocks before entering `shape_fused`.

All historical features must be point-in-time:

- current run may use snapshots available on or before the run timestamp;
- a backtest for snapshot `T` may only use snapshots available before or at
  `T`;
- current quote coverage and historical quote coverage must be logged
  separately;
- insufficient coverage must yield `UNSUPPORTED`, not `PASS`.

For each rolling-origin withholding experiment, the solver inputs and the
historical-prior builder must receive a masked point-in-time view: withheld
products at the origin snapshot are removed from `own_quotes`,
`neighbor_quotes`, and any `eex_history` rows or derived features that would
reveal the withheld target. Alternatively, historical priors for that
experiment must be fit only on snapshots strictly before the origin date. The
calibration report must state which masking rule was used.

### 4.6 Lambda Calibration

The lambda values are model parameters, not style settings. They must be
calibrated before production approval.

Calibration protocol:

1. Build rolling-origin backtests over historical EEX snapshots.
2. Start from historical same-snapshot quote sets with richer coverage, mask
   selected monthly/quarterly products from solver inputs and any same-snapshot
   history-derived features that would reveal the target, solve from the
   degraded quote set, and score primary calibration loss against the masked
   same-snapshot traded prices. Later-observed forward quotes are secondary
   stability diagnostics only and must not drive lambda selection unless
   de-leveled and explicitly justified.
3. Evaluate a reduced grid or Bayesian search over the active v1 degrees of
   freedom:

```text
lambda_smooth_month
lambda_smooth_yoy
lambda_shape
neighbor_shrinkage
history_lookback
```

Do not run an unconstrained seven-dimensional lambda search for v1. Normalize
penalty terms by historical variance or duration so most penalties are O(1),
then calibrate only the core trade-off:

```text
shape confidence vs smoothness
monthly smoothness vs year-on-year smoothness
```

The calibration report must explicitly quantify the regime mismatch between
monthly-rich `h+0/h+1` training examples and sparse `h+2+` deployment years.
If this mismatch is material, v1 defaults must favor maximum smoothness and
shrink shape priors rather than overfitting near-tenor monthly quotes.

4. Plot an L-curve / Pareto surface between:

```text
withheld_quote_error
smoothness / curvature
historical_shape_outlier_score
neighbor_panel_disagreement
```

5. Select a conservative knee point, not the visually nicest current curve.
6. Persist the chosen parameters with:

```text
calibration_window
withheld_products
metric_table
selected_config
selection_reason
approver
```

No merge to production defaults without this calibration report.

### 4.7 Peak Treatment

v1 scope:

- solve monthly BASE first;
- preserve existing PEAK calibration downstream;
- do not infer PEAK monthly shape unless CH PEAK quotes are sufficient.

v1 diagnostics must still report whether BASE monthly changes cause PEAK recalibration stress:

```text
peak_quote_key
peak_target
peak_pre_calibration_mean
peak_post_calibration_mean
peak_calibration_delta
offpeak_compensation_delta
max_hourly_delta_from_peak_recalibration
peak_residual_after_calibration
```

v2:

- joint BASE/PEAK monthly solve or spread solve:

```text
peak_month = base_month + peak_spread_month
```

with PEAK constraints and holiday calendars by delivery market.

## 5. Implementation Plan

### Phase A - Ground Truth and Fixtures

Create a reproducible synthetic/anonymized fixture preserving the sparse-market
geometry seen in the latest local forward snapshot:

```text
tests/fixtures/monthly_curve_sparse_2028_synthetic.parquet
```

If storing a full fixture is too heavy, create a minimal synthetic fixture preserving the critical sparse geometry:

- CH: `2028`, `2028-Q1`, `2029`, `2030`;
- CH relevant monthly/quarterly quotes where present;
- DE/FR/AT/IT current monthly/quarterly/calendar availability;
- historical snapshots sufficient for shape estimates.

Repository hygiene:

- full desk snapshots stay out of git;
- committed fixtures are minimal, synthetic or anonymized;
- generated solver reports/manifests live under ignored `output/` or
  `powerbi/data/`;
- `.planning/` can contain reviewed plans and audit summaries, but not large
  generated CSV/PNG work products.

Acceptance:

- tests do not depend on mutable local desk files;
- full local audit can still use `data/eex_forwards_history.parquet`.
- every fixture states whether it is synthetic, anonymized, or extracted from
  desk data;
- fixtures include expected active constraints and expected residual targets,
  not just source quotes.
- coverage report reproduces monthly, quarterly and calendar quote availability
  by market/horizon and drives prior reliability weights.

### Phase B - Constraint System

Implement in `monthly_forward_curve.py`:

```python
build_monthly_constraint_system(delivery_months, own_quotes, timezone="Europe/Zurich")
```

Required behavior:

- actual month hour counts;
- quote-aware non-overlap / residual handling;
- rank diagnostics;
- exact residual target calculation;
- human-readable constraint table.

Acceptance tests:

- `2028 + 2028-Q1` implies `2028-RESIDUAL` Apr-Dec target equal to energy residual;
- leap year February and DST do not break hour weighting;
- redundant `CAL + all four Q` is handled deterministically;
- infeasible quotes raise with useful diagnostics.
- adding a quoted month inside a quoted quarter changes the independent
  residual structure deterministically;
- all row labels are stable enough for audit reports and Power BI sidecars.

### Phase C - Priors

Implement:

```python
build_history_shape_prior(...)
build_neighbor_panel_shape_prior(...)
build_structural_monthly_shape_prior(...)
build_fused_shape_prior(...)
```

History prior:

- compute deviations from comparable parent products;
- require minimum snapshots;
- robust median and dispersion;
- output reliability weights.

Structural/climatological prior:

- estimate zero-mean CH monthly shape from realized spot, hydro/seasonal
  climatology, or near-tenor forward history;
- recenter inside comparable CH parent blocks;
- apply tenor-mismatch and source-reliability penalties;
- never affect absolute CH levels.

Panel prior:

- build neighbor monthly curves or implied monthly deviations;
- recenter to CH comparable parent block;
- robust aggregation across available markets;
- output contribution diagnostics per market/month.

Fused shape prior:

- combine panel, CH forward history and CH structural/climatological shape into
  one `shape_fused` object;
- reliability weights are functions of quote coverage, horizon, tenor mismatch
  and source dispersion;
- `shape_fused` carries status labels such as `PANEL_MULTI_MARKET`,
  `DE_SINGLE_MARKET`, `STRUCTURAL_ONLY`, or `UNSUPPORTED`.

Acceptance tests:

- adding +1000 to neighbor levels leaves prior deviations unchanged;
- missing AT/IT degrades gracefully to DE/FR/CH history;
- one outlier market does not dominate robust panel median;
- prior has zero weighted mean inside each CH parent block after recentering.
- if no neighbor or history prior has sufficient support, the solver still
  reprices CH quotes but diagnostics mark shape evidence as `UNSUPPORTED`.

### Phase D - Solver

Implement:

```python
solve_monthly_forward_curve(...)
```

Use either:

- a small dense KKT solve for monthly variables; or
- `QuantShapeOptimizer` adapted to monthly index.

Monthly dimension is small, so correctness and diagnostics matter more than micro-optimization.

Acceptance:

- hard constraint residual max <= `1e-8`;
- stationarity residual <= `1e-7`;
- deterministic solution;
- no silent fallback to legacy cascade.
- objective terms are reported in original units and normalized units;
- changing only neighbor absolute levels leaves the CH solution unchanged within
  numerical tolerance;
- changing CH calendar quotes moves the monthly solution through constraints,
  not through an external post-processor.

### Phase D1 - Sparse-Year Proof Of Shape

Before the full lambda-calibration and governance package, generate a
diagnostic 2028-2030 monthly BASE curve from:

```text
hard CH constraints
nullspace/KKT solve
maximum smoothness
DE_SINGLE_MARKET or STRUCTURAL_ONLY shape_fused prior
variance-normalized provisional lambdas
```

Purpose:

- prove that the core formulation removes the visible 2028-2030 monthly defect;
- verify exact CH EEX repricing, no neighbor level leakage and comparable-block
  logic;
- inspect the candidate before investing in full calibration infrastructure.

This proof run is not production approval. If it fails visually or
quantitatively, fix the model formulation before adding governance scaffolding.

### Phase D0 - Lambda Calibration

This phase is required before using the monthly solver as a production default.

Artifacts:

```text
.planning/phases/14-lt-audit-remediation/lambda_grid.yaml
.planning/phases/14-lt-audit-remediation/monthly_curve_selected_config.json
.planning/phases/14-lt-audit-remediation/LAMBDA-CALIBRATION-SUMMARY.md
output/monthly_curve_calibration/monthly_curve_lambda_backtest.csv
output/monthly_curve_calibration/monthly_curve_lcurve.png
```

Backtest design:

- rolling-origin snapshots from `eex_forwards_history.parquet`;
- only point-in-time data available at each origin;
- start from historical snapshots with richer quote coverage;
- mask selected monthly/quarterly products from `own_quotes`,
  `neighbor_quotes`, and any same-snapshot history-derived features that would
  reveal the target;
- solve the monthly curve from the degraded quote set;
- score primary calibration loss against the masked same-snapshot traded
  prices.

Later-observed forward quotes may only be used as a secondary stability
diagnostic, reported separately from primary withheld-product loss. They must
not drive lambda selection unless de-leveled and explicitly justified, because
otherwise the score mixes curve-construction error with market moves.

Candidate grid:

```yaml
lambda_smooth_month: [0.0, 0.1, 1.0, 10.0, 100.0]
lambda_smooth_yoy: [0.0, 0.1, 1.0, 10.0]
lambda_shape: [0.0, 0.25, 1.0, 4.0]
neighbor_shrinkage: [0.25, 0.5, 0.75]
history_lookback_years: [3, 5, 6]
```

Do not run an unconstrained seven-dimensional lambda search for v1. Normalize
penalty terms by historical variance or duration so most penalties are O(1),
then calibrate only the core trade-off:

```text
shape confidence vs smoothness
monthly smoothness vs year-on-year smoothness
```

The calibration report must explicitly quantify the regime mismatch between
monthly-rich `h+0/h+1` training examples and sparse `h+2+` deployment years.
If this mismatch is material, v1 defaults must favor maximum smoothness and
shrink shape priors rather than overfitting near-tenor monthly quotes.

Scoring table columns:

```text
config_hash
origin_date
withheld_product
withheld_load_type
market
target_price
predicted_price
abs_error
signed_error
constraint_residual_max
curvature_score
same_month_rank_score
historical_outlier_score
neighbor_disagreement_score
unsupported_gate_count
critical_gate_count
```

Selection rule:

- require every known-bad fixture to produce at least one `CRITICAL` row on the
  targeted governance gates;
- exclude any config that lets a known-bad fixture `PASS` required gates or
  returns `UNSUPPORTED` for the targeted failure mode;
- exclude any config that creates `CRITICAL` gates on known-coherent fixtures;
- exclude any config with unstable or ill-conditioned KKT diagnostics;
- choose the L-curve/Pareto knee that minimizes withheld-product error without
  materially increasing curvature or historical outlier scores;
- persist the selected config hash and exact selection reason.

Governance test:

- changing production lambda defaults without regenerating the selected-config
  artifact fails CI or the production approval script.

### Phase E - Integration Behind Flag

Add config:

```yaml
forwards:
  monthly_curve_solver:
    enabled: false
    markets: ["DE", "FR", "AT", "IT"]
    lambda_prior: ...
    lambda_smooth_month: ...
    lambda_smooth_yoy: ...
    lambda_shape: ...
    neighbor_shrinkage: ...
```

Integrate in:

```text
pfc_shaping/pipeline/production_phases.py
scripts/export_local_test_ch_hourly_csv.py
```

The export script and production path must call the same monthly solver when enabled.

When `monthly_curve_solver.enabled=true`, the monthly solver is the level
authority:

- production must pass the monthly BASE dictionary to `PFCAssembler`;
- production must pass `assembler_base_prices` as solved monthly BASE keys plus
  original CH PEAK/OFFPEAK/traded non-BASE keys required by the existing PEAK
  calibration path;
- synthetic monthly BASE keys must not enter `quoted_keys`;
- `quoted_keys` must contain only original traded CH BASE/PEAK keys;
- fail fast if any quoted PEAK key is requested but missing from
  `assembler_base_prices`;
- legacy cascading inside `PFCAssembler.build` must be disabled;
- `msfc_spline.smooth_base_prices(...)` must be disabled for BASE level
  construction;
- final calibration may use only original quoted CH keys.

Add an explicit integration switch:

```text
monthly_level_authority = "solver"
skip_legacy_level_cascade = true
skip_legacy_base_smoothing = true
```

The export flags for quote-aware monthly smoothing, neighbor anchors,
cross-year shaping, final monthly smoothing, and seam nullspace smoothing are
mutually exclusive with the new solver unless they run in `diagnostic_only`
mode. `diagnostic_only` means they may compute sidecar diagnostics from a copy
of the curve, but they must not mutate price columns or write changed CSV/PFC
outputs.

Acceptance:

- flag OFF is byte-identical for production PFC outputs and local export price
  outputs. Only explicitly requested diagnostic sidecars/logs may differ;
- flag ON emits identical `monthly_solution_hash`, `active_constraints_hash`
  and monthly BASE values in production dry run and local export for the same
  `MonthlyCurveInputs` fixture;
- generated report includes monthly solver diagnostics.
- flag ON writes a run manifest beside the generated CSV;
- local export refuses incompatible combinations of legacy post-processors and
  the monthly solver unless explicitly marked `diagnostic_only`.

### Phase F - Audit Gates

Add or upgrade audits in:

```text
scripts/audit_ch_hfc_seasonal_coherence.py
scripts/audit_ch_pfc_hourly_shape.py
scripts/build_powerbi_exports.py
```

New gates:

1. `hard_monthly_curve_repricing`
2. `neighbor_level_leakage`
3. `residual_vs_implied_comparable_block`
4. `same_month_rank_consistency`
5. `calendar_spread_seasonal_decomposition`
6. `historical_quantile_shape_outlier`
7. `monthly_shape_regression_2028_2030`
8. `lambda_calibration_artifact_present`
9. `point_in_time_data_contract`
10. `production_export_path_parity`

Thresholds:

- warning above historical P90;
- critical above historical P97.5;
- if historical sample insufficient, status must be `UNSUPPORTED`, not `PASS`.

Required gate output schema:

```text
gate_id
status
severity
market
load_type
year
month
product
parent_block_id
parent_block_type
parent_hours
parent_mean
month_price
month_deviation_from_parent
metric_name
metric_value
threshold_warning
threshold_critical
threshold_source
n_history
n_neighbors
evidence
remediation_hint
```

Required threshold file:

```text
historical_thresholds.csv
```

Columns:

```text
gate_id
metric
market
delivery_bucket
lookback_start
lookback_end
n_snapshots
min_required_n
p50
p90
p975
max_observed
regime_filter
status
```

Any cross-year audit table missing `parent_block_id`, `parent_block_type`,
`parent_hours`, `parent_mean`, or `month_deviation_from_parent` is
`UNSUPPORTED`.

#### 5.F.0 Gate Specification Table

Every gate must be documented in code and report output with:

```text
gate_id
input_artifacts
population
metric_formula
threshold_source
status_logic
required_evidence_files
unit_test_fixture
required_for_promotion
```

Minimum specification:

| gate_id | metric_formula | status_logic | fixture | required_for_promotion |
|---|---|---|---|---|
| `hard_monthly_curve_repricing` | `max(abs(A @ x - q))` | `PASS <= 1e-8`, otherwise `CRITICAL` | sparse 2028 | true |
| `neighbor_level_leakage` | `max(abs(solution(neighbor+C)-solution(neighbor)))` | `PASS <= 1e-8`, otherwise `CRITICAL` | neighbor +1000 | true |
| `residual_vs_implied_comparable_block` | same-month deviation delta between seasonal sub-block parents (`residual|calendar` or `quarter|calendar`) and full-CAL parents, thresholded by parent-type pair | `WARNING > P90`, `CRITICAL > P97.5`, insufficient parent-type sample `UNSUPPORTED` | 2028 CAL+Q1 residual and 2028 Q4 vs 2029 CAL | true |
| `same_month_rank_consistency` | sign/rank z-score after comparable-parent adjustment | `WARNING > P90`, `CRITICAL > P97.5`, insufficient sample `UNSUPPORTED` | Apr/Dec 2028-2029 | true |
| `calendar_spread_seasonal_decomposition` | decomposition residual between CAL spread and weighted seasonal block spreads | `WARNING > P90`, `CRITICAL > P97.5` | CAL28/CAL29 | true |
| `historical_quantile_shape_outlier` | `abs(metric - hist_median) / hist_dispersion` | historical P90/P97.5 | synthetic outlier | true |
| `monthly_shape_regression_2028_2030` | max targeted monthly gate severity for 2028-2030 focus population | `CRITICAL` if any targeted numerical gate is `CRITICAL` | known bad/coherent | true |
| `lambda_calibration_artifact_present` | selected config hash matches defaults | mismatch `CRITICAL` | config mutation | true |
| `point_in_time_data_contract` | all inputs satisfy `available_at <= run_timestamp` | violation `CRITICAL` | future quote fixture | true |
| `production_export_path_parity` | production/export monthly solver outputs hash-equal under same inputs | mismatch `CRITICAL` | parity fixture | true |

#### 5.F.1 Same-Month Cross-Year Rank Gate

This gate addresses the user's concrete observation: if `CAL28` is materially
above `CAL29`, then April/December relationships cannot be accepted blindly
when they contradict the economically comparable seasonal decomposition.

For each month `m` and adjacent years `y, y+1`:

```text
calendar_spread = CAL_y - CAL_{y+1}
month_spread = x_{y,m} - x_{y+1,m}
rank_metric(y,m) = x_{y,m} - parent_mean(y,m)
shape_delta = rank_metric(y,m) - rank_metric(y+1,m)
expected_spread_distribution =
  historical / panel distribution of same-month spreads conditional on
  comparable calendar and seasonal block spreads
```

Required decomposition columns:

```text
month
year_a
year_b
calendar_spread
parent_block_a
parent_block_b
parent_mean_a
parent_mean_b
parent_mix_adjustment
month_spread
shape_delta
expected_sign
actual_sign
z_score_or_quantile
quote_support_type
supporting_quote_keys
quote_support_value
quote_support_rule
status
```

Flag:

- `CRITICAL` if sign inversion is outside historical P97.5 or panel robust
  envelope and there is no deterministic active-quote support;
- `WARNING` if outside P90;
- `UNSUPPORTED` if the conditional sample is too small.

Quote support can suppress a critical only when `supporting_quote_keys` are
active hard CH constraints covering the exact month or parent block under test.
The rule must be encoded in `quote_support_rule`; free-text analyst reasoning
does not count as quote support.

This is not a hard rule that every month must preserve the calendar-spread
sign. It is a statistically calibrated evidence rule. Inversions are allowed
only when the quotes, comparable-block math, or robust panel/history evidence
supports them.

#### 5.F.2 Comparable-Block Decomposition Gate

Never compare:

```text
2028 Apr-Dec residual level
```

directly with:

```text
2029 full CAL level
```

without decomposing the full calendar into the same comparable seasonal block.
The gate must compute:

```text
implied_2029_AprDec_from_2029_CAL_and_shape_prior
2028_AprDec_residual_target
same_month_deviation_vs_parent
```

Then assess April, December and all other months on:

```text
month_deviation_from_parent
parent_block_spread
historical/panel support
```

This catches the "April 2028 near April 2029 despite different CALs" class
without imposing a naive monotonic-month rule.

#### 5.F.3 Visual Audit Is Evidence, Not The Gate

PNG diagnostics remain mandatory, but the CI gate must be numerical. The plot
inspection checklist is:

- monthly means by year;
- 2027-2030 focus;
- month-to-month deltas;
- same-month cross-year spreads;
- parent-block decomposition;
- neighbor/history envelope overlay.

The report must link each visual red flag to a numerical row in a CSV sidecar.
PNGs must annotate failed months from `audit_gates.csv`; they do not create
PASS/FAIL status by themselves.

#### 5.F.4 Required Evidence Artifacts

Every candidate PFC approval must emit:

```text
output/.../audit_manifest.json
output/.../audit_gates.csv
output/.../monthly_curve_constraints.csv
output/.../monthly_curve_priors.csv
output/.../monthly_curve_diagnostics.csv
output/.../historical_thresholds.csv
output/.../monthly_curve_diagnostics.xlsx
output/.../2028_2030_monthly_shape.png
powerbi/data/monthly_curve_diagnostics.csv
powerbi/data/audit_gates.csv
```

`audit_manifest.json` fields:

```text
monthly_curve_schema_version
git_sha
config_hash
input_quote_file_hash
history_file_hash
fixture_version
timezone
python_version
package_lock_hash
commands
generated_artifact_hashes
solver_status
gate_summary
```

All Power BI sidecars for monthly solver outputs must include
`monthly_curve_schema_version` for forward-compatible report parsing.

Acceptance:

- the original bad curve fails;
- the current cross-year post-processed curve is not automatically passed if it violates comparable-block logic;
- a synthetic economically coherent sparse curve passes.

No post-solve visual patching invariant:

Once `solve_monthly_forward_curve` returns `monthly_curve`, no downstream
module may alter monthly BASE levels except deterministic conversion to
hourly/15-minute granularity and final calibration to original CH traded
quotes. Visual artifacts are generated evidence linked to `audit_gates.csv`;
they do not create or block PASS/FAIL independently. If a numerical gate fails,
the model specification, priors, or weights must be changed and the curve
rerun; individual months or years must not be patched.

### Phase G - Migration / Deprecation

Initial release:

- flag OFF by default;
- local export and production both support flag ON;
- Power BI shows solver diagnostics.

Promotion criteria:

- 2028-2030 visual artifacts generated and linked to `audit_gates.csv` rows;
- `monthly_shape_regression_2028_2030` is `PASS` for calibrated, supportable
  populations, or `UNSUPPORTED` only for explicitly documented far-horizon
  populations where `historical_thresholds.csv` proves insufficient monthly
  market evidence;
- EEX residuals exact;
- gates `0 critical`;
- required gates have `0 unsupported` on near-horizon / historically
  calibrable populations;
- far-horizon `UNSUPPORTED` may be accepted only when all of the following
  hold: threshold generation was attempted point-in-time, the threshold row
  proves insufficient sample rather than pipeline failure, known-bad fixtures
  still fail, hard numerical gates still `PASS`, and the audit manifest names
  the residual model risk;
- analyst commentary may explain a `CRITICAL` or `UNSUPPORTED` status, but may
  not convert it to `PASS`;
- performance overhead < 5 seconds for monthly solve;
- no material regression in hourly shape audits.

After promotion:

- default flag ON for CH LT;
- document legacy cascade as fallback;
- remove or demote cross-year post-processor.

## 6. Required Tests

### Unit Tests

```text
tests/test_monthly_forward_curve_constraints.py
tests/test_monthly_forward_curve_priors.py
tests/test_monthly_forward_curve_solver.py
tests/test_monthly_forward_curve_integration.py
```

Must cover:

- hard repricing;
- residual bucket calculation;
- no neighbor absolute level leakage;
- robust outlier handling;
- DST/leap-year weights;
- sparse 2028 case;
- missing neighbor market data;
- infeasible quotes;
- flag OFF behavior.
- path parity between production and local export under the monthly-solver flag;
- fail-closed behavior for unsupported history/panel coverage;
- lambda calibration artifact required before production-default promotion.
- conflicting calendar plus sub-products raises infeasible diagnostics rather
  than dropping a quote;
- `x_prior` cannot depend on neighbor absolute levels or legacy post-processed
  output;
- KKT/nullspace solution is unique and reports condition number;
- lambda calibration is reproducible from fixed historical snapshots;
- numerical gate failure does not mutate the curve through a post-solver patch;
- monthly solver imports no `pfc_shaping.ct.*` and no deprecated
  `pfc_shaping.model.*` path.
- quote coverage by market/horizon is reproduced and stored;
- `shape_fused` reliability labels match available evidence
  (`PANEL_MULTI_MARKET`, `DE_SINGLE_MARKET`, `STRUCTURAL_ONLY`,
  `UNSUPPORTED`);
- sparse-year proof run fails if 2028-2030 remains monthly-incoherent despite
  exact EEX repricing.

### Integration Tests

Use the local export path and, if feasible, a production-phase dry run:

```powershell
pytest tests/test_export_local_test_ch_hourly_csv_script.py
pytest tests/test_audit_ch_hfc_seasonal_coherence_script.py
pytest tests/test_lt_monthly_curve_pipeline.py
```

### Audit Reproduction

For candidate approval:

```powershell
python scripts/export_local_test_ch_hourly_csv.py ... --enable-monthly-forward-curve-solver
python scripts/audit_ch_hfc_seasonal_coherence.py ... --cross-year-output ...
python scripts/audit_ch_pfc_hourly_shape.py ...
python scripts/audit_ch_hfc_vs_spot_shape.py ...
python scripts/plot_ch_hfc_diagnostics.py ...
powershell -File powerbi/refresh_powerbi_data.ps1 -Csv ...
```

## 7. Acceptance Checklist

Block merge if any item fails:

- CH BASE repricing exact for active quote-aware products.
- CH PEAK repricing exact when PEAK calibration is enabled.
- No neighbor level leakage under +1000 EUR/MWh shift test.
- 2028 sparse case no longer passes with cloned/incoherent monthly ranking.
- Monthly solver report identifies active constraints and priors.
- Production path and local export path call the same monthly solver under flag.
- No LT import from `pfc_shaping.ct.*`.
- Flag OFF behavior protected.
- Power BI summary points to intended CSV.
- Generated artifacts are not required for code tests.
- Lambda calibration report exists for the selected defaults.
- Run manifest proves point-in-time data usage.
- Same-month cross-year rank and comparable-block gates have no critical flags.
- Required gates have no `UNSUPPORTED` status on near-horizon / historically
  calibrable populations.
- Far-horizon `UNSUPPORTED` is accepted only when it is backed by an attempted
  point-in-time threshold calibration, explicit insufficient-sample evidence,
  and a named promotion risk entry. It must not hide a `CRITICAL` on known-bad
  fixtures.
- Any analyst override is explicit, named and traceable, but cannot change gate
  status.

## 8. Open Decisions for Desk / Analyst

1. Should v1 use all neighbor markets `DE, FR, AT, IT`, or a shrinkage prior where CH history dominates and neighbors contribute only when quote coverage is current?
2. Should the objective operate in EUR/MWh deviations or relative/log deviations? v1 recommends EUR/MWh for negative-price robustness.
3. What historical lookback is acceptable: 3y, 5y, 6y, or regime-weighted?
4. Should PEAK monthly shape be in v1 or explicitly v2?
5. Which optional, non-promotion diagnostics should be reported for analyst
   context beyond the required gates in Phase F?
6. Which statistic defines the robust panel center: weighted median,
   Huberized mean, or shrinkage median to CH history?
7. What minimum neighbor quote coverage is required before a market contributes
   to the current-shape prior?

## 9. Proposed Work Breakdown

Patch 1: constraints and tests.

- add `monthly_forward_curve.py` skeleton;
- implement month index, hour weights, active constraints, residual logic;
- add `DeliveryGrid`, `MarketQuote`, `MonthlyConstraintSystem` adapter;
- unit tests.

Patch 2: priors.

- history prior;
- coverage report and reliability weights;
- neighbor panel / DE_SINGLE_MARKET prior;
- structural/climatological CH shape prior;
- fused `shape_fused` prior;
- no-level-leakage tests.

Patch 3: solver.

- KKT solve;
- nullspace diagnostics;
- diagnostics;
- sparse 2028 fixture.

Patch 4: sparse-year proof candidate.

- diagnostic 2028-2030 monthly curve;
- exact repricing / no leakage / comparable-block checks;
- PNGs linked to numerical rows;
- no production promotion.

Patch 5: lambda calibration package.

- rolling-origin backtest harness;
- withheld-product scoring;
- L-curve / Pareto report;
- persisted selected defaults.

Patch 6: integration behind flag.

- production path;
- local export path;
- report tables.
- run manifest.

Patch 7: audits and Power BI sidecars.

- comparable-block gates;
- historical quantile thresholds;
- summary metrics.
- same-month cross-year spread diagnostics.

Patch 8: frozen governance fixtures.

- compact known-bad monthly curve preserving the 2028-2030 failure;
- compact known-coherent sparse curve;
- tests proving the bad fixture fails and coherent fixture passes.

Patch 9: candidate generation and audit package.

- regenerate PFC;
- PNGs;
- audit reports;
- analyst sign-off note.

## 10. Definition of 10/10 Plan

This plan is 10/10 only if external auditors agree that:

- it fixes the monthly level formulation, not just a 2028 symptom;
- it is literature-aligned;
- it preserves CH EEX as hard constraints;
- it uses neighbors in shape space only;
- it is calibrated to actual quote coverage by market/horizon instead of
  assuming far-horizon monthly quotes exist;
- it unifies production and local export paths;
- it has explicit tests for the observed failure mode;
- it has a safe migration path with flag OFF protection;
- it bypasses legacy cascade/MSFC level rewriting when solver mode is active;
- it defines objective weights and thresholds through masked point-in-time
  calibration/backtest, not taste;
- it has known-bad and known-coherent fixtures;
- it fails closed on required `CRITICAL` gates and on `UNSUPPORTED` gates for
  calibrable populations, while explicitly documenting far-horizon
  `UNSUPPORTED` where the market history cannot support P90/P97.5 thresholds;
- it emits reproducible manifests and machine-readable evidence for PNG,
  workbook and Power BI views.
