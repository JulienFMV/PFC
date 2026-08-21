# PFC 2026 literature evidence and falsifiable gates

Date: 2026-07-17

Status: normative scientific doctrine for Phase 14 challengers and future
production-readiness work. It refines, but does not weaken, the
`PFC-FMV-PRODUCT-QUALITY-CHARTER-20260713.md` invariants.

Production decision: **NO_GO**. This document does not approve a model,
candidate, data vintage, package, runtime or promotion.

## Scope and product definition

The delivered LT object is an hourly / 15-minute Swiss **forward valuation
curve**. It must reprice the market products used as hard constraints. It is
not a claim to know future spot prices. Spot, weather, hydro, load, renewable
and neighbouring-market information may improve the allocation of a fixed
market-consistent level across delivery intervals, or help quantify risk, but
may not silently replace the CH EEX level authority.

The product is accepted only when all three layers are separately defensible:

1. **Market level:** eligible point-in-time CH EEX products and the governed
   monthly solver.
2. **Conditional shape:** zero-mean-within-authority-bucket adjustments whose
   incremental value is proven out of sample.
3. **Uncertainty/scenarios:** calibrated predictive distributions and coherent
   paths, assessed independently of the point curve.

No score in layers 2 or 3 can compensate for failed market repricing,
provenance, freshness, solver residuals, atomic publication or rollback.

## Evidence docket

### Supplied local sources

The source PDFs were read from the user's desktop and were not modified.
Extracted text was written outside the repository under
`%LOCALAPPDATA%\Temp\pfc-literature-20260717`.

| Source | SHA-256 | Pages | Evidentiary use | Limitation |
| --- | --- | ---: | --- | --- |
| `Empirical Methods Swiss Market.pdf` | `2fae3e1d892fed3dd706851675fc64085828e71c56f069edd7afaa406ab08999` | 74 | Swiss hydro, interconnection, calendar, weather, BASE/PEAK calibration and robust/nonlinear shape challengers | Historical study around 2010; it predates usable CH futures in its setup and is not current production evidence |
| `energies-17-05797.pdf` | `78693f11d201b875cd564baed0605b21b5e2e6aad99eae33dc9812aa51715796` | 37 | Model taxonomy, exogenous drivers, regime adaptation, ensembles, interpretability and cross-validation | Narrative 2024 review mixing load and short-term spot-price tasks, markets, data and metrics; not proof for a Swiss LT HPFC |
| `Benth Shaping.pdf` | `a9352e38dc8d3cdd071f5ecb62bfc27154d26902efcfa105885cf373ffb869aa` | 5 | Smooth forward construction around an explicit seasonal prior under exact market-average constraints | Foundational excerpt; it does not supply FMV-specific operational thresholds or current data evidence |

Three additional user-supplied works were reviewed in the preceding session:
Audun Saethero's HFC thesis (`HFC1_DissASSaethroe.pdf`), Iago Sichinel
Chavarry's HFC report (`HFC2_ELE_Iago Sichinel Chavarry.pdf`) and Kiesel,
Paraschiv and Saethero, *On the construction of hourly price forward curves
for electricity prices* (`OffprintKieselSoetheroParaschiv.pdf`, DOI
`10.1007/s10287-018-0300-6`). The PDFs are not copied into this repository:
they may be copyright-restricted and are evidence inputs, not runtime assets.
Their former desktop paths were no longer present when the docket was resumed
on 2026-07-23, so hashes must be captured again if the local documents are
reattached; no unverifiable hash is asserted here.

Their actionable contribution is retained as falsifiable doctrine:

- curve construction is a constrained inverse problem whose regularisation
  and basis complexity are model-risk parameters, not implementation details;
- equivalent cascades of the same market information must lead to the same
  final curve, within declared numerical tolerance;
- the quote-to-curve Jacobian must be reported and stressed so that local quote
  changes cannot create unexplained remote oscillations;
- flexibility must be governed by effective degrees of freedom, conditioning,
  stability and out-of-sample gain, not only in-sample fit;
- scenarios and sensitivities must operate in the null space of hard market
  constraints, preserving monthly solver authority and exact repricing.

### Corroborating primary literature

- Fleten and Lemming, *Constructing forward price curves in electricity
  markets* (Energy Economics): market bid/ask information constrains a
  high-resolution curve while a bottom-up model supplies seasonality and a
  quadratic objective supplies smoothness.
  <https://doi.org/10.1016/S0140-9883(03)00039-2>
- Lago, Marcjasz, De Schutter and Weron, *Forecasting day-ahead electricity
  prices: a review of state-of-the-art algorithms, best practices and an
  open-access benchmark* (Applied Energy): fair comparison requires common
  datasets, long out-of-sample periods, strong simple benchmarks, suitable
  metrics and statistical comparison.
  <https://doi.org/10.1016/j.apenergy.2021.116983>
- Cerasa and Zani, *Enhancing electricity price forecasting accuracy: A novel
  filtering strategy for improved out-of-sample predictions* (Applied Energy,
  2025): robust filtering is evaluated in rolling windows on six markets, with
  code and data available for replication.
  <https://doi.org/10.1016/j.apenergy.2025.125357>
- Gneiting and Raftery, *Strictly Proper Scoring Rules, Prediction, and
  Estimation* (JASA): probabilistic forecasts require proper scores that reward
  honest distributional forecasts.
  <https://doi.org/10.1198/016214506000001437>
- Scheuerer and Hamill, *Variogram-Based Proper Scoring Rules for
  Probabilistic Forecasts of Multivariate Quantities*: energy and variogram
  scores assess ensemble distributions, with the variogram score adding
  sensitivity to dependence errors.
  <https://doi.org/10.1175/MWR-D-14-00269.1>
- Hilger et al., *Multivariate scenario generation of day-ahead electricity
  prices using normalizing flows* (Applied Energy, 2024): full-path scenarios
  and periodic retraining address temporal dependence and changing regimes.
  <https://doi.org/10.1016/j.apenergy.2024.123241>
- Phan et al., *Electricity price forecasting across Norway's five bidding
  zones in the post-crisis era* (2026 preprint): strictly causal rolling-origin
  evaluation, feature-group ablation and conditional hydro/gas regime analysis
  are useful current challenger patterns. As a preprint and a day-ahead Nordic
  study, it is corroboration rather than a normative authority.
  <https://arxiv.org/abs/2604.26634>

## Direct conclusions from the literature

1. A smooth HPFC is an inverse problem: observed products provide aggregate
   constraints and an explicit prior provides unobserved fine structure.
2. Smooth the **deviation from** the seasonal prior, not the full curve blindly;
   otherwise useful seasonality can be erased.
3. Market-product averages are constraints (or explicit bid/ask feasibility
   intervals), not soft targets that a later shaping layer may overwrite.
4. The seasonal/bottom-up prior is model risk. It must be versioned,
   sensitivity-tested and prevented from leaking absolute level.
5. Swiss shape drivers include hydro/reservoir state, cross-border dynamics,
   temperature, load, calendar, holidays and intraday/weekly seasonality.
6. A nonlinear or ensemble model is only a challenger. Heterogeneous papers do
   not establish universal superiority over robust linear or seasonal models.
7. Honest comparison requires common vintages and origins, genuinely unseen
   evaluation, strong simple benchmarks, regime analysis and uncertainty on
   performance differences.
8. Point metrics alone do not validate uncertainty. Marginal calibration and
   multivariate temporal dependence require distinct diagnostics.
9. Equivalent quote decompositions and cascades are metamorphic tests of the
   solver specification; repricing alone cannot detect a decomposition-sensitive
   regulariser.
10. Quote sensitivity is part of curve governance: every accepted quote needs
    a finite, reproducible bump response with bounded spillover and exact
    derivative identities for constrained products.

## FMV 2026 uplift (explicit inference, not a quoted universal standard)

The following gates combine the literature with FMV's valuation, security and
operability requirements. Numerical tolerances are product policy and must not
be presented as universal values supplied by the papers.

### G0 — immutable point-in-time evidence

- Every input row used by training, selection or generation is admissible at
  the declared origin: `available_at <= as_of`; revisions and acquisition time
  remain reconstructible.
- The run binds source snapshot, schema, transform code, configuration,
  environment/image and output hashes.
- Leakage tests deliberately inject future revisions, late publications and
  same-file aliases; admission must fail closed.
- Stale, partial or ambiguous critical inputs produce a governed failure, not
  an imputed success. Missing is never encoded as economic zero.

### G1 — exact market authority

- Reprice every supported hard CH EEX BASE and PEAK delivery product on the
  **final** delivered curve to `1e-6 EUR/MWh`; verify implied OFFPEAK identities
  to the same tolerance.
- Verify constraint feasibility, rank/conditioning, KKT/stationarity residuals
  and finite outputs. Numerical fallback is not an implicit pass.
- When `monthly_level_authority="solver"`, every downstream component is
  zero-mean within each governed monthly bucket. No month patch is permitted.
- Neighbour, history, hydro and spot features may shape but cannot provide
  absolute monthly level. A counterfactual constant shift in any such feature
  must not change governed monthly means.
- If bid/ask inputs are later governed, feasibility intervals and the policy
  selecting midpoint/side must be manifest-bound; do not silently reinterpret
  them as exact settlements.
- Rebuild the same market state through every accepted equivalent
  annual/quarter/month cascade. Final interval prices, monthly means and hard
  product repricing must be invariant to `1e-8 EUR/MWh`, or to a tighter
  solver-derived tolerance recorded in the manifest.
- Emit a quote-to-curve Jacobian (or deterministic finite-difference audit)
  for every governed quote. Bumps must preserve all unaffected hard identities,
  reproduce the bumped product derivative and remain finite; spillover by
  delivery distance is reported and thresholded before candidate selection.

### G2 — deterministic shaping evidence

- Use nested rolling-origin evaluation: feature/model/hyperparameter selection
  occurs inside each training origin; the future holdout remains untouched.
- Compare all challengers on identical origins, available-at vintages, target
  rows and eligibility masks against at least: market-constrained seasonal
  naive, robust regularised linear/GAM, current incumbent and a justified
  nonlinear/tree challenger.
- Report MAE and RMSE in EUR/MWh, bias, tail loss and weighted product/regime
  errors. Avoid MAPE as a primary gate because electricity prices can be near
  zero or negative.
- Report fold distribution and paired loss differences with confidence
  intervals or an appropriate predictive-accuracy test; an aggregate average
  alone cannot promote.
- Stratify at minimum by horizon, month/season, weekday/holiday, hour/ramp,
  BASE/PEAK/OFFPEAK, hydro/reservoir regime, cross-border/commodity stress,
  price spike and negative-price regime.
- Run leave-one-feature-group-out ablations. A complex feature family is
  retained only if its point-in-time incremental value or diagnostic value is
  demonstrated and documented.
- Report effective degrees of freedom, active basis dimension, regularisation
  weights, constraint rank/condition diagnostics and quote-bump stability for
  every challenger. Complexity is selected inside each training origin and is
  rejected when added flexibility lacks paired out-of-sample value.
- Preserve the locked T057 definition exactly. After it matures, evaluate it
  once as pre-registered; do not tune on T057. A new candidate requires a new
  future holdout.

### G3 — probabilistic and scenario evidence

- Quantiles are monotone and all scenarios preserve market-product identities
  and monthly-level authority pathwise or by a documented exact projection.
- Evaluate pinball loss by quantile and CRPS/WIS overall. Also report empirical
  interval coverage, interval width, PIT/rank diagnostics and calibration by
  horizon and regime.
- Evaluate full trajectories with energy score **and** variogram score (or a
  documented dependence-sensitive equivalent); marginally good quantiles do
  not prove coherent hourly paths.
- Compare against transparent residual/bootstrap and quantile-regression
  baselines on identical rolling origins.
- Stress scenarios for hydro scarcity/abundance, interconnector constraints,
  high/low load and renewable output, commodity shocks, spikes and negative
  prices. Scenario probabilities and conditioning data must be point-in-time.
- Promotion requires both statistical improvement and a pre-specified economic
  value test for FMV use (valuation/hedge/hydro decision), with transaction and
  imbalance assumptions explicit.

### G4 — governance and industrialisation

- One LT package and entrypoint; LT code remains independent from
  `pfc_shaping.ct.*`.
- Independent production, export and selected-artifact manifests agree on the
  exact candidate and inputs before atomic promotion.
- The publisher runtime, dependency closure and external CAS fail closed under
  alias, shadow-path, tamper, race and process-tree attacks.
- A locked, non-root, read-only-input container is reproducible by digest;
  CI runs import isolation, unit, contract, repricing, rolling-origin smoke,
  packaging, publication and deterministic artifact tests.
- Structured logs and metrics expose freshness, coverage, constraint
  residuals, drift, regime errors, calibration, runtime, publication state and
  rollback readiness. Alerts have owners and tested runbooks.
- Reproduce and rollback drills are promotion prerequisites, not post-launch
  cleanup.

## Candidate promotion decision rule

A candidate remains `NO_GO` unless:

1. every G0/G1/G4 invariant passes on fresh, immutable evidence;
2. deterministic gates in the quality charter pass across folds and critical
   regimes against frozen baselines;
3. the relevant locked future holdout has matured and passed exactly once;
4. probabilistic/scenario claims, if exposed, pass G3 independently;
5. Security, IT/Operations and Quant/Data roasts are independent, read-only,
   evidence-backed and have no unresolved P0/P1;
6. the exact artifact, manifests, image/dependency closure and rollback target
   are bound together before atomic promotion.

The current work may build and audit candidates but must not promote any of
them to production.
