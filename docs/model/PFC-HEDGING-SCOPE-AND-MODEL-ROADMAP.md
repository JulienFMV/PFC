# PFC hedging scope and model roadmap

## Business target and current boundary

The target use is market valuation and hedge-risk management for CH and DE,
with FR, AT and IT-North first serving as observation/risk markets and becoming
execution markets only after access, liquidity, products, credit and risk
limits are independently admitted.

This target must not be confused with current implementation authority:

- the hard monthly BASE solver is configured only for CH;
- the pipeline also builds a DE branch, but its monthly level still follows the
  legacy path and it has no CH-equivalent governed monthly-solver promotion;
- FR, AT and IT-North are not active LT output branches;
- no curve, including CH, is empirically promoted while the governed data and
  future-holdout gates remain blocked.

Each future market curve must therefore have its own forward-level authority.
Neighbour prices and spreads may shape another market only after zero-mean
normalization; they cannot set its level.

## Validated FMV target products

The user validated the following target architecture on 2026-09-02. The LT
program must produce separate, traceable products rather than make one curve
carry incompatible meanings:

1. `market_central_ch`: a quarter-hour Swiss valuation curve representing a
   forward-consistent central view of future spot prices;
2. `fundamental_scenario_ch`: explicitly labelled conditional curves for dry,
   wet, cold, mild, PV-penetration, electrification, capacity, outage and
   cross-border regimes;
3. `stochastic_spot_paths_ch`: temporally coherent paths around each admitted
   scenario for tail, negative-price and capture-risk analysis;
4. `portfolio_valuation_and_hedge`: a downstream use of the curves with FMV
   hydro/PV production, Valais client and large-industrial load profiles, open
   positions and tradable hedge instruments.

The central market curve is not a claim to predict the exact future spot
realisation. It is a market-consistent conditional central surface. A
shape-only scenario remains zero-mean inside the applicable solver bucket. A
scenario that changes monthly, quarterly or annual levels is a separately
labelled fundamental scenario and must be resolved upstream under its own
explicit assumptions; it must never overwrite the central EEX curve after the
monthly solve.

CH is the first production target. DE is the first hedge and spread market,
but requires its own governed EEX level authority before it can be promoted as
a peer output. FR, AT and IT-North remain observation/risk markets until
market-access, liquidity, products, credit and risk gates are independently
admitted.

## Verified current model inventory

| Layer | Code path actually called | Audit interpretation |
|---|---|---|
| CH monthly level | constrained quadratic monthly solver / direct KKT solve | sole governed level authority |
| Hourly shape | configured `ShapeHourlyMLP`, 64×64 ReLU | active champion candidate, not empirically proven champion |
| Day-type shape | normalized calendar ratios carried by hourly layer | shape only |
| Quarter-hour shape | DE-LU post-2025-10 Huber base plus OOS-gated Ridge corrections | robust transfer baseline; CH transfer risk remains |
| Market hydro shape | `WaterValueCorrection` fitted on aggregate reservoir anomaly and CH prices | market-shape proxy, not FMV asset water value |
| Future fundamentals | ENTSO-E climatology | conservative baseline, not a scenario model |
| Probabilistic output | disabled | deterministic candidate only |

One implementation caveat is material for future comparison: the MLP computes
180-day decay weights, but `MLPRegressor.fit` receives no sample weights.  The
weights are used while averaging quarter-hours inside each hourly observation;
they do not materially reweight older hourly observations against recent ones.
The statement that the final MLP fit has an effective 180-day half-life is
therefore unproven.  This must be tested as a preregistered challenger or
ablation, not patched into the baseline before the evaluation protocol is
frozen.

## Consequences for the ENTSO-E export work

1. `realized_final` is an ex-post target/control contract.  It may support
   shape training, spread evaluation and hedge-risk metrics only after exact
   finality and independent price reconciliation.
2. `causal_asof` is a feature/backtest contract.  It may contain only values
   demonstrably available at each rolling origin.
3. Market purpose is explicit per selected field: valuation/hedging,
   observation/risk or future candidate.  Purpose never grants trading,
   model-input or production authority.
4. Five-zone qualification remains useful for coupled-market consistency, but
   a CH/DE bounded export need not be blocked merely because an observation-only
   market is absent.
5. A Swiss market month is delimited in Europe/Zurich and may span two UTC
   partitions; DST cannot be reduced to 24 hours per day.

## Execution order

1. Finish the governed EEX and ENTSO-E/LSEG export chain, including source
   truth, availability, revision/finality and immutable manifests.
2. Freeze target definitions, rolling origins, metrics, thresholds and a new
   untouched future holdout.
3. Replay the current CH candidate unchanged as the baseline champion.
4. Compare preregistered hourly challengers on identical origins: calendar
   baseline, Ridge/ARX, GAM, LightGBM, current MLP, correctly recency-weighted
   MLP, then a restrained ensemble.
5. Select the quarter-hour layer separately, with native CH truth as the
   decisive gate and DE-LU transfer as a baseline.
6. Build probabilistic P scenarios and market-consistent Q valuation curves as
   distinct products.
7. Feed those price scenarios into a separate FMV hydro optimizer.  Reservoir
   cascades, inflows, efficiencies, pumping/turbining, terminal water value and
   risk appetite belong there, not in the market PFC level.
8. Promote only after independent holdout, shadow run, manifest parity and
   rollback evidence.

After those price-model gates, build two separate downstream decision layers:

9. an FMV hydro optimiser receiving price and inflow scenarios and returning
   water values and operating decisions without rewriting the market PFC;
10. a hedge optimiser receiving open physical/contract positions, CH/DE price
    scenarios, liquidity, bid-ask, credit and risk limits and initially
    producing shadow recommendations only.

No model retraining or selection is authorized by this roadmap.  The global
status remains `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`, and T057 stays
sealed.
