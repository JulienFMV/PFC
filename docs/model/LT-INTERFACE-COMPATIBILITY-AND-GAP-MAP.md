# LT interface compatibility and gap map

## Scope and outcome

This audit covers the ten LT interfaces named in the 2026-09-02 restart
handoff.  It does not inspect or change CT, retrain a model, access Databricks,
open T057, or alter the CH monthly solver.  The current pipeline produces one
untyped deterministic CH curve plus a legacy-level DE branch.  It does not yet
produce an admitted fundamental scenario, stochastic path set, portfolio
valuation, FMV hydro decision, or hedge recommendation.

The compatible insertion point is a standalone metadata contract under
`pfc_shaping.lt`.  `pfc_shaping.lt.curve_products` therefore defines scenario
and curve-product identity without being imported by
`production_phases.py`.  Its level-source fields are requirements only and
grant no model, publication, production, trade, probability, calendar, or
monthly-level authority.

## Current interface map

| Current interface | Inputs | Outputs | Target-product interpretation | Gap that must remain explicit |
|---|---|---|---|---|
| `monthly_forward_curve.py` | Delivery months/timezone, own CH BASE quotes, optional zero-mean shape prior, solver weights | Hard-constrained monthly BASE series, constraints, priors and KKT diagnostics | Monthly level source for `market_central_ch` | Result has no curve-product ID, information timestamp or complete provenance envelope; it is not a scenario factory |
| `production_phases.py` | Governed input frames/receipts, explicit reference timestamp, config and CH solver evidence | `LongTermArtifacts` with CH/DE branch DataFrames and CH monthly manifest | Orchestrates the current central CH candidate; DE remains legacy-level | Branch artifacts are not typed as any target product; no scenario/path/portfolio/hydro/hedge output exists |
| `assembler.py` | Forward levels, delivery grid, hourly/intraday/hydro shapes and optional exogenous forecasts | Quarter-hour `price_shape` plus factor diagnostics and disabled P10/P90 columns | Numerical core of the current `market_central_ch` candidate | DataFrame carries no scenario ID, information timestamp, normalization-bucket ID or provenance contract |
| `shape_hourly.py` | CH price/calendar history and optional hydro history | Unit-mean hourly/day-type factors and trends | Shape-only input to the central curve | Not a scenario definition; normalization is local to cells/days and must still be recentered to solver months |
| `shape_hourly_mlp.py` | CH history, calendar, hydro and outage features | Daily-normalized hourly factors | Active baseline shape candidate for the central curve | No final sample reweighting by the computed 180-day weights; no product/scenario authority |
| `shape_intraday.py` | Native price truth, calendar and optional ENTSO-E/hydro features | Mean-one quarter-hour factors | Quarter-hour shape input to the central curve | Current production fit uses DE-LU transfer truth; no CH scenario or product identity |
| `water_value.py` | Aggregate CH prices, reservoir anomaly and calendar | Market-price shape factor or additive delta | Hydro-related market-shape proxy only | Not FMV asset water value, dispatch, inflow scenario or hydro optimisation |
| `msfc_spline.py` | Flat forward base levels and delivery index | Smoothed base series with period-mean checks | Legacy central-level smoothing only | Skipped on the governed solver path; must not become a scenario level authority |
| `uncertainty.py` | Historical cell residuals and a deterministic PFC | Pointwise P10/P90 bands | No target product currently; production disables it | Bands are not coherent `stochastic_spot_paths_ch` and lack Swiss rolling-origin admission |
| `pfc_flavors.py` | Mid curve and fixed additive commercial premiums | Mid/client/production price variants | Legacy downstream presentation helper | Not portfolio exposure/valuation, FMV hydro optimisation, or hedge recommendations; additive level shifts are not curve authority |

## Six target products

| Target product | Existing compatible material | Current status |
|---|---|---|
| `market_central_ch` | CH monthly solver, assembler, hourly and intraday layers | Numerical candidate exists; typed product metadata was missing |
| `fundamental_scenario_ch` | Shape mechanisms can later consume admitted assumptions | No scenario engine or admitted real-data mapping exists |
| `stochastic_spot_paths_ch` | None; legacy residual bands are not paths | Not implemented and not authorized |
| FMV portfolio exposure and valuation | A central curve can eventually be an input | Not implemented; `PFCFlavors` is not a substitute |
| FMV hydro optimisation and water values | Market hydro proxy may remain an upstream price-shape feature | Asset optimiser not implemented; no market-level rewrite permitted |
| Hedge recommendations | CH curve and a future governed DE peer may become inputs | Not implemented; market access and risk controls remain separate gates |

## Required metadata boundary

Every future typed curve needs the following metadata outside the numerical
DataFrame:

- stable product and scenario identities;
- a UTC information timestamp bounded by source availability;
- a required level source that is explicitly not an authority grant;
- the normalization bucket (`solver_month` for shape-only effects, or a
  separate upstream level solve for level-changing scenarios);
- content-addressed provenance without embedding source values;
- an immutable authority-negative state.

Shape-only scenarios must descend from `market_central_ch`, retain the CH
monthly BASE solver as required level source, and normalize within each solver
month.  Level-changing scenarios must have a separately identified upstream
fundamental solve and cannot descend from or overwrite the central curve.
Stochastic paths must descend from a typed fundamental scenario and inherit
its scenario identity, level source and normalization bucket.

## Deliberately excluded from this milestone

No current pipeline call site is changed.  The contract does not hold curve
values, execute a solver, assign scenario probabilities, infer calendars,
select models, assemble candidates, publish outputs, authorize production or
authorize trades.  The global status remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`, and T057 remains sealed.
