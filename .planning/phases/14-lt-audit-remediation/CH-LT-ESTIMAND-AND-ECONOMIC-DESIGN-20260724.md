# CH LT estimand and economic design v1

## Decision

The canonical JSON beside this note is a structural, fail-closed draft. It
defines the full delivered CH LT target before any new candidate, data
acquisition, MDE decision, or T057 consumption. Schema v1 can validate its own
shape, but can never authorize evaluation, production, or promotion.

Canonical identities after the independent roast fixes:

- source document SHA-256:
  `4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931`;
- canonical semantic SHA-256:
  `41ce07d1cf04e77a6936dc2d6f6fece387cbb415aa4588fa40406072e741b384`;
- contract ID:
  `da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344`;
- validator policy SHA-256:
  `52c90167c51779724509d8a69ecc368c77a547f2eb2f3f55dbac10b98185a276`.

The schema closes the exact top-level and lifecycle key inventories. Unknown
approval or authority fields are rejected even if the document is rehashed.

## Estimand layers

1. **Market consistency**: the delivered curve must reprice the CH EEX products
   available at the origin. The monthly solver remains the level authority.
2. **Hourly shape**: direct CH day-ahead truth is scored only after prediction
   and truth are separately energy-centered inside each complete local delivery
   month.
3. **Quarter-hour shape**: the same centered estimand is used, but only against
   native CH 15-minute truth for one market product frozen before development,
   selection, tuning or scoring.
   Duplicated or interpolated hourly prices are not quarter-hour truth.
4. **Probabilistic scenarios**: joint paths, not independent marginal noise,
   preserve the solver forward as their energy-weighted ensemble expectation.
   Scenario-specific monthly-level risk is permitted only with ensemble-zero
   mean and a frozen covariance design; fixed-level pathwise scenarios are a
   separate shape-only diagnostic. CRPS/WIS/pinball,
   calibration/coverage, energy and variogram scores, dependence/coherence and
   Monte-Carlo error are all required by a frozen external design.

The primary shape loss is level-neutralized MAE in EUR/MWh, equally weighted by
origin after within-origin energy weighting. Repricing is a separate hard gate;
an unsupported truth layer cannot disappear inside an aggregate pass.

Origins are unique `as_of_utc` inference units. Their cadence, issuance
calendar, overlap clusters, minimum counts by horizon/stratum and multiplicity
family must be fixed by the statistical design manifest. Adding dense origins
cannot silently change the weighting or inflate effective sample size.

## Economic populations

No FMV profile value is invented in this contract. FIL and ACC require exact,
versioned volume rows from an external population manifest. BLOC 13 is treated
as a contract payoff requiring frozen legs, strikes/collars, settlement and
sign conventions; it is not approximated as a generic physical profile.
Hydro dispatch uses the same optimizer class, information set and physical
constraints across models, while allowing each model to choose its own
non-anticipative actions. Regret is against a feasible clairvoyant policy with
the same initial/terminal state and constraints; ex-post recourse is forbidden
unless the frozen policy allowed it. FIL/ACC cashflows use full delivered prices
and an explicit generation/consumption sign. Every CHF metric requires a frozen
FX fixing and conversion convention.

`pfc_shaping.lt.model.pfc_flavors.DEFAULT_CAPTURE_PREMIUM` is explicitly
forbidden as validation or economic-materiality evidence. OMPEX remains a
benchmark only and is never truth, model input, selection evidence, or a gate.

## Current fail-closed blockers (15)

- market layer: CH EEX vintage, solver configuration, candidate/baseline
  identity and post-evaluation market-consistency audit manifests absent;
- truth/design: direct CH hourly and native quarter-hour truth, exact
  statistical design, versioned calendar/strata and probabilistic scenario
  manifests absent;
- economics: exact FMV FIL/ACC/Bloc 13 population/payoff, hydro policy,
  valuation/FX convention and FMV Risk MDE approval absent;
- sampling/governance: exact origin/target/mask inventory and independent
  external admission envelope absent.

Until all bindings are independently supplied and a later schema implements
external admission, scientific execution and production remain `NO_GO`.

## Supported audit command

The installed entry point is `pfc-lt-audit-estimand`; checkout-compatible module
forms are `python -m pfc_shaping.cli.audit_ch_lt_estimand_contract` and
`python -m scripts.audit_ch_lt_estimand_contract`. `validate-draft` returns 0
only for structural validity; default `admit-evaluation` returns 3. Invalid
input returns 2. Durable output additionally requires absolute `--audit-root`
and `--output` paths in the pre-provisioned
`<audit-root>/phase14/ch_lt_estimand_contract_audits/*.json` namespace.
