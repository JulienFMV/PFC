# Session handoff — EEX short-tenor outcome-blind scaffold (D223)

Date: 2026-08-05  
Scope: LT only, local structural/algebraic evidence, no Databricks execution

## Outcome

D223 retains the dormant D222 combination algebra and adds a fail-closed,
outcome-blind future-selection scaffold. It fixes candidate/baseline families,
native-regime loss, mandatory diagnostics, fold separation, dependence and
multiplicity policy while deliberately leaving every evidence-dependent numeric
decision unset.

Policy status:
`PASS_LOCAL_STRUCTURE_ONLY_NO_EMPIRICAL_EXECUTION`.

Canonical policy content ID:
`00a2b2087589d14ef1330cfa0de109fe9dfcce81436e0a0265fb96067d10fbb6`.

Selected proof bundle:

`build/databricks-eex-daily/2026-08-05/short-tenor-combination-contract-proofs/c06dfcf5fbda16cd5bab04005d4581e35216116c30c6c9df6cef94f2d949f67b/`

The earlier D222 proof `21c557df...` is mathematically green but no longer
selected: inspection proves that its manifest binds the descriptive D220
identity `7a4c2a...`. D223 selects `c06dfcf...`, whose manifest binds the
governed D220 bundle `ae7b962c...` and manifest SHA-256 `fd5af41e...`.

## Frozen structural decisions

Candidate families:

- `ROBUST_REGULARIZED_LINEAR_GAM_SHAPE_V1`;
- `MONOTONE_HIST_GRADIENT_BOOSTING_SHAPE_V1`.

Baselines:

- `MARKET_CONSTRAINED_SEASONAL_NAIVE_V1`;
- `INCUMBENT_CH_LT_FROZEN_V1`.

Primary loss:
`NATIVE_REGIME_WITHIN_MONTH_SHAPE_MAE_EUR_MWH`.

Mandatory diagnostics include bias, RMSE, median absolute error, ramps, tails,
negative-price, peak/offpeak, weekday/weekend/holiday, season, DST, horizon and
vintage stability. Selection must be nested inside each monthly outer origin,
use the same targets/masks, refit coefficients inside the origin, use a
contrast-specific overlap-aware studentized stationary block bootstrap and
Holm FWER control. An MCS is diagnostic only.

## Deliberately unset numeric decisions

The following remain `null` and cannot act as defaults:

- minimum inner origins and effective clusters;
- embargo months;
- coefficient lattice and coefficient cap;
- per-component and combined-signal caps;
- superiority and noninferiority margins;
- marginal power floors;
- required unique outer origins by contrast.

They require a direct-CH dependence/power design and external trusted-time
freeze before outcomes. OMPEX cannot supply any of them.

## Fail-closed policy checks

`validate_short_tenor_combination_policy` rejects:

- the non-selected D220 proof identity or stale successor-core binding;
- a premature numeric sample size, embargo, cap or margin;
- candidate-family, baseline, primary-loss or mandatory-diagnostic drift;
- OMPEX training/tuning/selection/cap/margin use;
- AFRY numeric use, sealed T057 access or future-holdout opening;
- local non-PIT EEX as rolling-origin evidence;
- legacy or synthetic ENTSO-E substitution;
- model-input, assembly or production authority escalation.

The algebraic combiner additionally rechecks the explicit combined-signal cap
after the final D219 projection. It still supplies no defaults, performs no
silent clipping, authenticates no receipt and remains absent from LT
orchestration.

## Scientific alignment

Primary sources used for the structural design:

- Lago et al. (2021), doi `10.1016/j.apenergy.2021.116983`;
- Ziel and Weron (2018), doi `10.1016/j.eneco.2017.12.016`;
- Giacomini and White (2006), doi `10.1111/j.1468-0262.2006.00718.x`;
- Romano and Wolf (2005), doi `10.1111/j.1468-0262.2005.00615.x`;
- Hansen, Lunde and Nason (2011), doi `10.3982/ECTA5771`.

These references justify common-sample out-of-sample evaluation, point-in-time
forecast-origin information sets, dependence-aware inference, multiple-testing
control and explicit selection uncertainty. They do not justify a coefficient
or performance claim from the current local panel.

## Evidence and hashes

Selected proof:

- content ID:
  `c06dfcf5fbda16cd5bab04005d4581e35216116c30c6c9df6cef94f2d949f67b`;
- manifest SHA-256:
  `b76eb64b493a154c66def95a2dc13c448f2f93f048f2a3a45e5caf057e7783c8`;
- summary SHA-256:
  `5f68ca8c6089df2cacb76db202160720655fe6b3e3ffdcf2522e04e41e9b6db3`;
- residual Parquet SHA-256:
  `4232684e7dda0fb223474f8a74a1bb953f80a9118662f4f953fb388a9c48be27`.

Implementation:

- combination contract SHA-256:
  `0be8ab62cbaba4c28de14179a749d47046b8e41aa967da8c06a2d49a5bc250ba`;
- module SHA-256:
  `311a34c154a93cb3389debee2edb948117428b94c1fd3890b7b28bcac5eb7f6e`;
- tests SHA-256:
  `f5cb2bc621d897af54c741b4383d48de1f9c16a00a2634a07b47c7bb8a54dcb8`;
- Ruff-clean build-only materializer SHA-256:
  `a17304d563df43e911568af67dd17fa3628f51e33ea23b0ea0da21563fa172c6`;
- D219 projection dependency SHA-256:
  `951274f220ac7b5d3dc4a992ec83e672cc10e6b0b28036ecfca1336a2cbb981e`.

Three materializer replays created or re-adopted the same selected content ID.
The six persisted algebraic cases cover 1h/15min, both DST transitions,
BASE-only and positive/negative/zero coefficients. Maximum residuals:

- linearity: `0 EUR/MWh`;
- active constraints: `8.31279489688086e-15 EUR/MWh`;
- monthly means: `2.6981399962886943e-15 EUR/MWh`.

## Commands and results

Focused:

```text
python -I -B -m pytest tests/test_short_tenor_combination_contract.py ... -q
33 passed in 1.50s
```

Adjacent D219/D220/OMPEX/ENTSO-E/LT matrix:

```text
127 passed, 1 skipped in 49.91s
```

Ruff:

```text
All checks passed!
```

The materializer was run through Python with all mutable paths under `build/`.
Databricks request count, Warehouse starts, proof network calls and remote writes
were all zero.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-COMBINATION-CONTRACT-V1.json`;
- `pfc_shaping/lt/model/short_tenor_combination_contract.py`;
- `tests/test_short_tenor_combination_contract.py`;
- build-only `build/databricks-eex-daily/materialize_short_tenor_combination_contract.py`;
- `docs/research/forwards_sources.md`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D223);
- `.planning/HANDOFF.md`;
- this handoff.

## Authority and remaining blockers

- Point-in-time EEX availability: not proven.
- Governed ENTSO-E empirical input: unavailable.
- Numeric preregistration decisions: intentionally unset.
- Training and selection receipts: unauthenticated.
- Future independent holdout: not frozen and not consumed.
- Model input, candidate assembly, promotion and production: false.

The monthly solver remains sole level authority. OMPEX remains closed during
fit/selection and post-freeze benchmark-only. AFRY remains descriptive, T057
sealed and local/synthetic ENTSO-E substitution forbidden.

## Next safe batch

Define and adversarially test the hash-bound training/selection receipt schema
that will prove per-origin PIT cutoffs, fold isolation, source identities,
candidate-grid identity and no outcome access. Keep all numeric hyperparameters
null until direct-CH power evidence and external freeze exist.
