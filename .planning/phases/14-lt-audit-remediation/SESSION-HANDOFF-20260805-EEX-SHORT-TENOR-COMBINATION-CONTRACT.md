# Session handoff — EEX short-tenor combination contract (D222)

Date: 2026-08-05  
Scope: LT only, algebraic fixtures only, no Databricks/network execution

## Outcome

D222 admits an inactive explicit combination boundary for future D220
short-tenor components. It proves that named components with explicit scalar
coefficients can be combined without losing D219 solver neutrality. It cannot
learn weights, choose caps, authenticate receipts, assemble a candidate or
activate production.

Selected proof bundle:

`build/databricks-eex-daily/2026-08-05/short-tenor-combination-contract-proofs/21c557df75260ce162c15af6fbe0c91de4c8a6fcd233564403a04e7b1fc53ad0/`

Status:
`PASS_LOCAL_MATHEMATICAL_COMBINATION_ONLY_NO_MODEL_AUTHORITY`.

## Contract behavior

`combine_short_tenor_shape_components` requires:

- a non-empty mapping of exact named `pandas.Series` components;
- an exact coefficient key for every component and no extra key;
- finite non-boolean coefficients;
- explicit positive finite absolute caps for coefficients, individual
  contributions and the combined signal;
- six valid pairwise-distinct SHA-256 content IDs: normalization, D216 audit,
  D220 contrast bundle, policy, training receipt and selection receipt;
- complete local months, a valuation timestamp before delivery and exact D219
  monthly levels/active PEAK geometry.

It then:

1. validates and projects each component independently through D219;
2. rejects any raw or projected contribution outside its explicit cap;
3. computes the deterministic coefficient-sorted weighted sum;
4. rejects a combined-signal cap breach;
5. projects the sum through D219 again;
6. requires the final projection to equal the sum of individually projected
   components within `1e-9 EUR/MWh`.

No default coefficient, component discovery, silent clipping or
post-projection clipping exists. Receipt IDs are bound for replay but are not
authenticated and grant no authority.

`validate_short_tenor_combination_policy` separately rejects policy drift,
stale source bindings and non-outcome-blind numeric decisions. The declared
candidate/baseline families and successor-core binding are structural only;
all numeric coefficients, caps and grids remain null, and execution authority
is false.

## Scientific alignment

The contract records two primary references already consistent with the local
CH literature gate:

- Lago, Marcjasz, De Schutter & Weron (2021), Applied Energy 293, 116983,
  doi `10.1016/j.apenergy.2021.116983`: rigorous common-sample out-of-sample
  comparisons, strong transparent baselines and significance testing;
- Ziel & Weron (2018), Energy Economics 70, 396-420,
  doi `10.1016/j.eneco.2017.12.016`: regularized high-dimensional structures
  and combinations may help, but no structure dominates all seasons/hours.

Therefore D222 proves that a combination can preserve market constraints; it
does not claim that a combination improves the PFC. Future weights and caps
must be selected training-only inside every governed nested outer origin,
using common targets/masks, the declared embargo, transparent baselines, a
justified nonlinear challenger and feature-group ablations.

## Evidence and hashes

Selected bundle:

- content ID:
  `21c557df75260ce162c15af6fbe0c91de4c8a6fcd233564403a04e7b1fc53ad0`;
- manifest SHA-256:
  `c2cddd1ba59b3f0cf77ff16f5debcbf915c46b9373dd6a36e238d144d58bb21c`;
- summary SHA-256:
  `5f68ca8c6089df2cacb76db202160720655fe6b3e3ffdcf2522e04e41e9b6db3`;
- residual Parquet SHA-256:
  `45123d43c5978d115838308cb8028a4c13a3b842cebb34486d5c60a3286bed08`.

Implementation:

- `pfc_shaping/lt/model/short_tenor_combination_contract.py` SHA-256
  `311a34c154a93cb3389debee2edb948117428b94c1fd3890b7b28bcac5eb7f6e`;
- `tests/test_short_tenor_combination_contract.py` SHA-256
  `f5cb2bc621d897af54c741b4383d48de1f9c16a00a2634a07b47c7bb8a54dcb8`;
- `EEX-CH-SHORT-TENOR-COMBINATION-CONTRACT-V1.json` SHA-256
  `0be8ab62cbaba4c28de14179a749d47046b8e41aa967da8c06a2d49a5bc250ba`;
- build-only materializer SHA-256
  `a17304d563df43e911568af67dd17fa3628f51e33ea23b0ea0da21563fa172c6`;
- D219 dependency module SHA-256
  `951274f220ac7b5d3dc4a992ec83e672cc10e6b0b28036ecfca1336a2cbb981e`.

Canonical outcome-blind policy content ID:
`00a2b2087589d14ef1330cfa0de109fe9dfcce81436e0a0265fb96067d10fbb6`.

Two exact materializations returned the same content ID.

## Verification

Focused tests after fail-closed policy and provenance hardening:

```text
33 passed in 21.63s
```

Adjacent matrix covering D219, D220, D222, monthly solver,
`ch_lt_pit_preregistration`, LT/CT imports and LT package contract:

```text
209 passed, 4 skipped in 22.93s
```

Persisted six-case proof:

- 1h and 15min;
- spring and autumn DST;
- BASE-only active geometry;
- positive, negative and zero coefficients;
- negative algebraic monthly levels;
- maximum linearity residual: `0 EUR/MWh`;
- maximum constraint residual: `8.31279489688086e-15 EUR/MWh`;
- maximum monthly residual: `2.6981399962886943e-15 EUR/MWh`.

Independent random-coefficient roast:

- 24 cases over 12 months in 2027-2029;
- 1h and 15min;
- three components with PCG64 seed `20260805`;
- positive/negative coefficients and negative/zero/positive fixture levels;
- maximum linearity residual: `0 EUR/MWh`;
- maximum constraint residual: `1.4432899320127035e-14 EUR/MWh`;
- maximum monthly residual: `4.8110657888588664e-15 EUR/MWh`.

JSON validation passed. A 100-character mechanical line check is not claimed
for long SHA-256 identifiers and artifact paths. Ruff is not installed in the
repo-local runtime, so no Ruff pass is claimed.

## Roast findings and disposition

The first focused suite had 19 green tests. Manual fail-closed review then
identified two under-tested ambiguity paths:

- Python booleans could otherwise coerce to numeric coefficients/caps;
- one content ID could otherwise be reused for multiple provenance roles.

The implementation rejected both and initially reached 20 tests. The
outcome-blind policy validator then expanded the suite to 32 tests. A final
review corrected a stale D220 bundle binding and made the contract's
pairwise-provenance declaration mandatory, reaching 33 tests. The earlier
`cc6a8241...` proof is superseded by `21c557df...`, which binds the final D220
selection and current policy/module/test bytes. No failing model behavior was
observed after hardening.

## Changed files

- added `pfc_shaping/lt/model/short_tenor_combination_contract.py`;
- added `tests/test_short_tenor_combination_contract.py`;
- added
  `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-COMBINATION-CONTRACT-V1.json`;
- added build-only
  `build/databricks-eex-daily/materialize_short_tenor_combination_contract.py`;
- added D222 to `DECISION-LOG.md`;
- updated `docs/research/forwards_sources.md`;
- updated `.planning/HANDOFF.md`;
- added this handoff.

## Authority and risks

- Databricks requests: 0.
- SQL Warehouse starts: 0.
- Network calls during proof execution: 0.
- Remote writes: 0.
- The literature lookup was web-only and did not contact Databricks.
- Training/selection receipt authentication: false.
- PIT availability / rolling-origin / model input / assembly / production:
  false.
- The proof contains no market/vendor value, selected coefficient or selected
  cap. Fixture coefficient hashes are not candidate artifacts.
- Current empirical gate remains
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.
- Signed EEX PIT vintages, governed ENTSO-E, exact origin/target/inner-fold
  inventories, independent admission and a new future holdout remain required.
- OMPEX/AFRY remain benchmark-only, T057 remains sealed, monthly solver remains
  sole level authority, and LT/CT separation is unchanged.

## Next safe batch

Do not select weights yet. The next offline batch may define a strict
short-tenor candidate-grid/preregistration supplement with empty numeric
hyperparameters and explicit blockers, or harden the future training-receipt
schema. Any empirical fit must wait for governed PIT inputs and external
admission.

## Supersession note — D223

D223 supersedes the current proof selection in this handoff. Direct manifest
inspection showed that `21c557df...` binds the non-selected descriptive D220
identity `7a4c2a...`, despite the final-selection wording above. The corrected
current proof is `c06dfcf...`, which binds the governed D220 identity
`ae7b962c...`. See
`SESSION-HANDOFF-20260805-EEX-SHORT-TENOR-OUTCOME-BLIND-SCAFFOLD.md` and
D-20260805-223. The D222 algebra remains valid; only its current source/proof
selection and the incomplete preregistration scaffold are superseded.
