# Session handoff - LT scenario and curve authority-negative contract - 2026-09-02

## Outcome

The existing LT interfaces were mapped against the six FMV target products.
The current code has numerical material compatible with an untyped central CH
candidate, but no admitted fundamental scenario, coherent stochastic path set,
portfolio valuation, FMV hydro optimiser or hedge recommender.

A standalone metadata-only contract now types scenario definitions and the
three CH curve products. It is deliberately dormant and authority-negative:
it does not hold prices, import into the production pipeline, run a solver,
assign probabilities or grant any operational authority.

## Pre-implementation checkpoint

Before new code, the previously qualified ENTSO-E/LSEG chain was committed as
`1fd5595a65` (`feat(lt): checkpoint governed day-ahead contracts`). The index
was compared exactly against a closed 38-file allowlist. `git add -A` was not
used. The checkpoint included no CT file, heavy data file or `build/` artifact,
and left the worktree clean.

## Files added

- `pfc_shaping/lt/curve_products.py`
  - SHA-256
    `b4b16508e4c13b29edeb0b46cd5195b4c5bf3f216c715e23c0251d634b7e3b96`;
  - scenario axes and shape-only versus level-changing semantics;
  - content-addressed provenance bounded by a UTC information timestamp;
  - typed `market_central_ch`, `fundamental_scenario_ch` and
    `stochastic_spot_paths_ch` relationships;
  - frozen authority-negative state and metadata-only manifests.
- `tests/test_lt_curve_products.py`
  - SHA-256
    `5bdbb7d8bcc1bf83124431724086e5c13fe7dcbaa2e546175ca19d0d20cc6aaa`;
  - eight synthetic tests for immutability, probability rejection, solver-month
    neutrality, separate upstream level solves, parent inheritance and
    point-in-time provenance.
- `docs/model/LT-INTERFACE-COMPATIBILITY-AND-GAP-MAP.md`
  - SHA-256
    `086eab89f175564ea1afa634b415b39f49f1397256f71e538a19f1819a14c617`;
  - compatibility/gap map for all ten audited interfaces and six target
    products.
- this handoff.

## Files updated

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - added D-20260902-270.
- `.planning/HANDOFF.md`
  - updated restart order and current state.

`pfc_shaping/pipeline/production_phases.py`, every `pfc_shaping/ct/*` file, the
CH monthly solver and all data files are unchanged.

## Compatibility and authority rules

1. `market_central_ch` requires the CH monthly BASE solver as level source and
   cannot carry a scenario or parent product.
2. A shape-only `fundamental_scenario_ch` must descend from the central curve,
   use the same required level source and normalize within `solver_month`.
3. A level-changing `fundamental_scenario_ch` must name a separate upstream
   solve requirement and cannot descend from or mutate the central curve.
4. `stochastic_spot_paths_ch` must descend from a typed fundamental scenario
   and inherit its scenario identity, level source and normalization bucket.
5. Required sources are lineage requirements only. All probability, calendar,
   monthly-authority grant, model, assembly, publication, production and trade
   flags are structurally false and non-overridable.

## Verification

All commands ran from the canonical root through
`python -m scripts.run_workspace_local`; mutable paths and receipts remained
below `build/`.

```text
python -m pytest tests/test_lt_curve_products.py -q
8 passed in 0.20s

python -m pytest tests/test_lt_ct_imports.py tests/test_lt_curve_products.py -q
25 passed, 1 skipped in 75.13s

python -m ruff check pfc_shaping/lt/curve_products.py tests/test_lt_curve_products.py
All checks passed!

python -m ruff format --check pfc_shaping/lt/curve_products.py tests/test_lt_curve_products.py
2 files already formatted
```

Final receipt paths and SHA-256 values:

- `build/workspace-local-runs/scntestfinal/execution-receipt.json`:
  `2bfd8a21857f81dbdc5e188ec82192380cc0129d67a7483ac7e7e825b1418f16`;
- `build/workspace-local-runs/scnltimports/execution-receipt.json`:
  `a5ec6b8f1c8b6633c25147e087e306bc0031e4a38ad5a66f6b5c4bdfc0736bda`;
- `build/workspace-local-runs/scncheckfinal/execution-receipt.json`:
  `2410cbeaa69a3bb704880e0205b7d3897f6fb88ae8b78cb1a46b4fa9fbcbceba`;
- `build/workspace-local-runs/scnfmtfinal/execution-receipt.json`:
  `24774810d95b97132998ceababc201e52790820dc69458206e75d03fcbac93df`.

The 1 skip is the pre-existing optional TensorFlow import boundary.

## Explicit non-actions and cost

- model training or retraining: 0;
- Databricks connections/statements/business rows: `0/0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- T057 reads or writes: 0;
- CT changes: 0;
- production pipeline or CH solver changes: 0;
- remote writes and incremental cost: `0/zero`.

## Residual blockers and next admissible task

The contract supplies typing only. It does not prove source admission, a real
scenario mapping, a fundamental solve, stochastic calibration or any
downstream decision model. Global status remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`; T057 remains sealed.

The next admissible work remains closure of governed EEX/ENTSO-E/LSEG data and
future-holdout gates. Do not wire these definitions into
`production_phases.py`, select a model, train a scenario engine or generate
paths until a separate governed decision and the required evidence exist.
