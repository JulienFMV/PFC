# Session handoff - LT synthetic evaluation engine v1 - 2026-09-02

## Outcome

The next offline implementation increment is complete. The LT evaluation
protocol is source-bound v2, all four challengers exist behind a
synthetic-only API, and the scorer enforces monthly level neutrality, common
rows and explicit horizon coverage without ranking candidates.

No local dataset, future truth, production model or external compute was used.
The user's note that local data may extend through 31 August was recorded but
not used as an authority or pseudo-holdout.

## Changed files

- `pfc_shaping/lt/evaluation_challengers.py` (new): immutable synthetic
  training/prediction inputs, exact recency weights, weighted MLP, Ridge,
  spline-Ridge and lazy deterministic LightGBM.
- `pfc_shaping/lt/evaluation_engine.py` (new): common-row, local-month-neutral
  synthetic scoring with complete lead-bucket reporting.
- `pfc_shaping/lt/evaluation_protocol.py`: v2 candidate source bindings,
  implementation parameters, engine/package/runtime identities and new
  semantic hash.
- `pfc_shaping/package_contract.py`: evaluation modules added to the governed
  wheel positive inventory.
- `tests/test_lt_evaluation_challengers.py` (new): synthetic temporal,
  immutability, parameter, gradient, fit and deterministic-runtime tests.
- `tests/test_lt_evaluation_engine.py` (new): synthetic authority, metric,
  neutralization, complete-case, horizon and reconciliation tests.
- `tests/test_lt_evaluation_protocol.py`: v2 source-binding and no-data-access
  tests.
- `tests/test_lt_package_contract.py`: evaluation inventory assertion.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`: v2 operator contract.
- `.planning/HANDOFF.md`, Phase 14 `DECISION-LOG.md`, and this handoff.

No CT, heavy desk data, solver logic, production flag, model weight or T057
file was opened or changed.

## Frozen identities

- protocol semantic SHA-256:
  `1134a5e24cfabc797d8931a986bce87ac983dcaaec5dd39680929463c62bdf3e`;
- challenger source normalized-LF SHA-256:
  `887f3b00d33231b52c58395ef43b5310624222922e955a6425d723e885cb5e19`;
- evaluation engine normalized-LF SHA-256:
  `034a06c14ec5aff337ab58cf4ab2e79a63c49f2f1d656dbc5fb3bd950c310ffc`;
- package contract normalized-LF SHA-256:
  `c16379bcb37d7da5af50715e11d522dd36161f3dfb3aa828047f71b59a0159d3`;
- `pyproject.toml` normalized-LF SHA-256:
  `c61b2261c4ee0048d72a70ddb183b99b6a50cb52bf677bfbc663441db232987f`.

The incumbent source/config hashes remain unchanged from v1. The future
cohort remains `ch-lt-future-cohort-2026-10-v1`: 12 scheduled origins and zero
countable origins.

## Implementation guarantees

- synthetic inputs are copied read-only and require unique ordered UTC rows;
- training observations must be strictly earlier than their frozen origin;
- feature schemas and tuning-grid values must match exactly;
- recency weights are origin-relative, finite, positive and mean-normalized;
- the weighted MLP analytic gradient is checked against central differences;
- all stochastic-capable implementations use fixed seeds; LightGBM is CPU,
  deterministic and single-worker with exact version 4.6.0;
- all five candidates share one complete-case intersection;
- prediction and truth are centered separately by energy-weighted Swiss local
  delivery month;
- every report contains `M01_M06`, `M07_M12`, `M13_M24`, `M25_M36` and
  reconciles bucket rows/energy to its aggregate;
- empty buckets and unimplemented product/economic metrics remain
  `UNSUPPORTED_NEVER_PASS`;
- fitted synthetic models and reports expose no winner and carry immutable
  negative authority.

## Verification

Focused protocol/challenger/scoring/package matrix:

```text
58 passed in 4.27s
```

Expanded matrix covering the focused tests plus estimand, dependence/power,
origin registry, curve products, LT/CT imports, hourly shaping and quant
optimizer KKT/no-leakage/contracts:

```text
168 passed, 1 skipped in 30.15s
```

The skip is the pre-existing optional TensorFlow boundary. Targeted Ruff check
and Ruff format check passed.

Audit corrections made during the sequence:

- a package test initially treated every allowed wheel file as mandatory;
  the over-strong test assertion was removed and the contract retained;
- the first expanded command named a nonexistent quant test, so pytest ran no
  tests; the actual KKT/no-leakage/contract filenames were resolved and the
  matrix passed;
- PowerShell lacked `Convert.ToHexString`; the read-only hash helper was rerun
  with a compatible byte formatter;
- intentional semantic-hash failures exposed each v2 manifest change before
  the final hash was frozen.

## Authority and cost

- local data rows opened: `0`;
- real model training/retraining: `0`;
- real truth rows opened: `0`;
- Databricks connections/statements: `0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- model artifacts written: `0`;
- CT changes: `0`;
- CH monthly solver-authority changes: `0`.

Synthetic fitting in unit tests is code qualification only. It creates no
persisted model and grants no empirical, scientific or operational authority.

## Next action

1. Prepare the externally registrable origin information-set envelope and
   signed schedule payload without opening truth.
2. Resume the bounded governed EEX/ENTSO-E source admission when the external
   platform is available; local file presence is not admission.
3. Only after those gates pass, add a separate governed real-data runner with
   exact runtime/artifact bindings and nested development-origin selection.
4. Keep future-cohort truth closed until its independently registered maturity
   event.

Durable decision: D-20260902-273.
