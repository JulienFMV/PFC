# Session handoff - ENTSO-E LT model-input admission composite

Date: 2026-08-06  
Decision: D-20260806-280  
Status: synthetic composite PASS; real model-input admission remains false

## Outcome

D280 now composes the exact cadence-bound D244/D245 package with D253 family
coverage, D272 physical-series semantics, D254/D255 temporal zones and D273
PIT-safe feature roles. It admits only exact, structurally eligible primitive
physical selections and preserves every predecessor authority boundary.

The deterministic synthetic proof covers:

- 82 physical series and 82 mapped semantic signatures;
- 1,968 expected native hourly targets and zero missing targets;
- three selected primitives: realized training input, operational forecast
  prediction input and lagged realized input;
- zero blocker codes and identical results over two replays.

This proves composition only. Real source evidence, real PIT, predictive value,
model input, model selection, candidate, promotion and production authorities
all remain false.

## Files changed

- `.planning/phases/14-lt-audit-remediation/ENTSOE-LT-MODEL-INPUT-ADMISSION-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_lt_model_input_admission.py`
- `tests/test_entsoe_lt_model_input_admission.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`
  - engineer-facing cardinality requirement is self-contained: exact
    `COUNT(DISTINCT SeriesID)` reconciliation by source-semantic tuple; it
    contains no internal `D###` decision reference.
  - subsequently roasted into a standalone extraction contract covering the
    exact dimension/latest/vintage sources, forecasts, native-cadence history,
    immutable deliverables, raw timestamp semantics and acceptance evidence;
    all feature/model instructions and internal project vocabulary were
    removed.
- `docs/research/forwards_sources.md`
- `.planning/HANDOFF.md`
- this handoff

Generated local-only materializer:

- `build/d280_materialize.py`

## Exact identities

- D280 contract raw SHA-256:
  `59e3f61f0cd6f44e7efe0c990eee9834d94fe5b45616af8058bb3da7e1e6d30c`
- D280 contract canonical content ID:
  `20144a9715eb6892756f8648c1e25090d4635247ac9683dd3a8ea896166f7d34`
- validator SHA-256:
  `1a3f26f0697e3652d8827ce25b0c9e49a13a7950cf442e88dac85c974f20fbae`
- tests SHA-256:
  `ece46e36db94491b191afd57e6066bc197155f753a0d38b7f059429ca08c7cfd`
- proof ID:
  `c9821d3666f1f402c1214ad49f62dcb1a26d2661a18e16fd5fd5ce553a129753`
- assessment content ID:
  `f576186cc902aba357e3b223cdde77a554e22192b778ffd60fe37c52dfc77310`
- proof path:
  `build/databricks-eex-daily/2026-08-06/entsoe-lt-model-input-admission-proofs/c9821d3666f1f402c1214ad49f62dcb1a26d2661a18e16fd5fd5ce553a129753/manifest.json`

## Verification

- governed run `entmod280f`: `19 passed in 12.52s`, receipt
  `TARGET_EXIT_ZERO_NOT_AUTHORITY`, target exit `0`, complete output, token not
  forwarded.
- Adjacent D244/D245/D243/D253/D254/D255/D261/D270/D272/D273/D280 matrix:
  `200 passed`.
- Exact 27-file ENTSO-E contractual regression matrix, including governed
  EEX/ENTSO-E acquisition and external-time contracts: `607 passed` in
  `49.76s`; stable execution status `TARGET_EXIT_ZERO_NOT_AUTHORITY`, target
  and runner exit codes `0`.
- Ruff check and format check on validator and tests: pass.
- Materializer replay count: `2`, deterministic.

One deliberately broader 40-file text-reference sweep is not admission
evidence and exposed unrelated qualification debt: `1230 passed`, `5 skipped`,
`9 failed`. Eight failures are AFRY diagnostic runtime-location checks under
the external workstation interpreter; one is governed snapshot external
publication authentication. D280 and its exact ENTSO-E matrix stayed green.

Mutation coverage includes missing zone binding, leaked feature chronology,
missing direct selection, derived forecast error without transform lineage,
selection/dimension detachment, wrong target-grid alignment or duration,
selection predating predecessor evidence, cadence TOCTOU/duplicate-key
mutations and authority escalation.

## Execution and safety

- Databricks connections/statements/writes: `0/0/0`.
- Warehouse starts: `0`.
- Network calls: `0`.
- `H:` accesses: `0`.
- Real value rows opened/persisted: `0/0`.
- No CT module was imported or changed.

## Remaining real-data blockers

- No independently governed real D244/D245/D270 export has yet satisfied the
  complete D253/D272/D255/D273/D280 chain.
- Real owner identity, exact zone registry authority and historical PIT
  availability remain unverified.
- Forecast-error transforms still need their own content-addressed primitive
  dependency and transformation-lineage contract before use.
- Predictive value must later be established only through the independently
  frozen rolling-origin holdout. T057 remains sealed.

## Next safe batch

Define and roast the value-free transform-lineage gate for derived ENTSO-E
features, beginning with lagged forecast error, without admitting the feature
or opening real values. Real execution stays deferred until the governed local
export is available; do not retry Databricks merely to advance this batch.
