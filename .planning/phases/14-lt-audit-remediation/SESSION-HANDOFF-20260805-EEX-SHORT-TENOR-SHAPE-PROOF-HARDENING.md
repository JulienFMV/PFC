# Session handoff - EEX short-tenor shape proof hardening

Date: 2026-08-05

## Outcome

D219 retains the dormant D217 solver-neutral contract unchanged and replaces
its selected three-case qualification with a stronger content-addressed proof
bundle. The additional evidence proves price-level invariance, affine
preservation of an already solver-admissible curve, partial-PEAK and BASE-only
geometry, DST safety and absence of any LT orchestration wiring.

The status remains
`PASS_LOCAL_MATHEMATICAL_CONTRACT_ONLY_NO_MODEL_AUTHORITY`. Nothing was fitted,
activated or connected to production. No Databricks request, SQL Warehouse
start, network call or remote write occurred.

## Changed files

- `tests/test_short_tenor_shape_contract.py`
  - adds forward-level invariance;
  - proves that adding the delta preserves an affine solver solution;
  - proves the module is absent from LT production orchestration;
  - focused coverage increases from 11 to 14 tests.
- build-only proof materializer:
  `build/databricks-eex-daily/materialize_short_tenor_shape_contract.py`.
- `docs/research/forwards_sources.md`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D219).
- `.planning/HANDOFF.md`.
- this handoff.

The D217 contract JSON and
`pfc_shaping/lt/model/short_tenor_shape_contract.py` are unchanged. No CT,
AFRY numeric, Power BI, protected desk-data, monthly-solver implementation,
assembler or production flag was modified.

## Selected proof bundle

Root:

`build/databricks-eex-daily/2026-08-05/short-tenor-shape-contract-proofs/4da441b09c5559d772fd507214b2989489314bd9114a9328752a150f3957450a/`

- content ID:
  `4da441b09c5559d772fd507214b2989489314bd9114a9328752a150f3957450a`;
- manifest SHA-256:
  `b6892ba9b4de7c165ad5f70a01800cdd2a471b51f21fa6b94a7ebbacc97d5bb6`;
- summary SHA-256:
  `7d2071c4ad123fa50e37d40ea83ddd15905ef0d48ef067a48b80b6ee28a752b4`;
- 36 constraint residual rows, SHA-256
  `eaec1f992fdf08c488718ec232280fdaf48176a4e9e71d6ab4a0531eff188a04`;
- 20 monthly residual rows, SHA-256
  `f7ba2876b7a07221bd6f932251d49a3b940cfa68c0a0b6ffa2fa8df5bb7980fe`.

Implementation bindings:

- D217 contract SHA-256
  `cddea7b1a47e81bfa3e85a1ed70bfb1adfa39ed56cc6b546bad415f9c9dce9ee`;
- contract module SHA-256
  `951274f220ac7b5d3dc4a992ec83e672cc10e6b0b28036ecfca1336a2cbb981e`;
- shape-constraint module SHA-256
  `9298534b42d7fde3c867394a237c4b4665dd15273bde56568e5077eaf27798ad`;
- tests SHA-256
  `7f29aaa91982633a60f929cb5e202e90c54cdaa3606934d0de2a4c1a5a38dd97`;
- proof materializer SHA-256
  `1ad3193b31f44608f4dd30e035d717acedcaa83af1e8449fe32ecf467aadb81e`.

The D217 qualification
`1c3b833e2c7abc8ddc60867c2cdee8b152e2128a9699e4f38958a3b7062b8343`
remains valid mathematical evidence but is no longer the selected proof. D218
OMPEX structural-benchmark evidence is independent and is not superseded by
D219.

## Proof scope and results

Five persisted algebraic cases cover:

- complete hourly 2027 with monthly PEAK constraints;
- spring DST 2028 at 15 minutes;
- autumn DST 2028 at 15 minutes;
- four hourly months with alternating active PEAK constraints;
- two hourly BASE-only months.

Maximum persisted constraint residual:
`6.772360450213455e-15` EUR/MWh. Maximum persisted monthly weighted-mean
residual: `2.9844704963041844e-15` EUR/MWh. No market/vendor value and no
synthetic projected price curve are persisted; only algebraic audits and
near-zero residuals are stored.

A separate deterministic 30-case roast varied cadence, calendar windows,
negative/positive monthly levels and PEAK policies. Worst constraint residual
was `1.0058620603103918e-13`; worst monthly residual was
`7.474069153304495e-14` EUR/MWh.

## Commands and verification

Every command verified exact cwd and Git top-level
`C:\Users\jbattaglia\PFC_LT`; mutable paths remained below `build/`.

- Focused contract tests:
  `build\pytest-runtime-v2-final\python.exe -m pytest -q tests/test_short_tenor_shape_contract.py`
  -> `14 passed in 1.17s`.
- Adjacent LT matrix covering monthly audit/constraints/integration/solver,
  shape-hourly, assembler profile, LT package and LT/CT boundaries:
  -> `256 passed, 4 skipped in 37.57s`.
- Proof materializer ran twice and returned the same content ID while checking
  exact existing bytes.
- The earlier D217 qualification materializer was replayed once and returned
  its original content ID, confirming continuity.
- `git diff --check` must be green at final handoff.

## Authority and next safe step

D219 is mathematical boundary evidence only. Model input, training, candidate
assembly, selection, promotion and production authority remain false. The CH
monthly solver remains sole level authority; no later month patch is allowed.
Signed EEX PIT vintages, governed ENTSO-E, preregistered rolling-origin and a
new independent holdout remain mandatory. AFRY and OMPEX remain separate
benchmark-only evidence; T057 stays sealed.

Next safe offline batch: specify a pure same-vintage constructor for complete
DAY-versus-WEEK and WEEKEND-versus-WEEK additive contrasts. It must emit only
native day/block signals accepted by this dormant contract, with untrained
amplitude and activation false.
