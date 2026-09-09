# Session handoff - candidate execution contract

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decision: D-20260904-294

## Outcome

The candidate-execution ambiguity is closed without model fitting or a new
runtime. The exact path is native incumbent replay plus common challenger
observations. The frozen incumbent cannot be replaced by a generic MLP because
its source-bound implementation has native dataframe interfaces, constructs a
daily-normalized hourly factor target internally and applies its own factor
post-processing.

The four challengers may consume the common nine-column matrix, but they must
learn the incumbent-equivalent `f_H`, not raw EUR/MWh prices. The learning
transform is frozen as direct CH quarter-hour price divided by Swiss-local
daily mean, daily mean strictly above 5 EUR/MWh, ratio clipped to `[0.2, 3.0]`
and incumbent-weighted aggregation by Swiss-local date and clock hour. The
autumn repeated clock hour remains merged to match the frozen source rather
than silently changing the baseline.

All candidate predictions are factors until they have passed through the same
post-processing, solver-owned monthly levels and downstream layers. Only full
price curves in EUR/MWh may enter the evaluation engine, which separately
neutralizes local-month levels for shape scoring.

The common input adapter now requires `target_f_h` and rejects the earlier
ambiguous `target_eur_mwh` name. D293 timestamp alignment, availability,
integrity and one-mask semantics are unchanged.

No Databricks statement, Warehouse action, real-data read, model fit, truth
opening, GPU execution or remote write occurred. Every model, monthly-level,
selection, publication and production authority remains false.

## Changed files

- `pfc_shaping/lt/evaluation_execution_contract.py`
  - metadata-only native-incumbent/common-estimand decision;
  - semantic SHA-256:
    `91129fd87a945050d11f724461fa6677726a56dfd8da366143ac7fe61f3cf976`;
  - file SHA-256:
    `34dc57b874edf2484178685534179db2296a1f30da7fa03dc1b5f361bd6fb8d7`.
- `tests/test_lt_evaluation_execution_contract.py`
  - path, estimand, source hash/interface, scoring boundary and authority tests;
  - SHA-256:
    `36269b221ee2b308daee2db115dbd13955957278f7d3e33cd08bf49bd4012e53`.
- `pfc_shaping/lt/evaluation_inputs.py`
  - replaces the ambiguous raw-price target column with exact `target_f_h`;
  - SHA-256:
    `e6a70e6fcef4137e740d79cd398513966465e3be18a5ce7598096d281325ddca`.
- `tests/test_lt_evaluation_inputs.py`
  - updates synthetic fixtures and rejection checks to the factor target;
  - SHA-256:
    `327af1e4eb504fac0f00247a320af7ccaff73a5e63cfc587cb6616ef1190c565`.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`
  - documents the distinct learning, prediction, assembly and scoring spaces.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D294 and explicitly supersedes only D293's target label.
- `.planning/HANDOFF.md`
  - points to D294 and this handoff.

All preceding uncommitted ENTSO-E/LSEG and D289-D293 work remains preserved.

## Verification

Every shell action first checked that current directory and Git root both
resolve exactly to `C:\Users\jbattaglia\PFC_LT`. Python tests and formatting
used `scripts.run_workspace_local` with repo-local mutable paths.

- expected initial semantic-hash freeze test: `1 failed, 22 passed`; the sole
  failure disclosed the canonical hash, which was then frozen;
- post-freeze focused contract/input suite: `24 passed`;
- post-format focused contract/input suite: `24 passed`;
- expanded evaluation, feature-availability, materialization, LT/CT-import and
  package matrix: `186 passed, 1 skipped`;
- targeted Ruff check: pass;
- targeted Ruff format: pass after formatting.

The skipped case is the existing optional dependency/runtime boundary.

## Next safe step

Implement the first item of the frozen sequence only: a pure in-memory
constructor for incumbent-equivalent historical `target_f_h` observations,
with synthetic parity tests against the hash-bound incumbent transformation.

The constructor must preserve:

- direct CH quarter-hour source price and explicit availability;
- Swiss-local day grouping and the `daily_mean > 5` rule;
- ratio clipping before hourly aggregation;
- the incumbent's within-hour recency weighting and repeated-fallback-hour
  merge;
- deterministic row identity and exact value/timestamp hashes;
- native exclusion of missing source prices; feature nulls remain preserved
  until the existing common mask;
- all authorities false.

Do not yet add a real-data model runner, fit any candidate, open future truth,
start a Warehouse or use the GPU. A new window is not needed for this pure
constructor. Start a new window before implementing broad real-data execution
or common full-curve orchestration.

## Invariants

- The source-bound incumbent is replayed natively and never relabelled through
  a surrogate.
- Generic challengers learn `f_H`, not raw price, and are scored only after
  common full-price assembly.
- The synthetic challenger lab remains synthetic-only.
- PRD Databricks remains the enterprise-validated source boundary; upstream
  ENTSO-E recovery and freshness are Data Engineering responsibilities.
- The CH monthly BASE solver remains sole level authority.
- LT remains independent from CT, T057 remains sealed and every operational
  authority remains false.
