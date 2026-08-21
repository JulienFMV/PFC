# Session handoff - D228 short-tenor signed replay

Date: 2026-08-05

## Outcome

D228 adds a deterministic, value-free, path-based signed replay above D227.
It verifies local DSSE signature integrity, a real in-toto Statement v1 / SLSA
provenance v1 structure, exact D227 training-selection linkage and signed
execution-time observations. It does not authenticate external organizations,
establish independently trusted time, prove PIT availability or authorize an
empirical model.

Seven caller-anchored files are mandatory: two receipt envelopes, three public
keys and two time-observation envelopes. All paths are absolute, link-free,
single-link and pairwise distinct; all bytes match caller-held SHA-256 values.
Training uses the execution key, selection the governance key, and time
observations a third key. Key IDs are checked only after the caller-selected key
verifies the signature.

The SLSA replay binds exact receipt external parameters, output subject,
role-specific dependencies and run details. Selection dependencies bind the
training receipt, grid, fold inventory, loss commitment, policy and successor
core; they do not fabricate a training material inventory. Training and
selection execution-attestation hashes must be distinct.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-SIGNED-REPLAY-CONTRACT-V1.json`
- `pfc_shaping/validation/eex_short_tenor_signed_replay.py`
- `tests/test_eex_short_tenor_signed_replay.py`
- `build/databricks-eex-daily/materialize_short_tenor_signed_replay.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, heavy desk-data or AFRY numeric file was touched.

## Canonical identities

- signed-replay contract file SHA-256:
  `f9f447f31458a88aaaf4ce5863f9f23dbd6a77d1683b7759885783457780a957`
- signed-replay contract content ID:
  `09dfd36125132a1a1c8089f603a381414853c393d23dda633354938843382606`
- validator SHA-256:
  `b3e845797582751e8b2ba741d1d3add9a9f438c1dfa6880980c44b57e8feb4c8`
- tests SHA-256:
  `39aa2f9ed8a83dee213aa5f172a54d7d3bf6bb2577a9009e4cc782aae9542553`
- build-only materializer SHA-256:
  `78ea89afb578c9b9a54d9917e238ef851505591c9f3e31fe887e4a2a009289a2`
- research note SHA-256:
  `9845d734b6bc18c8095f0cf2ab50ef44782ebf09c3ef1ceff7f50e2aa5937908`

Reproducible build-only proof:

- content ID:
  `eed94d79109a5f196f49cf2bb11950b79889c8384cc82c1ef70a33a69cdebc5e`
- directory:
  `build/databricks-eex-daily/2026-08-05/short-tenor-signed-replay-proofs/eed94d79109a5f196f49cf2bb11950b79889c8384cc82c1ef70a33a69cdebc5e`
- manifest SHA-256:
  `18b5c3c394ddba05393fefd7b62fa2fbe0eb60c497c3e093d5a73e677ad2d8e9`
- assessment SHA-256:
  `d8b54043da2eba9062bfe385eeb639f0d46aafdaa5a2f367d93eda230cae51ef`

The manifest records zero Databricks requests, Warehouse starts, network calls,
remote writes and real receipts. It persists no numeric price, coefficient,
cap, loss or margin.

## Verification

Every shell command first verified exact cwd and normalized Git top-level
`C:\Users\jbattaglia\PFC_LT`. Mutable paths remained below `build/`.

- focused signed-replay tests: `27 passed in 13.81s`;
- expanded adjacent EEX/PIT/monthly/ENTSO-E/OMPEX/LT matrix:
  `455 passed, 2 skipped in 192.89s`;
- independent no-contention repeat of the same expanded matrix:
  `455 passed, 2 skipped in 35.51s`;
- materializer executed twice and returned the same content ID;
- AST and JSON parsing passed; relevant code lines remain below 100 characters;
- Ruff passed on the validator, tests and build-only materializer.

## Authority and risks

- Local signature integrity is not external key-owner identity or a governed
  key lifecycle.
- A signed observation is not independently trusted time until an approved
  external time authority and verification policy exist.
- Bound EEX/ENTSO-E materials and source-time receipts remain unopened. Their
  governed same-snapshot replay is a future mandatory layer.
- Training, selection, model input, candidate assembly, promotion and
  production remain strict `NO_GO`.
- OMPEX remains post-candidate benchmark-only; restricted AFRY stays
  descriptive; T057 stays sealed.
- The monthly solver remains sole level authority and LT remains independent
  from CT.

## Next safe batch

Define a value-free external trust-anchor registry and key-lifecycle contract,
including role ownership, issuance, rotation, revocation, compromise handling
and trusted-time policy. Keep it local and non-authoritative until the relevant
organizational owners provide independently governed anchors. Do not query or
start Databricks without explicit user authorization.
