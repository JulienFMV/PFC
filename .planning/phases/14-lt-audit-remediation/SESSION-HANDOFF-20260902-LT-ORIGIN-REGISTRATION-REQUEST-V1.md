# Session handoff - LT origin registration request v1 - 2026-09-02

## Outcome

Offline construction now covers the exact origin registration request-v2
signing surface. A request is built only after reverifying the signed schedule
and information-set envelope, and its cryptographic signature can be verified
without importing or loading a private key.

External FMV approval and real independent signatures were correctly treated
as future live-origin countability gates, not blockers to local construction.
No real signature, registry receipt or trusted-time admission was created.

## Changed files

- `pfc_shaping/lt/origin_registration_request.py` (new): exact request-v2
  builder, origin/request identity derivation, caller-supplied Ed25519
  signature assembly, public-key verification, cross-binding checks and
  explicit receipt-contract non-readiness.
- `pfc_shaping/lt/evaluation_protocol.py`: source-bound v4 identity and new
  request-module/package bindings; candidate, metric, cohort and authority
  semantics remain unchanged.
- `pfc_shaping/package_contract.py`: request module added to the governed-wheel
  positive inventory.
- `tests/test_lt_origin_registration_request.py` (new): synthetic field,
  identity, signature, chronology, CAS predecessor, signer-role,
  substitution, mutation and negative-authority tests.
- `tests/test_lt_evaluation_protocol.py` and
  `tests/test_lt_package_contract.py`: source/package binding assertions.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`: request boundary and
  explicit receipt-wire gap.
- `.planning/HANDOFF.md`, Phase 14 `DECISION-LOG.md`, and this handoff.

No CT, heavy desk data, monthly solver logic, production flag, model weight or
T057 outcome file was opened or changed.

## Frozen identities

- evaluation protocol v4 semantic SHA-256:
  `3fb4f3d2d1ba178d4217f96fc641727abda585d7bb5fe22768fa4cdc59668fbb`;
- evaluation protocol normalized-LF source SHA-256:
  `8339c031f17dcf9bf2a5ae0068850c8a0566c11453c9873c4903b7a377eca0d4`;
- origin request normalized-LF source SHA-256:
  `72323a7ca673cc53c09e6f6f8159771a3da845c055790acd2e5ff24292d33755`;
- origin envelope normalized-LF source SHA-256:
  `0c74cff02e886075b25d1f0893ec5b756a4e6661c88ee6cfd57aa8f07caaff8d`;
- package contract normalized-LF SHA-256:
  `7035638f9ae4fafab6c29798542ecba94ec525a480160f7d378415973392901e`;
- runtime specification normalized-LF SHA-256:
  `c61b2261c4ee0048d72a70ddb183b99b6a50cb52bf677bfbc663441db232987f`.

The future cohort remains `ch-lt-future-cohort-2026-10-v1`: 12 scheduled
origins and zero countable origins.

## Contract guarantees

- The request contains exactly the fields frozen by
  `ch_lt_origin_registration_request.v2`.
- Origin and request IDs use the frozen domain-separated hash derivations.
- The request reverifies exact envelope and schedule bytes and binds all
  information-set commitments, schedule timestamps, first target, trusted-time
  receipt hash and caller-held compare-and-append expectation.
- Genesis requires sequence one and no predecessor; later requests require an
  exact SHA-256 predecessor. Operation IDs are canonical UUIDs.
- Capture/origin/deadline/first-target chronology is checked independently.
- Request and schedule signer public-key identities must be disjoint.
- Re-signing a modified commitment does not bypass the exact-envelope binding.
- Trusted-time bytes are opaque and hash-bound only. Their semantics remain
  externally unadmitted.
- A verified signature reports cryptographic integrity but keeps registration,
  countability, truth opening, training, selection, scientific admission,
  production and promotion false.
- External receipt verification remains
  `UNSUPPORTED_EXTERNAL_RECEIPT_WIRE_CONTRACT_INCOMPLETE_NO_GO` until receipt
  ID/signature/trust and HEAD wire clauses are frozen. The local SQLite
  reference is not reused as production authority.

## Verification

Dedicated request-v2 matrix:

```text
14 passed in 0.33s
```

Focused request, envelope, evaluation protocol and package matrix:

```text
69 passed in 3.42s
```

Expanded evaluation, origin registry, estimand, dependence/power, curve
product, LT/CT import, hourly shape and optimizer matrix:

```text
314 passed, 1 skipped in 124.95s
```

Required LT minimum:

```text
58 passed, 1 skipped in 6.41s
```

The skip is the pre-existing optional TensorFlow boundary. Targeted Ruff
checks and format checks pass for the new and directly modified implementation
files.

Audit corrections made during the sequence:

- the plan initially described receipt verification too broadly; protocol
  audit showed that production receipt identity/signature and HEAD wire rules
  are not frozen, so the implementation records explicit non-readiness;
- an initial source-capability test matched the English word `requests.` in a
  docstring; it was narrowed to actual import/network call patterns;
- request attachment was hardened to validate chronology, origin-ID binding,
  HEAD semantics and hashes before accepting caller-supplied signature bytes;
- request and schedule signer-role disjointness was added after review;
- one minimum-test process completed after its output session was lost; the
  exact matrix was rerun with tracked output and passed.

## Authority and cost

- local data rows opened: `0`;
- real signatures or registry receipts created: `0`;
- production private keys loaded: `0`;
- real model training/retraining: `0`;
- real truth rows opened: `0`;
- Databricks connections/statements: `0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- model artifacts written: `0`;
- CT changes: `0`;
- CH monthly solver-authority changes: `0`.

## Next action

1. Continue construction by freezing a production external receipt and fresh
   HEAD wire contract: exact canonical bytes, ID domains, signer identities,
   trust registry, nonce/freshness and request/receipt/CAS bindings.
2. Only then implement the public-key-only receipt/HEAD verifier with synthetic
   adversarial tests; do not promote the SQLite reference schema.
3. In parallel when available, resume bounded governed EEX/ENTSO-E source
   admission. Local files through 31 August remain non-authoritative.
4. Keep future truth closed and add no real-data runner until registry and
   source-admission gates pass.

Durable decision: D-20260902-275.
