# Session handoff - LT origin receipt/HEAD wire v1 - 2026-09-02

## Outcome

Offline construction now covers strict public-key-only verification of a CH LT
origin registration receipt and nonce-bound fresh registry HEAD. The complete
request/envelope/schedule chain is reverified before either document is
accepted. The wire draft, source files and evaluation protocol are hash-bound.

This is construction evidence only. A synthetic receipt may carry the parent
protocol's signed `countable_prospective_origin=true` claim, but every local
result keeps external registration, countability, truth opening, training,
selection, scientific admission, production and promotion false. External FMV
approval is not required to continue local construction; it remains a future
live-origin admission gate.

No real signature, registry append, HEAD observation, trusted-time admission,
data access or model artifact was created.

## Changed files

- `.gitattributes`: the exact-byte wire JSON is pinned to LF so Windows
  `core.autocrlf` cannot alter its frozen SHA-256 on a fresh checkout.
- `.planning/phases/14-lt-audit-remediation/CH-LT-ORIGIN-REGISTRY-RECEIPT-HEAD-WIRE-CONTRACT-DRAFT-V1-20260902.json`
  (new): exact receipt/HEAD field inventories, canonicalization, identity and
  signature domains, key separation, nonce/freshness rules and explicit
  negative authority.
- `pfc_shaping/lt/origin_registration_receipt.py` (new): pure public-key-only
  builders/assemblers/verifiers for externally supplied receipt and HEAD
  signatures; no private key, I/O, clock, nonce generation or registry write.
- `tests/test_lt_origin_registration_receipt.py` (new): synthetic round-trip,
  authority, domain separation, mutation, resigning, key substitution,
  chronology, nonce, receipt-hash, TTL, canonical JSON and capability tests.
- `pfc_shaping/lt/origin_registration_request.py`: readiness now reports the
  local verifier implemented and hash-frozen while external trust admission
  remains missing.
- `pfc_shaping/lt/evaluation_protocol.py`: source-bound v5 identity with exact
  receipt-verifier and wire-contract bindings; candidate, metric, cohort,
  solver and authority semantics are unchanged.
- `pfc_shaping/package_contract.py`: the pure receipt verifier was added to the
  governed-wheel positive inventory; the SQLite reference remains excluded.
- `tests/test_lt_origin_registration_request.py`,
  `tests/test_lt_evaluation_protocol.py` and
  `tests/test_lt_package_contract.py`: updated readiness and exact source/file
  binding assertions.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`: v5 receipt/HEAD
  boundary and authority distinction.
- `.planning/HANDOFF.md`, Phase 14 `DECISION-LOG.md`, and this handoff.

No CT, heavy desk data, monthly solver logic, production flag, model weight or
T057 outcome file was opened or changed.

## Frozen identities

- evaluation protocol v5 semantic SHA-256:
  `800c1db06f3a760998d318553fcece605d86581991b2548132fb18e61e5ff9d7`;
- evaluation protocol normalized-LF source SHA-256:
  `ddc0d165b34689dc0ddc73863715a48e41f9fd5c317a08902bef678e13660d65`;
- wire contract semantic ID:
  `290b108770dc799eeb83ca9f0a046aa8436878858224b68f757a5d9bfacbcc33`;
- wire contract exact file SHA-256:
  `1f9d1f6495f716b7afb3497195626b24fe427b7dedf3f6f001c00051d6688047`;
- receipt/HEAD verifier normalized-LF source SHA-256:
  `d3224f53aa1912301ac87f5db0c4721547978c1da17358411d37adc558299eb7`;
- origin request normalized-LF source SHA-256:
  `e4dc1366ac1e13c1f2b4083ac6c8d5f24036cd743cae678359bae8f1435ed748`;
- origin envelope normalized-LF source SHA-256:
  `0c74cff02e886075b25d1f0893ec5b756a4e6661c88ee6cfd57aa8f07caaff8d`;
- package contract normalized-LF SHA-256:
  `cfb123ad7a0549dabfceec933fa8056aa81dad0328872fc097156ba68f1901b8`;
- runtime specification normalized-LF SHA-256:
  `c61b2261c4ee0048d72a70ddb183b99b6a50cb52bf677bfbc663441db232987f`.

The future cohort remains `ch-lt-future-cohort-2026-10-v1`: 12 scheduled
origins and zero countable origins.

## Contract guarantees

- Receipt and HEAD inputs must be exact canonical ASCII JSON with no duplicate
  keys, floats, alternate base64 or extra/missing fields.
- Receipt ID, receipt signature, HEAD ID and HEAD signature use four distinct
  domain-separated preimages.
- The receipt reverifies exact request, information-set envelope, signed
  schedule and opaque trusted-time bytes before checking all parent bindings.
- Receipt sequence/predecessor, operation, request ID and exact request hash,
  slot, schedule entry, capture window, origin and commit chronology must agree.
- Registry, request and schedule public-key identities must be disjoint.
- The HEAD binds exact signed receipt bytes, sequence and a caller-supplied
  lowercase SHA-256 nonce. The verifier never creates a nonce.
- Freshness uses an explicit caller-supplied UTC whole-second verification
  time and requires
  `committed <= observed <= verification_time <= expires`; TTL is in `(0, 300s]`.
- A verified receipt claim and a verified fresh HEAD are reported as
  cryptographic facts only. They cannot mutate any authority boolean to true.
- Local filesystem, cached HEADs and the SQLite reference never become
  registry authority.

## Verification

Dedicated receipt/HEAD matrix:

```text
17 passed in 1.41s
```

Focused receipt, request, envelope, evaluation and package matrix after final
format/hash closure:

```text
107 passed in 16.56s
```

Expanded evaluation, registry, estimand, dependence/power, curve-product,
LT/CT-import, hourly-shape and optimizer matrix:

```text
348 passed, 1 skipped in 186.70s
```

Required LT minimum:

```text
58 passed, 1 skipped in 11.17s
```

The skip is the pre-existing optional TensorFlow boundary. Targeted Ruff
checks and format checks pass. The emitted pytest `cache_dir` warning is a
pre-existing configuration mismatch; cache creation was disabled explicitly
and all test basetemps remained below `build/`.

Audit corrections made during the sequence:

- the initial test shell command contained a malformed cache-directory token;
  PowerShell rejected it before Ruff or pytest ran, and the corrected command
  was used;
- format-check found three unformatted files; they were mechanically
  normalized, all source hashes and the v5 semantic hash were recomputed, and
  the focused and mandatory matrices were rerun;
- the verifier's authority manifest was made constant-false rather than
  reflecting instance attributes, and sequence/context validation was
  simplified during code review;
- receipt `countable=true` is explicitly represented as an untrusted signed
  claim, never conflated with the verifier result's `countable_origin=false`.
- exact-file hashing review found `core.autocrlf=true`; a path-specific LF
  attribute was added and verified to keep the wire-contract bytes portable.

## Authority and cost

- local data rows opened: `0`;
- real signatures, receipts or HEAD observations created: `0`;
- production private keys loaded: `0`;
- real model training/retraining: `0`;
- real truth rows opened: `0`;
- Databricks connections/statements: `0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- model artifacts written: `0`;
- CT changes: `0`;
- CH monthly solver-authority changes: `0`.

## Next action

1. Continue offline construction without waiting for operational approval by
   freezing the registry trust-bundle/key lifecycle and transport-neutral
   `get_head` / compare-and-append / immutable operation-lookup conformance
   contract. It must remain an authority-negative interface and must not turn
   the local SQLite reference into the service implementation.
2. Implement only a deterministic in-memory synthetic conformance harness for
   linearizability, idempotent identical retry, divergent retry rejection,
   duplicate-slot rejection and rejected-operation retention. Keep network,
   filesystem, credentials and live registry operations absent.
3. External approval, registry service deployment, admitted trust keys and
   real signatures belong only to a later operationalization phase.
4. When governed EEX/ENTSO-E inputs become available, resume their bounded
   admission separately. Local files through 31 August remain non-authoritative.
5. Keep future truth closed; do not train or add a real-data runner while
   source and registry admission gates remain open.

Durable decision: D-20260902-276.
