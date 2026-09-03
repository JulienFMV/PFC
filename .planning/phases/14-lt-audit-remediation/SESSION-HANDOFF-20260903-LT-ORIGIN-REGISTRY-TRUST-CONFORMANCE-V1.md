# Session handoff - LT origin registry trust/conformance v1 - 2026-09-03

## Outcome

Offline construction now covers the registry trust-bundle/key-lifecycle and
transport-neutral state-machine gap left after the receipt/HEAD wire. Exact
caller-signed public-key bundles and their append-only rotation chain are
verified locally. A deterministic thread-safe in-memory harness qualifies
HEAD, compare-and-append, idempotence, uniqueness and immutable operation
lookup semantics with synthetic bytes only.

This remains authority-negative construction evidence. The code does not own
a private key, generate a signature, nonce or time, load credentials, access a
filesystem/database/network service, or claim remote CAS/WORM behaviour. No
trust root or registry service is admitted. All origins remain noncountable
and future truth remains closed.

No current ENTSO-E data was needed. No local data through 31 August was opened.
No Databricks connection or Warehouse action occurred.

## Changed files

- `.gitattributes`: pins the new exact-byte contract JSON to LF for portable
  hashing under the Windows `core.autocrlf=true` checkout.
- `.planning/phases/14-lt-audit-remediation/CH-LT-ORIGIN-REGISTRY-TRUST-TRANSPORT-CONFORMANCE-DRAFT-V1-20260903.json`
  (new): exact trust-bundle, key-lifecycle, transport-neutral operation,
  sanitized-rejection and negative-authority contract.
- `pfc_shaping/lt/origin_registry_conformance.py` (new): public-key-only trust
  bundle verification, append-only lifecycle verification, receipt-key
  selection, full receipt-chain candidate preparation and synthetic in-memory
  state-machine conformance.
- `tests/test_lt_origin_registry_conformance.py` (new): synthetic/adversarial
  signature, bundle, rotation, revocation, integration, retry, uniqueness,
  stale-HEAD, lookup and concurrency tests.
- `pfc_shaping/lt/evaluation_protocol.py`: source-bound v6 identity and exact
  conformance implementation/contract bindings; candidates, metrics, cohort,
  solver boundary and negative authorities are unchanged.
- `pfc_shaping/package_contract.py`: adds only the pure LT conformance module
  to the governed-wheel positive inventory.
- `tests/test_lt_evaluation_protocol.py` and
  `tests/test_lt_package_contract.py`: exact local-byte and package assertions.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`: v6 hashes and the
  authority-negative registry trust/transport boundary.
- `.planning/HANDOFF.md`, Phase 14 `DECISION-LOG.md`, and this handoff.

No CT file, heavy desk data, model weight, monthly solver logic, production
flag or T057 outcome file was opened or changed.

## Frozen identities

- evaluation protocol v6 semantic SHA-256:
  `152b453db2e19e4ea16007865cfabb91bf9147cef46105da1f9f584956769fbd`;
- evaluation protocol normalized-LF source SHA-256:
  `516edb82a74f14289db59839c40147f7c3531845b9725b89a0ad5c1102d2c0ee`;
- trust/transport semantic contract ID:
  `84839a92bc62426c964da9eaefcf719be80808c79089b7ac2142af14d033206b`;
- trust/transport exact LF file SHA-256:
  `57ce79771d25cf6a858a2c2fef585d202b66d05aff6a107b834f2afb9407df39`;
- registry conformance normalized-LF source SHA-256:
  `e7defb0700665bb68ca4debe490b6920816739f6882850752f304827b17cfdf1`;
- registry conformance test normalized-LF SHA-256:
  `9670a7b92625d8c27a110e47a102ce14c7661a12cff2f5b5f08ea81b36067c7f`;
- package contract normalized-LF SHA-256:
  `c9408d30b371393b4c5dbf46f94af650e6c64b5639ab39033b1a95fb98f23278`;
- parent receipt/HEAD exact wire-contract SHA-256:
  `1f9d1f6495f716b7afb3497195626b24fe427b7dedf3f6f001c00051d6688047`.

The future cohort remains `ch-lt-future-cohort-2026-10-v1`: 12 scheduled
origins and zero countable origins.

## Contract guarantees

- Trust bundles are exact canonical ASCII JSON with exact fields, duplicate
  rejection, canonical base64 and domain-separated identity/signature bytes.
- The caller-supplied trust root public key must differ from every registry
  signer. The module never receives a root or registry private key.
- A bundle has exactly one active signer valid at bundle issuance. Key records
  are ordered by ID and unique.
- A bundle chain begins at revision 1, advances by one, names the exact prior
  bundle, has strictly increasing issuance time and retains every prior key's
  bytes and validity window.
- Active keys may become historical, revoked or compromised; historical keys
  may only remain historical or become denied. Revoked and compromised states
  are terminal. Receipt verification rejects either denied state.
- Receipt-key selection uses the receipt's exact commit second and the key's
  half-open validity window. Integrated candidate preparation then reverifies
  the complete receipt/request/envelope/schedule chain.
- The synthetic harness uses one process-local lock for operation retry,
  uniqueness, sequence/predecessor comparison and append.
- Identical committed retries return the same immutable object and exact
  bytes. Divergent retries fail without replacing prior state.
- Duplicate request/slot/origin/as-of and stale-HEAD attempts fail closed and
  retain only a deterministic sanitized rejection. Request/receipt bytes,
  exception text and secrets are absent from that record.
- Every manifest and result keeps external trust, CAS/WORM, registration,
  countability, truth, training, selection, scientific, production and
  promotion authority false.

## Verification

Targeted style checks:

```text
python -m ruff check <5 changed Python/test files>
All checks passed!
python -m ruff format --check <5 changed Python/test files>
5 files already formatted
```

Dedicated trust/transport tests:

```text
16 passed, 1 warning in 0.17s
```

Focused trust, receipt, request, envelope, evaluation and package matrix:

```text
123 passed, 1 warning in 25.14s
```

Expanded evaluation, registry, estimand, dependence/power, curve-product,
LT/CT-import, hourly-shape and optimizer matrix:

```text
364 passed, 1 skipped, 1 warning in 342.84s
```

This expanded pass preceded the final narrowing of `get_head`; the affected
module, tests, protocol bindings and package surface were all rerun in the
final dedicated and focused matrices above.

Required LT minimum:

```text
58 passed, 1 skipped, 1 warning in 18.74s
```

The skip is the pre-existing optional TensorFlow boundary. The warning is the
pre-existing unknown pytest `cache_dir` option; cache creation was explicitly
disabled and all pytest basetemps were below
`build/origin-registry-conformance/`.

Audit corrections made during the sequence:

- an early root-guard command had a misspelled .NET type and stopped before
  executing project tools;
- an early patch orchestration variable was misspelled and made no change;
- the lifecycle-regression fixture initially failed on chain time ordering;
  its times/windows were corrected so the test now reaches and proves the
  intended irreversible-transition guard;
- a caller-settable synthetic cryptographic-success flag was removed;
- direct trust-bundle construction was tightened to validate ordered unique
  inventory, exactly one active signer, active-window issuance and root/signer
  role separation;
- final interface review found that `get_head` returned an internal commit
  object while the contract promised exact receipt bytes; the method and tests
  were narrowed to return only those exact bytes, then all dependent hashes
  were recomputed;
- the focused matrix was rerun to capture a complete result after its first
  detached output handle was not retained.

## Authority and cost

- local data rows opened: `0`;
- real signatures, trust bundles, receipts, HEADs or appends created: `0`;
- production private keys or credentials loaded: `0`;
- real model training/retraining: `0`;
- real truth rows opened: `0`;
- Databricks connections/statements: `0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- network or external registry operations: `0`;
- model artifacts written: `0`;
- CT changes: `0`;
- CH monthly solver-authority changes: `0`.

## Residual risks and next action

1. The in-memory lock proves only deterministic state-machine semantics. It is
   not evidence of independent service identity, remote linearizability,
   durable/WORM retention, availability or trusted commit time.
2. The trust root is deliberately caller-supplied and externally unadmitted.
   No real key inventory, custody or rotation ceremony exists.
3. Continue offline only if useful by first auditing the remaining transport
   authentication, replay and bounded error-envelope gap. Any implementation
   must remain a pure authority-negative synthetic interface without network,
   credentials, real keys or service deployment.
4. Do not create a live registry client merely to make progress. External
   service construction, operational keys and real registration belong to a
   separately authorized operationalization phase.
5. Resume governed EEX/ENTSO-E admission separately when inputs are available;
   do not substitute local history or open future truth meanwhile.

Durable decision: D-20260903-278.
