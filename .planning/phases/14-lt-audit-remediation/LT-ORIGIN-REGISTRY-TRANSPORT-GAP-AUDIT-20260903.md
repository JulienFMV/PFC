# LT origin registry transport-gap audit - 2026-09-03

## Outcome

No additional runtime code is justified in the current construction phase.
The origin-registry modules already close every transport-related invariant
that can be proved without an independently operated transport and identity
authority. The remaining gap is operational, not a missing local algorithm.

Creating an origin-registry HTTP client, mTLS configuration, workload-identity
adapter or second signed error protocol now would select infrastructure that
the repository does not specify. It would also duplicate the distinct snapshot
publication client while proving none of the missing external properties.

Status: `STOP_BEFORE_SPECULATIVE_TRANSPORT_IMPLEMENTATION_NO_AUTHORITY_NO_GO`.

## Audited surfaces

| Concern | Existing origin-registry mechanism | Local conclusion | Remaining external evidence |
|---|---|---|---|
| Request integrity and admission role | `origin_registration_request` verifies an exact Ed25519-signed request and keeps its signer distinct from schedule and registry roles | Closed for exact bytes under a caller-supplied public key | Admission of the request authority and authenticated client/service-account identity |
| Registry response integrity | `origin_registration_receipt` verifies domain-separated receipt and HEAD signatures; `origin_registry_conformance` selects only a locally eligible bundle key | Closed for exact bytes and local key lifecycle | Admission of the trust root, registry service identity and key custody |
| Append replay/idempotence | Operation UUID, request ID, exact request/receipt bytes, sequence, predecessor and uniqueness are checked under one synthetic lock | Closed as an in-memory state-machine conformance test | Remote linearizability, durable operation lookup and WORM retention |
| HEAD replay/freshness | Caller-supplied nonce, exact receipt binding, strict chronology and a 300-second maximum TTL | Closed cryptographically without generating a nonce or reading a clock | Unpredictable caller nonce, trusted verification time and authenticated live response |
| Ambiguous append recovery | Exact immutable operation lookup is already part of the contract and harness | Required recovery primitive is closed synthetically | Live transport failure boundary and remotely authoritative lookup |
| Rejected operation retention | Deterministic bounded rejection fields; request/receipt bytes, exception text and secrets are absent | Closed synthetically | Durable independent retention and service-side access control |
| Availability and transport errors | No origin-registry client exists | Deliberately not implemented | Selected transport, endpoint/status contract, timeout/retry policy, monitoring, SLO and DR evidence |

## Relevant non-reusable precedent

`pfc_shaping.data.snapshot_anchor_client` is a complete mTLS network client for
the separate snapshot-publication domain. It imports `http.client`, `ssl`,
environment configuration, certificate/private-key paths, temporary files and
publication-specific contracts. Its useful semantic precedent is narrow:

- authentication failures are determinate;
- a transient or malformed response after compare-and-append dispatch is
  indeterminate because the commit may have occurred;
- an indeterminate append must be resolved by exact operation lookup before
  any retry;
- read-only unavailability grants no authority;
- a fresh HEAD observation uses a new challenge nonce.

The module itself must not be imported, copied or generalized into the pure LT
origin-registry surface. The publication and origin domains, keys, schemas and
operational owners are distinct. A generic transport abstraction before the
origin service profile exists would increase coupling and obscure authority.

## Required behaviour when operationalization is authorized

Before adding a live client, one exact externally owned profile must specify:

1. service endpoint and protocol version;
2. server identity and pinned trust material;
3. client identity mechanism, credential owner and least-privilege scope;
4. exact request/response content types, maximum sizes and status mapping;
5. connect/read/total deadlines and rate-limit semantics;
6. whether failure occurred before or after possible append linearization;
7. exact operation-lookup recovery after every indeterminate append;
8. fresh nonce generation and trusted verification time for every HEAD check;
9. retention, WORM, backup/restore, multi-host concurrency and DR guarantees;
10. structured redacted telemetry, SLOs, alerting and audit ownership.

The future client must preserve this failure taxonomy:

- `PRE_DISPATCH_REJECTED`: no request reached the service; correct the local
  input or authentication state before retrying;
- `DETERMINATE_REMOTE_REJECTION`: an authenticated response proves no append;
- `INDETERMINATE_APPEND`: a commit may have occurred; perform exact operation
  lookup and never blind-retry with different bytes;
- `READ_UNAVAILABLE`: no state or freshness conclusion may be drawn;
- `INTEGRITY_FAILURE`: an invalid, divergent or unauthenticated response grants
  no authority and must not update local state.

Transport status codes or exception classes are not frozen here because they
depend on the selected protocol. No caller-supplied boolean may stand in for
authenticated peer identity, trusted time, remote durability or availability.

## Scope and authority

This audit changes no protocol, source hash, package inventory or runtime
behaviour. It opens no data and performs no network operation. It grants no
trust, registration, countability, truth-opening, training, selection,
scientific, production or promotion authority.

The next safe action is to resume a data-independent area with a fully specified
local success criterion, or wait for the operational transport profile. Do not
manufacture the latter merely to keep implementation moving.
