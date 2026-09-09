# Session handoff - EEX/ENTSO-E external-time batch

Date: 2026-08-05  
Decision: D-20260805-235  
Status: `PASS_LOCAL_RFC9162_MERKLE_AND_RFC3161_SCHEMA_ONLY_NO_EXTERNAL_TIME`

## Outcome

D235 implements and adversarially roasts the exact four-envelope RFC 9162
Merkle batch required by D233. It binds the exact D233 governed package, D234
physical-mapping compiler and D229 source-role registry. It also freezes a
fail-closed RFC 3161 request and future validation-receipt schema.

This is deliberately not a real timestamp. No DER request, TSA policy approval,
network submission, response, token, real DSSE envelope or externally trusted
time exists in the proof.

## RFC 9162 implementation

The exact role order is:

1. `eex_acquisition`;
2. `eex_source_trusted_time`;
3. `entsoe_acquisition`;
4. `entsoe_source_trusted_time`.

Each compact sorted-key UTF-8 leaf payload binds the same capture batch and
challenge plus its role and exact envelope SHA-256. The implementation uses:

- leaf: `SHA256(0x00 || canonical_payload_bytes)`;
- node: `SHA256(0x01 || left || right)`;
- RFC 9162 largest-power-of-two recursive split;
- RFC 9162 inclusion-path generation and section 2.1.3.2 verification;
- proof direction derived from leaf index and tree size;
- constant-time final root comparison.

The generic implementation was roasted on all trees from one to sixteen leaves,
including unbalanced sizes. The governed batch still requires exactly four
pairwise-distinct envelope hashes and one exact proof per role.

The synthetic D235 root is
`496c92c29ab1b4f1ad0f3491cbd4e38f500936973bc6ff8e8ac8ebeb3b0aa85c`.
It has local test-vector authority only.

## RFC 3161 boundary

The future request must retain exact DER bytes and bind the Merkle root through
the SHA-256 `messageImprint`, a positive unpredictable nonce, explicit approved
TSA policy and `certReq`.

The frozen future validation receipt has 46 required fields. A real verifier
must independently parse and cryptographically replay the DER/CMS and TSTInfo
bytes and verify response status, signed attributes, content type, exact
imprint, nonce, policy, certificate chain, exclusive critical time-stamping EKU,
certificate validity at `genTime`, revocation, a root pinned before the request,
the governed challenge bracket and the one-hour maximum lag.

Receipt booleans alone can never establish authority. The D235 synthetic receipt
requires all DER/crypto/time-authority claims to remain false. `genTime` would
not prove exact source-envelope signature time or retroactive PIT for historical
rows.

References:

- `https://www.rfc-editor.org/rfc/rfc9162#section-2.1.1`
- `https://www.rfc-editor.org/rfc/rfc9162#section-2.1.3`
- `https://www.rfc-editor.org/rfc/rfc3161#section-2.4.1`
- `https://www.rfc-editor.org/rfc/rfc3161#section-2.4.2`

## Exact artifacts and hashes

- contract:
  `.planning/phases/14-lt-audit-remediation/EEX-ENTSOE-EXTERNAL-TIME-BATCH-CONTRACT-V1.json`
  - raw SHA-256:
    `e9da2a73dd1bd62b5357e1f507bff3ef8e48918806c918bd4fdb1f609a71d138`
  - canonical content ID:
    `8f0f13315c382da7163a1f451752a6ddf84857a50133d89d3a3a31f340b264d0`
- validator:
  `pfc_shaping/validation/eex_entsoe_external_time_batch.py`
  - SHA-256:
    `a0e715b5f6adaf97f985ba93d99d35612b488289c464c30f852a4c03579e879e`
- tests:
  `tests/test_eex_entsoe_external_time_batch.py`
  - SHA-256:
    `7aa6c2bf2640fb1176e8bd9d8543de340e8ac34b69d0254b2a36142356d085a0`
- materializer:
  `build/databricks-eex-daily/materialize_eex_entsoe_external_time_batch.py`
  - SHA-256:
    `55ca77d5c62cd8e7f50c48b98cfb42cefed69bc34ec87fb05d58f45173ee5595`
- deterministic proof directory:
  `build/databricks-eex-daily/2026-08-05/eex-entsoe-external-time-batch-proofs/93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1`
  - proof content ID / assessment SHA-256:
    `93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1`
  - manifest SHA-256:
    `a10bbfa380652365684213197862139bf958dc33e706d81635cc2a6ae4582f02`

## Commands and results

Every command used exact cwd/Git-root guards for
`C:\Users\jbattaglia\PFC_LT`, repo-local mutable paths, repo-local `TEMP`/`TMP`
and fresh pytest basetemps under `build/`.

- targeted D235 roast:
  - `50 passed in 0.16s`;
- adjacent D229/D231-D235 seven-file matrix, including physical mapping and
  registered signers:
  - `265 passed in 5.92s`;
- Ruff on validator, tests and materializer:
  - `All checks passed!`;
- materializer run twice with the same local output root:
  - both returned
    `93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1`.

## Access and authority audit

All counters are zero:

- Databricks connections and statements;
- Warehouse starts;
- network and TSA calls;
- `H:` accesses;
- remote writes;
- real DSSE envelopes opened;
- market-value rows opened.

The only true local claim is successful replay of the synthetic Merkle fixture.
Real envelopes, DER, a real token, RFC 3161 cryptographic verification,
independently trusted time, PIT, training, selection, model input, candidate,
promotion and production authority are all false.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-ENTSOE-EXTERNAL-TIME-BATCH-CONTRACT-V1.json`
- `pfc_shaping/validation/eex_entsoe_external_time_batch.py`
- `tests/test_eex_entsoe_external_time_batch.py`
- `build/databricks-eex-daily/materialize_eex_entsoe_external_time_batch.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY numeric data, OMPEX, T057, heavy desk-data file or `H:`
path was opened or changed.

## Remaining gaps and next safe batch

Real D232 metadata, owner-verified D234 physical mapping, ENTSO-E values, four
source envelopes, an approved TSA policy, a real token, prospective PIT evidence
and a new independently frozen holdout remain absent. Do not contact a TSA or
Databricks without explicit authorization.

The next zero-query batch can define a byte-level RFC 3161 verifier dependency
and conformance-vector admission contract: pin an independently governed parser,
certificate-path/revocation implementation and public positive/negative vectors
before any real token is accepted. It must not use shell OpenSSL on this managed
workstation, submit network requests or turn synthetic receipts into authority.

The monthly solver remains sole monthly-level authority. LT remains independent
from CT, OMPEX remains post-candidate benchmark-only and AFRY remains
descriptive.
