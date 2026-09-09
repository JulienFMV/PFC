# Session handoff - RFC 3161 request DER preflight

Date: 2026-08-05  
Decision: `D-20260805-237`  
Status: `PASS_LOCAL_REQUEST_DER_AND_CAPABILITY_PREFLIGHT_ONLY_NO_TSA_OR_TOKEN_AUTHORITY`

## Outcome

D237 turns the D235 RFC 3161 request schema into one exact byte-level
conformance vector and replays it fail closed. It does not contact a TSA,
generate a real nonce, select a real policy, parse a timestamp response or
authorize any request submission.

The timestamped datum is explicitly the exact four-leaf RFC 9162 final-node
preimage:

```text
0x01 || left_subtree_hash || right_subtree_hash
```

It is 65 bytes and its SHA-256 is the exact D235 Merkle root
`496c92c29ab1b4f1ad0f3491cbd4e38f500936973bc6ff8e8ac8ebeb3b0aa85c`.
This makes the RFC 3161 `messageImprint` relation explicit without silently
double-hashing the 32-byte root or calling the root its own unhashed datum.

## Exact request profile

The synthetic `TimeStampReq` is 89 bytes and contains:

- version 1;
- SHA-256 OID `2.16.840.1.101.3.4.2.1` with generated parameters absent;
- the exact 32-byte D235 root as `hashedMessage`;
- synthetic placeholder policy OID `1.3.6.1.4.1.55555.1.235`;
- fixed synthetic positive 129-bit nonce;
- canonical DER `certReq=TRUE`;
- no extensions.

Request DER SHA-256:
`75fc863d2083fb1bc6d816d993cfb762764c6735c33acdecc4645b9db88ab3c4`.

The parser rejects BER indefinite or non-shortest lengths, overruns, trailing
bytes, wrong tags/order, non-minimal or negative INTEGERs, non-canonical OIDs,
SHA-256 parameters in this exact generated vector, wrong hash length,
missing/false `certReq` and extensions. Canonical re-encoding must equal every
input byte.

RFC 5754 requires generators to omit SHA-2 parameters, but a general parser
must accept absent and NULL parameters. D237 therefore claims exact replay of
its self-generated request only; it is not a general response/CMS/TSP parser.

References:

- `https://www.rfc-editor.org/rfc/rfc3161#section-2.4.1`
- `https://www.rfc-editor.org/rfc/rfc5754#section-2`
- `https://www.rfc-editor.org/rfc/rfc5652#section-5.4`
- `https://www.rfc-editor.org/rfc/rfc5816`

## Dependency preflight

The current `cryptography==47.0.0` dependency is admitted only for SHA-256,
X.509 parsing and public-key signature primitives. Certificate extraction from
PKCS#7 is not complete token verification. The current stack does not expose
the full CMS `SignerInfo`/signed-attributes and `TSTInfo` replay, timestamping-
specific certificate path or offline revocation validation required by D235.

`asn1crypto==1.5.1` and `pyHanko==0.35.2` are recorded as candidates only.
Neither is installed, package-admitted or allowed to grant token authority.
Shell OpenSSL remains forbidden on this managed workstation.

## Canonical files and hashes

- contract:
  `.planning/phases/14-lt-audit-remediation/EEX-ENTSOE-RFC3161-REQUEST-DER-PREFLIGHT-CONTRACT-V1.json`
  - raw SHA-256:
    `d087736e548f579c03eaf84e89c2ee268edca0a9e8bf4e42db3044e533e8234b`
  - canonical content ID:
    `2333f35a9df0500cac5826a47105cf610c607cc64ad07ab904daf23038a68199`
- validator and allowlisted materializer module:
  `pfc_shaping/validation/eex_entsoe_rfc3161_request_der.py`
  - SHA-256:
    `bf265576afb518b92ae05f00d5425b203509f4bd37a15aa0a8316888b83cffb4`
- tests:
  `tests/test_eex_entsoe_rfc3161_request_der.py`
  - SHA-256:
    `b64b090554bbb1a934b07ed1217dbdb3dca2ed58281e8ebe96b235b882299dce`
- workspace runner after explicit module allowlisting:
  `scripts/run_workspace_local.py`
  - SHA-256:
    `c69382ec88ec998b5a8cc8987b33e62f05e13cef07d14a607e14d3ef25381243`

## Proof

- content ID / assessment SHA-256:
  `53e2222392f71541d28e05e1dfc912361c02003b4278f2770109827319d4e9c1`
- path:
  `build/databricks-eex-daily/2026-08-05/eex-entsoe-rfc3161-request-der-proofs/53e2222392f71541d28e05e1dfc912361c02003b4278f2770109827319d4e9c1/`
- manifest SHA-256:
  `c5d8be620f1c2f5012aa7174a5949adbc0d74708a162ac215c44f40f1f8dcdcf`

Two final materializations returned the same content ID. The earlier local
draft proof `2d4b658f...` is superseded because it did not yet distinguish the
exact self-replay parameter rule from RFC 5754 general-parser interoperability.
It has no authority.

## Verification

- focused D237 roast: `71 passed in 0.50s`;
- adjacent D229/D231-D237 acquisition/trust matrix:
  `367 passed in 6.94s`;
- workspace-runner matrix: `150 passed in 13.10s`;
- Ruff on validator, tests and workspace runner: `All checks passed!`;
- manifest artifact hashes/sizes, exact source hashes and authority profile
  replayed successfully.

All ten execution counters are integer zero: Databricks connections/statements,
Warehouse starts, network/TSA calls, `H:` accesses, remote writes, real DSSE
envelopes, real timestamp tokens and market-value rows. The only true authority
claim is successful replay of the synthetic request DER.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-ENTSOE-RFC3161-REQUEST-DER-PREFLIGHT-CONTRACT-V1.json`
- `pfc_shaping/validation/eex_entsoe_rfc3161_request_der.py`
- `tests/test_eex_entsoe_rfc3161_request_der.py`
- `scripts/run_workspace_local.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY numeric data, T057, OMPEX, heavy desk-data file or `H:`
path was opened or changed.

## Remaining gaps and next safe batch

A real request still requires an owner-approved TSA/policy, a genuinely random
nonce generated for that one request, governed package/runtime admission,
submission authorization and retained DER request/response bytes. Token
acceptance additionally requires an independently admitted CMS/TSTInfo parser,
signature and signed-attribute replay, `SigningCertificate(V2)`, TSA-specific
certificate path, pinned roots and offline revocation evidence at `genTime`.

The next zero-query batch may define the exact dependency/wheel and public
positive/negative conformance-vector admission for that response verifier.
It must not install a candidate silently, call a TSA or Databricks, use shell
OpenSSL, or turn parser output into time/model/production authority.

Real ENTSO-E metadata/mapping/values, prospective PIT evidence and a new frozen
holdout remain absent. Training, selection, candidate assembly and production
remain blocked by evidence, not by local code. The monthly solver remains sole
monthly-level authority; LT stays independent from CT; OMPEX remains a
post-candidate benchmark only; AFRY remains descriptive.
