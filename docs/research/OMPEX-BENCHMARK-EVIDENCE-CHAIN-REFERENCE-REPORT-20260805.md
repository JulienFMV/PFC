# OMPEX benchmark evidence chain: local reference only

## Technical summary

The repository now has a fail-closed, price-free reference verifier for the
three events required before an OMPEX comparison can become scientifically
credible: freeze the PFC candidate, authenticate the OMPEX vintage available
at the forecast origin, then register realised truth only after delivery.

The implementation passes 15 synthetic and adversarial tests. It has admitted
no real receipt, opened no price value and created zero countable forecast
origins. A valid local signature proves integrity relative to the supplied
public key; it does not prove who owns that key or that a timestamp came from
an independent trusted clock. The reference result therefore cannot authorize
scoring, model selection, an OMPEX superiority claim, promotion or production.

## What the reference verifies

| Receipt | Required evidence | Enforced time boundary |
|---|---|---|
| Candidate freeze | candidate, configuration and input hashes; origin and target; no prior OMPEX or truth access | data cutoff no later than origin; freeze after cutoff and no later than target start |
| OMPEX availability | exact OMPEX filename, size and hash; candidate receipt binding; timestamp-semantics contract; asserted at-origin availability | first availability no later than origin; opening after candidate freeze |
| Truth publication | exact truth source and hash; both preceding receipt IDs; hourly UTC EUR/MWh semantics; independent frozen revision | publication after target end; opening after publication |

All three receipts use exact canonical JSON, domain-separated Ed25519
signatures and distinct role keys. Caller-held hashes bind each receipt and
each underlying artifact without requiring the verifier to open candidate,
OMPEX or realised-price values.

## Why a passing chain is still non-production

[RFC 8032](https://www.rfc-editor.org/info/rfc8032/) specifies EdDSA/Ed25519,
which is suitable for detecting payload changes and possession of a private
key. It does not by itself establish that a supplied public key belongs to the
claimed desk, vendor, candidate registry or truth publisher. That assurance
requires an independently governed identity-to-key binding, consistent with
the public-key-owner assurance described by
[NIST SP 800-89](https://csrc.nist.gov/pubs/sp/800/89/final).

Likewise, a signed timestamp remains an assertion by its signer. An
independently verifiable time boundary requires a trusted timestamp authority
or an equivalent governed service; [RFC 3161](https://www.rfc-editor.org/info/rfc3161/)
defines the standard timestamp protocol. Append-only or WORM retention is also
needed to prevent silent replacement, replay or rollback after outcomes become
known.

For those reasons, the `.reference.v1` schemas and signing domains are
deliberately incompatible with any future production wire. Even a structurally
perfect local chain always returns
`PASS_LOCAL_REFERENCE_STRUCTURE_ONLY_NO_COUNTABLE_ORIGIN` and
`countable_origin_count = 0`.

## Adversarial validation

The 15 tests cover the passing price-free path and reject:

- data cutoffs after origin, freezes before cutoff or after target start;
- OMPEX first availability after origin or opening before candidate freeze;
- truth publication before delivery end or opening before publication;
- reused role keys, altered signatures and non-canonical receipt bytes;
- caller-held hash mismatches and broken cross-receipt links;
- truth that is not independent of candidate and OMPEX;
- output overwrite and any authority claim in the machine contract.

The tests are synthetic. They establish software behavior, not the existence,
authenticity or historical availability of an OMPEX vintage or realised-truth
publication.

## Production gap

Before one forecast origin can count, a new production contract must provide:

1. an externally governed registry binding organizations, roles and public
   keys, including rotation and revocation;
2. independent trusted timestamps and append-only/WORM retention;
3. separate candidate-registry, OMPEX desk/vendor and truth-publication
   authorities;
4. authenticated OMPEX hour-ending semantics and availability at the exact
   origin;
5. an independent frozen truth revision published only after the target
   window;
6. multiple future origins and preregistered, dependence-aware comparison
   rules before outcomes are opened.

The immediate next engineering step is to assign the three external owners and
define the production trust-anchor and timestamp services. Replaying this local
reference with self-generated keys would add no scientific evidence.

## Scope and execution boundary

This batch ran entirely under `C:\Users\jbattaglia\PFC_LT`. It did not read or
write `H:`, contact Databricks, write remotely or inspect any price values.
OMPEX remains an occasional explicit read-only benchmark source only.

The selected delivery surface is a governed technical Markdown report. An HTML
render was not produced because the workstation contract forbids launching
project executables and browser/Playwright qualification; no visualization is
appropriate with zero real admitted origins.

The machine-readable source of truth is
`OMPEX-BENCHMARK-EVIDENCE-CHAIN-REFERENCE-CONTRACT-V1.json`; the executable
reference is `pfc_shaping.validation.ompex_benchmark_evidence_chain` and its
local audit entry point is `scripts/audit_ompex_benchmark_evidence_chain.py`.
