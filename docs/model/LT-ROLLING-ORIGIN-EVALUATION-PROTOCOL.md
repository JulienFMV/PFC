# LT rolling-origin evaluation protocol v6

## Scope

`pfc_shaping.lt.evaluation_protocol` locally freezes the next CH hourly-shape
comparison without opening real truth. `evaluation_challengers` implements the
four challengers only behind immutable synthetic fixtures, and
`evaluation_engine` scores synthetic predictions without ranking them. The
three modules reuse the existing LT estimand, origin-registry v2 and
dependence/power design rather than creating a second statistical authority.

The protocol is metadata only. Local Git and semantic hashes make changes
visible, but they do not replace the required external registry, trusted time,
FMV risk margins, power calibration or source admission.

Canonical semantic SHA-256:
`152b453db2e19e4ea16007865cfabb91bf9147cef46105da1f9f584956769fbd`.

## Candidate family

The comparison contains exactly one incumbent and four challengers:

1. the current unweighted 64x64 MLP, bound to the current source and
   configuration through cross-platform UTF-8/LF-normalized SHA-256 values;
2. the same MLP family with true observation-level 180-day decay weighting;
3. standardized Ridge;
4. an additive cubic-spline Ridge GAM without interactions;
5. deterministic CPU LightGBM with an explicit bounded grid.

The incumbent has no tuning grid. Challenger tuning may occur only inside
nested, externally registered development origins. Future-holdout tuning is
forbidden. Version 6 binds every challenger to the normalized source hash
`887f3b00d33231b52c58395ef43b5310624222922e955a6425d723e885cb5e19`.
The scoring engine is independently bound to
`034a06c14ec5aff337ab58cf4ab2e79a63c49f2f1d656dbc5fb3bd950c310ffc`.
The origin-envelope boundary is bound to
`0c74cff02e886075b25d1f0893ec5b756a4e6661c88ee6cfd57aa8f07caaff8d`.
The request-preparation boundary is bound to
`e4dc1366ac1e13c1f2b4083ac6c8d5f24036cd743cae678359bae8f1435ed748`.
The receipt/HEAD verifier is bound to normalized source hash
`d3224f53aa1912301ac87f5db0c4721547978c1da17358411d37adc558299eb7`
and its exact wire-contract file to
`1f9d1f6495f716b7afb3497195626b24fe427b7dedf3f6f001c00051d6688047`.
The registry trust/conformance implementation is bound to normalized source
hash
`e7defb0700665bb68ca4debe490b6920816739f6882850752f304827b17cfdf1`
and its exact trust/transport contract file to
`57ce79771d25cf6a858a2c2fef585d202b66d05aff6a107b834f2afb9407df39`.

The weighted MLP uses an exact observation-level exponential loss, a frozen
180-day half-life, a deterministic 64x64 ReLU network and analytic gradients.
Ridge and additive spline-Ridge use training-only scaling. LightGBM requires
exactly version 4.6.0, deterministic CPU mode and one worker; a missing or
mismatched optional runtime fails explicitly.

## Metrics and authority

The primary metric is monthly-level-neutralized MAE in EUR/MWh. Secondary
metrics are inherited exactly from the existing estimand, including RMSE,
bias, tail error, BASE/PEAK/OFFPEAK error and the separately gated economic
metrics. All candidates use the same complete-case rows and equal origin
weighting after within-origin energy weighting.

Only neutralized MAE, RMSE, bias and weighted P95 have a sufficiently closed
formula for the synthetic engine. Product and economic metrics remain
`UNSUPPORTED_NEVER_PASS` with an explicit missing-input reason. Results are
reported for all four lead buckets. A bucket without common rows is retained
as unsupported, so an aggregate diagnostic cannot hide missing coverage.

Market consistency remains an uncompensated hard gate. Statistical margins
remain pending the existing FMV-risk MDE and power design. Insufficient power
or coverage is `UNSUPPORTED_NEVER_PASS`.

The CH monthly BASE solver remains the sole monthly-level authority. Shape
scores subtract the energy-weighted local delivery-month mean separately from
prediction and truth; they cannot reward a challenger for rewriting the
forward level.

## Prospective cohort

The first new cohort schedules 12 monthly origins from October 2026 through
September 2027. Each origin covers lead months 1 through 36 and the four
existing horizon buckets. The timestamps are locally frozen before their
windows, but every slot has:

- `externally_registered=false`;
- `countable_origin=false`;
- `truth_open_authorized=false`.

Therefore the scheduled count is 12 while the countable scientific count is
zero. A missed slot is never shifted, backfilled or reweighted. The cohort may
need extension after the outcome-blind power design determines the required
number of unique origins; twelve is not a promotion claim.

T057 is neither read, referenced nor reused. ENTSO-E recovery and admitted
prospective availability remain acquisition prerequisites, not reasons to
alter the schedule after observing outcomes.

## Synthetic execution boundary

Synthetic fixtures must use unique ordered UTC timestamps, a strict
pre-origin training information set, finite numeric features and an exact
feature schema. Inputs are copied read-only. The scorer accepts only the exact
five-candidate inventory and the same complete-case intersection for every
candidate.

Synthetic fits and reports carry the source fixture identifier and an
immutable negative authority object. Their manifests state
`real_data_training_performed=false`, `real_truth_opened=false`,
`countable_origin=false` and `ranking_or_selection_performed=false`. The code
has no file, network, Databricks or CT access path.

## External-registration preparation boundary

`pfc_shaping.lt.origin_registration_envelope` closes the local preparation
step without claiming an external registration. It builds exact canonical
schedule-entry bytes with domain-separated identities for an independent
Ed25519 signer, attaches only a caller-supplied signature, and verifies all
twelve individually signed entries against one caller-held public key. It
accepts no private key.

The signed schedule must match the exact ordered cohort, each EEX trading day
must belong to its cadence month, every capture window must contain the frozen
origin, and the registry deadline must precede the first Swiss local delivery
month. The Europe/Zurich-to-UTC conversion is checked across CET/CEST rather
than inferred from an abbreviation.

An information-set envelope requires exact hashes for the structural origin
inventory, causal EEX inventory and vintage, solver configuration, candidates,
predictions, scenarios, universe, ex-ante mask rule, calendar/strata, runtime,
wheel and source revision. Verification requires the exact signed schedule
bytes and trust key again; a different valid schedule cannot be substituted.

Even after schedule-signature verification, the envelope records the trusted
origin-time receipt, independent request signature, remote compare-and-append
receipt and fresh remote HEAD observation as missing. Every operational,
scientific, truth-opening and production authority remains false. Synthetic
signatures in tests qualify only the verifier and are not written as project
evidence.

## Registration-request construction boundary

`pfc_shaping.lt.origin_registration_request` translates one reverified
information-set envelope into the exact field inventory of
`ch_lt_origin_registration_request.v2`. It derives the origin ID and request ID
with the protocol domains, binds the expected compare-and-append predecessor,
the selected signed schedule entry, every information-set commitment and the
SHA-256 of exact opaque trusted-time receipt bytes.

The production module accepts only public keys and caller-supplied signature
bytes. It rejects noncanonical UUIDs, invalid sequence/predecessor pairs,
request/schedule signer-role collapse, chronology violations, malformed
base64, changed hashes and requests re-signed after information-set mutation.

Successful verification proves only cryptographic integrity under the supplied
keys. Trusted-time semantics and the external request-signer role remain
unadmitted; no origin becomes registered or countable.

## Receipt and fresh-HEAD construction boundary

`pfc_shaping.lt.origin_registration_receipt` implements a local, hash-frozen
wire draft for `ch_lt_origin_registration_receipt.v2` and
`ch_lt_origin_registry_head_observation.v1`. Receipt and observation IDs use
separate domain-separated SHA-256 derivations; Ed25519 signatures use distinct
signature domains and exact canonical bytes. The verifier requires a registry
public key distinct from the request and schedule keys and reverifies the
complete request/envelope/schedule chain.

The HEAD binds the exact receipt bytes, sequence and caller-supplied SHA-256
nonce. Freshness requires
`committed <= observed <= caller_verification_time <= expires`, with a strictly
positive TTL no longer than 300 seconds. The library does not generate a
nonce, own a private key, read a clock, perform I/O or execute registry CAS.

A cryptographically valid receipt must carry the protocol's exact
`countable_prospective_origin=true` wire claim. This claim is reported
separately from local authority: even with a valid fresh HEAD,
`externally_registered`, `countable_origin`, truth opening, training,
selection, scientific admission, production and promotion all remain false.
They require external approval of the wire contract, registry-key trust
admission, an independently operated remote CAS/WORM service, trusted commit
time and conformance/security evidence.

The wire contract ID is
`290b108770dc799eeb83ca9f0a046aa8436878858224b68f757a5d9bfacbcc33`;
its exact file SHA-256 is
`1f9d1f6495f716b7afb3497195626b24fe427b7dedf3f6f001c00051d6688047`.
The incompatible SQLite reference remains test-only and is not promoted.

## Registry trust and transport-conformance boundary

`pfc_shaping.lt.origin_registry_conformance` verifies exact signed trust
bundles under a caller-supplied Ed25519 root public key. Bundles contain a
strictly ordered public registry-key inventory, exactly one currently active
signer, half-open validity windows and an append-only lifecycle chain.
Historical keys may verify receipts only inside their frozen window;
revoked or compromised keys are always rejected. The trust root and registry
signer roles must remain cryptographically distinct.

The same module provides a deterministic, thread-safe, in-memory model of the
transport-neutral `get_head`, compare-and-append and operation-lookup
semantics. It exercises atomic sequence/predecessor comparison, uniqueness,
exact idempotent retry, divergent-retry rejection, immutable committed bytes
and sanitized rejected-operation retention. Candidate preparation reverifies
the complete receipt/request/envelope/schedule chain against the selected
bundle key before this synthetic state-machine test.

This is not a registry client or service. It owns no private key, credential,
clock or nonce; performs no filesystem, database or network I/O; and cannot
provide remote linearizability, WORM retention, service identity or trusted
commit time. A locally valid trust-root signature admits neither that root nor
the registry service. All registration, countability, truth-opening,
training, selection, scientific, production and promotion authorities remain
false.

The trust/transport contract ID is
`84839a92bc62426c964da9eaefcf719be80808c79089b7ac2142af14d033206b`;
its exact file SHA-256 is
`57ce79771d25cf6a858a2c2fef585d202b66d05aff6a107b834f2afb9407df39`.
