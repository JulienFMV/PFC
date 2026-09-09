# Session handoff - LT origin registration envelope v1 - 2026-09-02

## Outcome

The next outcome-blind offline increment is complete. The LT runtime can
prepare exact bytes for independent schedule-entry signing, verify a complete
signed 12-slot schedule and bind one frozen origin information set to that
schedule without acquiring any external, scientific or production authority.

No real schedule was signed. All Ed25519 private keys and signatures used by
the tests were deterministic in-memory synthetic fixtures only. No fixture
signature was persisted as project evidence.

## Changed files

- `pfc_shaping/lt/origin_registration_envelope.py` (new): canonical payload
  construction, caller-supplied signature assembly, Ed25519 verification,
  exact-cohort validation, Europe/Zurich delivery-start checks and immutable
  negative information-set authority.
- `pfc_shaping/lt/evaluation_protocol.py`: source-bound v3 identity and new
  origin-envelope/package bindings; candidates, metrics, cohort and authority
  remain unchanged.
- `pfc_shaping/package_contract.py`: new LT preparation module added to the
  governed-wheel positive inventory.
- `tests/test_lt_origin_registration_envelope.py` (new): synthetic positive
  and adversarial signature, identity, chronology, cohort, DST, substitution,
  commitment and authority tests.
- `tests/test_lt_evaluation_protocol.py` and
  `tests/test_lt_package_contract.py`: source/package binding assertions.
- `docs/model/LT-ROLLING-ORIGIN-EVALUATION-PROTOCOL.md`: v3 registration
  preparation and authority boundary.
- `.planning/HANDOFF.md`, Phase 14 `DECISION-LOG.md`, and this handoff.

No CT, heavy desk data, monthly solver logic, production flag, model weight or
T057 file was opened or changed.

## Frozen identities

- evaluation protocol v3 semantic SHA-256:
  `e73d0b63160835f7c10b582152435361e51ac9568d433a721ded92327fac7137`;
- evaluation protocol normalized-LF source SHA-256:
  `101107387b767605e77f8e9a361b3d752951998c9b3cbd2e4713691c46a8d5a8`;
- origin registration envelope normalized-LF source SHA-256:
  `0c74cff02e886075b25d1f0893ec5b756a4e6661c88ee6cfd57aa8f07caaff8d`;
- package contract normalized-LF SHA-256:
  `43608e2a1f07743bb9a6a3ef247847086505bc617ceb6af18ae154ff46bc7efc`;
- runtime specification normalized-LF SHA-256:
  `c61b2261c4ee0048d72a70ddb183b99b6a50cb52bf677bfbc663441db232987f`;
- origin registry v2 protocol SHA-256:
  `6ea896ccdb35414b52237f2bcf1065755c3c10444b308ce905b60f472e68c697`;
- origin registry v2 protocol ID:
  `0fcf13246c50f6bc79d203437f7c5294495dc233f93873abbe2aeaf3dc282204`.

The challenger and scoring-engine hashes remain unchanged. The future cohort
remains `ch-lt-future-cohort-2026-10-v1`: 12 scheduled origins and zero
countable origins.

## Contract guarantees

- Production LT code imports only `Ed25519PublicKey`; it has no private-key,
  filesystem, network, truth-loading, fitting or registry-write path.
- Schedule-entry IDs are domain-separated hashes of exact canonical cores.
- Signatures require exact Ed25519 algorithm, raw-public-key identity,
  canonical standard base64 and exactly 64 decoded bytes.
- Every signed entry binds the frozen origin-registry protocol, official EEX
  calendar and settlement-definition hashes, explicit UTC windows and the EEX
  trading day.
- The schedule contains exactly the ordered October 2026 to September 2027
  cohort. Missing, duplicate, reordered or displaced origins fail closed.
- The first delivery boundary is recomputed as Swiss local month start and
  checked in UTC across CET/CEST.
- An information-set envelope binds the exact schedule manifest, selected
  entry, signer and all registry-v2 artifact commitments. Verification
  requires the exact schedule and key again.
- A different correctly signed schedule, malformed canonical bytes, changed
  commitment, recomputed identity with positive authority, or chronology
  violation fails closed.
- Trusted origin time, request signature, remote compare-and-append receipt
  and fresh remote HEAD observation remain missing. All authorities remain
  false.

## Verification

Dedicated synthetic mutation matrix:

```text
18 passed in 0.30s
```

Expanded evaluation, origin registry, estimand, dependence/power, curve
product, LT/CT import, hourly shape and optimizer matrix:

```text
300 passed, 1 skipped in 42.80s
```

Required LT minimum:

```text
58 passed, 1 skipped in 6.52s
```

The skip is the pre-existing optional TensorFlow boundary. Targeted Ruff
checks passed. Ruff formatting passes for the new implementation and tests;
the existing package test was intentionally left in its pre-existing layout
after limiting its diff to one relevant assertion.

Audit corrections made during the sequence:

- the first broad text search was too large and truncated; targeted reads
  verified the exact reference signing and schedule rules;
- a failed patch matched pre-format source context and made no change; it was
  reapplied against the exact current lines;
- a whole-file Ruff format on the package test produced unrelated churn; that
  churn was removed and only the new inventory assertion retained;
- envelope verification was tightened to require the exact signed schedule
  bytes and trust key again, closing valid-schedule substitution;
- first-target validation was tightened from simple chronology to the exact
  Europe/Zurich local-month boundary.

## Authority and cost

- local data rows opened: `0`;
- real schedule signatures created or persisted: `0`;
- production private keys loaded: `0`;
- real model training/retraining: `0`;
- real truth rows opened: `0`;
- Databricks connections/statements: `0/0`;
- Warehouse starts/resizes/creates: `0/0/0`;
- model artifacts written: `0`;
- CT changes: `0`;
- CH monthly solver-authority changes: `0`.

## Next action

1. Obtain FMV approval and an official versioned EEX calendar/settlement-event
   definition for the exact cohort; do not infer them from local dates.
2. Have the independent schedule authority sign the prepared entry bytes and
   freeze the exact manifest in the external trust domain.
3. At each live origin, add the trusted-time receipt, independent request
   signature, remote compare-and-append receipt and fresh HEAD observation.
4. Resume bounded governed EEX/ENTSO-E source admission when the platform is
   available. Local files through 31 August remain non-authoritative.
5. Only after those gates pass, add a separate governed real-data runner and
   keep future truth closed until independently registered maturity.

Durable decision: D-20260902-274.
