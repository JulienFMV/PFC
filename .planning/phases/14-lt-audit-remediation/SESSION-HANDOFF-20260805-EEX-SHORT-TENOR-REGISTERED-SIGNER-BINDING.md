# Session handoff - D230 registered signer binding

Date: 2026-08-05

## Outcome

D230 composes the exact D228 signed replay and D229 threshold trust registry
without opening market values. The `training_execution`,
`selection_governance` and `trusted_time` public keys must be the exact same
absolute files and caller-held hashes in both replays. Registry resolution is
derived from the two D228 signed-payload `observed_at_utc` fields; callers
cannot inject alternate role times. D229 is replayed twice so the trusted-time
key must resolve at both observations. The registry head must postdate the
asserted events, be valid at caller reference time and have no more than 31
days of local publication lag.

The roast identified and preserved a critical boundary. D228 does not contain
an independently timestamped DSSE signature-generation time.
`observed_at_utc` concerns the execution-attestation observation, and for the
time envelope it remains a signed payload assertion. D230 therefore proves a
local registered-key identity binding at asserted payload times only. Actual
signature time, backdating resistance, external time and every model authority
remain false.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-REGISTERED-SIGNER-BINDING-CONTRACT-V1.json`
- `pfc_shaping/validation/eex_short_tenor_registered_signer_binding.py`
- `tests/test_eex_short_tenor_registered_signer_binding.py`
- `build/databricks-eex-daily/materialize_short_tenor_registered_signer_binding.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, heavy desk-data or restricted AFRY numeric file was touched.
No OMPEX file or `H:` path was opened.

## Canonical identities

- D230 contract file SHA-256:
  `849cb6cea28913b87b69cea463d54ea9f110150467343cbd8f27f602a07fca23`
- D230 contract canonical content ID:
  `2fa6d60542f513faf132bc33dbf6ea17c98693b7658c37ad9e92195dcdd053f0`
- validator SHA-256:
  `98b3fda4e7f2ca6e83ca9bf4f7e72582c7402f7010722583c4588939ff02ab15`
- tests SHA-256:
  `d0cbe3770c0e939c72afa089c994137112bfdcde6b95599cfa79a47cdb9f699d`
- build-only materializer SHA-256:
  `0b8e9d6da3918a7dda3bc95c475427e79564e406e3a753878f362d4d70eafbfc`
- research note SHA-256:
  `9e2d4e3a0abfd0c707ebd404807f606f3ec269656ee4bdcf94e0dc5a17783289`

Reproducible build-only proof:

- content ID:
  `db365ca3045f989cd7257594f37ccec0c8b306475e9f4149b525f1732fff2ad3`
- directory:
  `build/databricks-eex-daily/2026-08-05/short-tenor-registered-signer-binding-proofs/db365ca3045f989cd7257594f37ccec0c8b306475e9f4149b525f1732fff2ad3`
- manifest SHA-256:
  `e2c2cfb19ada0f864fa655ffb42de267397294a6b80ed35fb96efe9dd6e9a240`
- assessment SHA-256:
  `a72fa04eb2b0660b8557b35e8294b823be453e00fc9f8a4e8a81588f928d188a`

The manifest binds exact D228 proof
`eed94d79109a5f196f49cf2bb11950b79889c8384cc82c1ef70a33a69cdebc5e`
and D229 proof
`72483a8aee28241db716a07355ed27ac4065b049d74e42f1c2224809821cbf61`.

## Verification

Every shell command first verified exact cwd and normalized Git top-level
`C:\Users\jbattaglia\PFC_LT`. Mutable paths remained below `build/`.

- focused D230 roast: `25 passed in 4.25s`;
- six-file D227-D230/governed-acquisition/snapshot-publication adjacency:
  `228 passed, 1 skipped in 45.11s`;
- expanded 28-file EEX/PIT/monthly/ENTSO-E/OMPEX/LT adjacency:
  `491 passed, 1 skipped in 192.64s`;
- independent final nine-file D227-D230/EEX-vintage/governed-acquisition/
  ENTSO-E confirmation: `266 passed, 1 skipped in 108.08s`;
- Ruff with the hash-checked repo-local binary and `--no-cache`: passed on the
  validator, tests and materializer;
- two exact materializations returned the same proof content ID;
- proof execution counters are zero for Databricks requests, Warehouse starts,
  network calls, `H:` accesses and remote writes;
- no private key, real registry, real receipt or numeric market/model value is
  persisted by the proof.

The final reclosure additionally fail-closes if either upstream validator
returns a drifted contract ID, a missing integrity flag, a newly true authority
or a non-zero Databricks/network/access counter. This prevents a monkeypatched
green sub-result from smuggling authority into the combined assessment.

A transient parallel `registry_bound_replay` draft duplicated D230 with weaker
publication semantics. Its useful upstream-drift checks were merged into this
canonical implementation, then its contract, validator, tests and materializer
were removed. Any ignored content-addressed draft proof left below `build/` is
non-authoritative and must not supersede the canonical proof above.

One existing warning in `test_governed_lt_acquisition.py` reports pandas
timezone loss when converting to a monthly PeriodIndex. It is unrelated to
D230 and the test passes.

## Authority and remaining gaps

- Actual DSSE signature-generation time is absent and unverified.
- A signed `observed_at_utc` is not RFC3161-equivalent external time and does
  not prove non-backdating.
- Registry owner strings and governance roots are not externally verified.
- D227/D228 source-key sets are not mapped to named signed EEX and ENTSO-E
  acquisition/source-time receipts.
- Same-snapshot EEX/ENTSO-E PIT availability remains missing.
- Training, selection, model input, candidate assembly, promotion and
  production remain strict `NO_GO`.
- The batch makes no empirical PFC-quality or OMPEX-superiority claim.
- The monthly solver remains sole monthly-level authority, LT stays independent
  from CT, OMPEX remains post-candidate benchmark-only, AFRY descriptive and
  T057 sealed.

## Next safe batch

Do not continue inventing local trust authority. Prepare, without executing,
the cheapest governed acquisition package that names each EEX/ENTSO-E source
role and specifies the exact external timestamp or transparency evidence
required for the four DSSE envelopes. Keep Databricks execution disabled until
the user explicitly authorizes a bounded query budget. If the providers cannot
produce such evidence, record the downgrade rather than treating local fixture
timestamps as authoritative.
