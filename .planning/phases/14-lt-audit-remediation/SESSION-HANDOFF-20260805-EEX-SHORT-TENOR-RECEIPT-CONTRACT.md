# Session handoff - D227 short-tenor receipt contract

Date: 2026-08-05

## Outcome

D227 freezes a value-free structural receipt boundary for future CH EEX
short-tenor training and selection. It does not authenticate a receipt, prove
point-in-time availability, run a fit or grant model/production authority.

The boundary binds, per outer origin, the exact D219-D225 chain, EEX and
ENTSO-E PIT catalogs, origin/target mask, candidate grid, inner-fold inventory,
implementation closure and runtime lock. Each inner fold separately binds its
inventory entry, EEX cutoff receipt, ENTSO-E cutoff receipt and input snapshot.
The outer origin is forbidden from inner training and validation. Training
binds the frozen candidate allowlist and loss commitment; selection must reuse
the exact origin, receipt, grid, folds and loss commitment.

The profile uses in-toto Attestation Framework v1.2, Statement v1, DSSE and the
SLSA provenance v1 predicate structure. The FMV local profile deliberately
rejects unknown fields, duplicate keys and unsafe logical names, while generic
SLSA consumers may ignore unrecognised fields. No cryptographic or trusted-time
authentication is claimed.

References:

- https://github.com/in-toto/attestation/blob/main/spec/README.md
- https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md
- https://github.com/in-toto/attestation/blob/main/spec/v1/envelope.md
- https://slsa.dev/spec/v1.2/build-provenance

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-RECEIPT-CONTRACT-V1.json`
- `pfc_shaping/validation/eex_short_tenor_receipts.py`
- `tests/test_eex_short_tenor_receipts.py`
- `build/databricks-eex-daily/materialize_short_tenor_receipt_contract.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI or heavy desk-data file was touched.

## Canonical identities

- receipt contract file SHA-256:
  `3168d75e245f033daa44f4c6dab9061f51e06f2b993797b7fb7d9c46dbe6b665`
- canonical contract content ID:
  `d4f7e4e87b8e059255e1a85a589f41d54596756d731ce4a9781b2dbc46d61068`
- validator SHA-256:
  `12e54315ac26702c9effd06c7ffd135cfe2fed51a2e79e5e19a6df3998631c97`
- tests SHA-256:
  `7de01220936dbbd2b35fa376a2e8bf9652795d3df439a951da374941a52f6ae0`
- build-only materializer SHA-256:
  `a9012fd6f96bf599941c831febdb1ee8654557e1b2feaa154ed870cdd00de6d8`
- research note SHA-256:
  `cf3c13cd9ff0da0a9390d060d11411ed321c7a3aa71cafeb698667a232cd44dc`

Reproducible build-only proof:

- content ID:
  `398b6ff48128a577f06a0106473a3cf71cdc6d411e3d1e877270d433cc6091cc`
- directory:
  `build/databricks-eex-daily/2026-08-05/short-tenor-receipt-contract-proofs/398b6ff48128a577f06a0106473a3cf71cdc6d411e3d1e877270d433cc6091cc`
- manifest SHA-256:
  `ad564fea14b6adcd126f93c745c08d3b9719c25826fa66f05e8d32a3707b5c3d`
- assessment SHA-256:
  `f84599b5996b629c68ee684833650b24be14b0af97b38d693b65baa89742c8b7`

The manifest records `databricks_request_count=0`,
`warehouse_start_count=0`, `network_call_count=0` and
`remote_write_count=0`.

## Verification

Every shell command first verified exact cwd and normalized Git top-level
`C:\Users\jbattaglia\PFC_LT`. `TEMP`, `TMP`, pytest basetemps and generated
outputs remained below `build/`.

- focused receipt tests: `55 passed in 0.18s`;
- adjacent EEX/PIT/fold/monthly/ENTSO-E/OMPEX/LT matrix:
  `329 passed, 4 skipped in 24.27s`;
- materializer executed twice and returned the same content ID;
- AST parsing passed; maximum relevant code line length is 98 characters;
- Ruff is unavailable in the governed repo-local runtime and is not claimed.

## Risks and authority

- The receipt schemas remain structural and unauthenticated. A receipt ID or
  self-declared key ID is not external identity or trusted time.
- Governed signed EEX PIT vintages, governed ENTSO-E, exact future origin/fold
  inventories and a newly frozen independent holdout remain mandatory before
  any empirical training or selection.
- OMPEX remains post-candidate benchmark-only; restricted AFRY remains
  descriptive; T057 remains sealed.
- The monthly solver remains sole monthly-level authority. Short-tenor layers
  may only add zero-mean within-month shape. LT stays independent from CT.
- Training, selection, model input, assembly, promotion and production remain
  strict `NO_GO`.

## Next safe batch

Design and test a path-based signed-envelope replay adapter against synthetic,
value-free receipts, with externally supplied trust anchors and trusted-time
interfaces left mandatory. Do not connect to Databricks or fit empirical
weights until the user explicitly authorises a costed query and the governed
PIT inputs plus independent holdout exist.
