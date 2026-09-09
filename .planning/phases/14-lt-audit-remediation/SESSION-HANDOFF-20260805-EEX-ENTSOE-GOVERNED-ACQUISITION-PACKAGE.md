# Session handoff - governed EEX/ENTSO-E acquisition package

Date: 2026-08-05  
Decision: D-20260805-233  
Status: `PREPARED_D233_LOCAL_ZERO_EXECUTION_NO_MODEL_AUTHORITY`

## Outcome

D233 freezes the complete local evidence package expected before any governed
EEX/ENTSO-E value acquisition. It composes D231 and D232, reuses the existing
EEX `prd.gold` snapshot without a new query, and keeps the future ENTSO-E
capture ceiling inactive pending real schema admission and a new explicit,
costed user authorization.

EEX explicitly covers `DAY`, `WEEK`, `WEEKEND`, `MONTH`, `QUARTER` and `YEAR`
in `EUR/MWh`. `DAY/WEEK/WEEKEND` remain zero-mean within each solver month;
the monthly solver remains the sole monthly-level authority. ENTSO-E cadence
must follow the applicable Swiss market-time regime: historical hourly Swiss
prices are not classified as a missing native 15-minute truth.

No Databricks statement or write, Warehouse start, network call, `H:` access,
remote write or market-value row opening occurred.

## Changed files

- `.planning/phases/14-lt-audit-remediation/EEX-ENTSOE-GOVERNED-ACQUISITION-PACKAGE-CONTRACT-V1.json`
- `pfc_shaping/validation/eex_entsoe_governed_acquisition_package.py`
- `tests/test_eex_entsoe_governed_acquisition_package.py`
- `build/databricks-eex-daily/materialize_eex_entsoe_governed_acquisition_package.py`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057 or heavy desk-data file was opened or
changed by D233.

## Canonical identities

- contract raw SHA-256:
  `97edaf93c639953365d20d7539380a706f0f8fdbec9476af94262fb1d4b2204d`
- contract canonical content ID:
  `4f6e644b15593e8d4ae2ca196e1d52a002d4255bdf42e7a137982edb854f0fa0`
- validator SHA-256:
  `95b6875d9d4b42c164473d0f36127dfbcd4cecbcde7b6597784bb48f04704ff8`
- tests SHA-256:
  `c34f70c903875ec02492e2e768299e15bc1fbdbc508cf301668a5b044e20de94`
- materializer SHA-256:
  `da576f365ba162788119161ad50558e062d3a4a66a366d4f5b10dd740e433b80`
- corrected ENTSO-E engineer note SHA-256:
  `bf8f07d224a341176f96a6efb3c1b04c68ab718c278a87daa7bdf847291496f0`

Reproducible proof:

- content ID / assessment SHA-256:
  `314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53`
- manifest SHA-256:
  `d7f6ad3af60d2efd087718e0515c8b73a6534387ab12c7c7a5f88f42aeadf2b4`
- path:
  `build/databricks-eex-daily/2026-08-05/eex-entsoe-governed-acquisition-package-proofs/314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53/`

Two final materializations returned the same content ID.

## Verification

- D233 focused roast: `44 passed`.
- D231+D232+D233 focused matrix: `109 passed`.
- expanded acquisition, receipt, trust, PIT and ENTSO-E matrix:
  `418 passed, 3 skipped, 1 warning in 130.60s`.
- Ruff on validator, tests and materializer: passed.
- The warning is the pre-existing timezone-to-period warning in
  `ingest_energy_charts.py`; it is unrelated to D233.
- Local-boundary replay verifies exact D231/D232/D227/D229/D230/intake bytes
  and the existing EEX manifest/artifact hashes without parsing price rows.

## Frozen future package

- Four DSSE/in-toto roles are mandatory: `eex_acquisition`,
  `eex_source_trusted_time`, `entsoe_acquisition` and
  `entsoe_source_trusted_time`.
- One unpredictable governed challenge precedes capture.
- The four exact envelope hashes form one RFC 9162 Merkle root, with one
  inclusion proof per envelope.
- Exactly one external-time profile is allowed per batch: RFC 3161 or a
  governed transparency log. It proves bounded existence, not exact signature
  time and not retroactive historical PIT availability.
- The inactive future ceiling is one capture per Europe/Zurich day, one
  Warehouse start, at most three read-only ENTSO-E statements, zero retries and
  64 GiB total local export. It requires a new monetary estimate, currency cap
  and explicit user authorization before activation.

## Remaining blockers

- No real D231 metadata result has passed D232.
- No physical ENTSO-E column mapping or value SQL is authorized.
- No real four-envelope source package or external-time proof exists.
- Historical EEX and backfilled ENTSO-E rows do not establish retroactive PIT.
- Governed ENTSO-E values, prospective captures and a new independently frozen
  future holdout are still missing.
- Training, selection, model input, candidate assembly, promotion and
  production remain false. T057 stays sealed; OMPEX remains post-candidate
  benchmark-only and AFRY remains descriptive.
