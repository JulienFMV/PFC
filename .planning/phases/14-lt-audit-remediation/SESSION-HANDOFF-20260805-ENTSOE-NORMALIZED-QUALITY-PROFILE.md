# Session handoff - ENTSO-E normalized quality profile

Date: 2026-08-05  
Decision: `D-20260805-239`  
Status: `PASS_SYNTHETIC_VALUE_FREE_QUALITY_PROFILE_MODEL_NO_GO`

## Outcome

D239 implements the local quality algorithm that will be applied only after a
future independently hash-admitted ENTSO-E package exists. It binds the exact
D233-D238 source chain and the current ENTSO-E intake contract, but opens no
real artifact and imports no connector.

The validator enforces:

- exact fields and grains for series dimension, latest and vintages;
- unique keys, non-null required fields, finite values, non-negative integer
  revisions and complete dimension/fact series joins;
- all required Swiss groups, six generation components and CH-DE/FR/IT/AT for
  physical flow, scheduled exchange and day/month/year-ahead NTC;
- exact `MW`, `MWh` and `EUR/MWh` units, admitted native resolutions and one
  explicit sign convention per series;
- UTC native-grid alignment and no implicit resampling/fill;
- availability before target for forecast/price/schedule/capacity series and
  not before target for actual/storage series;
- non-decreasing revision/load chronology and exact equality between latest
  and the last vintage on every overlapping target key;
- rolling-origin replay using only `as_of_utc <= origin_utc`.

The per-series output is value-free. It records expected/observed native slots,
missing slots, complete UTC days, earliest trustworthy `as_of`, backfill status
and overlap coverage. A clean structural profile remains distinct from the
730-complete-day seasonal threshold and from model authority.

The data-quality method materially influenced D239 by separating hard blockers
(grain, joins, leakage, units) from decision caveats (coverage, backfill and
holdout), and by reporting rates/coverage rather than only counts.

## Fail-closed authority boundary

D239 rejects `REAL_HASH_ADMITTED_ARTIFACTS` even when a caller passes booleans
claiming that hashes and a receipt were verified. A future real path must
compose an independent receipt/artifact verifier; self-declared booleans cannot
bootstrap trust. The current fixture exists only to test validator behaviour.

No Databricks connection or statement, Warehouse start, network call, `H:`
access, remote write, real artifact path or real ENTSO-E value was used. D239
generated no Databricks cost.

## Decision-number reconciliation

While the batch was active, the live worktree added canonical D237 for RFC 3161
request DER and D238 for the daily capture reservation. The quality profile was
therefore assigned D239 and updated to bind both exact proofs. No existing
decision or artifact was overwritten.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-NORMALIZED-QUALITY-PROFILE-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_normalized_quality.py`
- `tests/test_entsoe_normalized_quality.py`
- `build/databricks-eex-daily/materialize_entsoe_normalized_quality_profile.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057 or heavy desk-data file was opened or
changed by D239. Existing unrelated worktree changes were preserved.

## Canonical identities

- contract raw SHA-256 / canonical content ID:
  `3cfe59d8700e8b29662918de3dc2941678267520353fc271c852874321f345e1` /
  `7ff2d5b7808f636f2c181fa92b80cc4ae86dccdcaad84f6da8dc3d13499736ab`
- validator SHA-256:
  `bf08642a7ce6641c47f8c47eff81d676e4e35b87d195a7398479cf77dde090cd`
- tests SHA-256:
  `d2dfd393ba581dfba3d20f2c4eb46a5816b979be50e3aabe955c0db2f0249517`
- materializer SHA-256:
  `5f6cd352ca868a00e2b4fcc909a00cd7c32c199be678f1b2c6f73119d518b42c`

Selected reproducible proof:

- content ID:
  `5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b`
- manifest SHA-256:
  `2d733c356b34ad0280802ed9d77b47f4d445e4d82e2ac6c45595f26f81787789`
- assessment SHA-256:
  `4dffdaa6abc48c25c80009e6c88e4d8c401053dd074580be34b50e74de07ccb0`
- value-free series profile SHA-256:
  `b77f37f74594737e21fc7dffa5dc15ca39f196f77ba2a23ffa325bebd1c62415`
- path:
  `build/databricks-eex-daily/2026-08-05/entsoe-normalized-quality-profile-proofs/5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b/`

The selected proof binds exact proof content IDs / manifest SHA-256 values:

- D233:
  `314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53` /
  `d7f6ad3af60d2efd087718e0515c8b73a6534387ab12c7c7a5f88f42aeadf2b4`
- D234:
  `bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766` /
  `7d86330e2143e2009947b486eb1e8428dd6022735ebc33c9aa4cfc4b871c5268`
- D235:
  `93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1` /
  `a10bbfa380652365684213197862139bf958dc33e706d81635cc2a6ae4582f02`
- D236:
  `772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247` /
  `9d76631395176718b600ca0de8f7303ad8faea931c24298f514a54354da3d6e0`
- D237:
  `53e2222392f71541d28e05e1dfc912361c02003b4278f2770109827319d4e9c1` /
  `c5d8be620f1c2f5012aa7174a5949adbc0d74708a162ac215c44f40f1f8dcdcf`
- D238:
  `2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8` /
  `922a31ce0bf54b28e00b352bae1e4fa1d66fa357adf687c9a5f456e94d4511f6`

The earlier local proof
`113583549d591233eef33a89ac8b9a4ec92215dffefc0ba9f82e503b02870487`
preceded the D237/D238 binding and is superseded, not selected.

## Synthetic fixture and expected findings

The fixture contains 33 series, 792 latest rows and 792 vintage rows over one
complete UTC day. Latest/vintage overlap is exact and native-grid completeness
is 100%. This does not create empirical evidence. The expected findings are:

- `SYNTHETIC_FIXTURE_NOT_EMPIRICAL_EVIDENCE` - critical;
- `SEASONAL_COMPLETE_DAY_THRESHOLD_NOT_MET` - high;
- `BACKFILLED_SERIES_NOT_RETROACTIVE_PIT_EVIDENCE` - high;
- `NEW_INDEPENDENT_FUTURE_HOLDOUT_MISSING` - critical.

## Verification

Every shell action verified cwd and Git top-level as the canonical
`C:\Users\jbattaglia\PFC_LT`. Mutable outputs stayed below `build/`.

- Ruff on validator, tests and materializer: passed.
- focused D239: `16 passed in 1.74s`.
- D231-D239 acquisition chain: `344 passed in 3.02s`.
- final local quality matrix with legacy-readiness guards:
  `31 passed in 13.94s`.
- materializer executed twice after D237/D238 reconciliation: identical proof
  content ID.
- proof counters: zero Databricks connection/statement, Warehouse start,
  network call, `H:` access and remote write.

Focused test command:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe -m pytest tests\test_entsoe_normalized_quality.py -q --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-d239-focused
```

Chain command:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe -m pytest tests\test_databricks_zero_query_acquisition_plan.py tests\test_entsoe_metadata_admission.py tests\test_eex_entsoe_governed_acquisition_package.py tests\test_entsoe_physical_mapping_compiler.py tests\test_eex_entsoe_external_time_batch.py tests\test_entsoe_bounded_execution_receipt.py tests\test_eex_entsoe_rfc3161_request_der.py tests\test_entsoe_daily_capture_reservation_ledger.py tests\test_entsoe_normalized_quality.py -q --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-d231-d239
```

Materialization command, executed twice:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe build\databricks-eex-daily\materialize_entsoe_normalized_quality_profile.py --output-root C:\Users\jbattaglia\PFC_LT\build\databricks-eex-daily\2026-08-05\entsoe-normalized-quality-profile-proofs
```

## Remaining gaps and next permitted step

D239 does not read artifact paths and cannot admit real values. The next safe
local batch is an independent real-receipt/artifact admission verifier that
hashes exact repo-local bytes before opening them, rejects truncation/caps and
then passes only descriptor-bound admitted content into this quality profiler.
It can be built and roasted without Databricks.

Real metadata/mapping ownership, sign semantics, artifact hashes, normalized
values, 730 complete days, trustworthy vintage depth, same-snapshot PIT and a
new independent future holdout remain absent. Training, selection, model input,
candidate assembly, promotion and production remain false; T057 stays sealed.
The monthly solver remains sole level authority, LT remains independent from
CT, OMPEX remains benchmark-only and AFRY descriptive.
