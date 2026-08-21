# Session handoff - ENTSO-E artifact-package integrity preflight

Date: 2026-08-06  
Decision: `D-20260806-240`  
Status: `PASS_SYNTHETIC_PACKAGE_INTEGRITY_MODEL_NO_GO`

## Outcome

D240 verifies exact local artifact bytes before parsing and then runs D239 in
memory. It binds the exact D233, D236, D238 and D239 proofs plus the current
ENTSO-E intake contract. It imports no Databricks connector and opens no real
artifact.

The verifier admits only one content-addressed `SYNTHETIC_TEST_ONLY` package
below `build/governed-source-intake/synthetic-fixtures/<snapshot_id>`. It
requires exactly three ordered UTF-8 NDJSON files and checks exact inventory,
paths, hashes, byte sizes, row counts, columns, logical types, normalized schema
hashes, LF framing and resource caps before parsing. It rejects duplicate JSON
keys, non-finite values, malformed types, unsafe/link paths, truncation and
source-proof drift.

After integrity succeeds, D239 profiles the frames in memory. The durable proof
contains only descriptors, counts and hashes, never the synthetic values. The
data-quality method materially influenced this boundary: technical integrity,
analytical quality and real evidence authority are reported separately. A
green synthetic integrity result cannot self-declare empirical trust.

## Fail-closed authority and cost boundary

Only `synthetic_artifact_integrity_verified` is true. Real receipt/hash/source
authenticity, external time, same-snapshot PIT, seasonal diagnostics, training,
selection, model input, candidate assembly, promotion and production are all
false. Real large-Parquet streaming is deliberately not implemented.

No Databricks connection or statement, Warehouse start, network call, `H:`
access, remote write, real artifact path or real ENTSO-E value was used. D240
generated no Databricks cost.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-ARTIFACT-PACKAGE-INTEGRITY-PREFLIGHT-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_artifact_package_integrity.py`
- `tests/test_entsoe_artifact_package_integrity.py`
- `build/databricks-eex-daily/materialize_entsoe_artifact_package_integrity.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057, `H:` or heavy desk-data file was opened or
changed by D240. Existing unrelated worktree changes were preserved.

## Canonical identities

- contract raw SHA-256 / canonical content ID:
  `26631c82495df6f4616c2841bc73cbb5b9da13f5a265091fac5bcfcb5a21fe49` /
  `c4b69dd9e1ba790a5bfe3ec3c2ddc7c6555d907902f7ff429d582b820d43369e`
- validator SHA-256:
  `d485f6a3c39c4aacab2979640fd4eefec61a197d83ed0a6f352e1c62246188a6`
- tests SHA-256:
  `a5ea73ea9e239ff59e6347424ed148e3668b5545fadcf2028468af37766126ac`
- materializer SHA-256:
  `a9701a5c1c8995ca0db73b3a0d45b33ec54ca87637b97d0deb6a6fc620e9bf2c`

Selected reproducible proof:

- content ID:
  `5595a20c5b997485bbaa0e3aa41f90b131190e3e89d812b1acfdfaaebc88536b`
- manifest SHA-256:
  `97fc381771033c31da4f79789750638c70676bf32d1d3e844edc2609b3448ec3`
- assessment SHA-256:
  `eaee7527695aadcd9a648a676baddd8220fe04c0b8fcc5b9bc17395f97f5e220`
- integrity-receipt SHA-256 / canonical content ID:
  `a2b069a54fe722611d697565ec53ecdce270199311fcf74db7dc12cb1b97b462` /
  `f3e4ea1f550c63af578ca31d09db56404694fe4b9fcb0b3cefd8990b3180c104`
- synthetic package manifest raw SHA-256 / canonical content ID:
  `9ddd44261b06a68d8aeced75638ad2979ca1c1d45ff005437b2790f4dc06b680` /
  `ae28c79fba5ff66b0c06ad920d25df5742a35d7e671d236f19ae3593ca9ace7d`
- path:
  `build/databricks-eex-daily/2026-08-06/entsoe-artifact-package-integrity-proofs/5595a20c5b997485bbaa0e3aa41f90b131190e3e89d812b1acfdfaaebc88536b/`

The proof binds source proof content ID / manifest SHA-256 pairs:

- D233: `314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53` /
  `d7f6ad3af60d2efd087718e0515c8b73a6534387ab12c7c7a5f88f42aeadf2b4`
- D236: `772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247` /
  `9d76631395176718b600ca0de8f7303ad8faea931c24298f514a54354da3d6e0`
- D238: `2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8` /
  `922a31ce0bf54b28e00b352bae1e4fa1d66fa357adf687c9a5f456e94d4511f6`
- D239: `5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b` /
  `2d733c356b34ad0280802ed9d77b47f4d445e4d82e2ac6c45595f26f81787789`

## Synthetic fixture

The fixture contains 33 series-dimension rows, 792 latest rows and 792 vintage
rows. All 1,617 rows pass exact-byte verification before the existing D239
quality profiler runs. This is algorithm test evidence, not empirical ENTSO-E
history and not a substitute for governed real data.

## Verification

Every shell action verified cwd and Git top-level as the canonical
`C:\Users\jbattaglia\PFC_LT`. Mutable outputs stayed below `build/`.

- Ruff on validator, tests and materializer: passed.
- focused D240: `14 passed in 3.88s`.
- D231-D240 acquisition chain: `358 passed in 8.47s`.
- materializer executed twice: identical proof content ID.
- proof counters: zero Databricks connection/statement, Warehouse start,
  network call, `H:` access and remote write.

Focused test command:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe -m pytest tests\test_entsoe_artifact_package_integrity.py -q --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-d240-focused
```

Materialization command, executed twice:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe build\databricks-eex-daily\materialize_entsoe_artifact_package_integrity.py --output-root C:\Users\jbattaglia\PFC_LT\build\databricks-eex-daily\2026-08-06\entsoe-artifact-package-integrity-proofs
```

## Remaining gaps and next permitted step

D240 is intentionally limited to small synthetic NDJSON. Before real exports
can be admitted, a separate bounded streaming verifier must handle the governed
Parquet bytes without loading unbounded content, bind a genuinely independent
real receipt and preserve the same fail-closed path, hash, schema and authority
rules. That can be designed and roasted locally without querying Databricks.

Real metadata/mapping ownership, sign semantics, artifact hashes, normalized
values, 730 complete days, trustworthy vintage depth, same-snapshot PIT and a
new independent future holdout remain absent. Training, selection, model input,
candidate assembly, promotion and production remain false; T057 stays sealed.
The monthly solver remains sole level authority, LT remains independent from
CT, OMPEX remains benchmark-only and AFRY descriptive.
