# Session handoff - ENTSO-E incremental quality profile

Date: 2026-08-06  
Decision: `D-20260806-245`  
Status: `PASS_SYNTHETIC_INCREMENTAL_QUALITY_MODEL_NO_GO`

## Outcome

D245 implements the bounded analytical-quality layer that D244 deliberately
left open. It is entirely offline, repo-local and synthetic. D244 first verifies
the exact content-addressed Parquet package, hashes and schemas. D245 then makes
a second complete Arrow-batch pass and accumulates the normalized quality
checks without importing pandas or materializing a whole table.

The accumulator uses a repo-local SQLite spill capped at 1 GiB. It stores only
structural keys, UTC nanoseconds and SHA-256 row-value fingerprints; it never
stores raw numeric price, load, generation, flow, capacity or storage values.
The temporary database is removed in a `finally` path on success and failure.

The implemented checks cover exact dimension/latest/vintage grains, duplicate
rates, orphan joins, required groups, Swiss borders and generation breakdown,
units, native-grid alignment and completeness, sign semantics, snapshot and
availability timing, revision/load chronology, per-series vintage depth and
exact latest-versus-last-vintage agreement. The value-free series profile
contains only a hashed series ID, semantic metadata, counts, rates, dates and
backfill status.

The data-quality method materially shaped this batch: byte integrity,
batch-level validity, analytical-quality rates/findings and real-data authority
remain separate verdicts. D245 passes only the synthetic incremental mechanism.
Critical real-source gaps remain explicit instead of being averaged into one
green score.

## Authority and cost boundary

Only `synthetic_incremental_quality_profile_verified` is true. Real receipt,
real artifact hashes/values, real series inventory, source authenticity, real
PIT, training, selection, model input, candidate assembly, promotion and
production are false.

D245 made zero Databricks connections or statements, zero Warehouse starts,
zero network calls, zero `H:` accesses and zero remote writes. It therefore
generated no Databricks cost. D246/D247 are separate concurrent decisions; the
single D247 control-plane GET and stopped-Warehouse outcome do not belong to
D245's execution counters.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-INCREMENTAL-QUALITY-PROFILE-CONTRACT-V1.json`
- `pfc_shaping/validation/entsoe_incremental_quality.py`
- `tests/test_entsoe_incremental_quality.py`
- `build/databricks-eex-daily/materialize_entsoe_incremental_quality.py`
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

No CT, Power BI, AFRY, OMPEX, T057, `H:` or heavy desk-data file was opened or
changed by D245. Concurrent D246, D247 and D250 work was preserved.

## Canonical identities

- contract raw SHA-256 / canonical content ID:
  `15b9c24bb5b0cfefb27fef038497227baa926edb1433434ce0d0f7018f019c7c` /
  `1baf72727872e0013fd6a70c1e04c2ef0eeabd0faf66659a7f17fb10956c4184`
- validator SHA-256:
  `892e50a560f5af6810c1c8ba72d06a0a85a546ad8775d1dc17e09243e9ffe98c`
- tests SHA-256:
  `2cae80f49da6a13e56b7618387e3e1f4abafcc0897671059ae5dea61cf203e6a`
- materializer SHA-256:
  `2085304a3d4f3a97d4b2be23faf8436ee5f2362ba3d5528122153a4f5bf0b485`

Selected reproducible proof:

- content ID:
  `a5840e92c5ea783d9931069a8725352398e882831faf4e3fd6beb8f80a9653a1`
- manifest SHA-256:
  `9bf0f6333bbb024352ee8006441ad5b54e2280c65f8cdb147bcc578a4d55ded5`
- assessment SHA-256:
  `e32ed42f5c88d2ee05b7f0c5c6d24418a9fbe1ac1a4d76394d1e163f4ec8e520`
- summary SHA-256 / canonical content ID:
  `7d2811221e7e70b05ea8f25a09b2eec2cd8b220a029902e9c193ea42e24b80e6` /
  `a8fbdab0eba7d64d90148d64170918374eb71511c268298b7162ed0378b07f26`
- series profile SHA-256 / canonical content ID:
  `613bad2856b32592c565cdceee34a3da5a1c375b3014191b1b1608adb43b9b6e` /
  `2ae2b2f9d072d6114b04beb4abb45f6a57b58df288a2b1bd66cb343121918352`
- synthetic package manifest raw SHA-256 / canonical content ID:
  `3769c2a897b24a60d6d6275773a10bca665a1a675c7f592996f835877ba5b707` /
  `a3800d633e60c736de6a19816b3ce381a377e97ccb2bc4d01c15aab2a013b4de`
- synthetic quality context raw SHA-256 / canonical content ID:
  `a35fec05614b38d9b7699b9be439cd0723c656fdacf9716bb3d5d33467519a25` /
  `b61c52d3dd9d18f39429453e50901bbad5523af77e12be138cec67f4abc9d80e`
- D244 runtime receipt content ID:
  `dd91625a8ae28270bf70800e22daa3d910c3842daf83a2de9177fc1a1400c6cc`
- path:
  `build/databricks-eex-daily/2026-08-06/entsoe-incremental-quality-proofs/a5840e92c5ea783d9931069a8725352398e882831faf4e3fd6beb8f80a9653a1/`

Bound predecessor evidence remains exact:

- D239 proof content ID / manifest SHA-256:
  `5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b` /
  `2d733c356b34ad0280802ed9d77b47f4d445e4d82e2ac6c45595f26f81787789`
- D243 inventory contract content ID / contract / query SHA-256:
  `1e183fc51f2673cfc3fed0035dc4a5e3d84664f51ff8447a355264e5e819ddc6` /
  `ba8f6945b4a43b54762fa475228edabea4018bfea5871f2581dc53536c0743a1` /
  `16a989d2b1528f79b3ecb7a2d9f8f221a6be67cc1c4d3c622516c8bd0dde95e7`
- D244 proof content ID / manifest SHA-256:
  `42c1065bf66117a2be2c792424f08d08511056d0df5f3b4b706ac57af1fcf564` /
  `53e4fe281d405fea64e6f6084d0ae129f608c4f6ff88e944d60be3bf904177d4`

## Synthetic fixture and findings

The exact fixture has 33 series, 792 latest rows, 792 vintage rows and 792
latest/last-vintage overlap keys. The minimum grid ratio is 1.0, the minimum
complete history is one UTC day and SQLite peaks at 303,104 bytes. All D245
staging directories are empty after execution.

This remains algorithm evidence, not an empirical ENTSO-E sample. The proof
therefore preserves these findings:

- `SYNTHETIC_FIXTURE_NOT_EMPIRICAL_EVIDENCE` (`CRITICAL`)
- `D241_REAL_SCHEMA_MAPPING_REMAINS_INCOMPATIBLE` (`CRITICAL`)
- `D243_REAL_SERIES_INVENTORY_NOT_EXECUTED` (`HIGH`)
- `SEASONAL_COMPLETE_DAY_THRESHOLD_NOT_MET` (`HIGH`)
- `NEW_INDEPENDENT_FUTURE_HOLDOUT_MISSING` (`CRITICAL`)
- `BACKFILLED_SERIES_NOT_RETROACTIVE_PIT_EVIDENCE` (`HIGH`)

## Defects roasted

The first focused run reported `10 passed, 2 failed`. Both defects were fixed
before the final evidence was selected:

- an orphan latest-series ID triggered a `KeyError` while constructing the
  value-free profile; orphan rows are now excluded from the presentation layer
  after being counted as an exact hard failure;
- future-load validation used the verifier's wall-reference time instead of
  the package snapshot; all row-availability checks and the output snapshot now
  use the exact context/package snapshot, while the later verifier reference is
  retained only for evidence-time validity.

Final focused result: `12 passed in 3.52s`.

## Verification

Every shell action verified cwd and Git top-level as the canonical
`C:\Users\jbattaglia\PFC_LT`. Mutable inputs and outputs remained below
`build/`.

- Ruff on validator, tests and materializer: passed.
- focused D245: `12 passed in 3.52s`.
- D231-D245 acquisition chain: `386 passed in 18.66s`.
- materializer executed twice: identical proof content ID.
- proof content ID recomputed from canonical manifest core: exact match.
- proof counters: zero Databricks connection/statement, Warehouse start,
  network call, `H:` access and remote write.
- only `synthetic_incremental_quality_profile_verified` is true.
- repo-local incremental-quality staging entry count after all runs: zero.

Focused command:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe -m pytest tests\test_entsoe_incremental_quality.py -q --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-d245-focused-rerun
```

Materialization command, executed twice:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v2-final\python.exe build\databricks-eex-daily\materialize_entsoe_incremental_quality.py --output-root C:\Users\jbattaglia\PFC_LT\build\databricks-eex-daily\2026-08-06\entsoe-incremental-quality-proofs
```

## Remaining gaps and next permitted step

D245 cannot admit D241's current raw 11/8/10-column Unity Catalog tables.
D250 now specifies the interval-start transformation and preserves unknown PIT
semantics, but governed per-series cadence, owner-approved mapping, sign,
lineage, quality, revision and PIT construction remain required. D243's exact
series inventory remains unexecuted and the D247 daily attempt correctly
stopped before SQL because the Warehouse was stopped.

No Databricks action should be taken merely to advance D245. The next safe
offline step is to compose D250's mapping rules with a fail-closed normalized
export adapter contract, still without opening real values. A future real
profile may run only after independently governed normalization, artifact
receipt/hash admission, an already-running Warehouse/cost authorization where
applicable, and a new frozen future holdout.

T057 remains sealed. The monthly solver remains sole level authority, LT
remains independent from CT, OMPEX remains post-freeze benchmark-only and AFRY
descriptive only.
