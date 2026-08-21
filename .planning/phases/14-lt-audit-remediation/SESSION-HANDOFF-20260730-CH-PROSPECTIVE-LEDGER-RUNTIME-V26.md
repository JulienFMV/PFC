# Session handoff - CH prospective ledger and launcherless runtime v26

Date: 2026-07-30

Branch: `fix/lt-audit-remediation`

Observed HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`

Workspace: `C:\Users\jbattaglia\PFC_LT`

Production: strict `NO_GO`

## Outcome

The workstation permission problem is closed operationally: ordinary work no
longer requires user approvals. All mutable Conda prefixes, wheel builds,
caches, TEMP/TMP, test basetemps, runtime staging and evidence remain below
repo `build/`. The preinstalled ProgramData Anaconda/Conda executables were
used read-only. No admin right, elevation, Defender/ASR exception, AppData
write, project executable, Playwright or legacy `H:` checkout was used.

Two real local CH hourly captures now form one exact, contiguous, replayed
ledger. This is structural local evidence only. It is not trusted PIT,
scientific truth, official product identity, rolling-origin evidence or a
model input. T057 and future holdout outcomes were not consumed.

## Protected workspace state

Before every shell action, both cwd and Git top-level were checked to be
exactly `C:\Users\jbattaglia\PFC_LT`. The worktree remains intentionally very
dirty. Nothing was reset, cleaned, restored, staged, committed or promoted.
No CT or Power BI path was changed. `data/eex_forwards_history.parquet` was not
written or staged.

## Prospective source evidence

The canonical EEX workbook remained unchanged at SHA-256
`b3e213f1512890ea72af1cee03015fcc942ba9946ec89c7a5d4745603eb5eb0f`;
no fake new vintage was created.

The new CH capture covers `2026-07-28T22:00:00Z` through
`2026-07-29T22:00:00Z`, with 24 hourly provider observations and 96 stepwise
15-minute proxy rows. Provider body SHA-256 is
`9b458f1af39aac42611f9362a6976ad82b4eadaf6730f1d910151174bdb02a1d`.
The networkless Builder manifest SHA-256 is
`262d54ff3df4ce026876ac6020bdfd5492ae3b1176beca19d8995e300210ab83`.
The isolated verifier audit SHA-256 is
`18eb6272fbcc359ac57fbd1ff34d4606f508382b1865bf79acff5c186189db99`
and its runtime receipt SHA-256 is
`5aae9ef9747c646cf4f1661b30737895f0a3f86a491cd2a5e1b6a893582b18fe`.

Together with the prior 720-hour capture, the selected ledger covers 744
strictly contiguous native-hourly observations and 2,976 stepwise proxy rows.
The request SHA-256 is
`7b855b50c1dc3f0ef1c1a50c04cbabc75b63b4fac38c645c9129683f0c8fa4b4`.

Security findings closed in source include exact provider replay instead of
self-consistent metadata, single retained body bytes across verification,
honest receipt-claimed verifier hashes, physical-alias and ADS rejection, and
generic positive authority-claim rejection including signature, verified,
official, authoritative, authority and exact `valid` vocabulary.

## Selected ledger and supersession

Machine-readable selection:

`.planning/phases/14-lt-audit-remediation/CH-LT-PROSPECTIVE-CAPTURE-LEDGER-SELECTION-20260730.json`

Only v5 is selected for local structural diagnostics:

- ledger path:
  `build/prospective-ledgers/ch-hourly-local-ledger-20260730-v5.json`;
- SHA-256
  `089aaa82d1025fd550cd9cdceba6a20cfb1aef1871a57f4cdca2c2e23470bdb6`;
- ledger ID
  `b1698c22e28df075aacfecd04c752e7c028d9283229b64ca0c34b4555352264d`;
- durable execution receipt SHA-256
  `439b3a155c8912f8b96a088aa720e1eb0180d84695f65b25b7de06d2ec6cfd37`.

V1/v2 have historical hourly-truth overclaims, v3 predates replay and claim
hardening, and v4 lacks the v26-bound execution receipt. All are retained and
explicitly non-selectable. V4/v5 ledger content happens to be byte-identical;
selection is based on the separately bound v26 construction evidence.

## Reproducible wheels and runtime v26

Wheel commands used separate repo-local TEMP/TMP, pip cache, bytecode cache,
build and dist directories and `SOURCE_DATE_EPOCH=1783987200`:

```powershell
python -B setup.py build --build-base build\wheel-build-ag bdist_wheel --dist-dir build\wheel-dist-ag
python -B setup.py build --build-base build\wheel-build-ah bdist_wheel --dist-dir build\wheel-dist-ah
```

Both 91-member wheels are byte-identical, contract PASS, with SHA-256
`7f4801114f6e247505110030fa02af21fbd4978dccabdc8767a4454e2ea6d4b3`
and source revision
`98f51af0f735db7faa7e5f4156686e9e29fba2d8daaeb15ce72440f93b21ea48`.

Conda replayed the existing 19-package repo-local archive lock with
`--offline --copy --file`; no solver or network was used. Canonical v26
evidence:

- prefix `build/conda-runtime-v26-ledger-base`;
- Python manifest SHA-256
  `3b350dffb39c736983bd2db6e038f47afcb1810af4902b0c05baee8db2b514e4`;
- prefix receipt SHA-256
  `32eca4fc967032bc8ec1532d3308a6f4b1b56d1cbeb04e1e3875bd598092789f`;
- runtime receipt SHA-256
  `0ac27e95ec28835e3a34e17d62f8903655874737d580ab01656502d5991fa87f`;
- closure: 8,495 files / 19 distributions, tree
  `51bcb269f99739819ab32845091d8cade15036217f222b8a1595f9dd14d50e03`;
- exact `sys.path`, in order: runtime `Lib`, `DLLs`,
  `governed-site-packages`;
- explicit installed origin for
  `pfc_shaping.cli.build_ch_lt_prospective_capture_ledger` below the governed
  site-packages root;
- local quality true, production authorization false.

The installed v26 CLI produced v5 and its execution receipt, then an exact
retry exited zero with both files byte-identical. The durable receipt binds
request, ledger, project source revision, cwd, exact `sys.path` and runtime
receipt. A divergent existing target is rejected by tests.

## Terminal tests

- focused ledger and runtime-builder matrix: `38 passed`;
- final runtime/packaging matrix: `256 passed, 12 skipped, 2 deselected`;
- final publication/external-CAS matrix:
  `195 passed, 12 skipped, 1 deselected`;
- Ruff PASS;
- `git diff --check` exit 0, with only pre-existing LF-to-CRLF checkout
  warnings.

All tests used fresh repo-local basetemps and cache paths. No T057 outcome,
future truth, CT, Power BI or protected heavy data was read by this slice.

## Independent reviews and remaining blockers

Terminal read-only re-roasts all report P0/P1/P2 `0/0/0` for the selected
local slice:

- Security verified the hardened claim scanner, exact replay, source/wheel/
  installed equality, three-root `sys.path`, v26 seal, execution receipt and
  v1-v5 selection;
- IT/Operations verified standard-user confinement, runbook, non-ambiguous
  supersession, installed module origin, durable receipt, exact retry and
  terminal matrices;
- Quant/Data verified 744 hourly observations, 2,976 non-independent stepwise
  proxy rows, zero rolling-origin authority, no T057 access, solver-level
  authority and OMPEX benchmark-only invariants.

Only local continuity, native-hourly cadence and exact local replay are
admissible claims.

Production remains blocked by trusted capture time and revision lineage,
provider-authenticated product/auction/session/settlement semantics,
independent signatures, builder-inaccessible CAS/WORM/fresh monotone HEAD,
multi-season rolling origins and dependence-aware power, sealed future T057,
service identity, Windows CI/ASR qualification, SBOM/license/vulnerability
scans, structured supervision/observability, rollback and disaster recovery.

Monthly solver remains the sole level authority, OMPEX remains benchmark-only,
and no candidate or production promotion occurred.
