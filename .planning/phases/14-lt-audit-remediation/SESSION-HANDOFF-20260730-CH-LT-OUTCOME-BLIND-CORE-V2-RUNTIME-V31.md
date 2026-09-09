# Session handoff 2026-07-30 — CH LT outcome-blind core v2 and runtime v31

## Outcome

This slice closes the local standard-user packaging boundary and authors an
outcome-blind CH LT successor policy core. It does **not** admit a scientific
candidate and grants no production or promotion authority. Production remains
strict `NO_GO`.

All commands ran from `C:\Users\jbattaglia\PFC_LT` after exact cwd and Git-root
checks. Mutable state stayed below `build/`. No administrator right, elevation,
Defender/ASR exception, system installation, project `.exe`, Playwright, `H:`
path, CT file or Power BI file was used. The protected pre-existing dirty file
`data/eex_forwards_history.parquet` was not touched or staged; its SHA-256 is
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Permission-prompt closure

The obsolete v9 command that referenced a publisher wheelhouse below
`AppData` is revoked. `scripts/run_workspace_local.py`, the runtime builder and
their tests reject external mutable roots. Every mutable Conda, pip, pytest,
temporary and runtime-staging path is now repo-local. Read-only invocation of
an existing user or system interpreter is allowed; creation or mutation of an
environment outside the workspace is not.

The candidate audit runner allowlist accepts only the local script with exact
`python -B`, the exact v2 core path/hash and no additional arguments. The
packaged CLI is intentionally not runner-allowlisted because the runner
scrubs the runtime receipt environment; it is covered directly by the sealed
installed-runtime foreign-cwd test.

## Outcome-blind scientific core

- T057 tombstone:
  `.planning/phases/14-lt-audit-remediation/T057-OUTCOME-BLIND-TOMBSTONE-20260730.json`,
  SHA-256 `e7b7524375d431004c2dfe1aff9d7fea10a9bf7742c4044f53a9cda3cdb594ea`,
  ID `9d2afbe3593c5525bf67f16a0d4482fa24304d547f3649b84b74c7549daa3380`.
- Core v2:
  `.planning/phases/14-lt-audit-remediation/CH-LT-SUCCESSOR-CANDIDATE-CORE-V2-20260730.json`,
  SHA-256 `a2ec1d758043b7ed4bf111e99ef4f87d814ad292b7b3c115c36de30d1da4011e`,
  ID `7042b909a1ca0aeb63f26f76fccdea7256b526651685e4e1db713ca469cc3065`.
- Corrected readiness update:
  `.planning/phases/14-lt-audit-remediation/CH-LT-PIT-SUCCESSOR-READINESS-UPDATE-V2-20260730.json`,
  SHA-256 `93dd94ad7f2fc2734a81a73ab80fc5f373c9b9674e0895857f2ab8019c037497`,
  ID `8825f83e8a190198d909c85eda8142b868e937969eb29c07702c0b51493647f4`.

The v1 assessment is contaminated because its validator opened the
outcome-bearing T057 registry. V1 execution/admission is permanently
forbidden. Its JSON is retained only as a policy scaffold, and legacy T057
confirmation cannot be reused. Core v2 reads the outcome-blind tombstone only;
a new independent future holdout is required.

Core v2 corrects the origin-overlap rule to 35 overlapping lags and a minimum
block length of 36 with no cap; freezes exact selection, tie, Holm and
gatekeeping rules; separates native hourly/pre-transition and native
15-minute/post-transition scoring graphs; distinguishes positive- and
zero-margin Monte Carlo error gates; and makes conditioning thresholds and
future episode counts conditional on pre-frozen direct-CH error/power designs.
The monthly solver remains level authority and OMPEX remains benchmark-only.

The core is only `LOCAL_HASH_CLOSED_RECEIPT_FREE_CORE_NOT_EXTERNALLY_FROZEN`.
All ten evidence slots are `MISSING`: governed CH/EEX PIT evidence, native CH
truth, fold inventories, frozen candidates/baselines/gates, dependence/power,
probabilistic/Monte Carlo design, FMV economic materiality, qualified CPU
oracle and CPU/GPU parity, hash-bound independent reviews, and an external
admission envelope/one-shot ledger.

## Reproducible package and runtime v31

- Wheels C and D:
  `build/wheel-dist-core30v2-{c,d}/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
  96 members, 535,973 bytes, byte-identical SHA-256
  `3e3f4e24c23f36ac0f4bd43a77d46d23ed23e846c0f329ba922c8534cb89f00b`.
- Runtime prefix: `build/conda-runtime-v31-core-base`.
- Python manifest:
  `build/launcherless-python-runtime-manifest-20260730-v31-base.json`,
  SHA-256 `dbab8c99685967907957117b4ad14978548391ce0cc5df5f6e52b5b256eb8638`.
- Prefix receipt:
  `build/launcherless-conda-prefix-receipt-20260730-v31.json`,
  SHA-256 `704853b90ba051bf3551b131944870054d27a3ddc0753d0489123382766adf97`,
  ID `dccc1600d18b6918bdea8aa14efa0f3f12216bd60b1ee7364dbf83a9b0f9d0b0`.
- Runtime receipt:
  `build/launcherless-runtime-receipt-20260730-v31.json`,
  SHA-256 `4cc404d6fdceee4f1b41384ce1366cb9177b318c8074bbf740515519130e5323`.
- Closure: 8,500 files, 19 distributions, tree SHA-256
  `d3d070a0ba9d9e5da99b65ef218960bd27d2eca007cfb9e2c83b6f5d1670b4a2`.
- Exact `sys.path`: runtime `Lib`, runtime `DLLs`, and
  `governed-site-packages`; no checkout or additional application root.

The runtime authorizes local-quality execution only. It grants no scientific,
promotion or production authority.

## Commands, results and receipts

All target commands below were invoked through
`python -B -m scripts.run_workspace_local --run-id <id> -- <target>`; the
execution receipt preserves the complete exact command and repo-local mutable
paths.

| Run ID | Exact target/action | Result | Receipt SHA-256 |
|---|---|---:|---|
| `man31core` | `python -B -m scripts.build_launcherless_python_runtime_manifest --runtime-prefix ...v31-core-base --output ...v31-base.json` | exit 0 | `78911ae0c9a31f3edbe0c8c97cadb4a0c989263187b29718a0e065aa2f1d3153` |
| `aud31core` | Anaconda `python -B -m scripts.build_launcherless_conda_archive_lock audit-prefix` with exact v22 lock/spec and v31 manifest | exit 0 | `6dd457490c9d9f052043d4a1cf50a36a110ccfafc411036b5bcad5c1eabe4f13` |
| `asm31core` | Anaconda `python -B -m scripts.build_launcherless_local_runtime` with only repo-local wheelhouses/receipts | exit 0 | `2af4f3473b8fa334fcfd4c2fdf30193d5258783357099f3c15d6b9ac5b29dbf0` |
| `pkg31smk` | `python -B -m pytest tests\\test_ch_lt_successor_candidate_core_installed_v31.py -q -p no:cacheprovider` | 1 passed in 20.88 s | `b7d5855bd36fd53cd94335c0275c2631ae813dde2885ec49e12cbebf9be89893` |
| `sec30fix2` | focused core/readiness/runner/package tests | 156 passed | `4d0da66afde7b2261f198cdec92776f13f0291c0a83259dada278da572313a62` |
| `v2pro2` | prospective/core/package matrix | 415 passed, 2 deselected | `d1eeba37446bf04fe606319875ab54644da8e9461465fe512560cd94b7aa9f91` |
| `v2run2` | runtime/packaging matrix including installed v31 smoke | 305 passed, 12 skipped, 2 deselected | `dbef0c878f91798eb856b372f3cf79ec78a3cdf6656c865eb3cdcced9f5847b0` |
| `v2cas2` | external publication/CAS matrix | 200 passed | `ad7a807418101f84162ddadeea6a4329dd26b7b74bc6b50ca5d9b893af4a75ad` |
| `v2cand2` | candidate/atomic-publication matrix | 181 passed, 2 skipped | `22c5b4dc1c2546bffbb3e1f0a6f12314a5aa25543e33a1320d64b5e716ceae75` |
| `v2ruff3` | Ruff over the changed core/runner/package surface | pass | `418199140b62b68aabcf487fa5b8e7010fced03215556e712948cefed9fc2486` |

One known timezone warning remains non-failing. Historical v30 smoke failures
are retained as negative evidence and are superseded by v31, not erased.

## Independent read-only re-roasts

- Security: P0/P1/P2 `0/0/0` after the readiness supersession, docstring and
  runner-route corrections.
- IT/Operations: P0/P1/P2 `0/0/0`; explicitly verified `aud31core`,
  `asm31core`, `pkg31smk`, exact `sys.path`, reproducible wheels and current
  matrices.
- Quant/Data: P0/P1/P2 `0/0/0` for the authored policy core; confirms the ten
  missing evidence slots, monthly solver authority, OMPEX benchmark-only and
  strict `NO_GO`.

These reviews cover this local slice only. They are not the future hash-bound
admission reviews required by the core evidence slot.

## Main files changed in this slice

- `pfc_shaping/validation/ch_lt_successor_candidate_core.py`
- `pfc_shaping/validation/ch_lt_successor_candidate_core_v2.py`
- `pfc_shaping/validation/ch_lt_successor_readiness.py`
- `pfc_shaping/validation/ch_lt_preregistration_supersession.py`
- `pfc_shaping/cli/audit_ch_lt_successor_candidate_core.py`
- `scripts/audit_ch_lt_successor_candidate_core.py`
- `scripts/run_workspace_local.py`
- `pfc_shaping/package_contract.py`
- `tests/test_ch_lt_successor_candidate_core.py`
- `tests/test_ch_lt_successor_candidate_core_v2.py`
- `tests/test_ch_lt_successor_candidate_core_installed_v31.py`
- `tests/test_ch_lt_successor_readiness.py`
- `tests/test_ch_lt_preregistration_supersession.py`
- `tests/test_run_workspace_local_script.py`
- `tests/test_lt_package_contract.py`
- the three planning artifacts identified above.

The broader worktree was already intentionally very dirty. No reset, clean,
restore, staging or commit was performed.

## Residual blockers and next safe work

Production remains strict `NO_GO`. Remaining blockers include external trusted
time and signatures; builder-inaccessible CAS/WORM and fresh-HEAD admission;
authenticated fresh prospective CH and exact EEX product/session/settlement
evidence; direct-CH power and rolling-origin evidence; a new independent
holdout; final exact EEX repricing; SBOM/provenance; Windows CI under ASR;
service identity; observability; rollback; and the general production
import-before-self-admission boundary. The setup.py build path also carries a
future PEP 517 migration debt.

Next, acquire and admit fresh point-in-time prospective data, close the direct
CH power/episode design, validate the candidate in rolling-origin, then freeze
a new independent holdout and external candidate envelope. Do not reopen T057
and do not promote production.
