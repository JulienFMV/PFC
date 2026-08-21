# Session handoff - 2026-07-28 - Conda archive-payload runtime v17

## Status

- Canonical repo only: `C:\Users\jbattaglia\PFC_LT`.
- Branch: `fix/lt-audit-remediation`.
- Session-start HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree intentionally very dirty; no reset, clean, restore, stage or commit.
- `data/eex_forwards_history.parquet` was not touched. Its observed SHA-256
  remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- No CT, Power BI, old `H:` repo, Playwright, project executable, Defender
  exception, admin action, production publication or production promotion.
- Local runtime v17: `PASS` for local execution/quality inspection only.
- Production: strict `NO_GO`.

## Why v14 was not closure

The independent Security roast invalidated the byte-provenance claim in D169:
v14 trusted target `conda-meta/files`, did not derive installed bytes from the
retained archive payloads, accepted arbitrary receipt IDs in unit fixtures and
did not fully replay the caller-held lock at launch. D169 is retained as the
historical v14 decision but is superseded by D170 for current evidence.

The correction introduced:

- canonical archive-lock JSONL schema v2;
- independent parsing of `.conda` and `.tar.bz2` archives;
- exact `info/index.json`, `info/paths.json` and `info/link.json` validation;
- archive payload tree/file counts, prefix-replacement count and noarch type;
- byte comparison between each archive payload and the replayed prefix;
- explicit segregation of 406 generated noarch `pip` files as non-runtime;
- full lock/ID/spec/history/prefix replay in the runtime builder;
- launch-time rehash of every retained archive, recomputation of archive-set,
  lock and prefix-receipt IDs, and exact target `conda-meta` binding;
- canonical path ancestry checks before reads/writes;
- runtime receipt schema v4 and prefix receipt schema v2.

On Windows, the auditor implements the observed Conda semantics: text prefix
replacement uses forward slashes; non-pyzzer binary payloads remain unchanged.
Pyzzer binary prefix replacement is deliberately unsupported and fails closed.
Metadata-only packages with an explicit empty `paths` list are valid. Patched
target dependency strings are recorded and bound to the installed
`conda-meta`; archive/source dependency strings remain bound to the lock.
Neither string set is used as an installed-byte authority.

## Exact source/test delta for this closure

- `scripts/build_launcherless_conda_archive_lock.py`;
- `scripts/build_launcherless_local_runtime.py`;
- `pfc_shaping/pipeline/governed_release_cli_contract.py`;
- `tests/test_launcherless_conda_archive_lock.py`;
- `tests/test_launcherless_local_runtime.py`;
- `tests/test_launcherless_runtime_admission.py`.

No scientific LT, monthly solver, CT, OMPEX, holdout or data code was changed
by this v14-to-v17 packaging correction.

## Conda archive lock v2

The lock was built and independently verified with the existing Anaconda
Python, which contains the zstandard reader required for `.conda` payloads:

```powershell
C:\ProgramData\anaconda3\python.exe -B -m scripts.build_launcherless_conda_archive_lock build `
  --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v12 `
  --package-cache-root C:\Users\jbattaglia\.conda\pkgs `
  --output C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-archive-lock-20260728-v15.json
```

Evidence:

- lock path:
  `build/launcherless-conda-archive-lock-20260728-v15.json`;
- lock SHA-256:
  `346c6edcce71dea86816ec6938a1d6a87872a3cd30a01984c8577e7800c33fdb`;
- archive lock ID:
  `7fd90fd7a1a2f672da1890cb308870f5c7d1df5e38dbafe91f6da27ca486a813`;
- archive-set ID:
  `f3cd775e79648df9a9926a01eb97eadc8e951c5055c778ca9ca92b60bc8068e7`;
- 19 packages / 38,648,096 retained archive bytes;
- build: exit 0 in 92.1 s;
- independent verify: exit 0 in 89.3 s;
- explicit spec:
  `build/launcherless-conda-explicit-20260728-v15.txt`, 20 lines,
  2,035 bytes, SHA-256
  `c1da60c9c3474453ebc21580cecc15c556c9bcd82e9113c73bdaa62ec883e4eb`.

The archive paths still point to `C:\Users\jbattaglia\.conda\pkgs`; this is a
local retained cache, not a durable IT CAS/WORM. It remains an operations P1.

## Prefix v17 and archive-to-prefix proof

The final prefix was created in a new namespace with no solver or network:

```powershell
$env:CONDA_OVERRIDE_CUDA='0'
$env:CONDARC='C:\Users\jbattaglia\PFC_LT\build\condarc-runtime-v6.yml'
C:\ProgramData\anaconda3\Scripts\conda.exe create --offline --copy `
  --prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v17-archive-payload-audited-base `
  --file C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260728-v15.txt `
  --yes --json
```

Result: exit 0 in 161.9 s. Conda could not update its optional global
`C:\Users\jbattaglia\.conda\environments.txt` registry because it was not
writable; the repo-local prefix completed and no elevation was requested.

The Python manifest was captured before first target execution:

- `build/launcherless-python-runtime-manifest-20260728-v17-base.json`;
- SHA-256
  `48f47b44882c47eb7aa118ff6bcf4ee8ac45d715d794d82fc4b30cdb7617282e`;
- 6,285 files;
- tree SHA-256
  `75af523ee789bfd0d8bb180ab8aea5470bc7e7cec3fd2acd143508d4d9cebf6b`.

The archive-to-prefix audit passed in 335.6 s:

- prefix receipt:
  `build/launcherless-conda-prefix-receipt-20260728-v17.json`;
- receipt SHA-256:
  `3ad67d66d277a6783a2b1731fed93a847ff4f346283670a0a5c12047d6b91834`;
- prefix receipt ID:
  `145e28c66ce49e4127d130e78e7354483d74c6c8bf9dbb9d99847608ba498b2e`;
- 5,859 archive-verified installed files;
- 406 explicitly classified generated non-runtime files;
- 19 target package records and 6,285 total manifest files;
- `production_authorization=false`.

## Reproducible wheels O/P

Both wheels were built with `SOURCE_DATE_EPOCH=1783987200`, separate build,
dist and TEMP/TMP directories below `build/`, then audited:

- 464,417 bytes / 84 members;
- byte-identical SHA-256
  `2eb23e57c45bedb7c65ca44fbe99df3bda41bc9a78b1698b1c90bd5f759c72e4`;
- embedded source revision
  `d41e6a3524d673c84450d4e4327588994d0bf2f74b769854f8c9a256d5e656c2`;
- both wheel audits `PASS`, `promotion_eligible=false`.

The retained `setup.py` deprecation warning is still an IT packaging concern.

## Runtime v17 assembly and installed admission

The runtime builder used only repo-local inputs except for read-only retained
Conda archives already bound by the lock. Publisher inputs were
`build/runtime-inputs-20260728-repolocal-v1`; TEMP/TMP was
`build/runtime-temp-v17`.

Result:

- assembly exit 0 in 1,319.0 s;
- runtime receipt:
  `build/launcherless-runtime-receipt-20260728-v17.json`;
- runtime receipt SHA-256:
  `dc944cdd9d13f96ee7dfa9d20010d3905670a3547931542eb914c1ce12300e19`;
- schema `fmv_lt_launcherless_local_runtime.v4`;
- closure: 19 distributions / 8,488 files, tree SHA-256
  `e733f26c6a1120f6b09e284fa7d74cc30764bc3fe3029741fef812c20453d30a`;
- project source revision:
  `d41e6a3524d673c84450d4e4327588994d0bf2f74b769854f8c9a256d5e656c2`;
- `local_quality_authorization=true`;
- `production_authorization=false`.

Installed admission command:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH = `
  'C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260728-v17.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256 = `
  'dc944cdd9d13f96ee7dfa9d20010d3905670a3547931542eb914c1ce12300e19'
& 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v17-archive-payload-audited-base\python.exe' `
  -I -B -m pfc_shaping.cli.governed_release --version
```

Result: exit 0 in 140.0 s, version `0.14.0`, exact embedded source revision.
Admission occurs before argparse exposes `--version`.

The live isolated probe reports exactly four `sys.path` entries:

1. `...\python311.zip`;
2. `...\Lib`;
3. `...\DLLs`;
4. `...\governed-site-packages`.

Prefix root count is zero, checkout count is zero, user/system site count is
zero and governed application-root count is one.

## Fail-closed attempts retained

- v15 built successfully but installed admission rejected
  `Conda archive package record is invalid`; the lightweight validator omitted
  valid `noarch: generic` used by `tzdata`.
- v16 built successfully but installed admission rejected
  `Conda target metadata record is invalid`; the validator incorrectly
  required target patched dependency strings to equal the archive/source
  record. The source correction was tested directly against the real v16
  receipt before wheels O/P were built:
  `SOURCE_CONTRACT_REAL_V16_PASS`.
- Neither v15 nor v16 was repaired, relabelled or reused. Both namespaces and
  receipts are retained as negative evidence.

## Verification matrices

- focused archive/runtime/admission: `43 passed`;
- final runtime/packaging: `147 passed, 12 skipped, 2 deselected`;
- external publication/snapshot/anchors split after one non-conclusive
  five-minute combined timeout: `66 passed` and `59 passed`;
- atomic promotion: `116 passed, 2 skipped`;
- candidate bundle/evidence/assembler: `65 passed`;
- governed release/script/quality/monthly manifests: `267 passed`;
- targeted Ruff: `All checks passed!`;
- `git diff --check`: exit 0, informational LF/CRLF warnings only;
- staged files: zero.

No production promotion was exercised. The one combined publication timeout
is not counted as evidence; only the two terminal split runs are counted.

## Independent read-only re-roasts

Pending final Security, IT/Operations and Quant/Data addenda for v17. Their
results must be appended here before session closure.

## Open gates

- Same-user target code is still imported before its self-admission. This is
  an external trust-boundary P1 requiring an independently admitted bootstrap,
  read-only service identity or external execute allowlist.
- The Conda cache and Conda executable are not in a signed durable CAS/WORM;
  two clean runner replays are not yet demonstrated.
- Conda writes directly into the final prefix namespace; atomic prefix staging,
  kill recovery and job-object supervision remain absent.
- Build/admission execution sidecars, Windows standard-user CI under the FMV
  ASR policy, signed SBOM/provenance, structured logs, SLOs, active-runtime CAS
  and rollback drills remain required.
- V11 documentary revocation and attempt5 incident closure remain IT work.
- Fresh prospective point-in-time EEX/CH data, exact final EEX repricing,
  preregistered rolling-origin/T057/holdout evidence and probabilistic/scenario
  gates remain required before a new CH candidate.

Monthly solver authority, LT/CT separation and OMPEX benchmark-only status are
unchanged. Production remains strict `NO_GO`.
