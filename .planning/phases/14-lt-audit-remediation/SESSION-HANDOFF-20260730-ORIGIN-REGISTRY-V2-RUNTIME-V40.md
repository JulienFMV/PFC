# Session handoff - origin-registry protocol v2 and runtime v40

Date: 2026-07-30

Branch: `fix/lt-audit-remediation`

HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`

Decision: D196

Status: local-quality slice accepted; production strict `NO_GO`.

## Scope and workstation contract

This session closed the demonstrated origin-registry protocol and
standard-user packaging P1s. Work ran only from
`C:\Users\jbattaglia\PFC_LT`. Every shell action checked the exact cwd and Git
top-level. No command ran from `H:`, requested admin/elevation, changed ACLs,
requested Defender/ASR exceptions, built/launched a project `.exe`, used
Playwright, or intentionally wrote mutable state outside canonical repo
`build/`.

The worktree was already intentionally very dirty and was preserved. Nothing
was reset, cleaned, restored, staged or committed. LT/CT separation, monthly
solver level authority and OMPEX benchmark-only status remain unchanged.
`pfc_shaping/ct/*`, Power BI and the protected forward-history parquet were not
touched by this slice.

## What changed

Protocol and installed contract:

- `.planning/phases/14-lt-audit-remediation/CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V2-20260730.json`
- `pfc_shaping/validation/ch_lt_origin_registry_protocol.py`
- `pfc_shaping/cli/audit_ch_lt_origin_registry_protocol.py`
- `scripts/audit_ch_lt_origin_registry_protocol.py`
- `tests/test_ch_lt_origin_registry_protocol.py`
- `tests/test_ch_lt_origin_registry_protocol_installed_v40.py`

Standard-user/runtime chain:

- `scripts/build_repo_local_conda_prefix.py`
- `scripts/build_launcherless_conda_archive_lock.py`
- `scripts/build_launcherless_local_runtime.py`
- `pfc_shaping/pipeline/governed_release_cli_contract.py`
- `tests/test_build_repo_local_conda_prefix_script.py`
- `tests/test_launcherless_conda_archive_lock.py`
- `tests/test_launcherless_local_runtime.py`
- `tests/test_launcherless_runtime_admission.py`
- `pfc_shaping/tools/OPERATIONS.md`

Governance records:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff

Protocol v2 closes the v37 cross-field ordering and request/mask ambiguity.
The prefix builder now rejects links/junctions/reparse points on a new target
and its full parent chain before creation/subprocess. Prefix receipt v3 and
runtime receipt v5 chain the exact caller-held standard-user confinement
receipt and external-guard digest through installed launch-time admission.

## Canonical selected artifacts

- protocol v2: 15,589 bytes; SHA-256
  `6ea896ccdb35414b52237f2bcf1065755c3c10444b308ce905b60f472e68c697`;
  protocol ID
  `0fcf13246c50f6bc79d203437f7c5294495dc233f93873abbe2aeaf3dc282204`;
- Conda prefix:
  `build/conda-runtime-v40-origin-registry-v2-chain`;
- work root:
  `build/conda-prefix-work-v40-origin-registry-v2-chain`;
- repo-local confinement receipt:
  `build/conda-prefix-build-receipt-20260730-v40-origin-registry-v2-chain.json`,
  SHA-256
  `cbafa53aec714bd6f1b1430b7c9e649491b83350e645f83095a38824aeba4451`;
- Python manifest:
  `build/launcherless-python-runtime-manifest-20260730-v40-origin-registry-v2-chain.json`,
  SHA-256
  `e96d7258944abc038f141c71a363b89b619fc6d4fe5b731f1aabff2b92a74f8f`,
  6,285 files, tree
  `083ac24a2c621622d43e4504ecfbbe2183201d260e695f336410bd738d64c050`;
- prefix receipt v3:
  `build/launcherless-conda-prefix-receipt-20260730-v40-origin-registry-v2-chain.json`,
  SHA-256
  `753ce5af1975320c2ab02cd9a73e267710e3cca0b13320d0a701922ded3b3346`,
  prefix ID
  `b08d09f31cf4bbfff5b73227fd81fd892796915bc4afa3d5c6fa4734f0e8f8d9`,
  archive-set ID
  `f3cd775e79648df9a9926a01eb97eadc8e951c5055c778ca9ca92b60bc8068e7`;
- wheels:
  `build/wheel-dist-reg40a/fmv_pfc_lt-0.14.0-py3-none-any.whl` and
  `build/wheel-dist-reg40b/fmv_pfc_lt-0.14.0-py3-none-any.whl`, each 564,873
  bytes / 103 members, byte-identical SHA-256
  `369028f0983b9bb719284b91881a874af7896f4b4cb80d24e3b319a1edf26615`,
  source revision
  `6ee9a8457a8e831f62c110c2d2774a22998a277199d7eb2c7fbc66ea55012284`;
- runtime receipt v5:
  `build/launcherless-runtime-receipt-20260730-v40-origin-registry-v2-chain.json`,
  SHA-256
  `651c8caa548d2e1fdd874f7173397c6f2a05a5d2f3b01ae4a084fbf49468f561`,
  8,507 files / 19 distributions, closure tree
  `914b0c683a0f83319584db3af46bfa6fcb3a202e5aa7bba2d890e2797244d31e`.

The runtime reports `local_quality_authorization=true` and
`production_authorization=false`. Its exact live `sys.path`, in order, is:

1. `C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain\Lib`
2. `C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain\DLLs`
3. `C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain\governed-site-packages`

The checkout root, prefix root and user/AppData site do not enter `sys.path`.

## Exact construction commands and results

All commands below were wrapped by the canonical cwd/Git-root guard. The
ProgramData Anaconda executables were used read-only; mutable paths were
redirected below repo `build/`.

Fresh prefix recipe:

```powershell
python -B -m scripts.build_repo_local_conda_prefix --conda-python C:\ProgramData\anaconda3\python.exe --explicit-spec C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain --work-root C:\Users\jbattaglia\PFC_LT\build\conda-prefix-work-v40-origin-registry-v2-chain --receipt-output C:\Users\jbattaglia\PFC_LT\build\conda-prefix-build-receipt-20260730-v40-origin-registry-v2-chain.json --timeout-seconds 600
```

Result: target exit zero,
`PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION`, `prefix_ready=true`, network,
admin, promotion and production false. Real user guards were identical before
and after:

- `.conda/environments.txt`: 1,048 bytes, mtime_ns
  `1785432628955290100`, SHA-256
  `554db8db49b573851d3d299b962ad8048cfe404231d032791c0ce0d58d2bc92d`;
- `.condarc`: 86 bytes, mtime_ns `1737962786274685100`, SHA-256
  `998e455f1c09e5fd8abbee8560e1e576030cbdc0dd1b5c00bc27ca06d99c6f6e`.

Manifest, prefix audit and assembly target commands recorded in the immutable
workspace-run receipts:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -I -B -m scripts.build_launcherless_python_runtime_manifest --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain --output C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260730-v40-origin-registry-v2-chain.json

C:\ProgramData\anaconda3\python.exe -B -m scripts.build_launcherless_conda_archive_lock audit-prefix --lock C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-archive-lock-20260729-v22.json --expected-lock-sha256 020735fa21744772aedd71a7c99b33775ee27042c9a6c2dd953b15b6b9b720d8 --explicit-spec C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt --expected-explicit-spec-sha256 88266ae90c163470a9bcca09d4ef043bde2c33d5b8446f6536ff2df8cedabd46 --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain --python-runtime-manifest C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260730-v40-origin-registry-v2-chain.json --expected-python-runtime-manifest-sha256 e96d7258944abc038f141c71a363b89b619fc6d4fe5b731f1aabff2b92a74f8f --repo-local-build-receipt C:\Users\jbattaglia\PFC_LT\build\conda-prefix-build-receipt-20260730-v40-origin-registry-v2-chain.json --expected-repo-local-build-receipt-sha256 cbafa53aec714bd6f1b1430b7c9e649491b83350e645f83095a38824aeba4451 --output C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-prefix-receipt-20260730-v40-origin-registry-v2-chain.json

C:\ProgramData\anaconda3\python.exe -B -m scripts.build_launcherless_local_runtime --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v40-origin-registry-v2-chain --project-wheel C:\Users\jbattaglia\PFC_LT\build\wheel-dist-reg40b\fmv_pfc_lt-0.14.0-py3-none-any.whl --publisher-wheelhouse C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-wheelhouse --publisher-dependency-root C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-site-packages --publisher-receipt C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-dependency-closure-receipt.json --additional-wheel-directory C:\Users\jbattaglia\PFC_LT\build\launcherless-wheelhouse-20260727-v1 --python-runtime-manifest C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260730-v40-origin-registry-v2-chain.json --expected-python-runtime-manifest-sha256 e96d7258944abc038f141c71a363b89b619fc6d4fe5b731f1aabff2b92a74f8f --conda-prefix-build-receipt C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-prefix-receipt-20260730-v40-origin-registry-v2-chain.json --expected-conda-prefix-build-receipt-sha256 753ce5af1975320c2ab02cd9a73e267710e3cca0b13320d0a701922ded3b3346 --receipt-output C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v40-origin-registry-v2-chain.json --lock-path C:\Users\jbattaglia\PFC_LT\uv.lock
```

Execution receipt SHA-256 values and terminal results:

- `man40chain`: `7d687bbf026522b360ce79eb1506eb991c08578fb8322e1c72e7978cbfbae682`,
  target exit zero;
- `aud40chain`: `be1e491963e875689de8c91244a566565c147e834ae7b6a698f24a097764dac6`,
  target exit zero;
- `asm40chain`: `554181c2b372b58910d2df5976a611ad46813932eddfde094a82f23af422d759`,
  target exit zero.

## Test matrices

Focused protocol/installed-chain/anti-reparse/runtime matrix:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -I -B -m pytest tests\test_ch_lt_origin_registry_protocol.py tests\test_ch_lt_origin_registry_protocol_installed_v40.py tests\test_build_repo_local_conda_prefix_script.py tests\test_launcherless_conda_archive_lock.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py -q -p no:cacheprovider -m "not slow" --basetemp C:\Users\jbattaglia\PFC_LT\build\wpt-reg40pt\basetemp
```

Result: `80 passed` in 25.312 seconds, zero failure/error/skip; receipt
`build/workspace-local-runs/reg40pt/execution-receipt.json`, SHA-256
`112a1d907223df8a6a85dfc973165855aea4ad4add859a055ab14b979ff8bdcf`.

Historical installed v36 isolated from monolithic contention:

```powershell
C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe -I -B -m pytest tests\test_ch_lt_origin_target_mask_inventory_installed_v36.py -q -p no:cacheprovider -m "not slow" --basetemp C:\Users\jbattaglia\PFC_LT\build\wpt-reg40v36\basetemp
```

Result: `1 passed` in 383.651 seconds; receipt SHA-256
`2bc2deb4f67ec69fb2c90f1430230f5af82ed308cb488e59b37afb06ca9f2dee`.

Complementary science/runtime/packaging/publication/CAS matrix excluded the
two installed tests already covered above. Exact test list is preserved in
`build/workspace-local-runs/reg40mx2/execution-receipt.json`. Result: `825
passed, 18 skipped, 2 deselected` in 312.899 seconds, zero failure/error;
receipt SHA-256
`90db23a123db0a2dcd089ec913cc8db2ceec5a7f34b54d4a75dccb9855a167f2`.

The non-overlapping union is `827 passed, 18 skipped, 2 deselected`, zero
failure/error. This is a partition, not an arithmetic sum of every focused
run. All three terminal test receipts bind source tree SHA-256
`20ad68d37bbad30c7b4f6c3487e74be22505b8ab6da5a96da08c6d1db4c66c22`.

## Negative evidence retained

- v37: terminal roasts Security `0/2/3`, IT/Operations `0/1/4`, Quant/Data
  `0/1/1`; missing temporal/request/mask invariants and demonstrated user
  Conda-registry timestamp mutation. Superseded.
- v38: protocol semantic P1s and user-registry mutation were closed, but IT
  found no cryptographic chain from confinement receipt into runtime evidence
  and no reparse-parent rejection. Superseded.
- v39 builder/runtime receipt passed, but installed v39 rejected runtime
  schema v5 because its packaged validator still expected v4. Superseded.
- `reg40mx`: `826 passed, 18 skipped, 2 deselected, 1 failed`; historical
  installed v36 exceeded its 600-second child timeout under monolithic
  contention. Receipt SHA-256
  `aad3c65adbe05659f3955fa475507467f0a9ae2f69e0de6f021afd261bc74310`.
  Never cite it as PASS.
- Earlier audit attempts `aud39chain`, `aud39chain2` and `aud39chain3` failed
  respectively for missing `zstandard`, source-module isolation and missing
  `zstandard`; `aud39chain4` established the safe read-only ProgramData
  Anaconda route. No dependency was installed outside the workspace.

## Independent terminal re-roasts

- Security/Governance: P0/P1/P2 `0/0/2`. Remaining P2: loaded-byte
  pre-import TOCTOU; no global wall timeout/Job Object/process-tree kill/stale
  reconciliation.
- IT/Operations: P0/P1/P2 `0/0/3`, local-quality GO only. Remaining P2:
  supervision/resource telemetry; cost/contention and absent robust CI SLO;
  non-atomic/non-resumable prefix plus missing Windows power-loss,
  active-runtime CAS/rollback and recovery-observability drills.
- Quant/Data: P0/P1/P2 `0/0/0`. Protocol hash, temporal/request/mask
  invariants and outcome blindness verified. No T057 outcome was read.

No P0/P1 remains on this local slice. Production remains strict `NO_GO`.

## Open production blockers and next work

The external registry service and lease/CAS semantics are not implemented.
FMV has not approved/frozen an exact UTC cadence. Official event evidence,
trusted time/signatures, independent identities/keyrings/ACLs,
builder-inaccessible CAS/WORM/fresh HEAD, PIT commitments, future truth and
independent standard-user CI remain absent. Countable prospective origins are
zero and eleven external evidence slots are missing.

Next work, without production promotion:

1. close process supervision/observability and atomic/resumable runtime
   lifecycle debt in a CI-capable design;
2. commission the independent external origin registry and exact FMV UTC
   cadence;
3. capture genuinely fresh prospective CH/EEX evidence point-in-time;
4. register new outcome-blind rolling origins and a new future holdout; T057
   remains permanently ineligible for confirmatory reuse;
5. build a new auditable CH candidate, preserve monthly solver level authority
   and prove final delivered BASE/PEAK/OFFPEAK repricing, shaping and
   probabilistic/scenario quality before any promotion decision.

## Final perimeter audit

The isolated pytest runtime does not package Ruff; its attempted
`python -I -B -m ruff` exited immediately with `No module named ruff` and made
no project change. The repo-contained executable was then used read-only with
cache disabled:

```powershell
C:\Users\jbattaglia\PFC_LT\.venv\Scripts\ruff.exe check --no-cache scripts\build_repo_local_conda_prefix.py scripts\build_launcherless_conda_archive_lock.py scripts\build_launcherless_local_runtime.py pfc_shaping\pipeline\governed_release_cli_contract.py pfc_shaping\validation\ch_lt_origin_registry_protocol.py pfc_shaping\cli\audit_ch_lt_origin_registry_protocol.py scripts\audit_ch_lt_origin_registry_protocol.py tests\test_ch_lt_origin_registry_protocol.py tests\test_ch_lt_origin_registry_protocol_installed_v40.py tests\test_build_repo_local_conda_prefix_script.py tests\test_launcherless_conda_archive_lock.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py
```

Result: `All checks passed!`.

Final guarded checks:

- `git diff --check`: exit zero; line-ending warnings only;
- staged paths: zero;
- `pfc_shaping/ct` status paths: zero;
- `powerbi` status paths: zero;
- protected `data/eex_forwards_history.parquet` SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- user Conda guard hashes remain exactly
  `554db8db49b573851d3d299b962ad8048cfe404231d032791c0ce0d58d2bc92d`
  and
  `998e455f1c09e5fd8abbee8560e1e576030cbdc0dd1b5c00bc27ca06d99c6f6e`;
- selected runtime receipt is schema v5/PASS, local-quality true,
  production false and exposes exactly the three governed `sys.path` roots;
- workspace Python/Conda task-process count: zero;
- this handoff and the RFC amendment have no trailing whitespace and end with
  a newline.

No production promotion was attempted or authorized.
