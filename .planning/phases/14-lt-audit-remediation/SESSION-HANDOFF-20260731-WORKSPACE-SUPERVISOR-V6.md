# Session handoff - workspace supervisor v6 - 2026-07-31

## Executive status

The standard-user Windows workspace supervisor v6 packaging closure is
accepted for local quality only. Current code and evidence close every
demonstrated P0/P1 finding from the runtime-v40 supervision debt. Production,
scientific evaluation, publication, promotion and runtime authorities remain
false and production remains strict `NO_GO`.

This slice did not build or launch a project executable, use Playwright, access
the network, request admin/ACL/Defender rights, mutate AppData/ProgramData, read
a T057 outcome, touch CT or Power BI, modify the protected EEX history parquet,
create a commit or promote any artifact.

Launcherless runtime v40 remains the selected local-quality LT runtime. The
workspace supervisor is a laptop-only Python harness around tests/build/audit
modules; it is not packaged in the governed LT wheel and is not a production
runtime.

## Governing decision and documents

- D197 in `DECISION-LOG.md` selects supervisor v6 for local packaging quality.
- `LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md` contains the
  2026-07-31 supervisor-v6 amendment.
- `pfc_shaping/tools/OPERATIONS.md` documents the exact standard-user command,
  finite admission budget, receipts, Job containment, telemetry and recovery.
- D196 and
  `SESSION-HANDOFF-20260730-ORIGIN-REGISTRY-V2-RUNTIME-V40.md` continue to
  govern protocol v2 and launcherless runtime v40.

## Files changed in this slice

- `scripts/run_workspace_local.py`
- `tests/test_run_workspace_local_script.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260731-WORKSPACE-SUPERVISOR-V6.md`
- `.planning/HANDOFF.md`

The worktree was already intentionally very dirty. Existing unrelated changes
were preserved. Nothing was reset, cleaned, restored, staged or committed.

## Current code identities

- runner SHA-256:
  `0c570e1865f3684572574528834c69540dd0e9dd64861a21acd3338413791912`;
- runner tests SHA-256:
  `5b1bd4b444b15e61ca0a73188ffce884b5ea695c2c3ac073144649bc6d36ceee`;
- reviewed `OPERATIONS.md` SHA-256:
  `71fb91c614ad936c162d1fbb0acbb39593795e4a082cadf4166664009577ddba`;
- source-tree schema `pfc_lt_workspace_source_tree.v1`, 536 files,
  9,523,936 bytes, tree SHA-256
  `8c259783d414a17636de583e3a6341c94cff1d8a3df87dcf4d7eece044ab8c7a`.

## Implementation closure

### Complete-cycle supervision

- Execution receipt schema is `pfc_lt_workspace_local_execution.v6`.
- Supervisor receipt schema is `pfc_lt_workspace_local_supervisor.v1`.
- Reconciliation schema is `pfc_lt_workspace_local_reconciliation.v1`.
- `--wall-timeout-seconds` is finite, defaults to 1,800 and is capped at
  86,400 seconds.
- The parent budget starts before source/capability/bootstrap preflight and
  covers worker preflight, target execution and post-byte verification.
- A postflight or first-terminal-fsync deadline crossing is exit 124 and no
  authority. Cleanup/fsync may overshoot; the receipt records the overshoot and
  explicitly states `strict_return_bound=false`.

### Capability and process containment

- The parent captures exact worker bytes under
  `build/workspace-local-supervisors/<run-id>`.
- The one-shot capability binds run ID, worker-source hash, exact worker argv
  hash, exact wall budget, supervisor PID/start token and all five false
  authorities.
- The worker verifies its actual direct parent using
  `NtQueryInformationProcess`, writes an exclusive hash-bound admission
  sidecar, then consumes/deletes the capability.
- The parent requires exact admission run ID/capability/argv/worker/parent
  bindings and capability absence before accepting worker exit zero.
- Windows worker and target processes are created suspended, assigned to
  nested `KILL_ON_JOB_CLOSE` Jobs, then resumed. Timeout/interrupt/descendant
  leakage requires zero active Job members.
- Stale reconciliation requires exact PID plus process-creation token
  inactivity and writes a new exclusive sidecar without editing the original
  pending receipt. Retry disposition is always a fresh run ID.

### Standard-user paths and telemetry

- Ambient `APPDATA`, `LOCALAPPDATA` and `PROGRAMDATA` are removed and
  redirected to fresh repo-local roots for parent and worker.
- `HOME`, `USERPROFILE`, `TEMP`, `TMP`, Conda/pip/uv, pytest and scientific
  library caches remain below `build/`.
- Explicit `C:\ProgramData` interpreter use remains read-only only.
- Windows Job CPU, kernel time, page faults, process counts and peak memory are
  recorded. I/O operations/bytes use
  `JobObjectBasicAndIoAccountingInformation` class 8; peak memory uses class 9.
- Declared mutable-root logical file/byte telemetry never follows links or
  reparse points. It is not a global filesystem sandbox.

## Exact commands and results

Every shell action used the literal cwd/Git-root guard for
`C:\Users\jbattaglia\PFC_LT`. Mutable paths were repo-local.

### Static and direct focused tests

```powershell
python -B -m ruff check scripts\run_workspace_local.py tests\test_run_workspace_local_script.py

build\pytest-runtime-v1\python.exe -I -B -m pytest `
  tests\test_run_workspace_local_script.py -q -p no:cacheprovider `
  --basetemp build\pytest-runner-v6-dev17
```

Result: Ruff pass and `131 passed in 12.36s` before the final supervised
replay. The final supervised replay repeats all 131 current tests.

### Negative complete-cycle timeout

```powershell
build\pytest-runtime-v1\python.exe -I -B -m `
  scripts.run_workspace_local --run-id sv6cur1 `
  --wall-timeout-seconds 300 -- `
  build\pytest-runtime-v1\python.exe -I -B -m pytest `
  tests\test_run_workspace_local_script.py -q -p no:cacheprovider
```

The child printed 131 passed, but the post-verification crossed the parent
budget. This is negative evidence only:

- supervisor/execution status
  `SUPERVISOR_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY`;
- exit 124, 300.093 seconds, declared overshoot 0.093 seconds;
- 52 total / 0 active processes; capability consumed then absent;
- supervisor receipt SHA-256
  `08bc35b92afc2f85bddf0f8d19ff70adbff8983b87a09645500d0913f72b2d90`;
- execution receipt SHA-256
  `e8eaf3674719797aa354207b4504520a5d3064eed7c240cdbdcfee56081a6542`.

Never cite `sv6cur1` as a pass.

### Selected focused supervisor proof

```powershell
build\pytest-runtime-v1\python.exe -I -B -m `
  scripts.run_workspace_local --run-id sv6cur2 `
  --wall-timeout-seconds 900 -- `
  build\pytest-runtime-v1\python.exe -I -B -m pytest `
  tests\test_run_workspace_local_script.py -q -p no:cacheprovider
```

Result:

- `131 passed`, zero failure/error/skip/deselect;
- supervisor `WORKER_EXIT_ZERO_NOT_AUTHORITY`, execution
  `TARGET_EXIT_ZERO_NOT_AUTHORITY`;
- supervisor receipt SHA-256
  `7c259be55205544bac2b96824a841269f40216fece127b5cf9bbffdae7a91c1d`;
- execution receipt SHA-256
  `474425a985667ebafece69165a7c2ade3eefea781c3083d2a04834fb3859350f`;
- native result SHA-256
  `84fbd43301e7738c2d3e9f1949b4f67954d8b5009bd38761674b64e1abcf7d70`;
- JUnit SHA-256
  `53217d97b304ce222932086a4c1886f94884a9d2a40f7166766908fc7fb3d8a3`;
- 575.562 seconds total, peak Job memory 1,477,988,352 bytes,
  1,078,173,908 read bytes / 1,308,513 write bytes, 52 total / 0 active
  processes;
- capability consumed and absent; admission SHA-256
  `93fb23845c226d64ecea94976a7ce525c71488064b6a52aef3c726bb8a6497ea`;
- all authorities false.

### Selected integrated runtime/packaging/publication matrix

```powershell
build\pytest-runtime-v1\python.exe -I -B -m `
  scripts.run_workspace_local --run-id sv6mx1 `
  --wall-timeout-seconds 1500 -- `
  build\pytest-runtime-v1\python.exe -I -B -m pytest `
  tests\test_run_workspace_local_script.py `
  tests\test_build_repo_local_conda_prefix_script.py `
  tests\test_launcherless_conda_archive_lock.py `
  tests\test_launcherless_local_runtime.py `
  tests\test_launcherless_runtime_admission.py `
  tests\test_lt_package_contract.py `
  tests\test_lt_ct_imports.py `
  tests\test_snapshot_publisher_artifact.py `
  tests\test_snapshot_publisher_runtime_closure.py `
  tests\test_snapshot_publication_external_contract.py `
  tests\test_candidate_bundle.py `
  tests\test_candidate_evidence.py `
  tests\test_candidate_evidence_assembler.py `
  tests\test_atomic_promotion.py `
  -q -p no:cacheprovider -m "not slow"
```

Result:

- `481 passed, 18 skipped, 1 deselected in 247.28s`, zero failure/error;
- total cycle 311.672 seconds;
- supervisor receipt SHA-256
  `f56958989827f5e7c0c2b6a3f628740963846845df52081cd0694de60964d983`;
- execution receipt SHA-256
  `cc21a0eed9ce8afee7ed946996571a60cce6254beab66302604b192ad1968a9a`;
- peak Job memory 3,012,988,928 bytes; read/write transfers
  1,586,611,167 / 55,280,382 bytes; 0 active processes;
- capability consumed and absent, deadline not exceeded, all authorities
  false.

The 18 skips are 4 absent optional CT extras, 10 publisher-wheelhouse cases,
2 publisher symlink cases and 2 atomic-promotion symlink cases. They do not
qualify CT runtime, publisher wheelhouse, Windows symlink policy or independent
CI.

## Real E2E process evidence selected inside `sv6mx1`

- zero: `e2ezb62dffa7b`, supervisor/execution SHA-256
  `c89a25861aa65afa2c8944887b20d22034d6a0d6a0443ec230c291ffede3087c` /
  `81f18a0c96146ec913946870af4ed699b66d3909be19a4247402abc053101233`;
- timeout before admission: `e2eb7c50f0061`, supervisor SHA-256
  `d04617b7e3c116738221c5fa2c0e2daaecc9a00b8dd3ce2df5ec6f7fcc25eee7`,
  capability revoked/absent and no admission;
- timeout after admission: `e2ea8508b90c4`, supervisor/execution SHA-256
  `e2ea3ae2230a729a907181fc35a5b062d9e328d9bcd9704e8e47c3df61465472` /
  `41fbe409f332d580d5ec090cdae2c618bc1e3b1a049bdb809d1c10b292064647`,
  capability absent and 0 active processes;
- abrupt supervisor death: `e2ece989de712`; its intentionally pending
  supervisor/execution receipts remain unchanged, worker identity is exited,
  capability is absent, admission exists, and exclusive reconciliation
  SHA-256 is
  `2bc2ce5f296c493623c9afc06fb2963ce7409ee55c477e3c8e225e270cd78b31`;
- Job-owner crash: `jobc627df770c`; exact Job primitives from the current
  runner created the child suspended, assigned before resume, and both child
  PID 54192 and descendant PID 46004 were absent after abrupt owner death.

## Intermediate negative/development evidence

- `sv6e2ez1` exposed a missing supervisor-bootstrap `appdata` path and stopped
  before worker launch; the capability was revoked. The fresh `sv6e2ez2`
  preflight then passed after the fix. Neither is selected terminal matrix
  evidence.
- Early `e2ec*` namespaces (`e2ec505f7c1da`, `e2ec252202622`,
  `e2ecb996c1a83`, `e2ec09328dd3e`, and later test-generated variants) are
  retained development/crash evidence. They were never reused or relabelled.
- Historical `rnv6sup1`, `rnv6mx1`, `supv6*` and `rnv6pt1` receipts bind older
  runner bytes and are not current qualification evidence.

## Independent terminal roasts

### Security/Governance

Verdict P0/P1/P2 `0/0/1`. The residual P2 is that the initial supervisor
module executes same-user bytes before it captures the worker source. The
worker is exactly attested, but already executed parent bytes are not protected
against pre-capture same-user TOCTOU. This is non-blocking only because the
harness is explicitly non-authoritative and not a security sandbox.

### IT/Operations

Local-quality verdict P0/P1 `0/0`; packaging closure accepted. Six P2 remain:

1. no cold/warm SLO by command class, capacity alert, resource quota or
   governed evidence-retention policy;
2. a reader could observe the first terminal receipt briefly before a
   deadline-crossing second fsync rewrites it to timeout;
3. no parent-directory flush/power-loss machine drill or formal RPO/recovery;
4. declared-root telemetry is not exhaustive filesystem auditing;
5. supervisor crash/reconcile and child-plus-descendant Job crash are separate
   proofs rather than one post-`EXECUTION_STARTED` capstone;
6. the POSIX process-group branch does not provide parent-death termination and
   must not be reused as CI containment.

### Quant/Data

Verdict P0/P1/P2 `0/0/0`. Runner code has no LT/CT solver logic and no T057
reference. Monthly solver authority, protocol v2, origin/target/mask semantics
and LT-to-CT independence are unchanged. These matrices prove local
runtime/packaging/publication behavior only, not scientific admission.

## Final scope audit

- Ruff and `git diff --check`: final pass after documentation finalization;
  explicit trailing-whitespace/final-LF checks also pass for untracked files.
- staged path count: 0.
- `pfc_shaping/ct/*` status: empty.
- Power BI status: empty.
- protected `data/eex_forwards_history.parquet` SHA-256:
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- user `.conda/environments.txt` SHA-256:
  `554db8db49b573851d3d299b962ad8048cfe404231d032791c0ce0d58d2bc92d`.
- user `.condarc` SHA-256:
  `998e455f1c09e5fd8abbee8560e1e576030cbdc0dd1b5c00bc27ca06d99c6f6e`.
- no task process remains after excluding the inspection shell itself.

## Residual production blockers and next direction

Supervisor v6 closes the laptop packaging slice only. Production remains
`NO_GO` until independent Windows CI/ASR/SBOM/signing, immutable bootstrap,
external CAS/WORM/fresh HEAD, exact atomic prefix recovery, production
observability/SLO/rollback and the P2 above are resolved.

Scientific blockers remain unchanged: external registry/lease, FMV-approved
UTC cadence, trusted time/signatures, PIT prospective commitments, official
market-event semantics, zero countable origins, future truth and independent
admission. T057 remains unopened and permanently ineligible for confirmatory
reuse.

After this packaging closure, return to fresh prospective CH inputs and the
outcome-blind origin registry. Accumulate independently registered rolling
origins, preserve T057 until its governed one-time evaluation, and build a new
auditable CH candidate only when fresh point-in-time evidence and all model
gates are available. Do not promote production.
