# Session Handoff 2026-07-17 - Publisher Runtime v6 Paused

## Pause state

- Canonical repo and cwd: `C:\Users\jbattaglia\PFC_LT`.
- Branch: `fix/lt-audit-remediation`.
- Starting HEAD remains `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. Do not reset, clean, restore or
  mass-stage it.
- No commit and no production promotion were performed.
- `data/eex_forwards_history.parquet` was not touched or staged.
- CT and Power BI were not touched.
- Production remains strictly `NO_GO`.
- User requested a pause to shut down/extend the workstation. The final v6
  real reproducibility test was deliberately interrupted.
- Only the session-owned test processes PID `33180` and `32500` were stopped.
  Protected unrelated PID `3572` was not touched.

## Exact real inputs

- Wheelhouse:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252`
- Dependency root (the exact `site-packages`, never the bundle parent):
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages`
- Receipt (sibling of `site-packages`):
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json`
- Bound source tree SHA-256 from the prior handoff:
  `0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`
- Receipt SHA-256:
  `69c0843b4b9a2e202c838ed04027eec1e5008ebec04c0b3c30f37217b0fb45a1`

## Work completed before pause

### Initial v4 proof and explicit sys.path test

- The corrected real v4 test passed:
  `1 passed, 16 deselected in 600.52s`; external wall
  `603.2791512s`.
- Its real zipapp SHA-256 was
  `2368789f906440b3f9bbd133822c3dc231c7ad5f3993cab748a18eed3aee057a`.
- Real dependency-tree and tampered-zipapp attacks passed:
  `2 passed, 15 deselected in 17.62s`.
- Added a direct test proving that admission appends only the private captured
  root and never the environment source root.

### Independent read-only roasts and accepted findings

Security and IT/Operations independently returned `NO_GO` until demonstrated
P1s were corrected. Accepted findings were:

1. lexical/case/path aliases could leave the source closure importable;
2. another base-relative Python path could shadow the captured dependencies;
3. closure builder cleanup could delete a racing unowned target;
4. `PFC_DATA_ROOT` in the environment contract was not consumed;
5. operational `OSError` could escape as traceback/exit 1 instead of JSON/50;
6. in-process Windows cleanup leaked loaded `.pyd`/`.dll` files;
7. the first v5 worker marker was forgeable through environment injection;
8. the first v5 parent did not own a kill-on-close process tree.

The IT roast observed legacy residue before v6: each orphan retained 59 `.pyd`
and 7 `.dll` files / `106040248` bytes. Do not delete these by glob while any
publisher Python may be active. The runbook now requires a proven outage and
process inventory. No legacy residue was deleted in this session.

### Runtime v6 implementation now in the worktree

Runtime contract is now `fmv_lt_snapshot_publisher_runtime.v6`:

- exact lexical plus physical `samefile` source-root rejection before and
  after capture;
- exact stdlib/zipapp `sys.path` allowlist; the base root is removed and
  arbitrary base-relative shadow paths are rejected;
- one-shot worker capability bound to parent PID, exact artifact and random
  supervisor scratch;
- reserved internal worker variables are forbidden deployment inputs;
- parent-owned Windows `KILL_ON_JOB_CLOSE` Job Object;
- POSIX worker session/group termination on controlled interruption;
- parent cleanup only after worker exit/native DLL unload;
- cleanup-race ownership guard (`target_created`) in the closure builder;
- `FMV_DATA_ROOT` contract correction;
- admission-boundary `OSError` normalization to structured JSON exit `50`;
- explicit external attestation scope for artifact, source closure, scratch,
  capture, dedicated token and absence of untrusted same-token co-processes.

Focused v6 verification before pause:

- runtime/closure lightweight matrix: `25 passed, 8 skipped in 3.95s`;
- targeted v6 security/contract tests: `9 passed, 15 deselected`;
- Ruff: `All checks passed`;
- `py_compile`: pass;
- Job Object probe: a 60-second worker exited immediately when the parent-owned
  job handle closed (`exit_after_job_close_seconds=0.0`).

Historical v5 evidence, superseded by v6:

- one real `--help`: exit `0`, `164.6683994s`, temp set `12 -> 12`, no new
  residue, artifact SHA
  `c8b683973bc467918efcb5037d23a04d12f2c2d076322cfbf44b72915147fe33`;
- full v5 reproducibility: `1 passed, 19 deselected in 789.18s`, wall
  `790.6631659s`.

## Files changed in this resumed slice

- `pfc_shaping/publisher_runtime_admission.py`
- `scripts/build_snapshot_publisher_runtime_closure.py`
- `scripts/build_snapshot_publisher_zipapp.py`
- `deploy/publisher/runtime-contract.json`
- `deploy/publisher/environment-contract.json`
- `deploy/publisher/README.md`
- `pfc_shaping/tools/OPERATIONS.md`
- `tests/test_snapshot_publisher_runtime_closure.py`
- `tests/test_snapshot_publisher_artifact.py`
- `.planning/HANDOFF.md`
- this handoff

These coexist with a much larger pre-existing dirty Phase 14 stack. Audit the
exact diff before any future commit and keep the heavy parquet excluded.

## Interrupted command - rerun from the beginning

The following final v6 real test was interrupted at the user's pause request.
Its partial execution is not evidence and must be rerun from the beginning:

```powershell
$env:PFC_TEST_PUBLISHER_WHEELHOUSE='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_ROOT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_RECEIPT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json'
$timer=[System.Diagnostics.Stopwatch]::StartNew()
python -m pytest -q tests\test_snapshot_publisher_artifact.py -k "reproducible_runnable_and_signer_free" -p no:cacheprovider
$code=$LASTEXITCODE
$timer.Stop()
Write-Output ("WALL_SECONDS={0:F7}" -f $timer.Elapsed.TotalSeconds)
exit $code
```

## Resume order

1. Reverify exact cwd/root, branch, HEAD, dirty status and absence of session
   test processes. Do not touch PID 3572.
2. Rerun the interrupted real v6 reproducibility test and record the new pyz
   SHA, pytest duration, wall time and before/after temp inventory.
3. Rerun real dependency-tree/tampered-member attacks.
4. Rerun runtime (`test_snapshot_publisher_runtime_closure.py` plus artifact),
   publication/CAS (five snapshot publication files), packaging
   (`test_lt_package_contract.py`), targeted Ruff, `py_compile`, and
   `git diff --check`.
5. Ask Security and IT/Operations for a final read-only re-roast of v6. Require
   explicit confirmation that the environment capability, stdlib allowlist,
   Job Object and cleanup ownership P1s are closed.
6. Correct only demonstrated remaining findings, then rerun affected matrices.
7. Update the external-CAS RFC and append a durable decision to
   `DECISION-LOG.md`; update this handoff with final commands, hashes, results,
   residual P2s and risks.
8. Perform final scope/diff audit. Do not commit without that audit and do not
   promote production.

## Residual risks / next product work

- Startup remains very slow (v5 single admission about 165 seconds; full
  v5 test about 13 minutes). Define an operational startup SLO and timeout
  policy; this remains at least P2 even if v6 is functionally green.
- Signed machine-verifiable IT release attestation, real external anchor,
  service ACL/HSM/WORM, exact-volume multi-host/power-loss/backup/DR evidence,
  and provider-raw prospective acquisition remain absent.
- After packaging closure, return to fresh prospective data, the locked T057
  future holdout and a new auditable CH candidate. Monthly solver authority,
  no post-solve month patching and OMPEX benchmark-only status remain
  invariant.
