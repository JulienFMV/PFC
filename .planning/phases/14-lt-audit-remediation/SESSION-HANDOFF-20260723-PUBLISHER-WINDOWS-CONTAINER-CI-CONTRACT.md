# Session Handoff - Publisher Windows Container/CI Contract (2026-07-23)

## Status

Canonical workspace: `C:\Users\jbattaglia\PFC_LT` only.

- branch: `fix/lt-audit-remediation`;
- HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- worktree intentionally very dirty; staged file list is empty;
- `data/eex_forwards_history.parquet` was observed as pre-existing modified and
  was never edited or staged in this slice;
- no CT or Power BI file was changed;
- no commit, image push, snapshot publication, T057 attempt or candidate
  promotion occurred;
- software verdict for this reviewed packaging slice: conditional `GO`, no
  demonstrated P0/P1 after Security and IT/Operations re-roast;
- packaging/infrastructure and production verdict: strict `NO_GO` because
  Docker and the external operational proofs are absent.

This handoff supersedes the packaging portion of
`SESSION-HANDOFF-20260723-PUBLISHER-POST-ADMISSION-T057-FINAL-RECLOSURE.md`.
That earlier handoff remains authoritative for the frozen T057 one-shot plan.

## Implemented contract

### Image and build boundary

- One lane only: Windows `amd64`, Docker build/run with
  `--isolation process`; Hyper-V and Linux are not equivalent lanes.
- Builder and runtime share one fully qualified digest-pinned base.
- CI inspects base and final image architecture/OS, host/base Windows
  compatibility, inherited/final volumes, user, workdir, entrypoint, command,
  healthcheck and exact OCI/FMV labels.
- The built image is addressed by its immutable iidfile image ID. This workflow
  neither authorizes a mutable tag nor pushes an image.
- The Dockerfile declares no `VOLUME`, creates explicit scratch/data/evidence/
  input directories, checks every native `icacls` exit, runs negative and
  positive write probes, exposes all four help surfaces and requires a second
  internal zipapp build to reproduce both artifact and receipt bytes.

### Context, wheelhouse and secret claims

- `.dockerignore` is an exact positive list with 11 wheel filenames; no broad
  `*.whl` rule, CT, Power BI, data tree or environment file is admitted.
- `stage_snapshot_publisher_container_wheelhouse.py` selects the exact lock
  inventory, rejects links and unstable multi-link reads, verifies size/hash,
  durably publishes a canonical manifest and supports immediate `--audit`.
- The context checker binds every required artifact/contract by hash, rejects
  extra entries, links, zip duplicate names and case collisions, and scans
  decompressed zipapp members as well as closure/receipts.
- The scan claim is deliberately narrow:
  `PASS_NO_SYNTACTIC_PEM_OR_OPENSSH_PRIVATE_KEY_BLOCKS`. A match needs paired
  exact begin/end labels, strict Base64 lines, successful decode and at least
  32 decoded bytes. Marker-only constants in `pyarrow\arrow.dll` and
  `cryptography\...\ssh.py` do not match; complete direct and deflated blocks
  do. DER/PFX/JWK, semantic key validity and the base image are not covered.

### Runtime, operations and rollback

- Only the process-private captured dependency root may enter `sys.path`.
  Deployment/source roots are removed and post-import path mutation is
  rejected.
- Scratch is the only general RW mount; input is RO; data/evidence mounts are
  phase-specific; no anonymous volumes are permitted.
- Secret directories must exist at start, but signing and mTLS paths remain
  deferred until the PID/token-bound post-admission one-shot capability.
- The sealed admission metric is emitted on stderr after success. If admitted
  post-admission authority delivery fails, the sealed metric is preserved on
  stdout and the sole failure JSON remains on stderr.
- `restartPolicy=Never`; only exit `41` and `52` retry; five retries/six total
  executions, backoffs `5,15,30,60,120`, one stable operation ID.
- Rollback requires immutable target digest, compatible schema/protocol and
  trust registry/domain plus fresh authenticated CAS HEAD. External HEAD is
  never rewound.

## Files in this packaging slice

New:

- `.dockerignore`;
- `.github/workflows/publisher-runtime-v6.yml`;
- `deploy/publisher/Dockerfile`;
- `deploy/publisher/operations-contract.json`;
- `deploy/publisher/README.md`;
- `scripts/check_snapshot_publisher_container_context.py`;
- `scripts/stage_snapshot_publisher_container_wheelhouse.py`;
- `tests/test_snapshot_publisher_container_contract.py`;
- this handoff.

Modified:

- `deploy/publisher/environment-contract.json`;
- `pfc_shaping/publisher_runtime_admission.py`;
- `tests/test_snapshot_publisher_artifact.py`;
- `.planning/HANDOFF.md`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`.

Other dirty files predate or belong to adjacent Phase 14 work. Do not infer
that the full dirty worktree belongs to this slice, and do not reset it.

## Retained real inputs

- wheelhouse:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252`;
- dependency root:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages`;
- dependency receipt:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json`;
- audited container wheelhouse:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-container-validation-20260723\publisher-container-wheelhouse`;
- staging manifest:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-container-validation-20260723\publisher-container-wheelhouse-manifest.json`;
- all final logs:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-container-validation-20260723`.

The audited wheelhouse has 11 files and 58,746,108 bytes. Its canonical
manifest SHA-256 is
`f79357a48cf83c1d3887da7b6eab03dc2d86d5a05355e5e36a3e82e9590addcf`;
the source `uv.lock` SHA-256 is
`efcea25267644da75c8736b3ede0dfaaf4b6ee8e58b982a61e87edb1064eb5d6`.

## Commands and final results

Every command was prefixed by an exact, case-sensitive assertion that both cwd
and `git rev-parse --show-toplevel` equal
`C:\Users\jbattaglia\PFC_LT`.

### Real zipapp/context

The real combined run selected:

```powershell
$env:PFC_TEST_PUBLISHER_WHEELHOUSE='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_ROOT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_RECEIPT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json'
python -m pytest tests\test_snapshot_publisher_artifact.py::test_real_publisher_bundle_seals_a_non_authorizing_container_context tests\test_snapshot_publisher_artifact.py::test_optimized_publisher_zipapp_prepares_real_provider_raw_v3_bundle -q -p no:cacheprovider
```

Result: the optimized real provider-raw prepare passed; the context test alone
failed because the original marker-only scan found format strings inside
`arrow.dll` and `cryptography\...\ssh.py`: `1 failed, 1 passed in 497.30s`.
After narrowing the claim to complete syntactic key blocks, the exact context
test rerun passed: `1 passed in 337.97s`. Logs:
`real-artifact.stdout.log` and `real-context-rerun.stdout.log`.

### Final matrices on frozen current source

```powershell
python -m pytest tests\test_snapshot_publisher_runtime_closure.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_container_contract.py -q -p no:cacheprovider
```

Result: `70 passed, 13 skipped in 35.35s`.

```powershell
python -m pytest tests\test_snapshot_publisher_artifact.py::test_runtime_admission_appends_only_private_captured_dependency_root tests\test_snapshot_publisher_artifact.py::test_isolated_sys_path_removes_runtime_root_from_import_search tests\test_snapshot_publisher_artifact.py::test_dependency_import_sys_path_mutation_is_rejected -q -p no:cacheprovider
```

Result: `3 passed in 0.24s`. This is the explicit proof that only the captured
private root enters import search and that reintroduction/mutation fails.

Publication command covered anchor client/reference, bootstrap signer,
external publication contract, governed snapshot/acquisition and artifact.
Result: `202 passed, 14 skipped in 198.93s`; log
`publication-final-20260723-192934.stdout.log`.

Packaging command covered package contract, governed release/script,
candidate evidence/assembler/bundle, atomic promotion, quality gate, manifest
promotion, governed forwards, LT input sources and lambda calibration. Result:
`544 passed, 4 skipped, 2 warnings in 857.17s`; log
`packaging-final-20260723-193312.stdout.log`. The warnings are the expected
All-NaN/empty-mean NumPy warnings in
`test_fail_closed_if_history_is_insufficient`.

The prebuild replay command was:

```powershell
python -m scripts.stage_snapshot_publisher_container_wheelhouse --audit --source 'C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-container-validation-20260723\publisher-container-wheelhouse' --manifest-output 'C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-container-validation-20260723\publisher-container-wheelhouse-manifest.json'
```

Result: `status=PASS`, 11 exact wheels and the lock/hash values above.

Workflow YAML parsing, targeted Ruff and targeted `py_compile` pass. Docker and
Buildx discovery reports both unavailable. Final documentation-aware
`git diff --check` and scope audit are recorded below after this handoff is
written.

## Current file hashes before documentation-only edits

- `.dockerignore`:
  `78e3fcf264fc2ece2bf80f317d6edfa4f15bfc1e62ab09b813da3d683c10e9f7`;
- `.github/workflows/publisher-runtime-v6.yml`:
  `561f7ebcac6381063d5114ba878e027f0e720a0ce5a9afcd9af1b2660a354554`;
- `deploy/publisher/Dockerfile`:
  `2f76babd380541ad11e2d5208222ca7b7d19eb951ea7e80f2273d0c7914002c7`;
- `deploy/publisher/environment-contract.json`:
  `46ba580b40d68b7b4a8b6c50508b775c835ba1e4f78f01ce287d382e73be8949`;
- `deploy/publisher/operations-contract.json`:
  `8d1af62295835df056cedb7faab8040ddbf0fdcfdeaba99f5288f066d33bf433`;
- `deploy/publisher/README.md`:
  `df68efeedd3041a268d7f9806401911f5e6da8c79e14972d1481a20ef1934b33`;
- `pfc_shaping/publisher_runtime_admission.py`:
  `ec1db4ecf1ed86c33c77a641b2d6805df5936210666fd67b82d1eb5c14188f0f`;
- `scripts/check_snapshot_publisher_container_context.py`:
  `b867d7260196ba8d5442f2199efdf18b1c945f49d4362b1dc92f17de15c35ce6`;
- `scripts/stage_snapshot_publisher_container_wheelhouse.py`:
  `b96b41616d247005e8c0e483e17ad373c4c80ba1a780178b163ff59da98fe033`;
- `tests/test_snapshot_publisher_artifact.py`:
  `4064edf01920a6a18fc5b459166f570ca42395c75497f8266507683b215d8470`;
- `tests/test_snapshot_publisher_container_contract.py`:
  `df5e6249b633ab1f2a96602a58434896859a9eb65a083105f8a6ad8bcb269c66`.

## Independent roasts

Security final verdict: no P1 introduced by the corrected scanner. The
syntactic claim is honest, the zipapp scan is artifact-hash-bound,
decompressed and duplicate/case-collision-safe. Base-image and binary/semantic
secret scanning remain explicitly external.

IT/Operations final verdict: no demonstrated P0/P1 remains in post-admission
metric delivery, stream separation or retry lifecycle. Five retries/six total
executions and the operations-contract hash above agree across contract,
validator, tests and runbook.

Both verdicts are software-scope only and preserve the external `NO_GO`.

## External blockers and next action

Required before production:

- approved signed Windows base digest and vulnerability/secret scan;
- protected locked host Python plus isolated/JIT Windows runner and daemon;
- real process-isolation image build and all image-inspection checks;
- true containerized PREPARE, external CAS and FINALIZE drill, including
  failures, restart/retry, rollback and fresh HEAD behavior;
- independent second-builder final-image reproducibility;
- signed SBOM, provenance, image signature and registry admission policy;
- service identity, exact ACL and post-admission HSM/KMS/broker delivery;
- metric collection/alerts, audit retention, multi-host/power-loss,
  backup/restore and DR evidence.

Do not manufacture these proofs on a host without Docker. Production remains
strict `NO_GO`.

T057 remains frozen and must not be acquired or scored before
`2026-07-24T00:00:00Z`. After that maturity boundary, use only the canonical
one-shot sidecar route from the prior handoff, without retuning. Then return to
fresh point-in-time prospective inputs and a new auditable CH candidate. The
monthly solver remains the level authority and OMPEX remains benchmark-only.

## Final documentation and scope audit

Audit at `2026-07-23T17:53:47Z`:

- cwd and root still exactly `C:\Users\jbattaglia\PFC_LT`;
- branch still `fix/lt-audit-remediation`, HEAD still
  `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- full `git diff --check`: exit `0` (only expected Windows LF/CRLF notices);
- explicit trailing-whitespace scan across every file in this slice: empty;
- staged file list: empty;
- protected parquet status remains only
  ` M data/eex_forwards_history.parquet` and it was not staged;
- `git status --short -- pfc_shaping/ct powerbi`: empty;
- pytest/publisher worker process audit: empty;
- canonical T057 output
  `output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724`
  does not exist; maturity is still in the future at this audit time;
- no commit or production action was performed.

## Post-packaging T057 readiness addendum

At `2026-07-23T18:02:36Z`, T057 remained pre-maturity. No provider call was
made and neither the canonical output directory nor the attempt seal existed.

Read-only evidence:

- frozen plan SHA-256 remains
  `f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd`;
- baseline exists and matches
  `12447bbaa9828c0ffed871e62c35f90b8c100fcfab8c80b00468ac846848d895`;
- adjusted T056 t005 exists and matches
  `5e603a4d5926f9265ca564615e69d0d7ee39f778f6f19b495706ab1b89cf69b6`;
- canonical route, runner, independent audit and policy matrix:
  `47 passed in 265.77s`; log
  `t057-readiness-final-20260723-195750.stdout.log` under the retained log root.

The execution sidecar had one documentation drift: its blocking inventory
listed provider raw, Parquet, fetch summary and capture seal but omitted the
attempt seal added by D150. The sidecar now states that
`energy_charts_locked_holdout_attempt_seal.json` is created exclusively,
flushed and `fsync`ed before the provider call, consumes the local one-shot
even after crash/failure, and is hash-bound by the capture chain. Sidecar
SHA-256 after this documentation-only correction:
`188d58e22bfe6fdb21b309b872db907168aea888a12650e4231f33fbaab32e9e`.

No Python code, frozen plan byte, input artifact, selection, threshold or
production authorization changed. T057 must still wait until
`2026-07-24T00:00:00Z` and then use only the exact sidecar command.
