# Session handoff - Provider-raw PIT and Windows namespace hardening

Date: 2026-07-23

## Canonical state and prohibitions

- Workspace and Git root: `C:\Users\jbattaglia\PFC_LT`.
- Branch: `fix/lt-audit-remediation`.
- HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. Nothing was reset, restored,
  cleaned, staged or committed.
- `data/eex_forwards_history.parquet` remains locally modified and unstaged; it
  was not read, changed or staged in this slice.
- `pfc_shaping/ct/*` and Power BI remain untouched.
- Monthly solver remains sole monthly level authority; OMPEX is benchmark-only.
- No production snapshot, candidate, image or flag was promoted. Status is
  strict `NO_GO`.

Every shell command was guarded by exact cwd and Git-root equality against the
canonical `C:` path. No command used the retired `H:` repository.

## Changed files in this slice

- `pfc_shaping/cli/governed_acquisition_builder.py`;
- `pfc_shaping/data/governed_lt_acquisition.py`;
- `pfc_shaping/durable_artifact.py`;
- `pfc_shaping/tools/OPERATIONS.md`;
- `tests/test_governed_lt_acquisition.py`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`;
- this handoff and `.planning/HANDOFF.md`.

Final pre-documentation hashes:

- Builder: `f7db1c395ea8f30210e5a12859d410dd17e9e491e3a8c62412ef9fea386eec82`;
- acquisition transforms: `777f6b89826ea61ba7757b8312675606ab9b427be5c61d53d34a3f3fcdd75761`;
- durable artifact: `3c7935d3709e90b5635406846352db6bc81172f6ae134e1f6239e85898d44063`;
- acquisition tests: `5c5189426dc4013510e3bce75ed74e27ba022609e00f5245e4faf34bce3cdde1`;
- Operations runbook: `6e8ad772704c3575cc9efa3dc3d72e663906cdb0242091593b15215c87314343`.

## Closed local findings

### Provider PIT

- Energy Charts permits at most exactly 48 hours of day-ahead lead; `48h+1s`
  fails.
- ENTSO-E request/chunk end must not exceed `received_at`.
- SFOE checks the maximum timestamp in the complete raw CSV before any window
  filter, including DST cases.
- Replay re-runs the allow-listed provider transform and compares the produced
  frame exactly with the bounded Parquet payload.

### Windows namespace and TOCTOU

- Raw spec, body and output paths must be absolute and canonical.
- UNC, ADS, reparse traversal, non-fixed drives, indirect DOS/SUBST targets,
  `~` aliases, trailing aliases, reserved devices, control characters and
  `< > " | ? *` are rejected.
- `QueryDosDeviceW('C:')` resolved to `\Device\HarddiskVolume3` during the
  IT/Operations roast.
- All output-parent ancestors are pinned without delete sharing. Staging is
  pinned with `DELETE` and without `FILE_SHARE_DELETE`.
- Windows promotion uses `NtSetInformationFile`, information class 10, with
  `RootDirectory=parent_pin.native_handle` and a validated simple name.
- A test changes the process cwd to an attacker-controlled directory and proves
  the output remains under the pinned parent. A separate test proves staging
  substitution is blocked.

An intermediate experiment used `RootDirectory=NULL`; Windows resolved the
simple destination name from the process cwd and created exactly
`C:\Users\jbattaglia\PFC_LT\unsigned-build`. The five known generated files
were inspected, the exact directory path was resolved, and only that generated
directory was deleted. The regression test prevents recurrence.

### Crash recovery and concurrency

- Builder-only exact hardlink recovery requires a proved writer lease.
- Windows proves exclusion through the staging handle share mode; POSIX now
  acquires `flock(LOCK_EX|LOCK_NB)` and refuses a second writer.
- A complete exact two-link residue is revalidated and recovered. Divergent or
  ambiguous residues are retained and rejected.
- The global `write_durable_exact_bytes` remains concurrency-safe and does not
  apply the Builder's exclusive-recovery rule.

### Authority boundary

Builder manifests are explicitly `UNSIGNED`, `NOT_PUBLISHED` and mutable
quarantine. Operations requires Builder shutdown/token revocation, independent
exact-byte copy into a Builder-inaccessible namespace, post-copy replay and
hash validation, independent signature, retained `icacls` evidence and
negative write/delete/rename/hardlink probes.

This is specified but not yet executed. IT/Operations therefore retains a P1
release blocker outside the local Builder; no software test is reported as a
substitute for that external proof.

## Commands and results

All commands below were executed from the exact canonical root with the guard
described above.

Acquisition and lease/path closure:

```powershell
python -m pytest tests\test_governed_lt_acquisition.py -q -p no:cacheprovider
```

Final result: `66 passed, 1 skipped in 2.31s`. The skip is the externally
parameterized installed-wheel test when no wheel variables are supplied.

Package and acquisition contract:

```powershell
python -m pytest tests\test_governed_lt_acquisition.py tests\test_lt_package_contract.py -q -p no:cacheprovider
```

Result: `88 passed, 1 skipped in 5.10s`.

Independent Quant/Data command:

```powershell
python -m pytest tests\test_governed_lt_acquisition.py tests\test_governed_lt_input_snapshot_v2.py tests\test_lt_input_sources.py tests\test_lt_replay_transforms.py -q -p no:cacheprovider
```

Result: `152 passed, 3 skipped in 214.43s`.

Publication/CAS command:

```powershell
python -m pytest tests\test_governed_lt_input_snapshot_v2.py tests\test_snapshot_anchor_client.py tests\test_snapshot_anchor_reference.py tests\test_snapshot_bootstrap_signer.py tests\test_snapshot_publication_external_contract.py tests\test_snapshot_publisher_artifact.py -q -p no:cacheprovider
```

Result: `165 passed, 13 skipped in 198.26s`. This includes the repaired anchor
client concurrency test. Later edits were confined to Builder path grammar and
did not modify publication code.

Final runtime, artifact and container contract:

```powershell
python -m pytest tests\test_snapshot_publisher_runtime_closure.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_container_contract.py -q -p no:cacheprovider
```

Result on final source: `70 passed, 13 skipped in 41.11s`.

Explicit import-search proof:

```powershell
python -m pytest tests\test_snapshot_publisher_artifact.py::test_runtime_admission_appends_only_private_captured_dependency_root tests\test_snapshot_publisher_artifact.py::test_isolated_sys_path_removes_runtime_root_from_import_search tests\test_snapshot_publisher_artifact.py::test_dependency_import_sys_path_mutation_is_rejected -q -p no:cacheprovider
```

Result: `3 passed in 0.29s`. Only the process-private captured dependency root
is appended; deployment/source roots are removed and post-import mutation is
rejected.

Real optimized zipapp command used the exact retained handoff paths:

```powershell
$env:PFC_TEST_PUBLISHER_WHEELHOUSE='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_ROOT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_RECEIPT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json'
python -m pytest tests\test_snapshot_publisher_artifact.py::test_optimized_publisher_zipapp_prepares_real_provider_raw_v3_bundle -q -p no:cacheprovider
```

Result: `1 passed in 324.71s`.

Two fresh wheels were built with `uv build --quiet --wheel` into separate
outputs and audited with `python -m scripts.check_lt_wheel_contract`. Both
audits returned PASS, 79 members, identical bytes:

- SHA-256: `05f8fb6d39f7aea55522406f408b149c5ed34a4310bbb75519de024beec3f7bc`;
- size: 426,307 bytes;
- embedded source revision:
  `3658cca4abdda274401bad12e6018aaf8681be7fb09ddcf8f6bef7acd27e2391`;
- wheel A:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\acquisition-pit-toctou-wheel-final-20260723-212310-a\fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- wheel B:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\acquisition-pit-toctou-wheel-final-20260723-212310-b\fmv_pfc_lt-0.14.0-py3-none-any.whl`.

With wheel A and the exact retained dependency root, the installed
`pfc-lt-build-acquisition` entrypoint test returned `1 passed in 4.61s`.

The first pre-fix double wheel build correctly failed its audit because the
literal physical-device regex looked like an embedded UNC path. That source
literal was removed; the source portability scan, package matrix and both new
wheel audits pass. The rejected wheels under the `...-20260723-212101-{a,b}`
directories are non-authoritative and must not be used.

Ruff on the four changed Python/test files, `py_compile`, the source portability
scan and targeted `git diff --check` pass. Git reports only the existing
Windows LF-to-CRLF notice for `OPERATIONS.md`, not a whitespace error.

## Independent roasts

- Security final: no demonstrated P0/P1/P2; targeted path/lease/hardlink tests
  `11 passed`, acquisition plus anchor `111 passed, 1 skipped`.
- Quant/Data: PASS; no P0/P1 in PIT bounds, timestamp chain or exact replay.
- IT/Operations: no local Builder P0/P1. Production remains `NO_GO` because
  the independent Builder-inaccessible handoff, signed ACL/freeze receipt and
  service-identity post-copy replay have not been executed.

## Current blockers and next action

At `2026-07-23T19:25:11Z`, T057 was still pre-maturity. The canonical output
directory did not exist, and no provider call was made. Do not execute before
`2026-07-24T00:00:00Z`; after maturity use only the frozen one-shot sidecar,
without retuning or caller-selected clock.

Before production, execute under real service identities on the target volume:

- Builder stop/revocation and independent namespace handoff with signed
  ACL/freeze receipt and negative capability probes;
- kill/restart at write/fsync/hardlink/rename boundaries, concurrent exact and
  divergent writers, disk-full/memory-kill and live namespace remap drills;
- real container PREPARE/CAS/FINALIZE, external CAS/WORM, HSM/KMS or broker,
  monitoring/alerts, backup/restore, power-loss and DR evidence.

After the maturity boundary and packaging closure, execute T057 once, then
return to fresh point-in-time prospective inputs and a new auditable Swiss CH
candidate. Never promote until the complete external and scientific evidence
chain passes.
