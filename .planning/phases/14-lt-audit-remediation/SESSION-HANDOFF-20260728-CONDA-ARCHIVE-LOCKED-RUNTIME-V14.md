# Session handoff - 2026-07-28 - Conda archive-locked launcherless runtime v14

## Scope and immutable constraints

Work was performed only in the canonical repository
`C:\Users\jbattaglia\PFC_LT`, on branch `fix/lt-audit-remediation`, observed
HEAD `2f68125bff869ccb21c1e20df0201ad024ed27d3`. The worktree remained
intentionally very dirty. Nothing was reset, cleaned, restored, staged or
committed. No production publication or promotion was attempted. No CT or
Power BI file was touched. Monthly solver authority and OMPEX benchmark-only
status are unchanged.

`data/eex_forwards_history.parquet` was never touched or staged. Its final
observed SHA-256 remained
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

Every shell action was preceded by a separate exact cwd/Git-root guard. No
admin right, network access, Defender/ASR exception, Playwright/browser
automation or project executable was used.

## Problem closed in this local slice

D168 left two local packaging gaps: the CPython Conda base was only manifested
after the fact, and the publisher wheelhouse/closure inputs still lived under
`AppData\Local`, which caused unnecessary workspace permission prompts. This
session:

1. built and replayed an exact lock over all retained Conda archive bytes;
2. materialized a deterministic local `@EXPLICIT` spec;
3. created a fresh prefix with Conda `--offline --copy --file`, without solver,
   network or admin rights;
4. captured the complete prefix before first target-Python execution;
5. audited the prefix against the exact archives/spec/history and emitted a
   caller-held prefix-build receipt;
6. copied retained publisher inputs into a fresh repo-local staging namespace;
7. upgraded the local runtime receipt/admission contract from v2 to v3 so the
   prefix-build receipt is mandatory;
8. built and admitted a new installed runtime v14 containing the current
   CLI/high-level/atomic transition guard.

This closes exact local Conda provenance and the unnecessary `AppData` build
boundary. It does **not** close same-user verification before import and does
not confer production authority.

## Changed source and test files in this slice

- `scripts/build_launcherless_conda_archive_lock.py` (new);
- `scripts/build_launcherless_local_runtime.py` (new/current v3 contract);
- `pfc_shaping/pipeline/governed_release_cli_contract.py`;
- `tests/test_launcherless_conda_archive_lock.py` (new);
- `tests/test_launcherless_local_runtime.py` (new/current);
- `tests/test_launcherless_runtime_admission.py` (new/current);
- `tests/test_atomic_promotion.py`;
- `.planning/HANDOFF.md`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`;
- `pfc_shaping/tools/OPERATIONS.md`;
- this handoff.

The rest of the dirty worktree predates or is outside this focused slice and
must be preserved.

## Exact Conda archive and base-prefix evidence

Archive lock:

- path:
  `build/launcherless-conda-archive-lock-20260728-v12.json`;
- size: 19,748 bytes;
- SHA-256:
  `451ce7b960683414e8fad74b3668e2bb0ea530c644c14aa30fb08aef4e8d2f31`;
- archive lock ID:
  `235edd15a174e84cab2bb8b126a7e259a80c6c9e8af9f55ef0fbc643d5c91ee5`;
- archive-set ID:
  `05d73b203f0401f0034c42e7b29c43ef04f54a2632455648e254f2e8581d8535`;
- 19 packages, 38,648,096 retained archive bytes;
- replay against every retained archive: `PASS`.

Explicit spec:

- path: `build/launcherless-conda-explicit-20260728-v12.txt`;
- size: 2,035 bytes / 20 lines including `@EXPLICIT`;
- SHA-256:
  `c1da60c9c3474453ebc21580cecc15c556c9bcd82e9113c73bdaa62ec883e4eb`.

The first replay target
`build/conda-runtime-v13-archive-locked-base` timed out during Conda creation
and contains an incomplete 6,036-file/19-record prefix. No Conda process
remained. It is preserved as negative evidence and must never be repaired,
relaunched or relabelled.

The successful fresh prefix is
`build/conda-runtime-v14-archive-locked-base`. Exact Conda command:

```powershell
conda create --offline --copy `
  --prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v14-archive-locked-base `
  --file C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260728-v12.txt `
  --yes --json
```

It exited 0 in 83.6 seconds. The only warning was inability to update the user
Conda `environments.txt`; it did not affect prefix bytes. Target Python had not
been executed.

Caller-held pre-execution manifest:

- path:
  `build/launcherless-python-runtime-manifest-20260728-v14-base.json`;
- size: 911,003 bytes;
- SHA-256:
  `b10daeaf63691f37a72db80e37d690d3c44c42108b12d58308efebdacc55afb5`;
- 6,285 files;
- tree SHA-256:
  `4ab93b1dbe6b33eab88907a97cde3eb13ded0e5c0dcfe33302fdea7aaf43b4e5`;
- `python.exe` SHA-256:
  `50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`;
- `python311.dll` SHA-256:
  `026dfdf86464bf589085e5c37904f1df91c1b59c989fcaee36be66fdd29e773a`.

Prefix-build receipt:

- path:
  `build/launcherless-conda-prefix-build-receipt-20260728-v14.json`;
- size: 11,640 bytes;
- SHA-256:
  `fb333a8f855284672c3a42a8832a49615d3d4c416908e458d594419dc4bcd98e`;
- prefix receipt ID:
  `1cb82d795cc53c8d1ba291065f97725de5fba89ebe55972ccc2c37b551c09f83`;
- schema/status:
  `fmv_lt_launcherless_conda_prefix_build_receipt.v1` /
  `PASS_LOCAL_EXPLICIT_ARCHIVE_REPLAY_NOT_PRODUCTION`;
- `production_authorization=false`, `promotion_gate=false`.

Exact archive set and complete installed-file inventory are authoritative for
this local replay. Source/target dependency-string differences caused by Conda
patched repodata are recorded but deliberately non-authoritative.

## Repo-local publisher inputs and non-admin build

The retained publisher wheelhouse, closure and receipt were copied without
modifying their source into:

- `build/runtime-inputs-20260728-repolocal-v1/publisher-wheelhouse`;
- `build/runtime-inputs-20260728-repolocal-v1/publisher-site-packages`;
- `build/runtime-inputs-20260728-repolocal-v1/publisher-dependency-closure-receipt.json`.

The existing publisher verifier subsequently replayed wheel and closure bytes
inside the runtime builder. All v14 build inputs, outputs, `TEMP` and `TMP`
were below the canonical repo. The command generated no permission prompt and
used no elevation.

## Reproducible wheels I/J

Commands, both with `SOURCE_DATE_EPOCH=1783987200` and separate repo-local
build/temp directories:

```powershell
python -B setup.py build --build-base build\wheel-build-i `
  bdist_wheel --dist-dir build\wheel-dist-i
python -B setup.py build --build-base build\wheel-build-j `
  bdist_wheel --dist-dir build\wheel-dist-j
```

Both wheels:

- 461,476 bytes / 84 members;
- byte-identical SHA-256:
  `48bdb58134506422a76669ed4343b5b81c9b7eef000204ce5303a4a59b6e3734`;
- embedded source revision:
  `51946b9b60f024c8230d004b87a75fe9646c6c32c7a020f8247facedfd49bcb6`;
- wheel audit `PASS`, `promotion_eligible=false`.

The retained `setup.py` deprecation warning remains an IT/CI concern. This
fixed-directory local fallback avoids the known Windows temporary ACL defect;
it is not the future clean-tree PEP 517 production build.

## Runtime v14 assembly and exact sys.path

The builder was invoked with only repo-local inputs:

```powershell
python -B -m scripts.build_launcherless_local_runtime `
  --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v14-archive-locked-base `
  --project-wheel C:\Users\jbattaglia\PFC_LT\build\wheel-dist-i\fmv_pfc_lt-0.14.0-py3-none-any.whl `
  --publisher-wheelhouse C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-wheelhouse `
  --publisher-dependency-root C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-site-packages `
  --publisher-receipt C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-dependency-closure-receipt.json `
  --additional-wheel-directory C:\Users\jbattaglia\PFC_LT\build\launcherless-wheelhouse-20260727-v1 `
  --python-runtime-manifest C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260728-v14-base.json `
  --expected-python-runtime-manifest-sha256 b10daeaf63691f37a72db80e37d690d3c44c42108b12d58308efebdacc55afb5 `
  --conda-prefix-build-receipt C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-prefix-build-receipt-20260728-v14.json `
  --expected-conda-prefix-build-receipt-sha256 fb333a8f855284672c3a42a8832a49615d3d4c416908e458d594419dc4bcd98e `
  --receipt-output C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260728-v14.json `
  --lock-path C:\Users\jbattaglia\PFC_LT\uv.lock
```

Result: exit 0 in 863.5 seconds under one 20-minute supervisor, with no
overlapping retry. The path name retains the `-base` suffix but now contains
the completed runtime; do not rename it because all manifests bind the lexical
path.

Runtime receipt:

- schema `fmv_lt_launcherless_local_runtime.v3`;
- path: `build/launcherless-runtime-receipt-20260728-v14.json`;
- size: 6,931 bytes;
- SHA-256:
  `6ec9638fade90c2730f90079af794d63452428afe6db830fd22b67fa4702bccf`;
- closure: 8,488 files / 19 distributions;
- closure tree SHA-256:
  `09bca4a65c0bc6c9b39a036be2cc49384f0371851625f539e8d9a9ebabfcfcba`;
- `local_quality_authorization=true`;
- `production_authorization=false`.

Exact receipt `sys.path`, in order:

1. `...\conda-runtime-v14-archive-locked-base\python311.zip`;
2. `...\conda-runtime-v14-archive-locked-base\Lib`;
3. `...\conda-runtime-v14-archive-locked-base\DLLs`;
4. `...\conda-runtime-v14-archive-locked-base\governed-site-packages`.

There are exactly four entries. Prefix root count is zero, checkout count is
zero, user/system site count is zero, former publisher-source-root count is
zero and governed closure count is one. Every governed module origin is below
that single closure.

Installed launch-time admission command:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH = `
  'C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260728-v14.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256 = `
  '6ec9638fade90c2730f90079af794d63452428afe6db830fd22b67fa4702bccf'
& 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v14-archive-locked-base\python.exe' `
  -I -B -m pfc_shaping.cli.governed_release --version
```

Result: exit 0 in 607.3 seconds and
`0.14.0 source_revision=51946b9b60f024c8230d004b87a75fe9646c6c32c7a020f8247facedfd49bcb6`.
Admission occurs before argparse exposes even `--version`.

## Demonstrated test-harness finding and correction

The first complete atomic-promotion matrix reported
`115 passed, 2 skipped, 1 failed`. The failing test intended to restore
`release_cli_contract.assert_production_transition_runtime_authorized`, but the
autouse sealed-runtime fixture had already replaced that attribute with a
lambda. The test therefore did not exercise the real production guard and
reached a later REGISTER/workflow error.

`tests/test_atomic_promotion.py` now captures the real guard at module import,
before fixture monkeypatching, and restores that stable reference in the
negative test. Production code was not changed. The targeted test passed, and
the complete atomic matrix then passed `116 passed, 2 skipped`.

## Verification matrices

- focused Conda lock/runtime/admission:
  `39 passed in 3.32s`;
- final runtime/packaging, including Conda archive lock:
  `143 passed, 12 skipped, 2 deselected in 152.53s`;
- external publication/anchors/bootstrap signer:
  `77 passed in 9.53s`;
- atomic promotion after harness correction:
  `116 passed, 2 skipped in 440.90s`;
- candidate bundle/evidence/assembler:
  `65 passed in 305.55s`;
- governed release/script/monthly-manifest promotion:
  `240 passed in 459.43s`;
- targeted Ruff: `All checks passed!`;
- targeted `git diff --check`: exit 0, only informational LF/CRLF warnings.

No production publication or promotion was exercised by these tests.

## Independent read-only roasts

Final current-v14 Security and IT/Operations verdicts are recorded below after
their independent read-only inspection. Quant/Data independently confirmed
that this packaging slice does not touch CT, solver level authority, OMPEX
status, prospective truth or T057; the current-source I/J wheel addendum is
required before treating the earlier G/H identifiers as v14 evidence.

## Open gates and next work

Production remains strict `NO_GO`.

Packaging/runtime P1/P2 gates:

1. same-user writable target code is imported before self-admission; only an
   independently admitted pre-import supervisor or externally read-only service
   identity can close it;
2. v11 remains physically executable and its documentary revocation needs an
   independent IT quarantine/execute-deny or immutable external allowlist;
3. Conda-prefix creation is not atomic/resumable; v13 is direct evidence;
4. build 863.5 s and per-launch admission 607.3 s need an explicit SLO and an
   independently trusted optimization design;
5. Windows CI, ASR qualification, clean-tree PEP 517, signed SBOM/provenance,
   job-object supervision, structured logs, active-runtime CAS and rollback
   drills remain absent;
6. first verifier attempt5 scratch residue needs formal incident closure, and
   the acquisition needs a separate immutable execution sidecar.

After packaging handoff, return to the scientific/data path without treating
this local runtime as authority:

1. independently admit fresh licensed PIT CH EEX forwards and the retained
   hourly CH acquisition; never use the protected parquet as new evidence;
2. bind exact quote/product conventions and replay EEX repricing, cascade
   invariance and quote-to-curve sensitivity under monthly solver authority;
3. preregister rolling-origin/holdout/T057 rules, dependence/power/MDE and
   one-shot ledger before consuming future truth;
4. build a new auditable CH candidate with coherent probabilities/scenarios,
   then obtain fresh independent Security, IT/Operations and Quant/Data roasts;
5. never promote before independent manifests, external CAS and every gate are
   proven.
