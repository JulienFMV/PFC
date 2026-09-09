# Session handoff - launcherless runtime v12 security reclosure (2026-07-28)

## Status and perimeter

- Canonical repository only: `C:\Users\jbattaglia\PFC_LT`.
- Branch: `fix/lt-audit-remediation`.
- HEAD observed throughout:
  `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. No reset, clean, restore, staging,
  commit or production promotion occurred.
- `data/eex_forwards_history.parquet` was not modified or staged. Its observed
  SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- No `pfc_shaping/ct/*`, Power BI, Defender policy, admin setting or Playwright
  surface was touched.
- Monthly solver remains the monthly-level authority. OMPEX remains
  benchmark-only.
- Production remains strict `NO_GO`.

## Supersession and revocation

Runtime v7 is revoked. Its Security roast demonstrated mutable extracted-cache
provenance, `.` in `sys.path` and verification/import TOCTOU. Runtime v9 is also
revoked because its closure inherited an unreadable ACL and the independent
probe failed with `ModuleNotFoundError`. V8 and v10 are incomplete Conda
prefixes. Preserve all four as negative evidence and never launch, repair,
delete, relabel or promote them.

Runtime v11 closed the original-wheel, root-in-`sys.path`, full-prefix manifest,
staging-resume and launch-time replay findings. Its receipt SHA-256 was
`d0c06d62321c0aaf9022536373f78aebc889ada91912a417f3faa6a791380f18`.
The final Security and IT/Operations roasts nevertheless demonstrated that a
local receipt with `production_authorization=false` could satisfy the runtime
precondition used by `promote_candidate` and rollback. V11 is therefore
superseded by v12 and must not be used for transitions.

IT/Operations correctly notes that this is not a technical revocation. The
preserved v11 prefix, interpreter and receipt remain executable by the
workstation user and still contain the superseded transition check. Do not
delete, move, repair or ACL-edit this evidence from the user-space project.
Independent IT quarantine/execute denial or an immutable external runtime
allowlist is required before v11 revocation is operationally effective.

## Demonstrated product fix

`pfc_shaping.pipeline.governed_release_cli_contract` now exposes a separate
`assert_production_transition_runtime_authorized()` guard. It first requires
the exact local runtime admission, then unconditionally rejects because the
caller-held local receipt is not an independently signed, IT-admitted
production capability. `atomic_promotion.promote_candidate()` and
`rollback_to_candidate()` call that guard before path resolution or mutation.

The local unit fixture stubs this new capability only for tests of promotion
mechanics. Tests that exercise runtime rejection explicitly restore the real
guard. The two-process test also stubs it inside its isolated child harness;
production code is never relaxed.

`scripts/build_launcherless_local_runtime.py` now reads `python311._pth` as a
stable mono-linked file, compares its exact fixed payload and hashes those
captured bytes. It then replays the closure once more before receipt emission.

## Files changed in this closure

Product, builder and package contract:

- `pfc_shaping/pipeline/governed_release_cli_contract.py`;
- `pfc_shaping/pipeline/atomic_promotion.py`;
- `pfc_shaping/pipeline/governed_release.py`;
- `pfc_shaping/cli/governed_release.py`;
- `scripts/build_launcherless_local_runtime.py`;
- `PACKAGE.md`.

Regression tests:

- `tests/conftest.py`;
- `tests/test_atomic_promotion.py`;
- `tests/test_launcherless_local_runtime.py`;
- `tests/test_launcherless_runtime_admission.py`;
- `tests/test_run_governed_lt_release_script.py`.

Governance and operations records:

- `.planning/HANDOFF.md`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`;
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260727-LAUNCHERLESS-RUNTIME-CH-ACQUISITION.md`;
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260728-LAUNCHERLESS-RUNTIME-V12-SECURITY-RECLOSURE.md`;
- `pfc_shaping/tools/OPERATIONS.md`.

## Reproducible wheels G/H

Exact build commands, run twice into new directories:

```powershell
$env:SOURCE_DATE_EPOCH='1783987200'
python setup.py build --build-base build\wheel-build-g `
  bdist_wheel --dist-dir build\wheel-dist-g

$env:SOURCE_DATE_EPOCH='1783987200'
python setup.py build --build-base build\wheel-build-h `
  bdist_wheel --dist-dir build\wheel-dist-h
```

Both wheels:

- path: `build/wheel-dist-{g,h}/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- byte-identical SHA-256:
  `a461403b7db0a37fb8ef570e8a5fb698aa972c5b86879c9eeea3ff3e74aff6c5`;
- 84 members;
- embedded source revision:
  `3784cf5b3ddfe50d6249f213660514b0de31d0328220b4bf9d2fac156fc3c764`;
- both wheel audits: `PASS`, `promotion_eligible=false`.

The `setup.py` deprecation warning is retained. This path is a local fixed-dir
fallback to avoid the known Windows temporary-directory ACL defect, not the
future CI build design.

## Runtime v12 construction and evidence

Conda base creation was user-space, offline and copy-only:

```powershell
$env:CONDA_OVERRIDE_CUDA='0'
$env:CONDARC='C:\Users\jbattaglia\PFC_LT\build\condarc-runtime-v6.yml'
& 'C:\ProgramData\anaconda3\Scripts\conda.exe' create --offline --copy `
  --prefix 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v12' `
  'python=3.11.13' --yes --json
```

Result: `success=true`, `FETCH=[]`. The nonfatal warning about the sandboxed
user-wide `.conda/environments.txt` remains. No admin or system-environment
mutation occurred.

Before the first target-Python execution:

```powershell
python -B -m scripts.build_launcherless_python_runtime_manifest `
  --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v12 `
  --output C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260728-v12.json
```

Caller-held Python manifest:

- SHA-256:
  `54a7eeb6519d9eb4056efc36e4a0cd0039dc47a0745978c4667dd275e8494e6e`;
- 6,285 files;
- tree SHA-256:
  `985d82b94f8890af53219f793a0374d1252aa3873c8cde911b4900870554db4b`;
- CPython 3.11.13 executable SHA-256:
  `50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`.

The closure builder used the exact v12 prefix, wheel G, the retained publisher
wheelhouse/root/receipt, `build/launcherless-wheelhouse-20260727-v1`, the
manifest and hash above, and `uv.lock`:

```powershell
python -B -m scripts.build_launcherless_local_runtime `
  --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v12 `
  --project-wheel C:\Users\jbattaglia\PFC_LT\build\wheel-dist-g\fmv_pfc_lt-0.14.0-py3-none-any.whl `
  --publisher-wheelhouse C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252 `
  --publisher-dependency-root C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages `
  --publisher-receipt C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json `
  --additional-wheel-directory C:\Users\jbattaglia\PFC_LT\build\launcherless-wheelhouse-20260727-v1 `
  --python-runtime-manifest C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260728-v12.json `
  --expected-python-runtime-manifest-sha256 54a7eeb6519d9eb4056efc36e4a0cd0039dc47a0745978c4667dd275e8494e6e `
  --receipt-output C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260728-v12.json `
  --lock-path C:\Users\jbattaglia\PFC_LT\uv.lock
```

Operational anomaly retained: the supervising shell timed out at 904.1 s
after closure and `_pth` publication but before observing the result. The child
finished later and emitted the receipt. An overlapping exact retry correctly
failed once on a transient prefix-manifest divergence; after all child
processes ended, an independent 6,285-file comparison found zero changed,
missing or extra base files and the exact original tree. A later exact retry
refused the already-existing receipt. This proves byte integrity but also
demonstrates that job supervision/orphan handling is not yet production-ready.

Runtime receipt:

- path: `build/launcherless-runtime-receipt-20260728-v12.json`;
- SHA-256:
  `2050b2a6b84ea941f7a4b4609029c1f55903db56e83f1f7d5b976dea0f3a316f`;
- schema/status: `fmv_lt_launcherless_local_runtime.v2` / `PASS`;
- closure: 19 distributions, 8,488 files, tree SHA-256
  `c041dca7ad37def239b83d775fe59572aadd92035b51a273ae984490f350c93e`;
- `python311._pth` SHA-256:
  `bc068376ebd9405b9a7db3894a78d685fecbf9508d6f5c028f53d44232cf8bdf`;
- `sys.path`: `python311.zip`, `Lib`, `DLLs`, and exactly one
  `governed-site-packages` root;
- prefix-root count `0`, closure count `1`, project-launcher count `0`;
- no closure staging residue;
- `local_quality_authorization=true`;
- `production_authorization=false`.

An independent installed-runtime admission passed in 114.5 s. A real installed
wheel probe then called both public transition APIs with deliberately invalid
relative paths and returned:

```text
PROMOTE_LOCAL_RUNTIME_REJECTED=PASS
ROLLBACK_LOCAL_RUNTIME_REJECTED=PASS
```

The rejection occurred before path I/O.

## Verification matrices

Focused correction suite:

```powershell
python -m pytest tests\test_launcherless_local_runtime.py `
  tests\test_launcherless_runtime_admission.py `
  tests\test_atomic_promotion.py::test_two_process_promotions_on_same_expected_head_commit_exactly_one `
  -q -p no:cacheprovider
```

Result after the initial edit: `24 passed`. The two final sensitive nodes
(unsealed checkout and two-process race) then passed `2 passed`.

Final runtime/packaging matrix:

```powershell
python -m pytest tests\test_lt_provider_verifier_artifact.py `
  tests\test_snapshot_publisher_artifact.py `
  tests\test_snapshot_publisher_runtime_closure.py `
  tests\test_lt_package_contract.py `
  tests\test_audit_provider_acquisition_quarantine_script.py `
  tests\test_audit_legacy_provider_resolution_script.py `
  tests\test_launcherless_local_runtime.py `
  tests\test_launcherless_runtime_admission.py `
  -q -p no:cacheprovider -m "not slow"
```

Result: `127 passed, 12 skipped, 2 deselected in 155.31s`.

Final publication/CAS/candidate inventory was split into the same four
wall-time-safe groups:

- external publication/anchors/bootstrap signer: `77 passed in 10.28s`;
- atomic promotion: `116 passed, 2 skipped in 443.36s`;
- candidate bundle/evidence/assembler: `65 passed in 307.02s`;
- governed release/script/monthly manifests: `242 passed in 467.80s`.

Aggregate: `500 passed, 2 skipped`. Targeted Ruff passed.

## CH acquisition evidence retained

The real capture remains
`output/phase14/ch_da_hourly_capture_20260727_attempt2_curl`. V11 produced
`output/phase14/ch_da_hourly_acquisition_20260728_attempt5_launcherless_v11`.
Its deterministic manifest remains byte-identical to attempt4 at SHA-256
`c580b0e9472dd281258eb4969ecd10cb414eb6035c5c99df147aeea8c9d7077f`:
720 native hourly observations and 2,880 stepwise quarter-hour proxy rows,
never native quarter-hour truth.

Verifier retry3 evidence is retained under
`output/phase14/ch_da_hourly_acquisition_audit_20260728_attempt5_v14_retry3`:

- business audit SHA-256
  `2af291e944752822124e8f39d9e2cf3e4d9e491f3b99f607bf84595109803f09`;
- runtime receipt SHA-256
  `4781d62fe08fe5585489961ffda5db206cac93d21b10dc46c256a439032687ff`;
- status `VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION`;
- exact captured/source artifact and dependency path counts `1/0/1/0`;
- `runtime_authority=false`, `production_authorization=false`.

The deterministic acquisition manifest intentionally contains exact
capture/parser/config/bronze bytes but does not attribute the execution to the
v11 receipt, Python manifest, command or exit code. A separate immutable
execution sidecar is still required; do not alter the deterministic manifest
to hide this gap.

## Independent roast verdict and open P1 gates

Security v11/v12 re-roast: no P0. The product P1 allowing a local receipt to
cross the atomic promote/rollback runtime gate is closed and proven in v12.
The stable `_pth` re-read P2 is also closed. This closure is artifact-specific;
the executable v11 residue keeps operational revocation open as a P1.

IT/Operations: no P0, but production remains `NO_GO` because:

1. v11 remains physically executable and no independent IT quarantine or
   immutable runtime allowlist technically enforces its revocation;
2. the `-m pfc_shaping.cli.*` target imports code from the mutable closure
   before that same code performs admission; same-user writable ACLs mean the
   dependency-verification-before-import boundary is not independently trusted;
3. the Conda base is post-hoc manifested rather than built from retained exact
   archive/build/channel/SHA locks;
4. exact recovery covers the closure staging, not atomic Conda-prefix creation;
5. the v12 shell timeout demonstrated incomplete parent/child supervision;
6. the main launcherless runtime still lacks Windows CI, Docker/base-digest,
   ASR qualification, logs/SLO, active-runtime CAS and rollback drills;
7. the first verifier attempt5 scratch residue needs a formal incident and IT
   quarantine/closure; retry3 does not erase that history;
8. attempt5 still needs a separate execution-provenance sidecar.

`python311.zip` is named first in the current `_pth` but does not exist. Adding
it would be detected by post-import prefix replay, but it is another concrete
reason the current same-user pre-import boundary is not production authority.
Do not claim the packaging phase closed until an independently admitted
bootstrap/supervisor or external read-only service identity prevents execution
of altered bytes before verification.

## Post-roast source delta (not embedded in runtime v12)

The current checkout now invokes the production-transition guard at all three
public layers: CLI parsing, high-level governed-release functions and atomic
promotion/rollback. CLI and high-level rejection occurs before failure-root,
release-root or authorization-path I/O. The targeted regression matrix reports
`164 passed in 147.29s`. Two obsolete parametrized assertions were narrowed to
finalize/register/audit because promote/rollback now have their own stronger
production-capability tests.

Post-delta verification is green:

- runtime/packaging: `130 passed, 12 skipped, 2 deselected in 158.76s`;
- governed release/CLI/monthly-manifest promotion: `240 passed in 477.93s`.

The earlier four-group `500 passed, 2 skipped` inventory remains the artifact
v12 baseline. The current affected-group result is reported separately because
two obsolete transition cases moved out of that group into the strengthened
runtime-admission suite.

Runtime v12 was built before this outer-layer change. Its installed probe is
valid evidence only for the atomic public APIs. Do not relabel v12; a fresh
reproducible wheel/runtime is required after the independent pre-import design
is implemented.

## Next work

1. Have IT technically quarantine/deny execution of v11 or enforce an
   immutable external runtime allowlist without changing retained evidence.
2. Design the independent pre-import supervisor/service-identity boundary;
   do not simulate trust with another caller-controlled path/hash pair.
3. Add an exact Conda archive lock and atomically staged base-prefix builder,
   with kill/recovery tests and structured logs.
4. Produce the attempt5 execution sidecar and formal verifier-residue incident
   record without deleting or rewriting existing evidence.
5. Only after packaging reclosure, return to fresh prospective CH EEX/spot
   acquisition, successor rolling-origin/future holdout and a new auditable CH
   candidate. T057 and run15 remain scientific `NO_GO`.
