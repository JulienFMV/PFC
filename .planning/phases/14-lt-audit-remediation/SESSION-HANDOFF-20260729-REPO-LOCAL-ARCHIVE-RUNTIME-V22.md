# Session Handoff - Repo-local archive runtime v22

Date: 2026-07-29

Branch: `fix/lt-audit-remediation`

Observed HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`

Workspace: `C:\Users\jbattaglia\PFC_LT`

Production: strict `NO_GO`

## Outcome

Runtime v22 is the sole current launcherless runtime for local quality
execution. It contains the exact reproducible W/X wheel bytes and was built
from a Conda lock, retained archives and explicit spec whose mutable payload
roots are all below repo `build/`. It requires no administrator right, network,
Defender/ASR exception, project executable or Playwright.

Security, IT/Operations and Quant/Data independently roasted v22 in read-only
mode. After the runbook correction all report local P0/P1 = 0 and accept a
limited local-quality GO. This is not production or scientific authority.
Import-before-self-admission and same-user mutation remain production P1s that
require an independently admitted pre-import supervisor or a read-only service
identity. External signature, CAS/WORM/HEAD, official EEX semantics, rolling-
origin/T057 and a new CH candidate also remain open.

## Workspace and protected state

Before every shell action the session ran the exact guard:

```powershell
$expected='C:\Users\jbattaglia\PFC_LT'; $cwd=(Get-Location).Path; $root=(git rev-parse --show-toplevel).Trim().Replace('/','\'); if ($cwd -cne $expected -or $root -cne $expected) { throw "Workspace mismatch: cwd=$cwd root=$root" }; Write-Output "WORKSPACE_OK=$root"
```

Every guard returned
`WORKSPACE_OK=C:\Users\jbattaglia\PFC_LT`. The worktree remains intentionally
very dirty. Nothing was reset, cleaned, restored, staged, committed or
promoted. `data/eex_forwards_history.parquet` was not touched by this slice.
No CT or Power BI path was changed.

## Retained negative evidence

- V20 prefix:
  `build/conda-runtime-v20-eex-handoff-base`.
  `PYTHONPYCACHEPREFIX` was set during Conda creation and displaced 406
  generated pip bytecode files outside the prefix. Manifest SHA-256
  `6116efe4a87d18036db204afc5b79e4adcc4df7d9a171b4d0f2ceece9458aca2`;
  prefix audit failed. Preserve unchanged.
- V21 prefix:
  `build/conda-runtime-v21-eex-handoff-base`.
  Its runtime was internally coherent, but the reused archive lock/spec named
  payload paths under `C:\Users\jbattaglia\.conda\pkgs`. Security classified
  this as a local P1 against the repo-root contract. Preserve unchanged and do
  not select it.
- The first v21 assembly attempt through the ppa interpreter failed before
  mutation because that interpreter lacks `zstandard`; terminal receipt SHA
  `b35d8d7f5d4803e274689b76dd4290747816dd35a144f076b2262c90809f7cd5`.

## V22 archive and prefix evidence

All 19 locked archives already existed under
`build/conda-pkgs-runtime-v6`; validation found zero missing and zero hash
mismatch. The new lock was constructed from original HTTPS package metadata
and those repo-local payload bytes only. A first attempted rebuild using v21
Conda metadata failed before output because its `file://` URLs were not valid
original source metadata; no v22 lock was written by that failed attempt.

Canonical evidence:

- lock:
  `build/launcherless-conda-archive-lock-20260729-v22.json`
  - SHA-256
    `020735fa21744772aedd71a7c99b33775ee27042c9a6c2dd953b15b6b9b720d8`
  - lock ID
    `7fd90fd7a1a2f672da1890cb308870f5c7d1df5e38dbafe91f6da27ca486a813`
  - archive set ID
    `f3cd775e79648df9a9926a01eb97eadc8e951c5055c778ca9ca92b60bc8068e7`
  - 19 packages / 38,648,096 bytes;
- explicit spec:
  `build/launcherless-conda-explicit-20260729-v22.txt`, SHA-256
  `88266ae90c163470a9bcca09d4ef043bde2c33d5b8446f6536ff2df8cedabd46`;
- build-only Conda config:
  `build/condarc-runtime-v22.yml`, offline and repo-local;
- fresh prefix:
  `build/conda-runtime-v22-repolocal-archive-base`;
- pre-first-execution manifest:
  `build/launcherless-python-runtime-manifest-20260729-v22-base.json`,
  SHA-256
  `a3b12fe143a5af1f2e6d0db5a57308ea9596a568b187e15d5fd08fc241e1bfc7`,
  6,285 files, tree
  `c4b7d86442a455a7cd602eeb333e882f0e3f89886265f83074dae6bd5232ab65`;
- prefix receipt:
  `build/launcherless-conda-prefix-receipt-20260729-v22.json`, SHA-256
  `8155d0878a669437a91072ce71f4083b14bd33be31846e0df0c7c832776db571`,
  19 packages, 5,859 archive-verified files plus 406 declared generated
  non-runtime files, production false.

Conda created the prefix in 106.2 seconds with `--offline --copy` and the v22
spec. `CONDA_PKGS_DIRS`, `CONDA_ENVS_PATH`, `TEMP` and `TMP` were repo-local.
No `PYTHONPYCACHEPREFIX` was set during prefix creation. The only external
executable was the existing Anaconda/Conda tool under `ProgramData`, used
read-only; no payload, environment, cache or output was created there.

Explicit inspections reported zero `.conda`, `AppData` or `ProgramData`
archive/cache reference in the v22 lock/spec. All 19 lock cache roots and all
archive/file URIs resolve to `build/conda-pkgs-runtime-v6`.

## Wheel and runtime evidence

The two retained wheels are byte-identical:

- `build/wheel-dist-w/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- `build/wheel-dist-x/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- SHA-256
  `07b8228426c2857b30682228181245a7d2367cb31add87a1580f54388ce3b136`;
- 86 members / 483,369 bytes;
- embedded source revision
  `691139df0d2b941823d9c80c3825440a28d1af1d3095ae50f6330a23c130f15e`;
- both wheel audits PASS with `promotion_eligible=false`.

Runtime assembly used the read-only preinstalled Anaconda interpreter because
the ppa interpreter has no `zstandard`. All mutable environment paths remained
under `build/`. Assembly completed in 639.1 seconds:

- receipt:
  `build/launcherless-runtime-receipt-20260729-v22.json`;
- SHA-256
  `2e45ce409c027395b38096ab5718425d459917c83b66910eb7ddbf13e1d766bf`;
- 8,490 files / 19 distributions;
- closure tree
  `6fec62264ce247e249acd1c63cf9119048decf6cfb29d0c5bb05860ae25093e8`;
- exact `sys.path`, in order:
  1. `...\Lib`
  2. `...\DLLs`
  3. `...\governed-site-packages`
- prefix root, checkout, user site, system site and phantom
  `python311.zip` are absent;
- `local_quality_authorization=true`;
- `production_authorization=false`.

## Installed probes and matrix

Installed admission command:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH='C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260729-v22.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256='2e45ce409c027395b38096ab5718425d459917c83b66910eb7ddbf13e1d766bf'
$env:TEMP='C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v22-temp'; $env:TMP=$env:TEMP
& 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v22-repolocal-archive-base\python.exe' -I -B -m pfc_shaping.cli.governed_release --version
```

Result: exit 0 in 22.8 seconds; version 0.14.0 and exact source revision.

Installed fail-closed EEX probe:

```powershell
& 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v22-repolocal-archive-base\python.exe' -I -B -m pfc_shaping.cli.eex_forward_vintage_builder --runtime-receipt 'C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260729-v22.json' --expected-runtime-receipt-sha256 '2e45ce409c027395b38096ab5718425d459917c83b66910eb7ddbf13e1d766bf' --intake-spec 'C:\Users\jbattaglia\PFC_LT\build\runtime-v22-eex-probe\missing-spec.json' --expected-spec-sha256 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' --source-document 'C:\Users\jbattaglia\PFC_LT\build\runtime-v22-eex-probe\missing-source.xlsx' --trusted-time-receipt 'C:\Users\jbattaglia\PFC_LT\build\runtime-v22-eex-probe\missing-time.json' --trusted-time-public-key 'C:\Users\jbattaglia\PFC_LT\build\runtime-v22-eex-probe\missing-time.pem' --trusted-time-journal-id 'runtime-v22-negative-probe' --output-directory 'C:\Users\jbattaglia\PFC_LT\build\runtime-v22-eex-probe\output'
```

Result: exit 50 in 21.9 seconds, `EEX trusted-time public key is unavailable`,
`catalog_signed=false`, `external_cas_admitted=false`,
`production_authorization=false`; output directory absent.

Runtime/packaging matrix command:

```powershell
& 'C:\Users\jbattaglia\.conda\ppa_env\python.exe' -B -m scripts.run_workspace_local --run-id runtime6 -- 'C:\Users\jbattaglia\.conda\ppa_env\python.exe' -B -m pytest tests\test_lt_provider_verifier_artifact.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_runtime_closure.py tests\test_lt_package_contract.py tests\test_launcherless_conda_archive_lock.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py tests\test_run_workspace_local_script.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_audit_legacy_provider_resolution_script.py -q -p no:cacheprovider -m 'not slow'
```

Result: `180 passed, 12 skipped, 2 deselected in 103.59s`.
Receipt:
`build/workspace-local-runs/runtime6/execution-receipt.json`, SHA-256
`c05f80b7569413b60d6a1cf88f9d40daf06ada7a869fb81a6b2f5850f4339198`,
status `TARGET_EXIT_ZERO_NOT_AUTHORITY`; every mutable path is below `build/`
and all scientific/evaluation/runtime/promotion/production authorities are
false.

## Independent roasts

Security:

- local P0/P1/P2 = `0/0/0`;
- confirmed 19/19 payloads, paths, sizes and hashes under the repo-local root,
  coherent lock/spec/prefix/runtime bindings, exact `sys.path`, and EEX
  fail-closed behavior;
- local sealed unsigned/quarantined runtime GO; production `NO_GO`.

IT/Operations, after runbook correction:

- local P0/P1 = `0/0`, local-quality GO;
- confirms no mutable `.conda`, `AppData`, `ProgramData`, admin, Defender,
  project executable or Playwright dependency;
- retained P2s: stdout/stderr, counts and runner/interpreter hashes are not
  embedded in receipts; installed probes are terminal-only; `module_origins`
  does not separately enumerate `eex_forward_vintage_builder` despite exact
  wheel/closure binding and the installed probe;
- production `NO_GO`.

Quant/Data:

- local delta P0/P1/P2 = `0/0/0`;
- v21/v22 have the same wheel, source revision, 8,490-file executable closure
  and closure tree; 82/82 `pfc_shaping` members are installed byte-identically;
- zero CT, T057, OMPEX/HFC or heavy data payload; no model, monthly authority,
  data, T057 or candidate CH change;
- packaging GO only, with no scientific or production inference.

## Documentation changed in this closure

- `pfc_shaping/tools/OPERATIONS.md` selects v22, marks v21 negative and binds
  repo-local archive/spec requirements and v22 hashes.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` adds D178.
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
  adds the v22 amendment.
- `.planning/HANDOFF.md` points to this handoff and states current v22 status.

## Residual blockers and next direction

Local packaging closure does not authorize production. Required next controls:

- pre-import supervisor or read-only service identity and protection against
  same-user mutation;
- independently signed provenance and builder-inaccessible CAS/WORM/HEAD;
- atomic prefix construction, kill recovery, active-runtime CAS and rollback;
- Windows CI/ASR qualification, two-builder reproduction, SBOM/license/
  vulnerability scans, process supervision, structured logs and SLOs;
- provider-authenticated EEX product semantics and fresh prospective PIT bytes;
- rolling-origin evidence and the sealed future T057 one-shot holdout;
- a new CH candidate that preserves monthly solver authority and demonstrates
  exact final EEX repricing, calibrated uncertainty and auditable shaping.

Do not promote production. Continue to keep monthly solver authority, LT/CT
separation, OMPEX benchmark-only status and protected data invariants.
