# Session handoff - launcherless runtime and CH acquisition v2 (2026-07-27)

> **REVOKED / SUPERSEDED 2026-07-28.** The v7 local-runtime admission in this
> handoff failed later Security review and must not be used. Preserve it as
> negative evidence only. Current evidence and unresolved gates are in
> `SESSION-HANDOFF-20260728-LAUNCHERLESS-RUNTIME-V12-SECURITY-RECLOSURE.md`
> and decision D-20260728-168.

## Status

- Canonical repository: `C:\Users\jbattaglia\PFC_LT` only.
- Branch: `fix/lt-audit-remediation`.
- HEAD observed: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. No reset, clean, restore, staging
  or commit occurred.
- No production publication, candidate promotion or production promotion
  occurred. Production remains strict `NO_GO`.
- `data/eex_forwards_history.parquet` was not touched or staged; its observed
  SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- No `pfc_shaping/ct/*` or Power BI file was changed.
- Monthly solver remains monthly-level authority. OMPEX remains benchmark-only.

## Security/endpoint decision

The old project console launchers are removed from the package contract. The
endpoint uses no PFC `.exe`, `.cmd`, `.bat` or `.ps1`. All five governed
commands use the absolute admitted Python path with `-I -B -m`; the provider
verifier remains the dedicated `python -I -S -B <verifier.pyz>` artifact.

This followed Defender ASR blocks on unsigned setuptools-generated PFC
launchers. No Defender exclusion or weakening, admin operation, blocked-file
retry/copy/rename/delete, Playwright or browser automation occurred. The exact
ASR rule is not claimed without the Defender 1121/1122 event details.

## Changed files owned by this slice

- `pyproject.toml`
- `pfc_shaping/package_contract.py`
- `scripts/check_lt_wheel_contract.py`
- `scripts/build_launcherless_local_runtime.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `deploy/publisher/README.md`
- `pfc_shaping/cli/audit_ch_lt_compute_runtime.py`
- `pfc_shaping/cli/audit_ch_lt_compute_runtime_manifest.py`
- `pfc_shaping/cli/audit_ch_lt_estimand_contract.py`
- `pfc_shaping/cli/governed_release.py`
- `tests/test_lt_package_contract.py`
- `tests/test_launcherless_local_runtime.py`
- `tests/test_ch_lt_compute_runtime.py`
- `tests/test_ch_lt_compute_runtime_manifest.py`
- `tests/test_ch_lt_estimand_contract.py`
- `tests/test_run_governed_lt_release_script.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff.

Other dirty files predate or belong to adjacent Phase 14 work. Do not infer
ownership of the full worktree from this list.

## Reproducible wheel evidence

Two independent fixed-directory builds produced:

- `build/launcherless-wheel-c-20260727/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- `build/launcherless-wheel-d-20260727/fmv_pfc_lt-0.14.0-py3-none-any.whl`.

Both have 84 members, no `entry_points.txt`, no project launcher, identical
SHA-256
`f0bd93e3b37f98553d184c457c87b53c586ac8eb69bb2869014f956959bcbef3`
and embedded source revision
`fcfa51808ff7116f5b1af24cc98e95bc143af5f9bf8c16644b8d71ee82150c64`.
Both wheel-contract audits returned `PASS`, `promotion_eligible=false`.

## Launcherless runtime evidence

Runtime prefix:
`build/conda-runtime-v7`.

Receipt:
`build/launcherless-runtime-receipt-20260727-v2.json`, SHA-256
`5227f2162b1e285351692c82d04f65b8d1d3d26c88323a4c13c17dcac731c0a6`.

Bound properties:

- CPython `3.11.13`, 64 bit, SHA-256
  `50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`;
- `uv.lock` SHA-256
  `efcea25267644da75c8736b3ede0dfaaf4b6ee8e58b982a61e87edb1064eb5d6`;
- 19 exact distributions, 8,488 files;
- closure tree SHA-256
  `dffb264fbef6b3be33a7ca7714dfdc114f37b8f34f2d08727a222960fd25796d`;
- publisher subclosure tree SHA-256
  `0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`;
- all governed module origins are under
  `build/conda-runtime-v7/governed-site-packages`;
- exact `sys.path` contains the standard-library runtime roots and exactly one
  governed site-packages root; checkout count is zero;
- five module `--help` routes from a foreign cwd all exited `0`;
- local-quality authorisation is true, production authorisation is false.

Seven dependencies came from UV extracted archives. Every file was verified
against wheel `RECORD`; cache reference and HTTP metadata were bound to the
exact wheel hash in `uv.lock`; the sources were revalidated after capture and
copied into new non-hardlinked files. This does not replace a production
retained-wheel/SBOM/signature attestation.

## Real CH acquisition and verifier evidence

Capture retained unchanged:
`output/phase14/ch_da_hourly_capture_20260727_attempt2_curl`.

New acquisition:
`output/phase14/ch_da_hourly_acquisition_20260727_attempt4_launcherless`.

Hashes:

- manifest:
  `c580b0e9472dd281258eb4969ecd10cb414eb6035c5c99df147aeea8c9d7077f`;
- bronze:
  `a4d3da3f02522f96d5e2f757ca781eb62fa0ee2169c994eeb34254e5bbd46fe1`;
- provider parser:
  `b6dc574cebd9521c222b1a7022e61aa520f9c575923535950f2e4217c4d39f89`;
- raw envelope:
  `85afeb0b8140bbfb3675d82e828de794f6e7b31f0dbdb0e78d39187cd111a611`;
- transform config:
  `a3c26206c78fee2ce8945ff1a46f1832266482175ac0669d94ffcfc6f1f2ea7d`.

The manifest is `lt_provider_acquisition_build.v2`. It records 720 native
hourly observations and 2,880 stepwise QH transport rows. Native QH truth is
false; scientific use is `NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY`.

Verifier v14 outputs:

- business audit:
  `output/phase14/ch_da_hourly_acquisition_audit_20260727_attempt4_v14/business-audit.json`,
  SHA-256
  `f6d97d72753679d47d4f1cce5912c58bcaab26342cbb77ce72e68b0ebf60bd7f`;
- runtime receipt:
  `output/phase14/ch_da_hourly_acquisition_audit_20260727_attempt4_v14/runtime-receipt.json`,
  SHA-256
  `0bae2d7b056ed90356ead384afa10b589107dc0161f67e43e6c7c2a9c7eababb`.

The receipt binds the business-audit hash, reports exact captured/source
artifact/dependency counts `1/0/1/0`, has zero `vv-*` scratch residue and keeps
`runtime_authority=false`, `production_authorization=false`.

## Commands and results

Every command was preceded by the canonical cwd/root guard documented in the
previous handoff.

Conda offline creation used:

```powershell
$env:CONDA_OVERRIDE_CUDA='0'
$env:CONDARC='C:\Users\jbattaglia\PFC_LT\build\condarc-runtime-v6.yml'
& 'C:\ProgramData\anaconda3\Scripts\conda.exe' create --offline `
  --prefix 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v7' `
  'python=3.11.13' --yes --json
```

Result: exit `0`, `FETCH=[]`. Conda could not update the user-wide
`environments.txt` under sandbox permissions, but the explicit prefix is
complete and executable.

The closure build used
`python -B -m scripts.build_launcherless_local_runtime` with the exact v7
prefix, wheel C, retained publisher wheelhouse/closure/receipt, UV cache and
`uv.lock`. Result: exit `0` in `252.6s` and the canonical receipt above.

The acquisition used:

```powershell
build\conda-runtime-v7\python.exe -I -B -m `
  pfc_shaping.cli.governed_acquisition_builder `
  --capture-spec <absolute-attempt2-capture-spec.json> `
  --output-directory <absolute-attempt4-launcherless-directory>
```

Result: exit `0` in `3.1s`.

The verifier used `python -I -S -B build/provider-verifier-20260727-v14.pyz
audit-acquisition` with the retained publisher dependency root and a new v2
scratch root. Result: exit `0` in `129.8s`; both outputs and zero residue were
verified independently.

Final tests:

- targeted Ruff: `All checks passed!`;
- launcherless/runtime focused:
  `211 passed in 18.47s`;
- runtime/packaging:
  `116 passed, 12 skipped, 2 deselected in 67.55s`;
- publication/CAS/candidate:
  `500 passed, 2 skipped in 386.08s`.

## Failed/rejected attempts retained as evidence

- Pip wheel build failed before wheel creation because the Python-created
  temporary directory inherited a defective Windows ACL. No escalation was
  requested.
- The first direct build-meta call used the wrong `SOURCE_DATE_EPOCH`; the
  package contract rejected it. Its Python-created temporary also exposed the
  ACL defect. Fixed-directory `setup.py build ... bdist_wheel` was then used
  only to build wheels, never to install the project.
- `build/conda-runtime-v6` timed out after copying 8,488 files. It has no
  `python311._pth` and no receipt and is rejected. It was not deleted or
  reused. The clean v7 namespace completed and was the only prefix admitted by
  this now-revoked session; D-20260728-168 later revoked it.
- Acquisition attempt2 is legacy v1 and was correctly rejected by verifier
  v14. Attempt3 was never created because the ambient `ppa_env` dependency
  versions did not match the package contract.

## Remaining gates and next work

- Await and record final read-only Security and IT/Operations re-roasts of the
  launcherless slice; correct only demonstrated findings and rerun affected
  matrices.
- Add the local runtime builder to a reproducible CI/build job using retained
  original wheels rather than UV extracted-cache fallback.
- Obtain external Python/base image, wheelhouse, SBOM, vulnerability scan and
  release signatures; qualify the absolute Python module commands with FMV
  Security/IT. Do not request Defender exceptions.
- Complete independent service identity, ACL/WORM/external-CAS, monitoring,
  timeout, backup/restore and rollback drills.
- Return to fresh governed CH EEX forwards and prospective CH hourly truth,
  then execute the successor preregistered rolling-origin/future-holdout
  programme. T057 and the run15 fixture-backed candidate remain scientific
  `NO_GO` and must not be reused as confirmatory evidence.
