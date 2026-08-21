# Session handoff - PEAK/OFFPEAK repricing and launcherless runtime v34

Date: 2026-07-30  
Branch: `fix/lt-audit-remediation`  
HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production: strict `NO_GO`

## Outcome

The demonstrated nested-PEAK weighting defect is fixed. PEAK residual targets
are now derived from contractual PEAK delivery intervals, not all BASE hours.
The fresh local EEX audit covers BASE, PEAK and implied OFFPEAK mechanics while
keeping final delivered-candidate repricing, source semantics and every
production authority open.

Runtime v34 packages the exact corrected LT module. Its prefix and every
mutable build path are below `C:\Users\jbattaglia\PFC_LT\build`; no admin,
Defender/ASR exception, AppData mutation, project executable or Playwright was
used. The external Anaconda interpreters were read-only bootstrap processes.
Final audit and test evidence does not execute those interpreters: it targets
the captured repo-local `build/pytest-runtime-v1/python.exe` only.

## Demonstrated defect and correction

Before the correction, a synthetic Q1 2030 system with nested January PEAK
could report an internal constraint residual of `8.38e-13` while the raw Q1
PEAK quote missed by `0.22990282685543662` EUR/MWh. Root cause:
`calibration_buckets(local, peak_prices)` used all delivery hours before the
PEAK mask was applied.

`build_base_peak_offpeak_constraint_system` now calls
`calibration_buckets(local.loc[peak_mask], peak_prices)` and reindexes the
bucket labels onto the full horizon. The OFFPEAK complement set is constructed
with one precomputed PEAK index set. The regression spans the 23-hour Swiss
spring DST transition and checks raw Q1/January BASE, PEAK and implied
OFFPEAK means.

## Current EEX mechanical evidence

Selected source:

- registry SHA-256:
  `5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778`;
- source SHA-256:
  `fb71338f51334128878526877b802e48819b555639913a786a35d710a6b151e5`;
- quote identity commitment:
  `74ba58f6d00c8734ea668d487c8fb48d6e12e35642045be7a2c834daddbdfc95`;
- quote value commitment:
  `399d524c2c002b55bf844a5410791d0fa33ba7fc4ed843896f01b26769ece258`.

July 2026 is excluded because delivery had started. Over 2026-08..2032-12:

- 56,281 hourly intervals;
- 34 independent rows, rank 34: 17 PEAK and 17 implied OFFPEAK;
- active-row max error: `1.5234036254696548e-11` EUR/MWh;
- active PEAK max: `2.1742607714259066e-12`;
- active shared implied OFFPEAK max: `7.87281351222191e-12`;
- tolerance: `1e-9`.

The redundant parents `2026-Q4` and `2027` remain conflicts. Maximum
all-source errors are BASE `0.0017745586238220312`, PEAK
`0.0031034482764908944`, implied OFFPEAK `0.003105151728362898` EUR/MWh.
No tick/rounding inference is made. Explicit OFFPEAK quotes are absent.

Fresh audit:

- canonical closure-bound run `cl35au1`;
- report ID
  `aead7df174d501307b0840a26b650414be94bac3bbfcf69903af038ae77527bf`;
- execution receipt SHA-256
  `65dc1d46e82c669c4cb47f5b62c448c5fbfac1981caa069224e5922346c28022`;
- stdout SHA-256
  `29c7638d3824532d25a2d037a71c446673d877587611adbf998ef441b64969fd`.

## Reproducible wheels and runtime v34

Wheels:

- `build/wheel-dist-peak34a/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- `build/wheel-dist-peak34b/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- 99 members, 547,965 bytes, byte-identical SHA-256
  `cfd9a8db3c7a05154dfac45b473201aa29e0069c7e3e005a8fa7acc2e6ae4388`;
- source revision
  `545fe6d9be82aba0069766685a0165de7b6fea20d487dd2d29cbd345feb2834c`;
- wheel-contract receipts `whl34pa`/`whl34pb`:
  `cf9ccb75adb0788b57dc14d16b60ec076411a964bc787bf9734a01c12b1e7914`
  and
  `24caf7820e80a776f8add3554a1dc997771b8ba805e4207c95fb5c9d5dc2059b`.

Runtime:

- prefix `build/conda-runtime-v34-peak-base`;
- Python manifest SHA-256
  `f50d83fab7cd3ceff0f2d303990e682f47c55e590fa1b55bbdf35f8be100a463`,
  6,285 files, tree
  `ca27db7083a959f614c2c6c032c22c27409e6d33ac7e5e9a56cf1bc899d28a72`;
- Conda prefix receipt SHA-256
  `bd348f5145963bfc3ae4c157b723807d52c5a24adbd6a98589f2b1b3d891a233`,
  prefix ID
  `55d88614db56e6f30a7a115d61a5e70aeb59745075020f034c352134c4f777fb`;
- runtime receipt SHA-256
  `8f6f36e36a2707b1c71ff98cce20691e8c648bdda02eeb5f6ffa1335c8c4d405`;
- closure: 8,503 files, 19 distributions, tree
  `d51e3a4c333d3dd64b78a4319564201842703e662569bf649f2df9333875f787`;
- exact `sys.path`: runtime `Lib`, runtime `DLLs`, then exactly one
  `governed-site-packages`;
- installed shape module SHA-256 equals source:
  `9298534b42d7fde3c867394a237c4b4665dd15273bde56568e5077eaf27798ad`;
- `local_quality_authorization=true`, `production_authorization=false`.

Runner receipts:

- `man34pk`:
  `cbd7fbf6316e856ccf272b0167149af7d1abfbfbc77cf98d3baf8caab4bde64a`;
- `aud34pk`:
  `8fa4d89ae44c6bd796129e0691fd70e24398ad28468eba645a7ba98ad76bf13a`;
- `asm34pk`:
  `984bb2fe99843dbfa05cdb2f8bceba2264fe4ad90a573a5bf12b58a5fa08db3f`;
- installed smoke `pk34sm1`, `1 passed`:
  `ca863510dadf50e16feb347bf4a6ffc493e136e935bc57983d834cd697a193e8`.

The initial parent wait for `asm34pk` timed out at 300 seconds. No retry was
started. The single child continued, was observed progressing, then
terminalized its original runner receipt at exit zero. Final residual process
count was zero.

## Repo-local pytest harness

Final workstation tests use only
`build/pytest-runtime-v1/python.exe`. Its `python311._pth` resolves the
repo-local runtime `Lib` and `DLLs`, the captured `test-site-packages`, the
workspace source root, and runtime `governed-site-packages`, in that exact
order. Receipt schema v5 content-hashes every import root plus resolved module
origins before and after target execution. The launcher contains 1,429 files /
45,398,809 bytes with canonical tree SHA-256
`49f0bfef8cf6751e0826e80da6402fd08b9ad7adb315e74878ec359029af97e0`;
`python311._pth` SHA-256 is
`c6fadd6c79ecb9cf96ed3092b22cfbbc5ba8eb71363b9c3dc7c5dec50ad74d96`.
The import-root inventories and their durable digest algorithm are recorded
in D193 and in every `cl35*` execution receipt.

Pytest and its direct closure were captured byte-for-byte from the existing
user Conda environment. Packaging collection then failed closed on missing
`python-dotenv` (`rl34pk1`) and `requests` (`rl34pk2`). Those packages and the
`requests` transitives `certifi`, `charset-normalizer`, `idna` and `urllib3`
were copied to `build/pytest-runtime-v1/test-site-packages`; every copied file
had equal source and destination SHA-256. The external environment was read
only and was neither executed nor mutated for canonical final evidence.
Security and IT/Operations subsequently demonstrated that early `rl34*` runs
predated the frozen closure and did not bind it. Receipt v5 and the complete
`cl35*` replay supersede those runs. The prose-only `4f3ae5...` digest is
retired.

Exact canonical commands, after the mandatory cwd/Git-root guard:

```powershell
build\pytest-runtime-v1\python.exe -I -B -m scripts.run_workspace_local --run-id cl35u1 -- build\pytest-runtime-v1\python.exe -I -B -m pytest tests\test_run_workspace_local_script.py -q -p no:cacheprovider

build\pytest-runtime-v1\python.exe -I -B -m scripts.run_workspace_local --run-id cl35t1 -- build\pytest-runtime-v1\python.exe -I -B -m pytest tests\test_lt_quant_contract_matrix.py tests\test_peak_offpeak_installed_v34.py -q -p no:cacheprovider

build\pytest-runtime-v1\python.exe -I -B -m scripts.run_workspace_local --run-id cl35au1 -- build\pytest-runtime-v1\python.exe -I -B -m scripts.audit_ch_eex_current_repricing_sensitivity --registry .planning\phases\14-lt-audit-remediation\CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json --expected-registry-sha256 5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778

build\pytest-runtime-v1\python.exe -I -B -m scripts.run_workspace_local --run-id cl35mx1 -- build\pytest-runtime-v1\python.exe -I -B -m pytest tests\test_audit_ch_eex_current_repricing_sensitivity_script.py tests\test_audit_ch_eex_current_local_capture_script.py tests\test_monthly_curve_sensitivity.py tests\test_monthly_forward_curve_constraints.py tests\test_monthly_forward_curve_solver.py tests\test_monthly_forward_curve_priors.py tests\test_monthly_forward_curve_integration.py tests\test_monthly_forward_curve_audit.py tests\test_lt_quant_contract_matrix.py tests\test_peak_offpeak_installed_v34.py tests\test_run_workspace_local_script.py tests\test_lt_package_contract.py tests\test_lt_ct_imports.py tests\test_lt_provider_verifier_artifact.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_runtime_closure.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_audit_legacy_provider_resolution_script.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py tests\test_launcherless_conda_archive_lock.py tests\test_snapshot_publication_external_contract.py tests\test_candidate_bundle.py tests\test_candidate_evidence.py tests\test_candidate_evidence_assembler.py tests\test_ch_lt_local_candidate_quality.py tests\test_snapshot_anchor_client.py tests\test_snapshot_anchor_reference.py tests\test_snapshot_bootstrap_signer.py tests\test_atomic_promotion.py -q -p no:cacheprovider -m "not slow"
```

## Final matrices

All final audit/test receipts share source-tree SHA-256
`ea55935f9e5a3ae02cf14ffdca860258b3330678f4695171747071ee40fdef68`
and the same closure identities listed above.

| Run | Scope | Result | Execution receipt SHA-256 |
|---|---|---:|---|
| `cl35u1` | runner v5 closure/TOCTOU unit matrix | 94 passed | `a420ff3284e2768a90942e5210d7ca198195a7379afc673eb7be96c5338dfe19` |
| `cl35t1` | focused quantitative + installed smoke | 13 passed | `2238190d8ab4481dc8149c2bd0666a8e03bd0dc88aa2ff56e683e1a5986517ba` |
| `cl35au1` | current EEX audit | exit 0, NO-GO | `65dc1d46e82c669c4cb47f5b62c448c5fbfac1981caa069224e5922346c28022` |
| `cl35mx1` | unified scientific/packaging/publication/CAS | 680 passed, 18 skipped, 2 deselected | `12c2ad50f5e198babae1ea36a43272cf38311ef24d50d601c5f77d48336310d7` |

Every row above records target interpreter
`C:\Users\jbattaglia\PFC_LT\build\pytest-runtime-v1\python.exe`, exact `_pth`,
five ordered import roots and content identities before and after execution.
JUnit partitions `cl35mx1` into scientific `275 passed, 4 skipped`, packaging
`169 passed, 12 skipped`, publication `88 passed`, and CAS
`193 passed, 2 skipped`, all with zero failures/errors. The four
scientific skips are CT-only optional imports (`lightgbm`, `torch`,
`tensorflow`). Nine packaging skips require an approved publisher wheelhouse;
three packaging and two CAS skips require Windows symlink privilege unavailable
to the standard user. Two slow packaging tests were explicitly deselected by
`-m "not slow"`. No skip or deselection grants authority.

The earlier `peak34*` receipts targeted the existing user Conda interpreter;
the `rl34*` receipts predated or failed to bind the frozen dependency closure.
Both families are superseded as final workstation-execution evidence and
retained only for historical comparison. The old scoped Ruff
receipt is likewise not part of the canonical local replay; `git diff --check`
is the canonical current-tree formatting check.

`peak30pb1` is abandoned non-evidence. Its receipt SHA-256 is
`1aaae1d0c6868c2037bfd302c9e02b50d3e2062babae80301a69b0f910cf3cea`
and remains `EXECUTION_PENDING` after external termination. It is not a pass
and exposes a remaining stale-run reconciliation/process-supervision gap.

## Files changed in this slice

- `pfc_shaping/lt/model/shape_constraints.py`;
- `scripts/audit_ch_eex_current_repricing_sensitivity.py`;
- `scripts/run_workspace_local.py`;
- `tests/test_lt_quant_contract_matrix.py`;
- `tests/test_audit_ch_eex_current_repricing_sensitivity_script.py`;
- `tests/test_peak_offpeak_installed_v34.py`;
- `tests/test_run_workspace_local_script.py`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`;
- this handoff and `.planning/HANDOFF.md`.

No CT or Power BI file was touched. Protected
`data/eex_forwards_history.parquet` remained at SHA-256
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

## Roasts and remaining NO-GO

Initial read-only roasts found the stale standalone receipt, stale runtime
v33 and repeated-set complexity issue. A later Quant/Data re-roast returned
`0/0/0`, while Security found the unbound test import closure and IT/Operations
found that early `rl34*` receipts predated its final capture. D193 implements
content-hashed pre/post closure identity and the complete `cl35*` replay.

Terminal read-only verdicts after D193:

- Security: P0/P1/P2 `0/0/0`; closure finding closed after independent
  recomputation of `_pth`, five roots, launcher, origins and receipts.
- Quant/Data: `0/0/0`; D193 has no quantitative regression and the PEAK-only
  mechanics, solver authority and explicit NO-GO limits remain valid.
- IT/Operations: local packaging/runtime slice GO, `0/0/2`. P2 items are the
  roughly 0.98 GB double content read per sealed run and the absent global
  timeout/stale-run reconciliation, process/resource telemetry and rollback
  exercise. Production remains NO-GO.

Still open:

- official EEX product/session/settlement/tick semantics and an approved
  source-conflict policy;
- trusted PIT time/signature and builder-inaccessible external CAS/WORM/fresh
  HEAD;
- repricing of a final delivered hourly candidate, not only the mechanical
  oracle;
- prospective multi-season rolling origins, calibrated probabilistic
  scenarios and a new independently sealed future holdout;
- stale-run reconciliation, process/resource observability, CI/SBOM and
  rollback drills.

Monthly solver remains BASE level authority. OMPEX was not used. T057 was not
consumed. No candidate, publication, promotion or production transition was
performed. Production remains strict `NO_GO`.
