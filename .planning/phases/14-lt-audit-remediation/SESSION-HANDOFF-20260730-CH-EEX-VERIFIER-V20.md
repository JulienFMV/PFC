# Session handoff — 2026-07-30 CH EEX current selection and verifier v20

## Outcome

The fresh local CH EEX quarantine is now selected and reproducibly replayed
through a pre-import admitted Python zipapp. The demonstrated
dependency/import TOCTOU P1 and the subsequent output-boundary P2 are closed
for this local slice. Security and IT/Operations final read-only re-roasts each
report P0/P1/P2 `0/0/0`.

This is local diagnostic evidence only. It does not prove trusted point in
time, source authenticity, official EEX product/session/settlement semantics,
current-price repricing, T057 validity or production readiness. No candidate,
publication, promotion or production transition occurred. Production remains
strict `NO_GO`.

## Workspace and protected state

- canonical cwd and Git root: `C:\Users\jbattaglia\PFC_LT`;
- branch: `fix/lt-audit-remediation`;
- starting HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- worktree remains intentionally very dirty; no reset, clean, restore, stage
  or commit was performed;
- `data/eex_forwards_history.parquet` was not touched or staged; final observed
  SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- no `pfc_shaping/ct/*`, Power BI, project `.exe`, Playwright, AppData mutable
  path, admin right, ACL takeover or Defender/ASR exception was used;
- existing `C:\Users\jbattaglia\.conda\ppa_env\python.exe` was executed
  read-only under `-I -S -B`; dependency closure, `TEMP`/`TMP`, scratch,
  pytest basetemps and outputs remained below repo `build/`.

## Current EEX selection

Registry:

- `.planning/phases/14-lt-audit-remediation/CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json`;
- SHA-256
  `5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778`;
- selection ID
  `3be51903d7ed2774d464f8bfd49b20fe283fb5d1b2c0bb93f677e99fe4884667`.

Selected current capture:

- ID `eex-ch-20260730-v1`;
- exact source 62,490 bytes, SHA-256
  `fb71338f51334128878526877b802e48819b555639913a786a35d710a6b151e5`;
- manifest SHA-256
  `823eeb1095e48a49db6df28cda4fd6f96e0f054154252016be76cdbeca4e1801`;
- workspace receipt SHA-256
  `d191a275fa73828ffa5085e8e92090fa2636e0e2f3fda0a7882100561b9c67ba`;
- latest CH workbook row `2026-07-29`, 40 unique LT quotes: 20 BASE and
  20 PEAK across 12 CAL, 14 QUARTER and 14 MONTH identities;
- identity commitment
  `74ba58f6d00c8734ea668d487c8fb48d6e12e35642045be7a2c834daddbdfc95`;
- price-sensitive value commitment
  `399d524c2c002b55bf844a5410791d0fa33ba7fc4ed843896f01b26769ece258`;
- quote values are not emitted by the registry or audit.

`eex-ch-20260729-v2` is historical only. The registry and audit keep every
scientific, monthly-solver, training, selection, T057, candidate, publication,
promotion and production authority false.

## Software changes

Primary new/changed files for this closure:

- `scripts/audit_ch_eex_current_local_capture.py`
  (`6ee5ce9343b293ee1fc2f9d6c0e197f25a4cd9b3fcaea38e302755f134255a76`):
  exact registry/capture/parser/commitment replay, strict execution identity,
  complete non-truncated target-log re-read and hash validation;
- `pfc_shaping/verifier_package_contract.py`
  (`ac78a0b251b7197a109cab17aa80bb129d65ef93cb4960a73aac3fb8f6a73fb7`):
  positive 26-member inventory, `audit-current-eex` command and frozen parser
  hashes;
- `pfc_shaping/verifier_runtime_admission.py`
  (`1be238784f26c9137fba448c0d359ec8b9ba8376f5a47c886d667352dc43daf2`):
  pre-import supervised dispatch, process-private exact-byte copies, exact
  output root/names, protected evidence roots, post-audit replay and complete
  runtime executable fingerprint in the receipt;
- `deploy/verifier/runtime-contract.json`
  (`281e3e8b4e990569da786b4515d63f8df8f4a7383f4dfd401d733e33f3f11452`):
  adds only locked `openpyxl==3.1.5` and `et-xmlfile==2.0.0` for exact XLSX
  replay;
- `scripts/build_snapshot_publisher_runtime_closure.py`
  (`dcb381216f6005a2fc4c9bef96c59247c643c067d1f2eb81929a44cfd8ee09ec`):
  accepts an explicit stable runtime contract while retaining the publisher
  contract as the default;
- `scripts/build_lt_provider_verifier_zipapp.py`
  (`d223a606d51a9d6eab889e3d3261c8a4955cd823e474eb95f3c50c06bc48b3a8`):
  validates the verifier-specific closure and frozen parser sources;
- `tests/test_audit_ch_eex_current_local_capture_script.py`: deterministic and
  price-sensitive commitments plus missing/tampered/truncated log negatives;
- `tests/test_lt_provider_verifier_artifact.py`: command/inventory,
  reproducibility, exact `sys.path`, dedicated output root and four
  no-residue evidence-root negatives;
- `scripts/run_workspace_local.py` and
  `tests/test_run_workspace_local_script.py`: exact checkout diagnostic route
  and fail-closed grammar, including rejection of launcherless build paths
  outside repo `build/`;
- `pfc_shaping/tools/OPERATIONS.md`: exact standard-user v20 command, hashes,
  output boundary, scratch/retry/quarantine rules;
- D187 in `DECISION-LOG.md` and the v20 amendment in the external-CAS RFC.

## Runtime and artifacts

Dedicated local dependency evidence:

- wheelhouse `build/verifier-eex-runtime-inputs-v2/wheelhouse`, 13 wheels
  already present under repo `build/`; no network/install was used;
- closure `build/verifier-eex-runtime-inputs-v2/site-packages`;
- receipt
  `build/verifier-eex-runtime-inputs-v2/dependency-closure-receipt.json`,
  5,386 bytes, SHA-256
  `999e5f5a31631a4562f0c415f1f8ee2172a988007fd94609c524e004cd843d38`;
- 4,928 files, tree SHA-256
  `9eb10eecc91ed6e676e5605d77eb50c2c15b5dcb6dba311a3b14ad9cda6f1541`;
- exact `uv.lock` SHA-256
  `efcea25267644da75c8736b3ede0dfaaf4b6ee8e58b982a61e87edb1064eb5d6`.

Selected verifier:

- `build/provider-verifier-20260730-eex-v20a.pyz` and `...v20b.pyz`;
- both 111,858 bytes and byte-identical at SHA-256
  `efd896c8c19dc3e4ad1cb04270c09605d86a83e4a126386db4ab8084053a153c`;
- 26 exact members, source revision
  `6e6ac43f935060bc0495de9d5c401f6d2dfdf548e471bf3256b03c2cc216c9c6`;
- admitted audit
  `build/provider-verifier-eex-results/v20-audit-v1/current-eex-audit.json`,
  1,124 bytes, SHA-256
  `917dfd899e12c2795b5b3546eb785efdbfb4f4e8c20d5e31cf0840ad2a940dbf`;
- runtime receipt in the same directory, 2,373 bytes, SHA-256
  `a3ed33c946048bd0d742b518dcaa288bad49d133390f5ade21530db75deb36d7`;
- CPython executable/base SHA-256
  `50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`;
- captured artifact/dependency `sys.path` counts `1/1`; source artifact/
  dependency counts `0/0`; process-private exact-byte import mode;
- terminal scratch residue `0`; no targeted verifier process remained.

## Commands and results

Every shell command was preceded by an exact cwd/Git-root guard. Mutable
paths were explicitly placed under `build/`.

1. Anti-AppData permission contract:

   `pytest tests/test_run_workspace_local_script.py tests/test_launcherless_local_runtime.py ... -k "launcherless_runtime_paths ..."`

   Result: `4 passed, 97 deselected`. An external `AppData` wheelhouse is
   rejected; the repo-`build/` form is accepted. No approval prompt occurred.

2. Initial audit/verifier/closure unit matrix:

   Result after correcting one test expectation/import order: `33 passed, 1
   deselected`; scoped Ruff passed.

3. Closure build:

   `python -B -m scripts.build_snapshot_publisher_runtime_closure --wheel-directory ... --output ... --receipt-output ... --runtime-contract deploy/verifier/runtime-contract.json`

   The shell timeout fired at 122.6 seconds just after the terminal receipt was
   written. No Python process remained. Exact replay with
   `validate_runtime_dependency_closure(..., runtime_contract_path=...)`
   returned `PASS`, 13 distributions, 4,928 files and the hashes above. Do not
   misstate the wrapper timeout as a closure failure.

4. V19 reproduced and closed import-before-admission, but Security found a P2
   output-boundary residue/pollution path. V19 is retained as roast evidence,
   not selected. The boundary and runtime-receipt fixes produced v20.

5. Real v20 audit command is recorded exactly in Operations section 9. It
   exited zero with status `LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY`, produced
   the hashes above and left scratch empty.

6. A first slow verifier pytest used an unnecessarily long basetemp and was
   correctly refused with `insufficient Windows path headroom` after 25 other
   tests passed. Re-running the exact slow test with short basetemp `build/pv2`
   passed in 311.90 seconds. This is fail-closed path policy, not a functional
   regression.

7. Terminal matrices after v20:

- scoped Ruff: pass;
- v20 unit/non-slow: `30 passed, 1 deselected`;
- runtime/packaging:
  `221 passed, 12 skipped, 2 deselected in 38.95s`;
- publication contracts:
  `155 passed, 2 skipped in 120.62s`;
- no external publication or production authority was contacted or mutated.

## Independent re-roasts

- Security final read-only verdict: P0/P1/P2 `0/0/0`. It verified output
  admission before import/write, exact governed root/names, all protected
  evidence roots, four negative no-residue tests, 26-member reproducibility,
  frozen parsers, closure, `1/0/1/0` path counts and empty scratch.
- IT/Operations final read-only verdict: P0/P1/P2 `0/0/0`. It verified the
  receipt/log closure, full executable fingerprint, standard-user boundaries,
  explicit read-only Conda interpreter, absence of AppData/ProgramData/H:/
  Playwright/project exe/admin activity, matrix commands and runbook retry.
- Quant/Data earlier current-selection verdict: P0/P1/P2 `0/0/0` for exact
  local commitments. It explicitly warned that no independent EEX semantics,
  current-price solver repricing or scientific PIT claim is proved.

## Durable invariants and next work

- Monthly solver remains sole monthly-level authority. Do not feed this fresh
  capture into hard constraints until acquisition/time/semantics admission is
  independent and signed.
- OMPEX remains benchmark-only. LT must not import `pfc_shaping.ct.*`.
- T057 remains unconsumed and independent rolling-origin count remains zero.
- Next scientific step is not promotion: obtain independent trusted time,
  provider-authenticated product/session/settlement/unit/calendar semantics
  and builder-inaccessible external CAS/WORM/fresh HEAD for current CH EEX,
  then run exact current-price repricing and sensitivity evidence.
- After that admission, return to fresh prospective outcomes, governed
  rolling origins, sealed future T057 and a new auditable CH candidate.
- External service identity, signed release/SBOM, Windows CI/ASR, monitoring,
  atomic recovery, rollback and disaster-recovery drills remain production
  blockers. Production is strict `NO_GO`.

## 2026-07-30 standard-user no-prompt addendum

The obsolete launcherless runtime-v9 command naming an `AppData` publisher
wheelhouse must not be constructed or submitted. VS Code can request
permission before the builder's own path validation executes. `AGENTS.md` now
requires all command arguments naming mutable state to stay below the
canonical workspace and forbids requesting a sandbox override or approval.
The existing builder and workspace runner remain independent fail-closed
enforcement layers.

Exact local revalidation, with the cwd/Git-root guard and a fresh repo-local
namespace:

```powershell
python -B -m scripts.run_workspace_local --run-id permfix30a -- `
  python -B -m pytest tests\test_launcherless_local_runtime.py `
    tests\test_run_workspace_local_script.py -q -p no:cacheprovider
```

Result: `101 passed in 4.28s`, no approval request. Receipt:
`build/workspace-local-runs/permfix30a/execution-receipt.json`, SHA-256
`a90f46a684d6002d9c70ca87879f3366ee6d38f573f81a05a98d4b9819c69ee8`,
status `TARGET_EXIT_ZERO_NOT_AUTHORITY`, target exit `0`. No model, data,
runtime selection, candidate, promotion or production authority changed.

Final independent read-only re-roasts:

- Security: P0/P1/P2 `0/0/0`; the agent-side rule closes pre-submission
  prompting, while the independent builder/runner checks remain mandatory for
  any other caller or bypass.
- IT/Operations: P0/P1/P2 `0/0/0`; zero mutable path outside `build/`, zero
  reparse point and zero residual process were observed. The external Conda
  interpreter was read/executed only; all mutable state remained repo-local.
