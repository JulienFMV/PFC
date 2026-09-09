# Session handoff - publisher runtime v6 P1 closed

Date: 2026-07-17

## State and decision

- Canonical repo: `C:\Users\jbattaglia\PFC_LT` only.
- Branch: `fix/lt-audit-remediation`.
- Starting/current HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. Do not reset, clean, restore or
  mass-stage it.
- No commit and no production promotion were performed.
- `data/eex_forwards_history.parquet` was not touched or staged by this work.
- CT and Power BI were not touched.
- Monthly solver authority and OMPEX benchmark-only status are unchanged.
- Production remains strictly `NO_GO`.

The P1 dependency verification/import TOCTOU and the demonstrated publisher
runtime/packaging findings are closed at code level. Independent final Security
and IT/Operations re-roasts found no remaining P0/P1 in this slice. External IT
attestations and fresh auditable data/candidate evidence remain missing.

## Exact real build inputs

- Wheelhouse:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252`
- Dependency root (exact `site-packages`, not its parent):
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages`
- Receipt:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json`
- Receipt SHA-256 from the preceding checkpoint:
  `69c0843b4b9a2e202c838ed04027eec1e5008ebec04c0b3c30f37217b0fb45a1`

## Closed findings

The initial v6 re-roasts demonstrated, and the current work closes:

1. source tree could change after receipt validation and be rebound with stale
   wheel/RECORD provenance;
2. artifact audit could hash a second path read instead of the audited bytes;
3. artifact resolution, scratch and capability setup could escape the
   JSON/exit-50 boundary;
4. Windows worker ran before Job Object assignment;
5. `FMV_DATA_ROOT` could be silently overridden by `--data-root`;
6. capability wording overclaimed one-shot/scratch binding;
7. runbook did not explicitly cover abnormal-parent supervisor residue.

Current v6 requires receipt/tree equality, a stable mono-linked artifact image,
fully structured supervisor admission, suspended-before-Job startup, consumed
scratch-bound capability and authoritative `FMV_DATA_ROOT`.

## Exact final real command and result

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

Result:

- `1 passed, 31 deselected in 613.58s (0:10:13)`;
- `WALL_SECONDS=614.7710607`;
- both `publisher-a.pyz` and `publisher-b.pyz`: 85,451 bytes;
- both SHA-256:
  `854f3a7b738f34b30d72e81e9b620f3a7bf2d7fe4cc04da0f65b712f2f8fb663`;
- capture inventory remained exactly `12 -> 12`;
- the 12 entries are documented pre-v6 legacy `fmv-pfc-publisher-runtime-*`
  directories; no new supervisor directory and no related process remained.

## Other final verification

- Explicit sys.path tests: only the captured private root is appended; source
  root absent; distinct base-relative shadow rejected; runtime base removed.
  Result `4 passed, 1 skipped` (physical symlink creation unavailable on this
  Windows identity; lexical and real tree attacks pass).
- Real tree/metadata/`.pth`/hardlink, duplicate zip member and post-capture
  artifact-substitution attacks: `3 passed, 28 deselected in 163.41s`.
- Runtime matrix:
  `31 passed, 9 skipped in 3.07s`.
- Publication/CAS matrix:
  `120 passed in 46.64s`.
- Packaging matrix:
  `22 passed in 3.95s`.
- Targeted Ruff: `All checks passed!`.
- Targeted `py_compile`: pass.
- Final `git diff --check`: exit `0`; only expected Windows CRLF notices.

## Files changed in the runtime/packaging slice

- `pfc_shaping/publisher_runtime_admission.py`
- `scripts/build_snapshot_publisher_runtime_closure.py`
- `scripts/build_snapshot_publisher_zipapp.py`
- `scripts/publish_governed_lt_input_snapshot.py`
- `deploy/publisher/runtime-contract.json`
- `deploy/publisher/environment-contract.json`
- `deploy/publisher/README.md`
- `pfc_shaping/tools/OPERATIONS.md`
- `tests/test_snapshot_publisher_runtime_closure.py`
- `tests/test_snapshot_publisher_artifact.py`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

These coexist with a much larger pre-existing dirty Phase 14 stack. Audit exact
scope before any commit. Never stage the heavy EEX parquet.

## Scientific doctrine added

The three supplied PDFs were read without modifying them and are hash-recorded
in:

`.planning/phases/14-lt-audit-remediation/PFC-2026-LITERATURE-EVIDENCE-AND-GATES-20260717.md`

That doctrine separates forward valuation from spot forecasting and turns the
literature into falsifiable G0-G4 gates: immutable PIT evidence, exact final
EEX repricing, solver-level authority, nested rolling-origin shaping evidence,
calibrated/dependence-aware scenarios, atomic governance and IT reproducibility.
The product quality charter links to it. Numerical FMV thresholds are labelled
as policy, not universal claims from the papers.

## Residual risks and next work

Production remains blocked by the real independent CAS, service identities and
ACLs, signed release attestation, approved wheelhouse/SBOM/scans, HSM/KMS,
WORM/retention, true service-manager Job proof, multi-host/power-loss,
monitoring, backup/restore and DR.

The locked T057 window is 2026-07-10 through 2026-07-24 and is not mature on
2026-07-17. Do not edit, peek-tune or prematurely score its plan. After the
window matures and fresh provider-raw/PIT data are governed, evaluate T057 once
as pre-registered. Then build a new auditable CH candidate under the monthly
solver and no-OMPEX invariants. Do not promote it.

P2 packaging follow-ups, not P1 code blockers, include a direct capability
write-failure/cleanup test, exact structural validation of the environment
contract, richer dual-error reporting, startup SLO/timeout/health monitoring
and a target-host supervisor-residue drill.
