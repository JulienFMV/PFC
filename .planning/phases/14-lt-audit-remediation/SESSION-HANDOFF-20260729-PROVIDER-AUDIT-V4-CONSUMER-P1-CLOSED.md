# Session handoff — provider audit v4 consumer P1 closed

Date: 2026-07-29  
Branch: `fix/lt-audit-remediation`  
Observed HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production: strict `NO_GO`

## Outcome

The demonstrated v3 arbitrary-bronze consumer P1 is closed locally. Audit v4
separates security, source-scientific and candidate-readiness gates; records
the honest one-episode evidence scope; and remains local diagnostic only. The
panel now binds the audit, runtime receipt and actual verifier artifact,
audits the verifier manifest, independently replays provider bytes and checks
all authority fields type-strictly. No training, T057, candidate, publication
or promotion occurred.

The sole selected provider verifier name is
`build/provider-verifier-20260729-v18.pyz`. V17 is byte-identical and retained
only as a reproducibility witness. Provider verifier v14-v16 and CH audits
v2/v3 are retained historical evidence and are non-selectable. The main
launcherless application runtime remains v22; its numbering is independent
from verifier v18.

## Changed files in this closure

- `pfc_shaping/cli/audit_provider_acquisition_quarantine.py`
- `scripts/build_local_intraday_calibration_panel.py`
- `scripts/backtest_intraday_shape_estimator.py`
- `scripts/build_local_test_ch_pfc.py`
- `scripts/build_launcherless_local_runtime.py`
- `scripts/run_workspace_local.py`
- `tests/test_audit_provider_acquisition_quarantine_script.py`
- `tests/test_build_local_intraday_calibration_panel_script.py`
- `tests/test_backtest_intraday_shape_estimator_script.py`
- `tests/test_build_local_test_ch_pfc_script.py`
- `tests/test_lt_provider_verifier_artifact.py`
- `tests/test_launcherless_local_runtime.py`
- `tests/test_run_workspace_local_script.py`
- `deploy/verifier/README.md`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- this handoff

No `pfc_shaping/ct/*`, Power BI or protected heavy data file was intentionally
modified. Nothing was staged or committed.

## Standard-user boundary correction

The obsolete v9 command was problematic because it named an `AppData`
wheelhouse. The current boundary is double fail-closed:

- `scripts.run_workspace_local` confines every explicit launcherless and
  local-panel path below repo `build/` and scrubs mutable caches/temp roots;
- direct launcherless and panel CLIs independently require exact cwd/Git root
  and reject paths outside canonical repo `build/`.

There is no admin, elevation, ACL takeover, Defender exception, project
`.exe`, Playwright, system install or mutable environment outside the repo.
The preinstalled user Conda Python is used read-only.

## Canonical artifacts

- verifier v17 and v18: SHA-256
  `7f17be8de8e78ba5a063903c7ea459baed0372b70a128ffab3fb8b17f69b19c5`,
  64,877 bytes, 17 members;
- verifier source revision
  `99be2ce84325789aeacda69a41997766f1f365abe6747fa04b7d21bdad6b9a34`;
- dependency tree
  `0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`;
- acquisition manifest
  `d1bcddc7d56bfc1c6ad9a2936e6e3b77f1ee662af74df20f63cf22193227f8e0`;
- audit v4
  `build/prospective-audits/ch-da-hourly-20260729-v1-audit-v4.json`, SHA-256
  `638a6fc8887b867957fb8cb0ba2cafcf07c00c33fd2620c51e3bca366c2bfc02`;
- v4 runtime receipt SHA-256
  `f28bd6d4e93235467ad4a8c4a2102d6aee5ef7231a96ae3f3a512ccbcb12d82e`.

The isolated runtime-check passed with captured artifact/dependency roots
`1/1`, source artifact/dependency roots `0/0`, CPython 3.11.13, dependency
import `PROCESS_PRIVATE_EXACT_BYTE_COPY`, both authorities false and zero
scratch residue.

Audit v4 reports one capture episode, 720 hourly source observations, 2,880
stepwise proxy rows, an independent-information upper bound of 720, no added
information from proxy rows and no established seasonal coverage. All
training/model-selection/confirmatory rolling-origin/future-holdout/T057/
candidate gates are false; `local_diagnostic_allowed=true` only.

## Exact verifier commands and results

Both deterministic builds used the read-only
`C:\Users\jbattaglia\.conda\ppa_env\python.exe`, repo-local `TEMP/TMP`, the
dependency root
`build/runtime-inputs-20260728-repolocal-v1/publisher-site-packages`, receipt
`publisher-dependency-closure-receipt.json` and wheelhouse
`publisher-wheelhouse`:

```powershell
python.exe -B -m scripts.build_lt_provider_verifier_zipapp build\provider-verifier-20260729-v17.pyz --dependency-root build\runtime-inputs-20260728-repolocal-v1\publisher-site-packages --dependency-receipt build\runtime-inputs-20260728-repolocal-v1\publisher-dependency-closure-receipt.json --wheel-directory build\runtime-inputs-20260728-repolocal-v1\publisher-wheelhouse
python.exe -B -m scripts.build_lt_provider_verifier_zipapp build\provider-verifier-20260729-v18.pyz --dependency-root build\runtime-inputs-20260728-repolocal-v1\publisher-site-packages --dependency-receipt build\runtime-inputs-20260728-repolocal-v1\publisher-dependency-closure-receipt.json --wheel-directory build\runtime-inputs-20260728-repolocal-v1\publisher-wheelhouse
python.exe -I -S -B build\provider-verifier-20260729-v18.pyz runtime-check
```

The first v17 parent capture timed out after 120 seconds while its child
continued and completed; the artifact was then audited and found identical to
v18. The first v17 runtime-check likewise completed just after the parent
capture deadline with zero scratch residue but no retained stdout. V18 was
rerun with a sufficient capture window and is the counted PASS.

The successful real audit command was:

```powershell
python.exe -I -S -B build\provider-verifier-20260729-v18.pyz audit-acquisition --acquisition-directory build\prospective-acquisitions\ch-da-hourly-20260729-v1 --expected-manifest-sha256 d1bcddc7d56bfc1c6ad9a2936e6e3b77f1ee662af74df20f63cf22193227f8e0 --output-json build\prospective-audits\ch-da-hourly-20260729-v1-audit-v4.json --runtime-receipt-json build\prospective-audits\ch-da-hourly-20260729-v1-audit-v4-runtime-receipt.json
```

It exited 0 in 107.5 seconds. One preceding command incorrectly passed the
raw capture directory instead of the built acquisition directory and failed
closed with `acquisition directory inventory is not exact`; it wrote no v4
output and is retained as terminal negative evidence.

## Real CH role rejection

The final invocation passed the actual audit, receipt and v18 artifact through
`scripts.run_workspace_local --run-id chrole02` into the DE-only panel. It
exited 1 with `audit JSON is not eligible for the local DE intraday panel`.
Receipt SHA-256 is
`c60630815773dbc41048b6628aff7c1f740c2eba8137e8d45a45066b054ca094`,
status `TARGET_EXIT_NONZERO`, target exit 1, all authorities false and zero
panel/manifest outputs.

## Terminal tests on the exact final tree

- `audv4f2`: focused audit/panel/downstream/verifier/runner,
  `125 passed, 1 skipped`; receipt SHA-256
  `5fe46b643a2987abc8ac2177c7e36425bb697e58c1c77827071257f786c8d8e0`;
- `audv4r4`: targeted Ruff pass; receipt SHA-256
  `86b1bcba5d9de71ec4394f81d8797bffe0f3fa440297d9f6be846f87fae59524`;
- `prospmat5`: `243 passed, 2 deselected`, one pre-existing timezone warning;
  receipt SHA-256
  `1d0380d5e2f402e524fda808b5112d397876e6810d788a7216afbc41ab5737eb`;
- `runtime11`: `221 passed, 12 skipped, 2 deselected`; receipt SHA-256
  `9860f6b691f1a9b2c5415367f06ee6e7ea1c9fc2241d30332599dc4abbc12797`;
- `pubcas8`: `200 passed`; receipt SHA-256
  `8793cb32a322aa90dcf90e797ef9cdf546884bed49472c2b89325ef5dd6ae36a`;
- `pubcand8`: `181 passed, 2 skipped`; receipt SHA-256
  `cc5c9be942b5f4f1429c97ae7a60cd9c8808234fc24eeee65391d861614da37f`.

Every runner receipt is local-only, with production, promotion, scientific,
evaluation and runtime authority false. Receipts retain command/exit identity
but not stdout/stderr or parsed test counts; that observability gap remains IT
P2. V17/v18 are same-host deterministic evidence, not independent release
reproducibility.

## Roasts and blockers

The first v4 roasts found no local P0/P1 after the core v3 consumer fix but
identified four Security P2s: claimed verifier identity without artifact
bytes, missing manifest semantic-hash comparison, partial legacy claims/replay
and unrestricted direct panel paths. All four are corrected and covered by
adversarial tests. Quant's two conceptual v3 P2s are closed; its missing real
rejection receipt is closed by `chrole02`. IT's documentation P1 is closed by
the explicit pins in the runbook, operations guide, D180, root handoff and
this file.

Final independent read-only re-roasts:

- Security code: P0=0, P1=0, P2=0. The four preceding findings are closed;
  the actual v18 bytes, fresh and legacy replays, `build/` confinement and
  real CH-to-DE-panel rejection were independently checked.
- IT/Operations: P0=0, P1=0. V18 selection, historical non-selection,
  hashes, path boundaries, matrix receipts, source-before-recertification
  ordering and zero forbidden output were independently checked. Two
  non-blocking P2 limitations remain: runner receipts do not persist
  stdout/stderr or structured pytest counts, and v17/v18 are same-host rather
  than independently reproduced release artifacts.
- Quant/Data: P0=0, P1=0, P2=0. Exact replay and local hourly exploratory QA
  are accepted; training, scientific admission, confirmatory rolling-origin,
  T057, candidate authority, probabilistic calibration and production remain
  disallowed.

The consolidated verdict is `GO` only for governed local diagnostics and
`NO_GO` for scientific/candidate admission, T057, promotion and production.

Final scope audit after appending these verdicts: `git diff --check` passed;
HEAD remained `2f68125bff869ccb21c1e20df0201ad024ed27d3` on
`fix/lt-audit-remediation`; the protected forward-history SHA-256 remained
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
staged files, CT diffs and Power BI diffs were all zero. The LF-to-CRLF notices
were warnings on the intentionally dirty worktree, not `diff --check`
failures. No commit or promotion was performed.

External blockers remain: signed host/runtime/release attestation, trusted
available-at/revision time, official product/session/settlement identity,
capture-time TLS/CA attestation, independent signer, Builder-inaccessible
immutable freeze, external CAS/WORM/fresh monotone HEAD, read-only service
identity, Windows CI/ASR, SBOM/scans, logs/SLOs, rollback and incident drills.

## Next work

1. Obtain fresh governed prospective EEX PIT hard-level evidence with official
   semantics and independent time/signature/CAS authorities.
2. Accumulate preregistered multi-season, multi-regime unique origins with
   dependence-aware power; do not treat proxy rows as new observations.
3. Keep T057 sealed until its preregistered one-shot conditions are met.
4. Only then assemble and compare a new CH candidate under monthly solver
   level authority, exact EEX repricing and probabilistic calibration gates.
5. Do not promote production before independent Security, IT/Operations and
   Quant/Data evidence is complete.
