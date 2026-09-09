# Session handoff — workspace runner v4 structured evidence

Date: 2026-07-29

Branch: `fix/lt-audit-remediation`

HEAD observed throughout:
`2f68125bff869ccb21c1e20df0201ad024ed27d3`

Production status: strict `NO_GO`

## Outcome

The standard-user laptop runner now produces a bounded, structured and
fail-closed receipt for normally completed local observations without admin,
Defender exceptions, project executables, Playwright, `AppData`, `ProgramData`
or the legacy `H:` checkout. Every mutable namespace is below canonical repo
`build/`.

Receipt schema `pfc_lt_workspace_local_execution.v4` captures and fsyncs
stdout/stderr, JUnit and an append-only native pytest result; cross-checks
target exit and all pytest outcome categories; rejects caller evidence paths,
pytest configuration/plugin injection and argument files; binds evidence by
path plus descriptor identity; bounds inherited-pipe drainage; and binds then
revalidates runner bytes, interpreter bytes, Git HEAD/branch and the selected
dirty Python/config source tree.

A final Security roast demonstrated a residual short-option grouping bypass
(`-qc...`, `-qo...`, `-qp...`). The bypass was fixed and covered by four
additional adversarial cases. The terminal post-fix roasts report no P0/P1/P2
on the completed-run local contract.

This is not crash-safe process-tree supervision, independent attestation,
scientific evidence, candidate qualification, release authority or production
evidence. All five receipt authorities remain false and successful child runs
remain `TARGET_EXIT_ZERO_NOT_AUTHORITY`.

## Exact selected bytes

- `scripts/run_workspace_local.py`
  - bytes: `58,301`
  - SHA-256:
    `cce8fdd620d4fc8e0f9f2c3c678a5c50de20292f089bff9601691a0aead2dcf8`
- `tests/test_run_workspace_local_script.py`
  - bytes: `39,666`
  - SHA-256:
    `eeb3d8ee5b38f567d88ffd228d9880c00a4edcca911affb3a4f6a74be9edde4e`
- selected dirty source tree, common to all five terminal receipts
  - schema: `pfc_lt_workspace_source_tree.v1`
  - files: `487`
  - bytes: `8,814,949`
  - SHA-256:
    `1f3656419f4c3953479c2274d04c792d0d44f7b2c1c2cdd431a43cc159986753`
- target interpreter SHA-256:
  `50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac`

The source-tree field is an aggregate over selected code/config bytes. It is
not a retained per-file inventory, installed-environment admission, market
data/evidence-tree hash or scientific reproducibility proof.

## Terminal commands and results

Every shell command was preceded by the literal cwd/Git-root guard required by
`AGENTS.md` and executed with workdir `C:\Users\jbattaglia\PFC_LT`.

Focused lint and adversarial tests:

```powershell
python -B -m ruff check scripts\run_workspace_local.py tests\test_run_workspace_local_script.py
$env:TEMP='C:\Users\jbattaglia\PFC_LT\build\tmp-runner-v4-grouped3'
$env:TMP=$env:TEMP
python -B -m pytest tests\test_run_workspace_local_script.py -q -p no:cacheprovider --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-runner-v4-grouped3
```

Result: Ruff pass; `62 passed in 3.11s`.

Prospective/software-contract matrix:

```powershell
python -B -m scripts.run_workspace_local --run-id obsv4pro6 -- python -B -m pytest tests\test_run_workspace_local_script.py tests\test_capture_public_energy_charts_lt_script.py tests\test_governed_lt_acquisition.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_lt_provider_verifier_artifact.py tests\test_build_local_intraday_calibration_panel_script.py tests\test_ch_lt_prospective_acquisition_plan.py tests\test_ch_lt_local_candidate_quality.py tests\test_ch_lt_pit_preregistration.py tests\test_ch_lt_preregistration_supersession.py tests\test_dependence_power_supersession.py tests\test_verify_t057_evidence_supersession_script.py tests\test_ch_lt_compute_runtime_manifest.py tests\test_lt_package_contract.py -q -p no:cacheprovider -m "not slow"
```

Result: `278 passed, 2 deselected, 1 warning in 59.27s`.
Receipt SHA-256:
`8fc9e19a420b21767f06ab4f6ca76166a092a34352b68160dbdb02ba42ba6f49`.
The warning is the known timezone-to-period warning in the explicitly
non-admitted legacy Energy Charts loader.

Runtime/packaging matrix:

```powershell
python -B -m scripts.run_workspace_local --run-id obsv4run6 -- python -B -m pytest tests\test_lt_provider_verifier_artifact.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_runtime_closure.py tests\test_lt_package_contract.py tests\test_launcherless_conda_archive_lock.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py tests\test_run_workspace_local_script.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_audit_legacy_provider_resolution_script.py -q -p no:cacheprovider -m "not slow"
```

Result: `225 passed, 12 skipped, 2 deselected in 39.22s`.
Receipt SHA-256:
`23d7cb0b542fa210da099e6306ae977d4287fedaf97b6a8005a373010f7f0b3e`.

External-CAS/publication contract matrix:

```powershell
python -B -m scripts.run_workspace_local --run-id obsv4cas6 -- python -B -m pytest tests\test_snapshot_publication_external_contract.py tests\test_snapshot_anchor_client.py tests\test_snapshot_anchor_reference.py tests\test_snapshot_bootstrap_signer.py tests\test_governed_release.py tests\test_check_monthly_curve_promotion_from_manifests.py -q -p no:cacheprovider -m "not slow"
```

Result: `200 passed in 165.04s`.
Receipt SHA-256:
`adfa157bcbef5ea9bb5425ae4d0fc6f4d1bb3609855aa96966f3a35e95be1374`.

Candidate/atomic contract matrix:

```powershell
python -B -m scripts.run_workspace_local --run-id obsv4cand6 -- python -B -m pytest tests\test_atomic_promotion.py tests\test_candidate_bundle.py tests\test_candidate_evidence.py tests\test_candidate_evidence_assembler.py -q -p no:cacheprovider -m "not slow"
```

Result: `181 passed, 2 skipped in 225.85s`.
Receipt SHA-256:
`ae4953d700375881eaad4dd422d16f1a390c4af36836f217d450dc8c89cf4446`.

Terminal Ruff receipt:

```powershell
python -B -m scripts.run_workspace_local --run-id obsv4ruff5 -- python -B -m ruff check scripts\run_workspace_local.py tests\test_run_workspace_local_script.py
```

Result: pass. Receipt SHA-256:
`408d4b5720bda1cac876bedb2f4c3b815263efc8ef7a480e3547a7ffbc1ce049`.

Every terminal receipt has target/runner exit zero, complete non-truncated
streams, the same source-tree identity, and `runtime`, `evaluation`,
`scientific`, `production` and `promotion` authority false.

## Superseded and negative evidence

- `obsv4pro5`, `obsv4run5`, `obsv4cas5`, `obsv4cand5` and `obsv4ruff4` were
  valid observations of the pre-fix bytes but are superseded by the `*6`/
  `ruff5` receipts after the grouped-short-option fix.
- Earlier `*4`/`ruff3` receipts are also historical only.
- The first two focused grouped-option iterations each produced one test
  failure because rejection classification order returned the wrong error
  category. No receipt was selected from those attempts. The third iteration
  passes `62/62`.
- No historical receipt or run root was rewritten, repaired or relabelled.

## Independent roasts

Post-fix Security, read-only:

- P0 `0`, P1 `0`, P2 `0` on completed local runs.
- Grouped `-qc`, `-qo`, `-qp` forms are rejected and the exact
  `-p no:cacheprovider` allowlist remains usable.

Post-fix Quant/Data, read-only:

- P0 `0`, P1 `0`, P2 `0` new.
- The delta changes only harness integrity. Matrix names do not establish a
  prospective observation, candidate quality, rolling-origin or T057.
- Monthly solver remains level authority; OMPEX remains benchmark-only;
  production remains strict `NO_GO`.

Post-fix IT/Operations, read-only:

- P0 `0`, P1 `0`, P2 `0` new/demonstrated inside the completed-run scope.
- GO for normally completed same-user local observations under the exact
  selected v4 bytes and receipts.
- The documentation P1 is closed: D181, RFC, Operations, the root pointer and
  this handoff select the same terminal artifacts and preserve older receipts
  only as historical evidence.
- Crash/process-tree/independent CI, scientific/release/promotion and
  production remain strict `NO_GO`.

## Changed files in this closure

- `scripts/run_workspace_local.py`
- `tests/test_run_workspace_local_script.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff

No CT file, Power BI file, protected heavy data file, production manifest or
promotion target was intentionally changed.

## Residual blockers and next work

- The completed-run harness is not crash-safe. Global wall timeout, Windows
  Job Object/process-tree kill-on-close, Ctrl+C/crash/power-loss receipt
  finalization, stale-pending reconciliation, partial-log incident binding,
  directory fsync, quota and retention remain governed CI/IT work.
- Pipe joins are bounded to five seconds per stream and can therefore add
  about ten seconds sequentially.
- The source-tree aggregate does not retain a per-file inventory and excludes
  non-Python fixtures/resources and the installed dependency tree.
- Same-user mutability and signature gaps block independent trust.
- No fresh independently governed EEX prospective vintage, rolling-origin
  scientific admission, future holdout/T057 acceptance, auditable CH candidate
  or independent manifest promotion exists yet.
- Resume with fresh point-in-time prospective data and successor
  preregistration, then rolling-origin/T057 and a new CH candidate. Do not
  promote production before the independent evidence chain is complete.
