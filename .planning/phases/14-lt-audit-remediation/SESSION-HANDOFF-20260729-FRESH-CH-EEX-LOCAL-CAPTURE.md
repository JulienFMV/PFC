# Session handoff - fresh CH EEX local capture

Date: 2026-07-29

## Outcome

The current FMV desk workbook was preserved exactly as a local, one-shot,
non-authoritative handoff at
`build/eex-forward-local-captures/eex-ch-20260729-v2`. It must not enter the
monthly solver, shaping, training, selection, T057, candidate assembly,
publication or production.

No admin/elevation, Defender exception, project `.exe`, Playwright, system
install, `AppData` mutable path or `H:` checkout was used. The existing user
Conda interpreter was executed read-only; declared mutable destinations were
redirected below canonical repo `build/`. This harness is not a filesystem
sandbox and does not prove absence of arbitrary writes by hostile code.

## Changed files

- `pfc_shaping/data/eex_forward_local_capture.py`
- `scripts/capture_eex_forward_local.py`
- `scripts/run_workspace_local.py`
- `tests/test_eex_forward_local_capture.py`
- `tests/test_run_workspace_local_script.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff

The pre-existing dirty worktree was preserved. No reset, clean, restore,
stage, commit or promotion occurred. LT/CT separation, monthly-solver level
authority and OMPEX benchmark-only status are unchanged.

## Selected artifacts

Canonical read-only source:
`\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx`.

- source/copy SHA-256:
  `b3e213f1512890ea72af1cee03015fcc942ba9946ec89c7a5d4745603eb5eb0f`;
- size: 61,813 bytes;
- attempt SHA-256:
  `2ffe46c4de791bc202a3f3cd77c88b269747bded0e899b31ef8f88f0dcc648a0`;
- manifest SHA-256:
  `3691df3ad92a18692cd49aa920705f135acf672415d4e9d9faf427464752db4e`;
- logical CH row: `2026-07-28`;
- inventory: 40 syntactically admissible BASE/PEAK CAL, MONTH and QUARTER
  identities; no quote values in the manifest.

All clocks are untrusted. Trusted time, independent acquisition signature,
builder-inaccessible immutable copy, external CAS/WORM plus fresh HEAD and
official product/session/settlement semantics are absent. Every scientific,
solver, training, selection, holdout, candidate, promotion and production flag
is false.

Capture v1/`eexcap02` is superseded because the initial Security roast found
that the direct API accepted an arbitrary external same-name file. The API now
pins the exact UNC. V1 remains historical non-authoritative evidence.
`eexcap01` remains negative pre-execution evidence; receipt SHA-256 is
`06e48c1160e7a31c8ebaf9b799276c99dc57fc8b96958ebe7a46f9e5462376d7`.

## Exact selected capture command

The canonical workspace guard was executed separately immediately before each
shell action.

```powershell
python -B -m scripts.run_workspace_local --run-id eexcap03 -- python -B -m scripts.capture_eex_forward_local --source-document "\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx" --expected-source-sha256 b3e213f1512890ea72af1cee03015fcc942ba9946ec89c7a5d4745603eb5eb0f --capture-id eex-ch-20260729-v2
```

Result: exit 0, `TARGET_EXIT_ZERO_NOT_AUTHORITY`; receipt SHA-256
`18ef170eb2ed5982946a34a9a4185da1243c99df0bbf07c1060e1a52d11051d9`.

## Exact final matrices

Prospective/software:

```powershell
python -B -m scripts.run_workspace_local --run-id eexpro02 -- python -B -m pytest tests\test_run_workspace_local_script.py tests\test_capture_public_energy_charts_lt_script.py tests\test_eex_forward_local_capture.py tests\test_eex_forward_vintage_intake.py tests\test_governed_lt_acquisition.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_lt_provider_verifier_artifact.py tests\test_build_local_intraday_calibration_panel_script.py tests\test_ch_lt_prospective_acquisition_plan.py tests\test_ch_lt_local_candidate_quality.py tests\test_ch_lt_pit_preregistration.py tests\test_ch_lt_preregistration_supersession.py tests\test_dependence_power_supersession.py tests\test_verify_t057_evidence_supersession_script.py tests\test_ch_lt_compute_runtime_manifest.py tests\test_lt_package_contract.py -q -p no:cacheprovider -m "not slow"
```

Result: `316 passed, 2 deselected, 1 known warning`; receipt
`433baa4278e979029f224dcc08d475c0f959d7e2adbde9a86bf97bf0c67f52db`.

Runtime/packaging:

```powershell
python -B -m scripts.run_workspace_local --run-id eexrun02 -- python -B -m pytest tests\test_lt_provider_verifier_artifact.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_runtime_closure.py tests\test_lt_package_contract.py tests\test_launcherless_conda_archive_lock.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py tests\test_run_workspace_local_script.py tests\test_eex_forward_local_capture.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_audit_legacy_provider_resolution_script.py -q -p no:cacheprovider -m "not slow"
```

Result: `239 passed, 12 skipped, 2 deselected`; receipt
`c9ef70aab360aa330502b4e374bcab324845153d9092b5fb1be9e948d5fb6de7`.

External-CAS/publication:

```powershell
python -B -m scripts.run_workspace_local --run-id eexcas02 -- python -B -m pytest tests\test_snapshot_publication_external_contract.py tests\test_snapshot_anchor_client.py tests\test_snapshot_anchor_reference.py tests\test_snapshot_bootstrap_signer.py tests\test_governed_release.py tests\test_check_monthly_curve_promotion_from_manifests.py -q -p no:cacheprovider -m "not slow"
```

Result: `200 passed`; receipt
`a7a25ffdaf665a24573e83a285e4be13f5ea55d46813ea322bdef132b2c51127`.

Candidate/atomic:

```powershell
python -B -m scripts.run_workspace_local --run-id eexcand02 -- python -B -m pytest tests\test_atomic_promotion.py tests\test_candidate_bundle.py tests\test_candidate_evidence.py tests\test_candidate_evidence_assembler.py -q -p no:cacheprovider -m "not slow"
```

Result: `181 passed, 2 skipped`; receipt
`8faacfa4e6b21334590f363f1d7df61f6975d23d548fb9861a84f4626ee43d7d`.

Ruff:

```powershell
python -B -m scripts.run_workspace_local --run-id eexruff02 -- python -B -m ruff check scripts\run_workspace_local.py scripts\capture_eex_forward_local.py pfc_shaping\data\eex_forward_local_capture.py tests\test_run_workspace_local_script.py tests\test_eex_forward_local_capture.py
```

Result: pass; receipt
`88aa97e9c5c088f4fef04da55b61f7726f0b5a161c98cb841201bf224feb92d5`.

All selected receipts share runner SHA-256
`dd25057023b0c5c9d890dcec02b66d3aebd38b9494c752aaf12e74412d90390d`
and source-tree SHA-256
`45849de4ec032c78831ac1ac1bb3c847d4f70513436bf07a9425d556a335cb4d`
over 490 files / 8,838,775 bytes. Every receipt is local-only.

## Roasts and blockers

Initial Security: P0=0, P1=0, P2=1 for the direct-API source bypass; fixed and
all matrices rerun. Initial IT/Operations: P0=0; the missing runbook/handoff P1
was fixed here. Initial Quant/Data found one stale-current-identity P2 in the
root handoff; it was corrected.

Terminal read-only verdicts on the exact final local bytes are:

- Security P0/P1/P2: `0/0/0`;
- IT/Operations P0/P1/P2: `0/0/0`;
- Quant/Data P0/P1/P2: `0/0/0`.

Known production blockers remain deliberately open: no global wall timeout,
Job Object descendant containment, crash/stale-run reconciliation, filesystem
sandbox, CI/service identity, independent trusted time/signature, builder-
inaccessible immutable copy or external CAS/WORM/fresh HEAD. Production
remains strict `NO_GO`.

## Next work

1. Obtain independent trusted-time/acquisition authority for a fresh provider
   delivery; never upgrade this local capture retroactively.
2. Transfer exact bytes to builder-inaccessible external CAS/WORM and obtain a
   fresh-HEAD receipt.
3. Obtain official product/session/settlement/timezone/PEAK-calendar semantics
   before EEX repricing or solver admission.
4. Finish the probabilistic successor preregistration and rolling-origin power
   design without consuming T057.
5. Build a new CH candidate only after all data/scientific gates pass. Never
   promote before complete independent evidence.
