# Session handoff - CH hourly ledger v10 and recursive current evidence

Date: 2026-07-31  
Branch: `fix/lt-audit-remediation`  
Observed HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`  
Production: strict `NO_GO`

## Outcome

A fresh local CH hourly capture extends the selected contiguous diagnostic
ledger through `2026-07-31T06:00:00Z`. The selected ledger contains 776 native
hourly observations and 3,104 stepwise QH transport rows across five capture
episodes. Independent rolling-origin count remains zero. All captures are
post-delivery according to an untrusted workstation clock; no PIT, revision,
product/session, scientific, training, candidate or economic authority exists.

Security demonstrated two P1s in the first current-evidence auditor: nested
ledger evidence was not recursively reopened and lexical traversal was checked
before path normalization. The final auditor reconstructs the complete ledger
from the hash-bound request, rehashes/replays all five evidence chains, exact-
compares the result, binds supersession V4 -> V3 -> V2 -> V1 and rejects
ambiguous Windows paths including traversal, drive-relative and NTFS ADS
forms. Final Security re-roast reports no local P0/P1/new P2.

Laptop-model qualification V5 remains frozen on its original evidence
boundary. Nothing in this session retrains or requalifies the model. T057 and
the future holdout were not opened or consumed. No candidate, publication or
promotion occurred.

## Changed files

- `.planning/phases/14-lt-audit-remediation/CH-LT-PROSPECTIVE-CAPTURE-LEDGER-SELECTION-V4-20260731.json`
- `.planning/phases/14-lt-audit-remediation/CH-LT-CURRENT-EVIDENCE-SELECTION-V3-20260731.json`
- `scripts/audit_ch_lt_current_evidence_selection.py`
- `tests/test_audit_ch_lt_current_evidence_selection_script.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff

Generated evidence below `build/`:

- `prospective-captures/ch-da-hourly-20260731-v2/`
- `prospective-acquisitions/ch-da-hourly-20260731-v2/`
- `prospective-audits/ch-da-hourly-20260731-v2-audit-v4*.json`
- `prospective-ledgers/ch-hourly-local-ledger-request-20260731-v5.json`
- `prospective-ledgers/ch-hourly-local-ledger-20260731-v10.json`
- `prospective-ledgers/ch-hourly-local-ledger-execution-receipt-20260731-v10.json`
- `workspace-local-runs/cev31v3h/`
- `workspace-local-supervisors/cev31v3h/`

Protected `data/eex_forwards_history.parquet` was not touched or staged. Its
SHA-256 remains
`21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
No LT change imported or modified `pfc_shaping/ct/*`; Power BI was untouched.

## Capture, acquisition and isolated audit

Capture command, run from the exact canonical root through the repo-local
runner:

```powershell
build\pytest-runtime-v1\python.exe -B -m scripts.run_workspace_local --run-id chcap31v2 --wall-timeout-seconds 300 -- build\pytest-runtime-v1\python.exe -B -m scripts.capture_public_energy_charts_lt --role epex_ch --start-utc 2026-07-31T00:00:00Z --end-utc 2026-07-31T06:00:00Z --raw-cadence-minutes 60 --acquisition-id ch-da-hourly-20260731t0000z-20260731t0600z-v1 --output-directory C:\Users\jbattaglia\PFC_LT\build\prospective-captures\ch-da-hourly-20260731-v2 --ca-bundle C:\certs\git-ca-plus-fmv.crt
```

Capture supervisor/execution receipt SHA-256 values:

- `411bdb4743e0af5d046eff3850ac967dbe937faf1e6282ad09878ac8475f8f87`;
- `779c7f5e3c43fc493cc56dbe99235d77935c03839f5aca920bce8dbb244e4c76`.

Capture summary SHA-256 is
`71a19f45382a2d818a0bea0612f06af5e6fc0b2b07793e3f43355062d34495a6`.
It records six source observations, 24 output rows, native hourly cadence and
`UPSAMPLED_STEPWISE_PROXY`; native QH truth is false.

The installed builder used runtime v40 with both
`PFC_LT_RUNTIME_RECEIPT_PATH` and
`PFC_LT_RUNTIME_RECEIPT_SHA256=651c8caa548d2e1fdd874f7173397c6f2a05a5d2f3b01ae4a084fbf49468f561`,
then:

```powershell
build\conda-runtime-v40-origin-registry-v2-chain\python.exe -I -B -m pfc_shaping.cli.governed_acquisition_builder --capture-spec C:\Users\jbattaglia\PFC_LT\build\prospective-captures\ch-da-hourly-20260731-v2\capture-spec.json --output-directory C:\Users\jbattaglia\PFC_LT\build\prospective-acquisitions\ch-da-hourly-20260731-v2
```

Acquisition manifest SHA-256 is
`37e057ba4191023affadf4d552879fbc313e66803fe91d3c30b1f28e6d2700b3`.
The direct builder has no runner-v6 process receipt; this remains an IT P2.

The successful isolated verifier used the existing `.conda` Python read-only,
repo-local dependency root and fresh repo-local scratch:

```powershell
$env:PFC_LT_VERIFIER_DEPENDENCY_ROOT='C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-site-packages'
$env:PFC_LT_VERIFIER_SCRATCH_ROOT='C:\Users\jbattaglia\PFC_LT\build\verifier-scratch-ch31v2-a4'
C:\Users\jbattaglia\.conda\ppa_env\python.exe -I -S -B C:\Users\jbattaglia\PFC_LT\build\provider-verifier-20260729-v18.pyz audit-acquisition --acquisition-directory C:\Users\jbattaglia\PFC_LT\build\prospective-acquisitions\ch-da-hourly-20260731-v2 --expected-manifest-sha256 37e057ba4191023affadf4d552879fbc313e66803fe91d3c30b1f28e6d2700b3 --output-json C:\Users\jbattaglia\PFC_LT\build\prospective-audits\ch-da-hourly-20260731-v2-audit-v4.json --runtime-receipt-json C:\Users\jbattaglia\PFC_LT\build\prospective-audits\ch-da-hourly-20260731-v2-audit-v4-runtime-receipt.json
```

It exited zero after 235.3 seconds. Audit/runtime-receipt SHA-256 values are:

- `89ea389a54bea8e774590d0c1a032800c0579eac6b1de12242dbef6384d7408f`;
- `988ceae574054adef8d39acfdb0936948d1100e83563f42ad62c784f2883e822`.

Runtime admission reports one captured artifact root, one captured dependency
root, zero source roots and `PROCESS_PRIVATE_EXACT_BYTE_COPY`. Scratch residue
is zero. Three preceding configurations failed closed before output: missing
dependency-root environment, wrong ProgramData Python fingerprint, and an
installed runtime whose extra `sys.path` closure violated verifier isolation.

## Ledger and selection identities

Request v5 SHA-256:
`bc79c1f8a793ed3391ae3dd4a7df5b7793b3319144162b7ecaf55d56430fb136`.

Construction command under installed runtime v40:

```powershell
build\conda-runtime-v40-origin-registry-v2-chain\python.exe -I -B -m pfc_shaping.cli.build_ch_lt_prospective_capture_ledger --repo-root C:\Users\jbattaglia\PFC_LT --request C:\Users\jbattaglia\PFC_LT\build\prospective-ledgers\ch-hourly-local-ledger-request-20260731-v5.json --expected-request-sha256 bc79c1f8a793ed3391ae3dd4a7df5b7793b3319144162b7ecaf55d56430fb136 --output C:\Users\jbattaglia\PFC_LT\build\prospective-ledgers\ch-hourly-local-ledger-20260731-v10.json --execution-receipt-output C:\Users\jbattaglia\PFC_LT\build\prospective-ledgers\ch-hourly-local-ledger-execution-receipt-20260731-v10.json
```

- ledger SHA-256:
  `3805b71f1368a0e742c8627d4995a85c791de1360257f97b78be0b1665723140`;
- ledger ID:
  `cd759fb51a042d0a9f7b3a0c67f2badb3c6aadb098b33a37800bc6bb8c4f25df`;
- ledger execution-receipt SHA-256:
  `cd6f4e5134293546e81ae6f4628e9d31211d3598afd3f1bf2829572e75e9f575`;
- selection V4 SHA-256:
  `1adcf532d4df2508491dd4a1fb7ed5429d111a8d3721d900eb9529e4594575be`;
- current registry V3 SHA-256 / registry ID:
  `6518f7e876ce1c233fc055d3d20ad213088d361d784afbe5aa16d4f165e744f7` /
  `748aba2e2f85711d0a5dcdb07e0acacbf8dbce7a76ab4a4b07ef48371ec25488`.

The selected window is
`[2026-06-28T22:00:00Z, 2026-07-31T06:00:00Z)`: 776 native hours, 3,104 QH
transport proxies, five episodes, zero independent origins. Every authority is
false.

## Security fixes and final tests

Final auditor/test SHA-256 values:

- `scripts/audit_ch_lt_current_evidence_selection.py`:
  `017f17e6d41edd6b47e39b2f3af3650ab4ef368d3a4ab97bf27a71f922300f40`;
- `tests/test_audit_ch_lt_current_evidence_selection_script.py`:
  `5e157d6770b9460ba329612e4c6711f783fa2ce66b8209afe1f1c3cd84891395`.

Final direct matrix:

```powershell
build\pytest-runtime-v1\python.exe -B -m pytest tests\test_capture_public_energy_charts_lt_script.py tests\test_governed_lt_acquisition.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_ch_lt_prospective_capture_ledger.py tests\test_audit_ch_lt_current_evidence_selection_script.py tests\test_ch_market_time_regime.py tests\test_run_workspace_local_script.py -q -p no:cacheprovider -m "not slow" --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-data31v10-final
```

Result: `270 passed, 1 deselected`, zero failure; one known pandas warning in a
legacy loader explicitly tested as non-admitted. Scoped Ruff passes.

Final supervised command:

```powershell
build\pytest-runtime-v1\python.exe -B -m scripts.run_workspace_local --run-id cev31v3h --wall-timeout-seconds 900 -- build\pytest-runtime-v1\python.exe -B -m pytest tests\test_audit_ch_lt_current_evidence_selection_script.py tests\test_ch_lt_prospective_capture_ledger.py -q -p no:cacheprovider
```

Result: `32 passed`, zero skip/failure/error/deselection. Source tree SHA-256 is
`4b54a1879e71b52dc542b35bff975b25f22727016984d940cd401429dada0b3c`.
Execution/supervisor receipt SHA-256 values are:

- `c0c63129c36b2e6c25070a3851defa822e060ab89ca7817c9e7021fc43f26994`;
- `6a5663c6b24a7b68effbedfe10638da2e95ca36f2667c36a30e469a25f364a04`.

Captured import status is `BOUND_REPO_LOCAL_PTH`; the canonical checkout root
occurs exactly once in `sys.path`. Active processes are zero, deadline was not
exceeded, terminal receipt writes equal one and all authorities are false.

## Final independent roasts

- Security/Governance: P0/P1/new-P2 `0/0/0` after recursive reconstruction,
  supersession binding and Windows path probes.
- IT/Operations: `0/0/4`. Retained P2 are external authority/power-loss,
  denylist rather than filesystem sandbox, missing runner-v6 receipts on each
  direct construction step, and missing start/status/wait plus SLO contract.
- Quant/Data: `0/0/0` for admitted local claims. The five post-delivery
  episodes are not forecast origins; there is no seasonal, QH, PIT, economic or
  model-quality evidence.

## Next highest-value work

1. Produce and independently register a real outcome-blind prediction
   commitment before its first delivery interval, bound to exact solver,
   quotes, shaping, scenario and runtime identities.
2. Preserve its target mask and predictions without reading outcomes. Use the
   existing non-production registry only as a protocol rehearsal; it cannot
   create a countable origin.
3. After delivery, acquire outcomes through a trusted PIT/revision-aware path
   and score the frozen commitment with dependence-aware rolling-origin
   metrics and calibrated probabilistic diagnostics.
4. Keep T057 sealed. It may be evaluated only under its governed one-shot
   contract after the prospective process is sufficiently mature.
5. Build a new CH candidate and revisit promotion only after fresh governed EEX
   levels, independent origins, future holdout and probabilistic gates exist.

External trusted time/signatures, official product identity, CAS/WORM/fresh
HEAD, multi-season origins, CI/ASR/SBOM, filesystem isolation, power-loss
durability, observability and rollback remain mandatory. Production is strict
`NO_GO`.
