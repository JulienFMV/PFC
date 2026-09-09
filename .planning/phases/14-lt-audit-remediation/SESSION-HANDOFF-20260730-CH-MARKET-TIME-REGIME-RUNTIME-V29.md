# Session handoff - CH market-time regime and launcherless runtime v29

Date: 2026-07-30 (Europe/Zurich)

## Outcome

The Swiss auction market-time transition is now governed by a fail-closed,
point-in-time contract and an installed launcherless audit. The locally
admitted diagnostic state remains native DA MTU 60 minutes and native IDA
auction MTU 60 minutes. EPEX's 3 November 2026 statement is recorded only as a
planned first trading day. Actual go-live and the exact effective first
delivery UTC boundary are not proven. The 15-minute model valuation grid is
not native quarter-hour market truth.

No production promotion occurred. T057 was not read or launched. Monthly
solver level authority, LT/CT separation and OMPEX benchmark-only status are
unchanged. Production is strict `NO_GO`.

The workstation route is standard-user only. Ordinary repo-local commands no
longer request approval. Every mutable environment, cache, temp, wheelhouse,
runtime and output path is below `C:\Users\jbattaglia\PFC_LT\build`. Never use
the obsolete v9/AppData command, admin/elevation, ACL takeover, Defender/ASR
exceptions, project `.exe` files or Playwright.

## Read order

1. `AGENTS.md`
2. `.planning/HANDOFF.md`
3. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D185)
4. this handoff
5. `.planning/phases/14-lt-audit-remediation/CH-MARKET-TIME-REGIME-CONTRACT-20260730.json`
6. `pfc_shaping/tools/OPERATIONS.md` section 12

## Changed files for this closure

- `.planning/phases/14-lt-audit-remediation/CH-MARKET-TIME-REGIME-CONTRACT-20260730.json` (new)
- `pfc_shaping/validation/ch_market_time_regime.py` (new)
- `pfc_shaping/cli/audit_ch_market_time_regime.py` (new)
- `scripts/audit_ch_market_time_regime.py` (new checkout wrapper)
- `tests/test_ch_market_time_regime.py` (new)
- `pfc_shaping/package_contract.py`
- `scripts/check_lt_wheel_contract.py`
- `tests/test_lt_package_contract.py`
- `scripts/build_launcherless_local_runtime.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff

The worktree was already intentionally very dirty and remains so. Nothing was
reset, cleaned, restored, staged or committed. `pfc_shaping/ct/*`, Power BI and
heavy desk data were not touched.

## Official evidence and contract

Local evidence namespace:
`build/market-time-regime-evidence-20260730-v3`.

- EPEX HTML: 68,453 bytes, SHA-256
  `73c0c7b6010d3b8d14ade3936233f940abe4fb956fd7a9623fc32a5f5f42edec`.
- Swissgrid roadmap PDF: 9,006,825 bytes, SHA-256
  `879b287916d5be571b3cbffc1a50b7af891c6b63685d8ba59dd2732ff25faef7`.
- BGM report PDF: 6,129,398 bytes, SHA-256
  `60d321d39b5544bba91e4fe521552c01d322fdfea104cfd625bc26bdf15482c0`.

Capture v1 failed because Windows Schannel revocation status was unavailable;
v2 received HTTP 403; v3 succeeded with best-effort revocation behavior and a
browser user agent. All are retained as historical evidence.

Contract SHA-256:
`71711ae80b64556b8deab88e70581e1c7a0ef7c684d672b2cb550f2058c19c25`.
Contract ID:
`898d35e37a2df9dc9814039698eb605fdf220ece33d2035c7c7a2ea1b7bc9dba`.

The contract requires all eight blockers printed by the installed audit. It
separates DA/IDA and pre/post regimes, records DA CPM `[15,30,60]` and IDA CPM
`NONE`, requires exact four-QH arithmetic averaging for an hourly DA index,
forbids proxy truth and cross-boundary resampling, binds per-origin notice and
revision vintages, and requires DST cardinalities 92/96/100. All authority
flags remain false.

## Wheels and runtime

Wheels AI/AJ are 93-member, 518,505-byte, byte-identical artifacts:

- `build/wheel-dist-ai/fmv_pfc_lt-0.14.0-py3-none-any.whl`
- `build/wheel-dist-aj/fmv_pfc_lt-0.14.0-py3-none-any.whl`
- SHA-256
  `896a24ede95ce38110942ce6217d1af994dc003589f9c0da35505a85be51652b`
- source revision
  `9718dceb841e8dc221cd3f785b083b30971b0cc410b4407a9524ba40d52c3255`

Runtime v29 prefix:
`build/conda-runtime-v29-market-time-base`.

- explicit spec:
  `build/launcherless-conda-explicit-20260729-v22.txt`, SHA-256
  `88266ae90c163470a9bcca09d4ef043bde2c33d5b8446f6536ff2df8cedabd46`;
- Python manifest SHA-256
  `2f8e27d1a2057a59dec104b177546e2fbe4aa2247ca926b5db5da3cf01afd77b`,
  tree
  `a05074aefeb9618a2411339117aeebe1a4d9974f48c48039138eb9b828a1f497`;
- prefix receipt SHA-256
  `4b57594869347b9bc1d1655be98afee7a3ab746208683a733b703eddaf41b509`,
  prefix receipt ID
  `7a70e7c9c40676c8863c2f571e5140db0fd609574c0e559aa47836121113b962`;
- final runtime receipt SHA-256
  `f4ff1d309a7800056e254fe506b61cea5a46e691308e9263d1a9ab701825e8c3`;
- final closure: 8,497 files, 19 distributions, tree
  `c9b90f183ce23c36cc1dfce88a1956384efd9c22a5c9ae70d05b12b5f695bacb`;
- exact `sys.path`: runtime `Lib`, `DLLs`, `governed-site-packages` only;
- installed CLI and validator origins are explicit and source-hash equal;
- `production_authorization=false`.

The v27 create process outlived its shell timeout and produced no selectable
command result. V28 used an incorrect Conda option order (`--file` after other
arguments), so its history admission failed. Preserve both as non-selectable.
V29 used the correct standard-user, offline, copy-only command shape:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe create --offline --copy `
  --prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v29-market-time-base `
  --file C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt `
  --yes --json
```

`C:\ProgramData\anaconda3` was executed read-only. All writes stayed under the
repo. The runtime-builder shell timed out after 300 seconds while its process
continued, but the complete receipt was produced and an independent installed
audit revalidated the runtime twice. This is local-quality evidence only.

## Installed audit and exact replay

Durable output:
`build/market-time-regime-audits/ch-market-time-regime-audit-20260730-v1.json`.

- receipt SHA-256
  `9288b10f535974bb512ff33fabb330d3e86fc23a069c3344397c5874b9fdfa68`;
- audit operation ID
  `64d7251d52289f6f8425f8f0c81a695409a878f39fd9963460ac31ba521b8449`;
- normalized command SHA-256
  `65fe9e1a85f1b7ba03c969b173c88baa85949fb79c65efc00bbf285d4e1af533`;
- status `STRUCTURE_VALID_TRANSITION_NOT_ADMITTED`;
- `evidence_files_verified=true`;
- eight blockers; transition, scientific, execution, promotion and production
  authorities all false.

Exact terminal command, also recorded in `OPERATIONS.md` section 12:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH='C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v29.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256='f4ff1d309a7800056e254fe506b61cea5a46e691308e9263d1a9ab701825e8c3'
$env:TEMP='C:\Users\jbattaglia\PFC_LT\build\market-time-regime-audit-temp-v29'
$env:TMP=$env:TEMP

C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v29-market-time-base\python.exe `
  -I -B -m pfc_shaping.cli.audit_ch_market_time_regime `
  --repo-root C:\Users\jbattaglia\PFC_LT `
  --contract C:\Users\jbattaglia\PFC_LT\.planning\phases\14-lt-audit-remediation\CH-MARKET-TIME-REGIME-CONTRACT-20260730.json `
  --expected-contract-sha256 71711ae80b64556b8deab88e70581e1c7a0ef7c684d672b2cb550f2058c19c25 `
  --runtime-receipt C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v29.json `
  --expected-runtime-receipt-sha256 f4ff1d309a7800056e254fe506b61cea5a46e691308e9263d1a9ab701825e8c3 `
  --output C:\Users\jbattaglia\PFC_LT\build\market-time-regime-audits\ch-market-time-regime-audit-20260730-v1.json
```

The exact retry exited zero and retained the same receipt SHA. IT/Ops found
that the first runbook draft omitted the two required runtime options; they
were added, and the corrected documented command above was executed before
the terminal verdict.

## Final verification

Scientific matrix:

```powershell
python -B -m pytest tests\test_ch_market_time_regime.py `
  tests\test_ch_lt_prospective_acquisition_plan.py `
  tests\test_ch_lt_hourly_capture_contract_v2.py `
  tests\test_ch_lt_prospective_capture_ledger.py `
  tests\test_ch_lt_estimand_contract.py `
  tests\test_assembler_profile_type.py tests\test_intraday_amplitude.py `
  tests\test_lt_ct_imports.py -q -p no:cacheprovider `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-market-time-scientific-v1
```

Result: `113 passed, 1 skipped in 179.09s`.

Runtime/packaging matrix:

```powershell
python -B -m pytest tests\test_lt_package_contract.py `
  tests\test_launcherless_local_runtime.py `
  tests\test_launcherless_runtime_admission.py `
  tests\test_snapshot_publisher_runtime_closure.py `
  -q -p no:cacheprovider `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\pytest-market-time-packaging-v1
```

Result: `70 passed in 49.08s`.

Scoped Ruff result: `All checks passed!` for validator, CLI, wrapper,
runtime-builder, wheel checker and the two changed test modules.

Terminal read-only re-roasts after correction:

- Security: P0/P1/P2 `0/0/0`;
- IT/Operations: P0/P1/P2 `0/0/0`;
- Quant/Data: P0/P1/P2 `0/0/0`.

Final repository audit:

- `git diff --check`: pass;
- staged file count: zero;
- protected `data/eex_forwards_history.parquet` remains deliberately modified
  from before this closure and was not touched or staged; exact SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- cwd and Git top-level were exactly `C:\Users\jbattaglia\PFC_LT` before every
  shell action;
- no commit or production promotion was performed.

## Remaining blockers and next work

The eight market-time prerequisites remain open, as do trusted time,
independent signatures, external CAS/WORM/fresh monotone HEAD, licensed source
semantics, Windows CI/ASR, SBOM/scans, supervision, alerting and independent
atomic transition/rollback drills.

Next work returns to fresh prospective CH data, governed rolling-origin
evaluation and the sealed T057 path before any new auditable candidate. Do not
infer native quarter-hour truth, read T057 prematurely or promote production.
