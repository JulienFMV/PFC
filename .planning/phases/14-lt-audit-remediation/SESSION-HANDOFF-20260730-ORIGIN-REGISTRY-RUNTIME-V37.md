# Session Handoff - CH LT Origin Registry Protocol and Runtime v37

Date: 2026-07-30

## Terminal local conclusion

The local slice closes the previously missing *protocol* for globally unique
prospective origins, but deliberately does not claim that an external registry
exists. The hash-closed draft is
`CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V1-20260730.json`, schema
`ch_lt_origin_registry_protocol.v1`, SHA-256
`67e0b63a6eca5ec843847725b2f3097f7f0fbeeac615bcdb2b2c8be92e1b43fc`
and semantic protocol ID
`190044244ce09054f89ccda6b1387dbd3bbf331c500e6529a682d30a1100406c`.
It is outcome-blind, incomplete, locally hash-closed and non-authoritative.
Countable prospective origins remain zero.

Two independent governed project wheels are byte-identical at SHA-256
`62b1cb371f3e03e887488aebfe795c66a5b0b36332f009712e7428ef67bc6752`.
Launcherless runtime v37 is local-quality only. Its receipt SHA-256 is
`44a69422a06d9ee4ec9d01aa9989e1bb150b6e96b0188bad2bf906823d8a1dbb`.
The exact live `sys.path` is only runtime `Lib`, runtime `DLLs` and one
`governed-site-packages`; checkout, prefix root, user/system site, AppData and
phantom ZIP roots are absent. Production authorization is false.

The terminal installed test reports `1 passed`. The unified scientific,
runtime, packaging, publication and external-CAS fixture matrix reports
`823 passed, 18 skipped, 2 deselected`, with zero failure/error. No production
publication or promotion was attempted.

## Workstation and scope invariants

- cwd and Git root were checked before every shell action and remained exactly
  `C:\Users\jbattaglia\PFC_LT` on branch `fix/lt-audit-remediation`, HEAD
  `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- No command used `H:`, administrator elevation, ACL takeover, Defender/ASR
  exception, project executable, Playwright/browser runtime or mutable
  AppData/ProgramData path.
- Existing Anaconda and user Conda interpreters were executors only. Conda
  prefixes, package work areas, wheelhouses, pip caches, `TEMP`/`TMP`, pytest
  roots and receipts remained under repo `build/`.
- `data/eex_forwards_history.parquet` was not touched or staged; its SHA-256
  remained
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- Staged count remained zero. `pfc_shaping/ct/*` and Power BI status counts
  remained zero. Monthly solver authority, LT/CT separation and OMPEX
  benchmark-only status are unchanged.
- No outcome-bearing T057 evidence was read. The transitive test permits only
  `T057-OUTCOME-BLIND-TOMBSTONE-20260730.json`.

## Protocol semantics

The protocol freezes the following local design, pending independent FMV and
external-service authority:

- statistical cadence proposal: monthly;
- exact issuance schedule and UTC window: missing and must be externally
  frozen; CET/CEST abbreviation inference is forbidden;
- at most one countable origin per externally frozen slot;
- missed slots are not shifted, backfilled or reweighted;
- one external compare-and-append/CAS namespace rejects duplicate origins or
  slots, returns the identical receipt for an exact retry and rejects a
  divergent retry or alternate branch;
- local filesystem/SQLite is test-reference only and can never be authority;
- the registration request commits the origin/target/mask inventory,
  origin-available EEX inventory, prediction, scenarios, final mask, runtime,
  source/config/candidate identities and trusted origin time before the first
  target delivery begins;
- truth must remain unopened at registration. Truth opening, scoring,
  scientific admission, execution, publication, promotion and production are
  separate later transitions.

Official EEX web pages are planning rationale only. They do not satisfy an
evidence slot. FMV cadence approval, the official settlement/trading calendar,
trusted time, independent identities/keyrings/ACLs, builder-inaccessible
CAS/WORM/fresh HEAD, origin-available PIT inputs, sealed prediction/scenario/
mask commitments and independent reviews all remain missing.

## Files added or changed in this slice

- `.planning/phases/14-lt-audit-remediation/CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V1-20260730.json`
- `pfc_shaping/validation/ch_lt_origin_registry_protocol.py`
- `pfc_shaping/cli/audit_ch_lt_origin_registry_protocol.py`
- `scripts/audit_ch_lt_origin_registry_protocol.py`
- `tests/test_ch_lt_origin_registry_protocol.py`
- `tests/test_ch_lt_origin_registry_protocol_installed_v37.py`
- `pfc_shaping/package_contract.py`
- `tests/test_lt_package_contract.py`
- `scripts/run_workspace_local.py`
- `tests/test_run_workspace_local_script.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff.

No file was staged or committed.

## Standard-user / no-AppData closure

The historical runtime-v9 command naming an AppData publisher wheelhouse is
revoked and must never be submitted. The current builder independently rejects
every CLI build path outside canonical repo `build/`, and the workspace runner
rejects the same command before execution. The dedicated matrix

```powershell
python -B -m scripts.run_workspace_local --run-id noappdatav1 -- python -B -m pytest tests\test_launcherless_local_runtime.py tests\test_run_workspace_local_script.py -q -p no:cacheprovider -m "not slow"
```

reported `126 passed`. This needed no code change in this session because both
boundaries were already present and tested; the obsolete command was a stale
prompt, not a current workflow.

## Reproducible build and runtime evidence

### Project wheels

Both builds used
`C:\Users\jbattaglia\.conda\ppa_env\python.exe` read-only, with distinct
repo-local `TEMP`, pip cache, pycache and wheel directories, exact
`SOURCE_DATE_EPOCH=1783987200`, and:

```powershell
python.exe -B -m pip wheel --no-deps --no-build-isolation --no-cache-dir --wheel-dir <fresh-repo-build-dir> C:\Users\jbattaglia\PFC_LT
```

Selected witnesses:

- `build/wheel-dist-reg37e/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- `build/wheel-dist-reg37f/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- 562,637 bytes, 103 members, byte-identical SHA-256
  `62b1cb371f3e03e887488aebfe795c66a5b0b36332f009712e7428ef67bc6752`;
- source revision
  `317da4000afbcd0938af076659249088f648bd558ce5cbd23c0bae8db95da9ca`;
- both wheel audits: `PASS`; promotion eligible false.

### Offline Conda prefix and manifest

The fresh prefix was created offline from the exact repo-local explicit spec:

```powershell
C:\ProgramData\anaconda3\python.exe -B -m conda create --offline --copy --prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v37-origin-registry-base --file C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt --yes --json
```

`CONDARC`, `CONDA_PKGS_DIRS`, `CONDA_ENVS_PATH`, `TEMP` and `TMP` were all
explicit repo-local paths. The selected manifest is
`build/launcherless-python-runtime-manifest-20260730-v37-origin-registry-base-v2.json`,
SHA-256
`af7331d31565a237e6bfab36cbc3dcbb3f925557c4224b60fd655b0b606da4c1`,
6,285 files, tree
`e87c8e4587ab4ae67e2b4cdf0784f04336fa142cda86b013324bae7bbeedcc66`.
Runner `man37reg2` terminated at exit zero; receipt SHA-256
`daecb9634fad400ef41b359ce31edfbad778ad2ad53a949800bc392ef4b26d86`.

The prefix audit used archive-lock SHA-256
`020735fa21744772aedd71a7c99b33775ee27042c9a6c2dd953b15b6b9b720d8`
and explicit-spec SHA-256
`88266ae90c163470a9bcca09d4ef043bde2c33d5b8446f6536ff2df8cedabd46`.
Its receipt is
`build/launcherless-conda-prefix-receipt-20260730-v37-origin-registry.json`,
SHA-256
`6f4feecc2759e21a942a0f94534ff88b1ec76d7ce29d96d4ccc80e22290d0502`,
19 packages, prefix receipt ID
`78ebfe73c6c6b2db35968513ba685ea0d34200392b8482af63e5cbc731ffc5bd`
and archive set
`f3cd775e79648df9a9926a01eb97eadc8e951c5055c778ca9ca92b60bc8068e7`.
Runner `aud37reg` receipt SHA-256 is
`36a8874ee1023d46db7b608f98d05f0ae022a99fbd6bcfd235f5c78b09dffd67`.

### Runtime assembly

The exact assembly used only repo-local publisher and additional wheel inputs:

```powershell
C:\ProgramData\anaconda3\python.exe -B -m scripts.run_workspace_local --run-id asm37reg -- C:\ProgramData\anaconda3\python.exe -B -m scripts.build_launcherless_local_runtime --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v37-origin-registry-base --project-wheel C:\Users\jbattaglia\PFC_LT\build\wheel-dist-reg37e\fmv_pfc_lt-0.14.0-py3-none-any.whl --publisher-wheelhouse C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-wheelhouse --publisher-dependency-root C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-site-packages --publisher-receipt C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-dependency-closure-receipt.json --additional-wheel-directory C:\Users\jbattaglia\PFC_LT\build\launcherless-wheelhouse-20260727-v1 --python-runtime-manifest C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260730-v37-origin-registry-base-v2.json --expected-python-runtime-manifest-sha256 af7331d31565a237e6bfab36cbc3dcbb3f925557c4224b60fd655b0b606da4c1 --conda-prefix-build-receipt C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-prefix-receipt-20260730-v37-origin-registry.json --expected-conda-prefix-build-receipt-sha256 6f4feecc2759e21a942a0f94534ff88b1ec76d7ce29d96d4ccc80e22290d0502 --receipt-output C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v37-origin-registry.json --lock-path C:\Users\jbattaglia\PFC_LT\uv.lock
```

Result: `PASS` after 463.1 seconds. Execution-receipt SHA-256 is
`e4b0fc4fcf23a606773b0b8bbbcaefdcefa90883fdacdbf8e51d91208fe706ac`.
The runtime has 19 distributions and 8,507 governed files, tree
`5b5c6a64b41b6c36e9cc45f542aef2d86edb0023843e82b84ff26fdf46dc7826`.
Its exact ordered `sys.path` is:

1. `build/conda-runtime-v37-origin-registry-base/Lib`
2. `build/conda-runtime-v37-origin-registry-base/DLLs`
3. `build/conda-runtime-v37-origin-registry-base/governed-site-packages`

## Tests and terminal receipts

- initial protocol/package/runner focus: `155 passed`;
- scoped Ruff before build: `All checks passed`;
- installed v37 final run `reg37pt3`: `1 passed` in 21.11 seconds,
  source tree
  `d6e3f560d61ee73e65e6dd6d1cd2910623743d90a2ce1cc68c5cc5704e77e5ab`,
  receipt SHA-256
  `07e523a81f3e2adf260c26694c1733fa4e025a90d29faf8289cb3973d57de505`;
- unified `reg37mx`: `823 passed, 18 skipped, 2 deselected` in
  672.172 seconds, zero failure/error, same 532-file source tree, receipt
  SHA-256
  `27c645e7d2479e438c6fe25287ab8f87a77588d2f936c26bc8fef07b1ae116ac`;
- final scoped Ruff: pass;
- `git diff --check`: pass, with pre-existing LF-to-CRLF warnings only;
- staged count zero; protected parquet hash unchanged; CT and Power BI status
  counts zero; no task process remained.

## Negative/intermediate evidence not counted as PASS

- `reg37c` wheel build caller timed out and left no wheel. Its processes
  terminated without a selected artifact. Fresh `reg37e/f` are the witnesses.
- `man37reg` produced a byte-valid manifest, but its caller timeout closed the
  output handle and the runner ended `LAUNCH_FAILED` with `OSError [Errno 22]`.
  It is rejected. Fresh `man37reg2` is selected.
- direct packaged audit `reg37cli` failed closed because the workspace runner
  deliberately scrubbed ambient `PFC_LT_RUNTIME_RECEIPT_PATH`. This is expected
  authority isolation. The installed pytest proof injects the exact receipt
  only inside the test subprocess.
- `reg37pt` and `reg37pt2` demonstrated test-only expectation errors (canonical
  cwd, then exact NO-GO status string). Both were corrected without changing
  packaged bytes; only terminal `reg37pt3` is selected.

## Independent roast results

Terminal read-only results are Security/Governance `0/2/3`, IT/Operations
`0/1/4`, and Quant/Data `0/1/1` at P0/P1/P2. Security and Quant both reject
the missing executable cross-field ordering between the frozen schedule,
origin, commit deadline and first delivery. IT and Security reject the direct
v37 Conda build because it changed the timestamp of
`C:\Users\jbattaglia\.conda\environments.txt`. Quant additionally identifies
ambiguous request-ID/final-mask semantics. Residual P2 debt covers request and
mask binding, pre-import loaded-byte TOCTOU, bounded runner/process-tree
supervision, hash cost/resource telemetry, Windows power-loss/rollback and the
missing operational runbook.

This handoff is terminal as negative evidence only. Protocol v1 and runtime
v37 are superseded and not selectable; the successor must close both P1
findings with fresh protocol, wheels, prefix and runtime evidence.

## Strict NO-GO and next work

This slice creates no registry service, lease, origin, prediction, truth,
candidate, snapshot publication, promotion or production authority. Production
remains strict `NO_GO`.

Next highest-value work after terminal roasts is external coordination, not a
local authority simulation:

1. obtain FMV approval for the monthly issuance policy and exact externally
   frozen UTC schedule;
2. obtain official EEX calendar/settlement-event evidence and trusted time;
3. implement the independent compare-and-append/CAS registry with separate
   identities, ACLs/keyrings, builder-inaccessible storage, fresh HEAD and
   durable signed receipts;
4. bind a new prospective origin's PIT EEX/product inventory, prediction,
   scenarios and final target mask before delivery starts;
5. retain future native truth and evaluate multi-origin rolling performance
   plus a new independent holdout; never reuse T057;
6. continue final EEX delivered-candidate repricing, probabilistic/scenario
   calibration, Docker/CI/SBOM, bounded process supervision, telemetry,
   Windows power-loss recovery, observability and rollback drills.
