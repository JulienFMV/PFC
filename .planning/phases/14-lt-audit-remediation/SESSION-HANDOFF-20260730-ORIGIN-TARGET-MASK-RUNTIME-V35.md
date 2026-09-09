# Session handoff — CH LT origin/target/mask inventory and runtime v35

Date: 2026-07-30

Branch: `fix/lt-audit-remediation`

HEAD observed before and after the slice:
`2f68125bff869ccb21c1e20df0201ad024ed27d3`

Production status: **strict `NO_GO`**. No candidate, snapshot, publication,
promotion, T057 outcome or production transition was consumed.

## Workstation boundary

Every shell action was guarded by exact cwd and Git top-level equality with
`C:\Users\jbattaglia\PFC_LT`. No `H:`, administrator right, ACL takeover,
Defender/ASR exception, project executable, Playwright or browser was used.
All mutable Conda, pip, pytest, Ruff, wheel, runtime, cache and temporary
paths remained below repo `build/`.

The quoted historical v9 command containing an `AppData` publisher wheelhouse
is obsolete and must never be submitted. The direct runtime builder and the
workspace runner independently reject every explicit launcherless build path
outside canonical repo `build/`. Final boundary regression `wsboundary36`
reported `115 passed`; receipt SHA-256 is
`31719a5e3d46963e11a9fb719b58cb7a69b0eeee8c0821652339dc74b25c042c`.
It was followed and superseded for current source bytes by `otm35u3` and
`otm35mx1` below.

## Structural inventory decision

New package files:

- `pfc_shaping/validation/ch_lt_origin_target_mask_inventory.py`;
- `pfc_shaping/cli/audit_ch_lt_origin_target_mask_inventory.py`.

New checkout/test files:

- `scripts/audit_ch_lt_origin_target_mask_inventory.py`;
- `tests/test_ch_lt_origin_target_mask_inventory.py`;
- `tests/test_ch_lt_origin_target_mask_inventory_installed_v35.py`.

Updated files:

- `pfc_shaping/validation/ch_lt_estimand_contract.py` exports its frozen path,
  SHA-256 and contract ID;
- `pfc_shaping/package_contract.py` includes the new installed files;
- `scripts/run_workspace_local.py` allowlists and path-bounds checkout and
  installed inventory commands;
- `tests/test_run_workspace_local_script.py` and
  `tests/test_lt_package_contract.py` cover those boundaries.

One caller-supplied canonical UTC origin deterministically produces M01..M36
as full Europe/Zurich delivery months. Each target binds local and UTC
boundaries, lead bucket, expected hourly count and four-times quarter-hour
count. March and October DST months are 743/2,972 and 745/2,980 intervals.
The origin time is never inferred; FMV issuance cadence/calendar remain
`UNSET_REQUIRES_FMV_APPROVAL`.

The artifact is deliberately outcome-blind and non-countable. It contains no
prediction, score, loss, truth value or outcome path. `MARKET_CONSISTENCY` is
pending an origin-available CH EEX product inventory; `HOURLY_SHAPE` is
pending post-delivery direct native CH truth; `QUARTER_HOURLY_SHAPE` remains
`UNSUPPORTED_MARKET_TRANSITION_NOT_ADMITTED`; probabilistic scenarios remain
unsupported. The final truth label is
`NOT_AVAILABLE_OR_NOT_BOUND_NOT_READ`; the earlier local v1 wording
`SEALED_UNOPENED_NOT_READ` and artifact SHA-256 `59be0aac...abdd4` are
superseded and must not be selected.

The validator machine-checks the current estimand horizon/buckets and unique
origin unit, the candidate core v3, the contrast-aware power design and the
60-minute/unadmitted Swiss market-time regime. Every scientific, execution,
publication, promotion, T057 and production authority remains `false`.

Current installed-CLI artifact:

- path:
  `build/ch-lt-origin-target-mask-inventories/structural-dry-run-20260730T120000Z-v2.json`;
- SHA-256:
  `155ff555c8086b2dd671e63f779b0fa4b33ac12d7502ffbde7d0a3fe42876aa1`;
- bytes: `37,053`;
- inventory ID:
  `5a3851c1fd8cb7aed5827a05fe055d9ad151cf49fb090df03793126b09866a63`;
- origin ID:
  `fc4e9ac24f934897510c4b9bab802513a312fa1abd5dd013b0650795e4fe3388`;
- target count: `36`;
- `governance_bindings_verified=true`;
- `countable_prospective_origin=false`;
- production and promotion: `false`.

## Reproducible wheel and launcherless runtime v35

Two offline setuptools wheels were built with user Conda Python read-only,
`PIP_NO_INDEX=1`, `--no-build-isolation`, and every mutable path below
`build/`:

- `build/wheel-dist-otm35probe/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- `build/wheel-dist-otm35b/fmv_pfc_lt-0.14.0-py3-none-any.whl`.

They are byte-identical: 101 members / 555,837 bytes / SHA-256
`eb37d6a95179eb5468ce1afb93f1a3829521f2e788257d30f26068322f7df857`;
source revision is
`55396deaeb4529539420cf3184dd0160adf0865c06dbd297c2c519885037bcad`.
Wheel audit `whl35pr1` passed; receipt SHA-256 is
`b7d82e112e73f790ba18357b59831827d23ac4658befcceb744ee82dbd7cdabe`.

Selected prefix:
`build/conda-runtime-v35-origin-mask-final-base`.

- Python manifest SHA-256:
  `d8815297c0255f1b775e012b91e17e792764f53e8e090aee37a534e575a183f2`;
  6,285 files, tree
  `defc25f14a486bcf9e94895de010b535552bdfe0642006144d7d09e9b60b363d`;
- archive-derived prefix receipt SHA-256:
  `dc1d101bfe2cba03462db7871f7872fccc9787c2ae806a2928693ae88f0a989b`;
  19 packages, archive set
  `f3cd775e79648df9a9926a01eb97eadc8e951c5055c778ca9ca92b60bc8068e7`;
- runtime receipt SHA-256:
  `e4623969ff6a893b5c6f1aeb9d4ac0ebd618741da5a12e32bec69cf36bfea5c7`;
- governed closure: 8,505 files / 19 distributions / tree
  `7f9afd7b68117d1d2560d59a13b7e02142ce957baedd7809cd3410188258bfa2`;
- exact `sys.path`: runtime `Lib`, runtime `DLLs`, then one
  `governed-site-packages`; no checkout root;
- local quality `true`; production authorization `false`.

Rejected build attempts are negative evidence only:

- first v35 prefix used Conda hardlinks; `man35pk` failed mono-link admission;
- second prefix was copied but its command history placed `--file` last;
  `aud35cp2` rejected the history grammar;
- `aud35cp` used the minimal pytest runtime without `zstandard` and failed;
- the final prefix used exact offline command grammar
  `create --offline --copy --prefix <prefix> --file <spec> --yes --json`;
- `otm35ib1` proved that the workspace runner scrubs ambient runtime-receipt
  authority; installed admission is therefore exercised inside the final
  pytest receipt with explicit child environment, as in the existing v31/v33
  installed tests;
- `otm35u2` exposed a test-only cwd mismatch and is superseded by `otm35u3`.

No rejected prefix or nonzero receipt may be cited as success or repaired in
place.

## Exact principal commands

All commands below followed the mandatory cwd/Git-root guard.

```powershell
C:\Users\jbattaglia\.conda\ppa_env\python.exe -B -m pip wheel --no-deps --no-build-isolation --no-cache-dir --wheel-dir <fresh-build-dist> C:\Users\jbattaglia\PFC_LT

C:\ProgramData\anaconda3\python.exe -B -m conda create --offline --copy --prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v35-origin-mask-final-base --file C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt --yes --json

build\pytest-runtime-v1\python.exe -I -B -m scripts.run_workspace_local --run-id man35fn -- build\pytest-runtime-v1\python.exe -B -m scripts.build_launcherless_python_runtime_manifest --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v35-origin-mask-final-base --output C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260730-v35-origin-mask-final-base.json

C:\ProgramData\anaconda3\python.exe -B -m scripts.run_workspace_local --run-id aud35fn -- C:\ProgramData\anaconda3\python.exe -B -m scripts.build_launcherless_conda_archive_lock audit-prefix --lock C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-archive-lock-20260729-v22.json --expected-lock-sha256 020735fa21744772aedd71a7c99b33775ee27042c9a6c2dd953b15b6b9b720d8 --explicit-spec C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-explicit-20260729-v22.txt --expected-explicit-spec-sha256 88266ae90c163470a9bcca09d4ef043bde2c33d5b8446f6536ff2df8cedabd46 --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v35-origin-mask-final-base --python-runtime-manifest C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260730-v35-origin-mask-final-base.json --expected-python-runtime-manifest-sha256 d8815297c0255f1b775e012b91e17e792764f53e8e090aee37a534e575a183f2 --output C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-prefix-receipt-20260730-v35-origin-mask-final.json

C:\ProgramData\anaconda3\python.exe -B -m scripts.run_workspace_local --run-id asm35fn -- C:\ProgramData\anaconda3\python.exe -B -m scripts.build_launcherless_local_runtime --runtime-prefix C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v35-origin-mask-final-base --project-wheel C:\Users\jbattaglia\PFC_LT\build\wheel-dist-otm35probe\fmv_pfc_lt-0.14.0-py3-none-any.whl --publisher-wheelhouse C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-wheelhouse --publisher-dependency-root C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-site-packages --publisher-receipt C:\Users\jbattaglia\PFC_LT\build\runtime-inputs-20260728-repolocal-v1\publisher-dependency-closure-receipt.json --additional-wheel-directory C:\Users\jbattaglia\PFC_LT\build\launcherless-wheelhouse-20260727-v1 --python-runtime-manifest C:\Users\jbattaglia\PFC_LT\build\launcherless-python-runtime-manifest-20260730-v35-origin-mask-final-base.json --expected-python-runtime-manifest-sha256 d8815297c0255f1b775e012b91e17e792764f53e8e090aee37a534e575a183f2 --conda-prefix-build-receipt C:\Users\jbattaglia\PFC_LT\build\launcherless-conda-prefix-receipt-20260730-v35-origin-mask-final.json --expected-conda-prefix-build-receipt-sha256 dc1d101bfe2cba03462db7871f7872fccc9787c2ae806a2928693ae88f0a989b --receipt-output C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260730-v35-origin-mask-final.json --lock-path C:\Users\jbattaglia\PFC_LT\uv.lock
```

The installed CLI build/audit used runtime v35 under `-I -B -m`, with
`PFC_LT_RUNTIME_RECEIPT_PATH` and its exact SHA-256 supplied explicitly to the
child. The artifact path/hash are recorded above. The installed v35 smoke
replays that route inside the final pytest receipts.

## Final tests and receipts

- `otm35u3`: focused inventory, installed CLI, runner, launcherless,
  packaging and governance chain; `259 passed`; source tree
  `45e858224e55e7cd1ea6480af556ae048fd245ddf776620092eb92207af8747c`;
  receipt SHA-256
  `e6e15b06458651e8044c50fa08461c6855c7ee3fbc5bf5d8a74d152aa6f25292`.
- `otm35mx1`: unified scientific/runtime/packaging/publication/CAS matrix;
  `797 passed, 18 skipped, 2 deselected`, zero failure/error, 321.97 seconds;
  same source tree; receipt SHA-256
  `3e759e78596168b211e03c0f1cd9523cde9609790e3028c18fe14d43f514eec1`.
- Ruff: `python -B -m ruff check --no-cache <changed files>` with `TEMP/TMP`
  below `build/`: `All checks passed`. The earlier `otm36r1` missing-Ruff
  receipt is historical negative evidence, not final lint evidence.
- `git diff --check`: exit zero; warnings are LF/CRLF notices only.
- staged count: `0`.
- CT/Power BI status count: `0`.
- protected `data/eex_forwards_history.parquet` SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

The 18 skips and two slow deselections remain the previously documented
approved-publisher-wheelhouse, standard-user symlink/CI and real slow zipapp
qualification cases. They grant no authority.

## Independent terminal roasts

Terminal re-roasts are recorded in D194. They are read-only and do not access
outcome-bearing T057 material. Production remains strict `NO_GO` regardless
of local-slice verdict.

## Remaining scientific and production blockers

- This is one structural dry-run origin, not an FMV-approved monthly origin
  ledger. There is no cross-artifact uniqueness lease/registry, issuance
  cadence/calendar, origin-available EEX product inventory, sealed prediction,
  final complete-case mask, post-delivery native truth or independent external
  signature/trusted-time/CAS/WORM/fresh HEAD.
- Independent rolling-origin count remains zero. The exact dependence/power
  threshold cannot be evaluated until real pre-sealed predictions and mature
  post-origin outcomes accumulate.
- Legacy T057 remains permanently ineligible for confirmatory reuse. A new
  independent future holdout is required.
- Swiss quarter-hour market truth remains unsupported until exact go-live and
  first-delivery UTC boundary are independently admitted; planned dates do not
  upgrade truth.
- Final CH candidate repricing, probabilistic/scenario calibration, profile
  economics, container/CI/SBOM, global timeout/process-tree supervision,
  resource telemetry, external publication CAS, rollback and power-loss drills
  remain open.
- The monthly solver remains sole level authority; LT remains independent of
  CT; OMPEX remains benchmark-only.

Next best action: design the externally governed multi-origin registry and
FMV-approved issuance cadence, then capture the first real origin's PIT EEX
inventory plus prediction commitments before any future truth is opened.
