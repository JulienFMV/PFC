# Session handoff - 2026-07-28 - Hardened Conda payload runtime v18

## Status

- Canonical repo only: `C:\Users\jbattaglia\PFC_LT`.
- Branch: `fix/lt-audit-remediation`.
- Session-start HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree intentionally very dirty; no reset, clean, restore, stage or commit.
- `data/eex_forwards_history.parquet` was not touched; observed SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- No CT, Power BI, old `H:` repo, Playwright, project executable, Defender
  exception, admin action, production publication or production promotion.
- Runtime v18: `PASS` for local execution/quality inspection only.
- Production: strict `NO_GO`.

## Why v18 supersedes v17

The independent v17 re-roasts closed the original archive-to-prefix local byte
provenance P1 but demonstrated four local hardening findings. V18:

- rejects all Conda `pre-link`, `post-link`, `pre-unlink` and `post-unlink`
  spellings with hyphens or underscores;
- replaces the simple build-time reader with
  `read_stable_single_link_file`, the shared exact double-read mono-link
  primitive, in both the archive auditor and runtime builder;
- uses `paths_overlap_by_identity` for caller-held artifacts required outside
  the runtime prefix, including launch-time checks;
- removes the nonexistent `python311.zip` entry from `python311._pth` and from
  the exact launch contract.

The resulting live `sys.path` has only three entries: `Lib`, `DLLs` and exactly
one governed application root. The same-user import-before-admission P1 is not
closed by this change and remains a production blocker.

Changed source/test files for the v17-to-v18 correction:

- `scripts/build_launcherless_conda_archive_lock.py`;
- `scripts/build_launcherless_local_runtime.py`;
- `pfc_shaping/pipeline/governed_release_cli_contract.py`;
- `tests/test_launcherless_conda_archive_lock.py`;
- `tests/test_launcherless_local_runtime.py`;
- `tests/test_launcherless_runtime_admission.py`.

No scientific LT, monthly solver, CT, OMPEX, holdout or data code was changed.

## Reused immutable archive evidence

- archive-lock v2:
  `build/launcherless-conda-archive-lock-20260728-v15.json`;
- archive-lock SHA-256:
  `346c6edcce71dea86816ec6938a1d6a87872a3cd30a01984c8577e7800c33fdb`;
- archive-lock ID:
  `7fd90fd7a1a2f672da1890cb308870f5c7d1df5e38dbafe91f6da27ca486a813`;
- archive-set ID:
  `f3cd775e79648df9a9926a01eb97eadc8e951c5055c778ca9ca92b60bc8068e7`;
- 19 packages / 38,648,096 archive bytes;
- explicit spec:
  `build/launcherless-conda-explicit-20260728-v15.txt`;
- explicit-spec SHA-256:
  `c1da60c9c3474453ebc21580cecc15c556c9bcd82e9113c73bdaa62ec883e4eb`.

## Reproducible wheels Q/R

Both builds used `SOURCE_DATE_EPOCH=1783987200` and separate repo-local build,
dist and TEMP/TMP directories:

```powershell
python -B setup.py build --build-base build\wheel-build-q `
  bdist_wheel --dist-dir build\wheel-dist-q
python -B setup.py build --build-base build\wheel-build-r `
  bdist_wheel --dist-dir build\wheel-dist-r
```

Both wheels have 84 members / 464,396 bytes, are byte-identical at SHA-256
`5b2f993ef7d9408458ec6cb445a6daef4deace50666b8f12b695bc8b1ed26ed2`
and embed source revision
`756d0a594f868994bb532c25cf3f45060551f5c596a88e8ef3c479496425cf34`.
Both wheel audits are `PASS` with `promotion_eligible=false`. The direct
`setup.py` path remains an IT P2 pending clean PEP 517 builds with retained
frontend/backend identities.

## Fresh offline Conda prefix and receipts

Conda created the fresh namespace
`build/conda-runtime-v18-archive-payload-hardened-base` with exact
`--offline --copy --file`, no solve, network or elevation. It exited 0 in
209.9 seconds. The optional global
`C:\Users\jbattaglia\.conda\environments.txt` registry was not writable and
emitted a warning only; the repo-local prefix completed.

The caller-held pre-target-execution manifest is
`build/launcherless-python-runtime-manifest-20260728-v18-base.json`:

- SHA-256
  `05c36bc0947cf737e76276eab311a946089ef26f69d968799666ee48ab636a2a`;
- 6,285 files;
- tree SHA-256
  `3bccfafe4d6a30e95eb5e2ce3ffb6c6a36875131733a408c0222c31b1ea703a0`;
- capture completed in 144.1 seconds.

The hardened archive-to-prefix audit completed in 396.5 seconds. Prefix receipt
`build/launcherless-conda-prefix-receipt-20260728-v18.json` has:

- SHA-256
  `08e1d492917fd6a99d479f789ceb5f011a91bd53a779f74df5574927ff901bcc`;
- prefix receipt ID
  `1d54ae5e598e0b8081ed3ee8e2932c5dcfab5398fa58a9d39cfdae8f66fa10e4`;
- 5,859 archive-verified installed files;
- 406 generated non-runtime files;
- 19 packages / 6,285 total manifest files;
- `production_authorization=false`.

## Runtime v18 and installed admission

The first builder invocation used the current shell's `.conda\ppa_env` Python
and failed closed in 29.6 seconds with
`zstandard is required to audit .conda archive payloads`. It failed before any
receipt, closure or staging path existed. The terminal rerun used the existing
Anaconda Python that contains the required reader and completed in 1,683.2
seconds.

Runtime receipt `build/launcherless-runtime-receipt-20260728-v18.json`:

- schema `fmv_lt_launcherless_local_runtime.v4`;
- SHA-256
  `1124ba70f8e2903fe801a1ff5e39dd5df2afa362a38996a8dc7105895f29334d`;
- 19 distributions / 8,488 closure files;
- closure tree SHA-256
  `881abd1ebc68d38961dbb7bcf384d3aa042184bdc3c8efb7c00fde19a8ebf900`;
- `python311._pth` SHA-256
  `ad0983507279f6d9d54f667eb0f11de4fbe60c7361404c7b5e9854c36a7f8d90`;
- `local_quality_authorization=true`;
- `production_authorization=false`.

Installed admission command:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH = `
  'C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260728-v18.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256 = `
  '1124ba70f8e2903fe801a1ff5e39dd5df2afa362a38996a8dc7105895f29334d'
& 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v18-archive-payload-hardened-base\python.exe' `
  -I -B -m pfc_shaping.cli.governed_release --version
```

Result: exit 0 in 224.2 seconds, version `0.14.0`, exact embedded source
revision. Admission occurs before argparse exposes `--version`.

The live isolated `sys.path` is exactly:

1. `...\Lib`;
2. `...\DLLs`;
3. `...\governed-site-packages`.

The prefix root, checkout, user/system site and phantom ZIP root are absent;
the governed application-root count is exactly one.

## Final verification matrices

- focused hardening: `53 passed`;
- runtime/packaging: `157 passed, 12 skipped, 2 deselected`;
- governed release/script/quality/monthly manifests: `267 passed`;
- atomic promotion: `116 passed, 2 skipped`;
- candidate bundle/evidence/assembler: `65 passed`;
- external publication/snapshot: `66 passed`;
- anchor client/reference/bootstrap signer: `59 passed`;
- targeted Ruff: `All checks passed!`.

All test promotion/publication operations remained inside test namespaces. No
candidate, data or production transition was performed.

## Independent re-roasts

Final v18 Security, IT/Operations and Quant/Data addenda are pending and must be
appended before closing this session.

## Open gates

- Same-user target code is imported before its self-admission. Production
  requires a minimal independently admitted bootstrap/supervisor or an
  externally read-only/signature-enforced execution root.
- Archives, Conda executable, lock and receipts are not in an independently
  signed durable CAS/WORM; two clean Windows runner replays are absent.
- Conda prefix creation is not atomic; kill recovery, Job Object supervision
  and deterministic rename/recovery are absent.
- Immutable execution sidecars, Windows standard-user CI under FMV ASR,
  signed SBOM/provenance, structured logs/SLOs, active-runtime CAS and rollback
  drills remain required. V11 revocation and attempt5 incident closure remain
  independent IT work.
- The 406 generated `pip` files remain outside `sys.path` and are classified
  non-runtime, but production should remove them or reproduce/attest them.
- Fresh prospective point-in-time CH/EEX data, exact final EEX repricing,
  preregistered rolling-origin/T057/holdout evidence and probabilistic/scenario
  gates remain required before a new CH candidate.

Monthly solver authority, LT/CT separation and OMPEX benchmark-only status are
unchanged. Production remains strict `NO_GO`.
