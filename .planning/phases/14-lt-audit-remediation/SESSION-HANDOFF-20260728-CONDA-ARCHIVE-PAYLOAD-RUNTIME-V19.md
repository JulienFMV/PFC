# Session handoff - 2026-07-28 - Root-receipt-hardened runtime v19

## Status and read order

Read this file after `AGENTS.md`, `.planning/HANDOFF.md`, the Phase 14 decision
log, the external-CAS RFC and the v18 handoff. This addendum supersedes v18 only
for current local-quality runtime selection.

- Repo: `C:\Users\jbattaglia\PFC_LT`, branch `fix/lt-audit-remediation`.
- Session-start HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree intentionally very dirty; no reset/clean/restore/stage/commit.
- No admin, network, Defender exception, Playwright, project EXE, CT, Power BI,
  old `H:` repo, candidate transition or production promotion.
- Protected parquet untouched; observed SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- V19 is `PASS_LOCAL_QUALITY`; production is strict `NO_GO`.

## Delta v18 to v19

The final v18 Security roast found one narrow physical-alias omission: after
reading and validating the top-level caller-held runtime receipt, launch
admission derived the runtime prefix but did not compare that receipt path
itself with the prefix by physical identity. All nested caller-held artifacts
already used the physical comparison.

V19 adds, immediately after prefix derivation:

```python
if _same_or_nested_runtime_path(receipt_path, prefix):
    raise ReleaseCliIdentityError(...)
```

`_same_or_nested_runtime_path` delegates to `paths_overlap_by_identity`, so the
check covers lexical, 8.3/drive/UNC identity and ancestor aliases while still
rejecting links/reparse points. The pure regression
`test_top_level_runtime_receipt_rejects_physical_prefix_alias` passes without a
temporary filesystem dependency.

Only these packaged/test bytes changed from v18:

- `pfc_shaping/pipeline/governed_release_cli_contract.py`;
- `tests/test_launcherless_runtime_admission.py`.

All v18 archive lifecycle, stable double-read, prefix identity and three-entry
`sys.path` hardening remains intact. No model, solver, shaping, data, CT, OMPEX
or holdout code changed.

## V19 artifacts

Reproducible wheels S/T:

- paths: `build/wheel-dist-s/fmv_pfc_lt-0.14.0-py3-none-any.whl` and
  `build/wheel-dist-t/fmv_pfc_lt-0.14.0-py3-none-any.whl`;
- 84 members / 464,412 bytes;
- byte-identical SHA-256
  `875c75a229cc8d97ecaa2948d831453549542502adba103b84eb176158f0f92f`;
- embedded source revision
  `05fb283040fea0a44de9be2565f1b6128b5c21614a8a994e4936e76a88825553`;
- both wheel audits `PASS`, `promotion_eligible=false`.

Fresh prefix:

- `build/conda-runtime-v19-root-receipt-hardened-base`;
- Conda exact `--offline --copy --file`, exit 0 in 213.8 seconds;
- optional global `C:\Users\jbattaglia\.conda\environments.txt` warning only,
  with no elevation or prefix failure.

Caller-held Python manifest:

- `build/launcherless-python-runtime-manifest-20260728-v19-base.json`;
- SHA-256
  `037a73b61e45a3a6d2cc87addd6b2e786404d5a0a74c658b26fae24f50ad01fb`;
- 6,285 files, tree
  `60383a302b3b10096d46fe226d1da481521d73d9547d1c43935444fd64d833eb`;
- capture exit 0 in 129.1 seconds.

Prefix receipt:

- `build/launcherless-conda-prefix-receipt-20260728-v19.json`;
- SHA-256
  `ac8461b1d67aab3249e163f40910f2e0c15895dd6fa7c6f49135bebc80f5cd39`;
- prefix ID
  `35ba7064504f3098be553e9f4b802bd29f50d0118bb41f911b7d8d95ced77f01`;
- 19 packages, 5,859 archive-verified files, 406 generated non-runtime
  files and 6,285 total files;
- hardened audit exit 0 in 286.8 seconds.

Runtime receipt:

- `build/launcherless-runtime-receipt-20260728-v19.json`;
- SHA-256
  `c55ebc97dc006aa93d7043f8b5944f8c336187e22db649c6fdda612c4ca4772e`;
- schema v4, `status=PASS`, `local_quality_authorization=true`,
  `production_authorization=false`;
- 8,488 closure files / 19 distributions;
- closure tree
  `0cf67c43b3246c0d4d048387fc8d55580a25ecda3407e2e11e6abe534c689af9`;
- `python311._pth` SHA-256
  `ad0983507279f6d9d54f667eb0f11de4fbe60c7361404c7b5e9854c36a7f8d90`;
- assembly exit 0 in 1,159.5 seconds.

Installed admission:

```powershell
$env:PFC_LT_RUNTIME_RECEIPT_PATH = `
  'C:\Users\jbattaglia\PFC_LT\build\launcherless-runtime-receipt-20260728-v19.json'
$env:PFC_LT_RUNTIME_RECEIPT_SHA256 = `
  'c55ebc97dc006aa93d7043f8b5944f8c336187e22db649c6fdda612c4ca4772e'
& 'C:\Users\jbattaglia\PFC_LT\build\conda-runtime-v19-root-receipt-hardened-base\python.exe' `
  -I -B -m pfc_shaping.cli.governed_release --version
```

Result: exit 0 in 138.1 seconds, version `0.14.0`, exact source revision. The
receipt is physically outside the prefix. Exact `sys.path` is only:

1. `...\Lib`;
2. `...\DLLs`;
3. `...\governed-site-packages`.

## Tests and Windows harness incident

- New root-receipt regression: `1 passed`.
- Targeted Ruff: `All checks passed!`.
- V18 terminal matrices immediately before the one-check v19 delta remain:
  focused `53 passed`; runtime/packaging `157 passed, 12 skipped, 2
  deselected`; governed release `267 passed`; atomic `116 passed, 2 skipped`;
  candidate `65 passed`; publication `66 + 59 passed`.

Two v19 focused full-suite attempts under `build/pytest-*`, one root
`.pytest-*` attempt and a later multi-test `build/scratch-*` attempt became
non-conclusive because pytest lost read access to its own `basetemp` with
`WinError 5`. A single `tmp_path` test in a fresh `scratch-*` namespace and the
new pure regression both passed. No ACL was changed, no admin right was
requested and inaccessible namespaces were not deleted. Do not report the
non-conclusive attempts as functional failures or as passing matrices.

## Independent read-only re-roasts

Security verified the exact v19 wheel/source/runtime bytes and the new check's
placement immediately after prefix derivation. Verdict: P0 none, root-receipt
P2 closed, no new P1/P2 demonstrated in the v19 code delta,
`PASS_LOCAL_TARGETED_CLOSURE`; the structural import-before-admission P1 still
blocks production.

IT/Operations verified wheels S/T, the fresh standard-user offline prefix,
manifest/receipts, exact three-entry `sys.path`, external real receipt path,
assembly and installed admission. Verdict: 0 P0 / 0 new P1 / 0 P2 for the v19
delta, `GO` local-quality and production `NO_GO`. The PEP 517 dual-build P2 and
six historical production workstreams remain open. Its RFC observation was
made before the v18/v19 RFC amendment was appended; the RFC is now updated.

Quant/Data proved by exhaustive wheel diff that only the contract,
`build_identity.py` and `RECORD` changed. Monthly solver, assembler, shaping,
estimand, EEX vintage, CT exclusion, OMPEX benchmark-only and T057 isolation
remain byte-identical. Verdict: packaging P0/P1/P2 = 0/0/0, `GO` for the local
non-authoritative packaging slice, and `NO_GO` for scientific validation,
promotion or production.

All three roasts explicitly classify the v19 full-suite ACL runs as
non-conclusive. None reran a target runtime, model, test or promotion.

## Residual gates

- P1 import-before-self-admission/same-user remains open.
- Independently signed CAS/WORM for Conda executable, archives, wheels and
  receipts; atomic prefix staging/recovery; execution sidecars; clean Windows
  CI/ASR; PEP 517 dual builds; signed SBOM/provenance; observability/SLO;
  active-runtime CAS and rollback drills remain open.
- The 406 generated `pip` files are declared non-runtime and outside exact
  `sys.path`, but must be removed or reproducibly attested for production.
- Test outputs are terminal console observations, not independently signed,
  hash-bound execution sidecars.
- Fresh prospective PIT CH/EEX data, exact EEX repricing, preregistered
  rolling-origin/T057/future holdouts and probabilistic/scenario validation are
  still required before a new candidate.

Monthly solver authority, LT/CT separation and OMPEX benchmark-only status are
unchanged. No production promotion occurred.
