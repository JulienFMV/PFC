# Session Handoff 2026-07-17 - Publisher Runtime TOCTOU Capture

## Canonical workspace

- Canonical repo: `C:\Users\jbattaglia\PFC_LT`
- Branch: `fix/lt-audit-remediation`
- HEAD: `2f68125bff869ccb21c1e20df0201ad024ed27d3`
- The old `H:` workspace is no longer canonical and caused repeated sandbox
  elevation prompts. Start the next Codex session from the `C:` folder itself.
- The worktree is intentionally very dirty. Do not reset, clean, restore or
  mass-stage it.
- `data/eex_forwards_history.parquet` is a large local data modification. Do
  not revert, stage or commit it.

## Persistent product goal

Deliver a Swiss FMV reference PFC that is scientifically defensible and
IT-industrializable: one fail-closed LT pipeline, fresh point-in-time traceable
data, exact final EEX product repricing, robust rolling-origin and future
holdout shaping validation, coherent probabilistic scenarios, atomic
manifest-governed promotion, and reproducible Docker/CI packaging with
observability and rollback. Every structural step must receive independent
Security, IT/Operations and Quant/Data roasts.

Production remains `NO-GO`. No real independent external anchor, signed IT
release attestation, fresh prospective raw acquisition or promotion-ready CH
candidate has been proven.

## Invariants

- Do not touch `pfc_shaping/ct/*`.
- Do not touch `powerbi/data/*` or `powerbi/PFC_QA.*`.
- Never commit heavy desk data.
- With `monthly_level_authority="solver"`, the monthly solver owns BASE
  levels. Never patch individual months after solving.
- OMPEX remains benchmark-only and forbidden as model input.
- Active trust authorizes new work; historical keyrings authenticate exact
  committed replay only.
- Keep production fail-closed and curate code/tests/docs only.

## Completed in the current uncommitted stack

### Wheel and zipapp provenance

- `scripts/build_snapshot_publisher_runtime_closure.py` builds directly from
  the eleven exact CPython 3.11 Windows wheels selected from `uv.lock`.
- Archive hashes/sizes, `METADATA`, `WHEEL`, full `RECORD`, tags and
  installed paths are replayed. `.pth` is forbidden; recorded NumPy bytecode
  and nested-wheel residue are verified then excluded.
- Real wheelhouse:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252`
- Real closure bundle:
  `C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511`
- Dependency root is the bundle's `site-packages` child. The canonical receipt
  is its sibling `dependency-closure-receipt.json`; do not point the root at
  the bundle parent.
- Bound tree SHA-256:
  `0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`
- Receipt SHA-256:
  `69c0843b4b9a2e202c838ed04027eec1e5008ebec04c0b3c30f37217b0fb45a1`
- Prior real zipapp SHA-256 before the latest runtime schema change:
  `272f7b48263ec4e4855dd4eb0035b5de5b33fa3cb282912f501518acd09181bf`

### Process-lifetime trust and durable evidence

- Public trust files and historical inventories are captured as immutable
  process-lifetime bytes; downstream verifiers no longer reopen mutable trust.
- Durable evidence never replaces a divergent target, including truncated or
  noncanonical bytes.
- Exact authoritative bytes are staged only under
  `.pfc-repair-candidates` and return
  `durable_artifact_repair.v2` / `LOCAL_ARTIFACT_REPAIR_REQUIRED`.
- A noncooperating writer attack test proves that the target remains untouched.
- Targeted durable matrices:
  - `8 passed, 37 deselected`
  - `45 passed` for the complete bootstrap-signer/anchor-client pair.

### Dependency verification/import TOCTOU

Current files:

- `pfc_shaping/publisher_runtime_admission.py`
- `scripts/build_snapshot_publisher_zipapp.py`
- `deploy/publisher/runtime-contract.json`
- `deploy/publisher/README.md`
- `pfc_shaping/tools/OPERATIONS.md`
- `tests/test_snapshot_publisher_artifact.py`

Implemented behavior:

- Runtime schema is now `fmv_lt_snapshot_publisher_runtime.v4`.
- The environment dependency root is verification input only and is never
  appended to `sys.path`.
- After exact source verification, every file is copied through a stable open
  descriptor into a random process-private root.
- Source tree, captured tree and captured distribution inventory are checked
  against the embedded contract.
- The capture is made read-only at application level and rehashed before only
  that root is added to `sys.path`.
- Contract policy explicitly keeps storage-enforced read-only host isolation as
  a required external signed IT attestation.
- Per-file `fsync` was removed from this ephemeral copy after a real run showed
  multi-minute startup cost. Integrity is still checked by full post-copy
  rehash; crash durability is not a requirement for a disposable runtime copy.

Attack tests added:

- Mutation of the source after capture cannot change captured bytes.
- Mutation during copy fails admission.

Current verification:

- targeted capture/contract: `3 passed, 13 deselected`
- lightweight publisher artifact matrix: `9 passed, 7 skipped`
- Ruff targeted: `All checks passed`
- targeted `py_compile`: pass

## Real artifact test note

An initial real test used the bundle parent as dependency root. It correctly
failed because the receipt and zipapp then appeared as foreign closure files.
This was an operator path error, not a runtime regression.

The corrected reproducibility test was started with:

```powershell
$env:PFC_TEST_PUBLISHER_WHEELHOUSE='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-wheelhouse-cp311-efcea252'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_ROOT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\site-packages'
$env:PFC_TEST_PUBLISHER_DEPENDENCY_RECEIPT='C:\Users\jbattaglia\AppData\Local\pfc-lt-build\publisher-closure-d2d9b7fb0ad4443f93456b7bcf466511\dependency-closure-receipt.json'
python -m pytest -q tests\test_snapshot_publisher_artifact.py -k "reproducible_runnable_and_signer_free"
```

That process embedded the pre-optimization per-file `fsync` implementation
and completed successfully: `1 passed, 15 deselected in 723.68 s`. It proves
the real end-to-end path but is not performance evidence for current source.
Rerun against the optimized source and record the new duration. The unrelated
long-lived Python process PID 3572 must not be touched.

## Immediate next actions

1. Rerun the corrected real zipapp test against current source and record
   runtime plus artifact SHA-256.
2. Run the real dependency-tree attack and tampered-zipapp tests.
3. Add/confirm an explicit test that `verify_publisher_runtime` appends only
   the captured root and never the environment source root.
4. Run Ruff, `git diff --check`, runtime-closure tests, publisher tests,
   snapshot publication matrices and package contract tests.
5. Launch independent read-only Security and IT/Operations roasts. Require
   concrete findings with file/line evidence; do not self-certify.
6. Update the external-CAS RFC, decision log and this handoff with accepted
   findings and exact commands/results.
7. Keep production `NO-GO`; after packaging security closes, return to fresh
   prospective data, T057 future holdout and a fresh auditable CH candidate.

## New-session permission hygiene

Open `C:\Users\jbattaglia\PFC_LT` using **File > Open Folder** in a new VS
Code window, then start a new Codex chat there. Verify that the Codex workspace
root/cwd is `C:\Users\jbattaglia\PFC_LT` before running commands. If it still
shows `H:`, close that chat and create a new one from the `C:` window.
