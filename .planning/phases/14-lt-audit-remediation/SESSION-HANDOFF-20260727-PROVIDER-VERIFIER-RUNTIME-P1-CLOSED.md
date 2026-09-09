# Session Handoff - Provider verifier runtime P1 closed (2026-07-27)

## Status

- Canonical repo: `C:\Users\jbattaglia\PFC_LT`.
- Branch: `fix/lt-audit-remediation`.
- HEAD observed before and after the slice:
  `2f68125bff869ccb21c1e20df0201ad024ed27d3`.
- Worktree remains intentionally very dirty. Do not reset, clean or restore.
- No commit, staging, snapshot publication, candidate promotion or production
  promotion occurred.
- Software/documentation verdict for this bounded provider-verifier TOCTOU
  slice: no demonstrated P0/P1 after final Security and IT/Operations re-roast.
- Production/infrastructure/science verdict: strict `NO_GO`.

## Protected invariants

- `data/eex_forwards_history.parquet` was observed pre-existing modified and
  was not touched or staged. Its SHA-256 remained
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.
- No `pfc_shaping/ct/*` or Power BI file was changed.
- Monthly solver remains the authority of monthly level.
- OMPEX remains benchmark-only and is never a model input.
- The local verifier cannot authorize production.

## Closed demonstrated findings

The provider verifier now:

1. lives in a positive-inventory isolated zipapp and is excluded from the
   general governed LT wheel;
2. rejects direct caller-supplied runtime claims; checkout script wrappers are
   fail-closed stubs;
3. captures exact artifact and dependency bytes before import;
4. binds a one-shot worker capability to parent PID, source bytes, captured
   artifact, scratch and supervisor roots;
5. creates the Windows worker suspended, assigns it to a kill-on-close Job
   Object, then resumes it;
6. terminates on timeout, interruption and supervisor error, with bounded
   cleanup retries;
7. passes only a minimal environment and drops caller secrets;
8. requires exact `sys.path` counts `1/0/1/0` for captured/source artifact and
   captured/source dependency root;
9. pre-admits both outputs against every protected root and revalidates all
   acquisition/prior evidence after the business audit before emitting the
   runtime receipt;
10. separates direct business evidence from an unsigned hash-bound runtime
    observation, with both authority flags false;
11. limits zip members to 16 MiB each and 64 MiB total uncompressed;
12. documents the writer namespace as mutable quarantine and requires an
    independently ordered WORM/external-CAS retention handoff.

The final IT/Operations re-roast specifically confirmed no P0/P1 after the
runbook was corrected to avoid the impossible claim that a same-identity
hardlink writer could delete its temporary name while lacking delete authority
over the published name.

## Exact changed files in this slice

- `deploy/verifier/README.md`
- `deploy/verifier/runtime-contract.json`
- `pfc_shaping/verifier_runtime_admission.py`
- `pfc_shaping/publisher_runtime_admission.py`
- `pfc_shaping/cli/audit_provider_acquisition_quarantine.py`
- `pfc_shaping/cli/audit_legacy_provider_resolution.py`
- `pfc_shaping/package_contract.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `scripts/audit_provider_acquisition_quarantine.py`
- `scripts/audit_legacy_provider_resolution.py`
- `scripts/build_lt_provider_verifier_zipapp.py`
- `scripts/check_lt_wheel_contract.py`
- `tests/test_lt_provider_verifier_artifact.py`
- `tests/test_audit_provider_acquisition_quarantine_script.py`
- `tests/test_audit_legacy_provider_resolution_script.py`
- `tests/test_lt_package_contract.py`
- `tests/test_snapshot_publisher_artifact.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`
- `.planning/HANDOFF.md`
- this handoff.

Other dirty files predate or belong to adjacent Phase 14 work. Do not infer
ownership of the whole dirty worktree from this list.

## Reproducible artifacts

Provider verifier:

- `build/provider-verifier-20260727-v13.pyz`
- `build/provider-verifier-20260727-v14.pyz`
- each 63,939 bytes, 17 members;
- identical SHA-256:
  `b9afe8358492658214d4bcf01ad1207084ec992df545611c9cb0f02cd0dfa3b5`;
- source revision:
  `dac885fa91157cbadca37c6525d77b986b212ce6341b100cee59514bcf8101c2`;
- dependency tree SHA-256:
  `0ecb7997997cc124375e92614ca08d9c5274c683c6738448b9bd3c5eafaf78f1`.

Governed LT wheels:

- `build/runtime-verifier-wheel-20260727-c/fmv_pfc_lt-0.14.0-py3-none-any.whl`
- `build/runtime-verifier-wheel-20260727-d/fmv_pfc_lt-0.14.0-py3-none-any.whl`
- each 440,221 bytes, 81 members;
- identical SHA-256:
  `841113ef3134113464c2749ccbe0860eacb5ed3bac550ea85f49cf229c637f95`;
- embedded source revision:
  `deb555b1880942518d73cf86af395864ad238e8c3eec6bb70b46307bfcc02dc8`;
- verifier runtime and provider/legacy audit modules are absent;
- wheel contract reports `PASS`, `promotion_eligible=false`.

## Real runtime and business evidence

The real v14 runtime check returned exit `0` in `159.5221386s`:

- `captured_artifact_sys_path_count=1`;
- `source_artifact_sys_path_count=0`;
- `captured_dependency_root_sys_path_count=1`;
- `source_dependency_root_sys_path_count=0`;
- `runtime_authority=false`;
- `production_authorization=false`;
- zero new `vv-*` residue.

Positive pinned v2 fixture:

- directory:
  `output/phase14/provider_verifier_e2e_fixture_20260727/acquisition-v2-pinned`;
- manifest SHA-256:
  `87396febd322e3fbb519a3fbc04a7312edbcd73dc35763068d6304669cdb05f0`;
- business audit:
  `output/phase14/provider_verifier_e2e_fixture_20260727/audit-v2-pinned-v14.json`;
- audit SHA-256:
  `42ed85c7d8e82cd96b0d49b4882fe475d68484a990fbc9820fc46381b36c8b7c`;
- runtime receipt:
  `output/phase14/provider_verifier_e2e_fixture_20260727/audit-v2-pinned-v14-runtime-receipt.json`;
- receipt SHA-256:
  `18e734d6f3b15819e34bfe6632066837cf274534f75658eca28e11b28dbd9e7d`;
- exit `0`, wall `178.3092308s`, zero residue.

The ambient `acquisition-v2` fixture intentionally fails under the pinned
closure because its runtime fingerprint differs. It is retained as negative
evidence and was not relabeled.

Real legacy failure:

- acquisition:
  `output/phase14/prospective_public_quarantine_20260724/epex_ch_recent`;
- manifest SHA-256:
  `160aed6566e8edb8d4fdb7edcad6a0ff54e67419255bce2cd7a284b47792a372`;
- prior audit SHA-256:
  `3dc5aacef13b29a481629c360a3540b630e9089aa6c7117e459bc192fd22f8e4`;
- expected exit `50`: `provider transform runtime fingerprint mismatch`;
- wall `149.6808129s`;
- no business output, no receipt and zero residue.

## Exact final verification commands and results

Canonical guard, run separately before every command:

```powershell
$expected='C:\Users\jbattaglia\PFC_LT'
$cwd=(Get-Location).Path
$root=(git rev-parse --show-toplevel).Trim().Replace('/','\')
if ($cwd -cne $expected -or $root -cne $expected) {
  throw "Workspace mismatch: cwd=$cwd root=$root"
}
```

Real optimized current-source publisher proof used the retained wheelhouse,
dependency root and sibling receipt from the 2026-07-17 handoff. A local pytest
bootstrap under ignored `build/` supplied those three environment values so
the already-approved direct pytest prefix could avoid the defective sandbox
ACL:

```powershell
python -m pytest -p build.publisher_real_paths_plugin `
  tests\test_snapshot_publisher_artifact.py::test_optimized_publisher_zipapp_prepares_real_provider_raw_v3_bundle `
  -q -p no:cacheprovider `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\p10
```

Result: `1 passed in 348.22s`.

Final runtime/packaging:

```powershell
python -m pytest `
  tests\test_lt_provider_verifier_artifact.py `
  tests\test_snapshot_publisher_artifact.py `
  tests\test_snapshot_publisher_runtime_closure.py `
  tests\test_lt_package_contract.py `
  tests\test_audit_provider_acquisition_quarantine_script.py `
  tests\test_audit_legacy_provider_resolution_script.py `
  -q -p no:cacheprovider -m "not slow" `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\p11
```

Result: `100 passed, 12 skipped, 2 deselected in 116.72s`.

Final publication/CAS/candidate:

```powershell
python -m pytest `
  tests\test_snapshot_publication_external_contract.py `
  tests\test_snapshot_anchor_client.py `
  tests\test_snapshot_anchor_reference.py `
  tests\test_snapshot_bootstrap_signer.py `
  tests\test_atomic_promotion.py `
  tests\test_candidate_bundle.py `
  tests\test_candidate_evidence.py `
  tests\test_candidate_evidence_assembler.py `
  tests\test_governed_release.py `
  tests\test_run_governed_lt_release_script.py `
  tests\test_check_monthly_curve_promotion_from_manifests.py `
  -q -p no:cacheprovider `
  --basetemp C:\Users\jbattaglia\PFC_LT\build\p12
```

Result: `499 passed, 2 skipped in 554.17s`.

Additional results retained from the same current-source closure:

- acquisition/replay/panel: `178 passed, 3 skipped in 278.11s`; one
  pre-existing timezone Period conversion warning;
- initial broad publication/candidate run: `437 passed, 2 skipped, 3 failed`
  only because an overlong Windows basetemp caused `WinError 3`; the three exact
  tests then passed with short `build/p8`, and the complete short-path rerun is
  the `499 passed` result above;
- atomic-link race focused test: `1 passed, 53 deselected in 0.27s`;
- targeted Ruff: `All checks passed!`;
- `git diff --check`: exit `0`, only informational LF-to-CRLF warnings.

## Independent read-only roasts

Security found no residual P0/P1 after direct runtime-claim forgery, output
poisoning, worker-orphan and import-TOCTOU findings were closed. Its remaining
P2 notes are same-token capability construction and host isolation, both
explicitly outside local runtime authority.

IT/Operations initially found the missing runbook, then demonstrated three
documentation errors: understated durable-writer rights, wrong supervisor/
scratch disjointness, and impossible identical audit JSON paths. After those
were corrected it further demonstrated the hardlink/DACL immutability conflict
and an unsafe attestation order. The final runbook now says mutable local
quarantine and orders retention as ingest, hash, seal/revoke, control probes,
stable final hash, then signed receipt. Final re-roast: no P0/P1 documentary
finding remains.

## Remaining external and scientific blockers

- Real dedicated verifier and retention identities with no untrusted
  same-token process.
- Signed owner/DACL and negative-probe evidence under those identities.
- Approved WORM or external monotone CAS and signed retention receipt.
- Effective network denial and absence of signing/mTLS/HSM credentials.
- Signed Python, wheelhouse, SBOM and release provenance.
- Real Job Object, timeout, locked-file, concurrent-write, SMB, power-loss,
  monitoring, backup/restore and disaster-recovery drills.
- Fresh prospective point-in-time CH inputs, independent direct CH truth,
  scientifically admissible rolling-origin design and new future holdout.
- T057 remains revoked as historical `12/12` evidence; effective historical
  sample size is one.
- A new candidate CH must bind exact EEX vintage, solver configuration,
  deterministic sensitivity/cascade gates, probabilistic design, economic
  profiles/payoffs and independent manifests before any promotion review.

## Next action

Do not reopen this local P1 slice unless new evidence contradicts it. Move to
the fresh prospective-data/admission envelope and a new auditable CH candidate,
while keeping every production and publication gate false. No production
promotion is authorized.
