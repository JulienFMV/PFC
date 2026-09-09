# Session handoff - EEX PIT hard-level cross-binding and standard-user reclosure

Date: 2026-07-29 (Europe/Zurich)

Branch / source baseline:

- branch: `fix/lt-audit-remediation`;
- requested baseline HEAD:
  `2f68125bff869ccb21c1e20df0201ad024ed27d3`;
- canonical cwd and Git root:
  `C:\Users\jbattaglia\PFC_LT`;
- worktree intentionally very dirty; no reset, clean, restore, stage or commit;
- protected `data/eex_forwards_history.parquet` was not written. Its terminal
  SHA-256 remains
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`.

Production status: strict `NO_GO`. No data, candidate, publication, promotion
or production transition occurred.

## Outcome

Two local defects are closed in code and tests.

1. The standard-user runner now rejects every explicit launcherless runtime
   input/output outside repo `build/` before target execution. This includes
   an `AppData` wheelhouse even when readable. `uv.lock` is the sole allowed
   root-level path. No administrator right, elevation, Defender/ASR exception,
   project executable or Playwright runtime is used.
2. When the CH monthly solver is the level authority, its hard levels now come
   from the exact verified catalog-backed PIT vintage selection. The parallel
   `inputs.eex_report_path` workbook is not read in this branch. Authenticated
   vintage `quote_id`, revision, acquisition and source identities reach the
   solver constraints. `EEX_VINTAGE` is hard-quote eligible after exact raw
   workbook reparse, but remains explicitly non-promotable without independent
   external-CAS admission.

The new prospective intake adapter is local normalization only. It does not
capture provider bytes, mint trusted time, sign a catalog or publish external
CAS state. Its result keeps calibration, external-CAS and production authority
false.

## Changed implementation and tests in this slice

- `pfc_shaping/data/eex_forward_vintage_intake.py` (new): canonical
  caller-hash-bound spec, strict quote inventory, exact raw/receipt bindings,
  XLSX resource preflight, physical latest-row audit, append-only bitemporal
  revisions and signed journal extension checks;
- `pfc_shaping/data/eex_historical_vintage.py`: exact catalog/entry/receipt/
  parser-config fields and claim-smuggling rejection; bounded prospective
  catalog reads;
- `pfc_shaping/data/ingest_forwards.py`: public product normalization and
  fail-closed rejection of identical duplicate product columns;
- `pfc_shaping/data/forward_proxy.py`: verified-vintage snapshot factory,
  authenticated vintage quote IDs, exact workbook reparse and EEX-vintage
  lineage validation;
- `pfc_shaping/pipeline/production_phases.py`: solver branch consumes the
  verified PIT vintage snapshot directly;
- `pfc_shaping/pipeline/monthly_curve_authority.py`: accepts `EEX_VINTAGE` for
  hard constraints while forcing its local promotion eligibility false;
- `pfc_shaping/package_contract.py`: packages the new intake module;
- `scripts/run_workspace_local.py`: explicit launcherless path confinement;
- `pfc_shaping/tools/OPERATIONS.md`: executable standard-user rule;
- `tests/test_eex_forward_vintage_intake.py` (new),
  `tests/test_eex_historical_vintage.py`, `tests/test_forward_proxy.py`,
  `tests/test_lt_package_contract.py`,
  `tests/test_run_workspace_local_script.py`.

No CT or Power BI file was touched by this slice.

## Demonstrated adversarial closures

Tests reject:

- raw provider-byte mutation after the frozen spec;
- frozen quote inventory omission;
- silent fallback from a later invalid physical row;
- unknown quoted products and identical duplicate product columns;
- non-genesis initial trusted-time receipts and journal sequence gaps;
- same-date product-set change without governed tombstones/schema migration;
- integer `0/1` substituted for strict boolean authority claims;
- oversized intake specs before JSON parsing;
- a re-signed catalog carrying an injected `production_authorization` field;
- detached hard-level prices or quote IDs;
- any launcherless wheelhouse/prefix/receipt/manifest outside repo `build/`.

The end-to-end solver regression proves:

- `forward_source_kind == "EEX_VINTAGE"`;
- `promotion_eligible is False`;
- the own forward quote ID equals the authenticated vintage row `quote_id`;
- every generated constraint source quote ID equals that vintage ID.

## Exact terminal matrices

All commands below were run through `scripts.run_workspace_local` after the
separate literal cwd/Git-root guard.

```powershell
python -B -m scripts.run_workspace_local --run-id permfix1 -- python -B -m pytest tests\test_run_workspace_local_script.py -q -p no:cacheprovider
```

Result: `17 passed in 0.90s`. Receipt SHA-256:
`c439950540029ba0b6ba9af910626b547f97a691a18b056450dd6db60aaaf76d`.

```powershell
python -B -m scripts.run_workspace_local --run-id eexpit1 -- python -B -m pytest tests\test_eex_historical_vintage.py tests\test_forward_proxy.py tests\test_monthly_forward_curve_integration.py -q -p no:cacheprovider
```

Result: `74 passed in 44.03s`. Receipt SHA-256:
`211d463f41090840fe3432ae603065f806c1fb5a614223dab5a6ce228dbd8296`.

```powershell
python -B -m scripts.run_workspace_local --run-id eexpit2 -- python -B -m pytest tests\test_eex_historical_vintage.py::test_signed_catalog_binds_history_and_every_archived_source_document -q -p no:cacheprovider
```

Result: `1 passed in 18.09s`. Receipt SHA-256:
`8d87b7d6b04903396037ada931cb5d395a7adf1312b4c6a6413363165334e2b9`.

```powershell
python -B -m scripts.run_workspace_local --run-id eexadv3 -- python -B -m pytest tests\test_eex_forward_vintage_intake.py -q -p no:cacheprovider
```

Result: `13 passed in 5.17s`. Receipt SHA-256:
`6dc5e548405d77549443bd9d08e4c41cec6c0a0c4632af3b68f1a2fd4dcf1e2c`.

Two preceding adversarial runs are retained negative because their assertions,
not the implementation, were corrected: `eexadv1` exit 1 receipt
`d440f0abfa1f6c4147d3eac3e9fa7f7be628b5f833ea4c0a5a0d4955d353861b`;
`eexadv2` exit 1 receipt
`35d5b6f023c81dd43af55ae6741edb261a192f71973824dfd9cc9763a5bc9303`.

```powershell
python -B -m scripts.run_workspace_local --run-id eexmatrix3 -- python -B -m pytest tests\test_eex_forward_vintage_intake.py tests\test_eex_historical_vintage.py tests\test_forward_proxy.py tests\test_governed_forward_history.py tests\test_monthly_forward_curve_integration.py tests\test_monthly_forward_curve_solver.py tests\test_monthly_forward_curve_constraints.py tests\test_lt_package_contract.py tests\test_run_workspace_local_script.py -q -p no:cacheprovider -m "not slow"
```

Result: `188 passed in 34.36s`. Receipt SHA-256:
`3d5c7400c4bc711c9e92656d0cfdd356f17e4e6d0a46fd539837d16cdea6532b`.
The misspelled predecessor `eexmatrix2` is retained exit 4 and is not counted.

```powershell
python -B -m scripts.run_workspace_local --run-id runtime1 -- python -B -m pytest tests\test_lt_provider_verifier_artifact.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_runtime_closure.py tests\test_lt_package_contract.py tests\test_launcherless_conda_archive_lock.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py tests\test_run_workspace_local_script.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_audit_legacy_provider_resolution_script.py -q -p no:cacheprovider -m "not slow"
```

Result: `175 passed, 12 skipped, 2 deselected in 142.64s`. Receipt SHA-256:
`a3c858eb0ffd98ecab77702f4293ad475ad606f24eff8f7318aa2c0dd32864e9`.

The first monolithic publication run exceeded its 300-second parent timeout,
terminalized exit 120, and left no child process. It is retained negative with
receipt SHA-256
`b886fe12a3a1d167b7e422080e2321a4d590208d3bfce04b5940e9d01d5365dc`.
No failed namespace was reused. Fresh split runs produced:

```powershell
python -B -m scripts.run_workspace_local --run-id pubcand2 -- python -B -m pytest tests\test_atomic_promotion.py tests\test_candidate_bundle.py tests\test_candidate_evidence.py tests\test_candidate_evidence_assembler.py -q -p no:cacheprovider -m "not slow"
```

Result: `181 passed, 2 skipped in 575.00s`. Receipt SHA-256:
`30359a5c44d16e0c3ca793cefefe5eb2b745c632871413cf6692e0ecdeae2316`.

```powershell
python -B -m scripts.run_workspace_local --run-id pubcas2 -- python -B -m pytest tests\test_snapshot_publication_external_contract.py tests\test_snapshot_anchor_client.py tests\test_snapshot_anchor_reference.py tests\test_snapshot_bootstrap_signer.py tests\test_governed_release.py tests\test_check_monthly_curve_promotion_from_manifests.py -q -p no:cacheprovider -m "not slow"
```

Result: `200 passed in 336.30s`. Receipt SHA-256:
`23b90cc7461cb0f37c35d1b41cf68c9bd34f6831d5bca2b63b0f79a3210811b5`.

```powershell
python -B -m scripts.run_workspace_local --run-id eexruff3 -- python -B -m ruff check scripts\run_workspace_local.py tests\test_run_workspace_local_script.py pfc_shaping\data\eex_forward_vintage_intake.py pfc_shaping\data\eex_historical_vintage.py pfc_shaping\data\forward_proxy.py pfc_shaping\data\ingest_forwards.py pfc_shaping\pipeline\monthly_curve_authority.py pfc_shaping\pipeline\production_phases.py pfc_shaping\package_contract.py tests\test_eex_forward_vintage_intake.py tests\test_eex_historical_vintage.py tests\test_forward_proxy.py tests\test_lt_package_contract.py
```

Result: `All checks passed!`. Receipt SHA-256:
`efd32102cadde89e7347e2549f3af409356fcd60dc72b253aa6cdc73f9a122a7`.

### Post-roast final delta and superseding evidence

The first Security/Quant/IT delta roasts demonstrated three additional local
findings. They are corrected on the terminal source bytes:

- this EEX adapter now accepts only `EEX_SETTLEMENT_EUR_MWH`; a distinct
  adapter/schema is required for any FMV desk-mid source;
- zero is no longer silently treated as an absent quote by the prospective
  intake. Because provider quote-status metadata is not yet authenticated, any
  zero is ambiguous and fails closed;
- catalog/history/source/parser/proxy/monthly-authority reads are mono-link and
  bounded, and archived XLSX bytes pass the same ZIP resource preflight before
  replay;
- the standard-user runner scrubs ambient public-key, trusted-authority and
  journal variables in addition to private credentials.

```powershell
python -B -m scripts.run_workspace_local --run-id eexadv4 -- python -B -m pytest tests\test_eex_forward_vintage_intake.py -q -p no:cacheprovider
```

Result: `14 passed in 11.50s`. Receipt SHA-256:
`404cd701f79ea2c5c7de96758718eee42a4ec92d29e86e696d8f63cec8563243`.

The first post-roast expanded run (`finalmatrix1`) is retained negative: one
test still spied on the retired `Path.read_bytes` API. The test was corrected
to count the stable reader; no production behavior was weakened. Negative
receipt SHA-256:
`7b876a7687409ce56518f1b589839058e08c002739363cc69f6f3bd8e996f7b5`.

```powershell
python -B -m scripts.run_workspace_local --run-id finalmatrix2 -- python -B -m pytest tests\test_eex_forward_vintage_intake.py tests\test_eex_historical_vintage.py tests\test_forward_proxy.py tests\test_governed_forward_history.py tests\test_monthly_forward_curve_integration.py tests\test_monthly_forward_curve_solver.py tests\test_monthly_forward_curve_constraints.py tests\test_lt_package_contract.py tests\test_run_workspace_local_script.py -q -p no:cacheprovider -m "not slow"
```

Result: `189 passed in 41.94s`. Receipt SHA-256:
`75b450ae9407fdd01d5f94a003ffb601d301dfde95514f617b50efab08220e8f`.

```powershell
python -B -m scripts.run_workspace_local --run-id runtime2 -- python -B -m pytest tests\test_lt_provider_verifier_artifact.py tests\test_snapshot_publisher_artifact.py tests\test_snapshot_publisher_runtime_closure.py tests\test_lt_package_contract.py tests\test_launcherless_conda_archive_lock.py tests\test_launcherless_local_runtime.py tests\test_launcherless_runtime_admission.py tests\test_run_workspace_local_script.py tests\test_audit_provider_acquisition_quarantine_script.py tests\test_audit_legacy_provider_resolution_script.py -q -p no:cacheprovider -m "not slow"
```

Result: `175 passed, 12 skipped, 2 deselected in 122.10s`. Receipt SHA-256:
`7a19ca2ea27897bac2451eef301a53976b86e12496106a1635da4a1e68e23b69`.

Final Ruff result: `All checks passed!`. Receipt SHA-256:
`1d0e7f76118b98ad49f953d403c11116eb4f3c5066bd04e52e34f44afb1a44d1`.

Selected terminal source hashes:

- `scripts/run_workspace_local.py`: 22,723 bytes, SHA-256
  `31612b7f8fd15955a54e070c4ae9bc1465c1974f220570bcb242c6f23146c106`;
- `pfc_shaping/data/eex_forward_vintage_intake.py`: 40,212 bytes,
  SHA-256
  `b6de3f88dd8cf05ab472f84a0143a51406327e5cbbca55cda4412c2f0abdeb92`;
- `pfc_shaping/data/eex_historical_vintage.py`: 43,229 bytes, SHA-256
  `eb9c5c08225c18cac4ac13397716a859455c9c1e991b8d8666cf457efa2a7883`;
- `pfc_shaping/data/forward_proxy.py`: 51,756 bytes, SHA-256
  `14c01e2340e458879013c322e67a2534a32a8807fc3db02f8e7648f9a66d0e18`;
- `pfc_shaping/pipeline/monthly_curve_authority.py`: 31,524 bytes,
  SHA-256
  `767e8dbb6561defb5f13311825ec22c9004eb7e9fe8a5035029d6dc754d53e04`.

### Terminal strict-replay addendum

The final Quant/Security delta-roasts then demonstrated that a catalog built
outside the adapter could still use the generic historical replay. Terminal
code closes that bypass:

- prospective historical replay requires exact settlement convention, a
  SHA-256 intake ID and exact physical latest-row quote inventory;
- the hard-level `EEX_VINTAGE` factory rejects legacy parser configs, rebinds
  the supplied catalog mapping to the exact persisted catalog bytes/hash and
  accepts only prospective settlement admission;
- candidate/manifest verification, XLSX fallback, UNC fallback and exact
  workbook loading all use the stable 64 MiB + ZIP-preflight reader.

Legacy catalogs remain auditable under their historical schema but cannot
become CH hard-level authority.

`finalmatrix3` is retained negative: it proved the legacy end-to-end fixture
was rejected by the new gate; 230 other tests passed. Receipt SHA-256:
`a55eb6517a1e266b80f58ce2092b23d0c0310606f3dcd12fc6b4907649cd7227`.
The fixture was migrated to the exact prospective settlement contract.

```powershell
python -B -m scripts.run_workspace_local --run-id finalmatrix4 -- python -B -m pytest tests\test_eex_forward_vintage_intake.py tests\test_eex_historical_vintage.py tests\test_forward_proxy.py tests\test_governed_forward_history.py tests\test_monthly_forward_curve_integration.py tests\test_monthly_forward_curve_solver.py tests\test_monthly_forward_curve_constraints.py tests\test_lt_package_contract.py tests\test_run_workspace_local_script.py -q -p no:cacheprovider -m "not slow"
```

Result: `189 passed in 42.94s`. Receipt SHA-256:
`47f35e84bdf34c77eaf61f99c48943b7715070a04024b75e985612b655736b78`.

```powershell
python -B -m scripts.run_workspace_local --run-id candfinal1 -- python -B -m pytest tests\test_candidate_evidence_assembler.py::test_assembly_replays_hourly_export_from_staged_pfc tests\test_candidate_evidence_assembler.py::test_assembly_rejects_duplicated_forward_quote_snapshot -q -p no:cacheprovider
```

Result: `2 passed in 6.98s`. Receipt SHA-256:
`c9dae5beaf3fd018990f98f283f28dbe33e0ce3fae723889a47726f0c36e00cd`.

Terminal Ruff: `All checks passed!`. Receipt SHA-256:
`8757ccf1e0c193b45c52c073a634a2d6115b57654cabdf16ea23ac38d8bb107d`.

Terminal key hashes superseding earlier hashes in this document:

- intake: 40,252 bytes,
  `612943b555b3d4139e6cdc84554a928c6a03bb4664bf50e9d21a6b8b38edd3a1`;
- historical verifier: 45,104 bytes,
  `8ccc93731bfccaee0fa45678b406e75ed2f739d6e36deecc39ee761d5d52b286`;
- forward proxy: 53,887 bytes,
  `1848383b1f9e26de6750868745606e22f3d1c3f902e8e2a19b6bae749da0af7b`.

## Independent roast status

Final read-only Security/Governance, IT/Operations and Quant/Data re-roasts are
recorded in the session conclusion below. Their residual findings do not grant
production or scientific authority.

All three report P0=0. The demonstrated local convention/zero, resource
budget, prospective-replay and manifest-read P1 findings were corrected and
covered by terminal tests. Remaining P1 are production/industrialization or
external-scientific blockers: sealed trust roots and external CAS/HEAD, a
durable idempotent EEX CLI/staging/catalog workflow, transitive isolated
parser/runtime provenance and fresh wheels/runtime, supervised workers/CI,
real provider product semantics, multi-origin completeness and T057 evidence.

## Residual blockers and next work

- No fresh exact provider EEX workbook plus independently trusted receipt was
  available locally. The adapter is covered by fixtures, not fresh market data.
- Trust-anchor selection is still workstation/environment governed rather than
  an independently provisioned production bootstrap.
- Parser/intake/runtime dependency provenance is not yet a fully transitive,
  isolated import-before-hash closure.
- Settlement convention is now unique for this adapter, but authoritative
  provider contract/ISIN/session/calendar metadata is not yet bound. Ambiguous
  zero now fails closed; explicit quote status and tombstones are still absent,
  and week products remain excluded.
- External immutable CAS/WORM, signed monotone HEAD, concurrent publication,
  independent CI/ASR, SBOM/provenance, observability and rollback remain open.
- Rolling-origin shaping, T057, probabilistic/scenario calibration and a fresh
  auditable CH candidate were not run in this slice.

Next safe step: build the durable repo-local staging/CLI for a genuinely fresh
provider workbook and external trusted-time receipt, close transitive parser
and runtime provenance, then admit a signed cumulative catalog through
independent external CAS. Only after that should rolling-origin/T057 and a new
CH candidate be evaluated. Production remains strict `NO_GO`.
