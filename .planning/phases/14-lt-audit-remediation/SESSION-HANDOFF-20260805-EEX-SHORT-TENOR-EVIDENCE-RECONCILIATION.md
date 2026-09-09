# Session handoff — EEX short-tenor evidence reconciliation (D225)

Date: 2026-08-05  
Scope: LT only, local content-addressed evidence, no Databricks/network action

## Outcome

D225 removes a real D220-D223 evidence-identity contradiction without changing
any scientific contrast value or modelling formula.

Selected chain:

- D220 contrast bundle:
  `309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5`;
- D222/D223 combination proof:
  `122afd7292549b00a946a2893487d344e459c420a8bbdf99c11714c7308a8cfd`;
- outcome-blind combination policy:
  `189e7ab9d460daefbaa689f1dc8ae4c3bb269f7b5bcb5eae95e77b6d0d4d3e43`;
- D225 selection document content ID:
  `792149a7d46834b17507ea4ec6a15dbfeb2ab9233b058b9783673a58a57c26aa`;
- D225 proof bundle:
  `ca099abf04153724254663a880152ba9ed6979638b89eb3462b716a0af8fb041`;
- status:
  `PASS_LOCAL_EVIDENCE_CHAIN_RECONCILED_NO_MODEL_AUTHORITY`.

The independent OMPEX paired-truth work remains D224 and is not changed.

## Defect found and correction

The current repository contained four incompatible statements:

- the D220 decision selected contrast bundle `7a4c2a60...`;
- a later D220 handoff note selected `ae7b962c...`;
- D223 selected combination proof `c06dfcf...`, bound to `ae7b962c...`;
- the current combination-policy validator required `7a4c2a60...`.

The `ae7b...`, `7a4c...` and `309b...` manifests bind exactly the same
contrast, diagnostic and summary bytes. Their content IDs differ only because
they bind different historical test bytes. D225 selects `309b...` because it
binds the current contract/module/test bytes and two exact materializer runs
under the governed repo-local runtime re-adopted it. No market performance,
OMPEX, AFRY, T057 or ENTSO-E value participated in this choice.

The combination contract now binds `309b...` and decision D225. Two exact
replays generated `122afd...`; its manifest binds both `309b...` and policy
`189e7ab9...`. Old proofs `21c557df...` and `c06dfcf...` remain immutable but
are superseded because they bind superseded contrast identities.

## Fail-closed selection contract

Added:

- `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-EVIDENCE-SELECTION-V1.json`;
- `pfc_shaping/lt/model/short_tenor_evidence_selection.py`;
- `tests/test_short_tenor_evidence_selection.py`;
- build-only
  `build/databricks-eex-daily/materialize_short_tenor_evidence_selection.py`.

The validator requires:

- exact selected and superseded content IDs and manifest hashes;
- exact six-manifest set, with bounded stable UTF-8 JSON reads;
- selected combination proof bound to selected contrast and policy;
- superseded proofs bound only to superseded contrast bundles;
- identical contrast/diagnostic/summary hashes across the three D220 bundles;
- current contract/module/test hashes for selected evidence;
- zero Databricks/Warehouse/network/write counts;
- every PIT/model/candidate/promotion/production authority false.

It contains and persists no scientific payload values.

## Canonical hashes

Selected contrast:

- manifest SHA-256:
  `57b571bef311cf22955edc415e650f55671ad2f243141d4f334ccac425d8bb0f`;
- contrast/diagnostic/summary SHA-256:
  `99933b4e4fce7e208f3f57017f6ba1ad6b1342e97804958b64465cdc4e550dcc`,
  `f06f024de5e8f8d1e758195c7a260f64ff49bd9a488f10092c87294ee199e049`,
  `663bbc40827efe51e3a0f8e7b32b60b1f1679ea097a8d1e3a8816257bff42694`;
- contrast contract/module/tests/materializer SHA-256:
  `8d18107356ae60e1379baaee5392ff7f2fc2c79895140c6e4423105ad970a802`,
  `95ca6087f9ea213111abed2c06ba87978d21cb4d4c4221bb40e953abba626b0b`,
  `505d30aa8608af24619b5fcccf68d3e20f3b9520dc31ee73a97eb74c1d2ba858`,
  `481314d1aa07375bcf02c290306723384f35620542098b21200d0ed858958435`.

Selected combination proof:

- manifest/summary/residual SHA-256:
  `d9772a5e18eea9adb3f6aa9a085558fe0bf91534b03d2e172c4d3c16b0626dab`,
  `5f68ca8c6089df2cacb76db202160720655fe6b3e3ffdcf2522e04e41e9b6db3`,
  `45123d43c5978d115838308cb8028a4c13a3b842cebb34486d5c60a3286bed08`;
- combination contract/module/tests/materializer SHA-256:
  `367ae861ecb6273dbd72ff308fdd0523b8640e57f7dba122fade4d127e7b1aae`,
  `f87e72b56c6286c57dac74333fa379aa3756ba9674bd0a510a7f7f2f05274697`,
  `8a818a83633254c244409eac216b597aafabe93c69cbb3b38fb28c277f0f61ba`,
  `7c5289b1bb6afda0e8ddba7d8fb01ca286f7b7ecf393b3b6804865dfed5521bf`.

D225 selection:

- document SHA-256/content ID:
  `2517c106e5cfa9e6ab2dc39df301cb620df53aeb91948c652e58ef40015ac643`,
  `792149a7d46834b17507ea4ec6a15dbfeb2ab9233b058b9783673a58a57c26aa`;
- proof manifest/summary SHA-256:
  `6f0212a01490104c0e5fde40d3b4004ad0e9589424ca0766404eaff7c71b036d`,
  `a38294a2111d9dc3fae79e857fbbf980839f49dbadc94da822dbb15a73ff770f`;
- selection module/tests/materializer SHA-256:
  `7990a11afb05a7416a7decb5bdacd955525a105ab206464f722297e50ea7b27c`,
  `e34870c88cca6f524c59fc152a6274a7d447289e04858df668ab1dae9443f0fa`,
  `f0fb570579767523a87c92ff36a1ce8a25e1988a474214b506345d016e68eccf`.

## Commands and results

Every command first verified that both the current directory and normalized
Git top-level were exactly `C:\Users\jbattaglia\PFC_LT`. `TEMP`, `TMP`, pytest
basetemps and all outputs remained below `build/`.

Current D220 replay, executed twice:

```text
build/pytest-runtime-v2-final/python.exe -B
  build/databricks-eex-daily/materialize_short_tenor_contrasts.py
  --normalization-bundle build/databricks-eex-daily/2026-08-05/
    normalizations-all-products/2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f
  --output-root build/databricks-eex-daily/2026-08-05/short-tenor-contrast-panels
  --expected-normalization-id 2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f
  --expected-audit-id b09eb3250df5a3c0616eb169c512319c514ddf540251b405023d9351bd5d8bde
=> 309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5
=> identical second replay
```

Focused tests:

```text
python -m pytest tests/test_short_tenor_evidence_selection.py
  tests/test_short_tenor_combination_contract.py
  tests/test_short_tenor_contrasts.py -q
66 passed in 2.38s
```

Combination materializer, executed twice:

```text
python -B build/databricks-eex-daily/
  materialize_short_tenor_combination_contract.py
  --output-root build/databricks-eex-daily/2026-08-05/
    short-tenor-combination-contract-proofs
=> 122afd7292549b00a946a2893487d344e459c420a8bbdf99c11714c7308a8cfd
=> identical second replay
```

Selection materializer, executed twice:

```text
python -B build/databricks-eex-daily/
  materialize_short_tenor_evidence_selection.py
  --selection-document .planning/phases/14-lt-audit-remediation/
    EEX-CH-SHORT-TENOR-EVIDENCE-SELECTION-V1.json
  --evidence-root build/databricks-eex-daily/2026-08-05
  --output-root build/databricks-eex-daily/2026-08-05/
    short-tenor-evidence-selection-proofs
=> ca099abf04153724254663a880152ba9ed6979638b89eb3462b716a0af8fb041
=> identical second replay
```

Adjacent matrix:

```text
244 passed, 4 skipped in 14.77s
```

AST parsing, JSON parsing and the 100-character code-line audit passed. Ruff is
not installed in `build/pytest-runtime-v2-final`, so no Ruff pass is claimed.

## Superseded local artifacts

Historical durable selections retained but no longer current:

- contrast bundles `ae7b962c...` and `7a4c2a60...`;
- combination proofs `21c557df...` and `c06dfcf...`.

Pre-renumber local-only rehearsals `422edadd...` and `600bc3a9...` were created
before the concurrent OMPEX work occupied D224. They have no decision or
selection authority; D225 supersedes them with `122afd...` and `ca099...`.
Nothing was deleted.

## Changed files

- added
  `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-EVIDENCE-SELECTION-V1.json`;
- added `pfc_shaping/lt/model/short_tenor_evidence_selection.py`;
- added `tests/test_short_tenor_evidence_selection.py`;
- added build-only
  `build/databricks-eex-daily/materialize_short_tenor_evidence_selection.py`;
- updated
  `.planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-COMBINATION-CONTRACT-V1.json`;
- updated `pfc_shaping/lt/model/short_tenor_combination_contract.py`;
- updated `tests/test_short_tenor_combination_contract.py`;
- updated build-only
  `build/databricks-eex-daily/materialize_short_tenor_combination_contract.py`;
- added D225 to `DECISION-LOG.md` without changing OMPEX D224;
- updated `docs/research/forwards_sources.md` and `.planning/HANDOFF.md`;
- added this handoff.

## Authority and remaining blockers

- Databricks requests: 0.
- SQL Warehouse starts: 0.
- Network calls: 0.
- Remote writes: 0.
- Market performance used for selection: false.
- Point-in-time availability proven: false.
- Model input, candidate assembly, promotion and production: false.
- Monthly solver remains sole level authority.
- OMPEX remains post-candidate benchmark-only; AFRY remains descriptive; T057
  remains sealed.
- Signed EEX PIT vintages, governed ENTSO-E, exact origin/target/fold
  inventories, external freeze and a new future holdout remain mandatory.

## Next safe batch

Now that upstream evidence identity is unambiguous, define and adversarially
test the hash-bound future training/selection receipt schema. It must prove
per-origin PIT cutoffs, fold isolation, exact source/grid identities and no
outcome access while keeping all numeric hyperparameters null until the direct
CH dependence/power design and external freeze exist.
