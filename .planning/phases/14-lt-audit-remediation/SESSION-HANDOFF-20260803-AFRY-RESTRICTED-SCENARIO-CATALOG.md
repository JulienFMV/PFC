# Session handoff - AFRY restricted scenario catalog v6

Date: 2026-08-03
Branch: `fix/lt-audit-remediation`
Workspace: `C:\Users\jbattaglia\PFC_LT`
Production: strict `NO_GO`

## Outcome

The AFRY Switzerland 2026 Q2 delivery is available to local agents through a
restricted, content-addressed PyArrow catalog. It is useful for structural
scenario challenge and annual hour-slot shape diagnostics, but it is not an
approved model input, probability system, monthly level source, timestamped
calendar, 15-minute curve, or production artifact.

Current byte-identical local bundles:

- `build/afry-catalog-final-v6-a-20260803/a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454`
- `build/afry-catalog-final-v6-b-20260803/a7ca9238ad715e67269c73a862607c3a3301358821818bc5d2fa4f1581c54454`

Catalog SHA-256:
`f7aa9bd4d3cc0f13851225809b0271a138bb733c51fd563d9594508b32f65c9a`.
The catalog reports `PASS_LOCAL_INTEGRITY_ONLY` and
`NO_GO_MODEL_AND_PRODUCTION` separately.

## Source evidence

Restricted inputs remain outside Git:

- annual workbook SHA-256
  `460c5030b1f2c1b0e281dfd93fa5ef120a0b19c62af8a39d066313b7c7f926ea`;
- hourly workbook SHA-256
  `903850cff94738eba2845366b99423bef87c9c59fe122ae82de681d0199318f8`;
- Commodity and Modelling Annex SHA-256
  `9eaa00233074fac38efe9f9d048f0f9524d833a0ef9313d5acc50773a4acfbe1`;
- Quarterly Update Note SHA-256
  `dc0c0c2c06ddf95d523c418e7ae4899c0aed1ed0f8940566992e4008596769e8`.

The two PDFs are material. The QUN resolves scenario semantics, negative-price
and weather assumptions; the annex documents BID3 dispatch/investment,
hourly chronology, hydro and storage. They strengthen benchmark use while
showing that the data are vendor-model output rather than observed Swiss
truth. Their copies and extracted text remain restricted below `build/` and
must not be moved to docs or RAG.

## Files changed in this slice

- `.gitignore`
- `.planning/phases/14-lt-audit-remediation/AFRY-CH-2026-Q2-SOURCE-REGISTRATION.json`
- `.planning/phases/14-lt-audit-remediation/AFRY-CH-2026-Q2-SEMANTIC-CONTRACT.json`
- `.planning/phases/14-lt-audit-remediation/AFRY-CH-2026-Q2-AGENT-DATA-CONTEXT.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- `pfc_shaping/data/afry_scenarios.py`
- `scripts/materialize_afry_scenario_catalog.py`
- `tests/test_afry_scenarios.py`
- `tests/test_afry_scenario_extraction.py`
- this handoff

The worktree was already intentionally very dirty. No unrelated changes were
reset, cleaned, restored, staged or committed. CT and Power BI were untouched.

## Implemented controls

- exact release registration, July cadence, source dates, hashes and licence;
- separate hash-bound semantic contract without vendor numeric values;
- single-read source capture into immutable process memory, hash of the exact
  captured bytes, captured-container validation, capture-only parsing and
  post-parse rehash of those same bytes;
- OOXML rejection for macros, worksheet formulas, active/embedded content,
  encryption, traversal, excessive member count/size, aggregate compression
  and diluted per-member ZIP-bomb conditions;
- static PDF hash/header verification without executing or rendering PDFs;
- deterministic typed Parquet for hourly, annual, structural and hour-slot
  benchmark data plus JSON reconciliation profiles;
- exact schemas, annual semantics, physical balance/flow signs, hourly-to-
  annual rounding, negative-hour and representative-hour continuity checks;
- annual additive zero-mean hour-slot benchmark with all model, monthly-level
  and production authorities false;
- full-body catalog content ID, exact published member closure, artifact
  sizes/hashes, canonical LF checksum sidecar and fail-closed reuse;
- catalog policy verifier that rejects self-consistent authority flips and an
  ambiguous plain `PASS`;
- consumer verifier single-reads exact artefact bytes, recalculates structural
  and annual audits from Parquet rather than trusting JSON claims, and exposes
  a re-bound single-read byte API for downstream PyArrow consumption;
- producer module/entrypoint hashes, exact repo-local interpreter and a full
  1,429-file / 45,398,809-byte runtime-tree receipt bound into the catalog,
  with producer, registration, executable and runtime-tree rehash before
  publication;
- release-driven structural provenance instead of a hard-coded 2026 label.
- release-anchored contiguous annual horizon, exact observed future-domain
  reconciliation, exact scenario/year key uniqueness and row count;
- field-by-field modelled/interpolated status lineage, exact aggregate status,
  explicit null scenario weights and all authority columns false.

## Commands and results

Every shell command first asserted that cwd and Git top-level were exactly
`C:\Users\jbattaglia\PFC_LT`. All mutable TEMP/TMP, caches, basetemps and
outputs stayed below `build/`.

Targeted lint:

```powershell
& 'C:\Users\jbattaglia\.conda\ppa_env\Scripts\ruff.exe' check --no-cache pfc_shaping\data\afry_scenarios.py tests\test_afry_scenarios.py tests\test_afry_scenario_extraction.py scripts\materialize_afry_scenario_catalog.py
```

Result: `All checks passed!`

Targeted tests:

```powershell
& 'build\pytest-runtime-v2-final\python.exe' -I -B -m pytest tests\test_afry_scenarios.py tests\test_afry_scenario_extraction.py -q -p no:cacheprovider --basetemp build\pytest-afry-consumer-recompute-v4
```

Result: `72 passed`.

Adjacent matrix:

```powershell
& 'build\pytest-runtime-v2-final\python.exe' -I -B -m pytest tests\test_afry_scenarios.py tests\test_afry_scenario_extraction.py tests\test_electrification_scenarios_data.py tests\test_electrification_shape.py tests\test_lt_ct_imports.py tests\test_lt_package_contract.py -q -p no:cacheprovider -m 'not slow' --basetemp build\pytest-afry-adjacent-v5
```

Result: `149 passed, 4 skipped`.

Final materializations used the same command with distinct repo-local TEMP and
output roots:

```powershell
& 'build\pytest-runtime-v2-final\python.exe' -I -B -m scripts.materialize_afry_scenario_catalog --registration .planning\phases\14-lt-audit-remediation\AFRY-CH-2026-Q2-SOURCE-REGISTRATION.json --output-root build\afry-catalog-final-v6-a-20260803
```

and `build\afry-catalog-final-v6-b-20260803`. Results were identical content
ID `a7ca9238...54454`, local integrity pass and model/production `NO_GO`.
Producer hashes were also checked externally before/after each command.

Independent verification:

- all 11 member names, sizes and SHA-256 values identical across v6-A/v6-B;
- both complete bundles accepted by `verify_catalog_bundle`;
- registration SHA-256 is
  `e552302eb1921494663c7a86b502fd43c266f9c7b1a2619639cf3a5ceb1d66da`;
- producer hashes match the current module
  (`366603118ca25382e7f688d484154ab5e86f041a1cb9c528129c2bb4f477d9ed`)
  and entrypoint
  (`eac778cfc46f338fb14b2beb6f4c27b6fea97d7b52a8f1530cd2b837b675ba98`);
- rows: hourly 3,854,400; annual 127,329; structural 136; hour-slot 10,560;
- annual future domain is exactly the registered contiguous 2027--2060
  horizon for all four scenarios; structural keys are exactly 136/136 with
  no duplicates or scenario weights;
- field-level value-status maps, aggregate status and interpolated flags all
  reconcile exactly;
- runtime closure SHA-256
  `1695e210c93e93e5806f7d5e198576289016d1e384fd3403479fdc9e65f35caa`
  covers 1,429 plain files and 45,398,809 bytes and was stable in both runs;
- 440 scenario/delivery/weather groups, each with exactly the distinct ordered
  representative-hour set 1 through 8,760;
- 440 hour-slot groups satisfy the `1e-12` additive zero-mean tolerance.

## Detected and corrected findings

Security:

- existing CAS reuse originally checked only `catalog.json`; exact member,
  size and hash closure is now mandatory;
- a content-ID-consistent malicious authority flip was possible in principle;
  independent immutable non-authority policy checks now reject it;
- formula/external-link path matching missed case/backslash variants and ZIP
  compression was aggregate-only; paths are normalized and per-member limits
  are enforced;
- the checksum sidecar used Windows text newlines; it is now canonical LF.
- the canonical consumer context previously selected an obsolete bundle and
  reopened paths after verification; it now selects v6 and uses re-bound,
  single-read bytes that are hashed before PyArrow consumption.

IT/Operations:

- structural source provenance was hard-coded to 2026; it now follows the
  registered release ID;
- producer code was unbound and changed during a real long materialization.
  One v2 attempt failed closed. Code hashes and runtime versions are now in the
  catalog and sources are rehashed before publication.
- a signed registration could previously reduce the annual domain and hide
  extra workbook years; the v2 registration now anchors a 34-year contiguous
  horizon to release year + 1, checks hourly first/last anchors and compares
  the exact observed future domain globally and per scenario;
- executable binding alone did not close Python dependencies; v5 binds and
  double-scans the complete repo-local intake runtime tree.

Quant/Data:

- vendor documentation could be misread as empirical Swiss validation. The
  semantic contract now requires `observed_swiss_truth=false`, no independent
  validation, no probability calibration and no calendar authority;
- `High` is explicitly not a fast-transition/policy-compliance proxy;
- the Swiss policy difference remains unresolved rather than labelled purely
  political; scope and execution must be reconciled separately.
- structural key uniqueness, exact row count and null scenario weights are
  gated; value-status provenance is now field-by-field instead of a potentially
  over-broad scenario/year aggregate.
- the consumer previously trusted the derived audit JSON after checking only
  artifact hashes and schemas; v6 recalculates both structural and annual
  audits from the exact captured Parquet bytes and adversarially rejects a
  forged weight or unregistered year.

Final roast status for this local slice: Security P0/P1/P2 `0/0/0`,
Quant/Data P0/P1/P2 `0/0/0`, IT/Operations P0/P1/P2 `0/0/3`. The residual IT
P2 items are: no direct fault-injection test of the concurrent-rename recovery
branch, no orphan-staging startup reconciler/quarantine, and no durable
structured failure receipt for monitoring. They are next-batch operational
hardening, not silent production acceptance. Research and production gaps
remain explicit admission gates.

## Superseded evidence

The following build IDs are not current evidence:

- `21d426f9...`: pre-CAS-closure and pre-LF-sidecar catalog;
- `ee5d29f1...` and `e9398a2e...`: numerically byte-stable artifacts but
  different catalog governance because code changed between builds;
- `0bfc1f7a...`, `58cab90c...` and `d193d59a...`: successively hardened local
  catalogs superseded by the v6 consumer-recomputation closure;
- one v2 run was rejected with
  `AFRY producer source changed during materialization`.
- the intended `build/afry-catalog-final-v4-a` root was already present and
  empty when checked; it was refused, not deleted or reused. Fresh unique v5
  and then v6 roots were used instead.

Do not delete them during this dirty-worktree phase; they demonstrate that the
new fail-closed producer gate caught a real concurrent edit.

Final scope audit:

- `git diff --check`: exit `0`;
- protected `data/eex_forwards_history.parquet` SHA-256 unchanged at
  `21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013`;
- both raw AFRY workbooks are ignored by `data/AFRY_*.xlsx` and absent from
  `git ls-files`;
- no `pfc_shaping.ct` import or trailing whitespace exists in the AFRY slice;
- no staging residue exists below either selected v6 output root;
- no file was staged or committed and no model, candidate or production
  promotion was attempted.

## Remaining blockers and next work

- authoritative representative-hour to delivery-calendar mapping, including
  Swiss DST and leap years;
- vendor field definitions and policy-scope reconciliation;
- governed FMV scenario crosswalk and probability methodology;
- rolling-origin comparison against simple CH seasonal/history baselines,
  including capture-price and tail metrics;
- new externally frozen future holdout; T057 remains sealed and cannot be
  reused;
- stability/sensitivity across annual AFRY releases and weather years;
- licence/access/retention approval and independent runtime/CAS/CI evidence.
- direct concurrent-rename fault injection, orphan-staging reconciliation and
  durable structured failure receipts/monitoring.

Only after those gates may the AFRY benchmark be considered as a shaping
teacher or feature source. It must never replace solver monthly means. No
production promotion occurred.
