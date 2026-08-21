# Session Handoff - 2026-07-13 - Product Lead Quality And P0 Hardening

## Goal

An active Codex goal now owns the transformation of `PFC_LT` into FMV's
reference Swiss PFC: scientifically defensible, point-in-time, market exact,
promotion governed and straightforward for IT to industrialise.

Production and IT handoff remain `NO_GO`.

## Product Contract

Added
`.planning/phases/14-lt-audit-remediation/PFC-FMV-PRODUCT-QUALITY-CHARTER-20260713.md`.
It orders market/provenance/promotion invariants before statistical
optimisation, probabilistic quality and Docker. OMPEX remains advisory-only.

## Changes In This Slice

### Promotion capstone

Completed the delivered-product evidence integration in:

- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `scripts/create_lt_input_snapshot.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`

Production, export and selected-config must carry matching evidence bound to
the exact product-audit summary hash. Missing evidence is blocking.

### Final solver product projection

Changed:

- `pfc_shaping/lt/model/assembler.py`
- `pfc_shaping/lt/model/shape_constraints.py`
- `tests/test_monthly_forward_curve_integration.py`
- `tests/test_lt_quant_contract_matrix.py`

In solver-authority mode, the last mutation of `price_shape` is now a
fail-closed KKT projection. Solver months remain BASE constraints and fully
covered accepted PEAK quotes create disjoint PEAK/OFFPEAK constraints. The
projection uses no floor and no ridge and blocks residuals above `1e-6`.

Fully-covered redundant PEAK parents follow the Month > Quarter > Cal source
hierarchy. Partial PEAK parents are excluded and recorded. Explicit OFFPEAK is
rejected rather than ignored.

### Data fail-closed primitives

Changed:

- `pfc_shaping/data/forward_proxy.py`
- `pfc_shaping/pipeline/production_phases.py`
- `pfc_shaping/pipeline/quality_gate.py`
- `pfc_shaping/config.yaml`
- `tests/test_forward_proxy.py`
- `tests/test_quality_gate.py`

Nominal production branches use `allow_spot_proxy=False`. Core freshness is
fail-closed at EPEX 48 h, ENTSO-E 72 h and hydro 10 days. Outages are disabled
until failed acquisition can be distinguished from economic zero.

Actual preflight at `2026-07-13T12:00:00Z` blocks before fitting:

`QualityGateError: epex_ch: latest point is stale (119.6 days > 2.0 days)`

## Verification

- promotion-focused: `84 passed, 1 skipped`
- projection-focused after roast corrections: `36 passed`
- broader LT/P0 suite: `161 passed, 1 skipped in 26.12s`

## Permanent Expert Roast Status

Quant, data and IT agents reviewed the initial audit and this implementation.
Their post-change verdict remains `NO_GO` because:

1. `rolling_update.py` still bypasses the new gates and can publish fallback.
2. EEX provenance remains an unstructured description without trustworthy
   snapshot date, availability, hash or hard-quote eligibility.
3. Freshness is not yet calculated per latest finite critical feature.
4. Partial-product support and explicit OFFPEAK support need versioned policy.
5. Final projection evidence is not exported in production/export manifests.
6. No real producer emits the complete contract accepted by the capstone.
7. Selected canonical config hash and candidate manifest are not fully rebound.
8. Promotion output is not yet a durable full receipt.

## Next Mandatory Slice

Implement `ForwardSnapshot + Manifest Contract & Promotion Receipt v1`:

1. Structured source kind/path/hash, actual snapshot date, availability and
   hard-quote eligibility.
2. Monthly authority rejects absent, proxy, fallback, unknown and stale sources.
3. Remove `rolling_update` from publishing or apply the same preflight.
4. Add per-feature finite coverage/freshness gates.
5. Version production, export, selected-config and receipt schemas.
6. Recompute canonical config hash and bind every referenced artifact.
7. Persist final projection KKT/residual evidence.
8. Emit a complete receipt on PASS and BLOCK before the atomic release runner.

## Worktree Hygiene

No commit was made. Do not commit `data/eex_forwards_history.parquet`. It and
several audit-script changes predated this slice. Curate code, tests and
planning documents file by file.

## Continuation - ForwardSnapshot And Manifest Binding

Implemented after the initial handoff:

- `pfc_shaping/data/ingest_forwards.py`
  - parses captured workbook bytes;
  - selects the maximum quoted date, independent of physical row order;
  - blocks conflicting duplicate snapshot rows/products;
  - returns source row/column/product lineage.
- `pfc_shaping/data/forward_proxy.py`
  - immutable `ForwardSnapshot v1` with canonical quote/lineage/snapshot and
    observation hashes;
  - verified parser factory only for hard eligibility;
  - source reparse and exact price/date/lineage comparison;
  - immutable `ForwardEligibility v1`, policy hash and receipt recomputation;
  - no history-Parquet mutation during market-data loading.
- `pfc_shaping/calibration/monthly_forward_curve.py`
  - explicit quote/snapshot/observation/source IDs on `MarketQuote`;
  - complete residual-bucket source lineage and lineage hashes;
  - `quote_conflict_tolerance=0.01` separated from hard numerical tolerance.
- `pfc_shaping/pipeline/monthly_curve_authority.py`
  - production requires verified snapshot plus eligibility receipt;
  - exact snapshot/prices binding;
  - hard quote, diagnostics, constraint-provenance and hierarchy-policy hashes.
- `pfc_shaping/pipeline/production_phases.py`
  - workbook receipt propagated through CH solver/branch;
  - history Parquet is read and hashed from the same captured bytes;
  - `eex_as_of_date` is fail-closed for production until bitemporal receipts
    exist;
  - final product projection evidence is attached and hashed in the candidate
    manifest.
- `pfc_shaping/pipeline/quality_gate.py`
  - freshness and finite coverage are checked per critical feature, not only
    at the dataframe index.
- `scripts/audit_ch_product_normalization.py`
  - schema v3 binds the exact candidate snapshot quote-set and eligibility.
- `scripts/check_monthly_curve_promotion_from_manifests.py`
  - recomputes `canonical_config` hash;
  - reopens/hashes candidate manifest;
  - emits complete `promotion_receipt.v1`.

Configuration changes:

- `quality.freshness.eex_max_age_business_days=1`
- `quality.freshness.min_finite_fraction=0.95`
- `forwards.monthly_curve_solver.constraint_tolerance=1e-9`
- `forwards.monthly_curve_solver.quote_conflict_tolerance=0.01`

Real desk workbook evidence at 2026-07-13:

- source: `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx`
- source SHA-256: `62852e935927c53cb3f4c908ed6f1a993fd5344a1e4c7ec93fad03b47bcc6d17`
- selected snapshot date: `2026-07-10`
- quote count: `40`
- quote-set SHA-256: `e18cd372b6c47eaa4a77f1e65a68a93fe3366b61a020b993564f4304c37e2d5e`
- lineage SHA-256: `c4f3b1dee55967bf2e9cb42f57a4524ac3a6564823ef546b410d6f0f1ef94218`
- business age on Monday 2026-07-13: `1` day, PASS
- CAL quotes include 2027 through 2032.

Verification:

- `pytest` broad LT/P0 tranche: `238 passed, 1 warning in 65.17s`
- product-audit + capstone: `55 passed in 32.41s`
- `git diff --check`: no whitespace errors; CRLF conversion warnings only.

Global status remains `NO_GO`:

1. `rolling_update.py` / legacy scheduler can still publish outside this chain.
2. Output writes are direct, not candidate-bundle staging plus atomic pointer
   replacement and rollback.
3. Local EPEX/ENTSO/hydro caches are stale; production preflight blocks before
   fitting.
4. EEX holiday-aware freshness and bitemporal historical replay remain P1.
5. Shaping rolling-origin / locked holdout and probabilistic calibration have
   not yet started.

Next mandatory slice: one canonical LT entrypoint that stages an immutable
candidate bundle, runs all gates/audits, writes the receipt, and only then
atomically replaces `current.json` under a lock. Disable legacy publishing
before shadow-run comparison.

## Continuation - Immutable LT Candidate And Strict Promotion Root

Implemented:

- `pfc_shaping/pipeline/atomic_promotion.py`
  - immutable bundle manifest and full file-set hashes;
  - exclusive promotion lock;
  - canonical approved-receipt verification;
  - atomic `current.json` replacement and tested rollback.
- `pfc_shaping/pipeline/candidate_bundle.py`
  - LT-only candidate serialization under staging;
  - PFC/model/manifests written only inside the candidate;
  - exact EEX workbook archived in-bundle;
  - minimal quote-snapshot Parquet for delivered-product audit;
  - core input source hashes recorded without copying heavy Parquets.
- `scripts/build_lt_candidate.py`
  - strict LT-only candidate entrypoint;
  - explicit run ID, release root and reference timestamp;
  - no CT import, no promotion and no direct shared output save.
- `scripts/check_monthly_curve_promotion_from_manifests.py`
  - verifies bundle file set/content before and after evaluation;
  - requires production candidate, audited CSV/forwards and archived workbook
    to be hash-bound in the bundle;
  - recomputes forward identity/eligibility at the exact promotion timestamp;
  - validates constraint lineage semantics against hard quote IDs and solver
    policy, then requires production/export equality;
  - writes canonical JSON without non-finite values.
- `pfc_shaping/pipeline/quality_gate.py` and production preflight
  - recent finite coverage and maximum observation gaps;
  - production policy cannot be disabled or made more permissive by config.
- `pfc_shaping/lt/model/assembler.py`
  - a PEAK quote partially overlapping the delivered artifact now blocks;
  - fully outside PEAK quotes remain explicitly reported as out of scope.

Adversarial coverage includes:

- workbook hash/parse TOCTOU;
- permissive freshness receipt and production policy;
- substituted observation/source/policy/eligibility fields;
- BASE/PEAK `load_type` mutation;
- invented or divergent constraint provenance with coherently repinned hashes;
- permissive hierarchy policy with coherently repinned hash;
- stale eligibility replay at a later promotion timestamp;
- candidate bundle/file/receipt mutation and concurrent lock;
- partially covered PEAK quote.

Verification:

- focused LT provenance/promotion/candidate suite:
  `225 passed, 1 warning in 62.00s`;
- warning remains the known all-NaN insufficient-history test path in
  `monthly_curve_priors.py`;
- `git diff --check`: no whitespace errors, CRLF conversion warnings only.

Real shadow command:

```powershell
python scripts/build_lt_candidate.py --run-id 20260713-shadow-p0 `
  --release-root "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_RELEASES" `
  --reference-timestamp "2026-07-13T12:00:00+02:00"
```

Result: fail-closed before fitting/staging. `epex_ch.price_eur_mwh` had zero
recent coverage, an infinite recent gap and age `119.5 days > 2.0 days`.
The release root was not created.

Observed source endpoints at the shadow reference:

- EPEX CH: `2026-03-15 22:00 UTC`;
- EPEX DE: `2026-03-15 22:45 UTC`;
- ENTSO-E: `2026-04-23 11:00 UTC`;
- hydro: `2026-03-09 00:00 UTC`.

Global status remains `NO_GO`. The next operational blocker is fresh, governed
acquisition of EPEX/ENTSO/hydro. The next code slice is the unified
candidate -> delivered-product audit -> remaining gates -> receipt -> atomic
promotion runner, plus disabling legacy `rolling_update` publication.

## Continuation - Reusable External LT Data Snapshot Contract

The shared local root requested by the desk is now canonicalized at:

`C:\Users\jbattaglia\pfc_local_data`

Implemented:

- `pfc_shaping/data/lt_input_sources.py`
  - explicit `PFC_LT_DATA_ROOT` / CLI root selection;
  - `external_v2` `current.json` + immutable generation contract;
  - resolved-path confinement and contract hash/size verification;
  - byte receipts plus semantic DataFrame hashes;
  - portable logical paths in candidate evidence.
- `scripts/create_lt_input_snapshot.py`
  - immutable staging/finalization of one multi-file generation;
  - atomic pointer replacement;
  - migrated inputs defaulted to non-eligible unless explicitly governed.
- `pfc_shaping/pipeline/production_phases.py`
  - core data parsed from the same bytes that are hashed;
  - common generation eligibility checked before fit;
  - explicit EEX workbook override;
  - optional neighbors accepted only when declared by the generation contract.
- `pfc_shaping/pipeline/candidate_bundle.py`
  - source and semantic frame receipts revalidated;
  - exact config archived in-bundle;
  - absolute source paths removed from portable run evidence.
- `pfc_shaping/pipeline/quality_gate.py`
  - duplicate timestamps are blocking;
  - non-finite production freshness policy values are blocking.

External migration performed without modifying repo caches:

- copied byte-identical legacy EPEX CH/DE, ENTSO, outages, hydro and commodities
  into the reusable C: root;
- materialized immutable generation `20260713-migrated-seed-v2`;
- generation contract hash:
  `763e623a915a211a91a7c49b93dc9e53a1a41309321db9a8b1e4dbb35dd7a936`;
- `calibration_eligible=false` with provenance
  `MIGRATED_LEGACY_CACHE_UNVERIFIED_NOT_CALIBRATION_ELIGIBLE`.

The external-v2 preflight selects the C: root and blocks immediately because
the migrated seed is not calibration eligible. No release root/staging is
created.

Separate read-only discovery:

- `C:\Users\jbattaglia\PFC_CT_DATA` contains ENTSO/EPEX/hydro files through
  approximately 2026-05-22 plus forecasts/outages;
- they were not copied into the LT generation or used by the model because the
  LT/CT independence contract and common acquisition provenance are absent.

The reusable ENTSO-related parquets were subsequently copied, without model
activation, to:

`C:\Users\jbattaglia\pfc_local_data\entsoe\imports\pfc-ct-data-20260522-v2`

The hash-bound import uses `entso_reusable_import.v1` plus an independent
archive receipt, hashes and verifies every file/frame, records
rows/columns/index coverage, and enforces `calibration_eligible=false` plus
`model_activation=FORBIDDEN_UNTIL_GOVERNED_IMPORT`. The importer is
`scripts/archive_entso_dataset.py`; its focused test passes.

Final data and IT re-roasts returned GO on the archive v2 after verifying the
real 7-file archive, recursive exact file set, resolved-path confinement, and
same-payload byte/frame checks. No `ENTSOE_API_KEY` exists in the process or
known local `.env` files, so a fresh governed ENTSO-E acquisition remains an
operational blocker rather than being silently replaced by an incomplete
public proxy.

Verification after this slice:

- external root / candidate / quality focused: `31 passed`;
- broad LT provenance/promotion/candidate/archive suite after final hardening:
  `247 passed, 1 warning`;
- `git diff --check`: no whitespace errors, CRLF conversion warnings only.

Global production remains `NO_GO`: no fresh calibration-eligible acquisition
generation exists, and the independent promotion P0 findings on trusted
receipts/rollback/legacy publication remain open.

## Continuation - Shared ENTSO-E Archive Completion

The desk confirmed that ENTSO-E data must remain reusable across projects on
`C:` rather than belong to PFC_LT. The canonical shared area is:

`C:\Users\jbattaglia\pfc_local_data\entsoe`

An inventory found that the seven `PFC_CT_DATA` payloads were already archived
byte-identically in `pfc-ct-data-20260522-v2`, but `PFC_phase10_c` contained
three different payloads that had not yet been retained in the shared store.
They were archived, without overwrite or model activation, as:

`C:\Users\jbattaglia\pfc_local_data\entsoe\imports\pfc-phase10-20260528-v1`

Verified contents:

- `de_renewable_forecast.parquet`: 112,320 rows, through
  `2026-03-15 23:45 UTC`, SHA-256
  `2122fc36f81848c47b4b30f6fd14949e793fd30179eadca0dd8f13f850bb85fd`;
- `entso_15min.parquet`: 186,093 rows, through `2026-04-23 11:00 UTC`,
  SHA-256
  `adb14dd329a066bc78b9609bcd9b3b3dcad7b8b1e69e258a0237da00e992cad5`;
- `outages_15min.parquet`: 86,016 rows, through
  `2026-06-14 23:45 UTC`, SHA-256
  `723b27937f61a09a3b690822ec1b2d8895753157e753e9afc0999dca3090e3e7`.

`verify_entso_archive` passed on the real archive. Its manifest remains
`calibration_eligible=false` and
`model_activation=FORBIDDEN_UNTIL_GOVERNED_IMPORT`. The LT `current.json`
pointer is unchanged and still selects `20260713-migrated-seed-v2`; therefore
this archival completion has no model-side effect.

Permanent Data and IT roasts confirmed that the two receipt-bearing archives
are complete relative to their source directories and cannot be activated by
the model. The unverified predecessor `pfc-ct-data-20260522`, which had no
archive receipt, was moved without deletion to:

`C:\Users\jbattaglia\pfc_local_data\entsoe\quarantine\pfc-ct-data-20260522-legacy-unverified`

All 14 files in the two verified archive directories were marked read-only to
reduce accidental mutation. This is advisory local hardening only: the user
still owns the files with `FullControl`, and receipts are not independently
signed or stored append-only. Consequently the shared local archive is GO for
cross-project reuse and integrity checking, but remains NO-GO as an
enterprise immutable/WORM source until IT provides service-visible storage,
restricted ACLs and independently authenticated receipts.

Local reuse was made explicit on 2026-07-13:

- user environment `PFC_LT_DATA_ROOT` now points to
  `C:\Users\jbattaglia\pfc_local_data`;
- user environment `PFC_SHARED_DATA_ROOT` now points to
  `C:\Users\jbattaglia\pfc_local_data` for shared-data tools and non-PFC
  consumers;
- `C:\Users\jbattaglia\pfc_local_data\entsoe\CATALOG.md` documents the two
  immutable import IDs, coverage, integrity checks and consumer rules;
- all seven source payloads from `PFC_CT_DATA` and all three source payloads
  from `PFC_phase10_c` were rechecked byte-for-byte against their shared
  imports; every SHA-256 matched;
- `resolve_lt_input_paths` selected `external_v2` generation
  `20260713-migrated-seed-v2` from the user root and retained
  `calibration_eligible=false`;
- the low-level resolver now rejects a missing explicit root; legacy repo
  inputs require the explicit research opt-in `allow_legacy_repo=True` and
  remain ineligible;
- `verify_entso_archive` binds the hash-receipted `manifest.import_id` to the
  archive directory name, so copied/renamed imports are rejected;
- shared archive source and destination roots must be absolute;
- the formerly executable `pfc_shaping.pipeline.rolling_update` entrypoint and
  its retained implementation now raise before dotenv, logging, locking,
  ingestion or publication; module import no longer creates log/lock dirs;
- dashboard and operations guidance no longer recommend the retired publisher
  or its DuckDB/output/log success contract;
- focused archive/resolver/legacy entrypoint tests: `25 passed`;
- affected governance matrix after these fixes: `126 passed`.

Permanent Quant, Data and IT roasts found no remaining P0/P1 in shared-root
selection, archive completeness, import identity or model-eligibility
separation after correction. The IT production verdict remains NO-GO for a
fresh signed eligible acquisition, service-owned immutable storage, explicit
scheduler/container environment injection and governed service identities.

Existing source-project copies were not deleted. They are legacy duplicates,
not canonical data roots, and may be removed only as a separate curated desk
cleanup after checking that no scheduled consumer still references them.

## Continuation - Authenticated Evidence Replay And EEX Peakload Correction

The promotion capstone now reruns the delivered-product audit from the exact
CSV, quote Parquet and monthly candidate bytes bound in the finalized bundle.
It sanctions CH, BASE+PEAK and a `1e-6 EUR/MWh` hard tolerance in code. A
coherently modified and rehased audit summary is therefore rejected.

Data and authorization hardening:

- `pfc_shaping/data/acquisition_contract.py` requires an Ed25519 attestation
  for every `calibration_eligible=true` governed acquisition contract;
- every consumed role, including optional inputs, is bound to the common
  acquisition, byte/frame receipt, PIT `available_at_utc` and sanctioned
  cadence where defined;
- 15-minute EPEX/ENTSO grid coverage and daily hydro coverage are recomputed
  from archived frames, so hourly/2-hour sparse substitutions are blocking;
- `pfc_shaping/pipeline/quote_conflict_policy_contract.py` requires an
  independent Ed25519 authority before a production QUOTE_CONFLICT exception
  can be accepted by the capstone;
- caller-supplied promotion trust anchors were removed from public APIs.

Release governance hardening:

- `promotion_receipt.v2`, `promotion_event.v2` and `promotion_head.v1` are
  signed and independently verified;
- every historical event reopens its bundle and archived receipt;
- `PFC_PROMOTION_JOURNAL_ROOT` stores the signed monotone head outside the
  mutable release root;
- rewinding `current.json` from B to the old signed A head is rejected;
- candidate path, rollback semantics, receipt IDs, link/junction confinement
  and the complete event chain are checked by the governed resolver.

Quantitative correction:

- European EEX Peakload is now Monday-Friday 08:00-20:00 local time with
  public holidays included across solver constraints, cascading,
  arbitrage-free calibration, assembler projection, local export audit,
  delivered-product audit and perfect-foresight validation;
- CH calendar 2027 is locked at exactly `3,132` hourly PEAK intervals;
- holiday/cantonal signals remain spot-shaping features only;
- constraint and residual hours are reconstructed from product calendars and
  load type instead of trusting manifest-declared hours.

Shared ENTSO-E store re-verification:

- `pfc-ct-data-20260522-v2`: 7 files, byte and frame receipts PASS;
- `pfc-phase10-20260528-v1`: 3 files, byte and frame receipts PASS;
- both remain `calibration_eligible=false` and
  `FORBIDDEN_UNTIL_GOVERNED_IMPORT` while reusable by other projects under
  `C:\Users\jbattaglia\pfc_local_data\entsoe`.

Verification:

- capstone plus signed quote-policy contract: `38 passed`;
- atomic promotion/journal attacks: `22 passed`;
- EEX Peakload and affected LT quant modules: `185 passed`;
- required/broad LT matrix: `252 passed, 1 skipped in 132.19s`;
- archive verification rerun against both real C: imports: PASS.

Status remains `NO_GO` for production. No fresh, calibration-eligible and
attested EPEX/ENTSO/hydro acquisition exists. IT has not provisioned immutable
service configuration, separated signing identities, KMS/HSM keys or the
protected external journal. Legacy publishers are not yet disabled, and the
unified candidate -> audit -> receipt -> promotion runner plus rolling-origin,
locked holdout, probabilistic calibration and Docker/CI work remain pending.

## Continuation - Complete Freshness And Constraint Closure Replay

The permanent data and quant roasts found four coherent false-PASS paths:
optional inputs were not replayed at promotion, dense grids could be shifted
off the absolute UTC phase, duplicate/overlapping known buckets could be
rehash-consistent, and a hard quote could be absent from every parent row.

Corrections:

- `production_phases.py` now produces freshness reports for every consumed
  AT/FR/IT EPEX neighbor and reads outages only when the governed feature is
  enabled;
- the capstone replays valuation and promotion freshness for those optional
  roles and rejects timestamps not aligned to the sanctioned UTC grid;
- constraint products, hard quote IDs and source IDs must be unique;
- residual child periods must be strict, disjoint subsets of the parent;
- source lineage is independently closed; active rows are globally disjoint,
  and every in-grid hard quote must be completely covered and energy-repriced
  by active descendants.

Attack tests cover a 7-minute grid shift, outages becoming stale after
valuation, duplicate known months, duplicate provenance rows and a hard quote
with no parent constraint.

A second quant roast showed that a strict one-parent rule rejected the real
`redundant_consistent` solver case and that local residual closure did not
exclude overlaps between independent active rows. The final implementation
therefore serializes `delivery_months`, reconstructs every active row's month
set, enforces global disjointness, and checks each in-grid hard quote against
the complete descendant partition. A real 12 MONTH + consistent CAL fixture
passes; overlapping MONTH/QUARTER and omitted-child constructions fail.

Verification:

- focused capstone/quality suite: `63 passed`;
- quant constraint closure suite: `80 passed`;
- broad required and affected LT matrix: `323 passed, 1 skipped, 1 warning`;
- compileall: PASS;
- `git diff --check`: PASS with CRLF conversion warnings only;
- protected CT and Power BI paths: unchanged.

The producer now uses the same shared cadence/absolute-grid validator as the
promotion capstone, so hourly or phase-shifted neighbor data is rejected
before any candidate computation rather than only at promotion.

The shared ENTSO-E archives remain physically available under
`C:\Users\jbattaglia\pfc_local_data\entsoe`: the verified imports contain
seven and three Parquet payloads respectively. They remain deliberately
`calibration_eligible=false` and
`model_activation=FORBIDDEN_UNTIL_GOVERNED_IMPORT`; archival reuse is not model
activation.

## Continuation - IT Anti-Replay And Legacy Publisher Shutdown

The permanent IT roast reproduced two production P0s: restoring both an old
signed mutable head and its old `current.json` pointer was accepted, and the
legacy daily/direct publishers bypassed the candidate/capstone/promotion flow.

Corrections:

- every signed head now has a contiguous sequence and an exclusive immutable
  history entry under the external journal root;
- the full head history is verified and the mutable mirror must equal its
  latest entry;
- a regression test restores the old pointer and old mutable head after
  `A -> B` and confirms rejection;
- repo-local legacy inputs are always `calibration_eligible=false`, while
  `load_inputs` requires an explicit governed `external_v2` generation;
- `run_pfc_production.py`, `scripts/run_daily.py`, `run_daily_pfc.ps1` and
  `register_daily_task.ps1` terminate before legacy publication.

Verification for this slice: `39 passed` across atomic promotion, input root,
candidate bundle and legacy entrypoint tests.

The final IT follow-up found that `scripts/run_daily.py` could still ingest
before reaching its publication stop. `main()` now terminates before argument
parsing or ingestion, with a monkeypatched regression test proving
`ingest_data` is never called.

Final broad required and affected LT matrix after IT hardening:
`329 passed, 1 skipped, 1 known warning`.

This does not make the workstation journal WORM. Deleting immutable entries is
still possible for the owning Windows account, and an already registered task
outside this checkout must be disabled by IT. Production remains `NO_GO` until
the journal is service-owned append-only storage and the unified governed
runner replaces the installed scheduler.

## Continuation - Cross-Project ENTSO-E Root And Hidden MLP Input

The user reconfirmed that all reusable ENTSO-E data belongs on `C:` and must be
available to projects other than PFC_LT. A byte-level inventory rechecked the
10 unique payloads still present in PFC_LT, `PFC_phase10_c` and `PFC_CT_DATA`.
Every source SHA-256 matches one of the two immutable imports under:

`C:\Users\jbattaglia\pfc_local_data\entsoe\imports`

User-scoped environment variables now distinguish the contracts:

- `PFC_ENTSOE_DATA_ROOT=C:\Users\jbattaglia\pfc_local_data\entsoe` is the
  direct cross-project read-library root;
- `PFC_SHARED_DATA_ROOT=C:\Users\jbattaglia\pfc_local_data` is the common
  parent used by archive tooling;
- `PFC_LT_DATA_ROOT=C:\Users\jbattaglia\pfc_local_data` selects a complete LT
  snapshot through `current.json`.

No heavy legacy file was moved, deleted or modified. Those repo/project-local
copies remain non-canonical duplicates. Both shared imports passed
`verify_entso_archive`, including byte and semantic-frame receipts.

The review also found one active hidden-input defect: `ShapeHourlyMLP.fit()`
auto-loaded repo-local `outages_15min.parquet` whenever its caller passed
`None`. That could bypass both the governed snapshot and
`quality.freshness.outages_enabled=false`. The fallback was removed and
`run_long_term_phase` now injects `inputs.outages_all` explicitly. With the
feature disabled or absent, outage features are neutral zeros.

Changed files for this slice:

- `.env.example`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- this handoff
- `pfc_shaping/lt/model/shape_hourly_mlp.py`
- `pfc_shaping/data/lt_input_sources.py`
- `pfc_shaping/pipeline/production_phases.py`
- `scripts/archive_entso_dataset.py`
- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `tests/test_archive_entso_dataset_script.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `tests/test_lt_input_sources.py`
- `tests/test_lt_entsoe_data_contract.py`
- external catalog
  `C:\Users\jbattaglia\pfc_local_data\entsoe\CATALOG.md`

Verification:

```powershell
python -m pytest tests/test_lt_entsoe_data_contract.py tests/test_archive_entso_dataset_script.py tests/test_lt_input_sources.py tests/test_quality_gate.py -q -p no:cacheprovider
```

Result: `43 passed`.

Follow-up permanent Data/IT/Quant roasts reopened root-confinement and PIT
controls. The final local implementation for this slice now:

- supports the direct `PFC_ENTSOE_DATA_ROOT` contract in the archive CLI;
- removes the archive writer fallback to `PFC_LT_DATA_ROOT`;
- requires direct and parent roots to resolve to the same store when both are
  configured;
- checks junction confinement before creating `imports`;
- records the complete top-level source Parquet inventory and rejects unknown
  ENTSO-like Parquets instead of silently dropping them;
- computes all roles that the current LT configuration would consume and
  validates their individual eligibility, common attested cutoff and PIT
  availability before the first payload read;
- marks `entsoe\curated` as a legacy, non-authoritative convenience copy.

Final focused result for the corrected input/archive slice:
`53 passed, 1 skipped`. The skip is the Windows directory-symlink attack test
when the current account lacks symlink creation privilege. Expanded governed
release/input matrix: `179 passed, 1 skipped`.

The final Data/Quant follow-up required independent capstone parity and an
exhaustive archive inventory. Corrections:

- new archives use `entso_reusable_import.v2`;
- recursive source Parquets must be recognized or passed explicitly through
  repeatable `--exclude-parquet`; no filename heuristic remains;
- the v2 verifier enforces
  `source_inventory == archived_files union explicit_exclusions` and rejects
  overlap or duplicates;
- the capstone captures the exact archived `config.yaml` bytes, verifies the
  run-manifest and bundle hashes, derives all consumed roles with the same
  configuration rules as `load_inputs`, and requires exact receipt-set
  equality;
- the capstone independently replays every role cutoff against the common
  contract cutoff;
- a declared `commodities` role is now loaded with `read_required` and cannot
  disappear silently;
- an orchestration regression test proves PIT failure occurs before any
  `read_parquet_snapshot` call.

Adversarial tests cover an omitted config-enabled outages receipt, an earlier
but divergent role cutoff, an unclassified `load_forecast_15min.parquet`, a
tampered v2 inventory partition and a directory-link escape where Windows
permits link creation.

Final affected release/input matrix: `183 passed, 1 skipped`. Compileall and
`git diff --check` pass; only CRLF conversion warnings are emitted. The skip is
the Windows symlink test under accounts without link-creation privilege.

The permanent IT re-roast found no remaining code P0. Residual P1s are
deployment properties: a privileged actor can race/replace the user-owned
root, current v1 archives do not prove a recursively complete source
inventory, and acquisition is not a locked transaction across a changing
multi-file source. Those v1 archives remain research-only and explicitly
`UNVERIFIED_MULTI_FILE_IMPORT`.

Production remains `NO_GO`: the current LT generation is still a migrated,
unsigned, `calibration_eligible=false` seed; local user-profile storage is not
enterprise WORM; and the remaining release/evidence work is unfinished.

## Continuation - V2 Archive Materialization And Semantic EEX Replay

Final Data/Quant gaps were closed without changing model parameters:

- `entso_reusable_import.v2` now records recursive source receipts, rescans
  the complete source set before publication and recomputes rows, columns,
  index bounds and semantic frame hash during verification;
- producer and capstone call the same strict `consumed_lt_input_roles`, so
  numeric/string truthy flags cannot activate a hidden producer-only role;
- governed EEX history validation is shared and rejects missing columns,
  invalid/future dates, non-finite prices and duplicate quote identities;
- the capstone reopens captured `eex_forwards_history` bytes and independently
  replays that semantic PIT contract.

Adversarial tests now include coherent manifest rehash after row-count
falsification, source inventory mutation during archival, truthy non-boolean
configuration and a hash-consistent EEX history dated after valuation.

Two new immutable research archives were materialized on `C:`:

- `C:\Users\jbattaglia\pfc_local_data\entsoe\imports\pfc-ct-data-20260522-v3-inventory`
  has 11 source Parquets partitioned into 7 archived plus 4 explicit
  exclusions. Manifest SHA-256:
  `f07dc8a1bbcff4d296f2e17fc45d8c97c9ec2850f0a829e1914b92eb98f13040`.
- `C:\Users\jbattaglia\pfc_local_data\entsoe\imports\pfc-phase10-20260528-v2-inventory`
  has 6 source Parquets partitioned into 3 archived plus 3 explicit
  exclusions. Manifest SHA-256:
  `13dfc8611f72a08885b7157104a279a014f046263312313af99f83070d029a15`.

All 10 archived payload hashes equal the corresponding v1 archive hashes. The
v1 directories were not modified. The external `entsoe\CATALOG.md` designates
the new inventory-bound IDs as preferred reusable archives. They remain
`calibration_eligible=false` and cannot activate the LT model.

Verification:

- focused archive/input/capstone matrix: `91 passed, 1 skipped`;
- expanded governed release/input matrix: `191 passed, 1 skipped`;
- compileall: PASS;
- `git diff --check`: PASS with CRLF conversion warnings only;
- protected `pfc_shaping/ct` and Power BI paths: clean.

The pre-existing dirty heavy file `data/eex_forwards_history.parquet` was not
modified by this tranche and must stay outside any curated commit. Production
remains `NO_GO` until a fresh signed calibration-eligible acquisition exists
and storage, keys, journal and scheduler are service-owned.

## Continuation - Pre-Run Governance And Evidence Finalization Barrier

The permanent Quant re-roast gives the data/PIT tranche `GO`: DST fallback is
calendar-correct and the capstone reconstructs the EEX source summary from the
governed Parquet, requiring its exact payload and hash.

The release flow now fails closed before finalization:

- `build_lt_candidate.py` creates staging first, captures signed thresholds
  and selected-lambda evidence, then loads inputs and performs the single LT
  solve;
- it stops at `CANDIDATE_STAGED_EVIDENCE_PENDING`;
- atomic finalization rejects missing or invalid `candidate_evidence.v1`;
- independent inputs require Ed25519 role/hash/size/PIT receipts and must equal
  the exact bytes in `pre_run_model_governance.v1`;
- the finalizer revalidates selected decision/config parity and threshold
  gate/bucket/quantile/cutoff semantics;
- paths are unique and the evidence manifest binds the pre-seal run payload.

New or materially changed files in this continuation:

- `.env.example`
- `.planning/HANDOFF.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- this handoff
- `pfc_shaping/data/lt_input_sources.py`
- `pfc_shaping/pipeline/production_phases.py`
- `pfc_shaping/pipeline/monthly_curve_authority.py`
- `pfc_shaping/pipeline/model_governance_contract.py` (new)
- `pfc_shaping/pipeline/candidate_evidence.py` (new)
- `pfc_shaping/pipeline/candidate_bundle.py`
- `pfc_shaping/pipeline/atomic_promotion.py`
- `scripts/build_lt_candidate.py`
- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `tests/test_model_governance_contract.py` (new)
- `tests/test_candidate_evidence.py` (new)
- `tests/test_atomic_promotion.py`
- `tests/test_candidate_bundle.py`
- `tests/test_governed_release.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `tests/test_lt_entsoe_data_contract.py`
- `tests/test_legacy_production_entrypoints.py`

Focused results: model-governance/evidence `19 passed`, finalizer/release
integration `48 passed`, capstone `63 passed`. Final aggregate after threshold
sample/quantile closure and bounded Windows rename retries: `187 passed,
1 skipped`. Compileall, `git diff --check` and protected-path checks pass; the
skip is the sanctioned Windows symlink-creation test.

Permanent Data verdict: `GO` for pre-run capture/authentication, with no P0/P1
remaining in that precise scope;
complete evidence assembler remains `NO_GO`. Next P0 is deriving hourly
export, export manifest, monthly gates, run parity, conflict inventory/policy
and product audit from staged bytes, then making that DAG the only capstone and
`governed_release` authority.

Production remains `NO_GO`. No candidate production was generated and no
heavy data file was intentionally modified by this continuation.

Final permanent IT verdict matches Data: pre-run and the structural finalizer
guard are `GO code`; the complete all-evidence-before-finalization tranche is
`NO_GO`. Required next implementation is one locked, idempotent
`assemble_and_seal_candidate_evidence(staging, run_id)` transition with
canonical paths, per-role schemas, allowlisted producer contracts, explicit
parent SHA-256 bindings and durable recovery between evidence seal and run
manifest update. The trust anchor must be read-only and the authority private
key inaccessible to builder/finalizer identities in deployment.

## Continuation - Complete Evidence DAG And Canonical Release Seal

The next structural tranche is now implemented locally. No production
candidate was generated and the global status remains `NO_GO`.

Implemented contracts:

- the governed candidate clock reaches `PFCAssembler.build` and output dating;
- solver delivery months are exact, contiguous and CH/DST-aware;
- input pointer, signed contract, acquisition identity, receipts and consumed
  role set are independently replayed;
- hourly export, export manifest, monthly gates and selected-config parity are
  derived only from staged candidate bytes;
- delivered-product evidence requires the exact signed source-hierarchy
  policy and enforces BASE, PEAK, implied OFFPEAK and finite-price checks;
- the canonical assembled seal derives all release roles and binds itself to
  the run manifest;
- strict finalization runs under the promotion lock, recovers only recognized
  dead-PID temporaries, quarantines a failed post-rename replay and verifies
  again from the final directory;
- the sanctioned release CLI accepts no operator artifact paths and requires
  `assembled_candidate_seal.v1` for registration, audit and promotion.

New files in this continuation:

- `pfc_shaping/pipeline/candidate_evidence_assembler.py`
- `pfc_shaping/pipeline/candidate_product_evidence.py`
- `pfc_shaping/pipeline/governed_release.py`
- `pfc_shaping/pipeline/quote_conflict_policy_contract.py`
- `scripts/assemble_lt_candidate_product_evidence.py`
- `scripts/finalize_lt_candidate.py`
- `scripts/run_governed_lt_release.py`
- `tests/test_candidate_evidence_assembler.py`
- `tests/test_quote_conflict_policy_contract.py`

Materially changed files include:

- `pfc_shaping/lt/model/assembler.py`
- `pfc_shaping/data/lt_input_sources.py`
- `pfc_shaping/pipeline/production_phases.py`
- `pfc_shaping/pipeline/candidate_evidence.py`
- `pfc_shaping/pipeline/atomic_promotion.py`
- `scripts/audit_ch_product_normalization.py`
- `scripts/build_lt_candidate.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `tests/test_atomic_promotion.py`
- `tests/test_candidate_evidence.py`
- `tests/test_governed_release.py`
- `tests/test_long_term_branch.py`
- `tests/test_monthly_forward_curve_integration.py`
- `tests/test_country_tz_plumbing.py`

Adversarial coverage now includes non-finite in-scope and out-of-scope quotes,
unsigned/foreign/mutated policy, weakened audit CLI arguments, duplicate or
unknown products, input receipt-set substitution, acquisition/path attacks,
DST month boundaries, orphan seal recovery, active/dead/ambiguous temporary
files, Windows path length and post-rename strict replay failure.

Verification completed before the final permanent-agent verdict:

- product/evidence/finalizer/release tranche: `121 passed`;
- fresh crash/non-finite focus: `6 passed`;
- CLI pending, seal and temporary focus: `4 passed`;
- strict finalizer quarantine focus: `2 passed`;
- targeted Ruff check: PASS.

Pending at this handoff update: the expanded fresh test matrix and the final
Quant/Data/IT re-roast. The release remains blocked until those close and,
independently, until a real fresh signed calibration-eligible input generation
produces a candidate that passes the complete chain. No heavy data file should
be included in a curated commit.

### Final integration correction after permanent roast

The first final roast found two real integration P0s hidden by release mocks:

1. the capstone expected post-run product evidence inside the immutable
   pre-run selected-lambda decision;
2. flattened evidence copies broke bundle-relative product and forward paths,
   while the canonical export intentionally did not duplicate production
   forward/solver payloads.

Corrections:

- `selected_lambda_decision` remains immutable independent evidence;
- release role `selected_config_artifact` now points to the deterministic
  `selected_config_run_parity` artifact;
- that parity carries canonical config, approval, solution and constraint
  parity while hash-parenting the pre-run decision and production manifest;
- `product_normalization_evidence.json` is now an additional mandatory role
  of the assembled seal, without changing the generic legacy evidence role
  set;
- the hourly export manifest hash-parents the exact production manifest;
- assembled capstone mode verifies the strict seal, captures canonical bundle
  artifacts, resolves portable paths from the bundle root, and derives the
  product-evidence triad without mutating source artifacts;
- strict assembled evidence is replayed inside the promotion lock immediately
  before pointer publication;
- public Python audit/promotion APIs require assembled registration by default;
- workflow JSON publication and idempotent evidence copying use fsync plus
  exclusive hard-link publication;
- candidate temporaries include PID, process-start identity and UUID; PID reuse
  is distinguished from a live writer and `ACCESS_DENIED` is fail-closed;
- finalization is idempotent after an abrupt post-rename stop when the complete
  final candidate verifies.

A new no-mock integration test materializes a real signed `external_v2`
generation with fresh EPEX/ENTSO/hydro Parquets and semantic EEX history, then
runs candidate assembly, product policy, strict seal, finalization,
registration and the real capstone. All six governance gates are `PASS`; the
fixture remains globally rejected only because it intentionally lacks the
neighbor-leakage and far-horizon research gates. This distinction confirms
that the release integration is executable without claiming production
readiness.

Verification added after this correction:

- capstone plus real assembled integration: `64 passed, 1 warning`;
- targeted Ruff on all changed evidence/release/capstone files: PASS;
- final expanded evidence/release/capstone/monthly matrix:
  `266 passed, 1 warning in 151.46s`;
- `git diff --check`: PASS with line-ending warnings only;
- protected `pfc_shaping/ct` and Power BI paths: clean.

The remaining warning is a pandas `FutureWarning` in gate-frame concatenation;
it does not change current gate values or decisions. Final permanent-agent
verdicts were still running when this result was recorded.

### Final step 4+5 roast closure

The permanent roasts then found four residual P1 issues, all corrected:

- assembled registration no longer persists a request before comparing the
  exact single-read role payloads against the canonical seal;
- the strict finalizer reserves `evidence_contract`, and the public generic
  finalizer cannot forge the assembled contract label;
- `promote_candidate` requires assembled evidence by default; generic legacy
  compatibility is an explicit non-production opt-in;
- `.promotion.lock` records PID plus process-start identity and recovers a
  stale lock after PID reuse while retaining conservative legacy-lock support.

Verification after the first three corrections:

- focused finalizer/governed-release/assembler/capstone matrix:
  `130 passed, 1 warning in 140.64s`;
- expanded Phase 14 evidence/release/data/monthly matrix:
  `340 passed, 1 skipped, 1 warning in 153.50s`;
- targeted Ruff and `git diff --check`: PASS, with line-ending warnings only.

After the final generic-finalizer reservation correction:

- atomic promotion plus real assembled integration:
  `56 passed, 1 warning in 67.19s`;
- targeted Ruff: PASS.

The sanctioned Windows symlink test remains the one expected skip. The pandas
concat `FutureWarning` is unchanged and does not alter current gate values.
Quant, Data and IT were re-roasted read-only after the final correction. All
three returned `GO` for steps 4+5 with no remaining P0/P1. Their independent
focused matrices reported respectively 141, 73 and 146 passing tests. This GO
is limited to the evidence/finalization/release tranche and does not authorize
production promotion.

Global production remains `NO_GO`. The reusable local pointer still selects
`C:\Users\jbattaglia\pfc_local_data\snapshots\20260713-migrated-seed-v2`,
which is explicitly not calibration eligible. No release candidate or heavy
data artifact was generated by this closure.

Operational environment check at closure: the user-level variables
`PFC_LT_DATA_ROOT` and `PFC_SHARED_DATA_ROOT` both point to the shared C: root,
but the already-running Codex process has not inherited them; use an explicit
CLI root or a newly started process. `ENTSOE_API_KEY` is absent from process,
user and machine environments. The fresh governed acquisition blocker is
therefore unchanged.

## 2026-07-13 EEX Historical Vintage And Rolling-Origin Contract Closure

This tranche closes the code contract required before a scientifically valid
Tier 2 rolling-origin campaign can exist. It does not create historical data
and does not authorize production.

Changed code and tests:

- `pfc_shaping/data/acquisition_contract.py`
- `pfc_shaping/data/eex_historical_vintage.py` (new)
- `pfc_shaping/calibration/monthly_curve_lambda_calibration.py`
- `scripts/run_monthly_curve_lambda_calibration.py`
- `tests/test_eex_historical_vintage.py` (new)
- `tests/test_monthly_curve_lambda_calibration.py`
- `tests/test_run_monthly_curve_lambda_calibration_script.py` (new)

Contract now enforced:

- exact signed catalog/Parquet/XLSX/parser/config/lineage replay;
- distinct external timestamp authority, fixed journal ID and complete receipt
  chain; no receipt signer exists in product code;
- real `available_at`, global economic-quote uniqueness and one fully linked
  revision chain;
- canonical finite CH BASE calibration settings, complete-case comparison and
  clean identical source/dependency closure before and after a non-smoke run;
- no selected hash or candidate from smoke/unsupported runs;
- public writer restricted to content-sealed strict-path artifacts, one byte
  serialization, exclusive output directory, exact semantic inventory and
  quarantine on failed post-publication replay.

Verification:

- targeted Ruff on all tranche files: PASS;
- targeted pytest: `41 passed, 1 known warning`;
- expanded acquisition/calibration/candidate/promotion/LT matrix:
  `244 passed, 2 known warnings in 95.21s`;
- `git diff --check`: PASS, with only existing CRLF conversion warnings;
- protected `pfc_shaping/ct/*` and `powerbi/*`: clean.

The warnings are the existing all-NaN insufficient-history fixture in
`monthly_curve_priors.py` and the existing pandas concat FutureWarning in the
real capstone test. Neither changes a gate or decision.

Permanent Quant, Data and IT agents independently re-roasted the final exact
state. All three returned `GO` for this code contract with no P0/P1. Remaining
deployment requirements are an IT timestamp service that owns its clock, an
append-only/WORM canonical journal and protected read-only trust anchors, plus
container/lockfile binary reproducibility.

Global production and real Tier 2 calibration remain `NO_GO`. The existing
`data/eex_forwards_history.parquet` is legacy latest-revision data and was not
modified by this tranche. Do not synthesize receipts or availability. The next
methodical slice is a pre-registered `tier2_pfc_evaluation_plan.v1` and
`tier2_fold_result.v1` contract with disjoint selection/holdout windows and a
canonical market-constrained baseline; execution must wait for genuine future
vintages.

## Tier 2 Monthly EEX Preregistration Closure

Added:

- `pfc_shaping/calibration/tier2_monthly_eex_evaluation.py`
- `tests/test_tier2_monthly_eex_evaluation.py`

The implemented acceptance boundary is intentionally narrow. It authenticates
only `tier2_monthly_eex_evaluation_plan.v1` from three exact regular-file paths:
plan, model-governance receipt and independent trusted-time receipt. The same
plan byte snapshot is used for semantic validation and both signatures. JSON
duplicate keys, symlink/reparse traversal, relative trust anchors, shared
authority keys, non-canonical origin timestamps, mutable proof construction
and type-confused numeric settings fail closed.

Scientific contract:

- scope: `MONTHLY_EEX_MASKED_QUOTE_ONLY`;
- full hourly Tier 2: `UNSUPPORTED_REQUIRES_HOURLY_PIT_TRUTH`;
- atomic rank basis: delivery month x PEAK/OFFPEAK segment using actual CH DST
  hours and EEX Peakload weekdays 08:00-20:00, holidays included;
- mandatory buckets: MONTH and QUARTER crossed with H0, H1, H2 and H3+;
- origin IDs are derived from campaign plus canonical `fold_as_of_utc`;
- target delivery starts strictly after the origin;
- holdout origins are disjoint from selection and strictly postdate the
  external preregistration receipt.

Freeze, fold-result, campaign and index validators currently raise
`UNSUPPORTED` immediately and are not exported. This is deliberate: their
earlier mapping-only prototype was offensively shown to accept false PASS
campaigns without EEX data, exact fold identity or an authenticated causal
chain.

Verification:

- targeted: `19 passed, 1 skipped`;
- expanded final calendar/governance/vintage/calibration matrix:
  `87 passed, 1 skipped, 1 known warning`;
- targeted Ruff: PASS;
- `git diff --check` for the tranche: PASS;
- protected `pfc_shaping/ct/*` and `powerbi/*`: unchanged.

The skip is Windows symlink creation being unavailable; relative trust anchors
are tested and rejected, and reparse checks use Windows file attributes rather
than Python 3.12-only APIs. The warning is the known all-NaN insufficient-
history fixture in `monthly_curve_priors.py`.

Permanent Quant, Data and IT agents re-roasted the final exact design. Each
returned GO with no P0/P1 for the narrow preregistration/identifiability slice.
Global production and every downstream Tier 2 campaign remain NO_GO.

Next methodical slice: define a signed row-level fold evidence package tied to
the verified EEX vintage catalog. It must inventory selection context,
expanding historical context, evaluation snapshot and masked target; replay
document/snapshot/revision/quote identities and availability; reconstruct the
revealing set; and recompute prediction, baseline, repricing, conservation and
metrics before any campaign/freeze/index validator can be enabled.

## Tier 2 Monthly EEX Fold Data-Lineage Closure

Added:

- `pfc_shaping/calibration/tier2_monthly_eex_fold_evidence.py`
- `tests/test_tier2_monthly_eex_fold_evidence.py`

The public path-only boundary now authenticates one exact SELECTION fold package
against the governed plan, candidate grid and signed immutable EEX vintage
catalog. It reconstructs the latest available CH snapshot, masked target,
revealing overlaps, retained complement, PIT historical contexts and future
diagnostics from row hashes and source identities. Package, Parquet, source XLSX
and parser bytes are checked for stability across verification. Trust-anchor
paths are absolute/non-reparse; governance, acquisition and execution keys are
mutually distinct and disjoint from all time-authority keys.

The accepted claim is intentionally limited to
`DATA_LINEAGE_REPLAYED_MODEL_OUTPUT_UNVERIFIED`. The verified token exposes no
segments or metrics, only `diagnostic_result_sha256`; the row artifact records
`permitted_candidate_input_row_hashes`, not alleged actual model inputs.
Manifest and result enforce `model_output_verified=false`,
`campaign_eligible=false` and `production_approved=false`. Arithmetic,
identifiability, repricing and conservation are checked only for internal
diagnostic consistency and do not prove candidate generation.

Two roast findings materially changed the boundary:

- Data P0: a target-leaking segment plus recomputed metrics passed the former
  arithmetic-only proof. The proof was demoted to data lineage and metrics were
  removed from the token.
- IT P1: a manifest could be replaced after read but before the initial package
  snapshot. The snapshot now precedes reading, read bytes are bound to its hash,
  and the exact race has an offensive regression test.

Verification:

```powershell
python -m pytest tests/test_tier2_monthly_eex_evaluation.py tests/test_tier2_monthly_eex_fold_evidence.py -q
```

Result: `45 passed, 1 skipped`.

Expanded EEX vintage/calibration/forward-constraint matrix: `208 passed, 1
skipped, 1 known warning in 46.92s`. The warning is the existing all-NaN
insufficient-history fixture in `monthly_curve_priors.py`. Targeted Ruff and
`git diff --check` pass; protected `pfc_shaping/ct/*` and `powerbi/*` remain
unchanged.

Permanent Quant, Data and IT agents independently re-roasted the final state and
all returned GO with no P0/P1 for this data-lineage slice only. Global
production, model metrics, campaign closure, HOLDOUT and full hourly Tier 2
remain NO_GO.

Next methodical slice: implement deterministic replay of both the governed
monthly candidate solver and the canonical market-constrained baseline from the
permitted signed rows. The proof must bind executable/runtime/config/source
closure and compare generated segments byte-for-byte before metrics can enter a
verified token. Do not enable campaign or freeze validators yet.

## Tier 2 BASE-Only Deterministic Replay Core Closure

Added:

- `pfc_shaping/calibration/tier2_monthly_eex_base_replay.py`
- `tests/test_tier2_monthly_eex_base_replay.py`

Changed for the shared mathematical contract:

- `pfc_shaping/calibration/monthly_forward_curve.py`
- `tests/test_monthly_forward_curve_constraints.py`

The new private core generates candidate and canonical baseline monthly BASE
outputs without target, diagnostic or score inputs. It accepts only CH BASE
retained quotes, a complete strict candidate config and historical CH BASE
context. It counts one seasonal observation per snapshot and calendar month,
requires 24 distinct snapshots per month, rejects duplicate identities and
reprices every retained quote at `1e-9`.

Shared-solver corrections made during permanent roast:

- own-market level anchors and missing-month priors are isolated by delivery
  year;
- missing-month shape uses the raw prior relative to the represented raw prior,
  with Europe/Zurich delivery-hour weights;
- D2 smoothing cannot bridge represented buckets or delivery years;
- objective weights are real, non-boolean, finite and non-negative before any
  branch;
- shape index labels are normalized to monthly periods, while NaT and
  post-normalization collisions fail closed;
- non-finite objective/KKT/output diagnostics fail closed.

The governed core additionally caps objective weights at `1e12`, rejects KKT
condition above `1e12`, least-squares fallback and numerical ridge, and converts
solver failures to `Tier2MonthlyBaseReplayNumericalError`. PEAK/OFFPEAK remain
explicitly unsupported pending a joint solver.

Verification:

```powershell
python -m pytest tests/test_tier2_monthly_eex_base_replay.py tests/test_monthly_forward_curve_constraints.py tests/test_monthly_curve_lambda_calibration.py -q
```

Result: `90 passed, 2 known warnings`.

Expanded solver/prior/Tier2 matrix: `167 passed, 1 skipped, 2 known warnings`.
Impacted integration matrix excluding three known wall-clock-dependent fixtures:
`25 passed, 3 deselected`. The three full-file failures arise because temporary
workbook `available_at` timestamps are later than the fixed
`2026-07-13T20:00:00Z` valuation; this slice does not modify that path. Targeted
Ruff and `git diff --check` pass; protected CT and Power BI paths remain
unchanged.

Permanent Quant, Data and IT agents independently attacked the final state.
All returned GO with no P0/P1 for both the shared-solver changes and private
BASE-only replay core. This is not signed PIT model-output evidence. Global
model metrics, SELECTION/HOLDOUT campaigns, hourly quality and production remain
NO_GO.

Next methodical slice: build the path-only signed execution wrapper. It must
consume `VerifiedSelectionFoldDataLineage`, reconstruct only the exact permitted
rows, bind executable/runtime/config/source closure, run this pure core and
compare candidate and baseline outputs byte-for-byte with the signed package.
Do not enable campaign/freeze/index validators yet.

## Tier 2 Same-Snapshot Replay Prerequisite

Changed:

- `pfc_shaping/calibration/tier2_monthly_eex_fold_evidence.py`
- `tests/test_tier2_monthly_eex_fold_evidence.py`

The public fold-lineage API is unchanged. Internally it now delegates to one
token-protected primitive that retains immutable records for exactly the
verified historical-context and retained-snapshot roles from the same
authenticated in-memory frame. It never rereads catalog, history or row
artifacts after returning a public token, and it never carries target or
diagnostic roles.

Records and the recursively frozen candidate grid are materialized before the
final exact package snapshot. A new offensive test mutates the package during
record materialization and proves the final inventory check rejects it.

Verification: `48 passed, 1 skipped`; targeted Ruff and diff-check PASS.
Permanent Quant, Data and IT all returned GO with no P0/P1 for this narrow
same-snapshot prerequisite.

The wrapper itself remains unimplemented and global model-output verification
remains NO_GO. Its design must still close four items identified by all roasts:
trust anchors only from pinned IT policy, an exact full BASE config selected
from the preregistered grid, replayable source/runtime manifests checked before
and after solve, and a pre-existing separately signed output commitment whose
claim remains unverified until independent replay.
