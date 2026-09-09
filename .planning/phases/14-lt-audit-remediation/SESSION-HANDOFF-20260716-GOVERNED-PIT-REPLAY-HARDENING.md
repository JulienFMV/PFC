# Phase 14 Handoff - Governed PIT Replay Hardening (2026-07-16)

## Canonical locations

- Repository: `C:\Users\jbattaglia\PFC_LT`
- Branch: `fix/lt-audit-remediation`
- HEAD at start: `2f68125bff869ccb21c1e20df0201ad024ed27d3`
- Shared data root: `C:\Users\jbattaglia\pfc_local_data`
- `FMV_DATA_ROOT`, `PFC_LT_DATA_ROOT`, and `PFC_SHARED_DATA_ROOT` all point
  to that shared root.
- No old `H:` repository was changed or deleted by Codex after the manual
  transfer. Remaining `H:` strings are historical research evidence,
  business-source paths, or disabled legacy entry points.

The shared pointer still targets `20260713-migrated-seed-v2`. That generation
is `MIGRATED_UNVERIFIED`, is not calibration eligible, and is not evidence for
a prospective candidate.

## Scope completed in the working tree

1. Added three-authority v2 acquisition governance: acquisition, trusted time,
   and source-journal checkpoint, with distinct keys, contiguous receipts,
   exact bundle root, and cutoff equal to maximum governed availability.
2. Added immutable EEX historical-vintage catalog verification, PIT
   materialization, publisher closure, and exact consumed-frame binding.
3. Required a v2 mono-role acquisition contract for the hard-forward EEX
   workbook. Legacy v1 self-declared availability now fails closed.
4. Required distinct raw, derived, parser, parser-config, and quality-report
   paths for core roles.
5. Added allow-listed deterministic replay for core roles and capstone live
   bundle validation. This proves only bronze Parquet to derived Parquet. It
   does not prove exact provider response bytes to bronze or derived data.
6. Hardened publisher locking against hardlink confused-deputy attacks and
   bound the EEX catalog bytes before deriving the transitive copy closure.
7. Made the EEX vintage catalog mandatory at the general resolver boundary,
   including a fully re-signed negative test.
8. Preserved `schema_version` through candidate evidence reconstruction and
   removed a pandas concatenation deprecation from capstone gate replacement.

## Verification

- v2 governance: `17 passed in 98.37s`
- Fully re-signed EEX history without vintage catalog: `1 passed in 5.38s`
- Canonical candidate through capstone, FutureWarnings fatal:
  `1 passed in 72.30s`
- Earlier focused capstone matrix: `10 passed in 32.19s`
- Earlier acquisition/EEX/history/quality matrix: `78 passed in 48.51s`
- Post-roast v2/input-source/capstone matrix: `53 passed, 2 skipped in 53.11s`
- Targeted Ruff passed. `git diff --check` passed with expected Git LF/CRLF
  notices.
- `git fsck --connectivity-only` found no connectivity error. Dangling objects
  are recoverable Git residue, not repository corruption.

## Explicit non-closures

Production remains `NO_GO`.

- Operational EPEX uses Energy Charts JSON without retaining the exact body.
- Operational ENTSO-E may use Energy Charts or `entsoe-py`, but does not retain
  the exact JSON/XML response set.
- Hydro may use SFOE/BFE, Databricks, or cache state without an exact raw
  envelope and unambiguous source identity.
- Current replay starts from bronze Parquet and cannot independently detect an
  upstream parser or source-selection error.
- Publication events need a distinct signature and external monotone anchor
  for adversarial rollback defense.
- EEX revisions need explicit `UPSERT`/`DELETE` tombstones for withdrawals and
  re-keyed quotes.
- Exact-volume ACL/WORM, HSM/KMS, multi-host SMB, power-loss, backup/restore,
  and DR evidence remain external IT prerequisites.

## Next implementation slice

1. Define `lt_raw_envelope.v1` with exact response bytes, body hash, request
   parameters, media type, deterministic document ID/order, source identity,
   and trusted receipt time.
2. Add pure allow-listed transforms for Energy Charts price JSON, the named
   ENTSO-E XML bundle, and SFOE/BFE OGD17 CSV.
3. Add `pfc_shaping/data/governed_lt_acquisition.py` and a separate build
   script. The publisher must remain unable to acquire or sign data.
4. Treat each fallback as a separate governed source identity and eligibility
   policy. Do not merge ambient mutable caches into a calibration-grade role.
5. Bind quality evidence to raw, transform, config, derived frame, policy, and
   metrics. Pin pandas/numpy/pyarrow/tzdata runtime fingerprints.
6. Add real-format golden JSON/XML/CSV replay vectors, then repeat independent
   Quant, Security, and Data Architecture roasts.

## Git hygiene

`data/eex_forwards_history.parquet` is a pre-existing local heavy-data change.
Do not stage, modify, revert, or commit it. Do not stage generated candidates,
Power BI data, CT files, or shared-data artifacts.

## Post-roast closure

The independent Quant roast initially found a P0: optional neighbor EPEX roles
could be consumed without replay, and configured `commodities`/`outages` roles
could be consumed without a dedicated transform. The working tree now uses one
`REPLAY_GOVERNED_LT_INPUT_ROLES` authority for artifact separation and replay:
EPEX CH/DE/AT/FR/IT, ENTSO, and hydro. V2 consumption rejects commodities and
outages before model I/O until their dedicated replay exists. Fully re-signed
`epex_fr` alias and semantic-mutation attacks are covered. Quant re-roast:
requested P0 `CLOSED`, targeted merge correction `GO`.

The independent Security roast confirmed the hardlink lock, publisher EEX
closure, and mandatory resolver catalog fixes. It found no P0 but two P1:

- publication events remain unsigned and have no monotone anchor outside the
  writable data root;
- EEX verification previously reopened paths and did not explicitly equate the
  verified history hash with the consumed receipt.

The second P1 is now addressed with a shared descriptor-bound, single-link,
pre/open/post identity reader for catalog, history, source documents, and
parser code. Production requires the verified catalog hash to equal the first
captured catalog bytes and the verified history hash to equal the consumed
receipt before PIT materialization. EEX/history tests report `36 passed`, the
canonical capstone pair reports `2 passed`, and the hardlink reader attack
passes in rejection.

The first P1 remains open by design. A publication signature without an
external monotone anchor would still allow signed-chain truncation. Do not
merge or promote this complete governance slice until an IT-owned publication
authority and append-only/external anti-rollback checkpoint are specified,
implemented, and re-roasted.

## Signed-prefix prototype roast and D147

After migration to `C:\Users\jbattaglia\PFC_LT`, a signed external-directory
prototype was implemented in the working tree and attacked by independent
Security and Data Architecture roasters. Both returned merge `NO-GO`.

Merge-blocking findings:

- deleting the last signed anchor and restoring the previous pointer produces
  a valid signed prefix; a directory listing is not a monotone head;
- per-data-root locks cannot serialize the same publication domain across
  hosts or roots;
- the active migrated pointer has no explicit governed bootstrap transition;
- candidate/capstone archives do not yet preserve the publication-authority
  receipt and head proof;
- exact retry after a successful commit initially failed, and consumer reads
  were weaker than publisher reads.

The last two software defects were corrected locally: operation IDs are now
looked up before CAS expectation checks, exact retries repair/return the
existing commit, ID reuse with different inputs fails, staging is revalidated
around rename, and consumer pointer/contract/YAML/JSON/Parquet reads use stable
single-link descriptors. These corrections do not make the filesystem
prototype mergeable or monotone.

D147 replaces it with a three-authority protocol:
`PREPARE signed intent -> external linearizable CAS signed receipt -> FINALIZE
with fresh signed HEAD observation`. The local event/anchor filesystem chain is
to be removed, not promoted. Specification:
`LT-SNAPSHOT-PUBLICATION-EXTERNAL-CAS-RFC-20260716.md`.

Verification after transfer and roast corrections:

- governed v2 publication/attack suite: `26 passed`, then exact retry subset
  `4 passed, 24 deselected`;
- LT input source plus v2 suite after descriptor hardening:
  `61 passed, 2 skipped`;
- EEX/input-source/quality matrix: `95 passed, 2 skipped`;
- candidate assembler after correcting its raw/derived alias fixture:
  `40 passed`;
- canonical candidate/capstone pair: `2 passed`;
- nominal manifest-backed capstone: `1 passed`;
- targeted Ruff, Python compilation, CLI help and `git diff --check` passed.

No curated commit was created. The heavy local
`data/eex_forwards_history.parquet` modification remains untouched and must
stay excluded. Production and merge status for the publication slice remain
`NO_GO` until D147 is implemented and re-roasted.

## D147 final software closure

D147 has since been implemented and independently re-roasted. The production
protocol is publisher-signed intent, authenticated `get_head`, external
linearizable CAS receipt, fresh nonce-bound HEAD observation, then fail-closed
pointer projection. The reference authority uses SQLite transactions and is
explicitly `NON_PRODUCTION_TEST_ONLY`; its attack matrix covers competing
connections, stale genesis, alternative branches, exact retry, operation-ID
reuse, TTL/nonce and anchor-key rotation.

Security and Quant/Data re-roasts found no remaining P0 and returned software
merge GO after closure of their P1 findings. IT/Operations remains merge
`NO_GO` on bootstrap, key-rotation HEAD semantics, expired-observation/crash
recovery, status taxonomy and publisher packaging. The rejected filesystem writer is
not present in the sealed LT wheel, `ADOPT_LEGACY` is rejected at the signing
contract, mTLS pins only the configured CA, historical keyrings support replay,
and the capstone consumes descriptor-bound EEX and policy bytes. The broad
independent matrix reports `407 passed, 4 skipped`; focused software matrices
and Ruff are green.

Production remains `NO_GO` pending the independent IT backend and controls,
prospective provider-raw capture and a fresh audited CH candidate. EEX
withdrawal tombstones remain a separate PIT improvement. The heavy local
parquet remains excluded.
