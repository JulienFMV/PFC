# Session Handoff - 2026-07-14 - D143 Governed Release CAS, Journal And Rollback

## Scope

Completed the software-side Phase 14 release-transition hardening. No CT,
Power BI or heavy desk-data file belongs to this stage. No real LT candidate
was promoted.

## Implemented contracts

- Immutable signed event/head journal is authoritative; mutable head and
  `current.json` are projections.
- Explicit expected-current CAS, immutable full request IDs, exact idempotent
  retry and distinct exits `40/41/50/51`.
- Governed signed rollback to an exact earlier PROMOTE event.
- Historical keys are replay-only; active/historical event, receipt and
  rollback keyrings are pairwise disjoint.
- Transition locks are never automatically removed.
- Stable release-domain UUID replaces path-spelling hashes for signed domain
  and journal namespace, and is pinned by immutable `release_domain.json`.
- Candidate root/key paths reject links and candidate files reject hardlinks.
- Candidate tree flush precedes rename; parent flush follows rename where
  directory fsync is available.
- Governance-critical JSON/YAML parsing rejects duplicates, YAML merge
  collisions, non-finite numbers, cycles, excessive depth/cardinality and
  alias-DAG amplification.
- Crash-leftover immutable-write hardlinks are recovered only when strict name
  and inode/device match; concurrent cleanup is idempotent and fsynced.

## Changed files in D143

- `.env.example`
- `pfc_shaping/pipeline/atomic_promotion.py`
- `pfc_shaping/pipeline/promotion_contract.py`
- `pfc_shaping/pipeline/governed_release.py`
- `pfc_shaping/pipeline/governed_release_cli_contract.py`
- `pfc_shaping/pipeline/strict_structured_data.py`
- `pfc_shaping/pipeline/candidate_evidence.py`
- `pfc_shaping/pipeline/candidate_evidence_assembler.py`
- `scripts/run_governed_lt_release.py`
- `scripts/build_lt_candidate.py`
- `scripts/finalize_lt_candidate.py`
- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `scripts/audit_ch_product_normalization.py`
- `pfc_shaping/tools/OPERATIONS.md`
- `tests/test_atomic_promotion.py`
- `tests/test_promotion_contract.py`
- `tests/test_governed_release.py`
- `tests/test_run_governed_lt_release_script.py`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `tests/test_candidate_evidence.py`
- `tests/test_candidate_evidence_assembler.py`
- `tests/test_candidate_bundle.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Final verification

```text
python -m ruff check <D143 runtime and test files>
All checks passed!

python -m pytest tests/test_atomic_promotion.py tests/test_promotion_contract.py \
  tests/test_governed_release.py tests/test_run_governed_lt_release_script.py \
  tests/test_audit_ch_product_normalization_script.py \
  tests/test_check_monthly_curve_promotion_from_manifests.py \
  tests/test_candidate_evidence.py tests/test_candidate_evidence_assembler.py \
  tests/test_candidate_bundle.py -q
351 passed, 2 skipped, 1 warning in 243.88s

git diff --check
pass; expected LF/CRLF notices only
```

The skips are real Windows symlink creation tests; deterministic link and
junction simulations pass. The warning is the existing pandas concat future
warning in the real capstone assembly test.

## Roast history

Repeated read-only Systems, Security and IT reviews found and drove fixes for
receipt/event binding replay, historical key separation, candidate-root link
escape, tree durability, trust-anchor lexical links, key-rotation recovery,
timestamp precision, UNC alias domains, duplicate JSON/YAML keys and policy
loader bypasses, non-finite values, cyclic/deep/alias-amplified structures,
domain drift and immutable-write crash leftovers. Final Systems, Security and
replacement IT/Operations reviews returned software-stage GO with no P0/P1.

## Remaining production blockers

Production remains `NO_GO` until IT supplies evidence for:

- unique immutable domain UUID provisioning per logical release;
- service-account ACLs on releases, trust stores and journal;
- WORM storage or an independent monotonic anti-rewind witness;
- HSM/KMS-backed private-key custody and rotation procedures;
- kill/restart, concurrent-writer, projection-repair and rollback drills on the
  exact Windows/SMB volume and aliases used in production;
- hardlink/rename/durability behavior on that exact target filesystem;
- a fresh real candidate with signed independent inputs and all promotion gates
  passing. Do not use an allow-failed-gates path.

## Next action

Run the documented IT target-volume drill plan without promoting a real
candidate. If the storage contract passes, generate a fresh auditable CH LT
candidate under the governed UUID and execute register, audit and promote as
separate service identities. Keep global production `NO_GO` until every real
manifest and external infrastructure proof is archived.
