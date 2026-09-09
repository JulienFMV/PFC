# Session Handoff - 2026-07-15 - D145 Signed Admission And Publication

## Canonical Workspace

- Repo: `C:\Users\jbattaglia\PFC_LT`
- Branch: `fix/lt-audit-remediation`
- Parent HEAD at session start: `21901f4048b2a11751b194e320f792d84a4a9fa9`
- The former `H:` repo copy was not modified or deleted.
- No runtime path points to the old `H:\...\PFC_LT` repo or
  `C:\Users\jbattaglia\PFC_phase10_c`.
- The remaining `H:` paths in `pfc_shaping/config.yaml` are external EEX and
  OMPEX source locations. OMPEX is benchmark-only.

## Closed Software Scope

- Signed canonical v3 REGISTER request and workflow/release domain binding.
- CAS-safe candidate/request revalidation and exact replay after key rotation.
- Separate request, audit-result and promotion-result namespaces and ACL model.
- STATUS/AUDIT/PROMOTE read-only lookups do not create or repair governed roots.
- All post-CAS failures become exit `51`, `commit_status=COMMITTED`, and require
  exact projection repair.
- Installed runtime self-hash over an exact Python member allowlist.
- Positive-schema probabilistic dashboard export with finite/order/state gates.
- Operational runbook for wheel admission, IT provisioning, candidate ACL
  freeze, policy replay, forensic quarantine and target-volume qualification.

Principal runtime files:

- `pfc_shaping/pipeline/release_request_contract.py`
- `pfc_shaping/pipeline/atomic_promotion.py`
- `pfc_shaping/pipeline/governed_release.py`
- `pfc_shaping/pipeline/governed_release_cli_contract.py`
- `pfc_shaping/pipeline/probabilistic_output_governance.py`
- `pfc_shaping/package_contract.py`
- `dashboard/pages/2_pfc_curve.py`
- `dashboard/utils.py`
- `pfc_shaping/tools/OPERATIONS.md`

## Verification

- Final governance matrix:
  `360 passed, 2 skipped in 569.80s`.
- Final repository-wide suite:
  `1909 passed, 11 skipped, 23 warnings in 2439.58s`.
- Focused Ruff scope: pass.
- `git diff --check`: pass; Windows LF/CRLF notices only.
- Final IT/Operations roast: conditional software GO, no P0/P1; its independent
  release/request/atomic run reported `159 passed, 2 skipped`.
- Final Quant roast: GO for governed probabilistic publication, no P0/P1.
- No agent modified the main worktree during final roasts.

The full suite found and closed stale direct-export suffix tests, a
non-transactional `sys.modules` fixture, fixtures that did not provision the
new REGISTER domains/namespaces, and Windows storage-drill success budgets that
were shorter than 1,000 fsynced replacements under load. Runtime deadlines,
worker termination and injected timeout tests remain unchanged. Independent
IT re-roast returned GO with no P0/P1 or test weakening.

Reproducible package proof:

- members: `69`;
- wheel SHA-256:
  `d944cb56914fd4c6fe61c98e4398a1710ede30f365144a5ec49e26ba5c624fad`;
- embedded source revision:
  `2cca924acf184f6e442598c852325aa2a8944908f8d2f6e84eae7cc737f8a6c7`;
- both wheel audits: pass;
- isolated install and CLI smoke: exit `0`.

## Explicit Exclusions And Status

- Do not stage `data/eex_forwards_history.parquet`; it is a modified local heavy
  data artifact.
- No `pfc_shaping/ct/*`, `powerbi/data/*` or `powerbi/PFC_QA.*` change exists.
- No real candidate was generated, audited or promoted in D145.
- Shared data remains `20260713-migrated-seed-v2`,
  `MIGRATED_UNVERIFIED`, `calibration_eligible=false`, availability through
  2026-06-08.
- Global production status is `NO_GO` despite conditional software GO.

## Next Required Stage

1. Curate and commit all required code, tests and documentation while excluding
   heavy/generated data; verify the staged inventory before commit.
2. Run the full repository test suite from the canonical `C:` workspace.
3. Capture fresh PIT EEX/ENTSO-E inputs into the governed shared-data root.
4. Generate one fresh CH LT candidate with monthly solver and final PEAK
   calibration enabled.
5. Run delivered BASE/PEAK/implied-OFFPEAK audit and strict export without a
   failed-gate override.
6. Compare against OMPEX only as an imperfect independent benchmark.
7. Obtain new Quant and IT roasts before any promotion decision.
8. Require target SMB ACL/WORM, HSM/KMS, multi-host, power-loss,
   backup/restore and DR evidence before production authorization.
