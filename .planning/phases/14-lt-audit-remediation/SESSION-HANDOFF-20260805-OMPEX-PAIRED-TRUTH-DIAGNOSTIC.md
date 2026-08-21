# Session handoff - OMPEX paired-truth diagnostic

Date: 2026-08-05  
Decision: D-20260805-224  
Status: `DESCRIPTIVE_ONLY_NO_SUPERIORITY_DECISION`

## Outcome

The historical HPFC-to-OMPEX comparator was scientifically unsafe: it chose
the candidate/OMPEX alignment with the smallest direct distance and required
no realised truth. The entry point now requires a local hash-bound candidate,
OMPEX vintage and truth on one exact complete UTC hourly grid. It performs no
automatic alignment, inner join, imputation or duplicate aggregation.

The output contains paired errors of both forecasts against the same truth and
protected subgroup diagnostics, but it cannot decide superiority. Candidate
freeze chronology, OMPEX at-origin availability/timestamp semantics,
independent truth publication timing, multiple origins, preregistered margins,
dependence-aware inference and multiplicity control remain open.

No real price outcome was opened. All functional tests used synthetic local
fixtures. There was no `H:` or Databricks access and no remote write.

## Changed files

- `pfc_shaping/validation/ompex_truth_comparison.py`
- `scripts/compare_hpfc_ompex_benchmark.py`
- `tests/test_compare_hpfc_ompex_benchmark_script.py`
- `.planning/phases/14-lt-audit-remediation/OMPEX-PAIRED-TRUTH-DIAGNOSTIC-CONTRACT-V1.json`
- `docs/research/OMPEX-PAIRED-TRUTH-DIAGNOSTIC-REPORT-20260805.md`
- `docs/research/OMPEX-INDEPENDENT-BENCHMARK.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Exact hashes

- module:
  `dc62dc47a55af91f0ce42f8a5c0708066450676a0df8c4e785533a717ea4fec0`
- entry point:
  `0b8ab7b3e9a7483258d4a3de91e7fa28ca8db93a3fcaec78f06bb72e2a218f54`
- tests:
  `deeb69d69c143739cfac3f9c56c80d96062a11bce3f9657564b1660fa0688195`
- machine contract:
  `a3887bd8778f0e9b3342bd202d5b2cb731933c92f8fb6fdd5d77196e9b5f9754`
- technical report:
  `0b54db7a907c09ef7e538d24711949e4fedb8918d5cf14e81688270b9d2ddec8`
- updated independent-benchmark note:
  `b2d69a2a75f0165376d968b6563efb2623e90ef0ecce374ec660753b3631eed9`

## Validation

All commands ran from guarded cwd/Git root
`C:\Users\jbattaglia\PFC_LT`, with mutable temp/cache paths below `build/`.

- focused OMPEX module and script matrix: `19 passed`;
- adjacent historical EEX governance: `2 passed`;
- LT/CT import boundary: `17 passed, 1 skipped`;
- final combined matrix: `40 passed, 1 skipped`;
- Ruff passed on all changed Python files;
- contract JSON parsing, Python compilation and `git diff --check` are required
  in the final closure command.

Two broad pytest invocations timed out because each used a fresh repo-local
Python bytecode cache and left one verified child pytest process after the
shell timeout. Each process was identified by its exact workspace basetemp and
stopped; a read-only process check confirmed no other pytest process. Reusing
the established repo-local cache made the LT/CT matrix finish normally. This
was a test-harness performance issue, not a functional failure.

## Methodological basis and report surface

The contract records Diebold-Mariano paired predictive accuracy,
Giacomini-White out-of-sample/conditional ability, Romano-Wolf dependent
multiple-testing control and Lago et al. electricity-price forecasting best
practice. The full technical narrative is in
`docs/research/OMPEX-PAIRED-TRUTH-DIAGNOSTIC-REPORT-20260805.md`.

The Data Analytics validation/report workflow shaped the report around exact
definitions, denominators, failure modes and decision boundaries. A standard
interactive HTML artifact was not produced because the managed-workstation
contract forbids launching project executables and browser/Playwright
runtimes; the governed Markdown report records this concrete limitation.

## Next permitted batch

- Define the signed receipt schemas for candidate freeze, OMPEX at-origin
  availability/timestamp semantics and independent realised truth without
  choosing numeric margins from outcomes.
- Keep the D223 candidate selection scaffold dormant until governed EEX/
  ENTSO-E and direct-CH power/dependence evidence exist.
- Do not reopen T057, use AFRY numeric values, query Databricks for already
  local evidence or read `H:` during routine scoring.
