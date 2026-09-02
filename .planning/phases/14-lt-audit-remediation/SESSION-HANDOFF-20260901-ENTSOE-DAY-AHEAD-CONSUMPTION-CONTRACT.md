# Session handoff - ENTSO-E day-ahead consumption contract

Date: 2026-09-01

## Outcome

A pure LT consumer boundary now distinguishes final realized day-ahead truth
from observations causally usable at a stated as-of, handles A01/A03 intervals
deterministically and prevents spot from changing solver-authoritative monthly
means. The implementation is local, typed, fail-closed and connector-free. It
ran no Databricks query and consumed no restricted or heavy data.

The increment is code-ready but not real-data-admitted. Current ENTSO-E Silver
persists interval bounds and availability lineage but not the producer's
`curve_type`; effective-dated AT/DE-LU auction choices also need independently
admitted semantic and LSEG evidence. The consumer refuses both gaps rather
than inferring them.

## Functional contract

- `realized_final` requires `is_final=true`, accepts historical backfill truth
  and grants no PIT authority.
- `causal_asof` requires an explicit as-of. `FMV_FIRST_SEEN` is usable only at
  or after its actual first-seen time. `SOURCE_DOCUMENT_CREATED` is usable only
  with explicit proof that it is original day-ahead publication and precedes
  delivery. `UNKNOWN_BACKFILL` always fails causal use.
- A01 duration must equal native resolution. A03 duration must be a positive
  integer multiple and is expanded at native cadence. The output uses
  half-open UTC intervals and independently rejects overlaps and true gaps.
- Effective-dated rules must exactly cover the requested window, cannot
  overlap, cannot leave gaps and cannot be crossed by one source block. AT and
  DE-LU require explicit sequence 1 or 2. Markets with LSEG coverage require a
  passed independent control; IT-North stays explicitly ENTSO-E-only.
- Expanded rows retain market timezone/local offset, UTC and source bounds,
  native resolution, series/classification, curve type, availability basis and
  mode, quality/finality/history indicators and source provenance.
- `build_monthly_zero_mean_spot_shape` requires
  `monthly_level_authority="solver"` and duration-centers the spot component
  inside each local month. It cannot grant monthly-level authority.

## Files changed by this session

- `pfc_shaping/validation/entsoe_day_ahead_consumption.py` - new pure consumer
  contract.
- `tests/test_entsoe_day_ahead_consumption.py` - synthetic contract tests.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` - appended
  D-20260901-266.
- `.planning/HANDOFF.md` - current-state pointer updated.
- this handoff.

No pre-existing ENTSO-E/LSEG module, SQL template, CT file, Power BI file or
business-data file was modified.

## Tests and receipts

- `dauset1`: initial targeted run, `15 passed, 2 failed`; both failures were
  diagnostic-order defects for null/negative intervals and were corrected.
- `dauset2`: targeted contract, `17 passed`.
- `damatrix1`: ENTSO-E day-ahead, temporal diagnostic, LSEG reconciliation,
  availability and LT/CT matrix, `135 passed, 1 skipped`.
- `daltmin1`: required `test_arbitrage_free.py`, `test_cascading.py`,
  `test_water_value.py`, `test_lt_ct_imports.py`, `58 passed, 1 skipped`.
- `daruff1`: Ruff check passed; format check identified two files and they were
  formatted by `dafmt2`.
- Hostile review then closed effective-rule boundary crossing and retained the
  remaining causal/quality flags in output.
- `dauset3`: final targeted contract after that correction, `18 passed`.
- `daruff2`: final targeted Ruff check passed; `dafmt3` formatted the final
  module delta.

- `damatrix2`: final adjacent matrix after the hostile correction,
  `136 passed, 1 skipped`.
- `daltmin2`: final required minimum LT matrix, `58 passed, 1 skipped`.
- `daskip1`: isolated LT/CT boundary rerun, `17 passed, 1 skipped`; the sole
  skip is `tests/test_lt_ct_imports.py:145` because TensorFlow is not installed.
  This optional-import skip is unrelated to the day-ahead contract.
- `dapkg1`: LT package contract, `26 passed`.
- `daruff3` and `dafmt4`: final Ruff check and format check passed.
- Standard repository `git diff --check`: pass. A separate line scanner also
  found no trailing spaces in the three new files. Git reports only the
  pre-existing LF-to-CRLF checkout warning for the two already-modified
  canonical handoff files; no bulk line-ending rewrite was performed.

Final source hashes before this handoff-only evidence update:

- consumer module SHA-256:
  `5a35d417f469c1177f87db2ea19ebb7c829fd7df7c82337236376d17c9bec4d9`;
- targeted tests SHA-256:
  `2065d4457d202dd2b40d11d813f3ef259ceb3a2ae92fdd32816ea4ac4918f97f`.

All mutable test state and receipts are below
`build/workspace-local-supervisors/<run-id>/`.

## Databricks cost

- SQL statements: 0.
- Warehouse starts/resizes/creates: 0.
- business rows opened: 0.
- writes: 0.
- network calls: 0.
- incremental DBU/Azure exposure: zero.

## Residual risks and next external evidence

1. Preserve `curve_type` from producer Bronze into the governed Silver/export
   projection. Do not infer it from interval width in the LT consumer.
2. Freeze effective-dated AT and DE-LU classification-sequence rules from
   source semantics and independently admitted LSEG reconciliation. No default
   sequence exists.
3. Bind a real PRD export and its selection evidence into the existing v4
   replay/snapshot chain before model use.
4. Keep the model gate and T057 state unchanged until the governed input and
   independent future-holdout requirements are satisfied.

These are real-data admission blockers, not reasons to weaken the local
contract. A precise data-engineer request is now justified for Silver/export
`curve_type` preservation; no broad pipeline reinvestigation is needed.

## Hostile-review checklist

- look-ahead and late backfill: blocked by typed availability bases;
- technical vintage versus economic revision: never silently collapsed;
- sequence choice and source substitution: explicit rules only;
- duplicate/overlap/gap/right-edge handling: fail-closed half-open intervals;
- DST: UTC expansion plus explicit IANA market timezone, with 23/25-hour tests;
- monthly levels: duration-weighted zero-mean shape only;
- CT dependency: none;
- Databricks spend: zero.
