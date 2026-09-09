# Session handoff - LSEG/ENTSO-E July latest reconciliation

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decision: D-20260904-288

## Outcome

The existing July 2026 ENTSO-E `realized_latest_candidate` is now reconciled
against an independently extracted LSEG latest-observation frame for CH, AT,
DE-LU and FR. Every zone has all 744 expected UTC hours, full overlap, no gaps
and exactly zero price difference. IT-North has all 744 ENTSO-E hours but no
active LSEG EPEX curve and remains ENTSO-E-only.

The policy was frozen before business values were compared: at least 744
matched hours, overlap ratio 1.0, and maximum p95 absolute difference, absolute
bias and single-hour difference each `0.005 EUR/MWh`. No threshold changed
after the result.

This is latest-to-latest source consistency only. ENTSO-E remains
`realized_latest_candidate`, `is_final=false`, and every causal, model,
monthly-level, publication, production and trading authority remains false.

## Cost preflight and execution

The PBI Warehouse was already `RUNNING` at `2X-Small`, with a 45-minute
auto-stop. This client issued no start, resize or create request.

The preflight compared three catalog candidates:

- `prd.silver.ge_market_lseg_curve_value_vintages`: 195,575,749 rows and
  11,544,235,073 bytes, unpartitioned; rejected without a SQL query;
- `prd.silver.ge_market_lseg_curve_values`: 757,167 rows and 16,758,198 bytes;
  selected because it retains interval/provenance/DQ fields under the 32 MiB
  ceiling;
- `prd.gold.factlsegcurvevalueslatest`: 757,167 rows and 11,250,012 bytes;
  not selected because the direct Silver contract carries the required DQ and
  source metadata.

The first bounded statement
`01f1a842-46c7-15ce-a770-87b5252a9b24` failed at compilation because the latest
table has no physical `_curve_value_vintage_id`. It read zero bytes and
returned zero rows. There was no blind retry. The deployed producer formula
was inspected and the identifier was reconstructed deterministically in the
query.

The corrected statement
`01f1a842-8df8-1f79-93d3-90abb9bc505f` succeeded:

- read bytes: 13,141,107;
- files read: 1 / 16,768,719 bytes;
- source rows read: 757,816;
- native rows returned: 9,672;
- total time: 2,212 ms;
- remote-write, spill and failed-task counts: zero.

Incremental session totals were 15 Databricks REST calls, 2 statements, 9,672
business rows downloaded, zero Warehouse starts/resizes/creates and zero
Databricks writes. Cumulative outage-plan totals are now 101 REST calls, 8
statements, 21,755 business rows and 4 prior remote governance writes. The
prohibited 9.36 GB disambiguation query was not repeated.

## Code and contract changes

- `docs/data/sql/databricks_prd_lseg_epex_actuals_latest_extract.sql`
  - adds the bounded read-only four-curve latest-observation extraction;
  - exact SHA-256:
    `be9e94de41c9e0c65f1814be617e3abd591103d6b869c6c80444761e6de22fff`.
- `pfc_shaping/validation/spot_source_reconciliation.py`
  - validates the hash-bound LSEG latest extract without claiming PIT;
  - revalidates the exact raw ENTSO-E candidate before comparison;
  - shares the existing row-validation and aggregate reconciliation cores;
  - admits the latest-extract Swiss market-month spanning two UTC months while
    retaining the existing PIT builder's single-UTC-month contract and the
    31-day bound.
- `tests/test_spot_source_reconciliation.py`
  - tests the new SQL binding, exact-window parameters, pass path and raw-frame
    tamper rejection.
- `.planning/phases/14-lt-audit-remediation/LSEG-ENTSOE-JULY-LATEST-RECONCILIATION-V1-20260904.json`
  - freezes preflight, query, artifact, cost, policy, metrics and authorities;
  - canonical JSON SHA-256:
    `557f0965b9cfe60884d69c0709d3863030ee6aeba1adaab466b46d31a200b32b`;
  - file SHA-256:
    `6ae3cff82d0ff1f4a2ded5b85517c3e2c4721ca34eba02f7db3b402d47d44327`.
- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`
  - binds the completed reconciliation, updates measured execution totals and
    advances the next actions without changing global blocked status.
- `tests/test_lt_source_acquisition_outage_plan.py`
  - verifies the evidence hash, exact metrics, preflight rejection and negative
    authorities.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md` and
  `docs/data/ENTSOE-DAY-AHEAD-EXPORT-V2.md`
  - document the result and the latest/finality boundary.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D-20260904-288.
- `.planning/HANDOFF.md`
  - points to this outcome and handoff.

All preceding uncommitted 4 September ENTSO-E candidate files and evidence
remain in the same working tree and must be preserved.

## Local artifacts

Directory: `build/lseg-july-reconciliation-20260904/`

- `preflight.json`: SHA-256
  `16472e4342d50a92eca5fdbf26810546a20d066a5f4ce59393aba008bba05769`;
- `lseg-epex-latest.parquet`: 767,668 bytes, 9,672 rows, SHA-256
  `f99585749ad8e0bc113109a9f7a645e909dc4c12a7baed6a6ab18cb7c6b17e71`;
- LSEG semantic SHA-256:
  `a600519b206bda9cdd0f94688bd3bdb51909dad1ea169e9710df71d3847fd081`;
- `capture-manifest.json`: SHA-256
  `f193ca38914e682f2377173c9b97caaf5dd44911c6f68f2ce3f3098cca82eb74`;
- `reconciliation-report.json`: SHA-256
  `a941c9ca7f0076a844f972c07cc3149dde0c88d4857bdf1c7011400871e9cce0`.

The helpers and business-value artifacts remain ignored below `build/`. No raw
price value was copied into Git, documentation or evidence JSON.

## Verification

Workspace guards verified both current directory and Git top-level before each
command.

- the new reconciliation test initially failed at collection because the new
  callable did not yet exist; after implementation, one expectation exposed
  canonical timestamp precision (`.008000Z`) and was corrected;
- focused reconciliation module: `27 passed, 1 warning`;
- focused combined matrix:
  `tests/test_spot_source_reconciliation.py`,
  `tests/test_entsoe_day_ahead_export.py`,
  `tests/test_lt_source_acquisition_outage_plan.py` ->
  `60 passed, 1 warning`;
- adjacent ENTSO-E/LSEG matrix:
  `tests/test_entsoe_day_ahead_export.py`,
  `tests/test_entsoe_day_ahead_consumption.py`,
  `tests/test_entsoe_day_ahead_prd.py`,
  `tests/test_spot_source_reconciliation.py`,
  `tests/test_lt_source_acquisition_outage_plan.py` ->
  `129 passed, 1 warning`;
- targeted Ruff format and check: pass;
- `git diff --check`: pass.

The warning is the pre-existing unknown pytest `cache_dir` option.

## Next safe steps

1. Track the separate September CH backfill on
   `FMVSA/opendata-lakehouse#4`; do not conflate it with July.
2. Promote the same July bytes to `realized_final` only if external evidence
   binds the exact semantic hash, window and SeriesKeys. This is optional for
   local replay/reconciliation and not currently available.
3. Freeze a prospective causal origin protocol and independent future holdout
   before retraining or model admission.
4. Complete the remaining independently governed EEX admission evidence.

## Invariants

- Do not rerun the July ENTSO-E export or the 9.36 GB disambiguation query.
- Do not infer finality or causal availability from latest-rank equality.
- Do not treat IT-North as independently reconciled.
- Do not use silent LSEG substitution if a discrepancy appears.
- Keep every model and production authority false.
- The CH monthly BASE solver remains the sole monthly-level authority.
- LT remains independent from CT and T057 remains sealed.
