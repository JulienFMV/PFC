# Session handoff - EEX Swiss short-tenor contract semantics

Date: 2026-08-05

## Outcome

The exact public EEX Contract Details workbook dated 2026-07-22 resolves the
PEAK-calendar ambiguity for Swiss short-tenor futures. DAY PEAK exists on every
delivery day, including Saturday and Sunday, for 12 hours from 08:00 to
20:00. WEEK PEAK covers Monday-Friday for 60 hours total. WEEKEND PEAK covers
12 hours on Saturday plus 12 hours on Sunday, hence 24 hours total. Monthly
PEAK remains a distinct weekday-only calendar.

All work was performed locally from the frozen EEX snapshot and the public EEX
workbook. No Databricks request, Warehouse start or Databricks write occurred.

## Changed files

- `pfc_shaping/data/databricks_eex_daily_snapshot.py`
  - records the exact official workbook URL, date and SHA-256;
  - records short-tenor delivery-day and contract-size semantics;
  - retains DAY/WEEK/WEEKEND outside the CAL/Q/M monthly-solver view.
- `tests/test_databricks_eex_daily_snapshot.py`
  - covers weekend DAY PEAK and the published semantics receipt.
- `build/databricks-eex-daily/materialize_all_products_normalization.py`
  - verifies every relevant official contract row before materialization;
  - binds the public workbook and verification receipt into the manifest.
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D212)
- `.planning/HANDOFF.md`

## Exact source and evidence

- Official EEX workbook:
  `build/eex-public-contract-details/2026-07-22/eex-contract-details-xls-data_20260722.xlsx`
- Workbook SHA-256:
  `e03e51125b5e0b76668bc66bd736351799323abcb57b85e940ea574cd9ff1232`
- Rows checked: 14,082; contract-size mismatches: 0.
- Weekend DAY PEAK rows: 1,564.
- Selected normalization content ID:
  `2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`
- Normalization manifest SHA-256:
  `f91805f3004e746ac588e19aa6745ae0dc0f490a9ef5c49aae0918e7ec3f8f53`
- Selected live-CAL/Q/M v2 audit content ID:
  `bb1a09932b4bbff31dfdbb4ada561befb02050413ee819b03bf6c28f4858ab54`
- Audit manifest SHA-256:
  `7f5d565ad7191b5f455e7a29280de078422cd95b3a846d06dc8cc6f29bf131a0`

The normalization contains 72,175 live all-product rows, 34,105 live CAL/Q/M
rows and 10,377 quarantined rows. The latest all-product surface has 74 rows;
the solver view has 38. The v2 audit has 3,255 complete non-overlapping nested
comparisons, zero conflicts above 0.01 EUR/MWh and 10,766 BASE/PEAK identities
with zero recomposition failures.

## Verification

- Focused normalization tests: `7 passed`.
- Focused and adjacent LT roast: `138 passed, 4 skipped`.
- `git diff --check`: pass before documentation update.
- Two independent materializations returned the same normalization and audit
  content IDs.

## Authority and remaining blockers

This is local descriptive and pipeline-development evidence only. It does not
prove signed point-in-time availability and cannot authorize rolling-origin
selection, model selection, promotion or production. DAY/WEEK/WEEKEND cannot
rewrite monthly solver means. Governed ENTSO-E evidence, signed EEX vintages
and a new independently frozen future holdout remain required. AFRY and OMPEX
remain benchmark-only, T057 remains sealed, and production remains `NO_GO`.
