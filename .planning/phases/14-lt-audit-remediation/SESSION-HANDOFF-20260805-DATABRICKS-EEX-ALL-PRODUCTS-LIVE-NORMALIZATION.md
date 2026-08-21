# Session handoff — Databricks EEX all-products live normalization — 2026-08-05

## Outcome

The frozen one-statement `prd.gold` EEX capture was reworked entirely offline
after confirming that DAY, WEEK and WEEKEND are required evidence, not noise.
The selected normalization now exposes two explicit layers:

- all live products for research/shaping diagnostics;
- live CAL/Q/M only for the CH monthly-solver interface.

Selected normalization status:

`PASS_LOCAL_ALL_PRODUCTS_NORMALIZATION_WITH_QUARANTINE_NOT_PIT_OR_PROMOTION_AUTHORITY`

Selected surface-audit status:

`PASS_LOCAL_INTEGRITY_NO_MARKET_QUOTE_CONFLICTS`

No Databricks statement was issued in this batch. No Databricks object was
written or modified. The single local snapshot from 2026-08-05 remains the
only Warehouse extraction used for this work.

## Changed files

- `pfc_shaping/data/databricks_eex_daily_snapshot.py`
  - SHA-256
    `6d9194423d505bd8dcf4a31e397507b3f56ee694cba7e4449e76389d629ad189`;
  - retains DAY/WEEK/WEEKEND/MONTH/QUARTER/YEAR in the all-product layer;
  - exposes a separate CAL/Q/M solver layer;
  - requires quotation date strictly before delivery start;
  - quarantines invalid temporal or boundary rows with reason codes;
  - records the public EEX delivery-contract binding;
  - preserves settlement prices in EUR/MWh, including zero and negative
    values, and never falls back to `LastPrice`.
- `tests/test_databricks_eex_daily_snapshot.py`
  - SHA-256
    `77312ddea0774cdd3fc6aaad409be4564175c5fb7f9c6f34e1ecf812da1793a0`;
  - seven tests covering all-product retention, separate solver maps,
    boundary semantics, quarantine and source integrity.
- `pfc_shaping/validation/databricks_eex_surface_audit.py`
  - SHA-256
    `2c7a52149d41f09f6fd611c6352ea9ea77d402ebd3438786e3818f8ee65ac94b`;
  - offline CAL/Q/M nesting, coverage and BASE/PEAK recomposition audit.
- `tests/test_databricks_eex_surface_audit.py`
  - SHA-256
    `4f61b23cc634409063e74e05a9ebb2ffa16c0286399f2a48803803325b86d6bd`.
- `docs/research/forwards_sources.md`
  - updated to the live all-product contract and selected content IDs;
  - SHA-256
    `3ef908ff5f61e86a656e6925e2cc22d6768e96747f3281cd6c90f39a1fc80d19`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D-20260805-210;
  - SHA-256 before adding this handoff
    `f15a59e93308229ad745cd0324cc5f38237b06e17c4d50fd8c4b58f3e1844f0d`.
- build-only utilities:
  - `build/databricks-eex-daily/profile_delivery_conventions.py`;
  - `build/databricks-eex-daily/materialize_all_products_normalization.py`,
    SHA-256
    `f0c07b1d73ebb12a7dd0de2b20c054689fe28dc49776b41f661c766d10d3640c`;
  - `build/databricks-eex-daily/materialize_surface_audit.py`, SHA-256
    `a1bda3db926e8157b7e1b3e25656827807b68176709a1be9835d9b648ed2a3a3`;
  - public-document inspection assets below `build/eex-public-docs/`.
- this handoff.

No CT, Power BI, protected heavy desk-data, AFRY numeric artifact, production
flag or source-capture byte was modified.

## Source and public-contract bindings

Frozen source:

`build/databricks-eex-daily/2026-08-05/`

- rows: 82,552;
- quotation dates: 2019-01-02 through 2026-08-04;
- exact captured NDJSON SHA-256
  `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`;
- typed source Parquet SHA-256
  `cf0535420c16b97d28caa8002cd3dddc59d000d39011940aa9e02ec602e2c54d`;
- no null settlement and no duplicate source composite key.

Public EEX Contract Details:

- as-of 2026-07-22;
- URL
  `https://www.eex.com/fileadmin/EEX/Downloads/Trading/Specifications/Contract_Details/eex-contract-details-xls-data_20260722.xlsx`;
- local build-only SHA-256
  `e03e51125b5e0b76668bc66bd736351799323abcb57b85e940ea574cd9ff1232`;
- confirmed semantics: DAY one day; WEEK BASE Monday-Sunday; WEEK PEAK
  Monday-Friday; WEEKEND Saturday-Sunday; published last trading dates precede
  delivery starts.

## Selected normalization evidence

Path:

`build/databricks-eex-daily/2026-08-05/normalizations-all-products/bb258371a96f19a8e08b54f9126635c3a478314e3284b7bf1249dba3feaaaeb5/`

- content ID
  `bb258371a96f19a8e08b54f9126635c3a478314e3284b7bf1249dba3feaaaeb5`;
- manifest SHA-256
  `7806daf9f4b0572facc0b750c6d0b790d4ece611c8dcff92c9693ff3a233f4dd`;
- all-product live history: 72,175 rows, SHA-256
  `896f0c9f839b7fc9364398ed0848b0f1886c2ee54d277bf327ae1a414833c06e`;
- non-solver DAY/WEEK/WEEKEND live history: 38,070 rows;
- solver CAL/Q/M live history: 34,105 rows, SHA-256
  `fc5de85d1870937955dcc93cbf1cea0e1d2f85d892b286f490a6de11650d7a25`;
- latest all-product surface on 2026-08-04: 74 rows, SHA-256
  `b842551aebbf787a29f49ab29a8957bd139fdfdd6502ce9e50eb85328ad4fd97`;
- latest solver surface on 2026-08-04: 38 rows, SHA-256
  `3b8c9a9831a3ff44b8b9880e914fa1bb1e60c4e03d0610d1fd40ab8b83490aaf`;
- quarantine: 10,377 rows, SHA-256
  `d1328caac9fb860fc2bb0f4bf518ec59e792338f1d154d01c0067b937f2a21e5`.

Quarantine breakdown:

- 10,376 rows have quotation date on or after delivery start;
- one WEEK PEAK row has a Monday-Sunday boundary instead of Monday-Friday.

The two earlier content-addressed attempts
`1fea7719...280dbf8` and `2837dc48...36a3f` remain below `build/` as
superseded forensic evidence. They are not selected and were not deleted.

## Selected live-CAL/Q/M surface audit

Path:

`build/databricks-eex-daily/2026-08-05/surface-audits-live-cqm/8d301f964074b1030df3f30b00195c3cfe919a5f33647ca17ee1e8a1ffcbab3f/`

- content ID
  `8d301f964074b1030df3f30b00195c3cfe919a5f33647ca17ee1e8a1ffcbab3f`;
- manifest SHA-256
  `b0b9322698f9ffd31dea8e29a0ef5381a7b2403fa43d54e5f7abd86f95eda768`;
- 34,105 rows across 1,939 quotation dates;
- PEAK starts on 2023-06-26;
- 3,255 fully nested parent/child comparisons, zero conflicts above
  0.01 EUR/MWh;
- 10,766 BASE/PEAK identities, zero OFFPEAK recomposition failures;
- latest surface: 38 rows, 19 complete BASE/PEAK pairs, zero nesting conflict.

The D209 audit reported 437 conflicts from 36,753 CAL/Q/M rows. Those conflicts
disappear when the 2,648 non-live CAL/Q/M observations are quarantined. D210
therefore supersedes the D208/D209 derived bundles while preserving their
forensic history and the immutable raw source.

## Commands and results

Every shell action verified exact cwd and Git top-level
`C:\Users\jbattaglia\PFC_LT`. `TEMP`, `TMP` and pytest basetemps remained
below `build/`.

1. Focused normalizer tests after the replay-index repair:

   `python -I -B -m pytest tests/test_databricks_eex_daily_snapshot.py ...`

   Result: `7 passed in 0.29s`.

2. Combined normalizer and surface-audit tests:

   `python -I -B -m pytest tests/test_databricks_eex_daily_snapshot.py tests/test_databricks_eex_surface_audit.py ...`

   Result: `13 passed in 0.51s`.

3. All-product normalization was materialized twice from the frozen Parquet.
   Both successful runs returned content ID `bb258371...aaaeb5`; exact Parquet
   replay passed.

4. Live-CAL/Q/M surface audit was materialized twice. Both runs returned
   content ID `8d301f96...cbab3f`.

5. Adjacent regression matrix included both new test modules, governed forward
   history, LT/CT imports, LT package contract, and monthly curve audit,
   constraints, integration, priors and solver.

   Result: `177 passed, 4 skipped in 139.34s`.

6. `git diff --check` on the tracked documentation and decision log: PASS,
   with only the existing LF-to-CRLF checkout warning. The relevant Python
   files were imported and exercised by pytest.

## Failure found and repaired

The first exact replay of the new CAL/Q/M subset failed because Pandas retained
the source row index while Parquet correctly reloaded a range index. Row count,
values and schema were otherwise identical. The subset now resets its index at
the layer boundary. The focused test suite passed and two subsequent complete
materializations returned the same content ID.

This was an artifact-replay defect, not a price or product-identity correction.
The failed build-only slots were preserved and are not selected.

## Authority and remaining risks

- This snapshot is strong local engineering and descriptive evidence, not an
  independently signed point-in-time vintage catalog.
- No rolling-origin selection, candidate assembly or production promotion may
  use it as authority yet.
- Governed ENTSO-E Databricks inputs and a new independently frozen future
  holdout remain mandatory. Local ENTSO-E may be used only for schema/tooling,
  never as empirical substitution.
- DAY/WEEK/WEEKEND can support future short-horizon shaping diagnostics, but
  they cannot rewrite solver-authoritative monthly means.
- AFRY and OMPEX remain benchmark-only. T057 remains sealed. Production is
  strict `NO_GO` and LT/CT separation is unchanged.

## Next safe batch

Remain offline and build a grain-aware coverage/availability audit for the
38,070 live DAY/WEEK/WEEKEND rows. The output should quantify usable horizons,
BASE/PEAK pairing and temporal continuity without fitting, selecting or
promoting a model. In parallel, prepare the exact governed ENTSO-E data
contract for the data engineer; do not substitute local ENTSO-E values.
