# Session handoff — Databricks EEX offline normalization — 2026-08-05

## Outcome

The existing one-statement Swiss EEX snapshot from `prd.gold` was normalized
entirely offline into the CAL/Q/M BASE/PEAK identities consumed by the CH LT
monthly-solver path. No Databricks request was issued in this batch.

Selected status:

`PASS_LOCAL_NORMALIZATION_ONLY_NOT_PIT_OR_PROMOTION_AUTHORITY`

This proves local product identity, deterministic replay and solver
compatibility. It does not prove independently trusted historical availability
and grants no rolling-origin, candidate, promotion or production authority.

## Changed files

- `pfc_shaping/data/databricks_eex_daily_snapshot.py`
  - new offline-only normalizer; no connector or network path;
  - SHA-256
    `46c4b0709a8784d9a4298ecc9878e177424ab915e380aa75e3b0694afdb34357`;
  - strict 12-column source schema, source-key uniqueness, enum, date, currency
    and finite-settlement validation;
  - explicit BASE calendar bounds and PEAK first/last Monday-Friday bounds;
  - MONTH/QUARTER/YEAR identity reconstruction;
  - one-date quote-surface selection without fill-forward;
  - explicit separate BASE and PEAK solver maps;
  - all PIT/rolling-origin/promotion authority flags remain false.
- `tests/test_databricks_eex_daily_snapshot.py`
  - SHA-256
    `ea664c63b1f27589f1807dc611bcfe77c76caa0440e1aa8603d4e109ad381edc`;
  - five synthetic tests for BASE/PEAK bounds, exclusions, invalid enums,
    duplicate keys, no fill-forward, explicit solver maps and preservation of
    zero/negative settlement prices.
- `docs/research/forwards_sources.md`
  - SHA-256
    `ef904403258c56b4f5a62b0bbd7d8a0c2ea5babaf284ea67125a20681aabc194`;
  - documents the `prd.gold` source, local normalization, cost policy and
    non-authority boundary without disclosing price values.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D-20260805-208.
- this handoff.

No CT file, protected heavy desk data file, AFRY numeric artifact, Power BI
file, monthly solver production flag or existing source snapshot was modified.

## Selected local evidence

Source snapshot root:

`build/databricks-eex-daily/2026-08-05/`

Source bindings:

- exact captured NDJSON SHA-256
  `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`;
- local typed source Parquet SHA-256
  `cf0535420c16b97d28caa8002cd3dddc59d000d39011940aa9e02ec602e2c54d`;
- source rows: 82,552;
- source quotation coverage: 2019-01-02 through 2026-08-04;
- price field: settlement, EUR/MWh;
- no missing settlement values and no duplicate source composite keys.

Selected content-addressed normalization:

`build/databricks-eex-daily/2026-08-05/normalizations/25d40dcc74f17bae0486c8f27581fb4903bcfa8132098d2e6a1b79649c5780f6/`

- `manifest.json`
  - SHA-256
    `223d99a679908969bd308e49c3769f7afa1ba9cdfb1980415d91076efef7576b`;
- `eex_ch_cal_q_m_history.parquet`
  - 36,753 rows;
  - 222 distinct normalized product/load identities across history;
  - SHA-256
    `40f8b0e2add669a051840df6c21f80c7febc7e0c1289128d58275beffced5501`;
- `eex_ch_cal_q_m_latest.parquet`
  - observed quotation date 2026-08-04;
  - 20 BASE and 20 PEAK products;
  - SHA-256
    `d223c3055d07f12d6bfd4f479f53b948fe12892c946c4b5a9c5c8284d2796ce7`.

The earlier root-level derived files
`eex_ch_cal_q_m_history.parquet`, `eex_ch_cal_q_m_latest.parquet` and
`normalization-validation.json` are superseded attempts. Their first replay
correctly exposed a non-canonical Pandas string dtype. Content bundle
`8e796e72...88d3e` is also superseded by the final API-roasted bundle after
timezone-naive as-of enforcement and a no-mutation solver-map contract were
added. Neither superseded result is selected; the raw captured snapshot and
its original receipts remain unchanged.

## Scientific and semantic checks

- A naive all-calendar delivery-bound rule would have falsely rejected 3,023
  eligible PEAK rows.
- BASE uses the full calendar period.
- PEAK uses the first through last Monday-Friday delivery day, consistent with
  the existing project EEX Peakload definition of Monday-Friday, 08:00-20:00
  local time.
- With explicit load semantics, all 36,753 MONTH/QUARTER/YEAR rows validate.
- DAY/WEEK/WEEKEND are valid source data but excluded from the monthly-solver
  normalization: 45,799 rows.
- Zero and negative settlements are retained. `LastPrice` is never used as a
  fallback.
- The latest surface is one actually observed date; stale products are not
  filled forward from earlier dates.
- All 222 normalized identities round-trip through the existing canonical EEX
  desk-code parser and the monthly product-period parser.
- A real BASE solve over 77 delivery months completed with maximum hard-
  constraint residual `8.526512829121202e-14` EUR/MWh versus tolerance `1e-9`.

No price values were written to Git, documentation or console output.

## Commands and results

Every shell action first verified exact cwd and Git top-level
`C:\Users\jbattaglia\PFC_LT`; `TEMP` and `TMP` were set below `build/`.

1. Synthetic normalization tests:

   `python -I -B -m pytest tests/test_databricks_eex_daily_snapshot.py --basetemp build/pytest-databricks-eex-normalization-v3 -q`

   Result: `5 passed in 10.54s`.

2. Content-addressed materialization and exact local replay:

   Result: content ID
   `25d40dcc74f17bae0486c8f27581fb4903bcfa8132098d2e6a1b79649c5780f6`,
   exact Parquet frame equality, parser round-trip PASS and real monthly solver
   compatibility PASS.

3. Adjacent regression matrix:

   - new normalization tests;
   - monthly forward constraints;
   - monthly solver;
   - monthly forward integration;
   - governed forward history;
   - LT/CT import boundaries;
   - LT package contract.

   Result: `132 passed, 4 skipped in 73.28s`.

4. `git diff --check` on the new module, tests and documentation: PASS. Git
   reported only the existing Windows LF-to-CRLF warning for the documentation.

5. Ruff was attempted through the repo-local runtime and was unavailable:
   `No module named ruff`. No package was installed or fetched. Pytest imports,
   targeted behavior and adjacent contracts all passed.

## Failures found and repaired

1. The first calendar-bound hypothesis rejected 3,023 rows. Root cause: PEAK
   periods use first/last weekday delivery bounds. The contract was corrected
   and validated over all real CAL/Q/M rows.
2. The first derived Parquet replay found `object` versus canonical Pandas
   `string` dtype drift. The normalizer now constructs explicit string and UTC
   timestamp extension dtypes. A new content-addressed bundle was built and
   exact replay passed; no existing artifact was overwritten.

## Durable decision and invariants

D-20260805-208 selects the content-addressed bundle only for local research and
pipeline development.

Do not break:

- no repeated Databricks query for modelling; reuse the local snapshot;
- Databricks remains read-only and refreshes require deliberate user approval;
- monthly CH EEX levels remain solver-authoritative;
- local normalization is not signed PIT/vintage evidence;
- governed ENTSO-E Databricks evidence, signed EEX vintage availability and a
  new independent future holdout remain open;
- local or synthetic EEX/ENTSO-E substitution is forbidden for empirical
  validation;
- AFRY and OMPEX remain benchmark-only, T057 remains sealed, and production is
  strict `NO_GO`.

## Next safe batch

Stay offline. Build a non-promotional EEX surface audit that quantifies quote
coverage, nesting conflicts and BASE/PEAK implied-OFFPEAK feasibility by
quotation date, using the normalized history but making no rolling-origin or
predictive claim. This can improve solver diagnostics and data-engineer
requirements while ENTSO-E and trusted PIT availability remain open.
