# Session handoff — Databricks EEX daily local snapshot — 2026-08-05

## Outcome

One read-only Databricks SQL statement captured the complete visible Swiss
power EEX daily-price history from `prd.gold` to the canonical workstation.
No Databricks table, view, file or configuration was created or mutated.

The capture is suitable for local research and pipeline development. It is
not yet production or rolling-origin authority because independently trusted
point-in-time availability and the signed EEX vintage-catalog contract remain
open. ENTSO-E Databricks evidence also remains open; legacy local ENTSO-E may
be used only for schema/tooling preparation, not empirical substitution.

## Cost and mutation controls

- User policy: at most one Databricks repatriation attempt per local day.
- Today: exactly one extraction statement, `SELECT` only.
- Execution duration observed locally: 49.1 seconds.
- Result chunks: 2.
- Daily `_SUCCESS.json` guard was replay-tested: a second `-Execute` returned
  the existing artifact without sending another statement.
- The SQL allow-list rejects mutation verbs including `CREATE`, `INSERT`,
  `UPDATE`, `DELETE`, `MERGE`, `COPY`, `ALTER`, `DROP`, `TRUNCATE`, `OPTIMIZE`,
  `VACUUM`, `GRANT` and `REVOKE`.
- After capture, all validation and Parquet materialization ran locally.

## Source and scope

- `prd.gold.facteexpricedaily`
- `prd.gold.dimeexproduct`
- `prd.gold.dimeexdeliveryperiod`
- Filter: `Country = 'CH'` and `Commodity = 'POWER'`.
- Currency: EUR/MWh.
- Selected price authority for current research: `SettlementPrice`; `LastPrice`
  is retained but sparse.

## Local artifacts

Root:

`build/databricks-eex-daily/2026-08-05/`

- `_SUCCESS.json`: immutable successful-capture receipt.
- `manifest.json`: same capture manifest for direct inspection.
- `eex_ch_power.ndjson`: exact row-oriented Statement Execution result.
  - rows: 82,552
  - bytes: 29,763,661
  - SHA-256: `593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812`
- `eex_ch_power.parquet`: locally typed, sorted, Zstandard-compressed copy.
  - bytes: 542,030
  - SHA-256: `cf0535420c16b97d28caa8002cd3dddc59d000d39011940aa9e02ec602e2c54d`
- `local-validation.json`: local quality and authority receipt.
- `attempt-started.json`: daily attempt ledger.

All artifacts are below ignored `build/`; none contains the Databricks token.

Ephemeral helpers:

- `build/databricks-eex-daily/capture.ps1`
- `build/databricks-eex-daily/finalize_snapshot.py`

These helpers are local build evidence, not production code.

## Data-quality evidence

- rows: 82,552
- columns: 12
- distinct products: 5,286
- distinct delivery periods: 3,930
- quotation-date coverage: 2019-01-02 through 2026-08-04
- latest fact load: 2026-08-05T02:42:47.076Z
- delivery coverage: 2018-12-31 through 2032-12-31
- latest quotation surface: 80 rows, delivery through 2032-12-31
- product types: BASE, PEAK
- delivery-period types: DAY, WEEK, WEEKEND, MONTH, QUARTER, YEAR
- duplicate composite keys `(ProductID, DeliveryPeriodID, QuotationDateID)`: 0
- exact duplicate rows: 0
- missing `SettlementPrice`: 0
- missing `LastPrice`: 79,508; non-blocking because settlement is the selected
  price field.

## Commands and results

All shell commands first verified exact cwd and Git top-level
`C:\Users\jbattaglia\PFC_LT`.

1. Plan-only capture invocation: PASS; no Databricks call; one-statement plan.
2. `capture.ps1 -Execute`: PASS in 49.1 seconds; `_SUCCESS.json` produced.
3. Offline `finalize_snapshot.py`: PASS; local integrity and data quality.
4. Second `capture.ps1 -Execute`: daily success guard returned existing path;
   raw and success hashes unchanged; no second Databricks statement.
5. `git check-ignore -v`: capture, Parquet and receipt are ignored by the
   repository-wide `build/` rule.

## Durable decision

Decision: use one immutable daily EEX snapshot as the local research view and
reuse it for all LT work until a deliberately approved refresh is necessary.

Reason: LT modelling does not benefit from repeatedly querying an unchanged
forward surface; a local Parquet minimizes Databricks cost and improves
reproducibility and performance.

Rejected alternatives:

- live Databricks queries during model runs;
- repeated schema/profile probes before each calculation;
- more than one automatic capture attempt per day;
- writing derived tables or views to Databricks;
- treating local ENTSO-E as governed empirical substitution.

Invariants not to break:

- Databricks remains read-only.
- One extraction attempt maximum per local date.
- Monthly CH EEX levels remain solver-authoritative.
- This snapshot must not be represented as signed PIT/vintage or promotion
  evidence.
- T057 remains sealed and AFRY remains local diagnostic only.

## Next safe work

Work entirely offline from `eex_ch_power.parquet` to define the Databricks-to-
monthly-solver normalization, including product-code reconstruction from
delivery dates/types and explicit BASE/PEAK handling. Before any rolling-origin
selection or production admission, close the trusted availability/vintage
catalog and independent future-holdout requirements and obtain governed
ENTSO-E Databricks evidence.
