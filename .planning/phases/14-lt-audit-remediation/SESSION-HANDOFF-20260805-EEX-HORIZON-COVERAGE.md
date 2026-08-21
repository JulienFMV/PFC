# Session handoff - EEX local quote-horizon coverage

Date: 2026-08-05

## Outcome

The frozen local D212 EEX CAL/Q/M history now has a deterministic offline
horizon-coverage audit. It describes 34,105 live quotes across 1,939 quotation
dates and 220 product lifecycles without querying Databricks or starting a SQL
Warehouse.

Direct monthly evidence reaches at most 185 days for BASE and 187 days for
PEAK. There are 81 monthly quotes beyond 180 days and none beyond 365 days.
Far-horizon monthly shape is therefore not directly observed in this capture;
it must remain prior-driven and cannot rewrite the monthly solver means.

The selected status is
`PASS_LOCAL_DESCRIPTIVE_HORIZON_COVERAGE_WITH_GAP_ANOMALIES_NOT_PIT`.
`FactLoadTimestamp` is ingestion lineage only and does not authenticate
historical provider availability. Rolling-origin, model selection, promotion
and production authority remain false.

## Changed files

- `pfc_shaping/validation/databricks_eex_horizon_audit.py`
  - validates the exact normalized CAL/Q/M schema and lineage;
  - reconstructs delivery starts and strict positive lead times;
  - produces a zero-filled daily horizon grid, product lifecycles and a compact
    horizon summary without fill-forward or price-derived coverage;
  - records delayed ingestion and long-gap proxies without authority overclaim.
- `tests/test_databricks_eex_horizon_audit.py`
  - covers bucket boundaries, PEAK starts, live eligibility, zero coverage,
    prices, gaps, ingestion timing, duplicates and authority failures.
- `docs/research/forwards_sources.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` (D213)
- `.planning/HANDOFF.md`

## Exact sources and bindings

- D212 normalization content ID:
  `2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f`
- Normalization manifest SHA-256:
  `f91805f3004e746ac588e19aa6745ae0dc0f490a9ef5c49aae0918e7ec3f8f53`
- Local solver-history Parquet SHA-256:
  `fc5de85d1870937955dcc93cbf1cea0e1d2f85d892b286f490a6de11650d7a25`
- D212 surface-audit content ID:
  `bb1a09932b4bbff31dfdbb4ada561befb02050413ee819b03bf6c28f4858ab54`
- Surface-audit manifest SHA-256:
  `7f5d565ad7191b5f455e7a29280de078422cd95b3a846d06dc8cc6f29bf131a0`
- Horizon auditor SHA-256:
  `3f014dcd97d3dd336c0924bae3ef6dffe09f81b12e8edf0ee0f75ed3570c4a42`
- Horizon test SHA-256:
  `daf3626467bb937a95166e5719af56ff0d874157bd0d55951001db2f260c099f`

## Selected artifact

Root:

`build/databricks-eex-daily/2026-08-05/horizon-audits-live-cqm/435ecbc737f95268f03f7f347dfafc4163f5b5a4cb5b8dc9cec87e05f1645108/`

- Content ID:
  `435ecbc737f95268f03f7f347dfafc4163f5b5a4cb5b8dc9cec87e05f1645108`
- Manifest SHA-256:
  `949a1598710ad6198b569fd4cb00a99750707e162dbc0c709fefaa04f28e0ed1`
- `daily_horizon_coverage.parquet`: 49,194 rows, SHA-256
  `c5c0ffdc3616f8b8f4c2e2db853896c1f6d16f6e00472db476f096bb10f0f12a`
- `product_lifecycles.parquet`: 220 rows, SHA-256
  `8fabc4fb28c9393bed143c3c484f390c7a6639df64c336320fb23af1f159088f`
- `horizon_summary.parquet`: 36 rows, SHA-256
  `d4d401efa0095146317e6b87cf241316a78b140e60fe6817e0205c6c0d26e945`
- `summary.json`: SHA-256
  `57a589ad6d6625ee70292194222a235212508dc078503c89d75973aa5569cf0c`

The earlier internal draft
`d59a37a67bbf4cff9970f346dc1904daec75389e325da55fc8f91a6a1839c4f1`
is superseded and unselected. Its three analytical frames are semantically
identical, but it is bound to obsolete D210/D211 source identities.

## Data-quality findings

- Monthly max lead: BASE 185 days; PEAK 187 days.
- Monthly quotes over 180 days: 81; over 365 days: 0.
- One lifecycle exceeds the 20-weekday proxy threshold: BASE `2023-Q2`, with
  one quote before a 172-weekday proxy gap (2021-07-29 to 2022-03-29) and 259
  quotes after it. This requires upstream explanation but does not prove
  market inactivity; the proxy omits the EEX holiday calendar.
- There are 61 distinct fact-load timestamps. Quote-to-load lag has p50 657
  days, p90 2,116 days and max 2,688 days; 91.15% of rows exceed 30 days and
  65.07% exceed 365 days. This is a delayed-load/backfill-or-restatement
  signature, not provider-availability evidence.

## Verification commands and results

All commands ran from `C:\Users\jbattaglia\PFC_LT` with `TEMP`, `TMP` and
pytest basetemp below `build/`.

- Focused:
  `build\pytest-runtime-v2-final\python.exe -m pytest -q tests/test_databricks_eex_horizon_audit.py`
  -> `10 passed in 0.61s`.
- Adjacent LT matrix:
  `build\pytest-runtime-v2-final\python.exe -m pytest -q tests/test_databricks_eex_horizon_audit.py tests/test_databricks_eex_daily_snapshot.py tests/test_databricks_eex_surface_audit.py tests/test_monthly_forward_curve_priors.py tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_forward_curve_constraints.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_forward_curve_integration.py tests/test_governed_forward_history.py tests/test_audit_ch_product_normalization_script.py tests/test_lt_ct_imports.py tests/test_lt_package_contract.py`
  -> `248 passed, 4 skipped, 2 warnings in 28.53s`.
- The two warnings are the existing insufficient-history `All-NaN slice` and
  `Mean of empty slice` warnings in the lambda fail-closed test.
- Fresh-process replay regenerated all three Parquets and `summary.json` with
  exact byte-identical SHA-256 values.
- No Databricks request, SQL Warehouse start, Databricks write or network call
  occurred.

## Authority and remaining blockers

The CH monthly solver remains the sole level authority. This EEX bundle is
local descriptive evidence only, not an authenticated PIT vintage history.
Governed ENTSO-E evidence, signed EEX vintages and a new independently frozen
future holdout remain required before rolling-origin selection or promotion.
AFRY and OMPEX remain benchmark-only, T057 remains sealed, LT/CT separation is
unchanged, and production remains strict `NO_GO`.
