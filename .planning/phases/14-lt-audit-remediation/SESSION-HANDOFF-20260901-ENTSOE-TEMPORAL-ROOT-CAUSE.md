# Session handoff - ENTSO-E temporal root cause

Date: 2026-09-01

Branch: `fix/lt-audit-remediation`

## Outcome

The state-of-the-art value-blind root-cause diagnostic is complete. One
additional SQL statement ran on the already-active PBI Warehouse; it returned
seven aggregate rows, opened no prices and exactly reconciled the prior profile
capture.

The two root causes are now isolated:

1. Availability: all 17,925 rows have
   `publication_timestamp_utc > first_seen_pull_ts_utc`; publication, first
   seen and last seen are never null, and `first_seen > last_seen` never occurs.
   All publication timestamps are also 6.388 to 37.296 days after delivery
   start. The typical publication-after-first-seen lag is 29 seconds for CH,
   61 for AT, 118 for DE-LU, 150 for FR and 193 for IT-North. This strongly
   indicates a rebuild/transformation timestamp stored as publication time.
2. Intervals: all 404 failures are duration mismatches. There are no null
   bounds, `IntervalEndUtc`/`Date_Time_UTC` mismatches, non-positive intervals
   or unsupported resolutions. Most rows have the correct cadence; malformed
   durations are longer cadence multiples. This is strongly consistent with
   `IntervalEndUtc` being bridged to the next observed point across missing
   positions, but producer confirmation is still required.

The ENTSO-E incident notice dated 31 August does not explain either root cause:
the July publication lags infer a common event around 7 August, and publication
delay cannot change the structural interval duration.

## Evidence by series

| Series | Rows | Duration failures | Rate | Median / maximum duration | Typical publication after first seen |
|---|---:|---:|---:|---:|---:|
| `day_ahead_prices||at_price||1` | 2,935 | 24 | 0.818% | 15 / 75 min | 61 s |
| `day_ahead_prices||at_price||2` | 2,914 | 49 | 1.682% | 15 / 75 min | 61 s |
| `day_ahead_prices||ch_price` | 737 | 6 | 0.814% | 60 / 120 min | 29 s |
| `day_ahead_prices||de_lu_price||1` | 2,922 | 36 | 1.232% | 15 / 105 min | 118 s |
| `day_ahead_prices||de_lu_price||2` | 2,933 | 36 | 1.227% | 15 / 45 min | 118 s |
| `day_ahead_prices||fr_price` | 2,713 | 112 | 4.128% | 15 / 345 min | 150 s |
| `day_ahead_prices||it_nord_price` | 2,771 | 141 | 5.088% | 15 / 105 min | 193 s |

Total interval-failure rate: 2.254%.

## Cost and artifacts

Query history:

- final state `FINISHED`, seven rows, duration 1,619 ms;
- 2,781,806 bytes read from one file and 296,244 rows read;
- 1,515,748,324 bytes and 136 files pruned;
- zero remote-write bytes, zero Warehouse start, zero SQL retry.

Artifacts:

- `build/entsoe-day-ahead-prd-profile/20260901-july-temporal-diagnostic/diagnostic.json`
  - file SHA-256
    `9f6b798ec89e507ebd26524dbbc439abf347b9e36a9762855c00521f0841919d`;
  - content ID
    `a5f5f3b948212d65112cd2243633744cb1ef1872d827f7ff28312b2dfa80e3d4`;
  - offline replay `PASS_DIAGNOSTIC_CAPTURE_REPLAY`.
- `build/entsoe-day-ahead-prd-profile/20260901-july-temporal-diagnostic/query_history.json`
  - file SHA-256
    `9bb49a03d809fc255253f43646783ea0a23b8c417d5693b6fcc997a15de2fa34`.

## Changed files

- `docs/data/sql/databricks_prd_entsoe_day_ahead_temporal_diagnostic.sql`
  - per-series cause unions, lag distributions and duration distributions;
  - exact year/month partition fence and 101-row sentinel;
  - SHA-256
    `f20b0ac5681404834132897d48b491532ae226a7cec81a76abae7043721e5f04`.
- `pfc_shaping/validation/entsoe_day_ahead_temporal_diagnostic.py`
  - hash/read-only/value-blind SQL binding;
  - exact seven-series inventory, count/union/quantile validation;
  - exact reconciliation with the prior profile and authority-negative report;
  - SHA-256
    `2b7c860d62ff03d804f4d737dac8860edbdd0e518f631155e6b15050395288b1`.
- `scripts/capture_entsoe_day_ahead_temporal_diagnostic.py`
  - one-shot native-parameter capture, active-Warehouse guard and atomic receipt;
  - prior-profile binding and deterministic offline replay;
  - SHA-256
    `2e60b3c0b3e63382778e6273392acf4c9bf2d808ac6f30b4156f0b606846b4f8`.
- `scripts/capture_entsoe_day_ahead_prd_profile.py`
  - parameterized user-agent support reused by the diagnostic transport;
  - SHA-256
    `ce461587aa496b117d9b84576a5e2bb11a5530212fe01e587086a26ee86b3e65`.
- `tests/test_entsoe_day_ahead_temporal_diagnostic.py`
  - reconciliation, union, quantile, inventory, SQL and authority tests;
  - SHA-256
    `57990e50ea1544303dd04497cf58fc7c7b3d50e91a600eb055c55ddc45b45562`.
- `tests/test_capture_entsoe_day_ahead_temporal_diagnostic.py`
  - response schema, truncation, signed lag and negative-count tests;
  - SHA-256
    `9e4ad1270396c190bcafa93aec27e5ceb99d5e99eac8be8974d5043ed587e480`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - durable decision D-20260901-264.
- `.planning/HANDOFF.md`
  - current root-cause state and handoff pointer.

## Audit evidence

```text
dapdiagaud1
diagnostic + capture + prior profile tests
49 passed

dapdiagall1
all tests/test_*entsoe*.py -m "not slow"
784 passed

dapdiaglt1
tests/test_lt_ct_imports.py tests/test_lt_package_contract.py
43 passed, 1 skipped

dapdiagruff2 / dapdiagfmt3
Ruff check and format check
pass
```

## Single question for Jerome

The next external message should be limited to:

> Sur le profil PRD `day_ahead_prices` de juillet, les 17,925 lignes ont
> `publication_timestamp_utc` postérieur à `first_seen_pull_ts_utc` et au
> début de livraison de 6.4 à 37.3 jours, ce qui ressemble à un timestamp de
> rebuild/transformation plutôt qu'au timestamp de publication du document
> ENTSO-E. Peux-tu confirmer la source et la règle de calcul de ce champ ?
> Par ailleurs, les 404 seules anomalies d'intervalle sont des durées
> supérieures à la résolution déclarée, sans null, inversion ni divergence
> `IntervalEndUtc`/`Date_Time_UTC`; `IntervalEndUtc` est-il construit par
> `LEAD(Date_Time_UTC)` à travers des positions manquantes ? Si oui, peux-tu
> confirmer la correction prévue (`start + resolution`, gaps explicites) et
> le rebuild PRD associé ?

No further Databricks statement is justified before this answer. PIT/model
authority remains false, model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`, and T057 remains sealed.

Durable decision: D-20260901-264.
