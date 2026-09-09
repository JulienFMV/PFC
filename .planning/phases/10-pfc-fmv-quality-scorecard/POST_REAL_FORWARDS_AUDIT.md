# Post Real Forwards Audit

Generated: 2026-06-08.

## Source Swap

`data/forwards_history_phase10.parquet` was rebuilt from:

```text
H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Historique2019.xlsx
```

Import command:

```powershell
python scripts/import_fmv_forwards.py `
  --input "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Historique2019.xlsx" `
  --market DE `
  --output data/forwards_history_phase10.parquet `
  --source-tag real_eex_xlsx
```

Result:

| metric | value |
|---|---:|
| rows | 721 |
| vintages | 24 |
| keys per vintage | 29-31 |
| source | `real_eex_xlsx` |
| price min/max/mean | 51.5 / 131.9 / 80.8 EUR/MWh |

`data/eex_forwards_history.parquet` was also rebuilt from the canonical H:
workbooks:

* `Price_Report_EEX_CH_DE_Historique2019.xlsx`
* `Price_Report_EEX_Yearly.xlsx`

Result: 148,748 rows across DE, CH, FR, AT and IT, with data through
2026-06-05. The Yearly workbook wins on duplicate `(date, product, load_type,
market)` rows.

## Preflight

`python scripts/preflight_phase10.py` result:

| check | status |
|---|---|
| data freshness | PASS |
| forwards path | PASS, `forwards_source=real_eex_xlsx`, gate-eligible |
| disk space | FAIL, 2.14 GB free vs 5.0 GB threshold |
| runtime benchmark | PASS, full run estimate 0.36h |

Full Phase 10 scorecard was not run because preflight is `no-go` on disk space.

## Perfect-Foresight Smoke Run

Command:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'
python scripts/run_perfect_foresight.py --estimator sota_amp --no-figures
```

Result: completed using the real EEX parquet.

Key diagnostics for Cal 2025:

| metric | value |
|---|---:|
| `pf_cal` monthly corr | 0.8241 |
| `pf_cal_quarter` monthly corr | 0.9275 |
| `market` monthly corr | 0.8584 |
| `pf_cal_corr` median | 0.918 |
| `market_corr` median | 0.900 |
| best-vintage winter/summer ratio model/realized | 1.557 / 1.699 |
| best-vintage solar-bowl model/realized | 0.428 / 0.558 |
| best-vintage peak/off-peak spread model/realized | 7.158 / 6.407 |

## Next Gate

Free at least 5 GB on the output mount, then rerun:

```powershell
python scripts/preflight_phase10.py
python scripts/run_phase10_scorecard.py --epex-source parquet --output-dir .planning/phases/10-pfc-fmv-quality-scorecard
```

The expected first validation point is that the refreshed scorecard reports
`forwards_source=real_eex_xlsx` and is no longer diagnostic-only due to fallback
forwards.
