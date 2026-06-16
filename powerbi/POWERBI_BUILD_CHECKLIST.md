# PFC_QA.pbip - Build checklist

Power BI Desktop est installe via Microsoft Store. Depuis Codex, l'ouverture GUI
et l'edition visuelle ne sont pas fiables, donc le workflow robuste est:

1. Regenerer les donnees:

```powershell
.\powerbi\refresh_powerbi_data.ps1
```

2. Ouvrir le projet:

```powershell
.\powerbi\open_pfc_qa_powerbi.ps1
```

3. Dans Power BI Desktop, charger les tables depuis `powerbi\data\`:

| table Power BI | fichier |
|---|---|
| `HFC_Hourly` | `hfc_hourly_powerbi.csv` |
| `EEX_Residuals` | `eex_residuals.csv` |
| `Duck_Month_Hour` | `duck_month_hour.csv` |
| `Duck_Season_Hour` | `duck_season_hour.csv` |
| `Annual_Shape` | `annual_shape.csv` |
| `Negative_Low_Hours` | `negative_low_hours.csv` |
| `Summary_Metrics` | `summary_metrics.csv` |

Les requetes Power Query typées sont dans `powerbi\queries\*.pq`.

4. Ajouter les mesures DAX de `powerbi\measures\PFC_Measures.dax`.

5. Pages minimales:

## Executive QA

- Cards:
  - `EEX Max BASE Residual`
  - `EEX Max PEAK Residual`
  - `Weighted Negative Hours`
  - `Min Weighted Price`
  - `Peak Offpeak Spread`
  - `Evening Midday Spread`
- Table: `Summary_Metrics`.

## EEX Calibration

- Matrix:
  - Rows: `EEX_Residuals[load_type]`, `EEX_Residuals[product]`
  - Values: `target_eex_eur_mwh`, `csv_mean_eur_mwh`, `abs_error_eur_mwh`
- Conditional formatting on `abs_error_eur_mwh`.

## Duck Curves

- Line chart:
  - X: `Duck_Month_Hour[hour]`
  - Legend: `Duck_Month_Hour[month]`
  - Y: `Average price_weighted_mean_eur_mwh`
  - Slicer: `year`

## Seasonality

- Line/column chart:
  - X: `Period_Means[month]`
  - Y: `price_weighted_mean_eur_mwh`
  - Legend/slicer: `year`
- Table: `Annual_Shape`.

## Negative Tail

- Matrix heatmap:
  - Rows: `Negative_Low_Hours[month]`
  - Columns: `Negative_Low_Hours[hour]`
  - Values: `fast_negative_hours`, `p10_negative_hours`

6. Sauvegarder dans `powerbi\PFC_QA.pbip`.

## Refresh quotidien

Apres regeneration de `output\ch_hfc_hourly.csv`, lancer:

```powershell
.\powerbi\refresh_powerbi_data.ps1
```

Puis ouvrir Power BI et cliquer `Refresh`.
