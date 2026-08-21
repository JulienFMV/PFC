# CH HFC Hourly Validation Workbook

* workbook: `output\ch_hfc_hourly_20260616_20301231_validation_charts_v7_quote_aware.xlsx`
* source CSV: `output\ch_hfc_hourly_20260616_20301231_v7_quote_aware.csv`
* latest EEX CH BASE date: `2026-06-12`
* scope: `local/test validation`
* production approval: `NO`

## Workbook Blocks

| sheet | role |
|---|---|
| `Raw` | full hourly CSV with calendar fields and calibration product |
| `EEX_Means` | EEX product mean control for every price series |
| `Period_Means` | year > quarter > month average price controls |
| `Duck_Month` | hour x month duck curve data by year |
| `Duck_Season` | hour x season duck curve data by year |
| `Heatmap_Month_Hour` | month x hour weighted mean heatmap |
| `Structural_Width` | structural width by year/month/hour |
| `Negative_Low_Hours` | negative and very-low hour diagnostics |
| `Seasonal_Coherence` | January/October and winter/autumn consistency flags |
| `Quoted_EEX_Products` | residuals against every quoted overlapping EEX product |
| `Charts` | prebuilt charts for quick review |
