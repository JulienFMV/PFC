# Swissgrid NTC Baseline

* raw source: `C:\Users\jbattaglia\pfc_local_data\scenarios\swissgrid_ntc\Grenzfluesse-2026.csv`
* scenario output: `data\electrification_scenarios_composed_p0_real_sources_2030.parquet`
* component output: `data\electrification_scenarios_swissgrid_ntc_baseline_2026.parquet`
* status: `NON-PRODUCTION / OFFICIAL OBSERVED BASELINE / PROXY`
* publication date used for vintage checks: `2026-06-08`
* source method: Swissgrid 2026 annual cross-border NTC columns.
* model method: generic `ntc_ch_*_gw` uses `min(median export, median import)` in GW.

## Baseline Values

| border | ntc_field | export_median_gw | import_median_gw | symmetric_baseline_gw | observations |
| --- | --- | --- | --- | --- | --- |
| CH-AT | ntc_ch_at_gw | 1.2 | 0.9 | 0.9 | 15164 |
| CH-DE | ntc_ch_de_gw | 3.567 | 0.95 | 0.95 | 15164 |
| CH-FR | ntc_ch_fr_gw | 1.3 | 3.1 | 1.3 | 15164 |
| CH-IT | ntc_ch_it_gw | 3.225 | 1.81 | 1.81 | 14980 |

## Changed Scenario Cells

| country | scenario | delivery_year | column | value | border | method |
| --- | --- | --- | --- | --- | --- | --- |
| AT | central | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| AT | fast | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| AT | slow | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| CH | central | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| CH | fast | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| CH | slow | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| DE | central | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| DE | fast | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| DE | slow | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| FR | central | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| FR | fast | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| FR | slow | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| IT | central | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| IT | fast | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| IT | slow | 2030 | ntc_ch_at_gw | 0.9 | CH-AT | min(import/export median absolute NTC MW) |
| AT | central | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| AT | fast | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| AT | slow | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| CH | central | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| CH | fast | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| CH | slow | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| DE | central | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| DE | fast | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| DE | slow | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| FR | central | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| FR | fast | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| FR | slow | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| IT | central | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| IT | fast | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| IT | slow | 2030 | ntc_ch_de_gw | 0.95 | CH-DE | min(import/export median absolute NTC MW) |
| AT | central | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| AT | fast | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| AT | slow | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| CH | central | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| CH | fast | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| CH | slow | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| DE | central | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| DE | fast | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| DE | slow | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| FR | central | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| FR | fast | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| FR | slow | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| IT | central | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| IT | fast | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| IT | slow | 2030 | ntc_ch_fr_gw | 1.3 | CH-FR | min(import/export median absolute NTC MW) |
| AT | central | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| AT | fast | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| AT | slow | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| CH | central | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| CH | fast | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| CH | slow | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| DE | central | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| DE | fast | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| DE | slow | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| FR | central | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| FR | fast | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| FR | slow | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| IT | central | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| IT | fast | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |
| IT | slow | 2030 | ntc_ch_it_gw | 1.81 | CH-IT | min(import/export median absolute NTC MW) |

## Production Interpretation

This closes the numeric NTC P0 gap with a true Swissgrid source, but it is not a 2030 governed expansion assumption. Strict production governance must therefore remain failed until the risk committee approves long-term NTC values or a TYNDP/TSO LT capacity path.
