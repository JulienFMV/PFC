# NTC Baseline Input Audit

* source: `pfc_shaping\data\entso_15min.parquet`
* usable for proxy baseline: `NO`
* rule: no NTC baseline is produced unless local observations exist

## Column Coverage

| border | kind | column | available | non_null_rows | first_ts | last_ts | median_mw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DE | import | ntc_import_ch_de_mw | False | 0 |  |  |  |
| DE | export | ntc_export_ch_de_mw | False | 0 |  |  |  |
| DE | net | ntc_net_ch_de_mw | False | 0 |  |  |  |
| DE | total | ntc_total_ch_de_mw | False | 0 |  |  |  |
| FR | import | ntc_import_ch_fr_mw | False | 0 |  |  |  |
| FR | export | ntc_export_ch_fr_mw | False | 0 |  |  |  |
| FR | net | ntc_net_ch_fr_mw | False | 0 |  |  |  |
| FR | total | ntc_total_ch_fr_mw | False | 0 |  |  |  |
| IT | import | ntc_import_ch_it_mw | False | 0 |  |  |  |
| IT | export | ntc_export_ch_it_mw | False | 0 |  |  |  |
| IT | net | ntc_net_ch_it_mw | False | 0 |  |  |  |
| IT | total | ntc_total_ch_it_mw | False | 0 |  |  |  |
| AT | import | ntc_import_ch_at_mw | False | 0 |  |  |  |
| AT | export | ntc_export_ch_at_mw | False | 0 |  |  |  |
| AT | net | ntc_net_ch_at_mw | False | 0 |  |  |  |
| AT | total | ntc_total_ch_at_mw | False | 0 |  |  |  |

## Decision

The local ENTSO-E cache cannot currently support an NTC proxy baseline if any required CH border NTC column has zero observations. Final production still requires governed Swissgrid/JAO/TYNDP NTC assumptions.
