# Phase 13 Data Inventory: LT Electrification Scenario Shape

This inventory is the production data contract for scenario-driven LT HPFC
shape. Every row used by the model must be vintage-safe: either measured before
the pricing vintage, or published as a forward scenario before the pricing
vintage.

Canonical target file:

```text
data/electrification_scenarios.parquet
```

Derived model-facing feature file:

```text
data/hpfc_scenario_features.parquet
```

Local first-curve enriched smoke file:

```text
data/electrification_scenarios_ep2050_enriched.parquet
```

This enriched file is not a replacement for TYNDP/MaStR/Pronovo/NTC feeds. It is
a governed local overlay used to make a first PFC run fail-fast on the full
recommended schema while upstream production feeds are still being connected.

Required minimum columns:

```text
publication_date, source, scenario, country, delivery_year, delivery_month,
scenario_weight, quality_flag, demand_twh, pv_gw, wind_gw, battery_power_gw,
battery_energy_gwh, ev_twh, managed_charging_share, heatpump_twh, hydro_twh,
hydro_capacity_gw, hydro_reservoir_twh, nuclear_gw, dispatchable_gw, gas_gw,
coal_gw, ntc_ch_de_gw, ntc_ch_fr_gw, ntc_ch_it_gw
```

## Critical Data Matrix

| data | required fields | primary source | cadence | fallback if unavailable | fallout if missing |
|---|---|---|---|---|---|
| Swiss official energy pathway | `scenario`, `delivery_year`, `demand_twh`, `pv_gw`, `ev_twh`, `heatpump_twh`, `nuclear_gw`, `hydro_twh` | OFEN/BFE Energieperspektiven 2050+ | scenario publication | ENTSO-E TYNDP CH values; internal central scenario marked `fallback` | Cannot justify CH 2030 vs 2027 structural shape. Keep Phase 13 OFF for production. |
| European scenario backbone | `scenario`, `country`, `delivery_year`, `demand_twh`, `pv_gw`, `wind_gw`, batteries/flexibility and dispatchable capacity where available | ENTSO-E/ENTSOG TYNDP 2024 scenario data | TYNDP cycle | national NECP/TSO data, marked `fallback` | Cross-border scenario consistency lost; CH basis/fan chart weak. |
| European wind trajectory | `country in {DE, FR, IT, AT}`, `delivery_year/month`, `wind_gw` | ENTSO-E/ENTSOG TYNDP 2024 scenario data | TYNDP cycle | national TSO/NECP wind scenarios, marked `fallback` | Winter/night and storm-regime price shape not governed; keep wind coefficient at zero and widen structural fan chart. |
| CH PV actuals | `publication_date`, `country=CH`, `delivery_year/month`, `pv_gw` or generation proxy | Pronovo production and installation statistics | monthly | Swissolar/OFEN annual stats, marked `fallback` | PV trajectory cannot be actualized by vintage; use official forward scenario only. |
| CH PV production profile | monthly/quarterly PV output by technology, ideally 15-min | Pronovo monthly production by technology | monthly | ENTSO-E generation actuals or climatological PV capacity factor | Midday-bowl seasonal calibration weaker; keep PV coefficient conservative. |
| DE PV actuals | `country=DE`, `pv_gw`, commissioning date, publication/load date | Bundesnetzagentur MaStR | continuous/monthly extracts | BNetzA aggregate statistics | CH-DE coupling and import-price pressure less reliable. |
| DE battery actuals | `country=DE`, `battery_power_gw`, `battery_energy_gwh`, commissioning date | Bundesnetzagentur MaStR | continuous/monthly extracts | TYNDP / BNetzA aggregate storage statistics | Cannot model DE battery belly-refill spillover; widen scenario uncertainty. |
| CH battery actuals | `country=CH`, `battery_power_gw`, `battery_energy_gwh` | Pronovo/OFEN if available, utility/internal registry | monthly/quarterly | TYNDP CH battery assumptions, marked `fallback` | Battery term for CH must be scenario-only and low confidence. |
| EV trajectory | `ev_twh`, `managed_charging_share` | OFEN EP2050+, TYNDP demand scenarios | scenario publication | vehicle fleet projections with consumption conversion | Evening/night load term disabled or conservative; fan chart wider. |
| Heat-pump trajectory | `heatpump_twh`, optional seasonal profile | OFEN EP2050+, TYNDP heat/hybrid heat-pump data | scenario publication | building-sector electrification assumptions | Winter/evening seasonality term disabled or conservative. |
| Demand trajectory | `demand_twh`, optional monthly/seasonal demand | OFEN EP2050+, TYNDP demand data | scenario publication | historical demand trend + efficiency/electrification internal assumption | All normalized penetration features unstable; do not ship signal without at least fallback demand. |
| CH hydro flexibility | `hydro_twh`, `hydro_capacity_gw`, `hydro_reservoir_twh` | OFEN/BFE hydro statistics, Swissgrid/ENTSO-E actuals, FMV/internal hydro stack | weekly/monthly plus scenario publication | historical reservoir climatology and TYNDP CH hydro, marked `fallback` | Cannot defend CH winter/summer and evening flexibility shape; Phase 13 should remain diagnostic. |
| Nuclear availability | `nuclear_gw`, retirement/availability assumptions | OFEN/Swiss policy docs, TYNDP, operator disclosures | scenario/event-driven | internal availability scenario | Winter/evening scarcity term disabled; use market anchors only. |
| Dispatchable neighbouring capacity | `dispatchable_gw`, `gas_gw`, `coal_gw` by country/year | ENTSO-E TYNDP capacity scenario, national adequacy reports, operator disclosures | TYNDP/event-driven | aggregate dispatchable capacity from TYNDP, marked `fallback` | Evening scarcity and firm-capacity damping not governed; keep dispatchable coefficient zero. |
| Interconnection capacity | `ntc_ch_de_gw`, `ntc_ch_fr_gw`, `ntc_ch_it_gw` | ENTSO-E/TYNDP infrastructure data, Swissgrid/TSO data | TYNDP/event-driven | historical NTC average, marked `fallback` | Country-specific extremes not damped correctly; CH basis signal weak. |
| Scenario weights | `scenario_weight` by `source/scenario/year` | internal risk committee / desk governance | when scenarios approved | equal weights | Weighted curve/fan chart less economically meaningful, but scenario curves remain usable. |
| Publication metadata | `publication_date`, `source`, `quality_flag` | every upstream source | every load | none | Mandatory. Without this, data is not vintage-safe and must not be used. |

## Source URLs

| source | URL | usage |
|---|---|---|
| OFEN/BFE Energieperspektiven 2050+ | https://www.bfe.admin.ch/bfe/en/home/policy/energy-perspectives-2050-plus.html | Swiss official long-term demand, electrification and generation pathway |
| ENTSO-E/ENTSOG TYNDP 2024 scenarios | https://2024.entsos-tyndp-scenarios.eu/tyndp-2024-scenarios/ | Europe-wide scenario backbone |
| ENTSO-E/ENTSOG TYNDP 2024 downloads | https://2024.entsos-tyndp-scenarios.eu/download/ | scenario reports and downloadable data |
| ENTSO-E/ENTSOG scenario results | https://2024.entsos-tyndp-scenarios.eu/scenario-results/ | demand, flexibility, PV, heat-pump and EV scenario results |
| Pronovo monthly production by technology | https://pronovo.ch/news/publication-mensuelle-des-chiffres-de-production-par-technologie/ | CH renewable production actualization |
| Bundesnetzagentur MaStR | https://www.bundesnetzagentur.de/EN/Areas/Energy/CoreEnergyMarketDataRegister/start.html | DE installed generation/storage actualization |
| Swissgrid cross-border load flows / NTC | https://www.swissgrid.ch/en/home/operation/grid-data/cross-border-load-flows.html | official observed CH border NTC and commercial flow baseline |
| Swissgrid NTC values | https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc.html | source definition and operational publication channel for NTC values |
| Ember yearly electricity data | https://ember-energy.org/data/yearly-electricity-data/ | public historical/baseline generation, capacity, import and demand data |
| BFE/SFOE opendata.swiss organization | https://opendata.swiss/en/organization/bundesamt-fur-energie-bfe | complete official BFE public-data catalogue used to prioritize CH data ingestion |
| BFE Energiedashboard catalogue | https://opendata.swiss/en/organization/bundesamt-fur-energie-bfe?q=energiedashboard | daily CH production, consumption, cross-border flow, spot-price and forecast feeds |

## Production Rules

1. `publication_date` is mandatory for every row.
2. At vintage `v`, the model may only use rows with `publication_date <= v`.
3. Actual installation rows require a measurement or commissioning timestamp
   strictly before `v`.
4. Forward assumptions are allowed if their scenario publication date is before
   `v`.
5. Missing critical CH demand or PV scenario data means Phase 13 must stay OFF.
6. Missing batteries/EV/PAC does not block the whole layer, but the corresponding
   coefficient should be zeroed or the fan chart widened.
7. Rows without provenance are not data; they are assumptions and must be marked
   `quality_flag=fallback` or `quality_flag=internal`.
8. Proxy overlays must write a distinct table, must not overwrite official
   fields silently, must update `publication_date` to the assumption governance
   date, and must pass validation with `--require-recommended`.

## Minimum Viable Production Dataset

The smallest dataset that can justify enabling Phase 13 in production:

```text
publication_date
source
scenario in {slow, central, fast}
country = CH
delivery_year in {2027, 2028, 2029, 2030}
scenario_weight
demand_twh
pv_gw
wind_gw
battery_power_gw
battery_energy_gwh
ev_twh
managed_charging_share
heatpump_twh
hydro_twh
hydro_capacity_gw
hydro_reservoir_twh
nuclear_gw
ntc_ch_de_gw
ntc_ch_fr_gw
ntc_ch_it_gw
quality_flag
```

This is the minimum defendable FMV production dataset. `dispatchable_gw`,
`gas_gw` and `coal_gw` are strongly recommended for the neighbouring-country
rows; without them, the scenario can still run, but firm-capacity damping should
be read as incomplete.

## Gold HPFC Scenario Features

The model should consume normalized penetration indicators, not raw GW/TWh
directly. Build the gold table from the governed scenario inventory:

```powershell
$env:PYTHONPATH='.'
python scripts/build_hpfc_scenario_features.py `
  --path data/electrification_scenarios.parquet `
  --vintage 2026-01-15 `
  --output data/hpfc_scenario_features.parquet
```

For Databricks:

```powershell
$env:PYTHONPATH='.'
python scripts/build_hpfc_scenario_features.py `
  --databricks `
  --table-key electrification_scenarios `
  --vintage 2026-01-15 `
  --output data/hpfc_scenario_features.parquet
```

Target feature columns:

| feature | meaning |
|---|---|
| `pv_penetration` | PV energy share when `pv_twh` is available, otherwise `pv_gw / avg_load_gw` |
| `wind_penetration` | Wind energy share when `wind_twh` is available, otherwise `wind_gw / avg_load_gw` |
| `battery_penetration` | Composite of battery power penetration and battery energy cover |
| `battery_power_penetration` | `battery_power_gw / avg_load_gw` |
| `battery_energy_cover_h` | `battery_energy_gwh / avg_load_gw`, in equivalent average-load hours |
| `ev_penetration` | `ev_twh / demand_twh` |
| `heatpump_penetration` | `heatpump_twh / demand_twh` |
| `hydro_flexibility` | Composite of hydro energy share, hydro capacity penetration and reservoir winter cover |
| `import_dependency` | `net_import_twh / demand_twh`, or `(import_twh - export_twh) / demand_twh` |
| `ntc_penetration` | sum of `ntc_*_gw / peak_load_gw` |
| `firm_capacity_penetration` | `(nuclear_gw + dispatchable_gw + gas_gw + coal_gw) / peak_load_gw` |
| `managed_charging_share` | EV charging flexibility share in `[0, 1]` |

These features are the intended explanatory variables for the 2027 -> 2030
shape transition. Raw physical quantities remain in the scenario inventory for
auditability and source reconciliation.

## Local OFEN EP2050+ Extraction

The official OFEN/BFE hourly electricity ZIP can be cached outside the repo on
`C:` and parsed locally:

```powershell
$env:PYTHONPATH='.'
python scripts/import_ep2050_hourly.py `
  --cache-dir C:\Users\jbattaglia\pfc_local_data\ep2050 `
  --years 2025,2030 `
  --scenarios ZERO_Basis,WWB
```

The script downloads, or reuses, the official ZIP:

```text
https://pubdb.bfe.admin.ch/de/publication/download/11142
```

Raw and extracted files stay under `C:\Users\jbattaglia\pfc_local_data\ep2050`.
The normalized outputs are:

| file | content |
|---|---|
| `data/silver_ep2050_hourly.parquet` | long hourly table: scenario/year/datetime/technology/GWh/MWh |
| `data/electrification_scenarios_ep2050.parquet` | annual OFEN scenario rows in the governed scenario schema |
| `data/hpfc_scenario_features_ep2050.parquet` | gold penetration indicators derived from OFEN rows |

The OFEN workbooks report hourly values in `GWh/h`; annual scenario columns
named `*_twh` are therefore divided by 1000 during aggregation.

## Local Enrichment Overlay

For the first local PFC only, the EP2050 annual scenario rows can be enriched
with a documented proxy overlay:

```powershell
$env:PYTHONPATH='.'
python scripts/enrich_electrification_scenarios.py `
  --input data/electrification_scenarios_ep2050.parquet `
  --output data/electrification_scenarios_ep2050_enriched.parquet `
  --hourly data/silver_ep2050_hourly.parquet `
  --assumption-publication-date 2026-06-05
```

Profile: `ch_first_pfc_proxy_v0`.

Filled proxy fields:

| field | proxy rule |
|---|---|
| `battery_power_gw` | linear CH schedule: 0.4 in 2025, 2.0 in 2030, 3.5 in 2035 |
| `battery_energy_gwh` | linear CH schedule: 0.8 in 2025, 5.0 in 2030, 10.0 in 2035 |
| `managed_charging_share` | linear CH schedule: 0.20 in 2025, 0.35 in 2030, 0.50 in 2035 |
| `ntc_ch_de_gw`, `ntc_ch_fr_gw`, `ntc_ch_it_gw` | constant first-smoke capacities 4/4/3 GW |
| `dispatchable_gw`, `gas_gw`, `coal_gw` | 0 for CH first-smoke curve; neighbouring-country capacity feed still required for production |
| `winter_demand_twh` | derived from EP2050 hourly winter months when available |
| `hydro_reservoir_twh` | proxy `0.20 * winter_demand_twh`, clipped at 12 TWh |
| `scenario_weight` | equal within publication/country/year/month group |

The overlay preserves official EP2050 fields when present, stamps
`quality_flag=*_proxy_enriched`, appends the profile name to `source`, and sets
`publication_date` to the later of the original source publication date and the
assumption governance date. It is acceptable for local smoke/prod-readiness
work; it is not sufficient as the final committee-grade production scenario.

## Local Slow/Central/Fast EP2050 Workflow

The enriched OFEN inventory can be mapped to a local three-scenario production
smoke set:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/build_ep2050_multi_scenario_pfc.py `
  --scenario-input data/electrification_scenarios_ep2050_enriched.parquet `
  --scenario-output data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet `
  --features-output data/hpfc_scenario_features_ep2050_enriched_slow_central_fast.parquet `
  --market CH `
  --start-date 2030-01-01 `
  --horizon-days 365 `
  --output-prefix ep2050_pfc_2030 `
  --fan-chart-output output/ep2050_pfc_2030_structural_fan_chart.parquet `
  --summary .planning/phases/13-lt-electrification-scenario-shape/MULTI-SCENARIO-PFC.md
```

Mapping profile: `ep2050_slow_central_fast_mapping_v0`.

| target scenario | source rule | weight |
|---|---|---:|
| `slow` | alias of enriched OFEN `WWB` | 0.25 |
| `central` | explicit midpoint between enriched `WWB` and `ZERO_Basis` | 0.50 |
| `fast` | alias of enriched OFEN `ZERO_Basis` | 0.25 |

The midpoint is stamped `quality_flag=internal_midpoint_proxy_enriched`.
No missing physical field is converted to zero by the mapping step; missing
source values remain missing and must be caught by validation or downstream
governance.

Validate the expanded scenario table actually consumed by the PFC builds:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/validate_electrification_scenarios.py `
  --path output/ep2050_pfc_2030_scenario_expanded.parquet `
  --country CH `
  --scenarios slow,central,fast `
  --years 2030,2031 `
  --vintage 2026-06-05 `
  --require-recommended `
  --report .planning/phases/13-lt-electrification-scenario-shape/MULTI-SCENARIO-GATE.md
```

The runner builds each curve with `enable_electrification_shape=True`,
`enable_intraday_amplitude_shrinkage=True`, and
`require_electrification_scenarios=True`, then writes a separate weighted
structural fan chart. The Phase 13 production flags remain OFF by default in the
pipeline and assembler.

## Production Gate vs Smoke Gate

Two validation levels are intentionally separate:

| gate | command profile | expected local EP2050 proxy result | meaning |
|---|---|---|---|
| smoke/prod-readiness | `--require-recommended` | OK | schema and scenario coverage are sufficient to run a fail-fast local PFC |
| final production | `--require-production --required-countries CH,DE,FR,IT,AT` | FAILED | blocks proxy/internal CH-only data and requires complete country/scenario/year coverage |

Run the strict final-production gate on the canonical scenario file:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/validate_electrification_scenarios.py `
  --path data/electrification_scenarios.parquet `
  --country CH `
  --scenarios slow,central,fast `
  --years 2027-2031 `
  --vintage 2026-06-05 `
  --require-recommended `
  --require-production `
  --required-countries CH,DE,FR,IT,AT `
  --report .planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-PRODUCTION-GATE.md
```

The current local proxy should fail this gate until official multi-country
TYNDP/OFEN/Pronovo/MaStR/NTC/fuel-flexibility feeds are loaded. A failed
production gate is the correct outcome for local proxy data. The gate is
stricter than country presence: every required country must carry every
requested scenario and delivery year.

The same protection is available inside the model. Production orchestration can
construct:

```python
PFCAssembler(
    ...,
    enable_electrification_shape=True,
    require_electrification_scenarios=True,
    require_production_electrification_scenarios=True,
)
```

This keeps the default pipeline unchanged, but fails the build before shaping if
the loaded scenario inventory contains proxy/internal/fallback quality flags,
misses production metadata, or does not cover the required multi-country
production set.

## Official TYNDP 2024 Supply Component

The first official multi-country component is now cached locally and converted
into the scenario schema:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/import_tyndp2024_supply_inputs.py `
  --workbook "C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\20231103-Final-Supply-Inputs-for-TYNDP-2024-Scenarios.xlsx\20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx" `
  --output data/electrification_scenarios_tyndp2024_supply.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-SUPPLY-IMPORT.md `
  --countries CH,DE,FR,IT,AT `
  --scenarios slow,central,fast `
  --years 2030,2040 `
  --publication-date 2024-05-31 `
  --ingested-at-utc 2026-06-11
```

Local raw cache:

```text
C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\raw
```

Produced component:

| file | rows | status |
|---|---:|---|
| `data/electrification_scenarios_tyndp2024_supply.parquet` | 30 | official supply-side partial |
| `data/electrification_scenarios_tyndp2024_demand.parquet` | 16 | official neighbouring-country demand partial |
| `.planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-SUPPLY-IMPORT.md` | n/a | extraction report |
| `.planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-SUPPLY-PRODUCTION-GATE.md` | n/a | strict gate report, expected FAILED |
| `.planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-DEMAND-IMPORT.md` | n/a | demand extraction report |
| `.planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-DEMAND-PRODUCTION-GATE.md` | n/a | strict gate report, expected FAILED |

Coverage is `CH/DE/FR/IT/AT x slow/central/fast x 2030/2040` for PV,
wind, nuclear, battery energy, gas, coal and CO2. Missing critical demand,
NTC, battery power, hydro, EV, heat-pump and flexibility fields remain null.
The quality flag is `official_tyndp_supply_partial`, intentionally rejected by
the production gate. This prevents a supply-only component from being mistaken
for the final FMV scenario inventory.

The Demand Outputs workbook is also imported with `pyxlsb` into raw official
TYNDP scenario labels:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/import_tyndp2024_demand_outputs.py `
  --workbook "C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb" `
  --output data/electrification_scenarios_tyndp2024_demand.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-DEMAND-IMPORT.md `
  --countries AT,DE,FR,IT `
  --publication-date 2024-05-31 `
  --ingested-at-utc 2026-06-11
```

This yields `AT/DE/FR/IT x tyndp_distributed_energy/tyndp_global_ambition x
2040/2050` final electricity demand. CH is absent from the TYNDP demand output
and must remain sourced from OFEN/EP2050. No `slow/central/fast` mapping is
inferred inside the importer.

Neighbour 2030 demand is bridged as a non-production component:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/build_tyndp2024_neighbor_demand_bridge.py `
  --workbook "C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb" `
  --entso pfc_shaping/data/entso_15min.parquet `
  --output data/electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-NEIGHBOR-DEMAND-BRIDGE.md `
  --countries AT,DE,FR,IT `
  --vintage 2026-06-11 `
  --publication-date 2026-06-11 `
  --ingested-at-utc 2026-06-11
```

Method: interpolate official TYNDP REF 2019 to DE/GA 2040 at year 2030,
then set `slow=min(DE2030, GA2030)`, `fast=max(DE2030, GA2030)`, and
`central=midpoint`. The local ENTSO-E neighbour load columns are currently
empty, so `peak_load_gw` and `winter_demand_twh` remain null. The quality flag
is `internal_tyndp_demand_bridge_partial_proxy`, intentionally rejected by the
production gate.

NTC cannot be built from the local ENTSO-E cache because the local NTC columns
are empty:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/audit_ntc_baseline_inputs.py `
  --entso pfc_shaping/data/entso_15min.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/NTC-BASELINE-AUDIT.md
```

The local `ntc_*_ch_de/fr/it/at_mw` columns exist but have zero observations.
No ENTSO-E-derived local NTC baseline is generated from that cache.
The NTC gap is now closed numerically with the Swissgrid annual cross-border
CSV baseline described below. Final production still needs governed
Swissgrid/JAO/TYNDP long-term assumptions for `ntc_ch_de_gw`,
`ntc_ch_fr_gw`, `ntc_ch_it_gw`, and `ntc_ch_at_gw`.

## Composed Partial Multi-Country Inventory

The next prod-readiness layer composes available official components without
pretending they are complete:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/compose_lt_scenario_inventory.py `
  --tyndp-supply data/electrification_scenarios_tyndp2024_supply.parquet `
  --ch-ep2050 data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet `
  --neighbor-demand data/electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet `
  --output data/electrification_scenarios_composed_partial_2030.parquet `
  --features-output data/hpfc_scenario_features_composed_partial_2030.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/COMPOSED-PARTIAL-INVENTORY.md `
  --years 2030 `
  --vintage 2026-06-11 `
  --countries CH,DE,FR,IT,AT `
  --scenarios slow,central,fast
```

This writes:

| file | rows | status |
|---|---:|---|
| `data/electrification_scenarios_composed_partial_2030.parquet` | 15 | multi-country partial composed inventory |
| `data/hpfc_scenario_features_composed_partial_2030.parquet` | 15 | derived features for diagnostics only |
| `.planning/phases/13-lt-electrification-scenario-shape/COMPOSED-PARTIAL-INVENTORY.md` | n/a | remaining critical gap report |
| `.planning/phases/13-lt-electrification-scenario-shape/COMPOSED-PARTIAL-PRODUCTION-GATE.md` | n/a | strict gate report, expected FAILED |

Composition rules:

* TYNDP Supply is the multi-country base for PV, wind, nuclear, battery energy,
  fuels and CO2.
* CH EP2050 enriched rows overlay only CH demand, peak/winter demand, EV,
  heat-pump, hydro, import/export and battery power fields.
* The neighbour demand bridge fills `demand_twh` for AT/DE/FR/IT in 2030.
* Missing neighbour peak/winter demand, NTC, dispatchable capacity, flex and
  PV/wind TWh remain null. They are not converted to zero.
* The output is flagged `partial` / `proxy`, so `--require-production` must fail.

## Scenario Governance Gate

Final activation also requires an explicit governance manifest. The current
manifest is deliberately `draft`:

```text
.planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml
```

Run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/validate_scenario_governance.py `
  --inventory data/electrification_scenarios_composed_partial_2030.parquet `
  --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml `
  --vintage 2026-06-11 `
  --countries CH,DE,FR,IT,AT `
  --scenarios slow,central,fast `
  --years 2030 `
  --report .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-VALIDATION.md
```

The governance gate checks:

* `status: approved` and `approved_for_production: true`;
* `approval_date <= vintage` and non-empty `approved_by`;
* manifest scope equals the requested countries, scenarios and years;
* scenario weights exist and sum to one;
* source component paths exist and have vintage-safe publication dates;
* effective rows reference declared source components through `source_id` or
  `component_ids`;
* no `proxy`, `partial`, `internal`, `fallback` or `synthetic` quality flags;
* no null critical production values in the latest as-of inventory rows.
* critical zero values are either explicitly justified or listed in
  `zero_allowed_fields`;
* configured physical bounds are respected.

Current result is expected `FAILED`. This is the formal list of what must be
changed before Phase 13 can be final FMV production.

The same failures can be converted into an actionable gap register:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/build_lt_data_gap_register.py `
  --inventory data/electrification_scenarios_prod_candidate_neutralized_2030.parquet `
  --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml `
  --vintage 2026-06-12 `
  --countries CH,DE,FR,IT,AT `
  --scenarios slow,central,fast `
  --years 2030 `
  --csv-output data/lt_scenario_governance_gap_register.csv `
  --report .planning/phases/13-lt-electrification-scenario-shape/LT-DATA-GAP-REGISTER.md
```

The original composed partial inventory had 42 blocking rows. After the P0
structural bridge, Swissgrid NTC baseline, Ember yearly baseline, conditional
cross-border contract and explicit P1 neutralisation policy, the current
prod-candidate register contains 18 blocking rows: 15 proxy/partial quality
flags and 3 governance decision items. The register is a backlog; it must not
be used to impute values or to zero-fill missing critical production fields.

Agent expert reviews are recorded as non-voting controls:

```text
.planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-EXPERT-REVIEWS.md
```

They can recommend go/no-go, but they are not production approvers. Production
approval requires accountable human sign-off from Model Owner, Data Owner,
Market Risk and independent Model Validation.

Build the consolidated approval pack with:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/build_scenario_governance_approval_pack.py `
  --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml `
  --governance-report .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-VALIDATION.md `
  --gap-register data/lt_scenario_governance_gap_register.csv `
  --expert-reviews .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-EXPERT-REVIEWS.md `
  --output .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-APPROVAL-PACK.md
```

Current pack recommendation is `NO-GO`, as intended.

Current composed rows carry row-level provenance:

| row family | `component_ids` |
|---|---|
| CH rows | `tyndp2024_supply,ch_ep2050_enriched,internal_p0_structural_bridge,swissgrid_ntc_2026_baseline,ember_yearly_2026_baseline,lt_field_neutralization_policy_20260612` |
| AT/DE/FR/IT rows | `tyndp2024_supply,tyndp2024_neighbor_demand_bridge,internal_p0_structural_bridge,swissgrid_ntc_2026_baseline,ember_yearly_2026_baseline,lt_field_neutralization_policy_20260612` |

## Local-Test Agent Governance

Production governance remains strict: agents are reviewers, not accountable
human approvers. For local/test only, Phase 13 now has a separate manifest that
allows expert-agent approval while keeping `approved_for_production: false`:

```text
.planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml
```

Run the local/test gate:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/validate_scenario_governance.py `
  --inventory data/electrification_scenarios_prod_candidate_neutralized_2030.parquet `
  --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml `
  --vintage 2026-06-12 `
  --countries CH,DE,FR,IT,AT `
  --scenarios slow,central,fast `
  --years 2030 `
  --mode local-test `
  --report .planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-GOVERNANCE-GATE.md
```

Current result: `OK`, with 15 effective rows and no gate issues. This does not
change the production gate; it creates a controlled local/test approval tier.

Build the local/test CH PFC:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/build_local_test_ch_pfc.py `
  --inventory data/electrification_scenarios_prod_candidate_neutralized_2030.parquet `
  --manifest .planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml `
  --vintage 2026-06-12 `
  --market CH `
  --start-date 2030-01-01 `
  --horizon-days 365 `
  --summary .planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-CH-PFC.md
```

Generated artefacts:

| output | rows | role |
|---|---:|---|
| `output/local_test_ch_pfc_2030_slow.parquet` | 35,040 | CH 2030 slow local/test curve |
| `output/local_test_ch_pfc_2030_central.parquet` | 35,040 | CH 2030 central local/test curve |
| `output/local_test_ch_pfc_2030_fast.parquet` | 35,040 | CH 2030 fast local/test curve |
| `output/local_test_ch_pfc_2030_structural_fan_chart.parquet` | 35,040 | weighted structural fan chart |
| `output/local_test_ch_pfc_2030_scenario_expanded.parquet` | 6 | CH 2030/2031 scenario rows consumed by the build |
| `data/hpfc_scenario_features_local_test_2030.parquet` | 6 | derived features for the local/test build |
| `.planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-CH-PFC.md` | n/a | build report and limitations |

Local/test PFC summary:

| scenario | mean | min | p05 | p95 | max | midday_mean | evening_mean | night_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `slow` | 68.7398 | 19.2840 | 32.4119 | 105.9044 | 124.1458 | 62.7839 | 71.2973 | 69.2571 |
| `central` | 68.7398 | 19.2811 | 32.2993 | 105.9026 | 124.2926 | 62.4696 | 71.5419 | 69.3596 |
| `fast` | 68.7398 | 19.2780 | 32.1647 | 105.8288 | 124.4613 | 62.1221 | 71.8032 | 69.4787 |

Fan chart summary:

| metric | value |
|---|---:|
| weighted_mean | 68.7398 |
| mean structural width | 0.3445 |
| p95 structural width | 0.9886 |
| max structural width | 1.3482 |

The runner keeps the proxy/partial/internal flags visible, requires
`require_electrification_scenarios=True`, and writes a report that explicitly
forbids production activation.

### Hourly CSV export from tomorrow to 2030-12-31

The local/test PFC can be exported as an hourly CSV on a Europe/Zurich delivery
clock. For the current run, "tomorrow" is 2026-06-13. The export enforces
EEX freshness by default: the latest CH BASE snapshot in
`data/eex_forwards_history.parquet` must equal the previous business day of
`--valuation-date`. For the 2026-06-12 run this requires the 2026-06-11 EEX
close.

Refresh EEX before the run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/rebuild_forwards_history.py `
  --history "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Hist.xlsx" `
  --yearly "H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx"
```

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/export_local_test_ch_hourly_csv.py `
  --valuation-date 2026-06-12 `
  --local-start-date 2026-06-13 `
  --local-end-date 2030-12-31 `
  --output output/ch_pfc_hourly_20260613_20301231.csv `
  --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-CSV-20260613-20301231.md `
  --prefix local_test_ch_pfc_20260613_20301231
```

Generated artefacts:

| output | rows | role |
|---|---:|---|
| `output/ch_pfc_hourly_20260613_20301231.csv` | 39,913 | hourly CH local/test PFC CSV, Europe/Zurich delivery hours |
| `.planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-CSV-20260613-20301231.md` | n/a | export report with EEX calibration audit |
| `output/local_test_ch_pfc_20260613_20301231_structural_fan_chart.parquet` | 159,744 | 15-minute source fan chart used for hourly aggregation |

CSV columns:

```text
timestamp_ch, utc_offset_ch, timestamp_utc,
price_slow_eur_mwh, price_central_eur_mwh, price_fast_eur_mwh,
price_weighted_mean_eur_mwh,
structural_p10_eur_mwh, structural_p50_eur_mwh, structural_p90_eur_mwh,
structural_width_eur_mwh
```

`timestamp_ch` and `timestamp_utc` are exported in Excel-friendly
`dd.mm.yyyy hh:mm` format. `utc_offset_ch` keeps the Swiss DST offset as text
(`UTC+02:00` or `UTC+01:00`) so the duplicated local hour on autumn DST days
remains auditable.

The CSV is post-calibrated to the required EEX CH BASE snapshot
`2026-06-11`. The selected calibration products are monthly for the partial
2026 horizon, quarterly for 2027 where all quarters are available, and calendar
products for 2028-2030. The audit report confirms zero residual mean error on
these products after calibration. `--allow-stale-forwards` exists only for
explicit diagnostics/backtests and must not be used for a requested fresh run.

### Hourly CSV V4 local/test shape upgrade

The expert-agent audit found that the baseline CSV is well calibrated but too
conservative on long-horizon duck-curve persistence and structural fan width.
The V4 local/test run keeps EEX product means exact, then applies an explicit
non-production Swiss hydro/PV/electrification overlay with zero mean by EEX
calibration product. The overlay is disabled by default and must be requested
with `--enable-structural-shape-upgrade`.

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/export_local_test_ch_hourly_csv.py `
  --valuation-date 2026-06-12 `
  --local-start-date 2026-06-13 `
  --local-end-date 2030-12-31 `
  --output output/ch_pfc_hourly_20260613_20301231_v4_shape.csv `
  --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-CSV-20260613-20301231-V4-SHAPE.md `
  --prefix local_test_ch_pfc_20260613_20301231_v2_shape `
  --skip-build `
  --enable-structural-shape-upgrade `
  --structural-shape-upgrade-intensity 1.0 `
  --structural-scenario-spread-intensity 2.0

python scripts/audit_ch_pfc_hourly_shape.py `
  --csv output/ch_pfc_hourly_20260613_20301231_v4_shape.csv `
  --forwards data/eex_forwards_history.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-SHAPE-AUDIT-20260613-20301231-V4.md
```

V4 local/test metrics:

| metric | value |
|---|---:|
| audit score | 9.00 / 10 |
| rows | 39,913 |
| max EEX product mean error | 0.000000 EUR/MWh |
| structural width mean | 8.6762 EUR/MWh |
| structural width p95 | 19.8651 EUR/MWh |
| 2030 evening-minus-midday | 20.3082 EUR/MWh |
| 2030 weekend-minus-weekday | -4.7730 EUR/MWh |
| abs hourly ramp p99 | 27.7071 EUR/MWh |
| abs product-boundary jump p95 | 18.9450 EUR/MWh |

V4 is the current recommended local/test diagnostic curve. It is not a
production FMV curve: the overlay is expert-agent approved only for local/test,
and the remaining high boundary/ramp metrics still require upstream smoothing
before production use.

### Hourly CSV V5 negative-price capture

V5 adds an explicit bounded negative-price capture layer on top of V4. It is
designed for plausible PV-capture pockets in spring/summer midday hours, mostly
in `fast` / `structural_p10`, while preserving every EEX product mean exactly.
The weighted mean remains non-negative in this local/test run.

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/export_local_test_ch_hourly_csv.py `
  --valuation-date 2026-06-12 `
  --local-start-date 2026-06-13 `
  --local-end-date 2030-12-31 `
  --output output/ch_pfc_hourly_20260613_20301231_v5_negative_capture.csv `
  --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-CSV-20260613-20301231-V5-NEGATIVE-CAPTURE.md `
  --prefix local_test_ch_pfc_20260613_20301231_v2_shape `
  --skip-build `
  --enable-structural-shape-upgrade `
  --structural-shape-upgrade-intensity 1.0 `
  --structural-scenario-spread-intensity 2.0 `
  --enable-negative-price-capture `
  --negative-price-capture-intensity 1.0 `
  --negative-price-floor -30

python scripts/audit_ch_pfc_hourly_shape.py `
  --csv output/ch_pfc_hourly_20260613_20301231_v5_negative_capture.csv `
  --forwards data/eex_forwards_history.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/CH-PFC-HOURLY-SHAPE-AUDIT-20260613-20301231-V5-NEGATIVE-CAPTURE.md
```

V5 local/test metrics:

| metric | value |
|---|---:|
| audit score | 9.00 / 10 |
| rows | 39,913 |
| max EEX product mean error | 0.000000 EUR/MWh |
| weighted mean negative hours | 0 |
| structural p10 negative hours | 82 |
| minimum `price_fast_eur_mwh` | -2.5032 EUR/MWh |
| minimum `price_central_eur_mwh` | -0.0926 EUR/MWh |
| minimum weighted mean | 3.6040 EUR/MWh |
| structural width mean | 9.2906 EUR/MWh |
| structural width p95 | 23.4592 EUR/MWh |
| 2030 evening-minus-midday | 23.1611 EUR/MWh |

V5 is the recommended local/test diagnostic curve when negative-price capture
is desired. V4 remains the recommended local/test curve when a strictly
non-negative export is required.

### Excel validation workbook

An Excel workbook is generated for human validation of the HFC/PFC shape:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/build_ch_hfc_validation_workbook.py `
  --csv output/ch_hfc_hourly_20260616_20301231_v5_negative_capture.csv `
  --forwards data/eex_forwards_history.parquet `
  --output output/ch_hfc_hourly_20260616_20301231_validation.xlsx `
  --report .planning/phases/13-lt-electrification-scenario-shape/CH-HFC-HOURLY-VALIDATION-WORKBOOK-20260616-20301231.md
```

Workbook blocks:

| sheet | role |
|---|---|
| `Raw` | full hourly HFC with calendar fields, calibration product and Excel-native timestamp cells |
| `EEX_Means` | EEX product mean control for all price series |
| `Period_Means` | year > quarter > month mean prices |
| `Duck_Month` | month/hour duck-curve table by year |
| `Duck_Season` | season/hour duck-curve table by year |
| `Heatmap_Month_Hour` | month x hour weighted-mean heatmap |
| `Structural_Width` | structural width by year/month/hour |
| `Negative_Low_Hours` | negative and very-low hour diagnostics |
| `Charts` | prebuilt charts for monthly/seasonal duck curve, structural width and negative hours |

The `Raw` sheet writes `timestamp_ch` and `timestamp_utc` as real Excel
datetimes formatted `dd.mm.yyyy hh:mm`. `utc_offset_ch` remains text to audit
summer/winter time and DST duplicated local hours.

If desktop Excel reports workbook repair on the charted file, use the safe
workbook:

```text
output/ch_hfc_hourly_20260616_20301231_validation_safe.xlsx
```

The safe workbook removes embedded Excel tables and charts while preserving all
validation sheets, Excel-native timestamp cells and heatmap conditional
formatting. A French-locale CSV companion is also available:

```text
output/ch_hfc_hourly_20260616_20301231_v5_negative_capture_excel_fr.csv
```

It uses `;` as separator, comma decimals and UTF-8 BOM for direct Excel opening.

## P0 Structural Bridge

Some P0 gaps can be closed numerically with traceable local bridges. This is
still non-production because the bridge remains proxy/internal, but it removes
avoidable nulls before focusing on genuinely missing external sources.

Run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/bridge_lt_p0_structural_fields.py `
  --input data/electrification_scenarios_composed_partial_2030.parquet `
  --output data/electrification_scenarios_composed_p0_bridge_2030.parquet `
  --features-output data/hpfc_scenario_features_composed_p0_bridge_2030.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/P0-STRUCTURAL-BRIDGE.md `
  --entso pfc_shaping/data/entso_15min.parquet `
  --demand-workbook C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024\extracted\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb\Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb `
  --countries CH,DE,FR,IT,AT
```

Closed by the bridge:

* `peak_load_gw`: CH historical peak/average load ratio;
* `winter_demand_twh`: CH historical Nov-Mar demand share;
* `pv_twh` and `wind_twh`: explicit country capacity-factor bridge;
* `battery_power_gw`: four-hour duration bridge from battery energy;
* `ev_twh` and `heatpump_twh`: TYNDP Demand REF2019-to-2040 bridge;
* `dispatchable_gw`: nuclear-only lower-bound only when nuclear capacity is
  strictly positive.

Not closed by this internal bridge:

* CH-AT/DE/FR/IT NTC;
* neighbour hydro energy/capacity/reservoir;
* neighbour import/export/net-import balance;
* dispatchable capacity outside the nuclear lower-bound;
* production-quality flags and human approval.

The bridge reduces the gap register from 42 to 35 blockers, but remains a
production `NO-GO`.

## Public Baseline Sources Wired

Two true public sources are now wired locally as additive, vintage-safe
baseline components. They reduce avoidable numeric nulls, but remain
`proxy` because they are observed/historical baselines, not approved 2030
scenario paths.

### Swissgrid NTC baseline

Raw local file:

```text
C:\Users\jbattaglia\pfc_local_data\scenarios\swissgrid_ntc\Grenzfluesse-2026.csv
```

Run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/apply_swissgrid_ntc_baseline.py `
  --input data/electrification_scenarios_composed_p0_bridge_2030.parquet `
  --swissgrid-csv C:\Users\jbattaglia\pfc_local_data\scenarios\swissgrid_ntc\Grenzfluesse-2026.csv `
  --output data/electrification_scenarios_composed_p0_real_sources_2030.parquet `
  --features-output data/hpfc_scenario_features_composed_p0_real_sources_2030.parquet `
  --component-output data/electrification_scenarios_swissgrid_ntc_baseline_2026.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/SWISSGRID-NTC-BASELINE.md
```

Applied values use `min(median export, median import)` in GW for each Swiss
border, avoiding overstatement of a non-directional model field:

| field | baseline GW |
|---|---:|
| `ntc_ch_at_gw` | 0.90 |
| `ntc_ch_de_gw` | 0.95 |
| `ntc_ch_fr_gw` | 1.30 |
| `ntc_ch_it_gw` | 1.81 |

### Ember yearly baseline

Raw local file:

```text
C:\Users\jbattaglia\pfc_local_data\scenarios\ember_yearly\yearly_full_release_long_format.csv
```

Run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/apply_ember_yearly_baseline.py `
  --input data/electrification_scenarios_composed_p0_real_sources_2030.parquet `
  --ember-csv C:\Users\jbattaglia\pfc_local_data\scenarios\ember_yearly\yearly_full_release_long_format.csv `
  --output data/electrification_scenarios_composed_p0_public_sources_2030.parquet `
  --features-output data/hpfc_scenario_features_composed_p0_public_sources_2030.parquet `
  --component-output data/electrification_scenarios_ember_yearly_baseline_2026.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/EMBER-YEARLY-BASELINE.md
```

Closed where positive/source-present:

* `hydro_twh` and `hydro_capacity_gw` for AT/DE/FR/IT;
* `net_import_twh` for AT/DE/FR/IT;
* `gas_gw`, `coal_gw` where non-zero;
* missing `dispatchable_gw` using a gas+coal+nuclear capacity lower-bound.

Deliberately still not filled in the public-source inventory:

* `import_twh` and `export_twh`, because Ember gives net imports, not gross
  country import/export paths. The production contract now treats the
  cross-border balance as conditional: a governed `net_import_twh` is sufficient
  for the model, while gross import/export remain diagnostics;
* `hydro_reservoir_twh` for neighbouring countries, because Ember does not
  provide reservoir energy capacity. The field remains production-critical for
  CH only and is explicitly neutralised for neighbours until a governed source
  is wired;
* zero coal capacity rows, because critical missing data must not become
  silent zero values.

Current gap register after public baselines contains 23 blockers. The previous
field-level P0 numeric gaps on gross import/export and non-CH reservoir capacity
are closed by explicit contract rules, not by imputation. Remaining public-source
blockers are 15 proxy/partial quality flags, 3 governance decision items and 5
P1 numeric/flex fields.

### Explicit LT neutralization policy

Run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/apply_lt_neutralization_policy.py
```

Outputs:

| file | role |
|---|---|
| `data/electrification_scenarios_prod_candidate_neutralized_2030.parquet` | prod-candidate scenario inventory with explicit P1 zero neutralisations |
| `data/hpfc_scenario_features_prod_candidate_neutralized_2030.parquet` | derived HPFC scenario features with no silent hydro-reservoir zero in `hydro_flexibility` |
| `data/electrification_scenarios_neutralization_audit_2030.csv` | row/field audit of every neutralised value |
| `.planning/phases/13-lt-electrification-scenario-shape/LT-NEUTRALIZATION-POLICY.md` | human-readable neutralisation report |

This policy filled 66 missing values with explicit `*_zero_justification`
columns: `coal_gw`, `dsm_gw`, `electrolysis_twh`, `p2x_gw` and
`managed_charging_share`. It never overwrites non-null source values. On this
candidate, the gap register contains 18 blockers: 15 proxy/partial quality
flags and 3 manifest approval items. There are no remaining critical numeric
null blockers.

### BFE Energiedashboard daily actuals

The BFE/SFOE opendata.swiss Energiedashboard catalogue adds six daily public
CSV feeds useful for CH historical calibration and diagnostics:

| feed | local variable family | use |
|---|---|---|
| `BFE-DS-0093` electricity production Swissgrid | production by hydro ROR, hydro storage, nuclear, solar, thermal, wind | CH daily production calibration and ENTSO-E cross-check |
| `BFE-DS-0096` national and final consumption | `national_consumption_gwh`, `final_consumption_gwh` | CH demand/winter/peak diagnostics |
| `BFE-DS-0094` daily electricity import/export flows | bilateral gross flows CH-AT/DE/FR/IT and `net_import_gwh` | historical cross-border balance; gross flow diagnostics |
| `BFE-DS-0087` day-ahead base spot price | `spot_baseload_eur_mwh` | public price sanity check; not a replacement for EPEX/EEX curves |
| `BFE-DS-0082` SDSC consumption forecast | final consumption forecast p2.5/mean/p97.5 | only vintage-safe if archived daily without overwrite |
| `BFE-DS-0095` model-based estimated national consumption | recent estimated national consumption | latest CH load nowcast diagnostic |

Run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/import_bfe_energiedashboard.py `
  --raw-dir C:\Users\jbattaglia\pfc_local_data\scenarios\bfe_energiedashboard `
  --output data/bfe_energiedashboard_daily.parquet `
  --report .planning/phases/13-lt-electrification-scenario-shape/BFE-ENERGYDASHBOARD.md `
  --ingested-at-utc 2026-06-12T00:00:00Z
```

Current local import:

| output | rows | date coverage |
|---|---:|---|
| `data/bfe_energiedashboard_daily.parquet` | 70,610 | 2015-01-01 to 2026-06-24 depending on feed |
| `.planning/phases/13-lt-electrification-scenario-shape/BFE-ENERGYDASHBOARD.md` | report | source summary, variable list, vintage limitations |

Quant use:

* improves CH daily historical calibration with official BFE/Swissgrid actuals;
* gives true historical gross import/export flows, but not LT 2030 gross-flow
  scenarios;
* can replace local empty ENTSO-E flow diagnostics where daily energy is
  sufficient;
* does not approve the scenario inventory for production because rows have no
  row-level publication timestamp and are not forward-looking scenario paths.

### BFE full opendata.swiss catalogue audit

The full BFE/SFOE opendata.swiss organization page is now scanned into a
machine-readable discovery register:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/audit_bfe_opendata_catalog.py `
  --output data/bfe_opendata_catalog_audit.csv `
  --report .planning/phases/13-lt-electrification-scenario-shape/BFE-OPENDATA-CATALOG-AUDIT.md
```

Outputs:

| output | role |
|---|---|
| `data/bfe_opendata_catalog_audit.csv` | discovery register for 146 BFE datasets, with priority/family classification and first resource URL |
| `.planning/phases/13-lt-electrification-scenario-shape/BFE-OPENDATA-CATALOG-AUDIT.md` | human-readable shortlist of 54 P0 and 39 P1 candidates |

This register is not a model input. A listed dataset becomes model-usable only
after a dedicated importer, raw cache path under `C:`, schema validation,
vintage metadata and a validation report.

Further opendata.swiss candidates identified but not yet ingested:

| dataset | why it matters | expected use |
|---|---|---|
| BFE monthly/annual electricity balance | production, import/export, consumption monthly/annual | reconcile daily Energiedashboard and EP2050 annual totals |
| BFE/geo.admin charging requirements 2035 | municipal plug-in vehicle stock for 2035 | EV scenario cross-check, not direct 2030 demand without conversion |
| BFE PV winter production profiles / PV large plants | winter PV shape and alpine PV potential | calibrate winter/summer PV shape and slow/fast PV scenario narrative |

### BFE structural actuals wired

Three P0 BFE candidates from the catalogue audit are now imported locally:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/import_bfe_structural_actuals.py `
  --raw-dir C:\Users\jbattaglia\pfc_local_data\scenarios\bfe_structural_actuals `
  --ingested-at-utc 2026-06-12T00:00:00Z
```

Raw cache:

```text
C:\Users\jbattaglia\pfc_local_data\scenarios\bfe_structural_actuals
```

Outputs:

| output | rows | role |
|---|---:|---|
| `data/bfe_hydro_reservoir_weekly.parquet` | 20,700 | weekly CH reservoir content/capacity/fill ratio by region from 2000-01-03 to 2026-06-08 |
| `data/bfe_ch_production_plants.parquet` | 327,223 | BFE in-operation electricity production plant register, plant-level |
| `data/bfe_ch_installed_capacity_actuals.parquet` | 7 | CH installed capacity actuals by technology |
| `data/bfe_wasta_hydro_plants.parquet` | 728 | WASTA hydro plant technical actuals |
| `data/bfe_wasta_hydro_summary.parquet` | 4 | hydro type summary: ROR, pure pumped, storage and mixed pumped-storage |
| `data/bfe_ch_structural_actuals.parquet` | 12 | model-facing long actuals for calibration and scenario actualization |
| `.planning/phases/13-lt-electrification-scenario-shape/BFE-STRUCTURAL-ACTUALS.md` | n/a | import report and source URLs |

Latest model-facing actuals:

| variable | value | unit | measurement date | source |
|---|---:|---|---|---|
| `hydro_reservoir_twh` | 2.229 | TWh | 2026-06-08 | BFE Speicherseen |
| `hydro_reservoir_capacity_twh` | 8.895 | TWh | 2026-06-08 | BFE Speicherseen |
| `hydro_reservoir_fill_ratio` | 0.2506 | ratio | 2026-06-08 | BFE Speicherseen |
| `hydro_capacity_gw` | 17.4973 | GW | 2025-12-31 | BFE WASTA |
| `hydro_twh` | 41.3565 | TWh | 2025-12-31 | BFE WASTA |
| `pumped_storage_power_gw` | 4.0669 | GW | 2025-12-31 | BFE WASTA |
| `pv_gw` | 8.4511 | GW | 2026-06-12 snapshot | BFE Electricity production plants |
| `nuclear_gw` | 3.0146 | GW | 2026-06-12 snapshot | BFE Electricity production plants |
| `wind_gw` | 0.1088 | GW | 2026-06-12 snapshot | BFE Electricity production plants |
| `thermal_capacity_gw` | 0.2987 | GW | 2026-06-12 snapshot | BFE Electricity production plants |

These actuals are now declared as governance source components. They are
official and useful for calibration/actualization, but they are not a governed
2030 scenario path. They must not overwrite published scenario values silently
and they must not flip the Phase 13 production gate to GO.

## Local Canonical Model Wiring

To wire the best available local Phase 13 data into the canonical paths consumed
by the model, materialize the local LT data contract:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONUTF8='1'
python scripts/materialize_lt_data_contract.py `
  --local-root C:\Users\jbattaglia\pfc_local_data `
  --scenario-source data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet `
  --scenario-output data/electrification_scenarios.parquet `
  --features-output data/hpfc_scenario_features.parquet `
  --years 2027-2031 `
  --vintage 2026-06-05 `
  --country CH `
  --scenarios slow,central,fast `
  --report .planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-LOCAL.md
```

This creates the local cache folders under:

```text
C:\Users\jbattaglia\pfc_local_data
```

and writes:

| file | role |
|---|---|
| `data/electrification_scenarios.parquet` | canonical scenario inventory read by `ElectrificationScenarioStore()` when no explicit path is passed |
| `data/hpfc_scenario_features.parquet` | canonical gold feature table derived from the scenario inventory |
| `.planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-LOCAL.md` | local readiness report and external data gap list |

The current local materialization is a proxy bridge, not final production data:
it uses enriched EP2050 `WWB` / `ZERO_Basis` plus an internal midpoint. It is
valid for local smoke/prod-readiness runs with
`require_electrification_scenarios=True`; it is not enough to enable Phase 13 as
final production signal until TYNDP, Pronovo, MaStR, governed NTC and
vintage-safe forecast feeds are loaded.

## Operational Gate

Before enabling `enable_electrification_shape=True` in a production run, validate
the inventory at the pricing vintage:

```powershell
$env:PYTHONPATH='.'
python scripts/validate_electrification_scenarios.py `
  --path data/electrification_scenarios.parquet `
  --vintage 2026-01-15 `
  --country CH `
  --scenarios slow,central,fast `
  --years 2027-2030 `
  --report .planning/phases/13-lt-electrification-scenario-shape/scenario_inventory_validation.md
```

For Databricks-backed validation:

```powershell
$env:PYTHONPATH='.'
python scripts/validate_electrification_scenarios.py `
  --databricks `
  --table-key electrification_scenarios `
  --vintage 2026-01-15 `
  --country CH `
  --scenarios slow,central,fast `
  --years 2027-2030
```

Exit code `0` means the vintage/scenario/year coverage is present. Exit code
`2` means coverage is missing and Phase 13 must remain OFF, or production must
run with `require_electrification_scenarios=True` so the curve build fails fast
instead of silently using the identity correction.
