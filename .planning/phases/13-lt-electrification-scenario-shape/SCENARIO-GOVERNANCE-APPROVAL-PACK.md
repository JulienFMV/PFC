# Scenario Governance Approval Pack

## Decision Status

* decision_id: `phase13_lt_ch_2030_slow_central_fast_v0`
* manifest_status: `draft`
* approved_for_production: `False`
* approval_date: `None`
* approved_by: `[]`
* pack_recommendation: `NO-GO`

### Blocking Conditions

| blocker |
| --- |
| manifest status is not approved |
| approved_for_production is not true |
| approval_date is missing |
| approved_by is empty |
| 18 data/governance blockers remain in the gap register |

## Scenario Scope

```yaml
countries:
- CH
- DE
- FR
- IT
- AT
scenarios:
- slow
- central
- fast
years:
- 2030
```

## Scenario Weights

| scenario | weight |
| --- | --- |
| slow | 0.25 |
| central | 0.5 |
| fast | 0.25 |

## Source Components

| source_id | name | local_path | publication_date | role | status |
| --- | --- | --- | --- | --- | --- |
| tyndp2024_supply | ENTSO-E/ENTSOG TYNDP 2024 Supply Inputs | data/electrification_scenarios_tyndp2024_supply.parquet | 2024-05-31 | multi-country PV, wind, nuclear, battery energy, fuel and CO2 component | official_partial |
| tyndp2024_neighbor_demand_bridge | TYNDP 2024 Demand REF2019-to-2040 bridge | data/electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet | 2026-06-11 | non-production neighbour demand bridge for 2030 | internal_proxy_partial |
| ch_ep2050_enriched | CH EP2050 enriched slow/central/fast local mapping | data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet | 2026-06-05 | CH demand, hydro, heat-pump, EV and battery-power overlay | local_proxy_partial |
| internal_p0_structural_bridge | Internal P0 structural bridge | data/electrification_scenarios_composed_p0_bridge_2030.parquet | 2026-06-11 | non-production bridge for peak/winter, PV/wind energy, battery power, EV/PAC and dispatchable lower bound | internal_proxy_partial |
| swissgrid_ntc_2026_baseline | Swissgrid 2026 cross-border NTC baseline | data/electrification_scenarios_swissgrid_ntc_baseline_2026.parquet | 2026-06-08 | official observed baseline for CH-AT/DE/FR/IT NTC fields; not a governed 2030 expansion assumption | official_observed_baseline_proxy |
| ember_yearly_2026_baseline | Ember yearly electricity data baseline | data/electrification_scenarios_ember_yearly_baseline_2026.parquet | 2026-04-24 | official historical baseline for hydro, net imports and dispatchable thermal/nuclear capacity; not a governed 2030 scenario path | official_historical_baseline_proxy |
| lt_field_neutralization_policy_20260612 | LT field neutralization policy | data/electrification_scenarios_prod_candidate_neutralized_2030.parquet | 2026-06-12 | explicit zero neutralization for P1 flex fields and missing coal capacity with field-level justifications | governed_neutralization_draft |
| bfe_energiedashboard_daily_20260612 | BFE Energiedashboard daily actuals | data/bfe_energiedashboard_daily.parquet | 2026-06-12 | official CH daily actuals for production, consumption, cross-border gross flows, net imports and day-ahead base price; historical calibration only | official_actual_not_lt_scenario |
| bfe_speicherseen_weekly_20260612 | BFE weekly Swiss storage-lake reservoir content | data/bfe_hydro_reservoir_weekly.parquet | 2026-06-12 | official CH weekly hydro reservoir actuals for calibration and scenario actualization; not a governed 2030 reservoir path | official_actual_not_lt_scenario |
| bfe_electricity_production_plants_20260612 | BFE electricity production plants in operation | data/bfe_ch_installed_capacity_actuals.parquet | 2026-06-12 | official CH installed capacity actuals by technology for calibration and baseline actualization; not a governed 2030 capacity path | official_actual_not_lt_scenario |
| bfe_wasta_hydropower_plants_20260612 | BFE WASTA hydropower plant statistics | data/bfe_wasta_hydro_summary.parquet | 2026-06-12 | official CH hydropower capacity, expected production and pumped-storage actuals for calibration and baseline actualization | official_actual_not_lt_scenario |

## Gap Summary

| priority | family | gap_count |
| --- | --- | --- |
| P0 | data_quality | 15 |
| P0 | governance_decision | 3 |

## Field-Level Required Actions

| priority | field | owner | source_candidate | required_action | missing_rows |
| --- | --- | --- | --- | --- | --- |
| P0 | quality_flag | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | 0 |
| P0 | quality_flag | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | 0 |

## Expert Reviews

# Scenario Governance Expert Reviews

These reviews are advisory controls. AI agents and automated validators are
independent reviewers and control executors; they are not production approvers.
Production approval requires accountable human sign-off from Model Owner, Data
Owner, Market Risk and independent Model Validation.

## Quant Scenario Reviewer

Verdict: `NO-GO production`, `OK diagnostic/smoke/fan-chart non-prod`.

Findings:

* `slow/central/fast` is acceptable only with explicit `proxy/non-prod` labelling.
* The mapping mixes TYNDP LOW/BE/HIGH, a DE/GA neighbour-demand bridge and a
  local CH EP2050 overlay.
* CH `central` is an internal midpoint, not a published official trajectory.
* `0.25 / 0.50 / 0.25` is defendable only as a diagnostic symmetric prior, not
  as a market-probability weighting for expected PFC valuation.
* The fan chart must remain distinct from the expected PFC until weights are
  approved as probabilistic or clearly labelled as non-probabilistic stress
  weights.

Minimum conditions for recommendation:

* approved manifest with dated human approvers;
* documented physical/economic scenario narrative;
* no `partial/proxy/internal` source flags in production rows;
* governed NTC, peak/winter demand, PV/wind energy, dispatchable/flex, hydro and
  cross-border balance;
* green `validate_scenario_governance.py` gate.

## Data Engineering / Vintage-Safety Reviewer

Verdict: `NO-GO production`, gate design is directionally correct.

Required hardening:

* reject absent production columns in the governance gate;
* exclude `track=actual` rows when `measurement_date > vintage`;
* add explicit provenance for zeros or source-level field provenance before
  allowing critical zero values;
* link inventory rows to governed `source_components`, for example with
  `source_id` or `component_ids`;
* anchor manifest `local_path` resolution to the repo root;
* replace text-token quality flags with a strict enum before final production;
* add economic/plausibility bounds for PV/wind load factors, NTC, battery
  duration and flex fields.

Priority data order:

1. NTC CH-DE/FR/IT/AT.
2. Peak load and winter demand for neighbours.
3. PV TWh and wind TWh.
4. Import/export/net-import balance.
5. Hydro energy, capacity and reservoir assumptions.
6. Dispatchable/flex and thermal capacity.
7. EV/PAC neighbours after the core CH-shape drivers are governed.

## Model-Risk / FM Validation Reviewer

Verdict: `NO-GO production`, `OK to continue controlled research behind flag OFF`.

Findings:

* Agents can be reviewers, not production approvers.
* The current manifest is `draft`, `approved_for_production: false`, with no
  approval date and no accountable approvers.
* Production approval needs Model Owner, Data Owner, Market Risk and independent
  Model Validation sign-off.
* Each source needs an evidence pack: raw file, version, publication date,
  checksum, transformation, owner and usage/licence status.
* If a field remains absent, the only acceptable production path is an explicit,
  bounded and approved neutralisation; never a silent zero.

## Governance Validation Report

# Scenario Governance Validation

* manifest: `.planning\phases\13-lt-electrification-scenario-shape\SCENARIO-GOVERNANCE-MANIFEST.yaml`
* inventory: `data\electrification_scenarios_prod_candidate_neutralized_2030.parquet`
* vintage: `2026-06-12`
* mode: `production`
* governance gate: `FAILED`
* effective rows: `15`

## Issues

### manifest_not_approved

manifest status must be approved and approved_for_production must be true

_none_

### missing_approval_date

approval_date is required

_none_

### missing_approvers

approved_by must contain at least one approver

_none_

### unacceptable_quality_flags

proxy/partial/internal/fallback/synthetic flags are not production-governed

| country | scenario | delivery_year | quality_flag |
| --- | --- | --- | --- |
| CH | central | 2030 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| CH | fast | 2030 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| CH | slow | 2030 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| AT | central | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| AT | fast | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| AT | slow | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| DE | central | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| DE | fast | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| DE | slow | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| FR | central | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| FR | fast | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| FR | slow | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| IT | central | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| IT | fast | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| IT | slow | 2030 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |

## Effective Inventory Rows

| country | scenario | delivery_year | publication_date | quality_flag |
| --- | --- | --- | --- | --- |
| CH | central | 2030 | 2026-06-05 00:00:00+00:00 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| CH | fast | 2030 | 2026-06-05 00:00:00+00:00 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| CH | slow | 2030 | 2026-06-05 00:00:00+00:00 | official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| AT | central | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| AT | fast | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| AT | slow | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| DE | central | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| DE | fast | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| DE | slow | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| FR | central | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| FR | fast | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| FR | slow | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| IT | central | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| IT | fast | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
| IT | slow | 2030 | 2026-06-11 00:00:00+00:00 | official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit |
