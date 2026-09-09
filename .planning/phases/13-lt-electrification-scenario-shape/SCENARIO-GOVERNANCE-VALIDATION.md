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
