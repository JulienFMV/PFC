# LT Scenario Data Gap Register

* inventory: `data\electrification_scenarios_prod_candidate_neutralized_2030.parquet`
* manifest: `.planning\phases\13-lt-electrification-scenario-shape\SCENARIO-GOVERNANCE-MANIFEST.yaml`
* vintage: `2026-06-12`
* blocking gaps: `18`

## Gap Summary

| priority | family | gap_count |
| --- | --- | --- |
| P0 | data_quality | 15 |
| P0 | governance_decision | 3 |

## Field-Level Actions

| priority | field | owner | source_candidate | required_action | missing_rows |
| --- | --- | --- | --- | --- | --- |
| P0 | quality_flag | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | 0 |
| P0 | quality_flag | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | 0 |

## Full Register

| priority | issue | family | field | missing_rows | effective_rows | country | scenario | delivery_year | owner | source_candidate | required_action | prod_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | manifest_not_approved | governance_decision |  | 0 | 15 |  |  |  | risk_committee | scenario governance manifest | manifest status must be approved and approved_for_production must be true | BLOCKING |
| P0 | missing_approval_date | governance_decision |  | 0 | 15 |  |  |  | risk_committee | scenario governance manifest | approval_date is required | BLOCKING |
| P0 | missing_approvers | governance_decision |  | 0 | 15 |  |  |  | risk_committee | scenario governance manifest | approved_by must contain at least one approver | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | CH | central | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | CH | fast | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | CH | slow | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_ch_ep2050_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | AT | central | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | AT | fast | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | AT | slow | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | DE | central | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | DE | fast | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | DE | slow | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | FR | central | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | FR | fast | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | FR | slow | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | IT | central | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | IT | fast | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
| P0 | unacceptable_quality_flags | data_quality | quality_flag | 0 | 15 | IT | slow | 2030 | scenario_data | official governed scenario component | Replace or approve source so quality_flag is production-governed: official_component_partial_neighbor_demand_proxy_p0_structural_bridge_proxy_swissgrid_ntc_baseline_proxy_ember_yearly_baseline_proxy_neutralized_explicit | BLOCKING |
