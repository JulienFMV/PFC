# LT Neutralization Policy

* source_id: `lt_field_neutralization_policy_20260612`
* publication_date: `2026-06-12 00:00:00+00:00`
* input: `data\electrification_scenarios_composed_p0_public_sources_2030.parquet`
* output: `data\electrification_scenarios_prod_candidate_neutralized_2030.parquet`
* hpfc features: `data\hpfc_scenario_features_prod_candidate_neutralized_2030.parquet`
* audit: `data\electrification_scenarios_neutralization_audit_2030.csv`

## Scope

This artefact closes numeric P1 gaps through explicit, auditable zero neutralisations. It does not approve the scenario inventory for production and does not remove proxy/partial quality flags.

## Neutralized Fields

| field | neutralized_rows |
| --- | --- |
| coal_gw | 6 |
| dsm_gw | 15 |
| electrolysis_twh | 15 |
| managed_charging_share | 15 |
| p2x_gw | 15 |
