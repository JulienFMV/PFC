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
