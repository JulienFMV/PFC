# Session handoff - ENTSO-E series-to-zone temporal binding

Date: 2026-08-06  
Decision: D-20260806-255  
Scope: offline structure and data-quality gate only

## Outcome

D255 composes the exact D243 series inventory, D253 family mapping and D254
effective-dated zone registry. A non-directional coupled-zone series is now
accepted structurally only when its exact raw `FromZone` or `ToZone` value,
logical family, registry entry and complete validity interval agree. The
binding is content-addressed to all three inputs and becomes stale when any of
them changes.

The gate reports missing signatures and temporal gaps. It never infers a zone
from a display label, never lets one interval cross a code/configuration
change, and never routes directional flows, scheduled exchanges or NTC around
D253's border-and-direction controls.

## Files changed

- `pfc_shaping/validation/entsoe_zone_configuration.py`
  - added exact interval resolution and canonical registry-entry identity;
- `pfc_shaping/validation/entsoe_series_zone_binding.py`
  - new content-bound series/family/zone composition gate;
- `tests/test_entsoe_zone_configuration.py`
  - interval resolution and code-change rejection;
- `tests/test_entsoe_series_zone_binding.py`
  - exact five-zone price coverage, raw-field, input-content, temporal,
    direction, authority and template mutation tests;
- `docs/data/templates/README-ENTSOE-ZONE-BINDING.md`;
- `docs/data/templates/entsoe_zone_configuration_registry.v1.json.template`;
- `docs/data/templates/entsoe_series_zone_binding.v1.json.template`;
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`;
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`;
- `.planning/HANDOFF.md`.

## Verified commands and outputs

All commands ran from the canonical workspace through the repo-local governed
wrapper.

- `entzbind255final`: focused inventory/family/zone/binding/coverage matrix,
  `73 passed in 0.42s`;
- `entzall255a`: every `tests/*entsoe*.py`, `520 passed in 16.08s`;
- `entzruff255final`: Ruff on both validators and both test modules, passes.

Canonical SHA-256:

- zone registry validator:
  `f4a685d54d20c57a1464ab5497c46841c2f91cba3bed45cb76d2c529250ee624`;
- series-zone binding validator:
  `eb1953dc7e44b9b1f8090a33634e66f46bbdce58756ccf6cbc33a4a2178b8dd1`;
- zone tests:
  `92757274c0c9c30b35b0bcd9a3ca46056c1fb20bfa7fc23049e68408029fcb0f`;
- binding tests:
  `bda28238301dbaed074c8b8f76c786d7ad9841675a774b2943bdcbd4e395cf68`;
- template README / registry / binding:
  `edaf09a068267c478fe249c36dbe873d6438b4e85ace39f70038659fb02317c7` /
  `37565aa6f19b569461f44c7af917a66d18d962e2808e4a47bfecce38b524c591` /
  `f218bbf486aecd6f4ec025ddd5275bde729dc85f1c399c70d87ecbeb7d6ee745`.

## Databricks and data-cost receipt

- Databricks requests: 0;
- SQL statements: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- real ENTSO-E value rows opened: 0;
- same-day retry: not attempted and still forbidden.

## Current evidence-based ENTSO-E verdict

The 2026-08-03 `dev.gold` audit observed most core macro-families: actual load,
day-ahead prices, actual/forecast generation, solar/wind forecasts, hydro
storage, physical flows, scheduled exchanges and day/month/year NTC. It did
not freeze enough exact metadata to prove all required series.

Still unproven: separate load forecast, renewable forecast horizon, full
technology split, exact five-zone price coverage, all four CH borders in both
useful directions, effective cadence, explicit `EUR/MWh` metadata mapping,
sign, lineage, quality, revision and PIT availability.

High-value candidates if available: generation and network outages,
installed/available capacity, balancing energy/capacity prices and volumes,
imbalance price/system imbalance, redispatch/countertrading, net positions and
intraday cross-zonal capacity. Second tier: per-unit generation, curtailment,
load forecast margin, flow-based parameters and cross-zonal balancing
capacity. These candidates shape regimes only and cannot rewrite solver
monthly means.

## Risks and next safe action

The templates are intentionally value-free and fail admission until an
accountable owner fills them from the exact D243 inventory and governed EIC
history. Owner assertion still does not prove official registry membership,
real row coverage, predictive value or PIT safety.

On a later reserved Europe/Zurich day, and only if the Warehouse is already
running, capture the bounded D243 inventory once, export it locally with an
immutable manifest, then complete D253/D254/D255 evidence. Do not start a
Warehouse, write to Databricks or retry on 2026-08-06. D260/D261 cadence
regimes remain an independent concurrent gate and must be joined only after
their exact content IDs are frozen.

T057 remains sealed; AFRY remains descriptive; the monthly solver remains the
sole monthly-level authority; LT stays independent from CT.
