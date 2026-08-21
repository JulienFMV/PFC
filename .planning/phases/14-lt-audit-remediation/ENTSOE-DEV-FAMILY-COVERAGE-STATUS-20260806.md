# ENTSO-E dev family coverage status

Date: 2026-08-06  
Decision: D-20260806-247

## Verdict

All ENTSO-E families needed by the CH LT PFC are **not yet proven** in
`dev.gold`. The 2026-08-03 SQL audit proves that most core macro-families exist,
but it does not provide the exact, current `GroupName`/`FieldName` inventory
needed to distinguish every forecast horizon, technology, border, unit and
directional convention.

The reserved 2026-08-06 attempt stopped before SQL because the configured
Warehouse was `STOPPED`, classic `2X-Small`, non-serverless, with 45-minute
auto-stop. It was not started.

## Coverage supported by the 2026-08-03 dev audit

The following macro-families were observed in `dev.gold`:

- actual load;
- day-ahead prices;
- actual generation;
- generation forecasts;
- solar and wind forecasts;
- hydro reservoir storage;
- physical cross-border flows;
- scheduled exchanges;
- day-, month- and year-ahead NTC.

This is positive discovery evidence, not current model admission. In
particular, the prior audit did not freeze a replayable full dimension
inventory.

## Still unproven or ambiguous

- load forecast as a separately identified family;
- day-ahead versus intraday renewable forecasts;
- exact actual/forecast generation split by solar, wind, nuclear, run-of-river,
  reservoir and pumped storage;
- complete border coverage for CH-DE, CH-FR, CH-IT and CH-AT in both relevant
  directions;
- native resolution per `SeriesID`;
- explicit `EUR/MWh` metadata: the business unit is confirmed as `EUR/MWh`,
  but the 2026-08-03 dev dimension labelled it `EUR`; this metadata mismatch
  must be corrected or governed by an explicit mapping;
- positive-direction sign convention for flows and exchanges;
- source endpoint and immutable ENTSO-E document ID;
- source quality/reason and revision number;
- canonical PIT rule from publication, pull and load timestamps.

The 2026-08-06 real schema capture independently confirms that these cadence,
lineage, quality, revision and sign fields are not directly available in the
three raw table schemas.

The exact mapping gate now treats border and direction as separate checks. For
each cross-border family, `CH_TO_NEIGHBOR_ONLY` and
`NEIGHBOR_TO_CH_ONLY` are required on every CH-DE/FR/IT/AT border. Only a
physical-flow series may cover both directions in one series, and only when
the owner explicitly documents its positive sign direction. NTC and scheduled
exchange series cannot claim both directions from one record.

## Additional high-value ENTSO-E families to request

Priority 1 for CH hourly shape and scarcity regimes:

1. planned and unplanned generation unavailability;
2. transmission/network unavailability;
3. installed and available generation capacity by technology;
4. activated balancing energy prices and volumes for aFRR, mFRR and RR,
   separately for upward and downward direction;
5. imbalance price and system imbalance;
6. procured reserve capacity and capacity prices;
7. redispatch, countertrading and congestion actions;
8. net positions and evolving intraday cross-zonal capacity.

Official ENTSO-E documentation also exposes a second exploration tier that was
not in the first checklist and may help explain scarcity or congestion regimes:

- actual generation per unit, where governance and confidentiality permit it;
- available generation capacity, kept distinct from installed capacity;
- generation curtailment and load forecast margin;
- flow-based cross-zonal capacity parameters;
- allocated and used cross-zonal balancing capacity.

These are useful candidate features, not prerequisites for the first governed
CH LT baseline. Their absence must not block the initial model if the core
families above are complete and PIT-safe.

The legacy local archive shows that outage signals, neighbouring actuals, NTC,
physical flows, scheduled exchanges, DE renewable forecasts and FR nuclear
forecasts are potentially useful. It does not prove that those families exist
in current `dev.gold`, and it cannot be substituted for governed real evidence.

## Derived variables after PIT admission

- load, solar, wind and generation forecast errors;
- residual load and residual-load ramps;
- net imports and border congestion indicators;
- generation capacity factors;
- hydro scarcity and flexibility indicators;
- balancing/scarcity regime flags.

These are shape features only. They cannot rewrite monthly solver means.

## Local evidence context

The existing local archive remains `NO_GO` for empirical selection:

- forecast availability timestamps are absent;
- raw provider lineage is not replayable;
- 11 nominally duplicated fundamentals series diverge across views;
- neighbour actuals, physical flows and scheduled exchanges have only about
  55 days in the short local segment;
- local outages contain five series and are promising for future governed use;
- all model, rolling-origin and production authorities remain false.

## Cost receipt for the 2026-08-06 attempt

- local reservation content ID:
  `baaa32ea74e36e13280ea30ced5adbacbe58b35bc92f0758d5e6b66ba29ee770`;
- local reservation marker SHA-256:
  `ff61e1383fd4fdfb225424923dac769181be768c0c39c31a8d3311ecc1b3fb4a`;
- control-plane GETs in this attempt: 1;
- SQL statements: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- ENTSO-E rows opened: 0;
- same-day retry authorized: false.

The next real inventory attempt must wait for another reserved Europe/Zurich
day and an already-running Warehouse.
