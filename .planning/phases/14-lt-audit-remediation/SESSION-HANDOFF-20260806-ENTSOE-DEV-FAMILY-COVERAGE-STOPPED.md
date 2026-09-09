# Session handoff - ENTSO-E dev family coverage stopped safely

Date: 2026-08-06  
Decision: D-20260806-247

## Outcome

The user requested a real confirmation of all ENTSO-E data types in dev. The
D246 local reservation was created first and consumed the Europe/Zurich day.
One control-plane GET then showed the configured Warehouse as `STOPPED`,
classic `2X-Small`, non-serverless, auto-stop 45 minutes. No SQL was submitted
and the Warehouse was not started.

The durable coverage report therefore reconciles only already available real
evidence:

- 2026-08-03 SQL audit: core macro-families found;
- 2026-08-06 D241 control-plane schema capture: exact 11/8/10 raw columns and
  missing normalization semantics;
- 2026-08-05 legacy local readiness: potentially useful outage, forecast,
  neighbour, NTC, flow and exchange signals, but explicit `NO_GO` for empirical
  selection.

## Coverage conclusion

Observed in dev on 2026-08-03: actual load, day-ahead prices, actual and
forecast generation, solar/wind forecasts, reservoir storage, physical flows,
scheduled exchanges and day/month/year-ahead NTC.

Not yet proven exactly: load forecast, renewable day-ahead versus intraday,
technology breakdown, all CH borders/directions, native cadence, dev metadata
consistent with the confirmed `EUR/MWh` business unit, signs,
document/endpoint lineage, quality, revisions and canonical PIT availability.

Additional high-value families: generation/network outages, installed and
available capacity, aFRR/mFRR/RR prices and volumes up/down, imbalance price
and system balance, reserve capacity/prices, redispatch/countertrading, net
positions and intraday cross-zonal capacity.

## Cost and execution receipt

- reservation content ID:
  `baaa32ea74e36e13280ea30ced5adbacbe58b35bc92f0758d5e6b66ba29ee770`;
- marker SHA-256:
  `ff61e1383fd4fdfb225424923dac769181be768c0c39c31a8d3311ecc1b3fb4a`;
- control-plane GETs in this attempt: 1;
- SQL statements: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- rows opened: 0;
- same-day retry: forbidden.

## Changed files

- `build/databricks-control-plane/entsoe-daily-reservations/2026-08-06.json`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DEV-FAMILY-COVERAGE-STATUS-20260806.md`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DATA-ENGINEER-GAPS-20260805.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

## Next safe action

No further Databricks attempt on 2026-08-06. On a later reserved day, execute
the exact D243 dimension query only if the Warehouse is already `RUNNING`.
Then map raw labels with the data owner and classify every required/additional
family as present, absent or ambiguous before fact-value profiling.

Predecessor:
`SESSION-HANDOFF-20260806-ENTSOE-LOCAL-DAILY-CAPTURE-GUARD.md`.
