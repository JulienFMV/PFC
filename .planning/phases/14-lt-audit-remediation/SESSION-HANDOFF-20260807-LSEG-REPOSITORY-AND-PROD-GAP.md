# Session handoff — LSEG repository and PROD gap

Date: 2026-08-07  
Decision: D-20260807-296  
Status: Swiss LT HPFC confirmed in DEV source; PROD release absent

## Outcome

`FMVSA/lseg-lakehouse` confirms that measure `110181967` is the Swiss
`ContPwrPriceForward.forward.Price`: `CHE`, hourly values, daily issues,
EUR/MWh, Point Carbon, scenario 0, with a requested four-year value horizon.
This is a valid independent vendor HPFC benchmark candidate.

It is absent from `prd` because the release path has not been completed:

- `dev` head: `ebc3f23ff0a7e62e65471d861e4be993f35fdde1`;
- `main`: `89abd94cd50e7827dd4094d5a0944cfebfa864d8`, 44 commits behind;
- only PR #1 exists and it merged into `dev`;
- no `dev -> main` PR, release tag, staging deployment or PROD deployment
  exists;
- the successful 2026-08-05 workflow deployed DEV only.

## Benchmark interpretation

- `continuous_forward/CHE` is the LT HPFC benchmark.
- `pmt_spot_forecast/CHE` is a separate 16-day CT forecast.
- `epex_actuals` are scoring truth, not an LT forecast.
- `ForecastDateTimeUtc` is the LSEG forecast publication timestamp.
- `KnownAtTimestampUtc` is local first curated availability derived from
  Bronze ingest/pull time. It begins only 2026-06-15, so earlier source
  vintages are vendor backfill rather than historical FMV receipt evidence.
- DEV content currently ends 2028-12-31 despite a four-year request window;
  this gap must be explained before full N+3 scoring.

Use LSEG first as an independent benchmark. If later used as a teacher/input,
reserve disjoint origins/targets for independent scoring. It never overrides
hard CH EEX monthly constraints or the solver's monthly-level authority.

## Data-engineer action

Follow the repository's existing path: PR `dev -> main`, stage and run the
post-backfill validation, create a `vX.Y.Z` tag from `main`, deploy PROD, then
run/validate the governed PROD backfill. Do not ask the PFC workstation to
perform that deployment.

## Execution

- Repository cloned under ignored `build/data-engineer-repos/lseg-lakehouse`.
- GitHub DEV deploy at current head: successful.
- Databricks cumulative LSEG analysis: 28 control-plane GETs, zero SQL, zero
  business-row reads, zero Warehouse starts, zero writes.
- No CT, Power BI, AFRY, OMPEX, T057, `H:` or heavy desk-data file was opened
  or changed.

