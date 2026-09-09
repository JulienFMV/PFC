# Session handoff — LSEG vintage availability gap

Date: 2026-08-07  
Decision: D-20260807-297  
Status: source horizon explained; source availability timestamps incomplete

## Outcome

The data engineer's LSEG UI evidence confirms that the 2028-12-31 horizon is
the vendor's currently delivered boundary for Swiss curve `110181967`. It is
not a pipeline truncation and must not be extended synthetically.

The same evidence exposes a more important PIT gap. LSEG displays separate
`Forecast Date`, `Updated` and `Corrected Date` fields. The current Bronze
parser accepts only `valueDate`, `forecastDate`, `value` and `scenarioID`, and
maps `forecastDate` directly to `source_timestamp` / Gold
`ForecastDateTimeUtc`. It does not persist source update/correction times. One
displayed forecast nominally dated 00:00 was updated only around 03:32, so the
nominal forecast date cannot by itself authorize as-of availability.

## Required source contract

Preserve, without substitution:

- nominal forecast/origin timestamp;
- source updated timestamp;
- nullable source corrected timestamp;
- FMV pull timestamp;
- FMV first-ingest/known-at timestamp;
- source timezone or explicit UTC normalization evidence.

The exact LSEG API fields and timezone must be confirmed by the data engineer.
Do not scrape the UI if the authenticated API or source metadata endpoint
provides them.

## Volume decision

Continuous-forward vintages are expected to dominate storage because every
daily vintage repeats an hourly curve through the vendor horizon. The source
lakehouse may retain its governed history. The PFC workstation will later
export only curve `110181967` and the explicitly selected origins/windows; it
does not need a full 17-curve vintage dump.

## Execution

Only the local ignored repository and the user-provided screenshot were
inspected. Databricks SQL, business-row reads, Warehouse starts and writes were
all zero.

