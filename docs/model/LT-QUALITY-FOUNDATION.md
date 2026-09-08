# CH PFC quality foundation and business scope

D313, 8 September 2026. User-authorized roadmap refinement after D312.

## Active scope

The common PFC requires market repricing, solver monthly conservation, qualified
source chronology, hourly shape evidence and attributable daily revisions.
Business valuation and hydro decision claims require the corresponding real
population and physical constraints. No generic named portfolio is inferred.

**BLOC13 is not an established FMV requirement.** Its references in older
estimand/successor contracts are documentary requirements, not evidence of a
real business contract. It is excluded from the current quality roadmap and
missing-data request unless FMV confirms a use case. FIL/ACC labels likewise
do not authorize an invented mapping from plant or portfolio identifiers.

Frozen earlier scientific contracts and their hashes are retained as historical
evidence. This scope refinement does not bypass their validators or claim a new
scientific admission. Any later admitted successor must implement its explicit
confirmed product scope. All six current authorities remain false.

## Verified source distinctions

The D312 EEX snapshot has 38 current quotes, all on 7 September and traceable to
one raw source key each. Six parent-versus-finer-product conflicts reconstruct
from the original settlements and Swiss delivery hours. Three OFFPEAK alerts
follow from those BASE/PEAK conflicts. A declared half-cent rounding hypothesis
is compatible with these six discrepancies but is not vendor confirmation and
does not accept any conflict. An unsigned, exact-hash hierarchy proposal and
its expected authentication rejection are saved in the D313 review package.

The metadata inventory confirms accessible Databricks candidates including:

| Purpose | Existing source | Qualification still needed |
| --- | --- | --- |
| EEX levels | prd.gold.facteexpricedaily | Expected publication calendar, historical revision/availability evidence and independently approved hierarchy |
| CH hourly prices/fundamentals | prd.gold.factentsoetimeserieslatest, prd.gold.dimentsoeseries | Exact CH series/product, coverage, retained vintages and finalization policy |
| LSEG external curve | prd.gold.factlsegcurvevalueslatest | Actual dated capture; latest is not historical vintage evidence |

Metadata and client-specific coverage diagnostics remain local. They do not
qualify national Swiss demand or generation. The national PFC roadmap in
PFC-CH-AUDIT-ENTRYPOINT-20260908.md takes priority; optional business profile
valuation requires separately confirmed perimeter, units and versions.

## Daily collection command and boundary

From the canonical workspace, with the standard repo-local runtime/cache
environment described in AGENTS.md and the D313 handoff:

```text
python -B -m scripts.collect_lt_benchmark_day --config build/lt-quality-foundation-20260908/collector-config.json --registry build/lt-matched-vintages-20260908/registry --output build/lt-quality-foundation-20260908/daily-YYYYMMDD-attempt01
```

The output must be a fresh build directory. The command uses the actual date;
there is no backdating option. It checks the registry before credentials or
network access, locks a due capture against concurrent runs, reads exactly one
current-day OMPEX workbook, captures one EEX delta and the frozen LSEG source,
then calls the existing D304 daily builder. Missing sources produce a durable
failure; no stale filename fallback is selected. Source bytes and receipts
remain local and all authorities remain false.

Only returned EEX quotation dates replace historical rows. Dates absent from
the response preserve their earlier source lineage; this is not a freshness
pass or proof that the source has no revision on those days. Repeated vendor
issues on successive collection days stay repeated issues, not independent
forecast origins. No provider publication calendar is inferred from weekdays.

Budget: at most six SELECTs, at most 30,000 rows per statement (smaller EEX and
year-sliced LSEG limits), 180 seconds per statement including result retrieval,
600 seconds for one authorized Warehouse start, and configured auto-stop in
(0,45] minutes. Manual stop HTTP403 is not retried. A STARTING/RUNNING receipt
does not prove shutdown. The collector runs once and exits; it installs no
scheduler. Deployment to an owned execution service remains pending, as do the
19 actual future pilot captures. The local workbook path needs visibility in
that service's user context; a mapped drive is not an unattended-access proof.

A leftover .capture.lock after a hard process interruption blocks another run.
Inspect the associated attempt and process ownership before recovery; the
collector never steals a lock or overwrites a failed attempt.

## Economic diagnostic contract

`value_fixed_profile` accepts exactly one identified/versioned profile on an
ordered common native hourly or quarter-hour grid. Interval volumes are finite
nonnegative MWh with a separate GENERATION/CONSUMPTION direction. Profile
availability must precede valuation and delivery must follow valuation. It
retains negative prices and computes full-price EUR capture and signed
cashflows. It rejects missing/duplicate intervals, mixed versions and invalid
chronology. Zero total volume is UNSUPPORTED, not a passed metric.

This numeric function does not authenticate source documents, prove the native
cadence, establish FMV population ownership, calculate BLOC13, or optimize
hydro. It grants no economic admission. CHF valuation needs a separately frozen
FX convention. No capture-premium default is validation evidence.

## Next order of work

D315 recenters the next task on national CH input coverage and availability.
Reuse D300 national sources and existing contracts. Client profile mapping
is optional economic work, not a prerequisite for the next hourly benchmark.
Actual daily capture, truth and independent registration remain separately
tracked. D304, the solver and all model-admission invariants are retained.

Report and missing-data checklist:
`build/lt-quality-foundation-20260908/`.
