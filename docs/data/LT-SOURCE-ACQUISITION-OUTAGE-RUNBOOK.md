# LT source acquisition during an ENTSO-E service incident

## Current disposition

The public ENTSO-E incident does not make historical materialized Silver rows
invalid, and it does not prove that those rows are available. It only blocks a
claim that a fresh source pull is currently complete. The local workstation
must not test source recovery by querying Databricks or starting compute. A
later explicitly authorized, bounded construction comparison used an already
running Warehouse; it did not test or establish source recovery or freshness.

The machine-readable disposition is
`.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`.
It grants no execution, model, monthly-level, publication, production or
trading authority.

## Independent lanes

### EEX

Reuse the existing 5 August local capture. Its artifact and manifest hashes
match their recorded identities. Do not issue a replacement statement merely
because ENTSO-E is degraded, and do not switch to the unqualified direct EEX
API route.

The next admissible work is evidence completion around those exact bytes:

1. bind the exact three-table join SQL, predicates and physical table
   identities;
2. obtain independent source-time and signed-envelope evidence;
3. convert the causal daily snapshots into the existing signed EEX vintage
   catalogue rather than creating a second authority.

Local status on 2026-09-03: item 1 is complete under D-20260903-280. The
zero-query validator now binds the exact reviewed SQL bytes and SHA-256 to the
exact historical manifest, three ordered PRD tables, CH/POWER predicates,
12-column result schema and opaque artifact declaration. It does not open the
price artifact. Items 2 and 3 remain external/governed work and all model,
selection and production authorities remain false.

Until then the snapshot is useful local evidence, not a model input.

### ENTSO-E

Use only rows already materialized before the incident. On 2026-09-03 the PBI
SQL Warehouse was independently observed `RUNNING`; one explicitly authorized
bounded aggregate comparison ran without starting, resizing or creating
compute. This plan still authorizes no Warehouse start.

The first request is deliberately a July 2026 `realized_final` smoke export.
It exercises the v2 adapter and the LSEG reconciliation on a complete Swiss
market month without pretending that the August backfill was known in July.
Before values are delivered, the producer or data owner must provide the
effective-dated AT and DE-LU series selections. The export must use the exact
v2 realized SQL hash recorded in the plan and carry value-bound finality
evidence.

Local status on 2026-09-03: the metadata-only owner request is frozen at
`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-REQUEST-V1-20260903.json`
with canonical JSON SHA-256
`6794566035e40b69eb5d104d09e317896b69502ca800513ef094ca46673d6cfb`.
It covers exactly the July Swiss-local window and the two admitted
classification candidates for each of AT and DE-LU. It requests no business
values. It has not been transmitted, no owner response has been received and
the request itself authorizes no selection. Defaults, averaging and consumer
inference remain forbidden. Each selected key must cover the whole request
window; an in-window classification change blocks this request and requires a
separately reviewed segmented-export plan.

The construction-only selection question was subsequently resolved without an
owner response. The deployed producer code proves that sequences 1 and 2 are
distinct A44 auction identities with no implicit default. A bounded comparison
against the independently configured LSEG EPEX day-ahead curves then found:

- AT sequence 1: exact equality for all 2,976 July quarter-hours; sequence 2
  MAE `11.299412 EUR/MWh`;
- DE-LU sequence 1: exact equality for all 2,976 July quarter-hours; sequence 2
  MAE `9.549943 EUR/MWh`.

Therefore the July construction smoke export uses
`day_ahead_prices||at_price||1` and
`day_ahead_prices||de_lu_price||1`. The aggregate query returned no raw price
rows, but it read 9,364,142,086 bytes; do not repeat it. Its evidence is frozen
in
`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-EVIDENCE-V1-20260903.json`.
This parity selects the construction reference only. It does not prove source
publication time, realized finality, causal availability, model input or
production authority.

The 4 September export preflight is frozen in
`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-REALIZED-FINAL-EXPORT-PREFLIGHT-V1-20260904.json`.
It binds the exact five selected SeriesKeys, July Swiss-local window, two UTC
partitions and realized-export SQL hash. Three control-plane metadata GETs
opened no business rows. They observed the PBI SQL Warehouse `STOPPED` and
confirmed that the managed Delta source is partitioned by `_year` and
`_month`, but exposed neither a current byte size nor a file count. Therefore
the physical scan upper bound remains unproven and SQL execution is stopped.
The earlier 47.9 MB value-blind profile is a useful reference, not a hard bound
for this two-partition, wider export. No scan or cost ceiling has been approved.

The next operator must obtain a current hard scan upper bound or platform
export quote and observe the Warehouse already running for another authorized
workload. Only then may a human review the exact ceiling. The operator must not
start, resize or create compute and must not repeat statement
`01f1a79c-0849-1281-af0c-ee155e8346ce`.

The quote/contract request is now open as private producer-repository issue
`FMVSA/opendata-lakehouse#4`. It requests the hard scan bound or platform
quote, DBU/cloud-cost ceiling, output terms, assessment-cutoff rule and
value-bound finality contract. The issue is coordination only: it authorizes
no SQL and no Warehouse start. Wait for a complete response before cost review
or export execution.

After explicit user authorization on 4 September, a value-blind freshness and
cost check used the PBI Warehouse once it was already `STARTING`; this client
issued no start request. `DESCRIBE DETAIL` proves that Silver was modified at
`2026-09-04T06:42:56Z` and currently contains 137 files totalling
1,548,216,106 bytes. The bounded August/September day-ahead watermark query
read 7,798,287 bytes from two files in 3.7 seconds and returned no prices.

AT sequence 1, DE-LU sequence 1, FR and IT-North cover delivery through
`2026-09-04T22:00:00Z`; CH stops at `2026-09-02T22:00:00Z`. All five report
zero `dq_failed` rows in the bounded check. Thus the table is physically
updated, but current CH completeness is not proven and a two-day delivery gap
is visible, consistent with the recent source outage. The exact evidence is
frozen in
`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-FRESHNESS-COST-CHECK-V1-20260904.json`.

The Warehouse had no active sessions or queries after the check. The client
lacks permission to stop it: one explicit stop request returned HTTP 403, and
the Warehouse remained `RUNNING` with its 45-minute auto-stop. This failure
must not be retried; a platform owner may stop it sooner. The findings were
posted to issue `FMVSA/opendata-lakehouse#4` without authorizing the July
export at that time.

The user then explicitly authorized advancing the historical July work while
the September source recovery remains pending. A normalized, value-blind
coverage query over the exact Swiss-local July window and frozen SeriesKeys
read 65,771,005 bytes. Each of CH, AT sequence 1, DE-LU sequence 1, FR and
IT-North expands to exactly 2,976 quarter-hours with zero missing,
overlapping or invalid native intervals. This proves that the September CH gap
does not affect the already-materialized July candidate.

One bounded v2 latest-revision statement then read 31,967,426 bytes and
returned 12,083 native rows through one 6,653,296-byte Arrow result chunk. The
result is quarantined below
`build/entsoe-july-candidate-20260904/latest-revision-candidate.parquet`; it is
564,727 bytes with SHA-256
`046ae86ab84a72c61ea44547cc386f85e49d37d325352f4bfca208a08e5a9baa`
and semantic SHA-256
`5a6d72b5531e843dc4e7920c519cc3d93189e70d91277662a763f6575967374e`.
No value was printed, transmitted to GitHub or committed.

Local validation accepts the exact SQL binding, five SeriesKeys, window,
intervals, quality and raw-frame contract, then fails closed only on
`evidence does not exactly cover selected SeriesKeys`. Therefore this is a
latest-revision candidate, not yet `realized_final`. The exact non-value
evidence is frozen in
`.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-JULY-CANDIDATE-CAPTURE-V1-20260904.json`.

A subsequent producer-contract audit confirmed that the source intentionally
retains the latest observed revision and uses a seven-day rolling lookback to
capture corrections. It exposes no signed finality service. Waiting for such a
receipt is therefore not a prerequisite for local adapter or reconciliation
work. The explicit `realized_latest_candidate` lane now validates and replays
the existing bytes while keeping `is_final=false`, consumer authority false and
all model/publication/production authorities false.

The self-contained replay is frozen below
`build/entsoe-july-candidate-20260904/latest-revision-candidate-replay/` with
build ID
`b00f2b725c66d91e7b8ec681b578fd080be77837a6e75f9b1de675caade20a26`.
It replays all 12,083 rows exactly without Databricks. Do not rerun the export.
Issue 4 may still provide a finality assertion or source correction status, but
only that optional evidence could promote the same hash-bound bytes to
`realized_final`; its absence no longer blocks local replay or independent
reconciliation.

If the internal materialized Silver snapshot is unavailable, stop. Wait for
producer recovery; do not replace ENTSO-E with legacy local or synthetic data.
If the snapshot is available, the public outage still forbids any claim of
freshness beyond its documented watermark.

### Causal history and future holdout

The July rebuild is a backfill. It cannot become historical causal truth.
`causal_asof` resumes only with a genuinely prospective capture whose
availability is known at each frozen origin. Freeze the origin schedule and a
new independent future holdout before collecting that evidence or retraining
anything.

### LSEG

The matched July reconciliation is complete. Its thresholds were frozen before
opening the business values: 744 matched hours and full overlap per zone, with
maximum p95, absolute bias and single-hour difference of `0.005 EUR/MWh`.
CH, AT sequence 1, DE-LU sequence 1 and FR each matched the LSEG EPEX latest
curve for all 744 hours with zero missing hours and exactly zero difference.
IT-North remains complete in ENTSO-E but has no active LSEG EPEX cross-check.

The preflight rejected the unpartitioned 11,544,235,073-byte LSEG vintage table
without querying it. The bounded read instead used
`prd.silver.ge_market_lseg_curve_values`, whose catalog size was 16,758,198
bytes, below the frozen 32 MiB ceiling. The successful statement read
13,141,107 bytes, returned 9,672 native rows and wrote nothing remotely. PBI
was already running; no Warehouse was started, resized or created. Do not
repeat the earlier 9.36 GB disambiguation query.

This result validates latest-source consistency only. The LSEG extract is not
point-in-time evidence, it does not prove ENTSO-E finality or original
publication time, and it grants no model, monthly-level, publication or
production authority. Any future discrepancy blocks; it never licenses silent
source substitution. The frozen evidence is
`.planning/phases/14-lt-audit-remediation/LSEG-ENTSOE-JULY-LATEST-RECONCILIATION-V1-20260904.json`.

## Incident recovery check

On recovery, record the provider notice and first complete post-recovery
watermark separately from the historical materialized snapshot. A public
status page, successful HTTP response or absence of a banner does not by
itself prove source completeness, original publication time or finality.

All later products retain the existing authority boundary: the CH monthly
BASE solver remains the sole monthly-level authority, ENTSO-E/LSEG may only
provide realized truth, controls or zero-mean shape, LT remains independent
from CT, and T057 remains sealed.
