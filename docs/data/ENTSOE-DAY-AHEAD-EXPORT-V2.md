# ENTSO-E day-ahead consumer export v2

## Outcome

The corrected interval consumer is fed by two distinct, bounded Silver-vintage
exports.  Neither export changes or supersedes the historical nine-column PIT
query and its hash-bound receipts.

| Contract | Temporal selection | Additional authority required | Consumer meaning |
|---|---|---|---|
| `causal_asof` | latest eligible vintage with `availability_timestamp_utc <= as_of_utc` | original-publication evidence for every `SOURCE_DOCUMENT_CREATED` series; none for `FMV_FIRST_SEEN` | value demonstrably available at the simulated origin |
| `realized_latest_candidate` | latest observed revision inside a frozen assessment cutoff | none | reproducible local snapshot for adapter tests and independent reconciliation; `is_final=false` |
| `realized_final` | latest observed revision inside a frozen assessment cutoff | finality evidence covering the exact delivery window and selected SeriesKeys | ex-post realized truth; never a PIT feature |

Latest-revision ranking is not finality proof.  A returned-document
`createdDateTime` is not original historical day-ahead publication proof.
The candidate lane exists because the producer exposes retained latest
revisions and rolling correction capture, not a signed finality service. It
prevents that missing service from blocking local replay while preserving the
meaning of `realized_final`.

The selected market scope is explicit.  CH and DE-LU may be labelled as the
valuation/hedging scope; FR, AT and IT-North may be labelled as
observation/risk markets or future market candidates.  These labels describe
business purpose and grant no trading or publication authority.  Five-zone
source qualification remains a separate global control, not a requirement
that every bounded consumer export contain every zone.

This scope label is not a claim that both curves have equal implementation or
promotion status.  The current governed monthly solver is CH-only.  The DE
branch requires its own level-authority and promotion evidence before it can be
treated as equivalent to the governed CH candidate.

## Column lineage

| Consumer column | Authority |
|---|---|
| `field_name`, `series_key`, `classification_sequence` | direct Silver |
| `interval_start_utc`, `interval_end_utc`, `native_resolution` | direct producer-normalized Silver interval |
| `price_eur_per_mwh` | direct Silver `field_value`, restricted to A44 EUR/MWh |
| publication, first-seen, historicity and source provenance | direct Silver |
| `market_zone`, `market_timezone` | deterministic, closed mapping from admitted `field_name` |
| `quality_status` | `PASSED` only after exact `dq_failed is False` validation |
| `original_publication_proven` | separate scoped evidence; never inferred from `availability_basis` |
| `is_final` | separate scoped finality evidence; never inferred from revision rank |

Every external evidence item also binds the semantic SHA-256 of the exact raw
export rows it covers.  Matching only the market, window or SeriesKey is
insufficient: any value, interval, revision or provenance change invalidates
the evidence.

`curve_type` is absent from both exports.  The export validates normalized
half-open bounds and resolution; it neither reconstructs nor labels the XML
representation.

## Cost and execution fence

Both SQL templates are read-only, restricted to a bounded delivery window,
an explicit non-empty subset of the five SeriesKeys, at most two enumerated
`_year`/`_month` partitions and a 20,001-row rejection sentinel.  Two UTC
partitions are necessary because one Swiss market month starts and ends at
local midnight and normally straddles UTC month partitions; DST is never
reduced to a 24-hour-day assumption.  `LIMIT` does not cap scanned bytes.
Execution remains stopped until
all of the following are independently true:

1. the Warehouse was already running for another authorized workload;
2. partition pruning is proven;
3. a hard scan upper bound is known and does not exceed the approved ceiling;
4. a human explicitly authorizes the batch.

The local preflight performs no connection, statement, Warehouse start or
business-row read.  A ready preflight is review evidence, not execution
authority.

## Downstream authority

Passing `realized_latest_candidate` validation authorizes only deterministic
local replay and independent reconciliation; its consumer contract authority
and `is_final` flag remain false. Passing either governed consumer export
validation authorizes only construction of the existing day-ahead consumer
frame. No lane grants monthly-level, model-input, model selection, promotion or
production authority. The monthly BASE solver remains the sole level authority;
spot values may become only realized targets, independent controls or
duration-weighted zero-mean shapes after the remaining gates pass.

An exact comparison with another provider's latest-observation curve validates
cross-source consistency, not finality. In particular, matching LSEG latest
values cannot turn a `realized_latest_candidate` into `realized_final`, prove
historical point-in-time availability or cover a zone absent from LSEG.
