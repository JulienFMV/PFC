# Session handoff - LSEG vendor known-at DEV code review (2026-08-11)

## Outcome

Audited `FMVSA/lseg-lakehouse` DEV commit
`e3bd19746abd69b98a001d1f365ca56754f16950`, subject
`feat(lseg): add vendor-backed known-at metadata`, against the point-in-time
requirements of D-20260810-298.

The commit is a material source-contract improvement but remains
`NO_GO_PROD` for point-in-time model use. The vendor field is now acquired and
propagated, but value and metadata payloads are independently change-suppressed
and are not associated by a durable source-capture identity.

No Databricks SQL, Warehouse, cluster or job was started by this audit.

## Exact reviewed state

- Local ignored checkout:
  `build/data-engineer-repos/lseg-lakehouse`, clean detached HEAD at
  `e3bd19746abd69b98a001d1f365ca56754f16950`.
- Parent commit: `ebc3f23ff0a7e62e65471d861e4be993f35fdde1`.
- Diff: six files, 521 insertions and 177 deletions.
- Modified contract surfaces: pull, Landing-to-Bronze, Bronze-to-Silver,
  Silver-to-Gold, post-backfill validation and LSEG data-contract docs.

## What is correctly implemented

- Forecast curves call `GET /CurveForecastList/{CurveID}` and parse forecast,
  scenario, min/max value dates and `lastUpdateTime`.
- Actual curves call `GET /CurveSummary/Values/{CurveID}` and parse the curve
  summary `lastUpdateTime`.
- Landing raw JSON remains immutable and Bronze points distinguish
  `VALUE_POINT`, `FORECAST_METADATA` and `CURVE_SUMMARY_METADATA`.
- Silver and Gold vintages expose vendor update time, known-at time, source
  basis and an unknown/fallback flag.
- The vintage business key preserves curve, scenario, forecast origin,
  delivery interval, value and unit; vendor metadata does not manufacture a
  new value version when the value is unchanged.
- Gold exposes the fields on `FactLsegCurveValueVintages`; the latest fact
  remains a non-PIT latest-value surface.

## Blocking functional findings

### P1 - value/metadata association is not atomic

The pull notebook maintains a separate response hash/state file per
`payload_kind` and writes only changed responses. Thus a changed value payload
can be landed without its unchanged metadata response, and a changed metadata
response can be landed without a value payload.

Landing discards the envelope's source pull `run_id` as a parsed field and
assigns its own notebook `_run_id`. Silver then joins forecast metadata on
`curve_id, scenario_id, source_timestamp, _run_id` and curve summaries on
`curve_id, _run_id`. The equality join therefore succeeds only when both
independently suppressed payloads happen to be processed by the same landing
run. Consequences:

- a value correction can incorrectly become `pipeline_fallback` although
  vendor metadata exists;
- a metadata-only update cannot enrich the already stored value vintage;
- a replay or inbox retry can change metadata attribution merely by changing
  landing batch composition.

Required correction: preserve `source_pull_run_id` and land companion payloads
atomically, or build an as-of lookup from all durable Bronze metadata using the
business identity and the latest metadata capture not later than the value
pull.

### P1 - curve summary timestamp is not point-level actual availability

The bundled authenticated Swagger describes
`CurveSummary/Values/{CurveID}` as returning the last update time plus the
minimum and maximum dates available for the curve. It does not claim a
per-value update time. Applying that timestamp to every EPEX historical point
must therefore not be labelled as the time that each delivered version first
became available. Use an explicit FMV first-seen basis for actual points unless
a point-level vendor timestamp is available; retain curve-summary update time
as curve-level lineage only.

### P1 - duplicate-vintage provenance fields can become incoherent

During a full rebuild, duplicate value vintages are collapsed with independent
aggregations: minimum known-at, minimum non-null vendor update, maximum source
rank and a separate availability flag. If an early FMV fallback observation is
followed by a later vendor-attributed observation, the output can combine the
earlier fallback timestamp with the later vendor source label. Select one
coherent evidence struct or expose FMV first-seen and vendor update as separate
columns/bases.

### P1 - validation permits complete forecast fallback

Validation checks non-null known-at values, allowed source labels and internal
fallback-flag consistency, but does not require vendor coverage for forecast
groups and does not enforce vendor/known-at/pull temporal coherence. Add:

- forecast vendor-attribution coverage by group and curve, with an explicit
  blocking threshold;
- vendor source implies non-null vendor timestamp and known-at equal to that
  timestamp;
- vendor update must not be later than FMV pull except for a documented small
  clock tolerance;
- actual curve-summary lineage must not be admitted as point-level known-at.

## Static and CI evidence

- Exact current notebook JSON parsed successfully.
- Read-only Python AST parse: 47 ordinary code cells parsed, zero syntax
  failures, zero magic cells skipped.
- GitHub Actions `31498365811`: success; bundle validation and DEV deployment
  only.
- GitHub Actions `31498365873`: success; YAML lint only.
- No repository test suite was found, and neither workflow executed the
  notebooks or validated produced data.

## Required producer response

1. Correct the source-capture/as-of association and coherent evidence tuple.
2. Separate actual curve-summary lineage from point-level availability.
3. Add the coverage and temporal validation gates.
4. Run a clean DEV rebuild and provide a bounded receipt at the corrected
   commit, including vendor/fallback coverage by group and the temporal-check
   results.

## Files changed in PFC_LT

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`: added
  D-20260811-300.
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260811-LSEG-VENDOR-KNOWN-AT-CODE-REVIEW.md`:
  this handoff.

No model, CT, Power BI or heavy data file was changed.

## Follow-up closure at DEV `5bb89a6`

The DEV branch was refreshed to
`5bb89a6c1a3ff82704877f1daa07ea97da7aeef0`, subject
`fix(lseg): harden vendor known-at provenance`. The six contract files changed
again: 361 insertions and 123 deletions relative to `e3bd1974`.

Direct inspection closes all four prior P1 code findings:

1. The pull collects both value and companion metadata responses for a measure
   and writes both whenever either response changed. It preserves a common
   `source_pull_run_id` and `source_pull_batch_ts_utc`; response-state hashes
   are advanced only after the companion writes complete.
2. Landing carries that identity into raw Bronze and Bronze points. Silver
   prefers an exact source-pull match and otherwise selects the latest matching
   forecast metadata capture not later than the value capture.
3. Vintage deduplication selects one complete row ordered by earliest known-at,
   vendor-evidence rank, pull and ingest. It no longer combines independently
   aggregated timestamp, source and flag fields.
4. Curve-summary metadata is retained only as Bronze lineage for actuals.
   Silver/Gold actual value rows use pipeline fallback, while forecast curves
   require vendor update evidence.
5. Post-backfill validation now requires 100% vendor attribution per expected
   forecast curve, equality of vendor known-at/update, vendor update no later
   than pull plus five minutes, coherent fallback flags, and absence of vendor
   point lineage on actual Gold rows.

Residual non-blocking observation: the Bronze value parser still converts the
LSEG actuals placeholder `forecastDate=2000-01-01` into a non-null Silver
`source_timestamp`, despite updated comments stating actuals remain null.
Gold already converts that exact placeholder to null, and group-aware Gold
validation protects the final modeling surface. The mismatch can distort
Silver diagnostic counts based on `source_timestamp IS NULL`; normalize the
placeholder in Bronze/Silver in a later hygiene change.

Static and CI evidence:

- 47 ordinary notebook code cells parsed with Python AST; zero failures.
- GitHub run `31503992225`: successful bundle validation and DEV deployment.
- GitHub run `31503992147`: successful YAML lint.
- Neither workflow executed the pipeline or inspected business rows.

Final code verdict:
`CODE_GO_PENDING_REAL_DEV_REBUILD_AND_RECEIPT`. The producer must now run the
clean DEV rebuild and committed fail-closed validation. Required evidence is
zero failed checks, 100% vendor coverage for `continuous_forward` and
`pmt_spot_forecast`, coherent timestamp checks, and actual rows remaining
pipeline fallback with no vendor point timestamp.

No Databricks SQL, Warehouse, cluster or job was started by this follow-up.
