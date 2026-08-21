# Session handoff - ENTSO-E DEV code review (2026-08-11)

## Outcome

Reviewed the new `FMVSA/opendata-lakehouse` DEV commit
`8549319ff944bfcf2e8123b05907ac025a8f23b8` against the blockers recorded in
D-20260810-298. The change is a material partial remediation but remains
`NO_GO_PROD` pending an explicit existing-table migration/rebuild and an
executable DEV validation receipt. `FMVSA/lseg-lakehouse` DEV is unchanged at
`ebc3f23ff0a7e62e65471d861e4be993f35fdde1`; its prior blockers remain.

No Databricks SQL statement was submitted, no Warehouse or job was started,
and no Databricks write or cost-generating validation was performed.

## Exact repository state

- `build/data-engineer-repos/opendata-lakehouse`: detached clean review of
  `8549319ff944bfcf2e8123b05907ac025a8f23b8`, commit subject
  `fix: preserve entsoe series identity and interval semantics`.
- Diff from the prior audited DEV commit `55c3c436271d320b771d8a126b284249a878abdc`:
  six files, 299 insertions and 60 deletions.
- `build/data-engineer-repos/lseg-lakehouse`: `origin/dev` unchanged at
  `ebc3f23ff0a7e62e65471d861e4be993f35fdde1`.
- No PR exists for `FMVSA/opendata-lakehouse`; the new commit is not contained
  in a local tag.

## What the ENTSO-E commit fixes

- Parses `interval_start_utc` and `interval_end_utc` and retains native
  `resolution`; `DateTimeUtc` remains the documented right edge.
- Propagates `series_key` and `source_time_series_id` to Silver and Gold.
- Adds first/last publication, pull and ingest timestamps to Gold vintages.
- Extends dense validation defaults to physical flows and day/week/month/year
  NTC groups.
- Updates hash/parity and null/duplicate validation for the new contract.

## Blocking findings

1. Existing Bronze points tables are not explicitly ALTERed for the interval
   columns. Existing Silver tables are not ALTERed for `series_key`, source
   TimeSeries ID, interval edges or resolution. Existing Gold dimensions are
   not ALTERed for `SeriesKey` and `SourceTimeSeriesId`.
2. `mode=full` and `overwrite_silver` are independent. The notebook defaults
   the latter to false and the bundle job does not pass it, so a standard full
   job still MERGEs Silver. Old vintage rows are not updated with the new grain
   fields, and changed keys can coexist with legacy rows.
3. Auto Loader appends points and retains its checkpoint. A full source pull
   does not by itself prove replay of already processed files. A governed raw
   replay/table rebuild is required.
4. Entity-grain fallback `series_index:N` is positional, not a durable source
   identity. The parser also does not preserve registered-resource/asset mRID
   or EIC separately for outage/per-unit analysis.
5. The validation notebook still defaults `fail_on_error=false`; no final run
   receipt with zero failed checks, commit, run IDs and Delta versions exists.
6. GitHub `Deploy DEV` only validated and deployed the bundle. `CI - Validate
   YAML` only linted YAML. Neither workflow ran the pipeline or Python tests.
7. `python -m pytest tests/test_entsoe_retry_classifier.py` fails during
   collection with `AttributeError: pyspark.sql.types.StructType`. No tests
   were added in the remediation commit.

## Verification performed

- GitHub runs `31473553752` and `31473553771`: both successful, but their step
  inventories confirm bundle deployment and YAML lint only.
- `python -m py_compile src/entsoe_pipeline.py src/entsoe_validation.py`:
  passed.
- Independent parser smoke test with a small in-memory XML payload: passed for
  PT15M and PT60M, three points, explicit interval starts/ends and right-edge
  timestamps.
- The existing pytest module: failed at collection as described above.

All mutable test paths were kept below
`build/tmp/opendata-code-review-20260811`.

## Files changed in PFC_LT

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`: added D-20260811-299.
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260811-ENTSOE-DEV-CODE-REVIEW.md`: this handoff.

No model code, CT code, Power BI file or heavy local data file was changed.

## Required next producer evidence

1. Commit an explicit existing-table migration/rebuild path, including raw XML
   replay and Silver/Gold overwrite semantics.
2. Replace positional identity fallback for entity families and retain the
   physical resource identifier.
3. Repair/add unit tests for interval semantics, multi-TimeSeries preservation,
   migration and vintage selection.
4. Execute the DEV rebuild and final validation with `fail_on_error=true`, then
   provide the bounded receipt before PR/staging/PROD review.

## Functional-only re-review requested by the producer

On a later 2026-08-11 refresh, `origin/dev` was still exactly
`8549319ff944bfcf2e8123b05907ac025a8f23b8`; no newer commit existed. The
functional vintage design was re-reviewed independently of rebuild, test and
deployment concerns.

- For ordinary numeric series, the design is sound prospectively: the
  historical pull supplies the currently exposed value, a changed value gets
  a new semantic vintage, and an unchanged value extends first/last-seen pull
  and ingest timestamps.
- No ENTSO-E extraction request parameter for historical revision/as-of time
  was found in the official extraction guide. Historical backfill must
  therefore be labelled as one as-retrieved state, not reconstructed PIT
  history; only daily capture from its start date is governed prospective PIT
  evidence.
- Outages remain the functional exception. The parser does not retain
  `docStatus`, Reason, production/generating-unit identifiers or
  `Asset_RegisteredResource`; all three outage requests filter
  `docStatus=A05`. A cancellation/withdrawal or asset/status-only revision can
  consequently disappear from the active response without creating a
  tombstone or distinct semantic vintage, leaving a stale active latest row.
- Required functional correction: preserve document mRID/revision/status and
  outage resource associations, ingest active/cancelled/withdrawn states (or
  reconcile disappearance explicitly), and include status/resource identity
  in outage version semantics. A same-value revision need not create a new
  value vintage for ordinary series, but it must remain auditable and must
  change outage state when its status or resource association changes.

## Superseding functional review of DEV `a44bab0` and `6bf9bb9`

The preceding outage paragraph is superseded by the two later DEV commits:

- `a44bab0402af66fc718f6ac22fa59bd05b876ae1`, `fix(entsoe): preserve
  outage lineage and durable keys`;
- `6bf9bb9581d8a66d9a27c8b2cd1acf892837a1fa`, `fix(entsoe): classify
  availability timing semantics`.

The local review checkout was refreshed explicitly from `refs/heads/dev` and
was clean and detached at `6bf9bb9`. These commits remove the `doc_status=A05`
filters from the three outage groups, preserve document status, reasons and
resource details, expose `BridgeEntsoeSeriesResources`, and make availability
basis/knowledge/timestamp explicit. The former findings that those fields were
missing are therefore closed. A stale active outage is now a residual source
behaviour risk only if the API silently removes an outage instead of returning
its cancellation/withdrawal state; it is not a demonstrated defect by
construction.

The remaining functional outage gap has two linked parts:

1. `outage_vintage_marker_expr()` and the vintage hash do not include a
   canonical resource signature. A resource association can therefore change
   without producing a new semantic vintage if MW, status, revision and reason
   are unchanged.
2. `BridgeEntsoeSeriesResources` is grouped only by `SeriesKey` and resource
   identity across all vintages. It therefore represents the union of every
   resource ever observed for a series and cannot identify the resource set
   belonging to a specific vintage. Merely adding a resource signature to
   `VintageID` would not close point-in-time resource attribution. The bridge
   must be version-aware, preferably one row per `VintageID` and canonical
   resource identity, or the facts and bridge must share an explicit
   `ResourceSignature`.

The resource signature must be deterministic and order-insensitive, derived
from stable identity fields such as resource role, resource mRID, power-system
resource mRID and market-generating-unit mRID. Mutable names, locations and
nominal power should remain attributes rather than primary identity inputs,
except for a documented fallback where all stable identifiers are absent.

`source_document_mrid` is retained as lineage, but adding it unconditionally
to the semantic vintage hash requires one empirical check: identical repeated
API pulls must preserve the same document mRID, and successive revisions of
one outage must retain a stable event/document identity. If the extraction API
generates a new response-document mRID per pull, hashing it would manufacture
one vintage per daily capture even when the business state is unchanged.

One actual A78 XML payload must also validate that every
`Asset_RegisteredResource` is populated in `resource_details`. The parser
currently searches direct dotted element names, whereas the official XML
examples also show a nested `Asset_RegisteredResource` structure. If the DEV
payload is nested, traversal must occur per resource block; if the bridge is
fully populated on a representative A78 payload, this concern is closed.

Availability semantics are now sufficiently explicit for consumption, with
one invariant: strict FMV-observed point-in-time backtests must use
`AvailabilityBasis=FMV_FIRST_SEEN`; `SOURCE_DOCUMENT_CREATED` is source-stated
publication timing, while `UNKNOWN_BACKFILL` is not known point-in-time
availability. Historical backfill remains one as-retrieved version because no
historical revision/as-of API was identified; prospective daily captures build
the governed vintage history from their start date.

This superseding pass was functional only, as requested. It did not reassess
rebuild, tests, deployment or PROD promotion and submitted no Databricks SQL or
job.

## Functional closure at DEV `c482130`

The DEV branch was refreshed again and is now headed by:

- `3c62fee6856347e6ebd0736c9063553d90911de9`, `fix(entsoe): version
  outages by document and resource`;
- `c482130677223aa4bb992a2ea6ce07084577f7ca`, `fix(entsoe): link outage
  resources to vintages`.

Direct inspection confirms that the outage marker now contains document mRID
and an order-insensitive resource-detail signature. The Gold resource bridge
now carries `VintageID`, includes it in `BridgeID`, declares a non-enforced FK
to `FactEntsoeTimeSeriesVintages(VintageID)`, and is reconstructed from Silver
at vintage-resource grain. Post-backfill validation checks bridge existence,
required fields, duplicate keys, orphan SeriesID/VintageID and a deterministic
Silver-to-Gold bridge contract comparison.

Verdict: the previously identified outage vintage/resource modeling gap is
closed in code. The two remaining ENTSO-E items are empirical runtime evidence,
not further modeling changes:

1. prove on representative real A78 XML that every expected
   `Asset_RegisteredResource` populates `resource_details` and the Gold bridge;
2. prove on a real cancellation/withdrawal that A09/A13 is retained, or detect
   that the source silently removes the row and then define reconciliation.

Read-only AST parsing of the exact `origin/dev` versions of
`src/entsoe_pipeline.py` and `src/entsoe_validation.py` passed with `AST_OK`.
This is only syntax evidence; no PySpark unit/integration execution, DEV
rebuild, Databricks SQL, Warehouse or job was run from PFC_LT.

## Data-backed review at DEV `e7db715` — 2026-08-12

The ignored review clone was refreshed to
`e7db715370a3aa85f037854f019eca97dc59d885` (`fix: guard silver overwrite on
partial entsoe replays`). The overwrite guard is functionally sound: a Silver
overwrite is refused when the selected Bronze row/group scope is smaller than
the full Bronze table. GitHub YAML/deploy workflows reported success, but they
do not constitute Python parser tests.

The data engineer supplied the output of DEV post-backfill validation run
`efe9a1aa-f5cd-4534-9b78-8eecf4227f62`: 56/59 checks passed. The A78
Silver-to-Gold resource bridge passed for all 576 parsed vintage-resource rows.
However, the supplied evidence exposed one remaining parser defect:
`doc_status` is null for every A77/A78/A80 Bronze and Silver row. Official
ENTSO-E outage schemas model `docStatus` as an optional `Action_Status` complex
element whose code is in child `value`; the current parser reads only direct
text from `docStatus`. Both `parse_entsoe_points` and
`parse_entsoe_document_metadata` must read `docStatus.value`, with flat
`docStatus` as a compatibility fallback. Until rebuilt after that fix, absence
of A09/A13 cannot validate cancellation/withdrawal behavior.

The three validation failures represent two underlying issues:

1. `bronze_points_dense_group_gaps` reports 56 series. The check is
   functionally invalid for week/month/year-ahead NTC because it assumes every
   interval between min/max timestamps must exist at the reported resolution;
   ENTSO-E defines these auction products as one value for the whole
   week/month/year unless a higher resolution is published. Replace this with
   product-cadence/auction-period coverage. Keep dense checks for actual flows
   and day-ahead NTC; the very large Swiss-border gaps there remain real
   unexplained coverage gaps and must be localized against raw API responses.
2. Silver and Gold repeat the same source-level freshness failure for CH, DE
   and IT solar intraday forecasts: latest business date 2026-08-11 versus
   required 2026-08-12. Recheck after the expected publication window and
   classify raw response status before attributing this to ENTSO-E.

The comparison `1481 raw XML files containing Asset_RegisteredResource` versus
`576 Silver vintage rows with resource_details` is not a valid completeness
ratio because file and vintage-point grains differ. Required closure is a
snapshot/file-level Raw-to-Bronze anti-join, followed by the already-passing
Silver-to-Gold bridge check. No Databricks query or job was started by Codex;
all data evidence came from the attached engineer output.

## Pre-rebuild audit at DEV `b7d7f6e` — 2026-08-12

Commit `b7d7f6e716f2a09b994bfca8a1972e6e6ee97a1b` correctly adds
`_doc_status_code()` and uses it for root metadata, point parsing and the
TimeSeries fallback. An isolated AST execution against namespaced complex
`<docStatus><value>A09</value></docStatus>`, flat A13, missing status and a
non-direct nested status passed. The new default dense-group list also
correctly removes week/month/year-ahead NTC while retaining actual load,
reservoir storage, physical flows and day-ahead NTC.

Do not rebuild yet: the commit accidentally replaced the notebook bootstrap
cell in `notebooks/entsoe/90_post_backfill_validation.ipynb` with a duplicate
of the configuration cell. The notebook now invokes `load_pipeline_conf`,
`WORKSPACE_REPO_ROOT` and `json` before defining/importing them and creates the
`env` widget twice. Restore the parent bootstrap cell (`import json`, `sys`,
repo-root discovery, `src.utils.load_pipeline_conf`) as the first code cell;
move the dense-group edit into the following configuration cell. That original
configuration cell is byte-for-byte unchanged at `b7d7f6e` and still includes
`ntc_week_ahead`, `ntc_month_ahead` and `ntc_year_ahead`, so the advertised
restriction is not effective in a repaired/runnable notebook unless it is
applied there. No parser unit test was added. GitHub YAML/deploy workflows are
green but do not execute the Python/notebook logic. No Databricks SQL, job or
rebuild was launched.

## CurveType A03 verification — 2026-08-12

The core concern is confirmed and is rebuild-blocking, but the quoted timeline
and workaround are not supported by the official material reviewed. ENTSO-E's
official migration notice, updated 2026-01-16, states that REST responses from
the newer platform are optimized with A03 variable-sized blocks and lists
product migrations primarily across 2024-2025 (prices 2024-10-04, physical
flows 2025-10-07, renewable forecasts 2025-10-23, forecast transfer capacity
2025-11-04, load 2025-11-10; outage products already used A03). No official
evidence was found for a global August-2026 default switch. The official REST
parameter list does not list `curveType`, so forcing `curveType=A01` must not be
used as the solution without explicit vendor confirmation.

The current parser does not read or persist `curveType`. It maps each reported
position to exactly one `[start, start + resolution)` interval. Under A03 a
reported position starts a block whose value continues until the next reported
position, or to the Period end for the final block. A synthetic A03 period from
00:00 to 04:00, PT60M, with positions 1=100 and 3=200 produced only 00:00-01:00
and 02:00-03:00 instead of four hourly intervals. Thus emitted block-change
start positions are correct, but coverage, interval ends and right-edge value
timestamps are incomplete/wrong. This can explain part of the earlier dense-gap
failures and makes the current rebuild unsafe.

Required functional fix: retain `curve_type`, parse the Period end, implement
A03 carry-forward from each position to the next change/end, and expose a dense
resolution-grain series for Silver/Gold while preserving source-block lineage
(`source_position`, block bounds and an expansion flag). Keep A01 behavior
unchanged. Add tests for single and multiple A03 blocks, last-block-to-period-end
and PT15M/PT60M. No ENTSO-E API call, Databricks SQL, job or rebuild was run.

## Functional review of the 22-group scalar audit — 2026-08-12

The engineer's narrower conclusion that 15 of 22 configured groups already
read the expected scalar XML element is plausible, but it is not equivalent to
Gold modeling correctness. In DEV `b7d7f6e`, non-entity `series_key` is only
`group_name || field_name`; it omits source TimeSeries mRID and semantic
dimensions such as business type, process type and value category. Although
these dimensions participate in the vintage hash, the Latest observation key
uses only `series_key`, interval start and value timestamp, and
`DimEntsoeSeries` groups by `series_key`. Distinct semantic TimeSeries can
therefore collapse in Latest/Gold.

The audit's item findings were checked against current ENTSO-E r3 file-library
specifications and the parser/config:

- A75 actual generation exposes both actual generation output and actual
  consumption. Because the parser reads the first generic scalar and the
  configured field is shared per zone/PSR type, this is a Gold collision, not
  merely a descriptive warning. Preserve separate measures/identities; derive
  net output only explicitly.
- A85 requires positive and negative imbalance prices and can include separate
  scarcity, incentive and financial-neutrality components. The current parser
  retains only the first `imbalance_Price.amount` and not its category, so the
  audit's failure is correct.
- A86 requires total imbalance volume plus optional difference, situation
  (surplus/deficit/symmetric) and status. The first-value parser collapses these
  semantics, so the audit's failure is correct.
- A77/A80 publish available capacity. Configured names and value kinds call it
  unavailable MW, which reverses the business meaning. Preserve raw available
  capacity; calculate unavailable capacity only from a valid resource-level
  nominal/reference capacity with provenance.
- A78 currently publishes `NewNTC[MW]`; the configured
  `transmission_unavailable_mw` label is not authoritative. Combined with its
  documented A03 use, this is a failure, not a mild warning.
- A71/A33 requires production-unit code/name, validity range, status, type,
  location, voltage and update time. The outage-only resource-detail parser
  does not provide this complete per-unit identity/lifecycle contract.
- A09 is explicitly requested with contract type A01 in the current config, so
  its scalar assessment can remain provisionally correct. It is still covered
  by the cross-cutting semantic-key and A03 tests; do not invent a separate
  missing contract-filter defect.

Before rebuild, the functional blockers are: correct A03 expansion; a canonical
semantic series identity that prevents same-timestamp collisions; explicit
multi-measure parsing for A75/A85/A86; corrected A77/A80/A78 measurement names
and derivations; and complete A71/A33 unit identity/lifecycle. Currency and
measure unit should be retained as separate fields rather than selecting the
first of quantity unit, price unit or currency. Required fixture tests cover
A01/A03, multiple semantic TimeSeries at one timestamp, and the expected
measure/dimensions for all 22 configured groups. No Databricks SQL, job or
rebuild was run for this review.

## Current DEV `d7dcaed` and supplied runtime log — 2026-08-17

The remote branch advanced beyond the requested scheduling commit `f2c5a46` to
`d7dcaed788507196adaa6f9002439edc2d382e85` (`fix(entsoe): harden API response
handling and failure alerting`). `f2c5a46` is an ancestor. The current commit
correctly changes A71/A33 request chunks from 1100 to 365 days, adds failure
notifications to the five scheduled entry-point jobs, detects plain versus ZIP
responses, preserves every XML member in multi-member ZIP responses and makes
failed landing writes visible. The unit-test workflow discovers all tests, but
GitHub run status could not be queried: the local `gh` credential is invalid and
the API proxy is unavailable. No local project test runtime or Databricks job was
started by Codex.

The supplied read-only log `C:\Users\jbattaglia\Downloads\entsoe_log.txt` has
SHA-256
`36B813D27B97A10581FE98499D0099817CB9A9AC70734B185F769D43E4D00454`.
It is a successful **backfill** validation, not evidence from the newly scheduled
`operational` job: reference time is 2026-08-13T04:05:43Z, 59 checks passed, four
strict-cutoff checks were skipped, one coverage check was informational and no
check failed. Its latest Bronze ingestion began 2026-08-13 21:00; latest Silver
and Gold runs began 2026-08-14 07:25 and 08:08. It therefore does not prove that
the schedules introduced/restored later on August 14 and August 15 ran
successfully.

Four material findings remain:

1. `validation_mode=operational` changes only the reference timestamp. It does
   not scope the input tables to recent partitions. The validation contains 56
   `assert_zero_df` calls, each performing a count, plus summaries and full
   reconciliation hashes, with no cache/persist. The supplied tables contain
   about 17.0M Bronze points, 16.7M Silver latest/vintage rows and 16.7M Gold
   latest/vintage rows. Running this full-history audit daily on a fresh
   one-worker cluster is an avoidable Databricks cost. Keep the full audit for
   backfill/weekly qualification and create a recent-run/partition operational
   gate for daily use.
2. `dev_local` is the default bundle target and has schedules `UNPAUSED`; shared
   `dev` is also `UNPAUSED`. Deploying both can create duplicate scheduled pulls,
   validations and compute costs. Keep `dev_local` paused and make shared `dev`
   the single schedule owner.
3. Evening ingestion starts at 18:30 and the independent validation starts at
   19:15, but there is no dependency on completion of the evening core job. API
   retries, cluster startup or queueing can make validation inspect stale or
   partially refreshed state. Trigger it after successful core completion or
   explicitly verify the expected evening run reached Gold before validating.
4. A03 source-block boundaries are now represented correctly, but the parser
   deliberately emits one row per source block, not one row per resolution slot.
   This compressed storage is acceptable and cost-efficient only if its contract
   is explicit and local PFC ingestion expands `[IntervalStartUtc,
   IntervalEndUtc)` deterministically. The current dense-gap check counts rows as
   slots and therefore cannot distinguish A03 compression from real gaps. It
   reports 34 informational series, including Swiss day-ahead NTC missing ratios
   of 89.56% to 97.86% on CH-DE_LU/FR/AT and 91.24% on CH-IT_NORD. Coverage must
   be recomputed as interval-union coverage after A03 expansion before these are
   called source gaps.

Two prior functional gaps are still visible in the current tree: A78 remains
configured as `transmission_unavailable_mw` although the source measure is
`NewNTC[MW]`; A71/A33 retains unit mRID/name but not the complete validity,
status, location and voltage lifecycle contract. The daily validation checks
internal table consistency and does not detect either semantic error. The
publication-cadence document also still says 123 weekly requests while the job
and tests now correctly calculate 129.

## Current DEV `71c963a` static review during rebuild — 2026-08-17

Remote DEV is now exactly
`71c963a06e76770362710aaddf8fc74432ade44d` (`feat(entsoe): harden
operational validation for prod readiness`), parent `d7dcaed`. The read-only
review clone was detached at that commit. `git diff --check 71c963a^ 71c963a`
reported no whitespace error. The CI definition still runs
`python -m unittest discover -s tests -v`, but no CI result or Databricks run
was queried by Codex. The user reported that a DEV rebuild was already in
progress; Codex did not start, stop or inspect it.

The commit closes the four issues recorded immediately above:

- evening validation is a dependent task after successful Gold completion;
- `dev_local` schedules are paused while shared `dev` owns the schedules;
- A03 coverage is computed from merged `[interval_start_utc,
  interval_end_utc)` intervals rather than point counts;
- A78 old field suffixes are canonicalized in Silver to
  `_transmission_new_ntc_mw`, and the Gold Dim validator reuses the production
  semantic contract.

The A03 implementation matches the official ENTSO-E curve definition: every
position starts a block, the next position closes it and the final block ends
at the Period end. The current Transparency Platform A78 view labels the value
`New NTC`, so the canonical rename is appropriate for the currently observed
publication mode. A real DEV A78 payload remains required after rebuild;
ENTSO-E MoP v3r5 also foresees other transmission-unavailability publication
modes, so a future payload with available/installed capacity or net-position
impact must not be relabelled as New NTC.

Two PROD-readiness blockers remain in the operational gate:

1. The expected recent group set is the union of groups already observed in
   scoped Raw, Bronze, Silver and Gold. If a requested group is absent from
   every layer, the union omits it and every layer can pass. An empty scope also
   yields an empty required set and can pass all zero-row checks. This is
   realistic because ENTSO-E `No matching data found` is intentionally treated
   as a successful empty response. The gate also has no configured expected
   field/entity manifest, so one present border or zone can make a group look
   present while another requested border, direction, zone or PSR type is
   absent. The evening wrapper's exact seven request groups are not passed to
   the validation job. Required fix: persist/pass the exact expanded request
   manifest for the causally preceding core run and fail on missing group plus
   missing semantic field/entity, with an explicit non-empty-scope assertion.
2. Operational scope selects **all** successful
   `00_landing_to_bronze_entsoe` runs whose `start_ts` falls in the previous 26
   hours. `dq_ingestion_runs` does not record/filter `mode` or request groups.
   A recent `full` rebuild is therefore included in the next daily operational
   validation, potentially causing a full backfill-sized audit and avoidable
   Databricks cost. It also means the validation is not tied specifically to
   the evening run that triggered it. Required fix: record at least `mode`,
   `request_groups`/manifest and the parent/core run identity, then select the
   exact successful incremental run. Until this is deployed, do not let the
   evening validation run automatically within 26 hours of the current full
   rebuild.

The timestamp fallback is bounded but currently fail-open: metadata read errors
are swallowed and a zero-row fallback scope is not itself a failed check. It
must either fail closed for the production gate or record a mandatory failed
scope-integrity check. The new tests confirm the current union/fallback
behaviour but do not test that empty scope, a missing whole group, a missing
border/zone, or a recent full rebuild causes failure/exclusion.

Verdict at `71c963a`: rebuild may continue and the parsing/model changes are
materially improved, but **NO-GO for PROD promotion** until the two operational
gate blockers above are corrected and a successful post-rebuild backfill
validation plus one true incremental evening validation are supplied.
