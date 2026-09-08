# D313 - quality foundation, real source coverage and revised FMV scope

Date: 8 September 2026. Read AGENTS.md, .planning/HANDOFF.md, this handoff and
docs/model/LT-QUALITY-FOUNDATION.md. D304-D312 evidence remains preserved.

## Result and scope boundary

The priority foundation lot is implemented and locally verified. D304, solver
settings, assembler and EEX projection are unchanged. No model fit, prospective
truth opening, AFRY, T057, CT, Power BI, protected desk-data mutation or release.
All six authorities remain false. GPU authorized but not useful for this work.

User explicitly authorized execution of the proposed plan, then instructed us
to verify existing ENTSO-E/Databricks data, provide a precise missing-data list,
challenge BLOC13 and optimize the roadmap. BLOC13 is excluded from the active
required scope: repository mentions are documentary requirements, not proof of
a real FMV contract. FIL/ACC mappings are not inferred from asset names. Earlier
hash-frozen estimand/successor contracts remain intact and cannot be bypassed
for admission. A future admitted successor must reflect confirmed use cases.

Common PFC quality comes first; economic profile and hydro decision claims need
their respective populations and constraints. The still-missing LT volume
source does not stop market/shape/source-quality work. No economic gain claimed.

## EEX source and conflict results

D312 raw source:164rows, no duplicate raw key;38current Sep7quotes all trace to
one original ProductID/DeliveryPeriodID/quotation-date row and match settlement
and load timestamp. Capture receipt09:38:54UTC precedes D31209:45:22valuation.
All38 match the explicitly expected Sep7quotation date for that morning; this
does not establish a general exchange holiday/publication calendar or historic
PIT authority.155LastPrice values missing, none used as settlement substitutes.

Six parent/finer-product BASE/PEAK identities reconstruct from Swiss hours and
raw quotes. Max absolute direct residual0.004062500000003411EUR/MWh; all six
fit the predeclared half-cent rounding hypothesis. Vendor rounding remains
unconfirmed. Three OFFPEAK consequences reconstruct exactly, max0.0045363048.
Gates unchanged80PASS/9QUOTE_CONFLICT/0CRITICAL, solver means max1.02034e-11.
The exact candidate/forwards/conflict-identity-bound hierarchy proposal is
VALID_NOT_PRODUCTION_APPROVED; zero conflicts accepted. Existing authentication
rejects its missing independent signature/trust. No approval question is needed
to prepare this review package; no approval or waiver has been fabricated.

## Business-source investigation retained locally

Client-source metadata and aggregate coverage diagnostics were captured
under bounded read-only plans. Their granular details remain local. They
are not national Swiss evidence or a prerequisite for national PFC work.
Source schemas or recent values alone cannot establish economic ownership
or pre-origin availability. D315 clarifies the active national priority.

## New implementation

- pfc_shaping/validation/lt_source_quality.py: explicit source freshness and
  child-partition/Swiss-hour conflict reconstruction, no acceptance authority.
- pfc_shaping/validation/lt_economic_profiles.py: exact-grid, versioned,
  prevaluation-available MWh profiles with generation/consumption sign, negative
  prices retained, zero volumeUNSUPPORTED; full EUR valuation only. No invented
  FMV population, BLOC13, CHF convention, hydro policy or capture premium.
- scripts/collect_lt_benchmark_day.py: bounded source collection and existing
  scripts.run_lt_benchmark_day composition; actual time, no backdate option.
  Same-day guard before credentials/network; source schema and chronology,
  finite SQL/results/deadlines, one current-day OMPEX workbook, frozen LSEG,
  fresh build paths, concurrency lock and durable attempts/failures/manifests.
- tests/test_lt_quality_foundation.py:17tests.
- tests/test_collect_lt_benchmark_day.py:23tests.
- docs/model/LT-QUALITY-FOUNDATION.md: active scope, source map, command, budgets,
  conditional economic uses and remaining deployment/qualification work.

The collector calls the existing frozen D304 daily builder. It installs no
scheduler or background process.20day pilot stays1actual day;19future days and
an owned execution context remain pending. An interrupted owner's .capture.lock
is not stolen. Workbook mapped-drive access needs qualification in any future
execution context; the current script is a canonical-workspace local tool.

## Tests, independent verification and corrected failures

Final complete matrix134passed/4skipped, including40new tests, existing daily
builder/registry, EEX/OMPEX, quote-policy authentication and mandatory LT tests.
Initial focused17+18pass and intermediate matrices132then134pass are overlapping
checks, not extra independent test counts. Four skips are inherited fixtures.

Real same-day CLI:ALREADY_CAPTURED_TODAY,0network calls, no new registry entry.
Full local HTTP replay uses preserved real OMPEX/EEX/LSEG bytes with mocked
transport, not a new source observation. It exercises all6capture queries and
normalizers. V1found34903versus35015EEXrows: replacing the entire requested
window erased112saved rows on dates absent from the reply. Fixed to replace
only actual returned quotation dates, added regression tests, retained V1,
and V2verified35015EEXrows and19753LSEGhours. Quote values/lineage fields match;
source_snapshot_sha256 appropriately binds the newly replayed raw file bytes.
No failure was hidden by updating the model or saved original history.

Independent arithmetic reviewer uses raw delivery start/end dates and source
IDs, not the new quality helper:38quote identities,6direct/3derived residuals,
monthly means, exact CSV export hashes and62savedD312task files unchanged.
Full workspace baseline preservation is checked in the final closure, allowing
only the two shared governance files to change among pre-existing files.
Live profile JSON-response/Parquet parity is checked separately.

Report runtime copied and hash-verified belowbuild from installed plugin/node;
no external cache mutation, npm install, browser or project executable build.
Official portable report builder/delivery API is used. AGENTS forbids browser
launch, so its documented dependency seam routes to the unchanged structural
verifier. Final HTML validation/package passed; verificationSTRUCTURAL_ONLY,
browser rendering/interactions/SVG extraction are not certified.
Packaging corrections:local plugin manifest initially omitted; validator
required a native chart and actual SQL provenance for chart/tables; an UTF-8
read initially used Windows cp1252. Copied exact missing manifest, declared
UTF-8, added one grouped bar without removing report sections, and executed
small SQLite-in-memory projections of the real conflict register and missing
requirements. No Databricks queries were added for rendering. All limitations
and failure summaries remain in task artifacts; report revisions preserved.

## Commands and artifacts

All shell actions use an exact cwd/Git-root guard for C:\Users\jbattaglia\PFC_LT.
Python:build/conda-runtime-v41-model-source/python.exe -B. Mutable TEMP/TMP,
APPDATA/LOCALAPPDATA, matplotlib/XDG/Numba/joblib/Python/pip caches are below
build/lt-quality-foundation-20260908/runtime/<variable>. CPU thread caps4,
CUDA_VISIBLE_DEVICES=-1, PYTHONDONTWRITEBYTECODE=1. Credentials loaded only in
memory via the existing helper; never printed or copied into reports.

Task root:build/lt-quality-foundation-20260908/.
Executed Python helpers (in order, with dependent corrections noted above):
prepare.py; discover.py; discover_business.py; profile_business.py (timeout);
audit.py; replay_collector.py (v1failure then v2pass); verify.py;
prepare_report_runtime.py; report.py; finish_report_chart.py; bind_chart_query.py;
warehouse_check.py; profile_business_running.py; profile_summary.py;
update_report_business.py; bind_chart_query.py; close.py and validate_closure.py.
Report delivery:task/report-runtime/node.exe task/render_report.mjs using the
official delivery API, local runtime and explicit no-browser verification seam.

Pytest module command uses the source tests listed above plus:
tests/test_lt_benchmark_day.py tests/test_lt_benchmark_snapshots.py
tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py
tests/test_lt_ct_imports.py tests/test_databricks_eex_daily_snapshot.py
tests/test_ompex_benchmark.py tests/test_quote_conflict_policy_contract.py
with -q --tb=short, task-local cache/basetemp, and tests-validation.xml for the
final matrix. Earlier tests-quality/collector/matrix/final.xml retained.
Actual collector command:
python -B -m scripts.collect_lt_benchmark_day --config build/lt-quality-foundation-20260908/collector-config.json --registry build/lt-matched-vintages-20260908/registry --output build/lt-quality-foundation-20260908/collector-same-day-check

Key artifacts:
- plan.json,scope-amendment.json,workspace-before.json,prior-artifacts.json.
- audit-v1/:input-pins,conflicts,conflict-register,source-lineage,freshness-by-product,
  summary,source-hierarchy-proposal,policy-evaluation,policy-authentication,manifest.
- metadata-v1/,business-metadata-v1/,business-profile-v1/(timeout receipts),
  business-profile-v2/(plans,SQL,responseJSON,Parquets,summary,manifest,Warehouse receipts).
- collector-config.json,collector-same-day-check/,collector-replay-v1/,collector-replay-v2/.
- candidate-exports/:unchanged D312 D304 hourly and repeated-QH CSVs,manifest.
- review-v1/:independentverification,manifest;tests-*.xml.
- DONNEES-MANQUANTES.md,missing-inputs.json,RAPPORT-QUALITE.md,artifact.json,
  report.html,report-delivery.json,report-notes.json,audit-notebook.ipynb,
  chart-source.sql,missing-source.sql,failures-and-corrections.json.
- report-runtime-pins.json,report-runtime-extra-pins.json;copied runtime is
  reproducibility tooling, not model output. Earlier report revisions retained.
- closure.json,closure-validation.json bind final source/docs/results/hashes.

Retained candidate export SHA256:
hourly ff01b2f2f5c984fd7022b7e21eabe756a7561b2600959ee4dbc0af10cccc2109
repeated-QH 06647f4b9d66b5a25eafbeb614c966a8cb9e0dff2d777b7169696e2728b7c9d4
Retained D312 recipe SHA256:
afeefbb98ca6f0349efe3ed7f275aae0e374d88140f687a07af4d21519c82c3f.

## Next actionable work

D315 supersedes business-first sequencing. Continue national CH source
qualification from D300 and the existing availability contracts. Client profiles
remain optional economic validation and are never national system proxies.
The daily pilot and independent truth/custody are tracked separately. D304,
monthly solver, all-false authorities, noAFRY/T057 and LT/CT isolation persist.
