# D316 — National source readiness and structural event proposal

Continue from AGENTS.md, .planning/HANDOFF.md and the Phase 14 decision log.
The user requested continuation and a GitHub checkpoint for their Claude audit,
then asked about grid projects, Swiss price effects and an event watchlist.
National CH drivers take priority; customer profiles are optional exposure
validation. No additional customer file is needed for this national work.

## Published audit checkpoint

Code commit: dee652bc919d06345f71304d1f1eaacf0edf7bc7.
Repository: https://github.com/JulienFMV/PFC, PUBLIC.
Branch: fix/lt-audit-remediation. Push succeeded and ls-remote matched HEAD.
139 files committed, including prior LT code/test/protocol work; no force push.
Audit entry: docs/model/PFC-CH-AUDIT-ENTRYPOINT-20260908.md.
User starts Claude; no external message or audit request was sent.

Six worktree documents deliberately retain richer local business notes than
their public Git versions: .planning/HANDOFF.md, Phase 14 DECISION-LOG.md,
SESSION-HANDOFF-20260908-LT-VOLUME-DISCOVERY.md,
SESSION-HANDOFF-20260908-QUALITY-FOUNDATION.md,
docs/model/LT-QUALITY-FOUNDATION.md and LT-VOLUME-SOURCE-QUALIFICATION.md.
Do not indiscriminately stage these local bytes. Public-safe versions were
staged through repo-local files and git hash-object/update-index; worktree
notes were preserved. D316 public governance additions follow the same rule.

## D316 result

21 national families, 94 dictionary groups, 260 history groups and 14 future
groups. History count 3,124,297 vintage rows is not hours/energy/completeness.
Actual generation types B10/B11/B12/B14/B16/B19; capacity-by-type dictionary
only B10/B11/B12/B14 in this scope. No global PV/wind source absence asserted.
The future per-unit capacity subset reaches end-2028; it is not the entire
Swiss fleet or a 2029 system forecast. Annual NTC support reaches end-2026.
4,179 future outage version rows lack the top-level resource identifier;
nested/raw identifiers remain to inspect. Available capacity is not MW lost.
No future rows in the four operational forecast families in the exact query.

History selects interval starts before 2026-08-31T22:00Z: 20 groups include
intervals ending after that cutoff. `closed-history` is a filename, not a
fully closed training dataset. Earliest recorded Databricks observation is
2026-08-07T07:03:13Z, so no independent pre-August retrospective PIT claim.
Current revised-history exploration remains possible if explicitly labelled.

D300 reused and hash-verified: six prepared files, including 35,040 physical
QH rows, and six OFEN hydro source/replay artifacts, 922 weeks through Aug31.
D300 prepared source root: build/local-pfc-source-preflight-20260907/.
No model fit, new curve or candidate export; existing candidates preserved.

## Exact execution and evidence

Task root: build/lt-national-readiness-20260908/.
Interpreter: build/conda-runtime-v41-model-source/python.exe -B.
Every shell action checks exact cwd and Git top C:\Users\jbattaglia\PFC_LT.
Runtime TEMP/TMP/caches/user dirs are under task-root/runtime; CPU threads4,
CUDA_VISIBLE_DEVICES=-1, PYTHONDONTWRITEBYTECODE=1. No external mutable paths.

Build helpers run as:
`python -B -c "import runpy; runpy.run_path('build/lt-national-readiness-20260908/NAME.py',run_name='__main__')"`.
Executed names: prepare.py, stage_checkpoint.py, finish_checkpoint_review.py,
record_checkpoint.py, capture_dictionary.py, profile_national_families.py,
profile_national_coverage.py, analyze_national.py, verify_national.py.
Final documentation/preservation helpers and Git receipts are in the same root.

Frozen SQL budget4, max2001 returned rows/statement, deadline180s, up to6
metadata GETs and at most1 bounded Warehouse start. Four SELECTs submitted:

| Statement | Result | Local source |
|---|---|---|
| 01f1ab84-7861-18c7-8008-a3ff689e4601 | Truncated at2001; rejected, no Parquet | national-dictionary-v1/ |
| 01f1ab84-acb5-1a2e-a158-9510b538f46a | Complete,94 groups | national-families-v2/ |
| 01f1ab84-e622-1961-b6d2-e7f4e5d69b7e | Complete,260 groups | national-coverage-v1/closed-history* |
| 01f1ab84-ea44-1c8e-b607-7c8408b878a6 | Complete,14 groups | national-coverage-v1/future-support* |

SQL files, request plans, raw responses and final receipts are preserved.
Capture origin for history/future eligibility: 2026-09-08T12:57:50Z.
Queries are successive, not an atomic VERSION AS OF snapshot. No budget left
for an additional SELECT in this lot. Zero Warehouse starts, no table/config
writes; last state RUNNING, 2XSmall, auto-stop45min, shutdown unconfirmed.
The earlier stop403 was not retried. No outstanding SQL or background job.

Checkpoint tests:34 modules,591passed/5skipped/0failed,35.13s.
Exact module list: checkpoint-test-files.json; tests-checkpoint.xml/log.
pytest flags: -q --tb=short,
-o cache_dir=build/lt-national-readiness-20260908/pytest-cache,
--basetemp=build/lt-national-readiness-20260908/pytest-checkpoint,
--junitxml=build/lt-national-readiness-20260908/tests-checkpoint.xml.
One existing timezone-drop warning in the explicit legacy EnergyCharts test.
Initial standard staged whitespace check flagged22 existing Markdown hard
breaks on Date/Base lines of Sep4 handoffs. Exact22-only review preserved
them; command-local core.whitespace=-blank-at-eol check passed. No global
configuration change or unrelated formatting cleanup. D316 changes no code.

Independent verify_national.py uses raw JSON arrays, integers and datetime,
without importing analyze_national. Three JSON/Parquet reconciliations and
12 D300 source/prepared hash checks pass. Review verdict:
VERIFIED_WITH_EXPLICIT_SOURCE_LIMITS; all six authorities false.
ENTSO-E SKOS ZIP could not be read by the web tool; no current-version binary
verification claimed. A historical official code list confirms only the
selected stable type meanings. Outage semantics use official ENTSO-E help.
The documentation helper first stopped on an assertion because the preserved
handoff contains mixed line endings. No shared file had yet been changed;
the guard was corrected to detect the actual heading separator, and resumed
against the verified unchanged watchlist. No unrelated newline conversion.

## Outputs and next work

Public reports: docs/data/CH-NATIONAL-INPUT-READINESS-20260908.md,
CH-STRUCTURAL-EVENT-REGISTRY-PROPOSAL.md,
CH-STRUCTURAL-EVENT-WATCHLIST-20260908.json.
Local report/concise needs: RAPPORT-NATIONAL.md and
DONNEES-NATIONALES-A-QUALIFIER.md under the task root.
Seven analysis JSON/CSV tables plus source-review.json are under analysis-v1/;
review-v1/verification.json records independent checks.

The event proposal uses four official Swissgrid examples, with actual present
registration time, unknown MW impacts/probabilities and no historical byte-pin
claim. Beznau–Tiengen/PST programme2040 is not a precise commissioning year;
Gotthard2030 and Bickigen2031 are outside current2026–2029 deliveries. No new
database service, scheduler, model overlay or autonomous surveillance installed.

Next: qualify national grids/availability, map outage asset/quantity semantics,
and freeze missing future drivers; then implement the smallest versioned event
registry with source captures if continuing this proposal. Separate physical
capacity from commercially available capacity. Do not infer MW from kV or
apply announcement-based EUR/MWh patches. Register dependencies and revisions.
Benchmark candidates globally before results with a new independent future
holdout; separate monthly level and hourly shape, reuse assembler/projection.

D304 reference; D305–D307 comparisons; solver sole monthly-level authority.
No per-month model choice, CT, protected data changes, AFRY values or T057.
GPU/Warehouse authorization persists but neither was newly started here.
Production/promotion/scientific_admission/trading/externally_registered/
countable_origin remain false. Prior D313/D314 artifacts and local notes are
preserved; closure records exact permitted shared-governance updates.
