# Session handoff - local PFC PRD input readiness

Date: 2026-09-07. HEAD: `e4bfd457d3d37d654be54abfbf10d821640c1650`.
Decision: D-20260907-299. Existing dirty work preserved; no commit or promotion.

## Objective and authorization

The user clarified the immediate objective: a reproducible local FMV PFC from
appropriate PRD data, with useful preparation for later scientific evaluation.
After being told that the native pipeline recalibrates its components at every
run, the user explicitly authorized CPU calibration of the existing model and
all work necessary for that PFC. Do not ask for that authorization again.

D298 remains the historical read-only audit of the real scientific evaluation
runner. Its independent-origin/holdout requirements are not blanket prerequisites
for an exploratory local calculation. Do not conflate local calibration permission
with retrospective PIT evidence, independent model selection or production
promotion. No source-admission flag, solver authority or production gate was
changed in this session.

Success criteria for the eventual PFC: source/version/configuration traceability,
correct markets/units/dates, explicit missing coverage, native component
calibration, hard EEX product constraints and preservation of solver monthly
means. No new assembly adapter, speculative model or scenario inventory.

## Implemented correction

Live Unity Catalog metadata confirms `prd.gold.facteexpricedaily.QuotationDateID`
is `INT`. The August NDJSON carries corresponding compact strings. A typed SQL
or Parquet reader therefore reaches a real interface mismatch: pandas treated
integer `20260804` as epoch nanoseconds and normalization turned it into
`1970-01-01`. This affected both the daily normalizer and the pre-origin filter.

Both existing entry points now parse quotation keys through their string calendar
representation. Mixed compact/ISO representations remain equivalent. Invalid
dates are rejected; delivery dates retain the previous parsing policy. No
settlement value, normalization formula or source authority was changed.

Regression tests first reproduced three failures. The final seven-file matrix
passes `104 passed, 4 skipped` in 3.16 seconds. Skips are CT optional imports:
LightGBM (two), Torch and TensorFlow; no LT assertion was skipped for this fix.

## Real source work completed

All new artifacts are below `build/local-pfc-prd-preflight-20260907/`.

| Evidence | Observed result | Meaning for the local PFC |
|---|---|---|
| `control-plane.json` | Warehouse `STOPPED`; EEX Gold and ENTSO-E Silver vintages visible | Current observation, not the old September 4 state; no SQL/start performed |
| `entsoe-metadata.json` | `prd.gold.dimentsoeseries` and `prd.gold.factentsoetimeserieslatest` visible | Exact current dictionary/latest interfaces available for bounded source preparation |
| EEX catalog statistics | maximum quotation key `20260903`; recorded total size 49,617,642 bytes | More recent PRD quotations exist in catalog statistics; the August capture is not current. Statistics are not a transaction-bound hard scan guarantee |
| ENTSO-E vintage statistics | recorded total size 1,549,891,183 bytes; `_year`/`_month` partitions | Prepare exact role/window/SeriesKey extraction; table-wide timestamps do not prove a market's coverage |
| `final-source-replay/eex-materialization-audit.json` | 82,552 captured rows; 34,105 normalized solver-history rows; 10,377 quarantined rows; quotation range 2019-01-02 to 2026-08-04 | Existing materializer successfully consumes the real capture with the PRD integer type; quarantine is preserved, not silently admitted |
| `hydro-table-discovery.json` | Full visible name inventories: 114 Gold and 107 Silver tables, no next page | Found asset hydro measures/reservoir dimensions; names alone do not establish national CH reservoir semantics. No claim of exhaustive semantic absence |
| `sfoe-source-observation.json`, `sfoe-reservoir.csv` | Existing documented SFOE OGD17 URL returned 83,741 bytes; 1,392 weekly observations, 2000-01-03 to 2026-08-31 | The already prescribed national hydro source is reachable; no need to silently substitute FMV asset measures or invent a Gold mapping |
| `sfoe-screening.json`, `sfoe-reconciliation-exceptions.json` | Monday cadence complete, national bounds valid; two regional-sum discrepancies of +1/-1 GWh on 2004-05-03 and 2008-05-19 | Preliminary screening only. These dates precede a modern ten-year reference window; do not turn them into a blanket claim that current hydro data is unusable. No clipping, correction, tolerance relaxation or full provider replay was performed |

The previously captured ENTSO-E July package remains a one-month
`realized_latest_candidate`; it does not supply all seasonal price, load,
generation and flow history needed by the native calibration. Its recorded
coverage/LSEG reconciliation was not re-queried.

Databricks calls: eight control-plane GETs, zero SQL statements, zero Warehouse
starts/resizes/creates/stops and zero remote writes. One guessed ENTSO-E dimension
FQN returned 404; the repository's exact `prd.gold.dimentsoeseries` then returned
success. SFOE: one bounded GET, maximum permitted response 4 MiB. Credentials
were read locally, never logged or saved in receipts.

## Ownership and concrete continuation

- Data engineering owns producer identities, transformations, delivery units,
  revision/freshness/coverage quality and the meaning of published PRD series.
  Consume their evidence; do not rebuild their entire quality framework.
- The PFC consumer owns exact market/product/SeriesKey selection, the date at
  which inputs were available, the usable calibration window, missing-data
  behaviour and EEX/output invariants. The date-key regression is our defect.
- Next, prepare a single coherent current input set: reuse the existing EEX
  three-table query, obtain exact ENTSO-E price/fundamental selections from the
  current dictionary and existing contracts, and define the common calibration
  window before any expensive export. Bind a fresh scan/cost preflight to those
  actual queries and snapshots; the stopped Warehouse observation is not a
  completed extraction plan. Keep all outputs below a fresh `build/` directory.
- Reuse SFOE OGD17 for national hydro as prescribed by the existing materialization
  contract. Resolve selected-window parsing and validate the causal ten-year
  water-value reference against the captured bytes. The preliminary download
  omitted full HTTP response metadata and is not a complete provider envelope;
  do not invent headers or promote it as a signed acquisition.
- After the input controls pass, use the existing native component/assembler
  path for the authorized local calibration. `run_long_term_phase` currently
  calls three fits; no complete traced fitted component bundle was established
  in the inspected model locations. The old Phase 13 local-test scenario runner
  is not a PRD-ready shortcut. Operational `governed_release` admission remains
  a separate boundary; do not weaken it to make an exploratory run pass.

No PFC was generated in this session. The remaining work is concrete input
preparation and local execution, not waiting for an independent future holdout.
Independent registry/truth-opening evidence is still required before claiming
prospective evaluation or promoting a model. T057 remains sealed.

## Exact changed files and hashes

- `pfc_shaping/data/databricks_eex_daily_snapshot.py`:
  `afbd83ef4ac14fade8c910227d390391037a5a98e9948ce01c7dd794a1036943`.
- `pfc_shaping/data/databricks_lt_materialization.py`:
  `cb85f8c37cb0f7d0b08d90dfa42f1ceed58f538516c10998be4b66dbad401c4f`.
- `tests/test_databricks_eex_daily_snapshot.py`:
  `67be2d0162cf1c0d7be0a5a71088d916bbf05f8791ccf1da4873a7590df2cac6`.
- `tests/test_databricks_lt_materialization.py`:
  `34ffa84cebbf2a1ee11cc3c29bf36b5d4aa1d40e1614c379a77ddc936d4e820a`.
- `docs/data/DATABRICKS-LT-MATERIALIZATION.md`: quotation-key contract clarified.
- `.planning/HANDOFF.md`: current scope and continuation updated.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`: append D299.
- This session handoff.

New local artifacts also include the one-off `materialize_eex.py` replay script,
preliminary EEX replay and `final-source-replay/` tied to the final source bytes.
The two normalized Parquet outputs are byte-identical, SHA-256
`ff25b4310060c37f6ca46a2a370f75bd49261283b10489855aa93a8b1698bb9a`.
Use the final-source audit for current code provenance; earlier audit is retained
as an intermediate observation. The old source capture remains SHA-256
`593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812` and its
manifest remains `f8ec096be43851d85b16ec2b678d4a695fb0521c2c651e8bcf7c2491a29b50c1`.
SFOE source SHA-256:
`793fd38b216d547c30c098d93f5ee1e886bf33290570bec590e851f82ba255c0`.

Existing source-bound replay/signed bundles are not rewritten or re-admitted
after this parser change. New replay evidence binds the new code explicitly.

## Commands and verification

Every shell action begins with canonical cwd/Git-root guards. Python uses
repo-local `build/conda-runtime-v41-model-source/python.exe -B`, whose `_pth`
references repo-local v34 libraries/test dependencies; mutable paths remain
under `build/`.

Test command prefix:
`python.exe -B -m scripts.run_workspace_local --run-id <id> -- python.exe -B -m pytest`.
Both interpreter occurrences denote that exact repo-local interpreter.

- `d299red`: the two changed test files, `-k 'prd_date_keys or prd_integer_date_key' -q`:
  `3 failed, 2 passed, 36 deselected`; observed erroneous epoch dates.
- `d299green`: two changed test files plus `tests/test_databricks_lt_snapshot.py`,
  `tests/test_arbitrage_free.py`, `tests/test_cascading.py`,
  `tests/test_water_value.py`, `tests/test_lt_ct_imports.py`, `-q`:
  `101 passed, 4 skipped`.
- `d299final`: same seven-file matrix after invalid-date tests and limiting mixed
  parsing to quotation keys: `104 passed, 4 skipped` in 3.16 seconds.
- Receipts/logs/JUnit: `build/workspace-local-runs/<id>/`; supervisor evidence:
  `build/workspace-local-supervisors/<id>/`. Initial runtime inventory preflight
  took several minutes; tests themselves were short. No elevation was used.
- Real EEX replay: `python.exe -B build/local-pfc-prd-preflight-20260907/materialize_eex.py`;
  final source run succeeded, source hashes checked before decoding, all four
  materializer authorities false. No fit, solver or real scoring.
- Control-plane reads used `Invoke-RestMethod -Method Get`; the bounded SFOE
  download used `HttpWebRequest` with 30-second timeout and 4 MiB byte limit.
  `Import-Csv` screening emitted dates/counts and isolated reconciliation errors.
- `git diff --check` and final artifact/source integrity checks complete the
  documentation-only closeout after the tested code changes.

Final observations: `git diff --check` passed. Native incumbent, common assembly
and production pipeline hashes remain exactly those in D297/D298. Final EEX
audit SHA-256 is `390d7fd3ca04799ba4490f5923a9b535002daa326bb818b80f60032d9317e85d`.
Control-plane receipt SHA-256 is
`115a85736395f96788d6e4a0f1c42ee0c9970b22589d8eddef6fada8eaf2a048`;
correct ENTSO-E metadata receipt SHA-256 is
`2ed3bb951c341184b88437db7166830857961260cfc50fa61acbdd6d38402cd2`.
The erroneous initial dimension lookup was
`GET /api/2.1/unity-catalog/tables/prd.gold.dimentsoetimeseries`; the successful
one used `prd.gold.dimentsoeseries`. Visible-name discovery used
`GET /api/2.1/unity-catalog/tables?catalog_name=prd&schema_name=<gold|silver>&max_results=1000`.

Minor command failures: several initial path/glob searches named nonexistent
paths or used Windows wildcards where `rg -g` was needed; corrected without
mutation. One tool script failed JavaScript parsing before the SFOE request;
the corrected request ran once. These did not grant authority or change inputs.
No GPU, model selection, production flag, CT code, protected desk data, AFRY
values or T057 truth was touched. No message was sent to a data engineer.
