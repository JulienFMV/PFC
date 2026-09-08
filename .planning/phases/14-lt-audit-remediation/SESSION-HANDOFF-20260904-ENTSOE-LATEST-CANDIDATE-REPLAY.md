# Session handoff - ENTSO-E July latest-candidate replay

Date: 2026-09-04  
Base commit: `e4bfd457d3d37d654be54abfbf10d821640c1650`  
Durable decision: D-20260904-287

## Outcome

The prior statement that all local work had to wait for a platform-signed
finality receipt was corrected. The producer contract exposes retained latest
revisions and a seven-day correction lookback, but no signed finality service.
Missing finality therefore blocks only the `realized_final` label.

The existing July Parquet was not requeried. It now passes the explicit
`realized_latest_candidate` validator and a self-contained replay reproduces
all 12,083 raw and consumer rows exactly. The candidate has `is_final=false`,
`consumer_contract_authorized=false` and every model, selection, monthly-level,
publication, production and trading authority false.

No Databricks request, SQL statement, Warehouse start or new business-row
download occurred in this correction.

## Producer-contract audit

Read-only GitHub API inspection used producer repository
`FMVSA/opendata-lakehouse`:

- main commit: `545c56c5d08b9c360120eb5014f0d5a7fe7fd2f2`;
- tree: `ae253a9b352d8df087e0464158530175d995b553`;
- `docs/data-contracts/entsoe.md` blob:
  `3f425866266c6fdede6ec8890e82dbbaaa9fb213`;
- `docs/data-contracts/entsoe-publication-cadence.md` blob:
  `c062e5002b8ebcd7a8306415169eebfc0d779ddd`;
- `src/entsoe_pipeline.py` blob:
  `fc01d27e78fa2744d81b1b8ec78fd855a69f4c5e`.

The producer calls the serving state `Latest`, retains source revision lineage
and uses `default_incremental_lookback_days = 7` to capture late corrections.
No signing, receipt or finality facility was present in the recursive repository
inventory or the inspected contract/pipeline sources.

Issue 4 remained open with no platform response. A scope correction was posted:
`https://github.com/FMVSA/opendata-lakehouse/issues/4#issuecomment-5538187585`.
It makes a finality response optional for local replay and keeps the September
CH recovery as a separate request.

## Code and contract changes

- `pfc_shaping/validation/entsoe_day_ahead_export.py`
  - adds export-only usage `realized_latest_candidate`;
  - adds `validate_realized_latest_candidate`;
  - emits non-final consumer-shaped rows with consumer authority false;
  - replays the candidate without a finality-evidence object;
  - normalizes raw SQL transport timestamps before the replay builder compares
    the semantic hash and archives source Parquet.
- `tests/test_entsoe_day_ahead_export.py`
  - proves a string-timestamp transport frame validates and replays;
  - proves `is_final` and finality/consumer authorities remain false.
- `docs/data/ENTSOE-DAY-AHEAD-EXPORT-V2.md`
  - documents the third, authority-negative candidate lane.
- `docs/data/LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md`
  - removes the nonexistent receipt as a local replay wait condition.
- `.planning/phases/14-lt-audit-remediation/ENTSOE-DAY-AHEAD-JULY-LATEST-CANDIDATE-REPLAY-V1-20260904.json`
  - freezes producer evidence, artifact hashes, replay result and negative
    authorities;
  - canonical JSON SHA-256:
    `6dd70193fcfa4e7554519b3106554f3416af479b693b51654a7c6e4c73aea542`;
  - file SHA-256:
    `0c55b75dfcf21e9d5883bcf5d7e17d70c0518c733b435954b90ade36fcfe7491`.
- `.planning/phases/14-lt-audit-remediation/LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json`
  - marks candidate replay complete;
  - makes separately authorized matched LSEG reconciliation the next local
    evidence step;
  - leaves global admission blocked.
- `tests/test_lt_source_acquisition_outage_plan.py`
  - binds the new evidence and updated execution order.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - adds D-20260904-287.
- `.planning/HANDOFF.md`
  - points to this outcome and handoff.

The earlier 4 September preflight, freshness, candidate capture, outage-plan and
handoff changes remain uncommitted in the same working tree and must be
preserved.

## Local replay artifacts

Directory:
`build/entsoe-july-candidate-20260904/latest-revision-candidate-replay/`

- build ID:
  `b00f2b725c66d91e7b8ec681b578fd080be77837a6e75f9b1de675caade20a26`;
- `manifest.json`:  SHA-256
  `c2c733cac80ae87f7b8b56572720ba02447a574d030313530f953c92d4b8b8a2`;
- `source/day-ahead-export.parquet`: 988,395 bytes, SHA-256
  `a2a759bbadacbd34ae96dcc9fbfb3c8b66ef494eb8c65ac24c5b456a514fc76c`;
- `consumer/day-ahead-source.parquet`: 133,891 bytes, SHA-256
  `3aadd9e8cbdeeba51d8b8cd56b42c6d9b90f914042d720ced154b80b0da90bda`;
- `evidence/export-audit.json`: 1,903 bytes, SHA-256
  `937ecb22b58d5c849c61b279e1ec507d05206b7cae5c6c113c60b078df008689`.

The build helper
`build/entsoe-july-candidate-20260904/build_candidate_replay.py` is ignored
under `build/` and contains no credentials or price literals.

## Commands and results

Workspace guards verified both current directory and Git top-level before each
command.

- initial focused test:
  `python -B -m scripts.run_workspace_local --run-id entcand01 --wall-timeout-seconds 600 -- python -B -m pytest tests/test_entsoe_day_ahead_export.py -q -p no:cacheprovider`
  -> `15 passed, 1 warning`;
- first replay launch without `PYTHONPATH` -> failed before reading the source
  with `ModuleNotFoundError: pfc_shaping`;
- second replay launch -> exposed that the builder compared pre-normalized
  transport dtypes with a normalized semantic audit and failed closed with
  `raw export differs from its audit`;
- after normalizing inside the builder, replay launch ->
  `VERIFIED_SELF_CONTAINED_DAY_AHEAD_EXPORT_REPLAY`, 12,083/12,083 rows;
- combined focused test before one expectation update -> `31 passed`, one
  expected plan-list assertion failure;
- focused regression after update -> `32 passed, 1 warning`;
- one adjacent command named nonexistent
  `tests/test_entsoe_day_ahead_capture.py` and collected no tests;
- corrected adjacent matrix:
  `python -B -m scripts.run_workspace_local --run-id entcand06 --wall-timeout-seconds 900 -- python -B -m pytest tests/test_entsoe_day_ahead_export.py tests/test_entsoe_day_ahead_consumption.py tests/test_entsoe_day_ahead_prd.py tests/test_spot_source_reconciliation.py tests/test_lt_source_acquisition_outage_plan.py -q -p no:cacheprovider`
  -> `127 passed, 1 warning`;
- targeted Ruff check -> pass;
- Ruff format initially identified only
  `tests/test_lt_source_acquisition_outage_plan.py`; it was formatted and the
  focused regression remained green.

The warning is the pre-existing unknown pytest `cache_dir` option.

## Next safe step

The July adapter/replay work is no longer blocked. A matched CH/AT/DE-LU/FR
LSEG reconciliation may be requested only with separate authorization and must
use the exact replay window and values. It cannot cover IT-North and cannot by
itself promote the five-market snapshot to `realized_final`.

Separately, monitor issue 4 for the September CH backfill. Before any retraining
or model admission, governed EEX evidence, prospective causal ENTSO-E captures
and a new independently frozen future holdout remain required.

## Invariants

- Do not rerun the July candidate or the 9.36 GB disambiguation statement.
- Do not infer finality from latest rank, age, DQ status or replay success.
- Do not treat the candidate as `causal_asof`, final truth, model input or a
  holdout.
- Keep all model and production authorities false.
- The CH monthly BASE solver remains the sole monthly-level authority.
- LT remains independent from CT and T057 remains sealed.
