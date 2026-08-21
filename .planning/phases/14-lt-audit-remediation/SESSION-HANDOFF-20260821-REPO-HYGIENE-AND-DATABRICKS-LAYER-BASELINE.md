# Session handoff - clean repository and Databricks LT baseline

Date: 2026-08-21

Baseline commit:
`9b73e6cfcf242c70f095441945dcb2f84e6be6f6`

## Outcome

The accumulated Phase 14 code, tests, contracts and decision evidence are
integrated into one local Git baseline. Generated outputs, local business data,
model weights and copied research PDFs are no longer repository content.

The active data architecture is:

- EEX Gold forward facts/dimensions: hard monthly solver constraints;
- Gold `DimEntsoeSeries` + `FactEntsoeTimeSeriesLatest`: current ENTSO-E
  serving;
- Silver `ge_power_entsoe_time_series_vintages`: canonical ENTSO-E PIT,
  revision, source-document and resource history;
- Gold `BridgeEntsoeSeriesResources`: optional current asset enrichment;
- LSEG `continuous_forward/CHE`, curve `110181967`: independent benchmark.

Legacy Gold-only D293 contracts remain labeled historical replay only. They
are not the current intake contract.

## Versioned cleanup

Deleted obsolete root files:

- `AUDIT_b52f790.md`, `AUDIT_DEEP_2026-03-23.md`, `CODEX_AGENT_BRIEF.md`,
  `HANDOFF_FMV.md`;
- `error_analysis.py`, `run_backtest_delu.py`;
- `eval.log`, `results.tsv`, `results_lear.tsv`.

`.planning/HANDOFF.md` was reduced from a duplicated historical transcript to
one current pointer plus permanent invariants. Historical detail remains in
the decision log and dated handoffs.

The generated ENTSO-E HTML report and its rendering artifact were removed;
`docs/research/ENTSOE-DEV-PFC-DATA-COVERAGE-REPORT-20260806.md` remains the
source report.

## Files retained locally but removed from Git

Physical bytes were preserved for:

- ten local data/database inputs: EEX, commodity, EPEX, ENTSO-E, hydro,
  renewable, outage, workbook and DuckDB files;
- seven copied research PDFs;
- two `safetensors` model artifacts.

They are now covered by explicit `.gitignore` rules. Deterministic Parquet test
fixtures and small Phase 10 evidence Parquets remain versioned.

Tracked files previously below `pfc_shaping/output/` were removed from Git and
the directory now keeps only `.gitkeep`.

## Physical cleanup

The guarded `scripts/clean_repo_generated.ps1` performed two cleanups:

- 35 cache/test directories: 42,901 files, 249.6 MiB;
- `output/` and `pfc_shaping/output/`: 14,543 generated files,
  15,345.4 MiB.

The second cleanup is not directly recoverable; those files are reproducible
runtime/model outputs. No source, governed evidence, local business input or
tracked path was deleted. The script now supports a dry-run/execute
`-IncludeOutputs` scope and refuses tracked paths, reparse points, unexpected
parents and non-canonical workspaces.

## Current contracts and files

- `README.md`: concise LT/CT and source-layer architecture.
- `docs/data/README.md`: current versus historical data-document index.
- `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`: current selective immutable
  local export contract.
- `docs/data/SHARED-DATA-PLATFORM.md`: shared source-layer policy.
- `pfc_shaping/validation/databricks_pfc_layer_acceptance.py`: mixed-layer v2
  acceptance validator.
- `tests/test_databricks_pfc_layer_acceptance.py`: six contract tests.

## Verification

No Databricks SQL, Warehouse start or remote write occurred.

- mixed-layer acceptance: `6 passed`;
- ENTSO-E/mixed-layer/legacy/import set: `65 passed, 1 skipped`;
- mandatory LT minimum: `58 passed, 1 skipped`;
- targeted Ruff: pass;
- full Python compile of `pfc_shaping`, `scripts` and `tests`: pass;
- assigned-secret scan: no finding; the only private-key marker is an
  intentional negative security fixture;
- focused post-cleanup smoke: `8 passed`;
- post-cleanup Ruff on the adjusted data-boundary files: pass;
- full pytest collection exceeded the 60-second foreground bound and was
  stopped before a test run;
- the full local pytest suite was then run with a 20-minute bound but did not
  emit a terminal verdict before timeout. Its Python child exited immediately
  after the timeout check; no orphan remained. This is `INCONCLUSIVE`, not a
  pass or a demonstrated test failure. Split it by governed matrix or run it
  on the independently governed CI runner with per-test duration reporting.

## Residual status

- No push or production promotion is implied by this local baseline.
- Global historical-suite qualification remains inconclusive because of the
  20-minute local timeout; the focused LT/data/import matrices listed above
  are the positive evidence for this cleanup.
- Data/model admission remains
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.
- T057 remains sealed.
- The next substantive step is the governed incremental local export from
  Databricks, followed by PIT/data-quality admission and only then model
  selection or calibration.

Durable decisions: D-20260821-248, D-20260821-249 and D-20260821-250.
