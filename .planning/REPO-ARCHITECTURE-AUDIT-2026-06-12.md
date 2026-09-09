# Repo Architecture Audit - 2026-06-12

Scope: whole `PFC_LT` worktree, with special attention to LT/CT boundaries,
Phase 13 production readiness, source governance, packaging and repository
hygiene.

## Executive Verdict

The repo has a credible quant core and the LT/CT split is now mostly sound.
Phase 13 governance is moving in the right direction: new scenario behavior is
flagged OFF by default, vintage safety is explicit, and proxy data correctly
fails production gates.

The repo is not yet "state-of-the-art production clean" as a software asset.
The main blockers are packaging/tooling, artifact governance, oversized core
modules, stale/encoded documentation, and script sprawl. None of these require
changing the model math first; they require repository industrialization.

## Findings

| severity | area | finding | evidence | closure criterion |
|---|---|---|---|---|
| High | Packaging / CI | There is no discovered `pyproject.toml`, `setup.cfg`, `setup.py`, `pytest.ini`, `ruff.toml`, `mypy.ini` or `.github` workflow. The project therefore depends on ad hoc local environment state. | repo scan returned no packaging/tooling files; `requirements.txt`, `pfc_shaping/requirements.txt`, and `dashboard/requirements.txt` differ materially. | Add `pyproject.toml` with package metadata, pytest config, ruff format/lint, optional dependency groups (`lt`, `ct`, `dashboard`, `dev`), and a minimal CI/smoke command. |
| High | Data / artifacts | Large generated or cache artifacts are versioned inside source paths. `.gitignore` now ignores `data/*.parquet`, `output/*.parquet`, and `pfc_shaping/data/*.duckdb`, but tracked files still include `pfc_shaping/data/pfc_local.duckdb`, several parquet caches and `data/eex_forwards_history.parquet`. | `.gitignore:16-23`, `.gitignore:54-56`; `git ls-files data pfc_shaping/data` lists 20 data artifacts including `pfc_shaping/data/pfc_local.duckdb`. | Move raw/cache artifacts to `C:\Users\jbattaglia\pfc_local_data`; keep only small deterministic fixtures or manifest-pinned gold samples in repo. |
| High | Maintainability | `PFCAssembler` is too large and carries garbled encoded comments/log strings. It mixes orchestration, factor application, calibration, bridge logic, telemetry, calendar logic and consistency checks. | `pfc_shaping/lt/model/assembler.py:187`, `pfc_shaping/lt/model/assembler.py:413`, `pfc_shaping/lt/model/assembler.py:833`, `pfc_shaping/lt/model/assembler.py:1211`, `pfc_shaping/lt/model/assembler.py:1299`; mojibake at `pfc_shaping/lt/model/assembler.py:6-36`. | Split into `assembler_core`, `calibration_adapter`, `bridge_factor`, `energy_consistency`, `market_calendar`; repair encoding; keep public API stable with tests. |
| Medium | CLI / operations | `scripts/` has 45 top-level scripts mixing production, research, Phase 10, Phase 13, CT and LT utilities. This makes runbooks fragile and hides ownership boundaries. | `scripts/` inventory includes `run_phase10_*`, PriceFM, TYNDP, EP2050, governance, healthcheck and eval scripts side-by-side. | Introduce package CLIs under `pfc_shaping.cli.lt`, `pfc_shaping.cli.ct`, `pfc_shaping.cli.data`; leave wrappers in `scripts/` only for compatibility. |
| Medium | Data access safety | `load_electrification_scenarios_from_databricks` appends `where_sql` verbatim. The docstring says callers should keep it static, but the API surface permits unsafe dynamic SQL composition. | `pfc_shaping/data/electrification_scenarios.py:469-480`. | Replace with a constrained filter builder for vintage/country/scenario/year, or require a typed query object; keep raw SQL behind an explicitly named expert-only function. |
| Medium | Docs / onboarding | README is stale relative to the LT/CT split and contains legacy import paths plus encoding artifacts in the core docs. | README old paths at `README.md:175`, `README.md:189`, `README.md:317`; model doc mojibake at `pfc_shaping/lt/model/assembler.py:6-36`. | Regenerate README architecture map from `AGENTS.md`; remove old `pfc_shaping/model/*` references except in the migration note. |
| Medium | Dependency management | Three requirements files disagree on versions and optional dependencies. Some production-critical imports such as `holidays`, `pytest`, `databricks-sql-connector`, `statsmodels` appear only in `pfc_shaping/requirements.txt`, while root requirements pins a different surface. | `requirements.txt`; `pfc_shaping/requirements.txt`; `dashboard/requirements.txt`. | Single source of dependency truth in `pyproject.toml`; export lock files only if needed for deployment surfaces. |
| Low | Architecture boundary | LT/CT boundary is mostly respected. No `pfc_shaping.ct` imports were found under `pfc_shaping/lt` or `pfc_shaping/calibration`. CT imports appear only in `pfc_shaping/pipeline/swiss_short_term.py`, which is the CT overlay pipeline. | `rg "pfc_shaping\.ct" pfc_shaping/lt pfc_shaping/calibration pfc_shaping/pipeline` only returns `pfc_shaping/pipeline/swiss_short_term.py`. | Add this grep to CI or `tests/test_lt_ct_imports.py` so the boundary remains enforced automatically. |
| Low | Feature flag discipline | Phase 13 flags are correctly default-OFF and guarded in tests. | `pfc_shaping/lt/model/assembler.py:253`, `pfc_shaping/lt/model/assembler.py:257`; tests at `tests/test_electrification_shape.py:482-486`. | Keep this as a release invariant; add a flag-OFF byte-identity regression for the production assembly path. |

## Phase 13 Data Governance Status

Current best local inventory:

```text
data/electrification_scenarios_composed_p0_public_sources_2030.parquet
data/hpfc_scenario_features_composed_p0_public_sources_2030.parquet
```

Public/provenance components now wired:

| component | output | status |
|---|---|---|
| Swissgrid NTC 2026 baseline | `data/electrification_scenarios_swissgrid_ntc_baseline_2026.parquet` | official observed baseline proxy |
| Ember yearly 2026 baseline | `data/electrification_scenarios_ember_yearly_baseline_2026.parquet` | official historical baseline proxy |

Strict governance remains `NO-GO`, as intended:

| metric | value |
|---|---:|
| scenario rows | 15 |
| tests | 82 passed |
| gap register blockers | 26 |
| remaining numeric P0 fields | `import_twh`, `export_twh`, `hydro_reservoir_twh` |
| remaining non-data P0 blockers | human approval missing; proxy/partial quality flags |

This is correct model-risk behavior: true sources reduced nulls, but historical
baselines did not get promoted to approved 2030 assumptions.

## Recommended Closure Plan

1. Create `pyproject.toml` and CI smoke.
   Include ruff, pytest config, package metadata and extras. Make the current
   Phase 13 82-test suite one named smoke target.

2. Move data artifacts out of package paths.
   Keep `C:\Users\jbattaglia\pfc_local_data` as raw/cache home. Keep repo data
   only for deterministic small fixtures or governed sample manifests.

3. Split `PFCAssembler`.
   First extraction should be no-behavior-change: calendar helpers, energy
   consistency, bridge/rebalance factor and calibration adapter.

4. Promote script families to package CLIs.
   Keep thin wrappers under `scripts/`, but put logic under importable modules
   so tests do not depend on script-path behavior.

5. Repair documentation encoding and stale paths.
   README should mirror `AGENTS.md`, not the deprecated pre-split layout.

6. Harden Databricks filtering.
   Replace free-form `where_sql` in normal code paths with typed filters and
   keep raw SQL for expert/debug use only.

7. Close remaining Phase 13 production P0.
   Required external/governed inputs are gross import/export, reservoir energy
   capacity, and accountable approval of the scenario manifest.
