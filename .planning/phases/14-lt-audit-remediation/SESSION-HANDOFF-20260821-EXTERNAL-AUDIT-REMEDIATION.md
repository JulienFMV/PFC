# Session handoff - external audit remediation and source authority

Date: 2026-08-21

Starting commit:
`d743d24947`

Branch:
`fix/lt-audit-remediation`

## Outcome

The pasted independent Claude audit was checked against the actual execution
path. Four material findings and the legacy-entrypoint inconsistency were
confirmed and remediated locally. Two recommendations were rejected because
they conflict with the model architecture or misread a reachable code path.

Databricks was not queried. No Warehouse was started and no remote write was
performed.

## Source authority

The active architecture is role-specific:

- EEX forwards: Gold fact and dimensions, hard monthly solver constraints;
- spot: Gold interval fact and dimension, realized truth;
- ENTSO-E current serving: Gold dimension/latest;
- ENTSO-E PIT and revisions: Silver
  `ge_power_entsoe_time_series_vintages`;
- ENTSO-E Gold resource bridge: optional current-state enrichment;
- LSEG curve `110181967`: independent benchmark only;
- weather and Swissgrid Gold facts: candidate exogenous inputs after separate
  admission.

The model does not read these tables live. A bounded extraction creates an
immutable local Parquet snapshot plus manifest under `FMV_DATA_ROOT`; the
model consumes the admitted snapshot roles.

Important implementation boundary: the deterministic transformer from the
new Gold/Silver source-export roles to the model-facing `external_v2` roles is
not implemented yet. `scripts/create_lt_input_snapshot.py` only copies
pre-curated legacy files and publishes them as `MIGRATED_UNVERIFIED` with
`calibration_eligible=false`. It is not the Databricks adapter.

## Confirmed findings and changes

1. Forward front edge: current CAL/Q products could create past delivery
   months. `monthly_curve_authority.py` now selects only wholly undelivered
   CAL/Q/M products, validates the exact delivery grid and manifests included,
   started and unsupported products. `production_phases.py` builds from the
   filtered surface.
2. KKT fail-closed: `monthly_forward_curve.py` no longer falls back to least
   squares and now raises immediately on constraint or stationarity residual
   breaches.
3. Test environment: `pyproject.toml` declares `ingest` and `test` extras;
   `uv.lock` is synchronized.
4. CI: the unavailable exact Python pin is replaced by `3.11`; a bounded Linux
   LT/PIT workflow is added.
5. Legacy orchestration: `run_pfc_production.py` is reduced to a disabled
   sentinel and documentation names `pfc_shaping.cli.governed_release` as the
   governed entry point.

## Rejected audit claims

- Do not wire seam smoothing as a post-solver patch. D-20260709-109 requires
  seam work upstream in the zero-mean monthly/residual formulation, followed
  by unchanged EEX constraint verification.
- `_build_monthly_solver_contracts` is not dead. It is reached from
  `_build_non_overlapping_contracts` when solver-mode flags are active.

## Changed files

- `.github/workflows/lt-model.yml`
- `.github/workflows/publisher-runtime-v6.yml`
- `AGENTS.md`
- `README.md`
- `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`
- `pfc_shaping/calibration/monthly_forward_curve.py`
- `pfc_shaping/pipeline/monthly_curve_authority.py`
- `pfc_shaping/pipeline/production_phases.py`
- `pyproject.toml`
- `run_pfc_production.py`
- `tests/test_monthly_forward_curve_integration.py`
- `tests/test_monthly_forward_curve_solver.py`
- `uv.lock`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff

## Verification

All mutable pytest paths were below `build/`.

- full collection: `4324 tests collected`;
- targeted new front-edge/KKT and legacy-entrypoint tests: `54 passed`;
- bounded minimum LT, solver and PIT/data matrix: `293 passed, 3 skipped`;
- assembler/seam-lab matrix: `30 passed`;
- final combined bounded matrix: `333 passed, 3 skipped in 227.98s`;
- targeted Ruff: pass;
- `git diff --check`: pass before the final documentation update;
- `uv.lock` regenerated with a repo-local cache.

## Residual items

- The full historical suite remains unsuitable as a universal clean-clone
  gate: the external run reported local-evidence fixtures, governed runtime
  assumptions and cross-platform golden tolerances as 207 failures and 73
  errors. The bounded CI is positive regression evidence, not a claim that the
  full suite is green.
- Neutral ENTSO-E feature fills and legacy DE base interpolation remain
  explicit follow-up topics. For the CH governed path, missing admitted
  features should be handled by snapshot coverage/admission rather than hidden
  model defaults.
- Candidate governance still requires independent EEX acquisition evidence in
  addition to the local `external_v2` snapshot. Before simplifying this, decide
  whether that evidence remains a source cross-check or should be replaced by
  a content-bound Databricks export receipt. It is not the CH monthly level
  authority.
- Implement and test the Gold/Silver-to-model materializer. It must produce a
  reconciled current ENTSO-E view and origin-aware PIT feature views from
  Silver; a flat latest `entso` frame is insufficient for scientific backtests.
- Actual scientific admission still requires the governed local exports,
  PIT/data-quality checks and a new independently frozen future holdout.

Durable decision: D-20260821-251.
