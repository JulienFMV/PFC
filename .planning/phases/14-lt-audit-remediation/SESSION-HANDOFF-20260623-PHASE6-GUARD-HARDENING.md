# Session Handoff - 2026-06-23 - Phase 6 Guard Hardening

## Branch

- Worktree: `\\fmvfs2\Data\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase5_structural_width`
- Branch: `clean/phase6-guard-hardening`
- Base: `clean/phase5-structural-width` at `79680dc0`
- Upstream before push: none at time of handoff creation.

## Objective

Close the audit findings raised after Phase 5 around solver-authority local/test
exports:

- `--skip-build` could mutate a prebuilt solver-authority fan chart.
- Fresh `--enable-monthly-forward-curve-solver` builds still passed through
  export-time EEX calibration.
- Solver authority could be asserted by a weak/minimal manifest or CLI
  declaration.
- Power BI fallback could preserve stale legacy structural pseudo-quantiles.
- Seam nullspace smoothing used pandas labels as numpy positions.

## Expert Audit Loop

Read-only roaster agents were run repeatedly after each correction round.

Findings fixed:

- Fresh solver builds no longer rerun export-time `calibrate_hourly_to_eex`.
- Solver-authority exports block structural shape upgrade, post-calibration
  negative/peak rebalancers, EEX PEAK calibration, and all monthly/final
  mutators that can rewrite monthly means.
- Solver manifests must be adjacent to the fan parquet and complete enough for
  audit: market, forward snapshot date, solver config, `solver_config_hash`,
  `active_config_hash`, source hashes, active constraint hash, solution hash,
  KKT diagnostics, and legacy-bypass flags.
- `active_config_hash` is recalculated with `config_hash(solver_config)`.
- `solver_config`, `source_hashes`, and `solver_kkt` must be non-empty; KKT
  required fields are present, finite where numeric, and bool where boolean.
- Skip-build without a manifest now fails closed unless explicitly declared as
  `legacy`.
- Power BI recomputes legacy structural aliases from canonical structural
  scenario columns, or from ordered slow/central/fast scenario prices when no
  canonical bracket is present.

Final audit result:

- Agent `Godel`: `GO`, no P0/P1/P2 findings. Scope: solver manifest hash/KKT
  validation and test coverage.

Follow-up roast after commit `1e33d3b5e` found additional solver-authority proof
gaps:

- P0: a complete but stale solver manifest was not bound to the exported fan
  parquet.
- P0: EEX BASE validation used only non-overlapping finest calibration buckets,
  so a complete CAL quote could be missed when monthly quotes existed.
- P1: export-time structural bracket recompute could rewrite
  `price_weighted_mean_eur_mwh` under solver authority.
- P1: `source_hashes`, `solver_config_hash`, and KKT diagnostics were too weak.

Second correction round:

- Added `fan_chart_sha256` and `fan_chart_weighted_monthly_hash` to the local
  solver fan manifest.
- Export recomputes and validates fan parquet hash, fan weighted monthly hash,
  `source_hashes.forwards_path`, `solver_config_hash`, and `active_config_hash`.
- Export validates every fully covered Month/Quarter/CAL BASE quote
  independently, including parent quotes when finer quotes also exist.
- Solver-authority export now rejects windows with no fully covered EEX BASE
  quote.
- KKT gates reject residuals above `1e-6`, non-positive or `>1e12` condition
  number, non-integer rank/nullspace/dropped rows, and ridge/lstsq fallback.
- Final structural alias recompute restores solver `price_weighted_mean_eur_mwh`
  after alias recomputation.
- Fresh solver build passes export `--weights` through to
  `build_local_test_ch_pfc.py`.
- Power BI long labels changed from `P10/P90` to `Structural low/high`.

Final follow-up roasters:

- Agent `Hubble`: `GO`, no P0/P1/P2 in scope.
- Agent `Socrates`: `GO`, no P0/P1/P2 in scope. Observation only: no real
  `*.monthly_curve_manifest.json` artifact is present in this worktree, so this
  is implementation/test evidence, not production promotion evidence.

## Decision Log

Updated:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added `D-20260623-07 - Solver Authority Export Is Manifest-Backed And Non-Mutating`.

## Changed Files

Code:

- `scripts/export_local_test_ch_hourly_csv.py`
  - Added manifest-backed monthly authority resolution for prebuilt and fresh
    solver fan charts.
  - Added strict solver manifest validation, including `active_config_hash`
    recomputation and KKT field checks.
  - Added fan parquet hash, fan weighted monthly hash, forwards parquet hash,
    full-quote EEX BASE validation, and KKT threshold checks.
  - Blocks solver-authority post-processors that can mutate monthly levels.
  - Skips export-time EEX BASE recalibration for solver-authority fan charts.
  - Preserves solver `price_weighted_mean_eur_mwh` during final structural alias
    recomputation.
  - Makes the report explicit about monthly level authority and manifest path.
- `scripts/build_local_test_ch_pfc.py`
  - Writes `fan_chart_sha256`, `fan_chart_weighted_monthly_hash`, and
    `scenario_weights` to the solver fan manifest.
- `scripts/build_powerbi_exports.py`
  - Recomputes structural legacy aliases from canonical scenario bracket when
    present, otherwise from ordered scenario prices.
  - Labels structural low/high as structural brackets rather than P10/P90 in
    long Power BI exports.
- `pfc_shaping/lt/model/structural_scenario_bracket.py`
  - Rejects missing, extra, negative, zero-sum, NaN, or non-finite scenario
    weights.
- `pfc_shaping/lt/model/seam_nullspace_smoothing.py`
  - Maps pandas labels to positional indices before indexing interval-hour
    arrays.

Tests:

- `tests/test_export_local_test_ch_hourly_csv_script.py`
  - Covers skip-build solver mutator rejection.
  - Covers complete solver manifest acceptance and minimal/empty/hash-mismatch
    manifest rejection.
  - Covers stale fan hash, stale forwards hash, stale fan monthly hash,
    full-quote BASE constraint violations including CAL with monthly quotes, and
    rejection when no complete quote exists in the solver export window.
  - Covers fresh solver build skipping export recalibration.
  - Covers fresh solver build forwarding non-default `--weights`.
  - Covers fresh solver post-processor rejection.
  - Covers skip-build without manifest requiring explicit legacy authority.
- `tests/test_build_local_test_ch_pfc_script.py`
  - Covers reproducible fan manifest hash fields.
- `tests/test_build_powerbi_exports_script.py`
  - Covers canonical structural bracket precedence and stale legacy alias
    recomputation.
- `tests/test_structural_scenario_bracket.py`
  - Covers invalid scenario weight rejection.
- `tests/test_seam_nullspace_smoothing.py`
  - Covers non-default pandas index with BASE and PEAK constraints.

Planning:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-PHASE6-GUARD-HARDENING.md`

## Verification

Commands run from the clean worktree:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python -m pytest tests/test_export_local_test_ch_hourly_csv_script.py tests/test_build_powerbi_exports_script.py -q
```

Result:

- `54 passed in 52.48s`

Updated targeted result after follow-up fixes:

- `69 passed in 27.25s`

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python -m pytest tests/test_structural_scenario_bracket.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_build_powerbi_exports_script.py tests/test_cross_year_seasonal_shape.py tests/test_quote_aware_monthly_smoothing.py tests/test_seam_nullspace_smoothing.py tests/test_build_ep2050_multi_scenario_pfc_script.py tests/test_build_local_test_ch_pfc_script.py tests/test_lt_ct_imports.py -q
```

Result:

- Initial result: `98 passed, 1 skipped in 97.66s`
- Final result after follow-up fixes: `111 passed, 1 skipped in 28.38s`

## Risks And Non-Goals

- This phase does not promote monthly solver production flags.
- This phase does not activate solar shape modulation; prior audit evidence
  remains gate-off.
- This phase does not touch CT modules, Power BI data files, or heavy desk data.
- `solver_config_hash` is required as producer evidence but the active
  governance equality check uses `active_config_hash == config_hash(solver_config)`.
