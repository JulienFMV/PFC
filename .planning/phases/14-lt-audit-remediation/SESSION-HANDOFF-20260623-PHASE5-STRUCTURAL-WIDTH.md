# Session Handoff - 2026-06-23 - Phase 5 Structural Width

## Branch

- Worktree: `\\fmvfs2\Data\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase5_structural_width`
- Branch: `clean/phase5-structural-width`
- Upstream: `origin/clean/phase5-structural-width`
- Base state before this phase: branch already pushed with integrated clean commits:
  - `274bf40 Add LT product normalization audit`
  - `2a68ae9 Harden delivered hourly export gates`
  - `d4e8ba2 Harden solver structural prior governance`
  - `834a968 Harden Q4 comparable-block audit`

## Objective

Close P1-3 from `AGENT-PROMPT.md`: the structural fan chart must no longer
compute or present three deterministic slow/central/fast scenario points as
probabilistic p10/p50/p90 quantiles.

## Expert Audit Inputs

Three read-only subagents were launched:

- Structural-column surface audit: found producers, consumers, tests, docs, and
  recommended canonical scenario bracket plus legacy aliases.
- Quant roast: confirmed the main defect is the CSV/export layer and mutators
  recomputing p10/p50/p90 over only three scenario points.
- Integration/export audit: warned not to rename the global production LT
  `p10/p90` uncertainty contract; restrict this phase to structural local/test
  fan-chart columns.

## Decisions

Updated:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added `D-20260623-06 - Structural Scenario Bracket Is Not A Probability Fan`.

Decision summary:

- Canonical local/test structural fan columns:
  - `structural_scenario_low_eur_mwh`
  - `structural_scenario_central_eur_mwh`
  - `structural_scenario_high_eur_mwh`
  - `structural_scenario_spread_eur_mwh`
- Legacy `structural_p10/p50/p90/width_eur_mwh` columns remain aliases only.
- Canonical columns win when canonical and legacy aliases conflict.
- The global LT probabilistic `p10/p90` contract remains out of scope.

## Changed Files

Code:

- `pfc_shaping/lt/model/structural_scenario_bracket.py`
  - New shared LT helper.
  - Computes weighted mean plus ordered scenario bracket.
  - Populates legacy aliases from canonical bracket when requested.
- `pfc_shaping/lt/model/cross_year_seasonal_shape.py`
  - Removed local `_weighted_quantile_row`.
  - Recomputes fan columns through `recompute_structural_scenario_bracket`.
- `pfc_shaping/lt/model/quote_aware_monthly_smoothing.py`
  - Same replacement for quote-aware smoothing.
- `pfc_shaping/lt/model/seam_nullspace_smoothing.py`
  - Same replacement for seam smoothing.
- `scripts/export_local_test_ch_hourly_csv.py`
  - Imports shared bracket helper.
  - Exports canonical scenario bracket columns before legacy aliases.
  - Recomputes final structural aliases from canonical scenario bracket.
  - Report width prefers `structural_scenario_spread_eur_mwh`.

Tests:

- `tests/test_structural_scenario_bracket.py`
  - New contract test: low/central/high/spread, legacy aliases, and central not
    replaced by median.
- `tests/test_export_local_test_ch_hourly_csv_script.py`
  - Exact column contract now includes canonical structural scenario columns.
  - Conflict test verifies canonical bracket wins over legacy aliases.

Planning:

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-PHASE5-STRUCTURAL-WIDTH.md`

## Verification Completed

Targeted mutator/export suite:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python -m pytest tests/test_structural_scenario_bracket.py tests/test_cross_year_seasonal_shape.py tests/test_quote_aware_monthly_smoothing.py tests/test_seam_nullspace_smoothing.py tests/test_export_local_test_ch_hourly_csv_script.py -q
```

Result:

- `45 passed in 25.05s`

Consumer/reporting smoke suite:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python -m pytest tests/test_electrification_shape.py tests/test_build_ep2050_multi_scenario_pfc_script.py tests/test_build_local_test_ch_pfc_script.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_pfc_hourly_shape_script.py tests/test_audit_ch_hfc_vs_spot_shape_script.py tests/test_build_ch_hfc_validation_workbook_script.py tests/test_plot_ch_hfc_diagnostics_script.py tests/test_lt_ct_imports.py -q
```

Result:

- `58 passed, 1 skipped, 13 warnings in 56.66s`
- Warnings are existing matplotlib/pyparsing deprecations.

Search guard:

```powershell
rg -n '_weighted_quantile_row|np\.quantile\(matrix|structural_p10_eur_mwh.*quantile|structural_p90_eur_mwh.*quantile' pfc_shaping/lt/model scripts/export_local_test_ch_hourly_csv.py
```

Result:

- No matches in the touched structural mutator/export path.

## Remaining Before Commit

- Run the integration suite used for this branch after the handoff update.
- Run `git diff --check`.
- Commit and push if clean.

## Risks / Follow-Up

- Power BI semantic labels still call legacy aliases `P10/P90`; this phase did
  not touch `powerbi/*` or report layouts because AGENTS restricts Power BI
  changes unless explicitly requested.
- The CSV/export layer now carries both canonical and legacy structural columns.
  A later reporting migration should rename UI labels to `Low/High` or
  `Structural scenario low/high`.
- Do not apply this change to the global probabilistic LT `p10/p90` export
  contract.
