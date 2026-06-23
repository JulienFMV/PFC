# Session Handoff - 2026-06-23 - Phase 3 Curated

## Scope

Worktree:
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase3_curated`

Branch:
`clean/phase3-hourly-shaping-curated`

Base:
`origin/fix/lt-audit-remediation` at `c7e8ab6`.

Goal: curate Phase 3 delivered-hourly shaping fixes from the dirty
`PFC_LT` worktree into a clean branch, without committing generated outputs,
Power BI project files, CT files, or heavy data.

No commit was performed.

## Workflow Contract

Pre-commit workflow requested by user:

1. Audit scope and branch hygiene.
2. Review tests and behavioral evidence.
3. Roast the implementation for hidden regressions.
4. Only then consider a commit.

This session stopped before commit after audit/review/roast.

## Files Changed

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added final PEAK projection decision.
  - Added ordered structural bridge / strict diagnostic export decision.
- `scripts/export_local_test_ch_hourly_csv.py`
  - Ordered `structural_scenario_low/central/high/spread` bridge columns now
    map to structural P10/P50/P90/width.
  - Ordered scenario columns take precedence over legacy structural columns when
    both exist.
  - `calibrate_hourly_to_eex_base_peak(...)` solves OFFPEAK level from quoted
    BASE energy and quoted PEAK energy.
  - Final `--enable-eex-peak-calibration` projection runs after final seam
    mutators and before CSV write.
  - Missing structural export columns are rebuilt from row-wise ordered
    slow/central/fast scenario values before CSV write.
- `scripts/build_powerbi_exports.py`
  - Structural columns are optional for input discovery.
  - Missing or null structural columns are rebuilt column-by-column from
    row-wise ordered slow/central/fast values without overwriting valid
    existing brackets.
  - Shape/seasonal/spot audits run against a temporary normalized CSV carrying
    the synthesized structural columns, so optional structural discovery does
    not fail later inside audit scripts.
  - Strict quality gates block sidecar export unless `--allow-failed-gates` is
    explicitly set.
  - Summary metrics record Power BI quality gate status and issues.
- `tests/test_export_local_test_ch_hourly_csv_script.py`
  - Added ordered structural bridge regression.
  - Added final-mutator PEAK calibration regression.
- `tests/test_build_powerbi_exports_script.py`
  - New tests for Power BI quality gate issue generation, row-wise ordered
    structural fallback, partial structural schema preservation, real
    `build_exports(...)` blocking, diagnostic summary writing, and CLI
    `--allow-failed-gates` pass-through.

## Audit

Commands:

```powershell
git status --short --branch
rg "pfc_shaping\.ct|powerbi/PFC_QA|powerbi/data|pfc_shaping/data/.*\.(parquet|duckdb)|data/epex_hourly\.parquet|data/eex_forwards_history\.parquet" scripts/export_local_test_ch_hourly_csv.py scripts/build_powerbi_exports.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_build_powerbi_exports_script.py .planning/phases/14-lt-audit-remediation/DECISION-LOG.md .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-PHASE3-CURATED.md
git diff --check
```

Findings:

- No CT import was introduced.
- No `powerbi/PFC_QA.*` file was modified.
- No `powerbi/data/*` file was modified.
- No heavy parquet/duckdb file was modified.
- Remaining references to `data/eex_forwards_history.parquet`,
  `data/epex_hourly.parquet`, and `powerbi/data` are existing CLI defaults or
  messages in scripts, not artifact edits.
- `git diff --check` reported no whitespace errors; PowerShell displayed only
  line-ending normalization warnings for tracked text files.

## Review / Tests

Focused regression commands:

```powershell
python -m pytest tests/test_export_local_test_ch_hourly_csv_script.py::test_to_hourly_csv_frame_prefers_ordered_structural_bracket_columns tests/test_export_local_test_ch_hourly_csv_script.py::test_final_eex_peak_calibration_runs_after_final_mutators tests/test_build_powerbi_exports_script.py -q
python -m pytest tests/test_build_powerbi_exports_script.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_audit_ch_pfc_hourly_shape_script.py tests/test_lt_ct_imports.py -q
```

Output:

```text
.......                                                                  [100%]
7 passed in 1.83s
........................................................s.               [100%]
57 passed, 1 skipped in 105.86s
```

Phase 3 suite:

```powershell
python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_build_powerbi_exports_script.py tests/test_export_local_test_ch_hourly_csv_script.py -q
```

Output:

```text
......................F................................................  [100%]
FAILED tests/test_monthly_forward_curve_integration.py::test_monthly_solver_defaults_include_structural_template_fallback
1 failed, 70 passed in 20.22s
```

Interpretation: after agent roast, the monthly structural fallback default was
removed from this Phase 3 branch as cross-scope. This existing solver-prior test
therefore remains a known prerequisite for a separate solver/lambda branch.

Second shaping/export suite:

```powershell
python -m pytest tests/test_monthly_forward_curve_solver.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_audit_ch_pfc_hourly_shape_script.py tests/test_build_powerbi_exports_script.py -q
```

Output:

```text
..........................                                               [100%]
26 passed in 4.03s
```

Broad guardrail:

```powershell
$env:PYTHONPATH='.'; $files = Get-ChildItem tests -Filter 'test_monthly_forward_curve_*.py' | ForEach-Object { $_.FullName }; python -m pytest $files tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_long_term_branch.py tests/test_lt_ct_imports.py tests/test_export_local_test_ch_hourly_csv_script.py tests/test_build_powerbi_exports_script.py tests/test_audit_ch_pfc_hourly_shape_script.py -q
```

Output:

```text
Not rerun after splitting the solver-prior default out of Phase 3. The scoped
Phase 3 suite above is green; the wider monthly integration suite is blocked by
the intentionally split solver-prior prerequisite.
```

## Roast

Three read-only roast agents reviewed this branch after the first curation.

Roast findings fixed:

- The dirty-worktree implementation mapped ordered
  `structural_scenario_*` columns before legacy structural columns, but the
  generic column loop could still overwrite them if both schemas coexisted. The
  curated branch now writes a target structural column only once, so ordered
  scenario brackets truly take precedence. The regression fixture intentionally
  includes crossed legacy structural columns to prove this.
- High: Power BI accepted CSVs with missing structural columns but then audited
  the original CSV, causing downstream audit scripts to fail before sidecars.
  Fixed by auditing a temporary normalized CSV produced from `load_hourly(...)`.
- Medium: Power BI rebuilt all structural columns when any one was missing.
  Fixed by filling missing/null structural columns individually and preserving
  valid existing brackets.
- Medium: strict `build_exports(...)` blocking and CLI
  `--allow-failed-gates` pass-through lacked end-to-end tests. Added both.

Roast finding split out of scope:

- Monthly structural fallback defaults
  (`allow_template_structural_fallback=True`, amplitude `110`,
  `structural_weight=1.0`) are solver-prior behavior, not delivered-hourly
  Phase 3 export behavior. They were removed from this branch and should be
  handled in a separate solver/lambda branch with its own evidence.

Residual risk:

- No real candidate regeneration was run in this clean worktree. Prior real
  evidence remains from the dirty-worktree Phase 3 handoff:
  best diagnostic score `6.75/10`, BASE/PEAK residuals `0`, quantiles ordered,
  structural width too narrow, strict Power BI still blocked.
- This branch improves delivered-hourly bridge correctness and strict gating;
  it does not solve Phase 4 cross-year Q4 or fan-chart structural width.
- Diagnostic sidecars from the hourly export wrapper still require either strict
  pass or a separate direct `scripts/build_powerbi_exports.py
  --allow-failed-gates` run. The wrapper remains strict by default.

## Current Status

Curated Phase 3 code/test patch is scoped to delivered-hourly export/Power BI
bridge behavior and is green on the scoped Phase 3 tests. It is not
promotion-ready as a PFC candidate because the model still needs:

- separate solver/lambda structural fallback governance;
- Phase 4 Q4 comparable-block solver work;
- a governed structural width model.

Next recommended branch:
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase4_q4`
on `clean/phase4-cross-year-q4`.

## Post-Roast Final Verification

After applying roast fixes and splitting solver-prior defaults out of this
branch, final status was limited to:

```text
.planning/phases/14-lt-audit-remediation/DECISION-LOG.md
scripts/build_powerbi_exports.py
scripts/export_local_test_ch_hourly_csv.py
tests/test_export_local_test_ch_hourly_csv_script.py
?? .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-PHASE3-CURATED.md
?? tests/test_build_powerbi_exports_script.py
```

Final command:

```powershell
python -m pytest tests/test_build_powerbi_exports_script.py tests/test_export_local_test_ch_hourly_csv_script.py::test_to_hourly_csv_frame_prefers_ordered_structural_bracket_columns tests/test_export_local_test_ch_hourly_csv_script.py::test_final_eex_peak_calibration_runs_after_final_mutators tests/test_lt_ct_imports.py -q
```

Output:

```text
.........................s.                                              [100%]
26 passed, 1 skipped in 9.76s
```
