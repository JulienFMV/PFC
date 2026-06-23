# Session Handoff - 2026-06-23 - Solver Structural Prior

## Scope

Worktree:
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_solver_structural_prior`

Branch: `clean/solver-structural-prior`, tracking `origin/fix/lt-audit-remediation`.

Purpose: split the solver structural-prior default/manifest work out of the
Phase 3 hourly shaping branch, make fallback structural shaping auditable, and
repair the selected-lambda-vs-production manifest hash contract before Phase 4.

No commit was made.

## Files changed

- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-SOLVER-STRUCTURAL-PRIOR.md`
- `.planning/phases/14-lt-audit-remediation/lambda_grid.yaml`
- `pfc_shaping/calibration/monthly_curve_lambda_calibration.py`
- `pfc_shaping/calibration/monthly_curve_priors.py`
- `pfc_shaping/config.yaml`
- `pfc_shaping/pipeline/monthly_curve_authority.py`
- `scripts/check_monthly_curve_promotion_from_manifests.py`
- `scripts/run_monthly_curve_sparse_year_proof.py`
- `tests/fixtures/monthly_curve_phase_e_parity_baseline.json`
- `tests/test_check_monthly_curve_promotion_from_manifests.py`
- `tests/test_monthly_curve_lambda_calibration.py`
- `tests/test_monthly_forward_curve_integration.py`
- `tests/test_monthly_forward_curve_priors.py`
- `tests/test_run_monthly_curve_sparse_year_proof_script.py`

## Implementation

- Enabled audited template structural fallback in runtime config and code
  defaults:
  - `allow_template_structural_fallback=true`
  - `structural_amplitude_eur_mwh=110.0`
  - `history_weight=0.5`
  - `structural_weight=1.0`
- Aligned `lambda_grid.yaml`, `LambdaCalibrationSettings`, and
  `LambdaCandidateConfig` defaults with the same structural prior contract.
- Made `robust_panel_quantile=0.5` explicit in runtime config, monthly
  authority defaults, and `lambda_grid.yaml` so canonical payloads match exactly.
- Aligned `lambda_grid.yaml` with runtime for `constraint_tolerance=1e-9` and
  added `lambda_smooth_yoy=0.25`; runtime now has exactly one matching grid
  candidate.
- Extended `LambdaCandidateConfig` and canonical `config_hash()` payload to
  include:
  - neighbor markets;
  - `min_structural_snapshots`;
  - template fallback flag;
  - structural amplitude;
  - panel/history/structural prior weights.
- Changed monthly authority `active_config_hash` to use the same canonical
  `config_hash(settings)` contract as selected lambda artifacts.
- Changed manifest-backed promotion fallback hashing to use
  `config_hash(solver_config)` directly instead of rebuilding the old narrow
  monthly config payload.
- Changed sparse-year proof active config fallback hashing to include CLI
  structural knobs, markets, weights, lookback and robust quantile. Its defaults
  are now aligned with the runtime structural-prior contract.
- Passed `structural_amplitude_eur_mwh` into
  `build_structural_monthly_shape_prior_from_history`.
- Added `structural_prior_summary` to the monthly authority manifest.
- Moved fallback reason ownership into the structural prior builder. Template
  fallback diagnostics now carry:
  - `fallback_reason` = `empty_history`, `no_month_cal_history`, or
    `insufficient_history`;
  - per-month `n_history`.
- Avoided `RuntimeWarning` when parent residual diagnostics are all NaN by
  reporting a zero max residual for that degenerate diagnostic case.
- Enriched the synthetic monthly parity fixture with:
  - `active_config_hash`;
  - `structural_status`;
  - `fused_status`;
  - full `structural_prior_summary`.

Frozen fixture values:

- `monthly_solution_hash`:
  `5902cdb14fc6190c1aa40fb35eab968cea22f899dcd83bc8e28e0c262b02eb37`
- `active_config_hash`:
  `19e36cbce0011696531c2b024b208f509d67426f1734f5c30c3056c9429b1d62`

Runtime config/grid match:

```text
active_config_hash 75e2d1f117db0546672008a0062ec13052b939563e98ace520ebf1a63b5360cc
candidate_count 901
matching_candidates 1
matched_hash 75e2d1f117db0546672008a0062ec13052b939563e98ace520ebf1a63b5360cc
```

## Tests

Focused post-roast tests:

```powershell
python -m pytest tests/test_monthly_forward_curve_priors.py::test_structural_template_fallback_reports_insufficient_history_reason tests/test_monthly_curve_lambda_calibration.py::test_config_hash_includes_structural_prior_knobs tests/test_monthly_curve_lambda_calibration.py::test_candidate_config_hash_uses_canonical_structural_payload tests/test_monthly_forward_curve_integration.py::test_monthly_authority_manifest_records_structural_template_summary tests/test_monthly_forward_curve_integration.py::test_monthly_authority_active_config_hash_includes_structural_prior_knobs -q
```

Result: `5 passed in 1.83s`.

Broader focused suite:

```powershell
python -m pytest tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_check_monthly_curve_promotion_from_manifests.py -q
```

Initial result after hash/fallback fixes: `60 passed, 1 warning in 4.07s`.

After removing the warning:

```powershell
python -m pytest tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_check_monthly_curve_promotion_from_manifests.py -q
```

Result: `60 passed in 4.04s`.

Post-governance-roast targeted checks:

```powershell
python -m pytest tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_monthly_forward_curve_integration.py::test_monthly_authority_direct_and_history_paths_hash_equal tests/test_check_monthly_curve_promotion_from_manifests.py -q
```

Result: `8 passed in 1.84s`.

Final LT guardrail:

```powershell
python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_lt_ct_imports.py -q
```

Earlier result: `91 passed, 1 skipped in 10.54s`.

Final result after grid/runtime/sparse-proof corrections:
`94 passed, 1 skipped in 27.52s`.

Mechanical audit:

```powershell
git diff --check
```

Result: no whitespace errors; Git reported CRLF working-copy warnings only.

```powershell
rg -n "pfc_shaping\.ct|powerbi/PFC_QA|powerbi/data|pfc_shaping/data/.*\.(parquet|duckdb)|data/epex_hourly\.parquet" .planning/phases/14-lt-audit-remediation/DECISION-LOG.md .planning/phases/14-lt-audit-remediation/lambda_grid.yaml scripts/check_monthly_curve_promotion_from_manifests.py scripts/run_monthly_curve_sparse_year_proof.py pfc_shaping/calibration/monthly_curve_lambda_calibration.py pfc_shaping/calibration/monthly_curve_priors.py pfc_shaping/config.yaml pfc_shaping/pipeline/monthly_curve_authority.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_forward_curve_integration.py tests/test_monthly_forward_curve_priors.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/fixtures/monthly_curve_phase_e_parity_baseline.json
```

Result: no matches.

## Roast Results

Initial functional roast found:

- Blocker: `active_config_hash` was no longer comparable to selected lambda
  artifacts because monthly authority used a local `_sha256_json()` hash while
  lambda artifacts used `config_hash()`.
- High: strong fallback defaults needed explicit selected-artifact/audit
  contract.
- High: `structural_prior_summary.fallback_reasons` inferred
  `empty_history` and could lie for non-empty unsupported history.
- Medium: parity fixture only blessed solution hash drift, not the auditable
  config/status/summary contract.

Initial hygiene roast independently confirmed:

- active/selected hash mismatch risk;
- fallback reason audit gap;
- missing fixture assertions for `active_config_hash`,
  `structural_status`, `fused_status`, and `structural_prior_summary`.

Corrections made:

- One canonical `config_hash()` contract for production manifest and selected
  lambda artifacts.
- Canonical payload includes structural fallback knobs and weights.
- Fallback reasons and `n_history` are emitted by prior diagnostics.
- Fixture and tests freeze active hash and manifest structural summary.
- Calibration settings, candidate config defaults and `lambda_grid.yaml` are
  aligned with runtime defaults.

Final read-only roast verdict:

> No blockers/high findings found. Readiness verdict: ready from read-only diff
> review. The four requested fixes are present: shared `config_hash` contract,
> true fallback diagnostics, frozen parity/test fields, and scoped uncommitted
> LT/calibration/planning/tests changes only.

Additional expert roasts requested after that verdict:

- Mathematical/solver roast: no blockers/high. Residual medium risks were that
  the template fallback is a strong default, insufficient month support falls
  back for all structural months, and amplitude/weight are fixed calibration
  settings rather than searched grid dimensions.
- Governance roast found a blocker: runtime `active_config_hash` had no
  matching `lambda_grid.yaml` candidate because the grid missed
  `lambda_smooth_yoy=0.25` and used `constraint_tolerance=0.01`. It also found
  a high issue: promotion checker fallback hashing still rebuilt the old narrow
  payload.
- Test/maintainability roast found no blockers/high. It flagged a medium risk
  in `run_monthly_curve_sparse_year_proof.py`, where fallback hashing and CLI
  defaults could recreate the old mismatch class.

Corrections after additional roasts:

- `lambda_grid.yaml` now includes the runtime tuple and exactly one grid
  candidate matches runtime `active_config_hash`.
- Promotion checker fallback uses `config_hash(solver_config)`.
- Sparse-year proof fallback active hash includes structural CLI knobs and
  defaults are aligned with runtime.
- Targeted tests and full guardrail pass after these changes.

## Current Status

```text
## clean/solver-structural-prior...origin/fix/lt-audit-remediation
 M .planning/phases/14-lt-audit-remediation/DECISION-LOG.md
 M .planning/phases/14-lt-audit-remediation/lambda_grid.yaml
 M pfc_shaping/calibration/monthly_curve_lambda_calibration.py
 M pfc_shaping/calibration/monthly_curve_priors.py
 M pfc_shaping/config.yaml
 M pfc_shaping/pipeline/monthly_curve_authority.py
 M scripts/check_monthly_curve_promotion_from_manifests.py
 M scripts/run_monthly_curve_sparse_year_proof.py
 M tests/fixtures/monthly_curve_phase_e_parity_baseline.json
 M tests/test_check_monthly_curve_promotion_from_manifests.py
 M tests/test_monthly_curve_lambda_calibration.py
 M tests/test_monthly_forward_curve_integration.py
 M tests/test_monthly_forward_curve_priors.py
 M tests/test_run_monthly_curve_sparse_year_proof_script.py
?? .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260623-SOLVER-STRUCTURAL-PRIOR.md
```

## Codex Audit - 2026-06-23

Scope: read-only roast of the uncommitted `clean/solver-structural-prior`
patch after locating the correct worktree. The original `PFC_LT` worktree was
on `fix/lt-audit-remediation`; the audited worktree was:

```text
\\fmvfs2\Data\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_solver_structural_prior
```

Audit findings:

- No blocker or high-severity issue found in the solver structural-prior patch.
- `active_config_hash` now uses the shared `config_hash(settings)` payload and
  covers structural fallback activation, amplitude, prior weights, structural
  snapshot support, history lookback, robust panel quantile and neighbor
  markets.
- Runtime defaults and `lambda_grid.yaml` still have exactly one matching
  candidate hash.
- Structural fallback diagnostics expose source, fallback reason, amplitude,
  per-month `n_history`, zero-mean parent-space flag and parent residual
  summary in the monthly authority manifest.
- Promotion checker fallback hashing uses the canonical config hash instead of
  rebuilding the old narrow monthly config payload.
- Sparse-year proof fallback hashing includes CLI structural knobs and aligned
  runtime defaults.
- No LT import from `pfc_shaping.ct.*` was found in the touched code paths.
- No Power BI files or heavy data files were modified in this worktree.

Commands and results:

```powershell
git status --short --branch
```

Result: branch `clean/solver-structural-prior` tracking
`origin/fix/lt-audit-remediation`; same modified/untracked files listed above.

```powershell
python -m pytest tests/test_monthly_forward_curve_priors.py tests/test_monthly_forward_curve_integration.py tests/test_monthly_forward_curve_solver.py tests/test_monthly_curve_lambda_calibration.py tests/test_monthly_curve_promotion.py tests/test_run_monthly_curve_sparse_year_proof_script.py tests/test_check_monthly_curve_promotion_from_manifests.py tests/test_lt_ct_imports.py -q
```

Result: `94 passed, 1 skipped in 80.40s`.

```powershell
git diff --check
```

Result: no whitespace errors; Git emitted CRLF working-copy warnings only.

```powershell
@'
from pathlib import Path
from pfc_shaping.calibration.monthly_curve_lambda_calibration import load_lambda_grid, iter_candidate_configs, config_hash
from pfc_shaping.pipeline.monthly_curve_authority import DEFAULT_MONTHLY_SOLVER_CONFIG

grid = load_lambda_grid(Path('.planning/phases/14-lt-audit-remediation/lambda_grid.yaml'))
runtime_hash = config_hash(DEFAULT_MONTHLY_SOLVER_CONFIG)
hashes = [config_hash(c) for c in iter_candidate_configs(grid)]
print('runtime_hash', runtime_hash)
print('candidate_count', len(hashes))
print('matching_candidates', sum(h == runtime_hash for h in hashes))
print('first_match_index', next((i for i,h in enumerate(hashes) if h == runtime_hash), None))
'@ | python -
```

Result:

```text
runtime_hash 75e2d1f117db0546672008a0062ec13052b939563e98ace520ebf1a63b5360cc
candidate_count 901
matching_candidates 1
first_match_index 456
```

```powershell
rg -n "pfc_shaping\.ct|from pfc_shaping\.model|import pfc_shaping\.model|powerbi/PFC_QA|powerbi/data|pfc_shaping/data/.*\.(parquet|duckdb)|data/epex_hourly\.parquet|data/eex_forwards_history\.parquet" <touched-files>
```

Result: only text references to `data/eex_forwards_history.parquet` in config,
script defaults and tests; no heavy data or Power BI modification.

## Residual Risks

- This branch improves solver prior auditability and hash governance; it does
  not solve Phase 4 cross-year Q4 comparability.
- It does not solve the structural fan-chart width problem.
- Template fallback is now an active soft solver prior by default. Promotion
  still requires regenerated production/export/selected-lambda manifests and
  strict gates.
- Branch tracking points at `origin/fix/lt-audit-remediation`; confirm the push
  target before any future commit/push.

## Next Steps

1. Commit this solver-prior branch separately if accepted.
2. Resume Phase 4 in
   `h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase4_q4`
   on `clean/phase4-cross-year-q4`.
3. Implement comparable-block Q4/cross-year audit and solver fixes without
   touching CT, Power BI files, or heavy data.
4. Regenerate candidate solver ON + PEAK calibration only after Phase 4 and
   structural width work are both ready for strict promotion evidence.
