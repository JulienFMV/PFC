# Session Handoff - LT Product Normalization Audit - 2026-06-22

## Scope

Worktree:
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_lt_product_audit`

Branch: `clean/lt-product-normalization-audit`

HEAD at start/end of implementation: `c7e8ab657a8aa115e1e40de98de33fa3c3074db5`

Goal: implement Phase P1 read-only audit of CH LT product normalization on a
delivered hourly curve against traceable EEX CH forwards.

No CT, Power BI, desk parquet/duckdb data, production flag, commit, reset,
checkout, clean, or global stash operation was performed.

## Initial Guard Commands

```powershell
git rev-parse --show-toplevel
```

Output:

```text
//fmvfs2/Data/Energy/GeCom/CONTROLLING RISK/Analyses diverses/Python - JB/PFC_LT_clean_lt_product_audit
```

```powershell
git branch --show-current
```

Output:

```text
clean/lt-product-normalization-audit
```

```powershell
git status --short --branch
```

Output before modification:

```text
## clean/lt-product-normalization-audit...origin/fix/lt-audit-remediation
```

```powershell
git worktree list
```

Output:

```text
//fmvfs2/Data/Energy/GeCom/CONTROLLING RISK/Analyses diverses/Python - JB/PFC_LT                        c7e8ab6 [fix/lt-audit-remediation]
//fmvfs2/Data/Energy/GeCom/CONTROLLING RISK/Analyses diverses/Python - JB/PFC_LT_clean_lt_product_audit c7e8ab6 [clean/lt-product-normalization-audit]
//fmvfs2/Data/Energy/GeCom/CONTROLLING RISK/Analyses diverses/Python - JB/PFC_phase10                   6a77c87 [claude/clean-lt-ct-integration]
```

`AGENTS.md` was read. `CLAUDE.md` was read and only points to `AGENTS.md`; it
does not duplicate or contradict the root contract.

## Files Changed

- `docs/lt_monthly_curve_program/phase-p1-product-normalization-audit.md`
  - Short P1-specific usage and evidence note.
- `scripts/audit_ch_product_normalization.py`
  - New read-only CLI and importable audit function.
  - Requires explicit CSV, forwards source, selected manifest/artifact, and
    tolerance. Missing manifest is a `CRITICAL` gate.
  - Records input hashes, manifest hash, and selected forwards snapshot date.
  - Validates UTC-aware hourly timestamps, duplicate/gap/monotonicity gates.
  - Checks direct BASE/PEAK/OFFPEAK means on exact product windows.
  - Computes implied OFFPEAK from same-product BASE+PEAK using exact
    Europe/Zurich hour counts.
  - Builds Month > Quarter > Calendar quote-aware residual buckets.
  - Fails closed on conflicting duplicate quotes, empty evidence, missing PEAK
    quotes, missing files, absent products, and partial product windows.
  - Writes Markdown/JSON/CSV outputs before returning non-zero unless
    `--allow-failed-gates` is explicitly passed.
- `tests/test_audit_ch_product_normalization_script.py`
  - Synthetic-only tests for required P1 gates.
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-lt-product-normalization-audit.md`
  - This handoff.

## Commands Run

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py -q
```

First run result:

```text
...F...F.F                                                               [100%]
3 failed, 7 passed in 15.13s
```

Failures found and fixed:

- Markdown report used `DataFrame.to_markdown`, which required missing optional
  dependency `tabulate`.
- Timestamp duplicate/gap gates were emitted, but downstream peak-mask checks
  still ran on invalid indexes.
- One CAL+Q1 test fixture did not satisfy BASE and PEAK means simultaneously.

Second focused run after adding missing-input fail-closed coverage:

```text
...........                                                              [100%]
11 passed in 3.75s
```

Read-only roast then found and fixed:

- missing manifest could pass silently;
- duplicate exact-quote message incorrectly referenced tolerance;
- DST/leap product-window tests were too circular;
- PEAK calendar tests were too circular.

Final run:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py -q
```

Final output:

```text
..............                                                           [100%]
14 passed in 3.33s
```

## Test Coverage

Covered synthetic cases:

- complete PASS for BASE/PEAK/OFFPEAK;
- broken PEAK direct quote gives `CRITICAL`;
- direct parent quote failure is not hidden by a passing residual bucket;
- CAL+Q1 residual bucket is explicitly reported;
- partial window gives `UNSUPPORTED`;
- empty evidence gives `CRITICAL`;
- missing CSV/forwards evidence writes outputs and gives `CRITICAL`;
- missing selected manifest gives `CRITICAL`;
- missing PEAK quote on a covered BASE product gives `UNSUPPORTED`;
- timezone-naive UTC column, duplicate timestamp, and gap give `CRITICAL`;
- conflicting duplicate quote gives `CRITICAL`;
- fixed DST/leap hour counts for 2028-02, 2028-03, 2028-10, and 2028 CAL;
- fixed EEX CH PEAK checks for Aug 1 CH holiday and local 08:00-20:00
  boundaries;
- CLI writes outputs then returns non-zero on failed gates.

## Artifacts And Hashes

No real external audit was launched in this session.

No production/local/lambda manifests were consumed. No real CSV or forwards
hashes were generated beyond temporary pytest fixtures under the test temp
directory.

The script will record `csv_sha256`, `forwards_sha256`,
`forward_snapshot_date`, and `manifest_sha256` when run on explicit real
inputs.

## Gate Status

Real-run gate statuses: not applicable; no real audit run was executed.

Known synthetic gate statuses are asserted in
`tests/test_audit_ch_product_normalization_script.py`.

## Decisions

Decision: CLI requires `--tolerance-eur-mwh`.

Reason: tolerance must be explicit; no hidden default should turn a hard product
normalization miss into a pass.

Rejected alternative: defaulting to a small tolerance in the CLI.

Invariant: failed gates remain `CRITICAL`/`UNSUPPORTED`; diagnostic mode changes
only process exit code.

Decision: timestamp integrity gates block downstream product checks.

Reason: duplicate, non-monotone, gap, or non-UTC timestamps invalidate the
calendar/product masks, and continuing can raise misleading secondary errors.

Rejected alternative: sort/deduplicate/reindex inside the audit.

Invariant: the delivered CSV is read-only evidence and must not be repaired by
the audit.

Decision: duplicate quote prices must be strictly identical to be documented as
exact duplicates.

Reason: conflicting same as-of/product/load_type evidence is a source problem,
not a residual-tolerance problem.

Rejected alternative: using the EUR/MWh residual tolerance for duplicate quote
classification.

Invariant: any price variation across duplicate quote rows is `CRITICAL`.

Decision: selected manifest/artifact is required evidence for a PASS-capable
audit.

Reason: P1 evidence must identify the delivered curve, the forwards source, and
the selected manifest/artifact when judging delivered production-style outputs.

Rejected alternative: treating an omitted manifest as informational metadata.

Invariant: omitted or missing manifest remains a `CRITICAL` gate; the audit may
still produce a diagnostic report.

## Risks

- The audit supports EEX product keys `YYYY`, `YYYY-Qn`, and `YYYY-MM`; other
  product formats are `UNSUPPORTED`.
- The manifest is not discovered automatically. If omitted or missing, the
  audit reports `CRITICAL` rather than scanning broad output/data trees.
- No real delivered CSV or desk forwards file was read during this session, so
  external evidence compatibility remains to be proven by a real diagnostic run.

## Next Steps

1. Run the new CLI on explicitly selected real CSV, real EEX CH forwards source,
   and selected manifest/artifact.
2. Review the emitted Markdown/JSON/CSV gates.
3. Treat any `CRITICAL` or `UNSUPPORTED` as blocking evidence unless explicitly
   documented as accepted diagnostic scope.

## Continuation - 2026-06-22 Codex

Scope held to Phase P1 CH LT product normalization audit in worktree
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_lt_product_audit`
on branch `clean/lt-product-normalization-audit`.

Read before action:

- `AGENTS.md`
- `CLAUDE.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-lt-product-normalization-audit.md`
- `docs/lt_monthly_curve_program/phase-p1-product-normalization-audit.md`
- `scripts/audit_ch_product_normalization.py`
- `tests/test_audit_ch_product_normalization_script.py`

Guard/status commands:

```powershell
git rev-parse --show-toplevel
git branch --show-current
git status --short --branch
```

Observed status:

```text
//fmvfs2/Data/Energy/GeCom/CONTROLLING RISK/Analyses diverses/Python - JB/PFC_LT_clean_lt_product_audit
clean/lt-product-normalization-audit
## clean/lt-product-normalization-audit...origin/fix/lt-audit-remediation
?? .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-lt-product-normalization-audit.md
?? docs/lt_monthly_curve_program/
?? scripts/audit_ch_product_normalization.py
?? tests/test_audit_ch_product_normalization_script.py
```

Verification command:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py -q
```

Output:

```text
..............                                                           [100%]
14 passed in 12.04s
```

Real-run evidence availability check, using path metadata only:

```powershell
@('output/local_test_ch_pfc_hourly_20260613_20301231.csv','output/local_test_ch_pfc_hourly_20260613_20301231.monthly_curve_manifest.json','data/eex_forwards_history.parquet') | ForEach-Object { [pscustomobject]@{Path=$_; Exists=(Test-Path -LiteralPath $_); Length=if (Test-Path -LiteralPath $_) {(Get-Item -LiteralPath $_).Length} else {$null}} } | Format-Table -AutoSize
```

Output:

```text
Path                                                                          Exists Length
----                                                                          ------ ------
output/local_test_ch_pfc_hourly_20260613_20301231.csv                          False
output/local_test_ch_pfc_hourly_20260613_20301231.monthly_curve_manifest.json  False
data/eex_forwards_history.parquet                                               True 444766
```

Conclusion: no PASS-capable real P1 audit run was launched in this continuation.
The explicit delivered hourly CSV and selected manifest/artifact required by
the P1 evidence rules are absent from the clean worktree. The tracked
`data/eex_forwards_history.parquet` exists, but it was not read; only file
metadata was inspected. No CT, Power BI, heavy desk parquet/duckdb data,
production flag, commit, reset, checkout, clean, or stash operation was
performed.

Remaining next step: provide or generate an explicit delivered hourly CH LT CSV
and its selected manifest/artifact in this worktree, then run
`scripts/audit_ch_product_normalization.py` with `--allow-failed-gates` only if
the run is diagnostic.

## Continuation - 2026-06-22 P1 Real Evidence And Remediation

User asked to continue with the next step. Scope remained Phase P1 CH product
normalization only. No CT, Power BI refresh, production flag, commit, reset,
checkout, clean, global stash, or Power BI data edit was performed.

### Failed Clean Export Attempt

Attempted to generate a fresh clean-worktree solver export:

```powershell
python scripts/export_local_test_ch_hourly_csv.py --local-start-date 2026-06-13 --local-end-date 2030-12-31 --output output/p1_product_norm_ch_hfc_hourly_20260613_20301231.csv --report .planning/phases/14-lt-audit-remediation/P1-PRODUCT-NORM-EXPORT-20260622.md --prefix p1_product_norm_20260622 --forwards data/eex_forwards_history.parquet --required-forward-date 2026-06-17 --enable-monthly-forward-curve-solver --skip-powerbi-refresh
```

Result: failed before generation because the clean worktree lacks
`data/electrification_scenarios_prod_candidate_neutralized_2030.parquet`.
No CSV/manifest was written by this command.

### Selected Real Artifact From Main Worktree

Because the clean worktree had no delivered output, selected the latest matching
main-worktree local-test solver artifact:

- CSV:
  `h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\ch_hfc_hourly_20260622_20321231_monthly_solver_lambda110_peakcal.csv`
- Manifest:
  `h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.monthly_curve_manifest.json`
- Forwards:
  `h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet`

The selected manifest records:

```text
forward_snapshot_date: 2026-06-19
monthly_level_authority: solver
skip_legacy_level_cascade: true
skip_legacy_base_smoothing: true
monthly_solution_hash: 9c8bc0cda233a463680eb3df0a795911e876fd6b3121bb7d433bad2af7eec5b1
active_constraints_hash: 554cfae0e419da72a190cdfb9ce4db9149383abcda1e57ec6d3ee5a036a62c18
source_hashes.forwards_path: f5a2f1dc2c5c8f2215e8c59137f7aab8c163eb5bab83e4755928fd2b9deb8c4f
```

The clean worktree and main worktree `data/eex_forwards_history.parquet` hashes
differ. The audit therefore used the main-worktree forwards path to match the
selected manifest.

### Original Delivered Artifact Audit

Command:

```powershell
python scripts/audit_ch_product_normalization.py --csv "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\ch_hfc_hourly_20260622_20321231_monthly_solver_lambda110_peakcal.csv" --forwards "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet" --manifest "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.monthly_curve_manifest.json" --as-of 2026-06-19 --tolerance-eur-mwh 0.000001 --report .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260622-LAMBDA110-PEAKCAL.md --gates-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-GATES-20260622-LAMBDA110-PEAKCAL.csv --json-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260622-LAMBDA110-PEAKCAL.json --allow-failed-gates
```

Output:

```text
[product-normalization-audit] pass=0 critical=1 unsupported=0
```

Gate:

```text
CRITICAL timestamp_utc_timezone: timestamp column contains timezone-naive values, row_count=57241
```

Evidence hashes:

```text
csv_sha256: 5da6da30e9efe3145710582c71e58e1a4d43827645f8c682d69a1c932e498b9a
forwards_sha256: f5a2f1dc2c5c8f2215e8c59137f7aab8c163eb5bab83e4755928fd2b9deb8c4f
manifest_sha256: cc6b85255d361f0eac62f830a77834cc8c63b09860535c82be4bcbfe377701a1
forward_snapshot_date: 2026-06-19
```

Conclusion: the original delivered artifact is not PASS-capable because
`timestamp_utc` is timezone-naive.

### Code Changes

Files changed in this continuation:

- `scripts/export_local_test_ch_hourly_csv.py`
  - `timestamp_utc` now exports as UTC-aware text:
    `%Y-%m-%dT%H:%M:%S%z`.
- `scripts/audit_ch_product_normalization.py`
  - Added `source_quote_parent_child_consistency` gates for complete
    Month/Quarter/Calendar parent-child quote sets.
  - Added `partial_bucket_window` handling so quote-aware direct buckets are
    not incorrectly scored as price residuals when the delivered CSV only
    partially covers the product.
- `tests/test_export_local_test_ch_hourly_csv_script.py`
  - Added assertion for UTC-aware export timestamp.
- `tests/test_audit_ch_product_normalization_script.py`
  - Added source quote parent-child inconsistency coverage.
  - Strengthened partial-window coverage to assert no `CRITICAL`.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added UTC-aware export timestamp decision.
  - Added source quote parent/child consistency gate decision.
- This handoff.

### Remediation Export From Selected Fan Chart

Generated a remediation CSV from the already-selected main-worktree solver fan
chart, without rebuilding the model and without refreshing Power BI:

```powershell
python scripts/export_local_test_ch_hourly_csv.py --local-start-date 2026-06-22 --local-end-date 2032-12-31 --output output/p1_product_norm_ch_hfc_hourly_20260622_20321231_lambda110_peakcal_tzfixed.csv --report .planning/phases/14-lt-audit-remediation/P1-PRODUCT-NORM-EXPORT-20260622-LAMBDA110-PEAKCAL-TZFIXED.md --prefix p1_product_norm_lambda110_peakcal_tzfixed --forwards "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet" --required-forward-date 2026-06-19 --skip-build --fan-chart-output "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.parquet" --enable-eex-peak-calibration --skip-powerbi-refresh
```

Output:

```text
[hourly-csv] rows=57241
[hourly-csv] output -> output\p1_product_norm_ch_hfc_hourly_20260622_20321231_lambda110_peakcal_tzfixed.csv
[hourly-csv] report -> .planning/phases/14-lt-audit-remediation/P1-PRODUCT-NORM-EXPORT-20260622-LAMBDA110-PEAKCAL-TZFIXED.md
```

This remediation CSV is diagnostic/regenerated evidence, not the original
delivered artifact.

### Remediation CSV Audit

Command:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/p1_product_norm_ch_hfc_hourly_20260622_20321231_lambda110_peakcal_tzfixed.csv --forwards "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet" --manifest "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.monthly_curve_manifest.json" --as-of 2026-06-19 --tolerance-eur-mwh 0.000001 --report .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260622-LAMBDA110-PEAKCAL-TZFIXED.md --gates-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-GATES-20260622-LAMBDA110-PEAKCAL-TZFIXED.csv --json-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260622-LAMBDA110-PEAKCAL-TZFIXED.json --allow-failed-gates
```

Final output after audit-script improvements:

```text
[product-normalization-audit] pass=96 critical=15 unsupported=6
```

Artifact hashes:

```text
csv_sha256: 47efd6d886ddf58fb548ff0a1fafb5c576a855d44702b46b41df2b54e2dea16c
forwards_sha256: f5a2f1dc2c5c8f2215e8c59137f7aab8c163eb5bab83e4755928fd2b9deb8c4f
manifest_sha256: cc6b85255d361f0eac62f830a77834cc8c63b09860535c82be4bcbfe377701a1
forward_snapshot_date: 2026-06-19
```

Gate summary:

```text
48 PASS       quote_aware_bucket_mean
48 PASS       direct_quote_mean
9 CRITICAL    direct_quote_mean
6 CRITICAL    source_quote_parent_child_consistency
3 UNSUPPORTED partial_product_window
3 UNSUPPORTED partial_bucket_window
```

Source quote conflicts in selected snapshot:

```text
BASE 2026-Q3 target=98.35 child_weighted=98.34858695652174 residual=-0.0014130434782515522
BASE 2026-Q4 target=125.06 child_weighted=125.061607062019 residual=0.0016070620189907459
BASE 2027 target=94.2 child_weighted=94.19946575342466 residual=-0.0005342465753415127
PEAK 2026-Q3 target=95.4 child_weighted=95.40151515151516 residual=0.0015151515151501371
PEAK 2026-Q4 target=138.76 child_weighted=138.65753846153848 residual=-0.10246153846151174
PEAK 2027 target=97.07 child_weighted=97.06023166023166 residual=-0.00976833976832836
```

Direct quote criticals match the source conflicts plus implied OFFPEAK:

```text
BASE 2026-Q3 residual=-0.001413052083336197
BASE 2026-Q4 residual=0.0016070502489782257
BASE 2027 residual=-0.0005342344748839878
PEAK 2026-Q3 residual=0.0015151338383674329
PEAK 2026-Q4 residual=-0.1024615410256331
PEAK 2027 residual=-0.009768296975551038
OFFPEAK 2026-Q3 residual=-0.00305085098871416
OFFPEAK 2026-Q4 residual=0.05841145976208395
OFFPEAK 2027 residual=0.004543519639071292
```

Unsupported gates are all due to the delivered/remediation window starting on
2026-06-22 while the selected snapshot contains 2026-06 products:

```text
partial_product_window: BASE/PEAK/OFFPEAK 2026-06, actual product rows=216, expected base rows=720
partial_bucket_window: BASE 2026-06 actual=216 expected=720
partial_bucket_window: PEAK 2026-06 actual=84 expected=264
partial_bucket_window: OFFPEAK 2026-06 actual=132 expected=456
```

### Tests

Commands:

```powershell
python -m pytest tests/test_export_local_test_ch_hourly_csv_script.py::test_to_hourly_csv_frame_filters_local_window_and_averages tests/test_audit_ch_product_normalization_script.py -q
python -m pytest tests/test_audit_ch_product_normalization_script.py tests/test_export_local_test_ch_hourly_csv_script.py::test_to_hourly_csv_frame_filters_local_window_and_averages -q
```

Outputs:

```text
...............                                                          [100%]
15 passed in 3.38s

................                                                         [100%]
16 passed in 2.95s
```

### Current P1 Status

P1 is not PASS for the selected delivered artifact.

Blocking findings:

1. Original delivered CSV has timezone-naive `timestamp_utc`.
2. After timestamp remediation, product checks expose selected-snapshot source
   quote conflicts for overlapping CH parent/child products.
3. June 2026 products are only partially covered because the selected CSV
   starts on 2026-06-22.

Next remediation should not patch individual months. Recommended next steps:

1. Regenerate the selected export with UTC-aware `timestamp_utc`.
2. Decide the canonical quote hierarchy/evidence policy for inconsistent
   overlapping EEX quotes. If Month > Quarter > Calendar is authoritative, keep
   parent/child conflicts as source gates and ensure the manifest lists the
   active/dropped quote rows.
3. Run P1 on a full-product delivery window or explicitly document June 2026
   partial-window `UNSUPPORTED` as diagnostic-only.

## Continuation - 2026-06-22 Active Quote Set Policy

Implemented the active quote set policy proposed after reviewing the EEX source
conflicts.

Decision implemented:

- Direct product checks now use the active quote set under
  `Month > Quarter > Calendar`.
- A parent with a complete finer child set is dropped from direct checks and
  emitted as `active_quote_set_parent_dropped`.
- Dropped parents include `dropped_reason`, `child_products`, target,
  child-weighted implied value, residual and tolerance.
- `parent_child_conflict` remains `CRITICAL`.
- Quote-aware bucket checks use the same active quote set as direct checks.

Files changed in this continuation:

- `scripts/audit_ch_product_normalization.py`
  - Added `complete_child_products(...)` and `active_quote_set(...)`.
  - Direct checks, implied OFFPEAK checks and quote-aware bucket checks now use
    `active_forwards`.
  - Metadata now records `active_forward_quote_rows` and
    `dropped_forward_quote_rows`.
- `tests/test_audit_ch_product_normalization_script.py`
  - Updated parent/child conflict test to assert the parent is dropped from
    direct checks while finer month products remain checked.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added `D-20260622-07 - P1 Active Quote Set Direct Checks`.
- `docs/lt_monthly_curve_program/phase-p1-product-normalization-audit.md`
  - Documented active quote set direct checks and dropped parent evidence.
- This handoff.

Tests:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py tests/test_export_local_test_ch_hourly_csv_script.py::test_to_hourly_csv_frame_filters_local_window_and_averages -q
```

Output:

```text
................                                                         [100%]
16 passed in 2.97s
```

### Re-audit Of 2026-06-22 Start Remediation CSV

Command rerun:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/p1_product_norm_ch_hfc_hourly_20260622_20321231_lambda110_peakcal_tzfixed.csv --forwards "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet" --manifest "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.monthly_curve_manifest.json" --as-of 2026-06-19 --tolerance-eur-mwh 0.000001 --report .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260622-LAMBDA110-PEAKCAL-TZFIXED.md --gates-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-GATES-20260622-LAMBDA110-PEAKCAL-TZFIXED.csv --json-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260622-LAMBDA110-PEAKCAL-TZFIXED.json --allow-failed-gates
```

Output:

```text
[product-normalization-audit] pass=96 critical=6 unsupported=6
```

Interpretation:

- Direct active quote checks are no longer duplicated as unavoidable parent
  residual failures.
- The 6 `CRITICAL` rows are only
  `active_quote_set_parent_dropped` with `dropped_reason=parent_child_conflict`.
- The 6 `UNSUPPORTED` rows are due to 2026-06 products being only partially
  covered by a CSV that starts 2026-06-22.

### Diagnostic Full-Month Start From 2026-07-01

Generated a diagnostic CSV from the same selected fan chart, without rebuilding
the model and without refreshing Power BI:

```powershell
python scripts/export_local_test_ch_hourly_csv.py --local-start-date 2026-07-01 --local-end-date 2032-12-31 --output output/p1_product_norm_ch_hfc_hourly_20260701_20321231_lambda110_peakcal_tzfixed.csv --report .planning/phases/14-lt-audit-remediation/P1-PRODUCT-NORM-EXPORT-20260701-LAMBDA110-PEAKCAL-TZFIXED.md --prefix p1_product_norm_lambda110_peakcal_tzfixed_july --forwards "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet" --required-forward-date 2026-06-19 --skip-build --fan-chart-output "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.parquet" --enable-eex-peak-calibration --skip-powerbi-refresh
```

Output:

```text
[hourly-csv] rows=57025
[hourly-csv] output -> output\p1_product_norm_ch_hfc_hourly_20260701_20321231_lambda110_peakcal_tzfixed.csv
[hourly-csv] report -> .planning/phases/14-lt-audit-remediation/P1-PRODUCT-NORM-EXPORT-20260701-LAMBDA110-PEAKCAL-TZFIXED.md
```

Audit command:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/p1_product_norm_ch_hfc_hourly_20260701_20321231_lambda110_peakcal_tzfixed.csv --forwards "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet" --manifest "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.monthly_curve_manifest.json" --as-of 2026-06-19 --tolerance-eur-mwh 0.000001 --report .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260701-LAMBDA110-PEAKCAL-TZFIXED.md --gates-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-GATES-20260701-LAMBDA110-PEAKCAL-TZFIXED.csv --json-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260701-LAMBDA110-PEAKCAL-TZFIXED.json --allow-failed-gates
```

Output:

```text
[product-normalization-audit] pass=96 critical=6 unsupported=3
```

Gate summary:

```text
48 PASS       direct_quote_mean
48 PASS       quote_aware_bucket_mean
6 CRITICAL    active_quote_set_parent_dropped
3 UNSUPPORTED quoted_product_absent
```

Metadata:

```text
csv_sha256: f926e2ce8cefe238b708c75ee7a90189a426fbed6e229eee88d833ab9f381682
forwards_sha256: f5a2f1dc2c5c8f2215e8c59137f7aab8c163eb5bab83e4755928fd2b9deb8c4f
manifest_sha256: cc6b85255d361f0eac62f830a77834cc8c63b09860535c82be4bcbfe377701a1
forward_quote_rows: 40
active_forward_quote_rows: 34
dropped_forward_quote_rows: 6
forward_snapshot_date: 2026-06-19
```

Remaining non-PASS rows:

- `CRITICAL active_quote_set_parent_dropped`:
  - BASE 2026-Q3 residual `-0.0014130434782515522`
  - BASE 2026-Q4 residual `0.0016070620189907459`
  - BASE 2027 residual `-0.0005342465753415127`
  - PEAK 2026-Q3 residual `0.0015151515151501371`
  - PEAK 2026-Q4 residual `-0.10246153846151174`
  - PEAK 2027 residual `-0.00976833976832836`
- `UNSUPPORTED quoted_product_absent`:
  - BASE/PEAK/OFFPEAK 2026-06, because the diagnostic CSV starts 2026-07-01
    while the selected snapshot contains 2026-06 quotes.

Current status:

- The selected curve satisfies all direct active quote checks and quote-aware
  bucket checks for covered active products at `1e-6` EUR/MWh.
- P1 remains not PASS because parent/child quote conflicts are intentionally
  `CRITICAL`, and 2026-06 is outside the July diagnostic window.

Next step:

- Decide governance status for `parent_child_conflict`: either keep it
  promotion-blocking, or introduce an explicitly approved waiver/accepted
  status only when a complete finer child set exists and the parent is dropped
  from the active quote set.
- Separately, run with a source snapshot/window contract that excludes products
  outside the delivered window, or treat absent 2026-06 as diagnostic-only.

## Continuation - 2026-06-23 CSV Window Scope Contract

Context: session resumed with workspace root pointing at the main
`PFC_LT` worktree, which had many unrelated dirty files including Power BI and
heavy data artifacts. Work continued in the clean P1 worktree:
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_lt_product_audit`.

Read before action:

- `AGENTS.md`
- `CLAUDE.md`
- this handoff
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`

Implemented explicit CSV-window scoping for P1:

- New `audit(..., scope_to_csv_window=False)` parameter.
- New CLI flag: `--scope-forwards-to-csv-window`.
- Default remains fail-closed.
- When the flag is used, only quoted products with no overlap with the
  delivered CSV window are excluded and emitted as `INFO out_of_scope_quote`.
- Partial products remain `UNSUPPORTED`.
- `status_counts` and Markdown/JSON output now include `INFO`.

Files changed in this continuation:

- `scripts/audit_ch_product_normalization.py`
  - Added `scope_forwards_to_csv_window(...)`.
  - Added `scope_to_csv_window` metadata.
  - Added `out_of_scope_forward_quote_rows` and `scoped_forward_quote_rows`
    metadata when scoping is used.
  - Added CLI flag `--scope-forwards-to-csv-window`.
  - Reports `INFO` counts.
- `tests/test_audit_ch_product_normalization_script.py`
  - Added synthetic test proving no-overlap quotes are `UNSUPPORTED` by default
    and `INFO out_of_scope_quote` only with explicit scoping.
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
  - Added `D-20260623-01 - Explicit CSV Window Scope For P1`.
- `docs/lt_monthly_curve_program/phase-p1-product-normalization-audit.md`
  - Documented the scoping option and its limits.
- This handoff.

Tests:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py tests/test_export_local_test_ch_hourly_csv_script.py::test_to_hourly_csv_frame_filters_local_window_and_averages -q
```

Output:

```text
.................                                                        [100%]
17 passed in 14.39s
```

### Scoped Diagnostic Audit

Command:

```powershell
python scripts/audit_ch_product_normalization.py --csv output/p1_product_norm_ch_hfc_hourly_20260701_20321231_lambda110_peakcal_tzfixed.csv --forwards "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\data\eex_forwards_history.parquet" --manifest "h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\output\local_test_ch_pfc_20260622_20321231_monthly_solver_lambda110_peakcal_structural_fan_chart.monthly_curve_manifest.json" --as-of 2026-06-19 --tolerance-eur-mwh 0.000001 --scope-forwards-to-csv-window --report .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260701-LAMBDA110-PEAKCAL-TZFIXED-SCOPED.md --gates-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-GATES-20260701-LAMBDA110-PEAKCAL-TZFIXED-SCOPED.csv --json-output .planning/phases/14-lt-audit-remediation/CH-P1-PRODUCT-NORMALIZATION-AUDIT-20260701-LAMBDA110-PEAKCAL-TZFIXED-SCOPED.json --allow-failed-gates
```

Output:

```text
[product-normalization-audit] pass=96 critical=6 unsupported=0
```

Gate summary:

```text
48 PASS       direct_quote_mean
48 PASS       quote_aware_bucket_mean
6 CRITICAL    active_quote_set_parent_dropped
2 INFO        out_of_scope_quote
```

Metadata:

```text
csv_sha256: f926e2ce8cefe238b708c75ee7a90189a426fbed6e229eee88d833ab9f381682
forwards_sha256: f5a2f1dc2c5c8f2215e8c59137f7aab8c163eb5bab83e4755928fd2b9deb8c4f
manifest_sha256: cc6b85255d361f0eac62f830a77834cc8c63b09860535c82be4bcbfe377701a1
forward_quote_rows: 40
scoped_forward_quote_rows: 38
out_of_scope_forward_quote_rows: 2
active_forward_quote_rows: 32
dropped_forward_quote_rows: 6
scope_to_csv_window: true
forward_snapshot_date: 2026-06-19
```

The two scoped-out quotes are BASE and PEAK `2026-06`, whose product window
ends exactly at the CSV start (`2026-06-30 22:00:00+00:00`). No OFFPEAK source
quote exists; OFFPEAK is implied only after source scoping.

Remaining blockers:

- `CRITICAL active_quote_set_parent_dropped` for conflicting dropped parents:
  - BASE `2026-Q3`, `2026-Q4`, `2027`
  - PEAK `2026-Q3`, `2026-Q4`, `2027`

Current P1 status after this continuation:

- Covered active direct quote checks: PASS.
- Covered active quote-aware bucket checks: PASS.
- Out-of-window June source quotes: INFO only with explicit scoping.
- P1 still not PASS because `parent_child_conflict` remains intentionally
  `CRITICAL` pending governance decision.

## Continuation - 2026-06-23 Pre-Commit Audit / Review / Roast

User requested that future commit workflow always include audit, review and
roast before commit. No commit was performed in this continuation.

Worktree remained:
`h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_lt_product_audit`

Branch remained:
`clean/lt-product-normalization-audit`

### Audit

Commands run:

```powershell
git status --short --branch
git diff --stat
rg "pfc_shaping\.ct|powerbi|PFC_QA|eex_forwards_history\.parquet|epex_hourly\.parquet|\.duckdb" scripts/audit_ch_product_normalization.py tests/test_audit_ch_product_normalization_script.py docs/lt_monthly_curve_program .planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-lt-product-normalization-audit.md .planning/phases/14-lt-audit-remediation/DECISION-LOG.md
python -m compileall scripts/audit_ch_product_normalization.py scripts/export_local_test_ch_hourly_csv.py
git diff --check
```

Findings:

- No CT import was introduced.
- No Power BI file was modified.
- No heavy data file was modified.
- Mentions of `data/eex_forwards_history.parquet` are documentation/evidence
  references in the handoff and P1 usage doc, not file edits.
- `git diff --check` reported no whitespace errors; PowerShell displayed only
  existing line-ending normalization warnings for tracked text files.

### Review

Additional test command:

```powershell
python -m pytest tests/test_audit_ch_product_normalization_script.py tests/test_export_local_test_ch_hourly_csv_script.py::test_to_hourly_csv_frame_filters_local_window_and_averages tests/test_lt_ct_imports.py -q
```

Output:

```text
.................................s.                                      [100%]
34 passed, 1 skipped in 58.63s
```

Reviewed generated reports without rereading the heavy forwards parquet.
Evidence status remains:

- Original delivered artifact:
  - `PASS=0`, `CRITICAL=1`, `UNSUPPORTED=0`
  - blocker: `timestamp_utc_timezone`, `row_count=57241`
- Scoped July diagnostic:
  - `PASS=96`, `CRITICAL=6`, `UNSUPPORTED=0`, `INFO=2`
  - all active covered direct and quote-aware product gates pass;
  - 2 no-overlap June source quotes are explicit `INFO out_of_scope_quote`;
  - remaining blockers are 6
    `active_quote_set_parent_dropped` rows with
    `dropped_reason=parent_child_conflict`.

### Roast

Roast finding fixed in this continuation:

- The decision log had a naming ambiguity: D-20260622-06 described a provisional
  `source_quote_parent_child_consistency` gate name, while the final
  implementation emits `active_quote_set_parent_dropped` with
  `dropped_reason=parent_child_conflict`. D-20260622-07 was clarified to state
  that it supersedes the provisional naming while preserving fail-closed
  severity.

Residual risks intentionally not fixed here:

- P1 is still not promotion-PASS until governance decides whether
  `parent_child_conflict` remains a hard blocker or can be explicitly accepted
  when a complete finer active quote set exists.
- Generated P1 report artifacts are useful evidence but should be curated before
  commit; they are not needed for runtime code.
- The main `PFC_LT` worktree remains dirty and should not be committed
  wholesale.

### Commit Candidate Scope

Candidate code/doc/test files for a future curated P1 commit:

- `scripts/audit_ch_product_normalization.py`
- `tests/test_audit_ch_product_normalization_script.py`
- `scripts/export_local_test_ch_hourly_csv.py`
- `tests/test_export_local_test_ch_hourly_csv_script.py`
- `docs/lt_monthly_curve_program/phase-p1-product-normalization-audit.md`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260622-lt-product-normalization-audit.md`

Evidence artifacts under `.planning/phases/14-lt-audit-remediation/CH-P1-*`
and `P1-PRODUCT-NORM-EXPORT-*` should be included only if the PR wants embedded
audit evidence. Do not include generated `output/*.csv`, Power BI files, CT
files, or heavy parquet/duckdb data.

### Branch Hygiene Worktrees

Created separate clean worktrees for the next two scopes:

```text
h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase3_curated
  branch: clean/phase3-hourly-shaping-curated
  tracking: origin/fix/lt-audit-remediation

h:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT_clean_phase4_q4
  branch: clean/phase4-cross-year-q4
  tracking: origin/fix/lt-audit-remediation
```

Initial parallel creation attempt was stopped by two environment issues:

- Phase 3 tried to smudge LFS object
  `models/chronos-2/model.safetensors` and failed certificate verification.
- Phase 4 hit a transient Git config lock while both `git worktree add`
  commands ran at the same time.

Recovery:

```powershell
git branch --set-upstream-to=origin/fix/lt-audit-remediation clean/phase4-cross-year-q4
$env:GIT_LFS_SKIP_SMUDGE='1'; git worktree add "..\PFC_LT_clean_phase3_curated" clean/phase3-hourly-shaping-curated
$env:GIT_LFS_SKIP_SMUDGE='1'; git worktree add "..\PFC_LT_clean_phase4_q4" clean/phase4-cross-year-q4
```

Verification:

```powershell
git status --short --branch
```

Outputs:

```text
## clean/phase3-hourly-shaping-curated...origin/fix/lt-audit-remediation
## clean/phase4-cross-year-q4...origin/fix/lt-audit-remediation
```

No code/data changes were made in the Phase 3 or Phase 4 worktrees.
