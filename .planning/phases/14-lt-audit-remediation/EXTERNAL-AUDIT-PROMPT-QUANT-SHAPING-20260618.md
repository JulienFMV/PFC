# External Audit Prompt - CH/DE LT PFC Quant Shaping Changes - 2026-06-18

You are an external quantitative engineering auditor reviewing changes made on the `fix/lt-audit-remediation` branch of the PFC repository.

Your mandate is to perform a critical audit of all changes described below. Treat this as a pre-merge/pre-production audit. Do not assume that a green test suite or a Power BI refresh is sufficient. The output must be a ranked finding list with severity, file/line references, reproducible evidence, and concrete remediation steps.

## 1. Repository Context

Repo: `JulienFMV/PFC`

Working branch: `fix/lt-audit-remediation`

Reference commit: after pulling the branch, record `git rev-parse --short HEAD` in the audit report.

Domain: long-term electricity forward curve generation for Switzerland and Germany, with EEX calibration and CH hourly HFC/PFC exports.

Relevant project convention:

- LT code lives in `pfc_shaping/lt/model/`, `pfc_shaping/calibration/`, and LT orchestration scripts.
- LT must not import `pfc_shaping.ct.*`.
- `run_pfc_production.py` / LT orchestration is the boundary where production generation should eventually be integrated.
- Current work is still local/test diagnostics unless explicitly promoted.

Primary user concern that triggered the latest iteration:

> The monthly shaping plot was visibly wrong. In particular, 2028 had a severe Q1-to-residual break and an implausible Apr-Dec residual shape. The curve passed earlier numerical gates but looked market-incoherent.

## 2. Files Changed In Scope

Audit these files as the primary code diff:

- `scripts/export_local_test_ch_hourly_csv.py`
- `scripts/audit_ch_hfc_seasonal_coherence.py`
- `scripts/audit_ch_pfc_hourly_shape.py`
- `scripts/build_powerbi_exports.py`
- `scripts/build_ch_hfc_validation_workbook.py`
- `tests/test_export_local_test_ch_hourly_csv_script.py`
- `tests/test_audit_ch_hfc_seasonal_coherence_script.py`
- `tests/test_audit_ch_pfc_hourly_shape_script.py`

Also inspect but do not assume they are authoritative:

- `data/eex_forwards_history.parquet` was locally rebuilt from desk EEX Excel sources and contains latest CH/DE/FR/AT/IT forward snapshot dated `2026-06-17`.
- `powerbi/data/*` sidecars were regenerated locally by `powerbi/refresh_powerbi_data.ps1`.
- `output/*` and `.planning/phases/14-lt-audit-remediation/*` report artifacts are generated diagnostics and may not be committed.

## 3. Current Final Candidate Curve

Power BI was last refreshed against this CSV:

`output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv`

Important: this final candidate was produced by directly applying the new annual residual anchor to the prior CSV, then recalibrating BASE and PEAK. The full CLI path with the new `--enable-neighbor-annual-residual-shape-anchor` flag was attempted and appeared to hang before writing the CSV. Treat this as a current P0 integration/performance issue.

Important artifact availability note:

- `output/*` files are local/generated artifacts and may not be present after cloning GitHub.
- If you are auditing only the GitHub branch, audit the code and tests, and treat final-candidate metrics as reported evidence that must be independently reproduced once artifacts are provided.
- If you are auditing the final candidate curve itself, request or regenerate the following local artifacts:
  - `output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_pathsmooth.csv`
  - `output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv`
  - `output/qa_neighbor_anchor/residualanchor_direct_monthly.csv`
  - `output/qa_neighbor_anchor/residualanchor_direct_monthly_split.csv`
  - `output/qa_neighbor_anchor/residualanchor_direct_monthly_path.csv`
  - `output/qa_neighbor_anchor/residualanchor_direct_calendar.csv`
  - `output/hfc_diagnostics_residualanchor_direct/*.png`
- Do not mark the candidate `PASS` if you cannot inspect either the regenerated CSV or the provided CSV artifact.

The direct correction script used locally was equivalent to:

```python
from pathlib import Path
import pandas as pd
from scripts.export_local_test_ch_hourly_csv import (
    _parse_timestamp_ch,
    _latest_eex_prices_by_load_type,
    _historical_month_deviations,
    apply_neighbor_annual_residual_shape_anchor,
    calibrate_hourly_to_eex,
    calibrate_hourly_to_eex_base_peak,
)

inp = Path("output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_pathsmooth.csv")
out = Path("output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv")
forwards = Path("data/eex_forwards_history.parquet")
hourly = pd.read_csv(inp)
ts = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
latest, ch = _latest_eex_prices_by_load_type(forwards, market="CH")
_, de = _latest_eex_prices_by_load_type(forwards, market="DE")
hist = {
    "BASE": _historical_month_deviations(
        forwards,
        market="CH",
        load_type="BASE",
        min_date=pd.Timestamp(latest).normalize() - pd.DateOffset(years=6),
    )
}
weights = {"slow": 0.25, "central": 0.5, "fast": 0.25}
hourly, audit = apply_neighbor_annual_residual_shape_anchor(
    hourly,
    ts_ch=ts,
    base_forward_prices=ch["BASE"],
    neighbor_base_forward_prices=de["BASE"],
    historical_base_deviations=hist["BASE"],
    neighbor_current_weight=0.65,
    weights=weights,
    intensity=1.0,
    max_month_delta_eur_mwh=30.0,
    negative_price_floor=-30.0,
    max_weighted_negative_hours=300,
)
hourly, _ = calibrate_hourly_to_eex(hourly, forward_prices=ch["BASE"])
hourly, _ = calibrate_hourly_to_eex_base_peak(
    hourly,
    base_forward_prices=ch["BASE"],
    peak_forward_prices=ch.get("PEAK", {}),
    weights=weights,
    negative_price_floor=-30.0,
    max_weighted_negative_hours=300,
)
hourly.to_csv(out, index=False)
```

## 4. EEX Market Facts Used In The Latest Correction

Latest CH BASE snapshot date: `2026-06-17`.

Relevant CH quotes:

```text
2027      93.90
2027-Q1  122.02
2027-Q2   71.95
2027-Q3   75.01
2027-Q4  107.01
2028      80.40
2028-Q1  109.97
2029      72.41
2030      69.22
```

Relevant DE BASE quotes:

```text
2028      81.10
2028-01  103.74
2028-02  100.86
2028-03   83.51
2028-04   68.42
2028-05   66.14
2028-06   67.32
2028-Q1   95.94
2028-Q2   67.28
2028-Q3   73.12
2028-Q4   88.06
```

Mechanics:

- CH has `2028` Calendar and `2028-Q1` quoted.
- Calibration creates a `2028-RESIDUAL` bucket for Apr-Dec.
- `2028-RESIDUAL` target is approximately `70.6209801545`.
- A visible break from Q1 to Apr is partly unavoidable because Q1 is directly quoted at `109.97` while the Apr-Dec residual bucket must average about `70.62`.
- The model defect was the shape inside Apr-Dec, not the existence of a Q1-to-Apr break.

Before latest residual anchor, 2028 monthly weighted means were roughly:

```text
Jan 117.62
Feb 115.62
Mar  97.02
Apr  52.50
May  50.93
Jun  51.20
Jul  56.62
Aug  60.62
Sep  71.71
Oct  83.09
Nov  97.93
Dec 110.68
```

After latest direct residual anchor candidate:

```text
Jan 117.62
Feb 115.62
Mar  97.02
Apr  62.88
May  60.60
Jun  61.77
Jul  65.45
Aug  65.69
Sep  71.69
Oct  78.38
Nov  84.39
Dec  84.68
```

Audit whether this final shape is actually market-plausible. Do not accept it merely because it is smoother.

## 5. Summary Of Implemented Changes

### 5.1 Weighted Negative Prices In Central/Weighted Curve

File: `scripts/export_local_test_ch_hourly_csv.py`

Implemented optional weighted negative capture:

- New CLI arg: `--weighted-negative-capture-intensity`.
- Applied through `apply_post_calibration_negative_rebalancer(...)`.
- Default is `0.0`, so flag-OFF behavior should remain unchanged.
- Intended to allow bounded negative weighted prices in solar belly hours while preserving EEX bucket means.

File: `scripts/audit_ch_pfc_hourly_shape.py`

Implemented a localized negative gate:

- Weighted negative prices can pass if constrained to Apr-Sep, hours 10-16.
- Min weighted price must be `>= -15`.
- Min any price level must be `>= -30`.
- Share must be `<= 0.50%`.
- Outside allowed hours must be zero.
- Max per month `<= 48`.
- Max consecutive run `<= 8`.
- P10 and fast negative shares `<= 2%`.
- Gate exposes metrics in `shape_metrics`: `negative_gate_status`, share, localization, outside hours, min weighted price, etc.

Audit questions:

- Is this gate quantitatively defensible for CH long-term forward curves?
- Are thresholds too permissive or too hard-coded?
- Does it allow negative weighted prices only where historically and structurally plausible?
- Does it accidentally reward reducing P10/fast negative tails rather than preserving realistic tail risk?

### 5.2 Neighbor Monthly Spread Anchor For Unquoted Quarter Months

File: `scripts/export_local_test_ch_hourly_csv.py`

Implemented:

- `_quarter_month_numbers`
- `_quarter_year`
- `_month_key`
- `_guided_month_targets`
- `_historical_month_deviations`
- `apply_neighbor_monthly_spread_anchor`

CLI args:

- `--enable-neighbor-monthly-spread-anchor`
- `--neighbor-monthly-market` default `DE`
- `--neighbor-monthly-anchor-intensity`
- `--neighbor-monthly-current-weight` default `0.65`
- `--neighbor-monthly-historical-market` default `CH`
- `--neighbor-monthly-max-delta-eur-mwh`

Intended behavior:

- For unquoted CH monthly splits inside a quoted CH quarter, use neighbor monthly deviations from the neighbor quarter, optionally blended with historical CH month-quarter deviations.
- Use deviations/shape, not neighbor absolute level.
- Recenter exactly to the CH parent bucket.
- BASE anchor applies flat monthly additive deltas.
- PEAK anchor applies deltas to contractual EEX peak hours and compensates offpeak inside the month so monthly BASE is preserved.
- Recalibrate EEX BASE and optionally EEX PEAK after application.

Audit questions:

- Does this leak neighbor absolute levels into CH?
- Does the re-centering properly preserve CH bucket means under unequal month hour counts and DST?
- Is the PEAK/offpeak compensation correct with Swiss holidays and EEX peak definitions?
- Should AT/FR/IT be part of the guide, or is DE-only acceptable?

### 5.3 EEX Peak Holiday Calendars

File: `scripts/export_local_test_ch_hourly_csv.py`

`_eex_peak_mask(ts_ch, country=...)` now supports `CH`, `DE`, `AT`, `FR`, `IT`, and raises `ValueError` for unsupported country.

Audit questions:

- Is the `holidays` package calendar definition aligned with EEX contractual peak calendars for all supported markets?
- Are timezone/local-date conversions correct for non-CH markets, given `ts_ch` is still Europe/Zurich?

### 5.4 Seasonal, Monthly Split, Monthly Path, And Calendar Audits

File: `scripts/audit_ch_hfc_seasonal_coherence.py`

Added:

- `monthly_split_checks(...)`
- `calendar_coherence_checks(...)`
- `monthly_path_checks(...)`
- CLI outputs:
  - `--monthly-split-output`
  - `--monthly-path-output`
  - `--calendar-output`
  - `--neighbor-market`
- Report sections:
  - Seasonal checks
  - Unquoted monthly split checks
  - Monthly path checks
  - Calendar coherence checks
  - Quoted EEX product residuals

Current intended gates:

- Seasonal: flags annual-only January below October / Q1-Q4 anomalies.
- Monthly split: compares unquoted CH month-vs-parent deviations against neighbor market shape, not absolute levels.
- Monthly path: flags large adjacent jumps and sharp local reversals inside synthetic annual/residual buckets only.
- Calendar: flags implausible weekend premium and large week-to-week jumps.

Audit questions:

- Are thresholds calibrated or arbitrary?
- Does the monthly split audit miss annual residual shape defects across Q2/Q3/Q4?
- Does the monthly path audit wrongly ignore cross-bucket jumps that are economically implausible but not directly quoted?
- Does the calendar audit correctly handle DST weeks, partial first/last months, and holidays?

### 5.5 Power BI And Workbook Sidecars

File: `scripts/build_powerbi_exports.py`

Added outputs:

- `seasonal_coherence.csv`
- `monthly_split_diagnostics.csv`
- `monthly_path_diagnostics.csv`
- `calendar_coherence.csv`

Added summary metrics:

- `negative_gate_status`
- `weighted_negative_share_pct`
- `weighted_negative_outside_allowed_hours`
- `negative_localization_pct`
- `min_weighted_eur_mwh`
- seasonal/monthly split/monthly path/calendar critical and warning counts

File: `scripts/build_ch_hfc_validation_workbook.py`

Added workbook sheets:

- `Monthly_Path_Checks`
- `Monthly_Split_Checks`
- `Calendar_Coherence`

Audit questions:

- Are Power BI sidecars correctly refreshed whenever a new `ch_hfc_hourly*.csv` is generated?
- Are these new sidecars included in the Power BI semantic model/report, or merely written to disk?
- Does `resolve_csv_path()` still pick the latest expected `output/ch_hfc_hourly*.csv` and avoid unrelated files?

### 5.6 Synthetic Monthly Path Smoothing

File: `scripts/export_local_test_ch_hourly_csv.py`

Implemented:

- `apply_synthetic_monthly_path_smoothing(...)`
- `_solve_month_path_delta(...)`
- `_independent_rows(...)`

CLI args:

- `--enable-final-monthly-path-smoothing`
- `--final-monthly-path-smoothing-intensity`
- `--final-monthly-path-smoothing-lambda`
- `--final-monthly-path-max-delta-eur-mwh`

Intended behavior:

- Smooth monthly means only inside synthetic annual or residual buckets.
- Preserve the synthetic bucket mean by construction.
- Recompute weighted fan columns.
- Enforce negative floor and max weighted negative hours.
- Recalibrate EEX BASE and PEAK after application.

Important concern:

- This smoothing fixed intra-bucket zigzags but did not solve the 2028 residual level allocation by itself.
- It should not be used to hide an economically wrong residual shape.

Audit questions:

- Is the linear system correct?
- Does it preserve bucket means exactly enough after rounding?
- Does it accidentally distort directly quoted quarters/months?
- Should this exist at all if the annual residual anchor is the better correction?

### 5.7 Neighbor Annual Residual Shape Anchor

File: `scripts/export_local_test_ch_hourly_csv.py`

Implemented:

- `apply_neighbor_annual_residual_shape_anchor(...)`
- `_annual_residual_month_targets(...)`

CLI args:

- `--enable-neighbor-annual-residual-shape-anchor`
- `--neighbor-annual-residual-anchor-intensity`
- `--neighbor-annual-residual-max-delta-eur-mwh`

Intended behavior:

- Applies only to buckets named like `YYYY-RESIDUAL`, e.g. `2028-RESIDUAL`.
- Uses neighbor current monthly prices where available, otherwise neighbor quarter plus historical CH month-quarter deviations.
- Uses neighbor shape only; re-centers target months exactly to CH residual bucket target.
- Adds deltas to scenario columns and recomputes weighted fan columns.
- Enforces negative floor and max negative hours.
- Requires subsequent EEX BASE and PEAK recalibration.

Current empirical effect on 2028:

- Raises Q2 residual months from ~51-52 to ~61-63.
- Lowers Nov/Dec from ~98/111 to ~84/85.
- Keeps `2028-RESIDUAL` mean at about `70.62`.

Known critical issue:

- Running the full export CLI with `--enable-neighbor-annual-residual-shape-anchor` appeared to hang before writing output, even though direct application of `apply_neighbor_annual_residual_shape_anchor` plus recalibration completed in about 21 seconds.
- The auditor must identify why the full CLI path hangs or is excessively slow before approving the implementation.

Audit questions:

- Is annual residual anchoring mathematically correct with unequal month counts and DST?
- Should the guide be DE-only, or a multi-market robust blend using DE/AT/FR/IT?
- Does fallback to neighbor quarter + historical CH deviations behave sensibly where neighbor monthly products are missing?
- Is the max delta of `30 EUR/MWh` too wide?
- Should this operate on BASE only, or also PEAK residual shape?

## 6. Commands Already Run Locally

Prerequisites:

- Python environment with the repo dependencies installed.
- `data/eex_forwards_history.parquet` refreshed to include CH/DE/FR/AT/IT latest snapshot `2026-06-17`.
- `data/epex_hourly.parquet` available for spot-shape audit.
- Local generated CSV artifacts under `output/`, or enough raw data to regenerate them.

Unit tests:

```powershell
pytest tests/test_export_local_test_ch_hourly_csv_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_audit_ch_pfc_hourly_shape_script.py -q
```

Observed result:

```text
34 passed
```

Shape audit on final direct candidate:

```powershell
python scripts/audit_ch_pfc_hourly_shape.py --csv output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv --forwards data/eex_forwards_history.parquet --report .planning/phases/14-lt-audit-remediation/CH-HFC-HOURLY-SHAPE-AUDIT-RESIDUALANCHOR-DIRECT.md
```

Observed:

```text
[shape-audit] score=8.00/10
```

Seasonal/monthly/calendar audit:

```powershell
python scripts/audit_ch_hfc_seasonal_coherence.py --csv output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv --forwards data/eex_forwards_history.parquet --neighbor-market DE --report .planning/phases/14-lt-audit-remediation/CH-HFC-SEASONAL-COHERENCE-AUDIT-RESIDUALANCHOR-DIRECT.md --monthly-output output/qa_neighbor_anchor/residualanchor_direct_monthly.csv --hour-month-output output/qa_neighbor_anchor/residualanchor_direct_hour_month.csv --monthly-split-output output/qa_neighbor_anchor/residualanchor_direct_monthly_split.csv --monthly-path-output output/qa_neighbor_anchor/residualanchor_direct_monthly_path.csv --calendar-output output/qa_neighbor_anchor/residualanchor_direct_calendar.csv
```

Observed:

```text
[seasonal-audit] critical=0 warning=0
```

HFC vs spot-shape audit:

```powershell
python scripts/audit_ch_hfc_vs_spot_shape.py --csv output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv --spot data/epex_hourly.parquet --report .planning/phases/14-lt-audit-remediation/CH-HFC-VS-SPOT-SHAPE-AUDIT-RESIDUALANCHOR-DIRECT.md
```

Observed:

```text
[hfc-vs-spot] score=8.25/10
```

Power BI refresh:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\powerbi\refresh_powerbi_data.ps1 -Csv "output\ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv"
```

Observed summary metrics from `powerbi/data/summary_metrics.csv`:

```text
source_csv: output\ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv
shape_score_10: 8
hfc_vs_spot_score_10: 8.25
max_eex_base_error_eur_mwh: 0.000000
max_eex_peak_error_eur_mwh: 0.000000
weighted_negative_hours: 42
negative_gate_status: PASS
weighted_negative_share_pct: 0.11
weighted_negative_outside_allowed_hours: 0.000000
negative_localization_pct: 100
min_weighted_eur_mwh: -7.23
p10_negative_hours: 110
min_price_eur_mwh: -16.80
seasonal_critical_flags: 0
seasonal_warning_flags: 0
monthly_split_critical_flags: 0
monthly_split_warning_flags: 0
monthly_path_critical_flags: 0
monthly_path_warning_flags: 0
calendar_critical_flags: 0
calendar_warning_flags: 0
```

PNG diagnostics generated:

```powershell
python scripts/plot_ch_hfc_diagnostics.py --csv output/ch_hfc_hourly_20260618_20301231_weightedneg_i200_de65_chhist35_residualanchor_direct.csv --forwards data/eex_forwards_history.parquet --output-dir output/hfc_diagnostics_residualanchor_direct
```

## 7. Hard Constraints That Must Still Hold

1. EEX BASE calibration must be exact within floating/rounding tolerance for every active quote-aware bucket.
2. EEX PEAK calibration must be exact for quoted CH PEAK products when `--enable-eex-peak-calibration` is used.
3. No absolute neighbor level leakage into CH. Neighbor markets may provide shape/deviation guides only.
4. Flag-OFF behavior must remain byte-identical, or differences must be explicitly justified and tested.
5. New local/test features must remain disabled by default.
6. LT code must not import CT modules.
7. No test should be weakened to make a curve pass.
8. Weighted negative prices may be allowed only when bounded, localized, and explicitly justified.
9. Directly quoted EEX months/quarters must be treated as market signals, not overridden by smoothing.
10. Residual bucket shaping must preserve residual mean under unequal hour counts, DST, and partial horizon boundaries.
11. Power BI must read the intended latest `output/ch_hfc_hourly*.csv` or the explicit CSV passed to refresh.
12. Production approval remains `NO` until the full run path, audit path, and governance path are reviewed.

## 8. Required Audit Work

Please perform the following:

1. Read the diffs in all files listed in Section 2.
2. Reproduce the tests and the three audits in Section 6.
3. Inspect the final PNGs, especially:
   - `01_monthly_means_by_year.png`
   - `02_focus_2027_2028_eex_buckets.png`
   - `03_month_to_month_deltas.png`
   - `09_executive_qa_summary.png`
4. Verify the direct candidate's 2028 residual bucket math:
   - `2028-Q1` exact at `109.97`.
   - `2028-RESIDUAL` exact at about `70.6209801545`.
   - Apr-Dec shape plausible and not absolute-level copied from DE.
5. Investigate why the full CLI path with `--enable-neighbor-annual-residual-shape-anchor` hangs or exceeds 180 seconds before writing the CSV.
6. Decide whether the direct candidate should be considered a valid diagnostic artifact only, or whether the code path is ready to be integrated into normal generation after fixing the hang.
7. Check whether the new audits are sufficient to catch the visual defect that triggered this request. If not, specify a better gate.
8. Review all thresholds and identify those that require market calibration, historical backtest, or analyst approval.
9. Verify that Power BI refresh sidecars and summary metrics reflect the intended CSV and not stale cached data.
10. Identify any missing tests, especially:
    - annual residual anchoring mean preservation;
    - no neighbor level leakage;
    - full CLI no-hang regression;
    - flag-OFF byte identity;
    - PEAK calibration after residual anchoring;
    - DST/partial horizon residual bucket preservation.

## 9. Expected Auditor Output Format

Return:

1. `Verdict`: one of `BLOCK`, `CONDITIONAL PASS`, `PASS`.
2. `Top Findings`: ordered by severity. Each finding must include:
   - Severity: `P0`, `P1`, `P2`, or `P3`.
   - File and line reference.
   - Evidence.
   - Why it matters quantitatively.
   - Required remediation.
3. `Reproduction Log`: exact commands run and outputs.
4. `Quant Assessment`: whether the final 2028 shape is market-plausible, with numeric justification.
5. `Constraint Checklist`: pass/fail for all constraints in Section 7.
6. `Recommended Next Patch`: minimal next changes before merge.

## 10. Known Issues To Treat As Suspect Until Proven Otherwise

- P0 candidate: full CLI path hangs with the new annual residual anchor flag.
- P1 candidate: annual residual anchor currently uses BASE only; PEAK shape may still rely on later PEAK recalibration rather than a neighbor PEAK residual shape.
- P1 candidate: DE-only guide may be too narrow; AT/FR/IT might provide useful cross-checks or robust blending.
- P1 candidate: thresholds in negative gate and monthly/calendar gates are hand-set, not historically calibrated.
- P2 candidate: Power BI writes sidecars, but semantic model/report binding may not expose every new diagnostic.
- P2 candidate: generated reports/PNGs are local artifacts and may not be available to a remote auditor unless regenerated.

## 11. Prompt Self-Audit Checklist

This prompt should be considered 10/10 only if it gives the external auditor:

- exact branch and domain context;
- complete file scope;
- complete list of implemented changes;
- exact current final candidate;
- the known direct-vs-CLI discrepancy;
- reproducible commands;
- numerical before/after evidence;
- hard constraints;
- explicit expected output format;
- known issues and suspected failure modes.

If any of these are missing, improve the prompt before using it.
