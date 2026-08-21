# MIT Quant Shaping Calibration And Reaudit - 2026-06-18

## Verdict

`CONDITIONAL / NOT PRODUCTION`

The annual-only quant smoothness quick win is valid and improves the visible 2029-2030 monthly shape, but it does not honestly reach the requested internal `8/10` shape gate. Final candidate for this pass:

`output/ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv`

Selected lambdas:

| parameter | value | rationale |
|---|---:|---|
| `lambda_smooth_h` | 0.0 | preserve intraday/duck prior; `h=30` over-smoothed the 2030 duck curve |
| `lambda_smooth_m` | 10000.0 | smooth annual-only month means in the constraint nullspace |
| `lambda_seam` | 10000.0 | stabilize month-boundary curvature inside annual-only years |

## Calibration Decision

The first L-curve diagnostic selected `h30_m10000_s10000`, but the full regenerated PFC audit rejected it:

| candidate | shape_score | vs_spot_score | seasonal_flags | EEX BASE/PEAK | key defect |
|---|---:|---:|---:|---|---|
| baseline `mit_reaudit` | 7.75 | 7.25 | 0 critical / 4 warning | exact / exact | 2029-2030 monthly shape visibly wrong |
| `h30_m10000_s10000` | 7.00 | 8.25 | 0 / 0 | exact / exact | duck curve collapsed: 2030 evening-midday `12.64` |
| `h0_m10000_s10000` | 7.75 | 8.25 | 0 / 0 | exact / exact | monthly fixed; seam/ramp gate still weak |
| quote-aware all-year direct probe | 6.25 | 8.25 | 1 critical / 3 warning | BASE exact / PEAK error `0.262` | 2027-2028 distorted; rejected |

Conclusion: L-curve roughness alone is insufficient. The accepted diagnostic lambda set is the one passing the full generation, visual monthly audit, EEX audit, and spot-shape audit without suppressing the intraday duck signal.

## Final Candidate Metrics

Power BI was refreshed explicitly against:

`output\ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv`

Summary metrics:

| metric | value |
|---|---:|
| shape_score_10 | 7.75 |
| hfc_vs_spot_score_10 | 8.25 |
| max_eex_base_error_eur_mwh | 0.000000 |
| max_eex_peak_error_eur_mwh | 0.000000 |
| negative_gate_status | PASS |
| weighted_negative_share_pct | 0.10 |
| seasonal_critical_flags | 0 |
| seasonal_warning_flags | 0 |
| monthly_path_critical_flags | 0 |
| monthly_path_warning_flags | 0 |
| calendar_critical_flags | 0 |
| calendar_warning_flags | 0 |
| latest_hfc_shape_corr_vs_spot | 0.92 |

## Monthly Shape Evidence

Before annual-only quant smoothing:

| year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2029 | 83.30 | 86.87 | 73.24 | 57.11 | 50.12 | 47.71 | 59.85 | 55.84 | 74.90 | 83.87 | 98.82 | 98.30 |
| 2030 | 79.46 | 85.40 | 70.31 | 54.60 | 47.29 | 44.21 | 59.24 | 53.23 | 75.16 | 82.41 | 93.28 | 87.28 |

After final candidate:

| year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2029 | 82.39 | 75.30 | 68.28 | 62.18 | 58.20 | 57.20 | 59.43 | 64.45 | 71.86 | 80.62 | 89.89 | 99.11 |
| 2030 | 79.29 | 72.53 | 65.77 | 59.91 | 56.16 | 55.34 | 57.64 | 62.41 | 69.17 | 76.70 | 84.26 | 91.50 |

2028 remains unchanged by design because it is constrained by quoted `2028-Q1` and the `2028-RESIDUAL` Apr-Dec bucket. The visible Mar-Apr break is economically imposed by the quote structure, not a free annual-only shape error.

## Reproduction Log

Full generation:

```powershell
python scripts/export_local_test_ch_hourly_csv.py --local-start-date 2026-06-18 --local-end-date 2030-12-31 --output output/ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv --report .planning/phases/14-lt-audit-remediation/CH-HFC-HOURLY-CSV-MIT-QUANTANNUAL-H0.md --prefix local_test_ch_pfc_20260613_20301231 --forwards data/eex_forwards_history.parquet --required-forward-date 2026-06-17 --enable-structural-shape-upgrade --enable-post-calibration-negative-rebalancer --weighted-negative-capture-intensity 2 --max-weighted-negative-hours 300 --enable-post-calibration-peak-shape-rebalancer --enable-eex-peak-calibration --enable-neighbor-monthly-spread-anchor --enable-neighbor-annual-residual-shape-anchor --enable-final-monthly-path-smoothing --enable-final-quant-annual-smoothness --skip-powerbi-refresh
```

Audits:

```powershell
python scripts/audit_ch_pfc_hourly_shape.py --csv output/ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv --forwards data/eex_forwards_history.parquet --report .planning/phases/14-lt-audit-remediation/CH-HFC-HOURLY-SHAPE-AUDIT-MIT-QUANTANNUAL-H0.md
python scripts/audit_ch_hfc_seasonal_coherence.py --csv output/ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv --forwards data/eex_forwards_history.parquet --neighbor-market DE --report .planning/phases/14-lt-audit-remediation/CH-HFC-SEASONAL-COHERENCE-AUDIT-MIT-QUANTANNUAL-H0.md --monthly-output output/qa_mit_quantannual_h0/monthly.csv --hour-month-output output/qa_mit_quantannual_h0/hour_month.csv --monthly-split-output output/qa_mit_quantannual_h0/monthly_split.csv --monthly-path-output output/qa_mit_quantannual_h0/monthly_path.csv --calendar-output output/qa_mit_quantannual_h0/calendar.csv
python scripts/audit_ch_hfc_vs_spot_shape.py --csv output/ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv --spot data/epex_hourly.parquet --report .planning/phases/14-lt-audit-remediation/CH-HFC-VS-SPOT-SHAPE-AUDIT-MIT-QUANTANNUAL-H0.md
python scripts/plot_ch_hfc_diagnostics.py --csv output/ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv --forwards data/eex_forwards_history.parquet --output-dir output/hfc_diagnostics_mit_quantannual_h0
powershell -NoProfile -ExecutionPolicy Bypass -File .\powerbi\refresh_powerbi_data.ps1 -Csv "output\ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv"
pytest tests/test_export_local_test_ch_hourly_csv_script.py tests/test_audit_ch_hfc_seasonal_coherence_script.py tests/test_audit_ch_pfc_hourly_shape_script.py tests/test_lt_quant_curve_continuity.py tests/test_lt_quant_contract_matrix.py tests/test_lt_quant_optimizer_kkt.py -q
```

Observed:

```text
shape-audit: 7.75/10
seasonal-audit: critical=0 warning=0
hfc-vs-spot: 8.25/10
pytest: 57 passed
```

## Remaining Blockers To 8/10+

1. The shape gate still caps this candidate at `7.75` because `boundary_jump_abs_p95_eur_mwh=28.37` remains above the `20` threshold.
2. A naive all-year quote-aware smooth solve reduced `boundary_p95` to `18.79`, but failed PEAK exactness and distorted 2027-2028. It is rejected.
3. The next real patch must add a local seam-smoothing operator that works around quoted bucket boundaries while preserving all active BASE and PEAK constraints exactly.
4. The L-curve harness must be upgraded so the selection objective includes intraday/duck preservation or spot-shape backtest terms, not only roughness.
5. Production remains `NO` until the seam operator is implemented, calibrated, and audited visually/quantitatively.
