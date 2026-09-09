# LT Quant Smoothness Lambda Calibration

* source_csv: `output\ch_hfc_hourly_20260618_20301231_patched_reanchor_after_smoothing.csv`
* metrics_csv: `output\lt_quant_lambda_calibration\lambda_sensitivity_report.csv`
* status: diagnostic calibration evidence, not production sign-off
* full-reaudit decision: initial L-curve-only candidate rejected; operational diagnostic candidate is `h0_m10000_s10000`

## Accepted Diagnostic Candidate After Full Reaudit

| metric | value |
|---|---:|
| label | h0_m10000_s10000 |
| lambda_smooth_h | 0.0 |
| lambda_smooth_m | 10000.0 |
| lambda_seam | 10000.0 |
| reason | preserves intraday duck shape while fixing 2029-2030 annual-only monthly path |
| full_generation_csv | output/ch_hfc_hourly_20260618_20301231_mit_quantannual_h0.csv |
| shape_score_10 | 7.75 |
| hfc_vs_spot_score_10 | 8.25 |
| seasonal_critical_flags | 0 |
| seasonal_warning_flags | 0 |
| max_eex_base_error_eur_mwh | 0.000000 |
| max_eex_peak_error_eur_mwh | 0.000000 |

## Initial L-Curve Candidate Rejected By Full Reaudit

| metric | value |
|---|---:|
| label | h30_m10000_s10000 |
| lambda_smooth_h | 30.0 |
| lambda_smooth_m | 10000.0 |
| lambda_seam | 10000.0 |
| prior_rmse_eur_mwh | 8.399472624641618 |
| boundary_jump_abs_p95_eur_mwh | 5.635313494353935 |
| residual_boundary_jump_abs_max_eur_mwh | 2.103676083469736 |
| month_adjacent_jump_abs_p95_eur_mwh | 8.508583641125803 |
| month_curvature_abs_p95_eur_mwh | 2.4203358566987485 |
| max_constraint_abs_error_eur_mwh | 5.463192565002828e-09 |
| lcurve_selection_score | 0.43828962459615123 |

## Historical Spot Monthly-Shape Benchmark

| metric | value |
|---|---:|
| spot_month_adjacent_jump_abs_p50_eur_mwh | 11.637390 |
| spot_month_adjacent_jump_abs_p95_eur_mwh | 31.434719 |
| spot_month_adjacent_jump_abs_max_eur_mwh | 41.520675 |

## +/-25% Stability Perturbations

| label | month_p95_change | residual_seam_change | prior_rmse |
|---|---:|---:|---:|
| h30_m10000_s10000_lambda_smooth_h_x0.75 | 0.000152 | 0.186051 | 7.997026 |
| h30_m10000_s10000_lambda_smooth_h_x1.25 | -0.000135 | -0.147231 | 8.702967 |
| h30_m10000_s10000_lambda_smooth_m_x0.75 | 0.368474 | -0.026968 | 8.182883 |
| h30_m10000_s10000_lambda_smooth_m_x1.25 | -0.325666 | 0.021419 | 8.606779 |
| h30_m10000_s10000_lambda_seam_x0.75 | 0.000000 | 0.000219 | 8.399472 |
| h30_m10000_s10000_lambda_seam_x1.25 | -0.000000 | -0.000131 | 8.399473 |

## Interpretation

- Hard EEX constraints remain non-negotiable; candidates with non-zero residuals are invalid.
- The L-curve score is a documented heuristic over distance-to-prior and shape roughness; by itself it over-selected hourly smoothness.
- Full-generation backtest/visual audit rejected `lambda_smooth_h=30` because it suppressed the 2030 intraday duck signal.
- Desk/independent quant approval is still required before using these lambdas in production.
