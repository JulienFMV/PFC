# LT Next SOTA Validation: Intraday Amplitude Shrinkage

## Scope

Branch: `feat/lt-next-sota`.

Head at sync: `0652a65 docs(agent-briefs): worktree-safe sync via feature branch from origin`.

Delivered lever: additive `sota_amp` estimator. It enables
`PFCAssembler(enable_intraday_amplitude_shrinkage=True)` inside the scoped
perfect-foresight benchmark only. The production/default path keeps the flag
`False`.

Local data caveat: this workspace did not have the phase-10 data files at first
run. `data/epex_hourly.parquet` was bootstrapped from tracked
`pfc_shaping/data/epex_15min.parquet`, and
`data/forwards_history_phase10.parquet` was generated with
`forwards_source=fallback_diagnostic`. Results below are diagnostic for the
residual and implementation, not a formal market-data gate.

## Commands

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/run_perfect_foresight.py --ab
$env:PYTHONPATH='.'; pytest tests/test_intraday_amplitude.py -q
$env:PYTHONPATH='.'; pytest tests/test_perfect_foresight.py::test_build_curve_rejects_unknown_estimator_string tests/test_perfect_foresight.py::test_sota_swap_restores_on_exception tests/test_perfect_foresight.py::test_sota_swap_patches_and_restores_shapehourly_halflife -q
$env:PYTHONPATH='.'; python -m py_compile pfc_shaping/lt/model/intraday_amplitude.py pfc_shaping/lt/model/assembler.py pfc_shaping/validation/perfect_foresight.py scripts/run_perfect_foresight.py
```

## A/B Result

Cal-2025 A/B benchmark:

| metric | baseline | SOTA | SOTA+solar | SOTA+amp |
|---|---:|---:|---:|---:|
| median `pf_cal_corr` | 0.7447 | 0.9182 | 0.9182 | 0.9182 |
| min variant `pf_cal_corr` | n/a | n/a | 0.8608 | 0.8606 |
| median delta vs SOTA | n/a | n/a | +0.0000 | +0.0000 |
| vintages improved vs baseline | n/a | 12/12 | 12/12 | 12/12 |
| Wilcoxon p, SOTA > baseline | n/a | 0.0002 | n/a | n/a |

Best-trained physical sub-KPIs:

| sub-KPI | SOTA | SOTA+solar | SOTA+amp | realized |
|---|---:|---:|---:|---:|
| solar-bowl depth | 0.449 | 0.437 | 0.428 | 0.558 |
| peak/offpeak spread, EUR/MWh | 18.56 | 18.57 | 7.16 | 6.41 |

Interpretation: `sota_amp` preserves the monthly-shape gate while closing 94%
of the measured peak/offpeak amplitude gap. It is a null or negative result for
the solar-bowl KPI, so it should be treated as a targeted peak/offpeak fix, not
as a replacement for the solar residual work.

## Contracts

Reproducibility: default constructor flag is `False`; tests assert this default,
and the new module is only imported inside guarded `if
enable_intraday_amplitude_shrinkage` blocks. OFF-path behavior is therefore
additive and unreachable by default.

No leakage: the layer consumes only calendar labels and `peak_base_spreads_`
already fitted by the cascader on the pre-vintage SOTA path. It does not read
realized target-year residuals or benchmark KPIs.

Additive changes: new module, one assembler kwarg, one perfect-foresight
estimator variant, benchmark/report wiring, and focused tests.

Known validation caveat: the diagnostic build emits existing energy-consistency
telemetry on several variants. Because the available forwards file here is
fallback-derived, these logs are recorded but not used as a formal gate.
