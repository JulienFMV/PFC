# Chronos-2 future covariates report

## Implementation

- `FoundationForecaster.forecast(..., future_covariates=...)` now threads known-future covariates to Chronos-2 through `predict_df(future_df=...)`.
- The installed Chronos API was verified in `C:\Users\jbattaglia\.conda\ppa_env`: `predict_df(df, future_df=None, ..., prediction_length=..., quantile_levels=...)`.
- `future_df` uses the synthetic hourly grid immediately after the synthetic context timestamps; real-calendar covariate values are mapped positionally.
- `LEARForecaster(use_future_covariates=False, use_de_renewable_future=False)` keeps both flags default OFF.
- Calendar covariates are deterministic. DE renewable forecast covariates are separately gated and documented as suspect because the parquet has no `as_of` timestamp.

## Smoke validation

Command:

```powershell
C:\Users\jbattaglia\.conda\ppa_env\python.exe scripts/eval_lear_feature_ab.py --n-days 1 --horizon 1 --bootstrap 10 --use-foundation-model --output-dir pfc_shaping/output/chronos2_future_cov_smoke_fm
```

Input/output:

- Local Chronos path used after fixing root model discovery: `models/chronos-2`.
- JSON/parquet artifacts were generated under `pfc_shaping/output/chronos2_future_cov_smoke_fm/` for inspection and then removed to avoid committing generated outputs.
- Scope: smoke only, D+1 for 1 day (`n=24`). This is not a ship gate.

| arm | final MAE | final WAPE | delta MAE vs A | delta WAPE vs A |
|---|---:|---:|---:|---:|
| A baseline | 10.2887 | 7.5348% | 0.0000 | 0.0000 pp |
| B calendar future | 10.3157 | 7.5545% | +0.0269 | +0.0197 pp |
| C calendar + DE future | 10.2000 | 7.4697% | -0.0888 | -0.0650 pp |

Foundation-only attribution:

| arm | FM-only MAE | FM-only WAPE |
|---|---:|---:|
| A baseline | 4.8512 | 3.5527% |
| B calendar future | 5.2714 | 3.8604% |
| C calendar + DE future | 3.9014 | 2.8571% |

Bootstrap CI in this smoke is degenerate because the run has one 24-hour block and only 10 bootstrap draws; it is not statistically meaningful.

## Reproducibility check

Unit coverage verifies the flag-OFF no-op plumbing: with `use_future_covariates=False`, LEAR does not build future covariates; `FoundationForecaster` only sends `future_df` when valid future covariates are supplied.

An exact `atol=0` LEAR backtest comparison using separately repeated backtests was attempted, but the existing LEAR backtest retraining path is not byte-deterministic in this environment even with `use_foundation_model=False` and unchanged flags. Differences appeared at floating precision scale and are attributable to the existing model retraining path, not the future-covariate branch. A full ship gate should therefore use the established reproducible harness over a controlled window and compare against a frozen artifact or parent-commit artifact.

## Leakage caveat

Realized CH load/solar/wind and neighbor prices remain past-only. Calendar features are safe future covariates. DE renewable forecasts are only included when `use_de_renewable_future=True`; the file lacks `as_of`, so any measured edge for C may be spurious if the file is realization-stamped.
