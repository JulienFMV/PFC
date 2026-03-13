# PFC Operations Runbook

## 1) Preconditions
- `C:\Users\jbattaglia\.conda\pfc311\python.exe` exists
- `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx` reachable
- `.env` contains:
  - `ENTSOE_API_KEY=...`
  - `TEAMS_WEBHOOK_URL=...` (recommended)

## 2) Daily run (manual)
```powershell
cd "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC"
C:\Users\jbattaglia\.conda\pfc311\python.exe -m pfc_shaping.pipeline.rolling_update
```

## 3) Scheduler setup (Windows Task Scheduler)
```powershell
cd "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC"
powershell -ExecutionPolicy Bypass -File pfc_shaping\tools\register_daily_task.ps1 -TaskName "PFC_Daily_Update" -RunAt "06:15"
```

## 4) Success criteria
- Latest run exists in DuckDB table `runs`
- `calibrated = true`
- `quality.benchmark_gate_ok = true` in `run_report_*.json`
- Fresh file in `pfc_shaping/output/pfc_15min_<run_id>.csv`

## 5) If run fails
- Check latest log in `pfc_shaping/logs/rolling_update_YYYYMMDD.log`
- Check Teams alert payload (`PFC run FAILED`)
- Typical causes:
  - network path unreachable
  - corrupted EEX report format
  - benchmark gate violation (strict mode ON)

## 6) Release policy
- `quality.fail_on_benchmark: true` must stay enabled in prod
- Any threshold change (`max_mae_eur_mwh`, `max_rmse_eur_mwh`, `max_abs_bias_eur_mwh`) requires release note
