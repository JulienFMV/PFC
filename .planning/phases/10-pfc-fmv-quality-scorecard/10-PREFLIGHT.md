# Plan 10-04 — Preflight Go/No-Go [C6 REVIEWS]

**Generated** : 2026-06-08T13:56:28Z
**Script** : scripts/preflight_phase10.py
**Verdict** : **no-go**

## Python environment
- sys.executable : `C:\Users\jbattaglia\.conda\ppa_env\python.exe`
- Xcode 3.9 shadowing detected : False
- statsmodels + matplotlib importable : True

## Check 1 — Data freshness
- epex_hourly.parquet present : True
- epex_hourly.parquet : mtime 2026-06-08T11:38:17Z, age 0.1 days, threshold ≤ 14 days
- forwards_history_phase10.parquet present : True
- forwards_history_phase10.parquet : mtime 2026-06-08T13:54:20Z, age 0.0 days, threshold ≤ 14 days
- **Status** : PASS (PASS)

## Check 2 — Forwards path
- forwards_source unique value : `real_eex_xlsx`
- Gate-eligible : Y
- Decision : continue (gate-eligible)
- **Status** : PASS (PASS)

## Check 3 — Disk space
- Free space at output_dir mount : 2.14 GB
- Threshold ≥ 5.0 GB → FAIL
- Budget output total : ~2.5 GB (96 PFC caches × 25 MB + KPIs + figures)
- **Status** : FAIL (FAIL)
- Reason : free space 2.1 GB < 5.0 GB threshold

## Check 4 — Runtime benchmark
- Micro-run (Config 4, vintage 2024-06-28, no uncertainty) : 9.61 sec
- Extrapolation full 96-build × 1.4 (Pillar 3 overhead) : 1291.2 sec = 0.36 h
- Budget hard cap : 2.5h ; soft target : 2.0h
- **Status** : PASS (PASS)

## Final Verdict
- **no-go**
- Next step : addresser le(s) check(s) FAIL avant re-run preflight : disk_space: free space 2.1 GB < 5.0 GB threshold
