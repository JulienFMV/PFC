# Claude counter-review of D317 — 8 September 2026

Scope: verify the remediation commits `c5b80a80f3` and `d07aca6c17` against
the 33 findings of `docs/model/PFC-CH-AUDIT-REPORT-20260908.md`. Read-only
review; no code, data, artifact or authority changed. Deliverable:
`docs/model/PFC-CH-AUDIT-COUNTER-REVIEW-20260908.md`.

## Result

- All demonstrated defects (F-13, F-14, F-16, F-18, F-19, F-20, F-21, F-25,
  F-27, F-29, F-30, F-31, F-32) are fixed in code with Git-replayable
  regression tests; F-17, F-26, F-28 are correctly re-labelled; F-23 partial as
  declared; F-01/F-05/F-06/F-10 deferred to the next benchmark as declared.
- The diff touches no file under `pfc_shaping/lt/`, `pfc_shaping/calibration/`
  or `pfc_shaping/pipeline/`, so D304, the solver, the assembler and the EEX
  projection are unchanged by construction.
- Linux replay of the checkpoint matrix plus the new snapshot/publisher/input
  tests: 681 passed, 0 failed, 20 skipped. Remote CI green on both workflows at
  `d07aca6c17`.
- No new ruff violations introduced by the fix commits (per-file before/after
  counts identical).
- Four new minor observations (N-1 to N-4); N-1 is an operating decision for
  pilot day 2 (same-day partial EEX rows would abort the day; bound the query
  to `QuotationDateID < day` or document the abort).
- The D317 challenge on the level-conditioned shape proposal is accepted; the
  training conditioning variable must be known at origin (EEX-implied monthly
  level at a fixed lead), with the realized mean kept as an oracle lane only.

## Exact commands (Linux, scratch venv)

```
git fetch origin fix/lt-audit-remediation
git checkout -B claude/pfc-ch-audit-15kn4o d07aca6c17
python -m pytest -q -p no:cacheprovider -rs <34 checkpoint files> \
  tests/test_databricks_lt_snapshot.py tests/test_snapshot_publisher_container_contract.py \
  tests/test_lt_input_sources.py
ruff check --output-format concise <13 changed .py files>   # before/after per file
```

Not independently replayed: `build/lt-audit-response-20260908/` receipts, the
75-month frozen solver replay, the five `independent-v1` probes, and the
Windows 716/5 matrix (its exact 38 paths live only in the local command JSON).

Authorities: production, promotion, scientific_admission, trading,
externally_registered, countable_origin all remain false.
