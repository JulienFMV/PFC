# T057 locked one-shot execution sidecar - 2026-07-23

Status: frozen pre-score protocol. Production authorization: false.

This sidecar does not modify the frozen T057 plan. It supersedes only the
operational command templates embedded in that plan, which predate provider-raw
semantic replay, exact-support centering and first-capture sealing.

## Unique authorized route

Do not acquire or score before the end-exclusive maturity instant
`2026-07-24T00:00:00Z`. After that instant, use exactly:

```powershell
python scripts\run_energy_charts_epex_locked_holdout.py `
  --plan-json .planning\phases\14-lt-audit-remediation\locked_holdout_plan_t057_t056_asof20260709.json `
  --expected-plan-sha256 f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd `
  --output-dir output\phase14\t057_locked_t056_future_holdout\energy_charts_locked_runner_20260724 `
  --bzn CH
```

The wrapper enforces the exact plan id, SHA-256, CH bidding zone and canonical
output directory. The direct backtest and direct audit templates inside the
frozen plan are obsolete as operator entrypoints; they remain frozen historical
bytes and must not be edited.

## First-capture rule

The first post-maturity provider attempt is the only local T057 capture. The
wrapper refuses any later execution as soon as one of these exists:

- `energy_charts_locked_holdout_attempt_seal.json`;
- `energy_charts_epex_spot_provider_raw.json`;
- `energy_charts_epex_spot_hourly.parquet`;
- `energy_charts_epex_spot_fetch_summary.json`;
- `energy_charts_locked_holdout_capture_seal.json`.

Immediately before the first provider call, the wrapper creates the attempt
seal with exclusive creation, flush and `fsync`. It binds the plan hash, CH
bidding zone, canonical output directory and trusted system attempt time. Its
existence consumes the local T057 attempt even if the provider call then fails
or the process crashes.

On full coverage the wrapper also seals plan, acquisition, provider-raw,
Parquet and attempt-seal hashes before invoking the locked runner. The runner,
independent audit and promotion policy all require and independently replay
that chain for the T057 plan. Never overwrite, refresh, relocate or cherry-pick
these files. A crash or incomplete first attempt is a fail-closed event:
preserve the evidence and register a new future holdout rather than retrying
T057.

This is only local anti-selection evidence. It is not an external CAS, WORM or
production authority. Production remains NO-GO until the external publication
and release controls are independently proven.
