# ENTSO-E real local daily capture guard

Date: 2026-08-06  
Decision: D-20260806-246

## Outcome

The workstation now has a real repo-local cost guard that atomically consumes
one Europe/Zurich day before any future ENTSO-E Warehouse or SQL action.

The marker is created exclusively at
`build/databricks-control-plane/entsoe-daily-reservations/YYYY-MM-DD.json`.
Any existing marker, including a partial or corrupt one, means the day is
consumed. A publication or `fsync` failure after file creation also consumes
the day. The marker is never deleted or replaced automatically.

## Closed risk

D238 demonstrated the daily rule only with synthetic ledgers and explicitly
had no real reservation authority. D246 implements the workstation-local
mechanism needed for cost control. A 32-way concurrent roast produced exactly
one winner and 31 refusals.

The reservation binds the exact D243 query and contract, a hashed Warehouse
identifier, a fresh whole-second UTC time and its Europe/Zurich date. It stores
no token, host, HTTP path or plaintext Warehouse identifier.

## Limits

This is a local workstation guard, not an external distributed lock:

- cross-host uniqueness is not proven;
- the clock is local and unattested;
- Windows file `fsync` is used, but directory-fsync and power-loss durability
  are not proven;
- Warehouse state is not checked here;
- no SQL execution is authorized;
- no data, PIT, model or production authority is created.

The next execution batch must first reserve the day, then verify that the
selected Warehouse is already `RUNNING`, and only then may submit the exact
D243 read-only statement once.

## Evidence

- contract raw SHA-256:
  `20deb61e362362c46ab5864020a3a70d9f16bcebc41cb729030707cbea7de5e3`;
- contract content ID:
  `66f846bd91ad23e9111a216acff3ed8058929269175dc5da8d893054a176c312`;
- implementation SHA-256:
  `09a9dec36db450054bba56f09a3767983273b4e03a9ae6f28f6105af98f2c0a0`;
- tests SHA-256:
  `f305679dea992ea427a23a33fd84cd1256979f9b7700251f493471c0df7aece2`;
- focused roast: `16 passed`;
- adjacent D238/D243/D244/D246 slice: `67 passed`;
- Ruff: pass;
- D246 Databricks requests, SQL statements, Warehouse starts and writes: zero.

