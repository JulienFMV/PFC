# Session handoff - ENTSO-E local daily capture guard

Date: 2026-08-06  
Decision: D-20260806-246

## Outcome

Implemented a real workstation-local, repo-local, atomic daily reservation
marker for future ENTSO-E capture cost control. D238 remains synthetic evidence;
D246 is the first mechanism that actually consumes the local day on this
workspace.

The exact root is
`build/databricks-control-plane/entsoe-daily-reservations`. Each day maps to one
exclusive `YYYY-MM-DD.json` marker. Existing, corrupt, partial or divergent
markers all consume the day. Same-day replay, replacement and deletion are not
authorized. File creation is `O_EXCL`, single-link regular-file checked and
file-fsynced. Failure after creation leaves the day consumed.

Reservations bind D243 contract/query identities, a hashed Warehouse ID,
whole-second UTC and the corresponding Europe/Zurich day. No secret, host,
HTTP path or plaintext Warehouse identifier is persisted.

## Evidence and roast

- contract raw SHA-256:
  `20deb61e362362c46ab5864020a3a70d9f16bcebc41cb729030707cbea7de5e3`;
- contract content ID:
  `66f846bd91ad23e9111a216acff3ed8058929269175dc5da8d893054a176c312`;
- implementation/tests SHA-256:
  `09a9dec36db450054bba56f09a3767983273b4e03a9ae6f28f6105af98f2c0a0` /
  `f305679dea992ea427a23a33fd84cd1256979f9b7700251f493471c0df7aece2`;
- focused: `16 passed in 0.19s`;
- adjacent D238/D243/D244/D246: `67 passed in 5.45s`;
- concurrency roast: 32 contenders, exactly one successful reservation;
- injected `fsync` failure: first attempt fails but marker remains; second
  attempt is refused as already consumed;
- Ruff: pass.

## Execution counters

- Databricks requests: 0;
- SQL statements: 0;
- Warehouse starts: 0;
- Databricks writes: 0;
- ENTSO-E rows or values opened: 0.

## Residual limits

- local system UTC is unattested;
- no external monotone clock or cross-host lock;
- Windows directory-fsync and power-loss durability are unproven;
- Warehouse state and query execution are deliberately outside D246;
- no PIT, model-input or production authority.

## Changed files

- `.planning/phases/14-lt-audit-remediation/ENTSOE-LOCAL-DAILY-CAPTURE-GUARD-CONTRACT-V1.json`
- `.planning/phases/14-lt-audit-remediation/ENTSOE-LOCAL-DAILY-CAPTURE-GUARD-20260806.md`
- `pfc_shaping/data/entsoe_local_daily_capture_guard.py`
- `tests/test_entsoe_local_daily_capture_guard.py`
- `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
- `.planning/HANDOFF.md`
- this handoff.

## Next safe batch

Build the execution orchestrator with injected transport. It must reserve the
day first, perform one read-only control-plane state check, refuse every state
other than `RUNNING`, submit the exact D243 statement once, sanitize the result
locally and never retry on the same day. Do not execute it while the Warehouse
is stopped.

Predecessors:

- `SESSION-HANDOFF-20260806-ENTSOE-PARQUET-STREAMING-INTEGRITY-PREFLIGHT.md`;
- `SESSION-HANDOFF-20260806-ENTSOE-SERIES-INVENTORY-PREFLIGHT.md`.

