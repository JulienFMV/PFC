# Session handoff - Databricks Silver/Gold readiness

Date: 2026-08-31

Branch: `fix/lt-audit-remediation`

## Outcome

The user confirmed the role-specific Silver/Gold architecture for the next LT
data admission pass. The repository already implements the required offline
boundary:

- Gold EEX facts and dimensions for monthly-solver constraint candidates;
- Gold spot and ENTSO-E dimension/latest for current serving and realized
  price truth;
- Silver ENTSO-E vintages for point-in-time availability and revision history;
- authority-negative deterministic materialization, exact replay and
  `lt_input_snapshot.v4` publication validation.

This confirmation does not admit data by layer name alone. No real PRD v4
export is present under `build/databricks-exports/`, no current cost-preflight
receipt exists, and no model or production gate was changed.

## Read-only readiness evidence

- `.env` contains non-empty Databricks host, Warehouse path and token entries,
  but the variables are not injected into the current process.
- `.env` still declares `dev.gold`; this is not evidence for the newly reported
  PRD promotion.
- `pfc_shaping/config.yaml` retains the separate legacy
  `fmv_prod.market_data` mapping and has no embedded credentials. It is not the
  Phase 14 Gold/Silver v4 source contract.
- The Databricks SQL connector, PyArrow and YAML runtime dependencies are
  available locally.
- One read-only Databricks control-plane `GET_WAREHOUSE` call reported a
  stopped Classic `2X-Small` Warehouse, one fixed cluster, 45-minute auto-stop
  and serverless disabled.
- No SQL statement, Warehouse start request, business-row read, Databricks
  write or remote write occurred.

Under the existing cost contract, a stopped Warehouse yields
`STOP_NO_ACTIVE_WAREHOUSE`. The workstation must not start it merely to perform
the audit. The next real export must either reuse a Warehouse already running
for a separately authorized workload with an accepted cost fence, or be
produced by the platform/data engineer under their own governed cost
authority.

## Verification

All mutable test paths were below
`build/pytest-databricks-silver-gold-20260831/`.

- Silver/Gold layer acceptance, materialization, snapshot v4, cost preflight
  and zero-query acquisition plan: `101 passed in 9.56s`.
- Databricks control-plane calls: `1` read-only metadata GET.
- Databricks SQL statements / Warehouse starts / business rows / writes:
  `0 / 0 / 0 / 0`.
- No restricted AFRY value was opened or reproduced.

## Next admissible step

1. Obtain a fresh value-blind PRD cost-preflight receipt or wait for the target
   Warehouse to be already running for a separately authorized workload.
2. Reconcile the exact PRD catalog, Gold/Silver table FQNs and selected columns
   with `docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`.
3. Execute one bounded read-only full snapshot with query, predicate, PIT
   watermark, row/byte/hash, query-history and cost evidence.
4. Generate the ENTSO-E `SeriesKey` mapping from the admitted Gold dimension,
   replay Silver vintages at several origins, and bind the EEX three-table join
   into the signed vintage catalogue.
5. Only after data-quality admission, freeze a new independent future holdout
   and begin rolling-origin model qualification. T057 remains sealed.

## Residual status

Model admission remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`. The CH monthly solver remains
the sole monthly-level authority; Silver/Gold data, weather, AFRY and LSEG may
not rewrite solver monthly means.

Durable decision: D-20260831-256.
