# FMV Power Forward Curve

This repository builds the Swiss long-term Power Forward Curve (PFC) and keeps
the short-term forecasting overlay in a separate package boundary.

The current development status is research and governed local qualification.
No artifact is production-authorized merely because it can be built locally.

## Non-negotiable model rules

- The LT and CT implementations are independent. LT code never imports
  `pfc_shaping.ct.*`.
- The Swiss monthly BASE solver is the only monthly-level authority.
- EEX forward prices are hard monthly constraints. ENTSO-E, weather,
  neighbouring markets and benchmarks may shape a month only with zero mean.
- An hourly or quarter-hour layer cannot change a monthly mean produced by the
  solver.
- Promotion evidence must bind independent production, local-export and
  selected-model manifests.
- Restricted AFRY and OMPEX data are benchmarks only. LSEG is also a benchmark
  until a separate promotion decision says otherwise.

The complete contributor contract is in [AGENTS.md](AGENTS.md).

## Architecture

```text
pfc_shaping/
  calibration/       EEX cascading, arbitrage-free monthly solver
  lt/model/          LT hourly and 15-minute shaping, hydro, uncertainty
  ct/model/          D+1..D+10 overlay; never imported by LT
  data/              source adapters and immutable local intake contracts
  pipeline/          top-level orchestration and governed publication
  validation/        PIT, quality, backtest and promotion gates
scripts/             bounded entry points and workspace-local operations
tests/               unit, contract and adversarial tests
docs/data/           current data architecture and intake contracts
docs/research/       dated research evidence; not current authority by itself
.planning/phases/14-lt-audit-remediation/
                     durable decisions, receipts and session handoffs
```

`pfc_shaping/model/` is only the deprecated import shim. New code belongs in
`pfc_shaping/lt/model/` or `pfc_shaping/ct/model/`.

## Databricks and local data contract

The model never queries Databricks while fitting or generating a PFC. A
separate bounded extraction publishes immutable Parquet plus a manifest to the
consumer-neutral local data root selected by `FMV_DATA_ROOT`.

| Domain | Databricks authority | PFC role |
|---|---|---|
| EEX forwards | `prd.gold.facteexpricedaily` plus product and delivery-period dimensions | hard monthly solver constraints |
| Spot | Gold interval fact and product dimension | realized-price truth for calibration and backtests |
| ENTSO-E current | Gold dimension and latest fact | current serving and operational features |
| ENTSO-E resources | Gold current series-resource bridge | optional current outage/equipment enrichment |
| ENTSO-E vintages | `silver.ge_power_entsoe_time_series_vintages` | canonical PIT history, revisions and historical resource mappings |
| LSEG HPFC | selected `continuous_forward/CHE` Gold curve, latest and bounded vintages | external benchmark only |
| Weather/Swissgrid | governed Gold facts | candidate exogenous features after their own admission gates |

The former duplicate ENTSO-E Gold vintage fact is not a required PFC source.
The Gold bridge is current-state only; it cannot replace Silver
`resource_details` in point-in-time backtests.

See [LT Databricks snapshot intake](docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md)
and [shared data platform](docs/data/SHARED-DATA-PLATFORM.md). The
[data documentation index](docs/data/README.md) separates current contracts
from historical replay material.

## Main LT flow

```text
governed local snapshots
        |
        v
EEX contract selection and cascading
        |
        v
monthly BASE solver (level authority)
        |
        v
zero-mean structural shaping
        |
        v
hourly / 15-minute curve
        |
        v
independent validation and governed publication
```

`run_pfc_production.py` is the top-level orchestration entry point. Operational
commands and the allowlisted standard-user runner are documented in
[OPERATIONS.md](pfc_shaping/tools/OPERATIONS.md). On the managed workstation,
all mutable test and build output must remain below `build/`.

## Current evidence boundary

The empirical model gate remains
`BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS` until the governed local
exports, point-in-time checks and a new independently frozen future holdout are
available. Legacy local or synthetic EEX/ENTSO-E data cannot substitute for
that evidence, and T057 remains sealed.

## Where decisions live

- Current rules: [AGENTS.md](AGENTS.md)
- Durable Phase 14 decisions:
  [.planning/phases/14-lt-audit-remediation/DECISION-LOG.md](.planning/phases/14-lt-audit-remediation/DECISION-LOG.md)
- Current project handoff: [.planning/HANDOFF.md](.planning/HANDOFF.md)
- Packaging boundary: [PACKAGE.md](PACKAGE.md)
