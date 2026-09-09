# Session handoff - FMV PFC target and next-session restart - 2026-09-02

## Start here

This is the canonical restart document for the next session. Read, in order:

1. repository-root `AGENTS.md`;
2. this handoff;
3. `docs/model/PFC-HEDGING-SCOPE-AND-MODEL-ROADMAP.md`;
4. `SESSION-HANDOFF-20260902-ENTSOE-EXPORT-HEDGING-SCOPE.md`;
5. durable decisions D-20260902-267 through D-20260902-269.

Do not begin by retraining, opening T057, starting Databricks compute or adding
a new deep model.

## User-validated business objective

Build a Swiss quarter-hour PFC useful to FMV's primarily Swiss hydro and
alpine-PV activity, Valais client portfolios, large industrial consumers and
hedging on CH and DE markets. FR, AT and IT-North may first improve observation,
spread and risk analysis and may become hedge-execution markets only after
FMV's access and the required market controls are admitted.

The PFC should approach a plausible central view of hypothetical future Swiss
spot prices while supporting explicit regimes such as dry/wet hydro, cold/mild
weather, PV penetration, electrification, new or retired plants, outages,
congestion and other forces that alter intraday, daily, monthly, quarterly or
annual shape.

## Settled product architecture

Keep these products separate:

1. `market_central_ch`: EEX-forward-consistent valuation curve;
2. `fundamental_scenario_ch`: conditional physical/structural price curves;
3. `stochastic_spot_paths_ch`: coherent paths around admitted scenarios;
4. FMV portfolio exposure and valuation;
5. FMV hydro optimisation and water values;
6. hedge recommendations using admitted CH/DE instruments.

The central curve is not the exact future spot. Shape-only effects remain
zero-mean inside the solver bucket. A level-changing scenario must be a
separate upstream fundamental solve and must not overwrite the central curve.

## Verified current truth

- The constrained CH monthly solver is the sole current level authority.
- The DE branch exists but follows a legacy monthly path and is not a governed
  peer of CH.
- The active hourly baseline is a 64x64 ReLU MLP, but its computed 180-day
  weights do not reweight samples in the final `MLPRegressor.fit`.
- DE-LU Huber/Ridge quarter-hour shaping is a transfer baseline, not proven CH
  truth.
- Aggregate reservoir regression is a market hydro-shape proxy, not an FMV
  asset water-value optimiser.
- ENTSO-E climatology is a conservative future-input baseline, not a structural
  scenario engine.
- Probabilistic P10/P90 output is disabled pending Swiss rolling-origin
  calibration.
- LT must not import `pfc_shaping.ct.*`; LEAR/Chronos remain separate CT work.
- Producer-normalised ENTSO-E intervals are consumed without `curve_type`.
- `causal_asof` and `realized_final` are separate, value-bound contracts.

## Current implementation evidence

The preceding export handoff records exact files and hashes. Its final local
qualification was:

- export/evidence/replay: 14 passed;
- adjacent ENTSO-E/Databricks matrix: 153 passed;
- complete non-slow ENTSO-E/LSEG/spot matrix: 854 passed;
- LT minimum plus package contract: 84 passed, 1 optional skip;
- targeted Ruff and hostile static checks: pass.

No Databricks connection, query, Warehouse start, business row, write or
incremental cost was used for that final export increment. The observed
Warehouse remained stopped.

## First next-session task

Before writing model code, produce a compatibility and gap map for the existing
LT interfaces by inspecting only the relevant files:

- `pfc_shaping/calibration/monthly_forward_curve.py`;
- `pfc_shaping/pipeline/production_phases.py`;
- `pfc_shaping/lt/model/assembler.py`;
- `pfc_shaping/lt/model/shape_hourly.py` and `shape_hourly_mlp.py`;
- `pfc_shaping/lt/model/shape_intraday.py`;
- `pfc_shaping/lt/model/water_value.py`;
- `pfc_shaping/lt/model/msfc_spline.py`;
- `pfc_shaping/lt/model/uncertainty.py`;
- `pfc_shaping/lt/model/pfc_flavors.py`.

Map each current input/output to the six target products above. Identify where
scenario identity, information timestamp, level authority, normalization
bucket and provenance must enter without changing the governed baseline.

The first safe coding milestone should then be an authority-negative contract
and synthetic-test layer for scenario definitions and curve-product typing.
It must not alter `production_phases.py`, select a champion or claim real-data
admission. Proposed paths must be checked against the existing package layout
before creating files.

## Subsequent execution order

1. Close governed EEX/ENTSO-E/LSEG data and future-holdout gates.
2. Freeze targets, origins, metrics, thresholds and tie-break rules.
3. Replay the current CH candidate unchanged.
4. Compare calendar/Ridge/GAM/LightGBM/current-MLP/weighted-MLP hourly shapes.
5. Select the quarter-hour model separately using native CH truth.
6. Build typed P and Q scenario curves and coherent stochastic paths.
7. Build a DE curve with its own EEX level authority and CH-DE dependence.
8. Add FMV exposure, hydro optimisation and hedge optimisation downstream.
9. Run independent holdout, shadow, manifest and rollback gates before any
   promotion.

## Non-negotiable blockers and invariants

- Global model status:
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`.
- T057 remains sealed.
- AFRY Batch 4 and AFRY-driven rolling-origin selection remain blocked.
- No individual post-solver month patches.
- No neighbouring, hydro, weather, client or scenario layer may silently
  rewrite central CH solver means.
- No market or scenario label grants model, publication, trade or production
  authority.
- Preserve all existing tracked and untracked user/session work.

## Suggested new-session prompt

> Read AGENTS.md and
> `.planning/phases/14-lt-audit-remediation/SESSION-HANDOFF-20260902-PFC-FMV-TARGET-AND-NEXT-SESSION.md`
> completely. Resume from its first next-session task. Audit the existing LT
> interfaces, update the plan, and implement only the authority-negative
> scenario/curve contract milestone with synthetic tests. Do not retrain,
> query Databricks, start a Warehouse, open T057, touch CT or change the CH
> solver's monthly-level authority.
