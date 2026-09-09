# Phase 5bis: Shape Hourly Seasonal — Context

> **STATUS: SUPERSEDED.** This pre-doc was the original Phase 5bis context (gathered 2026-05-17). Post-2026-05-18 adversarial panel review, Phase 5bis was split into:
> - **5bis-A** (no-op infrastructure refactor) — see .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/ (DELIVERED 2026-05-18).
> - **5bis-B** (math change: bowl-deepening via 3 levers) — see .planning/phases/05C-shape-hourly-bowl-deepening/ (IN PLANNING 2026-05-19).

**Gathered:** 2026-05-18 (pre-discussion, ready for `/gsd-discuss-phase 5bis`)
**Status:** Ready for discussion → planning → execution

## Phase Boundary

Extend `pfc_shaping/lt/model/shape_hourly.py` so that the hourly shape factor
`f_H` is indexed by `(saison, type_jour, hour)` instead of `(saison, type_jour)
→ np.array[24]`. Today's single `f_H[hour]` is global across all (season,
day-type) cells; the new version stores **one factor per (cell, hour)** so the
solar-bowl in summer h12-h15 and the evening peak in winter h17-h19 are no
longer collapsed onto the same hourly factor.

**In scope**:
- `pfc_shaping/lt/model/shape_hourly.py` (fit + apply + save + load)
- `pfc_shaping/lt/model/assembler.py` (consumption of `self.sh.apply()`)
- Feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` (env var)
- New `tests/test_shape_hourly_seasonal.py`
- Acceptance backtest on synthetic data inside the test suite

**Out of scope** (phases that come after):
- Negative-PFC support (Phase 5)
- Per-block probabilistic output (Phase 5ter)
- HFC-OMPEX backtest harness (Phase 10 refondu, can be parallelized)
- Fundamentals-driven shape (Phase 5quater, much later)
- Anything in `pfc_shaping/ct/*`

## Implementation Decisions (pre-discuss — to refine in `/gsd-discuss-phase`)

### f_H storage format
- **Proposal D-01**: `factors_` becomes `dict[tuple[str, str, int], float]`,
  keyed by `(saison, type_jour, hour)`. Memory cost: 4 saisons × 5 type_jour ×
  24 heures = 480 floats. Trivial.
- Alternative considered: keep nested dict `dict[tuple[str, str],
  np.ndarray[24]]` but learn per-cell-per-hour values instead of smoothing
  across hours. **Rejected** because it muddles the upgrade story (same
  type, different semantics).

### Fit method
- **Proposal D-02**: In `ShapeHourly.fit()`, current code already groups
  by `(saison, type_jour)` and computes `hourly_mean[h]` per cell. We just
  stop the gaussian smoothing across hours **across cells boundaries**
  (the smoothing within a cell stays). Each cell keeps its own 24 values.
- The current code already does this almost — line 129-152. We just need to
  store the 24-vector as 24 dict entries instead of one array.

### `apply()` interface
- **Proposal D-03**: `ShapeHourly.apply(timestamps, calendar_df,
  reference_date)` API unchanged. Return signature unchanged (Series aligned
  on 15min index). Internal lookup goes from `factors_[(saison, type_jour)][h]`
  to `factors_[(saison, type_jour, h)]`.

### Backward compatibility for save/load
- **Proposal D-04**: `save()` writes the new 3D parquet (3 key columns
  + `f_H` value). `load()` detects format by checking columns: if `heure` is
  among keys, load 3D; else load 2D (legacy) and replicate `factors_2D[h]`
  for each `h` in `range(24)` of the same cell — effectively a no-op upgrade.
- Existing fitted models on disk continue to load and produce identical
  outputs.

### Feature flag
- **Proposal D-05**: `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=0` (default OFF).
  When OFF, `factors_` is built/loaded in 2D mode (current behavior).
  When ON, `factors_` is 3D. The assembler reads via `self.sh.apply()` which
  doesn't care — it just gets a Series back.
- Flag lives in `ShapeHourly.__init__(self, use_seasonal_hourly: bool = False)`,
  defaulting from env var if not set explicitly.
- Production rollout: deploy code with flag OFF, run A/B backtest, flip ON
  if KPI passes.

### Tests
- **Proposal D-06**: New `tests/test_shape_hourly_seasonal.py` with:
  - Synthetic bowl test: inject prices `price[Ete, h=12] = 40`,
    `price[Hiver, h=12] = 100`, verify `factors_[(Ete, Ouvrable, 12)] !=
    factors_[(Hiver, Ouvrable, 12)]`.
  - Save/load round-trip: write 3D, read back, compare exactly.
  - Lazy upgrade: write legacy 2D parquet, read with new code, verify it
    works and produces identical predictions.
  - Mean-preservation: `mean(factors_[(saison, type_jour, h)] for h in 24) ≈ 1.0`
    per (saison, type_jour) cell (the energy invariant).
- Existing 142 tests in `tests/` stay green.

### Claude's Discretion
- Exact storage choice in the parquet (long format vs wide). The migration
  story is what matters; the on-disk shape is implementation detail.
- Whether to also touch `ShapeHourlyMLP` (the neural alternative): out of
  scope for this phase unless tests reveal a coupling.

## Specific Ideas

### Backtest acceptance criteria

To validate Phase 5bis before flipping the feature flag ON in production:

1. **Walk-forward backtest** on `pfc_shaping/data/epex_15min.parquet` (CH) for
   2023-2025 (training on 2-year rolling window, monthly recalibration).
2. **Block-MAE per period** (1 year holdout) on:
   - Block 10-15 weekday (mid-day)
   - Block 18-9 weekday (night, customer's nighttime block)
   - Block 12-14 weekend summer (deep bowl)
   - Global hourly MAE (no regression sentinel)
3. **DM test** (one-sided, Newey-West) on the block-MAE improvement vs the
   current baseline.
4. **Decision rule**: flip flag ON if:
   - All targeted blocks improve by at least -1.5 €/MWh (-2.0 for summer
     weekend midday).
   - Hourly MAE global doesn't regress (Δ ≥ -0.2 €/MWh tolerance).
   - DM p-value < 0.05 on at least 2 of the 3 targeted blocks.

This backtest harness is **Phase 10 refondu**, can be scaffolded in parallel
with the Phase 5bis code change (uses the current PFC as baseline, no
dependency on the new code).

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phase 5bis improves block-MAE but degrades hourly MAE | Feature flag rollback; investigate via Phase 5 (negative PFC enables proper bowl) |
| Save/load lazy upgrade fails on production artifact | Add `save_legacy_2d()` escape hatch + integration test |
| Performance regression (480 dict lookups vs 1 array index) | Profile; if hot, switch to numpy 3D array internally |
| `ShapeHourlyMLP` (neural alternative) becomes incompatible | Out of scope; current production uses table-based `ShapeHourly` |

## Notes for the planner

When `/gsd-plan-phase` runs after `/gsd-discuss-phase`, decompose into:
1. Refactor `ShapeHourly.factors_` storage from 2D to 3D (with flag).
2. Update `fit()` to write 3D when flag ON, keep 2D when OFF.
3. Update `apply()` to read 3D when flag ON.
4. Update `save/load` with lazy upgrade.
5. Write `tests/test_shape_hourly_seasonal.py`.
6. Run full `pytest tests/` to confirm no regression.
7. Commit + push (1 atomic commit, message: `feat(LT): f_H seasonal hourly indexing (Phase 5bis)`).

Phase 10 backtest scaffolding is a separate phase plan.
