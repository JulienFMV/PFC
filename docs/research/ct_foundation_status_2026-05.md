# CT Foundation Status — 2026-05

## Scope

This note records the current status of foundation-model usage for the Swiss
short-term forecaster after the May 2026 CT benchmark refresh.

It is intentionally operational. The goal is to prevent research paths from
being confused with the production baseline.

## Decision

The Swiss CT production path is now:

- `LEAR` as the core forecaster
- `DE` price exogenous input as required
- `FR / AT / IT` as optional exogenous inputs
- `Chronos-2 finetuned` blend enabled by default on `J+1` only
- no FutureBoost dependency by default

In code, the Swiss CT branch now enables the foundation path by default and
allows opt-out via:

- `PFC_CT_ENABLE_FOUNDATION=0`

The foundation blend is capped to `J+1` through
`foundation_blend_max_horizon_days=1`, so longer horizons keep the standard
LEAR path.

## Why

Recent internal CT benchmarks on the Swiss worktree showed:

- `CH + DE` improves on `CH-only` for `J+1` and `J+3`
- adding `FR / AT / IT` did not improve the benchmark materially
- `FutureBoost` underperformed both the LEAR baseline and the simple
  PriceFM blends on the tested overlap
- `Chronos-2 finetuned` improves the Swiss `J+1` baseline materially when
  blended after LEAR post-processing, while a `J+1`-only cap prevents
  unnecessary degradation on longer horizons

The current evidence therefore justifies a production dependency on the local
Chronos path for `J+1`, but not a broader promotion of the foundation path
across all horizons.

## Current status of the foundation path

### What is kept

- `pfc_shaping/model/foundation_forecaster.py`
- `scripts/finetune_chronos2.py`
- local `chronos2_finetuned/` adapter support

These remain valid as R&D tooling.

### What is not assumed anymore

- Chronos is not assumed to be the best Swiss CT model on every horizon
- Chronos is not assumed to replace LEAR as the core Swiss CT forecaster
- FutureBoost is not assumed to be a viable promotion path into production

## R&D role

The foundation path is no longer pure R&D. It is a promoted production
component on `J+1`, while still remaining a challenger track outside that
scope.

It remains useful for:

- controlled benchmark campaigns beyond `J+1`
- testing covariate-aware zero-shot / light-adaptation workflows
- future work on inference-time adaptation or causal normalization

Any broader promotion beyond `J+1` still requires:

1. a rolling Swiss CT benchmark against `LEAR CH+DE`
2. a stable gain on the target horizon
3. no degradation in stressed regimes
4. a reproducible runtime and dependency story

## Recommended next CT priorities

1. push `LEAR CH+DE + Chronos(J+1)` further
2. improve stressed regimes:
   - negative prices
   - solar surplus
   - congestion / cross-border stress
   - atypical calendars
3. benchmark a tabular challenger before widening foundation usage beyond `J+1`

## 2026-05-13 snapshot refresh

After refreshing the live Swiss CT snapshot through `2026-05-12`, the main
failures shifted toward spring daytime low-price regimes, with the most severe
errors concentrated on `8h-17h` during `1-2 May 2026`.

The operational consequence is narrow:

- keep foundation bounded to `J+1`
- do not widen it to longer horizons
- give the finetuned Chronos blend more weight on `8h-17h` for `J+1`

This is a targeted response to the current live midday stress regime, not a
general change in the long-horizon CT architecture.
