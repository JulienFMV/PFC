# CT Foundation Status — 2026-05

## Scope

This note records the current status of foundation-model usage for the Swiss
short-term forecaster after the May 2026 CT benchmark refresh.

It is intentionally operational. The goal is to prevent research paths from
being confused with the production baseline.

## Decision

The Swiss CT production path is:

- `LEAR` as the core forecaster
- `DE` price exogenous input as required
- `FR / AT / IT` as optional exogenous inputs
- no foundation-model dependency by default
- no FutureBoost dependency by default

In code, the production pipeline now treats the foundation path as opt-in via:

- `PFC_CT_ENABLE_FOUNDATION=1`

If the variable is unset, the Swiss CT branch runs the LEAR baseline without
Chronos.

## Why

Recent internal CT benchmarks on the Swiss worktree showed:

- `CH + DE` improves on `CH-only` for `J+1` and `J+3`
- adding `FR / AT / IT` did not improve the benchmark materially
- `FutureBoost` underperformed both the LEAR baseline and the simple
  PriceFM blends on the tested overlap

So the current evidence does not justify a production dependency on the
foundation path.

## Current status of the foundation path

### What is kept

- `pfc_shaping/model/foundation_forecaster.py`
- `scripts/finetune_chronos2.py`
- local `chronos2_finetuned/` adapter support

These remain valid as R&D tooling.

### What is not assumed anymore

- Chronos is not assumed to be the best Swiss CT model
- Chronos is not assumed to be required for production
- FutureBoost is not assumed to be a viable promotion path into production

## R&D role

The foundation path is now a challenger track.

It can still be useful for:

- controlled benchmark campaigns against `LEAR CH+DE`
- testing covariate-aware zero-shot / light-adaptation workflows
- future work on inference-time adaptation or causal normalization

But any promotion back toward production requires:

1. a rolling Swiss CT benchmark against `LEAR CH+DE`
2. a stable gain on `J+1..J+3`
3. no degradation in stressed regimes
4. a reproducible runtime and dependency story

## Recommended next CT priorities

1. push `LEAR CH+DE` further
2. improve stressed regimes:
   - negative prices
   - solar surplus
   - congestion / cross-border stress
   - atypical calendars
3. benchmark a tabular challenger before re-promoting foundation work

