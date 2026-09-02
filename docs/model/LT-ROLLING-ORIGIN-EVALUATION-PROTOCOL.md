# LT rolling-origin evaluation protocol v1

## Scope

`pfc_shaping.lt.evaluation_protocol` locally freezes the next CH hourly-shape
comparison without fitting a model or opening truth. It reuses the existing LT
estimand, origin-registry v2 and dependence/power design rather than creating a
second statistical authority.

The protocol is metadata only. Local Git and semantic hashes make changes
visible, but they do not replace the required external registry, trusted time,
FMV risk margins, power calibration or source admission.

Canonical semantic SHA-256:
`c2705a8d175bfe7421e2722d316bee4a5eb5631506f284dde03363ab561cb26b`.

## Candidate family

The comparison contains exactly one incumbent and four challengers:

1. the current unweighted 64x64 MLP, bound to the current source and
   configuration through cross-platform UTF-8/LF-normalized SHA-256 values;
2. the same MLP family with true observation-level 180-day decay weighting;
3. standardized Ridge;
4. an additive cubic-spline Ridge GAM without interactions;
5. deterministic CPU LightGBM with an explicit bounded grid.

The incumbent has no tuning grid. Challenger tuning may occur only inside
nested, externally registered development origins. Future-holdout tuning is
forbidden. Challenger implementation hashes remain absent until the code and
runtime exist; consequently this contract cannot authorize training.

## Metrics and authority

The primary metric is monthly-level-neutralized MAE in EUR/MWh. Secondary
metrics are inherited exactly from the existing estimand, including RMSE,
bias, tail error, BASE/PEAK/OFFPEAK error and the separately gated economic
metrics. All candidates use the same complete-case rows and equal origin
weighting after within-origin energy weighting.

Market consistency remains an uncompensated hard gate. Statistical margins
remain pending the existing FMV-risk MDE and power design. Insufficient power
or coverage is `UNSUPPORTED_NEVER_PASS`.

The CH monthly BASE solver remains the sole monthly-level authority. Shape
scores subtract the energy-weighted local delivery-month mean separately from
prediction and truth; they cannot reward a challenger for rewriting the
forward level.

## Prospective cohort

The first new cohort schedules 12 monthly origins from October 2026 through
September 2027. Each origin covers lead months 1 through 36 and the four
existing horizon buckets. The timestamps are locally frozen before their
windows, but every slot has:

- `externally_registered=false`;
- `countable_origin=false`;
- `truth_open_authorized=false`.

Therefore the scheduled count is 12 while the countable scientific count is
zero. A missed slot is never shifted, backfilled or reweighted. The cohort may
need extension after the outcome-blind power design determines the required
number of unique origins; twelve is not a promotion claim.

T057 is neither read, referenced nor reused. ENTSO-E recovery and admitted
prospective availability remain acquisition prerequisites, not reasons to
alter the schedule after observing outcomes.
