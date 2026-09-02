# LT rolling-origin evaluation protocol v2

## Scope

`pfc_shaping.lt.evaluation_protocol` locally freezes the next CH hourly-shape
comparison without opening real truth. `evaluation_challengers` implements the
four challengers only behind immutable synthetic fixtures, and
`evaluation_engine` scores synthetic predictions without ranking them. The
three modules reuse the existing LT estimand, origin-registry v2 and
dependence/power design rather than creating a second statistical authority.

The protocol is metadata only. Local Git and semantic hashes make changes
visible, but they do not replace the required external registry, trusted time,
FMV risk margins, power calibration or source admission.

Canonical semantic SHA-256:
`1134a5e24cfabc797d8931a986bce87ac983dcaaec5dd39680929463c62bdf3e`.

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
forbidden. Version 2 binds every challenger to the normalized source hash
`887f3b00d33231b52c58395ef43b5310624222922e955a6425d723e885cb5e19`.
The scoring engine is independently bound to
`034a06c14ec5aff337ab58cf4ab2e79a63c49f2f1d656dbc5fb3bd950c310ffc`.

The weighted MLP uses an exact observation-level exponential loss, a frozen
180-day half-life, a deterministic 64x64 ReLU network and analytic gradients.
Ridge and additive spline-Ridge use training-only scaling. LightGBM requires
exactly version 4.6.0, deterministic CPU mode and one worker; a missing or
mismatched optional runtime fails explicitly.

## Metrics and authority

The primary metric is monthly-level-neutralized MAE in EUR/MWh. Secondary
metrics are inherited exactly from the existing estimand, including RMSE,
bias, tail error, BASE/PEAK/OFFPEAK error and the separately gated economic
metrics. All candidates use the same complete-case rows and equal origin
weighting after within-origin energy weighting.

Only neutralized MAE, RMSE, bias and weighted P95 have a sufficiently closed
formula for the synthetic engine. Product and economic metrics remain
`UNSUPPORTED_NEVER_PASS` with an explicit missing-input reason. Results are
reported for all four lead buckets. A bucket without common rows is retained
as unsupported, so an aggregate diagnostic cannot hide missing coverage.

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

## Synthetic execution boundary

Synthetic fixtures must use unique ordered UTC timestamps, a strict
pre-origin training information set, finite numeric features and an exact
feature schema. Inputs are copied read-only. The scorer accepts only the exact
five-candidate inventory and the same complete-case intersection for every
candidate.

Synthetic fits and reports carry the source fixture identifier and an
immutable negative authority object. Their manifests state
`real_data_training_performed=false`, `real_truth_opened=false`,
`countable_origin=false` and `ranking_or_selection_performed=false`. The code
has no file, network, Databricks or CT access path.
