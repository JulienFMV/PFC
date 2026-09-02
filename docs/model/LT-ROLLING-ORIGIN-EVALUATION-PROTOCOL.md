# LT rolling-origin evaluation protocol v3

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
`e73d0b63160835f7c10b582152435361e51ac9568d433a721ded92327fac7137`.

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
forbidden. Version 3 binds every challenger to the normalized source hash
`887f3b00d33231b52c58395ef43b5310624222922e955a6425d723e885cb5e19`.
The scoring engine is independently bound to
`034a06c14ec5aff337ab58cf4ab2e79a63c49f2f1d656dbc5fb3bd950c310ffc`.
The origin-envelope boundary is bound to
`0c74cff02e886075b25d1f0893ec5b756a4e6661c88ee6cfd57aa8f07caaff8d`.

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

## External-registration preparation boundary

`pfc_shaping.lt.origin_registration_envelope` closes the local preparation
step without claiming an external registration. It builds exact canonical
schedule-entry bytes with domain-separated identities for an independent
Ed25519 signer, attaches only a caller-supplied signature, and verifies all
twelve individually signed entries against one caller-held public key. It
accepts no private key.

The signed schedule must match the exact ordered cohort, each EEX trading day
must belong to its cadence month, every capture window must contain the frozen
origin, and the registry deadline must precede the first Swiss local delivery
month. The Europe/Zurich-to-UTC conversion is checked across CET/CEST rather
than inferred from an abbreviation.

An information-set envelope requires exact hashes for the structural origin
inventory, causal EEX inventory and vintage, solver configuration, candidates,
predictions, scenarios, universe, ex-ante mask rule, calendar/strata, runtime,
wheel and source revision. Verification requires the exact signed schedule
bytes and trust key again; a different valid schedule cannot be substituted.

Even after schedule-signature verification, the envelope records the trusted
origin-time receipt, independent request signature, remote compare-and-append
receipt and fresh remote HEAD observation as missing. Every operational,
scientific, truth-opening and production authority remains false. Synthetic
signatures in tests qualify only the verifier and are not written as project
evidence.
