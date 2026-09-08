# LT rolling-origin evaluation protocol v6

## Scope

`pfc_shaping.lt.evaluation_protocol` locally freezes the next CH hourly-shape
comparison without opening real truth. `evaluation_challengers` implements the
four challengers only behind immutable synthetic fixtures, and
`evaluation_engine` scores synthetic predictions without ranking them. The
companion `evaluation_reference` contract places the transparent seasonal
baseline outside the five-model selection inventory. These modules reuse the
existing LT estimand, origin-registry v2 and dependence/power design rather
than creating a second statistical authority.

The protocol is metadata only. Local Git and semantic hashes make changes
visible, but they do not replace the required external registry, trusted time,
FMV risk margins, power calibration or source admission.

Canonical semantic SHA-256:
`152b453db2e19e4ea16007865cfabb91bf9147cef46105da1f9f584956769fbd`.

## Candidate family

The comparison contains exactly one incumbent and four challengers:

1. the current unweighted 64x64 MLP, bound to the current source and
   configuration through cross-platform UTF-8/LF-normalized SHA-256 values;
2. the same MLP family with true observation-level 180-day decay weighting;
3. standardized Ridge;
4. an additive cubic-spline Ridge GAM without interactions;
5. deterministic CPU LightGBM with an explicit bounded grid.

The market-constrained seasonal baseline is the primary promotion reference,
not a sixth selectable model. It is scored separately before candidate ranking
on the exact same origins, rows, masks, energy weights, metrics and horizon
buckets. It has no hyperparameter tuning and cannot enter candidate selection.
This preserves the exact five-model protocol while making the product charter's
simple-baseline requirement explicit. Its companion semantic contract is
`pfc_shaping.lt.evaluation_reference`; the real-data implementation and scorer
remain pending.

The incumbent has no tuning grid. Challenger tuning may occur only inside
nested, externally registered development origins. Future-holdout tuning is
forbidden. Version 6 binds every challenger to the normalized source hash
`887f3b00d33231b52c58395ef43b5310624222922e955a6425d723e885cb5e19`.
The scoring engine is independently bound to
`034a06c14ec5aff337ab58cf4ab2e79a63c49f2f1d656dbc5fb3bd950c310ffc`.
The origin-envelope boundary is bound to
`0c74cff02e886075b25d1f0893ec5b756a4e6661c88ee6cfd57aa8f07caaff8d`.
The request-preparation boundary is bound to
`e4dc1366ac1e13c1f2b4083ac6c8d5f24036cd743cae678359bae8f1435ed748`.
The receipt/HEAD verifier is bound to normalized source hash
`d3224f53aa1912301ac87f5db0c4721547978c1da17358411d37adc558299eb7`
and its exact wire-contract file to
`1f9d1f6495f716b7afb3497195626b24fe427b7dedf3f6f001c00051d6688047`.
The registry trust/conformance implementation is bound to normalized source
hash
`e7defb0700665bb68ca4debe490b6920816739f6882850752f304827b17cfdf1`
and its exact trust/transport contract file to
`57ce79771d25cf6a858a2c2fef585d202b66d05aff6a107b834f2afb9407df39`.

The weighted MLP uses an exact observation-level exponential loss, a frozen
180-day half-life, a deterministic 64x64 ReLU network and analytic gradients.
Ridge and additive spline-Ridge use training-only scaling. LightGBM requires
exactly version 4.6.0, deterministic CPU mode and one worker; a missing or
mismatched optional runtime fails explicitly.

Available GPU compute does not change the frozen LightGBM identity or the
seasonal reference: both remain deterministic CPU executions. GPU acceleration
is reserved for separately qualified nonlinear work, frozen-weight inference,
scenario transforms and repeated scoring. The current runtime contract forbids
GPU fit or model selection until CPU/GPU parity, deterministic settings and
runtime evidence have passed; the CPU float64 implementation remains the hard
gate oracle.

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
candidate. The seasonal reference is deliberately absent from that selection
inventory; its separate scorer is not yet implemented, so no synthetic or real
reference score is claimed by v6.

Synthetic fits and reports carry the source fixture identifier and an
immutable negative authority object. Their manifests state
`real_data_training_performed=false`, `real_truth_opened=false`,
`countable_origin=false` and `ranking_or_selection_performed=false`. The code
has no file, network, Databricks or CT access path.

## Common PRD input boundary

`pfc_shaping.lt.evaluation_inputs` is the pure bridge between already
materialized enterprise PRD snapshots and the future real-data evaluation
runner. It accepts no connector or query configuration. For one frozen origin,
it requires exact training and prediction frames plus the explicit ordered
hourly feature inventory frozen by `pfc_shaping.lt.evaluation_feature_inventory`.

Training rows contain row identity, delivery time, maximum dependency
availability, an incumbent-equivalent hourly `target_f_h` and features; both
delivery and availability must be strictly before the origin. Raw
`target_eur_mwh` is not accepted at this boundary. Prediction rows contain no
target, must be delivered at or after the origin, and every feature dependency
must be available no later than the origin. Source roles are bound by snapshot
SHA-256 only.

One complete-case mask per split is applied before any model sees the arrays.
The resulting feature order and eligible rows are common to the seasonal
reference, incumbent and all challengers. Outputs are detached read-only arrays
with value and identity hashes, but all acquisition, training, truth-opening,
selection, monthly-level, publication and production authorities remain false.

The hourly candidate matrix contains exactly nine entries, in order:
`hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos`,
`is_holiday`, `hydro_fill` and `years_ahead`. Calendar and maturity values are
deterministic from delivery/origin. Historical hydro is allowed only when its
realized value was available before the origin; future hydro is the week-of-year
climatology fitted strictly before that origin. Nulls are preserved until the
one common mask is formed.

The incumbent's three outage positions are not candidate data columns. The
governed configuration disables outages, so they remain explicit fixed
compatibility constants inside the incumbent replay. This is a disabled-feature
state, not permission to turn missing observations or forecasts into zero.
Raw PRD ENTSO-E load, solar, wind and cross-border actuals are excluded because
their future same-target values do not exist at the origin. The causal
`solar_regime`, `load_deviation` and `flow_deviation` climatologies remain
separate quarter-hour shaping context applied identically downstream of every
hourly candidate; they are not extra challenger-only predictors.

`pfc_shaping.lt.evaluation_feature_builder` constructs this exact matrix from
an already materialized three-column input: delivery UTC, hydro availability
UTC and normalized hydro fill. It verifies the frozen origin, split-specific
delivery side, hydro role, availability and prediction-climatology cutoff. It
uses Europe/Zurich plus the existing Valais/German holiday precedence and calls
the incumbent encoder directly. `hydro_fill` must already be a fraction in
`[0,1]`; percentages, infinities and non-numeric values fail instead of being
silently normalized. Missing hydro remains null for the downstream common
mask. Exact input columns prevent raw ENTSO-E actuals or outage fields from
entering this boundary.

`prepare_constructed_prd_origin_inputs` performs the next and only assembly
step. Training metadata supplies row identity, delivery, target availability
and `target_f_h`; prediction metadata supplies only identity and delivery. The
adapter requires exact one-to-one delivery timestamp sets, aligns feature
values by timestamp rather than caller row position and computes training
availability as `max(target availability, hydro availability)`. Prediction
availability is the admitted hydro-climatology availability. Before delegation
to the common PRD validator it reverifies feature shape, units, read-only state,
negative authority and the value/delivery hashes of both constructed batches.
No second masking or feature implementation is introduced. Construction and
parity verification of `target_f_h` remain a separate prerequisite; callers
cannot fit these arrays directly from raw price levels.

## Candidate execution interface boundary

`pfc_shaping.lt.evaluation_execution_contract` freezes the only compatible
path found by the source audit. The incumbent is replayed through the
hash-bound native `ShapeHourlyMLP.fit/apply` interfaces. It is not replaced by
a generic MLP. The four challengers may consume the common nine-column matrix,
but their learning target must reproduce the incumbent's hourly `f_H`
construction: direct CH quarter-hour price divided by the Swiss-local daily
mean, days with mean at most 5 EUR/MWh excluded, ratios clipped to `[0.2, 3.0]`
and aggregated by Swiss-local date and clock hour with the incumbent weights.
The repeated autumn clock hour is merged because that is the frozen incumbent
behaviour; silently correcting it would change the baseline.

At prediction time every candidate produces `f_H` on the native quarter-hour
UTC delivery grid. Positivity floor, Swiss-local daily normalization and final
`[0.4, 2.0]` clipping are applied exactly once. Those factors are not valid
inputs to the EUR/MWh scoring engine. Each candidate must first pass through
the same solver-level and downstream curve assembly; only the resulting full
price curves are scored after separate energy-weighted local-month centering.
The CH monthly BASE solver remains unchanged and authoritative.

This contract is metadata-only and authority-negative. It does not authorize
real-data fitting, use of the synthetic challenger laboratory as a real
runner, truth opening, Warehouse or GPU execution, ranking, publication or
production.

`pfc_shaping.lt.evaluation_target_builder` implements the first pure step. It
accepts only already-materialized CH delivery UTC, price-availability UTC and
quarter-hour price columns. It checks the frozen origin and quarter-hour grid,
sorts deterministically, reproduces the incumbent daily-ratio, clipping,
recency-weighted local-hour aggregation and repeated-autumn-hour merge, and
emits detached `target_f_h` metadata with exact identity, timestamp and value
hashes. Each hourly target is represented by its earliest contributing UTC
timestamp and is available only when its latest contributing price is
available. Missing prices follow the incumbent's pre-target exclusion;
infinities fail closed. Synthetic capture tests compare its target and first
nine features directly with the preprocessing arrays passed by the hash-bound
native incumbent to a no-training test double.

`pfc_shaping.lt.evaluation_factor_postprocess` closes the corresponding
challenger-only prediction seam. It accepts raw factors only for one of the
four frozen challenger IDs and rejects the incumbent explicitly, preventing
its native post-processing from being applied twice. On a timezone-aware
quarter-hour grid at the frozen origin it applies the exact native sequence:
floor at `0.1`, arithmetic normalization within each Swiss-local day, then
clip to `[0.4, 2.0]`. The detached result binds raw values, delivery timestamps
and final factors separately. A synthetic fixed-prediction test proves exact
equality with `ShapeHourlyMLP.apply`; no model is fitted and the factors still
carry no EUR/MWh, scoring, selection or production authority.

`pfc_shaping.lt.evaluation_curve_assembly` closes the final local numerical
seam without adding a second curve formula. It requires one solver-authority
`PFCAssembler` retaining the exact native `ShapeHourlyMLP` and the exact four
postprocessed challenger batches. A shallow assembler copy calls the incumbent
`apply` natively; challenger copies override only that returned `f_H`. The
override carries immutable copies of the incumbent `f_W` maps, while every
candidate reuses the same monthly solver prices, quoted products, quarter-hour
ENTSO-E context, intraday model and water-value input.

The adapter rejects legacy monthly level paths, uncertainty output and the
unfrozen solar, electrification and amplitude layers. Outage context is not an
input because D291 excludes those disabled positions from the common candidate
inventory. After assembly it requires bit-identical `B`, `f_S`, `f_W`, `f_Q`,
`f_WV`, `delta_wv` and `f_bridge`, a completed final solver projection and
EUR/MWh monthly means equal to `B`. It emits five detached read-only price
vectors accepted directly by `SyntheticEvaluationSet`; it does not fit, open
truth, score, rank or grant authority.

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

## Registration-request construction boundary

`pfc_shaping.lt.origin_registration_request` translates one reverified
information-set envelope into the exact field inventory of
`ch_lt_origin_registration_request.v2`. It derives the origin ID and request ID
with the protocol domains, binds the expected compare-and-append predecessor,
the selected signed schedule entry, every information-set commitment and the
SHA-256 of exact opaque trusted-time receipt bytes.

The production module accepts only public keys and caller-supplied signature
bytes. It rejects noncanonical UUIDs, invalid sequence/predecessor pairs,
request/schedule signer-role collapse, chronology violations, malformed
base64, changed hashes and requests re-signed after information-set mutation.

Successful verification proves only cryptographic integrity under the supplied
keys. Trusted-time semantics and the external request-signer role remain
unadmitted; no origin becomes registered or countable.

## Receipt and fresh-HEAD construction boundary

`pfc_shaping.lt.origin_registration_receipt` implements a local, hash-frozen
wire draft for `ch_lt_origin_registration_receipt.v2` and
`ch_lt_origin_registry_head_observation.v1`. Receipt and observation IDs use
separate domain-separated SHA-256 derivations; Ed25519 signatures use distinct
signature domains and exact canonical bytes. The verifier requires a registry
public key distinct from the request and schedule keys and reverifies the
complete request/envelope/schedule chain.

The HEAD binds the exact receipt bytes, sequence and caller-supplied SHA-256
nonce. Freshness requires
`committed <= observed <= caller_verification_time <= expires`, with a strictly
positive TTL no longer than 300 seconds. The library does not generate a
nonce, own a private key, read a clock, perform I/O or execute registry CAS.

A cryptographically valid receipt must carry the protocol's exact
`countable_prospective_origin=true` wire claim. This claim is reported
separately from local authority: even with a valid fresh HEAD,
`externally_registered`, `countable_origin`, truth opening, training,
selection, scientific admission, production and promotion all remain false.
They require external approval of the wire contract, registry-key trust
admission, an independently operated remote CAS/WORM service, trusted commit
time and conformance/security evidence.

The wire contract ID is
`290b108770dc799eeb83ca9f0a046aa8436878858224b68f757a5d9bfacbcc33`;
its exact file SHA-256 is
`1f9d1f6495f716b7afb3497195626b24fe427b7dedf3f6f001c00051d6688047`.
The incompatible SQLite reference remains test-only and is not promoted.

## Registry trust and transport-conformance boundary

`pfc_shaping.lt.origin_registry_conformance` verifies exact signed trust
bundles under a caller-supplied Ed25519 root public key. Bundles contain a
strictly ordered public registry-key inventory, exactly one currently active
signer, half-open validity windows and an append-only lifecycle chain.
Historical keys may verify receipts only inside their frozen window;
revoked or compromised keys are always rejected. The trust root and registry
signer roles must remain cryptographically distinct.

The same module provides a deterministic, thread-safe, in-memory model of the
transport-neutral `get_head`, compare-and-append and operation-lookup
semantics. It exercises atomic sequence/predecessor comparison, uniqueness,
exact idempotent retry, divergent-retry rejection, immutable committed bytes
and sanitized rejected-operation retention. Candidate preparation reverifies
the complete receipt/request/envelope/schedule chain against the selected
bundle key before this synthetic state-machine test.

This is not a registry client or service. It owns no private key, credential,
clock or nonce; performs no filesystem, database or network I/O; and cannot
provide remote linearizability, WORM retention, service identity or trusted
commit time. A locally valid trust-root signature admits neither that root nor
the registry service. All registration, countability, truth-opening,
training, selection, scientific, production and promotion authorities remain
false.

The trust/transport contract ID is
`84839a92bc62426c964da9eaefcf719be80808c79089b7ac2142af14d033206b`;
its exact file SHA-256 is
`57ce79771d25cf6a858a2c2fef585d202b66d05aff6a107b834f2afb9407df39`.
