# CH LT PIT and probabilistic preregistration — draft 2026-07-24

Status: `DRAFT_BLOCKED_NOT_EXECUTABLE`  
Production: strict `NO_GO`  
Protocol name: `ch_lt_pit_outer24_future4_20260724_v1`  
Canonical draft JSON: `CH-LT-PIT-PROBABILISTIC-PREREGISTRATION-DRAFT-20260724.json`  
Canonical plan id: `ae5557fd7e58a6ee4164e7f8a949cb379fc2d8ac23766e17a1873c4de420c5f6`  
Current JSON file SHA-256: `aba798530084b7031a0ac38b1c48b20cff575d6082edbcf37c9a04528900ba61`

## Decision

Do not run another CH challenger, rolling-origin campaign or future holdout from
the current local evidence. First turn this draft into a separately governed,
trusted-time-frozen plan with an exact origin inventory, exact future episodes,
admitted PIT data manifests and independent review receipts.

The contract validator is
`pfc_shaping.validation.ch_lt_pit_preregistration`; the fail-closed preflight is
`scripts.audit_ch_lt_pit_preregistration`. A structurally valid draft can be
checked with `--mode validate-draft`. The default `admit-execution` mode always
exits non-zero in schema v1. This validator is deliberately incapable of
turning self-declared receipt hashes or booleans into execution authority. A
future, distinct path-based envelope must authenticate the immutable plan core,
independent signatures, trusted time, CAS/HEAD, runtime, reviews and one-shot
state before issuing any non-production capability. No such envelope exists.

## Why current evidence is insufficient

The direct CH Energy Charts capture covers only 2026-06-24 through 2026-07-24,
has hourly provider cadence, uses an untrusted workstation timestamp, is
unsigned and remains in a Builder-mutable local namespace. Exact local replay
does not convert it into an independent point-in-time authority. The DE-LU
15-minute panel is mixed-authority, contains an ungoverned historical base and
has no demonstrated DE-to-CH transfer validity. It is diagnostic-only.

Consequently, none of these files is admissible for training, model selection
or holdout scoring under this protocol. In particular, repeating hourly CH
values over quarter-hours cannot validate native 15-minute shape.

## Draft quantitative hypotheses — not frozen

These values are initial research hypotheses encoded in the blocked schema v1.
They are not yet a defensible freeze design. In particular, the independent
Quant/Data roast rejected `n=24`, the block length and the HAC lag as
unsupported without an LT horizon-specific power/MDE and dependence study. A
future schema must replace them rather than treating them as promotion gates:

- 24 distinct chronological monthly outer origins, six per season;
- at least four origins in each pre-defined regime: hydro low/high, negative
  price, price spike, renewable high and cross-border stress; overlaps are
  allowed, but regime definitions must be hash-bound before evaluation;
- the origin is the inference unit; hours and quarter-hours are not counted as
  independent experiments;
- nested model, feature and hyperparameter selection inside every outer origin,
  at least 12 inner origins and a one-month embargo;
- four untouched 14-day future episodes, one per season, each strictly after
  the independent trusted freeze and executed once in append-only namespaces;
- primary deterministic metric: monthly-level-neutralized MAE, aggregated
  within origin before equal-origin aggregation;
- provisional statistical gate: at least 2% MAE improvement, no aggregate
  RMSE regression, at least 70% positive origins, no mandatory-stratum
  degradation above 2%, and a strictly positive 95% lower confidence bound;
- family-wise alpha 5% with Holm adjustment; circular moving-block bootstrap
  on origins, block length three, 10,000 PCG64 replicates, seed 20260724;
  Newey-West/HLN DM with lag two is a declared cross-check, not a substitute
  for origin-level inference;
- deterministic comparators on identical evidence: market-constrained
  seasonal naive, current incumbent, robust regularized linear/GAM and one
  justified nonlinear/tree challenger;
- OMPEX is diagnostic benchmark-only and forbidden from fit, selection and
  truth;
- quantiles `0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99` and 1,000
  PCG64 scenarios with seed 20260724;
- pinball, WIS, CRPS, coverage, sharpness and PIT/rank diagnostics for
  marginals; energy and variogram scores for paths; the variogram uses order
  0.5 and quarter-hour lags 1, 4, 16, 48 and 96 with inverse-lag weights;
- every scenario is projected pathwise to the monthly solver authority within
  `1e-9 EUR/MWh` and preserves hard product identities. Marginal quantile
  curves are required to be non-crossing but are not separately forced to
  reprice the monthly mean: doing so at every quantile would collapse any
  non-degenerate ordered distribution;
- final EEX repricing `1e-6`, monthly solver and downstream monthly-level
  preservation `1e-9`, cascade invariance `1e-8`, plus the D155
  support/year/bucket/row-L1 sensitivity gates `2 / 2 / sqrt(5) / 3`;
- effective degrees of freedom, active basis, condition number, regularization
  and feature-group ablations are mandatory. Complexity is selected inside
  each origin and must earn paired out-of-sample value.

The 2%/70%/2% statistical values are an explicit FMV research hypothesis, not
universal constants claimed by the literature. They did not survive the first
independent Quant/Data roast as an executable specification: the full
hypothesis family, score formulas, dependence calibration, Monte-Carlo error,
strata and power must be frozen first.

## Data admission boundary

Every unconditional role — exact EEX forward source, governed forward-history
vintage and direct CH spot — must have exact provider bytes, source-specific
replay equality, complete `available_at <= origin` evidence, independent
trusted time and signature, a Builder-inaccessible immutable namespace, and an
external-CAS receipt plus fresh authenticated HEAD. Conditional hydro,
ENTSO-E, commodity, outage and neighbouring-market roles must satisfy the same
controls whenever a candidate consumes them. Missing critical data is
`UNSUPPORTED` or failure, never economic zero.

## Economic decision boundary

The statistical gates are not an FMV risk appetite. Before freeze, Risk must
approve and sign:

- the exact Fil, Acc and dispatch profile population;
- transaction and imbalance assumptions;
- a strictly positive minimum improvement in profile capture-value error,
  expressed in EUR/MWh;
- the materiality aggregation and tie policy.

No numerical economic threshold is invented in this draft. Its absence is an
intentional execution blocker.

## Current blockers reported by the validator

1. no independent governance/trusted-time freeze;
2. no governed PIT manifests for the unconditional data roles;
3. no exact 24-origin inventory or frozen regime-definition manifest;
4. no exact four-episode future inventory;
5. no FMV Risk-approved capture-value materiality threshold;
6. no independent Security, IT/Operations and Quant/Data review receipts for
   the frozen bytes.

The post-roast validator additionally and unconditionally blocks execution
until all of the following are implemented in a new contract version:

- a receipt-free immutable plan core plus a separate, path-verified external
  admission envelope, avoiding circular signatures over bytes that contain
  their own receipts;
- a dependence-aware power/MDE and effective-sample-size study;
- exact LT horizon, target, truth, eligibility-mask and nested inner-fold
  inventories for every outer origin;
- direct CH 15-minute truth or an explicit `UNSUPPORTED_15MIN` claim, plus
  post-episode outcome receipts that cannot exist at initial plan freeze;
- the exact multiplicity family, calibration tolerances, score formulas,
  confidence intervals and Monte-Carlo error/coupling policy;
- exact Fil/Acc/dispatch profile artifacts, volume/calendar/sign conventions,
  capture-value formula and per-profile non-regression;
- a durable attempt seal/ledger and a CPU/GPU runtime, determinism and parity
  contract.

## GPU boundary

The workstation GPU can later accelerate a justified nonlinear challenger,
scenario generation and repeated scoring. It does not relax provenance,
baseline, nesting, determinism or replay requirements. CPU and GPU execution
must produce equivalently governed artifacts, with device/runtime versions and
determinism settings bound in the run manifest; a GPU-only gain must still beat
the transparent baselines on the same frozen origins.

## Replacement core/envelope sequence

1. Obtain governed CH and EEX PIT evidence with the required independent
   authority chain.
2. Run a frozen power/MDE and dependence study, then derive the resulting exact
   retrospective origin count, LT target horizons and regime minima. `24` is
   not accepted merely because it appeared in this first draft.
3. Choose four future windows that begin strictly after the trusted freeze;
   do not inspect their outcomes.
4. Freeze a receipt-free plan core containing the complete candidate grid,
   origin/target/inner-fold inventories, runtime contract and FMV economic
   assumptions; compute its final `plan_id` once.
5. Obtain independent read-only Security, IT/Operations and Quant/Data reviews
   of that immutable core and close every P0/P1.
6. Build a separate external admission envelope whose signed receipts point to
   the final core `plan_id` and exact bytes. Verify receipt bytes, issuers,
   authority separation, trusted time, CAS/HEAD, ACL/freeze and review links;
   never insert those receipts back into the core they attest.
7. Create the durable one-shot attempt seal before any future outcome fetch.
   Only then may a new verifier issue a non-production execution capability.
8. Execute the non-production rolling-origin campaign. A separate
   candidate, manifest chain and promotion review remain mandatory afterward.

This draft does not repair or reuse T057. T057 remains corrected historical
evidence with effective `n=1` and production `NO_GO`.
