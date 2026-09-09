# Proposed next step after D309 — causal origin/delivery training

Recommendation only in response to the user's next-step question. No adopted
D310 decision, frozen benchmark, model execution or authority change. D309
remains complete; its closure-bound files are unchanged. Only this document
is added, using repository findings already read in this conversation.

## Recommended next bounded experiment

Prioritize a correctly constructed origin/delivery training experiment over
another recency or smoothing grid. The earlier D301 review established that
years_ahead was constant at zero in training. This is a concrete formulation
limit, not evidence that a learned maturity effect will improve forecasting.

Construct historical pairs (information origin, delivery hour) at multiple
lead times. For an outer evaluation origin O, every training delivery label
and its complete-month signed target must be observed before O. Features for
each training pair must use only information at its own earlier origin o.
Future deterministic calendar information is allowed; realized future hydro,
load, generation and later EEX quotes cannot become pair-origin predictors.
Retained revised histories remain retrospective/non-PIT evidence even when
delivery cutoffs are respected. Do not manufacture historical publication dates.

Reuse the D304 signed reference at each historical information origin. Propose
one small regularized residual learner, with globally fixed calendar basis and
explicit maturity interactions, to predict corrections to the signed D304
shape. Freeze the precise feature inventory, regularization, origin cadence,
minimum history, support, CPU cap and sample weighting before new results.
Duplicate delivery labels across information origins must not silently gain
weight; report distinct hours and origin/hour pairs separately.

Three planned comparisons isolate the formulation:

1. D304 unchanged.
2. Same residual learner and training pairs with maturity inputs disabled.
3. Same residual learner and pairs with maturity inputs enabled.

No model/month or model/horizon switching. A single global recipe applies at
all deliveries. Monthly-neutral correction, existing assembler and EEX
projection, solver monthly levels preserved. This requires its own explicit
experiment contract; do not silently alter earlier frozen feature inventories.

Retain D308 regime checks and D309 seam/revision checks, pre-origin scales,
negative/near-zero/tail diagnostics and support flags. Report paired shape
accuracy and revisions before/after projection, with level errors separate.
Unsupported lead-time training support cannot be represented as learned
maturity. No automatic extrapolation or qualification of three-year accuracy.

Deliverable if accepted: bounded CPU benchmark, candidate curves, independent
verification, report and an explicit gain/failure decision. No promised gain.
D304 remains reference absent qualifying evidence. Past assessment origins
remain exposed development data, not a new independent test.

## Parallel qualification and later roadmap

Prepare successive D304 forecast snapshots at genuinely new governed local
valuations when available. Same-valuation recomputation is not a new vintage.
Complete independent custody and truth-finalization requirements for the
existing prospective holdout draft; no registration or countable-origin claim
without those requirements. No T057 reuse.

The structural 2027–2030 roadmap still matters for causal future evolution;
this bounded maturity experiment does not replace it or invent its missing
drivers. Scenario implementation requires the mandatory source/restricted-data
contracts before any data access. No AFRY, Warehouse or GPU authorization is
introduced. All six authority fields stay false.

## Evidence and scope

Basis: D301/D303 reviewed constant training maturity; D304 signed-reference
advantage; D307/D308 recency failures; D309 negative smoothing trigger and
revision attribution. This sequencing is an engineering proposal, not a
literature claim or a result of a new experiment. No web research, shell action,
data read, fit or tests performed in this recommendation-only turn. No code,
existing handoff pointer, decision log or frozen artifact changed.
