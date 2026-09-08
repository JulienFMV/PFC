# Expert proposal after D308 — knowledge cutoff 31 August 2026

User asked for the next expert PFC recommendation. This is a proposal only,
not a frozen new benchmark, an adopted D309 decision or authorization change.
D308 remains complete and its closure-bound files are untouched. No new fit,
scoring, source extraction, data download, Warehouse/GPU/AFRY/T057 or agents.

## Recommendation

Keep D304 primary local signed reference and D305–D308 comparisons. Stop
expanding the recency grid. Prioritize an audit of revision stability for the
same delivery hours and month/season boundary behavior, then a distinct causal
shape hypothesis if supported. Preserve monthly solver levels and assembler.

1. Audit curve revisions for common delivery hours across available local
   vintages. Separate solver monthly-level changes, raw shape changes and the
   final EEX projection effect (including new PEAK information). A market-driven
   update is not automatically instability. Paired counterfactual replays with
   fixed levels/quote surfaces may isolate shaping effects, but must be labelled
   counterfactual, never fabricated historical point-in-time vintages. Actual
   historical availability requires archived source evidence. Missing local
   snapshots remain missing, with no Warehouse substitution or invented inputs.
2. Audit month/season seams explicitly. D308 ramp scoring deliberately excludes
   Swiss month boundaries and UTC gaps. Preserve that metric and add a separate
   decomposition of boundary price changes into monthly-level and shape terms;
   do not smooth raw prices across solver months. Verify leap/DST populations,
   delivery years, horizon and negative/near-zero/tail controls. Freeze revision
   budgets/thresholds from pre-origin information before any candidate results.
3. If the audit establishes a recurring calendar discontinuity or systematic
   profile bias, test one transparent signed calendar challenger: a penalized
   smooth annual/hourly profile with Swiss day types, using globally fixed
   structure/regularization and monthly-neutral signed EUR/MWh output. This is
   a proposed hypothesis, not a claim of improvement. Retain D304 and the same
   assembler/EEX projection, no model/month selection. Do not conflate the
   earlier ratio-space GAM with a signed, explicitly specified new experiment.
   Construct causal origin/horizon training examples before claiming learned
   maturity; a constant years_ahead feature cannot establish horizon learning.
4. Follow with the separately governed structural-shape milestone for 2027–2030:
   known-at-origin physical drivers and coherent chronology for renewables,
   demand, hydro, storage/flexibility and exchanges. Public assumptions must be
   dated and distinguished from governed data. This turn does not open scenario
   data or execute that work; read the mandatory restricted-data context and
   source/semantic contracts before any later scenario implementation. AFRY
   remains excluded. Any central-PFC correction has zero monthly mean; a
   fundamental level outlook belongs to a separate product. Avoid an isolated
   battery kernel with no price-formation or PFC validation.
5. Qualify uncertainty as a separate conditional-shape product: joint hourly
   trajectories at fixed solver levels, with dependence, negative-hour episodes,
   ramps and coverage scored. Do not force the central curve to reproduce spot
   tails. Project/recenter trajectories through existing constraints and score
   their distributions after projection. Marginal quantile curves are not
   themselves coherent scenarios. Any unconditional level uncertainty needs
   its own product/contract and must not overwrite the solver.

A prospective holdout should be frozen before future labels are observed;
start/cutoff/population and truth finalization require their own governed
contract. The six already exposed origins cannot be made independent by
renaming folds. This qualification can be prepared while local development
continues; T057 stays sealed, all six authorities false.

## State of the art: verified dated primary sources

- FETS v2,17 July2026:
  https://arxiv.org/abs/2604.22328v2
  Chronos-2/TiRex-2 lead aggregate results on54 energy datasets. This motivates
  bounded candidate qualification, not a claim of three-year CH PFC superiority.
- Cross-border EPF v1,17 August2026:
  https://arxiv.org/abs/2608.17091v1
  N-HiTS/NBEATSx competitive under limited data; Transformers require adaptation.
  Target experiment is day-ahead DE-LU2024, not long-term CH forward shaping.
- REMIND–PyPSA-Eur v1,5 October2025 (journal reference2026):
  https://arxiv.org/abs/2510.04388v1
  Couples transition pathways with hourly operation and flexibility; illustrates
  the importance of coherent physical drivers. Germany-focused2045 scenarios
  do not validate a Swiss2030 model or prescribe the local implementation.

All three versioned abstract pages reopened successfully this turn. Broad search
also returned September2026 work; it is excluded from the cutoff-based proposal.
An older mid-/long-term probabilistic paper publisher page returned403 and is
not needed for the final recommendation. No exhaustive-literature-review claim.
The proposed sequencing is engineering judgment from repository evidence and
these sources, not a result directly demonstrated by any cited paper.

NBEATSx is a possible later bounded CPU challenger once a causal LT formulation
is fixed; Chronos-2/TiRex-2 require compatible horizon formulation and pretraining
contamination checks. No automatic forecast-length extrapolation or model API
send. The recommendation prioritizes measurable product defects before these
more complex comparisons.

## Actions and exact file scope

Read the existing post-D301 SOTA proposal and D303 priority-review handoffs;
used the D308 evidence already in session. Web read-only source verification.
Only this new handoff file is added. No modification to .planning/HANDOFF.md,
DECISION-LOG.md or D308 code/docs/reports/closure; their frozen hashes persist.
No model tests rerun for a recommendation-only document. Check new document
whitespace and git diff --check from the canonical cwd/Git root before closing.
All shell actions verify C:\Users\jbattaglia\PFC_LT as cwd and Git top-level.
