# D309 — fixed revision and calendar-seam audit

8 September 2026. Local CPU audit and next-hypothesis recommendation only.
D304 primary, D305–D308 evidence preserved, all six authorities false.
No product-code change, model selection, new model fit, solver correction,
Warehouse, GPU, AFRY, T057, protected-data access or external publication.

## Scope and challenge before results

The admitted D300–D308 lineage contains one current valuation (7 September2026,
4September quotes) and six reconstructed annual origins. D301 explicitly uses
LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT. Same-valuation model variants are not
successive vintages. A top-level local directory inventory found no additional
D304 release series in this admitted lineage. This does not prove that no other
archive exists elsewhere. Report real daily D304 vintage stability UNSUPPORTED.

Audit all eight D308 hourly controls/candidates on their original full grids.
Six adjacent origin pairs are fixed, including2026->current. Empty intersections
remain UNSUPPORTED; never extrapolate a saved historical curve into overlap.
For every nonempty pair, reapply the older D304 training targets to the newer
full delivery grid, with newer solver levels/quoted keys/reference date through
the existing assembler and EEX projection. Expected four new counterfactual
assemblies, maximum six. No curve is labelled a historical production vintage.

For common delivery hours, decompose final revision exactly into level,
pre-projection shape and projection revisions. Signed-reference counterfactual
additionally separates updated-history contribution at fixed newer market/grid
from the remaining market/grid contribution. Show their projection interaction;
do not claim causal attribution to solar, hydro or economic fundamentals.

## Thresholds fixed from pre-origin history

All thresholds for all seven origins frozen in plan.json before seam/revision
results. Use the same closed-month target frames as D308. Save native absolute
midnight-ramp p95, month-boundary-ramp p95, and signed-target boundary-ramp p95,
with counts. Midnight excludes UTC gaps; no interpolation across absent hours.
At least12 training boundaries required for a boundary-scale flag, otherwise
UNSUPPORTED. These are diagnostic scales, not production/trading limits.

Revision scale: compare equal-history D304 with the same reference excluding
the latest12 closed training months, on a deterministic next12-month calendar.
Use p95 absolute centered-shape difference. Require at least24 closed months and
at least12 retained older months. Both histories end before the origin. This
is a pre-origin pseudo-update scale, not observed vintage volatility. Freeze
it and its support; flag input-history revisions against the earlier origin's
scale. Never judge raw market-level changes against a shape-update budget.
Budget preparation: at most14 calendar calculations, plus at most6 replay
calculations; zero statistical estimator fits. Four CPU threads.

## Seams and comparison

Keep D308 ramp statistics intact. New event tables retain consecutive UTC-hour
transitions at Swiss midnight, month and season boundaries, including future
unscored events. Add within-month midnight controls. Levels come from saved
solver maps. FMV seasons are the existing calendar contract: winter Nov–Mar,
spring Apr–May, summer Jun–Sep, autumn Oct; boundaries Apr/Jun/Oct/Nov. Do not
substitute meteorological quarters. A pre-freeze test initially assumed those
quarters and was corrected against the existing source, without code change.
Decompose price step into solver-level, pre-projection-shape and
final-projection steps. For signed curves, further expose raw calendar step,
monthly-centering step and any assembly remainder. No cross-month smoothing.

Score only events whose two hours belong to eligible complete truth months in
the frozen D308 diagnostics. Report full ramp error, shape ramp error and level
error step; verify full=shape+level, without claiming an MSE orthogonality identity
on boundary subsets. Months' realized means are evaluation labels only.
Groups: all month boundaries, seasonal boundaries, other month boundaries and
within-month midnights. Break out delivery year, Swiss season, horizons1–6/
7–12/13–24/25–36/37+, negative/near-zero/positive truth and shape tails using
the unchanged D308 pre-origin thresholds. Empty error populations UNSUPPORTED.

Recommendation trigger (not adoption): a single smooth-calendar challenger is
worth specifying if D304's assessment seasonal boundaries have >=8 events,
other month boundaries >=8, each across >=3 origins, and seasonal MAE exceeds
other-month MAE by >10% for BOTH full and shape ramp error. Additionally the
raw-calendar absolute contribution must exceed half of the sum of absolute
raw/centering/projection contributions at seasonal boundaries. This identifies
a hypothesis, not proof of avoidable error: hour/day-type changes also contribute.
If unmet, do not force a new model from this audit. Keep all adverse results.

## Verification and deliverables

Separate saved-output verifier checks hashes, exact timestamp populations,
independent NumPy differences/decompositions/thresholds, old-history raw replay,
solver levels<=1e-9 and independent product repricing. All existing D308 curves
remain byte-identical. Counterfactual exports carry explicit non-vintage status;
the current replay includes hourly and repeated-QH CSVs, no adopted candidate.
Unit tests cover gaps/DST/leap, month-vs-season labels, revision decomposition,
origin cutoff, missing populations and recommendation support/vetoes. Run LT
required tests and relevant signed assembly diagnostics, retaining known failures.

Prepare a non-registered prospective holdout draft binding the existing D304
current forecast for October2026–September2027. Independent custodian and truth
finalization/availability evidence remain required, not fabricated. This is
future near-horizon preparation, not three-year qualification or T057 reopening.
Write French report, all CSV/Parquet evidence, decision and handoff with hashes,
exact commands, limitations and failures. No result-driven model or month patch.
