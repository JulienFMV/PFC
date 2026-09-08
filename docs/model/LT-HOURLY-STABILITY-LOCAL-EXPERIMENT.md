# D308 — fixed CH hourly stability benchmark, 8 September 2026

Frozen before new candidate results. CPU local only; all six authorities false.
D304 equal-history signed shape remains primary reference. D305 hydro and
D306 intraday remain preserved comparisons, not components activated here;
D307 one/two-year recency and MLP are paired hourly controls.

## Challenge and bounded design

D307 aggregate gains did not establish stability. Selecting its best half-life
or selecting a weight per delivery month would exploit already exposed outcomes.
Use both existing half-lives (365.25 and 730.5 days), each at global recent
weights 0.25 and 0.50. Four candidates only; no weight optimization, model fit,
new feature, hydro, intraday residual, clipping or horizon/season/month switch.
This is a conservative shrinkage experiment, not a claim that blending must
improve performance. Retain the same 2% gain requirement despite smaller weights.

Reuse hash-verified D307 raw predictions and training targets. Blend raw signed
EUR/MWh shapes as (1-alpha)*D304 + alpha*recent, then center per full Swiss
delivery month and call the existing assembler and EEX projection. Never blend
monthly levels or correct a month after the solver. Four new assemblies for
each of six historical origins plus current = 28; 28 cached control curves;
zero statistical estimator fits. Four CPU threads, no remote or GPU execution.
No scored rerun or extra weight after results. Failed unscored attempts retained.

## Pre-origin controls and success criteria

Same frozen D307 origins/populations: 2021–2022 diagnostic only, 2023–2026
already-exposed development assessment. No independent significance or holdout.
Freeze all seven sets of numerical thresholds in plan.json before assembly or
scoring. Use only retained complete training months ending before each origin.
Retain raw-price p95 and absolute within-month contiguous UTC ramp p95.
Add signed monthly training-residual p05/p95 and absolute residual p95. Assess
lower/upper/absolute shape-tail events against these thresholds; future realized
monthly means define evaluation labels only and never enter predictions.
Do not lower an unsupported threshold after observing results.

Save before/after-projection shape, full-price, monthly-level and ramp errors,
MAE/RMSE/bias/absolute-error p95 and counts. Retain D307 masks: negative truth,
abs(price)<=5, positive>5, high raw price, large ramps, Swiss seasons,
delivery years (distinct from origin), horizons 1–6/7–12/13–24/25–36 months,
first eight months, peak/offpeak and weekend/holiday. Add three shape-tail masks.
UTC gaps and Swiss month boundaries excluded from ramp scoring; DST preserved.

Equal-origin assessment summary and hourly pooled summary both reported.
Favorable local screen requires >=2% shape MAE and RMSE improvement over D304,
>=3/4 origin MAE wins, and no >5% shape/ramp MAE or RMSE regression on a declared
segment with >=168 hours across >=2 origins. Add explicit per-origin ALL
shape/ramp veto at >5% with >=168 hours; do not hide an adverse year in averages.
Each of the three shape-tail masks must have >=168 hours across >=2 origins
for both shape and ramp diagnostics; otherwise stability screen is incomplete.
HIGH_PRICE unsupported stays explicit; it never becomes passed spike evidence.
Every result retained; no automatic adoption even if the local screen passes.

Verify complete paired populations; full_error=shape_error+level_error;
MSE(full)=MSE(shape)+MSE(level) per complete month; identical monthly levels
across candidates and solver residual <=1e-9. No CRITICAL EEX gate; unchanged
quote conflicts. Monthly forward-versus-spot error is distinct from constraints.

## Verification, exports and limitations

Independent saved-output verifier reconstructs component calendar predictions
with a separate NumPy/dictionary calculation, blending/centering, pre-origin
thresholds, masks, populations, every metric, EEX gates and CSV prices/timestamps.
Hash-check inputs, source snapshots, outputs and prior D301 inventory. Test
alignment, convex weights, finite signed values, causality, tails, DST, gaps,
level decomposition and screening failures. Run the required LT test matrix;
retain known Phase5 fixture failures without editing fixtures or waivers.

Current CSVs: hourly Oct2026–Dec2029 and repeated-hour 15min transport; full
Parquet through2032. Retain valuation 2026-09-07T08Z / September4 quotes, not a
fresh September8 mark. D304 and all comparisons unchanged; all artifacts kept.
No Warehouse, GPU, AFRY, T057, CT/protected-data mutation or external publication.
No native CH15min accuracy, future structural2030 accuracy, PnL or calibrated
risk-band claims. Revised histories and overlapping exposed years remain local
development evidence. Record decisions, exact evidence and handoff at closure.
