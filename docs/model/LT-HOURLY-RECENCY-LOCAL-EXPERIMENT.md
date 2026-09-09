# D307 — hourly CH recency comparison, 8 September 2026

Protocol fixed before new results. CPU only on retained D300/D301/D304 inputs;
no Warehouse, GPU, AFRY, T057, CT or new data/model authority. D304/D305/D306
remain references. No new estimator family, assembly adapter or month patch.

The user prioritizes observable hourly CH day-ahead quality. The EPEX July2026
index specification identifies CH day-ahead as60min (p40). JAO's30June2026
announcement postpones Swiss-border15min MTU/block-bid implementation to2027;
it does not fix end2027 or the domestic energy-auction launch. Keep market
transition dates separate from the resolution of retained observations. No
hard-coded2027 switch; intraday15min trading already exists. Source review:
`docs/data/CH-DAY-AHEAD-RESOLUTION-20260908.md`.

## Frozen candidates and populations

Six original origins2021–2026 and identical saved solver/grid/quote surfaces.
2021–2022 diagnostic,2023–2026 exposed development assessment; no fresh holdout.
Candidates: cached D301 corrected MLP, D304 equal-history signed seasonal,
signed seasonal with exponentially decaying weights of half-life365.25days,
and730.5days. Only the mean within existing calendar/backoff cells changes.
Use all the same finite pre-origin complete-month training labels, including
negative prices; w=2^(-(last_training_timestamp-t)/half_life). No outcome-based
window selection, exclusions, regime switch, clipping, fitted maturity or hydro.
Reuse calendar_cell_reference with an explicit optional aligned positive weight.
Default unweighted behavior must reproduce D304. Same signed monthly centering
and PFCAssembler/final EEX projection; unchanged solver monthly authority.

Save raw predictions, training weights and effective sample support per calendar
cell/backoff; concentrated support is a diagnostic, not an adaptive fallback.
Pre-origin training95th percentiles of raw hourly price and absolute within-month
hourly ramps define point/ramp masks; no future-derived thresholds.

## Measurements and interpretation

Same complete truth-month populations as D304, before September1 2026 Swiss
time; one real hour is one observation, not four repeated CH quarters.
Reuse native score_curves on pre/final projection for direct D304 comparison.
Also save common hourly frames and diagnostics for:

- Shape error: (prediction - predicted month mean) - (truth - truth month mean).
- Full-price error: prediction - truth. Monthly level error: predicted mean -
  realized mean. Verify full error=shape error+level error and pooled
  MSE(full)=MSE(shape)+MSE(level) on whole months. Realized levels are evaluation
  labels only. Solver errors remain distinct from historical source/PIT defects.
- Within-month consecutive UTC-hour ramp error, excluding month boundaries
  and gaps. Quantify MAE/RMSE/bias/P95 with counts; do not claim PnL.
- Negative truth (<0), near-zero(abs<=5), positive(>5), high-price truth above
  pre-origin95th percentile; large absolute truth ramps above pre-origin95th
  percentile; four seasons, peak/offpeak, weekend/holiday, each delivery year,
  horizons1–6/7–12/13–24/25–36months, plus common first8-month view across the
  four assessment origins. Empty populations remain UNSUPPORTED.

Primary comparisons use equal-origin MAE/RMSE on2023–2026, plus pooled errors,
origin wins and full per-origin/segment outcomes. Already exposed and overlapping
years are dependent local development evidence, never independent promotion.
For favorable local screening, require >=2% improvement in both shape MAE and
RMSE versus D304, at least3/4 origin MAE wins, and no>5% MAE/RMSE regression
versus D304 on any supported declared segment (>=168hours across>=2origins),
including ramp error. Keep per-origin regressions visible even if aggregate
gates pass. MLP comparisons remain mandatory; no automatic adoption regardless
of screening. Monthly level MAE cannot be improved by hourly recentering.

## Exports, verification and closure

Build all three signed current variants on the retained valuation
2026-09-07T08Z/September4 quotes, not a September8 revaluation. Reuse existing
current hourly object, solver, full grid and EEX projection. Export native
hourly CSV plus15min repeated-hour transport for Oct2026–Dec2029 and full
Parquet through2032. Keep MLP/D304 controls and D305/D306 alternatives intact.
No claim of native15min truth, validated2030 physics or risk bands.

Require unchanged default D304 raw/assembled predictions, finite complete grids,
hourly repeated quarters, monthly level error<=1e-9, noCRITICAL EEX gate and
unchanged quote-conflict statuses. Independent saved-output review must verify
input/source/output hashes, independently calculate weighted cell predictions,
all errors/masks/summary gates, EEX repricing, exports and training cutoffs.
Tests cover weighting arithmetic, default parity, index/nonfinite/negative-weight
rejection, future-data rejection, DST/leap preservation and error decomposition.
Retain prior known Phase5 fixture failures; do not alter goldens. Update decision
log and handoff with exact commands, hashes, failures, results and limitations.
