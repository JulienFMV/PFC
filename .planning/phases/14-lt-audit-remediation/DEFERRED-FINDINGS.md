# Deferred LT Audit Findings

These findings are documented for follow-up because they are lower priority than
the P0/P1 remediation in this pass. They should not be treated as production
acceptance.

## Arbitrage Calibration

`pfc_shaping/calibration/arbitrage_free.py` still resolves inconsistent
overlapping products through least-squares smearing. A Cal-vs-Q/M conflict can
leave each quoted block repriced imperfectly even when the aggregate residual is
small. Follow-up: add an explicit policy for inconsistent overlapping products
and make convergence assert per-block residuals, not only aggregate residuals.

## Peak Mask Consistency

`pfc_shaping/calibration/cascading.py` fits peak spreads on an hour mask that can
omit holidays differently from the hour counts used by the calibrator. Follow-up:
share one peak/offpeak calendar mask between spread fitting and block repricing.

## Intraday Amplitude Target

`pfc_shaping/lt/model/intraday_amplitude.py` compares a peak-base target against
peak-offpeak spread, which can over-compress the enabled experimental layer.
Follow-up: either redefine the target as peak-offpeak or keep the layer gated off
until the metric and target are aligned.

## Robust Seasonal Ratios

`pfc_shaping/lt/model/robust_seasonal_ratios.py` uses a Kish effective sample
size that is scale-invariant, so down-weighted crisis years do not reduce the
shrinkage-to-prior weight as intended. The in-sample median anchor can also
self-anchor to a crisis-heavy window. Follow-up: wire an exogenous anchor and use
a crisis-sensitive effective sample size.

## Season Definitions

Season definitions differ across modules: the calendar path uses a wider winter
definition than the electrification scenario path. Follow-up: centralize season
definitions in one shared LT calendar helper and migrate all LT modules to it.

## Resolved In This Pass

The `ShapeHourly.apply()` clip-after-renormalization issue was promoted to P1 and
fixed in this remediation. Clipping now happens before the local-day mean
renormalization, with a regression test for `mean_h(f_H)=1` on clipped days.
