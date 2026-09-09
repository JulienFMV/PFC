# Signed composition: local experiment D305

Objective: qualify the D304 signed seasonal hourly candidate with hydro and
intrahour components, preserving the existing CH monthly solver and final
BASE/PEAK projection. No assembly adapter or default-model replacement.

Protocol fixed before results on 7 September 2026:

1. Replay all six D304 origins (2021–2026), including four development
   assessment origins (2023–2026). Compare cached hourly MLP and signed seasonal
   forecasts with/without the existing additive WaterValueCorrection. Fit hydro
   sensitivity on each origin's complete pre-origin CH months and pre-origin
   reservoir history. Recompute the existing causal five-year hydro anomaly
   within each origin; exclude unsupported neutral sentinels (not observations).
   Use the latest supported anomaly at most14 days old, including ISO week53
   fallback to the preceding supported week. Forecast using the existing civil-week
   linear return-to-zero recipe, over the original delivery window. No future
   observed reservoir values. Save forecasts, fits, errors and product gates.
2. On native DE quarter-hour history, compare flat, existing ShapeIntraday,
   its existing sparse-support regularization, and an additive seasonal
   reference. Eight monthly origins January–August2026, expanding history
   strictly before each origin; no exogenous realized future features. The
   target is price minus its parent-hour mean. For factor models the *observed*
   parent-hour price is supplied: this is conditional disaggregation accuracy,
   explicitly not a price forecast benchmark. Report monthly and negative-hour
   errors, equal-month MAE/RMSE and sample counts. No tuning.
3. The additive reference averages signed EUR/MWh residuals by season, day
   type, hour and quarter, falling back to season/hour/quarter, hour/quarter,
   then quarter. Recenter every predicted parent hour. No ratio, price floor,
   invented far-horizon damping or fitted maturity effect.
4. Generate current local PFC alternatives on the exact D300 monthly levels
   and retained forecast contexts. Include the D300 control, signed hourly,
   signed plus hydro, and signed plus hydro plus additive DE intraday transfer.
   Freeze all alternatives; no outcome-chosen model switch or operational
   promotion. CH truth is hourly repeated four times, so CH quarter-hour
   predictive improvement cannot be asserted from these inputs.

Composition contract: explicit hourly-neutral signed quarter-hour residuals
may be added to the signed hourly lane; non-neutral legacy f_Q is rejected.
Existing water-value delta is computed against solver BASE, remains monthly
neutral and is applied once before the common final projection. Both optional
inputs require exact finite aligned data; no ignored hydro forecast or model.
Hourly gains, intrahour conservation and market constraints are separate checks.

Success: exact D300/D304 control reproduction; finite complete candidate grids;
monthly level error <=1e-9 EUR/MWh; no new CRITICAL market gate; preserved
intrahour residuals through hourly product projection; deterministic arithmetic
recomputed from saved outputs; positive/negative/DST/leap/rejection tests pass.
Existing Phase5 fixture failures must remain documented, not waived or rewritten.

All data are retained, revised PRD extracts, not certified historical vintages.
These previously exposed years are local development evidence. All production,
promotion, scientific, trading, origin-registration and countable-origin
authorities remain false. No Warehouse, GPU, AFRY, T057 or new dependencies.
Probabilistic uncertainty and future structural trajectories remain separate
unqualified work; no validated risk bands are attached to these point curves.
