# D306: existing price-conditioned intraday composition

Protocol fixed before new results, 7 September 2026. Local CPU development only;
D304/D305 remain references. No adoption, tuning, Warehouse, GPU, AFRY or T057.
All authorities remain false. Existing revised extracts are not PIT vintages.

Reuse ShapeIntraday.apply, without changing its estimator or defaults. Convert
its factors to EUR/MWh residuals as parent_hour_price * (f_Q - 1), then subtract
the residual mean within each UTC parent hour. Require exactly four quarters,
finite aligned hourly prices, explicit origin and matching calendar. This is
price conditioning by scaling, not a learned nonlinear price-regime model.
Zero parent price implies zero residual: report this limitation explicitly.

Compare flat, native factors, existing sparse-regularized factors and D305's
unconditional additive reference on exactly the same timestamps. No hydro.
Reuse the eight saved pre-origin D305 factor fits (Jan-Aug 2026); evaluate every
origin on all remaining complete months through August (36 origin-month pairs).
The eight lead-zero populations must reproduce D305 to numerical tolerance.
For each fold compute both observed-parent conditional decomposition and an
effective forecast: hourly calendar-cell mean fitted strictly on pre-origin DE
hourly prices, with the existing calendar reference (Swiss calendar semantics
explicitly retained for this transparent proxy). No realized monthly level or
future price enters that forecast. This proxy is not an EEX-based DE PFC.
Both lanes share train, truth, masks and candidate populations. Keep D305
conditional results separate from effective price prediction.

Report MAE/RMSE, sample size and bias by origin/month, true parent price <0,
abs(parent)<=5 EUR/MWh (overlap intentional), parent>5, each observed season,
and leads 0, 1-2, 3-5, 6-7 months; unsupported autumn and longer forecast
horizons are explicit. Also report predicted-parent negative/near-zero regimes
for effective forecasts. Pooling repeated origin observations is not an
independent sample: retain equal-fold results and fold wins as well as pooled
errors. No confidence or significance claim from these exposed populations.

Gate interpretation frozen: no favorable overall qualification if either MAE
or RMSE worsens >5% versus flat in any supported truth-price, season or horizon
segment (>=96 quarters), or versus native for regularized factors. Aggregate
gain alone never qualifies a candidate. Report all per-fold regressions too;
these gates are local screening only, never promotion authority.

Current CH exports: retain D305 signed and MLP controls; generate signed+native
and signed+regularized intraday on the exact retained solver levels/full grid.
Fit only these two current DE components on prices strictly before the retained
valuation instant, without exogenous/hydro layers. Condition on the saved signed
PFC hourly price, call the existing assembler with signed_intraday_shape and
existing final EEX projection. No adapter or month patch. Save full Parquet and
Oct2026-Dec2029 CSVs plus annual/horizon/regime diagnostics through 2032.
Replay all six historical D304 signed CH curves with each origin's pre-origin
DE fit when supported; use flat explicit unsupported fallback if history is
empty. CH observations are repeated hourly: assert unchanged hourly scores,
never interpret quarter-hour deviations from them as transfer accuracy.

Success: saved predictions/model replay and independent error recalculation;
prior manifests unchanged; hourly mean drift and monthly solver errors <=1e-9;
no CRITICAL EEX gate; saved CSV/Parquet parity; negative/zero/positive, invalid
alignment, DST, leap and long-horizon conservation tests. Retain existing
Phase5 known fixture failures without changing goldens. Save source/input pins,
test logs, comparative report, exports and handoff/decision before closure.
