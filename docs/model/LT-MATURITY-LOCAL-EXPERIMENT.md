# D310 origin/delivery residual experiment

Frozen recipe before new results. Retrospective revised local CH hourly data,
not point-in-time evidence. D304 remains reference; all six authorities false.
User authorizes GPU when useful; this small dense ridge solve uses four CPU
threads. No Warehouse, AFRY, T057, CT, production or protected desk-data work.

Three curves: unchanged D304, calendar residual ridge, identical ridge with
maturity interactions. Quarterly inner origins are twelve hours before Swiss
quarter start. At least twelve complete historical months are required. D304
at each pair origin uses only months closed before that origin. Labels are
complete months closed before the outer origin, at leads 1 through 36 months.
Repeated delivery labels receive inverse pair-count weights so each distinct
hour contributes total weight one. No candidate selection by month or horizon.

Fixed basis: sine/cosine harmonics 1,2,3 of Swiss hour, interacted separately
with each of the four existing FMV seasons and five existing day types: 54
columns. Maturity adds the same columns times min(lead_months,36)/36: 108
columns. Columns are centered within each complete Swiss delivery month.
No future prices or physical realized predictors; historical realized monthly
means are labels only. Ridge minimizes weighted mean squared residual plus
0.1 times squared coefficient norm, all coefficients penalized, no intercept,
no data-dependent feature scaling, hyperparameter search or early stopping.
At most fourteen fits and fourteen assemblies, seven retained outer origins.

The capped formula applies globally; months beyond 36 or beyond the outer
training lead maximum are explicitly unsupported for learned maturity. This
does not qualify extrapolation. Training lead counts and distinct hours are
reported. Earlier tuning/assessment labels retain their roles but all are
exposed development evidence, not a new independent holdout.

Reuse assembler, EEX projection and solver levels exactly. Compare pre/final
projection shape, full price and monthly level separately. D308 thresholds and
screen remain frozen: >=2% shape MAE and RMSE gain, >=3/4 assessment origin
wins, no >5% supported regime/origin shape or ramp regression, supported tails.
Report all years, FMV seasons, horizons, negative, near-zero, ramps and tails.
D309 pre-origin seam scales retained; report seam and revision decomposition.
Revision comparisons are common-delivery retrospective recomputations, not
archived daily vintage evidence. Seam veto: >5% pooled assessment month-boundary
shape or full ramp MAE/RMSE regression, support >=8 events over >=3 origins.
Maturity claim additionally requires improvement in both global shape errors
versus the calendar-only residual; no adoption even if exploratory screen passes.
Current exports retain the September 7 valuation and solver; no new vintage.

Independent checks reconstruct weighted sufficient statistics using saved pair
labels, independent calendar reference replay and eigen-decomposition solve;
recompute diagnostics, monthly constraints, EEX product gates and seam/revision
identities. Tests cover origin cutoffs, varying maturity, duplicate weights,
neutrality, DST and fixed far-horizon cap. Existing source/artifact hashes stay
bound and are verified before/after execution.
