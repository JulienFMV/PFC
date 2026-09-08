# Benchmark next-step proposal — 7 September 2026

## Scope and status

The user first asked what to do next to introduce better models and benchmark
them, then explicitly accepted proceeding ("on fais la suite") and asked
whether to continue here, start a new session or compact. Recommendation:
`/compact` in the same conversation, then resume this implementation directly.
The local CPU development benchmark is authorized; do not ask again for that
permission. No protocol revision has been implemented or real benchmark run.
D300's local curves, fitted models and source evidence are unchanged. This
planning/authorization update performed no fit, scoring, Databricks call, GPU,
promotion or external message.

## Repository findings

Read AGENTS.md, current root/D300 handoffs, the LT rolling-origin protocol,
candidate execution/challenger/reference/target/full-price assembly modules and
the FMV quality charter. Four challengers already exist in synthetic-only code:
true 180-day weighted MLP, Ridge, additive spline Ridge GAM and CPU LightGBM
4.6.0. The seasonal reference is outside candidate selection. Real execution
and seasonal-reference scoring remain unfinished. Do not route real arrays
through SyntheticTrainingSet or silently change its provenance.

D300's HydroAlignedShapeHourlyMLP differs from the frozen v6 incumbent. A local
comparison must explicitly version this baseline/feature correction and retain
the old incumbent as a separate diagnostic. No silent frozen-hash replacement.

## Proposed next implementation outcome

Build a bounded local development benchmark reusing the existing assembler and
candidate mathematics, starting with CH hourly shape. Freeze sources, temporal
splits, features, budgets, metrics and current corrected-model identity before
new results. Implement the actual model execution and reference scoring; do not
add another assembly adapter. Refit every learned component/climatology on the
training side of each split. Never reuse D300's all-history fitted weights to
claim held-out historical performance.

Report current corrected MLP and the four challengers against a transparent
seasonal reference on common rows and solver/product constraints. Measure
monthly-neutralized MAE/RMSE, bias, tails, regime/horizon coverage and compute
time. Preserve the charter's goals (2% MAE/RMSE gain, 70% eligible folds, regime
guardrails) as policy targets, not evidence already achieved or statistical
significance. No winner is predetermined.

The current source snapshot is latest-observed history, not historically
available vintages. A chronological replay can guide local development, but
must disclose revision/availability limitations and cannot be called a causal
or independently validated historical PFC backtest. Restrict metrics/horizons
to actual support. Independent prospective registration, admitted source timing
and holdout still govern scientific confirmation; T057 remains sealed.

Assess intraday regularization separately after isolating hourly effects. The
native DE quarter-hour history is under one year and cannot substantiate broad
multi-year CH quarter-hour performance. Negative-price capability requires a
separate target/shape-architecture experiment: swapping the hourly regressor
behind the same positive-factor clipping does not itself solve that limitation.

Consumer/model team owns benchmark code, leakage checks, common inputs,
calibration, metrics and failure reporting. Data engineering owns producer gaps,
vintage/publication meaning and usable outage sources. Product owner owns
business loss priorities and subsequent independent acceptance decisions.

## Verification and references

Read-only shell actions checked canonical cwd and Git root. No code/runtime
change or new test was needed for this planning update. Only this note and the root
handoff pointer changed. The previous session inventory remains an immutable
snapshot of D300; its documentation hashes predate this proposal.

Primary methodological references consulted: scikit-learn TimeSeriesSplit
documentation (chronological separation) and Lago et al., Applied Energy 2021,
doi:10.1016/j.apenergy.2021.116983 (electricity forecast comparison practices).
The latter concerns day-ahead forecasting and is not evidence that any model
will improve FMV's long-term shape. All scientific/production authorities remain
false. Authorization covers implementing and running the explicitly labeled
local development comparison; it does not establish signed scientific source
admission, external origin registration, independent holdout or production
promotion. Keep the frozen scientific v6 protocol intact while making the
corrected local benchmark identity explicit.

## First actions after compact

Read AGENTS.md, root HANDOFF.md, this note, and the D300 source-integration
handoff. Inspect the existing candidate mathematics and execution/reference
interfaces before choosing the smallest implementation. Define chronological
development splits, supported horizons and common feature/target handling;
separate the corrected local MLP identity from the frozen incumbent. Implement
and test real local CPU candidate execution and seasonal reference scoring,
then run a bounded retrospective development comparison on retained source
bytes. Refit all learned components within each split and report unsupported
coverage honestly. Reuse the existing full-price assembler; do not add another
assembly adapter or disguise real inputs as synthetic fixtures.
