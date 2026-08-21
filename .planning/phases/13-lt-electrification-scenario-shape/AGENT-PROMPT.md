# Agent Prompt: Phase 13 LT Electrification Scenario Shape

## 0. Role and Standard

You are a senior power-market quant engineer operating at Axpo desk pragmatism
and ETH/EPFL modelling rigor. Your task is to add the next LT HPFC frontier:
scenario-conditioned structural shape evolution for 2027-2030+, driven by
electrification fundamentals.

Do not open a PR. Do not touch any other worktree. Work only in the current PFC
worktree.

## 1. Mandatory Reading

Read these first:

```text
.planning/phases/11-ct-chronos2-future-covariates/AGENT-ONBOARDING.md
.planning/phases/12-lt-next-sota/AGENT-PROMPT.md
.planning/phases/12-lt-intraday-amplitude-shrinkage/RESEARCH.md
.planning/phases/12-lt-intraday-amplitude-shrinkage/VALIDATION.md
.planning/phases/13-lt-electrification-scenario-shape/SPEC.md
.planning/phases/13-lt-electrification-scenario-shape/LITERATURE.md
pfc_shaping/lt/model/solar_modulation.py
pfc_shaping/lt/model/assembler.py
pfc_shaping/validation/perfect_foresight.py
scripts/run_perfect_foresight.py
```

Understand the already shipped stack:

* regime-aware seasonal ratios;
* hydro-aware peak spreads;
* SOTA `ShapeHourly` half-life 90d;
* `sota_solar`;
* `sota_amp`;
* perfect-foresight A/B reporting.

Do not redo those.

## 2. Mission

Deliver a flag-gated module that makes 2030 structurally different from 2027
when scenario fundamentals justify it.

Core concept:

```text
f_H_future = f_H_SOTA
           + constrained block response to PV, batteries, EVs, heat pumps,
             demand, nuclear, and interconnection scenario drivers
```

The market still owns the level and all quoted delivery-product means.
Electrification only redistributes shape within allowed degrees of freedom.

## 3. Non-Negotiable Contracts

Reproducibility:

* new behavior behind a flag defaulting OFF;
* OFF path byte-identical, prove `atol=0`;
* no imports of the new module when the flag is disabled if practical.

No leakage:

* scenario rows require `publication_date <= vintage`;
* realized installation rows require measurement timestamp `< vintage`;
* future official assumptions are allowed only if published before vintage;
* no realized future price, capacity, generation, or target-year KPI can enter
  training or features.

Additive architecture:

* prefer a new module `pfc_shaping/lt/model/electrification_shape.py`;
* add one assembler flag, do not change baseline semantics;
* add one estimator variant, e.g. `sota_electrification`;
* preserve local-day mean `f_H = 1`;
* preserve period means after arbitrage-free calibration.

Anti-overfit:

* do not tune to Cal-2025 alone;
* use low-dimensional block basis and sign-constrained/ridge coefficients;
* validate with faux-future splits.

## 4. Required Plan-Gate Before Code

Before implementation, write:

```text
.planning/phases/13-lt-electrification-scenario-shape/RESEARCH.md
```

It must contain:

1. measured residuals from the current `--ab` output;
2. why this lever is not a retune of Phase 12 amplitude or Phase 10 solar;
3. data sources and exact vintage/as-of semantics;
4. mathematical model and sign constraints;
5. integration point and OFF-path reproducibility proof plan;
6. validation plan with ship thresholds;
7. leakage analysis and rejected alternatives.

Then implement only the scoped plan.

## 5. Implementation Blueprint

Create:

```text
pfc_shaping/lt/model/electrification_shape.py
tests/test_electrification_shape.py
```

Suggested API:

```python
class ElectrificationScenarioStore:
    def __init__(self, path): ...
    def asof(self, vintage): ...
    def get(self, *, country, scenario, delivery_period): ...

class StructuralDriverProjector:
    def transform(self, scenario_row, calendar_df): ...

class ElectrificationFHCorrection:
    def fit(self, spot_history, driver_history, *, vintage): ...
    def apply(self, f_H, calendar_df, *, vintage, scenario, scenario_store): ...

def electrification_modulate(
    f_H,
    calendar_df,
    shape_hourly,
    *,
    vintage,
    scenario,
    scenario_store,
): ...
```

Assembler flags:

```python
enable_electrification_shape: bool = False
electrification_scenario: str | None = None
electrification_scenario_path: str | Path | None = None
```

Perfect-foresight estimator:

```text
sota_electrification
```

The estimator composes the SOTA stack with the electrification flag, mirroring
`_sota_solar_estimator()` and `_sota_amp_estimator()`.

## 6. Model Constraints

Use block-level constrained coefficients. Minimum block set:

```text
NIGHT
MORNING_RAMP
MIDDAY_BOWL
AFTERNOON_SHOULDER
EVENING_RAMP
```

Required signs:

```text
PV:          MIDDAY_BOWL <= 0
battery:    MIDDAY_BOWL >= 0 and EVENING_RAMP <= 0
managed EV: MIDDAY_BOWL >= 0 or NIGHT >= 0
unmanaged EV: EVENING_RAMP >= 0
heat pumps: WINTER MORNING_RAMP >= 0 and WINTER EVENING_RAMP >= 0
```

Normalize after applying coefficients:

```text
mean(f_H_adj | local day) = 1
```

Never let scenario modulation change the market-implied Cal/Q/M average.

## 7. Validation Commands

At minimum run:

```powershell
$env:PYTHONPATH='.'; pytest tests/test_electrification_shape.py -q
$env:PYTHONPATH='.'; pytest tests/test_phase10_reproducibility.py -q
$env:PYTHONPATH='.'; pytest tests/test_perfect_foresight.py::test_build_curve_rejects_unknown_estimator_string -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/run_perfect_foresight.py --ab
```

If runtime is high, run a one-vintage diagnostic first, then the full `--ab`.

## 8. Expected Evidence

Report:

* OFF identity max absolute delta;
* scenario rows used after `asof(vintage)`;
* coefficient table with signs;
* 2027 vs 2030 block deltas under slow/central/fast;
* perfect-foresight A/B:
  - baseline;
  - SOTA;
  - SOTA+solar;
  - SOTA+amp;
  - SOTA+electrification;
* faux-future validation metrics;
* structural fan chart width by delivery year.

Ship only if:

```text
OFF identity exact
no leakage tests pass
monthly-shape gate remains >= 0.85
faux-future de-leveled metrics non-worse
scenario monotonicity tests pass
```

If the empirical effect is weak, ship a null-result report and keep the flag OFF.

## 9. Final Response Template

End with:

```text
Branch/HEAD:
Files changed:
Flag default:
Leakage proof:
Validation:
Decision: shipped / null-result / blocked
```

