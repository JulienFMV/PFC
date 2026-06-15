# Phase 13 RESEARCH: Electrification-Scenario Shape

## 1. Diagnosed Residual

Current branch: `feat/lt-next-sota`.

Latest diagnostic command:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/run_perfect_foresight.py --ab
```

Local data caveat: the run used `data/epex_hourly.parquet` bootstrapped from
tracked EPEX 15-minute data and a locally reconstructed
`data/forwards_history_phase10.parquet` with
`forwards_source=fallback_diagnostic`. The numbers are valid for residual
triage and implementation diagnostics, not a final market-data gate.

Cal-2025 result:

| metric | baseline | SOTA | SOTA+solar | SOTA+amp |
|---|---:|---:|---:|---:|
| median `pf_cal_corr` | 0.7447 | 0.9182 | 0.9182 | 0.9182 |
| min variant `pf_cal_corr` | n/a | n/a | 0.8608 | 0.8606 |
| median delta vs SOTA | n/a | n/a | +0.0000 | +0.0000 |
| improved vintages vs baseline | n/a | 12/12 | 12/12 | 12/12 |
| SOTA Wilcoxon p vs baseline | n/a | 0.0002 | n/a | n/a |

Best-trained CH physical sub-KPIs:

| sub-KPI | SOTA | SOTA+solar | SOTA+amp | realized |
|---|---:|---:|---:|---:|
| solar-bowl depth | 0.449 | 0.437 | 0.428 | 0.558 |
| peak/offpeak spread, EUR/MWh | 18.56 | 18.57 | 7.16 | 6.41 |

The immediate Cal-2025 peak/offpeak residual is now mostly closed by Phase 12
`sota_amp`. The remaining frontier is not another 2025 amplitude tweak. It is
the long-horizon structural question: how should a 2030 delivery shape differ
from 2027 when PV, batteries, EVs, heat pumps, demand, nuclear availability, and
cross-border flexibility follow different published trajectories?

This is not directly measurable in the single fully-realized Cal-2025 year. The
right evidence is therefore:

* faux-future validation on historical as-of splits;
* monotonic scenario contrast tests for 2027 vs 2030;
* preservation of market anchor constraints;
* explicit scenario uncertainty decomposition.

## 2. Method

Implement a low-dimensional, sign-constrained structural modulation layer on
top of the existing LT SOTA stack:

```text
log f_H_struct[t] = log f_H_base[t] + gamma[cell(t)] . (x_scenario[t] - x_ref)
```

where:

* `cell(t)` is a pooled season x day-type x block label;
* blocks are `NIGHT`, `MORNING_RAMP`, `MIDDAY_BOWL`,
  `AFTERNOON_SHOULDER`, `EVENING_RAMP`;
* `x_scenario` contains scenario drivers such as PV penetration, battery power
  share, battery energy cover, EV demand share, heat-pump winter share, nuclear
  availability, and NTC proxies;
* coefficients are ridge-shrunk and sign-constrained.

The first implementation should focus on deterministic transformation and
governance, not on a high-DoF estimator:

1. versioned scenario store with strict `publication_date <= vintage`;
2. block-level structural feature projection;
3. monotone coefficient table with conservative defaults;
4. local-day normalization of `f_H`;
5. optional scenario ensemble output in a later iteration.

The model uses published scenario assumptions as legitimate forward information
only when they are available as of the vintage. It never uses realized future
capacity or realized future prices.

## 3. Literature Justification

* Sensfuss, Ragwitz, Genoese (2008): RES moves the merit order and depresses
  hours with abundant renewable output.
* Hirth (2013): VRE market value declines with penetration, giving a direct
  economic rationale for PV-driven shape cannibalization.
* Denholm, Brinkman, Jorgenson (2015): high PV creates midday overgeneration and
  evening ramp stress; storage and flexibility are the operational counterforce.
* Seel, Mills, Wiser (2018): high-VRE futures materially alter wholesale hourly
  price patterns.
* Schmalensee (2022): competitive storage is a system-cost response to the duck
  curve and should compress spreads when deployed at sufficient power/energy.
* ENTSO-E/ENTSOG TYNDP 2024 and OFEN Energieperspektiven 2050+: published
  scenario sources for long-term European and Swiss electrification pathways.

## 4. Why This Is Not Re-Doing Earlier Work

| existing work | why Phase 13 is distinct |
|---|---|
| SOTA seasonal ratios | solves historical monthly shape and CH prior shrinkage |
| hydro-aware peak spreads | calibrates historical peak/base spread under CH hydro regime |
| solar modulation | uses realized/projected solar penetration from local history, capped to observed support |
| intraday amplitude shrinkage | compresses Cal-2025 peak/offpeak residual toward pre-vintage fitted spreads |

Phase 13 targets unobserved future regimes. It makes 2030 differ from 2027
because the published system trajectory differs, not because 2025 residuals are
retuned.

## 5. Integration Point

New module:

```text
pfc_shaping/lt/model/electrification_shape.py
```

Assembler kwargs, default OFF:

```python
enable_electrification_shape: bool = False
electrification_scenario: str | None = None
electrification_scenario_path: str | Path | None = None
```

Placement in `PFCAssembler.build()`:

1. `ShapeHourly.apply()` produces `f_H`;
2. existing `enable_solar_modulation` may apply;
3. `enable_electrification_shape` applies scenario modulation;
4. local-day `mean_h f_H = 1`;
5. existing damping, base, water value, and calibration continue.

Perfect-foresight estimator:

```text
sota_electrification
```

implemented as a scoped context-manager swap like `_sota_solar_estimator()` and
`_sota_amp_estimator()`.

## 6. Validation Plan

Unit tests:

* scenario store excludes rows with `publication_date > vintage`;
* missing required scenario columns fail fast;
* modulation preserves local-day mean exactly within numeric tolerance;
* PV-only central case deepens midday shape;
* adding battery energy/power refills midday and compresses evening ramp;
* constructor flag defaults OFF;
* OFF path does not import/call the module where practical.

Integration tests:

```powershell
$env:PYTHONPATH='.'; pytest tests/test_electrification_shape.py -q
$env:PYTHONPATH='.'; pytest tests/test_phase10_reproducibility.py -q
$env:PYTHONPATH='.'; pytest tests/test_perfect_foresight.py::test_build_curve_rejects_unknown_estimator_string -q
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/run_perfect_foresight.py --ab
```

Ship thresholds:

| gate | threshold |
|---|---|
| OFF identity | exact equality, `atol=0` |
| no-leakage | as-of and future-actual tests pass |
| Cal-2025 `pf_cal_corr` | no degradation below 0.85 gate |
| faux-future de-leveled metrics | non-worse vs SOTA/SOTA+solar |
| 2027 vs 2030 monotonicity | PV/battery/EV/PAC sign tests pass |
| scenario fan chart | structural width increases with horizon |

If empirical effect is weak, ship a null-result report with the flag OFF.

## 7. Leakage Analysis

Allowed at vintage `v`:

* spot history with timestamp `< v`;
* actual capacity or generation measurements with measurement timestamp `< v`;
* official scenario assumptions with `publication_date <= v`;
* calendar features derived from timestamps;
* market forward anchors observed as of `v`.

Forbidden:

* realized future spot prices;
* realized future capacity measurements;
* scenario rows whose `publication_date > v`;
* Cal-2025 or later benchmark residuals as fitted features;
* any data source without as-of metadata unless explicitly marked `fallback`
  and used only in diagnostics.

The scenario store must be responsible for the as-of filter; callers are not
trusted to filter correctly.

## 8. Rejected Alternatives

| alternative | reason rejected |
|---|---|
| retune Phase 12 amplitude | solves current Cal-2025 spread residual but not 2030 structural divergence |
| extrapolate `trend_per_hour_` harder | cannot see unbuilt assets |
| lift `solar_pen` cap by hand | ungoverned, non-reproducible, and one-driver only |
| full fundamental dispatch model | too large for this repo phase and risks replacing rather than shaping market anchors |
| free hourly ML model | too many degrees of freedom and poor interpretability for LT governance |

