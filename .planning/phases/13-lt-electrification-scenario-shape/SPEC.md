# Phase 13 SPEC: LT Electrification Scenario Shape

## 0. Executive Thesis

The next long-term HPFC frontier is not another historical shape retune. It is a
structural non-stationarity problem: a curve delivering 2027 and a curve
delivering 2030 should not share the same intra-day and seasonal shape merely
because the quoted forward anchors are calibrated. The shape must be conditioned
on the projected state of the power system.

Target capability:

```text
HPFC shape(delivery_year, vintage, scenario)
  = historical SOTA shape
  + leak-free structural modulation from projected electrification drivers
```

The module must stay additive and OFF by default. With the flag OFF, the
existing pipeline remains byte-identical. With the flag ON, the HPFC can produce
scenario-specific shapes for slow, central, and fast electrification pathways,
then aggregate them into a weighted curve and a widening structural fan chart.

## 1. Problem Statement

Current LT shape evolution is mostly endogenous:

| mechanism | current role | structural limit |
|---|---|---|
| `ShapeHourly.trend_per_hour_` | extrapolates past hourly profile trends by delivery horizon | linear extrapolation from a short historical regime |
| `PFCAssembler._shape_freedom()` | damps rich shapes toward the backbone with horizon | generic flattening, not physical scenario evolution |
| `solar_modulation.SolarPenetrationFeature.project()` | projects monthly solar penetration from realized history | capped at historical support and cannot express 2030 PV/storage buildout |

This is acceptable for near-term curve shaping but not for a 2030 delivery
curve. The 2030 shape is driven by assets and flexible loads that do not yet
exist in realized spot history: PV, batteries, EVs, heat pumps, demand-side
response, nuclear availability, and cross-border transmission.

## 2. Design Principle

Use the market forward curve for tradable energy anchors. Use historical data to
estimate low-dimensional response coefficients. Use scenario trajectories to
move the future shape.

The model must never overwrite the market:

```text
quoted Cal/Q/M/Peak/Offpeak means are constraints
electrification layer only redistributes energy within unconstrained dimensions
```

Concretely, the layer acts on `f_H` and optionally on a final
month-mean-preserving price residual. It must preserve:

* local-day mean of `f_H` equal to 1 after modulation;
* quoted period means after arbitrage-free calibration;
* no usage of realized future prices, realized future capacity, or post-vintage
  scenario publications.

## 3. Data Contract

Create a versioned scenario table. Preferred path:

```text
data/electrification_scenarios.parquet
```

Minimum schema:

| column | type | meaning |
|---|---|---|
| `publication_date` | timestamp UTC | source publication or internal approval date |
| `source` | string | `OFEN_EP2050`, `ENTSOE_TYNDP`, `PRONOVO`, `BNETZA_MASTR`, `INTERNAL_FAST`, etc. |
| `scenario` | string | `slow`, `central`, `fast`, or official scenario name |
| `country` | string | `CH`, `DE`, `FR`, `IT`, etc. |
| `delivery_year` | int | target year |
| `delivery_month` | int nullable | optional month-level trajectory |
| `scenario_weight` | float nullable | ex-ante scenario probability for aggregation |
| `pv_gw` | float nullable | installed PV capacity |
| `wind_gw` | float nullable | installed wind capacity |
| `battery_power_gw` | float nullable | battery discharge/charge power |
| `battery_energy_gwh` | float nullable | battery energy capacity |
| `ev_twh` | float nullable | annual EV electricity demand |
| `heatpump_twh` | float nullable | annual heat-pump electricity demand |
| `demand_twh` | float nullable | annual gross electricity demand |
| `nuclear_gw` | float nullable | available nuclear capacity |
| `ntc_ch_de_gw`, `ntc_ch_fr_gw`, `ntc_ch_it_gw` | float nullable | interconnection capacity proxies |
| `quality_flag` | string nullable | `official`, `internal`, `fallback`, `inferred` |

The store must expose only as-of views:

```python
scenario_store.asof(vintage).for_delivery(year, scenario, country)
```

Hard leakage rule:

```text
publication_date <= vintage
actual installed capacity rows must also have measurement timestamp < vintage
```

Official future assumptions published before the vintage are allowed. Future
realized capacity measured after the vintage is not.

## 4. Structural Features

Raw capacity is not enough. Convert drivers to normalized physical features:

| feature | definition | intuition |
|---|---|---|
| `pv_pen_y` | annual PV generation proxy / annual demand | midday cannibalization pressure |
| `pv_midday_share_m` | monthly PV production proxy / local midday load | seasonal bowl depth |
| `battery_energy_cover_h` | battery GWh / average daily load GW | ability to move midday surplus |
| `battery_power_share` | battery GW / peak load GW | ability to compress evening peak |
| `ev_load_share_y` | EV TWh / demand TWh | electrified transport load |
| `ev_managed_share` | managed charging share if available | whether EVs flatten or worsen evening |
| `hp_winter_share` | winter heat-pump load proxy / winter demand | winter seasonality and evening peaks |
| `residual_load_slope_proxy` | PV/wind/load scenario projection by block | ramp pressure |

Use scenario-provided generation if available. Otherwise compute simple capacity
to generation proxies with fixed, documented capacity factors and seasonal
profiles. The proxy assumptions must be versioned and testable.

## 5. Shape Basis

Use a small block basis. Do not estimate free hourly coefficients for a single
country/year.

Recommended blocks:

| block | hours local | purpose |
|---|---|---|
| `NIGHT` | 00-05, 22-23 | EV night charging, low PV |
| `MORNING_RAMP` | 06-09 | morning load and solar ramp |
| `MIDDAY_BOWL` | 10-15 | PV cannibalization and battery charging |
| `AFTERNOON_SHOULDER` | 16-17 | transition from PV to ramp |
| `EVENING_RAMP` | 18-21 | net-load ramp, batteries, EV/PAC demand |

Season grouping:

```text
WINTER = Dec/Jan/Feb
SHOULDER = Mar/Apr/May/Oct/Nov
SUMMER = Jun/Jul/Aug/Sep
```

Day-type grouping:

```text
WORKDAY vs WEEKEND/HOLIDAY
```

## 6. Model Form

Preferred first implementation: constrained additive log-shape modulation.

Let `x_yms` be structural driver vector for delivery month `m`, scenario `s`,
and delivery year `y`. Let `b(t)` be the block/season/day-type cell for local
timestamp `t`.

```text
log_f_H_adj[t] = log(f_H_base[t]) + gamma[b(t)] . (x_yms - x_ref)
```

Then:

1. exponentiate;
2. normalize `f_H_adj` to local-day mean 1;
3. pass through existing downstream layers;
4. after calibration, optionally apply a month-mean-preserving structural
   residual correction only for dimensions not quoted by market products.

Sign and monotonic constraints:

| driver | required sign pattern |
|---|---|
| PV | `MIDDAY_BOWL <= 0`, `MORNING_RAMP <= 0`, `EVENING_RAMP >= 0` only through normalization or explicit ramp term |
| battery energy/power | `MIDDAY_BOWL >= 0`, `EVENING_RAMP <= 0`, spread compression non-negative |
| managed EV | `MIDDAY_BOWL >= 0` or `NIGHT >= 0`, `EVENING_RAMP <= 0` if managed |
| unmanaged EV | `EVENING_RAMP >= 0`, `NIGHT >= 0` |
| heat pumps | `WINTER MORNING_RAMP >= 0`, `WINTER EVENING_RAMP >= 0`, winter/summer ratio increasing |
| nuclear exit | raises scarcity-sensitive winter/evening blocks unless offset by imports |
| interconnection | shrinks country-specific extremes and CH basis |

Coefficient estimation:

* use ridge or Bayesian hierarchical shrinkage;
* pool across CH/DE/FR/IT where data quality allows;
* include country fixed effects and market-year random effects;
* enforce signs via constrained least squares or projected gradient;
* target de-leveled hourly residuals, not raw prices.

Do not tune to Cal-2025 alone. Use Cal-2025 as a diagnostic, not as the sole
objective.

## 7. Integration Point

New module:

```text
pfc_shaping/lt/model/electrification_shape.py
```

Suggested public API:

```python
class ElectrificationScenarioStore:
    def __init__(self, path: str | Path): ...
    def asof(self, vintage: pd.Timestamp) -> "ElectrificationScenarioStore": ...
    def get(self, country: str, scenario: str, delivery_period: pd.Period) -> pd.Series: ...

class StructuralDriverProjector:
    def transform(self, scenario_row: pd.Series, calendar_df: pd.DataFrame) -> pd.DataFrame: ...

class ElectrificationFHCorrection:
    def fit(self, spot_history, driver_history, *, vintage) -> "ElectrificationFHCorrection": ...
    def apply(self, f_H, calendar_df, *, vintage, scenario, delivery_year) -> pd.Series: ...

def electrification_modulate(f_H, calendar_df, shape_hourly, *, vintage, scenario, scenario_store) -> pd.Series: ...
```

Assembler flag:

```python
enable_electrification_shape: bool = False
electrification_scenario: str | None = None
electrification_scenario_path: str | Path | None = None
```

Placement:

1. `ShapeHourly.apply()` produces `f_H`.
2. Existing `enable_solar_modulation` may run.
3. New `enable_electrification_shape` runs.
4. Normalize local-day `f_H`.
5. Existing damping, `f_Q`, `f_WV`, backbone, and calibration continue.

The layer should supersede the linear-capped future projection inside
`solar_modulation` only when explicitly enabled. It must not silently change
`sota_solar`.

## 8. Scenario Aggregation

The model must support single-scenario and ensemble outputs:

```text
curve_slow
curve_central
curve_fast
curve_weighted = sum_s weight_s * curve_s
structural_p10/p90 = quantiles across scenarios plus existing stochastic uncertainty
```

Scenario uncertainty is not the same as price residual volatility. It should be
reported separately:

```text
p10_total = combine(stochastic_p10, scenario_p10)
p90_total = combine(stochastic_p90, scenario_p90)
```

For a first version, keep combination transparent:

```text
total_band_width^2 = stochastic_band_width^2 + scenario_band_width^2
```

This is an approximation, but it is auditable and avoids pretending structural
uncertainty is Gaussian spot noise.

## 9. Validation Plan

Minimum validations:

1. OFF-path identity:
   * default flag is `False`;
   * module is not imported when disabled;
   * full pipeline output equal to baseline at `atol=0`.
2. No-leakage:
   * scenario rows with `publication_date > vintage` are excluded;
   * realized actual capacity after vintage is excluded;
   * module raises on naive future actuals without publication metadata.
3. Invariants:
   * local-day mean `f_H == 1` after modulation;
   * no quoted period mean drift after calibration;
   * no negative or exploding factors under fast scenario.
4. Faux-future backtest:
   * train as-of 2022, predict 2023/2024 shape;
   * train as-of 2023, predict 2024/2025 shape;
   * compare against SOTA and SOTA+solar on de-leveled diurnal metrics.
5. Perfect-foresight diagnostic:
   * extend `scripts/run_perfect_foresight.py --ab` with `sota_electrification`;
   * do not overfit to 2025; report it as one diagnostic year.
6. Scenario contrast:
   * generate 2027 vs 2030 central/fast scenarios;
   * verify qualitative monotonicity:
     - PV-only deepens midday bowl;
     - PV plus batteries partly refills midday and compresses evening spread;
     - heat pumps increase winter/evening structure.

Ship threshold:

| gate | threshold |
|---|---|
| OFF identity | exact equality, `atol=0` |
| Cal-2025 monthly `pf_cal_corr` | no degradation below current gate 0.85 |
| de-leveled diurnal RMSE | non-worse vs SOTA on faux-future backtests |
| 2027/2030 monotonicity | all sign tests pass |
| scenario fan chart | width increases with horizon and fast-vs-slow divergence |
| governance | all data sources have vintage/as-of metadata |

## 10. Rejected Shortcuts

| shortcut | rejection |
|---|---|
| Extrapolate `trend_per_hour_` further | cannot see unbuilt assets and saturates poorly |
| Increase solar beta by hand for 2030 | ungoverned and not reproducible |
| Use realized future PV capacity in validation | leakage |
| Fit free hourly neural net on 2025 | too many degrees of freedom for one realized year |
| Replace market anchors with fundamental level | breaks HPFC purpose and arbitrage-free contract |
| Merge scenarios into one central assumption only | hides structural uncertainty and risk premia |

## 11. Quant Deliverables

The implementation should produce:

* scenario store with as-of filtering;
* structural feature transformer;
* constrained block-coefficient correction;
* estimator variant `sota_electrification`;
* optional scenario ensemble runner;
* report tables:
  - driver values by scenario/year;
  - coefficient signs and magnitudes;
  - 2027/2030 block deltas;
  - A/B validation vs SOTA and SOTA+solar;
  - scenario fan chart decomposition.

## 12. Literature Backbone

See `LITERATURE.md` in this phase. The highest-value citations are:

* Sensfuss, Ragwitz, Genoese (2008) for merit-order effect.
* Hirth (2013) for declining market value of VRE.
* Denholm, Brinkman, Jorgenson (2015) for duck curve and overgeneration.
* Seel, Mills, Wiser (2018) for hourly price-pattern effects under high VRE.
* Schmalensee (2022) for competitive storage and duck-curve economics.
* Gowrisankaran, Reynolds, Samano (2016) for intermittency value.
* ENTSO-E/ENTSOG TYNDP 2024 and OFEN/EP2050+ for scenario data.

