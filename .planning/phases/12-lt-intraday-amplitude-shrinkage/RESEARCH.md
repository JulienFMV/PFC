# LT Next SOTA Plan-Gate: Intra-Day Amplitude Shrinkage

## 1. Diagnosed residual

Command run from `feat/lt-next-sota` at `0652a65`:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/run_perfect_foresight.py --ab
```

Local data caveat: `data/epex_hourly.parquet` was bootstrapped from tracked
`pfc_shaping/data/epex_15min.parquet`; `data/forwards_history_phase10.parquet`
was generated with `forwards_source=fallback_diagnostic`, so the forwards ladder
is diagnostic-only, not gate-eligible. The perfect-foresight Cal anchor and
CH-physical sub-KPIs are still usable for residual triage.

Measured Cal-2025 A/B:

| metric | baseline | sota | sota_solar |
|---|---:|---:|---:|
| median `pf_cal_corr` | 0.7447 | 0.9182 | 0.9182 |
| min `pf_cal_corr` | n/a | n/a | 0.8608 |
| vintage improvements vs baseline | n/a | 12/12 | 12/12 |
| Wilcoxon p, SOTA > baseline | n/a | 0.0002 | n/a |
| solar `pf_cal_corr` median delta | n/a | n/a | +0.0000 |

The remaining actionable residual is not monthly seasonal shape: `pf_cal_corr`
clears the 0.85 gate under SOTA. It is intra-day amplitude at the best-trained
vintage:

| sub-KPI | SOTA | SOTA+solar | realized | residual after SOTA+solar |
|---|---:|---:|---:|---:|
| solar-bowl depth | 0.449 | 0.437 | 0.558 | -0.121 |
| peak/off-peak spread, EUR/MWh | 18.56 | 18.57 | 6.41 | +12.16 |

Therefore the target is a measured, persistent over-amplitude of the local
08-20 peak premium and an under-deepened solar bowl. The existing solar layer did
not close this residual on the local production data; it moved bowl depth in the
wrong direction and left peak/off-peak unchanged.

## 2. Method and rejected alternatives

Proposed lever: a parsimonious `f_H` amplitude-shrinkage post-processor applied
after the current SOTA and solar hooks. It estimates one or two shrinkage slopes
on pre-vintage de-levelled realized residuals:

```text
f_H_adj[t] = 1 + a[cell_or_block] * (f_H[t] - 1)
```

with `a` constrained to `[0, 1]`, fitted by ridge/shrinkage toward `1` and
pooled by a small number of hour blocks. The layer then re-normalizes by local
day, preserving `mean_h f_H = 1`. Economically, this is a CH hydro-flexibility
correction: hydro storage arbitrage and solar flattening reduce thermal-style
peak/off-peak amplitude, consistent with Bevilacqua et al. 2022 and the
Karakatsani-Bunn / Wagner residual-demand literature. Statistically, the
low-degree parameterization follows Harrell's observations-per-parameter
discipline, as the repo already did for `solar_modulation.py`.

Rejected alternatives:

| alternative | reason rejected |
|---|---|
| Re-tune `RegimeAwareSeasonalRatios` | Monthly shape is already solved enough: median `pf_cal_corr` 0.9182, min SOTA+solar 0.8608. This would redo Phase 10 work. |
| Re-tune `HydroAwarePeakSpreads` | Peak spread calibration is already shipped; the residual remains after it, and the prompt explicitly says not to redo Section 3. |
| Re-tune solar betas | The solar layer is already implemented and currently worsens bowl depth on this run. A direct retune on the only realized year risks p-hacking. |
| Full residual-demand/GAM stack | Too many parameters for one fully realized delivery year and adds a wider leakage surface. |
| Far-horizon governance | The measured residual is in Cal-2025 intra-day amplitude under perfect-foresight, not Y+2/Y+3 seasonal damping. |

## 3. Integration point and reproducibility

Integration point: a new LT module, tentatively
`pfc_shaping/lt/model/intraday_amplitude.py`, with a pure function or small class
called from `PFCAssembler.build()` only when a new kwarg such as
`enable_intraday_amplitude_shrinkage=True` is set.

Placement:

1. `ShapeHourly.apply()` produces `f_H`.
2. Existing `enable_solar_modulation` may adjust `f_H`.
3. New amplitude shrinkage adjusts `f_H`.
4. Local-day normalization restores `mean_h f_H = 1`.
5. Existing damping, `f_W`, `f_Q`, `f_WV`, and calibration continue unchanged.

Default is `False`. The OFF path must not call the new module and must be
byte-identical to current output, proven with an `atol=0` test and preferably a
monkeypatch that raises if the new function is reached while disabled. Changes
are additive: a new module, one kwarg, one estimator variant such as
`sota_amp` or `sota_solar_amp`, and focused tests.

## 4. Validation plan and ship threshold

Primary validation:

```powershell
$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; python scripts/run_perfect_foresight.py --ab
```

Add a new estimator variant in `perfect_foresight.py` and benchmark against
`sota` and `sota_solar`.

Ship thresholds:

| metric | threshold |
|---|---|
| `pf_cal_corr` | remains >= 0.85 on all 12 Cal-2025 vintages |
| peak/off-peak spread | materially closer than 18.57 to realized 6.41; target <= 10 EUR/MWh without monthly-shape regression |
| solar-bowl depth | no worse than SOTA+solar 0.437; target moves toward realized 0.558 |
| significance | paired Wilcoxon or block-bootstrap CI on absolute-error improvement excludes zero where sample allows |
| reproducibility | flag OFF `max_abs_delta == 0.0` |

If effect size is weak or unstable, ship as a null-result report only.

## 5. Leakage analysis

Training data:

* realized EPEX prices strictly with `index < vintage`;
* calendar labels derived from timestamps only;
* no realized delivery-month information for periods `>= vintage`;
* optional physical covariates only if their publication/vintage semantics are
  explicit and filtered internally.

The module must enforce `index < vintage` inside fit/feature extraction,
independent of caller filtering. Forward delivery months receive only projected
or fitted coefficients from the pre-vintage training set. The correction cannot
use realized Cal-2025 sub-KPI residuals at production inference; those are only
validation targets in the perfect-foresight diagnostic.
