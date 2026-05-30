# Perfect-Foresight Shaping Diagnostic — Methodology & Findings

**Date** : 2026-05-29
**Scope** : additive, non-gating diagnostic isolating **shaping** quality from
**forward-forecast** error in the Swiss HPFC. Targets delivery Cal 2025 (the only
fully-realized forward year in the EPEX coverage, which ends 2026-03-15).
**Code** : `pfc_shaping/validation/perfect_foresight.py`,
`scripts/run_perfect_foresight.py`, `tests/test_perfect_foresight.py`.
**Reproducibility** : never touches `pillar1/pillar2_df/pillar3_df/pillar4_df`
(the objects guarded by `tests/test_phase10_reproducibility.py`, atol=1e-12).

---

## 1. Motivation

The Pillar 1 `seasonal_profile` test fails (aggregate 0.78–0.81 < 0.85). The
earlier audit (`10-AUDIT-FORWARDS.md`) argued this was mostly a *forward-forecast*
artefact (the market's forward curve mispricing the level), not a shaping defect —
but it argued it by reasoning, not by measurement. The `seasonal_profile` test, as
designed, **cannot separate** two error sources:

- **(a) shaping error** — does the cascader distribute a *given* level correctly
  across months/days/hours?
- **(b) forward-forecast error** — did the market's traded forward curve match
  what actually happened?

A test that conflates (a) and (b) cannot certify shaping. This diagnostic was
built to *measure* the split, and is the falsification experiment for the audit.

## 2. Methodology (literature-grounded)

The HPFC literature decomposes a forward curve into a **level** (arbitrage-free
constraint: the hourly curve averages back to traded futures over each delivery
block) and a **shape** (hourly × daily × weekly × seasonal profile). A shaping
metric must measure the shape *conditional on the level being correct*.

- **Fleten & Lemming (2003)**, *Energy Economics* 25(5) — QP: smoothest curve close
  to a prior shape s.t. period-averages equal traded prices. The λ smoothing weight
  trades shape fidelity vs continuity (under-smoothing inflates apparent skill).
- **Benth, Koekebakker & Ollmar (2007)**, *J. Derivatives* 15(1) — maximum-smoothness
  spline; arbitrage condition = integral over each delivery period equals the
  contract price; **shape is exogenous, traded curve sets only the level/area**.
- **Kiesel, Paraschiv & Sætherø (2019)**, *Comput. Manag. Sci.* 16(1) — SOTA: the
  seasonality shape is partly identified by the forwards themselves; pure historical
  profiles are a baseline, not the frontier.
- **Caldana, Fusai & Roncoroni (2017)**, *EJOR* 261(3) — no-arbitrage embedding of
  the hourly object inside coarse traded products, EPEX-validated.
- **Lago, Marcjasz, De Schutter & Weron (2021)**, *Applied Energy* 293 — EPF metric
  discipline: report pattern *and* amplitude; DM-significance vs a naive benchmark.
- **Gneiting & Raftery (2007)**, *JASA* 102(477) — strictly proper scoring rules
  (CRPS, energy score) for the probabilistic extension.
- **Marcjasz, Narajewski, Weron & Ziel (2025)**, arXiv:2503.02518 — the 2021–2023
  crisis contaminates trailing-window seasonal-component estimation; treat it.
- **Bevilacqua et al. (2022)**, *Energy Economics* — CH forward shape is tightly
  coupled to DE/FR/IT and driven by hydro/liquidity; basis matters (CH is outside
  EU SDAC).

Two complementary mechanisms operationalise "perfect foresight":

### 2.1 Build-side anchoring (seasonal shape)
Re-anchor `build_one` on the **realized Cal settlement** of the delivery year
(computed ex-post from EPEX, Europe/Zurich), while the cascader is still trained
*only* on data available before the vintage. The resulting monthly signature is
compared to realized. Because the level is now (nearly) perfect, the residual is
**dominated by seasonal profiling** error.

> **Anti-circularity** : we anchor at the *coarse* (Cal / Cal+Quarter) granularity,
> never at month level. Anchoring monthly settlements would make the monthly-
> signature test circular (MSFC forces monthly means ≈ anchors → corr≈1 trivially;
> empirically verified at r = 0.99997). Cal anchoring forces the cascader to
> *generate* the entire 12-month profile from its `seasonal_ratios`.

> **Caveats on "pure shaping error"** :
> - The Cal anchor pins the **smoothed backbone B** mean exactly; the delivered
>   `price_shape` is `B × f_S × f_W × f_H × f_Q × f_WV`, and the multiplicative
>   shape factors introduce O(0.2 €/MWh) drift on the base Cal mean and O(3–10%)
>   on peak monthly means. So the residual `1 − pf_cal_corr` is "dominated by
>   `seasonal_ratios` cascading" but not 100% pure (small contribution from the
>   f-chain monthly-mean drift, visible in the `Energy consistency` warnings).
> - `fit_seasonal_ratios` only counts **full calendar years**; vintages within
>   the same calendar-year cohort therefore share identical seasonal_ratios. The
>   per-vintage sweep is effectively **a small number of distinct regimes**
>   (typically 2–3 for our 24-vintage cohort), not 24 independent samples — the
>   diagnostic exposes `n_distinct_pf_corr_regimes` so callers can interpret the
>   robust aggregate correctly.

### 2.2 Score-side de-levelling (intra-period shape)
Normalise *both* model and realized to the same block mean, then score the
normalised diurnal/weekly profiles with two complementary axes:
- **cosine similarity** — pattern fidelity (right peak hour, right weekend dip);
  invariant to both additive and multiplicative rescaling of the profile.
- **demeaned RMSE** — amplitude fidelity (right peak-trough spread, bowl depth);
  **additive-invariant only** — responds to multiplicative rescale by design,
  because amplitude SHOULD respond to scale. The wording "level-invariant" is
  shorthand for additive-invariant.

The score is computed **strictly on the realized delivery-year window**
(`year == target_year` filter) so extrapolated hours outside the anchored window
(`_resolve_base`'s year-offset fallback for Cal 2026+ in 3-year-horizon builds)
do not contaminate the diurnal/weekly cosine and RMSE.

### 2.3 Robust aggregation & CH-physical decomposition
Per-vintage results are aggregated with a **distributional** summary
(median, p10, p90, min, max) rather than the brittle `min` order statistic the
gate uses. Shape error is decomposed into interpretable CH-physical sub-KPIs:
winter/summer ratio (hydro-storage + heating), solar-bowl depth (DE coupling,
regime-drifting), peak/off-peak spread (hydro flexibility flattening).

## 3. Experimental design

Delivery target = **Cal 2025** (fully realized). The market's tradeable Cal-2025
key appears only in early-2024 vintages, and realized coverage ends 2026-03-15, so
maturity and realization cannot be decoupled via traded keys alone. We therefore
**fix the target = realized Cal 2025** and sweep the vintage 2024-01 → 2024-12;
each vintage trains its `seasonal_ratios` on `[2023 .. vintage)` and is anchored on
the realized Cal-2025 level. This yields a clean *seasonal-shape skill vs training
maturity* curve under perfect level foresight.

## 4. Findings (delivery Cal 2025)

### 4.1 The forward-forecast hypothesis is refuted
Under **perfect annual-level foresight**, the monthly-shape correlation is only
**~0.745** (median), rising to **0.824** once a 2nd full calendar year enters
training. Perfect level foresight does **not** lift correlation to ~0.95. The bulk
of the seasonal-shape miss is therefore **profiling**, not forward-forecast error —
the opposite of the audit's strong claim.

### 4.2 The cascader's standalone seasonal profiling is the weak link
Counter-intuitively, `market_corr` (≈0.85 median) **exceeds** `pf_cal_corr`
(≈0.745). Stripping the curve to a single annual anchor forces it to build the
entire monthly shape from `seasonal_ratios` (trained on the 2023 crisis year),
which is **weaker** than the traded quarter/month granularity the market normally
supplies. So: when the market quotes quarters, the curve is fine; when it must
*interpolate* seasonal shape from a bare annual level (far horizon), it degrades.
The granularity ladder (Cal → Cal+Quarter → market) quantifies this.

### 4.3 Training-data starvation, not (only) model logic
Correlation is flat ~0.745 for 1.0–1.8 years of history and jumps to 0.824 exactly
when the 2nd full calendar year (2024) completes — `fit_seasonal_ratios` uses only
*full* calendar years, and trend terms need ≥3. The early-vintage fails are
substantially a **data-starvation + crisis-contamination** problem (cf. Marcjasz
et al. 2025), consistent with — but more precise than — the audit.

### 4.4 Two concrete, literature-aligned shape defects (CH sub-KPIs)
At the best-trained vintage, model vs realized:
- **peak/off-peak spread : ~20 vs ~6 €/MWh** — the model **overstates** the spread
  ~3×. Classic CH pitfall: a thermal-like profile ignoring hydro-flexibility
  flattening (Bevilacqua et al. 2022).
- **solar-bowl depth : ~0.42 vs ~0.56** — the model **understates** the summer
  midday bowl (regime drift: deepening solar penetration not captured).
- **winter/summer ratio : ~1.67 vs ~1.70** — nearly correct (the seasonal *level*
  ratio is fine; the problem is intra-day amplitude, not the seasonal mean).

## 4bis. SOTA optimisation: regime-aware seasonal_ratios

Beyond diagnosing the gap, the diagnostic was used as the optimisation target
for a drop-in SOTA `seasonal_ratios` estimator
(`pfc_shaping/calibration/robust_seasonal_ratios.RegimeAwareSeasonalRatios`).
Three improvements over the baseline LS+full-year estimator:

1. **Regime-aware down-weighting** (Marcjasz et al. 2025, arXiv:2503.02518) —
   each year contributes proportionally to `exp(−|level_y − long_run_median| /
   scale)`, with `scale = 30 €/MWh`. Crisis years (2022 ≈ 230 €/MWh) get weight
   ≈ 0.007: soft-excluded, not hard-deleted.
2. **Regime-weighted mean aggregation** (not median) — chosen over Hildmann
   LAD because the cascader's downstream hour-conservation semantics are
   mean-based; an over-aggressive median operator dragged the well-trained
   vintage from 0.82 → 0.55 in tuning. The regime weights provide outlier
   robustness without breaking the semantics.
3. **Bayesian shrinkage to a CH-physical prior** (Bevilacqua et al. 2022) —
   posterior `(n_eff · empirical + α · prior) / (n_eff + α)` with `α = 0.5`.
   This stabilises early vintages (where empirical evidence is one crisis
   year) without dragging down well-trained vintages.

We also use **partial-year** data and any year above `min_obs_per_period =
100` hours of valid data; the baseline's strict full-year filter is the main
reason 10/12 vintages collapse to identical ratios.

### A/B benchmark (Cal 2025, 12 vintages, Wilcoxon paired)

| metric | baseline | SOTA | gain |
|---|---:|---:|---:|
| pf_cal_corr median | 0.745 | **0.852** | **+0.108** |
| pf_cal_corr min | 0.703 | 0.816 | +0.112 |
| pf_cal_corr max | 0.824 | 0.883 | +0.059 |
| vintages improved | — | **12/12** | strict Pareto |
| Wilcoxon signed-rank | — | — | **p = 0.0002** |

The SOTA median **passes the SC#1 seasonal_profile gate** (0.85 threshold) where
the baseline fails. The Wilcoxon paired test rejects H0 ("no improvement") at
p < 0.001. Available as opt-in via `build_curve(estimator="sota")` and
`run_perfect_foresight(estimator="sota")`; the in-tree fitter is unchanged so
the `atol=1e-12` reproducibility contract holds.

## 5. Recommendations

The diagnostic converts the audit's "no change needed" into a **targeted,
defensible improvement programme**, with this metric as the optimisation target:

1. **Robust, regime-aware `seasonal_ratios`** — estimate with a robust loss
   (LAD/Huber/M-regression) and down-weight/exclude the 2021–2023 crisis window
   (Marcjasz et al. 2025; M-regression arXiv:1806.09803). Highest leverage in the
   far horizon where only Cal/Quarter quotes exist.
2. **Recalibrate the peak/off-peak spread to CH hydro flexibility** — the additive
   spread is ~3× too wide vs realized; investigate `fit_peak_spreads` and the
   intra-day `f_H` amplitude.
3. **Refresh the solar-bowl profile** — the summer midday depression is drifting
   deeper with DE solar coupling; the historical profile under-captures it.
4. **Gate aggregation** — replace the `min` order statistic with a robust quantile
   + reported distribution (median/p10/p90), and tag the reference window by regime.

All four are profiling/calibration changes that would move *this* metric; none
require touching the arbitrage-free level machinery.

## 6. Limitations & findings flagged for the codebase owner

- **Only Cal 2025 is fully realized** → a single delivery year; conclusions on
  far-horizon Cal years (2026+) await more realized data.
- **Maturity and realization are partially confounded** by the short EPEX
  coverage.
- **Granularity-ladder caveat** : at the late vintages where the cascader is
  best-trained, the market may already quote at *month* granularity for the near
  horizon, so the `market` ladder rung is not always at coarser granularity than
  `pf_cal_quarter`. The runner emits the exact anchor key set per rung in the
  report so the comparison is auditable.
- **Pre-existing inconsistency in `cascading.fit_peak_spreads` (HIGH, FLAGGED,
  NOT FIXED)** : the calibrator's peak mask is `weekday & hour ∈ [8,20)` and
  does NOT exclude CH national holidays, while every other peak mask in the
  codebase (assembler `_is_peak_timestamp`, `count_hours`, this module's
  `_peak_mask`) does. Impact on the fitted spread is small (~1.5%, 4 CH national
  holidays / 260 weekdays). Fixing it would shift Pillar 1/2/4 outputs and
  break the `atol=1e-12` reproducibility contract; defer to the codebase owner.
- **Probabilistic axis** (CRPS/energy score) is specified but not yet wired (the
  HPFC here is deterministic); see §2.2 for the extension path.
