# Solar-Aware Intra-Day Shape — Research Specs (pre-implementation)

**Date** : 2026-05-30
**Status** : research only — no code yet. Two parallel expert-agent investigations
have been committed here so that the implementation can resume in a fresh
session without re-running the research.

## Motivation (carried over from `10-PERFECT-FORESIGHT-SHAPING.md` §4ter)

The current SOTA shaping stack (regime-aware seasonal_ratios + hydro-aware
peak_spreads + intra-day half-life=90d) clears the 0.85 SC#1 gate on all 12
Cal-2025 vintages (min 0.861, median 0.918, +0.174 vs baseline). The
**residual peak/off-peak amplitude gap (~14 €/MWh: model 20 vs realized 6.4)
is invariant to the half-life** and was traced to a 2025 solar-regime shift
that a past-fit `f_H` cannot anticipate.

Empirical measurement (already in the methodology doc): monthly CH solar
penetration (`solar_mw / load_mw`) correlates with realized midday-bowl depth
at **Pearson r = 0.914, p ≈ 5e-16, n = 39 months**. This is the genuine
remaining lever toward BKW/Axpo-grade fidelity.

## Files

- `method_research.json` — PhD-level methodology spec. Headline decisions:
  - **Method (A)**: direct multiplicative scaling of `f_H` with 3–4 block-pooled
    per-hour coefficients (rejects GAM/residual-demand-stack/two-regime alternatives
    for n_months ≈ 39).
  - **Feature**: monthly `solar_pen_m = Σ solar_mw / Σ load_mw` (CH realized;
    forward projection via PRONOVO capacity × CF-climatology, NOT DE forecast).
  - **Integration**: post-processing layer between `ShapeHourly.apply()` and the
    assembler — NOT a 4th component of `_sota_estimator()` swap. Preserves
    `mean_h f_H = 1`, keeps `f_W` untouched, atol=1e-12 bit-identical when flag=False.
  - **Validation**: peak/off-peak ±2 €/MWh of realized; bowl_depth ±0.05;
    `pf_cal_corr ≥ 0.85` on 12/12 hard gate; summer demeaned RMSE −15 %;
    bootstrap 90 % CI excludes zero.
  - **CV**: leave-one-year-out (auditor-friendly), walk-forward secondary;
    random K-fold rejected as leakage-prone.
  - **Cited**: Karakatsani-Bunn 2008, Wagner 2014, Cludius 2014, Kiesel-Paraschiv
    2017, Bevilacqua et al. 2022, Marcjasz et al. 2023.

- `data_probe.json` — senior-data-engineer audit of the 6 candidate parquets.
  Headline:
  - **GO** for `pfc_shaping/data/entso_15min.parquet` (CH 2021–2026, 0 % missing
    in key cols, no NaN runs > 24h).
  - **CH↔DE solar correlation**: 2023 = 0.945, 2024 = 0.932, 2025 = 0.942 hourly.
  - **Critical caveat on `de_renewable_forecast.parquet`**: zero-lag perfect
    alignment with CH realized → **likely realization-stamped, not forecast**.
    Must add an `as_of` column before any forecast use; v1 uses climatology
    forecast.
  - **Collinearity flag**: `solar_pen_m` vs `total_re_pen_m` ρ = 0.9999 — pick
    one. Recommend `solar_pen_m`. Trend term needed (ρ ≈ 0.44 with time index).
  - **Leakage map**: `vintage_filter()` pattern documented for the 4 source
    parquets (entso/epex/hydro/forecast).
  - **Calendar hygiene**: DST transitions, leap year, holidays all correctly
    handled — no issue.
  - **Data gaps to close (post-v1)**: (a) `as_of` column on DE forecast,
    (b) ingest DE realized solar, (c) monthly alignment rule for weekly hydro.

## Integrity

| file | bytes | sha256[:16] |
|---|---:|---|
| `method_research.json` | 23,980 | `35a47abb9f807b44` |
| `data_probe.json`      | 52,669 | `2f9894e920ddaab4` |

These hashes are recomputable with
``python -c "import hashlib;print(hashlib.sha256(open('FILE','rb').read()).hexdigest()[:16])"``.
They match what the two agents independently reported.

## Next session — implementation checklist

1. Read both JSON specs in full (sections 1–7 of `method_research.json`,
   sections 1–8 of `data_probe.json`).
2. Implement `pfc_shaping/lt/model/solar_modulation.py` with:
   - `SolarPenetrationFeature` — leak-free `solar_pen_m` extractor with a
     `vintage` arg (training uses < vintage; forward months use climatology +
     trend).
   - `SolarBlockedFHCorrection` — fits 3–4 block-pooled per-hour β coefs
     against the de-levelled midday residual, with ridge regularisation.
   - Post-`ShapeHourly.apply()` hook: `f_H_adj = f_H * (1 + β · (s − s̄))` on
     daytime hours, renormalise to preserve `mean_h f_H = 1`.
3. Wire via `_sota_estimator()` as an *additional* swap (or as a kwarg to
   `ShapeHourly` if simpler), defaulting OFF for repro.
4. Tests under `tests/test_solar_modulation.py`:
   - Identity test (β = 0 → no change to f_H).
   - Mean-preservation test (`mean_h f_H_adj == 1` post-swap).
   - Leak test (extracting `solar_pen_m` at vintage `v` uses ONLY data < v).
   - End-to-end on best vintage 2024-12-31: check `peak_offpeak_spread` and
     `bowl_depth` shifts in the right direction; assert within ship thresholds.
5. Extend `scripts/run_perfect_foresight.py --ab` with a third estimator
   variant (`sota_solar`) for paired benchmark vs `sota`.
6. Five-agent QA audit pass (methodology, code, repro-isolation, numerical,
   statistical), same discipline as the previous round.
7. Doc update in `10-PERFECT-FORESIGHT-SHAPING.md` (§4quater).

## Provenance

Two parallel sub-agents launched in session
`session_01KAVRuVUAtpu9Lm4PQKuGy2`:
- `a62a52b31710f126f` — "SOTA solar-aware intra-day shape method" (29.5k tokens, 6 tools, 312s).
- `a9d791b4d7098c8a1` — "Solar/wind data leakage probe" (58.9k tokens, 14 tools, 376s).

Both were read-only research agents (no code modifications). The session's
stdout channel was unreliable, so the agents wrote JSON to `/tmp` and the
SHA-256 cross-check was used to verify integrity before committing here.
