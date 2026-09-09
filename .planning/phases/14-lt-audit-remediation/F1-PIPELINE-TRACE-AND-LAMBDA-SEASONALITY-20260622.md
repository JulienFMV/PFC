# F1 — Pipeline provenance trace & Λ(t) seasonality sizing — 2026-06-22

Scope: **strictly read-only**. No code/config changed, no flag flipped, no production generation.
Sole artifact written = this report. Branch: `fix/lt-audit-remediation`.
Purpose: (1) establish which pipeline actually feeds the delivered Power BI curve;
(2) measure the structurally defensible CH seasonal amplitude to size the missing
seasonality prior Λ(t) — **not** to fix the model yet (no implementation without a new go).

---

## 1. Provenance graph (verified, code-level)

Three distinct orchestrations exist; they diverge.

| Path | Entry | B(t) level source | Monthly solver | LEAR/CT | Writes |
|------|-------|-------------------|----------------|---------|--------|
| **A** | `run_pfc_production.py` → `pfc_shaping/pipeline/production_phases.py` | solver *iff* `config.yaml:forwards.monthly_curve_solver.enabled` | wired, **OFF by default** (`config.yaml:86`) | yes | `pfc_shaping/model/artifacts/*.parquet/.csv` + `production_monthly_curve_manifest.json` |
| **B** | `run_daily.py` → `python -m pfc_shaping.pipeline.rolling_update` (`run_daily.py:157-158`) | **legacy `ContractCascader` always** (`rolling_update.py:391`); builds its own `PFCAssembler` at `rolling_update.py:405` **without** `monthly_level_authority` | **absent** | **absent** | its own outputs |
| **C** | `scripts/export_local_test_ch_hourly_csv.py` → `ch_hfc_hourly*.csv` | solver (when `--enable-monthly-forward-curve-solver`) | yes | — | `ch_hfc_hourly*.csv`, `*.monthly_curve_manifest.json` |

**Power BI chain (verified verbatim):**

```
export_local_test_ch_hourly_csv.py  →  ch_hfc_hourly*.csv
        │
        ▼
build_powerbi_exports.py  (LATEST_HFC_GLOB="ch_hfc_hourly*.csv", :41,:95)
        │   groupby(year, month, hour).mean()          (:153)
        ▼
powerbi/data/duck_month_hour_long.csv   (:179)   +  hfc_hourly_powerbi.csv (:148)
        │
        ▼
Power BI semantic model  (Source = \\fmvfs2\…\PFC_LT\powerbi\data\duck_month_hour_long.csv)
        │
        ▼
"Monthly means by year — Duck Month Avg Weighted Price"   ← the chart that was flagged
```

`duck_month_hour_long.csv` is written **only** by the export family (`export_local_test_ch_hourly_csv.py`, `build_powerbi_exports.py`, `build_powerbi_semantic_model.py`); neither `run_pfc_production.py` (A) nor `rolling_update.py` (B) writes it.

**Conclusion C-1.** The delivered Power BI chart comes from **Path C (export)**, which runs the **monthly solver**. The reproduced solver/proof curve matches the chart to the euro (2027≈134, 2028≈115, 2029≈83, 2030≈74). The far-horizon flatness the desk sees **is** the monthly solver's far horizon, i.e. the **missing seasonality prior Λ(t)** — not a legacy-cascade artifact.

**Conclusion C-2 (open, organisational).** It is unverified from code alone whether Path C is the *sanctioned* daily production or an analyst-run export. What is certain: the artifact the dashboard reads is produced by C, and the daily cron entry `run_daily.py` routes to **B (legacy, no solver/LEAR/governance)**. The 14-commit monthly-solver + governance work touches A and C, **never B**, and is **OFF by default even on A**. → The new layer is not yet the authoritative daily path; this must be resolved (unify on one path) before any flag-ON.

---

## 2. Effective config & silent-degradation surface (read-only findings)

- `config.yaml:86 forwards.monthly_curve_solver.enabled: false` → on Path A, B(t) is built by **legacy `ContractCascader` + `msfc_spline`**, not the solver.
- **Silent fabricated forwards**: `forward_proxy.derive_base_prices` substitutes spot-derived forwards on any EEX-source miss (XLSX→UNC→Databricks→proxy), **INFO log only** (`forward_proxy.py:223-224`); no manifest/gate flag distinguishes a proxy run from a real-quote run.
- **Repository data**: the worktree carries only `data/eex_forwards_history.parquet` (CH/DE/FR/AT/IT, Cal/Quarter/Month, BASE+PEAK, 2020-05→2026-06; 145 302 rows) and `commodities_cache.parquet`. **No realized-spot EPEX parquet is in the repo** — spot history lives on the fileshare. Consequence: the *spot climatological* measure below cannot be computed in this environment and is flagged as to-be-run on the fileshare.

---

## 3. Forward-market seasonality measurement (the Λ(t) evidence)

Method (read-only): for each market, BASE only, ratio `r = Month_price / Cal_price` for every
`(snapshot_date, delivery_year, month)` where both the month and that year's Cal are quoted;
grouped by calendar month; the 12-vector hour-weight-renormalised to mean 1. This is the
"historical relations between traded forward months" estimator (KYOS/Montel standard).

### 3a. CH forward-market monthly ratio (partial coverage)

| Month | Market (norm.) | σ | n | assembler hard | priors hard |
|---|---|---|---|---|---|
| Jan | **1.270** | 0.206 | 372 | 1.18 | 1.20 |
| Feb | **1.305** | 0.185 | 251 | 1.12 | 1.18 |
| Mar | 1.082 | 0.128 | 238 | 1.02 | 1.05 |
| Apr | 0.859 | 0.032 | 98 | 0.90 | 0.95 |
| May | 0.735 | 0.036 | 65 | 0.85 | 0.85 |
| Jun | 0.749 | 0.051 | 22 | 0.88 | 0.82 |
| Jul–Dec | *(insufficient CH quotes)* | — | — | — | — |

- CH winter/summer (covered months) **W/S ≈ 1.72**, vs hard-coded **assembler 1.28** and **priors 1.40**.
- **Caveats**: (i) CH months are quoted mainly Jan–Jun for the near year → only a partial shape (no clean Jul/Aug trough or Nov/Dec peak); (ii) the 2020-2026 sample spans the 2021-2023 gas crisis → winter premia inflated, σ large (±0.21 on Jan). The crisis-robust value is lower (see DE).

### 3b. DE full-12-month witness — does the premium survive at far horizon?

| Horizon | n months | winter | summer | **W/S** |
|---|---|---|---|---|
| DE h+1 (next year) | 12 | 1.166 | 0.897 | **1.30** |
| DE h+2 (2 years out) | 12 | 1.227 | 0.861 | **1.42** |

**Finding F3-α (decisive).** The seasonal amplitude **does not decay with horizon — it persists and slightly increases** (W/S 1.30 → 1.42 from h+1 to h+2). This is direct forward-market evidence that a full winter premium is priced **2 years out**, contradicting both the model's flat far-horizon and the `_shape_freedom` shape-amplitude decay. It **empirically validates** the literature correction: Λ(t) must be applied at **constant relative amplitude across all horizons** (Benth's seasonality function is not smoothed/decayed; horizon attenuation belongs to volatility/uncertainty bands — Samuelson — not the central shape).

### 3c. Comparison to the two hard-coded CH tables

- `assembler._SEASONAL_RATIOS_CH` (W/S 1.28) — **too flat** vs market; understates the premium even where applied (annual-only timestamps).
- `monthly_curve_priors.DEFAULT_CH_STRUCTURAL_MONTHLY_RATIOS` (W/S 1.40) — **order-of-magnitude consistent** with DE h+2 (1.42); a defensible starting point, but currently `UNSUPPORTED`/off in the proof and hard-coded rather than data-derived.

---

## 4. Λ(t) sizing — the three evidence types (as requested)

| Evidence type | Status | Value |
|---|---|---|
| **Forward-market** | **measured (this report)** | CH partial W/S≈1.7 (crisis-inflated); DE full W/S 1.30 (h+1)→1.42 (h+2), **persists far-horizon** |
| **Spot climatological** | **UNSUPPORTED in sandbox** (no spot parquet; lives on fileshare) | to compute: CH EPEX monthly mean / annual mean over 10+ y, robust (down-weight 2021-23 crisis), mean-1 |
| **Unsupported zones** | CH forward months at h≥2 = none; Jul–Dec CH forward = sparse | far-horizon CH shape must come from spot climatology + DE/panel shape, **at constant amplitude** |

**Order-of-magnitude for Λ(t) (CH, relative, mean-1):** winter (Dec–Feb) ≈ **1.15–1.25**, summer (Jun–Aug) ≈ **0.80–0.88**, i.e. **W/S ≈ 1.4–1.5** — between the (too-flat) assembler table and the (crisis-inflated) raw CH forward sample, and consistent with the DE far-horizon witness. **Applied at constant relative amplitude at every delivery year** (no horizon decay). Exact monthly vector to be fixed by combining: robust CH spot climatology (fileshare) + CH/DE forward months where quoted, normalised mean-1, crisis-down-weighted (the existing `robust_seasonal_ratios` machinery is the right tool).

---

## 5. Conclusions & gate

1. **Delivered chart = Path C (export → build_powerbi_exports → Duck table).** The monthly solver IS in the delivered curve; the far-horizon flatness = missing Λ(t).
2. **Pipeline must be unified** (A/B/C diverge; B = legacy cron; solver OFF by default on A). This is the prerequisite remediation (F1) before any flag-ON.
3. **Λ(t) is needed and the market supports a constant-amplitude winter premium of W/S ≈ 1.4–1.5** out to at least h+2 (DE witness); the hard-coded assembler table understates it; `_shape_freedom`'s far-horizon shape decay is contradicted by the forward market.
4. Spot-climatology confirmation requires the fileshare EPEX history (not in sandbox).

**No implementation performed. No flag flipped. No production output generated.**
Next actions require a new explicit go: (a) unify the delivered pipeline; (b) implement data-derived
Λ(t) at constant relative amplitude; (c) neutralise far-horizon shape decay; (d) fix the prompt
repricing break (`_rebalance_near_term_bridge`). See the global hermeneutic audit for the ordered path.
