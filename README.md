# PFC

Swiss power price forecasting and price forward curve platform for FMV.

This repository implements an end-to-end stack to:

- forecast Swiss EPEX spot prices from `D+1` to `D+10`,
- construct an arbitrage-free Swiss Price Forward Curve from `M+1` to `Y+3`,
- blend structural market information, hydro signals, and modern forecasting models,
- expose outputs and diagnostics through a web dashboard.

The project is designed for production use in a Swiss utility context: hydro-sensitive, cross-border, and forward-market constrained.

## Executive Summary

PFC is an internal FMV market analytics platform for Swiss power.

It combines:

- short-term spot forecasting,
- long-term forward-curve construction,
- arbitrage-free calibration,
- hydro-aware structural modeling,
- and a web dashboard for diagnostics and operational transparency.

The objective is not only predictive accuracy, but decision-grade market curves that remain useful for trading, risk, valuation, and hydro optimization.

## Quick Start

Main production run:

```powershell
$env:PATH='C:\Users\jbattaglia\.conda\ppa_env;C:\Users\jbattaglia\.conda\ppa_env\Library\bin;C:\Users\jbattaglia\.conda\ppa_env\Scripts;' + $env:PATH
C:\Users\jbattaglia\.conda\ppa_env\python.exe run_pfc_production.py
```

Short-term evaluation harness:

```powershell
$env:PATH='C:\Users\jbattaglia\.conda\ppa_env;C:\Users\jbattaglia\.conda\ppa_env\Library\bin;C:\Users\jbattaglia\.conda\ppa_env\Scripts;' + $env:PATH
C:\Users\jbattaglia\.conda\ppa_env\python.exe autoresearch_eval_lear.py
```

## Why This Project Exists

Swiss power is not a purely local market. A credible forecast must reflect:

- Swiss hydro economics,
- coupling with `DE`, `FR`, `AT`, and `IT`,
- short-term market regimes and spikes,
- long-term forward consistency across `Cal / Quarter / Month`,
- physically and financially coherent shaping from hourly to 15-minute granularity.

This repository addresses that full problem, not only point forecasting.

## What The Platform Does

### 1. Short-Term Forecasting

The short-term engine produces hourly Swiss forecasts on `D+1..D+10` and supports:

- LEAR-style high-dimensional linear regression,
- LightGBM hour-wise nonlinear correction,
- Chronos-2 foundation model integration and fine-tuning,
- PriceFM graph-aware experimental forecasts,
- experimental meta-blending via a FutureBoost-style layer,
- probabilistic outputs and backtesting.

### 2. Long-Term PFC Construction

The long-term engine builds a Swiss 15-minute PFC using:

- forward contract cascading,
- structural shape factors,
- hydro water value correction,
- intraday shaping,
- smooth base-curve interpolation,
- arbitrage-free calibration against tradable forward products.

### 3. Product Layer

The repository also includes:

- a Streamlit dashboard for diagnostics and exploration,
- local DuckDB persistence for forecast artifacts,
- production scripts for daily runs,
- research scripts to benchmark experimental models.

## Architecture

At a high level, the curve is built as:

`P(t) = B(t) × f_S(t) × f_W(t) × f_H(t) × f_Q(t) × f_WV(t)`

Where:

- `B(t)` is the base forward level implied by cascaded contracts,
- `f_S(t)` is seasonal structure,
- `f_W(t)` is day-of-week structure,
- `f_H(t)` is the hourly shape,
- `f_Q(t)` is the 15-minute intrahour factor,
- `f_WV(t)` is the Swiss hydro water-value correction.

The resulting raw curve is then calibrated to be arbitrage-free with respect to annual, quarterly, monthly and peak products.

## Repository Structure

```text
dashboard/                         Streamlit app
docs/                              project notes and supporting docs
models/                            local model checkpoints (e.g. Chronos-2)
pfc_shaping/
  calibration/
    arbitrage_free.py              arbitrage-free calibration
    cascading.py                   forward cascading Cal/Q/M
  data/
    ingest_epex.py                 Swiss and neighboring spot ingestion
    ingest_entso.py                ENTSO-E features, borders, system data
    ingest_energy_charts.py        Energy Charts prices and power data
    ingest_forwards.py             forward ingestion
    ingest_hydro.py                hydro data and water value inputs
    ingest_outages.py              outages and availability
    ingest_smard.py                SMARD fallback/system data
  model/
    assembler.py                   structural PFC assembly
    foundation_forecaster.py       Chronos-based forecasting integration
    futureboost_experimental.py    experimental meta-layer
    lear_forecaster.py             core short-term LEAR forecaster
    msfc_spline.py                 smooth base curve
    pricefm_experimental.py        experimental PriceFM blend logic
    shape_hourly.py                hourly shape factor
    shape_hourly_mlp.py            neural hourly shape alternative
    shape_intraday.py              15-minute shaping
    uncertainty.py                 uncertainty estimation
    water_value.py                 hydro water value correction
  pipeline/
    autoresearch.py                automated search/evolution loop
scripts/
  finetune_chronos2.py             Chronos-2 LoRA fine-tuning
  generate_pricefm_experimental_forecast.py
  run_daily.py                     scheduled orchestration
run_pfc_production.py              end-to-end production run
autoresearch_eval_lear.py          fixed evaluation harness for CT model work
```

## Modeling Philosophy

This codebase is deliberately hybrid.

We do not assume that a single model class is sufficient for Swiss power. Instead:

- linear models provide robustness and interpretability,
- gradient boosting captures regime nonlinearity,
- foundation models capture richer temporal patterns,
- graph-aware models encode cross-border market structure,
- structural long-term shaping enforces economic consistency,
- calibration enforces tradability.

This is the right bias for a hydro utility: strong inductive structure first, machine learning where it adds measurable value.

## Short-Term Stack

### LEAR Core

The core short-term model lives in [`pfc_shaping/model/lear_forecaster.py`](pfc_shaping/model/lear_forecaster.py).

It combines:

- lagged Swiss prices,
- neighboring price information,
- weather and load proxies,
- border and system features,
- calendar effects,
- model averaging,
- short-term overlay logic into the PFC.

### Foundation Models

The repository supports Chronos-2 through [`pfc_shaping/model/foundation_forecaster.py`](pfc_shaping/model/foundation_forecaster.py) and LoRA fine-tuning through [`scripts/finetune_chronos2.py`](scripts/finetune_chronos2.py).

Chronos is used as a modern temporal foundation model, not as a replacement for market structure.

Current Swiss CT policy:

- production baseline is `LEAR + DE`
- foundation usage is opt-in, not default
- enable it explicitly with `PFC_CT_ENABLE_FOUNDATION=1`
- see [`docs/research/ct_foundation_status_2026-05.md`](docs/research/ct_foundation_status_2026-05.md)

### PriceFM

PriceFM is integrated experimentally as a graph-aware model specialized for interconnected power systems.

In this project, it is used as:

- a structural expert for the Swiss-Alpine neighborhood,
- a complementary forecast source,
- an input to experimental blends rather than the sole production model.

This is a better fit for Switzerland than a generic global TSFM-only approach, because topology matters.

### FutureBoost-Style Meta Layer

The repository also includes an experimental meta-layer inspired by recent research on using foundation-model forecasts as features for downstream learners.

Implemented in [`pfc_shaping/model/futureboost_experimental.py`](pfc_shaping/model/futureboost_experimental.py), it learns when and how to trust:

- LEAR,
- PriceFM,
- regime-aware signals,
- disagreement dynamics between models.

This is currently experimental and opt-in.

## Long-Term PFC Stack

### Base Curve

Forward prices are ingested and cascaded to consistent `Cal`, `Quarter`, and `Month` levels using [`pfc_shaping/calibration/cascading.py`](pfc_shaping/calibration/cascading.py).

### Structural Shaping

The long-term curve uses:

- hourly structural patterns from [`shape_hourly.py`](pfc_shaping/model/shape_hourly.py),
- 15-minute refinement from [`shape_intraday.py`](pfc_shaping/model/shape_intraday.py),
- Swiss hydro-specific correction from [`water_value.py`](pfc_shaping/model/water_value.py),
- smooth long-horizon governance in [`assembler.py`](pfc_shaping/model/assembler.py).

### Arbitrage-Free Calibration

The final PFC is calibrated with [`pfc_shaping/calibration/arbitrage_free.py`](pfc_shaping/calibration/arbitrage_free.py) so that the curve remains consistent with tradable forward contracts.

This is a hard constraint, not a cosmetic post-process.

## Dashboard

The Streamlit dashboard under [`dashboard/`](dashboard/) is not a side artifact. It is part of the product.

It provides:

- short-term forecast visualization,
- profile and regime diagnostics,
- backtest views,
- model comparison,
- experimental overlay documentation,
- operational transparency for research vs production behavior.

For a model-heavy analytics product, this approach is materially faster and more expressive than rebuilding the same logic in a BI tool.

## Data Sources

The platform is built around a mix of public institutional data, market data, and FMV internal data.

### Public and Semi-Public Sources

#### ENTSO-E Transparency

Used through [`pfc_shaping/data/ingest_entso.py`](pfc_shaping/data/ingest_entso.py).

Main uses:

- Swiss load, solar, wind,
- neighboring system signals for `DE_LU`, `FR`, `AT`, `IT_NORD`,
- Swiss border schedules and flows,
- NTC-style cross-border features,
- system-derived short-term explanatory variables.

This is a core source for operational market fundamentals.

#### Energy Charts / Fraunhofer ISE

Used through [`pfc_shaping/data/ingest_energy_charts.py`](pfc_shaping/data/ingest_energy_charts.py).

Main uses:

- Swiss day-ahead prices,
- German day-ahead prices,
- public power series,
- unified fallback data source for price and system information.

This source is especially useful because it is public, stable, and convenient for production-oriented ingestion.

#### SMARD

Used through [`pfc_shaping/data/ingest_smard.py`](pfc_shaping/data/ingest_smard.py).

Main uses:

- German fallback price data,
- German system-level signals when a dedicated public fallback is preferable.

SMARD is relevant mainly for German market context and redundancy.

#### Calendar / Holiday Data

Used through [`pfc_shaping/data/calendar_ch.py`](pfc_shaping/data/calendar_ch.py).

Main uses:

- Swiss holidays,
- German holidays,
- calendar classification of weekdays, Saturdays, Sundays, and market-sensitive holidays.

This is critical because Swiss power prices react strongly to German holiday effects.

#### Yahoo Finance / Public Commodity Proxy Layer

Used today in the dashboard and in short-term exogenous features as a pragmatic public-market proxy layer.

Relevant code:

- [`dashboard/utils.py`](dashboard/utils.py)
- [`pfc_shaping/model/lear_forecaster.py`](pfc_shaping/model/lear_forecaster.py)

Main instruments currently used:

- `TTF` gas,
- `Brent`,
- `CO2 EUA` proxy.

Main uses:

- commodity visualization in the dashboard,
- short-term exogenous signals,
- fuel-stack proxy construction in the LEAR model.

Important modeling note:

- `TTF` and `Brent` from Yahoo-style public feeds are acceptable research and monitoring proxies,
- `CO2 EUA` is currently proxied, not ingested from the canonical ICE EUA futures chain,
- therefore this layer is operationally useful, but not yet the most institutional implementation possible.

State-of-the-art target for FMV is:

- governed commodity curves served from Databricks Gold,
- sourced from approved market data feeds or internal validated market snapshots,
- with explicit contract mapping, roll logic, and currency/unit normalization.

In other words, the current setup is a practical public proxy layer; the target setup is an institutional commodity market data layer.

### Internal / FMV Sources

#### EEX Forward Data

Relevant code:

- [`pfc_shaping/data/ingest_forwards.py`](pfc_shaping/data/ingest_forwards.py)
- [`pfc_shaping/data/forward_proxy.py`](pfc_shaping/data/forward_proxy.py)

Main uses:

- loading annual, quarterly, and monthly forward prices,
- building the structural base curve,
- anchoring the PFC to tradable products.

This includes internal EEX reporting and FMV forward reference files.

#### Hydro / Reservoir Data

Relevant code:

- [`pfc_shaping/data/ingest_hydro.py`](pfc_shaping/data/ingest_hydro.py)
- [`pfc_shaping/model/water_value.py`](pfc_shaping/model/water_value.py)

Main uses:

- Swiss hydro fill,
- reservoir state,
- water-value correction,
- hydro-sensitive shaping of the Swiss forward curve.

This is one of the most structurally important Swiss-specific inputs in the repository.

#### Outages and Availability

Relevant code:

- [`pfc_shaping/data/ingest_outages.py`](pfc_shaping/data/ingest_outages.py)

Main uses:

- Swiss and neighboring availability constraints,
- short-term forecasting features,
- structural shaping support where capacity availability matters.

### Data Strategy

The target architecture is to industrialize these datasets in Databricks Gold and let the Python modeling and web application layer consume curated tables from there.

In other words:

- Databricks Gold should become the governed source of truth,
- Python remains the modeling, calibration, forecasting, and application layer.

This is the right split for FMV: governed data upstream, fast analytical product development downstream.

## Production Run

The main production entry point is:

```powershell
C:\Users\jbattaglia\.conda\ppa_env\python.exe run_pfc_production.py
```

This pipeline:

1. loads and updates data,
2. fits long-term components,
3. builds the structural PFC,
4. runs the short-term LEAR forecast,
5. optionally applies experimental PriceFM / FutureBoost overlays,
6. exports artifacts to `pfc_shaping/output`.

## Experimental Modes

The short-term production path supports three experimental overlay modes:

- `fixed`
- `regime`
- `futureboost`

Useful environment variables:

```powershell
$env:PFC_ENABLE_PRICEFM_EXPERIMENT='1'
$env:PFC_APPLY_PRICEFM_EXPERIMENT_TO_PFC='1'
$env:PFC_GENERATE_PRICEFM_EXPERIMENT='1'
$env:PFC_PRICEFM_PYTHON='C:\Users\jbattaglia\.conda\pricefm_tf\python.exe'
$env:PFC_PRICEFM_BLEND_MODE='futureboost'
$env:PFC_LEAR_BACKTEST_MODE='skip'
```

These modes are intentionally opt-in. Production should remain robust even if the experimental path is disabled.

## Evaluation

Short-term research is benchmarked through a fixed harness:

```powershell
C:\Users\jbattaglia\.conda\ppa_env\python.exe autoresearch_eval_lear.py
```

The project uses a composite score combining:

- MAE,
- RMSE,
- MAPE,
- correlation.

This is the correct philosophy for Swiss hydro dispatch and trading: pure MAE is not enough. Profile quality matters.

## What Makes This Repository Different

This is not a generic forecasting notebook collection.

It is a full-stack market modeling platform with:

- structural power-market knowledge,
- short-term ML and foundation models,
- graph-aware experimental modeling,
- forward-market consistency,
- calibration,
- web-based diagnostics,
- production orchestration.

For Swiss power, that combination is more defensible than a pure data-science stack or a pure structural stack taken alone.

## Roadmap

The next major improvement axes are:

- tighter CT-to-PFC overlay control in the prompt horizon,
- stronger operational use of Databricks Gold datasets,
- richer exogenous European regime signals,
- continued validation of PriceFM and FutureBoost-style meta-learning,
- broader live benchmarking across regimes and stress periods.

## Environment

Primary environment:

```powershell
C:\Users\jbattaglia\.conda\ppa_env\python.exe
```

Recommended PowerShell PATH prelude:

```powershell
$env:PATH='C:\Users\jbattaglia\.conda\ppa_env;C:\Users\jbattaglia\.conda\ppa_env\Library\bin;C:\Users\jbattaglia\.conda\ppa_env\Scripts;' + $env:PATH
```

Experimental PriceFM / TensorFlow environment:

```powershell
C:\Users\jbattaglia\.conda\pricefm_tf\python.exe
```

## License / Usage

Internal FMV project. Adapt licensing and publication policy before external distribution.
