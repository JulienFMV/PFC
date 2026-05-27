#!/usr/bin/env python3
"""Autopsy diagnostics for Swiss CT governed features.

Focus:
- horizon-by-horizon governed health and gating reasons
- per-hour governed feature availability and simple predictive utility
- sparse selection rate of governed features under a stronger L1 regime
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.lear_forecaster import LEARForecaster  # noqa: E402


DATA_FILES = {
    "epex_ch": ROOT / "pfc_shaping" / "data" / "epex_15min.parquet",
    "epex_de": ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet",
    "entso": ROOT / "pfc_shaping" / "data" / "entso_fundamentals_15min.parquet",
    "hydro": ROOT / "pfc_shaping" / "data" / "hydro_reservoir.parquet",
    "outages": ROOT / "pfc_shaping" / "data" / "outages_15min.parquet",
    "commodities": ROOT / "data" / "commodities_cache.parquet",
    "de_renewable_forecast": ROOT / "pfc_shaping" / "data" / "de_renewable_forecast.parquet",
    "multi_country_forecast": ROOT / "pfc_shaping" / "data" / "multi_country_forecast_15min.parquet",
    "fr_nuclear_forecast": ROOT / "pfc_shaping" / "data" / "fr_nuclear_forecast_15min.parquet",
    "weather_forecast": ROOT / "pfc_shaping" / "data" / "weather_forecast_hourly.parquet",
}

OUTPUT_PATH = ROOT / "pfc_shaping" / "output" / "ct_governed_autopsy_latest.json"


def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return None if df.empty else df


def _load_inputs() -> dict[str, pd.DataFrame | None]:
    return {
        "epex_ch": pd.read_parquet(DATA_FILES["epex_ch"]),
        "epex_de": pd.read_parquet(DATA_FILES["epex_de"]),
        "entso": pd.read_parquet(DATA_FILES["entso"]),
        "hydro": pd.read_parquet(DATA_FILES["hydro"]),
        "outages": _read_optional_parquet(DATA_FILES["outages"]),
        "commodities": _read_optional_parquet(DATA_FILES["commodities"]),
        "de_renewable_forecast": _read_optional_parquet(DATA_FILES["de_renewable_forecast"]),
        "multi_country_forecast": _read_optional_parquet(DATA_FILES["multi_country_forecast"]),
        "fr_nuclear_forecast": _read_optional_parquet(DATA_FILES["fr_nuclear_forecast"]),
        "weather_forecast": _read_optional_parquet(DATA_FILES["weather_forecast"]),
    }


def _fit_governed_model(inputs: dict[str, pd.DataFrame | None]) -> LEARForecaster:
    model = LEARForecaster(use_foundation_model=False, use_governed_forecast_features=True)
    model.fit(
        epex_15min=inputs["epex_ch"],
        entso_15min=inputs["entso"],
        outages_15min=inputs["outages"],
        commodities=inputs["commodities"],
        hydro=inputs["hydro"],
        epex_de_15min=inputs["epex_de"],
        de_renewable_forecast=inputs["de_renewable_forecast"],
        multi_country_forecast=inputs["multi_country_forecast"],
        fr_nuclear_forecast=inputs["fr_nuclear_forecast"],
        weather_forecast=inputs["weather_forecast"],
    )
    return model


def _gate_reason_payload(model: LEARForecaster, horizon: int, min_coverage_ratio: float) -> dict[str, object]:
    health = model.assess_governed_forecast_feature_health(
        horizon_days=horizon,
        min_coverage_ratio=min_coverage_ratio,
    )
    expected_sources = list(model.GOVERNED_FORECAST_COLUMNS) + list(model.GOVERNED_WEATHER_COLUMNS)
    retained_sources = [k for k, v in health.get("source_coverage", {}).items() if float(v) >= min_coverage_ratio]
    retained_core = [k for k in retained_sources if k in model.GOVERNED_FORECAST_CORE_COLUMNS]
    retained_weather = [k for k in retained_sources if k in model.GOVERNED_WEATHER_COLUMNS]
    return {
        "horizon": int(horizon),
        "enabled_for_run": bool(health.get("enabled_for_run", False)),
        "coverage_ratio": float(health.get("coverage_ratio", 0.0)),
        "min_coverage_ratio": float(health.get("min_coverage_ratio", min_coverage_ratio)),
        "max_supported_horizon_days": int(health.get("max_supported_horizon_days", 0)),
        "retained_source_count": int(len(retained_sources)),
        "retained_core_count": int(len(retained_core)),
        "retained_weather_count": int(len(retained_weather)),
        "missing_sources": list(health.get("missing_sources", [])),
        "source_coverage": dict(health.get("source_coverage", {})),
        "gate_reasons": {
            "coverage_ratio_below_threshold": float(health.get("coverage_ratio", 0.0)) < float(min_coverage_ratio),
            "missing_core_family": len(retained_core) == 0,
            "missing_weather_family": len(retained_weather) == 0,
            "max_supported_zero": int(health.get("max_supported_horizon_days", 0)) <= 0,
            "retained_source_share": float(len(retained_sources) / len(expected_sources)) if expected_sources else 1.0,
        },
    }


def _safe_corr(x: pd.Series, y: pd.Series) -> float | None:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 30:
        return None
    corr = df["x"].corr(df["y"])
    return None if pd.isna(corr) else float(corr)


def _per_hour_feature_diagnostics(model: LEARForecaster) -> dict[str, object]:
    complete = model._complete_days
    exog = model.exog_
    hours_payload: dict[str, object] = {}
    key_prefixes = (
        "forecast_load_ch_mw",
        "forecast_load_de_mw",
        "forecast_load_fr_mw",
        "forecast_load_it_mw",
        "forecast_solar_de_mw",
        "forecast_wind_de_mw",
        "forecast_fr_nuclear_available_mw",
        "wx_ch_zurich_shortwave_radiation",
    )

    for hour in range(24):
        X, y = model._build_features(model.prices_h_, exog, hour)
        governed_cols = [c for c in X.columns if c.startswith("forecast_") or c.startswith("wx_")]
        direct_cols = [c for c in governed_cols if "_d0_" in c]
        direct_corrs = []
        for col in direct_cols:
            corr = _safe_corr(X[col], y)
            if corr is not None:
                direct_corrs.append({"feature": col, "corr": corr, "abs_corr": abs(corr)})
        direct_corrs = sorted(direct_corrs, key=lambda row: row["abs_corr"], reverse=True)

        sparse_sample = min(len(X), 720)
        selection = {
            "n_features_governed": int(len(governed_cols)),
            "n_selected_governed": 0,
            "selected_governed_features": [],
            "selection_rate": 0.0,
        }
        if len(governed_cols) >= 2 and sparse_sample >= 120:
            X_sub = X.iloc[-sparse_sample:].copy()
            y_sub = y.iloc[-sparse_sample:].astype(float)
            en = make_pipeline(
                StandardScaler(),
                ElasticNetCV(
                    l1_ratio=0.5,
                    n_alphas=50,
                    cv=3,
                    random_state=42,
                    max_iter=10000,
                ),
            )
            en.fit(np.nan_to_num(X_sub.values.astype(float), nan=0.0), y_sub.values)
            coef = en.named_steps["elasticnetcv"].coef_
            selected = [
                col for col, w in zip(X_sub.columns, coef)
                if abs(float(w)) > 1e-10 and (col.startswith("forecast_") or col.startswith("wx_"))
            ]
            selection = {
                "n_features_governed": int(len(governed_cols)),
                "n_selected_governed": int(len(selected)),
                "selected_governed_features": selected[:20],
                "selection_rate": float(len(selected) / len(governed_cols)) if governed_cols else 0.0,
            }

        key_feature_corrs = {}
        for prefix in key_prefixes:
            candidates = [c for c in direct_cols if c.startswith(prefix)]
            if not candidates:
                continue
            best = None
            for col in candidates:
                corr = _safe_corr(X[col], y)
                if corr is None:
                    continue
                payload = {"feature": col, "corr": corr, "abs_corr": abs(corr)}
                if best is None or payload["abs_corr"] > best["abs_corr"]:
                    best = payload
            if best is not None:
                key_feature_corrs[prefix] = best

        hours_payload[f"h{hour:02d}"] = {
            "n_rows": int(len(X)),
            "n_governed_features": int(len(governed_cols)),
            "n_direct_governed_features": int(len(direct_cols)),
            "top_direct_correlations": direct_corrs[:10],
            "key_feature_correlations": key_feature_corrs,
            "sparse_selection": selection,
        }
    return hours_payload


def main() -> None:
    min_cov = float(0.98)
    inputs = _load_inputs()
    model = _fit_governed_model(inputs)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "vintage_schema_verified": bool(
            model._governed_vintage_status
            and all(bool(v.get("vintage_schema_verified", False)) for v in model._governed_vintage_status.values())
        ),
        "gating_by_horizon": {
            f"h{h}": _gate_reason_payload(model, horizon=h, min_coverage_ratio=min_cov)
            for h in range(1, 11)
        },
        "per_hour_feature_diagnostics": _per_hour_feature_diagnostics(model),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
