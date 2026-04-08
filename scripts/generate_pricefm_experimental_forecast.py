#!/usr/bin/env python3
"""
Generate a future CH PriceFM forecast using the best local experimental setup.

The script:
1. rebuilds the full CH-centric dataset from the latest parquets,
2. creates simple operational exogenous forecasts for load/solar/wind,
3. recursively projects neighbor prices with the Phase I model,
4. projects CH prices with the best Phase II model,
5. saves a timestamp-aligned forecast usable by the experimental blend path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfc_shaping.model.pricefm_experimental import BEST_PRICEFM_EXPERIMENT
from scripts.export_pricefm_ch_dataset import COUNTRY_SPECS, build_dataset


PRICEFM_REPO = ROOT / "__tmp_pricefm"
DEFAULT_QH_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_forecast_latest_15min.csv"
DEFAULT_HOURLY_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_forecast_latest.csv"
DEFAULT_META_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_forecast_latest_meta.json"

TRAIN_START = "2023-01-01"
TRAIN_END = "2025-06-30"
VAL_START = "2025-07-01"
VAL_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-03-25"

COUNTRIES = list(BEST_PRICEFM_EXPERIMENT.dataset_countries)
TARGET_COUNTRY = "CH"
LOCAL_TZ = "Europe/Zurich"
FEATURES = ["load", "solar", "wind"]
LAG_STEPS = 96
LEAD_STEPS = 96
QUANTILES = [0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90]


def _discover_countries(df: pd.DataFrame) -> list[str]:
    countries = sorted({col.split("-")[0] for col in df.columns if col != "time_utc"})
    valid = []
    for country in countries:
        required = {f"{country}-price", f"{country}-load", f"{country}-solar", f"{country}-wind"}
        if required.issubset(df.columns):
            valid.append(country)
    return valid


def _build_ch_adjacency(input_countries: list[str]) -> dict[str, list[str]]:
    base = {
        "CH": ["CH", "AT", "DE_LU", "FR", "IT_NORD"],
        "AT": ["AT", "CH", "DE_LU", "IT_NORD", "SI"],
        "DE_LU": ["AT", "BE", "CH", "CZ", "DK_1", "DK_2", "DE_LU", "FR", "NL", "NO_2", "PL", "SE_4"],
        "FR": ["BE", "CH", "DE_LU", "ES", "FR", "IT_NORD"],
        "IT_NORD": ["AT", "CH", "FR", "IT_CNOR", "IT_NORD", "SI"],
    }
    allowed = set(input_countries)
    adjacency: dict[str, list[str]] = {}
    for node, neighbors in base.items():
        if node not in allowed:
            continue
        adjacency[node] = [nb for nb in neighbors if nb in allowed]
        if node not in adjacency[node]:
            adjacency[node] = [node] + adjacency[node]
    return adjacency


def _weighted_analog_value(
    series: pd.Series,
    ts: pd.Timestamp,
    local_ts: pd.Timestamp,
) -> float:
    offsets = [1, 2, 7, 14, 21, 28]
    weights = [0.32, 0.18, 0.24, 0.12, 0.08, 0.06]
    values: list[float] = []
    value_weights: list[float] = []
    for days, weight in zip(offsets, weights):
        src = ts - pd.Timedelta(days=days)
        if src in series.index:
            val = series.at[src]
            if pd.notna(val):
                values.append(float(val))
                value_weights.append(weight)

    if values:
        return float(np.average(values, weights=value_weights))

    history = series.dropna()
    if history.empty:
        return 0.0

    hist_local = history.index.tz_convert(LOCAL_TZ)
    slot_mask = (
        (hist_local.dayofweek == local_ts.dayofweek)
        & (hist_local.hour == local_ts.hour)
        & (hist_local.minute == local_ts.minute)
    )
    slot_values = history.loc[slot_mask]
    if not slot_values.empty:
        return float(slot_values.tail(12).median())

    return float(history.tail(96 * 28).median())


def _forecast_exogenous(history: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=future_index)
    local_index = future_index.tz_convert(LOCAL_TZ)
    for country in COUNTRIES:
        for feature in FEATURES:
            col = f"{country}-{feature}"
            if col not in history.columns:
                raise KeyError(f"Missing required exogenous column: {col}")
            series = history[col].copy()
            values = [
                _weighted_analog_value(series, ts, local_ts)
                for ts, local_ts in zip(future_index, local_index)
            ]
            pred = pd.Series(values, index=future_index, name=col)
            if feature in {"solar", "wind", "load"}:
                pred = pred.clip(lower=0.0)
            out[col] = pred
    return out


def _load_actual_price_panel(index: pd.DatetimeIndex) -> pd.DataFrame:
    panel = pd.DataFrame(index=index)
    for country in COUNTRIES:
        spec = COUNTRY_SPECS[country]
        price_df = pd.read_parquet(spec["price_path"])
        price_series = price_df[spec["price_col"]].copy()
        if price_series.index.tz is None:
            price_series.index = price_series.index.tz_localize("UTC")
        panel[f"{country}-price"] = price_series.reindex(index)
    return panel


def _make_gate_vec(active_countries: list[str], input_countries: list[str]) -> np.ndarray:
    active = set(active_countries)
    return np.array([[1.0 if c in active else 0.0 for c in input_countries]], dtype="float32")


def _scaled_lag_window(
    state_df: pd.DataFrame,
    country: str,
    anchor_utc: pd.Timestamp,
    x_scaler,
    y_scaler,
) -> np.ndarray:
    lag_index = pd.date_range(anchor_utc - pd.Timedelta(minutes=15 * LAG_STEPS), periods=LAG_STEPS, freq="15min", tz="UTC")
    raw = state_df.loc[lag_index, [f"{country}-price", f"{country}-load", f"{country}-solar", f"{country}-wind"]].copy()
    scaled = np.empty((LAG_STEPS, 4), dtype="float32")
    scaled[:, 0] = y_scaler.transform(raw[[f"{country}-price"]]).reshape(-1)
    scaled[:, 1:] = x_scaler.transform(raw[[f"{country}-load", f"{country}-solar", f"{country}-wind"]]).astype("float32")
    return scaled


def _scaled_lead_window(
    state_df: pd.DataFrame,
    country: str,
    anchor_utc: pd.Timestamp,
    x_scaler,
) -> np.ndarray:
    lead_index = pd.date_range(anchor_utc, periods=LEAD_STEPS, freq="15min", tz="UTC")
    raw = state_df.loc[lead_index, [f"{country}-load", f"{country}-solar", f"{country}-wind"]]
    return x_scaler.transform(raw).astype("float32")


def _inverse_price(pred_scaled: np.ndarray, y_scaler) -> np.ndarray:
    mid_idx = int(np.argmin(np.abs(np.array(QUANTILES) - 0.5)))
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    return pred[0, :, mid_idx]


def _load_models_and_scalers(dataset_path: Path):
    import tensorflow as tf  # noqa: F401

    sys.path.insert(0, str(PRICEFM_REPO))
    from PriceFM import read_dataset, split_dataframe, scale_dataframe_per_country
    from PriceFM.data import get_k_hop_countries
    from PriceFM.model import load_model

    df = read_dataset(str(dataset_path))
    countries = _discover_countries(df)
    df_train, df_val, df_test = split_dataframe(
        df,
        TRAIN_START,
        TRAIN_END,
        VAL_START,
        VAL_END,
        TEST_START,
        TEST_END,
    )
    _, _, _, x_scalers, y_scalers = scale_dataframe_per_country(
        df_train, df_val, df_test, countries, FEATURES, "price"
    )
    adjacency = _build_ch_adjacency(countries)
    active_nodes = get_k_hop_countries(adjacency, countries, TARGET_COUNTRY, BEST_PRICEFM_EXPERIMENT.graph_degree)

    phase1 = load_model(str(PRICEFM_REPO / "Model" / "PhaseI_CH.keras"))
    phase2 = load_model(str(PRICEFM_REPO / "Model" / f"phase2_best_CH_deg{BEST_PRICEFM_EXPERIMENT.graph_degree}.keras"))
    return countries, x_scalers, y_scalers, phase1, phase2, active_nodes


def _predict_country_day(
    model,
    state_df: pd.DataFrame,
    countries: list[str],
    target_country: str,
    anchor_utc: pd.Timestamp,
    x_scalers,
    y_scalers,
    active_nodes: list[str],
) -> np.ndarray:
    x_lag_all = np.stack(
        [_scaled_lag_window(state_df, country, anchor_utc, x_scalers[country], y_scalers[country]) for country in countries],
        axis=0,
    )[None, ...]
    x_lead_all = np.stack(
        [_scaled_lead_window(state_df, country, anchor_utc, x_scalers[country]) for country in countries],
        axis=0,
    )[None, ...]
    gate = _make_gate_vec(active_nodes, countries)
    pred_scaled = model.predict(
        {"X_lag_all": x_lag_all, "X_lead_all": x_lead_all, "graph_gate": gate},
        verbose=0,
    )
    return _inverse_price(pred_scaled, y_scalers[target_country])


def generate_forecast(
    horizon_days: int,
    dataset_path: Path,
    qh_output: Path,
    hourly_output: Path,
    meta_output: Path,
) -> dict[str, object]:
    dataset, coverage = build_dataset(COUNTRIES, min_coverage=0.95)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)

    state_df = dataset.set_index("time_utc").sort_index()
    state_df.index = pd.to_datetime(state_df.index, utc=True)

    latest_common_ts = state_df.index.max()
    price_ends = []
    for country in COUNTRIES:
        spec = COUNTRY_SPECS[country]
        price_df = pd.read_parquet(spec["price_path"])
        price_index = price_df.index
        if price_index.tz is None:
            price_index = price_index.tz_localize("UTC")
        price_ends.append(price_index.max())
    latest_price_common_ts = min(price_ends)

    if latest_price_common_ts > latest_common_ts:
        bridge_index = pd.date_range(
            latest_common_ts + pd.Timedelta(minutes=15),
            latest_price_common_ts,
            freq="15min",
            tz="UTC",
        )
        bridge = _forecast_exogenous(state_df, bridge_index)
        bridge_prices = _load_actual_price_panel(bridge_index)
        bridge = pd.concat([bridge_prices, bridge], axis=1)
        state_df = pd.concat([state_df, bridge], axis=0)
        state_df = state_df[~state_df.index.duplicated(keep="last")].sort_index()

    countries, x_scalers, y_scalers, phase1, phase2, active_nodes = _load_models_and_scalers(dataset_path)

    last_ts = state_df.index.max()
    first_anchor_utc = last_ts.floor("D") + pd.Timedelta(days=1)
    anchor_utc_days = pd.date_range(first_anchor_utc, periods=horizon_days, freq="D", tz="UTC")

    future_qh_index = pd.date_range(anchor_utc_days[0], periods=horizon_days * LEAD_STEPS, freq="15min", tz="UTC")
    exog_future = _forecast_exogenous(state_df, future_qh_index)
    for price_col in [f"{country}-price" for country in countries]:
        exog_future[price_col] = np.nan
    state_df = pd.concat([state_df, exog_future], axis=0)
    state_df = state_df[~state_df.index.duplicated(keep="last")].sort_index()

    forecast_rows: list[dict[str, object]] = []
    for day_idx, anchor_utc in enumerate(anchor_utc_days, start=1):
        day_index = pd.date_range(anchor_utc, periods=LEAD_STEPS, freq="15min", tz="UTC")

        neighbor_predictions: dict[str, np.ndarray] = {}
        for country in countries:
            if country == TARGET_COUNTRY:
                continue
            neighbor_predictions[country] = _predict_country_day(
                phase1,
                state_df,
                countries,
                country,
                anchor_utc,
                x_scalers,
                y_scalers,
                [country],
            )

        ch_pred = _predict_country_day(
            phase2,
            state_df,
            countries,
            TARGET_COUNTRY,
            anchor_utc,
            x_scalers,
            y_scalers,
            active_nodes,
        )

        state_df.loc[day_index, "CH-price"] = ch_pred
        for country, values in neighbor_predictions.items():
            state_df.loc[day_index, f"{country}-price"] = values

        for ts, value in zip(day_index, ch_pred):
            forecast_rows.append(
                {
                    "timestamp": ts,
                    "pricefm": float(value),
                    "days_ahead": day_idx,
                }
            )

    qh = pd.DataFrame(forecast_rows).sort_values("timestamp").reset_index(drop=True)
    hourly = (
        qh.assign(timestamp=qh["timestamp"].dt.floor("h"))
        .groupby("timestamp", as_index=False)
        .agg(pricefm=("pricefm", "mean"), days_ahead=("days_ahead", "min"))
    )
    hourly["date"] = hourly["timestamp"].dt.tz_convert(LOCAL_TZ).dt.date.astype(str)
    hourly["hour"] = hourly["timestamp"].dt.tz_convert(LOCAL_TZ).dt.hour

    qh_output.parent.mkdir(parents=True, exist_ok=True)
    qh.to_csv(qh_output, index=False)
    hourly.to_csv(hourly_output, index=False)

    dated_hourly = hourly_output.with_name(f"pricefm_forecast_{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}.csv")
    dated_qh = qh_output.with_name(f"pricefm_forecast_{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}_15min.csv")
    hourly.to_csv(dated_hourly, index=False)
    qh.to_csv(dated_qh, index=False)

    payload = {
        "dataset_path": str(dataset_path),
        "countries": countries,
        "coverage": coverage,
        "history_end_utc": str(last_ts),
        "forecast_start_utc": str(hourly["timestamp"].min()),
        "forecast_end_utc": str(hourly["timestamp"].max()),
        "horizon_days": horizon_days,
        "graph_degree": BEST_PRICEFM_EXPERIMENT.graph_degree,
        "epochs": BEST_PRICEFM_EXPERIMENT.epochs,
        "hourly_output": str(hourly_output),
        "qh_output": str(qh_output),
    }
    meta_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a future CH PriceFM forecast.")
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--dataset-path", type=Path, default=ROOT / BEST_PRICEFM_EXPERIMENT.dataset_path)
    parser.add_argument("--qh-output", type=Path, default=DEFAULT_QH_OUTPUT)
    parser.add_argument("--hourly-output", type=Path, default=ROOT / BEST_PRICEFM_EXPERIMENT.forecast_latest_path)
    parser.add_argument("--meta-output", type=Path, default=DEFAULT_META_OUTPUT)
    args = parser.parse_args()

    payload = generate_forecast(
        horizon_days=args.horizon_days,
        dataset_path=args.dataset_path.resolve(),
        qh_output=args.qh_output.resolve(),
        hourly_output=args.hourly_output.resolve(),
        meta_output=args.meta_output.resolve(),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
