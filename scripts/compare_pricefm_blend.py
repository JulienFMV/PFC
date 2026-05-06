#!/usr/bin/env python3
"""
Compare a trained PriceFM CH model against the current LEAR backtest and test
simple convex blends on the overlapping period.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PRICEFM_REPO = ROOT / "__tmp_pricefm"
DEFAULT_DATASET = ROOT / "pfc_shaping" / "output" / "pricefm_ch_de_poc.csv"
DEFAULT_QH_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_de_predictions_15min.csv"
DEFAULT_HOURLY_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_de_predictions_hourly.csv"
DEFAULT_BLEND_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_de_blend_eval.json"


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
    adjacency = {}
    for node, neighbors in base.items():
        if node not in allowed:
            continue
        adjacency[node] = [nb for nb in neighbors if nb in allowed]
        if node not in adjacency[node]:
            adjacency[node] = [node] + adjacency[node]
    return adjacency


def _score_frame(df: pd.DataFrame, forecast_col: str, actual_col: str = "actual") -> dict[str, float]:
    err = df[forecast_col] - df[actual_col]
    abs_err = err.abs()
    ape = abs_err / df[actual_col].abs().clip(lower=1) * 100.0
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float(ape.mean())
    corr = float(df[forecast_col].corr(df[actual_col]))
    score = (
        0.35 * (mae / 15.0)
        + 0.30 * (rmse / 22.3)
        + 0.20 * (mape / 30.9)
        + 0.15 * (1.0 - corr)
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "corr": corr,
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PriceFM CH POC vs LEAR and test blends.")
    parser.add_argument("--pricefm-repo", type=Path, default=PRICEFM_REPO)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target-country", default="CH")
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--val-start", default="2025-07-01")
    parser.add_argument("--val-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-03-25")
    parser.add_argument("--graph-degree", type=int, default=1)
    parser.add_argument("--export-only", action="store_true", help="Only export PriceFM predictions, skip LEAR/blend evaluation.")
    parser.add_argument("--qh-output", type=Path, default=DEFAULT_QH_OUTPUT)
    parser.add_argument("--hourly-output", type=Path, default=DEFAULT_HOURLY_OUTPUT)
    parser.add_argument("--blend-output", type=Path, default=DEFAULT_BLEND_OUTPUT)
    args = parser.parse_args()

    args.pricefm_repo = args.pricefm_repo.resolve()
    args.dataset = args.dataset.resolve()
    args.qh_output = args.qh_output.resolve()
    args.hourly_output = args.hourly_output.resolve()
    args.blend_output = args.blend_output.resolve()

    if not args.pricefm_repo.exists():
        raise FileNotFoundError(f"PriceFM repo not found: {args.pricefm_repo}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        raise RuntimeError("TensorFlow is not available in the current environment.") from exc

    os.chdir(args.pricefm_repo)
    sys.path.insert(0, str(args.pricefm_repo))

    from PriceFM import (
        read_dataset,
        split_dataframe,
        scale_dataframe_per_country,
        separate_countries,
        make_rolling_window_samples,
    )
    from PriceFM.data import pack_dataset, get_k_hop_countries
    from PriceFM.evaluation import inverse_scale_y_pred, inverse_scale_y_true
    from PriceFM.model import load_model

    df = read_dataset(str(args.dataset))
    countries = _discover_countries(df)
    if args.target_country not in countries:
        raise ValueError(f"Target country {args.target_country} not present in dataset. Found: {countries}")

    features = ["load", "solar", "wind"]
    label_column = "price"
    lag_features = ["price", "load", "solar", "wind"]
    lead_features = ["load", "solar", "wind"]
    lag_window = 96
    lead_window = 96
    quantiles = [0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90]
    mid_idx = int(np.argmin(np.abs(np.array(quantiles) - 0.5)))

    df_train, df_val, df_test = split_dataframe(
        df,
        args.train_start,
        args.train_end,
        args.val_start,
        args.val_end,
        args.test_start,
        args.test_end,
    )
    df_train_s, df_val_s, df_test_s, _, y_scalers = scale_dataframe_per_country(
        df_train, df_val, df_test, countries, features, label_column
    )

    train_sep = separate_countries(df_train_s, countries, features, label_column)
    val_sep = separate_countries(df_val_s, countries, features, label_column)
    test_sep = separate_countries(df_test_s, countries, features, label_column)

    rolling_train = {}
    rolling_val = {}
    rolling_test = {}
    for country in countries:
        rolling_train[country] = {}
        rolling_val[country] = {}
        rolling_test[country] = {}
        for split_name, source, target in [
            ("train", train_sep, rolling_train),
            ("val", val_sep, rolling_val),
            ("test", test_sep, rolling_test),
        ]:
            x_lag, x_lead, y_arr, anchors = make_rolling_window_samples(
                source[country], country, lag_features, lead_features, label_column, lag_window, lead_window
            )
            target[country] = {"X_lag": x_lag, "X_lead": x_lead, "Y": y_arr, "t": anchors}

    adjacency_dict = _build_ch_adjacency(countries)
    active_nodes = get_k_hop_countries(adjacency_dict, countries, args.target_country, args.graph_degree)
    gate_fn = lambda _: active_nodes
    x_lag_te, x_lead_te, gate_te, y_te = pack_dataset(
        rolling_test, countries, [args.target_country], gate_fn
    )

    model_path = args.pricefm_repo / "Model" / f"phase2_best_{args.target_country}_deg{args.graph_degree}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained PriceFM model not found: {model_path}")

    model = load_model(str(model_path))
    pred_scaled = model.predict(
        {"X_lag_all": x_lag_te, "X_lead_all": x_lead_te, "graph_gate": gate_te},
        batch_size=256,
        verbose=0,
    )

    y_true = inverse_scale_y_true(y_te, y_scalers[args.target_country])
    y_pred = inverse_scale_y_pred(pred_scaled, y_scalers[args.target_country])
    median_pred = y_pred[..., mid_idx]

    freq = pd.infer_freq(df.index[:20]) or "15min"
    step = pd.to_timedelta(freq)
    anchors = rolling_test[args.target_country]["t"]
    qh_rows = []
    for sample_idx, anchor_ts in enumerate(anchors):
        for step_idx in range(median_pred.shape[1]):
            ts = pd.Timestamp(anchor_ts) + step_idx * step
            qh_rows.append(
                {
                    "anchor_ts": pd.Timestamp(anchor_ts),
                    "timestamp": ts,
                    "actual_qh": float(y_true[sample_idx, step_idx]),
                    "pricefm_qh": float(median_pred[sample_idx, step_idx]),
                }
            )

    qh = pd.DataFrame(qh_rows).sort_values("timestamp")
    qh["timestamp"] = pd.to_datetime(qh["timestamp"], utc=True)
    qh["anchor_ts"] = pd.to_datetime(qh["anchor_ts"], utc=True)
    qh["date_utc"] = qh["timestamp"].dt.date.astype(str)
    qh["hour_utc"] = qh["timestamp"].dt.hour
    args.qh_output.parent.mkdir(parents=True, exist_ok=True)
    qh.to_csv(args.qh_output, index=False)

    hourly = (
        qh.assign(timestamp=qh["timestamp"].dt.floor("h"))
        .groupby(["timestamp"], as_index=False)[["actual_qh", "pricefm_qh"]]
        .mean()
        .rename(columns={"actual_qh": "actual", "pricefm_qh": "pricefm"})
    )
    hourly["date_utc"] = hourly["timestamp"].dt.date.astype(str)
    hourly["hour_utc"] = hourly["timestamp"].dt.hour
    hourly.to_csv(args.hourly_output, index=False)

    if args.export_only:
        payload = {
            "dataset": str(args.dataset),
            "countries": countries,
            "graph_degree": args.graph_degree,
            "qh_predictions": str(args.qh_output),
            "hourly_predictions": str(args.hourly_output),
            "n_qh_rows": int(len(qh)),
            "n_hourly_rows": int(len(hourly)),
        }
        args.blend_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"qh_output:{args.qh_output}")
        print(f"hourly_output:{args.hourly_output}")
        print(f"blend_output:{args.blend_output}")
        print(json.dumps(payload, indent=2))
        return

    sys.path.insert(0, str(ROOT))
    from pfc_shaping.ct.model.lear_forecaster import LEARForecaster

    epex_ch = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_15min.parquet")
    epex_de = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet")
    entso = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "entso_15min.parquet")

    lear = LEARForecaster()
    lear.fit(epex_15min=epex_ch, entso_15min=entso, epex_de_15min=epex_de)
    bt = lear.backtest(n_days=30, horizon=1).copy()
    bt["timestamp"] = (
        pd.to_datetime(bt["date"])
        + pd.to_timedelta(bt["hour"], unit="h")
    )
    bt["timestamp"] = bt["timestamp"].dt.tz_localize("Europe/Zurich").dt.tz_convert("UTC")
    bt = bt[["timestamp", "actual", "forecast"]].rename(columns={"forecast": "lear"})

    merged = bt.merge(hourly[["timestamp", "pricefm"]], on="timestamp", how="inner")
    if merged.empty:
        raise ValueError("No timestamp overlap between LEAR backtest and PriceFM predictions.")

    blend_rows = []
    for w in np.linspace(0.0, 1.0, 21):
        col = f"blend_{w:.2f}"
        merged[col] = (1.0 - w) * merged["lear"] + w * merged["pricefm"]
        metrics = _score_frame(merged, col)
        metrics["weight_pricefm"] = float(w)
        blend_rows.append(metrics)

    blend_df = pd.DataFrame(blend_rows).sort_values("score", ascending=True).reset_index(drop=True)
    best = blend_df.iloc[0].to_dict()
    learn_metrics = _score_frame(merged, "lear")
    pricefm_metrics = _score_frame(merged, "pricefm")

    payload = {
        "dataset": str(args.dataset),
        "countries": countries,
        "graph_degree": args.graph_degree,
        "overlap_start": str(merged["timestamp"].min()),
        "overlap_end": str(merged["timestamp"].max()),
        "n_hours_overlap": int(len(merged)),
        "lear": learn_metrics,
        "pricefm": pricefm_metrics,
        "best_blend": best,
        "top5_blends": blend_df.head(5).to_dict(orient="records"),
        "qh_predictions": str(args.qh_output),
        "hourly_predictions": str(args.hourly_output),
    }
    args.blend_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"qh_output:{args.qh_output}")
    print(f"hourly_output:{args.hourly_output}")
    print(f"blend_output:{args.blend_output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
