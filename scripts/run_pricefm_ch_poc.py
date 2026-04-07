#!/usr/bin/env python3
"""
Run a CH-centric PriceFM POC using the local cloned PriceFM repository.

The script expects a dataset exported by `export_pricefm_ch_dataset.py` and
creates a minimal graph-aware benchmark centered on CH.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PRICEFM_REPO = ROOT / "__tmp_pricefm"
DEFAULT_DATASET = ROOT / "pfc_shaping" / "output" / "pricefm_ch_poc.csv"
DEFAULT_RESULT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_poc_metrics.json"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CH-centric PriceFM POC.")
    parser.add_argument("--pricefm-repo", type=Path, default=PRICEFM_REPO)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target-country", default="CH")
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--val-start", default="2025-07-01")
    parser.add_argument("--val-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-03-25")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--graph-degree", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()

    args.pricefm_repo = args.pricefm_repo.resolve()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()

    if not args.pricefm_repo.exists():
        raise FileNotFoundError(f"PriceFM repo not found: {args.pricefm_repo}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    try:
        import tensorflow  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not installed in the current environment. "
            "Install the versions from scripts/pricefm_requirements.txt first."
        ) from exc

    sys.path.insert(0, str(args.pricefm_repo))
    from PriceFM import (
        read_dataset,
        split_dataframe,
        scale_dataframe_per_country,
        separate_countries,
        make_rolling_window_samples,
        pipline_phase_I,
        pipline_phase_II,
    )

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
    emb_dim = 24 * len(quantiles)
    num_experts = 4

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
        x_lag_tr, x_lead_tr, y_tr, t_tr = make_rolling_window_samples(
            train_sep[country], country, lag_features, lead_features, label_column, lag_window, lead_window
        )
        x_lag_va, x_lead_va, y_va, t_va = make_rolling_window_samples(
            val_sep[country], country, lag_features, lead_features, label_column, lag_window, lead_window
        )
        x_lag_te, x_lead_te, y_te, t_te = make_rolling_window_samples(
            test_sep[country], country, lag_features, lead_features, label_column, lag_window, lead_window
        )
        rolling_train[country] = {"X_lag": x_lag_tr, "X_lead": x_lead_tr, "Y": y_tr, "t": t_tr}
        rolling_val[country] = {"X_lag": x_lag_va, "X_lead": x_lead_va, "Y": y_va, "t": t_va}
        rolling_test[country] = {"X_lag": x_lag_te, "X_lead": x_lead_te, "Y": y_te, "t": t_te}

    adjacency_dict = _build_ch_adjacency(countries)
    os.chdir(args.pricefm_repo)
    (args.pricefm_repo / "Model").mkdir(exist_ok=True)
    (args.pricefm_repo / "Result").mkdir(exist_ok=True)

    phase1 = pipline_phase_I(
        input_countries_pretrain=countries,
        output_countries_pretrain=countries,
        output_countries_test=[args.target_country],
        df_train=rolling_train,
        df_val=rolling_val,
        df_test=rolling_test,
        y_scalers=y_scalers,
        emb_dim=emb_dim,
        num_experts=num_experts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        quantiles=quantiles,
    )
    phase2 = pipline_phase_II(
        input_countries=countries,
        target_country=args.target_country,
        adjacency_dict=adjacency_dict,
        graph_degree=args.graph_degree,
        df_train=rolling_train,
        df_val=rolling_val,
        df_test=rolling_test,
        y_scalers=y_scalers,
        emb_dim=emb_dim,
        num_experts=num_experts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        quantiles=quantiles,
        model_choice=f"PhaseI_{args.target_country}",
    )

    payload = {
        "countries": countries,
        "target_country": args.target_country,
        "train_range": [args.train_start, args.train_end],
        "val_range": [args.val_start, args.val_end],
        "test_range": [args.test_start, args.test_end],
        "graph_degree": args.graph_degree,
        "phase1": phase1.get(args.target_country, {}),
        "phase2": phase2.get(args.target_country, {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"metrics_output:{args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
