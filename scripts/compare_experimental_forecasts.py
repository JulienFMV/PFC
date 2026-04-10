#!/usr/bin/env python3
"""
Compare experimental LEAR overlays (fixed / regime / futureboost) on the latest forecast window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.futureboost_experimental import apply_futureboost_experimental  # noqa: E402
from pfc_shaping.model.pricefm_experimental import blend_lear_with_pricefm, load_pricefm_forecast  # noqa: E402


DEFAULT_LEAR = ROOT / "pfc_shaping" / "output" / "lear_forecast_latest.csv"
DEFAULT_PRICEFM = ROOT / "pfc_shaping" / "output" / "pricefm_forecast_latest.csv"
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "experimental_forecast_compare.json"


def _summarize(df: pd.DataFrame, label: str) -> dict:
    out = {
        "label": label,
        "rows": int(len(df)),
        "mean": float(df["price_lear"].mean()),
        "min": float(df["price_lear"].min()),
        "max": float(df["price_lear"].max()),
        "std": float(df["price_lear"].std()),
    }
    local = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Zurich")
    peak_mask = local.dt.hour.isin([7, 8, 9, 17, 18, 19, 20]) & (local.dt.dayofweek < 5)
    if peak_mask.any():
        out["peak_mean"] = float(df.loc[peak_mask, "price_lear"].mean())
        out["offpeak_mean"] = float(df.loc[~peak_mask, "price_lear"].mean())
    return out


def build_experimental_compare_payload(lear_path: Path, pricefm_path: Path) -> dict:
    lear = pd.read_csv(lear_path)
    lear["timestamp"] = pd.to_datetime(lear["timestamp"], utc=True)
    pricefm = load_pricefm_forecast(pricefm_path)

    fixed = blend_lear_with_pricefm(lear.copy(), pricefm, weight_pricefm=0.15)
    regime = blend_lear_with_pricefm(lear.copy(), pricefm, weight_pricefm=None)
    futureboost, futureboost_meta = apply_futureboost_experimental(lear.copy(), pricefm)

    summaries = []
    for name, df in [
        ("lear", lear),
        ("fixed", fixed),
        ("regime", regime),
        ("futureboost", futureboost),
    ]:
        summaries.append(_summarize(df, name))

    base = summaries[0]
    deltas = []
    for summary in summaries[1:]:
        deltas.append(
            {
                "label": summary["label"],
                "delta_mean": summary["mean"] - base["mean"],
                "delta_peak_mean": summary.get("peak_mean", 0.0) - base.get("peak_mean", 0.0),
                "delta_offpeak_mean": summary.get("offpeak_mean", 0.0) - base.get("offpeak_mean", 0.0),
                "delta_std": summary["std"] - base["std"],
            }
        )

    return {
        "lear_path": str(lear_path.resolve()),
        "pricefm_path": str(pricefm_path.resolve()),
        "futureboost_meta": futureboost_meta,
        "summaries": summaries,
        "deltas_vs_lear": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experimental overlays on the latest LEAR forecast.")
    parser.add_argument("--lear", type=Path, default=DEFAULT_LEAR)
    parser.add_argument("--pricefm", type=Path, default=DEFAULT_PRICEFM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_experimental_compare_payload(args.lear, args.pricefm)
    args.output.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output:{args.output.resolve()}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
