#!/usr/bin/env python3
"""
Apply the frozen experimental PriceFM blend to an existing LEAR forecast file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.pricefm_experimental import (
    BEST_PRICEFM_EXPERIMENT,
    blend_lear_with_pricefm,
    load_pricefm_forecast,
)


DEFAULT_LEAR = ROOT / "pfc_shaping" / "output" / "lear_forecast_latest.csv"
DEFAULT_PRICEFM = ROOT / BEST_PRICEFM_EXPERIMENT.forecast_latest_path
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "lear_forecast_pricefm_experimental_latest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend LEAR forecast with experimental PriceFM forecast.")
    parser.add_argument("--lear", type=Path, default=DEFAULT_LEAR)
    parser.add_argument("--pricefm", type=Path, default=DEFAULT_PRICEFM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weight", type=float, default=BEST_PRICEFM_EXPERIMENT.blend_weight)
    parser.add_argument("--mode", choices=["fixed", "regime"], default="regime")
    args = parser.parse_args()

    lear = pd.read_csv(args.lear)
    pricefm = load_pricefm_forecast(args.pricefm)
    blended = blend_lear_with_pricefm(
        lear,
        pricefm,
        weight_pricefm=args.weight if args.mode == "fixed" else None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(args.output, index=False)
    print(f"output:{args.output}")
    print(f"rows:{len(blended)}")
    print(f"pricefm_rows:{int(blended['pricefm_used'].sum())}")
    print(f"mode:{args.mode}")
    if args.mode == "fixed":
        print(f"weight:{args.weight:.2f}")


if __name__ == "__main__":
    main()
