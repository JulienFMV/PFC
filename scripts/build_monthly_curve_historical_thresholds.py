"""Build Phase F historical threshold rows for monthly curve audit gates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.calibration.monthly_curve_audit import build_monthly_curve_historical_thresholds


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    history = pd.read_parquet(args.forwards)
    thresholds = build_monthly_curve_historical_thresholds(
        history,
        market=args.market,
        load_type=args.load_type,
        run_timestamp=pd.Timestamp(args.run_timestamp) if args.run_timestamp else None,
        lookback_years=args.lookback_years,
        min_required_n=args.min_required_n,
        timezone=args.timezone,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(output, index=False)
    summary = thresholds["status"].astype(str).value_counts().to_dict() if not thresholds.empty else {}
    print(f"[monthly-thresholds] rows={len(thresholds)} status={summary}")
    print(f"[monthly-thresholds] wrote {output}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forwards", type=Path, default=Path("data/eex_forwards_history.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/monthly_curve_calibration/historical_thresholds.csv"),
    )
    parser.add_argument("--market", default="CH")
    parser.add_argument("--load-type", default="BASE")
    parser.add_argument("--run-timestamp", default=None)
    parser.add_argument("--lookback-years", type=int, default=6)
    parser.add_argument("--min-required-n", type=int, default=24)
    parser.add_argument("--timezone", default="Europe/Zurich")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
