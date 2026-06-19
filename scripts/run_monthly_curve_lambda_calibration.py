"""Run offline lambda calibration for the monthly forward curve solver."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.calibration.monthly_curve_lambda_calibration import (  # noqa: E402
    FAIL_STATUSES,
    file_sha256,
    load_lambda_grid,
    run_lambda_calibration,
    write_calibration_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    grid = load_lambda_grid(args.grid)
    history = pd.read_parquet(args.forwards)
    source_hash = file_sha256(args.forwards)
    artifacts = run_lambda_calibration(
        history,
        grid=grid,
        source_file_hash=source_hash,
        max_origins=args.max_origins,
        max_configs=args.max_configs,
        smoke=bool(args.smoke),
        command_line=sys.argv if argv is None else [Path(__file__).as_posix(), *argv],
        input_parquet_path=str(args.forwards),
    )
    write_calibration_artifacts(artifacts, args.output_dir)

    final_status = str(artifacts.summary["final_status"])
    print(f"final_status={final_status}")
    print(f"production_approved={artifacts.summary['production_approved']}")
    print(f"scoring_rows={len(artifacts.scoring)}")
    print(f"output_dir={args.output_dir}")
    return 2 if final_status in FAIL_STATUSES else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forwards", type=Path, default=Path("data/eex_forwards_history.parquet"))
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path(".planning/phases/14-lt-audit-remediation/lambda_grid.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/monthly_curve_calibration"))
    parser.add_argument("--smoke", action="store_true", help="Run a deterministic small calibration subset.")
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
