"""Run offline lambda calibration for the monthly forward curve solver."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.calibration.monthly_curve_lambda_calibration import (  # noqa: E402
    load_lambda_grid,
    run_verified_lambda_calibration,
    write_calibration_artifacts,
)
from pfc_shaping.data.eex_historical_vintage import (  # noqa: E402
    EexHistoricalVintageError,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    grid = load_lambda_grid(args.grid)
    if not args.smoke and (args.max_origins is not None or args.max_configs is not None):
        print("final_status=FAIL_RESEARCH_REDUCTION_OUTSIDE_SMOKE")
        print("production_approved=False")
        return 2
    if args.vintage_catalog is None:
        print("final_status=FAIL_MISSING_HISTORICAL_VINTAGE_CATALOG")
        print("production_approved=False")
        return 2
    try:
        artifacts = run_verified_lambda_calibration(
            args.forwards,
            catalog_path=args.vintage_catalog,
            grid=grid,
            max_origins=args.max_origins,
            max_configs=args.max_configs,
            smoke=bool(args.smoke),
            command_line=sys.argv if argv is None else [Path(__file__).as_posix(), *argv],
        )
        write_calibration_artifacts(artifacts, args.output_dir)
    except (OSError, EexHistoricalVintageError) as exc:
        print("final_status=FAIL_CALIBRATION_INPUT_OR_PUBLICATION_CONTRACT")
        print("production_approved=False")
        print(f"reason={exc}")
        return 2

    final_status = str(artifacts.summary["final_status"])
    print(f"final_status={final_status}")
    print(f"production_approved={artifacts.summary['production_approved']}")
    print(f"scoring_rows={len(artifacts.scoring)}")
    print(f"output_dir={args.output_dir}")
    return 0 if final_status.startswith("PASS_") else 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forwards", type=Path, default=Path("data/eex_forwards_history.parquet"))
    parser.add_argument(
        "--vintage-catalog",
        type=Path,
        default=None,
        help="Signed eex_historical_vintage_catalog.v1 bound to the exact history bytes.",
    )
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
