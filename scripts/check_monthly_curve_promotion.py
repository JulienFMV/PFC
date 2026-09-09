"""Check monthly curve promotion evidence from audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.calibration.monthly_curve_promotion import evaluate_monthly_curve_promotion


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    audit_gates = pd.read_csv(args.audit_gates)
    historical_thresholds = pd.read_csv(args.historical_thresholds)
    manifest = {}
    if args.manifest is not None and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    decision = evaluate_monthly_curve_promotion(
        audit_gates,
        historical_thresholds,
        run_timestamp=pd.Timestamp(args.run_timestamp) if args.run_timestamp else None,
        far_horizon_min_years=int(args.far_horizon_min_years),
        required_governance_gates=_csv_set(args.require_governance_gates),
        manifest=manifest,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(decision.summary, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        decision.details.to_csv(args.details_output, index=False)

    print(json.dumps(decision.summary, sort_keys=True, default=str))
    return 0 if decision.approved else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-gates", type=Path, required=True)
    parser.add_argument("--historical-thresholds", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run-timestamp", default=None)
    parser.add_argument("--far-horizon-min-years", type=int, default=2)
    parser.add_argument(
        "--require-governance-gates",
        default="",
        help=(
            "Comma-separated governance gate ids that must be present and PASS, "
            "for example lambda_calibration_artifact_present,production_export_path_parity."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--details-output", type=Path, default=None)
    return parser.parse_args(argv)


def _csv_set(value: str) -> set[str]:
    return {part.strip() for part in str(value).split(",") if part.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
