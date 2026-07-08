"""Pre-register an EPEX shape-lab parameter sweep without using OMPEX.

The output is a lab-only execution plan.  It does not run candidates and does
not read OMPEX/HFC files.  Each planned trial must be executed and compared
with independent no-OMPEX diagnostics before any advisory OMPEX post-check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any


DEFAULT_GRID = {
    "weekend_intensity": [0.25, 0.5, 0.75],
    "low_tail_intensity": [0.25, 0.5, 0.75],
    "peak_subshape_intensity": [0.25, 0.5, 0.75],
}


def build_plan(
    *,
    candidate_csv: Path,
    spot_parquet: Path,
    output_root: Path,
    valuation_timestamp: str,
    max_abs_delta_eur_mwh: float,
    lookback_years: int = 5,
    grid: dict[str, list[float]] | None = None,
    plan_id: str = "epex_shape_lab_sweep_v1",
) -> dict[str, Any]:
    grid = grid or DEFAULT_GRID
    required = ["weekend_intensity", "low_tail_intensity", "peak_subshape_intensity"]
    missing = sorted(set(required) - set(grid))
    if missing:
        raise ValueError(f"sweep grid missing keys: {missing}")
    trials = []
    for idx, values in enumerate(product(*(grid[key] for key in required)), start=1):
        params = dict(zip(required, (float(v) for v in values), strict=True))
        trial_id = f"trial_{idx:03d}_w{params['weekend_intensity']:.2f}_l{params['low_tail_intensity']:.2f}_p{params['peak_subshape_intensity']:.2f}"
        trial_dir = output_root / trial_id
        adjusted_csv = trial_dir / "candidate_epex_shape_lab_adjusted.csv"
        comparison_dir = trial_dir / "independent_ab_comparison"
        governance_json = trial_dir / "governance_audit" / "epex_shape_lab_governance_audit.json"
        trials.append(
            {
                "trial_id": trial_id,
                "parameters": {
                    **params,
                    "max_abs_delta_eur_mwh": float(max_abs_delta_eur_mwh),
                    "lookback_years": int(lookback_years),
                },
                "output_dir": str(trial_dir),
                "commands": {
                    "run_ab": _join_command(
                        [
                            "python",
                            "scripts/run_epex_shape_lab_ab.py",
                            "--candidate-csv",
                            str(candidate_csv),
                            "--spot-parquet",
                            str(spot_parquet),
                            "--output-dir",
                            str(trial_dir),
                            "--valuation-timestamp",
                            valuation_timestamp,
                            "--weekend-intensity",
                            str(params["weekend_intensity"]),
                            "--low-tail-intensity",
                            str(params["low_tail_intensity"]),
                            "--peak-subshape-intensity",
                            str(params["peak_subshape_intensity"]),
                            "--max-abs-delta-eur-mwh",
                            str(float(max_abs_delta_eur_mwh)),
                            "--lookback-years",
                            str(int(lookback_years)),
                        ]
                    ),
                    "compare_independent": _join_command(
                        [
                            "python",
                            "scripts/compare_epex_shape_lab_ab.py",
                            "--baseline-csv",
                            str(candidate_csv),
                            "--adjusted-csv",
                            str(adjusted_csv),
                            "--output-dir",
                            str(comparison_dir),
                        ]
                    ),
                    "audit_governance_no_ompex": _join_command(
                        [
                            "python",
                            "scripts/audit_epex_shape_lab_governance.py",
                            "--lab-manifest",
                            str(trial_dir / "ab_lab_manifest.json"),
                            "--independent-summary",
                            str(comparison_dir / "ab_comparison_summary.json"),
                            "--output-json",
                            str(governance_json),
                        ]
                    ),
                },
            }
        )
    return {
        "plan_id": plan_id,
        "activation_status": "lab_only",
        "production_approved": False,
        "benchmark_policy": "pre_registered_independent_no_ompex",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_postcheck_allowed_after_selection": True,
        "candidate_csv": str(candidate_csv),
        "spot_parquet": str(spot_parquet),
        "candidate_csv_sha256": _sha256(candidate_csv),
        "spot_parquet_sha256": _sha256(spot_parquet),
        "valuation_timestamp": valuation_timestamp,
        "selection_basis": [
            "independent finite/quantile/negative checks",
            "monthly mean drift",
            "fan width drift",
            "calendar shape effects",
            "governance PASS without OMPEX",
        ],
        "forbidden_selection_inputs": ["OMPEX", "HFC_OMPEX", "external_HPFC_benchmark"],
        "grid": {key: [float(v) for v in grid[key]] for key in required},
        "max_abs_delta_eur_mwh": float(max_abs_delta_eur_mwh),
        "lookback_years": int(lookback_years),
        "trial_count": len(trials),
        "trials": trials,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _join_command(parts: list[str]) -> str:
    return " ".join(_quote(part) for part in parts)


def _quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _parse_grid(value: str | None) -> dict[str, list[float]] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--grid-json must decode to an object")
    return {str(key): [float(v) for v in values] for key, values in payload.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--spot-parquet", type=Path, default=Path("data/epex_hourly.parquet"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--valuation-timestamp", required=True)
    parser.add_argument("--max-abs-delta-eur-mwh", type=float, default=6.0)
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--grid-json", default=None)
    parser.add_argument("--plan-id", default="epex_shape_lab_sweep_v1")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan(
        candidate_csv=args.candidate_csv,
        spot_parquet=args.spot_parquet,
        output_root=args.output_root,
        valuation_timestamp=args.valuation_timestamp,
        max_abs_delta_eur_mwh=args.max_abs_delta_eur_mwh,
        lookback_years=args.lookback_years,
        grid=_parse_grid(args.grid_json),
        plan_id=args.plan_id,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"plan_id": plan["plan_id"], "trial_count": plan["trial_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
