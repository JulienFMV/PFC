from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pfc_shaping.pipeline.monthly_curve_authority import monthly_solver_settings
from pfc_shaping.pipeline.strict_structured_data import load_strict_yaml
from pfc_shaping.validation.solver_parity_perfect_foresight import (
    run_solver_parity_perfect_foresight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the non-promotional Cal perfect-foresight solver parity diagnostic."
    )
    parser.add_argument("--epex", type=Path, default=Path("data/epex_hourly.parquet"))
    parser.add_argument(
        "--eex-history",
        type=Path,
        default=Path("data/eex_forwards_history.parquet"),
    )
    parser.add_argument("--config", type=Path, default=Path("pfc_shaping/config.yaml"))
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument(
        "--vintage",
        default="2024-12-31T17:00:00Z",
        help="Timezone-aware point-in-time cutoff.",
    )
    parser.add_argument("--config-name", default="bowl_on_floors_off")
    parser.add_argument("--estimator", choices=("baseline", "sota"), default="baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_strict_yaml(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain a mapping")
    epex_frame = pd.read_parquet(args.epex)
    if "price_eur_mwh" not in epex_frame.columns:
        raise ValueError("EPEX input must contain price_eur_mwh")
    result = run_solver_parity_perfect_foresight(
        epex_frame["price_eur_mwh"],
        pd.read_parquet(args.eex_history),
        target_year=args.target_year,
        vintage=pd.Timestamp(args.vintage),
        solver_settings=monthly_solver_settings(config),
        config_name=args.config_name,
        estimator=args.estimator,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.pfc.to_parquet(args.output_dir / "solver_parity_pf_15min.parquet")
    summary = {
        "metadata": result.metadata,
        "metrics": result.metrics,
        "final_projection_report": result.final_projection_report,
        "monthly_authority_manifest": result.monthly_authority_manifest,
    }
    (args.output_dir / "solver_parity_pf_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metadata": result.metadata, "metrics": result.metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
