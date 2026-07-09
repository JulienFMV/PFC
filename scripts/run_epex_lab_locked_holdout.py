"""Run a locked EPEX lab holdout only after spot coverage is complete.

The runner is intentionally conservative: it first writes a coverage report and
stops without running the backtest unless the pre-registered holdout window is
fully covered by the supplied EPEX spot parquet. It never approves production
promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_epex_lab_locked_holdout import audit_holdout
from scripts.backtest_epex_shape_lab_against_spot import backtest_against_spot
from scripts.check_epex_lab_locked_holdout_coverage import check_coverage


def run_locked_holdout(
    *,
    plan_json: Path,
    spot_parquet: Path,
    output_dir: Path,
) -> dict[str, Any]:
    plan = _read_json(plan_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "coverage_status.json"
    coverage = check_coverage(plan_json=plan_json, spot_parquet=spot_parquet, output=coverage_path)
    run_summary: dict[str, Any] = {
        "schema_version": "epex_lab_locked_holdout_run.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "plan_json": str(plan_json),
        "spot_parquet": str(spot_parquet),
        "output_dir": str(output_dir),
        "coverage_status": str(coverage_path),
        "coverage_ready": bool(coverage.get("ready_to_run_backtest")),
        "coverage": coverage,
        "benchmark_policy": "locked_future_no_ompex_holdout",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
    }
    if coverage.get("ready_to_run_backtest") is not True:
        run_summary.update(
            {
                "status": "WAITING_FOR_FULL_SPOT_COVERAGE",
                "backtest_ran": False,
                "audit_ran": False,
                "next_action": coverage.get("next_action"),
            }
        )
        return _write_run_summary(output_dir, run_summary)

    backtest_cfg = plan.get("backtest") or {}
    backtest = backtest_against_spot(
        baseline_csv=Path(str(plan["baseline_csv"])),
        adjusted_csv=Path(str(plan["adjusted_csv"])),
        spot_parquet=spot_parquet,
        output_dir=output_dir,
        valuation_timestamp=str(backtest_cfg["valuation_timestamp_utc"]),
        lookback_years=int(backtest_cfg.get("lookback_years", 2)),
        eval_days=int(backtest_cfg.get("eval_days", 30)),
        embargo_days=int(backtest_cfg.get("embargo_days", 1)),
        max_auto_folds=int(backtest_cfg.get("max_auto_folds", 12)),
        min_eval_hours=int(backtest_cfg.get("min_eval_hours", 24)),
    )
    backtest_summary_path = output_dir / "spot_backtest_summary.json"
    audit_path = output_dir / "locked_holdout_audit.json"
    audit = audit_holdout(plan_json=plan_json, spot_backtest_summary=backtest_summary_path, output=audit_path)
    run_summary.update(
        {
            "status": "LOCKED_HOLDOUT_PASS" if audit.get("holdout_pass") is True else "NO_GO_LOCKED_HOLDOUT_FAIL",
            "backtest_ran": True,
            "audit_ran": True,
            "spot_backtest_summary": str(backtest_summary_path),
            "locked_holdout_audit": str(audit_path),
            "spot_backtest_status": backtest.get("status"),
            "holdout_audit_status": audit.get("status"),
            "holdout_pass": bool(audit.get("holdout_pass")),
            "strict_lab_gate_pass": bool(backtest.get("strict_lab_gate_pass")),
            "post_valuation_metrics": backtest.get("post_valuation_metrics"),
            "holdout_metrics": audit.get("holdout_metrics"),
            "next_action": (
                "Review holdout evidence; production still requires adjusted production/export/selected/capstone chain."
                if audit.get("holdout_pass") is True
                else "Do not promote; investigate holdout failure without retuning against this locked window."
            ),
        }
    )
    return _write_run_summary(output_dir, run_summary)


def _write_run_summary(output_dir: Path, run_summary: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "locked_holdout_run_summary.json"
    path.write_text(json.dumps(_jsonable(run_summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_summary["run_summary"] = str(path)
    return run_summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--spot-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_locked_holdout(plan_json=args.plan_json, spot_parquet=args.spot_parquet, output_dir=args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
