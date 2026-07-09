"""Audit a locked future no-OMPEX holdout backtest for an EPEX lab candidate.

The audit validates a completed spot backtest against a pre-registered locked
holdout plan. It reports whether the holdout evidence passes the plan criteria,
but it never approves production promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
SUMMARY_POLICY = "rolling_origin_epex_spot_no_ompex_lab_only"


def audit_holdout(
    *,
    plan_json: Path,
    spot_backtest_summary: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    plan = _read_json(plan_json)
    summary = _read_json(spot_backtest_summary)
    criteria = plan.get("pass_criteria") or {}
    post_csv = Path(str((summary.get("outputs") or {}).get("post_valuation_timestamp_residuals_csv", "")))
    if not post_csv.is_absolute():
        candidate = spot_backtest_summary.parent / post_csv
        if candidate.exists():
            post_csv = candidate
    post = _load_post_valuation(post_csv)
    window = _holdout_window(post, start_utc=str(plan.get("holdout_start_utc")), end_utc=str(plan.get("holdout_end_utc")))
    metrics = _window_metrics(window)
    checks = {
        "plan_schema": plan.get("schema_version") == PLAN_SCHEMA_VERSION,
        "plan_no_ompex": plan.get("ompex_used_in_model") is False
        and plan.get("ompex_used_in_selection") is False
        and plan.get("ompex_used_in_backtest") is False,
        "summary_policy": summary.get("benchmark_policy") == SUMMARY_POLICY,
        "summary_no_ompex": summary.get("ompex_used_in_model") is False
        and summary.get("ompex_used_in_selection") is False
        and summary.get("ompex_used_in_backtest") is False,
        "summary_lab_only": summary.get("promotion_gate") is False and summary.get("production_approved") is False,
        "strict_lab_gate_pass": summary.get("strict_lab_gate_pass") is criteria.get("strict_lab_gate_pass"),
        "baseline_csv_sha256_bound": (summary.get("source_hashes") or {}).get("baseline_csv")
        == criteria.get("baseline_csv_sha256"),
        "adjusted_csv_sha256_bound": (summary.get("source_hashes") or {}).get("adjusted_csv")
        == criteria.get("adjusted_csv_sha256"),
        "valuation_timestamp_bound": _utc_text(summary.get("valuation_timestamp_utc"))
        == _utc_text((plan.get("backtest") or {}).get("valuation_timestamp_utc")),
        "post_valuation_csv_present": post_csv.exists(),
        "holdout_window_hours": int(metrics["hours"]) >= int(criteria.get("min_holdout_hours", 1)),
        "holdout_non_degraded": float(metrics.get("residual_mae_improvement_eur_mwh", float("-inf")))
        >= float(criteria.get("min_residual_mae_improvement_eur_mwh", 0.0)),
    }
    status = "LOCKED_HOLDOUT_PASS" if all(checks.values()) else "NO_GO_LOCKED_HOLDOUT_FAIL"
    audit = {
        "schema_version": "epex_lab_locked_holdout_audit.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "status": status,
        "holdout_pass": bool(status == "LOCKED_HOLDOUT_PASS"),
        "plan_json": str(plan_json),
        "spot_backtest_summary": str(spot_backtest_summary),
        "post_valuation_timestamp_residuals_csv": str(post_csv),
        "checks": checks,
        "holdout_window": {
            "start_utc": plan.get("holdout_start_utc"),
            "end_utc": plan.get("holdout_end_utc"),
        },
        "holdout_metrics": metrics,
        "criteria": criteria,
        "note": (
            "A passing locked holdout is independent scientific evidence only. "
            "Production still requires the adjusted production/export/selected/"
            "capstone chain."
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _load_post_valuation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "timestamp_utc" not in frame.columns:
        raise ValueError(f"post valuation CSV missing timestamp_utc: {path}")
    out = frame.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    return out


def _holdout_window(frame: pd.DataFrame, *, start_utc: str, end_utc: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    mask = (frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] < end)
    return frame.loc[mask].copy()


def _window_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"hours": 0, "residual_mae_improvement_eur_mwh": None}
    baseline = pd.to_numeric(frame["baseline_abs_error_eur_mwh"], errors="raise")
    adjusted = pd.to_numeric(frame["adjusted_abs_error_eur_mwh"], errors="raise")
    return {
        "hours": int(len(frame)),
        "baseline_residual_mae_eur_mwh": float(baseline.mean()),
        "adjusted_residual_mae_eur_mwh": float(adjusted.mean()),
        "residual_mae_improvement_eur_mwh": float(baseline.mean() - adjusted.mean()),
    }


def _utc_text(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


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
    parser.add_argument("--spot-backtest-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = audit_holdout(
        plan_json=args.plan_json,
        spot_backtest_summary=args.spot_backtest_summary,
        output=args.output,
    )
    print(json.dumps(_jsonable(audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
