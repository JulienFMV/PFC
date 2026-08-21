"""Summarize EPEX shape-lab stability across frozen no-OMPEX A/B cases.

The summary is a read-only lab diagnostic.  It is not a production promotion
gate and it does not read OMPEX.  Each case is expected to point at an
independent A/B summary and its governance audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_stability(
    *,
    cases: list[tuple[str, Path, Path]],
    output_dir: Path,
    max_monthly_mean_delta: float = 1e-5,
    max_width_delta: float = 1e-9,
    max_implied_width_delta: float = 1e-9,
    max_reported_minus_implied_width: float = 1e-9,
    max_peak_offpeak_spread_delta: float = 5.0,
    max_month_hour_mean_delta: float = 3.5,
    max_boundary_delta_jump: float = 1.0,
    max_ramp_p99_increase: float = 1.0,
    min_adjusted_price: float = -10.0,
    max_p10_negative_hours: int = 240,
    max_p10_negative_cluster_hours: int = 48,
) -> dict[str, Any]:
    rows = []
    for label, summary_path, governance_path in cases:
        summary = _read_json(summary_path)
        governance = _read_json(governance_path)
        ramp_increase = float(summary["ramp_abs_p99_adjusted_eur_mwh"]) - float(
            summary["ramp_abs_p99_baseline_eur_mwh"]
        )
        row = {
            "label": label,
            "summary_json": str(summary_path),
            "governance_json": str(governance_path),
            "benchmark_policy": summary.get("benchmark_policy"),
            "ompex_used_in_model": bool(summary.get("ompex_used_in_model")),
            "ompex_used_in_selection": bool(summary.get("ompex_used_in_selection")),
            "governance_status": governance.get("status"),
            "governance_failed_count": int(governance.get("failed_count", -1)),
            "n_hours": int(summary.get("n_hours", 0)),
            "finite_adjusted_ok": bool(summary.get("finite_adjusted_ok")),
            "quantile_order_adjusted_ok": bool(summary.get("quantile_order_adjusted_ok")),
            "weighted_negative_hours_adjusted": int(summary.get("weighted_negative_hours_adjusted", -1)),
            "min_adjusted_price_eur_mwh": float(summary.get("min_adjusted_price_eur_mwh")),
            "max_abs_delta_eur_mwh": float(summary.get("max_abs_delta_eur_mwh")),
            "max_abs_monthly_mean_delta_eur_mwh": float(summary.get("max_abs_monthly_mean_delta_eur_mwh")),
            "max_abs_width_delta_eur_mwh": float(summary.get("max_abs_width_delta_eur_mwh")),
            "max_abs_implied_width_delta_eur_mwh": _float_or_none(
                summary.get("max_abs_implied_width_delta_eur_mwh")
            ),
            "max_abs_reported_minus_implied_width_adjusted_eur_mwh": _float_or_none(
                summary.get("max_abs_reported_minus_implied_width_adjusted_eur_mwh")
            ),
            "max_abs_peak_offpeak_spread_delta_eur_mwh": _float_or_none(
                summary.get("max_abs_peak_offpeak_spread_delta_eur_mwh")
            ),
            "max_abs_month_hour_mean_delta_eur_mwh": _float_or_none(
                summary.get("max_abs_month_hour_mean_delta_eur_mwh")
            ),
            "max_abs_boundary_delta_jump_eur_mwh": _float_or_none(
                summary.get("max_abs_boundary_delta_jump_eur_mwh")
            ),
            "p10_negative_hours_adjusted": _int_or_none(summary.get("p10_negative_hours_adjusted")),
            "p10_negative_cluster_max_hours": _int_or_none(summary.get("p10_negative_cluster_max_hours")),
            "ramp_abs_p99_baseline_eur_mwh": float(summary.get("ramp_abs_p99_baseline_eur_mwh")),
            "ramp_abs_p99_adjusted_eur_mwh": float(summary.get("ramp_abs_p99_adjusted_eur_mwh")),
            "ramp_abs_p99_increase_eur_mwh": ramp_increase,
        }
        row["case_pass"] = _case_pass(
            row,
            max_monthly_mean_delta=max_monthly_mean_delta,
            max_width_delta=max_width_delta,
            max_implied_width_delta=max_implied_width_delta,
            max_reported_minus_implied_width=max_reported_minus_implied_width,
            max_peak_offpeak_spread_delta=max_peak_offpeak_spread_delta,
            max_month_hour_mean_delta=max_month_hour_mean_delta,
            max_boundary_delta_jump=max_boundary_delta_jump,
            max_ramp_p99_increase=max_ramp_p99_increase,
            min_adjusted_price=min_adjusted_price,
            max_p10_negative_hours=max_p10_negative_hours,
            max_p10_negative_cluster_hours=max_p10_negative_cluster_hours,
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_csv = output_dir / "stability_cases.csv"
    frame.to_csv(cases_csv, index=False)
    failed = frame[~frame["case_pass"].astype(bool)] if not frame.empty else frame
    decision = {
        "schema_version": "epex_shape_lab_stability_summary.v1",
        "read_only": True,
        "promotion_gate": False,
        "benchmark_policy": "multi_date_independent_no_ompex",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "case_count": int(len(frame)),
        "passed_case_count": int(frame["case_pass"].sum()) if not frame.empty else 0,
        "failed_case_count": int(len(failed)),
        "status": "PASS" if len(frame) > 0 and failed.empty else "FAIL",
        "thresholds": {
            "max_monthly_mean_delta": float(max_monthly_mean_delta),
            "max_width_delta": float(max_width_delta),
            "max_implied_width_delta": float(max_implied_width_delta),
            "max_reported_minus_implied_width": float(max_reported_minus_implied_width),
            "max_peak_offpeak_spread_delta": float(max_peak_offpeak_spread_delta),
            "max_month_hour_mean_delta": float(max_month_hour_mean_delta),
            "max_boundary_delta_jump": float(max_boundary_delta_jump),
            "max_ramp_p99_increase": float(max_ramp_p99_increase),
            "min_adjusted_price": float(min_adjusted_price),
            "max_p10_negative_hours": int(max_p10_negative_hours),
            "max_p10_negative_cluster_hours": int(max_p10_negative_cluster_hours),
        },
        "outputs": {"cases_csv": str(cases_csv)},
        "note": (
            "This is stability evidence only. It does not approve production "
            "promotion and does not use OMPEX."
        ),
    }
    (output_dir / "stability_summary.json").write_text(
        json.dumps(_jsonable(decision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return decision


def _case_pass(
    row: dict[str, Any],
    *,
    max_monthly_mean_delta: float,
    max_width_delta: float,
    max_implied_width_delta: float,
    max_reported_minus_implied_width: float,
    max_peak_offpeak_spread_delta: float,
    max_month_hour_mean_delta: float,
    max_boundary_delta_jump: float,
    max_ramp_p99_increase: float,
    min_adjusted_price: float,
    max_p10_negative_hours: int,
    max_p10_negative_cluster_hours: int,
) -> bool:
    return bool(
        row["benchmark_policy"] == "independent_no_ompex"
        and row["ompex_used_in_model"] is False
        and row["ompex_used_in_selection"] is False
        and row["governance_status"] == "PASS"
        and row["governance_failed_count"] == 0
        and row["finite_adjusted_ok"] is True
        and row["quantile_order_adjusted_ok"] is True
        and row["weighted_negative_hours_adjusted"] == 0
        and row["min_adjusted_price_eur_mwh"] >= min_adjusted_price
        and _le(row["max_abs_monthly_mean_delta_eur_mwh"], max_monthly_mean_delta)
        and _le(row["max_abs_width_delta_eur_mwh"], max_width_delta)
        and _le(row["max_abs_implied_width_delta_eur_mwh"], max_implied_width_delta)
        and _le(
            row["max_abs_reported_minus_implied_width_adjusted_eur_mwh"],
            max_reported_minus_implied_width,
        )
        and _le(row["max_abs_peak_offpeak_spread_delta_eur_mwh"], max_peak_offpeak_spread_delta)
        and _le(row["max_abs_month_hour_mean_delta_eur_mwh"], max_month_hour_mean_delta)
        and _le(row["max_abs_boundary_delta_jump_eur_mwh"], max_boundary_delta_jump)
        and _le(row["ramp_abs_p99_increase_eur_mwh"], max_ramp_p99_increase)
        and _le(row["p10_negative_hours_adjusted"], max_p10_negative_hours)
        and _le(row["p10_negative_cluster_max_hours"], max_p10_negative_cluster_hours)
    )


def _parse_case(value: str) -> tuple[str, Path, Path]:
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError("case must use LABEL|AB_SUMMARY_JSON|GOVERNANCE_JSON")
    label, summary, governance = parts
    if not label:
        raise ValueError("case label must not be empty")
    return label, Path(summary), Path(governance)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _le(value: Any, threshold: float | int) -> bool:
    return value is not None and float(value) <= float(threshold)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case tuple as LABEL|AB_SUMMARY_JSON|GOVERNANCE_JSON. Repeat for each date.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-monthly-mean-delta", type=float, default=1e-5)
    parser.add_argument("--max-width-delta", type=float, default=1e-9)
    parser.add_argument("--max-implied-width-delta", type=float, default=1e-9)
    parser.add_argument("--max-reported-minus-implied-width", type=float, default=1e-9)
    parser.add_argument("--max-peak-offpeak-spread-delta", type=float, default=5.0)
    parser.add_argument("--max-month-hour-mean-delta", type=float, default=3.5)
    parser.add_argument("--max-boundary-delta-jump", type=float, default=1.0)
    parser.add_argument("--max-ramp-p99-increase", type=float, default=1.0)
    parser.add_argument("--min-adjusted-price", type=float, default=-10.0)
    parser.add_argument("--max-p10-negative-hours", type=int, default=240)
    parser.add_argument("--max-p10-negative-cluster-hours", type=int, default=48)
    args = parser.parse_args(argv)
    cases = [_parse_case(item) for item in args.case]
    decision = summarize_stability(
        cases=cases,
        output_dir=args.output_dir,
        max_monthly_mean_delta=args.max_monthly_mean_delta,
        max_width_delta=args.max_width_delta,
        max_implied_width_delta=args.max_implied_width_delta,
        max_reported_minus_implied_width=args.max_reported_minus_implied_width,
        max_peak_offpeak_spread_delta=args.max_peak_offpeak_spread_delta,
        max_month_hour_mean_delta=args.max_month_hour_mean_delta,
        max_boundary_delta_jump=args.max_boundary_delta_jump,
        max_ramp_p99_increase=args.max_ramp_p99_increase,
        min_adjusted_price=args.min_adjusted_price,
        max_p10_negative_hours=args.max_p10_negative_hours,
        max_p10_negative_cluster_hours=args.max_p10_negative_cluster_hours,
    )
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))
    return 0 if decision["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
