"""Audit EPEX shape-lab artifacts for lab-only governance.

This is not a production promotion gate.  It verifies that an EPEX A/B lab run
kept OMPEX out of model selection and preserved the required lab safeguards
before any advisory OMPEX post-check is interpreted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def audit_governance(
    *,
    lab_manifest: Path,
    independent_summary: Path,
    output_json: Path,
    ompex_metrics: Path | None = None,
    max_monthly_mean_delta: float = 1e-5,
    max_width_delta: float = 1e-9,
    max_after_constraint_error: float = 1e-5,
) -> dict[str, Any]:
    lab = _read_json(lab_manifest)
    independent = _read_json(independent_summary)
    ompex = _read_json(ompex_metrics) if ompex_metrics is not None else None

    checks: list[dict[str, Any]] = []
    _check(checks, "lab_activation_status", lab.get("activation_status") == "lab_only", lab.get("activation_status"))
    _check(checks, "lab_production_approved_false", lab.get("production_approved") is False, lab.get("production_approved"))
    _check(checks, "lab_ompex_not_selected", lab.get("ompex_used_in_selection") is False, lab.get("ompex_used_in_selection"))
    _check(
        checks,
        "lab_benchmark_policy_forbids_ompex",
        str(lab.get("benchmark_policy")) == "advisory_only_ompex_forbidden_as_input_target_loss_or_gate",
        lab.get("benchmark_policy"),
    )
    _check(
        checks,
        "monthly_base_constraints_present",
        int(lab.get("constraint_summary", {}).get("base_monthly_constraints", 0)) > 0,
        lab.get("constraint_summary", {}).get("base_monthly_constraints"),
    )
    _check(
        checks,
        "monthly_peak_constraints_present",
        int(lab.get("constraint_summary", {}).get("peak_monthly_constraints", 0)) > 0,
        lab.get("constraint_summary", {}).get("peak_monthly_constraints"),
    )
    _check(
        checks,
        "after_constraint_error_ok",
        _float(lab.get("constraint_summary", {}).get("max_after_abs_error_eur_mwh")) <= max_after_constraint_error,
        lab.get("constraint_summary", {}).get("max_after_abs_error_eur_mwh"),
    )
    _check(
        checks,
        "weighted_negative_hours_zero",
        int(lab.get("audit", {}).get("weighted_negative_hours", -1)) == 0,
        lab.get("audit", {}).get("weighted_negative_hours"),
    )

    _check(
        checks,
        "independent_policy_no_ompex",
        independent.get("benchmark_policy") == "independent_no_ompex",
        independent.get("benchmark_policy"),
    )
    _check(
        checks,
        "independent_ompex_not_model",
        independent.get("ompex_used_in_model") is False,
        independent.get("ompex_used_in_model"),
    )
    _check(
        checks,
        "independent_ompex_not_selection",
        independent.get("ompex_used_in_selection") is False,
        independent.get("ompex_used_in_selection"),
    )
    _check(
        checks,
        "independent_finite_adjusted",
        independent.get("finite_adjusted_ok") is True,
        independent.get("finite_adjusted_ok"),
    )
    _check(
        checks,
        "independent_quantile_order",
        independent.get("quantile_order_adjusted_ok") is True,
        independent.get("quantile_order_adjusted_ok"),
    )
    _check(
        checks,
        "independent_monthly_drift_ok",
        _float(independent.get("max_abs_monthly_mean_delta_eur_mwh")) <= max_monthly_mean_delta,
        independent.get("max_abs_monthly_mean_delta_eur_mwh"),
    )
    _check(
        checks,
        "independent_width_drift_ok",
        _float(independent.get("max_abs_width_delta_eur_mwh")) <= max_width_delta,
        independent.get("max_abs_width_delta_eur_mwh"),
    )
    _check(
        checks,
        "independent_weighted_negative_hours_zero",
        int(independent.get("weighted_negative_hours_adjusted", -1)) == 0,
        independent.get("weighted_negative_hours_adjusted"),
    )

    if ompex is not None:
        _check(checks, "ompex_policy_advisory", ompex.get("benchmark_policy") == "advisory", ompex.get("benchmark_policy"))
        _check(checks, "ompex_read_only", ompex.get("read_only") is True, ompex.get("read_only"))
        _check(checks, "ompex_not_model_input", ompex.get("ompex_used_in_model") is False, ompex.get("ompex_used_in_model"))
        _check(
            checks,
            "ompex_points_match_independent_overlap_or_less",
            int(ompex.get("n_points", 0)) <= int(independent.get("n_hours", 0)),
            {"ompex_n_points": ompex.get("n_points"), "independent_n_hours": independent.get("n_hours")},
        )

    failed = [row for row in checks if row["status"] != "PASS"]
    decision = {
        "audit": "epex_shape_lab_governance",
        "status": "PASS" if not failed else "FAIL",
        "production_approval": "NO",
        "promotion_gate": False,
        "ompex_role": "advisory_post_check_only" if ompex is not None else "not_checked",
        "failed_count": len(failed),
        "checks": checks,
        "inputs": {
            "lab_manifest": str(lab_manifest),
            "independent_summary": str(independent_summary),
            "ompex_metrics": str(ompex_metrics) if ompex_metrics is not None else None,
        },
        "thresholds": {
            "max_monthly_mean_delta": float(max_monthly_mean_delta),
            "max_width_delta": float(max_width_delta),
            "max_after_constraint_error": float(max_after_constraint_error),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check(rows: list[dict[str, Any]], name: str, passed: bool, value: Any) -> None:
    rows.append({"name": name, "status": "PASS" if bool(passed) else "FAIL", "value": value})


def _float(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-manifest", type=Path, required=True)
    parser.add_argument("--independent-summary", type=Path, required=True)
    parser.add_argument("--ompex-metrics", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-monthly-mean-delta", type=float, default=1e-5)
    parser.add_argument("--max-width-delta", type=float, default=1e-9)
    parser.add_argument("--max-after-constraint-error", type=float, default=1e-5)
    args = parser.parse_args(argv)
    decision = audit_governance(
        lab_manifest=args.lab_manifest,
        independent_summary=args.independent_summary,
        ompex_metrics=args.ompex_metrics,
        output_json=args.output_json,
        max_monthly_mean_delta=args.max_monthly_mean_delta,
        max_width_delta=args.max_width_delta,
        max_after_constraint_error=args.max_after_constraint_error,
    )
    print(json.dumps({"status": decision["status"], "failed_count": decision["failed_count"]}, sort_keys=True))
    return 0 if decision["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
