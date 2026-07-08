"""Check promotion readiness for a selected EPEX shape-lab artifact.

This checker intentionally separates strict diagnostic evidence from production
promotion evidence.  A lab artifact can pass product and Power BI diagnostics
while still being NO-GO production until it has its own production/export/
selected/capstone chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def check_readiness(
    *,
    lab_manifest: Path,
    governance_audit: Path,
    independent_summary: Path,
    product_summary: Path,
    powerbi_summary: Path,
    ompex_advisory_delta: Path | None = None,
    adjusted_production_manifest: Path | None = None,
    adjusted_export_manifest: Path | None = None,
    adjusted_selected_config: Path | None = None,
    adjusted_capstone: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    lab = _load_json(lab_manifest)
    governance = _load_json(governance_audit)
    independent = _load_json(independent_summary)
    product = _load_json(product_summary)
    powerbi = _load_powerbi_summary(powerbi_summary)
    ompex = _load_json(ompex_advisory_delta) if ompex_advisory_delta is not None else None

    checks = [
        _check("lab_activation_lab_only", lab.get("activation_status") == "lab_only", lab.get("activation_status")),
        _check("lab_not_production_approved", lab.get("production_approved") is False, lab.get("production_approved")),
        _check("lab_ompex_not_selection", lab.get("ompex_used_in_selection") is False, lab.get("ompex_used_in_selection")),
        _check("governance_pass", governance.get("status") == "PASS", governance.get("status")),
        _check(
            "independent_no_ompex",
            independent.get("benchmark_policy") == "independent_no_ompex"
            and independent.get("ompex_used_in_model") is False
            and independent.get("ompex_used_in_selection") is False,
            independent.get("benchmark_policy"),
        ),
        _check("product_all_gates_pass", product.get("all_gates_pass") is True, product.get("all_gates_pass")),
        _check("product_no_critical", int(product.get("critical_count", -1)) == 0, product.get("critical_count")),
        _check("product_no_unsupported", int(product.get("unsupported_count", -1)) == 0, product.get("unsupported_count")),
        _check(
            "product_no_blocking_quote_conflicts",
            int(product.get("blocking_quote_conflict_count", -1)) == 0,
            product.get("blocking_quote_conflict_count"),
        ),
        _check(
            "powerbi_quality_gate_pass",
            powerbi.get("powerbi_quality_gate_status") == "PASS",
            powerbi.get("powerbi_quality_gate_status"),
        ),
        _check(
            "powerbi_no_weighted_negative_hours",
            _float_value(powerbi.get("weighted_negative_hours")) == 0.0,
            powerbi.get("weighted_negative_hours"),
        ),
        _check(
            "powerbi_no_critical_flags",
            _powerbi_critical_count(powerbi) == 0,
            _powerbi_critical_count(powerbi),
        ),
    ]
    if ompex is not None:
        checks.append(
            _check(
                "ompex_advisory_not_selection",
                ompex.get("read_only") is True
                and ompex.get("ompex_used_in_model") is False
                and ompex.get("ompex_used_in_selection") is False,
                ompex.get("benchmark_policy"),
            )
        )

    production_paths = {
        "adjusted_production_manifest": adjusted_production_manifest,
        "adjusted_export_manifest": adjusted_export_manifest,
        "adjusted_selected_config": adjusted_selected_config,
        "adjusted_capstone": adjusted_capstone,
    }
    missing_production_evidence = [
        name for name, path in production_paths.items() if path is None or not path.exists()
    ]
    if adjusted_capstone is not None and adjusted_capstone.exists():
        capstone = _load_json(adjusted_capstone)
        capstone_approved = capstone.get("approved") is True
    else:
        capstone_approved = False
    if adjusted_export_manifest is not None and adjusted_export_manifest.exists():
        export_manifest = _load_json(adjusted_export_manifest)
        checks.extend(
            [
                _check(
                    "adjusted_export_manifest_bound",
                    _same_path(export_manifest.get("adjusted_csv"), (lab.get("outputs") or {}).get("adjusted_csv")),
                    export_manifest.get("adjusted_csv"),
                ),
                _check(
                    "adjusted_export_manifest_not_production_approved",
                    export_manifest.get("production_approved") is False,
                    export_manifest.get("production_approved"),
                ),
            ]
        )
    if adjusted_selected_config is not None and adjusted_selected_config.exists():
        selected_artifact = _load_json(adjusted_selected_config)
        checks.extend(
            [
                _check(
                    "adjusted_selected_artifact_bound",
                    _same_path(
                        selected_artifact.get("selected_adjusted_csv"),
                        (lab.get("outputs") or {}).get("adjusted_csv"),
                    ),
                    selected_artifact.get("selected_adjusted_csv"),
                ),
                _check(
                    "adjusted_selected_artifact_not_production_approved",
                    selected_artifact.get("production_promotion_approved") is False,
                    selected_artifact.get("production_promotion_approved"),
                ),
            ]
        )
    strict_diagnostics_pass = all(
        check["status"] == "PASS"
        for check in checks
        if check["name"]
        not in {
            "lab_activation_lab_only",
            "lab_not_production_approved",
        }
    )
    production_chain_pass = not missing_production_evidence and capstone_approved and lab.get("production_approved") is True
    approved = bool(strict_diagnostics_pass and production_chain_pass)
    status = (
        "PROMOTION_READY"
        if approved
        else "STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING"
        if strict_diagnostics_pass
        else "STRICT_DIAGNOSTICS_FAIL"
    )
    summary = {
        "schema_version": "epex_lab_promotion_readiness.v1",
        "approved": approved,
        "status": status,
        "strict_diagnostics_pass": bool(strict_diagnostics_pass),
        "production_chain_pass": bool(production_chain_pass),
        "missing_production_evidence": missing_production_evidence,
        "checks": checks,
        "selected_adjusted_csv": (lab.get("outputs") or {}).get("adjusted_csv"),
        "lab_manifest": str(lab_manifest),
        "governance_audit": str(governance_audit),
        "independent_summary": str(independent_summary),
        "product_summary": str(product_summary),
        "powerbi_summary": str(powerbi_summary),
        "ompex_advisory_delta": str(ompex_advisory_delta) if ompex_advisory_delta is not None else None,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


def _check(name: str, passed: bool, value: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "value": value}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_powerbi_summary(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    if not {"metric", "value"}.issubset(frame.columns):
        raise ValueError(f"Power BI summary must contain metric,value columns: {path}")
    return {str(row.metric): str(row.value) for row in frame.itertuples(index=False)}


def _float_value(value: Any) -> float:
    return float(value)


def _powerbi_critical_count(summary: dict[str, str]) -> int:
    total = 0
    for key, value in summary.items():
        if key.endswith("_critical_flags"):
            total += int(float(value))
    return total


def _same_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (OSError, TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-manifest", type=Path, required=True)
    parser.add_argument("--governance-audit", type=Path, required=True)
    parser.add_argument("--independent-summary", type=Path, required=True)
    parser.add_argument("--product-summary", type=Path, required=True)
    parser.add_argument("--powerbi-summary", type=Path, required=True)
    parser.add_argument("--ompex-advisory-delta", type=Path, default=None)
    parser.add_argument("--adjusted-production-manifest", type=Path, default=None)
    parser.add_argument("--adjusted-export-manifest", type=Path, default=None)
    parser.add_argument("--adjusted-selected-config", type=Path, default=None)
    parser.add_argument("--adjusted-capstone", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = check_readiness(
        lab_manifest=args.lab_manifest,
        governance_audit=args.governance_audit,
        independent_summary=args.independent_summary,
        product_summary=args.product_summary,
        powerbi_summary=args.powerbi_summary,
        ompex_advisory_delta=args.ompex_advisory_delta,
        adjusted_production_manifest=args.adjusted_production_manifest,
        adjusted_export_manifest=args.adjusted_export_manifest,
        adjusted_selected_config=args.adjusted_selected_config,
        adjusted_capstone=args.adjusted_capstone,
        output=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
