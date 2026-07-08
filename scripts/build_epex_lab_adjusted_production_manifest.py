"""Build the adjusted-production manifest contract for an EPEX lab artifact.

This script does not promote an EPEX lab curve by itself.  It packages the
hash-bound production-manifest schema expected by the readiness checker and
keeps approval flags false unless a real LT production path passes explicit
approval values through the Python API.  The CLI is intentionally NO-GO only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_manifest(
    *,
    lab_manifest: Path,
    baseline_monthly_manifest: Path,
    product_summary: Path,
    powerbi_summary: Path,
    source_hierarchy_policy: Path,
    independent_summary: Path,
    governance_audit: Path,
    output: Path,
    production_run_id: str | None = None,
    production_entrypoint: str | None = None,
    git_commit: str | None = None,
    production_approved: bool = False,
    production_promotion_approved: bool = False,
) -> dict[str, Any]:
    lab = _load_json(lab_manifest)
    monthly = _load_json(baseline_monthly_manifest)
    product = _load_json(product_summary)
    powerbi = _load_powerbi_summary(powerbi_summary)
    policy = _load_json(source_hierarchy_policy)
    independent = _load_json(independent_summary)
    governance = _load_json(governance_audit)

    adjusted_csv = Path(str((lab.get("outputs") or {}).get("adjusted_csv", "")))
    if not adjusted_csv.exists():
        raise FileNotFoundError(f"adjusted CSV not found: {adjusted_csv}")

    checks = _contract_checks(
        lab=lab,
        monthly=monthly,
        product=product,
        powerbi=powerbi,
        policy=policy,
        independent=independent,
        governance=governance,
    )
    contract_pass = all(check["status"] == "PASS" for check in checks)
    if production_approved or production_promotion_approved:
        missing_identity = [
            name
            for name, value in {
                "production_run_id": production_run_id,
                "production_entrypoint": production_entrypoint,
                "git_commit": git_commit,
            }.items()
            if not value
        ]
        if missing_identity:
            raise ValueError(
                "production approval requires complete run identity: "
                + ", ".join(missing_identity)
            )
    approved = bool(production_approved and production_promotion_approved and contract_pass)

    manifest: dict[str, Any] = {
        "schema_version": "epex_lab_adjusted_production_manifest.v1",
        "activation_status": "production_candidate" if approved else "production_candidate_no_go",
        "production_approved": approved,
        "production_promotion_approved": approved,
        "promotion_scope": "LT_EPEX_LAB_OFF_BY_DEFAULT_PRODUCTION_PATH" if approved else "LT_EPEX_LAB_PRODUCTION_CONTRACT_NO_GO",
        "production_run_id": production_run_id,
        "production_entrypoint": production_entrypoint,
        "git_commit": git_commit,
        "adjusted_csv": str(adjusted_csv),
        "adjusted_csv_sha256": _sha256(adjusted_csv),
        "lab_manifest": str(lab_manifest),
        "lab_manifest_sha256": _sha256(lab_manifest),
        "baseline_monthly_manifest": str(baseline_monthly_manifest),
        "baseline_monthly_manifest_sha256": _sha256(baseline_monthly_manifest),
        "monthly_level_authority": monthly.get("monthly_level_authority"),
        "monthly_solution_hash": monthly.get("monthly_solution_hash"),
        "active_constraints_hash": monthly.get("active_constraints_hash"),
        "active_config_hash": monthly.get("active_config_hash"),
        "source_hierarchy_policy": str(source_hierarchy_policy),
        "source_hierarchy_policy_sha256": _sha256(source_hierarchy_policy),
        "product_summary": str(product_summary),
        "product_summary_sha256": _sha256(product_summary),
        "powerbi_summary": str(powerbi_summary),
        "powerbi_summary_sha256": _sha256(powerbi_summary),
        "independent_summary": str(independent_summary),
        "independent_summary_sha256": _sha256(independent_summary),
        "governance_audit": str(governance_audit),
        "governance_audit_sha256": _sha256(governance_audit),
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "contract_pass": contract_pass,
        "checks": checks,
        "note": (
            "CLI-built manifests are NO-GO by default. Production approval "
            "requires a real off-by-default LT production path to call the API "
            "with approval flags and complete run identity."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _contract_checks(
    *,
    lab: dict[str, Any],
    monthly: dict[str, Any],
    product: dict[str, Any],
    powerbi: dict[str, str],
    policy: dict[str, Any],
    independent: dict[str, Any],
    governance: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _check("lab_activation_lab_only", lab.get("activation_status") == "lab_only", lab.get("activation_status")),
        _check("lab_not_production_approved", lab.get("production_approved") is False, lab.get("production_approved")),
        _check("lab_ompex_not_selection", lab.get("ompex_used_in_selection") is False, lab.get("ompex_used_in_selection")),
        _check("monthly_level_authority_solver", monthly.get("monthly_level_authority") == "solver", monthly.get("monthly_level_authority")),
        _check("source_hierarchy_policy_approved", policy.get("production_approved") is True, policy.get("production_approved")),
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
            float(powerbi.get("weighted_negative_hours", "nan")) == 0.0,
            powerbi.get("weighted_negative_hours"),
        ),
        _check("powerbi_no_critical_flags", _powerbi_critical_count(powerbi) == 0, _powerbi_critical_count(powerbi)),
    ]


def _check(name: str, passed: bool, value: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "value": value}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_powerbi_summary(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    if not {"metric", "value"}.issubset(frame.columns):
        raise ValueError(f"Power BI summary must contain metric,value columns: {path}")
    return {str(row.metric): str(row.value) for row in frame.itertuples(index=False)}


def _powerbi_critical_count(summary: dict[str, str]) -> int:
    total = 0
    for key, value in summary.items():
        if key.endswith("_critical_flags"):
            total += int(float(value))
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-manifest", type=Path, required=True)
    parser.add_argument("--baseline-monthly-manifest", type=Path, required=True)
    parser.add_argument("--product-summary", type=Path, required=True)
    parser.add_argument("--powerbi-summary", type=Path, required=True)
    parser.add_argument("--source-hierarchy-policy", type=Path, required=True)
    parser.add_argument("--independent-summary", type=Path, required=True)
    parser.add_argument("--governance-audit", type=Path, required=True)
    parser.add_argument("--production-run-id", default=None)
    parser.add_argument("--production-entrypoint", default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        lab_manifest=args.lab_manifest,
        baseline_monthly_manifest=args.baseline_monthly_manifest,
        product_summary=args.product_summary,
        powerbi_summary=args.powerbi_summary,
        source_hierarchy_policy=args.source_hierarchy_policy,
        independent_summary=args.independent_summary,
        governance_audit=args.governance_audit,
        production_run_id=args.production_run_id,
        production_entrypoint=args.production_entrypoint,
        git_commit=args.git_commit,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
