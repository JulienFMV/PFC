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
import re
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.epex_lab_selection_policy import (
    selection_policy_pass,
    selection_policy_value,
)


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
    selection_summary: Path | None = None,
    production_run_id: str | None = None,
    production_entrypoint: str | None = None,
    git_commit: str | None = None,
    source_provenance_manifest: Path | None = None,
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
    selection = _load_json(selection_summary) if selection_summary is not None else None
    source_provenance = _load_json(source_provenance_manifest) if source_provenance_manifest is not None else None

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
        source_provenance=source_provenance,
        selection=selection,
        adjusted_csv=adjusted_csv,
    )
    contract_pass = all(check["status"] == "PASS" for check in checks)
    selection_policy_ok = selection_policy_pass(selection, adjusted_csv=adjusted_csv)
    if production_approved or production_promotion_approved:
        identity_errors = _production_identity_errors(
            production_run_id=production_run_id,
            production_entrypoint=production_entrypoint,
            git_commit=git_commit,
            source_provenance_manifest=source_provenance_manifest,
        )
        if identity_errors:
            raise ValueError(
                "production approval requires complete valid run identity: "
                + ", ".join(identity_errors)
            )
    approved = bool(production_approved and production_promotion_approved and contract_pass)
    approved = bool(approved and selection_policy_ok)

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
        "selection_summary": str(selection_summary) if selection_summary is not None else None,
        "selection_summary_sha256": _sha256(selection_summary) if selection_summary is not None else None,
        "selection_policy_pass": selection_policy_ok,
        "source_provenance_manifest": str(source_provenance_manifest) if source_provenance_manifest is not None else None,
        "source_provenance_manifest_sha256": (
            _sha256(source_provenance_manifest) if source_provenance_manifest is not None else None
        ),
        "source_kind": (source_provenance or {}).get("source_kind") if source_provenance is not None else None,
        "source_promotion_eligible": (
            (source_provenance or {}).get("source_promotion_eligible") if source_provenance is not None else None
        ),
        "source_provenance_pass": _source_provenance_pass(source_provenance, adjusted_csv),
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
    source_provenance: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    adjusted_csv: Path,
) -> list[dict[str, Any]]:
    checks = [
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
        _check(
            "selection_policy_pass",
            selection_policy_pass(selection, adjusted_csv=adjusted_csv),
            selection_policy_value(selection, adjusted_csv=adjusted_csv),
        ),
    ]
    checks.append(
        _check(
            "source_provenance_manifest_present",
            source_provenance is not None,
            source_provenance is not None,
        )
    )
    if source_provenance is not None:
        checks.extend(_source_provenance_checks(source_provenance, adjusted_csv))
    return checks


def _production_identity_errors(
    *,
    production_run_id: str | None,
    production_entrypoint: str | None,
    git_commit: str | None,
    source_provenance_manifest: Path | None,
) -> list[str]:
    errors: list[str] = []
    if not str(production_run_id or "").strip():
        errors.append("production_run_id")
    if not str(production_entrypoint or "").strip():
        errors.append("production_entrypoint")
    if not str(git_commit or "").strip():
        errors.append("git_commit")
    elif re.fullmatch(r"[0-9a-f]{40}", str(git_commit).strip()) is None:
        errors.append("git_commit_must_be_40_hex")
    if source_provenance_manifest is None:
        errors.append("source_provenance_manifest")
    elif not source_provenance_manifest.exists():
        errors.append("source_provenance_manifest_missing")
    return errors


def _source_provenance_checks(source_provenance: dict[str, Any], adjusted_csv: Path) -> list[dict[str, Any]]:
    adjusted_path = str(source_provenance.get("adjusted_csv"))
    adjusted_sha = source_provenance.get("adjusted_csv_sha256")
    expected_sha = _sha256(adjusted_csv)
    source_path = Path(str(source_provenance.get("source_path", "")))
    staged_candidate = Path(str(source_provenance.get("staged_candidate_csv", "")))
    lab_manifest = Path(str(source_provenance.get("lab_manifest", "")))
    source_export_manifest = Path(str(source_provenance.get("source_export_manifest", "")))
    return [
        _check(
            "source_provenance_schema",
            source_provenance.get("schema_version") == "epex_lab_adjusted_lt_candidate_stage.v1",
            source_provenance.get("schema_version"),
        ),
        _check(
            "source_provenance_schema_role",
            source_provenance.get("schema_role") == "source_provenance",
            source_provenance.get("schema_role"),
        ),
        _check(
            "source_provenance_lab_only",
            source_provenance.get("activation_status") == "staged_lab_only"
            and source_provenance.get("production_approved") is False
            and source_provenance.get("production_promotion_approved") is False,
            {
                "activation_status": source_provenance.get("activation_status"),
                "production_approved": source_provenance.get("production_approved"),
                "production_promotion_approved": source_provenance.get("production_promotion_approved"),
            },
        ),
        _check(
            "source_provenance_candidate_csv",
            source_provenance.get("source_kind") == "candidate_csv",
            source_provenance.get("source_kind"),
        ),
        _check(
            "source_provenance_promotion_eligible",
            source_provenance.get("source_promotion_eligible") is True,
            source_provenance.get("source_promotion_eligible"),
        ),
        _check(
            "source_provenance_no_contract_blockers",
            list(source_provenance.get("production_contract_blockers") or []) == [],
            source_provenance.get("production_contract_blockers"),
        ),
        _check(
            "source_provenance_adjusted_csv_bound",
            _same_path(adjusted_path, adjusted_csv) or adjusted_sha == expected_sha,
            {"adjusted_csv": adjusted_path, "adjusted_csv_sha256": adjusted_sha},
        ),
        _check(
            "source_provenance_source_sha_bound",
            source_path.exists() and source_provenance.get("source_sha256") == _sha256(source_path),
            {"source_path": str(source_path), "source_sha256": source_provenance.get("source_sha256")},
        ),
        _check(
            "source_provenance_staged_candidate_sha_bound",
            staged_candidate.exists()
            and source_provenance.get("staged_candidate_csv_sha256") == _sha256(staged_candidate),
            {
                "staged_candidate_csv": str(staged_candidate),
                "staged_candidate_csv_sha256": source_provenance.get("staged_candidate_csv_sha256"),
            },
        ),
        _check(
            "source_provenance_lab_manifest_sha_bound",
            lab_manifest.exists() and source_provenance.get("lab_manifest_sha256") == _sha256(lab_manifest),
            {"lab_manifest": str(lab_manifest), "lab_manifest_sha256": source_provenance.get("lab_manifest_sha256")},
        ),
        _check(
            "source_provenance_source_export_manifest_sha_bound",
            source_export_manifest.exists()
            and source_provenance.get("source_export_manifest_sha256") == _sha256(source_export_manifest),
            {
                "source_export_manifest": str(source_export_manifest),
                "source_export_manifest_sha256": source_provenance.get("source_export_manifest_sha256"),
            },
        ),
        _check(
            "source_provenance_source_export_manifest_bound",
            source_export_manifest.exists()
            and _source_export_manifest_bound_to_csv(source_export_manifest, source_path),
            {"source_export_manifest": str(source_export_manifest), "source_path": str(source_path)},
        ),
        _check(
            "source_provenance_no_ompex",
            source_provenance.get("ompex_used_in_model") is False
            and source_provenance.get("ompex_used_in_selection") is False,
            {
                "ompex_used_in_model": source_provenance.get("ompex_used_in_model"),
                "ompex_used_in_selection": source_provenance.get("ompex_used_in_selection"),
            },
        ),
    ]


def _source_provenance_pass(source_provenance: dict[str, Any] | None, adjusted_csv: Path) -> bool:
    if source_provenance is None:
        return False
    return all(check["status"] == "PASS" for check in _source_provenance_checks(source_provenance, adjusted_csv))


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


def _same_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (OSError, TypeError, ValueError):
        return False


def _source_export_manifest_bound_to_csv(manifest_path: Path, source_csv: Path) -> bool:
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if not manifest.get("schema_version"):
        return False
    candidate_path_keys = [
        "output_csv",
        "candidate_csv",
        "source_csv",
        "adjusted_csv",
        "selected_adjusted_csv",
    ]
    for key in candidate_path_keys:
        if _same_path(manifest.get(key), source_csv):
            return True
    candidate_sha_keys = [
        "output_csv_sha256",
        "candidate_csv_sha256",
        "source_csv_sha256",
        "adjusted_csv_sha256",
        "selected_adjusted_csv_sha256",
    ]
    expected_sha = _sha256(source_csv) if source_csv.exists() else None
    return expected_sha is not None and any(manifest.get(key) == expected_sha for key in candidate_sha_keys)


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
    parser.add_argument("--selection-summary", type=Path, default=None)
    parser.add_argument("--production-run-id", default=None)
    parser.add_argument("--production-entrypoint", default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--source-provenance-manifest", type=Path, default=None)
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
        selection_summary=args.selection_summary,
        production_run_id=args.production_run_id,
        production_entrypoint=args.production_entrypoint,
        git_commit=args.git_commit,
        source_provenance_manifest=args.source_provenance_manifest,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
