"""Build approved export/selected/capstone artifacts for an approved EPEX lab production manifest.

This script does not approve a lab candidate by itself.  It only packages the
remaining production-chain artifacts after an adjusted production manifest is
already approved, contract-pass, source-provenance-pass, and run-identity-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def build_chain(
    *,
    adjusted_production_manifest: Path,
    output_dir: Path,
) -> dict[str, Path]:
    production = _load_json(adjusted_production_manifest)
    errors = _production_manifest_errors(production, adjusted_production_manifest)
    if errors:
        raise ValueError("approved adjusted production manifest required: " + ", ".join(errors))

    adjusted_csv = Path(str(production.get("adjusted_csv")))
    if not adjusted_csv.exists():
        raise FileNotFoundError(f"adjusted CSV not found: {adjusted_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    production_sha = _sha256(adjusted_production_manifest)
    run_identity = {
        "production_run_id": production.get("production_run_id"),
        "production_entrypoint": production.get("production_entrypoint"),
        "git_commit": production.get("git_commit"),
    }
    common = {
        "activation_status": "production_approved",
        "production_approved": True,
        "production_promotion_approved": True,
        "promotion_scope": "LT_EPEX_LAB_OFF_BY_DEFAULT_PRODUCTION_PATH",
        "adjusted_csv": str(adjusted_csv),
        "adjusted_csv_sha256": _sha256(adjusted_csv),
        "adjusted_production_manifest": str(adjusted_production_manifest),
        "adjusted_production_manifest_sha256": production_sha,
        **run_identity,
    }

    export_path = output_dir / "adjusted_export_manifest.json"
    selected_path = output_dir / "adjusted_selected_artifact.json"
    capstone_path = output_dir / "adjusted_production_capstone.json"

    export_manifest = {
        "schema_version": "epex_lab_adjusted_export_manifest.v1",
        **common,
        "baseline_monthly_manifest": production.get("baseline_monthly_manifest"),
        "baseline_monthly_manifest_sha256": production.get("baseline_monthly_manifest_sha256"),
        "source_hierarchy_policy": production.get("source_hierarchy_policy"),
        "source_hierarchy_policy_sha256": production.get("source_hierarchy_policy_sha256"),
        "product_summary": production.get("product_summary"),
        "product_summary_sha256": production.get("product_summary_sha256"),
        "powerbi_summary": production.get("powerbi_summary"),
        "powerbi_summary_sha256": production.get("powerbi_summary_sha256"),
        "independent_summary": production.get("independent_summary"),
        "independent_summary_sha256": production.get("independent_summary_sha256"),
        "governance_audit": production.get("governance_audit"),
        "governance_audit_sha256": production.get("governance_audit_sha256"),
        "source_provenance_manifest": production.get("source_provenance_manifest"),
        "source_provenance_manifest_sha256": production.get("source_provenance_manifest_sha256"),
    }
    export_path.write_text(json.dumps(export_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selected_artifact = {
        "schema_version": "epex_lab_selected_artifact.v1",
        **common,
        "selection_status": "PRODUCTION_APPROVED",
        "selected_adjusted_csv": str(adjusted_csv),
        "selected_adjusted_csv_sha256": _sha256(adjusted_csv),
        "adjusted_export_manifest": str(export_path),
        "adjusted_export_manifest_sha256": _sha256(export_path),
        "reason": "Selected EPEX lab adjusted artifact is bound to an approved adjusted production manifest.",
    }
    selected_path.write_text(json.dumps(selected_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    capstone = {
        "schema_version": "epex_lab_production_capstone.v1",
        "approved": True,
        "status": "PROMOTION_READY",
        "strict_diagnostics_pass": True,
        "production_chain_pass": True,
        "adjusted_csv": str(adjusted_csv),
        "adjusted_csv_sha256": _sha256(adjusted_csv),
        "adjusted_production_manifest": str(adjusted_production_manifest),
        "adjusted_production_manifest_sha256": production_sha,
        "adjusted_export_manifest": str(export_path),
        "adjusted_export_manifest_sha256": _sha256(export_path),
        "adjusted_selected_artifact": str(selected_path),
        "adjusted_selected_artifact_sha256": _sha256(selected_path),
        **run_identity,
        "note": "Capstone is valid only with the exact bound adjusted production/export/selected artifacts.",
    }
    capstone_path.write_text(json.dumps(capstone, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "adjusted_export_manifest": export_path,
        "adjusted_selected_artifact": selected_path,
        "adjusted_capstone": capstone_path,
    }


def _production_manifest_errors(production: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    if production.get("schema_version") != "epex_lab_adjusted_production_manifest.v1":
        errors.append("schema_version")
    for key in ["production_approved", "production_promotion_approved", "contract_pass", "source_provenance_pass"]:
        if production.get(key) is not True:
            errors.append(key)
    if not _same_path(production.get("adjusted_production_manifest"), path) and production.get("adjusted_production_manifest") is not None:
        errors.append("self_reference_path")
    if not re.fullmatch(r"[0-9a-f]{40}", str(production.get("git_commit") or "")):
        errors.append("git_commit")
    if not str(production.get("production_run_id") or "").strip():
        errors.append("production_run_id")
    if not str(production.get("production_entrypoint") or "").strip():
        errors.append("production_entrypoint")
    if production.get("ompex_used_in_model") is not False:
        errors.append("ompex_used_in_model")
    if production.get("ompex_used_in_selection") is not False:
        errors.append("ompex_used_in_selection")
    adjusted_csv = Path(str(production.get("adjusted_csv", "")))
    if not adjusted_csv.exists():
        errors.append("adjusted_csv")
    elif production.get("adjusted_csv_sha256") != _sha256(adjusted_csv):
        errors.append("adjusted_csv_sha256")
    errors.extend(_source_provenance_errors(production, adjusted_csv))
    return errors


def _source_provenance_errors(production: dict[str, Any], adjusted_csv: Path) -> list[str]:
    errors: list[str] = []
    provenance_text = str(production.get("source_provenance_manifest") or "").strip()
    if not provenance_text:
        return ["source_provenance_manifest"]
    provenance_path = Path(provenance_text)
    if not provenance_path.exists():
        return ["source_provenance_manifest_missing"]
    if production.get("source_provenance_manifest_sha256") != _sha256(provenance_path):
        errors.append("source_provenance_manifest_sha256")
        return errors
    try:
        provenance = _load_json(provenance_path)
    except (OSError, json.JSONDecodeError):
        return ["source_provenance_manifest_json"]
    expected_adjusted_sha = _sha256(adjusted_csv) if adjusted_csv.exists() else None
    if provenance.get("schema_version") != "epex_lab_adjusted_lt_candidate_stage.v1":
        errors.append("source_provenance_schema")
    if provenance.get("schema_role") != "source_provenance":
        errors.append("source_provenance_schema_role")
    if not (
        provenance.get("activation_status") == "staged_lab_only"
        and provenance.get("production_approved") is False
        and provenance.get("production_promotion_approved") is False
    ):
        errors.append("source_provenance_lab_only")
    if production.get("source_kind") != provenance.get("source_kind") or provenance.get("source_kind") != "candidate_csv":
        errors.append("source_provenance_source_kind")
    if production.get("source_promotion_eligible") is not True or provenance.get("source_promotion_eligible") is not True:
        errors.append("source_provenance_promotion_eligible")
    if list(provenance.get("production_contract_blockers") or []) != []:
        errors.append("source_provenance_contract_blockers")
    if provenance.get("adjusted_csv_sha256") != expected_adjusted_sha:
        errors.append("source_provenance_adjusted_csv_sha256")
    source_path = Path(str(provenance.get("source_path", "")))
    if not source_path.exists() or provenance.get("source_sha256") != _sha256(source_path):
        errors.append("source_provenance_source_sha256")
    staged_candidate = Path(str(provenance.get("staged_candidate_csv", "")))
    if not staged_candidate.exists() or provenance.get("staged_candidate_csv_sha256") != _sha256(staged_candidate):
        errors.append("source_provenance_staged_candidate_sha256")
    lab_manifest = Path(str(provenance.get("lab_manifest", "")))
    if not lab_manifest.exists() or provenance.get("lab_manifest_sha256") != _sha256(lab_manifest):
        errors.append("source_provenance_lab_manifest_sha256")
    source_export = Path(str(provenance.get("source_export_manifest", "")))
    if not source_export.exists() or provenance.get("source_export_manifest_sha256") != _sha256(source_export):
        errors.append("source_provenance_source_export_manifest_sha256")
    elif not _source_export_manifest_bound_to_csv(source_export, source_path):
        errors.append("source_provenance_source_export_manifest_bound")
    if provenance.get("ompex_used_in_model") is not False or provenance.get("ompex_used_in_selection") is not False:
        errors.append("source_provenance_ompex")
    return errors


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False


def _source_export_manifest_bound_to_csv(manifest_path: Path, source_csv: Path) -> bool:
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if not manifest.get("schema_version"):
        return False
    path_keys = [
        "output_csv",
        "candidate_csv",
        "source_csv",
        "adjusted_csv",
        "selected_adjusted_csv",
    ]
    for key in path_keys:
        if _same_path(manifest.get(key), source_csv):
            return True
    sha_keys = [
        "output_csv_sha256",
        "candidate_csv_sha256",
        "source_csv_sha256",
        "adjusted_csv_sha256",
        "selected_adjusted_csv_sha256",
    ]
    expected_sha = _sha256(source_csv) if source_csv.exists() else None
    return expected_sha is not None and any(manifest.get(key) == expected_sha for key in sha_keys)
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (OSError, TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjusted-production-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = build_chain(
        adjusted_production_manifest=args.adjusted_production_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
