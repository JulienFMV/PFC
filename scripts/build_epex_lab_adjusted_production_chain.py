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
