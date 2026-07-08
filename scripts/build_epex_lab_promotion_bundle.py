"""Build a local promotion-evidence bundle for an EPEX shape-lab artifact.

The bundle is intentionally non-production by default.  It records exact hashes
for the adjusted CSV, the lab manifest, strict diagnostics, and the baseline
monthly solver authority so that reviewers can see what is proven locally and
what still requires a real production/export/capstone promotion path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_bundle(
    *,
    lab_manifest: Path,
    baseline_monthly_manifest: Path,
    product_summary: Path,
    powerbi_summary: Path,
    source_hierarchy_policy: Path,
    independent_summary: Path,
    governance_audit: Path,
    output_dir: Path,
    ompex_advisory_delta: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lab = _load_json(lab_manifest)
    monthly = _load_json(baseline_monthly_manifest)
    product = _load_json(product_summary)
    adjusted_csv = Path(str((lab.get("outputs") or {}).get("adjusted_csv", "")))
    baseline_csv = Path(str(lab.get("candidate_csv", "")))
    if not adjusted_csv.exists():
        raise FileNotFoundError(f"adjusted CSV not found: {adjusted_csv}")
    if not baseline_csv.exists():
        raise FileNotFoundError(f"baseline CSV not found: {baseline_csv}")

    export_manifest_path = output_dir / "adjusted_export_manifest.json"
    selected_artifact_path = output_dir / "adjusted_selected_artifact.json"
    local_capstone_path = output_dir / "adjusted_local_capstone_no_go.json"

    export_manifest = {
        "schema_version": "epex_lab_adjusted_export_manifest.v1",
        "activation_status": "lab_only",
        "production_approved": False,
        "production_promotion_approved": False,
        "promotion_scope": "LOCAL_EPEX_LAB_STRICT_DIAGNOSTIC_ONLY",
        "adjusted_csv": str(adjusted_csv),
        "adjusted_csv_sha256": _sha256(adjusted_csv),
        "baseline_csv": str(baseline_csv),
        "baseline_csv_sha256": _sha256(baseline_csv),
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
        "ompex_advisory_delta": str(ompex_advisory_delta) if ompex_advisory_delta else None,
        "ompex_advisory_delta_sha256": _sha256(ompex_advisory_delta) if ompex_advisory_delta else None,
    }
    export_manifest_path.write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected_artifact = {
        "schema_version": "epex_lab_selected_artifact.v1",
        "selection_status": "LOCAL_STRICT_DIAGNOSTIC_SELECTED",
        "production_approved": False,
        "production_promotion_approved": False,
        "promotion_scope": "LOCAL_EPEX_LAB_STRICT_DIAGNOSTIC_ONLY",
        "selected_adjusted_csv": str(adjusted_csv),
        "selected_adjusted_csv_sha256": export_manifest["adjusted_csv_sha256"],
        "adjusted_export_manifest": str(export_manifest_path),
        "adjusted_export_manifest_sha256": _sha256(export_manifest_path),
        "baseline_monthly_solution_hash": monthly.get("monthly_solution_hash"),
        "baseline_active_constraints_hash": monthly.get("active_constraints_hash"),
        "source_hierarchy_policy_sha256": export_manifest["source_hierarchy_policy_sha256"],
        "product_all_gates_pass": product.get("all_gates_pass") is True,
        "reason": (
            "Selected EPEX lab artifact has strict local diagnostics but is not "
            "production approved without a production manifest and capstone."
        ),
    }
    selected_artifact_path.write_text(
        json.dumps(selected_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    local_capstone = {
        "schema_version": "epex_lab_local_capstone.v1",
        "approved": False,
        "status": "LOCAL_STRICT_DIAGNOSTICS_PASS_NO_PRODUCTION_APPROVAL",
        "strict_diagnostics_pass": bool(product.get("all_gates_pass") is True),
        "production_chain_pass": False,
        "adjusted_export_manifest": str(export_manifest_path),
        "adjusted_export_manifest_sha256": _sha256(export_manifest_path),
        "adjusted_selected_artifact": str(selected_artifact_path),
        "adjusted_selected_artifact_sha256": _sha256(selected_artifact_path),
        "missing_production_evidence": ["adjusted_production_manifest"],
        "note": (
            "This local capstone is evidence packaging only. It must not be "
            "treated as production promotion approval."
        ),
    }
    local_capstone_path.write_text(
        json.dumps(local_capstone, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "adjusted_export_manifest": export_manifest_path,
        "adjusted_selected_artifact": selected_artifact_path,
        "adjusted_local_capstone": local_capstone_path,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
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
    parser.add_argument("--ompex-advisory-delta", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = build_bundle(
        lab_manifest=args.lab_manifest,
        baseline_monthly_manifest=args.baseline_monthly_manifest,
        product_summary=args.product_summary,
        powerbi_summary=args.powerbi_summary,
        source_hierarchy_policy=args.source_hierarchy_policy,
        independent_summary=args.independent_summary,
        governance_audit=args.governance_audit,
        ompex_advisory_delta=args.ompex_advisory_delta,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
