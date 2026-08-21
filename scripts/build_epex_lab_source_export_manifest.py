"""Build a hash-bound source-export manifest for EPEX lab staging.

The manifest proves that a candidate hourly CSV used as an EPEX lab source is
bound to known local evidence.  It does not approve the adjusted EPEX lab curve
for production; it only makes the source CSV provenance verifiable before
staging can package a NO-GO production-contract artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_manifest(
    *,
    candidate_csv: Path,
    output: Path,
    baseline_monthly_manifest: Path | None = None,
    selected_config: Path | None = None,
    source_hierarchy_policy: Path | None = None,
    capstone: Path | None = None,
    production_approved: bool = False,
    production_promotion_approved: bool = False,
) -> dict[str, Any]:
    if not candidate_csv.exists():
        raise FileNotFoundError(f"candidate CSV not found: {candidate_csv}")
    artifacts = {
        "baseline_monthly_manifest": baseline_monthly_manifest,
        "selected_config": selected_config,
        "source_hierarchy_policy": source_hierarchy_policy,
        "capstone": capstone,
    }
    for name, path in artifacts.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    manifest: dict[str, Any] = {
        "schema_version": "epex_lab_source_export_manifest.v1",
        "activation_status": "source_export_for_lab_staging",
        "production_approved": bool(production_approved),
        "production_promotion_approved": bool(production_promotion_approved),
        "promotion_scope": "SOURCE_CSV_PROVENANCE_ONLY",
        "candidate_csv": str(candidate_csv),
        "candidate_csv_sha256": _sha256(candidate_csv),
        "output_csv": str(candidate_csv),
        "output_csv_sha256": _sha256(candidate_csv),
        "baseline_monthly_manifest": _path_or_none(baseline_monthly_manifest),
        "baseline_monthly_manifest_sha256": _sha_or_none(baseline_monthly_manifest),
        "selected_config": _path_or_none(selected_config),
        "selected_config_sha256": _sha_or_none(selected_config),
        "source_hierarchy_policy": _path_or_none(source_hierarchy_policy),
        "source_hierarchy_policy_sha256": _sha_or_none(source_hierarchy_policy),
        "capstone": _path_or_none(capstone),
        "capstone_sha256": _sha_or_none(capstone),
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "note": (
            "This manifest binds the source hourly CSV for EPEX lab staging. "
            "It is not adjusted-curve production approval."
        ),
    }
    if baseline_monthly_manifest is not None:
        monthly = _load_json(baseline_monthly_manifest)
        manifest.update(
            {
                "monthly_level_authority": monthly.get("monthly_level_authority"),
                "monthly_solution_hash": monthly.get("monthly_solution_hash"),
                "active_constraints_hash": monthly.get("active_constraints_hash"),
                "active_config_hash": monthly.get("active_config_hash"),
            }
        )
    if selected_config is not None:
        selected = _load_json(selected_config)
        manifest["selected_config_production_approved"] = selected.get("production_approved")
        manifest["selected_config_monthly_solution_hash"] = selected.get("monthly_solution_hash")
    if capstone is not None:
        capstone_payload = _load_json(capstone)
        manifest["source_capstone_approved"] = capstone_payload.get("approved")
        manifest["source_capstone_status"] = capstone_payload.get("status")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _sha_or_none(path: Path | None) -> str | None:
    return _sha256(path) if path is not None else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--baseline-monthly-manifest", type=Path)
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--source-hierarchy-policy", type=Path)
    parser.add_argument("--capstone", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        candidate_csv=args.candidate_csv,
        baseline_monthly_manifest=args.baseline_monthly_manifest,
        selected_config=args.selected_config,
        source_hierarchy_policy=args.source_hierarchy_policy,
        capstone=args.capstone,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
