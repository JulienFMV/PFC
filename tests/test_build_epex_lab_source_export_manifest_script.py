from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_epex_lab_source_export_manifest import build_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_build_epex_lab_source_export_manifest_binds_candidate_csv(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    candidate.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    monthly = tmp_path / "monthly.json"
    selected = tmp_path / "selected.json"
    policy = tmp_path / "policy.json"
    capstone = tmp_path / "capstone.json"
    _write_json(
        monthly,
        {
            "monthly_level_authority": "solver",
            "monthly_solution_hash": "m" * 64,
            "active_constraints_hash": "c" * 64,
            "active_config_hash": "a" * 64,
        },
    )
    _write_json(selected, {"production_approved": True, "monthly_solution_hash": "m" * 64})
    _write_json(policy, {"production_approved": True})
    _write_json(capstone, {"approved": True, "status": "PROMOTION_EVIDENCE_PASS"})

    manifest = build_manifest(
        candidate_csv=candidate,
        baseline_monthly_manifest=monthly,
        selected_config=selected,
        source_hierarchy_policy=policy,
        capstone=capstone,
        output=tmp_path / "source_export_manifest.json",
    )

    assert manifest["schema_version"] == "epex_lab_source_export_manifest.v1"
    assert manifest["production_approved"] is False
    assert manifest["production_promotion_approved"] is False
    assert manifest["candidate_csv"] == str(candidate)
    assert manifest["candidate_csv_sha256"] == _sha256(candidate)
    assert manifest["output_csv_sha256"] == _sha256(candidate)
    assert manifest["monthly_level_authority"] == "solver"
    assert manifest["source_capstone_approved"] is True
    assert manifest["ompex_used_in_model"] is False
    assert manifest["ompex_used_in_selection"] is False
    saved = json.loads((tmp_path / "source_export_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_build_epex_lab_source_export_manifest_requires_candidate_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="candidate CSV"):
        build_manifest(
            candidate_csv=tmp_path / "missing.csv",
            output=tmp_path / "source_export_manifest.json",
        )
