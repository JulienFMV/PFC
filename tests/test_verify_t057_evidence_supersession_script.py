from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.verify_t057_evidence_supersession as verifier_module
from scripts.verify_t057_evidence_supersession import (
    supersession_record_for_sha256,
    verify_supersession_registry,
)


def test_t057_supersession_registry_verifies_artifacts_and_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = {
        "legacy_energy_charts_run_summary": _artifact(tmp_path, "run.json", b"run"),
        "legacy_spot_backtest_summary": _artifact(tmp_path, "spot.json", b"spot"),
        "legacy_duplicated_rolling_folds_csv": _artifact(tmp_path, "folds.csv", b"folds"),
        "superseded_local_quality_report": _artifact(tmp_path, "quality.json", b"quality"),
        "superseded_correction_v1": _artifact(tmp_path, "correction-v1.json", b"v1"),
    }
    correction = tmp_path / "correction-v2.json"
    _write_json(
        correction,
        {
            "schema_version": "t057_fold_integrity_correction.v2",
            "status": "T057_HISTORICAL_ROLLING_EVIDENCE_REVOKED",
            "production_authorization": False,
            "promotion_gate": False,
            "source_contract": {
                "canonical_sources_only": True,
                "caller_anchors_match_canonical_registry": True,
                "candidate_spot_and_config_cross_bound": True,
                "metrics_recomputed_from_csv": True,
            },
            "legacy_evidence": {
                "run_summary_sha256": _sha256(artifacts["legacy_energy_charts_run_summary"]),
                "spot_backtest_summary_sha256": _sha256(
                    artifacts["legacy_spot_backtest_summary"]
                ),
                "rolling_folds_csv_sha256": _sha256(
                    artifacts["legacy_duplicated_rolling_folds_csv"]
                ),
            },
            "supersession": {
                "superseded_quality_report_sha256": _sha256(
                    artifacts["superseded_local_quality_report"]
                )
            },
        },
    )
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(
        verifier_module,
        "_EXPECTED_SUPERSEDED_PATHS",
        {kind: path.name for kind, path in artifacts.items()},
    )
    _write_json(
        registry,
        {
            "schema_version": "t057_evidence_supersession_registry.v1",
            "status": "ACTIVE_FAIL_CLOSED_SUPERSESSION",
            "plan_id": "t057_locked_t056_future_holdout",
            "production_authorization": False,
            "promotion_gate": False,
            "authoritative_correction": {
                "path": correction.name,
                "sha256": _sha256(correction),
                "schema_version": "t057_fold_integrity_correction.v2",
            },
            "superseded_evidence": [
                {
                    "kind": kind,
                    "path": path.name,
                    "sha256": _sha256(path),
                }
                for kind, path in artifacts.items()
            ],
            "effective_claims": {
                "historical_distinct_fold_count": 1,
                "historical_robustness_established": False,
                "future_holdout_episode_count": 1,
                "future_holdout_hours": 336,
                "materiality_established": False,
                "production_go": False,
            },
        },
    )

    original_read = verifier_module.read_stable_single_link_file
    registry_reads = 0

    def counted_read(path: Path, **kwargs):
        nonlocal registry_reads
        if Path(path) == registry:
            registry_reads += 1
            if registry_reads > 1:
                raise AssertionError("registry bytes were captured more than once")
        return original_read(path, **kwargs)

    monkeypatch.setattr(verifier_module, "read_stable_single_link_file", counted_read)
    result = verify_supersession_registry(registry=registry, repo_root=tmp_path)
    assert registry_reads == 1
    monkeypatch.setattr(verifier_module, "read_stable_single_link_file", original_read)
    record = supersession_record_for_sha256(
        _sha256(artifacts["superseded_local_quality_report"]),
        registry=registry,
    )

    assert result["status"] == "T057_SUPERSESSION_VERIFIED"
    assert result["superseded_evidence_count"] == 5
    assert result["production_authorization"] is False
    assert record is not None
    assert record["kind"] == "superseded_local_quality_report"

    incomplete = json.loads(registry.read_text(encoding="utf-8"))
    incomplete["superseded_evidence"].pop()
    _write_json(registry, incomplete)
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_supersession_registry(registry=registry, repo_root=tmp_path)

    substituted = json.loads(json.dumps(incomplete))
    substituted["superseded_evidence"] = [
        {
            "kind": kind,
            "path": path.name,
            "sha256": _sha256(path),
        }
        for kind, path in artifacts.items()
    ]
    shadow = _artifact(tmp_path, "shadow-run.json", artifacts["legacy_energy_charts_run_summary"].read_bytes())
    substituted["superseded_evidence"][0]["path"] = shadow.name
    substituted["superseded_evidence"][0]["sha256"] = _sha256(shadow)
    _write_json(registry, substituted)
    with pytest.raises(ValueError, match="mismatched_paths"):
        verify_supersession_registry(registry=registry, repo_root=tmp_path)


def test_t057_supersession_registry_rejects_tampered_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path, "run.json", b"before")
    correction = tmp_path / "correction.json"
    _write_json(correction, {})
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(
        verifier_module,
        "_EXPECTED_SUPERSEDED_PATHS",
        {"run": artifact.name},
    )
    _write_json(
        registry,
        {
            "schema_version": "t057_evidence_supersession_registry.v1",
            "status": "ACTIVE_FAIL_CLOSED_SUPERSESSION",
            "plan_id": "t057_locked_t056_future_holdout",
            "production_authorization": False,
            "promotion_gate": False,
            "authoritative_correction": {
                "path": correction.name,
                "sha256": _sha256(correction),
                "schema_version": "t057_fold_integrity_correction.v2",
            },
            "superseded_evidence": [
                {"kind": "run", "path": artifact.name, "sha256": _sha256(artifact)}
            ],
            "effective_claims": {},
        },
    )
    artifact.write_bytes(b"after")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_supersession_registry(registry=registry, repo_root=tmp_path)


def _artifact(root: Path, name: str, payload: bytes) -> Path:
    path = root / name
    path.write_bytes(payload)
    return path


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
