from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.check_monthly_curve_promotion_from_manifests import _active_config_hash, main


def test_manifest_backed_promotion_passes_with_matching_real_hashes(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--run-timestamp",
            "2026-06-17",
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 0
    augmented = pd.read_csv(paths["augmented"])
    strict = augmented[
        augmented["gate_id"].isin(
            {
                "lambda_calibration_artifact_present",
                "production_export_path_parity",
                "selected_config_manifest_parity",
                "selected_config_production_approval",
            }
        )
    ]
    assert set(strict["status"]) == {"PASS"}


def test_manifest_backed_promotion_blocks_export_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, export_solution_hash="different_solution")

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--run-timestamp",
            "2026-06-17",
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    parity = augmented[augmented["gate_id"].eq("production_export_path_parity")].iloc[0]
    assert parity["status"] == "CRITICAL"


def test_manifest_backed_promotion_blocks_non_prod_selected_config(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, selected_production_approved=False)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--run-timestamp",
            "2026-06-17",
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    approval = augmented[augmented["gate_id"].eq("selected_config_production_approval")].iloc[0]
    assert approval["status"] == "CRITICAL"


def test_manifest_backed_promotion_blocks_candidate_scope_selected_config(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, production_promotion_approved=False)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--run-timestamp",
            "2026-06-17",
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    approval = augmented[augmented["gate_id"].eq("selected_config_production_approval")].iloc[0]
    assert approval["status"] == "CRITICAL"
    assert "production_promotion_approved=False" in approval["evidence"]


def test_manifest_backed_promotion_blocks_negative_selection_status(tmp_path: Path) -> None:
    for selection_status in (
        "NOT_PRODUCTION_APPROVED",
        "DIAGNOSTIC_SELECTED_NOT_PRODUCTION_APPROVED",
        "production_approved",
        "Production_Approved",
    ):
        paths = _write_inputs(
            tmp_path / selection_status.lower(),
            selected_production_approved=True,
            selection_status=selection_status,
        )

        rc = main(
            [
                "--audit-gates",
                str(paths["audit_gates"]),
                "--historical-thresholds",
                str(paths["historical_thresholds"]),
                "--production-manifest",
                str(paths["production_manifest"]),
                "--export-manifest",
                str(paths["export_manifest"]),
                "--selected-config-artifact",
                str(paths["selected_config"]),
                "--run-timestamp",
                "2026-06-17",
                "--augmented-audit-gates",
                str(paths["augmented"]),
            ]
        )

        assert rc == 1
        augmented = pd.read_csv(paths["augmented"])
        approval = augmented[augmented["gate_id"].eq("selected_config_production_approval")].iloc[0]
        assert approval["status"] == "CRITICAL"


def test_manifest_backed_promotion_blocks_selected_solution_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, selected_solution_hash="stale_solution")

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--run-timestamp",
            "2026-06-17",
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    parity = augmented[augmented["gate_id"].eq("selected_config_manifest_parity")].iloc[0]
    assert parity["status"] == "CRITICAL"
    assert "monthly_solution_hash" in parity["evidence"]


def test_manifest_backed_promotion_blocks_selected_constraints_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, selected_constraints_hash="stale_constraints")

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--run-timestamp",
            "2026-06-17",
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    parity = augmented[augmented["gate_id"].eq("selected_config_manifest_parity")].iloc[0]
    assert parity["status"] == "CRITICAL"
    assert "active_constraints_hash" in parity["evidence"]


def _write_inputs(
    tmp_path: Path,
    *,
    export_solution_hash: str = "solution",
    selected_production_approved: bool = True,
    production_promotion_approved: bool | None = None,
    selection_status: str | None = None,
    selected_solution_hash: str = "solution",
    selected_constraints_hash: str = "constraints",
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    audit_gates = tmp_path / "audit_gates.csv"
    historical_thresholds = tmp_path / "historical_thresholds.csv"
    production_manifest = tmp_path / "production_manifest.json"
    export_manifest = tmp_path / "export_manifest.json"
    selected_config = tmp_path / "candidate_config.json"
    augmented = tmp_path / "augmented_audit_gates.csv"

    _audit_gates().to_csv(audit_gates, index=False)
    pd.DataFrame([{"gate_id": "same_month_rank_consistency", "status": "PASS"}]).to_csv(
        historical_thresholds,
        index=False,
    )
    production = _manifest(monthly_solution_hash="solution")
    export = _manifest(monthly_solution_hash=export_solution_hash)
    production_manifest.write_text(json.dumps(production), encoding="utf-8")
    export_manifest.write_text(json.dumps(export), encoding="utf-8")
    selected_payload = {
        "schema_version": "monthly_curve_selected_config.v1",
        "config_hash": _active_config_hash(production),
        "active_config_hash_from_candidate_manifest": _active_config_hash(production),
        "monthly_solution_hash": selected_solution_hash,
        "active_constraints_hash": selected_constraints_hash,
        "candidate_manifest": str(production_manifest),
        "production_approved": selected_production_approved,
        "selection_status": (
            selection_status
            if selection_status is not None
            else "PRODUCTION_APPROVED"
            if selected_production_approved
            else "DIAGNOSTIC_SELECTED_NOT_PRODUCTION_APPROVED"
        ),
    }
    if production_promotion_approved is not None:
        selected_payload["production_promotion_approved"] = production_promotion_approved
    selected_config.write_text(json.dumps(selected_payload), encoding="utf-8")
    return {
        "audit_gates": audit_gates,
        "historical_thresholds": historical_thresholds,
        "production_manifest": production_manifest,
        "export_manifest": export_manifest,
        "selected_config": selected_config,
        "augmented": augmented,
    }


def _audit_gates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"gate_id": "hard_monthly_curve_repricing", "status": "PASS", "product": "2028"},
            {"gate_id": "neighbor_level_leakage", "status": "PASS", "product": "neighbor_shift"},
            {
                "gate_id": "monthly_shape_regression_2028_2030",
                "status": "PASS",
                "product": "2028_2030_focus_population",
            },
        ]
    )


def _manifest(*, monthly_solution_hash: str) -> dict[str, object]:
    return {
        "monthly_solution_hash": monthly_solution_hash,
        "active_constraints_hash": "constraints",
        "solver_config": {
            "lambda_prior": 1e-6,
            "lambda_smooth_month": 1.0,
            "lambda_smooth_yoy": 0.25,
            "lambda_shape": 1.0,
            "neighbor_shrinkage": 0.5,
            "history_lookback_years": 6,
            "min_history_snapshots": 24,
            "constraint_tolerance": 1e-9,
            "stationarity_tolerance": 1e-7,
        },
    }
