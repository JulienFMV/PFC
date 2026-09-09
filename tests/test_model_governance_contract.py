from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from pfc_shaping.pipeline.model_governance_contract import (
    ModelGovernanceAuthenticationError,
    sign_model_governance_artifact_receipt,
    validate_selected_lambda_decision,
    verify_model_governance_artifact_receipt,
)


@pytest.fixture
def keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    key = Ed25519PrivateKey.generate()
    private = tmp_path / "private.pem"
    public = tmp_path / "public.pem"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH", str(public))
    return private, public


def _receipt(artifact: Path, private: Path, **overrides: str) -> dict[str, object]:
    return sign_model_governance_artifact_receipt(
        artifact,
        artifact_role=overrides.get("artifact_role", "selected_lambda_decision"),
        authority_id="FMV_MODEL_GOVERNANCE",
        issued_at_utc=overrides.get("issued_at_utc", "2026-07-12T00:00:00Z"),
        data_cutoff_utc=overrides.get("data_cutoff_utc", "2026-07-11T00:00:00Z"),
        private_key_path=private,
    )


def test_signed_governance_receipt_binds_role_bytes_and_pit(
    tmp_path: Path,
    keys: tuple[Path, Path],
) -> None:
    artifact = tmp_path / "decision.json"
    artifact.write_text(json.dumps({"lambda": 1.0}), encoding="utf-8")
    receipt = _receipt(artifact, keys[0])

    verified = verify_model_governance_artifact_receipt(
        receipt,
        artifact_path=artifact,
        expected_role="selected_lambda_decision",
        valuation_timestamp="2026-07-13T00:00:00Z",
    )

    assert verified["artifact_sha256"] == receipt["artifact_sha256"]
    assert verified["authority_id"] == "FMV_MODEL_GOVERNANCE"


def test_governance_receipt_rejects_artifact_mutation(
    tmp_path: Path,
    keys: tuple[Path, Path],
) -> None:
    artifact = tmp_path / "decision.json"
    artifact.write_text("before", encoding="utf-8")
    receipt = _receipt(artifact, keys[0])
    artifact.write_text("after", encoding="utf-8")

    with pytest.raises(ModelGovernanceAuthenticationError, match="hash mismatch"):
        verify_model_governance_artifact_receipt(
            receipt,
            artifact_path=artifact,
            expected_role="selected_lambda_decision",
            valuation_timestamp="2026-07-13T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"issued_at_utc": "2026-07-14T00:00:00Z"}, "issued after valuation"),
        ({"data_cutoff_utc": "2026-07-12T01:00:00Z"}, "not point-in-time"),
    ],
)
def test_governance_receipt_rejects_non_pit_authority_timing(
    tmp_path: Path,
    keys: tuple[Path, Path],
    overrides: dict[str, str],
    message: str,
) -> None:
    artifact = tmp_path / "decision.json"
    artifact.write_text("decision", encoding="utf-8")
    receipt = _receipt(artifact, keys[0], **overrides)

    with pytest.raises(ModelGovernanceAuthenticationError, match=message):
        verify_model_governance_artifact_receipt(
            receipt,
            artifact_path=artifact,
            expected_role="selected_lambda_decision",
            valuation_timestamp="2026-07-13T00:00:00Z",
        )


def test_governance_receipt_rejects_role_replay(
    tmp_path: Path,
    keys: tuple[Path, Path],
) -> None:
    artifact = tmp_path / "thresholds.csv"
    artifact.write_text("gate_id,status\na,PASS\n", encoding="utf-8")
    receipt = _receipt(artifact, keys[0], artifact_role="selected_lambda_decision")

    with pytest.raises(ModelGovernanceAuthenticationError, match="role mismatch"):
        verify_model_governance_artifact_receipt(
            receipt,
            artifact_path=artifact,
            expected_role="historical_thresholds",
            valuation_timestamp="2026-07-13T00:00:00Z",
        )


def test_governance_receipt_rejects_signature_mutation(
    tmp_path: Path,
    keys: tuple[Path, Path],
) -> None:
    artifact = tmp_path / "decision.json"
    artifact.write_text("decision", encoding="utf-8")
    receipt = _receipt(artifact, keys[0])
    receipt["authority_id"] = "ATTACKER"

    with pytest.raises(ModelGovernanceAuthenticationError, match="receipt_id mismatch"):
        verify_model_governance_artifact_receipt(
            receipt,
            artifact_path=artifact,
            expected_role="selected_lambda_decision",
            valuation_timestamp="2026-07-13T00:00:00Z",
        )


def test_selected_lambda_decision_rejects_post_run_binding() -> None:
    decision = {
        "schema_version": "monthly_curve_selected_lambda_decision.v1",
        "decision_id": "decision-001",
        "calibration_campaign_id": "campaign-001",
        "selection_reason": "rolling-origin holdout",
        "selection_status": "PRODUCTION_APPROVED",
        "production_approved": True,
        "canonical_config": {"lambda_prior": 1.0},
        "config_hash": "invalid-until-hash-check",
        "monthly_solution_hash": "post-hoc",
    }

    with pytest.raises(ModelGovernanceAuthenticationError, match="post-run fields"):
        validate_selected_lambda_decision(decision)


def test_selected_lambda_decision_must_match_active_config() -> None:
    from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash

    canonical = {"lambda_prior": 1.0}
    decision = {
        "schema_version": "monthly_curve_selected_lambda_decision.v1",
        "decision_id": "decision-001",
        "calibration_campaign_id": "campaign-001",
        "selection_reason": "rolling-origin holdout",
        "selection_status": "PRODUCTION_APPROVED",
        "production_approved": True,
        "canonical_config": canonical,
        "config_hash": config_hash(canonical),
    }

    with pytest.raises(ModelGovernanceAuthenticationError, match="active solver config"):
        validate_selected_lambda_decision(
            decision,
            expected_config={"lambda_prior": 2.0},
        )


def test_signed_thresholds_cannot_exceed_authorized_data_cutoff(
    tmp_path: Path,
    keys: tuple[Path, Path],
) -> None:
    rows: list[dict[str, object]] = []
    for gate_id, metric in (
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        (
            "residual_vs_implied_comparable_block",
            "comparable_block_shape_delta_abs_eur_mwh",
        ),
    ):
        for bucket in ["all", *(f"month_{month:02d}" for month in range(1, 13))]:
            rows.append(
                {
                    "gate_id": gate_id,
                    "metric": metric,
                    "market": "CH",
                    "delivery_bucket": bucket,
                    "lookback_start": "2098-01-01",
                    "lookback_end": "2099-01-01",
                    "n_snapshots": 0,
                    "min_required_n": 24,
                    "p50": None,
                    "p90": None,
                    "p975": None,
                    "max_observed": None,
                    "regime_filter": "test",
                    "status": "UNSUPPORTED",
                }
            )
    artifact = tmp_path / "historical_thresholds.csv"
    pd.DataFrame(rows).to_csv(artifact, index=False)
    receipt = _receipt(
        artifact,
        keys[0],
        artifact_role="historical_thresholds",
        data_cutoff_utc="2026-07-11T00:00:00Z",
    )

    with pytest.raises(ModelGovernanceAuthenticationError, match="lookback exceeds"):
        verify_model_governance_artifact_receipt(
            receipt,
            artifact_path=artifact,
            expected_role="historical_thresholds",
            valuation_timestamp="2026-07-13T00:00:00Z",
        )


def test_signed_thresholds_pass_requires_samples_and_ordered_statistics(
    tmp_path: Path,
    keys: tuple[Path, Path],
) -> None:
    rows: list[dict[str, object]] = []
    for gate_id, metric in (
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        (
            "residual_vs_implied_comparable_block",
            "comparable_block_shape_delta_abs_eur_mwh",
        ),
    ):
        for bucket in ["all", *(f"month_{month:02d}" for month in range(1, 13))]:
            rows.append(
                {
                    "gate_id": gate_id,
                    "metric": metric,
                    "market": "CH",
                    "delivery_bucket": bucket,
                    "lookback_start": "2020-01-01",
                    "lookback_end": "2026-07-01",
                    "n_snapshots": 0,
                    "min_required_n": 24,
                    "p50": 1e12,
                    "p90": 1e12,
                    "p975": 1e12,
                    "max_observed": 1e12,
                    "regime_filter": "test",
                    "status": "PASS",
                }
            )
    artifact = tmp_path / "historical_thresholds.csv"
    pd.DataFrame(rows).to_csv(artifact, index=False)
    receipt = _receipt(
        artifact,
        keys[0],
        artifact_role="historical_thresholds",
        data_cutoff_utc="2026-07-11T00:00:00Z",
    )

    with pytest.raises(ModelGovernanceAuthenticationError, match="insufficient samples"):
        verify_model_governance_artifact_receipt(
            receipt,
            artifact_path=artifact,
            expected_role="historical_thresholds",
            valuation_timestamp="2026-07-13T00:00:00Z",
        )
