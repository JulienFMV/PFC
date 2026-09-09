from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.pipeline.release_request_contract import (
    RELEASE_REQUEST_SCHEMA,
    ReleaseRequestAuthenticationError,
    ensure_workflow_domain,
    load_registered_release_request,
    sign_release_request,
)


def _key_pair(root: Path, name: str) -> tuple[Path, Path]:
    key = Ed25519PrivateKey.generate()
    private = root / f"{name}-private.pem"
    public = root / f"{name}-public.pem"
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
    return private, public


def _registered_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object]]:
    private, public = _key_pair(tmp_path, "register")
    monkeypatch.setenv(
        "PFC_RELEASE_WORKFLOW_DOMAIN_ID", "123e4567-e89b-42d3-a456-426614174010"
    )
    monkeypatch.setenv("PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH", str(public))
    workflow, domain_sha256 = ensure_workflow_domain(tmp_path / "workflow", create=True)
    request = sign_release_request(
        {
            "schema_version": RELEASE_REQUEST_SCHEMA,
            "run_id": "run-001",
            "registration_contract": "assembled_candidate_seal.v1",
            "workflow_domain_sha256": domain_sha256,
        },
        private_key_path=private,
    )
    path = (
        workflow
        / "run-001"
        / "requests"
        / str(request["request_id"])[:32]
        / "release_request.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    return workflow, path, request


def test_signed_request_loads_only_from_canonical_workflow_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, path, request = _registered_request(tmp_path, monkeypatch)

    assert load_registered_release_request(
        path, workflow_root=workflow, expected_run_id="run-001"
    ) == request


def test_request_payload_tamper_invalidates_register_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, path, request = _registered_request(tmp_path, monkeypatch)
    request["registration_contract"] = "forged"
    path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ReleaseRequestAuthenticationError, match="request_id mismatch"):
        load_registered_release_request(
            path, workflow_root=workflow, expected_run_id="run-001"
        )


def test_foreign_register_authority_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, path, request = _registered_request(tmp_path, monkeypatch)
    foreign_private, _ = _key_pair(tmp_path, "foreign")
    forged = sign_release_request(request, private_key_path=foreign_private)
    path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(ReleaseRequestAuthenticationError, match="signing key is not trusted"):
        load_registered_release_request(
            path, workflow_root=workflow, expected_run_id="run-001"
        )


def test_workflow_domain_configuration_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, path, _ = _registered_request(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "PFC_RELEASE_WORKFLOW_DOMAIN_ID", "123e4567-e89b-42d3-a456-426614174011"
    )

    with pytest.raises(ReleaseRequestAuthenticationError, match="does not match pinned marker"):
        load_registered_release_request(
            path, workflow_root=workflow, expected_run_id="run-001"
        )


def test_missing_workflow_domain_is_not_recreated_after_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "workflow"
    (workflow / "run-001" / "requests").mkdir(parents=True)
    monkeypatch.setenv(
        "PFC_RELEASE_WORKFLOW_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174010",
    )

    with pytest.raises(
        ReleaseRequestAuthenticationError,
        match="missing after workflow activity",
    ):
        ensure_workflow_domain(workflow, create=True)

    assert not (workflow / "workflow_domain.json").exists()


def test_signed_request_relocated_outside_workflow_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, path, request = _registered_request(tmp_path, monkeypatch)
    relocated = tmp_path / "alternate" / "release_request.json"
    relocated.parent.mkdir()
    relocated.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ReleaseRequestAuthenticationError, match="canonical workflow-domain"):
        load_registered_release_request(
            relocated, workflow_root=workflow, expected_run_id="run-001"
        )
