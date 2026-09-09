from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.data.snapshot_bootstrap_authority import (
    PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_ENV,
)
from pfc_shaping.data.snapshot_publication_contract import (
    BOOTSTRAP_REQUIRED_SOURCE_CLASS,
    BOOTSTRAP_TRANSITION_TYPE,
    BOOTSTRAP_USE_POLICY,
    PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
    PUBLICATION_DOMAIN_ID_ENV,
    publication_domain_sha256,
    verify_snapshot_bootstrap_authorization,
)
from pfc_shaping.data.snapshot_publication_state import canonical_json
from pfc_shaping.durable_artifact import (
    DurableArtifactRepairRequired,
    durable_artifact_durability_classification,
)
from scripts.sign_snapshot_bootstrap_authorization import sign_authorization


def test_offline_bootstrap_signer_stages_truncation_repair_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path, public_path = _keypair(tmp_path)
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    monkeypatch.setenv(
        PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_ENV,
        str(private_path),
    )
    input_path = tmp_path / "unsigned-bootstrap.json"
    output_path = tmp_path / "signed-bootstrap.json"
    input_path.write_bytes(canonical_json(_authorization_payload()))
    output_path.write_bytes(b"{")

    with pytest.raises(DurableArtifactRepairRequired) as first_error:
        sign_authorization(input_path, output_path)
    first = first_error.value.result
    candidate = Path(str(first["repair_candidate_path"]))
    first_payload = candidate.read_bytes()
    verified = verify_snapshot_bootstrap_authorization(
        json.loads(first_payload),
        trusted_public_key_path=public_path,
    )
    with pytest.raises(DurableArtifactRepairRequired) as second_error:
        sign_authorization(input_path, output_path)
    second = second_error.value.result

    assert first["schema_version"] == "durable_artifact_repair.v2"
    assert first["status"] == "LOCAL_ARTIFACT_REPAIR_REQUIRED"
    assert first["reason"] == "AUTHORITATIVE_BYTES_DIFFER_FROM_EXISTING_TARGET"
    assert first["durability_classification"] == (
        durable_artifact_durability_classification()
    )
    assert first == second
    assert output_path.read_bytes() == b"{"
    assert candidate.read_bytes() == first_payload
    assert len(str(verified["authorization_id"])) == 64
    assert first["automatic_replacement_performed"] is False
    assert first["repair_candidate_sha256"] == first["authoritative_sha256"]

    other = _authorization_payload()
    other["operation_id"] = "20000000-0000-4000-8000-000000000009"
    input_path.write_bytes(canonical_json(other))
    with pytest.raises(DurableArtifactRepairRequired) as other_error:
        sign_authorization(input_path, output_path)
    assert other_error.value.result["repair_candidate_path"] != str(candidate)
    assert output_path.read_bytes() == b"{"


def _authorization_payload() -> dict[str, object]:
    return {
        "schema_version": PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
        "publication_domain_sha256": publication_domain_sha256(),
        "transition_type": BOOTSTRAP_TRANSITION_TYPE,
        "use_policy": BOOTSTRAP_USE_POLICY,
        "operation_id": "10000000-0000-4000-8000-000000000009",
        "expected_anchor_sequence": 0,
        "expected_anchor_receipt_id": None,
        "legacy_pointer_sha256": "9" * 64,
        "generation_id": "migrated-seed",
        "contract_path": "snapshots/migrated-seed/lt_input_snapshot.json",
        "contract_sha256": "7" * 64,
        "generation_inventory_sha256": "8" * 64,
        "required_source_class": BOOTSTRAP_REQUIRED_SOURCE_CLASS,
        "required_calibration_eligible": False,
        "issued_at_utc": "2026-07-16T08:59:00+00:00",
        "expires_at_utc": "2026-07-16T09:05:00+00:00",
    }


def _keypair(root: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = root / "bootstrap-private.pem"
    public_path = root / "bootstrap-public.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path
