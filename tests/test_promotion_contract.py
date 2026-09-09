from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.pipeline.promotion_contract as promotion_contract
from pfc_shaping.pipeline.promotion_contract import (
    PROMOTION_POLICY_VERSION,
    ReceiptAuthenticationError,
    RollbackAuthorizationAuthenticationError,
    public_key_id_from_private_key_path,
    public_key_id_from_public_key_path,
    sign_promotion_receipt,
    sign_rollback_authorization,
    validate_required_gate_set,
    verify_rollback_authorization_signature,
)


def _key_pair(tmp_path: Path, name: str) -> tuple[Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / f"{name}.private.pem"
    public_path = tmp_path / f"{name}.public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def _payload() -> dict[str, object]:
    return {
        "schema_version": "rollback_authorization.v1",
        "operation_id": "rollback-op-001",
        "release_root_sha256": "4" * 64,
        "expected_current_event_id": "1" * 64,
        "target_promotion_event_id": "2" * 64,
        "target_run_id": "run-001",
        "target_bundle_manifest_sha256": "3" * 64,
        "reason_code": "QUALITY_REGRESSION",
        "incident_id": "INC-2026-001",
        "authorized_at_utc": "2026-07-14T10:00:00+00:00",
        "expires_at_utc": "2026-07-14T10:15:00+00:00",
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


@pytest.mark.parametrize("policy_version", [None, "lt_promotion_policy.v0"])
def test_promotion_receipt_signing_rejects_unsupported_policy_version(
    tmp_path: Path,
    policy_version: str | None,
) -> None:
    private_path, _ = _key_pair(tmp_path, "promotion")
    payload: dict[str, object] = {"schema_version": "promotion_receipt.v2"}
    if policy_version is not None:
        payload["promotion_policy_version"] = policy_version

    with pytest.raises(ReceiptAuthenticationError, match="policy version is unsupported"):
        sign_promotion_receipt(payload, private_key_path=private_path)


def test_gate_validation_is_bound_to_deterministic_only_policy() -> None:
    with pytest.raises(ReceiptAuthenticationError, match="policy version is unsupported"):
        validate_required_gate_set(
            {
                "promotion_policy_version": "lt_promotion_policy.v0",
                "governance_gates": [],
            }
        )

    assert PROMOTION_POLICY_VERSION == "lt_promotion_policy.deterministic_only.v1"


def test_historical_policy_is_replay_only(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = {
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "governance_gates": [
            {"gate_id": gate_id, "status": "PASS"}
            for gate_id in sorted(promotion_contract.REQUIRED_PROMOTION_GATE_IDS)
        ],
    }
    future = "lt_promotion_policy.deterministic_only.v2"
    monkeypatch.setattr(promotion_contract, "PROMOTION_POLICY_VERSION", future)
    monkeypatch.setitem(
        promotion_contract.PROMOTION_GATE_IDS_BY_POLICY,
        future,
        promotion_contract.REQUIRED_PROMOTION_GATE_IDS,
    )

    with pytest.raises(ReceiptAuthenticationError, match="is not current"):
        validate_required_gate_set(receipt)
    validate_required_gate_set(receipt, allow_historical_policy=True)


def _sign_unchecked(payload: dict[str, object], private_path: Path) -> dict[str, object]:
    private_key = serialization.load_pem_private_key(
        private_path.read_bytes(),
        password=None,
    )
    assert isinstance(private_key, Ed25519PrivateKey)
    authorization_id = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    signed = {**payload, "authorization_id": authorization_id}
    signed["signature"] = {
        "algorithm": "Ed25519",
        "key_id": public_key_id_from_private_key_path(private_path),
        "value_base64": base64.b64encode(
            private_key.sign(_canonical_json_bytes(signed))
        ).decode("ascii"),
    }
    return signed


def test_rollback_authorization_round_trip_and_key_ids(tmp_path: Path) -> None:
    private_path, public_path = _key_pair(tmp_path, "rollback")

    signed = sign_rollback_authorization(
        _payload(),
        private_key_path=private_path,
    )
    verified = verify_rollback_authorization_signature(
        signed,
        trusted_public_key_path=public_path,
    )

    assert verified == {key: value for key, value in signed.items() if key != "signature"}
    assert len(str(verified["authorization_id"])) == 64
    assert public_key_id_from_private_key_path(private_path) == (
        public_key_id_from_public_key_path(public_path)
    )


def test_rollback_authorization_rejects_tampered_payload(tmp_path: Path) -> None:
    private_path, public_path = _key_pair(tmp_path, "rollback")
    signed = sign_rollback_authorization(_payload(), private_key_path=private_path)
    signed["incident_id"] = "INC-SUBSTITUTED"

    with pytest.raises(
        RollbackAuthorizationAuthenticationError,
        match="authorization_id does not match canonical payload",
    ):
        verify_rollback_authorization_signature(
            signed,
            trusted_public_key_path=public_path,
        )


def test_rollback_authorization_rejects_wrong_trusted_key(tmp_path: Path) -> None:
    private_path, _ = _key_pair(tmp_path, "rollback")
    _, wrong_public_path = _key_pair(tmp_path, "wrong")
    signed = sign_rollback_authorization(_payload(), private_key_path=private_path)

    with pytest.raises(
        RollbackAuthorizationAuthenticationError,
        match="signing key is not trusted",
    ):
        verify_rollback_authorization_signature(
            signed,
            trusted_public_key_path=wrong_public_path,
        )


def test_rollback_authorization_rejects_tampered_signature(tmp_path: Path) -> None:
    private_path, public_path = _key_pair(tmp_path, "rollback")
    signed = sign_rollback_authorization(_payload(), private_key_path=private_path)
    signature = dict(signed["signature"])
    signature["value_base64"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    signed["signature"] = signature

    with pytest.raises(
        RollbackAuthorizationAuthenticationError,
        match="signature is invalid",
    ):
        verify_rollback_authorization_signature(
            signed,
            trusted_public_key_path=public_path,
        )


def test_rollback_authorization_rejects_extra_signature_field(tmp_path: Path) -> None:
    private_path, public_path = _key_pair(tmp_path, "rollback")
    signed = sign_rollback_authorization(_payload(), private_key_path=private_path)
    signature = dict(signed["signature"])
    signature["comment"] = "not-covered-by-the-signature"
    signed["signature"] = signature

    with pytest.raises(
        RollbackAuthorizationAuthenticationError,
        match="signature block is invalid",
    ):
        verify_rollback_authorization_signature(
            signed,
            trusted_public_key_path=public_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "rollback_authorization.v2", "only rollback_authorization.v1"),
        ("expected_current_event_id", "A" * 64, "not a canonical SHA-256"),
        ("target_promotion_event_id", "2" * 63, "not a canonical SHA-256"),
        ("target_bundle_manifest_sha256", "not-a-hash", "not a canonical SHA-256"),
        ("reason_code", "", "reason_code is invalid"),
        ("authorized_at_utc", "2026-07-14T10:00:00", "timezone-naive"),
    ],
)
def test_sign_rollback_authorization_rejects_malformed_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    private_path, _ = _key_pair(tmp_path, "rollback")
    payload = _payload()
    payload[field] = value

    with pytest.raises(RollbackAuthorizationAuthenticationError, match=message):
        sign_rollback_authorization(payload, private_key_path=private_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "rollback_authorization.v2", "only rollback_authorization.v1"),
        (
            "target_bundle_manifest_sha256",
            "f" * 63,
            "target_bundle_manifest_sha256 is not a canonical SHA-256",
        ),
    ],
)
def test_verify_rollback_authorization_rejects_signed_malformed_payload(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    private_path, public_path = _key_pair(tmp_path, "rollback")
    payload = _payload()
    payload[field] = value
    signed = _sign_unchecked(payload, private_path)

    with pytest.raises(
        RollbackAuthorizationAuthenticationError,
        match=message,
    ):
        verify_rollback_authorization_signature(
            signed,
            trusted_public_key_path=public_path,
        )


def test_rollback_authorization_rejects_extra_payload_field(tmp_path: Path) -> None:
    private_path, _ = _key_pair(tmp_path, "rollback")
    payload = _payload()
    payload["override"] = True

    with pytest.raises(
        RollbackAuthorizationAuthenticationError,
        match="field set mismatch",
    ):
        sign_rollback_authorization(payload, private_key_path=private_path)
