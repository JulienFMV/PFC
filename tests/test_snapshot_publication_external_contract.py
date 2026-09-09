from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.data import snapshot_publication_state as state_module
from pfc_shaping.data.snapshot_anchor_signing import (
    sign_snapshot_anchor_head_observation,
    sign_snapshot_anchor_receipt,
)
from pfc_shaping.data.snapshot_bootstrap_authority import (
    sign_snapshot_bootstrap_authorization,
)
from pfc_shaping.data.snapshot_publication_contract import (
    BOOTSTRAP_REQUIRED_SOURCE_CLASS,
    BOOTSTRAP_TRANSITION_TYPE,
    BOOTSTRAP_USE_POLICY,
    PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV,
    PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
    PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
    PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_DIR_ENV,
    PUBLICATION_DOMAIN_ID_ENV,
    PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV,
    SnapshotPublicationAuthenticationError,
    publication_domain_sha256,
    sign_snapshot_publication_intent,
    verify_snapshot_anchor_head_observation,
    verify_snapshot_anchor_receipt,
    verify_snapshot_bootstrap_intent_authorization,
    verify_snapshot_publication_intent,
)
from pfc_shaping.data.snapshot_publication_state import (
    SnapshotPublicationStateError,
    external_head_challenge_sha256,
    pointer_mapping_from_external_receipt,
    verify_external_publication_evidence,
    verify_publication_authority_separation,
)


@pytest.fixture(autouse=True)
def _domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "5338d76a-642e-4648-ac2d-a932844ec8cc",
    )
    monkeypatch.setattr(
        state_module,
        "_reference_utc",
        lambda _: datetime.fromisoformat("2026-07-16T09:03:00+00:00"),
    )


def test_external_publication_contracts_round_trip_with_distinct_authorities(
    tmp_path: Path,
) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    intent = sign_snapshot_publication_intent(
        _intent_payload(),
        private_key_path=request_private,
    )
    receipt = sign_snapshot_anchor_receipt(
        _receipt_payload(intent),
        private_key_path=anchor_private,
    )
    observation = sign_snapshot_anchor_head_observation(
        _observation_payload(receipt),
        private_key_path=anchor_private,
    )

    assert (
        verify_snapshot_publication_intent(
            intent,
            trusted_public_key_path=request_public,
        )["intent_id"]
        == intent["intent_id"]
    )
    assert (
        verify_snapshot_anchor_receipt(
            receipt,
            trusted_public_key_path=anchor_public,
        )["intent_id"]
        == intent["intent_id"]
    )
    assert (
        verify_snapshot_anchor_head_observation(
            observation,
            trusted_public_key_path=anchor_public,
        )["receipt_id"]
        == receipt["receipt_id"]
    )

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="not trusted",
    ):
        verify_snapshot_anchor_receipt(
            receipt,
            trusted_public_key_path=request_public,
        )


def test_historical_anchor_key_is_verified_from_sealed_keyring(tmp_path: Path) -> None:
    old_private, old_public = _keypair(tmp_path / "old-anchor")
    _, active_public = _keypair(tmp_path / "active-anchor")
    keyring = tmp_path / "anchor-keyring"
    keyring.mkdir()
    (keyring / "old.pem").write_bytes(old_public.read_bytes())
    intent = sign_snapshot_publication_intent(
        _intent_payload(),
        private_key_path=_keypair(tmp_path / "request")[0],
    )
    receipt = sign_snapshot_anchor_receipt(
        _receipt_payload(intent),
        private_key_path=old_private,
    )

    verified = verify_snapshot_anchor_receipt(
        receipt,
        trusted_public_key_path=active_public,
        trusted_public_key_dir=keyring,
    )

    assert verified["receipt_id"] == receipt["receipt_id"]


def test_normal_publication_cannot_bind_legacy_pointer(tmp_path: Path) -> None:
    private, _ = _keypair(tmp_path / "request")
    payload = _intent_payload()
    payload["legacy_pointer_sha256"] = "a" * 64

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="normal publication cannot bind legacy bootstrap evidence",
    ):
        sign_snapshot_publication_intent(payload, private_key_path=private)


def test_legacy_adoption_is_rejected_even_when_signed_directly(
    tmp_path: Path,
) -> None:
    private, _ = _keypair(tmp_path / "request")
    payload = _intent_payload()
    payload.update(
        {
            "transition_type": "ADOPT_LEGACY",
            "legacy_pointer_sha256": "b" * 64,
            "expected_anchor_sequence": 0,
            "expected_anchor_receipt_id": None,
        }
    )

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="legacy adoption is forbidden",
    ):
        sign_snapshot_publication_intent(payload, private_key_path=private)


def test_bootstrap_requires_exact_independent_it_authorization(tmp_path: Path) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    bootstrap_private, bootstrap_public = _keypair(tmp_path / "bootstrap")
    authorization = sign_snapshot_bootstrap_authorization(
        _bootstrap_authorization_payload(),
        private_key_path=bootstrap_private,
    )
    intent_payload = _bootstrap_intent_payload(authorization)
    intent = sign_snapshot_publication_intent(
        intent_payload,
        private_key_path=request_private,
        bootstrap_trusted_public_key_path=bootstrap_public,
    )

    verified = verify_snapshot_publication_intent(
        intent,
        trusted_public_key_path=request_public,
        bootstrap_trusted_public_key_path=bootstrap_public,
    )
    verified_authorization = verify_snapshot_bootstrap_intent_authorization(
        intent,
        trusted_public_key_path=bootstrap_public,
        admitted_at_utc="2026-07-16T09:02:00+00:00",
    )

    assert verified["transition_type"] == BOOTSTRAP_TRANSITION_TYPE
    assert verified_authorization["authorization_id"] == authorization["authorization_id"]


def test_explicit_no_bootstrap_keyring_does_not_fall_back_to_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    rogue_private, rogue_public = _keypair(tmp_path / "rogue-bootstrap")
    _, active_public = _keypair(tmp_path / "active-bootstrap")
    keyring = tmp_path / "bootstrap-keyring"
    keyring.mkdir()
    (keyring / "rogue.pem").write_bytes(rogue_public.read_bytes())
    monkeypatch.setenv(PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_DIR_ENV, str(keyring))
    authorization = sign_snapshot_bootstrap_authorization(
        _bootstrap_authorization_payload(),
        private_key_path=rogue_private,
    )
    intent = sign_snapshot_publication_intent(
        _bootstrap_intent_payload(authorization),
        private_key_path=request_private,
        bootstrap_trusted_public_key_path=rogue_public,
        bootstrap_trusted_public_key_dir=None,
    )

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="signing key is not trusted",
    ):
        verify_snapshot_publication_intent(
            intent,
            trusted_public_key_path=request_public,
            bootstrap_trusted_public_key_path=active_public,
            bootstrap_trusted_public_key_dir=None,
        )


def test_bootstrap_rejects_wrong_pointer_binding(tmp_path: Path) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    bootstrap_private, bootstrap_public = _keypair(tmp_path / "bootstrap")
    authorization = sign_snapshot_bootstrap_authorization(
        _bootstrap_authorization_payload(),
        private_key_path=bootstrap_private,
    )
    payload = _bootstrap_intent_payload(authorization)
    payload["legacy_pointer_sha256"] = "f" * 64

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="does not bind the exact intent",
    ):
        sign_snapshot_publication_intent(
            payload,
            private_key_path=request_private,
            bootstrap_trusted_public_key_path=bootstrap_public,
        )


def test_bootstrap_rejects_expired_authorization_at_admission(tmp_path: Path) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    bootstrap_private, bootstrap_public = _keypair(tmp_path / "bootstrap")
    authorization = sign_snapshot_bootstrap_authorization(
        _bootstrap_authorization_payload(),
        private_key_path=bootstrap_private,
    )
    intent = sign_snapshot_publication_intent(
        _bootstrap_intent_payload(authorization),
        private_key_path=request_private,
        bootstrap_trusted_public_key_path=bootstrap_public,
    )

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="not valid at admission",
    ):
        verify_snapshot_bootstrap_intent_authorization(
            intent,
            trusted_public_key_path=bootstrap_public,
            admitted_at_utc="2026-07-16T09:06:00+00:00",
        )


def test_bootstrap_rejects_request_signer_as_it_authority(tmp_path: Path) -> None:
    shared_private, shared_public = _keypair(tmp_path / "shared")
    authorization = sign_snapshot_bootstrap_authorization(
        _bootstrap_authorization_payload(),
        private_key_path=shared_private,
    )

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="request and IT bootstrap signer identities overlap",
    ):
        sign_snapshot_publication_intent(
            _bootstrap_intent_payload(authorization),
            private_key_path=shared_private,
            bootstrap_trusted_public_key_path=shared_public,
        )


def test_bootstrap_authorization_cannot_make_legacy_seed_eligible(tmp_path: Path) -> None:
    bootstrap_private, _ = _keypair(tmp_path / "bootstrap")
    payload = _bootstrap_authorization_payload()
    payload["required_calibration_eligible"] = True

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="must preserve calibration_eligible=false",
    ):
        sign_snapshot_bootstrap_authorization(
            payload,
            private_key_path=bootstrap_private,
        )


def test_receipt_rejects_predecessor_sequence_divergence(tmp_path: Path) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    anchor_private, _ = _keypair(tmp_path / "anchor")
    intent = sign_snapshot_publication_intent(
        _intent_payload(),
        private_key_path=request_private,
    )
    payload = _receipt_payload(intent)
    payload["previous_receipt_id"] = None

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="predecessor/sequence mismatch",
    ):
        sign_snapshot_anchor_receipt(payload, private_key_path=anchor_private)


def test_head_observation_rejects_nonpositive_lifetime(tmp_path: Path) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    anchor_private, _ = _keypair(tmp_path / "anchor")
    intent = sign_snapshot_publication_intent(
        _intent_payload(),
        private_key_path=request_private,
    )
    receipt = sign_snapshot_anchor_receipt(
        _receipt_payload(intent),
        private_key_path=anchor_private,
    )
    payload = _observation_payload(receipt)
    payload["expires_at_utc"] = payload["observed_at_utc"]

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match="expiry is invalid",
    ):
        sign_snapshot_anchor_head_observation(payload, private_key_path=anchor_private)


def test_contract_rejects_noncanonical_utc_timestamp(tmp_path: Path) -> None:
    private, _ = _keypair(tmp_path / "request")
    payload = _intent_payload()
    payload["operation_created_at_utc"] = "2026-07-16T09:00:00Z"

    with pytest.raises(
        SnapshotPublicationAuthenticationError,
        match=r"canonical \+00:00 UTC",
    ):
        sign_snapshot_publication_intent(payload, private_key_path=private)


def test_external_state_verifies_exact_cas_and_fresh_head_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(request_public))
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    intent = sign_snapshot_publication_intent(
        _intent_payload(),
        private_key_path=request_private,
    )
    intent_payload = _canonical(intent)
    receipt = sign_snapshot_anchor_receipt(
        _receipt_payload(intent),
        private_key_path=anchor_private,
    )
    receipt_payload = _canonical(receipt)
    pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=_sha256(receipt_payload),
    )
    observation_payload = _signed_observation_payload(
        receipt,
        receipt_payload=receipt_payload,
        challenge_sha256=external_head_challenge_sha256(
            pointer,
            challenge_nonce="4" * 64,
        ),
        private_key=anchor_private,
    )

    verified = verify_external_publication_evidence(
        intent_payload=intent_payload,
        receipt_payload=receipt_payload,
        observation_payload=observation_payload,
        pointer=pointer,
        require_current=True,
        expected_challenge_nonce="4" * 64,
    )

    assert verified["status"] == "VERIFIED_EXTERNAL_CAS_PUBLICATION"
    assert verified["receipt_id"] == receipt["receipt_id"]


def test_external_state_rejects_signed_observation_for_another_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _external_state_fixture(tmp_path, monkeypatch)
    wrong_observation = _signed_observation_payload(
        fixture["receipt"],
        receipt_payload=fixture["receipt_payload"],
        challenge_sha256="f" * 64,
        private_key=fixture["anchor_private"],
    )

    with pytest.raises(
        SnapshotPublicationStateError,
        match="does not select the receipt projection",
    ):
        verify_external_publication_evidence(
            intent_payload=fixture["intent_payload"],
            receipt_payload=fixture["receipt_payload"],
            observation_payload=wrong_observation,
            pointer=fixture["pointer"],
            require_current=True,
            expected_challenge_nonce="4" * 64,
        )


def test_external_state_rejects_expired_head_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _external_state_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        state_module,
        "_reference_utc",
        lambda _: datetime.fromisoformat("2026-07-16T09:06:00+00:00"),
    )

    with pytest.raises(SnapshotPublicationStateError, match="observation is expired"):
        verify_external_publication_evidence(
            intent_payload=fixture["intent_payload"],
            receipt_payload=fixture["receipt_payload"],
            observation_payload=fixture["observation_payload"],
            pointer=fixture["pointer"],
            require_current=True,
            expected_challenge_nonce="4" * 64,
        )


def test_current_head_requires_active_observation_key_after_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    old_anchor_private, old_anchor_public = _keypair(tmp_path / "old-anchor")
    _, active_anchor_public = _keypair(tmp_path / "active-anchor")
    keyring = tmp_path / "anchor-keyring"
    keyring.mkdir()
    (keyring / "old.pem").write_bytes(old_anchor_public.read_bytes())
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(request_public))
    monkeypatch.setenv(
        PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
        str(active_anchor_public),
    )
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV, str(keyring))
    intent = sign_snapshot_publication_intent(
        _intent_payload(),
        private_key_path=request_private,
    )
    intent_payload = _canonical(intent)
    receipt = sign_snapshot_anchor_receipt(
        _receipt_payload(intent),
        private_key_path=old_anchor_private,
    )
    receipt_payload = _canonical(receipt)
    pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=_sha256(receipt_payload),
    )
    old_observation = _signed_observation_payload(
        receipt,
        receipt_payload=receipt_payload,
        challenge_sha256=external_head_challenge_sha256(
            pointer,
            challenge_nonce="4" * 64,
        ),
        private_key=old_anchor_private,
    )

    with pytest.raises(
        SnapshotPublicationStateError,
        match="authentication failed",
    ):
        verify_external_publication_evidence(
            intent_payload=intent_payload,
            receipt_payload=receipt_payload,
            observation_payload=old_observation,
            pointer=pointer,
            require_current=True,
            expected_challenge_nonce="4" * 64,
        )


def test_external_state_rejects_acquisition_signer_reused_by_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _external_state_fixture(tmp_path, monkeypatch)
    intent = json.loads(fixture["intent_payload"])
    receipt = json.loads(fixture["receipt_payload"])
    observation = json.loads(fixture["observation_payload"])
    contract = {
        "acquisition_attestation": {
            "key_id": intent["signature"]["key_id"],
        }
    }

    with pytest.raises(
        SnapshotPublicationStateError,
        match="acquisition and publication authority signer identities overlap",
    ):
        verify_publication_authority_separation(
            contract,
            intent=intent,
            receipt=receipt,
            observation=observation,
        )


def _intent_payload() -> dict[str, object]:
    return {
        "schema_version": "lt_snapshot_publication_intent.v1",
        "publication_domain_sha256": publication_domain_sha256(),
        "operation_id": "10000000-0000-4000-8000-000000000001",
        "operation_created_at_utc": "2026-07-16T09:00:00+00:00",
        "transition_type": "PUBLISH",
        "expected_anchor_receipt_id": "0" * 64,
        "expected_anchor_sequence": 1,
        "generation_id": "generation-001",
        "contract_path": "snapshots/generation-001/lt_input_snapshot.json",
        "contract_sha256": "1" * 64,
        "generation_inventory_sha256": "2" * 64,
        "legacy_pointer_sha256": None,
        "bootstrap_authorization": None,
        "bootstrap_authorization_id": None,
        "bootstrap_authorization_sha256": None,
    }


def _bootstrap_authorization_payload() -> dict[str, object]:
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


def _bootstrap_intent_payload(
    authorization: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "lt_snapshot_publication_intent.v1",
        "publication_domain_sha256": publication_domain_sha256(),
        "operation_id": authorization["operation_id"],
        "operation_created_at_utc": "2026-07-16T09:00:00+00:00",
        "transition_type": BOOTSTRAP_TRANSITION_TYPE,
        "expected_anchor_receipt_id": None,
        "expected_anchor_sequence": 0,
        "generation_id": authorization["generation_id"],
        "contract_path": authorization["contract_path"],
        "contract_sha256": authorization["contract_sha256"],
        "generation_inventory_sha256": authorization["generation_inventory_sha256"],
        "legacy_pointer_sha256": authorization["legacy_pointer_sha256"],
        "bootstrap_authorization": authorization,
        "bootstrap_authorization_id": authorization["authorization_id"],
        "bootstrap_authorization_sha256": _sha256(_canonical(authorization)),
    }


def _receipt_payload(intent: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "lt_snapshot_anchor_receipt.v1",
        "publication_domain_sha256": publication_domain_sha256(),
        "sequence": 2,
        "previous_receipt_id": intent["expected_anchor_receipt_id"],
        "operation_id": intent["operation_id"],
        "intent_id": intent["intent_id"],
        "intent_sha256": _sha256(_canonical(intent)),
        "transition_type": intent["transition_type"],
        "generation_id": intent["generation_id"],
        "contract_sha256": intent["contract_sha256"],
        "generation_inventory_sha256": intent["generation_inventory_sha256"],
        "legacy_pointer_sha256": intent["legacy_pointer_sha256"],
        "bootstrap_authorization_id": intent["bootstrap_authorization_id"],
        "bootstrap_authorization_sha256": intent["bootstrap_authorization_sha256"],
        "committed_at_utc": "2026-07-16T09:00:01+00:00",
    }


def _observation_payload(receipt: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "lt_snapshot_anchor_head_observation.v1",
        "publication_domain_sha256": publication_domain_sha256(),
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": _sha256(_canonical(receipt)),
        "sequence": receipt["sequence"],
        "challenge_nonce": "4" * 64,
        "challenge_sha256": "3" * 64,
        "observed_at_utc": "2026-07-16T09:00:02+00:00",
        "expires_at_utc": "2026-07-16T09:05:02+00:00",
    }


def _signed_observation_payload(
    receipt: dict[str, object],
    *,
    receipt_payload: bytes,
    challenge_sha256: str,
    private_key: Path,
) -> bytes:
    payload = _observation_payload(receipt)
    payload["receipt_sha256"] = _sha256(receipt_payload)
    payload["challenge_sha256"] = challenge_sha256
    return _canonical(
        sign_snapshot_anchor_head_observation(
            payload,
            private_key_path=private_key,
        )
    )


def _external_state_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    request_private, request_public = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(request_public))
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    intent = sign_snapshot_publication_intent(
        _intent_payload(),
        private_key_path=request_private,
    )
    intent_payload = _canonical(intent)
    receipt = sign_snapshot_anchor_receipt(
        _receipt_payload(intent),
        private_key_path=anchor_private,
    )
    receipt_payload = _canonical(receipt)
    pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=_sha256(receipt_payload),
    )
    observation_payload = _signed_observation_payload(
        receipt,
        receipt_payload=receipt_payload,
        challenge_sha256=external_head_challenge_sha256(
            pointer,
            challenge_nonce="4" * 64,
        ),
        private_key=anchor_private,
    )
    return {
        "anchor_private": anchor_private,
        "intent_payload": intent_payload,
        "receipt": receipt,
        "receipt_payload": receipt_payload,
        "pointer": pointer,
        "observation_payload": observation_payload,
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _keypair(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    key = Ed25519PrivateKey.generate()
    private = root / "private.pem"
    public = root / "public.pem"
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
