from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.data.snapshot_anchor_reference import (
    REFERENCE_AUTHORITY_CLASSIFICATION,
    SnapshotAnchorReferenceAuthority,
    SnapshotAnchorReferenceConfig,
    SnapshotAnchorReferenceConflict,
    SnapshotAnchorReferenceError,
)
from pfc_shaping.data.snapshot_bootstrap_authority import (
    sign_snapshot_bootstrap_authorization,
)
from pfc_shaping.data.snapshot_publication_contract import (
    BOOTSTRAP_REQUIRED_SOURCE_CLASS,
    BOOTSTRAP_TRANSITION_TYPE,
    BOOTSTRAP_USE_POLICY,
    PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
    PUBLICATION_DOMAIN_ID_ENV,
    publication_domain_sha256,
    sign_snapshot_publication_intent,
    verify_snapshot_anchor_head_observation,
    verify_snapshot_anchor_receipt,
)
from pfc_shaping.data.snapshot_publication_state import (
    canonical_json,
    external_head_challenge_sha256,
    pointer_mapping_from_external_receipt,
)

_DOMAIN_ID = "de6bc6ac-1243-4f06-821b-c40ddb088edc"
_START = datetime(2026, 7, 16, 9, 0, tzinfo=timezone.utc)


def test_reference_authority_is_explicitly_non_production() -> None:
    assert REFERENCE_AUTHORITY_CLASSIFICATION == "NON_PRODUCTION_TEST_ONLY"
    assert SnapshotAnchorReferenceAuthority.NON_PRODUCTION is True


def test_reference_rejects_bootstrap_identity_reused_by_request_authority(
    authority_setup: AuthoritySetup,
) -> None:
    with pytest.raises(
        SnapshotAnchorReferenceError,
        match="trust roles must be globally disjoint",
    ):
        SnapshotAnchorReferenceAuthority(
            SnapshotAnchorReferenceConfig(
                database_path=authority_setup.database,
                request_public_key_path=authority_setup.request_public,
                anchor_private_key_path=authority_setup.anchor_private,
                bootstrap_public_key_path=authority_setup.request_public,
            )
        )


def test_reference_rejects_historical_keyring_overlap_across_authorities(
    authority_setup: AuthoritySetup,
    tmp_path: Path,
) -> None:
    bootstrap_keyring = tmp_path / "bootstrap-keyring"
    bootstrap_keyring.mkdir()
    (bootstrap_keyring / "historical-anchor.pem").write_bytes(
        authority_setup.anchor_public.read_bytes()
    )

    with pytest.raises(
        SnapshotAnchorReferenceError,
        match="trust roles must be globally disjoint",
    ):
        SnapshotAnchorReferenceAuthority(
            SnapshotAnchorReferenceConfig(
                database_path=authority_setup.database,
                request_public_key_path=authority_setup.request_public,
                anchor_private_key_path=authority_setup.anchor_private,
                bootstrap_public_key_path=authority_setup.bootstrap_public,
                bootstrap_public_key_dir=bootstrap_keyring,
            )
        )


def test_reference_head_is_empty_then_returns_exact_latest_receipt(
    authority_setup: AuthoritySetup,
) -> None:
    authority = authority_setup.authority()
    assert authority.get_head() is None
    intent_payload = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="generation-a",
        operation_id="00000000-0000-0000-0000-000000000001",
    )
    receipt_payload = authority.compare_and_append(intent_payload)

    assert authority.get_head() == receipt_payload


def test_database_is_bound_to_one_publication_domain(
    authority_setup: AuthoritySetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_setup.authority()
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "84c7c313-09ae-41ae-a27f-c83f579aa934",
    )

    with pytest.raises(
        SnapshotAnchorReferenceError,
        match="belongs to another publication domain",
    ):
        authority_setup.authority()


def test_two_connections_linearize_competing_genesis(
    authority_setup: AuthoritySetup,
) -> None:
    first = authority_setup.authority()
    second = authority_setup.authority()
    left = authority_setup.intent(expected=None, sequence=0, generation="left")
    right = authority_setup.intent(expected=None, sequence=0, generation="right")
    barrier = Barrier(2)

    def append(
        authority: SnapshotAnchorReferenceAuthority,
        payload: bytes,
    ) -> tuple[str, bytes | str]:
        barrier.wait()
        try:
            return "committed", authority.compare_and_append(payload)
        except SnapshotAnchorReferenceConflict as exc:
            return "conflict", str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(append, first, left),
            executor.submit(append, second, right),
        ]
        results = [future.result(timeout=10) for future in futures]

    assert sorted(status for status, _ in results) == ["committed", "conflict"]
    committed_payload = next(value for status, value in results if status == "committed")
    assert isinstance(committed_payload, bytes)
    committed = _mapping(committed_payload)
    assert committed["sequence"] == 1


def test_signed_genesis_rollback_and_alternative_branch_are_rejected(
    authority_setup: AuthoritySetup,
) -> None:
    authority = authority_setup.authority()
    genesis_intent = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="genesis",
    )
    genesis_receipt = _mapping(authority.compare_and_append(genesis_intent))
    second_intent = authority_setup.intent(
        expected=str(genesis_receipt["receipt_id"]),
        sequence=1,
        generation="second",
    )
    authority.compare_and_append(second_intent)

    fresh_signed_genesis = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="replacement-genesis",
    )
    stale_predecessor_branch = authority_setup.intent(
        expected=str(genesis_receipt["receipt_id"]),
        sequence=1,
        generation="alternative-second",
    )

    for attack in (fresh_signed_genesis, stale_predecessor_branch):
        with pytest.raises(
            SnapshotAnchorReferenceConflict,
            match="bootstrap is permitted only|predecessor is not the current authority HEAD",
        ):
            authority.compare_and_append(attack)


def test_exact_retry_survives_lost_response_and_reopen(
    authority_setup: AuthoritySetup,
) -> None:
    authority = authority_setup.authority()
    intent = authority_setup.intent(expected=None, sequence=0, generation="durable")

    lost_response = authority.compare_and_append(intent)
    reopened = authority_setup.authority(clock=_START + timedelta(minutes=10))

    assert reopened.compare_and_append(intent) == lost_response
    operation_id = str(_mapping(intent)["operation_id"])
    assert (
        reopened.lookup_operation(
            operation_id,
            expected_intent_payload=intent,
            expected_receipt_payload=lost_response,
        )
        == lost_response
    )


def test_request_rotation_allows_exact_replay_but_rejects_new_historical_admission(
    authority_setup: AuthoritySetup,
    tmp_path: Path,
) -> None:
    original = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="before-request-rotation",
    )
    original_receipt = authority_setup.authority().compare_and_append(original)
    receipt = _mapping(original_receipt)
    new_request_private, new_request_public = _keypair(tmp_path / "new-request")
    request_keyring = tmp_path / "request-keyring"
    request_keyring.mkdir()
    historical_request = request_keyring / "historical-request.pem"
    historical_request.write_bytes(
        authority_setup.request_public.read_bytes()
    )
    rotated = authority_setup.authority(
        request_public=new_request_public,
        request_keyring=request_keyring,
    )
    active_payload = new_request_public.read_bytes()
    new_request_public.write_bytes(authority_setup.request_public.read_bytes())
    historical_request.write_bytes(active_payload)
    rogue_request_private, rogue_request_public = _keypair(tmp_path / "rogue-request")
    (request_keyring / "post-start-rogue.pem").write_bytes(
        rogue_request_public.read_bytes()
    )

    assert rotated.compare_and_append(original) == original_receipt

    historical_new_operation = authority_setup.intent(
        expected=str(receipt["receipt_id"]),
        sequence=1,
        generation="historical-request-new-operation",
    )
    with pytest.raises(
        SnapshotAnchorReferenceError,
        match="active admission trust set",
    ):
        rotated.compare_and_append(historical_new_operation)

    rogue_new_operation = authority_setup.intent(
        expected=str(receipt["receipt_id"]),
        sequence=1,
        generation="post-start-rogue-request",
        request_private=rogue_request_private,
    )
    with pytest.raises(
        SnapshotAnchorReferenceError,
        match="active admission trust set",
    ):
        rotated.compare_and_append(rogue_new_operation)

    active_new_operation = authority_setup.intent(
        expected=str(receipt["receipt_id"]),
        sequence=1,
        generation="active-request-new-operation",
        request_private=new_request_private,
    )
    assert _mapping(rotated.compare_and_append(active_new_operation))["sequence"] == 2


def test_bootstrap_rotation_allows_exact_replay_but_rejects_new_historical_admission(
    authority_setup: AuthoritySetup,
    tmp_path: Path,
) -> None:
    original = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="before-bootstrap-rotation",
    )
    original_receipt = authority_setup.authority().compare_and_append(original)
    new_bootstrap_private, new_bootstrap_public = _keypair(tmp_path / "new-bootstrap")
    bootstrap_keyring = tmp_path / "bootstrap-keyring-rotation"
    bootstrap_keyring.mkdir()
    (bootstrap_keyring / "historical-bootstrap.pem").write_bytes(
        authority_setup.bootstrap_public.read_bytes()
    )
    rotated = authority_setup.authority(
        bootstrap_public=new_bootstrap_public,
        bootstrap_keyring=bootstrap_keyring,
    )

    assert rotated.compare_and_append(original) == original_receipt

    historical_new_operation = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="historical-bootstrap-new-operation",
        bootstrap_private=authority_setup.bootstrap_private,
        bootstrap_public=authority_setup.bootstrap_public,
    )
    empty_rotated = authority_setup.authority(
        bootstrap_public=new_bootstrap_public,
        bootstrap_keyring=bootstrap_keyring,
        database=tmp_path / "empty-rotated-anchor.sqlite3",
    )
    with pytest.raises(
        SnapshotAnchorReferenceError,
        match="active admission trust set",
    ):
        empty_rotated.compare_and_append(historical_new_operation)

    active_new_operation = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="active-bootstrap-new-operation",
        bootstrap_private=new_bootstrap_private,
        bootstrap_public=new_bootstrap_public,
    )
    assert _mapping(empty_rotated.compare_and_append(active_new_operation))["sequence"] == 1


def test_operation_id_reuse_with_different_input_is_conflict(
    authority_setup: AuthoritySetup,
) -> None:
    authority = authority_setup.authority()
    operation_id = str(uuid.uuid4())
    first = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="first",
        operation_id=operation_id,
    )
    conflicting = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="different",
        operation_id=operation_id,
    )

    authority.compare_and_append(first)

    with pytest.raises(
        SnapshotAnchorReferenceConflict,
        match="operation_id was already committed with different input",
    ):
        authority.compare_and_append(conflicting)


def test_head_observation_is_signed_nonce_bound_and_short_lived(
    authority_setup: AuthoritySetup,
) -> None:
    authority = authority_setup.authority(ttl=timedelta(seconds=45))
    intent_payload = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="observed",
    )
    receipt_payload = authority.compare_and_append(intent_payload)
    intent = _mapping(intent_payload)
    receipt = _mapping(receipt_payload)
    pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=_sha256(receipt_payload),
    )
    nonce = "a" * 64

    observation_payload = authority.observe_head(
        intent_payload=intent_payload,
        receipt_payload=receipt_payload,
        pointer=pointer,
        challenge_nonce=nonce,
    )
    observation = verify_snapshot_anchor_head_observation(
        _mapping(observation_payload),
        trusted_public_key_path=authority_setup.anchor_public,
    )

    assert observation["challenge_nonce"] == nonce
    assert observation["challenge_sha256"] == external_head_challenge_sha256(
        pointer,
        challenge_nonce=nonce,
    )
    observed = datetime.fromisoformat(str(observation["observed_at_utc"]))
    expires = datetime.fromisoformat(str(observation["expires_at_utc"]))
    assert expires - observed == timedelta(seconds=45)


def test_historical_receipt_survives_anchor_key_rotation(
    authority_setup: AuthoritySetup,
    tmp_path: Path,
) -> None:
    old_authority = authority_setup.authority()
    first_intent = authority_setup.intent(
        expected=None,
        sequence=0,
        generation="old-key",
    )
    first_receipt_payload = old_authority.compare_and_append(first_intent)
    first_receipt = _mapping(first_receipt_payload)

    new_private, new_public = _keypair(tmp_path / "new-anchor")
    new_authority = authority_setup.authority(
        anchor_private=new_private,
    )
    second_intent = authority_setup.intent(
        expected=str(first_receipt["receipt_id"]),
        sequence=1,
        generation="new-key",
    )
    second_receipt_payload = new_authority.compare_and_append(second_intent)
    second_receipt = _mapping(second_receipt_payload)

    keyring = tmp_path / "anchor-keyring"
    keyring.mkdir()
    (keyring / "old.pem").write_bytes(authority_setup.anchor_public.read_bytes())
    verified_old = verify_snapshot_anchor_receipt(
        first_receipt,
        trusted_public_key_path=new_public,
        trusted_public_key_dir=keyring,
    )
    verified_new = verify_snapshot_anchor_receipt(
        second_receipt,
        trusted_public_key_path=new_public,
    )

    assert verified_old["sequence"] == 1
    assert verified_new["sequence"] == 2
    assert (
        new_authority.lookup_operation(
            str(first_receipt["operation_id"]),
            expected_receipt_payload=first_receipt_payload,
        )
        == first_receipt_payload
    )


class AuthoritySetup:
    def __init__(
        self,
        *,
        database: Path,
        request_private: Path,
        request_public: Path,
        anchor_private: Path,
        anchor_public: Path,
        bootstrap_private: Path,
        bootstrap_public: Path,
    ) -> None:
        self.database = database
        self.request_private = request_private
        self.request_public = request_public
        self.anchor_private = anchor_private
        self.anchor_public = anchor_public
        self.bootstrap_private = bootstrap_private
        self.bootstrap_public = bootstrap_public

    def authority(
        self,
        *,
        anchor_private: Path | None = None,
        request_public: Path | None = None,
        request_keyring: Path | None = None,
        bootstrap_public: Path | None = None,
        bootstrap_keyring: Path | None = None,
        database: Path | None = None,
        ttl: timedelta = timedelta(minutes=1),
        clock: datetime = _START,
    ) -> SnapshotAnchorReferenceAuthority:
        return SnapshotAnchorReferenceAuthority(
            SnapshotAnchorReferenceConfig(
                database_path=database or self.database,
                request_public_key_path=request_public or self.request_public,
                anchor_private_key_path=anchor_private or self.anchor_private,
                request_public_key_dir=request_keyring,
                bootstrap_public_key_path=bootstrap_public or self.bootstrap_public,
                bootstrap_public_key_dir=bootstrap_keyring,
                head_observation_ttl=ttl,
            ),
            clock=lambda: clock,
        )

    def intent(
        self,
        *,
        expected: str | None,
        sequence: int,
        generation: str,
        operation_id: str | None = None,
        request_private: Path | None = None,
        bootstrap_private: Path | None = None,
        bootstrap_public: Path | None = None,
    ) -> bytes:
        operation = operation_id or str(uuid.uuid4())
        transition = "PUBLISH"
        legacy_pointer_sha256 = None
        bootstrap_authorization = None
        bootstrap_authorization_id = None
        bootstrap_authorization_sha256 = None
        if sequence == 0:
            transition = BOOTSTRAP_TRANSITION_TYPE
            legacy_pointer_sha256 = "3" * 64
            bootstrap_authorization = sign_snapshot_bootstrap_authorization(
                {
                    "schema_version": PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
                    "publication_domain_sha256": publication_domain_sha256(),
                    "transition_type": BOOTSTRAP_TRANSITION_TYPE,
                    "use_policy": BOOTSTRAP_USE_POLICY,
                    "operation_id": operation,
                    "expected_anchor_sequence": 0,
                    "expected_anchor_receipt_id": None,
                    "legacy_pointer_sha256": legacy_pointer_sha256,
                    "generation_id": generation,
                    "contract_path": f"snapshots/{generation}/lt_input_snapshot.json",
                    "contract_sha256": "1" * 64,
                    "generation_inventory_sha256": "2" * 64,
                    "required_source_class": BOOTSTRAP_REQUIRED_SOURCE_CLASS,
                    "required_calibration_eligible": False,
                    "issued_at_utc": (_START - timedelta(minutes=1)).isoformat(),
                    "expires_at_utc": (_START + timedelta(minutes=5)).isoformat(),
                },
                private_key_path=bootstrap_private or self.bootstrap_private,
            )
            bootstrap_authorization_id = bootstrap_authorization["authorization_id"]
            bootstrap_authorization_sha256 = _sha256(
                canonical_json(bootstrap_authorization)
            )
        document = sign_snapshot_publication_intent(
            {
                "schema_version": "lt_snapshot_publication_intent.v1",
                "publication_domain_sha256": publication_domain_sha256(),
                "operation_id": operation,
                "operation_created_at_utc": _START.isoformat(),
                "transition_type": transition,
                "expected_anchor_receipt_id": expected,
                "expected_anchor_sequence": sequence,
                "generation_id": generation,
                "contract_path": f"snapshots/{generation}/lt_input_snapshot.json",
                "contract_sha256": "1" * 64,
                "generation_inventory_sha256": "2" * 64,
                "legacy_pointer_sha256": legacy_pointer_sha256,
                "bootstrap_authorization": bootstrap_authorization,
                "bootstrap_authorization_id": bootstrap_authorization_id,
                "bootstrap_authorization_sha256": bootstrap_authorization_sha256,
            },
            private_key_path=request_private or self.request_private,
            bootstrap_trusted_public_key_path=bootstrap_public or self.bootstrap_public,
        )
        return canonical_json(document)


@pytest.fixture
def authority_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AuthoritySetup:
    monkeypatch.setenv(PUBLICATION_DOMAIN_ID_ENV, _DOMAIN_ID)
    request_private, request_public = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    bootstrap_private, bootstrap_public = _keypair(tmp_path / "bootstrap")
    return AuthoritySetup(
        database=tmp_path / "anchor.sqlite3",
        request_private=request_private,
        request_public=request_public,
        anchor_private=anchor_private,
        anchor_public=anchor_public,
        bootstrap_private=bootstrap_private,
        bootstrap_public=bootstrap_public,
    )


def _mapping(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    assert isinstance(value, dict)
    return value


def _sha256(payload: bytes) -> str:
    import hashlib

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
