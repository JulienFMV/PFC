from __future__ import annotations

import hashlib
import ssl
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import scripts.publish_governed_lt_input_snapshot as publish_cli
from pfc_shaping.data import snapshot_anchor_client as client_module
from pfc_shaping.data import snapshot_publication_state as state_module
from pfc_shaping.data.snapshot_anchor_client import (
    SnapshotAnchorAuthenticationError,
    SnapshotAnchorClient,
    SnapshotAnchorClientConfig,
    SnapshotAnchorClientError,
    SnapshotAnchorConflictError,
    SnapshotAnchorIndeterminateError,
    SnapshotAnchorNotFoundError,
    SnapshotAnchorProtocolError,
    SnapshotAnchorUnavailableError,
    load_anchor_client_config_from_environment,
)
from pfc_shaping.data.snapshot_anchor_signing import (
    sign_snapshot_anchor_head_observation,
    sign_snapshot_anchor_receipt,
)
from pfc_shaping.data.snapshot_publication_contract import (
    PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV,
    PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
    PUBLICATION_DOMAIN_ID_ENV,
    PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_DIR_ENV,
    PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV,
    publication_domain_sha256,
    publication_public_key_id,
    sign_snapshot_publication_intent,
)
from pfc_shaping.data.snapshot_publication_state import (
    canonical_json,
    external_head_challenge_sha256,
    pointer_mapping_from_external_receipt,
)
from pfc_shaping.data.snapshot_publisher import SnapshotPublicationRepairRequired
from pfc_shaping.durable_artifact import (
    DurableArtifactRepairRequired,
    durable_artifact_durability_classification,
    prepare_durable_json_directory,
    write_durable_exact_bytes,
)


def test_compare_and_append_verifies_exact_signed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(request_public))
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=request_private,
    )
    intent_payload = canonical_json(intent)
    receipt = sign_snapshot_anchor_receipt(
        _receipt(intent, intent_payload),
        private_key_path=anchor_private,
    )
    transport = _transport(monkeypatch, canonical_json(receipt))
    client = SnapshotAnchorClient(_config(tmp_path))

    assert client.compare_and_append(intent_payload) == canonical_json(receipt)
    assert transport.requests[0][0] == "POST"
    assert transport.requests[0][2] == intent_payload


def test_get_head_returns_only_an_authenticated_current_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=request_private,
    )
    intent_payload = canonical_json(intent)
    receipt_payload = canonical_json(
        sign_snapshot_anchor_receipt(
            _receipt(intent, intent_payload),
            private_key_path=anchor_private,
        )
    )
    transport = _transport(monkeypatch, receipt_payload)
    client = SnapshotAnchorClient(_config(tmp_path))

    assert client.get_head() == receipt_payload
    assert transport.requests[0][0] == "GET"
    assert transport.requests[0][1].endswith("/head")


def test_get_head_accepts_current_receipt_signed_by_rotated_historical_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    old_anchor_private, old_anchor_public = _keypair(tmp_path / "old-anchor")
    _, active_anchor_public = _keypair(tmp_path / "active-anchor")
    keyring = tmp_path / "anchor-keyring"
    keyring.mkdir()
    (keyring / "old.pem").write_bytes(old_anchor_public.read_bytes())
    monkeypatch.setenv(
        PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
        str(active_anchor_public),
    )
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV, str(keyring))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=request_private,
    )
    intent_payload = canonical_json(intent)
    receipt_payload = canonical_json(
        sign_snapshot_anchor_receipt(
            _receipt(intent, intent_payload),
            private_key_path=old_anchor_private,
        )
    )
    _transport(monkeypatch, receipt_payload)

    assert SnapshotAnchorClient(_config(tmp_path)).get_head() == receipt_payload


def test_observe_head_accepts_historical_receipt_with_active_observation_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    old_anchor_private, old_anchor_public = _keypair(tmp_path / "old-anchor")
    active_anchor_private, active_anchor_public = _keypair(tmp_path / "active-anchor")
    keyring = tmp_path / "anchor-keyring"
    keyring.mkdir()
    (keyring / "old.pem").write_bytes(old_anchor_public.read_bytes())
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(request_public))
    monkeypatch.setenv(
        PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
        str(active_anchor_public),
    )
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV, str(keyring))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    monkeypatch.setattr(
        state_module,
        "_reference_utc",
        lambda _: state_module.datetime.fromisoformat("2026-07-16T09:03:00+00:00"),
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=request_private,
    )
    intent_payload = canonical_json(intent)
    receipt = sign_snapshot_anchor_receipt(
        _receipt(intent, intent_payload),
        private_key_path=old_anchor_private,
    )
    receipt_payload = canonical_json(receipt)
    pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
    )
    challenge_nonce = "4" * 64
    observation_payload = canonical_json(
        sign_snapshot_anchor_head_observation(
            {
                "schema_version": "lt_snapshot_anchor_head_observation.v1",
                "publication_domain_sha256": publication_domain_sha256(),
                "receipt_id": receipt["receipt_id"],
                "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
                "sequence": receipt["sequence"],
                "challenge_nonce": challenge_nonce,
                "challenge_sha256": external_head_challenge_sha256(
                    pointer,
                    challenge_nonce=challenge_nonce,
                ),
                "observed_at_utc": "2026-07-16T09:00:02+00:00",
                "expires_at_utc": "2026-07-16T09:05:02+00:00",
            },
            private_key_path=active_anchor_private,
        )
    )
    _transport(monkeypatch, observation_payload)

    assert SnapshotAnchorClient(_config(tmp_path)).observe_head(
        intent_payload=intent_payload,
        receipt_payload=receipt_payload,
        pointer=pointer,
        challenge_nonce=challenge_nonce,
    ) == observation_payload


def test_compare_and_append_rejects_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    transport = _transport(monkeypatch, b"{}", status=302)
    client = SnapshotAnchorClient(_config(tmp_path))

    with pytest.raises(SnapshotAnchorClientError, match="HTTP 302"):
        client._request("POST", "/test", b"{}", expected_statuses={200})
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, SnapshotAnchorAuthenticationError),
        (404, SnapshotAnchorNotFoundError),
        (409, SnapshotAnchorConflictError),
        (429, SnapshotAnchorUnavailableError),
        (500, SnapshotAnchorUnavailableError),
    ],
)
def test_anchor_http_status_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    error_type: type[SnapshotAnchorClientError],
) -> None:
    _transport(monkeypatch, b"{}", status=status)
    client = SnapshotAnchorClient(_config(tmp_path))

    with pytest.raises(error_type, match=f"HTTP {status}"):
        client._request("POST", "/test", b"{}", expected_statuses={200})


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_compare_and_append_transient_http_response_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    _transport(monkeypatch, b"{}", status=status)
    client = SnapshotAnchorClient(_config(tmp_path))

    with pytest.raises(SnapshotAnchorIndeterminateError, match=f"HTTP {status}"):
        client._request(
            "POST",
            "/test",
            b"{}",
            expected_statuses={200},
            commit_may_have_occurred=True,
        )


def test_compare_and_append_rejects_noncanonical_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, request_public = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(request_public))
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=request_private,
    )
    intent_payload = canonical_json(intent)
    receipt = sign_snapshot_anchor_receipt(
        _receipt(intent, intent_payload),
        private_key_path=anchor_private,
    )
    _transport(monkeypatch, canonical_json(receipt) + b"\n")
    client = SnapshotAnchorClient(_config(tmp_path))

    with pytest.raises(
        SnapshotAnchorIndeterminateError,
        match="invalid success response after a possible commit",
    ):
        client.compare_and_append(intent_payload)


def test_compare_and_append_rejects_historical_request_key_before_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical_private, historical_public = _keypair(tmp_path / "historical-request")
    _, active_public = _keypair(tmp_path / "active-request")
    keyring = tmp_path / "request-keyring"
    keyring.mkdir()
    historical_key_id = publication_public_key_id(historical_public)
    (keyring / f"{historical_key_id}.pem").write_bytes(historical_public.read_bytes())
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(active_public))
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_DIR_ENV, str(keyring))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent_payload = canonical_json(
        sign_snapshot_publication_intent(
            _intent(),
            private_key_path=historical_private,
        )
    )
    transport = _transport(monkeypatch, b"{}")

    with pytest.raises(
        SnapshotAnchorAuthenticationError,
        match="active admission keys",
    ):
        SnapshotAnchorClient(_config(tmp_path)).compare_and_append(intent_payload)

    assert transport.requests == []


def test_operation_lookup_requires_exact_archived_receipt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=request_private,
    )
    intent_payload = canonical_json(intent)
    receipt_payload = canonical_json(
        sign_snapshot_anchor_receipt(
            _receipt(intent, intent_payload),
            private_key_path=anchor_private,
        )
    )
    transport = _transport(monkeypatch, receipt_payload)
    client = SnapshotAnchorClient(_config(tmp_path))

    assert (
        client.lookup_operation(
            str(intent["operation_id"]),
            expected_receipt_payload=receipt_payload,
        )
        == receipt_payload
    )
    assert transport.requests[0][0] == "GET"


def test_ambiguous_commit_recovery_accepts_committed_historical_request_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical_private, historical_public = _keypair(tmp_path / "historical-request")
    _, active_public = _keypair(tmp_path / "active-request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    keyring = tmp_path / "request-keyring"
    keyring.mkdir()
    historical_key_id = publication_public_key_id(historical_public)
    (keyring / f"{historical_key_id}.pem").write_bytes(historical_public.read_bytes())
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(active_public))
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_DIR_ENV, str(keyring))
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=historical_private,
    )
    intent_payload = canonical_json(intent)
    receipt_payload = canonical_json(
        sign_snapshot_anchor_receipt(
            _receipt(intent, intent_payload),
            private_key_path=anchor_private,
        )
    )
    _transport(monkeypatch, receipt_payload)
    client = SnapshotAnchorClient(_config(tmp_path))

    assert client.lookup_operation(
        str(intent["operation_id"]),
        expected_intent_payload=intent_payload,
    ) == receipt_payload


def test_operation_lookup_rejects_valid_but_divergent_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_private, _ = _keypair(tmp_path / "request")
    anchor_private, anchor_public = _keypair(tmp_path / "anchor")
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = sign_snapshot_publication_intent(
        _intent(),
        private_key_path=request_private,
    )
    intent_payload = canonical_json(intent)
    archived = sign_snapshot_anchor_receipt(
        _receipt(intent, intent_payload),
        private_key_path=anchor_private,
    )
    divergent_payload = _receipt(intent, intent_payload)
    divergent_payload["committed_at_utc"] = "2026-07-16T09:00:02+00:00"
    divergent = sign_snapshot_anchor_receipt(
        divergent_payload,
        private_key_path=anchor_private,
    )
    _transport(monkeypatch, canonical_json(divergent))
    client = SnapshotAnchorClient(_config(tmp_path))

    with pytest.raises(SnapshotAnchorClientError, match="differs from the archived receipt"):
        client.lookup_operation(
            str(intent["operation_id"]),
            expected_receipt_payload=canonical_json(archived),
        )


def test_environment_config_requires_every_mtls_binding() -> None:
    with pytest.raises(SnapshotAnchorClientError, match="configuration is missing"):
        load_anchor_client_config_from_environment(
            {"PFC_DATA_PUBLICATION_ANCHOR_BASE_URL": "https://anchor.invalid"}
        )


def test_cas_cli_persists_receipt_without_observing_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    receipt = {
        **_receipt(intent, canonical_json(intent)),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    intent_path = tmp_path / "intent.json"
    receipt_path = tmp_path / "receipt.json"
    intent_path.write_bytes(canonical_json(intent))

    class FakeClient:
        def compare_and_append(self, payload: bytes) -> bytes:
            assert payload == intent_path.read_bytes()
            return canonical_json(receipt)

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )

    result = publish_cli._commit_external_cas(
        SimpleNamespace(
            intent=intent_path,
            receipt_output=receipt_path,
        )
    )

    assert result["status"] == "COMMITTED_EXTERNAL_CAS_AWAITING_FRESH_HEAD"
    assert result["schema_version"] == "lt_snapshot_publication_cli_result.v1"
    assert result["command"] == "cas"
    assert result["commit_status"] == "COMMITTED"
    assert result["projection_status"] == "AWAITING_FRESH_HEAD"
    assert result["retry_disposition"] == "RUN_FINALIZE_WITH_FRESH_HEAD"
    assert receipt_path.read_bytes() == canonical_json(receipt)


def test_cas_cli_recovers_ambiguous_commit_by_exact_operation_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_payload = canonical_json(intent)
    receipt = {
        **_receipt(intent, intent_payload),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    receipt_payload = canonical_json(receipt)
    intent_path = tmp_path / "intent.json"
    receipt_path = tmp_path / "receipt.json"
    intent_path.write_bytes(intent_payload)

    class FakeClient:
        def compare_and_append(self, payload: bytes) -> bytes:
            raise SnapshotAnchorClientError("ambiguous transport failure")

        def lookup_operation(self, operation_id: str, **kwargs: object) -> bytes:
            assert operation_id == intent["operation_id"]
            assert kwargs["expected_intent_payload"] == intent_payload
            return receipt_payload

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )

    result = publish_cli._commit_external_cas(
        SimpleNamespace(
            intent=intent_path,
            receipt_output=receipt_path,
        )
    )

    assert result["anchor_receipt_id"] == receipt["receipt_id"]
    assert receipt_path.read_bytes() == receipt_payload


@pytest.mark.parametrize(
    ("commit_error", "expected_type"),
    [
        (
            SnapshotAnchorAuthenticationError("authentication failed"),
            SnapshotAnchorAuthenticationError,
        ),
        (
            SnapshotAnchorProtocolError("invalid signed response"),
            SnapshotAnchorProtocolError,
        ),
        (
            SnapshotAnchorIndeterminateError("transport timeout"),
            SnapshotAnchorIndeterminateError,
        ),
    ],
)
def test_cas_cli_preserves_primary_failure_class_when_exact_lookup_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit_error: SnapshotAnchorClientError,
    expected_type: type[SnapshotAnchorClientError],
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_path = tmp_path / "intent.json"
    intent_path.write_bytes(canonical_json(intent))

    class FakeClient:
        def compare_and_append(self, payload: bytes) -> bytes:
            raise commit_error

        def lookup_operation(self, operation_id: str, **kwargs: object) -> bytes:
            raise SnapshotAnchorNotFoundError("operation is absent")

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )

    with pytest.raises(expected_type):
        publish_cli._commit_external_cas(
            SimpleNamespace(
                intent=intent_path,
                receipt_output=tmp_path / "receipt.json",
            )
        )


def test_cas_cli_reports_committed_when_receipt_archive_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_payload = canonical_json(intent)
    receipt = {
        **_receipt(intent, intent_payload),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    receipt_payload = canonical_json(receipt)
    intent_path = tmp_path / "intent.json"
    intent_path.write_bytes(intent_payload)

    class FakeClient:
        def compare_and_append(self, payload: bytes) -> bytes:
            assert payload == intent_payload
            return receipt_payload

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )
    monkeypatch.setattr(
        publish_cli,
        "_write_exact_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(SnapshotPublicationRepairRequired) as caught:
        publish_cli._commit_external_cas(
            SimpleNamespace(
                intent=intent_path,
                receipt_output=tmp_path / "receipt.json",
            )
        )

    result = caught.value.result
    assert result["commit_status"] == "COMMITTED"
    assert result["projection_status"] == "RECEIPT_ARCHIVE_REPAIR_REQUIRED"
    assert result["retry_disposition"] == "RETRY_EXACT_OPERATION_LOOKUP"
    assert result["anchor_receipt_id"] == receipt["receipt_id"]


def test_cas_cli_exact_retry_does_not_create_head_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_payload = canonical_json(intent)
    receipt = {
        **_receipt(intent, intent_payload),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    receipt_payload = canonical_json(receipt)
    intent_path = tmp_path / "intent.json"
    receipt_path = tmp_path / "receipt.json"
    intent_path.write_bytes(intent_payload)
    receipt_path.write_bytes(receipt_payload)

    class FakeClient:
        def lookup_operation(self, operation_id: str, **kwargs: object) -> bytes:
            assert operation_id == intent["operation_id"]
            assert kwargs["expected_receipt_payload"] == receipt_payload
            return receipt_payload

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )
    result = publish_cli._commit_external_cas(
        SimpleNamespace(
            intent=intent_path,
            receipt_output=receipt_path,
        )
    )

    assert result["anchor_receipt_id"] == receipt["receipt_id"]
    assert "head_challenge_nonce" not in result


def test_cas_cli_stages_truncated_receipt_repair_from_exact_authority_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_payload = canonical_json(intent)
    receipt = {
        **_receipt(intent, intent_payload),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    receipt_payload = canonical_json(receipt)
    intent_path = tmp_path / "intent.json"
    receipt_path = tmp_path / "receipt.json"
    intent_path.write_bytes(intent_payload)
    receipt_path.write_bytes(b"{")

    class FakeClient:
        def lookup_operation(self, operation_id: str, **kwargs: object) -> bytes:
            assert operation_id == intent["operation_id"]
            assert kwargs["expected_intent_payload"] == intent_payload
            return receipt_payload

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )

    with pytest.raises(SnapshotPublicationRepairRequired) as caught:
        publish_cli._commit_external_cas(
            SimpleNamespace(intent=intent_path, receipt_output=receipt_path)
        )

    result = caught.value.result
    repair = result["local_repair"]
    candidate = Path(str(repair["repair_candidate_path"]))
    assert result["commit_status"] == "COMMITTED"
    assert result["projection_status"] == "RECEIPT_ARCHIVE_REPAIR_REQUIRED"
    assert result["anchor_receipt_id"] == receipt["receipt_id"]
    assert repair["reason"] == "AUTHORITATIVE_BYTES_DIFFER_FROM_EXISTING_TARGET"
    assert repair["automatic_replacement_performed"] is False
    assert receipt_path.read_bytes() == b"{"
    assert candidate.read_bytes() == receipt_payload


def test_finalize_renews_and_content_addresses_head_observation_on_every_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_payload = canonical_json(intent)
    receipt = {
        **_receipt(intent, intent_payload),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    receipt_payload = canonical_json(receipt)
    intent_path = tmp_path / "intent.json"
    receipt_path = tmp_path / "receipt.json"
    observations = tmp_path / "observations"
    intent_path.write_bytes(intent_payload)
    receipt_path.write_bytes(receipt_payload)
    ids = iter(["e" * 64, "f" * 64])
    observed_nonces: list[str] = []

    class FakeClient:
        def lookup_operation(self, operation_id: str, **kwargs: object) -> bytes:
            assert operation_id == intent["operation_id"]
            assert kwargs["expected_receipt_payload"] == receipt_payload
            assert kwargs["expected_intent_payload"] == intent_payload
            return receipt_payload

        def observe_head(self, **kwargs: object) -> bytes:
            nonce = str(kwargs["challenge_nonce"])
            observed_nonces.append(nonce)
            return canonical_json(
                {
                    "challenge_nonce": nonce,
                    "observation_id": next(ids),
                }
            )

    finalized: list[Path] = []
    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )
    monkeypatch.setattr(
        publish_cli,
        "finalize_governed_lt_snapshot",
        lambda **kwargs: finalized.append(Path(kwargs["head_observation_path"]))
        or {"status": "FINALIZED"},
    )
    args = SimpleNamespace(
        data_root=tmp_path,
        intent=intent_path,
        anchor_receipt=receipt_path,
        observation_directory=observations,
    )

    first = publish_cli._finalize_external_projection(args)
    second = publish_cli._finalize_external_projection(args)

    assert first["head_observation_path"] != second["head_observation_path"]
    assert len(observed_nonces) == 2
    assert observed_nonces[0] != observed_nonces[1]
    assert finalized == [observations / f"{'e' * 64}.json", observations / f"{'f' * 64}.json"]
    assert sorted(path.name for path in observations.glob("*.json")) == [
        f"{'e' * 64}.json",
        f"{'f' * 64}.json",
    ]


def test_finalize_stages_truncated_receipt_before_head_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_payload = canonical_json(intent)
    receipt = {
        **_receipt(intent, intent_payload),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    receipt_payload = canonical_json(receipt)
    intent_path = tmp_path / "intent.json"
    receipt_path = tmp_path / "receipt.json"
    intent_path.write_bytes(intent_payload)
    receipt_path.write_bytes(b"{")

    class FakeClient:
        def lookup_operation(self, operation_id: str, **kwargs: object) -> bytes:
            assert operation_id == intent["operation_id"]
            assert "expected_receipt_payload" not in kwargs
            assert kwargs["expected_intent_payload"] == intent_payload
            return receipt_payload

        def observe_head(self, **kwargs: object) -> bytes:
            raise AssertionError("HEAD observation must wait for local repair")

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )
    monkeypatch.setattr(
        publish_cli,
        "finalize_governed_lt_snapshot",
        lambda **_: {"status": "FINALIZED"},
    )

    with pytest.raises(SnapshotPublicationRepairRequired) as caught:
        publish_cli._finalize_external_projection(
            SimpleNamespace(
                data_root=tmp_path,
                intent=intent_path,
                anchor_receipt=receipt_path,
                observation_directory=tmp_path / "observations",
            )
        )

    result = caught.value.result
    repair = result["local_repair"]
    candidate = Path(str(repair["repair_candidate_path"]))
    assert result["commit_status"] == "COMMITTED"
    assert result["projection_status"] == "RECEIPT_ARCHIVE_REPAIR_REQUIRED"
    assert repair["reason"] == "AUTHORITATIVE_BYTES_DIFFER_FROM_EXISTING_TARGET"
    assert repair["automatic_replacement_performed"] is False
    assert receipt_path.read_bytes() == b"{"
    assert candidate.read_bytes() == receipt_payload


def test_finalize_post_lookup_failure_preserves_committed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "de6bc6ac-1243-4f06-821b-c40ddb088edc",
    )
    intent = {
        **_intent(),
        "intent_id": "a" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "b" * 64, "value_base64": "AA=="},
    }
    intent_payload = canonical_json(intent)
    receipt = {
        **_receipt(intent, intent_payload),
        "receipt_id": "c" * 64,
        "signature": {"algorithm": "Ed25519", "key_id": "d" * 64, "value_base64": "AA=="},
    }
    receipt_payload = canonical_json(receipt)
    intent_path = tmp_path / "intent.json"
    receipt_path = tmp_path / "receipt.json"
    intent_path.write_bytes(intent_payload)
    receipt_path.write_bytes(receipt_payload)

    class FakeClient:
        def lookup_operation(self, operation_id: str, **kwargs: object) -> bytes:
            assert kwargs["expected_receipt_payload"] == receipt_payload
            return receipt_payload

        def observe_head(self, **kwargs: object) -> bytes:
            raise SnapshotAnchorUnavailableError("anchor observe unavailable")

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )

    with pytest.raises(SnapshotPublicationRepairRequired) as caught:
        publish_cli._finalize_external_projection(
            SimpleNamespace(
                data_root=tmp_path,
                intent=intent_path,
                anchor_receipt=receipt_path,
                observation_directory=tmp_path / "observations",
            )
        )

    result = caught.value.result
    assert result["commit_status"] == "COMMITTED"
    assert result["projection_status"] == "LOCAL_PROJECTION_REPAIR_REQUIRED"
    assert result["retry_disposition"] == "RETRY_FINALIZE_WITH_FRESH_HEAD"
    assert result["error"]["type"] == "SnapshotAnchorUnavailableError"


def test_exact_output_recovers_crash_hardlink_residue(tmp_path: Path) -> None:
    payload = b'{"status":"committed"}'
    target = tmp_path / "receipt.json"
    temporary = tmp_path / ".receipt.json.crash.tmp"
    temporary.write_bytes(payload)
    publish_cli.os.link(temporary, target)
    assert target.stat().st_nlink == 2

    assert publish_cli._write_exact_output(target, payload) == target

    assert not temporary.exists()
    assert target.read_bytes() == payload
    assert target.stat().st_nlink == 1


def test_durable_directory_never_deletes_unowned_writer_temporary(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / ".pfc-0123456789ab-active.tmp"
    temporary.write_bytes(b"in-flight")

    with pytest.raises(DurableArtifactRepairRequired) as caught:
        prepare_durable_json_directory(tmp_path, label="publication journal")

    assert temporary.read_bytes() == b"in-flight"
    assert caught.value.result == {
        "schema_version": "durable_artifact_repair.v2",
        "status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
        "reason": "UNOWNED_TEMPORARY_RESIDUE",
        "path": str(temporary),
        "operator_action": "VERIFY_WRITER_STOPPED_THEN_QUARANTINE_RESIDUE",
        "automatic_deletion_performed": False,
        "durability_classification": durable_artifact_durability_classification(),
    }


def test_authoritative_repair_never_replaces_another_canonical_document(
    tmp_path: Path,
) -> None:
    target = tmp_path / "receipt.json"
    original = b'{"operation_id":"operation-a"}'
    divergent = b'{"operation_id":"operation-b"}'

    assert publish_cli._write_exact_output(target, original, repair_from_authority=True) == target
    with pytest.raises(ValueError, match="already exists with other bytes"):
        publish_cli._write_exact_output(target, divergent, repair_from_authority=True)

    assert target.read_bytes() == original


def test_concurrent_repairs_only_stage_distinct_candidates(
    tmp_path: Path,
) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"{")
    predicate_entered = Event()
    release_first_writer = Event()

    def delayed_repair(_: bytes) -> bool:
        predicate_entered.set()
        assert release_first_writer.wait(timeout=5)
        return True

    def write(payload: bytes, predicate: Callable[[bytes], bool]) -> Path:
        return write_durable_exact_bytes(
            target,
            payload,
            label="concurrent receipt",
            replace_existing_if=predicate,
        )

    first_payload = b'{"operation_id":"operation-a"}'
    second_payload = b'{"operation_id":"operation-b"}'
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(write, first_payload, delayed_repair)
        assert predicate_entered.wait(timeout=5)
        second = executor.submit(write, second_payload, lambda _: True)
        with pytest.raises(DurableArtifactRepairRequired):
            second.result(timeout=5)
        release_first_writer.set()
        with pytest.raises(DurableArtifactRepairRequired):
            first.result(timeout=5)

    assert target.read_bytes() == b"{"
    candidates = list((tmp_path / ".pfc-repair-candidates").glob("*.repair-candidate"))
    assert {candidate.read_bytes() for candidate in candidates} == {
        first_payload,
        second_payload,
    }


def test_noncooperating_writer_is_never_overwritten_during_repair(
    tmp_path: Path,
) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"{")
    authoritative = b'{"operation_id":"authoritative"}'
    concurrent = b'{"operation_id":"concurrent-writer"}'

    def replace_target_while_authorizing(_: bytes) -> bool:
        target.write_bytes(concurrent)
        return True

    with pytest.raises(DurableArtifactRepairRequired) as caught:
        write_durable_exact_bytes(
            target,
            authoritative,
            label="noncooperating writer receipt",
            replace_existing_if=replace_target_while_authorizing,
        )

    result = caught.value.result
    assert target.read_bytes() == concurrent
    assert Path(str(result["repair_candidate_path"])).read_bytes() == authoritative
    assert result["automatic_replacement_performed"] is False


def test_durable_directory_reports_unowned_repair_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".pfc-repair-0123456789ab.lock"
    lock.write_bytes(b"")

    with pytest.raises(DurableArtifactRepairRequired) as caught:
        prepare_durable_json_directory(tmp_path, label="publication journal")

    assert caught.value.result["reason"] == "UNOWNED_REPAIR_LOCK"
    assert caught.value.result["path"] == str(lock)
    assert lock.exists()


def test_prepare_discovers_current_external_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head_payload = canonical_json({"receipt_id": "a" * 64, "sequence": 7})

    class FakeClient:
        def get_head(self) -> bytes:
            return head_payload

    monkeypatch.setattr(publish_cli, "SnapshotAnchorClient", lambda _: FakeClient())
    monkeypatch.setattr(
        publish_cli,
        "load_anchor_client_config_from_environment",
        lambda _: object(),
    )

    assert publish_cli._prepare_head_expectation(
        SimpleNamespace(
            expected_anchor_sequence=None,
            expected_anchor_receipt_id=None,
            allow_empty_genesis=False,
        )
    ) == ("a" * 64, 7)


def test_finalize_repair_required_uses_distinct_operational_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {"status": "COMMITTED_REPAIR_REQUIRED", "operation_id": "op-001"}
    monkeypatch.setattr(
        publish_cli,
        "_parse_args",
        lambda _: SimpleNamespace(
            command="finalize",
            data_root=Path("C:/data"),
            intent=Path("C:/intent.json"),
            anchor_receipt=Path("C:/receipt.json"),
            observation_directory=Path("C:/observations"),
        ),
    )
    monkeypatch.setattr(
        publish_cli,
        "assert_snapshot_publisher_runtime_identity",
        lambda **_: None,
    )
    monkeypatch.setattr(
        publish_cli,
        "_finalize_external_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SnapshotPublicationRepairRequired(result)
        ),
    )

    assert publish_cli.main([]) == 51


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_exit"),
    [
        (SnapshotAnchorConflictError("stale HEAD"), "CAS_CONFLICT", 40),
        (SnapshotAnchorIndeterminateError("timeout"), "INDETERMINATE", 52),
        (SnapshotAnchorProtocolError("bad signature"), "INTEGRITY_FAILURE", 50),
    ],
)
def test_publication_cli_emits_structured_failure_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: SnapshotAnchorClientError,
    expected_status: str,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        publish_cli,
        "_parse_args",
        lambda _: SimpleNamespace(command="cas"),
    )
    monkeypatch.setattr(
        publish_cli,
        "assert_snapshot_publisher_runtime_identity",
        lambda **_: None,
    )
    monkeypatch.setattr(
        publish_cli,
        "_commit_external_cas",
        lambda *_: (_ for _ in ()).throw(error),
    )

    assert publish_cli.main([]) == expected_exit
    payload = publish_cli.json.loads(capsys.readouterr().err)
    assert payload["schema_version"] == "lt_snapshot_publication_cli_result.v1"
    assert payload["status"] == expected_status
    assert payload["production_authorization"] is False


def test_mtls_context_uses_only_pinned_ca_and_captured_credential_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ca = tmp_path / "ca.pem"
    certificate = tmp_path / "client.pem"
    private_key = tmp_path / "client-key.pem"
    ca.write_bytes(b"ORIGINAL CA")
    certificate.write_bytes(b"ORIGINAL CERTIFICATE")
    private_key.write_bytes(b"ORIGINAL PRIVATE KEY")
    validated = client_module._validated_config(
        SnapshotAnchorClientConfig(
            base_url="https://anchor.example.test",
            ca_bundle=ca,
            client_certificate=certificate,
            client_private_key=private_key,
        )
    )
    ca.write_bytes(b"REPLACED CA")
    certificate.write_bytes(b"REPLACED CERTIFICATE")
    private_key.write_bytes(b"REPLACED PRIVATE KEY")
    captured: dict[str, object] = {}

    class PinnedContext:
        def __init__(self, protocol: int) -> None:
            captured["protocol"] = protocol
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.minimum_version = None

        def load_verify_locations(self, *, cadata: str) -> None:
            captured["ca"] = cadata

        def load_cert_chain(self, *, certfile: str, keyfile: str) -> None:
            captured["certificate"] = Path(certfile).read_bytes()
            captured["private_key"] = Path(keyfile).read_bytes()

    monkeypatch.setattr(client_module.ssl, "SSLContext", PinnedContext)
    monkeypatch.setattr(
        client_module,
        "read_stable_single_link_file",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("credential reopened")),
    )

    context = client_module._build_ssl_context(validated)

    assert captured == {
        "protocol": ssl.PROTOCOL_TLS_CLIENT,
        "ca": "ORIGINAL CA",
        "certificate": b"ORIGINAL CERTIFICATE",
        "private_key": b"ORIGINAL PRIVATE KEY",
    }
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_publisher_mtls_material_is_staged_inside_supervisor_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "governed-supervisor"
    scratch.mkdir()
    monkeypatch.setenv("_PFC_SNAPSHOT_PUBLISHER_CAPTURE_ROOT", str(scratch))
    staged_parents: list[Path] = []

    class PinnedContext:
        def __init__(self, _protocol: int) -> None:
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.minimum_version = None

        def load_verify_locations(self, *, cadata: str) -> None:
            assert cadata == "CA"

        def load_cert_chain(self, *, certfile: str, keyfile: str) -> None:
            staged_parents.extend([Path(certfile).parent, Path(keyfile).parent])

    config = SnapshotAnchorClientConfig(
        base_url="https://anchor.example.test",
        ca_bundle=tmp_path / "unused-ca.pem",
        client_certificate=tmp_path / "unused-cert.pem",
        client_private_key=tmp_path / "unused-key.pem",
        _ca_payload=b"CA",
        _certificate_payload=b"CERT",
        _private_key_payload=b"KEY",
    )
    monkeypatch.setattr(client_module.ssl, "SSLContext", PinnedContext)

    client_module._build_ssl_context(config)

    assert len(staged_parents) == 2
    assert all(parent.parent == scratch for parent in staged_parents)
    assert all(not parent.exists() for parent in staged_parents)


class _FakeContext:
    minimum_version = None

    def load_cert_chain(self, *, certfile: str, keyfile: str) -> None:
        assert certfile
        assert keyfile


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int) -> None:
        self.status = status
        self._body = body

    def read(self, size: int) -> bytes:
        return self._body[:size]

    def getheader(self, name: str, default: str = "") -> str:
        return "application/json" if name == "Content-Type" else default


class _FakeTransport:
    def __init__(self, body: bytes, *, status: int) -> None:
        self.body = body
        self.status = status
        self.requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    def connection(self, *args: object, **kwargs: object) -> _FakeTransport:
        return self

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> None:
        self.requests.append((method, path, body, headers))

    def getresponse(self) -> _FakeResponse:
        return _FakeResponse(self.body, status=self.status)

    def close(self) -> None:
        return None


def _transport(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    *,
    status: int = 200,
) -> _FakeTransport:
    transport = _FakeTransport(body, status=status)
    monkeypatch.setattr(client_module, "_validated_config", lambda config: config)
    monkeypatch.setattr(client_module, "_build_ssl_context", lambda config: _FakeContext())
    monkeypatch.setattr(
        client_module.http.client,
        "HTTPSConnection",
        transport.connection,
    )
    return transport


def _config(tmp_path: Path) -> SnapshotAnchorClientConfig:
    return SnapshotAnchorClientConfig(
        base_url="https://anchor.invalid",
        ca_bundle=tmp_path / "ca.pem",
        client_certificate=tmp_path / "client.pem",
        client_private_key=tmp_path / "client-key.pem",
    )


def _intent() -> dict[str, object]:
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


def _receipt(
    intent: dict[str, object],
    intent_payload: bytes,
) -> dict[str, object]:
    return {
        "schema_version": "lt_snapshot_anchor_receipt.v1",
        "publication_domain_sha256": publication_domain_sha256(),
        "sequence": 2,
        "previous_receipt_id": intent["expected_anchor_receipt_id"],
        "operation_id": intent["operation_id"],
        "intent_id": intent["intent_id"],
        "intent_sha256": hashlib.sha256(intent_payload).hexdigest(),
        "transition_type": intent["transition_type"],
        "generation_id": intent["generation_id"],
        "contract_sha256": intent["contract_sha256"],
        "generation_inventory_sha256": intent["generation_inventory_sha256"],
        "legacy_pointer_sha256": None,
        "bootstrap_authorization_id": None,
        "bootstrap_authorization_sha256": None,
        "committed_at_utc": "2026-07-16T09:00:01+00:00",
    }


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
