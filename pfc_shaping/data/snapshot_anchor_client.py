"""mTLS client for the IT-owned LT snapshot publication anchor."""

from __future__ import annotations

import hashlib
import http.client
import os
import secrets
import ssl
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import quote, urlsplit

from pfc_shaping.data.snapshot_publication_contract import (
    configured_publication_anchor_keyring_directory,
    configured_publication_bootstrap_keyring_directory,
    configured_publication_request_keyring_directory,
    publication_domain_sha256,
    required_publication_anchor_public_key_path,
    required_publication_bootstrap_public_key_path,
    required_publication_request_public_key_path,
    verify_snapshot_anchor_head_observation,
    verify_snapshot_anchor_receipt,
    verify_snapshot_publication_intent,
)
from pfc_shaping.data.snapshot_publication_state import (
    canonical_json,
    external_head_challenge_sha256,
    verify_external_publication_evidence,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

_JSON_CONTENT_TYPE = "application/json"
_MAX_RESPONSE_BYTES = 1_048_576


class SnapshotAnchorClientError(RuntimeError):
    """Raised when the external anchor transport or response is invalid."""


class SnapshotAnchorConflictError(SnapshotAnchorClientError):
    """Raised for a determinate CAS or idempotency conflict."""


class SnapshotAnchorAuthenticationError(SnapshotAnchorClientError):
    """Raised when the authority rejects the authenticated caller."""


class SnapshotAnchorNotFoundError(SnapshotAnchorClientError):
    """Raised when an exact authority object is absent."""


class SnapshotAnchorUnavailableError(SnapshotAnchorClientError):
    """Raised when the authority explicitly reports temporary unavailability."""


class SnapshotAnchorIndeterminateError(SnapshotAnchorClientError):
    """Raised when transport failure leaves commit outcome unknown."""


class SnapshotAnchorProtocolError(SnapshotAnchorClientError):
    """Raised for an invalid or unauthenticated authority response."""


@dataclass(frozen=True)
class SnapshotAnchorClientConfig:
    base_url: str
    ca_bundle: Path
    client_certificate: Path
    client_private_key: Path
    timeout_seconds: float = 10.0
    max_response_bytes: int = _MAX_RESPONSE_BYTES
    _ca_payload: bytes | None = field(default=None, repr=False, compare=False)
    _certificate_payload: bytes | None = field(default=None, repr=False, compare=False)
    _private_key_payload: bytes | None = field(default=None, repr=False, compare=False)


class SnapshotAnchorClient:
    """Call one authenticated, linearizable external publication domain."""

    def __init__(self, config: SnapshotAnchorClientConfig) -> None:
        self._config = _validated_config(config)
        parsed = urlsplit(self._config.base_url)
        self._host = str(parsed.hostname)
        self._port = parsed.port or 443
        self._base_path = parsed.path.rstrip("/")
        self._ssl_context = _build_ssl_context(self._config)

    def compare_and_append(self, intent_payload: bytes) -> bytes:
        """Submit one exact signed intent and return its signed CAS receipt bytes."""

        try:
            intent = _strict_canonical_mapping(intent_payload, label="publication intent")
        except SnapshotAnchorClientError as exc:
            raise SnapshotAnchorProtocolError("publication intent is invalid") from exc
        try:
            verify_snapshot_publication_intent(
                intent,
                trusted_public_key_path=required_publication_request_public_key_path(),
                bootstrap_trusted_public_key_path=(
                    required_publication_bootstrap_public_key_path()
                    if intent.get("transition_type") == "BOOTSTRAP"
                    else None
                ),
                bootstrap_trusted_public_key_dir=None,
            )
        except ValueError as exc:
            raise SnapshotAnchorAuthenticationError(
                "publication intent is not authorized by the active admission keys"
            ) from exc
        receipt_payload = self._request(
            "POST",
            self._domain_path("compare-and-append"),
            intent_payload,
            expected_statuses={200, 201},
            commit_may_have_occurred=True,
        )
        try:
            receipt = _strict_canonical_mapping(
                receipt_payload,
                label="anchor receipt",
            )
            verified = verify_snapshot_anchor_receipt(
                receipt,
                trusted_public_key_path=required_publication_anchor_public_key_path(),
            )
            if not _receipt_closes_intent(
                verified,
                intent=intent,
                intent_sha256=hashlib.sha256(intent_payload).hexdigest(),
            ):
                raise SnapshotAnchorClientError(
                    "anchor receipt does not realize the submitted compare-and-append"
                )
        except Exception as exc:
            raise SnapshotAnchorIndeterminateError(
                "external anchor returned an invalid success response after a possible commit"
            ) from exc
        return receipt_payload

    def get_head(self) -> bytes:
        """Return and authenticate the authority's current signed HEAD receipt."""

        payload = self._request(
            "GET",
            self._domain_path("head"),
            None,
            expected_statuses={200},
        )
        receipt = _strict_canonical_mapping(payload, label="anchor HEAD receipt")
        verify_snapshot_anchor_receipt(
            receipt,
            trusted_public_key_path=required_publication_anchor_public_key_path(),
            trusted_public_key_dir=configured_publication_anchor_keyring_directory(),
        )
        return payload

    def observe_head(
        self,
        *,
        intent_payload: bytes,
        receipt_payload: bytes,
        pointer: Mapping[str, object],
        challenge_nonce: str | None = None,
    ) -> bytes:
        """Obtain and verify a fresh challenge-bound signed HEAD observation."""

        receipt = _strict_canonical_mapping(receipt_payload, label="anchor receipt")
        verify_snapshot_anchor_receipt(
            receipt,
            trusted_public_key_path=required_publication_anchor_public_key_path(),
            trusted_public_key_dir=configured_publication_anchor_keyring_directory(),
        )
        selected_nonce = challenge_nonce or secrets.token_hex(32)
        request_payload = canonical_json(
            {
                "schema_version": "lt_snapshot_anchor_head_request.v1",
                "publication_domain_sha256": publication_domain_sha256(),
                "expected_receipt_id": receipt["receipt_id"],
                "challenge_nonce": selected_nonce,
                "challenge_sha256": external_head_challenge_sha256(
                    pointer,
                    challenge_nonce=selected_nonce,
                ),
            }
        )
        observation_payload = self._request(
            "POST",
            self._domain_path("head-observations"),
            request_payload,
            expected_statuses={200},
        )
        observation = _strict_canonical_mapping(
            observation_payload,
            label="anchor HEAD observation",
        )
        verify_snapshot_anchor_head_observation(
            observation,
            trusted_public_key_path=required_publication_anchor_public_key_path(),
        )
        verify_external_publication_evidence(
            intent_payload=intent_payload,
            receipt_payload=receipt_payload,
            observation_payload=observation_payload,
            pointer=pointer,
            require_current=True,
            expected_challenge_nonce=selected_nonce,
        )
        return observation_payload

    def lookup_operation(
        self,
        operation_id: str,
        *,
        expected_receipt_payload: bytes | None = None,
        expected_intent_payload: bytes | None = None,
    ) -> bytes:
        """Return the immutable signed receipt for an exact operation retry."""

        if expected_receipt_payload is None and expected_intent_payload is None:
            raise SnapshotAnchorClientError(
                "operation lookup requires an expected receipt or intent"
            )
        canonical_operation = _canonical_operation_id(operation_id)
        payload = self._request(
            "GET",
            self._domain_path(f"operations/{quote(canonical_operation, safe='')}"),
            None,
            expected_statuses={200},
        )
        receipt = _strict_canonical_mapping(payload, label="anchor operation receipt")
        verified = verify_snapshot_anchor_receipt(
            receipt,
            trusted_public_key_path=required_publication_anchor_public_key_path(),
            trusted_public_key_dir=configured_publication_anchor_keyring_directory(),
        )
        if verified.get("operation_id") != canonical_operation:
            raise SnapshotAnchorClientError("anchor operation lookup returned another operation")
        if expected_receipt_payload is not None and payload != expected_receipt_payload:
            raise SnapshotAnchorClientError(
                "anchor operation lookup differs from the archived receipt"
            )
        if expected_intent_payload is not None:
            intent = _strict_canonical_mapping(
                expected_intent_payload,
                label="expected publication intent",
            )
            verify_snapshot_publication_intent(
                intent,
                trusted_public_key_path=required_publication_request_public_key_path(),
                trusted_public_key_dir=configured_publication_request_keyring_directory(),
                bootstrap_trusted_public_key_path=(
                    required_publication_bootstrap_public_key_path()
                    if intent.get("transition_type") == "BOOTSTRAP"
                    else None
                ),
                bootstrap_trusted_public_key_dir=(
                    configured_publication_bootstrap_keyring_directory()
                    if intent.get("transition_type") == "BOOTSTRAP"
                    else None
                ),
            )
            if not _receipt_closes_intent(
                verified,
                intent=intent,
                intent_sha256=hashlib.sha256(expected_intent_payload).hexdigest(),
            ):
                raise SnapshotAnchorClientError(
                    "anchor operation lookup does not close the expected intent"
                )
        return payload

    def _domain_path(self, suffix: str) -> str:
        return f"{self._base_path}/v1/publication-domains/{publication_domain_sha256()}/{suffix}"

    def _request(
        self,
        method: str,
        path: str,
        payload: bytes | None,
        *,
        expected_statuses: set[int],
        commit_may_have_occurred: bool = False,
    ) -> bytes:
        connection = http.client.HTTPSConnection(
            self._host,
            self._port,
            timeout=self._config.timeout_seconds,
            context=self._ssl_context,
        )
        headers = {
            "Accept": _JSON_CONTENT_TYPE,
            "Content-Type": _JSON_CONTENT_TYPE,
            "Cache-Control": "no-store",
        }
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            body = response.read(self._config.max_response_bytes + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise SnapshotAnchorIndeterminateError(
                "external anchor request failed with an indeterminate outcome"
            ) from exc
        finally:
            connection.close()
        if response.status not in expected_statuses:
            error_type: type[SnapshotAnchorClientError]
            if response.status in {409, 412}:
                error_type = SnapshotAnchorConflictError
            elif response.status in {401, 403}:
                error_type = SnapshotAnchorAuthenticationError
            elif response.status == 404:
                error_type = SnapshotAnchorNotFoundError
            elif response.status == 429 or 500 <= response.status <= 599:
                error_type = (
                    SnapshotAnchorIndeterminateError
                    if commit_may_have_occurred
                    else SnapshotAnchorUnavailableError
                )
            else:
                error_type = SnapshotAnchorProtocolError
            raise error_type(f"external anchor returned HTTP {response.status}")
        if len(body) > self._config.max_response_bytes:
            error_type = (
                SnapshotAnchorIndeterminateError
                if commit_may_have_occurred
                else SnapshotAnchorProtocolError
            )
            raise error_type("external anchor response exceeds size limit")
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip()
        if content_type != _JSON_CONTENT_TYPE:
            error_type = (
                SnapshotAnchorIndeterminateError
                if commit_may_have_occurred
                else SnapshotAnchorProtocolError
            )
            raise error_type("external anchor response is not JSON")
        return body


def load_anchor_client_config_from_environment(
    environment: Mapping[str, str],
) -> SnapshotAnchorClientConfig:
    """Build the explicit mTLS transport configuration from an environment."""

    names = {
        "base_url": "PFC_DATA_PUBLICATION_ANCHOR_BASE_URL",
        "ca_bundle": "PFC_DATA_PUBLICATION_ANCHOR_TLS_CA_PATH",
        "client_certificate": "PFC_DATA_PUBLICATION_ANCHOR_MTLS_CERT_PATH",
        "client_private_key": "PFC_DATA_PUBLICATION_ANCHOR_MTLS_KEY_PATH",
    }
    values = {field: str(environment.get(name, "")).strip() for field, name in names.items()}
    missing = sorted(names[field] for field, value in values.items() if not value)
    if missing:
        raise SnapshotAnchorClientError(
            f"external anchor client configuration is missing: {missing}"
        )
    return SnapshotAnchorClientConfig(
        base_url=values["base_url"],
        ca_bundle=Path(values["ca_bundle"]),
        client_certificate=Path(values["client_certificate"]),
        client_private_key=Path(values["client_private_key"]),
    )


def verify_external_operation_receipt(
    receipt_payload: bytes,
    *,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    """Require the live external authority to return the exact archived receipt."""

    receipt = _strict_canonical_mapping(
        receipt_payload,
        label="archived anchor operation receipt",
    )
    verify_snapshot_anchor_receipt(
        receipt,
        trusted_public_key_path=required_publication_anchor_public_key_path(),
        trusted_public_key_dir=configured_publication_anchor_keyring_directory(),
    )
    selected_environment = os.environ if environment is None else environment
    client = SnapshotAnchorClient(load_anchor_client_config_from_environment(selected_environment))
    return client.lookup_operation(
        str(receipt.get("operation_id", "")),
        expected_receipt_payload=receipt_payload,
    )


def _validated_config(config: SnapshotAnchorClientConfig) -> SnapshotAnchorClientConfig:
    parsed = urlsplit(str(config.base_url).strip())
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SnapshotAnchorClientError("external anchor base URL must be plain HTTPS")
    if config.timeout_seconds <= 0 or config.timeout_seconds > 60:
        raise SnapshotAnchorClientError("external anchor timeout is invalid")
    if config.max_response_bytes < 1 or config.max_response_bytes > _MAX_RESPONSE_BYTES:
        raise SnapshotAnchorClientError("external anchor response size limit is invalid")
    paths = []
    payloads = []
    for value in (
        config.ca_bundle,
        config.client_certificate,
        config.client_private_key,
    ):
        path = assert_absolute_path_has_no_links(Path(value).expanduser())
        payloads.append(
            read_stable_single_link_file(path, label="external anchor TLS credential")
        )
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise SnapshotAnchorClientError("external anchor TLS credential paths must be distinct")
    return SnapshotAnchorClientConfig(
        base_url=parsed.geturl().rstrip("/"),
        ca_bundle=paths[0],
        client_certificate=paths[1],
        client_private_key=paths[2],
        timeout_seconds=float(config.timeout_seconds),
        max_response_bytes=int(config.max_response_bytes),
        _ca_payload=payloads[0],
        _certificate_payload=payloads[1],
        _private_key_payload=payloads[2],
    )


def _build_ssl_context(config: SnapshotAnchorClientConfig) -> ssl.SSLContext:
    if (
        config._ca_payload is None
        or config._certificate_payload is None
        or config._private_key_payload is None
    ):
        config = _validated_config(config)
    assert config._ca_payload is not None
    assert config._certificate_payload is not None
    assert config._private_key_payload is not None
    ca_payload = config._ca_payload
    certificate_payload = config._certificate_payload
    private_key_payload = config._private_key_payload
    try:
        ca_text = ca_payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SnapshotAnchorClientError("external anchor TLS CA bundle is not PEM") from exc
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        context.load_verify_locations(cadata=ca_text)
        with tempfile.TemporaryDirectory(
            prefix="pfc-lt-mtls-",
            dir=_publisher_supervisor_scratch_root(),
        ) as temporary:
            temporary_root = Path(temporary)
            certificate_path = temporary_root / "client-certificate.pem"
            private_key_path = temporary_root / "client-private-key.pem"
            _write_private_temporary_file(certificate_path, certificate_payload)
            _write_private_temporary_file(private_key_path, private_key_payload)
            context.load_cert_chain(
                certfile=str(certificate_path),
                keyfile=str(private_key_path),
            )
    except (OSError, ssl.SSLError) as exc:
        raise SnapshotAnchorClientError(
            "external anchor TLS credentials cannot initialize mTLS"
        ) from exc
    return context


def _publisher_supervisor_scratch_root() -> str | None:
    raw = os.environ.get("_PFC_SNAPSHOT_PUBLISHER_CAPTURE_ROOT", "").strip()
    if not raw:
        return None
    try:
        root = assert_absolute_path_has_no_links(Path(raw).expanduser())
    except ValueError as exc:
        raise SnapshotAnchorClientError(
            "publisher mTLS scratch root is invalid"
        ) from exc
    if not root.is_dir():
        raise SnapshotAnchorClientError(
            "publisher mTLS scratch root is unavailable"
        )
    return str(root)


def _write_private_temporary_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotAnchorClientError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise SnapshotAnchorClientError(f"{label} is not canonical JSON")
    return value


def _receipt_closes_intent(
    receipt: Mapping[str, object],
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
) -> bool:
    return (
        receipt.get("intent_id") == intent.get("intent_id")
        and receipt.get("intent_sha256") == intent_sha256
        and receipt.get("operation_id") == intent.get("operation_id")
        and receipt.get("transition_type") == intent.get("transition_type")
        and receipt.get("generation_id") == intent.get("generation_id")
        and receipt.get("contract_sha256") == intent.get("contract_sha256")
        and receipt.get("generation_inventory_sha256") == intent.get("generation_inventory_sha256")
        and receipt.get("legacy_pointer_sha256") == intent.get("legacy_pointer_sha256")
        and receipt.get("bootstrap_authorization_id")
        == intent.get("bootstrap_authorization_id")
        and receipt.get("bootstrap_authorization_sha256")
        == intent.get("bootstrap_authorization_sha256")
        and receipt.get("sequence") == int(intent["expected_anchor_sequence"]) + 1
        and receipt.get("previous_receipt_id") == intent.get("expected_anchor_receipt_id")
    )


def _canonical_operation_id(value: object) -> str:
    raw = str(value or "").strip().lower()
    try:
        canonical = str(uuid.UUID(raw))
    except ValueError as exc:
        raise SnapshotAnchorClientError("operation_id is not a canonical UUID") from exc
    if raw != canonical:
        raise SnapshotAnchorClientError("operation_id is not a canonical UUID")
    return canonical
