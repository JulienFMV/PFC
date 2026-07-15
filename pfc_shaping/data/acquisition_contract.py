"""Authenticated provenance contract for calibration-eligible LT inputs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.path_safety import assert_absolute_path_has_no_links

TRUSTED_ACQUISITION_PUBLIC_KEY_ENV = "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH"
TRUSTED_TIME_PUBLIC_KEY_ENV = "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH"
TRUSTED_TIME_JOURNAL_ID_ENV = "PFC_DATA_TIMESTAMP_JOURNAL_ID"
SIGNATURE_ALGORITHM = "Ed25519"
TRUSTED_TIME_RECEIPT_SCHEMA = "eex_source_receipt.v1"
ATTESTABLE_ACQUISITION_SCHEMAS = {
    "lt_input_snapshot.v1",
    "eex_historical_vintage_catalog.v1",
}


class AcquisitionAuthenticationError(ValueError):
    """Raised when governed input provenance cannot be authenticated."""


def verify_acquisition_contract(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Verify against the IT-controlled trust anchor; callers cannot replace it."""

    key_value = os.environ.get(TRUSTED_ACQUISITION_PUBLIC_KEY_ENV)
    if not key_value:
        raise AcquisitionAuthenticationError(
            f"{TRUSTED_ACQUISITION_PUBLIC_KEY_ENV} is required for governed acquisition"
        )
    unsigned = dict(contract)
    attestation = unsigned.pop("acquisition_attestation", None)
    if unsigned.get("schema_version") not in ATTESTABLE_ACQUISITION_SCHEMAS:
        raise AcquisitionAuthenticationError(
            "acquisition contract schema cannot be verified"
        )
    if not isinstance(attestation, Mapping):
        raise AcquisitionAuthenticationError("acquisition attestation is missing")
    if attestation.get("algorithm") != SIGNATURE_ALGORITHM:
        raise AcquisitionAuthenticationError("acquisition attestation algorithm is unsupported")
    public_key = _select_trusted_public_key(
        attestation,
        primary_path=key_value,
        label="acquisition",
    )
    payload = _canonical_json_bytes(unsigned)
    if str(attestation.get("payload_sha256", "")) != hashlib.sha256(payload).hexdigest():
        raise AcquisitionAuthenticationError("acquisition payload hash mismatch")
    try:
        signature = base64.b64decode(
            str(attestation.get("value_base64", "")),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise AcquisitionAuthenticationError("acquisition signature is not valid base64") from exc
    try:
        public_key.verify(signature, payload)
    except InvalidSignature as exc:
        raise AcquisitionAuthenticationError("acquisition signature is invalid") from exc
    return unsigned


def verify_trusted_time_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    key_value = os.environ.get(TRUSTED_TIME_PUBLIC_KEY_ENV)
    if not key_value:
        raise AcquisitionAuthenticationError(
            f"{TRUSTED_TIME_PUBLIC_KEY_ENV} is required for historical EEX vintages"
        )
    expected_journal_id = os.environ.get(TRUSTED_TIME_JOURNAL_ID_ENV, "").strip()
    if not expected_journal_id:
        raise AcquisitionAuthenticationError(
            f"{TRUSTED_TIME_JOURNAL_ID_ENV} is required for historical EEX vintages"
        )
    unsigned = dict(receipt)
    attestation = unsigned.pop("trusted_time_attestation", None)
    if unsigned.get("schema_version") != TRUSTED_TIME_RECEIPT_SCHEMA:
        raise AcquisitionAuthenticationError("trusted-time receipt schema is unsupported")
    if str(unsigned.get("journal_id", "")) != expected_journal_id:
        raise AcquisitionAuthenticationError("trusted-time journal is not the governed journal")
    if not isinstance(attestation, Mapping):
        raise AcquisitionAuthenticationError("trusted-time attestation is missing")
    if attestation.get("algorithm") != SIGNATURE_ALGORITHM:
        raise AcquisitionAuthenticationError("trusted-time algorithm is unsupported")
    public_key = _select_trusted_public_key(
        attestation,
        primary_path=key_value,
        label="trusted-time",
    )
    payload = _canonical_json_bytes(unsigned)
    if str(attestation.get("payload_sha256", "")) != hashlib.sha256(payload).hexdigest():
        raise AcquisitionAuthenticationError("trusted-time payload hash mismatch")
    try:
        signature = base64.b64decode(
            str(attestation.get("value_base64", "")),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise AcquisitionAuthenticationError("trusted-time signature is not valid base64") from exc
    try:
        public_key.verify(signature, payload)
    except InvalidSignature as exc:
        raise AcquisitionAuthenticationError("trusted-time signature is invalid") from exc
    return unsigned


def _load_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(
            _read_stable_regular_file(path, label="trusted acquisition public key")
        )
    except AcquisitionAuthenticationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise AcquisitionAuthenticationError("cannot load trusted acquisition public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise AcquisitionAuthenticationError("trusted acquisition public key is not Ed25519")
    return key


def _absolute_trusted_key_path(value: str | Path) -> Path:
    try:
        return assert_absolute_path_has_no_links(value)
    except (OSError, ValueError) as exc:
        raise AcquisitionAuthenticationError(
            "trusted public key path must be absolute and contain no links"
        ) from exc


def _select_trusted_public_key(
    attestation: Mapping[str, object],
    *,
    primary_path: str | Path,
    label: str,
) -> Ed25519PublicKey:
    selected_key_id = str(attestation.get("key_id", ""))
    if len(selected_key_id) != 64 or any(
        character not in "0123456789abcdef" for character in selected_key_id
    ):
        raise AcquisitionAuthenticationError(f"{label} signing key id is invalid")
    primary = _load_public_key(_absolute_trusted_key_path(primary_path))
    if _public_key_id(primary) == selected_key_id:
        return primary
    raise AcquisitionAuthenticationError(
        f"{label} signing key is not the active trusted key; historical replay is unsupported"
    )


def _read_stable_regular_file(path: str | Path, *, label: str) -> bytes:
    descriptor: int | None = None
    try:
        lexical = assert_absolute_path_has_no_links(path)
        before = lexical.lstat()
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise AcquisitionAuthenticationError(
                f"{label} must be a regular file with exactly one link"
            )
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        flags |= int(getattr(os, "O_NOINHERIT", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(lexical, flags)
        opened_before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            raw = handle.read(64 * 1024 + 1)
            opened_after = os.fstat(handle.fileno())
        after = lexical.lstat()
    except AcquisitionAuthenticationError:
        raise
    except (AttributeError, OSError, ValueError) as exc:
        raise AcquisitionAuthenticationError(f"cannot load {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > 64 * 1024:
        raise AcquisitionAuthenticationError(f"{label} is unexpectedly large")
    identities = [
        _file_identity(metadata)
        for metadata in (before, opened_before, opened_after, after)
    ]
    if any(identity != identities[0] for identity in identities[1:]):
        raise AcquisitionAuthenticationError(f"{label} changed while it was read")
    return raw


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_nlink),
    )


def _public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
