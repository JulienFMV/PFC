"""Source-only signing authority for governed LT acquisition contracts."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.data.acquisition_contract import (
    ATTESTABLE_ACQUISITION_SCHEMAS,
    SIGNATURE_ALGORITHM,
    SOURCE_ACQUISITION_RECEIPT_SCHEMA,
    SOURCE_JOURNAL_CHECKPOINT_SCHEMA,
    TRUSTED_TIME_RECEIPT_SCHEMA,
    AcquisitionAuthenticationError,
    _canonical_json_bytes,
    _public_key_id,
    _read_stable_regular_file,
    source_acquisition_receipt_id,
)


def sign_acquisition_contract(
    contract: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    """Sign one acquisition contract outside the governed runtime wheel."""

    unsigned = dict(contract)
    unsigned.pop("acquisition_attestation", None)
    if unsigned.get("schema_version") not in ATTESTABLE_ACQUISITION_SCHEMAS:
        raise AcquisitionAuthenticationError(
            "acquisition contract schema cannot be attested"
        )
    private_key = _load_private_key(private_key_path)
    public_key = private_key.public_key()
    payload = _canonical_json_bytes(unsigned)
    signed = dict(unsigned)
    signed["acquisition_attestation"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(public_key),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    return signed


def sign_source_acquisition_receipt(
    receipt: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    """Sign one prospective PIT receipt with the independent time authority."""

    unsigned = dict(receipt)
    unsigned.pop("trusted_time_attestation", None)
    unsigned.pop("receipt_id", None)
    if unsigned.get("schema_version") != SOURCE_ACQUISITION_RECEIPT_SCHEMA:
        raise AcquisitionAuthenticationError("source receipt schema cannot be attested")
    unsigned["receipt_id"] = source_acquisition_receipt_id(unsigned)
    private_key = _load_private_key(private_key_path, label="trusted-time private key")
    public_key = private_key.public_key()
    payload = _canonical_json_bytes(unsigned)
    signed = dict(unsigned)
    signed["trusted_time_attestation"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(public_key),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    return signed


def sign_eex_trusted_time_receipt(
    receipt: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    """Sign one legacy-compatible immutable EEX workbook time receipt."""

    unsigned = dict(receipt)
    unsigned.pop("trusted_time_attestation", None)
    if unsigned.get("schema_version") != TRUSTED_TIME_RECEIPT_SCHEMA:
        raise AcquisitionAuthenticationError("EEX trusted-time receipt cannot be attested")
    private_key = _load_private_key(private_key_path, label="trusted-time private key")
    public_key = private_key.public_key()
    payload = _canonical_json_bytes(unsigned)
    signed = dict(unsigned)
    signed["trusted_time_attestation"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(public_key),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    return signed


def sign_source_journal_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    """Sign one externally governed source-journal checkpoint."""

    unsigned = dict(checkpoint)
    unsigned.pop("journal_attestation", None)
    unsigned.pop("checkpoint_id", None)
    if unsigned.get("schema_version") != SOURCE_JOURNAL_CHECKPOINT_SCHEMA:
        raise AcquisitionAuthenticationError("source journal checkpoint cannot be attested")
    unsigned["checkpoint_id"] = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    private_key = _load_private_key(private_key_path, label="source journal private key")
    public_key = private_key.public_key()
    payload = _canonical_json_bytes(unsigned)
    signed = dict(unsigned)
    signed["journal_attestation"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(public_key),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    return signed


def _load_private_key(
    path: str | Path,
    *,
    label: str = "acquisition private key",
) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(
            _read_stable_regular_file(path, label=label),
            password=None,
        )
    except AcquisitionAuthenticationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise AcquisitionAuthenticationError(f"cannot load {label}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AcquisitionAuthenticationError("acquisition private key is not Ed25519")
    return key
