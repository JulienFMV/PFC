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
    AcquisitionAuthenticationError,
    _canonical_json_bytes,
    _public_key_id,
    _read_stable_regular_file,
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


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(
            _read_stable_regular_file(path, label="acquisition private key"),
            password=None,
        )
    except AcquisitionAuthenticationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise AcquisitionAuthenticationError("cannot load acquisition private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AcquisitionAuthenticationError("acquisition private key is not Ed25519")
    return key
