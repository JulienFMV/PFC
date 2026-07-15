"""External authorization for production QUOTE_CONFLICT policy artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


TRUSTED_POLICY_PUBLIC_KEY_ENV = "PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH"
POLICY_SCHEMA = "ch_quote_conflict_source_hierarchy_policy.v1"


class QuoteConflictPolicyAuthenticationError(ValueError):
    """Raised when a production policy lacks independent authorization."""


def sign_quote_conflict_policy(
    policy: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    unsigned = dict(policy)
    unsigned.pop("approval_attestation", None)
    if unsigned.get("schema_version") != POLICY_SCHEMA:
        raise QuoteConflictPolicyAuthenticationError("quote-conflict policy schema mismatch")
    private_key = _load_private_key(private_key_path)
    payload = _canonical_json_bytes(unsigned)
    signed = dict(unsigned)
    signed["approval_attestation"] = {
        "algorithm": "Ed25519",
        "key_id": _public_key_id(private_key.public_key()),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    return signed


def verify_quote_conflict_policy(policy: Mapping[str, object]) -> dict[str, object]:
    selected = os.environ.get(TRUSTED_POLICY_PUBLIC_KEY_ENV)
    if not selected:
        raise QuoteConflictPolicyAuthenticationError(
            f"{TRUSTED_POLICY_PUBLIC_KEY_ENV} is required for production approval"
        )
    public_key = _load_public_key(selected)
    unsigned = dict(policy)
    attestation = unsigned.pop("approval_attestation", None)
    if not isinstance(attestation, Mapping):
        raise QuoteConflictPolicyAuthenticationError("quote-conflict approval attestation is missing")
    if attestation.get("algorithm") != "Ed25519":
        raise QuoteConflictPolicyAuthenticationError("quote-conflict approval algorithm is unsupported")
    if str(attestation.get("key_id", "")) != _public_key_id(public_key):
        raise QuoteConflictPolicyAuthenticationError("quote-conflict approval key is not trusted")
    payload = _canonical_json_bytes(unsigned)
    if str(attestation.get("payload_sha256", "")) != hashlib.sha256(payload).hexdigest():
        raise QuoteConflictPolicyAuthenticationError("quote-conflict policy payload hash mismatch")
    try:
        signature = base64.b64decode(str(attestation.get("value_base64", "")), validate=True)
    except (TypeError, ValueError) as exc:
        raise QuoteConflictPolicyAuthenticationError("quote-conflict signature is not valid base64") from exc
    try:
        public_key.verify(signature, payload)
    except InvalidSignature as exc:
        raise QuoteConflictPolicyAuthenticationError("quote-conflict approval signature is invalid") from exc
    return unsigned


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise QuoteConflictPolicyAuthenticationError("quote-conflict private key is not Ed25519")
    return key


def _load_public_key(path: str | Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise QuoteConflictPolicyAuthenticationError("quote-conflict public key is not Ed25519")
    return key


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
