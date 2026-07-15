"""Authenticated contract for LT candidate promotion receipts."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

REQUIRED_PROMOTION_GATE_IDS = frozenset(
    {
        "candidate_run_freshness",
        "lambda_calibration_artifact_present",
        "production_export_path_parity",
        "delivered_product_normalization_audit",
        "lt_probabilistic_output_governance",
        "ompex_training_selection_exclusion",
        "selected_config_manifest_parity",
        "selected_config_production_approval",
    }
)
PROMOTION_POLICY_VERSION = "lt_promotion_policy.deterministic_only.v1"
PROMOTION_GATE_IDS_BY_POLICY = {
    PROMOTION_POLICY_VERSION: REQUIRED_PROMOTION_GATE_IDS,
}
SIGNATURE_ALGORITHM = "Ed25519"
ROLLBACK_AUTHORIZATION_SCHEMA = "rollback_authorization.v1"
_CANONICAL_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLLBACK_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "operation_id",
        "release_root_sha256",
        "expected_current_event_id",
        "target_promotion_event_id",
        "target_run_id",
        "target_bundle_manifest_sha256",
        "reason_code",
        "incident_id",
        "authorized_at_utc",
        "expires_at_utc",
    }
)


class ReceiptAuthenticationError(ValueError):
    """Raised when a promotion receipt is unsigned, forged or incomplete."""


class EventAuthenticationError(ValueError):
    """Raised when a promotion transition event is unsigned or forged."""


class RollbackAuthorizationAuthenticationError(ValueError):
    """Raised when a rollback authorization is malformed, forged or untrusted."""


def sign_promotion_receipt(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    unsigned = dict(payload)
    unsigned.pop("receipt_id", None)
    unsigned.pop("signature", None)
    if unsigned.get("schema_version") != "promotion_receipt.v2":
        raise ReceiptAuthenticationError("only promotion_receipt.v2 can be signed")
    if unsigned.get("promotion_policy_version") != PROMOTION_POLICY_VERSION:
        raise ReceiptAuthenticationError(
            "promotion receipt policy version is unsupported"
        )
    if unsigned.get("authorization_mode") not in {
        "REGISTERED_PROMOTION",
        "DIAGNOSTIC_ONLY",
    }:
        raise ReceiptAuthenticationError(
            "promotion receipt authorization mode is unsupported"
        )
    receipt_id = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    signed_payload = dict(unsigned)
    signed_payload["receipt_id"] = receipt_id
    private_key = _load_private_key(private_key_path)
    public_key = private_key.public_key()
    signature = private_key.sign(_canonical_json_bytes(signed_payload))
    signed_payload["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": public_key_id(public_key),
        "value_base64": base64.b64encode(signature).decode("ascii"),
    }
    return signed_payload


def verify_promotion_receipt_signature(
    receipt: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, object]:
    payload = dict(receipt)
    signature_block = payload.pop("signature", None)
    if not isinstance(signature_block, Mapping):
        raise ReceiptAuthenticationError("promotion receipt signature is missing")
    if set(signature_block) != {"algorithm", "key_id", "value_base64"}:
        raise ReceiptAuthenticationError("promotion receipt signature block is invalid")
    if signature_block.get("algorithm") != SIGNATURE_ALGORITHM:
        raise ReceiptAuthenticationError("promotion receipt signature algorithm is unsupported")
    receipt_id = str(payload.get("receipt_id", ""))
    unsigned = dict(payload)
    unsigned.pop("receipt_id", None)
    expected_receipt_id = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if receipt_id != expected_receipt_id:
        raise ReceiptAuthenticationError("promotion receipt_id does not match canonical payload")
    public_key = _load_public_key(trusted_public_key_path)
    if str(signature_block.get("key_id", "")) != public_key_id(public_key):
        raise ReceiptAuthenticationError("promotion receipt signing key is not trusted")
    try:
        signature = base64.b64decode(
            str(signature_block.get("value_base64", "")),
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise ReceiptAuthenticationError("promotion receipt signature is not valid base64") from exc
    try:
        public_key.verify(signature, _canonical_json_bytes(payload))
    except InvalidSignature as exc:
        raise ReceiptAuthenticationError("promotion receipt signature is invalid") from exc
    return payload


def validate_required_gate_set(
    receipt: Mapping[str, object],
    *,
    require_pass: bool = True,
    allow_historical_policy: bool = False,
) -> None:
    policy_version = receipt.get("promotion_policy_version")
    if not isinstance(policy_version, str) or policy_version not in PROMOTION_GATE_IDS_BY_POLICY:
        raise ReceiptAuthenticationError(
            "promotion receipt policy version is unsupported"
        )
    if not allow_historical_policy and policy_version != PROMOTION_POLICY_VERSION:
        raise ReceiptAuthenticationError(
            "promotion receipt policy version is not current"
        )
    required_gate_ids = PROMOTION_GATE_IDS_BY_POLICY[policy_version]
    rows = receipt.get("governance_gates")
    if not isinstance(rows, list):
        raise ReceiptAuthenticationError("promotion receipt governance gates are missing")
    gate_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReceiptAuthenticationError("promotion receipt governance gate is invalid")
        gate_id = str(row.get("gate_id", ""))
        if not gate_id:
            raise ReceiptAuthenticationError("promotion receipt governance gate_id is missing")
        if require_pass and row.get("status") != "PASS":
            raise ReceiptAuthenticationError("promotion receipt contains a non-PASS governance gate")
        gate_ids.append(gate_id)
    if len(gate_ids) != len(set(gate_ids)):
        raise ReceiptAuthenticationError("promotion receipt contains duplicate governance gates")
    if set(gate_ids) != set(required_gate_ids):
        missing = sorted(set(required_gate_ids).difference(gate_ids))
        extra = sorted(set(gate_ids).difference(required_gate_ids))
        raise ReceiptAuthenticationError(
            f"promotion receipt governance gate set mismatch; missing={missing}; extra={extra}"
        )


def validate_promotion_receipt_decision(receipt: Mapping[str, object]) -> bool:
    """Validate the canonical capstone decision encoded in a signed receipt."""

    approved = receipt.get("approved")
    if not isinstance(approved, bool):
        raise ReceiptAuthenticationError("promotion receipt approved flag is invalid")
    summary = receipt.get("decision_summary")
    if not isinstance(summary, Mapping):
        raise ReceiptAuthenticationError("promotion receipt decision_summary is missing")
    if summary.get("approved") is not approved:
        raise ReceiptAuthenticationError("promotion receipt decision approved flag mismatch")
    expected_status = "PROMOTION_EVIDENCE_PASS" if approved else "BLOCKED"
    if str(summary.get("status", "")) != expected_status:
        raise ReceiptAuthenticationError("promotion receipt decision status mismatch")
    blocking_count = summary.get("blocking_count")
    if isinstance(blocking_count, bool):
        raise ReceiptAuthenticationError("promotion receipt blocking_count is invalid")
    try:
        blocking = int(blocking_count)
    except (TypeError, ValueError) as exc:
        raise ReceiptAuthenticationError("promotion receipt blocking_count is invalid") from exc
    if (blocking == 0) is not approved:
        raise ReceiptAuthenticationError("promotion receipt blocking_count contradicts approval")
    return approved


def sign_rollback_authorization(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    """Sign one exact rollback authorization without evaluating its freshness."""

    unsigned = dict(payload)
    unsigned.pop("authorization_id", None)
    unsigned.pop("signature", None)
    _validate_rollback_authorization_payload(unsigned)
    authorization_id = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    signed = {**unsigned, "authorization_id": authorization_id}
    private_key = _load_rollback_private_key(private_key_path)
    signed["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": public_key_id(private_key.public_key()),
        "value_base64": base64.b64encode(
            private_key.sign(_canonical_json_bytes(signed))
        ).decode("ascii"),
    }
    return signed


def verify_rollback_authorization_signature(
    authorization: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, object]:
    """Authenticate and structurally validate a rollback authorization."""

    payload = dict(authorization)
    signature_block = payload.pop("signature", None)
    if not isinstance(signature_block, Mapping):
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization signature is missing"
        )
    if set(signature_block) != {"algorithm", "key_id", "value_base64"}:
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization signature block is invalid"
        )
    if signature_block.get("algorithm") != SIGNATURE_ALGORITHM:
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization signature algorithm is unsupported"
        )

    authorization_id = str(payload.get("authorization_id", ""))
    if not _CANONICAL_SHA256_RE.fullmatch(authorization_id):
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization_id is not a canonical SHA-256"
        )
    unsigned = dict(payload)
    unsigned.pop("authorization_id", None)
    _validate_rollback_authorization_payload(unsigned)
    expected_id = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if authorization_id != expected_id:
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization_id does not match canonical payload"
        )

    public_key = _load_rollback_public_key(trusted_public_key_path)
    expected_key_id = public_key_id(public_key)
    signature_key_id = str(signature_block.get("key_id", ""))
    if not _CANONICAL_SHA256_RE.fullmatch(signature_key_id):
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization signing key_id is invalid"
        )
    if signature_key_id != expected_key_id:
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization signing key is not trusted"
        )
    try:
        signature = base64.b64decode(
            str(signature_block.get("value_base64", "")),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization signature is not valid base64"
        ) from exc
    try:
        public_key.verify(signature, _canonical_json_bytes(payload))
    except InvalidSignature as exc:
        raise RollbackAuthorizationAuthenticationError(
            "rollback authorization signature is invalid"
        ) from exc
    return payload


def sign_promotion_event(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    unsigned = dict(payload)
    unsigned.pop("event_id", None)
    unsigned.pop("signature", None)
    if unsigned.get("schema_version") != "promotion_event.v2":
        raise EventAuthenticationError("only promotion_event.v2 can be signed")
    event_id = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    signed = dict(unsigned)
    signed["event_id"] = event_id
    private_key = _load_private_key(private_key_path)
    signature = private_key.sign(_canonical_json_bytes(signed))
    signed["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": public_key_id(private_key.public_key()),
        "value_base64": base64.b64encode(signature).decode("ascii"),
    }
    return signed


def verify_promotion_event_signature(
    event: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, object]:
    payload = dict(event)
    signature_block = payload.pop("signature", None)
    if not isinstance(signature_block, Mapping):
        raise EventAuthenticationError("promotion event signature is missing")
    if set(signature_block) != {"algorithm", "key_id", "value_base64"}:
        raise EventAuthenticationError("promotion event signature block is invalid")
    if signature_block.get("algorithm") != SIGNATURE_ALGORITHM:
        raise EventAuthenticationError("promotion event signature algorithm is unsupported")
    event_id = str(payload.get("event_id", ""))
    unsigned = dict(payload)
    unsigned.pop("event_id", None)
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != event_id:
        raise EventAuthenticationError("promotion event_id does not match canonical payload")
    public_key = _load_public_key(trusted_public_key_path)
    if str(signature_block.get("key_id", "")) != public_key_id(public_key):
        raise EventAuthenticationError("promotion event signing key is not trusted")
    try:
        signature = base64.b64decode(str(signature_block.get("value_base64", "")), validate=True)
    except (TypeError, ValueError) as exc:
        raise EventAuthenticationError("promotion event signature is not valid base64") from exc
    try:
        public_key.verify(signature, _canonical_json_bytes(payload))
    except InvalidSignature as exc:
        raise EventAuthenticationError("promotion event signature is invalid") from exc
    return payload


def sign_promotion_head(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    unsigned = dict(payload)
    unsigned.pop("head_id", None)
    unsigned.pop("signature", None)
    if unsigned.get("schema_version") != "promotion_head.v1":
        raise EventAuthenticationError("only promotion_head.v1 can be signed")
    head_id = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    signed = {**unsigned, "head_id": head_id}
    private_key = _load_private_key(private_key_path)
    signed["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": public_key_id(private_key.public_key()),
        "value_base64": base64.b64encode(
            private_key.sign(_canonical_json_bytes(signed))
        ).decode("ascii"),
    }
    return signed


def verify_promotion_head_signature(
    head: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, object]:
    payload = dict(head)
    signature_block = payload.pop("signature", None)
    if not isinstance(signature_block, Mapping):
        raise EventAuthenticationError("promotion head signature is missing")
    if set(signature_block) != {"algorithm", "key_id", "value_base64"}:
        raise EventAuthenticationError("promotion head signature block is invalid")
    if signature_block.get("algorithm") != SIGNATURE_ALGORITHM:
        raise EventAuthenticationError("promotion head signature algorithm is unsupported")
    head_id = str(payload.get("head_id", ""))
    unsigned = dict(payload)
    unsigned.pop("head_id", None)
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != head_id:
        raise EventAuthenticationError("promotion head_id does not match canonical payload")
    public_key = _load_public_key(trusted_public_key_path)
    if str(signature_block.get("key_id", "")) != public_key_id(public_key):
        raise EventAuthenticationError("promotion head signing key is not trusted")
    try:
        signature = base64.b64decode(str(signature_block.get("value_base64", "")), validate=True)
    except (TypeError, ValueError) as exc:
        raise EventAuthenticationError("promotion head signature is not valid base64") from exc
    try:
        public_key.verify(signature, _canonical_json_bytes(payload))
    except InvalidSignature as exc:
        raise EventAuthenticationError("promotion head signature is invalid") from exc
    return payload


def public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def public_key_id_from_public_key_path(path: str | Path) -> str:
    """Return the canonical authority ID for an Ed25519 public-key PEM."""

    return public_key_id(_load_public_key(path))


def public_key_id_from_private_key_path(path: str | Path) -> str:
    """Return the canonical authority ID for an Ed25519 private-key PEM."""

    return public_key_id(_load_private_key(path).public_key())


def _validate_rollback_authorization_payload(payload: Mapping[str, object]) -> None:
    fields = set(payload)
    if fields != _ROLLBACK_AUTHORIZATION_FIELDS:
        missing = sorted(_ROLLBACK_AUTHORIZATION_FIELDS.difference(fields))
        extra = sorted(fields.difference(_ROLLBACK_AUTHORIZATION_FIELDS))
        raise RollbackAuthorizationAuthenticationError(
            f"rollback authorization field set mismatch; missing={missing}; extra={extra}"
        )
    if payload.get("schema_version") != ROLLBACK_AUTHORIZATION_SCHEMA:
        raise RollbackAuthorizationAuthenticationError(
            f"only {ROLLBACK_AUTHORIZATION_SCHEMA} can be signed"
        )
    for field in (
        "expected_current_event_id",
        "target_promotion_event_id",
        "target_bundle_manifest_sha256",
        "release_root_sha256",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not _CANONICAL_SHA256_RE.fullmatch(value):
            raise RollbackAuthorizationAuthenticationError(
                f"rollback authorization {field} is not a canonical SHA-256"
            )
    for field in (
        "operation_id",
        "target_run_id",
        "reason_code",
        "incident_id",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise RollbackAuthorizationAuthenticationError(
                f"rollback authorization {field} is invalid"
            )
    for field in ("authorized_at_utc", "expires_at_utc"):
        value = payload.get(field)
        if not isinstance(value, str):
            raise RollbackAuthorizationAuthenticationError(
                f"rollback authorization {field} is invalid"
            )
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RollbackAuthorizationAuthenticationError(
                f"rollback authorization {field} is invalid"
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise RollbackAuthorizationAuthenticationError(
                f"rollback authorization {field} is timezone-naive"
            )
        if timestamp.utcoffset().total_seconds() != 0:
            raise RollbackAuthorizationAuthenticationError(
                f"rollback authorization {field} is not UTC"
            )


def _load_rollback_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        return _load_private_key(path)
    except ReceiptAuthenticationError as exc:
        raise RollbackAuthorizationAuthenticationError(
            "cannot load Ed25519 rollback authorization private key"
        ) from exc


def _load_rollback_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        return _load_public_key(path)
    except ReceiptAuthenticationError as exc:
        raise RollbackAuthorizationAuthenticationError(
            "cannot load trusted Ed25519 rollback authorization public key"
        ) from exc


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise ReceiptAuthenticationError("cannot load Ed25519 promotion private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ReceiptAuthenticationError("promotion private key is not Ed25519")
    return key


def _load_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(Path(path).read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise ReceiptAuthenticationError("cannot load trusted Ed25519 promotion public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ReceiptAuthenticationError("trusted promotion public key is not Ed25519")
    return key


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
