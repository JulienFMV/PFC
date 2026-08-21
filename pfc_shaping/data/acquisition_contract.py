"""Authenticated provenance contract for calibration-eligible LT inputs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    process_immutable_file_bytes,
)

TRUSTED_ACQUISITION_PUBLIC_KEY_ENV = "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH"
TRUSTED_ACQUISITION_PUBLIC_KEY_DIR_ENV = "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_DIR"
TRUSTED_TIME_PUBLIC_KEY_ENV = "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH"
TRUSTED_TIME_PUBLIC_KEY_DIR_ENV = "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_DIR"
TRUSTED_TIME_JOURNAL_ID_ENV = "PFC_DATA_TIMESTAMP_JOURNAL_ID"
TRUSTED_JOURNAL_PUBLIC_KEY_ENV = "PFC_DATA_JOURNAL_TRUSTED_PUBLIC_KEY_PATH"
TRUSTED_JOURNAL_PUBLIC_KEY_DIR_ENV = "PFC_DATA_JOURNAL_TRUSTED_PUBLIC_KEY_DIR"
SIGNATURE_ALGORITHM = "Ed25519"
TRUSTED_TIME_RECEIPT_SCHEMA = "eex_source_receipt.v1"
SOURCE_ACQUISITION_RECEIPT_SCHEMA = "source_acquisition_receipt.v2"
SOURCE_JOURNAL_CHECKPOINT_SCHEMA = "source_journal_checkpoint.v1"
GOVERNED_LT_INPUT_SNAPSHOT_SCHEMA = "lt_input_snapshot.v2"
PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA = "lt_input_snapshot.v3"
GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS = frozenset(
    {GOVERNED_LT_INPUT_SNAPSHOT_SCHEMA, PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA}
)
ATTESTABLE_ACQUISITION_SCHEMAS = {
    "lt_input_snapshot.v1",
    GOVERNED_LT_INPUT_SNAPSHOT_SCHEMA,
    PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
    "eex_historical_vintage_catalog.v1",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPLAY_GOVERNED_LT_INPUT_ROLES = frozenset(
    {"epex_ch", "epex_de", "epex_at", "epex_fr", "epex_it", "entso", "hydro"}
)


class AcquisitionAuthenticationError(ValueError):
    """Raised when governed input provenance cannot be authenticated."""


def verify_acquisition_contract(
    contract: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path | None = None,
    trusted_public_key_dir: str | Path | None = None,
) -> dict[str, object]:
    """Verify against an explicit pinned anchor or the IT-controlled environment."""

    if trusted_public_key_dir is not None and trusted_public_key_path is None:
        raise AcquisitionAuthenticationError(
            "explicit acquisition trust requires trusted_public_key_path"
        )
    explicit_trust = trusted_public_key_path is not None
    key_value = (
        trusted_public_key_path
        if explicit_trust
        else os.environ.get(TRUSTED_ACQUISITION_PUBLIC_KEY_ENV)
    )
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
        keyring_path=(
            trusted_public_key_dir
            if explicit_trust
            else os.environ.get(TRUSTED_ACQUISITION_PUBLIC_KEY_DIR_ENV)
        ),
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
    if unsigned.get("schema_version") in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
        _verify_governed_lt_snapshot_receipts(
            unsigned,
            acquisition_key_id=_public_key_id(public_key),
            require_provider_raw=(
                unsigned.get("schema_version") == PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA
            ),
        )
    return unsigned


def verify_trusted_time_receipt(
    receipt: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path | None = None,
    trusted_public_key_dir: str | Path | None = None,
    trusted_journal_id: str | None = None,
) -> dict[str, object]:
    explicit_values_present = any(
        value is not None
        for value in (
            trusted_public_key_path,
            trusted_public_key_dir,
            trusted_journal_id,
        )
    )
    if explicit_values_present and (
        trusted_public_key_path is None or trusted_journal_id is None
    ):
        raise AcquisitionAuthenticationError(
            "explicit trusted-time trust requires trusted_public_key_path and "
            "trusted_journal_id"
        )
    explicit_trust = trusted_public_key_path is not None
    key_value = (
        trusted_public_key_path
        if explicit_trust
        else os.environ.get(TRUSTED_TIME_PUBLIC_KEY_ENV)
    )
    if not key_value:
        raise AcquisitionAuthenticationError(
            f"{TRUSTED_TIME_PUBLIC_KEY_ENV} is required for historical EEX vintages"
        )
    expected_journal_id = str(
        trusted_journal_id
        if explicit_trust
        else os.environ.get(TRUSTED_TIME_JOURNAL_ID_ENV, "")
    ).strip()
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
        keyring_path=(
            trusted_public_key_dir
            if explicit_trust
            else os.environ.get(TRUSTED_TIME_PUBLIC_KEY_DIR_ENV)
        ),
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


def verify_source_acquisition_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate one prospective, source-neutral PIT receipt."""

    unsigned, public_key = _verify_time_attestation(
        receipt,
        expected_schema=SOURCE_ACQUISITION_RECEIPT_SCHEMA,
    )
    receipt_id = str(unsigned.get("receipt_id", ""))
    identity = dict(unsigned)
    identity.pop("receipt_id", None)
    if receipt_id != hashlib.sha256(_canonical_json_bytes(identity)).hexdigest():
        raise AcquisitionAuthenticationError("source receipt_id mismatch")
    sequence = unsigned.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise AcquisitionAuthenticationError("source receipt sequence is invalid")
    previous = unsigned.get("previous_receipt_id")
    if sequence == 1:
        if previous is not None:
            raise AcquisitionAuthenticationError("source receipt genesis has a parent")
    elif not _is_sha256(previous):
        raise AcquisitionAuthenticationError("source receipt parent is invalid")
    for field in ("acquisition_id", "source_role", "source_system", "source_locator"):
        value = unsigned.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AcquisitionAuthenticationError(f"source receipt {field} is missing")
    if unsigned.get("acquisition_method") != "PROSPECTIVE_DIRECT":
        raise AcquisitionAuthenticationError("source receipt is not a prospective direct acquisition")
    if not _is_sha256(unsigned.get("raw_sha256")):
        raise AcquisitionAuthenticationError("source receipt raw_sha256 is invalid")
    raw_size = unsigned.get("raw_size_bytes")
    if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 1:
        raise AcquisitionAuthenticationError("source receipt raw_size_bytes is invalid")
    _utc_datetime(unsigned.get("received_at_utc"), label="source receipt received_at_utc")
    verified = dict(unsigned)
    verified["trusted_time_key_id"] = _public_key_id(public_key)
    return verified


def source_acquisition_receipt_id(receipt: Mapping[str, object]) -> str:
    """Return the canonical id for one unsigned v2 source receipt."""

    unsigned = dict(receipt)
    unsigned.pop("trusted_time_attestation", None)
    unsigned.pop("receipt_id", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def verify_source_journal_checkpoint(
    checkpoint: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Authenticate one external append-only journal checkpoint."""

    key_value = os.environ.get(TRUSTED_JOURNAL_PUBLIC_KEY_ENV)
    if not key_value:
        raise AcquisitionAuthenticationError(
            f"{TRUSTED_JOURNAL_PUBLIC_KEY_ENV} is required for governed source journals"
        )
    expected_journal_id = os.environ.get(TRUSTED_TIME_JOURNAL_ID_ENV, "").strip()
    unsigned = dict(checkpoint)
    attestation = unsigned.pop("journal_attestation", None)
    if unsigned.get("schema_version") != SOURCE_JOURNAL_CHECKPOINT_SCHEMA:
        raise AcquisitionAuthenticationError("source journal checkpoint schema is unsupported")
    if str(unsigned.get("journal_id", "")) != expected_journal_id:
        raise AcquisitionAuthenticationError("source journal checkpoint journal is not governed")
    if not isinstance(attestation, Mapping):
        raise AcquisitionAuthenticationError("source journal checkpoint attestation is missing")
    if attestation.get("algorithm") != SIGNATURE_ALGORITHM:
        raise AcquisitionAuthenticationError("source journal checkpoint algorithm is unsupported")
    public_key = _select_trusted_public_key(
        attestation,
        primary_path=key_value,
        keyring_path=os.environ.get(TRUSTED_JOURNAL_PUBLIC_KEY_DIR_ENV),
        label="source-journal",
    )
    payload = _canonical_json_bytes(unsigned)
    if str(attestation.get("payload_sha256", "")) != hashlib.sha256(payload).hexdigest():
        raise AcquisitionAuthenticationError("source journal checkpoint payload hash mismatch")
    try:
        signature = base64.b64decode(
            str(attestation.get("value_base64", "")),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise AcquisitionAuthenticationError(
            "source journal checkpoint signature is not valid base64"
        ) from exc
    try:
        public_key.verify(signature, payload)
    except InvalidSignature as exc:
        raise AcquisitionAuthenticationError("source journal checkpoint signature is invalid") from exc
    identity = dict(unsigned)
    checkpoint_id = str(identity.pop("checkpoint_id", ""))
    if checkpoint_id != hashlib.sha256(_canonical_json_bytes(identity)).hexdigest():
        raise AcquisitionAuthenticationError("source journal checkpoint_id mismatch")
    return unsigned, _public_key_id(public_key)


def _verify_time_attestation(
    receipt: Mapping[str, object],
    *,
    expected_schema: str,
) -> tuple[dict[str, object], Ed25519PublicKey]:
    key_value = os.environ.get(TRUSTED_TIME_PUBLIC_KEY_ENV)
    if not key_value:
        raise AcquisitionAuthenticationError(
            f"{TRUSTED_TIME_PUBLIC_KEY_ENV} is required for governed source receipts"
        )
    expected_journal_id = os.environ.get(TRUSTED_TIME_JOURNAL_ID_ENV, "").strip()
    if not expected_journal_id:
        raise AcquisitionAuthenticationError(
            f"{TRUSTED_TIME_JOURNAL_ID_ENV} is required for governed source receipts"
        )
    unsigned = dict(receipt)
    attestation = unsigned.pop("trusted_time_attestation", None)
    if unsigned.get("schema_version") != expected_schema:
        raise AcquisitionAuthenticationError("source receipt schema is unsupported")
    if str(unsigned.get("journal_id", "")) != expected_journal_id:
        raise AcquisitionAuthenticationError("source receipt journal is not governed")
    if not isinstance(attestation, Mapping):
        raise AcquisitionAuthenticationError("source receipt time attestation is missing")
    if attestation.get("algorithm") != SIGNATURE_ALGORITHM:
        raise AcquisitionAuthenticationError("source receipt time algorithm is unsupported")
    public_key = _select_trusted_public_key(
        attestation,
        primary_path=key_value,
        keyring_path=os.environ.get(TRUSTED_TIME_PUBLIC_KEY_DIR_ENV),
        label="trusted-time",
    )
    payload = _canonical_json_bytes(unsigned)
    if str(attestation.get("payload_sha256", "")) != hashlib.sha256(payload).hexdigest():
        raise AcquisitionAuthenticationError("source receipt time payload hash mismatch")
    try:
        signature = base64.b64decode(
            str(attestation.get("value_base64", "")),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise AcquisitionAuthenticationError(
            "source receipt time signature is not valid base64"
        ) from exc
    try:
        public_key.verify(signature, payload)
    except InvalidSignature as exc:
        raise AcquisitionAuthenticationError("source receipt time signature is invalid") from exc
    return unsigned, public_key


def _verify_governed_lt_snapshot_receipts(
    contract: Mapping[str, object],
    *,
    acquisition_key_id: str,
    require_provider_raw: bool,
) -> None:
    if contract.get("calibration_eligible") is not True:
        raise AcquisitionAuthenticationError("governed LT snapshot must be calibration eligible")
    if contract.get("source_class") != "GOVERNED_ACQUISITION":
        raise AcquisitionAuthenticationError("governed LT snapshot source class is invalid")
    acquisition_id = str(contract.get("acquisition_id", "")).strip()
    if not acquisition_id:
        raise AcquisitionAuthenticationError("governed LT snapshot acquisition_id is missing")
    snapshot_available = _utc_datetime(
        contract.get("available_at_utc"),
        label="governed LT snapshot available_at_utc",
    )
    files = contract.get("files")
    if not isinstance(files, Mapping) or not files:
        raise AcquisitionAuthenticationError("governed LT snapshot files are missing")
    role_available_times: list[datetime] = []
    receipt_ids: set[str] = set()
    verified_receipts: list[dict[str, object]] = []
    for role, raw_entry in files.items():
        if not isinstance(raw_entry, Mapping):
            raise AcquisitionAuthenticationError(f"governed LT role is invalid: {role}")
        entry = dict(raw_entry)
        if entry.get("calibration_eligible") is not True:
            raise AcquisitionAuthenticationError(f"governed LT role is ineligible: {role}")
        if str(entry.get("acquisition_id", "")) != acquisition_id:
            raise AcquisitionAuthenticationError(f"governed LT role acquisition diverges: {role}")
        receipt = entry.get("source_receipt")
        if not isinstance(receipt, Mapping):
            raise AcquisitionAuthenticationError(f"governed LT role receipt is missing: {role}")
        verified = verify_source_acquisition_receipt(receipt)
        if verified["trusted_time_key_id"] == acquisition_key_id:
            raise AcquisitionAuthenticationError(
                "acquisition and trusted-time authorities must use distinct keys"
            )
        receipt_id = str(verified["receipt_id"])
        if receipt_id in receipt_ids:
            raise AcquisitionAuthenticationError("governed LT snapshot repeats a source receipt")
        receipt_ids.add(receipt_id)
        verified_receipts.append(verified)
        if str(verified["source_role"]) != str(role):
            raise AcquisitionAuthenticationError(f"governed LT source role mismatch: {role}")
        if str(verified["source_system"]).upper() != str(
            entry.get("source_system", "")
        ).upper():
            raise AcquisitionAuthenticationError(f"governed LT source system mismatch: {role}")
        if str(verified["acquisition_id"]) != acquisition_id:
            raise AcquisitionAuthenticationError(f"governed LT receipt acquisition mismatch: {role}")
        raw_artifact = entry.get("raw_artifact")
        provider_derived_at: datetime | None = None
        if not isinstance(raw_artifact, Mapping):
            raise AcquisitionAuthenticationError(f"governed LT raw artifact is missing: {role}")
        _portable_relative_path(raw_artifact.get("path"), label=f"raw artifact {role}")
        if require_provider_raw and str(role) in REPLAY_GOVERNED_LT_INPUT_ROLES:
            provider_raw = entry.get("provider_raw_artifact")
            provider_derivation = entry.get("provider_derivation")
            if not isinstance(provider_raw, Mapping):
                raise AcquisitionAuthenticationError(
                    f"governed LT provider raw artifact is missing: {role}"
                )
            if not isinstance(provider_derivation, Mapping):
                raise AcquisitionAuthenticationError(
                    f"governed LT provider derivation is missing: {role}"
                )
            if str(provider_raw.get("sha256", "")) != str(verified["raw_sha256"]):
                raise AcquisitionAuthenticationError(
                    f"governed LT provider raw hash mismatch: {role}"
                )
            if provider_raw.get("size_bytes") != verified["raw_size_bytes"]:
                raise AcquisitionAuthenticationError(
                    f"governed LT provider raw size mismatch: {role}"
                )
            _portable_relative_path(
                provider_raw.get("path"), label=f"provider raw artifact {role}"
            )
            for field in ("parser_code_sha256", "parser_config_sha256"):
                if not _is_sha256(provider_derivation.get(field)):
                    raise AcquisitionAuthenticationError(
                        f"governed LT provider derivation {field} is invalid: {role}"
                    )
            _portable_relative_path(
                provider_derivation.get("parser_code_path"),
                label=f"provider parser code {role}",
            )
            _portable_relative_path(
                provider_derivation.get("parser_config_path"),
                label=f"provider parser config {role}",
            )
            provider_derived_at = _utc_datetime(
                provider_derivation.get("derived_at_utc"),
                label=f"governed LT provider derivation time {role}",
            )
        else:
            if str(raw_artifact.get("sha256", "")) != str(verified["raw_sha256"]):
                raise AcquisitionAuthenticationError(f"governed LT raw hash mismatch: {role}")
            if raw_artifact.get("size_bytes") != verified["raw_size_bytes"]:
                raise AcquisitionAuthenticationError(f"governed LT raw size mismatch: {role}")
        derivation = entry.get("derivation")
        if not isinstance(derivation, Mapping):
            raise AcquisitionAuthenticationError(f"governed LT derivation is missing: {role}")
        for field in ("parser_code_sha256", "parser_config_sha256"):
            if not _is_sha256(derivation.get(field)):
                raise AcquisitionAuthenticationError(
                    f"governed LT derivation {field} is invalid: {role}"
                )
        _portable_relative_path(
            derivation.get("parser_code_path"),
            label=f"parser code {role}",
        )
        _portable_relative_path(
            derivation.get("parser_config_path"),
            label=f"parser config {role}",
        )
        derived_at = _utc_datetime(
            derivation.get("derived_at_utc"),
            label=f"governed LT derivation time {role}",
        )
        received_at = _utc_datetime(
            verified.get("received_at_utc"),
            label=f"governed LT receipt time {role}",
        )
        role_available = _utc_datetime(
            entry.get("available_at_utc"),
            label=f"governed LT role available_at_utc {role}",
        )
        if (
            derived_at < received_at
            or role_available < derived_at
            or (
                provider_derived_at is not None
                and not (received_at <= provider_derived_at <= derived_at)
            )
        ):
            raise AcquisitionAuthenticationError(f"governed LT role time order is invalid: {role}")
        if role_available > snapshot_available:
            raise AcquisitionAuthenticationError(f"governed LT role is newer than snapshot: {role}")
        role_available_times.append(role_available)
        quality = entry.get("quality_evidence")
        if not isinstance(quality, Mapping) or quality.get("status") != "PASS":
            raise AcquisitionAuthenticationError(f"governed LT quality evidence failed: {role}")
        if not _is_sha256(quality.get("policy_sha256")) or not _is_sha256(
            quality.get("report_sha256")
        ):
            raise AcquisitionAuthenticationError(f"governed LT quality binding is invalid: {role}")
        _portable_relative_path(quality.get("report_path"), label=f"quality report {role}")
        if str(role) in REPLAY_GOVERNED_LT_INPUT_ROLES:
            artifact_paths = {
                "derived": _portable_relative_path(entry.get("path"), label=f"derived artifact {role}"),
                "raw": _portable_relative_path(
                    raw_artifact.get("path"), label=f"raw artifact {role}"
                ),
                "parser_code": _portable_relative_path(
                    derivation.get("parser_code_path"), label=f"parser code {role}"
                ),
                "parser_config": _portable_relative_path(
                    derivation.get("parser_config_path"), label=f"parser config {role}"
                ),
                "quality_report": _portable_relative_path(
                    quality.get("report_path"), label=f"quality report {role}"
                ),
            }
            if require_provider_raw:
                provider_raw = entry["provider_raw_artifact"]
                provider_derivation = entry["provider_derivation"]
                artifact_paths.update(
                    {
                        "provider_raw": _portable_relative_path(
                            provider_raw.get("path"),
                            label=f"provider raw artifact {role}",
                        ),
                        "provider_parser_code": _portable_relative_path(
                            provider_derivation.get("parser_code_path"),
                            label=f"provider parser code {role}",
                        ),
                        "provider_parser_config": _portable_relative_path(
                            provider_derivation.get("parser_config_path"),
                            label=f"provider parser config {role}",
                        ),
                    }
                )
            if len(set(artifact_paths.values())) != len(artifact_paths):
                raise AcquisitionAuthenticationError(
                    f"governed LT core role aliases raw, derived or evidence artifacts: {role}"
                )
    checkpoint = contract.get("source_journal_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise AcquisitionAuthenticationError("governed LT source journal checkpoint is missing")
    verified_checkpoint, journal_key_id = verify_source_journal_checkpoint(checkpoint)
    time_key_ids = {str(receipt["trusted_time_key_id"]) for receipt in verified_receipts}
    if len(time_key_ids) != 1:
        raise AcquisitionAuthenticationError("governed LT receipts mix trusted-time authorities")
    if journal_key_id in {acquisition_key_id, *time_key_ids}:
        raise AcquisitionAuthenticationError(
            "acquisition, trusted-time and journal authorities must use distinct keys"
        )
    expected_bundle_root = governed_snapshot_bundle_root_sha256(files)
    if str(verified_checkpoint.get("bundle_root_sha256", "")) != expected_bundle_root:
        raise AcquisitionAuthenticationError("source journal checkpoint bundle root mismatch")
    checkpoint_issued_at = _verify_snapshot_receipt_checkpoint(
        verified_receipts,
        verified_checkpoint,
        acquisition_id=acquisition_id,
    )
    if max(*role_available_times, checkpoint_issued_at) != snapshot_available:
        raise AcquisitionAuthenticationError(
            "governed LT snapshot availability is not the maximum governed availability"
        )


def _verify_snapshot_receipt_checkpoint(
    receipts: list[dict[str, object]],
    checkpoint: Mapping[str, object],
    *,
    acquisition_id: str,
) -> datetime:
    if str(checkpoint.get("acquisition_id", "")) != acquisition_id:
        raise AcquisitionAuthenticationError("source journal checkpoint acquisition mismatch")
    ordered = sorted(receipts, key=lambda item: int(item["sequence"]))
    sequences = [int(item["sequence"]) for item in ordered]
    first = checkpoint.get("first_sequence")
    last = checkpoint.get("last_sequence")
    if (
        not isinstance(first, int)
        or isinstance(first, bool)
        or not isinstance(last, int)
        or isinstance(last, bool)
        or sequences != list(range(first, last + 1))
    ):
        raise AcquisitionAuthenticationError("source receipt checkpoint sequence is incomplete")
    ordered_ids = [str(item["receipt_id"]) for item in ordered]
    if checkpoint.get("receipt_ids") != ordered_ids:
        raise AcquisitionAuthenticationError("source receipt checkpoint inventory mismatch")
    if str(checkpoint.get("head_receipt_id", "")) != ordered_ids[-1]:
        raise AcquisitionAuthenticationError("source receipt checkpoint head mismatch")
    expected_parent = checkpoint.get("previous_receipt_id")
    if ordered[0].get("previous_receipt_id") != expected_parent:
        raise AcquisitionAuthenticationError("source receipt checkpoint initial parent mismatch")
    for parent, child in zip(ordered[:-1], ordered[1:], strict=True):
        if str(child.get("previous_receipt_id", "")) != str(parent["receipt_id"]):
            raise AcquisitionAuthenticationError("source receipt chain is not linked")
        parent_time = _utc_datetime(parent["received_at_utc"], label="source receipt time")
        child_time = _utc_datetime(child["received_at_utc"], label="source receipt time")
        if child_time < parent_time:
            raise AcquisitionAuthenticationError("source receipt times are not monotonic")
    issued_at = _utc_datetime(
        checkpoint.get("issued_at_utc"),
        label="source journal checkpoint issued_at_utc",
    )
    latest_received = max(
        _utc_datetime(item["received_at_utc"], label="source receipt time")
        for item in ordered
    )
    if issued_at < latest_received:
        raise AcquisitionAuthenticationError("source journal checkpoint predates its receipts")
    return issued_at


def governed_snapshot_bundle_root_sha256(files: Mapping[str, object]) -> str:
    """Return the journal-signable root of every declared LT role artifact."""

    if not isinstance(files, Mapping) or not files:
        raise AcquisitionAuthenticationError("governed LT bundle root files are missing")
    normalized: dict[str, object] = {}
    for role in sorted(str(value) for value in files):
        entry = files.get(role)
        if not isinstance(entry, Mapping):
            raise AcquisitionAuthenticationError(f"governed LT bundle role is invalid: {role}")
        normalized[role] = dict(entry)
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def _portable_relative_path(value: object, *, label: str) -> str:
    raw = str(value or "").strip()
    path = Path(raw)
    if not raw or path.is_absolute() or path.drive or ".." in path.parts:
        raise AcquisitionAuthenticationError(f"{label} path is not portable and relative")
    return path.as_posix()


def _utc_datetime(value: object, *, label: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise AcquisitionAuthenticationError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcquisitionAuthenticationError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AcquisitionAuthenticationError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _is_sha256(value: object) -> bool:
    return bool(_SHA256.fullmatch(str(value)))


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
    keyring_path: str | Path | None,
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
    if keyring_path is not None:
        keyring = _absolute_trusted_key_path(keyring_path)
        if not keyring.is_dir():
            raise AcquisitionAuthenticationError(
                f"{label} trusted keyring is not a directory"
            )
        historical = _load_public_key(
            _absolute_trusted_key_path(keyring / f"{selected_key_id}.pem")
        )
        if _public_key_id(historical) != selected_key_id:
            raise AcquisitionAuthenticationError(
                f"{label} historical trusted key identity mismatch"
            )
        return historical
    raise AcquisitionAuthenticationError(
        f"{label} signing key is not the active trusted key; historical replay is unsupported"
    )


def _read_stable_regular_file(path: str | Path, *, label: str) -> bytes:
    try:
        frozen = process_immutable_file_bytes(path)
    except ValueError as exc:
        raise AcquisitionAuthenticationError(f"cannot load {label}") from exc
    if frozen is not None:
        if len(frozen) > 64 * 1024:
            raise AcquisitionAuthenticationError(f"{label} is unexpectedly large")
        return frozen
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
    ).encode("utf-8")
