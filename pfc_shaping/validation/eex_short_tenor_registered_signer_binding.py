"""Bind D228 local signers to D229 registry keys without claiming trusted time.

The combined replay is intentionally value-free.  It reuses the exact D228
and D229 path validators, requires the same caller-anchored public-key files on
both sides, and resolves the three actual D228 signer roles at the event times
asserted by the signed time-observation payloads.  Those payload assertions are
not signature-generation timestamps, so external time and every model authority
remain false.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pfc_shaping.validation.eex_short_tenor_signed_replay import (
    SIGNED_REPLAY_CONTRACT_CONTENT_ID,
    SIGNED_REPLAY_CONTRACT_SCHEMA,
    ShortTenorSignedReplayError,
    verify_signed_short_tenor_receipt_pair_paths,
)
from pfc_shaping.validation.eex_short_tenor_trust_registry import (
    REQUIRED_ROLES,
    TRUST_REGISTRY_CONTRACT_CONTENT_ID,
    TRUST_REGISTRY_CONTRACT_SCHEMA,
    ShortTenorTrustRegistryError,
    verify_trust_registry_chain_paths,
)

REGISTERED_SIGNER_BINDING_CONTRACT_SCHEMA = (
    "fmv-eex-ch-short-tenor-registered-signer-binding-contract-v1"
)
REGISTERED_SIGNER_BINDING_CONTRACT_STATUS = (
    "PASS_LOCAL_REGISTERED_SIGNER_AND_ASSERTED_EVENT_TIME_BINDING_ONLY_"
    "NO_EXTERNAL_TIME"
)
REGISTERED_SIGNER_BINDING_CONTRACT_CONTENT_ID = (
    "2fa6d60542f513faf132bc33dbf6ea17c98693b7658c37ad9e92195dcdd053f0"
)
REGISTERED_SIGNER_BINDING_CONTRACT_FILE_SHA256 = (
    "849cb6cea28913b87b69cea463d54ea9f110150467343cbd8f27f602a07fca23"
)
SIGNED_REPLAY_CONTRACT_FILE_SHA256 = (
    "f9f447f31458a88aaaf4ce5863f9f23dbd6a77d1683b7759885783457780a957"
)
TRUST_REGISTRY_CONTRACT_FILE_SHA256 = (
    "e7fbf4f099bcc355073d0f5b591d988100df4168c22b172ce8fc08e4c4a0ed9c"
)
MAXIMUM_REGISTRY_PUBLICATION_LAG_DAYS = 31

_SIGNED_REPLAY_INPUT_FIELDS = {
    "training_envelope_path",
    "selection_envelope_path",
    "training_execution_public_key_path",
    "selection_governance_public_key_path",
    "trusted_time_public_key_path",
    "training_time_envelope_path",
    "selection_time_envelope_path",
    "expected_training_envelope_sha256",
    "expected_selection_envelope_sha256",
    "expected_training_execution_public_key_sha256",
    "expected_selection_governance_public_key_sha256",
    "expected_trusted_time_public_key_sha256",
    "expected_training_time_envelope_sha256",
    "expected_selection_time_envelope_sha256",
    "expected_trusted_time_source_id",
}
_TRUST_REGISTRY_INPUT_FIELDS = {
    "registry_envelope_paths",
    "expected_registry_envelope_sha256s",
    "governance_public_key_paths",
    "expected_governance_public_key_sha256s",
    "registered_public_key_paths",
    "expected_registered_public_key_sha256s",
    "expected_registry_id",
    "expected_head_sequence",
    "expected_head_envelope_sha256",
    "caller_reference_time_utc",
}
_SHARED_ROLE_PATH_FIELDS = {
    "training_execution": "training_execution_public_key_path",
    "selection_governance": "selection_governance_public_key_path",
    "trusted_time": "trusted_time_public_key_path",
}
_SHARED_ROLE_HASH_FIELDS = {
    "training_execution": "expected_training_execution_public_key_sha256",
    "selection_governance": "expected_selection_governance_public_key_sha256",
    "trusted_time": "expected_trusted_time_public_key_sha256",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ShortTenorRegisteredSignerBindingError(ValueError):
    """Raised when D228 signers do not bind exactly to the D229 registry."""


def validate_registered_signer_binding_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact non-authoritative D230 binding contract."""

    contract = _mapping(document, "registered signer binding contract")
    _equal(
        contract.get("schema_version"),
        REGISTERED_SIGNER_BINDING_CONTRACT_SCHEMA,
        "registered signer binding contract schema",
    )
    _equal(
        contract.get("status"),
        REGISTERED_SIGNER_BINDING_CONTRACT_STATUS,
        "registered signer binding contract status",
    )
    selected = _mapping(contract.get("selected_boundaries"), "selected boundaries")
    _equal(
        selected.get("signed_replay_contract_schema"),
        SIGNED_REPLAY_CONTRACT_SCHEMA,
        "signed replay contract schema binding",
    )
    _equal(
        selected.get("signed_replay_contract_content_id"),
        SIGNED_REPLAY_CONTRACT_CONTENT_ID,
        "signed replay contract content binding",
    )
    _equal(
        selected.get("signed_replay_contract_sha256"),
        SIGNED_REPLAY_CONTRACT_FILE_SHA256,
        "signed replay contract byte binding",
    )
    _equal(
        selected.get("trust_registry_contract_schema"),
        TRUST_REGISTRY_CONTRACT_SCHEMA,
        "trust registry contract schema binding",
    )
    _equal(
        selected.get("trust_registry_contract_content_id"),
        TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "trust registry contract content binding",
    )
    _equal(
        selected.get("trust_registry_contract_sha256"),
        TRUST_REGISTRY_CONTRACT_FILE_SHA256,
        "trust registry contract byte binding",
    )
    profile = _mapping(contract.get("binding_profile"), "binding profile")
    _equal(
        profile.get("maximum_registry_publication_lag_days"),
        MAXIMUM_REGISTRY_PUBLICATION_LAG_DAYS,
        "maximum registry publication lag",
    )
    limitations = _mapping(
        contract.get("temporal_limitations"),
        "temporal limitations",
    )
    _equal(
        limitations.get("actual_signature_generation_time_lifecycle_acceptance_proven"),
        False,
        "actual signature-time authority",
    )
    authorities = _mapping(contract.get("authorities"), "authorities")
    for field in (
        "actual_signature_generation_time_verified",
        "external_owner_identity_verified",
        "external_registry_authority_verified",
        "independently_trusted_time_verified",
        "point_in_time_availability_proven",
        "training_authorized",
        "selection_authorized",
        "model_input_authorized",
        "candidate_assembly_authorized",
        "promotion_authorized",
        "production_authorized",
    ):
        _equal(authorities.get(field), False, f"authority {field}")
    content_id = _sha256_json(contract)
    _equal(
        content_id,
        REGISTERED_SIGNER_BINDING_CONTRACT_CONTENT_ID,
        "registered signer binding contract content id",
    )
    return {
        "schema_version": REGISTERED_SIGNER_BINDING_CONTRACT_SCHEMA,
        "status": REGISTERED_SIGNER_BINDING_CONTRACT_STATUS,
        "contract_content_id": content_id,
        "signed_replay_contract_content_id": SIGNED_REPLAY_CONTRACT_CONTENT_ID,
        "trust_registry_contract_content_id": TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "maximum_registry_publication_lag_days": (
            MAXIMUM_REGISTRY_PUBLICATION_LAG_DAYS
        ),
        "actual_signature_generation_time_verified": False,
        "independently_trusted_time_verified": False,
        "production_authorized": False,
    }


def verify_registered_signer_binding_paths(
    *,
    signed_replay_inputs: Mapping[str, object],
    trust_registry_inputs: Mapping[str, object],
) -> dict[str, object]:
    """Atomically compose exact D228 and D229 checks around shared key paths."""

    replay_inputs = _exact_input_mapping(
        signed_replay_inputs,
        _SIGNED_REPLAY_INPUT_FIELDS,
        "signed replay inputs",
    )
    registry_inputs = _exact_input_mapping(
        trust_registry_inputs,
        _TRUST_REGISTRY_INPUT_FIELDS,
        "trust registry inputs",
    )
    registry_inputs = _freeze_registry_inputs(registry_inputs)
    try:
        replay = verify_signed_short_tenor_receipt_pair_paths(**replay_inputs)
    except (ShortTenorSignedReplayError, TypeError) as exc:
        raise ShortTenorRegisteredSignerBindingError(
            "D228 signed replay failed during registered signer binding"
        ) from exc
    _assert_signed_replay_assessment(replay)

    receipt_bindings = _mapping(replay.get("receipt_bindings"), "receipt bindings")
    training_receipt = _mapping(
        receipt_bindings.get("training"),
        "training receipt binding",
    )
    selection_receipt = _mapping(
        receipt_bindings.get("selection"),
        "selection receipt binding",
    )
    time_bindings = _mapping(
        replay.get("time_observation_bindings"),
        "time observation bindings",
    )
    training_time = _mapping(
        time_bindings.get("training"),
        "training time binding",
    )
    selection_time = _mapping(
        time_bindings.get("selection"),
        "selection time binding",
    )
    training_event_text = _utc_text(
        training_time.get("observed_at_utc"),
        "training asserted event time",
    )
    selection_event_text = _utc_text(
        selection_time.get("observed_at_utc"),
        "selection asserted event time",
    )
    reference_text = _utc_text(
        registry_inputs.get("caller_reference_time_utc"),
        "caller reference time",
    )

    primary_event_times = {role: reference_text for role in REQUIRED_ROLES}
    primary_event_times.update(
        {
            "training_execution": training_event_text,
            "selection_governance": selection_event_text,
            "trusted_time": selection_event_text,
        }
    )
    training_time_event_times = dict(primary_event_times)
    training_time_event_times["trusted_time"] = training_event_text
    primary_registry = _verify_registry(
        registry_inputs,
        event_times=primary_event_times,
        label="primary asserted-event registry replay",
    )
    training_time_registry = _verify_registry(
        registry_inputs,
        event_times=training_time_event_times,
        label="training-time registry replay",
    )
    _require_same_registry_snapshot(primary_registry, training_time_registry)

    resolved = _mapping(
        primary_registry.get("locally_resolved_key_ids_by_role"),
        "primary resolved registry keys",
    )
    training_time_resolved = _mapping(
        training_time_registry.get("locally_resolved_key_ids_by_role"),
        "training-time resolved registry keys",
    )
    signing_key_ids = {
        "training_execution": _sha(
            training_receipt.get("signing_key_id"),
            "training receipt signing key",
        ),
        "selection_governance": _sha(
            selection_receipt.get("signing_key_id"),
            "selection receipt signing key",
        ),
        "trusted_time": _sha(
            training_time.get("signing_key_id"),
            "training time signing key",
        ),
    }
    _equal(
        selection_time.get("signing_key_id"),
        signing_key_ids["trusted_time"],
        "cross-observation trusted-time signing key",
    )
    for role in ("training_execution", "selection_governance", "trusted_time"):
        _equal(
            resolved.get(role),
            signing_key_ids[role],
            f"{role} registered key resolution",
        )
    _equal(
        training_time_resolved.get("trusted_time"),
        signing_key_ids["trusted_time"],
        "training-time registered key resolution",
    )

    _verify_shared_role_paths_and_hashes(
        replay_inputs=replay_inputs,
        registry_inputs=registry_inputs,
        signing_key_ids=signing_key_ids,
    )
    head_issued_text = _utc_text(
        primary_registry.get("head_issued_at_utc"),
        "registry head issued at",
    )
    head_issued = _utc(head_issued_text, "registry head issued at")
    event_times = {
        "training": _utc(training_event_text, "training asserted event time"),
        "selection": _utc(selection_event_text, "selection asserted event time"),
    }
    publication_lag_seconds: dict[str, int] = {}
    for label, event_time in event_times.items():
        lag = head_issued - event_time
        if lag < timedelta(0):
            raise ShortTenorRegisteredSignerBindingError(
                f"registry head predates {label} asserted signer event time"
            )
        if lag > timedelta(days=MAXIMUM_REGISTRY_PUBLICATION_LAG_DAYS):
            raise ShortTenorRegisteredSignerBindingError(
                f"registry publication lag exceeds maximum for {label}"
            )
        publication_lag_seconds[label] = int(lag.total_seconds())

    bindings = {
        "training_receipt": {
            "registry_role": "training_execution",
            "signing_key_id": signing_key_ids["training_execution"],
            "target_envelope_sha256": _sha(
                replay_inputs["expected_training_envelope_sha256"],
                "training target envelope",
            ),
            "asserted_event_time_utc": training_event_text,
            "asserted_event_time_basis": (
                "training_time_observation.observed_at_utc"
            ),
        },
        "selection_receipt": {
            "registry_role": "selection_governance",
            "signing_key_id": signing_key_ids["selection_governance"],
            "target_envelope_sha256": _sha(
                replay_inputs["expected_selection_envelope_sha256"],
                "selection target envelope",
            ),
            "asserted_event_time_utc": selection_event_text,
            "asserted_event_time_basis": (
                "selection_time_observation.observed_at_utc"
            ),
        },
        "training_time_observation": {
            "registry_role": "trusted_time",
            "signing_key_id": signing_key_ids["trusted_time"],
            "target_envelope_sha256": _sha(
                replay_inputs["expected_training_time_envelope_sha256"],
                "training time target envelope",
            ),
            "asserted_event_time_utc": training_event_text,
            "asserted_event_time_basis": "signed_payload.observed_at_utc",
        },
        "selection_time_observation": {
            "registry_role": "trusted_time",
            "signing_key_id": signing_key_ids["trusted_time"],
            "target_envelope_sha256": _sha(
                replay_inputs["expected_selection_time_envelope_sha256"],
                "selection time target envelope",
            ),
            "asserted_event_time_utc": selection_event_text,
            "asserted_event_time_basis": "signed_payload.observed_at_utc",
        },
    }
    return {
        "schema_version": (
            "fmv_eex_ch_short_tenor_registered_signer_binding_assessment.v1"
        ),
        "status": REGISTERED_SIGNER_BINDING_CONTRACT_STATUS,
        "registered_signer_binding_contract_content_id": (
            REGISTERED_SIGNER_BINDING_CONTRACT_CONTENT_ID
        ),
        "signed_replay_contract_content_id": SIGNED_REPLAY_CONTRACT_CONTENT_ID,
        "trust_registry_contract_content_id": TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "outer_origin_id": replay["outer_origin_id"],
        "outer_origin_as_of_utc": replay["outer_origin_as_of_utc"],
        "training_receipt_id": replay["training_receipt_id"],
        "selection_receipt_id": replay["selection_receipt_id"],
        "registry_id": primary_registry["registry_id"],
        "registry_head_sequence": primary_registry["head_sequence"],
        "registry_head_envelope_sha256": primary_registry["head_envelope_sha256"],
        "registry_head_payload_sha256": primary_registry["head_payload_sha256"],
        "registry_head_issued_at_utc": head_issued_text,
        "registry_head_expires_at_utc": primary_registry["head_expires_at_utc"],
        "caller_reference_time_utc": reference_text,
        "registered_signer_key_ids_by_role": signing_key_ids,
        "asserted_event_time_bindings": bindings,
        "registry_publication_lag_seconds_by_receipt": publication_lag_seconds,
        "registry_event_time_replay_count": 2,
        "local_d228_signature_integrity_verified": True,
        "local_d229_registry_integrity_verified": True,
        "local_registered_signer_identity_binding_verified": True,
        "local_asserted_event_time_resolution_verified": True,
        "post_event_registry_head_verified": True,
        "actual_signature_generation_time_verified": False,
        "backdating_resistance_verified": False,
        "source_signer_identity_or_event_time_bound": False,
        "external_owner_identity_verified": False,
        "external_registry_authority_verified": False,
        "independently_trusted_time_verified": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "receipt_materials_opened": False,
        "price_values_opened": False,
        "databricks_request_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "h_drive_access_count": 0,
        "remote_write_count": 0,
    }


def _verify_registry(
    inputs: Mapping[str, object],
    *,
    event_times: Mapping[str, str],
    label: str,
) -> dict[str, object]:
    try:
        assessment = verify_trust_registry_chain_paths(
            **dict(inputs),
            caller_event_time_utc_by_role=event_times,
        )
    except (ShortTenorTrustRegistryError, TypeError) as exc:
        raise ShortTenorRegisteredSignerBindingError(f"{label} failed") from exc
    _assert_registry_assessment(assessment, label)
    return assessment


def _assert_registry_assessment(
    assessment: Mapping[str, object],
    label: str,
) -> None:
    _equal(
        assessment.get("trust_registry_contract_content_id"),
        TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        f"{label} contract content id",
    )
    for field in (
        "local_threshold_signature_integrity_verified",
        "local_rollback_freeze_and_lifecycle_consistency_verified",
    ):
        _equal(assessment.get(field), True, f"{label} {field}")
    for field in (
        "external_owner_identity_verified",
        "external_registry_authority_verified",
        "governance_root_bootstrap_verified",
        "governance_root_rotation_verified",
        "independently_trusted_time_verified",
        "point_in_time_availability_proven",
        "training_authorized",
        "selection_authorized",
        "model_input_authorized",
        "candidate_assembly_authorized",
        "promotion_authorized",
        "production_authorized",
        "price_values_opened",
    ):
        _equal(assessment.get(field), False, f"{label} {field}")
    for field in (
        "databricks_request_count",
        "warehouse_start_count",
        "network_call_count",
        "h_drive_access_count",
        "remote_write_count",
    ):
        _equal(assessment.get(field), 0, f"{label} {field}")


def _assert_signed_replay_assessment(assessment: Mapping[str, object]) -> None:
    _equal(
        assessment.get("signed_replay_contract_content_id"),
        SIGNED_REPLAY_CONTRACT_CONTENT_ID,
        "signed replay contract content id",
    )
    for field in (
        "local_dsse_signature_integrity_verified",
        "in_toto_slsa_bindings_verified",
        "d227_structural_replay_verified",
        "training_selection_link_verified",
        "execution_time_observation_links_verified",
    ):
        _equal(assessment.get(field), True, f"signed replay {field}")
    for field in (
        "external_key_owner_identity_verified",
        "key_lifecycle_verified",
        "independently_trusted_time_verified",
        "point_in_time_availability_proven",
        "training_authorized",
        "selection_authorized",
        "model_input_authorized",
        "candidate_assembly_authorized",
        "promotion_authorized",
        "production_authorized",
        "receipt_materials_opened",
        "receipt_subject_artifacts_opened",
        "source_trusted_time_receipts_opened",
        "price_values_opened",
    ):
        _equal(assessment.get(field), False, f"signed replay {field}")
    for field in (
        "databricks_request_count",
        "warehouse_start_count",
        "network_call_count",
        "remote_write_count",
    ):
        _equal(assessment.get(field), 0, f"signed replay {field}")


def _require_same_registry_snapshot(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> None:
    for field in (
        "registry_id",
        "registry_sequence_count",
        "head_sequence",
        "head_envelope_sha256",
        "head_payload_sha256",
        "head_issued_at_utc",
        "head_expires_at_utc",
        "governance_policy_id",
        "governance_public_key_sha256",
        "registered_public_key_sha256",
        "chain_payload_sha256s",
        "chain_envelope_sha256s",
    ):
        _equal(left.get(field), right.get(field), f"cross-replay registry {field}")


def _verify_shared_role_paths_and_hashes(
    *,
    replay_inputs: Mapping[str, object],
    registry_inputs: Mapping[str, object],
    signing_key_ids: Mapping[str, str],
) -> None:
    registry_paths = _mapping(
        registry_inputs.get("registered_public_key_paths"),
        "registered public key paths",
    )
    registry_hashes = _mapping(
        registry_inputs.get("expected_registered_public_key_sha256s"),
        "registered public key hashes",
    )
    for role in ("training_execution", "selection_governance", "trusted_time"):
        key_id = signing_key_ids[role]
        if key_id not in registry_paths or key_id not in registry_hashes:
            raise ShortTenorRegisteredSignerBindingError(
                f"{role} signer key is absent from registered path inventory"
            )
        replay_path = _absolute_identity(
            replay_inputs[_SHARED_ROLE_PATH_FIELDS[role]],
            f"{role} signed replay key path",
        )
        registry_path = _absolute_identity(
            registry_paths[key_id],
            f"{role} registry key path",
        )
        _equal(replay_path, registry_path, f"{role} shared public key path")
        replay_hash = _sha(
            replay_inputs[_SHARED_ROLE_HASH_FIELDS[role]],
            f"{role} signed replay public key hash",
        )
        registry_hash = _sha(
            registry_hashes[key_id],
            f"{role} registered public key hash",
        )
        _equal(replay_hash, registry_hash, f"{role} shared public key hash")


def _exact_input_mapping(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> dict[str, object]:
    selected = _mapping(value, label)
    actual = set(selected)
    if actual != expected:
        raise ShortTenorRegisteredSignerBindingError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return dict(selected)


def _freeze_registry_inputs(inputs: Mapping[str, object]) -> dict[str, object]:
    frozen = dict(inputs)
    for field in (
        "registry_envelope_paths",
        "expected_registry_envelope_sha256s",
        "governance_public_key_paths",
        "expected_governance_public_key_sha256s",
    ):
        value = frozen[field]
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ShortTenorRegisteredSignerBindingError(
                f"trust registry {field} must be a sequence"
            )
        frozen[field] = tuple(value)
    for field in (
        "registered_public_key_paths",
        "expected_registered_public_key_sha256s",
    ):
        frozen[field] = dict(_mapping(frozen[field], f"trust registry {field}"))
    return frozen


def _absolute_identity(value: object, label: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ShortTenorRegisteredSignerBindingError(f"{label} must be path-like")
    path = Path(value)
    if not path.is_absolute():
        raise ShortTenorRegisteredSignerBindingError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ShortTenorRegisteredSignerBindingError(f"{label} cannot resolve") from exc
    return os.path.normcase(str(resolved))


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ShortTenorRegisteredSignerBindingError(f"{label} must be a mapping")
    return value


def _sha(value: object, label: str) -> str:
    selected = str(value)
    if not _SHA256_RE.fullmatch(selected):
        raise ShortTenorRegisteredSignerBindingError(
            f"{label} must be lowercase SHA-256"
        )
    return selected


def _utc(value: object, label: str) -> datetime:
    text = _utc_text(value, label)
    return datetime.fromisoformat(text[:-1] + "+00:00")


def _utc_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ShortTenorRegisteredSignerBindingError(
            f"{label} must use canonical UTC text"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ShortTenorRegisteredSignerBindingError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise ShortTenorRegisteredSignerBindingError(
            f"{label} must use whole-second UTC"
        )
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ShortTenorRegisteredSignerBindingError(
            f"{label} must use canonical UTC text"
        )
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ShortTenorRegisteredSignerBindingError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
