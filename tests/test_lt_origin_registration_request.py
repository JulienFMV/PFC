from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.origin_registration_envelope import (
    ScheduleEntryCore,
    assemble_signed_schedule_entry,
    build_schedule_entry_signature_payload,
    build_schedule_manifest_payload,
    canonical_json_bytes,
    prepare_origin_information_set_envelope,
    public_key_id,
)
from pfc_shaping.lt.origin_registration_request import (
    RECEIPT_STATUS,
    REQUEST_SCHEMA_VERSION,
    REQUEST_STATUS,
    OriginRegistrationRequestError,
    ReceiptContractReadiness,
    RegistrationHeadExpectation,
    VerifiedRegistrationRequest,
    assemble_signed_origin_registration_request,
    build_origin_registration_signature_payload,
    receipt_contract_readiness,
    verify_signed_origin_registration_request,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V2-20260730.json"
)


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label.encode()).digest())


def _commitments(salt: str = "") -> dict[str, str]:
    fields = (
        "origin_target_mask_inventory_sha256",
        "origin_target_mask_inventory_id",
        "origin_available_eex_product_inventory_sha256",
        "eex_vintage_manifest_sha256",
        "monthly_solver_configuration_sha256",
        "candidate_baseline_identity_manifest_sha256",
        "candidate_hypothesis_and_procedure_manifest_sha256",
        "prediction_commitment_sha256",
        "scenario_commitment_sha256",
        "structural_target_universe_commitment_sha256",
        "ex_ante_evaluation_mask_rule_commitment_sha256",
        "calendar_and_strata_manifest_sha256",
        "runtime_receipt_sha256",
        "project_wheel_sha256",
        "project_source_revision",
    )
    return {
        name: hashlib.sha256(f"synthetic:{salt}:{name}".encode()).hexdigest() for name in fields
    }


def _schedule(private: Ed25519PrivateKey, *, salt: str = "") -> bytes:
    entries: list[bytes] = []
    for index, slot in enumerate(default_evaluation_protocol().holdout.origin_slots):
        origin = slot.origin_as_of_utc
        core = ScheduleEntryCore(
            cadence_slot_id=slot.slot_id.removeprefix("origin-"),
            eex_trading_day=origin.date().isoformat(),
            capture_window_open_utc=(origin - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            capture_window_close_utc=(origin + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            latest_external_registry_commit_utc=(origin + timedelta(hours=2)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            official_eex_calendar_document_sha256=hashlib.sha256(
                f"synthetic-calendar:{salt}:{index}".encode()
            ).hexdigest(),
            official_settlement_event_definition_sha256=hashlib.sha256(
                f"synthetic-settlement:{salt}:{index}".encode()
            ).hexdigest(),
        )
        signing_payload = build_schedule_entry_signature_payload(core)
        entries.append(
            assemble_signed_schedule_entry(
                signature_payload=signing_payload,
                signer_key_id=public_key_id(private.public_key()),
                signature_base64=base64.b64encode(private.sign(signing_payload)).decode("ascii"),
            )
        )
    return build_schedule_manifest_payload(entries)


def _inputs(*, commitment_salt: str = "") -> tuple[bytes, bytes, Ed25519PrivateKey, bytes]:
    schedule_key = _private("schedule")
    schedule = _schedule(schedule_key)
    envelope = prepare_origin_information_set_envelope(
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=schedule_key.public_key(),
        slot_id="origin-2026-10",
        first_target_delivery_start_utc="2026-10-31T23:00:00Z",
        commitments=_commitments(commitment_salt),
    )
    trusted_time = canonical_json_bytes(
        {
            "classification": "SYNTHETIC_OPAQUE_TRUSTED_TIME_BYTES",
            "origin_as_of_utc": "2026-10-06T12:00:00Z",
        }
    )
    return envelope, schedule, schedule_key, trusted_time


def _head(sequence: int = 1) -> RegistrationHeadExpectation:
    return RegistrationHeadExpectation(
        registration_operation_id="21d3f4e7-a1b2-4c5d-8e9f-1029384756ab",
        expected_sequence=sequence,
        expected_previous_receipt_id=None if sequence == 1 else "9" * 64,
    )


def _signed_request() -> tuple[bytes, bytes, bytes, Ed25519PrivateKey, bytes, Ed25519PrivateKey]:
    envelope, schedule, schedule_key, trusted_time = _inputs()
    request_key = _private("request")
    signature_payload = build_origin_registration_signature_payload(
        information_set_envelope_payload=envelope,
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=schedule_key.public_key(),
        trusted_time_receipt_payload=trusted_time,
        head=_head(),
    )
    request = assemble_signed_origin_registration_request(
        signature_payload=signature_payload,
        signer_key_id=public_key_id(request_key.public_key()),
        signature_base64=base64.b64encode(request_key.sign(signature_payload)).decode("ascii"),
    )
    return request, envelope, schedule, schedule_key, trusted_time, request_key


def test_request_matches_the_exact_protocol_v2_field_inventory() -> None:
    request, *_ = _signed_request()
    document = json.loads(request)
    protocol = json.loads(PROTOCOL.read_bytes())
    required = protocol["registration_request_contract"]["required_fields"]

    assert document["schema_version"] == REQUEST_SCHEMA_VERSION
    assert set(document) == set(required)
    assert request == canonical_json_bytes(document)
    assert document["truth_opened"] is False
    assert document["future_holdout_consumed"] is False
    assert document["t057_consumed"] is False


def test_signed_request_verifies_but_all_external_authority_remains_negative() -> None:
    request, envelope, schedule, schedule_key, trusted_time, request_key = _signed_request()
    verified = verify_signed_origin_registration_request(
        request,
        trusted_request_public_key=request_key.public_key(),
        information_set_envelope_payload=envelope,
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=schedule_key.public_key(),
        trusted_time_receipt_payload=trusted_time,
    )
    manifest = verified.to_manifest()

    assert verified.status == REQUEST_STATUS
    assert verified.cryptographic_request_signature_verified is True
    assert verified.request_sha256 == hashlib.sha256(request).hexdigest()
    assert verified.expected_sequence == 1
    assert manifest["missing_external_evidence"] == [
        "TRUSTED_TIME_RECEIPT_SEMANTIC_ADMISSION",
        "REQUEST_SIGNER_EXTERNAL_ROLE_ADMISSION",
        "EXTERNAL_COMPARE_AND_APPEND_RECEIPT",
        "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION",
    ]
    false_authorities = (
        "externally_registered",
        "countable_origin",
        "trusted_time_semantically_admitted",
        "receipt_verified",
        "fresh_head_observation_verified",
        "truth_open_authorized",
        "model_training_authorized",
        "model_selection_authorized",
        "scientific_admission",
        "production_authorization",
        "promotion_gate",
    )
    assert all(manifest[field] is False for field in false_authorities)
    with pytest.raises(TypeError):
        VerifiedRegistrationRequest(  # type: ignore[call-arg]
            request_id="1" * 64,
            request_sha256="2" * 64,
            request_signer_key_id="3" * 64,
            registration_operation_id=str(uuid.uuid4()),
            expected_sequence=1,
            expected_previous_receipt_id=None,
            countable_origin=True,
        )
    with pytest.raises(FrozenInstanceError):
        verified.countable_origin = True  # type: ignore[misc]


def test_request_binds_exact_schedule_envelope_and_trusted_time_bytes() -> None:
    request, envelope, schedule, schedule_key, trusted_time, request_key = _signed_request()
    for changes in (
        {"trusted_time_receipt_payload": trusted_time + b"\n"},
        {"information_set_envelope_payload": _inputs(commitment_salt="other")[0]},
        {
            "schedule_manifest_payload": _schedule(schedule_key, salt="other"),
        },
    ):
        kwargs = {
            "trusted_request_public_key": request_key.public_key(),
            "information_set_envelope_payload": envelope,
            "schedule_manifest_payload": schedule,
            "trusted_schedule_public_key": schedule_key.public_key(),
            "trusted_time_receipt_payload": trusted_time,
        }
        kwargs.update(changes)
        with pytest.raises(ValueError):
            verify_signed_origin_registration_request(request, **kwargs)


@pytest.mark.parametrize("mutation", ["signature", "request_id", "commitment"])
def test_request_tampering_fails_closed_even_when_rehashed_or_resigned(
    mutation: str,
) -> None:
    request, envelope, schedule, schedule_key, trusted_time, request_key = _signed_request()
    document = json.loads(request)
    if mutation == "signature":
        document["request_signature"]["value_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
    elif mutation == "request_id":
        document["request_id"] = "0" * 64
    else:
        document["scenario_commitment_sha256"] = "7" * 64
        core = dict(document)
        core.pop("request_signature")
        core_without_id = dict(core)
        core_without_id.pop("request_id")
        document["request_id"] = hashlib.sha256(
            b"FMV_CH_LT_ORIGIN_REGISTRATION_REQUEST_V2"
            + b"\x00"
            + canonical_json_bytes(core_without_id)
        ).hexdigest()
        signing = dict(document)
        signing.pop("request_signature")
        document["request_signature"] = {
            "algorithm": "ED25519",
            "key_id": public_key_id(request_key.public_key()),
            "value_base64": base64.b64encode(
                request_key.sign(canonical_json_bytes(signing))
            ).decode("ascii"),
        }
    with pytest.raises(OriginRegistrationRequestError):
        verify_signed_origin_registration_request(
            canonical_json_bytes(document),
            trusted_request_public_key=request_key.public_key(),
            information_set_envelope_payload=envelope,
            schedule_manifest_payload=schedule,
            trusted_schedule_public_key=schedule_key.public_key(),
            trusted_time_receipt_payload=trusted_time,
        )


def test_wrong_request_key_and_noncanonical_request_fail_closed() -> None:
    request, envelope, schedule, schedule_key, trusted_time, request_key = _signed_request()
    kwargs = {
        "information_set_envelope_payload": envelope,
        "schedule_manifest_payload": schedule,
        "trusted_schedule_public_key": schedule_key.public_key(),
        "trusted_time_receipt_payload": trusted_time,
    }
    with pytest.raises(OriginRegistrationRequestError, match="trust binding"):
        verify_signed_origin_registration_request(
            request,
            trusted_request_public_key=_private("wrong").public_key(),
            **kwargs,
        )
    with pytest.raises(OriginRegistrationRequestError, match="canonical"):
        verify_signed_origin_registration_request(
            request + b"\n",
            trusted_request_public_key=request_key.public_key(),
            **kwargs,
        )
    duplicate = request.replace(
        b'{"cadence_contract_sha256":', b'{"x":1,"x":2,"cadence_contract_sha256":', 1
    )
    with pytest.raises(OriginRegistrationRequestError, match="duplicate key"):
        verify_signed_origin_registration_request(
            duplicate,
            trusted_request_public_key=request_key.public_key(),
            **kwargs,
        )


def test_request_and_schedule_signer_roles_must_be_disjoint() -> None:
    request, envelope, schedule, schedule_key, trusted_time, _ = _signed_request()

    with pytest.raises(OriginRegistrationRequestError, match="disjoint"):
        verify_signed_origin_registration_request(
            request,
            trusted_request_public_key=schedule_key.public_key(),
            information_set_envelope_payload=envelope,
            schedule_manifest_payload=schedule,
            trusted_schedule_public_key=schedule_key.public_key(),
            trusted_time_receipt_payload=trusted_time,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("capture_window_close_utc", "2026-10-06T11:59:59Z", "chronology"),
        ("origin_target_mask_inventory_id", "8" * 64, "origin id binding"),
    ],
)
def test_signature_assembly_rejects_rehashed_invalid_cross_fields(
    field: str, replacement: object, message: str
) -> None:
    request, *_ = _signed_request()
    signing = json.loads(request)
    signing.pop("request_signature")
    signing[field] = replacement
    core = dict(signing)
    core.pop("request_id")
    signing["request_id"] = hashlib.sha256(
        b"FMV_CH_LT_ORIGIN_REGISTRATION_REQUEST_V2" + b"\x00" + canonical_json_bytes(core)
    ).hexdigest()

    with pytest.raises(OriginRegistrationRequestError, match=message):
        assemble_signed_origin_registration_request(
            signature_payload=canonical_json_bytes(signing),
            signer_key_id="7" * 64,
            signature_base64=base64.b64encode(b"x" * 64).decode("ascii"),
        )


def test_head_expectation_enforces_sequence_and_canonical_predecessor() -> None:
    assert _head(2).expected_previous_receipt_id == "9" * 64
    with pytest.raises(OriginRegistrationRequestError, match="genesis"):
        RegistrationHeadExpectation(str(uuid.uuid4()), 1, "1" * 64)
    with pytest.raises(OriginRegistrationRequestError, match="SHA-256"):
        RegistrationHeadExpectation(str(uuid.uuid4()), 2, None)
    with pytest.raises(OriginRegistrationRequestError, match="sequence"):
        RegistrationHeadExpectation(str(uuid.uuid4()), True, None)
    with pytest.raises(OriginRegistrationRequestError, match="canonical UUID"):
        RegistrationHeadExpectation(str(uuid.uuid4()).upper(), 1, None)


def test_non_genesis_request_binds_expected_predecessor() -> None:
    envelope, schedule, schedule_key, trusted_time = _inputs()
    payload = build_origin_registration_signature_payload(
        information_set_envelope_payload=envelope,
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=schedule_key.public_key(),
        trusted_time_receipt_payload=trusted_time,
        head=_head(2),
    )
    document = json.loads(payload)

    assert document["expected_sequence"] == 2
    assert document["expected_previous_receipt_id"] == "9" * 64


def test_receipt_verification_stays_explicitly_unsupported_and_negative() -> None:
    readiness = receipt_contract_readiness()
    manifest = readiness.to_manifest()

    assert readiness == ReceiptContractReadiness()
    assert readiness.status == RECEIPT_STATUS
    assert manifest["verification_implemented"] is True
    assert manifest["local_wire_contract_hash_frozen"] is True
    assert manifest["external_trust_admitted"] is False
    assert manifest["countable_origin"] is False
    assert len(manifest["missing_contract_clauses"]) == 4
    assert all(
        manifest[field] is False
        for field in (
            "scientific_admission",
            "production_authorization",
            "promotion_gate",
        )
    )
    with pytest.raises(TypeError):
        ReceiptContractReadiness(verification_implemented=True)  # type: ignore[call-arg]


def test_runtime_module_has_no_private_key_io_data_training_or_ct_path() -> None:
    source = (ROOT / "pfc_shaping/lt/origin_registration_request.py").read_text(encoding="utf-8")
    forbidden = (
        "Ed25519PrivateKey",
        "pfc_shaping.ct",
        "pfc_shaping.pipeline",
        "databricks",
        "read_parquet",
        "read_csv",
        "open(",
        "Path(",
        "import requests",
        "requests.get",
        "requests.post",
    )

    assert not any(fragment in source for fragment in forbidden)
