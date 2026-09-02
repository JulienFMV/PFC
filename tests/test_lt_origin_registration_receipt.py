from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
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
from pfc_shaping.lt.origin_registration_receipt import (
    HEAD_OBSERVATION_SCHEMA_VERSION,
    HEAD_STATUS,
    RECEIPT_SCHEMA_VERSION,
    RECEIPT_STATUS,
    WIRE_CONTRACT_ID,
    WIRE_CONTRACT_SHA256,
    OriginRegistrationReceiptError,
    OriginRegistrationVerificationContext,
    VerifiedRegistrationReceipt,
    VerifiedRegistryHeadObservation,
    assemble_signed_origin_registration_receipt,
    assemble_signed_registry_head_observation,
    build_origin_registration_receipt_signature_payload,
    build_registry_head_observation_signature_payload,
    verify_signed_origin_registration_receipt,
    verify_signed_registry_head_observation,
)
from pfc_shaping.lt.origin_registration_request import (
    RegistrationHeadExpectation,
    assemble_signed_origin_registration_request,
    build_origin_registration_signature_payload,
)

ROOT = Path(__file__).resolve().parents[1]
WIRE_CONTRACT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-ORIGIN-REGISTRY-RECEIPT-HEAD-WIRE-CONTRACT-DRAFT-V1-20260902.json"
)
_RECEIPT_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRATION_RECEIPT_V2"
_RECEIPT_SIGNATURE_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRATION_RECEIPT_SIGNATURE_V1"
_HEAD_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRY_HEAD_OBSERVATION_V1"
_HEAD_SIGNATURE_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRY_HEAD_OBSERVATION_SIGNATURE_V1"


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label.encode()).digest())


def _commitments() -> dict[str, str]:
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
    return {name: hashlib.sha256(f"synthetic:{name}".encode()).hexdigest() for name in fields}


def _schedule(private: Ed25519PrivateKey) -> bytes:
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
                f"synthetic-calendar:{index}".encode()
            ).hexdigest(),
            official_settlement_event_definition_sha256=hashlib.sha256(
                f"synthetic-settlement:{index}".encode()
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


def _context() -> tuple[OriginRegistrationVerificationContext, Ed25519PrivateKey]:
    schedule_key = _private("receipt-schedule")
    request_key = _private("receipt-request")
    schedule = _schedule(schedule_key)
    envelope = prepare_origin_information_set_envelope(
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=schedule_key.public_key(),
        slot_id="origin-2026-10",
        first_target_delivery_start_utc="2026-10-31T23:00:00Z",
        commitments=_commitments(),
    )
    trusted_time = canonical_json_bytes(
        {
            "classification": "SYNTHETIC_OPAQUE_TRUSTED_TIME_BYTES",
            "origin_as_of_utc": "2026-10-06T12:00:00Z",
        }
    )
    request_signing = build_origin_registration_signature_payload(
        information_set_envelope_payload=envelope,
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=schedule_key.public_key(),
        trusted_time_receipt_payload=trusted_time,
        head=RegistrationHeadExpectation(
            registration_operation_id="21d3f4e7-a1b2-4c5d-8e9f-1029384756ab",
            expected_sequence=1,
            expected_previous_receipt_id=None,
        ),
    )
    request = assemble_signed_origin_registration_request(
        signature_payload=request_signing,
        signer_key_id=public_key_id(request_key.public_key()),
        signature_base64=base64.b64encode(request_key.sign(request_signing)).decode("ascii"),
    )
    return (
        OriginRegistrationVerificationContext(
            signed_request_payload=request,
            request_public_key=request_key.public_key(),
            information_set_envelope_payload=envelope,
            schedule_manifest_payload=schedule,
            schedule_public_key=schedule_key.public_key(),
            trusted_time_receipt_payload=trusted_time,
        ),
        _private("receipt-registry"),
    )


def _receipt(
    context: OriginRegistrationVerificationContext,
    registry_key: Ed25519PrivateKey,
    *,
    committed_at_utc: str = "2026-10-06T12:30:00Z",
) -> bytes:
    signing = build_origin_registration_receipt_signature_payload(
        context=context, committed_at_utc=committed_at_utc
    )
    return assemble_signed_origin_registration_receipt(
        signature_payload=signing,
        signer_key_id=public_key_id(registry_key.public_key()),
        signature_base64=base64.b64encode(registry_key.sign(signing)).decode("ascii"),
    )


def _head(
    context: OriginRegistrationVerificationContext,
    registry_key: Ed25519PrivateKey,
    receipt: bytes,
    *,
    nonce: str | None = None,
    observed_at_utc: str = "2026-10-06T12:31:00Z",
    expires_at_utc: str = "2026-10-06T12:35:00Z",
) -> tuple[bytes, str]:
    nonce = nonce or hashlib.sha256(b"synthetic-caller-nonce").hexdigest()
    signing = build_registry_head_observation_signature_payload(
        receipt_payload=receipt,
        registry_public_key=registry_key.public_key(),
        context=context,
        challenge_nonce=nonce,
        observed_at_utc=observed_at_utc,
        expires_at_utc=expires_at_utc,
    )
    payload = assemble_signed_registry_head_observation(
        signature_payload=signing,
        signer_key_id=public_key_id(registry_key.public_key()),
        signature_base64=base64.b64encode(registry_key.sign(signing)).decode("ascii"),
    )
    return payload, nonce


def _resign_receipt(document: dict[str, object], key: Ed25519PrivateKey) -> bytes:
    unsigned = dict(document)
    unsigned.pop("registry_signature", None)
    core = dict(unsigned)
    core.pop("receipt_id", None)
    unsigned["receipt_id"] = hashlib.sha256(
        _RECEIPT_ID_DOMAIN + b"\x00" + canonical_json_bytes(core)
    ).hexdigest()
    signing = _RECEIPT_SIGNATURE_DOMAIN + b"\x00" + canonical_json_bytes(unsigned)
    unsigned["registry_signature"] = {
        "algorithm": "ED25519",
        "key_id": public_key_id(key.public_key()),
        "value_base64": base64.b64encode(key.sign(signing)).decode("ascii"),
    }
    return canonical_json_bytes(unsigned)


def _resign_head(document: dict[str, object], key: Ed25519PrivateKey) -> bytes:
    unsigned = dict(document)
    unsigned.pop("registry_signature", None)
    core = dict(unsigned)
    core.pop("observation_id", None)
    unsigned["observation_id"] = hashlib.sha256(
        _HEAD_ID_DOMAIN + b"\x00" + canonical_json_bytes(core)
    ).hexdigest()
    signing = _HEAD_SIGNATURE_DOMAIN + b"\x00" + canonical_json_bytes(unsigned)
    unsigned["registry_signature"] = {
        "algorithm": "ED25519",
        "key_id": public_key_id(key.public_key()),
        "value_base64": base64.b64encode(key.sign(signing)).decode("ascii"),
    }
    return canonical_json_bytes(unsigned)


def test_wire_contract_is_hash_closed_and_exactly_authority_negative() -> None:
    payload = WIRE_CONTRACT.read_bytes()
    document = json.loads(payload)
    semantic = dict(document)
    assert semantic.pop("wire_contract_id") == WIRE_CONTRACT_ID
    assert (
        hashlib.sha256(
            json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        == WIRE_CONTRACT_ID
    )
    assert hashlib.sha256(payload).hexdigest() == WIRE_CONTRACT_SHA256
    assert document["receipt"]["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert document["head_observation"]["schema_version"] == HEAD_OBSERVATION_SCHEMA_VERSION
    assert document["lifecycle"]["countable_origin_authority"] is False
    assert all(
        document["local_verification_semantics"][field] is False
        for field in (
            "externally_registered",
            "countable_origin",
            "truth_open_authorized",
            "model_training_authorized",
            "model_selection_authorized",
            "scientific_admission",
            "production_authorization",
            "promotion_gate",
        )
    )


def test_receipt_matches_protocol_inventory_and_retains_negative_authority() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    result = verify_signed_origin_registration_receipt(
        receipt, registry_public_key=registry_key.public_key(), context=context
    )
    manifest = result.to_manifest()
    contract = json.loads(WIRE_CONTRACT.read_bytes())

    assert set(json.loads(receipt)) == set(contract["receipt"]["required_fields"])
    assert result.status == RECEIPT_STATUS
    assert result.cryptographic_receipt_verified is True
    assert result.receipt_claimed_countable_prospective_origin is True
    assert manifest["registry_public_key_trust_admitted"] is False
    assert manifest["fresh_head_observation_cryptographically_verified"] is False
    assert manifest["countable_origin"] is False
    assert manifest["externally_registered"] is False
    assert manifest["truth_open_authorized"] is False
    assert manifest["model_training_authorized"] is False
    assert manifest["model_selection_authorized"] is False
    assert manifest["scientific_admission"] is False
    assert manifest["production_authorization"] is False
    assert manifest["promotion_gate"] is False


def test_fresh_nonce_bound_head_verifies_but_never_counts_the_origin() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    head, nonce = _head(context, registry_key, receipt)
    result = verify_signed_registry_head_observation(
        head,
        receipt_payload=receipt,
        registry_public_key=registry_key.public_key(),
        context=context,
        expected_challenge_nonce=nonce,
        verification_time_utc=datetime(2026, 10, 6, 12, 32, tzinfo=timezone.utc),
    )
    manifest = result.to_manifest()
    contract = json.loads(WIRE_CONTRACT.read_bytes())

    assert set(json.loads(head)) == set(contract["head_observation"]["required_fields"])
    assert result.status == HEAD_STATUS
    assert result.cryptographic_receipt_verified is True
    assert result.cryptographic_head_signature_verified is True
    assert result.fresh_head_observation_cryptographically_verified is True
    assert manifest["countable_origin"] is False
    assert manifest["externally_registered"] is False
    assert "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION" not in manifest["missing_external_evidence"]
    assert "EXTERNAL_REGISTRY_PUBLIC_KEY_TRUST_ADMISSION" in manifest["missing_external_evidence"]


def test_signature_payloads_are_domain_separated_from_their_json_documents() -> None:
    context, registry_key = _context()
    receipt_signing = build_origin_registration_receipt_signature_payload(
        context=context, committed_at_utc="2026-10-06T12:30:00Z"
    )
    receipt_signature = registry_key.sign(receipt_signing)
    registry_key.public_key().verify(receipt_signature, receipt_signing)
    with pytest.raises(InvalidSignature):
        registry_key.public_key().verify(receipt_signature, receipt_signing.split(b"\x00", 1)[1])

    receipt = _receipt(context, registry_key)
    head_signing = build_registry_head_observation_signature_payload(
        receipt_payload=receipt,
        registry_public_key=registry_key.public_key(),
        context=context,
        challenge_nonce="4" * 64,
        observed_at_utc="2026-10-06T12:31:00Z",
        expires_at_utc="2026-10-06T12:35:00Z",
    )
    assert head_signing.startswith(_HEAD_SIGNATURE_DOMAIN + b"\x00")
    assert receipt_signing.startswith(_RECEIPT_SIGNATURE_DOMAIN + b"\x00")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("cadence_slot_id", "2099-01", "binding mismatch"),
        ("request_sha256", "7" * 64, "SHA-256 binding mismatch"),
        ("committed_at_utc", "2026-10-06T11:59:59Z", "chronology"),
    ],
)
def test_resigned_receipt_tampering_fails_closed(
    field: str, replacement: object, message: str
) -> None:
    context, registry_key = _context()
    document = json.loads(_receipt(context, registry_key))
    document[field] = replacement
    tampered = _resign_receipt(document, registry_key)

    with pytest.raises(OriginRegistrationReceiptError, match=message):
        verify_signed_origin_registration_receipt(
            tampered, registry_public_key=registry_key.public_key(), context=context
        )


def test_receipt_rejects_false_countability_even_before_signature_attachment() -> None:
    context, registry_key = _context()
    signing = build_origin_registration_receipt_signature_payload(
        context=context, committed_at_utc="2026-10-06T12:30:00Z"
    )
    prefix, payload = signing.split(b"\x00", 1)
    document = json.loads(payload)
    document["countable_prospective_origin"] = False
    core = dict(document)
    core.pop("receipt_id")
    document["receipt_id"] = hashlib.sha256(
        _RECEIPT_ID_DOMAIN + b"\x00" + canonical_json_bytes(core)
    ).hexdigest()
    invalid = prefix + b"\x00" + canonical_json_bytes(document)

    with pytest.raises(OriginRegistrationReceiptError, match="semantics"):
        assemble_signed_origin_registration_receipt(
            signature_payload=invalid,
            signer_key_id=public_key_id(registry_key.public_key()),
            signature_base64=base64.b64encode(registry_key.sign(invalid)).decode("ascii"),
        )


def test_wrong_registry_key_and_role_key_reuse_fail_closed() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    with pytest.raises(OriginRegistrationReceiptError, match="trust binding"):
        verify_signed_origin_registration_receipt(
            receipt, registry_public_key=_private("wrong-registry").public_key(), context=context
        )
    with pytest.raises(OriginRegistrationReceiptError, match="disjoint"):
        verify_signed_origin_registration_receipt(
            receipt, registry_public_key=context.request_public_key, context=context
        )
    with pytest.raises(OriginRegistrationReceiptError, match="disjoint"):
        verify_signed_origin_registration_receipt(
            receipt, registry_public_key=context.schedule_public_key, context=context
        )


def test_noncanonical_duplicate_and_signature_tampering_fail_closed() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    with pytest.raises(OriginRegistrationReceiptError, match="canonical"):
        verify_signed_origin_registration_receipt(
            receipt + b"\n", registry_public_key=registry_key.public_key(), context=context
        )
    duplicate = receipt.replace(b'{"cadence_slot_id":', b'{"x":1,"x":2,"cadence_slot_id":', 1)
    with pytest.raises(OriginRegistrationReceiptError, match="duplicate key"):
        verify_signed_origin_registration_receipt(
            duplicate, registry_public_key=registry_key.public_key(), context=context
        )
    document = json.loads(receipt)
    document["registry_signature"]["value_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
    with pytest.raises(OriginRegistrationReceiptError, match="signature is invalid"):
        verify_signed_origin_registration_receipt(
            canonical_json_bytes(document),
            registry_public_key=registry_key.public_key(),
            context=context,
        )


def test_receipt_reverification_detects_any_request_chain_byte_change() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    altered_context = OriginRegistrationVerificationContext(
        signed_request_payload=context.signed_request_payload + b"\n",
        request_public_key=context.request_public_key,
        information_set_envelope_payload=context.information_set_envelope_payload,
        schedule_manifest_payload=context.schedule_manifest_payload,
        schedule_public_key=context.schedule_public_key,
        trusted_time_receipt_payload=context.trusted_time_receipt_payload,
    )
    with pytest.raises(ValueError):
        verify_signed_origin_registration_receipt(
            receipt, registry_public_key=registry_key.public_key(), context=altered_context
        )


@pytest.mark.parametrize(
    ("verification_time", "message"),
    [
        (datetime(2026, 10, 6, 12, 30, 59, tzinfo=timezone.utc), "not yet valid"),
        (datetime(2026, 10, 6, 12, 35, 1, tzinfo=timezone.utc), "stale"),
    ],
)
def test_head_freshness_uses_caller_time_and_closed_interval(
    verification_time: datetime, message: str
) -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    head, nonce = _head(context, registry_key, receipt)

    with pytest.raises(OriginRegistrationReceiptError, match=message):
        verify_signed_registry_head_observation(
            head,
            receipt_payload=receipt,
            registry_public_key=registry_key.public_key(),
            context=context,
            expected_challenge_nonce=nonce,
            verification_time_utc=verification_time,
        )


def test_head_nonce_and_receipt_hash_rebindings_fail_even_when_resigned() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    head, nonce = _head(context, registry_key, receipt)
    document = json.loads(head)
    document["challenge_nonce"] = "8" * 64
    rebound_nonce = _resign_head(document, registry_key)
    with pytest.raises(OriginRegistrationReceiptError, match="challenge_nonce"):
        verify_signed_registry_head_observation(
            rebound_nonce,
            receipt_payload=receipt,
            registry_public_key=registry_key.public_key(),
            context=context,
            expected_challenge_nonce=nonce,
            verification_time_utc=datetime(2026, 10, 6, 12, 32, tzinfo=timezone.utc),
        )

    document = json.loads(head)
    document["receipt_sha256"] = "9" * 64
    rebound_receipt = _resign_head(document, registry_key)
    with pytest.raises(OriginRegistrationReceiptError, match="receipt_sha256"):
        verify_signed_registry_head_observation(
            rebound_receipt,
            receipt_payload=receipt,
            registry_public_key=registry_key.public_key(),
            context=context,
            expected_challenge_nonce=nonce,
            verification_time_utc=datetime(2026, 10, 6, 12, 32, tzinfo=timezone.utc),
        )


def test_head_builder_rejects_ttl_and_receipt_chronology_violations() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    common = {
        "receipt_payload": receipt,
        "registry_public_key": registry_key.public_key(),
        "context": context,
        "challenge_nonce": "4" * 64,
    }
    with pytest.raises(OriginRegistrationReceiptError, match="TTL"):
        build_registry_head_observation_signature_payload(
            **common,
            observed_at_utc="2026-10-06T12:31:00Z",
            expires_at_utc="2026-10-06T12:36:01Z",
        )
    with pytest.raises(OriginRegistrationReceiptError, match="chronology"):
        build_registry_head_observation_signature_payload(
            **common,
            observed_at_utc="2026-10-06T12:29:59Z",
            expires_at_utc="2026-10-06T12:34:00Z",
        )


def test_authority_booleans_cannot_be_injected_or_mutated() -> None:
    context, registry_key = _context()
    receipt = _receipt(context, registry_key)
    verified = verify_signed_origin_registration_receipt(
        receipt, registry_public_key=registry_key.public_key(), context=context
    )
    with pytest.raises(TypeError):
        VerifiedRegistrationReceipt(  # type: ignore[call-arg]
            receipt_id="1" * 64,
            receipt_sha256="2" * 64,
            request_id="3" * 64,
            registry_signer_key_id="4" * 64,
            sequence=1,
            previous_receipt_id=None,
            committed_at_utc="2026-10-06T12:30:00Z",
            countable_origin=True,
        )
    with pytest.raises(FrozenInstanceError):
        verified.countable_origin = True  # type: ignore[misc]

    head, nonce = _head(context, registry_key, receipt)
    verified_head = verify_signed_registry_head_observation(
        head,
        receipt_payload=receipt,
        registry_public_key=registry_key.public_key(),
        context=context,
        expected_challenge_nonce=nonce,
        verification_time_utc=datetime(2026, 10, 6, 12, 32, tzinfo=timezone.utc),
    )
    with pytest.raises(TypeError):
        VerifiedRegistryHeadObservation(  # type: ignore[call-arg]
            observation_id="1" * 64,
            receipt_id="2" * 64,
            receipt_sha256="3" * 64,
            request_id="4" * 64,
            registry_signer_key_id="5" * 64,
            sequence=1,
            challenge_nonce="6" * 64,
            observed_at_utc="2026-10-06T12:31:00Z",
            expires_at_utc="2026-10-06T12:35:00Z",
            verification_time_utc="2026-10-06T12:32:00Z",
            countable_origin=True,
        )
    with pytest.raises(FrozenInstanceError):
        verified_head.production_authorization = True  # type: ignore[misc]


def test_runtime_module_has_no_private_key_io_data_training_or_ct_path() -> None:
    source = (ROOT / "pfc_shaping/lt/origin_registration_receipt.py").read_text(encoding="utf-8")
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
        "model.fit",
    )

    assert not any(fragment in source for fragment in forbidden)
