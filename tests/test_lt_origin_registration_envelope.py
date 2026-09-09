from __future__ import annotations

import base64
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.origin_registration_envelope import (
    INFORMATION_SET_SCHEMA_VERSION,
    STATUS,
    OriginRegistrationEnvelopeError,
    ScheduleEntryCore,
    assemble_signed_schedule_entry,
    build_schedule_entry_signature_payload,
    build_schedule_manifest_payload,
    canonical_json_bytes,
    prepare_origin_information_set_envelope,
    public_key_id,
    verify_origin_information_set_envelope,
    verify_signed_schedule_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _private(label: str = "synthetic-schedule") -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label.encode()).digest())


def _signed_entry(slot_index: int, private: Ed25519PrivateKey, *, salt: str = "") -> bytes:
    slot = default_evaluation_protocol().holdout.origin_slots[slot_index]
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
            f"synthetic-calendar-{salt}-{slot_index}".encode()
        ).hexdigest(),
        official_settlement_event_definition_sha256=hashlib.sha256(
            f"synthetic-settlement-{salt}-{slot_index}".encode()
        ).hexdigest(),
    )
    signing_payload = build_schedule_entry_signature_payload(core)
    return assemble_signed_schedule_entry(
        signature_payload=signing_payload,
        signer_key_id=public_key_id(private.public_key()),
        signature_base64=base64.b64encode(private.sign(signing_payload)).decode("ascii"),
    )


def _schedule(
    private: Ed25519PrivateKey | None = None, *, salt: str = ""
) -> tuple[bytes, Ed25519PrivateKey]:
    selected = private or _private()
    entries = tuple(_signed_entry(index, selected, salt=salt) for index in range(12))
    return build_schedule_manifest_payload(entries), selected


def _commitments() -> dict[str, str]:
    names = (
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
    return {name: hashlib.sha256(f"synthetic:{name}".encode()).hexdigest() for name in names}


def _first_envelope() -> tuple[bytes, bytes, Ed25519PrivateKey]:
    schedule, private = _schedule()
    envelope = prepare_origin_information_set_envelope(
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=private.public_key(),
        slot_id="origin-2026-10",
        first_target_delivery_start_utc="2026-10-31T23:00:00Z",
        commitments=_commitments(),
    )
    return envelope, schedule, private


def test_schedule_signature_payload_is_canonical_domain_bound_and_deterministic() -> None:
    private = _private()
    signed = _signed_entry(0, private)
    document = json.loads(signed)
    signing_document = dict(document)
    signature = signing_document.pop("signature")

    assert signed == canonical_json_bytes(document)
    assert signature["algorithm"] == "ED25519"
    assert signature["key_id"] == public_key_id(private.public_key())
    private.public_key().verify(
        base64.b64decode(signature["value_base64"]),
        canonical_json_bytes(signing_document),
    )
    assert _signed_entry(0, private) == signed


def test_exact_signed_schedule_matches_all_twelve_frozen_slots() -> None:
    payload, private = _schedule()
    verified = verify_signed_schedule_manifest(payload, trusted_public_key=private.public_key())

    assert len(verified.entry_payloads) == 12
    assert verified.entry_for("2026-10")["cadence_slot_id"] == "2026-10"
    assert verified.entry_for("2027-09")["cadence_slot_id"] == "2027-09"
    assert verified.signer_key_id == public_key_id(private.public_key())
    assert verified.payload_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("mutation", ["signature", "entry", "identity"])
def test_schedule_signature_and_identity_tampering_fail_closed(mutation: str) -> None:
    payload, private = _schedule()
    document = json.loads(payload)
    entry = document["entries"][0]
    if mutation == "signature":
        entry["signature"]["value_base64"] = base64.b64encode(b"x" * 64).decode()
    elif mutation == "entry":
        entry["capture_window_close_utc"] = "2026-10-06T13:00:01Z"
    else:
        entry["schedule_entry_id"] = "0" * 64

    with pytest.raises(OriginRegistrationEnvelopeError):
        verify_signed_schedule_manifest(
            canonical_json_bytes(document), trusted_public_key=private.public_key()
        )


def test_wrong_schedule_key_and_noncanonical_payload_fail_closed() -> None:
    payload, private = _schedule()
    with pytest.raises(OriginRegistrationEnvelopeError, match="trust binding"):
        verify_signed_schedule_manifest(payload, trusted_public_key=_private("wrong").public_key())
    with pytest.raises(OriginRegistrationEnvelopeError, match="canonical"):
        verify_signed_schedule_manifest(payload + b"\n", trusted_public_key=private.public_key())
    duplicate = payload.replace(
        b'{"entries":',
        b'{"entries":[],"entries":',
        1,
    )
    with pytest.raises(OriginRegistrationEnvelopeError, match="duplicate key"):
        verify_signed_schedule_manifest(duplicate, trusted_public_key=private.public_key())


def test_schedule_requires_exact_order_complete_cohort_and_origin_window() -> None:
    payload, private = _schedule()
    document = json.loads(payload)
    document["entries"][0], document["entries"][1] = (
        document["entries"][1],
        document["entries"][0],
    )
    with pytest.raises(OriginRegistrationEnvelopeError, match="exact ordered"):
        verify_signed_schedule_manifest(
            canonical_json_bytes(document), trusted_public_key=private.public_key()
        )

    entries = tuple(_signed_entry(index, private) for index in range(1, 12))
    incomplete = build_schedule_manifest_payload(entries)
    with pytest.raises(OriginRegistrationEnvelopeError, match="exact ordered"):
        verify_signed_schedule_manifest(incomplete, trusted_public_key=private.public_key())

    first = json.loads(_signed_entry(0, private))
    signing = dict(first)
    signing.pop("signature")
    core = dict(signing)
    core.pop("schedule_entry_id")
    replacement = ScheduleEntryCore(
        cadence_slot_id=str(core["cadence_slot_id"]),
        eex_trading_day=str(core["eex_trading_day"]),
        capture_window_open_utc="2026-10-06T13:00:00Z",
        capture_window_close_utc="2026-10-06T14:00:00Z",
        latest_external_registry_commit_utc="2026-10-06T15:00:00Z",
        official_eex_calendar_document_sha256=str(core["official_eex_calendar_document_sha256"]),
        official_settlement_event_definition_sha256=str(
            core["official_settlement_event_definition_sha256"]
        ),
    )
    signing_payload = build_schedule_entry_signature_payload(replacement)
    displaced = assemble_signed_schedule_entry(
        signature_payload=signing_payload,
        signer_key_id=public_key_id(private.public_key()),
        signature_base64=base64.b64encode(private.sign(signing_payload)).decode(),
    )
    all_entries = [displaced] + [_signed_entry(index, private) for index in range(1, 12)]
    with pytest.raises(OriginRegistrationEnvelopeError, match="outside"):
        verify_signed_schedule_manifest(
            build_schedule_manifest_payload(all_entries),
            trusted_public_key=private.public_key(),
        )


def test_information_set_envelope_is_exact_hash_bound_and_authority_negative() -> None:
    payload, schedule, private = _first_envelope()
    document = verify_origin_information_set_envelope(
        payload,
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=private.public_key(),
    )
    authority = document["authority"]

    assert document["schema_version"] == INFORMATION_SET_SCHEMA_VERSION
    assert document["slot_id"] == "origin-2026-10"
    assert document["truth_opened"] is False
    assert document["realized_maturity_mask_present"] is False
    assert isinstance(authority, dict)
    assert authority["status"] == STATUS
    assert not any(value for key, value in authority.items() if key != "status")
    assert document["missing_registration_evidence"] == [
        "TRUSTED_ORIGIN_TIME_RECEIPT",
        "INDEPENDENT_REQUEST_SIGNATURE",
        "EXTERNAL_COMPARE_AND_APPEND_RECEIPT",
        "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION",
    ]


def test_envelope_rejects_incomplete_commitments_and_invalid_deadline() -> None:
    schedule, private = _schedule()
    incomplete = _commitments()
    incomplete.pop("scenario_commitment_sha256")
    with pytest.raises(OriginRegistrationEnvelopeError, match="fields are not exact"):
        prepare_origin_information_set_envelope(
            schedule_manifest_payload=schedule,
            trusted_schedule_public_key=private.public_key(),
            slot_id="origin-2026-10",
            first_target_delivery_start_utc="2026-10-31T23:00:00Z",
            commitments=incomplete,
        )
    with pytest.raises(OriginRegistrationEnvelopeError, match="does not match"):
        prepare_origin_information_set_envelope(
            schedule_manifest_payload=schedule,
            trusted_schedule_public_key=private.public_key(),
            slot_id="origin-2026-10",
            first_target_delivery_start_utc="2026-10-06T13:30:00Z",
            commitments=_commitments(),
        )


def test_first_delivery_start_is_verified_in_swiss_local_time_across_dst() -> None:
    schedule, private = _schedule()
    payload = prepare_origin_information_set_envelope(
        schedule_manifest_payload=schedule,
        trusted_schedule_public_key=private.public_key(),
        slot_id="origin-2027-07",
        first_target_delivery_start_utc="2027-07-31T22:00:00Z",
        commitments=_commitments(),
    )

    assert (
        verify_origin_information_set_envelope(
            payload,
            schedule_manifest_payload=schedule,
            trusted_schedule_public_key=private.public_key(),
        )["first_target_delivery_start_utc"]
        == "2027-07-31T22:00:00Z"
    )


def test_envelope_tampering_and_rehashed_authority_escalation_fail_closed() -> None:
    payload, schedule, private = _first_envelope()
    document = json.loads(payload)
    document["slot_id"] = "origin-2026-11"
    with pytest.raises(OriginRegistrationEnvelopeError, match="id mismatch"):
        verify_origin_information_set_envelope(
            canonical_json_bytes(document),
            schedule_manifest_payload=schedule,
            trusted_schedule_public_key=private.public_key(),
        )

    document = json.loads(payload)
    document["authority"]["externally_registered"] = True
    core = dict(document)
    core.pop("envelope_id")
    domain = b"FMV_LT_ORIGIN_INFORMATION_SET_ENVELOPE_V1"
    document["envelope_id"] = hashlib.sha256(
        domain + b"\x00" + canonical_json_bytes(core)
    ).hexdigest()
    with pytest.raises(OriginRegistrationEnvelopeError, match="authority"):
        verify_origin_information_set_envelope(
            canonical_json_bytes(document),
            schedule_manifest_payload=schedule,
            trusted_schedule_public_key=private.public_key(),
        )


def test_envelope_reverification_requires_the_exact_signed_schedule() -> None:
    payload, schedule, private = _first_envelope()
    replacement_schedule, _ = _schedule(private, salt="replacement")

    assert replacement_schedule != schedule
    with pytest.raises(OriginRegistrationEnvelopeError, match="protocol binding"):
        verify_origin_information_set_envelope(
            payload,
            schedule_manifest_payload=replacement_schedule,
            trusted_schedule_public_key=private.public_key(),
        )


def test_signature_assembly_rejects_malformed_or_noncanonical_base64() -> None:
    core = ScheduleEntryCore(
        cadence_slot_id="2026-10",
        eex_trading_day="2026-10-06",
        capture_window_open_utc="2026-10-06T11:00:00Z",
        capture_window_close_utc="2026-10-06T13:00:00Z",
        latest_external_registry_commit_utc="2026-10-06T14:00:00Z",
        official_eex_calendar_document_sha256="1" * 64,
        official_settlement_event_definition_sha256="2" * 64,
    )
    payload = build_schedule_entry_signature_payload(core)
    with pytest.raises(OriginRegistrationEnvelopeError, match="encoding"):
        assemble_signed_schedule_entry(
            signature_payload=payload,
            signer_key_id="3" * 64,
            signature_base64="not-base64",
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"eex_trading_day": "2026-11-01"}, "cadence slot"),
        ({"eex_trading_day": "2026-10-32"}, "trading day"),
        (
            {
                "capture_window_open_utc": "2026-10-06T14:00:00Z",
                "capture_window_close_utc": "2026-10-06T13:00:00Z",
            },
            "chronology",
        ),
    ],
)
def test_schedule_core_rejects_invalid_calendar_and_chronology(
    changes: dict[str, str], message: str
) -> None:
    values = {
        "cadence_slot_id": "2026-10",
        "eex_trading_day": "2026-10-06",
        "capture_window_open_utc": "2026-10-06T11:00:00Z",
        "capture_window_close_utc": "2026-10-06T13:00:00Z",
        "latest_external_registry_commit_utc": "2026-10-06T14:00:00Z",
        "official_eex_calendar_document_sha256": "1" * 64,
        "official_settlement_event_definition_sha256": "2" * 64,
    }
    values.update(changes)

    with pytest.raises(OriginRegistrationEnvelopeError, match=message):
        ScheduleEntryCore(**values)


def test_canonical_json_and_manifest_builder_reject_ambiguous_inputs() -> None:
    with pytest.raises(OriginRegistrationEnvelopeError, match="floats"):
        canonical_json_bytes({"value": 1.0})
    with pytest.raises(OriginRegistrationEnvelopeError, match="exact bytes"):
        build_schedule_manifest_payload([bytearray(b"{}")])  # type: ignore[list-item]


def test_module_has_no_signing_data_truth_training_or_ct_capability() -> None:
    source = (ROOT / "pfc_shaping/lt/origin_registration_envelope.py").read_text(encoding="utf-8")
    forbidden = (
        "Ed25519PrivateKey",
        "pfc_shaping.ct",
        "pfc_shaping.pipeline",
        "databricks",
        "read_parquet",
        "read_csv",
        "open(",
        "Path(",
        "requests.",
    )

    assert not any(fragment in source for fragment in forbidden)
    assert "t057" not in source.lower()
