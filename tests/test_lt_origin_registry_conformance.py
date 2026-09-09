from __future__ import annotations

import base64
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from pfc_shaping.lt.origin_registration_receipt import (
    OriginRegistrationVerificationContext,
    assemble_signed_origin_registration_receipt,
    build_origin_registration_receipt_signature_payload,
)
from pfc_shaping.lt.origin_registration_request import (
    RegistrationHeadExpectation,
    assemble_signed_origin_registration_request,
    build_origin_registration_signature_payload,
)
from pfc_shaping.lt.origin_registry_conformance import (
    CONTRACT_ID,
    CONTRACT_SHA256,
    OriginRegistryConformanceConflict,
    OriginRegistryConformanceError,
    RegistryKeyLifecycle,
    RegistryPublicKeySpec,
    SyntheticAppendCandidate,
    SyntheticOriginRegistryConformanceHarness,
    VerifiedRegistryTrustBundle,
    assemble_signed_registry_trust_bundle,
    build_registry_trust_bundle_signature_payload,
    prepare_synthetic_append_candidate,
    verify_registry_trust_bundle,
    verify_registry_trust_bundle_chain,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-ORIGIN-REGISTRY-TRUST-TRANSPORT-CONFORMANCE-DRAFT-V1-20260903.json"
)


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label.encode()).digest())


def _key_spec(
    key: Ed25519PrivateKey,
    lifecycle: RegistryKeyLifecycle = RegistryKeyLifecycle.ACTIVE,
    *,
    valid_from: str = "2026-01-01T00:00:00Z",
    valid_until: str = "2027-01-01T00:00:00Z",
) -> RegistryPublicKeySpec:
    return RegistryPublicKeySpec(
        public_key=key.public_key(),
        valid_from_utc=valid_from,
        valid_until_utc=valid_until,
        lifecycle_status=lifecycle,
    )


def _signed_bundle(
    root: Ed25519PrivateKey,
    specs: tuple[RegistryPublicKeySpec, ...],
    *,
    revision: int = 1,
    previous_bundle_id: str | None = None,
    issued_at: str = "2026-06-01T00:00:00Z",
) -> bytes:
    signing = build_registry_trust_bundle_signature_payload(
        revision=revision,
        previous_bundle_id=previous_bundle_id,
        issued_at_utc=issued_at,
        registry_keys=specs,
    )
    return assemble_signed_registry_trust_bundle(
        signature_payload=signing,
        signer_key_id=public_key_id(root.public_key()),
        signature_base64=base64.b64encode(root.sign(signing)).decode("ascii"),
    )


def _synthetic_candidate(
    label: str,
    *,
    sequence: int = 1,
    previous_receipt_id: str | None = None,
    slot: str = "2026-10",
    origin_at: str = "2026-10-06T12:00:00Z",
    operation_id: str | None = None,
    request_id: str | None = None,
    receipt_id: str | None = None,
    request_payload: bytes | None = None,
    receipt_payload: bytes | None = None,
) -> SyntheticAppendCandidate:
    return SyntheticAppendCandidate(
        request_payload=request_payload or canonical_json_bytes({"synthetic_request": label}),
        receipt_payload=receipt_payload or canonical_json_bytes({"synthetic_receipt": label}),
        registration_operation_id=operation_id
        or str(uuid.uuid5(uuid.NAMESPACE_DNS, f"op:{label}")),
        request_id=request_id or hashlib.sha256(f"request:{label}".encode()).hexdigest(),
        cadence_slot_id=slot,
        origin_id=hashlib.sha256(f"origin:{label}".encode()).hexdigest(),
        origin_as_of_utc=origin_at,
        expected_sequence=sequence,
        expected_previous_receipt_id=previous_receipt_id,
        receipt_id=receipt_id or hashlib.sha256(f"receipt:{label}".encode()).hexdigest(),
    )


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
    return {name: hashlib.sha256(f"conformance:{name}".encode()).hexdigest() for name in fields}


def _registration_context() -> tuple[OriginRegistrationVerificationContext, Ed25519PrivateKey]:
    schedule_key = _private("conformance-schedule")
    request_key = _private("conformance-request")
    registry_key = _private("conformance-registry")
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
                f"conformance-calendar:{index}".encode()
            ).hexdigest(),
            official_settlement_event_definition_sha256=hashlib.sha256(
                f"conformance-settlement:{index}".encode()
            ).hexdigest(),
        )
        signing = build_schedule_entry_signature_payload(core)
        entries.append(
            assemble_signed_schedule_entry(
                signature_payload=signing,
                signer_key_id=public_key_id(schedule_key.public_key()),
                signature_base64=base64.b64encode(schedule_key.sign(signing)).decode("ascii"),
            )
        )
    schedule = build_schedule_manifest_payload(entries)
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
            registration_operation_id="21d3f4e7-a1b2-4c5d-8e9f-1029384756ac",
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
        registry_key,
    )


def _registration_receipt(
    context: OriginRegistrationVerificationContext, registry_key: Ed25519PrivateKey
) -> bytes:
    signing = build_origin_registration_receipt_signature_payload(
        context=context, committed_at_utc="2026-10-06T12:30:00Z"
    )
    return assemble_signed_origin_registration_receipt(
        signature_payload=signing,
        signer_key_id=public_key_id(registry_key.public_key()),
        signature_base64=base64.b64encode(registry_key.sign(signing)).decode("ascii"),
    )


def test_contract_is_exactly_hash_closed_and_authority_negative() -> None:
    payload = CONTRACT.read_bytes()
    document = json.loads(payload)
    semantic = dict(document)
    assert semantic.pop("contract_id") == CONTRACT_ID
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
        == CONTRACT_ID
    )
    assert hashlib.sha256(payload).hexdigest() == CONTRACT_SHA256
    assert document["lifecycle"]["countable_origin_authority"] is False
    assert document["synthetic_harness"]["filesystem_or_network_io"] is False
    assert document["synthetic_harness"]["private_key_capability"] is False
    assert all(
        document["local_result_authority"][field] is False
        for field in (
            "trust_root_externally_admitted",
            "external_cas_worm_verified",
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


def test_signed_trust_bundle_verifies_but_never_admits_the_root_or_registry() -> None:
    root = _private("trust-root")
    registry = _private("trust-registry")
    payload = _signed_bundle(root, (_key_spec(registry),))
    bundle = verify_registry_trust_bundle(payload, trust_root_public_key=root.public_key())
    manifest = bundle.to_manifest()

    assert bundle == verify_registry_trust_bundle(payload, trust_root_public_key=root.public_key())
    assert bundle.active_key_id == public_key_id(registry.public_key())
    assert (
        bundle.public_key_for_receipt(
            key_id=bundle.active_key_id, committed_at_utc="2026-10-06T12:30:00Z"
        ).public_bytes_raw()
        == registry.public_key().public_bytes_raw()
    )
    assert manifest["cryptographic_trust_root_signature_verified"] is True
    assert manifest["registry_key_lifecycle_locally_verified"] is True
    assert manifest["trust_root_externally_admitted"] is False
    assert manifest["external_cas_worm_verified"] is False
    assert manifest["countable_origin"] is False
    assert manifest["production_authorization"] is False
    with pytest.raises(TypeError):
        VerifiedRegistryTrustBundle(  # type: ignore[call-arg]
            bundle_id="1" * 64,
            bundle_sha256="2" * 64,
            revision=1,
            previous_bundle_id=None,
            issued_at_utc="2026-06-01T00:00:00Z",
            trust_root_key_id="3" * 64,
            registry_keys=bundle.registry_keys,
            countable_origin=True,
        )
    with pytest.raises(FrozenInstanceError):
        bundle.production_authorization = True  # type: ignore[misc]


def test_trust_bundle_rejects_wrong_root_tampering_and_noncanonical_bytes() -> None:
    root = _private("trust-root")
    payload = _signed_bundle(root, (_key_spec(_private("registry")),))
    with pytest.raises(OriginRegistryConformanceError, match="trust binding"):
        verify_registry_trust_bundle(
            payload, trust_root_public_key=_private("wrong-root").public_key()
        )
    with pytest.raises(OriginRegistryConformanceError, match="canonical"):
        verify_registry_trust_bundle(payload + b"\n", trust_root_public_key=root.public_key())
    document = json.loads(payload)
    document["registry_keys"][0]["valid_until_utc"] = "2028-01-01T00:00:00Z"
    with pytest.raises(OriginRegistryConformanceError, match="bundle id mismatch"):
        verify_registry_trust_bundle(
            canonical_json_bytes(document), trust_root_public_key=root.public_key()
        )


def test_bundle_rejects_root_role_collapse_and_invalid_active_inventory() -> None:
    root = _private("shared-root-registry")
    payload = _signed_bundle(root, (_key_spec(root),))
    with pytest.raises(OriginRegistryConformanceError, match="disjoint"):
        verify_registry_trust_bundle(payload, trust_root_public_key=root.public_key())

    with pytest.raises(OriginRegistryConformanceError, match="exactly one active"):
        build_registry_trust_bundle_signature_payload(
            revision=1,
            previous_bundle_id=None,
            issued_at_utc="2026-06-01T00:00:00Z",
            registry_keys=(
                _key_spec(_private("historical-a"), RegistryKeyLifecycle.HISTORICAL),
                _key_spec(_private("historical-b"), RegistryKeyLifecycle.HISTORICAL),
            ),
        )


def test_complete_bundle_chain_allows_forward_rotation_and_historical_verification() -> None:
    root = _private("rotation-root")
    old = _private("rotation-old")
    new = _private("rotation-new")
    old_active = _key_spec(
        old,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2026-07-01T00:00:00Z",
    )
    first_payload = _signed_bundle(
        root,
        (old_active,),
        issued_at="2026-06-01T00:00:00Z",
    )
    first = verify_registry_trust_bundle(first_payload, trust_root_public_key=root.public_key())
    second_payload = _signed_bundle(
        root,
        (
            _key_spec(
                old,
                RegistryKeyLifecycle.HISTORICAL,
                valid_from="2026-01-01T00:00:00Z",
                valid_until="2026-07-01T00:00:00Z",
            ),
            _key_spec(
                new,
                valid_from="2026-07-01T00:00:00Z",
                valid_until="2027-07-01T00:00:00Z",
            ),
        ),
        revision=2,
        previous_bundle_id=first.bundle_id,
        issued_at="2026-07-01T00:00:00Z",
    )
    current = verify_registry_trust_bundle_chain(
        (first_payload, second_payload), trust_root_public_key=root.public_key()
    )

    assert current.revision == 2
    assert current.active_key_id == public_key_id(new.public_key())
    assert (
        current.public_key_for_receipt(
            key_id=public_key_id(old.public_key()),
            committed_at_utc="2026-06-30T23:59:59Z",
        ).public_bytes_raw()
        == old.public_key().public_bytes_raw()
    )
    with pytest.raises(OriginRegistryConformanceError, match="outside"):
        current.public_key_for_receipt(
            key_id=public_key_id(old.public_key()),
            committed_at_utc="2026-07-01T00:00:00Z",
        )


def test_bundle_chain_rejects_removed_keys_and_lifecycle_regression() -> None:
    root = _private("chain-root")
    old = _private("chain-old")
    new = _private("chain-new")
    old_window = {
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2027-01-01T00:00:00Z",
    }
    first_payload = _signed_bundle(root, (_key_spec(old, **old_window),))
    first = verify_registry_trust_bundle(first_payload, trust_root_public_key=root.public_key())
    removed = _signed_bundle(
        root,
        (
            _key_spec(
                new,
                valid_from="2026-07-01T00:00:00Z",
                valid_until="2027-07-01T00:00:00Z",
            ),
        ),
        revision=2,
        previous_bundle_id=first.bundle_id,
        issued_at="2026-07-01T00:00:00Z",
    )
    with pytest.raises(OriginRegistryConformanceError, match="append-only"):
        verify_registry_trust_bundle_chain(
            (first_payload, removed), trust_root_public_key=root.public_key()
        )

    historical_payload = _signed_bundle(
        root,
        (
            _key_spec(old, RegistryKeyLifecycle.HISTORICAL, **old_window),
            _key_spec(
                new,
                valid_from="2026-07-01T00:00:00Z",
                valid_until="2027-07-01T00:00:00Z",
            ),
        ),
        revision=2,
        previous_bundle_id=first.bundle_id,
        issued_at="2026-07-01T00:00:00Z",
    )
    historical = verify_registry_trust_bundle(
        historical_payload, trust_root_public_key=root.public_key()
    )
    regressed = _signed_bundle(
        root,
        (
            _key_spec(old, RegistryKeyLifecycle.ACTIVE, **old_window),
            _key_spec(
                new,
                RegistryKeyLifecycle.HISTORICAL,
                valid_from="2026-07-01T00:00:00Z",
                valid_until="2027-07-01T00:00:00Z",
            ),
        ),
        revision=3,
        previous_bundle_id=historical.bundle_id,
        issued_at="2026-08-01T00:00:00Z",
    )
    with pytest.raises(OriginRegistryConformanceError, match="regressed"):
        verify_registry_trust_bundle_chain(
            (first_payload, historical_payload, regressed),
            trust_root_public_key=root.public_key(),
        )


@pytest.mark.parametrize(
    "lifecycle",
    [RegistryKeyLifecycle.REVOKED, RegistryKeyLifecycle.COMPROMISED],
)
def test_revoked_or_compromised_registry_key_never_verifies_receipts(
    lifecycle: RegistryKeyLifecycle,
) -> None:
    root = _private(f"blocked-root:{lifecycle.value}")
    blocked = _private(f"blocked-key:{lifecycle.value}")
    active = _private(f"active-key:{lifecycle.value}")
    payload = _signed_bundle(
        root,
        (
            _key_spec(blocked, lifecycle),
            _key_spec(active),
        ),
    )
    bundle = verify_registry_trust_bundle(payload, trust_root_public_key=root.public_key())
    with pytest.raises(OriginRegistryConformanceError, match="not trusted"):
        bundle.public_key_for_receipt(
            key_id=public_key_id(blocked.public_key()),
            committed_at_utc="2026-06-01T00:00:00Z",
        )


def test_cryptographic_candidate_factory_binds_bundle_receipt_and_request_chain() -> None:
    context, registry_key = _registration_context()
    receipt = _registration_receipt(context, registry_key)
    root = _private("integration-trust-root")
    bundle_payload = _signed_bundle(root, (_key_spec(registry_key),))
    bundle = verify_registry_trust_bundle(bundle_payload, trust_root_public_key=root.public_key())
    candidate = prepare_synthetic_append_candidate(
        context=context,
        receipt_payload=receipt,
        trust_bundle=bundle,
    )
    outcome = SyntheticOriginRegistryConformanceHarness().compare_and_append(candidate)

    assert outcome.committed.to_manifest()["countable_origin"] is False
    assert outcome.committed.to_manifest()["production_authorization"] is False

    wrong_bundle_payload = _signed_bundle(root, (_key_spec(_private("unrelated-registry-key")),))
    wrong_bundle = verify_registry_trust_bundle(
        wrong_bundle_payload, trust_root_public_key=root.public_key()
    )
    with pytest.raises(OriginRegistryConformanceError, match="absent"):
        prepare_synthetic_append_candidate(
            context=context,
            receipt_payload=receipt,
            trust_bundle=wrong_bundle,
        )


def test_synthetic_append_exact_retry_head_and_operation_lookup_are_immutable() -> None:
    harness = SyntheticOriginRegistryConformanceHarness()
    candidate = _synthetic_candidate("first")
    first = harness.compare_and_append(candidate)
    retry = harness.compare_and_append(candidate)

    assert first.idempotent_retry is False
    assert retry.idempotent_retry is True
    assert retry.committed is first.committed
    assert harness.get_head() == candidate.receipt_payload
    assert (
        harness.lookup_operation(
            candidate.registration_operation_id,
            expected_request_payload=candidate.request_payload,
        )
        is first.committed
    )
    assert first.committed.receipt_payload == candidate.receipt_payload
    assert first.committed.to_manifest()["synthetic_state_transition"] is True
    assert first.committed.to_manifest()["externally_registered"] is False
    assert first.committed.to_manifest()["countable_origin"] is False


def test_divergent_committed_retry_fails_without_mutating_prior_result() -> None:
    harness = SyntheticOriginRegistryConformanceHarness()
    first = _synthetic_candidate("committed")
    committed = harness.compare_and_append(first).committed
    divergent = _synthetic_candidate(
        "divergent",
        operation_id=first.registration_operation_id,
        request_id=first.request_id,
    )
    with pytest.raises(OriginRegistryConformanceConflict, match="DIVERGENT_OPERATION_RETRY"):
        harness.compare_and_append(divergent)

    assert harness.get_head() == committed.receipt_payload
    assert harness.rejection_count == 0
    assert (
        harness.lookup_operation(
            first.registration_operation_id,
            expected_request_payload=first.request_payload,
        )
        is committed
    )


def test_duplicate_slot_rejection_is_sanitized_retained_and_nonreplaceable() -> None:
    harness = SyntheticOriginRegistryConformanceHarness()
    first = _synthetic_candidate("slot-first")
    committed = harness.compare_and_append(first).committed
    duplicate = _synthetic_candidate(
        "slot-duplicate",
        sequence=2,
        previous_receipt_id=first.receipt_id,
        slot=first.cadence_slot_id,
        origin_at="2026-10-07T12:00:00Z",
    )
    with pytest.raises(OriginRegistryConformanceConflict) as captured:
        harness.compare_and_append(duplicate)
    rejection = captured.value.rejection

    assert rejection is not None
    assert rejection.error_code == "CADENCE_SLOT_ID_CONFLICT"
    assert set(rejection.to_manifest()) == {
        "schema_version",
        "rejection_id",
        "registration_operation_id",
        "request_id",
        "request_sha256",
        "error_code",
        "expected_sequence",
        "expected_previous_receipt_id",
        "observed_head_sequence",
        "observed_head_receipt_id",
        "countable_origin",
        "production_authorization",
    }
    assert not hasattr(rejection, "request_payload")
    assert not hasattr(rejection, "receipt_payload")
    assert (
        harness.lookup_operation(
            duplicate.registration_operation_id,
            expected_request_payload=duplicate.request_payload,
        )
        == rejection
    )
    with pytest.raises(OriginRegistryConformanceConflict) as exact_retry:
        harness.compare_and_append(duplicate)
    assert exact_retry.value.rejection is rejection
    with pytest.raises(
        OriginRegistryConformanceConflict, match="DIVERGENT_REJECTED_OPERATION_RETRY"
    ):
        harness.compare_and_append(
            _synthetic_candidate(
                "replacement",
                sequence=2,
                previous_receipt_id=first.receipt_id,
                slot="2026-11",
                origin_at="2026-11-03T12:00:00Z",
                operation_id=duplicate.registration_operation_id,
            )
        )
    assert harness.get_head() == committed.receipt_payload
    assert harness.rejection_count == 1


def test_stale_head_rejection_does_not_advance_sequence() -> None:
    harness = SyntheticOriginRegistryConformanceHarness()
    first = _synthetic_candidate("head-first")
    committed = harness.compare_and_append(first).committed
    stale = _synthetic_candidate(
        "head-stale",
        slot="2026-11",
        origin_at="2026-11-03T12:00:00Z",
    )
    with pytest.raises(OriginRegistryConformanceConflict) as captured:
        harness.compare_and_append(stale)

    assert captured.value.error_code == "STALE_HEAD_CONFLICT"
    assert captured.value.rejection is not None
    assert captured.value.rejection.observed_head_sequence == 1
    assert captured.value.rejection.observed_head_receipt_id == first.receipt_id
    assert harness.get_head() == committed.receipt_payload


def test_concurrent_genesis_candidates_have_one_linearized_winner() -> None:
    harness = SyntheticOriginRegistryConformanceHarness()
    candidates = (
        _synthetic_candidate("race-a", slot="2026-10", origin_at="2026-10-06T12:00:00Z"),
        _synthetic_candidate("race-b", slot="2026-11", origin_at="2026-11-03T12:00:00Z"),
    )

    def submit(candidate: SyntheticAppendCandidate) -> str:
        try:
            harness.compare_and_append(candidate)
        except OriginRegistryConformanceConflict:
            return "rejected"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, candidates))

    assert sorted(results) == ["committed", "rejected"]
    assert harness.get_head() in {candidate.receipt_payload for candidate in candidates}
    assert harness.rejection_count == 1


def test_lookup_requires_exact_committed_bytes_or_rejection_hash() -> None:
    harness = SyntheticOriginRegistryConformanceHarness()
    candidate = _synthetic_candidate("lookup")
    harness.compare_and_append(candidate)
    with pytest.raises(OriginRegistryConformanceConflict, match="MISMATCH"):
        harness.lookup_operation(
            candidate.registration_operation_id,
            expected_request_payload=candidate.request_payload + b" ",
        )
    with pytest.raises(OriginRegistryConformanceError, match="UNKNOWN"):
        harness.lookup_operation(
            str(uuid.uuid5(uuid.NAMESPACE_DNS, "unknown")),
            expected_request_payload=b"unknown",
        )


def test_runtime_module_has_no_private_key_io_network_data_training_or_ct_path() -> None:
    source = (ROOT / "pfc_shaping/lt/origin_registry_conformance.py").read_text(encoding="utf-8")
    forbidden = (
        "Ed25519PrivateKey",
        "pfc_shaping.ct",
        "pfc_shaping.pipeline",
        "databricks",
        "sqlite",
        "read_parquet",
        "read_csv",
        "open(",
        "Path(",
        "import requests",
        "requests.get",
        "requests.post",
        "datetime.now",
        "uuid.uuid4",
        "model.fit",
    )

    assert not any(fragment in source for fragment in forbidden)
