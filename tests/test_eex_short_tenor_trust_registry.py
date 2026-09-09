from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.pipeline import production_phases
from pfc_shaping.validation.eex_short_tenor_signed_replay import (
    canonical_json_bytes,
    public_key_id,
)
from pfc_shaping.validation.eex_short_tenor_trust_registry import (
    GOVERNANCE_SIGNATURE_THRESHOLD,
    REGISTRY_PAYLOAD_TYPE,
    REGISTRY_SCHEMA,
    REQUIRED_ROLES,
    TRUST_REGISTRY_CONTRACT_CONTENT_ID,
    TRUST_REGISTRY_CONTRACT_STATUS,
    ShortTenorTrustRegistryError,
    assert_key_accepted_for_trusted_event_time,
    governance_policy_id,
    lifecycle_event,
    registry_dsse_pae,
    validate_registry_document,
    validate_trust_registry_contract,
    verify_trust_registry_chain_paths,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-CH-SHORT-TENOR-TRUST-REGISTRY-CONTRACT-V1.json"
)
OWNERS = {
    "eex_acquisition": "org-eex-acquisition",
    "eex_source_trusted_time": "org-eex-source-time",
    "entsoe_acquisition": "org-entsoe-acquisition",
    "entsoe_source_trusted_time": "org-entsoe-source-time",
    "selection_governance": "org-selection-governance",
    "training_execution": "org-training-execution",
    "trusted_time": "org-independent-time",
}


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label.encode()).digest())


def _pem(private: Ed25519PrivateKey) -> bytes:
    return private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _base_events(
    key_id: str,
    *,
    issued_at: str,
    valid_from: str,
) -> list[dict[str, object]]:
    return [
        lifecycle_event(
            key_id=key_id,
            event_type="ISSUED",
            effective_at_utc=issued_at,
            recorded_at_utc=issued_at,
            reason=None,
            incident_id=None,
        ),
        lifecycle_event(
            key_id=key_id,
            event_type="ACTIVATED",
            effective_at_utc=valid_from,
            recorded_at_utc=valid_from,
            reason=None,
            incident_id=None,
        ),
    ]


def _entry(
    role: str,
    private: Ed25519PrivateKey,
    *,
    issued_at: str = "2025-12-01T00:00:00Z",
    valid_from: str = "2026-01-01T00:00:00Z",
    valid_until: str | None = "2027-01-01T00:00:00Z",
    supersedes: str | None = None,
    terminal_type: str | None = None,
    terminal_effective: str | None = None,
    terminal_recorded: str | None = None,
    terminal_reason: str | None = None,
    incident_id: str | None = None,
) -> dict[str, object]:
    key_id = public_key_id(private.public_key())
    events = _base_events(key_id, issued_at=issued_at, valid_from=valid_from)
    if terminal_type is not None:
        if terminal_effective is None or terminal_recorded is None:
            raise AssertionError("terminal event times are required")
        events.append(
            lifecycle_event(
                key_id=key_id,
                event_type=terminal_type,
                effective_at_utc=terminal_effective,
                recorded_at_utc=terminal_recorded,
                reason=terminal_reason,
                incident_id=incident_id,
            )
        )
    return {
        "owner_organization_id": OWNERS[role],
        "role": role,
        "key_id": key_id,
        "public_key_pem_sha256": _sha(_pem(private)),
        "issued_at_utc": issued_at,
        "valid_from_utc": valid_from,
        "valid_until_utc": valid_until,
        "supersedes_key_id": supersedes,
        "events": events,
    }


def _sort_keys(document: dict[str, object]) -> None:
    document["keys"].sort(  # type: ignore[union-attr]
        key=lambda item: (item["role"], item["valid_from_utc"], item["key_id"])
    )


def _documents() -> tuple[
    list[dict[str, object]],
    dict[str, Ed25519PrivateKey],
    list[Ed25519PrivateKey],
]:
    operational = {role: _private(f"operational-{role}") for role in REQUIRED_ROLES}
    governance = [_private(f"registry-governance-{index}") for index in range(3)]
    policy_id = governance_policy_id(
        [public_key_id(private.public_key()) for private in governance]
    )
    version_one = {
        "schema_version": REGISTRY_SCHEMA,
        "registry_id": "fmv-short-tenor-external-trust-v1",
        "issuer_organization_id": "org-registry-governance",
        "sequence": 1,
        "previous_registry_payload_sha256": None,
        "issued_at_utc": "2026-07-01T00:00:00Z",
        "expires_at_utc": "2026-07-31T00:00:00Z",
        "governance_policy_id": policy_id,
        "policy_content_id": TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "keys": [_entry(role, operational[role]) for role in REQUIRED_ROLES],
    }
    _sort_keys(version_one)
    rotation_time = "2026-07-15T00:00:00Z"
    version_one_training = next(
        entry
        for entry in version_one["keys"]
        if entry["role"] == "training_execution"
    )
    version_one_training["valid_until_utc"] = rotation_time

    version_two = copy.deepcopy(version_one)
    version_two.update(
        {
            "sequence": 2,
            "issued_at_utc": "2026-07-15T00:00:00Z",
            "expires_at_utc": "2026-08-15T00:00:00Z",
        }
    )
    old_training = next(
        entry
        for entry in version_two["keys"]
        if entry["role"] == "training_execution"
    )
    old_training["events"].append(
        lifecycle_event(
            key_id=old_training["key_id"],
            event_type="RETIRED",
            effective_at_utc=rotation_time,
            recorded_at_utc=rotation_time,
            reason="ROTATED",
            incident_id=None,
        )
    )
    new_training = _private("operational-training-execution-v2")
    new_entry = _entry(
        "training_execution",
        new_training,
        issued_at="2026-07-01T00:00:00Z",
        valid_from=rotation_time,
        supersedes=old_training["key_id"],
    )
    version_two["keys"].append(new_entry)
    operational["training_execution_v2"] = new_training
    _sort_keys(version_two)
    version_two["previous_registry_payload_sha256"] = _sha(
        canonical_json_bytes(version_one)
    )
    return [version_one, version_two], operational, governance


def _signed_envelope(
    document: dict[str, object],
    governance: list[Ed25519PrivateKey],
    *,
    signer_indexes: tuple[int, ...] = (0, 1),
) -> bytes:
    payload = canonical_json_bytes(document)
    signatures = []
    for index in signer_indexes:
        private = governance[index]
        signatures.append(
            {
                "keyid": public_key_id(private.public_key()),
                "sig": base64.b64encode(
                    private.sign(registry_dsse_pae(payload))
                ).decode("ascii"),
            }
        )
    signatures.sort(key=lambda item: item["keyid"])
    return canonical_json_bytes(
        {
            "payloadType": REGISTRY_PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode("ascii"),
            "signatures": signatures,
        }
    )


def _write_bundle(
    tmp_path: Path,
    documents: list[dict[str, object]],
    operational: dict[str, Ed25519PrivateKey],
    governance: list[Ed25519PrivateKey],
    *,
    signer_indexes: tuple[int, ...] = (0, 1),
    relink: bool = True,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if relink:
        documents[0]["previous_registry_payload_sha256"] = None
        for index in range(1, len(documents)):
            documents[index]["previous_registry_payload_sha256"] = _sha(
                canonical_json_bytes(documents[index - 1])
            )
    envelope_paths = []
    envelope_hashes = []
    for index, document in enumerate(documents, start=1):
        raw = _signed_envelope(
            document,
            governance,
            signer_indexes=signer_indexes,
        )
        path = tmp_path / f"registry-{index}.dsse.json"
        path.write_bytes(raw)
        envelope_paths.append(path)
        envelope_hashes.append(_sha(raw))

    governance_paths = []
    governance_hashes = []
    for index, private in enumerate(governance):
        raw = _pem(private)
        path = tmp_path / f"governance-{index}.pem"
        path.write_bytes(raw)
        governance_paths.append(path)
        governance_hashes.append(_sha(raw))

    head_ids = {entry["key_id"] for entry in documents[-1]["keys"]}
    registered_paths: dict[str, Path] = {}
    registered_hashes: dict[str, str] = {}
    for private in operational.values():
        key_id = public_key_id(private.public_key())
        if key_id not in head_ids:
            continue
        raw = _pem(private)
        path = tmp_path / f"registered-{key_id}.pem"
        path.write_bytes(raw)
        registered_paths[key_id] = path
        registered_hashes[key_id] = _sha(raw)
    event_times = {role: "2026-07-16T00:00:00Z" for role in REQUIRED_ROLES}
    return {
        "registry_envelope_paths": envelope_paths,
        "expected_registry_envelope_sha256s": envelope_hashes,
        "governance_public_key_paths": governance_paths,
        "expected_governance_public_key_sha256s": governance_hashes,
        "registered_public_key_paths": registered_paths,
        "expected_registered_public_key_sha256s": registered_hashes,
        "expected_registry_id": documents[-1]["registry_id"],
        "expected_head_sequence": documents[-1]["sequence"],
        "expected_head_envelope_sha256": envelope_hashes[-1],
        "caller_reference_time_utc": "2026-08-05T00:00:00Z",
        "caller_event_time_utc_by_role": event_times,
    }


def _bundle(tmp_path: Path) -> dict[str, object]:
    documents, operational, governance = _documents()
    return {
        "documents": documents,
        "operational": operational,
        "governance": governance,
        "kwargs": _write_bundle(tmp_path, documents, operational, governance),
    }


def _rewrite_envelope(
    path: Path,
    document: dict[str, object],
    governance: list[Ed25519PrivateKey],
    kwargs: dict[str, object],
    index: int,
    *,
    signer_indexes: tuple[int, ...] = (0, 1),
) -> None:
    raw = _signed_envelope(document, governance, signer_indexes=signer_indexes)
    path.write_bytes(raw)
    kwargs["expected_registry_envelope_sha256s"][index] = _sha(raw)
    if index == len(kwargs["expected_registry_envelope_sha256s"]) - 1:
        kwargs["expected_head_envelope_sha256"] = _sha(raw)


def test_contract_is_exact_thresholded_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = validate_trust_registry_contract(document)

    assert result["status"] == TRUST_REGISTRY_CONTRACT_STATUS
    assert result["contract_content_id"] == TRUST_REGISTRY_CONTRACT_CONTENT_ID
    assert result["governance_signature_threshold"] == 2
    assert result["external_registry_authority_verified"] is False
    assert result["production_authorized"] is False


def test_two_version_threshold_chain_replays_without_authority(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = verify_trust_registry_chain_paths(**bundle["kwargs"])

    assert result["registry_sequence_count"] == 2
    assert result["head_sequence"] == 2
    assert result["governance_threshold"] == 2
    assert result["local_threshold_signature_integrity_verified"] is True
    assert result["local_rollback_freeze_and_lifecycle_consistency_verified"] is True
    assert result["governance_root_bootstrap_verified"] is False
    assert result["governance_root_rotation_verified"] is False
    assert result["independently_trusted_time_verified"] is False
    assert result["production_authorized"] is False
    assert result["databricks_request_count"] == 0
    assert result["h_drive_access_count"] == 0
    assert result["locally_resolved_key_ids_by_role"]["training_execution"] == (
        public_key_id(bundle["operational"]["training_execution_v2"].public_key())
    )


def test_historical_key_requires_independently_trusted_event_time() -> None:
    documents, operational, _ = _documents()
    old_id = public_key_id(operational["training_execution"].public_key())
    with pytest.raises(ShortTenorTrustRegistryError, match="independently trusted"):
        assert_key_accepted_for_trusted_event_time(
            documents[-1],
            role="training_execution",
            key_id=old_id,
            event_time_utc="2026-07-14T23:59:59Z",
            trusted_event_time_verified=False,
        )
    accepted = assert_key_accepted_for_trusted_event_time(
        documents[-1],
        role="training_execution",
        key_id=old_id,
        event_time_utc="2026-07-14T23:59:59Z",
        trusted_event_time_verified=True,
    )
    assert accepted["historical_signature_time_accepted"] is True
    with pytest.raises(ShortTenorTrustRegistryError, match="key selection"):
        assert_key_accepted_for_trusted_event_time(
            documents[-1],
            role="training_execution",
            key_id=old_id,
            event_time_utc="2026-07-15T00:00:00Z",
            trusted_event_time_verified=True,
        )


def test_compromise_cuts_acceptance_at_invalidity_time() -> None:
    documents, operational, _ = _documents()
    head = documents[-1]
    entry = next(item for item in head["keys"] if item["role"] == "trusted_time")
    invalid_since = "2026-07-10T00:00:00Z"
    entry["events"].append(
        lifecycle_event(
            key_id=entry["key_id"],
            event_type="COMPROMISED",
            effective_at_utc=invalid_since,
            recorded_at_utc="2026-07-14T00:00:00Z",
            reason="KEY_COMPROMISE",
            incident_id="INC-2026-0710",
        )
    )
    replacement = _private("operational-trusted-time-v2")
    head["keys"].append(
        _entry(
            "trusted_time",
            replacement,
            issued_at="2026-07-14T00:00:00Z",
            valid_from="2026-07-14T00:00:00Z",
            supersedes=entry["key_id"],
        )
    )
    _sort_keys(head)
    key_id = public_key_id(operational["trusted_time"].public_key())
    accepted = assert_key_accepted_for_trusted_event_time(
        head,
        role="trusted_time",
        key_id=key_id,
        event_time_utc="2026-07-09T23:59:59Z",
        trusted_event_time_verified=True,
    )
    assert accepted["lifecycle_state_at_registry_issue"] == "COMPROMISED"
    with pytest.raises(ShortTenorTrustRegistryError, match="exactly one key"):
        assert_key_accepted_for_trusted_event_time(
            head,
            role="trusted_time",
            key_id=key_id,
            event_time_utc="2026-07-10T00:00:00Z",
            trusted_event_time_verified=True,
        )
    replacement_id = public_key_id(replacement.public_key())
    replacement_accepted = assert_key_accepted_for_trusted_event_time(
        head,
        role="trusted_time",
        key_id=replacement_id,
        event_time_utc="2026-07-14T00:00:00Z",
        trusted_event_time_verified=True,
    )
    assert replacement_accepted["key_id"] == replacement_id


def test_incident_replacement_rejects_overlap_but_allows_fail_closed_gap() -> None:
    documents, _, _ = _documents()
    head = documents[-1]
    predecessor = next(
        item for item in head["keys"] if item["role"] == "trusted_time"
    )
    predecessor["events"].append(
        lifecycle_event(
            key_id=predecessor["key_id"],
            event_type="REVOKED",
            effective_at_utc="2026-07-10T00:00:00Z",
            recorded_at_utc="2026-07-14T00:00:00Z",
            reason="PRIVILEGE_WITHDRAWN",
            incident_id="INC-REVOKED-1",
        )
    )
    replacement = _entry(
        "trusted_time",
        _private("overlapping-trusted-time-replacement"),
        issued_at="2026-07-01T00:00:00Z",
        valid_from="2026-07-09T23:59:59Z",
        supersedes=predecessor["key_id"],
    )
    head["keys"].append(replacement)
    _sort_keys(head)

    with pytest.raises(ShortTenorTrustRegistryError, match="overlaps predecessor"):
        validate_registry_document(head)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_role", "role inventory"),
        ("duplicate_key", "globally unique"),
        ("owner_overlap", "owners overlap"),
        ("event_id", "event id binding"),
        ("missing_activation", "issued and activated"),
        ("terminal_not_last", "terminal lifecycle event must be last"),
        ("chronology", "chronology"),
        ("compromise_reason", "compromise reason differs"),
        ("missing_valid_until", "finite valid until"),
        ("no_active", "exactly one active"),
        ("two_active", "exactly one active"),
        ("rotation_owner", "rotation owner differs"),
        ("rotation_predecessor", "rotation predecessor differs"),
        ("rotation_boundary", "rotation boundary differs"),
    ],
)
def test_registry_lifecycle_and_owner_separation_fail_closed(
    mutation: str,
    message: str,
) -> None:
    documents, _, _ = _documents()
    head = documents[-1]
    keys = head["keys"]
    training = [item for item in keys if item["role"] == "training_execution"]
    if mutation == "missing_role":
        head["keys"] = [item for item in keys if item["role"] != "trusted_time"]
    elif mutation == "duplicate_key":
        keys[1]["key_id"] = keys[0]["key_id"]
        for event in keys[1]["events"]:
            event["event_id"] = lifecycle_event(
                key_id=keys[1]["key_id"],
                event_type=event["event_type"],
                effective_at_utc=event["effective_at_utc"],
                recorded_at_utc=event["recorded_at_utc"],
                reason=event["reason"],
                incident_id=event["incident_id"],
            )["event_id"]
        _sort_keys(head)
    elif mutation == "owner_overlap":
        for item in keys:
            if item["role"] == "training_execution":
                item["owner_organization_id"] = OWNERS["selection_governance"]
    elif mutation == "event_id":
        keys[0]["events"][0]["event_id"] = "0" * 64
    elif mutation == "missing_activation":
        keys[0]["events"] = keys[0]["events"][:1]
    elif mutation == "terminal_not_last":
        training[0]["events"].append(
            lifecycle_event(
                    key_id=training[0]["key_id"],
                    event_type="ISSUED",
                    effective_at_utc="2026-07-15T00:00:00Z",
                    recorded_at_utc="2026-07-15T00:00:00Z",
                reason=None,
                incident_id=None,
            )
        )
    elif mutation == "chronology":
        event = keys[0]["events"][1]
        event["recorded_at_utc"] = "2025-12-31T00:00:00Z"
        event["event_id"] = lifecycle_event(
            key_id=keys[0]["key_id"],
            event_type=event["event_type"],
            effective_at_utc=event["effective_at_utc"],
            recorded_at_utc=event["recorded_at_utc"],
            reason=event["reason"],
            incident_id=event["incident_id"],
        )["event_id"]
    elif mutation == "compromise_reason":
        entry = next(item for item in keys if item["role"] == "trusted_time")
        entry["events"].append(
            lifecycle_event(
                key_id=entry["key_id"],
                    event_type="COMPROMISED",
                    effective_at_utc="2026-07-10T00:00:00Z",
                    recorded_at_utc="2026-07-14T00:00:00Z",
                reason="SUPERSEDED",
                incident_id="INC-1",
            )
        )
    elif mutation == "missing_valid_until":
        keys[0]["valid_until_utc"] = None
    elif mutation == "no_active":
        entry = next(item for item in keys if item["role"] == "trusted_time")
        entry["events"].append(
            lifecycle_event(
                key_id=entry["key_id"],
                event_type="COMPROMISED",
                effective_at_utc="2026-07-10T00:00:00Z",
                recorded_at_utc="2026-07-14T00:00:00Z",
                reason="KEY_COMPROMISE",
                incident_id="INC-NO-REPLACEMENT",
            )
        )
    elif mutation == "two_active":
        training[0]["events"] = training[0]["events"][:2]
    elif mutation == "rotation_owner":
        training[1]["owner_organization_id"] = "org-other-training"
    elif mutation == "rotation_predecessor":
        training[1]["supersedes_key_id"] = "a" * 64
    elif mutation == "rotation_boundary":
        training[1]["valid_from_utc"] = "2026-07-14T00:00:00Z"
        training[1]["events"][1] = lifecycle_event(
            key_id=training[1]["key_id"],
            event_type="ACTIVATED",
                effective_at_utc="2026-07-14T00:00:00Z",
                recorded_at_utc="2026-07-14T00:00:00Z",
            reason=None,
            incident_id=None,
        )
        _sort_keys(head)
    with pytest.raises(ShortTenorTrustRegistryError, match=message):
        validate_registry_document(head)


def test_threshold_rejects_one_signature(tmp_path: Path) -> None:
    documents, operational, governance = _documents()
    kwargs = _write_bundle(
        tmp_path,
        documents,
        operational,
        governance,
        signer_indexes=(0,),
    )
    with pytest.raises(ShortTenorTrustRegistryError, match="2-of-3"):
        verify_trust_registry_chain_paths(**kwargs)


def test_threshold_rejects_duplicate_unsorted_or_unpinned_signature(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    kwargs = bundle["kwargs"]
    path = kwargs["registry_envelope_paths"][0]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["signatures"].append(copy.deepcopy(envelope["signatures"][0]))
    raw = canonical_json_bytes(envelope)
    path.write_bytes(raw)
    kwargs["expected_registry_envelope_sha256s"][0] = _sha(raw)
    with pytest.raises(ShortTenorTrustRegistryError, match="sorted and unique"):
        verify_trust_registry_chain_paths(**kwargs)

    bundle = _bundle(tmp_path / "unsorted")
    kwargs = bundle["kwargs"]
    path = kwargs["registry_envelope_paths"][0]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["signatures"].reverse()
    raw = canonical_json_bytes(envelope)
    path.write_bytes(raw)
    kwargs["expected_registry_envelope_sha256s"][0] = _sha(raw)
    with pytest.raises(ShortTenorTrustRegistryError, match="sorted and unique"):
        verify_trust_registry_chain_paths(**kwargs)

    documents, operational, governance = _documents()
    governance.append(_private("untrusted-fourth-governance"))
    kwargs = _write_bundle(
        tmp_path / "untrusted",
        documents,
        operational,
        governance[:3],
    )
    untrusted = governance[3]
    path = kwargs["registry_envelope_paths"][0]
    document = documents[0]
    raw = _signed_envelope(document, [untrusted, *governance[:2]])
    path.write_bytes(raw)
    kwargs["expected_registry_envelope_sha256s"][0] = _sha(raw)
    with pytest.raises(ShortTenorTrustRegistryError, match="pinned governance"):
        verify_trust_registry_chain_paths(**kwargs)


def test_signature_is_verified_before_declared_key_id(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = bundle["kwargs"]
    path = kwargs["registry_envelope_paths"][0]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    signature = envelope["signatures"][0]
    signature["keyid"] = envelope["signatures"][1]["keyid"]
    signature["sig"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    raw = canonical_json_bytes(envelope)
    path.write_bytes(raw)
    kwargs["expected_registry_envelope_sha256s"][0] = _sha(raw)
    with pytest.raises(ShortTenorTrustRegistryError, match="exactly one pinned"):
        verify_trust_registry_chain_paths(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("predecessor", "predecessor payload binding"),
        ("sequence", "registry sequence differs"),
        ("issuer", "registry chain issuer differs"),
        ("issued_after_expiry", "before predecessor expiry"),
        ("history_core", "historical key public_key_pem_sha256 differs"),
        ("history_event", "event history is not an immutable prefix"),
    ],
)
def test_registry_chain_continuity_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    documents, operational, governance = _documents()
    if mutation == "predecessor":
        documents[1]["previous_registry_payload_sha256"] = "a" * 64
    elif mutation == "sequence":
        documents[1]["sequence"] = 3
    elif mutation == "issuer":
        documents[1]["issuer_organization_id"] = "org-other-registry"
    elif mutation == "issued_after_expiry":
        documents[0]["expires_at_utc"] = "2026-07-15T00:00:00Z"
    elif mutation == "history_core":
        documents[1]["keys"][0]["public_key_pem_sha256"] = "a" * 64
    elif mutation == "history_event":
        event = documents[1]["keys"][0]["events"][0]
        event["recorded_at_utc"] = "2025-12-02T00:00:00Z"
        event["event_id"] = lifecycle_event(
            key_id=documents[1]["keys"][0]["key_id"],
            event_type=event["event_type"],
            effective_at_utc=event["effective_at_utc"],
            recorded_at_utc=event["recorded_at_utc"],
            reason=event["reason"],
            incident_id=event["incident_id"],
        )["event_id"]
    kwargs = _write_bundle(
        tmp_path,
        documents,
        operational,
        governance,
        relink=mutation != "predecessor",
    )
    with pytest.raises(ShortTenorTrustRegistryError, match=message):
        verify_trust_registry_chain_paths(**kwargs)


def test_head_checkpoint_expiry_ttl_and_event_time_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = bundle["kwargs"]
    kwargs["expected_head_envelope_sha256"] = "0" * 64
    with pytest.raises(ShortTenorTrustRegistryError, match="checkpoint"):
        verify_trust_registry_chain_paths(**kwargs)

    bundle = _bundle(tmp_path / "expired")
    kwargs = bundle["kwargs"]
    kwargs["caller_reference_time_utc"] = "2026-08-20T00:00:00Z"
    with pytest.raises(ShortTenorTrustRegistryError, match="not valid"):
        verify_trust_registry_chain_paths(**kwargs)

    documents, operational, governance = _documents()
    documents[1]["expires_at_utc"] = "2026-08-21T00:00:01Z"
    kwargs = _write_bundle(tmp_path / "ttl", documents, operational, governance)
    with pytest.raises(ShortTenorTrustRegistryError, match="TTL"):
        verify_trust_registry_chain_paths(**kwargs)

    bundle = _bundle(tmp_path / "future-event")
    kwargs = bundle["kwargs"]
    kwargs["caller_event_time_utc_by_role"]["trusted_time"] = (
        "2026-08-06T00:00:00Z"
    )
    with pytest.raises(ShortTenorTrustRegistryError, match="exceeds reference"):
        verify_trust_registry_chain_paths(**kwargs)


def test_exact_paths_hashes_and_key_sets_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = bundle["kwargs"]
    kwargs["expected_registry_envelope_sha256s"][0] = "0" * 64
    with pytest.raises(ShortTenorTrustRegistryError, match="SHA-256 differs"):
        verify_trust_registry_chain_paths(**kwargs)

    bundle = _bundle(tmp_path / "missing")
    kwargs = bundle["kwargs"]
    selected = next(iter(kwargs["registered_public_key_paths"]))
    kwargs["registered_public_key_paths"].pop(selected)
    with pytest.raises(ShortTenorTrustRegistryError, match="path set differs"):
        verify_trust_registry_chain_paths(**kwargs)

    bundle = _bundle(tmp_path / "wrong-key-hash")
    kwargs = bundle["kwargs"]
    selected = next(iter(kwargs["expected_registered_public_key_sha256s"]))
    kwargs["expected_registered_public_key_sha256s"][selected] = "0" * 64
    with pytest.raises(ShortTenorTrustRegistryError, match="caller hash binding"):
        verify_trust_registry_chain_paths(**kwargs)

    bundle = _bundle(tmp_path / "relative")
    kwargs = bundle["kwargs"]
    kwargs["registry_envelope_paths"][0] = Path("relative.json")
    with pytest.raises(ShortTenorTrustRegistryError, match="path is unsafe"):
        verify_trust_registry_chain_paths(**kwargs)


def test_private_key_material_and_governance_collision_are_rejected(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    kwargs = bundle["kwargs"]
    private = bundle["operational"]["training_execution_v2"]
    selected_id = public_key_id(private.public_key())
    selected_path = kwargs["registered_public_key_paths"][selected_id]
    private_raw = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    selected_path.write_bytes(private_raw)
    kwargs["expected_registered_public_key_sha256s"][selected_id] = _sha(private_raw)
    head_entry = next(
        entry
        for entry in bundle["documents"][-1]["keys"]
        if entry["key_id"] == selected_id
    )
    head_entry["public_key_pem_sha256"] = _sha(private_raw)
    _rewrite_envelope(
        kwargs["registry_envelope_paths"][-1],
        bundle["documents"][-1],
        bundle["governance"],
        kwargs,
        1,
    )
    with pytest.raises(ShortTenorTrustRegistryError, match="private key material"):
        verify_trust_registry_chain_paths(**kwargs)

    documents, operational, governance = _documents()
    target = next(
        entry
        for entry in documents[-1]["keys"]
        if entry["role"] == "training_execution"
        and entry["supersedes_key_id"] is not None
    )
    replacement = governance[0]
    old_id = target["key_id"]
    new_id = public_key_id(replacement.public_key())
    target["key_id"] = new_id
    target["public_key_pem_sha256"] = _sha(_pem(replacement))
    target["events"] = _base_events(
        new_id,
        issued_at=target["issued_at_utc"],
        valid_from=target["valid_from_utc"],
    )
    _sort_keys(documents[-1])
    operational.pop(
        next(
            label
            for label, key in operational.items()
            if public_key_id(key.public_key()) == old_id
        )
    )
    operational["governance-collision"] = replacement
    kwargs = _write_bundle(
        tmp_path / "collision",
        documents,
        operational,
        governance,
    )
    with pytest.raises(ShortTenorTrustRegistryError, match="external to registered"):
        verify_trust_registry_chain_paths(**kwargs)


def test_strict_canonical_envelope_and_payload_are_required(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = bundle["kwargs"]
    path = kwargs["registry_envelope_paths"][0]
    raw = path.read_bytes().replace(
        b'{"payload":',
        b'{"payloadType":"duplicate","payload":',
        1,
    )
    path.write_bytes(raw)
    kwargs["expected_registry_envelope_sha256s"][0] = _sha(raw)
    with pytest.raises(ShortTenorTrustRegistryError, match="strict UTF-8 JSON"):
        verify_trust_registry_chain_paths(**kwargs)

    bundle = _bundle(tmp_path / "noncanonical")
    kwargs = bundle["kwargs"]
    path = kwargs["registry_envelope_paths"][0]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    raw = json.dumps(envelope, indent=2).encode("utf-8")
    path.write_bytes(raw)
    kwargs["expected_registry_envelope_sha256s"][0] = _sha(raw)
    with pytest.raises(ShortTenorTrustRegistryError, match="canonical JSON"):
        verify_trust_registry_chain_paths(**kwargs)


def test_contract_hash_detects_any_policy_drift() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document["authorities"]["production_authorized"] = True
    with pytest.raises(ShortTenorTrustRegistryError, match="content id"):
        validate_trust_registry_contract(document)


def test_registry_replay_is_absent_from_lt_production() -> None:
    production_source = inspect.getsource(production_phases)
    verifier_source = inspect.getsource(verify_trust_registry_chain_paths).lower()

    assert "eex_short_tenor_trust_registry" not in production_source
    assert "verify_trust_registry_chain_paths" not in production_source
    assert "databricks.sql" not in verifier_source
    assert "read_excel" not in verifier_source
    assert GOVERNANCE_SIGNATURE_THRESHOLD == 2
