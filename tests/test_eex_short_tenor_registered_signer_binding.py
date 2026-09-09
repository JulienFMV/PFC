from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.pipeline import production_phases
from pfc_shaping.validation import eex_short_tenor_registered_signer_binding as binding
from pfc_shaping.validation.eex_short_tenor_registered_signer_binding import (
    MAXIMUM_REGISTRY_PUBLICATION_LAG_DAYS,
    REGISTERED_SIGNER_BINDING_CONTRACT_CONTENT_ID,
    REGISTERED_SIGNER_BINDING_CONTRACT_STATUS,
    ShortTenorRegisteredSignerBindingError,
    validate_registered_signer_binding_contract,
    verify_registered_signer_binding_paths,
)
from pfc_shaping.validation.eex_short_tenor_signed_replay import public_key_id
from pfc_shaping.validation.eex_short_tenor_trust_registry import (
    REGISTRY_SCHEMA,
    REQUIRED_ROLES,
    TRUST_REGISTRY_CONTRACT_CONTENT_ID,
    governance_policy_id,
    lifecycle_event,
)
from tests import test_eex_short_tenor_signed_replay as replay_fixtures
from tests import test_eex_short_tenor_trust_registry as registry_fixtures

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-CH-SHORT-TENOR-REGISTERED-SIGNER-BINDING-CONTRACT-V1.json"
)


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"D230-{label}".encode()).digest()
    )


def _bundle(tmp_path: Path) -> dict[str, object]:
    fixture_root = (tmp_path / "fixture").resolve()
    replay_root = fixture_root / "replay"
    registry_root = fixture_root / "registry"
    replay_root.mkdir(parents=True)
    registry_root.mkdir(parents=True)
    operational = {role: _private(f"operational-{role}") for role in REQUIRED_ROLES}
    replay = replay_fixtures._bundle(
        replay_root,
        training_key=operational["training_execution"],
        governance_key=operational["selection_governance"],
        time_key=operational["trusted_time"],
    )
    governance = [_private(f"governance-{index}") for index in range(3)]
    policy_id = governance_policy_id(
        [public_key_id(private.public_key()) for private in governance]
    )
    head = {
        "schema_version": REGISTRY_SCHEMA,
        "registry_id": "fmv-short-tenor-d230-reference-v1",
        "issuer_organization_id": "org-d230-registry-governance",
        "sequence": 1,
        "previous_registry_payload_sha256": None,
        "issued_at_utc": "2027-10-02T00:00:00Z",
        "expires_at_utc": "2027-11-01T00:00:00Z",
        "governance_policy_id": policy_id,
        "policy_content_id": TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "keys": [
            registry_fixtures._entry(
                role,
                operational[role],
                valid_until="2029-01-01T00:00:00Z",
            )
            for role in REQUIRED_ROLES
        ],
    }
    registry_fixtures._sort_keys(head)
    envelope_path = (registry_root / "registry-1.dsse.json").resolve()
    governance_paths = [
        (registry_root / f"governance-{index}.pem").resolve()
        for index in range(3)
    ]
    for path, private in zip(governance_paths, governance, strict=True):
        path.write_bytes(registry_fixtures._pem(private))
    registered_paths: dict[str, Path] = {}
    registered_hashes: dict[str, str] = {}
    replay_key_paths = replay["key_paths"]
    replay_key_payloads = replay["key_payloads"]
    for role, private in operational.items():
        key_id = public_key_id(private.public_key())
        if role in {"training_execution", "selection_governance", "trusted_time"}:
            path = replay_key_paths[role]
            payload = replay_key_payloads[role]
        else:
            path = (registry_root / f"registered-{role}.pem").resolve()
            payload = registry_fixtures._pem(private)
            path.write_bytes(payload)
        registered_paths[key_id] = path
        registered_hashes[key_id] = registry_fixtures._sha(payload)
    bundle = {
        "replay": replay,
        "head": head,
        "operational": operational,
        "governance": governance,
        "envelope_path": envelope_path,
        "signed_replay_inputs": replay_fixtures._kwargs(replay),
        "trust_registry_inputs": {
            "registry_envelope_paths": [envelope_path],
            "expected_registry_envelope_sha256s": ["0" * 64],
            "governance_public_key_paths": governance_paths,
            "expected_governance_public_key_sha256s": [
                registry_fixtures._sha(registry_fixtures._pem(private))
                for private in governance
            ],
            "registered_public_key_paths": registered_paths,
            "expected_registered_public_key_sha256s": registered_hashes,
            "expected_registry_id": head["registry_id"],
            "expected_head_sequence": 1,
            "expected_head_envelope_sha256": "0" * 64,
            "caller_reference_time_utc": "2027-10-15T00:00:00Z",
        },
    }
    _refresh_registry(bundle)
    return bundle


def _refresh_registry(
    bundle: dict[str, object],
    *,
    signer_indexes: tuple[int, ...] = (0, 1),
) -> None:
    raw = registry_fixtures._signed_envelope(
        bundle["head"],
        bundle["governance"],
        signer_indexes=signer_indexes,
    )
    bundle["envelope_path"].write_bytes(raw)
    digest = registry_fixtures._sha(raw)
    inputs = bundle["trust_registry_inputs"]
    inputs["expected_registry_envelope_sha256s"] = [digest]
    inputs["expected_head_envelope_sha256"] = digest


def _add_training_replacement(
    bundle: dict[str, object],
    *,
    terminal_type: str,
    effective_at: str,
    replacement_valid_from: str,
) -> tuple[str, str]:
    head = bundle["head"]
    old = next(
        entry for entry in head["keys"] if entry["role"] == "training_execution"
    )
    if terminal_type == "RETIRED":
        old["valid_until_utc"] = effective_at
        reason = "ROTATED"
        incident_id = None
        recorded_at = effective_at
    else:
        reason = "KEY_COMPROMISE"
        incident_id = "INC-D230-TRAINING"
        recorded_at = head["issued_at_utc"]
    old["events"].append(
        lifecycle_event(
            key_id=old["key_id"],
            event_type=terminal_type,
            effective_at_utc=effective_at,
            recorded_at_utc=recorded_at,
            reason=reason,
            incident_id=incident_id,
        )
    )
    replacement = _private(f"training-replacement-{terminal_type}")
    replacement_entry = registry_fixtures._entry(
        "training_execution",
        replacement,
        valid_from=replacement_valid_from,
        valid_until="2029-01-01T00:00:00Z",
        supersedes=old["key_id"],
    )
    head["keys"].append(replacement_entry)
    registry_fixtures._sort_keys(head)
    payload = registry_fixtures._pem(replacement)
    key_id = public_key_id(replacement.public_key())
    path = (bundle["envelope_path"].parent / f"registered-{key_id}.pem").resolve()
    path.write_bytes(payload)
    registry_inputs = bundle["trust_registry_inputs"]
    registry_inputs["registered_public_key_paths"][key_id] = path
    registry_inputs["expected_registered_public_key_sha256s"][key_id] = (
        registry_fixtures._sha(payload)
    )
    _refresh_registry(bundle)
    return old["key_id"], key_id


def _verify(bundle: dict[str, object]) -> dict[str, object]:
    return verify_registered_signer_binding_paths(
        signed_replay_inputs=bundle["signed_replay_inputs"],
        trust_registry_inputs=bundle["trust_registry_inputs"],
    )


def test_contract_is_exact_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = validate_registered_signer_binding_contract(document)

    assert result["status"] == REGISTERED_SIGNER_BINDING_CONTRACT_STATUS
    assert result["contract_content_id"] == REGISTERED_SIGNER_BINDING_CONTRACT_CONTENT_ID
    assert (
        result["maximum_registry_publication_lag_days"]
        == MAXIMUM_REGISTRY_PUBLICATION_LAG_DAYS
    )
    assert result["actual_signature_generation_time_verified"] is False
    assert result["independently_trusted_time_verified"] is False
    assert result["production_authorized"] is False


def test_valid_combined_replay_binds_three_roles_without_time_authority(
    tmp_path: Path,
) -> None:
    result = _verify(_bundle(tmp_path))

    assert result["registry_event_time_replay_count"] == 2
    assert result["local_d228_signature_integrity_verified"] is True
    assert result["local_d229_registry_integrity_verified"] is True
    assert result["local_registered_signer_identity_binding_verified"] is True
    assert result["local_asserted_event_time_resolution_verified"] is True
    assert result["post_event_registry_head_verified"] is True
    assert result["actual_signature_generation_time_verified"] is False
    assert result["backdating_resistance_verified"] is False
    assert result["source_signer_identity_or_event_time_bound"] is False
    assert result["independently_trusted_time_verified"] is False
    assert result["production_authorized"] is False
    assert result["databricks_request_count"] == 0
    assert result["warehouse_start_count"] == 0
    assert result["network_call_count"] == 0
    assert result["h_drive_access_count"] == 0
    assert result["remote_write_count"] == 0


def test_binding_uses_exact_payload_asserted_times(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = _verify(bundle)
    bindings = result["asserted_event_time_bindings"]

    assert bindings["training_receipt"]["asserted_event_time_utc"] == (
        "2027-09-01T00:00:00Z"
    )
    assert bindings["selection_receipt"]["asserted_event_time_utc"] == (
        "2027-10-01T00:00:01Z"
    )
    assert bindings["training_time_observation"]["registry_role"] == "trusted_time"
    assert bindings["selection_time_observation"]["registry_role"] == "trusted_time"
    assert result["registry_publication_lag_seconds_by_receipt"]["training"] == (
        31 * 24 * 60 * 60
    )


def test_shared_key_must_be_the_identical_path_not_a_byte_copy(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    replay = bundle["replay"]
    key_id = replay["key_ids"]["training_execution"]
    copied = (tmp_path / "copied-training-key.pem").resolve()
    copied.write_bytes(replay["key_payloads"]["training_execution"])
    bundle["trust_registry_inputs"]["registered_public_key_paths"][key_id] = copied

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="shared public key path differs",
    ):
        _verify(bundle)


def test_registry_role_key_must_match_verified_d228_signer(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    replacement = _private("unregistered-training-replacement")
    entry = next(
        item
        for item in bundle["head"]["keys"]
        if item["role"] == "training_execution"
    )
    old_id = entry["key_id"]
    replacement_entry = registry_fixtures._entry(
        "training_execution",
        replacement,
        valid_until="2029-01-01T00:00:00Z",
    )
    bundle["head"]["keys"].remove(entry)
    bundle["head"]["keys"].append(replacement_entry)
    registry_fixtures._sort_keys(bundle["head"])
    new_id = replacement_entry["key_id"]
    path = (tmp_path / "replacement-training.pem").resolve()
    payload = registry_fixtures._pem(replacement)
    path.write_bytes(payload)
    paths = bundle["trust_registry_inputs"]["registered_public_key_paths"]
    hashes = bundle["trust_registry_inputs"]["expected_registered_public_key_sha256s"]
    del paths[old_id]
    del hashes[old_id]
    paths[new_id] = path
    hashes[new_id] = registry_fixtures._sha(payload)
    _refresh_registry(bundle)

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="training_execution registered key resolution differs",
    ):
        _verify(bundle)


def test_registry_head_must_postdate_asserted_events(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle["head"]["issued_at_utc"] = "2027-09-15T00:00:00Z"
    bundle["head"]["expires_at_utc"] = "2027-10-15T00:00:00Z"
    bundle["trust_registry_inputs"]["caller_reference_time_utc"] = (
        "2027-10-10T00:00:00Z"
    )
    _refresh_registry(bundle)

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="registry head predates selection",
    ):
        _verify(bundle)


def test_registry_publication_lag_is_bounded(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle["head"]["issued_at_utc"] = "2027-10-03T00:00:00Z"
    bundle["head"]["expires_at_utc"] = "2027-11-02T00:00:00Z"
    _refresh_registry(bundle)

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="publication lag exceeds maximum for training",
    ):
        _verify(bundle)


def test_caller_cannot_inject_alternate_event_time_mapping(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle["trust_registry_inputs"]["caller_event_time_utc_by_role"] = {
        role: "2027-10-15T00:00:00Z" for role in REQUIRED_ROLES
    }

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="trust registry inputs fields differ",
    ):
        _verify(bundle)


def test_d228_envelope_tamper_remains_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = bundle["replay"]["paths"]["training_envelope"]
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="D228 signed replay failed",
    ):
        _verify(bundle)


def test_d229_threshold_weakening_remains_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _refresh_registry(bundle, signer_indexes=(0,))

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="asserted-event registry replay failed",
    ):
        _verify(bundle)


def test_training_signer_retired_after_asserted_event_remains_historically_valid(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    old_key_id, _ = _add_training_replacement(
        bundle,
        terminal_type="RETIRED",
        effective_at="2027-09-14T00:00:00Z",
        replacement_valid_from="2027-09-14T00:00:00Z",
    )
    assessment = _verify(bundle)
    assert assessment["registered_signer_key_ids_by_role"]["training_execution"] == (
        old_key_id
    )
    assert assessment["local_asserted_event_time_resolution_verified"] is True


def test_training_signer_compromised_at_asserted_event_fails_closed(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _add_training_replacement(
        bundle,
        terminal_type="COMPROMISED",
        effective_at="2027-09-01T00:00:00Z",
        replacement_valid_from="2027-09-02T00:00:00Z",
    )
    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="asserted-event registry replay failed",
    ):
        _verify(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("production_authorized", True),
        ("price_values_opened", True),
    ],
)
def test_d229_success_cannot_smuggle_authority_or_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    bundle = _bundle(tmp_path)
    original = binding.verify_trust_registry_chain_paths

    def forged_registry(**kwargs: object) -> dict[str, object]:
        assessment = original(**kwargs)
        assessment[field] = value
        return assessment

    monkeypatch.setattr(binding, "verify_trust_registry_chain_paths", forged_registry)
    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match=rf"registry replay {field} differs",
    ):
        _verify(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("production_authorized", True),
        ("receipt_materials_opened", True),
    ],
)
def test_d228_success_cannot_smuggle_authority_or_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    bundle = _bundle(tmp_path)
    original = binding.verify_signed_short_tenor_receipt_pair_paths

    def forged_replay(**kwargs: object) -> dict[str, object]:
        assessment = original(**kwargs)
        assessment[field] = value
        return assessment

    monkeypatch.setattr(
        binding,
        "verify_signed_short_tenor_receipt_pair_paths",
        forged_replay,
    )
    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match=rf"signed replay {field} differs",
    ):
        _verify(bundle)


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "d228_content",
        "d229_content",
        "lag",
        "time_authority",
        "production",
    ],
)
def test_contract_rejects_every_authority_or_boundary_mutation(mutation: str) -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if mutation == "status":
        document["status"] = "PASS"
    elif mutation == "d228_content":
        document["selected_boundaries"]["signed_replay_contract_content_id"] = "0" * 64
    elif mutation == "d229_content":
        document["selected_boundaries"]["trust_registry_contract_content_id"] = "0" * 64
    elif mutation == "lag":
        document["binding_profile"]["maximum_registry_publication_lag_days"] = 32
    elif mutation == "time_authority":
        document["authorities"]["independently_trusted_time_verified"] = True
    elif mutation == "production":
        document["authorities"]["production_authorized"] = True

    with pytest.raises(ShortTenorRegisteredSignerBindingError):
        validate_registered_signer_binding_contract(document)


def test_registered_signer_binding_is_absent_from_lt_production() -> None:
    source = inspect.getsource(production_phases)
    assert "eex_short_tenor_registered_signer_binding" not in source
    assert "verify_registered_signer_binding_paths" not in source


def test_source_contains_no_databricks_connector_or_private_key_fixture() -> None:
    source = inspect.getsource(
        __import__(
            "pfc_shaping.validation.eex_short_tenor_registered_signer_binding",
            fromlist=["dummy"],
        )
    )
    assert "databricks.sql" not in source
    assert "DATABRICKS_TOKEN" not in source
    assert "PRIVATE KEY" not in source
    assert "Ed25519PrivateKey" not in source


def test_contract_content_id_changes_for_unchecked_semantic_mutation() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    mutated["not_verified_by_local_replay"].append("invented authority")

    with pytest.raises(
        ShortTenorRegisteredSignerBindingError,
        match="contract content id differs",
    ):
        validate_registered_signer_binding_contract(mutated)
