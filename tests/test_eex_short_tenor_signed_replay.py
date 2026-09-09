from __future__ import annotations

import base64
import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.pipeline import production_phases
from pfc_shaping.validation.eex_short_tenor_receipts import (
    COMBINATION_PROOF_CONTENT_ID,
    COMBINATION_PROOF_MANIFEST_SHA256,
    CONTRAST_BUNDLE_CONTENT_ID,
    CONTRAST_BUNDLE_MANIFEST_SHA256,
    EVIDENCE_SELECTION_CONTENT_ID,
    EVIDENCE_SELECTION_DOCUMENT_SHA256,
    EVIDENCE_SELECTION_PROOF_CONTENT_ID,
    EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256,
    POLICY_CONTENT_ID,
    POLICY_DOCUMENT_SHA256,
    RECEIPT_STATUS,
    REQUIRED_MATERIAL_ROLES,
    SELECTION_RECEIPT_SCHEMA,
    SELECTION_RULE,
    SHAPE_PROJECTION_CONTENT_ID,
    SUCCESSOR_CORE_ID,
    TRAINING_RECEIPT_SCHEMA,
    with_receipt_id,
)
from pfc_shaping.validation.eex_short_tenor_receipts import (
    CONTRACT_CONTENT_ID as RECEIPT_CONTRACT_CONTENT_ID,
)
from pfc_shaping.validation.eex_short_tenor_signed_replay import (
    DSSE_PAYLOAD_TYPE,
    SHORT_TENOR_BUILD_TYPE,
    SIGNED_REPLAY_CONTRACT_CONTENT_ID,
    SIGNED_REPLAY_CONTRACT_STATUS,
    SLSA_PREDICATE_TYPE,
    STATEMENT_TYPE,
    ShortTenorSignedReplayError,
    canonical_json_bytes,
    dsse_pae,
    public_key_id,
    trusted_time_observation_statement,
    validate_signed_replay_contract,
    verify_signed_short_tenor_receipt_pair_paths,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-CH-SHORT-TENOR-SIGNED-REPLAY-CONTRACT-V1.json"
)
TIME_SOURCE_ID = "synthetic-reference-time-source-v1"


def _h(character: str) -> str:
    return character * 64


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(value: object) -> str:
    return _sha(canonical_json_bytes(value))


def _access_log() -> dict[str, bool]:
    return {
        "outer_target_truth_opened": False,
        "ompex_opened": False,
        "afry_numeric_values_accessed": False,
        "t057_accessed": False,
        "future_holdout_consumed": False,
        "post_origin_data_accessed": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "receipt_authenticated": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
    }


def _authority(
    *,
    governance_key_id: str,
    execution_key_id: str,
    trusted_time_key_id: str,
) -> dict[str, object]:
    return {
        "governance_key_id": governance_key_id,
        "execution_key_id": execution_key_id,
        "trusted_time_key_id": trusted_time_key_id,
        "source_acquisition_key_ids": [_h("4"), _h("5")],
        "source_trusted_time_key_ids": [_h("6"), _h("7")],
        "cryptographic_authentication_performed": False,
        "independent_authorities_proven": False,
    }


def _materials() -> list[dict[str, str]]:
    content_ids = {
        "EEX_PIT_CATALOG": _h("1"),
        "ENTSOE_PIT_CATALOG": _h("2"),
        "D225_SELECTION_DOCUMENT": EVIDENCE_SELECTION_CONTENT_ID,
        "D225_SELECTION_PROOF": EVIDENCE_SELECTION_PROOF_CONTENT_ID,
        "D220_CONTRAST_BUNDLE": CONTRAST_BUNDLE_CONTENT_ID,
        "D225_COMBINATION_PROOF": COMBINATION_PROOF_CONTENT_ID,
        "D219_PROJECTION_PROOF": SHAPE_PROJECTION_CONTENT_ID,
        "D225_COMBINATION_POLICY": POLICY_CONTENT_ID,
        "ORIGIN_TARGET_MASK": _h("6"),
        "CANDIDATE_GRID": _h("7"),
        "FOLD_INVENTORY": _h("8"),
        "IMPLEMENTATION_CLOSURE": _h("9"),
        "RUNTIME_LOCK": _h("a"),
    }
    digests = {
        role: _h(character)
        for role, character in zip(
            REQUIRED_MATERIAL_ROLES,
            "123456789abcd",
            strict=True,
        )
    }
    digests.update(
        {
            "D225_SELECTION_DOCUMENT": EVIDENCE_SELECTION_DOCUMENT_SHA256,
            "D225_SELECTION_PROOF": EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256,
            "D220_CONTRAST_BUNDLE": CONTRAST_BUNDLE_MANIFEST_SHA256,
            "D225_COMBINATION_PROOF": COMBINATION_PROOF_MANIFEST_SHA256,
            "D225_COMBINATION_POLICY": POLICY_DOCUMENT_SHA256,
            "CANDIDATE_GRID": _h("b"),
            "FOLD_INVENTORY": _h("c"),
        }
    )
    return [
        {
            "role": role,
            "name": role.lower(),
            "sha256": digests[role],
            "content_id": content_ids[role],
            "max_available_at_utc": "2027-07-31T00:00:00Z",
            "trusted_time_receipt_sha256": _h("f"),
        }
        for role in REQUIRED_MATERIAL_ROLES
    ]


def _inner_folds() -> list[dict[str, object]]:
    return [
        {
            "fold_id": "F01",
            "fold_as_of_utc": "2026-12-01T00:00:00Z",
            "training_origin_ids": ["O01"],
            "validation_origin_id": "V01",
            "training_data_max_available_at_utc": "2026-11-30T23:00:00Z",
            "validation_target_delivery_start_utc": "2027-01-01T00:00:00Z",
            "validation_target_delivery_end_utc": "2027-02-01T00:00:00Z",
            "validation_truth_available_at_utc": "2027-02-02T00:00:00Z",
            "fold_inventory_entry_sha256": _h("a"),
            "eex_pit_cutoff_receipt_sha256": _h("b"),
            "entsoe_pit_cutoff_receipt_sha256": _h("c"),
            "fold_input_snapshot_sha256": _h("d"),
        },
        {
            "fold_id": "F02",
            "fold_as_of_utc": "2027-05-01T00:00:00Z",
            "training_origin_ids": ["O01", "V01"],
            "validation_origin_id": "V02",
            "training_data_max_available_at_utc": "2027-04-30T23:00:00Z",
            "validation_target_delivery_start_utc": "2027-06-01T00:00:00Z",
            "validation_target_delivery_end_utc": "2027-07-01T00:00:00Z",
            "validation_truth_available_at_utc": "2027-07-02T00:00:00Z",
            "fold_inventory_entry_sha256": _h("b"),
            "eex_pit_cutoff_receipt_sha256": _h("c"),
            "entsoe_pit_cutoff_receipt_sha256": _h("d"),
            "fold_input_snapshot_sha256": _h("e"),
        },
    ]


def _run_details(
    *,
    selection: bool,
    access: Mapping[str, bool],
    trusted_time_payload_sha256: str,
) -> dict[str, object]:
    byproducts = {
        "access_log": _json_hash(access),
        "environment_manifest": _h("8"),
        "selection_log" if selection else "training_log": _h("9"),
    }
    if not selection:
        byproducts["inner_loss_commitment"] = _h("e")
    return {
        "builder_id": "https://fmv.example.invalid/builders/short-tenor-fixture-v1",
        "invocation_id": (
            "selection-fixture-001" if selection else "training-fixture-001"
        ),
        "started_at_utc": (
            "2027-09-01T00:00:00Z" if selection else "2027-08-01T00:00:00Z"
        ),
        "finished_at_utc": (
            "2027-10-01T00:00:00Z" if selection else "2027-09-01T00:00:00Z"
        ),
        "execution_attestation_sha256": _h("b" if selection else "a"),
        "trusted_time_receipt_sha256": trusted_time_payload_sha256,
        "byproduct_sha256s": byproducts,
    }


def _training_receipt(
    *,
    governance_key_id: str,
    execution_key_id: str,
    trusted_time_key_id: str,
    trusted_time_payload_sha256: str,
) -> dict[str, object]:
    access = _access_log()
    unsigned: dict[str, object] = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "receipt_role": "TRAINING",
        "status": RECEIPT_STATUS,
        "market": "CH",
        "currency_unit": "EUR/MWh",
        "campaign_type": "SELECTION",
        "outer_origin_id": "OUTER-2028-01",
        "outer_origin_as_of_utc": "2028-01-01T00:00:00Z",
        "outer_origin_entry_sha256": _h("0"),
        "plan_frozen_at_utc": "2026-01-01T00:00:00Z",
        "policy_content_id": POLICY_CONTENT_ID,
        "successor_core_id": SUCCESSOR_CORE_ID,
        "candidate_grid_sha256": _h("b"),
        "fold_inventory_sha256": _h("c"),
        "inner_loss_commitment_sha256": _h("e"),
        "materials": _materials(),
        "candidate_config_sha256_allowlist": [_h("d"), _h("e")],
        "inner_folds": _inner_folds(),
        "subject": {
            "role": "TRAINED_CANDIDATE_SET",
            "name": "trained-candidate-set.json",
            "sha256": _h("f"),
        },
        "run_details": _run_details(
            selection=False,
            access=access,
            trusted_time_payload_sha256=trusted_time_payload_sha256,
        ),
        "access_log": access,
        "authority": _authority(
            governance_key_id=governance_key_id,
            execution_key_id=execution_key_id,
            trusted_time_key_id=trusted_time_key_id,
        ),
        "claims": _claims(),
    }
    return with_receipt_id(unsigned)


def _selection_receipt(
    training: Mapping[str, object],
    *,
    governance_key_id: str,
    execution_key_id: str,
    trusted_time_key_id: str,
    trusted_time_payload_sha256: str,
) -> dict[str, object]:
    access = _access_log()
    unsigned: dict[str, object] = {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "receipt_role": "SELECTION",
        "status": RECEIPT_STATUS,
        "market": "CH",
        "currency_unit": "EUR/MWh",
        "campaign_type": "SELECTION",
        "outer_origin_id": training["outer_origin_id"],
        "outer_origin_as_of_utc": training["outer_origin_as_of_utc"],
        "outer_origin_entry_sha256": training["outer_origin_entry_sha256"],
        "policy_content_id": POLICY_CONTENT_ID,
        "successor_core_id": SUCCESSOR_CORE_ID,
        "training_receipt_id": training["receipt_id"],
        "training_receipt_sha256": _json_hash(training),
        "candidate_grid_sha256": training["candidate_grid_sha256"],
        "fold_inventory_sha256": training["fold_inventory_sha256"],
        "selected_candidate_config_sha256": _h("d"),
        "selection_rule": SELECTION_RULE,
        "inner_loss_commitment_sha256": training[
            "inner_loss_commitment_sha256"
        ],
        "subject": {
            "role": "SELECTED_SHORT_TENOR_CANDIDATE",
            "name": "selected-short-tenor-candidate.json",
            "sha256": _h("e"),
            "candidate_config_sha256": _h("d"),
        },
        "run_details": _run_details(
            selection=True,
            access=access,
            trusted_time_payload_sha256=trusted_time_payload_sha256,
        ),
        "access_log": access,
        "authority": _authority(
            governance_key_id=governance_key_id,
            execution_key_id=execution_key_id,
            trusted_time_key_id=trusted_time_key_id,
        ),
        "claims": _claims(),
    }
    return with_receipt_id(unsigned)


def _resolved_dependencies(receipt: Mapping[str, object]) -> list[dict[str, object]]:
    materials = receipt.get("materials")
    if isinstance(materials, list):
        return [
            {
                "name": material["name"],
                "digest": {"sha256": material["sha256"]},
            }
            for material in materials
        ]
    return [
        {
            "name": "candidate-grid",
            "digest": {"sha256": receipt["candidate_grid_sha256"]},
        },
        {
            "name": "fold-inventory",
            "digest": {"sha256": receipt["fold_inventory_sha256"]},
        },
        {
            "name": "inner-loss-commitment",
            "digest": {"sha256": receipt["inner_loss_commitment_sha256"]},
        },
        {
            "name": "selection-policy",
            "digest": {"sha256": receipt["policy_content_id"]},
        },
        {
            "name": "successor-core",
            "digest": {"sha256": receipt["successor_core_id"]},
        },
        {
            "name": "training-receipt",
            "digest": {"sha256": receipt["training_receipt_sha256"]},
        },
    ]


def _slsa_statement(receipt: Mapping[str, object]) -> dict[str, object]:
    run = receipt["run_details"]
    subject = receipt["subject"]
    byproducts = [
        {"name": name, "digest": {"sha256": run["byproduct_sha256s"][name]}}
        for name in sorted(run["byproduct_sha256s"])
    ]
    byproducts.extend(
        [
            {
                "name": "execution-attestation",
                "digest": {"sha256": run["execution_attestation_sha256"]},
            },
            {
                "name": "trusted-time-receipt",
                "digest": {"sha256": run["trusted_time_receipt_sha256"]},
            },
        ]
    )
    return {
        "_type": STATEMENT_TYPE,
        "subject": [
            {
                "name": subject["name"],
                "digest": {"sha256": subject["sha256"]},
            }
        ],
        "predicateType": SLSA_PREDICATE_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": SHORT_TENOR_BUILD_TYPE,
                "externalParameters": {
                    "receipt_contract_content_id": RECEIPT_CONTRACT_CONTENT_ID,
                    "receipt_role": receipt["receipt_role"],
                    "receipt_schema": receipt["schema_version"],
                    "receipt_sha256": _json_hash(receipt),
                    "receipt": dict(receipt),
                },
                "internalParameters": {},
                "resolvedDependencies": _resolved_dependencies(receipt),
            },
            "runDetails": {
                "builder": {"id": run["builder_id"]},
                "metadata": {
                    "invocationId": run["invocation_id"],
                    "startedOn": run["started_at_utc"],
                    "finishedOn": run["finished_at_utc"],
                },
                "byproducts": byproducts,
            },
        },
    }


def _signed_envelope(
    statement: Mapping[str, object],
    *,
    key: Ed25519PrivateKey,
) -> bytes:
    payload = canonical_json_bytes(statement)
    signature = key.sign(dsse_pae(DSSE_PAYLOAD_TYPE, payload))
    return canonical_json_bytes(
        {
            "payloadType": DSSE_PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode("ascii"),
            "signatures": [
                {
                    "keyid": public_key_id(key.public_key()),
                    "sig": base64.b64encode(signature).decode("ascii"),
                }
            ],
        }
    )


def _public_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _bundle(
    tmp_path: Path,
    *,
    training_observed_at: str = "2027-09-01T00:00:00Z",
    selection_observed_at: str = "2027-10-01T00:00:01Z",
    time_source_id: str = TIME_SOURCE_ID,
    training_key: Ed25519PrivateKey | None = None,
    governance_key: Ed25519PrivateKey | None = None,
    time_key: Ed25519PrivateKey | None = None,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = {
        "training_execution": training_key or Ed25519PrivateKey.generate(),
        "selection_governance": governance_key or Ed25519PrivateKey.generate(),
        "trusted_time": time_key or Ed25519PrivateKey.generate(),
    }
    key_ids = {name: public_key_id(key.public_key()) for name, key in keys.items()}
    training_time_statement = trusted_time_observation_statement(
        receipt_role="TRAINING",
        execution_attestation_sha256=_h("a"),
        observed_at_utc=training_observed_at,
        time_source_id=time_source_id,
    )
    selection_time_statement = trusted_time_observation_statement(
        receipt_role="SELECTION",
        execution_attestation_sha256=_h("b"),
        observed_at_utc=selection_observed_at,
        time_source_id=time_source_id,
    )
    training = _training_receipt(
        governance_key_id=key_ids["selection_governance"],
        execution_key_id=key_ids["training_execution"],
        trusted_time_key_id=key_ids["trusted_time"],
        trusted_time_payload_sha256=_json_hash(training_time_statement),
    )
    selection = _selection_receipt(
        training,
        governance_key_id=key_ids["selection_governance"],
        execution_key_id=key_ids["training_execution"],
        trusted_time_key_id=key_ids["trusted_time"],
        trusted_time_payload_sha256=_json_hash(selection_time_statement),
    )
    payloads = {
        "training_envelope": _signed_envelope(
            _slsa_statement(training),
            key=keys["training_execution"],
        ),
        "selection_envelope": _signed_envelope(
            _slsa_statement(selection),
            key=keys["selection_governance"],
        ),
        "training_time_envelope": _signed_envelope(
            training_time_statement,
            key=keys["trusted_time"],
        ),
        "selection_time_envelope": _signed_envelope(
            selection_time_statement,
            key=keys["trusted_time"],
        ),
    }
    key_payloads = {name: _public_key_bytes(key) for name, key in keys.items()}
    paths = {
        name: (tmp_path / f"{name}.json").resolve() for name in payloads
    }
    key_paths = {
        name: (tmp_path / f"{name}.pem").resolve() for name in key_payloads
    }
    for name, payload in payloads.items():
        paths[name].write_bytes(payload)
    for name, payload in key_payloads.items():
        key_paths[name].write_bytes(payload)
    return {
        "keys": keys,
        "key_ids": key_ids,
        "training": training,
        "selection": selection,
        "time_statements": {
            "training": training_time_statement,
            "selection": selection_time_statement,
        },
        "payloads": payloads,
        "key_payloads": key_payloads,
        "paths": paths,
        "key_paths": key_paths,
        "time_source_id": time_source_id,
    }


def _kwargs(bundle: Mapping[str, object]) -> dict[str, object]:
    paths = bundle["paths"]
    key_paths = bundle["key_paths"]
    payloads = bundle["payloads"]
    key_payloads = bundle["key_payloads"]
    return {
        "training_envelope_path": paths["training_envelope"],
        "selection_envelope_path": paths["selection_envelope"],
        "training_execution_public_key_path": key_paths["training_execution"],
        "selection_governance_public_key_path": key_paths["selection_governance"],
        "trusted_time_public_key_path": key_paths["trusted_time"],
        "training_time_envelope_path": paths["training_time_envelope"],
        "selection_time_envelope_path": paths["selection_time_envelope"],
        "expected_training_envelope_sha256": _sha(payloads["training_envelope"]),
        "expected_selection_envelope_sha256": _sha(payloads["selection_envelope"]),
        "expected_training_execution_public_key_sha256": _sha(
            key_payloads["training_execution"]
        ),
        "expected_selection_governance_public_key_sha256": _sha(
            key_payloads["selection_governance"]
        ),
        "expected_trusted_time_public_key_sha256": _sha(
            key_payloads["trusted_time"]
        ),
        "expected_training_time_envelope_sha256": _sha(
            payloads["training_time_envelope"]
        ),
        "expected_selection_time_envelope_sha256": _sha(
            payloads["selection_time_envelope"]
        ),
        "expected_trusted_time_source_id": bundle["time_source_id"],
    }


def _verify(bundle: Mapping[str, object]) -> dict[str, object]:
    return verify_signed_short_tenor_receipt_pair_paths(**_kwargs(bundle))


def _replace_payload(bundle: dict[str, object], name: str, payload: bytes) -> None:
    bundle["paths"][name].write_bytes(payload)
    bundle["payloads"][name] = payload


def _replace_receipt(
    bundle: dict[str, object],
    *,
    role: str,
    receipt: Mapping[str, object],
) -> None:
    name = "training_envelope" if role == "TRAINING" else "selection_envelope"
    key_name = (
        "training_execution" if role == "TRAINING" else "selection_governance"
    )
    payload = _signed_envelope(_slsa_statement(receipt), key=bundle["keys"][key_name])
    _replace_payload(bundle, name, payload)


def test_contract_is_exact_slsa_and_non_authoritative() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = validate_signed_replay_contract(contract)

    assert result["status"] == SIGNED_REPLAY_CONTRACT_STATUS
    assert result["contract_content_id"] == SIGNED_REPLAY_CONTRACT_CONTENT_ID
    assert contract["in_toto_slsa_profile"]["predicate_type"] == (
        SLSA_PREDICATE_TYPE
    )
    assert result["external_authority_verified"] is False
    assert result["independently_trusted_time_verified"] is False
    assert result["training_authorized"] is False
    assert result["selection_authorized"] is False
    assert result["production_authorized"] is False


def test_dsse_pae_and_spki_key_id_are_exact() -> None:
    assert dsse_pae(DSSE_PAYLOAD_TYPE, b"{}") == (
        b"DSSEv1 28 application/vnd.in-toto+json 2 {}"
    )
    key = Ed25519PrivateKey.generate().public_key()
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert public_key_id(key) == _sha(der)


def test_valid_replay_verifies_local_integrity_without_authority(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    result = _verify(bundle)
    after = sorted(path.name for path in tmp_path.iterdir())

    assert result["status"] == SIGNED_REPLAY_CONTRACT_STATUS
    assert result["local_dsse_signature_integrity_verified"] is True
    assert result["in_toto_slsa_bindings_verified"] is True
    assert result["d227_structural_replay_verified"] is True
    assert result["training_selection_link_verified"] is True
    assert result["execution_time_observation_links_verified"] is True
    assert result["execution_attestations_distinct"] is True
    assert result["role_keys_pairwise_distinct"] is True
    assert result["receipt_materials_opened"] is False
    assert result["source_trusted_time_receipts_opened"] is False
    assert result["price_values_opened"] is False
    assert result["external_key_owner_identity_verified"] is False
    assert result["independently_trusted_time_verified"] is False
    assert result["point_in_time_availability_proven"] is False
    assert result["training_authorized"] is False
    assert result["selection_authorized"] is False
    assert result["production_authorized"] is False
    assert result["databricks_request_count"] == 0
    assert result["warehouse_start_count"] == 0
    assert result["network_call_count"] == 0
    assert result["remote_write_count"] == 0
    assert before == after


def test_replay_rejects_caller_envelope_or_key_hash_drift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = _kwargs(bundle)
    kwargs["expected_training_envelope_sha256"] = _h("0")
    with pytest.raises(ShortTenorSignedReplayError, match="caller-held hash"):
        verify_signed_short_tenor_receipt_pair_paths(**kwargs)

    kwargs = _kwargs(bundle)
    kwargs["expected_trusted_time_public_key_sha256"] = _h("0")
    with pytest.raises(ShortTenorSignedReplayError, match="caller-held hash"):
        verify_signed_short_tenor_receipt_pair_paths(**kwargs)


def test_replay_rejects_invalid_signature_before_statement_parse(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    envelope = json.loads(bundle["payloads"]["training_envelope"])
    envelope["payload"] = base64.b64encode(b"not-json").decode("ascii")
    _replace_payload(bundle, "training_envelope", canonical_json_bytes(envelope))

    with pytest.raises(ShortTenorSignedReplayError, match="signature is invalid"):
        _verify(bundle)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_signature", "exactly one signature"),
        ("keyid_spoof", "key id differs"),
        ("unknown_field", "keys differ"),
        ("wrong_payload_type", "payload type differs"),
        ("noncanonical", "canonical JSON"),
        ("duplicate_key", "strict UTF-8 JSON"),
    ],
)
def test_replay_rejects_ambiguous_or_weakened_dsse(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle["payloads"]["training_envelope"]
    envelope = json.loads(original)
    if mutation == "extra_signature":
        envelope["signatures"].append(dict(envelope["signatures"][0]))
        payload = canonical_json_bytes(envelope)
    elif mutation == "keyid_spoof":
        envelope["signatures"][0]["keyid"] = _h("0")
        payload = canonical_json_bytes(envelope)
    elif mutation == "unknown_field":
        envelope["unexpected"] = False
        payload = canonical_json_bytes(envelope)
    elif mutation == "wrong_payload_type":
        envelope["payloadType"] = "application/json"
        payload = canonical_json_bytes(envelope)
    elif mutation == "noncanonical":
        payload = json.dumps(envelope, indent=2).encode("utf-8")
    else:
        payload = original.replace(
            b'"payloadType":"application/vnd.in-toto+json"',
            b'"payloadType":"application/vnd.in-toto+json",'
            b'"payloadType":"application/vnd.in-toto+json"',
        )
    _replace_payload(bundle, "training_envelope", payload)

    with pytest.raises(ShortTenorSignedReplayError, match=message):
        _verify(bundle)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("predicate_type", "SLSA predicate type differs"),
        ("subject", "subject digest differs"),
        ("dependency", "resolved dependencies differs"),
        ("metadata", "run metadata differs"),
        ("unknown_predicate", "predicate keys differ"),
    ],
)
def test_replay_rejects_slsa_binding_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    bundle = _bundle(tmp_path)
    statement = _slsa_statement(bundle["training"])
    if mutation == "predicate_type":
        statement["predicateType"] = "https://attacker.invalid/predicate"
    elif mutation == "subject":
        statement["subject"][0]["digest"]["sha256"] = _h("0")
    elif mutation == "dependency":
        statement["predicate"]["buildDefinition"]["resolvedDependencies"][0][
            "digest"
        ]["sha256"] = _h("0")
    elif mutation == "metadata":
        statement["predicate"]["runDetails"]["metadata"]["invocationId"] = (
            "substituted"
        )
    else:
        statement["predicate"]["unexpected"] = False
    payload = _signed_envelope(statement, key=bundle["keys"]["training_execution"])
    _replace_payload(bundle, "training_envelope", payload)

    with pytest.raises(ShortTenorSignedReplayError, match=message):
        _verify(bundle)


def test_replay_rejects_receipt_authority_or_training_link_drift(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    key_ids = bundle["key_ids"]
    time_statements = bundle["time_statements"]
    training = _training_receipt(
        governance_key_id=key_ids["selection_governance"],
        execution_key_id=_h("0"),
        trusted_time_key_id=key_ids["trusted_time"],
        trusted_time_payload_sha256=_json_hash(time_statements["training"]),
    )
    selection = _selection_receipt(
        training,
        governance_key_id=key_ids["selection_governance"],
        execution_key_id=_h("0"),
        trusted_time_key_id=key_ids["trusted_time"],
        trusted_time_payload_sha256=_json_hash(time_statements["selection"]),
    )
    _replace_receipt(bundle, role="TRAINING", receipt=training)
    _replace_receipt(bundle, role="SELECTION", receipt=selection)
    with pytest.raises(ShortTenorSignedReplayError, match="execution key binding"):
        _verify(bundle)

    bundle = _bundle(tmp_path / "link")
    selection = dict(bundle["selection"])
    selection["training_receipt_id"] = _h("0")
    selection = with_receipt_id(selection)
    _replace_receipt(bundle, role="SELECTION", receipt=selection)
    with pytest.raises(ShortTenorSignedReplayError, match="structural replay"):
        _verify(bundle)


def test_replay_rejects_outcome_access_even_when_validly_signed(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    training = dict(bundle["training"])
    training["access_log"] = dict(training["access_log"])
    training["access_log"]["ompex_opened"] = True
    training["run_details"] = dict(training["run_details"])
    training["run_details"]["byproduct_sha256s"] = dict(
        training["run_details"]["byproduct_sha256s"]
    )
    training["run_details"]["byproduct_sha256s"]["access_log"] = _json_hash(
        training["access_log"]
    )
    training = with_receipt_id(training)
    _replace_receipt(bundle, role="TRAINING", receipt=training)

    with pytest.raises(ShortTenorSignedReplayError, match="structural replay"):
        _verify(bundle)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"training_observed_at": "2027-09-01T00:00:01Z"},
            "later than selection start",
        ),
        (
            {"selection_observed_at": "2027-09-30T23:59:59Z"},
            "precedes run finish",
        ),
        (
            {"selection_observed_at": "2028-01-01T00:00:01Z"},
            "later than outer origin",
        ),
    ],
)
def test_replay_rejects_reference_time_travel(
    tmp_path: Path,
    kwargs: dict[str, str],
    message: str,
) -> None:
    bundle = _bundle(tmp_path, **kwargs)
    with pytest.raises(ShortTenorSignedReplayError, match=message):
        _verify(bundle)


def test_replay_rejects_time_source_payload_or_subject_drift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = _kwargs(bundle)
    kwargs["expected_trusted_time_source_id"] = "different-source"
    with pytest.raises(ShortTenorSignedReplayError, match="time source differs"):
        verify_signed_short_tenor_receipt_pair_paths(**kwargs)

    bundle = _bundle(tmp_path / "payload")
    statement = trusted_time_observation_statement(
        receipt_role="SELECTION",
        execution_attestation_sha256=_h("b"),
        observed_at_utc="2027-10-01T00:00:02Z",
        time_source_id=TIME_SOURCE_ID,
    )
    envelope = _signed_envelope(statement, key=bundle["keys"]["trusted_time"])
    _replace_payload(bundle, "selection_time_envelope", envelope)
    with pytest.raises(ShortTenorSignedReplayError, match="payload binding differs"):
        _verify(bundle)

    bundle = _bundle(tmp_path / "subject")
    statement = dict(bundle["time_statements"]["training"])
    statement["subject"] = [
        {
            "name": "training-execution-attestation",
            "digest": {"sha256": _h("0")},
        }
    ]
    envelope = _signed_envelope(statement, key=bundle["keys"]["trusted_time"])
    _replace_payload(bundle, "training_time_envelope", envelope)
    with pytest.raises(ShortTenorSignedReplayError, match="subject digest differs"):
        _verify(bundle)


def test_replay_rejects_reused_execution_attestation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    selection_time_statement = trusted_time_observation_statement(
        receipt_role="SELECTION",
        execution_attestation_sha256=_h("a"),
        observed_at_utc="2027-10-01T00:00:01Z",
        time_source_id=TIME_SOURCE_ID,
    )
    selection = dict(bundle["selection"])
    selection["run_details"] = dict(selection["run_details"])
    selection["run_details"]["execution_attestation_sha256"] = _h("a")
    selection["run_details"]["trusted_time_receipt_sha256"] = _json_hash(
        selection_time_statement
    )
    selection = with_receipt_id(selection)
    _replace_receipt(bundle, role="SELECTION", receipt=selection)
    envelope = _signed_envelope(
        selection_time_statement,
        key=bundle["keys"]["trusted_time"],
    )
    _replace_payload(bundle, "selection_time_envelope", envelope)

    with pytest.raises(ShortTenorSignedReplayError, match="must be distinct"):
        _verify(bundle)


def test_replay_rejects_reused_role_key_even_at_distinct_paths(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    reused = bundle["key_payloads"]["training_execution"]
    bundle["key_paths"]["trusted_time"].write_bytes(reused)
    bundle["key_payloads"]["trusted_time"] = reused

    with pytest.raises(ShortTenorSignedReplayError, match="must be distinct"):
        _verify(bundle)


def test_replay_rejects_relative_or_aliased_paths(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = _kwargs(bundle)
    kwargs["training_envelope_path"] = Path("relative.json")
    with pytest.raises(ShortTenorSignedReplayError, match="absolute and link-free"):
        verify_signed_short_tenor_receipt_pair_paths(**kwargs)

    kwargs = _kwargs(bundle)
    kwargs["selection_envelope_path"] = kwargs["training_envelope_path"]
    with pytest.raises(ShortTenorSignedReplayError, match="pairwise distinct"):
        verify_signed_short_tenor_receipt_pair_paths(**kwargs)


def test_contract_rejects_any_semantic_mutation() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["authorities"]["production_authorized"] = True
    with pytest.raises(ShortTenorSignedReplayError, match="content id differs"):
        validate_signed_replay_contract(contract)


def test_signed_replay_is_absent_from_lt_production() -> None:
    production_source = inspect.getsource(production_phases)
    verifier_source = inspect.getsource(
        verify_signed_short_tenor_receipt_pair_paths
    ).lower()

    assert "eex_short_tenor_signed_replay" not in production_source
    assert "verify_signed_short_tenor_receipt_pair_paths" not in production_source
    assert "databricks.sql" not in verifier_source
    assert "databricks.connect" not in verifier_source
    assert "read_excel" not in verifier_source
    assert "pyarrow" not in verifier_source
