from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pfc_shaping.validation.eex_entsoe_external_time_batch import (
    CONTRACT_CONTENT_ID,
    CONTRACT_FILE_SHA256,
    ROLES,
    ExternalTimeBatchError,
    build_four_leaf_batch,
    build_leaf_payload,
    canonical_json_bytes,
    load_strict_json,
    rfc9162_inclusion_path,
    rfc9162_leaf_hash,
    rfc9162_merkle_tree_hash,
    validate_external_time_batch_contract,
    validate_local_boundary,
    validate_synthetic_rfc3161_receipt_fixture,
    verify_rfc9162_inclusion,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-ENTSOE-EXTERNAL-TIME-BATCH-CONTRACT-V1.json"
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _fixture(document: dict[str, object] | None = None) -> dict[str, object]:
    selected = _contract() if document is None else document
    fixture = selected["synthetic_fixture"]
    assert isinstance(fixture, dict)
    return fixture


def _batch(document: dict[str, object] | None = None) -> dict[str, object]:
    fixture = _fixture(document)
    return build_four_leaf_batch(
        capture_batch_id=str(fixture["capture_batch_id"]),
        challenge_nonce_sha256=str(fixture["challenge_nonce_sha256"]),
        envelope_sha256_by_role=fixture["envelope_sha256_by_role"],  # type: ignore[arg-type]
    )


def test_exact_contract_bytes_content_and_fixture_validate() -> None:
    payload = CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CONTRACT_FILE_SHA256
    assessment = validate_external_time_batch_contract(_contract())
    assert assessment["contract_content_id"] == CONTRACT_CONTENT_ID
    assert assessment["fixture_root_sha256"] == (
        "496c92c29ab1b4f1ad0f3491cbd4e38f500936973bc6ff8e8ac8ebeb3b0aa85c"
    )
    assert assessment["all_four_inclusion_proofs_verified"] is True
    assert assessment["rfc3161_token_cryptographically_verified"] is False
    assert assessment["independently_trusted_time_verified"] is False


def test_exact_d233_d234_local_parent_boundaries_validate_without_values() -> None:
    assessment = validate_local_boundary(ROOT)
    assert assessment["status"] == "PASS_LOCAL_D233_D234_BOUNDARIES_ONLY"
    assert assessment["databricks_statements"] == 0
    assert assessment["network_calls"] == 0
    assert assessment["h_drive_accesses"] == 0
    assert assessment["real_timestamp_tokens_opened"] == 0


def test_leaf_canonical_json_is_compact_utf8_without_bom_or_newline() -> None:
    fixture = _fixture()
    envelopes = fixture["envelope_sha256_by_role"]
    assert isinstance(envelopes, dict)
    leaf = build_leaf_payload(
        capture_batch_id=str(fixture["capture_batch_id"]),
        challenge_nonce_sha256=str(fixture["challenge_nonce_sha256"]),
        envelope_role=ROLES[0],
        envelope_sha256=str(envelopes[ROLES[0]]),
    )
    payload = canonical_json_bytes(leaf)
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert not payload.endswith(b"\n")
    assert b" " not in payload
    assert payload == json.dumps(
        leaf,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def test_rfc9162_leaf_and_node_domain_separation_matches_fixture() -> None:
    fixture = _fixture()
    batch = _batch()
    proofs = batch["proofs_by_role"]
    assert isinstance(proofs, dict)
    for role in ROLES:
        expected = fixture["expected_leaf_sha256_by_role"]  # type: ignore[index]
        proof = proofs[role]
        assert proof["leaf_sha256"] == expected[role]  # type: ignore[index]


def test_all_four_generated_inclusion_paths_match_frozen_vectors() -> None:
    fixture = _fixture()
    batch = _batch()
    proofs = batch["proofs_by_role"]
    assert isinstance(proofs, dict)
    expected_paths = fixture["expected_inclusion_paths_by_role"]
    assert isinstance(expected_paths, dict)
    for role in ROLES:
        assert proofs[role]["inclusion_path_sha256s"] == expected_paths[role]  # type: ignore[index]
        assert proofs[role]["verified"] is True  # type: ignore[index]


def test_challenge_or_envelope_change_changes_root() -> None:
    fixture = _fixture()
    baseline = _batch()["root_sha256"]
    changed_challenge = build_four_leaf_batch(
        capture_batch_id=str(fixture["capture_batch_id"]),
        challenge_nonce_sha256="0" * 64,
        envelope_sha256_by_role=fixture["envelope_sha256_by_role"],  # type: ignore[arg-type]
    )
    assert changed_challenge["root_sha256"] != baseline

    envelopes = copy.deepcopy(fixture["envelope_sha256_by_role"])
    assert isinstance(envelopes, dict)
    envelopes[ROLES[0]] = "1" * 64
    changed_envelope = build_four_leaf_batch(
        capture_batch_id=str(fixture["capture_batch_id"]),
        challenge_nonce_sha256=str(fixture["challenge_nonce_sha256"]),
        envelope_sha256_by_role=envelopes,
    )
    assert changed_envelope["root_sha256"] != baseline


def test_missing_extra_or_duplicate_role_hash_fails_closed() -> None:
    fixture = _fixture()
    baseline = fixture["envelope_sha256_by_role"]
    assert isinstance(baseline, dict)
    missing = dict(baseline)
    missing.pop(ROLES[0])
    extra = dict(baseline)
    extra["unknown_role"] = "0" * 64
    duplicate = dict(baseline)
    duplicate[ROLES[1]] = duplicate[ROLES[0]]
    for selected in (missing, extra, duplicate):
        with pytest.raises(ExternalTimeBatchError):
            build_four_leaf_batch(
                capture_batch_id=str(fixture["capture_batch_id"]),
                challenge_nonce_sha256=str(fixture["challenge_nonce_sha256"]),
                envelope_sha256_by_role=selected,
            )


@pytest.mark.parametrize(
    ("batch_id", "challenge"),
    [
        ("BAD BATCH", "0" * 64),
        ("d235-valid-batch", "A" * 64),
        ("d235-valid-batch", "0" * 63),
    ],
)
def test_unsafe_batch_or_noncanonical_sha_fails_closed(
    batch_id: str, challenge: str
) -> None:
    fixture = _fixture()
    with pytest.raises(ExternalTimeBatchError):
        build_four_leaf_batch(
            capture_batch_id=batch_id,
            challenge_nonce_sha256=challenge,
            envelope_sha256_by_role=fixture["envelope_sha256_by_role"],  # type: ignore[arg-type]
        )


def test_generic_rfc9162_root_and_path_verify_for_all_four_leaves() -> None:
    fixture = _fixture()
    envelopes = fixture["envelope_sha256_by_role"]
    assert isinstance(envelopes, dict)
    entries = [
        canonical_json_bytes(
            build_leaf_payload(
                capture_batch_id=str(fixture["capture_batch_id"]),
                challenge_nonce_sha256=str(fixture["challenge_nonce_sha256"]),
                envelope_role=role,
                envelope_sha256=str(envelopes[role]),
            )
        )
        for role in ROLES
    ]
    root = rfc9162_merkle_tree_hash(entries)
    assert root.hex() == fixture["expected_root_sha256"]
    for index, entry in enumerate(entries):
        assert verify_rfc9162_inclusion(
            leaf_hash=rfc9162_leaf_hash(entry),
            leaf_index=index,
            tree_size=4,
            inclusion_path=rfc9162_inclusion_path(entries, index),
            expected_root=root,
        )


@pytest.mark.parametrize("tree_size", range(1, 17))
def test_rfc9162_generic_paths_verify_for_balanced_and_unbalanced_trees(
    tree_size: int,
) -> None:
    entries = [f"rfc9162-entry-{index}".encode() for index in range(tree_size)]
    root = rfc9162_merkle_tree_hash(entries)
    for index, entry in enumerate(entries):
        assert verify_rfc9162_inclusion(
            leaf_hash=rfc9162_leaf_hash(entry),
            leaf_index=index,
            tree_size=tree_size,
            inclusion_path=rfc9162_inclusion_path(entries, index),
            expected_root=root,
        )


def test_rfc9162_empty_tree_hash_matches_standard_definition() -> None:
    assert rfc9162_merkle_tree_hash([]) == hashlib.sha256(b"").digest()


def test_tampered_missing_extra_or_wrong_index_proof_fails_closed() -> None:
    batch = _batch()
    proofs = batch["proofs_by_role"]
    assert isinstance(proofs, dict)
    proof = proofs[ROLES[0]]
    assert isinstance(proof, dict)
    leaf = bytes.fromhex(str(proof["leaf_sha256"]))
    path = [bytes.fromhex(value) for value in proof["inclusion_path_sha256s"]]  # type: ignore[union-attr]
    root = bytes.fromhex(str(batch["root_sha256"]))
    assert not verify_rfc9162_inclusion(
        leaf_hash=bytes([leaf[0] ^ 1]) + leaf[1:],
        leaf_index=0,
        tree_size=4,
        inclusion_path=path,
        expected_root=root,
    )
    assert not verify_rfc9162_inclusion(
        leaf_hash=leaf,
        leaf_index=0,
        tree_size=4,
        inclusion_path=path[:-1],
        expected_root=root,
    )
    assert not verify_rfc9162_inclusion(
        leaf_hash=leaf,
        leaf_index=0,
        tree_size=4,
        inclusion_path=[*path, b"\x00" * 32],
        expected_root=root,
    )
    assert not verify_rfc9162_inclusion(
        leaf_hash=leaf,
        leaf_index=1,
        tree_size=4,
        inclusion_path=path,
        expected_root=root,
    )


@pytest.mark.parametrize("tree_size", [0, -1, True])
def test_invalid_tree_size_fails_closed(tree_size: int) -> None:
    with pytest.raises(ExternalTimeBatchError):
        verify_rfc9162_inclusion(
            leaf_hash=b"\x00" * 32,
            leaf_index=0,
            tree_size=tree_size,
            inclusion_path=[],
            expected_root=b"\x00" * 32,
        )


def test_contract_rejects_role_reordering_or_caller_direction_bits() -> None:
    document = _contract()
    roles = document["leaf_contract"]["canonical_role_order"]  # type: ignore[index]
    roles[0], roles[1] = roles[1], roles[0]  # type: ignore[index]
    with pytest.raises(ExternalTimeBatchError, match="leaf roles"):
        validate_external_time_batch_contract(document)

    document = _contract()
    document["rfc9162_merkle_contract"][  # type: ignore[index]
        "caller_supplied_direction_bits_authorized"
    ] = True
    with pytest.raises(ExternalTimeBatchError, match="Merkle contract"):
        validate_external_time_batch_contract(document)


def test_rfc3161_request_imprint_must_equal_exact_batch_root() -> None:
    document = _contract()
    request = document["synthetic_fixture"]["rfc3161_request_template"]  # type: ignore[index]
    request["message_imprint_hashed_message_hex"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ExternalTimeBatchError, match="fixture request imprint"):
        validate_external_time_batch_contract(document)


@pytest.mark.parametrize(
    "field",
    [
        "request_der_present",
        "network_submission_performed",
    ],
)
def test_fixture_cannot_claim_request_der_or_network_submission(field: str) -> None:
    document = _contract()
    request = document["synthetic_fixture"]["rfc3161_request_template"]  # type: ignore[index]
    request[field] = True  # type: ignore[index]
    with pytest.raises(ExternalTimeBatchError, match="fixture"):
        validate_external_time_batch_contract(document)


@pytest.mark.parametrize(
    "field",
    [
        "timestamp_token_der_present",
        "cms_signature_verified",
        "message_imprint_verified",
        "nonce_verified",
        "certificate_path_verified",
        "independently_trusted_time_verified",
        "actual_signature_generation_time_verified",
        "historical_point_in_time_retroactively_proven",
    ],
)
def test_synthetic_receipt_cannot_claim_real_validation(field: str) -> None:
    document = _contract()
    receipt = document["synthetic_fixture"][  # type: ignore[index]
        "rfc3161_validation_receipt_fixture"
    ]
    receipt[field] = True  # type: ignore[index]
    with pytest.raises(ExternalTimeBatchError, match="synthetic receipt"):
        validate_external_time_batch_contract(document)


def test_synthetic_receipt_root_binding_is_exact() -> None:
    receipt = _fixture()["rfc3161_validation_receipt_fixture"]
    assert isinstance(receipt, dict)
    with pytest.raises(ExternalTimeBatchError, match="receipt batch root"):
        validate_synthetic_rfc3161_receipt_fixture(
            receipt,
            expected_root_sha256="0" * 64,
        )


def test_future_real_receipt_schema_cannot_drop_crypto_or_revocation_fields() -> None:
    document = _contract()
    fields = document["future_real_rfc3161_validation_receipt_contract"][  # type: ignore[index]
        "required_fields"
    ]
    for field in (
        "cms_signature_verified",
        "exclusive_critical_timestamping_eku_verified",
        "tsa_not_revoked_at_gen_time_verified",
        "validator_artifact_sha256",
    ):
        mutated = copy.deepcopy(document)
        selected = mutated["future_real_rfc3161_validation_receipt_contract"][  # type: ignore[index]
            "required_fields"
        ]
        selected.remove(field)  # type: ignore[union-attr]
        with pytest.raises(ExternalTimeBatchError, match="future receipt fields"):
            validate_external_time_batch_contract(mutated)
    assert len(fields) == 46  # type: ignore[arg-type]


def test_access_counter_bool_or_nonzero_fails_closed() -> None:
    for value in (1, False):
        document = _contract()
        document["access_counters"]["h_drive_accesses"] = value  # type: ignore[index]
        with pytest.raises(ExternalTimeBatchError, match="integer zero"):
            validate_external_time_batch_contract(document)


def test_authority_escalation_fails_closed() -> None:
    document = _contract()
    authorities = document["authorities"]
    assert isinstance(authorities, dict)
    for field, value in authorities.items():
        if value is not False:
            continue
        mutated = copy.deepcopy(document)
        mutated["authorities"][field] = True  # type: ignore[index]
        with pytest.raises(ExternalTimeBatchError, match="authority"):
            validate_external_time_batch_contract(mutated)


def test_unknown_contract_or_receipt_field_fails_closed() -> None:
    document = _contract()
    document["self_declared_time"] = False
    with pytest.raises(ExternalTimeBatchError, match="contract fields differ"):
        validate_external_time_batch_contract(document)

    receipt = copy.deepcopy(_fixture()["rfc3161_validation_receipt_fixture"])
    assert isinstance(receipt, dict)
    receipt["extra"] = False
    with pytest.raises(ExternalTimeBatchError, match="receipt fields differ"):
        validate_synthetic_rfc3161_receipt_fixture(
            receipt,
            expected_root_sha256=str(_fixture()["expected_root_sha256"]),
        )


def test_strict_json_rejects_duplicates_nonfinite_and_bom() -> None:
    with pytest.raises(ExternalTimeBatchError, match="duplicate JSON field"):
        load_strict_json(b'{"x":1,"x":2}')
    with pytest.raises(ExternalTimeBatchError, match="non-finite JSON token"):
        load_strict_json(b'{"x":NaN}')
    with pytest.raises(ExternalTimeBatchError, match="invalid strict UTF-8 JSON"):
        load_strict_json(b"\xef\xbb\xbf{}")
