"""Verify the D235 RFC 9162 four-envelope batch without obtaining a timestamp."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

CONTRACT_SCHEMA = "fmv-eex-entsoe-external-time-batch-contract-v1"
CONTRACT_STATUS = (
    "PASS_LOCAL_RFC9162_MERKLE_AND_RFC3161_SCHEMA_ONLY_NO_EXTERNAL_TIME"
)
CONTRACT_FILE_SHA256 = (
    "e9da2a73dd1bd62b5357e1f507bff3ef8e48918806c918bd4fdb1f609a71d138"
)
CONTRACT_CONTENT_ID = (
    "8f0f13315c382da7163a1f451752a6ddf84857a50133d89d3a3a31f340b264d0"
)
D233_CONTRACT_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "EEX-ENTSOE-GOVERNED-ACQUISITION-PACKAGE-CONTRACT-V1.json"
)
D233_CONTRACT_FILE_SHA256 = (
    "97edaf93c639953365d20d7539380a706f0f8fdbec9476af94262fb1d4b2204d"
)
D233_CONTRACT_CONTENT_ID = (
    "4f6e644b15593e8d4ae2ca196e1d52a002d4255bdf42e7a137982edb854f0fa0"
)
D234_CONTRACT_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "ENTSOE-PHYSICAL-MAPPING-COMPILER-CONTRACT-V1.json"
)
D234_CONTRACT_FILE_SHA256 = (
    "f33bc696d8e7a535f7700b40c13dc90d73e2526073400edcd2588e57432a9b3a"
)
D234_CONTRACT_CONTENT_ID = (
    "5dc43e18dcf107020e24638307a707d8c7afbf894ca1131e62907f8680ad0951"
)
TRUST_REGISTRY_FILE_SHA256 = (
    "e7fbf4f099bcc355073d0f5b591d988100df4168c22b172ce8fc08e4c4a0ed9c"
)
TRUST_REGISTRY_CONTENT_ID = (
    "3ecf96f83be71c5beb24160793d5fcbce2d771edbda74e190bda3d5b851b54fa"
)
LEAF_SCHEMA = "fmv_source_envelope_timestamp_leaf.v1"
RECEIPT_SCHEMA = "fmv_rfc3161_batch_timestamp_validation_receipt.v1"
SYNTHETIC_STATUS = "SYNTHETIC_SCHEMA_FIXTURE_ONLY_NOT_A_REAL_TIMESTAMP"
SYNTHETIC_RECEIPT_STATUS = (
    "SYNTHETIC_SCHEMA_FIXTURE_ONLY_NOT_CRYPTOGRAPHICALLY_VALIDATED"
)
SHA256_OID = "2.16.840.1.101.3.4.2.1"
ROLES = (
    "eex_acquisition",
    "eex_source_trusted_time",
    "entsoe_acquisition",
    "entsoe_source_trusted_time",
)
LEAF_FIELDS = (
    "schema_version",
    "capture_batch_id",
    "challenge_nonce_sha256",
    "envelope_role",
    "envelope_sha256",
)
RECEIPT_REQUIRED_FIELDS = (
    "schema_version",
    "capture_batch_id",
    "challenge_receipt_sha256",
    "challenge_nonce_sha256",
    "batch_root_sha256",
    "tree_size",
    "canonical_role_order",
    "timestamp_request_der_sha256",
    "timestamp_response_der_sha256",
    "timestamp_token_der_sha256",
    "response_status",
    "failure_info",
    "tst_info_version",
    "tsa_policy_oid",
    "message_imprint_hash_algorithm_oid",
    "message_imprint_hashed_message_hex",
    "request_nonce_decimal",
    "response_nonce_decimal",
    "serial_number_hex",
    "gen_time_utc",
    "accuracy_micros",
    "ordering",
    "tsa_name",
    "tsa_leaf_certificate_sha256",
    "tsa_certificate_chain_sha256s",
    "trusted_root_bundle_sha256",
    "revocation_evidence_sha256s",
    "validated_at_utc",
    "validator_identity",
    "validator_version",
    "validator_artifact_sha256",
    "cms_signature_verified",
    "signed_attributes_verified",
    "content_type_and_tst_info_verified",
    "message_imprint_verified",
    "nonce_verified",
    "policy_verified",
    "certificate_path_verified",
    "exclusive_critical_timestamping_eku_verified",
    "tsa_certificate_time_valid_at_gen_time_verified",
    "tsa_not_revoked_at_gen_time_verified",
    "trusted_root_pinned_before_request_verified",
    "challenge_to_external_time_bracket_verified",
    "maximum_external_time_lag_verified",
    "actual_signature_generation_time_verified",
    "historical_point_in_time_retroactively_proven",
)
FALSE_ACCESS_FIELDS = (
    "databricks_connections",
    "databricks_statements",
    "warehouse_starts",
    "network_calls",
    "h_drive_accesses",
    "remote_writes",
    "timestamp_authority_requests",
    "real_dsse_envelopes_opened",
    "market_value_rows_opened",
)
FALSE_AUTHORITY_FIELDS = (
    "real_source_envelopes_present",
    "real_timestamp_request_der_present",
    "real_timestamp_response_der_present",
    "rfc3161_token_cryptographically_verified",
    "independently_trusted_time_verified",
    "same_snapshot_point_in_time_availability_proven",
    "actual_signature_generation_time_verified",
    "training_authorized",
    "selection_authorized",
    "model_input_authorized",
    "candidate_assembly_authorized",
    "promotion_authorized",
    "production_authorized",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BATCH_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{7,127}$")
_OID = re.compile(r"^[0-2](?:\.[0-9]+)+$")


class ExternalTimeBatchError(ValueError):
    """Raised when an external-time batch input is ambiguous or overclaims."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the exact compact UTF-8 JSON encoding frozen by D235."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExternalTimeBatchError("value is not canonical finite JSON") from exc


def build_leaf_payload(
    *,
    capture_batch_id: str,
    challenge_nonce_sha256: str,
    envelope_role: str,
    envelope_sha256: str,
) -> dict[str, str]:
    """Build one exact domain payload before RFC 9162 leaf hashing."""

    batch_id = _safe_batch_id(capture_batch_id)
    challenge = _sha256(challenge_nonce_sha256, "challenge nonce")
    if envelope_role not in ROLES:
        raise ExternalTimeBatchError("envelope role is not governed")
    envelope = _sha256(envelope_sha256, "envelope")
    return {
        "schema_version": LEAF_SCHEMA,
        "capture_batch_id": batch_id,
        "challenge_nonce_sha256": challenge,
        "envelope_role": envelope_role,
        "envelope_sha256": envelope,
    }


def rfc9162_leaf_hash(payload: bytes) -> bytes:
    """Compute ``SHA256(0x00 || payload)`` from RFC 9162 section 2.1.1."""

    if not isinstance(payload, bytes) or not payload:
        raise ExternalTimeBatchError("leaf payload must be non-empty bytes")
    return hashlib.sha256(b"\x00" + payload).digest()


def rfc9162_node_hash(left: bytes, right: bytes) -> bytes:
    """Compute ``SHA256(0x01 || left || right)`` from RFC 9162."""

    if not isinstance(left, bytes) or len(left) != 32:
        raise ExternalTimeBatchError("left child must be a SHA-256 digest")
    if not isinstance(right, bytes) or len(right) != 32:
        raise ExternalTimeBatchError("right child must be a SHA-256 digest")
    return hashlib.sha256(b"\x01" + left + right).digest()


def rfc9162_merkle_tree_hash(entries: Sequence[bytes]) -> bytes:
    """Compute RFC 9162 MTH using its largest-power-of-two split rule."""

    if isinstance(entries, (bytes, str)) or not isinstance(entries, Sequence):
        raise ExternalTimeBatchError("Merkle entries must be a sequence")
    if not entries:
        return hashlib.sha256(b"").digest()
    if len(entries) == 1:
        return rfc9162_leaf_hash(entries[0])
    split = _largest_power_of_two_less_than(len(entries))
    return rfc9162_node_hash(
        rfc9162_merkle_tree_hash(entries[:split]),
        rfc9162_merkle_tree_hash(entries[split:]),
    )


def rfc9162_inclusion_path(entries: Sequence[bytes], leaf_index: int) -> list[bytes]:
    """Generate RFC 9162 ``PATH(m, D_n)`` in leaf-to-root order."""

    if not entries or not isinstance(leaf_index, int) or isinstance(leaf_index, bool):
        raise ExternalTimeBatchError("inclusion path needs a non-empty tree and integer index")
    if leaf_index < 0 or leaf_index >= len(entries):
        raise ExternalTimeBatchError("leaf index is outside the tree")
    if len(entries) == 1:
        return []
    split = _largest_power_of_two_less_than(len(entries))
    if leaf_index < split:
        return rfc9162_inclusion_path(entries[:split], leaf_index) + [
            rfc9162_merkle_tree_hash(entries[split:])
        ]
    return rfc9162_inclusion_path(entries[split:], leaf_index - split) + [
        rfc9162_merkle_tree_hash(entries[:split])
    ]


def verify_rfc9162_inclusion(
    *,
    leaf_hash: bytes,
    leaf_index: int,
    tree_size: int,
    inclusion_path: Sequence[bytes],
    expected_root: bytes,
) -> bool:
    """Verify an RFC 9162 section 2.1.3.2 inclusion proof."""

    if not isinstance(leaf_hash, bytes) or len(leaf_hash) != 32:
        raise ExternalTimeBatchError("leaf hash must be 32 bytes")
    if not isinstance(expected_root, bytes) or len(expected_root) != 32:
        raise ExternalTimeBatchError("expected root must be 32 bytes")
    if not isinstance(tree_size, int) or isinstance(tree_size, bool) or tree_size <= 0:
        raise ExternalTimeBatchError("tree size must be a positive integer")
    if not isinstance(leaf_index, int) or isinstance(leaf_index, bool):
        raise ExternalTimeBatchError("leaf index must be an integer")
    if leaf_index < 0 or leaf_index >= tree_size:
        raise ExternalTimeBatchError("leaf index is outside the tree")
    if isinstance(inclusion_path, (bytes, str)) or not isinstance(
        inclusion_path, Sequence
    ):
        raise ExternalTimeBatchError("inclusion path must be a sequence")
    path: list[bytes] = []
    for node in inclusion_path:
        if not isinstance(node, bytes) or len(node) != 32:
            raise ExternalTimeBatchError("inclusion node must be 32 bytes")
        path.append(node)

    fn = leaf_index
    sn = tree_size - 1
    result = leaf_hash
    for proof_node in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            result = rfc9162_node_hash(proof_node, result)
            if not fn & 1:
                while fn != 0 and not fn & 1:
                    fn >>= 1
                    sn >>= 1
        else:
            result = rfc9162_node_hash(result, proof_node)
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(result, expected_root)


def build_four_leaf_batch(
    *,
    capture_batch_id: str,
    challenge_nonce_sha256: str,
    envelope_sha256_by_role: Mapping[str, object],
) -> dict[str, object]:
    """Build and self-verify the exact four-role D235 Merkle batch."""

    envelopes = _mapping(envelope_sha256_by_role, "envelope hashes")
    _exact_keys(envelopes, set(ROLES), "envelope hashes")
    normalized = {role: _sha256(envelopes[role], f"{role} envelope") for role in ROLES}
    if len(set(normalized.values())) != len(ROLES):
        raise ExternalTimeBatchError("envelope hashes must be pairwise distinct")
    payloads = [
        canonical_json_bytes(
            build_leaf_payload(
                capture_batch_id=capture_batch_id,
                challenge_nonce_sha256=challenge_nonce_sha256,
                envelope_role=role,
                envelope_sha256=normalized[role],
            )
        )
        for role in ROLES
    ]
    leaves = [rfc9162_leaf_hash(payload) for payload in payloads]
    root = rfc9162_merkle_tree_hash(payloads)
    proofs: dict[str, dict[str, object]] = {}
    for index, role in enumerate(ROLES):
        path = rfc9162_inclusion_path(payloads, index)
        if not verify_rfc9162_inclusion(
            leaf_hash=leaves[index],
            leaf_index=index,
            tree_size=len(ROLES),
            inclusion_path=path,
            expected_root=root,
        ):
            raise ExternalTimeBatchError(f"generated inclusion proof failed for {role}")
        proofs[role] = {
            "leaf_index": index,
            "tree_size": len(ROLES),
            "leaf_sha256": leaves[index].hex(),
            "inclusion_path_sha256s": [item.hex() for item in path],
            "verified": True,
        }
    return {
        "schema_version": "fmv_eex_entsoe_external_time_merkle_batch.v1",
        "capture_batch_id": _safe_batch_id(capture_batch_id),
        "challenge_nonce_sha256": _sha256(challenge_nonce_sha256, "challenge nonce"),
        "canonical_role_order": list(ROLES),
        "envelope_sha256_by_role": normalized,
        "tree_size": len(ROLES),
        "root_sha256": root.hex(),
        "proofs_by_role": proofs,
        "all_inclusion_proofs_verified": True,
    }


def validate_synthetic_rfc3161_receipt_fixture(
    document: Mapping[str, object],
    *,
    expected_root_sha256: str,
) -> dict[str, object]:
    """Validate only the non-authoritative fixture; never accept a real token claim."""

    receipt = _mapping(document, "synthetic RFC 3161 receipt")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "batch_root_sha256",
            "timestamp_request_der_present",
            "timestamp_response_der_present",
            "timestamp_token_der_present",
            "cms_signature_verified",
            "message_imprint_verified",
            "nonce_verified",
            "policy_verified",
            "certificate_path_verified",
            "revocation_verified",
            "independently_trusted_time_verified",
            "actual_signature_generation_time_verified",
            "historical_point_in_time_retroactively_proven",
        },
        "synthetic RFC 3161 receipt",
    )
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(receipt.get("status"), SYNTHETIC_RECEIPT_STATUS, "receipt status")
    _equal(
        receipt.get("batch_root_sha256"),
        _sha256(expected_root_sha256, "expected root"),
        "receipt batch root",
    )
    for field, value in receipt.items():
        if field in {"schema_version", "status", "batch_root_sha256"}:
            continue
        _equal(value, False, f"synthetic receipt {field}")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": SYNTHETIC_RECEIPT_STATUS,
        "rfc3161_token_cryptographically_verified": False,
        "independently_trusted_time_verified": False,
    }


def validate_external_time_batch_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate and replay the exact D235 contract and synthetic fixture."""

    contract = _mapping(document, "external-time contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "status",
            "purpose",
            "selected_boundaries",
            "leaf_contract",
            "rfc9162_merkle_contract",
            "rfc3161_request_contract",
            "future_real_rfc3161_validation_receipt_contract",
            "synthetic_fixture",
            "access_counters",
            "standards_basis",
            "authorities",
            "required_before_real_external_time_authority",
        },
        "external-time contract",
    )
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _validate_boundaries(_mapping(contract.get("selected_boundaries"), "boundaries"))
    _validate_leaf_contract(_mapping(contract.get("leaf_contract"), "leaf contract"))
    _validate_merkle_contract(
        _mapping(contract.get("rfc9162_merkle_contract"), "Merkle contract")
    )
    _validate_request_contract(
        _mapping(contract.get("rfc3161_request_contract"), "RFC 3161 request contract")
    )
    _validate_future_receipt_contract(
        _mapping(
            contract.get("future_real_rfc3161_validation_receipt_contract"),
            "future receipt contract",
        )
    )
    fixture_assessment = _validate_fixture(
        _mapping(contract.get("synthetic_fixture"), "synthetic fixture")
    )
    access = _mapping(contract.get("access_counters"), "access counters")
    _exact_keys(access, set(FALSE_ACCESS_FIELDS), "access counters")
    for field in FALSE_ACCESS_FIELDS:
        _integer_zero(access.get(field), f"access counter {field}")
    _exact_sequence(
        contract.get("standards_basis"),
        (
            "https://www.rfc-editor.org/rfc/rfc9162#section-2.1.1",
            "https://www.rfc-editor.org/rfc/rfc9162#section-2.1.3",
            "https://www.rfc-editor.org/rfc/rfc3161#section-2.4.1",
            "https://www.rfc-editor.org/rfc/rfc3161#section-2.4.2",
        ),
        "standards basis",
    )
    authorities = _mapping(contract.get("authorities"), "authorities")
    _exact_keys(
        authorities,
        {"synthetic_merkle_fixture_verified_after_successful_replay", *FALSE_AUTHORITY_FIELDS},
        "authorities",
    )
    _equal(
        authorities.get("synthetic_merkle_fixture_verified_after_successful_replay"),
        True,
        "synthetic Merkle authority",
    )
    for field in FALSE_AUTHORITY_FIELDS:
        _equal(authorities.get(field), False, f"authority {field}")
    required = contract.get("required_before_real_external_time_authority")
    if not isinstance(required, list) or len(required) != 9:
        raise ExternalTimeBatchError("real external-time gate inventory must contain nine items")
    if not all(isinstance(item, str) and item.strip() for item in required):
        raise ExternalTimeBatchError("real external-time gates must be non-empty text")
    content_id = _sha256_json(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "fixture_root_sha256": fixture_assessment["root_sha256"],
        "all_four_inclusion_proofs_verified": True,
        "rfc3161_request_der_present": False,
        "rfc3161_token_cryptographically_verified": False,
        "independently_trusted_time_verified": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_local_boundary(workspace_root: Path) -> dict[str, object]:
    """Verify exact D233/D234 parent contracts without opening source values."""

    root = assert_absolute_path_has_no_links(workspace_root.resolve(strict=True))
    boundaries = (
        (
            "D233",
            D233_CONTRACT_RELATIVE_PATH,
            D233_CONTRACT_FILE_SHA256,
            D233_CONTRACT_CONTENT_ID,
        ),
        (
            "D234",
            D234_CONTRACT_RELATIVE_PATH,
            D234_CONTRACT_FILE_SHA256,
            D234_CONTRACT_CONTENT_ID,
        ),
    )
    for label, relative, expected_file, expected_content in boundaries:
        selected = assert_absolute_path_has_no_links(
            (root / relative).resolve(strict=True)
        )
        try:
            selected.relative_to(root)
        except ValueError as exc:
            raise ExternalTimeBatchError(f"{label} boundary escapes workspace") from exc
        payload = read_stable_single_link_file(
            selected,
            label=f"D235 {label} parent contract",
            max_bytes=1_000_000,
        )
        _equal(hashlib.sha256(payload).hexdigest(), expected_file, f"{label} file hash")
        _equal(
            _sha256_json(load_strict_json(payload)),
            expected_content,
            f"{label} content ID",
        )
    return {
        "schema_version": "fmv_eex_entsoe_d235_local_boundary_validation.v1",
        "status": "PASS_LOCAL_D233_D234_BOUNDARIES_ONLY",
        "d233_contract_sha256": D233_CONTRACT_FILE_SHA256,
        "d233_contract_content_id": D233_CONTRACT_CONTENT_ID,
        "d234_contract_sha256": D234_CONTRACT_FILE_SHA256,
        "d234_contract_content_id": D234_CONTRACT_CONTENT_ID,
        "databricks_statements": 0,
        "network_calls": 0,
        "h_drive_accesses": 0,
        "real_timestamp_tokens_opened": 0,
    }


def load_strict_json(payload: bytes) -> Mapping[str, object]:
    """Load bounded UTF-8 JSON while rejecting duplicates and non-finite tokens."""

    if not payload or len(payload) > 1_000_000:
        raise ExternalTimeBatchError("JSON payload must be non-empty and <= 1 MB")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExternalTimeBatchError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ExternalTimeBatchError(f"non-finite JSON token forbidden: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalTimeBatchError("invalid strict UTF-8 JSON") from exc
    return _mapping(value, "JSON payload")


def _validate_boundaries(boundaries: Mapping[str, object]) -> None:
    expected = {
        "governed_acquisition_package_schema": "fmv-eex-entsoe-governed-acquisition-package-contract-v1",
        "governed_acquisition_package_status": "PREPARED_D233_LOCAL_ZERO_EXECUTION_NO_MODEL_AUTHORITY",
        "governed_acquisition_package_content_id": D233_CONTRACT_CONTENT_ID,
        "governed_acquisition_package_sha256": D233_CONTRACT_FILE_SHA256,
        "governed_acquisition_package_decision": "D-20260805-233",
        "entsoe_physical_mapping_contract_schema": "fmv_entsoe_physical_mapping_compiler_contract.v1",
        "entsoe_physical_mapping_contract_content_id": D234_CONTRACT_CONTENT_ID,
        "entsoe_physical_mapping_contract_sha256": D234_CONTRACT_FILE_SHA256,
        "entsoe_physical_mapping_decision": "D-20260805-234",
        "trust_registry_contract_content_id": TRUST_REGISTRY_CONTENT_ID,
        "trust_registry_contract_sha256": TRUST_REGISTRY_FILE_SHA256,
        "trust_registry_decision": "D-20260805-229",
    }
    _exact_keys(boundaries, set(expected), "boundaries")
    for field, expected_value in expected.items():
        _equal(boundaries.get(field), expected_value, f"boundary {field}")


def _validate_leaf_contract(leaf: Mapping[str, object]) -> None:
    _equal(leaf.get("schema_version"), LEAF_SCHEMA, "leaf schema")
    _equal(
        leaf.get("canonical_json_encoding"),
        "UTF8_NO_BOM_SORTED_KEYS_COMPACT_NO_TRAILING_NEWLINE",
        "leaf encoding",
    )
    _exact_sequence(leaf.get("required_fields"), LEAF_FIELDS, "leaf fields")
    _exact_sequence(leaf.get("canonical_role_order"), ROLES, "leaf roles")
    _equal(leaf.get("capture_batch_id_pattern"), _BATCH_ID.pattern, "batch ID pattern")
    _equal(leaf.get("sha256_encoding"), "LOWERCASE_HEX_64", "leaf SHA encoding")
    for field in (
        "same_capture_batch_and_challenge_required_for_all_leaves",
        "exactly_one_leaf_per_role",
    ):
        _equal(leaf.get(field), True, f"leaf contract {field}")
    for field in ("duplicate_envelope_sha256_authorized", "unknown_fields_authorized"):
        _equal(leaf.get(field), False, f"leaf contract {field}")


def _validate_merkle_contract(merkle: Mapping[str, object]) -> None:
    expected = {
        "hash_algorithm": "SHA-256",
        "hash_algorithm_oid": SHA256_OID,
        "tree_size": 4,
        "leaf_hash_formula": "SHA256(0x00 || canonical_leaf_payload_bytes)",
        "node_hash_formula": "SHA256(0x01 || left_child_hash || right_child_hash)",
        "inclusion_path_length_for_four_leaf_tree": 2,
        "root_sha256_encoding": "LOWERCASE_HEX_64",
    }
    for field, expected_value in expected.items():
        _equal(merkle.get(field), expected_value, f"Merkle contract {field}")
    for field in (
        "largest_power_of_two_split_rule_required",
        "proof_direction_derived_from_leaf_index_and_tree_size",
        "one_exact_inclusion_proof_per_role_required",
        "constant_time_root_comparison_required",
    ):
        _equal(merkle.get(field), True, f"Merkle contract {field}")
    for field in (
        "caller_supplied_direction_bits_authorized",
        "empty_or_non_four_leaf_batch_authorized",
        "proof_with_missing_extra_or_malformed_node_authorized",
    ):
        _equal(merkle.get(field), False, f"Merkle contract {field}")


def _validate_request_contract(request: Mapping[str, object]) -> None:
    expected = {
        "request_version": 1,
        "profile": "RFC3161_BATCH_ROOT",
        "timestamped_object": "RFC9162_MERKLE_TREE_HASH_OF_FOUR_CANONICAL_SOURCE_ENVELOPE_LEAVES",
        "message_imprint_hash_algorithm": "SHA-256",
        "message_imprint_hash_algorithm_oid": SHA256_OID,
    }
    for field, expected_value in expected.items():
        _equal(request.get(field), expected_value, f"RFC 3161 request {field}")
    for field in (
        "message_imprint_hashed_message_must_equal_exact_batch_root",
        "nonce_required",
        "nonce_positive_integer_decimal_string_required",
        "request_policy_oid_required",
        "cert_req_required",
        "request_der_sha256_required_before_network_submission",
    ):
        _equal(request.get(field), True, f"RFC 3161 request {field}")
    for field in (
        "network_submission_implemented_by_d235",
        "tsa_policy_selected_or_endorsed_by_d235",
        "self_signed_or_locally_self_timestamped_token_authorized",
    ):
        _equal(request.get(field), False, f"RFC 3161 request {field}")


def _validate_future_receipt_contract(receipt: Mapping[str, object]) -> None:
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "future receipt schema")
    _exact_sequence(
        receipt.get("required_fields"), RECEIPT_REQUIRED_FIELDS, "future receipt fields"
    )
    _exact_sequence(receipt.get("response_status_allowed"), ("GRANTED",), "response status")
    _equal(receipt.get("tst_info_version_required"), 1, "TSTInfo version")
    _equal(
        receipt.get("maximum_gen_time_lag_after_last_envelope_seconds"),
        3600,
        "genTime lag",
    )
    for field in (
        "failure_info_must_be_null_on_granted",
        "message_imprint_must_equal_request_and_exact_batch_root",
        "response_nonce_must_equal_request_nonce",
        "gen_time_must_not_precede_governed_challenge_or_last_envelope",
        "validation_time_must_not_precede_gen_time",
        "certificate_chain_and_revocation_evidence_required",
        "independent_token_byte_parsing_and_cryptographic_verification_required",
        "actual_signature_generation_time_verified_must_be_false",
        "historical_point_in_time_retroactively_proven_must_be_false",
    ):
        _equal(receipt.get(field), True, f"future receipt {field}")
    _equal(
        receipt.get("receipt_booleans_without_independent_token_replay_authorized"),
        False,
        "self-declared receipt booleans",
    )


def _validate_fixture(fixture: Mapping[str, object]) -> dict[str, object]:
    _equal(fixture.get("status"), SYNTHETIC_STATUS, "fixture status")
    batch = _safe_batch_id(fixture.get("capture_batch_id"))
    challenge = _sha256(fixture.get("challenge_nonce_sha256"), "fixture challenge")
    built = build_four_leaf_batch(
        capture_batch_id=batch,
        challenge_nonce_sha256=challenge,
        envelope_sha256_by_role=_mapping(
            fixture.get("envelope_sha256_by_role"), "fixture envelopes"
        ),
    )
    expected_leaves = _mapping(
        fixture.get("expected_leaf_sha256_by_role"), "expected leaves"
    )
    _exact_keys(expected_leaves, set(ROLES), "expected leaves")
    expected_paths = _mapping(
        fixture.get("expected_inclusion_paths_by_role"), "expected paths"
    )
    _exact_keys(expected_paths, set(ROLES), "expected paths")
    expected_root = _sha256(fixture.get("expected_root_sha256"), "fixture root")
    _equal(built["root_sha256"], expected_root, "fixture computed root")
    proofs = _mapping(built["proofs_by_role"], "computed proofs")
    for role in ROLES:
        proof = _mapping(proofs[role], f"{role} proof")
        _equal(proof.get("leaf_sha256"), expected_leaves[role], f"{role} leaf")
        _exact_sequence(
            proof.get("inclusion_path_sha256s"),
            _sequence(expected_paths[role], f"{role} expected path"),
            f"{role} inclusion path",
        )
    request = _mapping(fixture.get("rfc3161_request_template"), "request fixture")
    _validate_request_fixture(request, expected_root=expected_root)
    receipt = validate_synthetic_rfc3161_receipt_fixture(
        _mapping(fixture.get("rfc3161_validation_receipt_fixture"), "receipt fixture"),
        expected_root_sha256=expected_root,
    )
    return {
        "root_sha256": expected_root,
        "proofs_verified": True,
        "request_der_present": False,
        "receipt": receipt,
    }


def _validate_request_fixture(request: Mapping[str, object], *, expected_root: str) -> None:
    _equal(request.get("request_version"), 1, "fixture request version")
    _equal(request.get("profile"), "RFC3161_BATCH_ROOT", "fixture request profile")
    _equal(
        request.get("message_imprint_hash_algorithm_oid"),
        SHA256_OID,
        "fixture request algorithm",
    )
    _equal(
        request.get("message_imprint_hashed_message_hex"),
        expected_root,
        "fixture request imprint",
    )
    nonce = request.get("nonce_decimal")
    if not isinstance(nonce, str) or not nonce.isdigit() or int(nonce) <= 0:
        raise ExternalTimeBatchError("fixture request nonce must be a positive decimal string")
    policy = request.get("request_policy_oid")
    if not isinstance(policy, str) or _OID.fullmatch(policy) is None:
        raise ExternalTimeBatchError("fixture request policy must be an OID")
    _equal(request.get("request_policy_is_synthetic_placeholder"), True, "fixture policy")
    _equal(request.get("cert_req"), True, "fixture certReq")
    _equal(request.get("request_der_present"), False, "fixture request DER")
    _equal(request.get("network_submission_performed"), False, "fixture network")


def _largest_power_of_two_less_than(value: int) -> int:
    if value <= 1:
        raise ExternalTimeBatchError("Merkle split requires at least two entries")
    return 1 << ((value - 1).bit_length() - 1)


def _safe_batch_id(value: object) -> str:
    if not isinstance(value, str) or _BATCH_ID.fullmatch(value) is None:
        raise ExternalTimeBatchError("capture batch ID is unsafe")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExternalTimeBatchError(f"{label} must be lowercase SHA-256")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ExternalTimeBatchError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExternalTimeBatchError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ExternalTimeBatchError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _exact_sequence(value: object, expected: Sequence[object], label: str) -> None:
    selected = _sequence(value, label)
    if tuple(selected) != tuple(expected):
        raise ExternalTimeBatchError(
            f"{label} differs: expected {list(expected)!r}, received {list(selected)!r}"
        )


def _integer_zero(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != 0:
        raise ExternalTimeBatchError(f"{label} must be integer zero")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ExternalTimeBatchError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_SCHEMA",
    "CONTRACT_STATUS",
    "ExternalTimeBatchError",
    "ROLES",
    "build_four_leaf_batch",
    "build_leaf_payload",
    "canonical_json_bytes",
    "load_strict_json",
    "rfc9162_inclusion_path",
    "rfc9162_leaf_hash",
    "rfc9162_merkle_tree_hash",
    "rfc9162_node_hash",
    "validate_external_time_batch_contract",
    "validate_local_boundary",
    "validate_synthetic_rfc3161_receipt_fixture",
    "verify_rfc9162_inclusion",
]
