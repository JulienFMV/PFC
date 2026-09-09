from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pfc_shaping.validation import eex_entsoe_rfc3161_request_der as subject
from pfc_shaping.validation.eex_entsoe_rfc3161_request_der import (
    SYNTHETIC_NONCE_DECIMAL,
    SYNTHETIC_PREIMAGE_HEX,
    SYNTHETIC_REQUEST_DER_HEX,
    SYNTHETIC_REQUEST_DER_SHA256,
    SYNTHETIC_ROOT,
    Rfc3161RequestError,
    build_and_validate_synthetic_fixture,
    build_timestamp_request_der,
    canonical_json_bytes,
    load_contract,
    materialize_proof,
    parse_timestamp_request_der,
    validate_contract,
    validate_d235_parent,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-ENTSOE-RFC3161-REQUEST-DER-PREFLIGHT-CONTRACT-V1.json"
)
D235_CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-ENTSOE-EXTERNAL-TIME-BATCH-CONTRACT-V1.json"
)
D235_MANIFEST_PATH = (
    ROOT
    / "build"
    / "databricks-eex-daily"
    / "2026-08-05"
    / "eex-entsoe-external-time-batch-proofs"
    / "93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1"
    / "manifest.json"
)


def _document() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _request() -> bytes:
    return bytes.fromhex(SYNTHETIC_REQUEST_DER_HEX)


def _with_outer_length(data: bytes, length: int) -> bytes:
    assert length < 128
    return bytes((data[0], length)) + data[2:]


def _insert(data: bytes, index: int, payload: bytes) -> bytes:
    return data[:index] + payload + data[index:]


def test_contract_loads_and_validates() -> None:
    document, raw = load_contract(CONTRACT_PATH)
    assessment = validate_contract(document)
    assert hashlib.sha256(raw).hexdigest() == subject.CONTRACT_FILE_SHA256
    assert assessment["status"] == subject.CONTRACT_STATUS
    assert assessment["timestamp_authority_requests"] == 0
    assert assessment["network_calls"] == 0
    assert assessment["model_input_authorized"] is False
    assert assessment["production_authorized"] is False


def test_contract_content_id_is_frozen() -> None:
    document = _document()
    actual = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    assert actual == subject.CONTRACT_CONTENT_ID


def test_synthetic_request_matches_fixed_der_and_hash() -> None:
    fixture = build_and_validate_synthetic_fixture()
    parsed = fixture["request"]
    assert isinstance(parsed, dict)
    assert parsed["der_size_bytes"] == 89
    assert parsed["der_sha256"] == SYNTHETIC_REQUEST_DER_SHA256
    assert parsed["message_imprint_hashed_message_hex"] == SYNTHETIC_ROOT
    assert fixture["final_node_preimage_sha256"] == SYNTHETIC_ROOT
    assert fixture["real_request_authorized_for_submission"] is False
    assert fixture["timestamp_authority_requests"] == 0


def test_parser_recovers_exact_profile_fields() -> None:
    parsed = parse_timestamp_request_der(_request())
    assert parsed == {
        "version": 1,
        "message_imprint_hash_algorithm_oid": "2.16.840.1.101.3.4.2.1",
        "message_imprint_hashed_message_hex": SYNTHETIC_ROOT,
        "request_policy_oid": "1.3.6.1.4.1.55555.1.235",
        "nonce_decimal": SYNTHETIC_NONCE_DECIMAL,
        "nonce_bit_length": 129,
        "cert_req": True,
        "extensions_present": False,
        "der_size_bytes": 89,
        "der_sha256": SYNTHETIC_REQUEST_DER_SHA256,
    }


def test_builder_binds_final_node_preimage_to_root() -> None:
    request = build_timestamp_request_der(
        final_node_preimage=bytes.fromhex(SYNTHETIC_PREIMAGE_HEX),
        merkle_root=bytes.fromhex(SYNTHETIC_ROOT),
        request_policy_oid="1.3.6.1.4.1.55555.1.235",
        nonce=int(SYNTHETIC_NONCE_DECIMAL),
    )
    assert request == _request()


@pytest.mark.parametrize(
    ("preimage", "root", "policy", "nonce"),
    [
        (b"\x01" + b"\x00" * 63, bytes.fromhex(SYNTHETIC_ROOT), "1.2.3", 1),
        (b"\x00" + b"\x00" * 64, bytes.fromhex(SYNTHETIC_ROOT), "1.2.3", 1),
        (bytes.fromhex(SYNTHETIC_PREIMAGE_HEX), b"\x00" * 31, "1.2.3", 1),
        (bytes.fromhex(SYNTHETIC_PREIMAGE_HEX), b"\x00" * 32, "1.2.3", 1),
        (bytes.fromhex(SYNTHETIC_PREIMAGE_HEX), bytes.fromhex(SYNTHETIC_ROOT), "3.1", 1),
        (bytes.fromhex(SYNTHETIC_PREIMAGE_HEX), bytes.fromhex(SYNTHETIC_ROOT), "1.2.3", 0),
        (bytes.fromhex(SYNTHETIC_PREIMAGE_HEX), bytes.fromhex(SYNTHETIC_ROOT), "1.2.3", -1),
    ],
)
def test_builder_rejects_invalid_semantics(
    preimage: bytes,
    root: bytes,
    policy: str,
    nonce: int,
) -> None:
    with pytest.raises(Rfc3161RequestError):
        build_timestamp_request_der(
            final_node_preimage=preimage,
            merkle_root=root,
            request_policy_oid=policy,
            nonce=nonce,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda data: b"",
        lambda data: bytearray(data),
        lambda data: data + b"\x00",
        lambda data: data[:-1],
        lambda data: bytes((0x31,)) + data[1:],
        lambda data: bytes((0x3F,)) + data[1:],
        lambda data: bytes((data[0], 0x80)) + data[2:],
        lambda data: bytes((data[0], 0x81, data[1])) + data[2:],
        lambda data: bytes((data[0], 0x82, 0x00, data[1])) + data[2:],
        lambda data: bytes((data[0], data[1] + 1)) + data[2:],
    ],
)
def test_parser_rejects_outer_der_faults(mutator: object) -> None:
    corrupted = mutator(_request())  # type: ignore[operator]
    with pytest.raises(Rfc3161RequestError):
        parse_timestamp_request_der(corrupted)  # type: ignore[arg-type]


def test_parser_rejects_request_above_byte_ceiling() -> None:
    with pytest.raises(Rfc3161RequestError):
        parse_timestamp_request_der(b"\x30\x00" + b"\x00" * 4095)


@pytest.mark.parametrize(
    "index_value",
    [
        (4, 0x02),
        (4, 0x00),
        (4, 0x81),
        (5, 0x31),
        (7, 0x31),
        (19, 0x02),
        (20, 0x05),
        (54, 0x05),
        (67, 0x04),
        (86, 0x02),
        (88, 0x00),
        (88, 0x01),
    ],
)
def test_parser_rejects_field_mutations(index_value: tuple[int, int]) -> None:
    index, value = index_value
    mutated = bytearray(_request())
    mutated[index] = value
    with pytest.raises(Rfc3161RequestError):
        parse_timestamp_request_der(bytes(mutated))


def test_parser_rejects_nonminimal_version() -> None:
    data = _request()
    mutated = _with_outer_length(_insert(data, 4, b"\x00"), data[1] + 1)
    mutated = mutated[:3] + b"\x02" + mutated[4:]
    with pytest.raises(Rfc3161RequestError, match="minimally"):
        parse_timestamp_request_der(mutated)


def test_parser_rejects_sha256_null_parameters() -> None:
    data = bytearray(_insert(_request(), 20, b"\x05\x00"))
    data[1] += 2
    data[6] += 2
    data[8] += 2
    with pytest.raises(Rfc3161RequestError, match="parameters must be absent"):
        parse_timestamp_request_der(bytes(data))


def test_parser_rejects_truncated_hashed_message() -> None:
    data = bytearray(_request())
    del data[53]
    data[1] -= 1
    data[6] -= 1
    data[21] -= 1
    with pytest.raises(Rfc3161RequestError, match="32 bytes"):
        parse_timestamp_request_der(bytes(data))


def test_parser_rejects_missing_policy() -> None:
    data = _request()
    mutated = _with_outer_length(data[:54] + data[67:], data[1] - 13)
    with pytest.raises(Rfc3161RequestError):
        parse_timestamp_request_der(mutated)


def test_parser_rejects_noncanonical_policy_oid() -> None:
    data = bytearray(_insert(_request(), 57, b"\x80"))
    data[1] += 1
    data[55] += 1
    with pytest.raises(Rfc3161RequestError, match="OID is not minimally encoded"):
        parse_timestamp_request_der(bytes(data))


def test_parser_rejects_missing_nonce() -> None:
    data = _request()
    mutated = _with_outer_length(data[:67] + data[86:], data[1] - 19)
    with pytest.raises(Rfc3161RequestError):
        parse_timestamp_request_der(mutated)


def test_parser_rejects_nonminimal_nonce() -> None:
    data = bytearray(_insert(_request(), 69, b"\x00"))
    data[1] += 1
    data[68] += 1
    with pytest.raises(Rfc3161RequestError, match="minimally"):
        parse_timestamp_request_der(bytes(data))


def test_parser_rejects_missing_cert_req() -> None:
    data = _request()
    mutated = _with_outer_length(data[:-3], data[1] - 3)
    with pytest.raises(Rfc3161RequestError):
        parse_timestamp_request_der(mutated)


def test_parser_rejects_extensions() -> None:
    data = bytearray(_request())
    data[1] += 4
    mutated = bytes(data) + b"\xa0\x02\x30\x00"
    with pytest.raises(Rfc3161RequestError, match="extensions"):
        parse_timestamp_request_der(mutated)


def test_parser_rejects_noncanonical_long_nonce_length() -> None:
    data = bytearray(_request())
    data[1] += 1
    mutated = data[:68] + bytes((0x81, 0x11)) + data[69:]
    with pytest.raises(Rfc3161RequestError, match="shortest form"):
        parse_timestamp_request_der(bytes(mutated))


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("input_bindings", "external_time_batch_decision", "D-20260805-999"),
        ("request_profile", "version", 2),
        ("request_profile", "generated_sha2_algorithm_identifier_parameters", "NULL"),
        ("request_profile", "d237_exact_generated_request_replay_parameters", ["ABSENT", "NULL"]),
        ("request_profile", "general_rfc5754_interoperable_parser_must_accept_parameters", ["ABSENT"]),
        ("request_profile", "extensions_allowed", True),
        ("synthetic_fixture", "merkle_root_sha256", "00" * 32),
        ("synthetic_fixture", "request_der_sha256", "00" * 32),
        ("runtime_capability_preflight", "real_token_acceptance_with_current_stack", True),
        ("runtime_capability_preflight", "candidate_parser_admitted_by_d237", True),
        ("authority_boundary", "independently_trusted_time_verified", True),
        ("authority_boundary", "model_input_authorized", True),
        ("execution_budget", "network_calls", 1),
        ("execution_budget", "timestamp_authority_requests", 1),
        ("execution_budget", "h_drive_accesses", 1),
    ],
)
def test_contract_mutations_fail_closed(
    section: str,
    field: str,
    replacement: object,
) -> None:
    changed = copy.deepcopy(_document())
    changed_section = changed[section]
    assert isinstance(changed_section, dict)
    changed_section[field] = replacement
    with pytest.raises(Rfc3161RequestError):
        validate_contract(changed)


def test_contract_rejects_top_level_extra_field() -> None:
    changed = _document()
    changed["unexpected"] = True
    with pytest.raises(Rfc3161RequestError, match="fields differ"):
        validate_contract(changed)


def test_parent_d235_bytes_and_zero_execution_are_bound() -> None:
    result = validate_d235_parent(
        contract_path=D235_CONTRACT_PATH,
        proof_manifest_path=D235_MANIFEST_PATH,
    )
    assert result["d235_proof_content_id"] == subject.D235_PROOF_CONTENT_ID
    assert result["databricks_statements"] == 0
    assert result["network_calls"] == 0
    assert result["h_drive_accesses"] == 0
    assert result["timestamp_authority_requests"] == 0


def test_parent_d235_contract_tamper_fails(tmp_path: Path) -> None:
    target = tmp_path / "contract.json"
    target.write_bytes(D235_CONTRACT_PATH.read_bytes() + b" ")
    with pytest.raises(Rfc3161RequestError, match="D235 contract SHA-256"):
        validate_d235_parent(
            contract_path=target,
            proof_manifest_path=D235_MANIFEST_PATH,
        )


def test_parent_d235_manifest_tamper_fails(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_bytes(D235_MANIFEST_PATH.read_bytes() + b" ")
    with pytest.raises(Rfc3161RequestError, match="D235 proof manifest SHA-256"):
        validate_d235_parent(
            contract_path=D235_CONTRACT_PATH,
            proof_manifest_path=target,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b"\xef\xbb\xbf{}",
        b'{"a":NaN}',
        b"\xff",
    ],
)
def test_strict_json_rejects_ambiguous_inputs(raw: bytes) -> None:
    with pytest.raises(Rfc3161RequestError):
        subject._strict_json(raw, "fixture")


def test_contract_fixture_hashes_are_lowercase_sha256() -> None:
    fixture = _document()["synthetic_fixture"]
    assert isinstance(fixture, dict)
    for field in (
        "left_subtree_hash_hex",
        "right_subtree_hash_hex",
        "merkle_root_sha256",
        "request_der_sha256",
    ):
        value = fixture[field]
        assert isinstance(value, str)
        assert subject._HEX_64.fullmatch(value)


def test_materializer_is_deterministic_and_zero_execution(tmp_path: Path) -> None:
    output = tmp_path / "proofs"
    first = materialize_proof(output_root=output)
    second = materialize_proof(output_root=output)
    assert first == second
    manifest_path = first / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["content_id"] == first.name
    assert manifest["contract_content_id"] == subject.CONTRACT_CONTENT_ID
    assert set(manifest["execution"].values()) == {0}
    assert manifest["authorities"]["synthetic_request_der_verified"] is True
    assert all(
        value is False
        for key, value in manifest["authorities"].items()
        if key != "synthetic_request_der_verified"
    )
    for name, expected in manifest["artifacts"].items():
        payload = (first / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
        assert len(payload) == expected["size_bytes"]


def test_materializer_rejects_existing_artifact_tamper(tmp_path: Path) -> None:
    output = tmp_path / "proofs"
    proof = materialize_proof(output_root=output)
    (proof / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(Rfc3161RequestError, match="existing D237 artifact differs"):
        materialize_proof(output_root=output)
