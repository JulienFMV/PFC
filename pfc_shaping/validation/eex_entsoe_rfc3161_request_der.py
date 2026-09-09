"""Canonical RFC 3161 request DER preflight without TSA or token authority."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

CONTRACT_SCHEMA = "fmv_eex_entsoe_rfc3161_request_der_preflight_contract.v1"
CONTRACT_STATUS = (
    "PASS_LOCAL_REQUEST_DER_AND_CAPABILITY_PREFLIGHT_ONLY_NO_TSA_OR_TOKEN_AUTHORITY"
)
CONTRACT_DECISION = "D-20260805-237"
CONTRACT_FILE_SHA256 = (
    "d087736e548f579c03eaf84e89c2ee268edca0a9e8bf4e42db3044e533e8234b"
)
CONTRACT_CONTENT_ID = (
    "2333f35a9df0500cac5826a47105cf610c607cc64ad07ab904daf23038a68199"
)

D235_CONTRACT_SHA256 = (
    "e9da2a73dd1bd62b5357e1f507bff3ef8e48918806c918bd4fdb1f609a71d138"
)
D235_CONTRACT_CONTENT_ID = (
    "8f0f13315c382da7163a1f451752a6ddf84857a50133d89d3a3a31f340b264d0"
)
D235_PROOF_CONTENT_ID = (
    "93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1"
)
D235_PROOF_MANIFEST_SHA256 = (
    "a10bbfa380652365684213197862139bf958dc33e706d81635cc2a6ae4582f02"
)

SHA256_OID = "2.16.840.1.101.3.4.2.1"
SYNTHETIC_POLICY_OID = "1.3.6.1.4.1.55555.1.235"
SYNTHETIC_NONCE_DECIMAL = "340282366920938463463374607431768211507"
SYNTHETIC_LEFT_HASH = (
    "88a55a70db9afd42a35d7c09e5cbc6770e709345da4e5846b004c88ba7c42444"
)
SYNTHETIC_RIGHT_HASH = (
    "8e70e7aabed64df8ae293dd11a2f93ce902c26b3c54249e46c2f0fa7bbc01bd0"
)
SYNTHETIC_ROOT = (
    "496c92c29ab1b4f1ad0f3491cbd4e38f500936973bc6ff8e8ac8ebeb3b0aa85c"
)
SYNTHETIC_PREIMAGE_HEX = "01" + SYNTHETIC_LEFT_HASH + SYNTHETIC_RIGHT_HASH
SYNTHETIC_REQUEST_DER_HEX = (
    "3057020101302f300b06096086480165030402010420"
    + SYNTHETIC_ROOT
    + "060b2b0601040183b20301816b"
    + "02110100000000000000000000000000000033"
    + "0101ff"
)
SYNTHETIC_REQUEST_DER_SHA256 = (
    "75fc863d2083fb1bc6d816d993cfb762764c6735c33acdecc4645b9db88ab3c4"
)

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_OID = re.compile(r"^(?:0|1|2)(?:\.(?:0|[1-9][0-9]*))+$")
_MAX_CONTRACT_BYTES = 200_000
_MAX_REQUEST_BYTES = 4096
_PROOF_SCHEMA = "fmv_eex_entsoe_rfc3161_request_der_preflight_proof.v1"


class Rfc3161RequestError(ValueError):
    """Raised when the request contract or DER bytes fail closed."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the exact compact canonical JSON representation used for IDs."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Rfc3161RequestError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Rfc3161RequestError(
            f"{label} fields differ: missing={missing}, extra={extra}"
        )


def _equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise Rfc3161RequestError(f"{label} differs")


def _strict_json(raw: bytes, label: str) -> Mapping[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Rfc3161RequestError(f"{label} must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Rfc3161RequestError(f"{label} is not strict UTF-8") from exc

    def reject_constant(value: str) -> object:
        raise Rfc3161RequestError(f"{label} contains non-finite number {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Rfc3161RequestError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Rfc3161RequestError(f"{label} is not valid JSON") from exc
    return _mapping(parsed, label)


def load_contract(path: Path) -> tuple[Mapping[str, Any], bytes]:
    """Load the exact D237 contract from a stable single-link file."""

    raw = read_stable_single_link_file(
        path,
        label="D237 contract",
        max_bytes=_MAX_CONTRACT_BYTES,
    )
    actual_hash = _sha256_bytes(raw)
    if not hmac.compare_digest(actual_hash, CONTRACT_FILE_SHA256):
        raise Rfc3161RequestError("D237 contract SHA-256 differs")
    return _strict_json(raw, "D237 contract"), raw


def _encode_length(length: int) -> bytes:
    if length < 0:
        raise Rfc3161RequestError("DER length must not be negative")
    if length < 128:
        return bytes((length,))
    encoded = length.to_bytes((length.bit_length() + 7) // 8, "big")
    if len(encoded) > 4:
        raise Rfc3161RequestError("DER length is too large")
    return bytes((0x80 | len(encoded),)) + encoded


def _tlv(tag: int, value: bytes) -> bytes:
    if tag < 0 or tag > 0xFF or tag & 0x1F == 0x1F:
        raise Rfc3161RequestError("only low-tag-number DER is permitted")
    return bytes((tag,)) + _encode_length(len(value)) + value


def _encode_positive_integer(value: int) -> bytes:
    if value <= 0:
        raise Rfc3161RequestError("INTEGER must be positive")
    encoded = value.to_bytes((value.bit_length() + 7) // 8, "big")
    if encoded[0] & 0x80:
        encoded = b"\x00" + encoded
    return _tlv(0x02, encoded)


def _base128(value: int) -> bytes:
    if value < 0:
        raise Rfc3161RequestError("OID arc must not be negative")
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(chunks))


def _encode_oid(oid: str) -> bytes:
    if not _OID.fullmatch(oid):
        raise Rfc3161RequestError("OID text is not canonical")
    arcs = [int(part) for part in oid.split(".")]
    if len(arcs) < 2 or arcs[0] > 2 or (arcs[0] < 2 and arcs[1] > 39):
        raise Rfc3161RequestError("OID arcs are invalid")
    body = _base128(40 * arcs[0] + arcs[1])
    for arc in arcs[2:]:
        body += _base128(arc)
    return _tlv(0x06, body)


def _read_tlv(data: bytes, offset: int, label: str) -> tuple[int, bytes, int]:
    if offset < 0 or offset + 2 > len(data):
        raise Rfc3161RequestError(f"{label} is truncated")
    tag = data[offset]
    if tag & 0x1F == 0x1F:
        raise Rfc3161RequestError(f"{label} uses high-tag-number form")
    first_length = data[offset + 1]
    cursor = offset + 2
    if first_length == 0x80:
        raise Rfc3161RequestError(f"{label} uses BER indefinite length")
    if first_length < 0x80:
        length = first_length
    else:
        count = first_length & 0x7F
        if count == 0 or count > 4 or cursor + count > len(data):
            raise Rfc3161RequestError(f"{label} length is invalid")
        length_bytes = data[cursor : cursor + count]
        cursor += count
        if length_bytes[0] == 0:
            raise Rfc3161RequestError(f"{label} length has a leading zero")
        length = int.from_bytes(length_bytes, "big")
        if length < 128:
            raise Rfc3161RequestError(f"{label} length is not shortest form")
    end = cursor + length
    if end > len(data):
        raise Rfc3161RequestError(f"{label} value overruns input")
    return tag, data[cursor:end], end


def _expect_single_tlv(data: bytes, expected_tag: int, label: str) -> bytes:
    tag, value, end = _read_tlv(data, 0, label)
    if tag != expected_tag:
        raise Rfc3161RequestError(f"{label} has unexpected tag 0x{tag:02x}")
    if end != len(data):
        raise Rfc3161RequestError(f"{label} contains trailing bytes")
    return value


def _read_field(
    content: bytes,
    offset: int,
    expected_tag: int,
    label: str,
) -> tuple[bytes, int]:
    tag, value, end = _read_tlv(content, offset, label)
    if tag != expected_tag:
        raise Rfc3161RequestError(f"{label} has unexpected tag 0x{tag:02x}")
    return value, end


def _decode_positive_integer(value: bytes, label: str) -> int:
    if not value:
        raise Rfc3161RequestError(f"{label} INTEGER is empty")
    if value[0] & 0x80:
        raise Rfc3161RequestError(f"{label} INTEGER is negative")
    if len(value) > 1 and value[0] == 0 and not value[1] & 0x80:
        raise Rfc3161RequestError(f"{label} INTEGER is not minimally encoded")
    result = int.from_bytes(value, "big")
    if result <= 0:
        raise Rfc3161RequestError(f"{label} INTEGER must be positive")
    return result


def _decode_oid(value: bytes, label: str) -> str:
    if not value:
        raise Rfc3161RequestError(f"{label} OID is empty")
    subidentifiers: list[int] = []
    accumulator = 0
    at_start = True
    for octet in value:
        if at_start and octet == 0x80:
            raise Rfc3161RequestError(f"{label} OID is not minimally encoded")
        accumulator = (accumulator << 7) | (octet & 0x7F)
        at_start = False
        if not octet & 0x80:
            subidentifiers.append(accumulator)
            accumulator = 0
            at_start = True
    if not at_start or not subidentifiers:
        raise Rfc3161RequestError(f"{label} OID is truncated")
    first = subidentifiers[0]
    if first < 40:
        arcs = [0, first]
    elif first < 80:
        arcs = [1, first - 40]
    else:
        arcs = [2, first - 80]
    arcs.extend(subidentifiers[1:])
    oid = ".".join(str(arc) for arc in arcs)
    if _expect_single_tlv(_encode_oid(oid), 0x06, f"{label} OID replay") != value:
        raise Rfc3161RequestError(f"{label} OID does not round-trip")
    return oid


def _encode_request_fields(*, hashed_message: bytes, policy_oid: str, nonce: int) -> bytes:
    if len(hashed_message) != 32:
        raise Rfc3161RequestError("hashedMessage must be 32 bytes")
    algorithm_identifier = _tlv(0x30, _encode_oid(SHA256_OID))
    message_imprint = _tlv(
        0x30,
        algorithm_identifier + _tlv(0x04, hashed_message),
    )
    request_content = b"".join(
        (
            _encode_positive_integer(1),
            message_imprint,
            _encode_oid(policy_oid),
            _encode_positive_integer(nonce),
            _tlv(0x01, b"\xff"),
        )
    )
    request = _tlv(0x30, request_content)
    if len(request) > _MAX_REQUEST_BYTES:
        raise Rfc3161RequestError("request exceeds the byte ceiling")
    return request


def build_timestamp_request_der(
    *,
    final_node_preimage: bytes,
    merkle_root: bytes,
    request_policy_oid: str,
    nonce: int,
) -> bytes:
    """Build the strict D237 TimeStampReq profile without submitting it."""

    if len(final_node_preimage) != 65 or final_node_preimage[0] != 0x01:
        raise Rfc3161RequestError("RFC 9162 final-node preimage must be 65 bytes")
    if len(merkle_root) != 32:
        raise Rfc3161RequestError("Merkle root must be 32 bytes")
    if not hmac.compare_digest(hashlib.sha256(final_node_preimage).digest(), merkle_root):
        raise Rfc3161RequestError("final-node preimage does not hash to Merkle root")
    if request_policy_oid == "0.0" or not _OID.fullmatch(request_policy_oid):
        raise Rfc3161RequestError("request policy OID is invalid")

    return _encode_request_fields(
        hashed_message=merkle_root,
        policy_oid=request_policy_oid,
        nonce=nonce,
    )


def parse_timestamp_request_der(data: bytes) -> dict[str, object]:
    """Strictly parse and canonically replay the D237 TimeStampReq profile."""

    if not isinstance(data, bytes):
        raise Rfc3161RequestError("request DER must be immutable bytes")
    if not data or len(data) > _MAX_REQUEST_BYTES:
        raise Rfc3161RequestError("request DER size is outside the allowed range")
    content = _expect_single_tlv(data, 0x30, "TimeStampReq")
    cursor = 0

    version_bytes, cursor = _read_field(content, cursor, 0x02, "version")
    version = _decode_positive_integer(version_bytes, "version")
    if version != 1:
        raise Rfc3161RequestError("TimeStampReq version must be 1")

    imprint_bytes, cursor = _read_field(content, cursor, 0x30, "messageImprint")
    imprint_cursor = 0
    algorithm_bytes, imprint_cursor = _read_field(
        imprint_bytes,
        imprint_cursor,
        0x30,
        "hashAlgorithm",
    )
    algorithm_oid_bytes, algorithm_end = _read_field(
        algorithm_bytes,
        0,
        0x06,
        "hashAlgorithm OID",
    )
    if algorithm_end != len(algorithm_bytes):
        raise Rfc3161RequestError("SHA-256 AlgorithmIdentifier parameters must be absent")
    algorithm_oid = _decode_oid(algorithm_oid_bytes, "hashAlgorithm")
    if algorithm_oid != SHA256_OID:
        raise Rfc3161RequestError("messageImprint must use SHA-256")
    hashed_message, imprint_cursor = _read_field(
        imprint_bytes,
        imprint_cursor,
        0x04,
        "hashedMessage",
    )
    if imprint_cursor != len(imprint_bytes) or len(hashed_message) != 32:
        raise Rfc3161RequestError("hashedMessage must be exactly 32 bytes")

    policy_bytes, cursor = _read_field(content, cursor, 0x06, "reqPolicy")
    policy_oid = _decode_oid(policy_bytes, "reqPolicy")
    nonce_bytes, cursor = _read_field(content, cursor, 0x02, "nonce")
    nonce = _decode_positive_integer(nonce_bytes, "nonce")
    cert_req, cursor = _read_field(content, cursor, 0x01, "certReq")
    if cert_req != b"\xff":
        raise Rfc3161RequestError("certReq must be canonical DER TRUE")
    if cursor != len(content):
        raise Rfc3161RequestError("extensions or trailing fields are forbidden")

    rebuilt = _encode_request_fields(
        hashed_message=hashed_message,
        policy_oid=policy_oid,
        nonce=nonce,
    )
    if not hmac.compare_digest(rebuilt, data):
        raise Rfc3161RequestError("request is not the canonical D237 DER encoding")
    return {
        "version": version,
        "message_imprint_hash_algorithm_oid": algorithm_oid,
        "message_imprint_hashed_message_hex": hashed_message.hex(),
        "request_policy_oid": policy_oid,
        "nonce_decimal": str(nonce),
        "nonce_bit_length": nonce.bit_length(),
        "cert_req": True,
        "extensions_present": False,
        "der_size_bytes": len(data),
        "der_sha256": _sha256_bytes(data),
    }


def build_and_validate_synthetic_fixture() -> dict[str, object]:
    """Build the exact fixed vector and replay every semantic binding."""

    preimage = bytes.fromhex(SYNTHETIC_PREIMAGE_HEX)
    root = bytes.fromhex(SYNTHETIC_ROOT)
    request = build_timestamp_request_der(
        final_node_preimage=preimage,
        merkle_root=root,
        request_policy_oid=SYNTHETIC_POLICY_OID,
        nonce=int(SYNTHETIC_NONCE_DECIMAL),
    )
    if not hmac.compare_digest(request.hex(), SYNTHETIC_REQUEST_DER_HEX):
        raise Rfc3161RequestError("synthetic request DER differs from fixed vector")
    if not hmac.compare_digest(_sha256_bytes(request), SYNTHETIC_REQUEST_DER_SHA256):
        raise Rfc3161RequestError("synthetic request SHA-256 differs")
    parsed = parse_timestamp_request_der(request)
    _equal(parsed["message_imprint_hashed_message_hex"], SYNTHETIC_ROOT, "imprint")
    _equal(parsed["request_policy_oid"], SYNTHETIC_POLICY_OID, "policy")
    _equal(parsed["nonce_decimal"], SYNTHETIC_NONCE_DECIMAL, "nonce")
    return {
        "evidence_class": "SYNTHETIC_CONFORMANCE_VECTOR_ONLY",
        "final_node_preimage_sha256": _sha256_bytes(preimage),
        "request": parsed,
        "real_request_authorized_for_submission": False,
        "timestamp_authority_requests": 0,
    }


def validate_contract(contract: Mapping[str, Any]) -> dict[str, object]:
    """Validate every security-relevant D237 contract assertion."""

    _exact_keys(
        contract,
        {
            "schema_version",
            "status",
            "decision",
            "purpose",
            "input_bindings",
            "request_profile",
            "strict_der_rejections",
            "runtime_capability_preflight",
            "synthetic_fixture",
            "standards",
            "execution_budget",
            "authority_boundary",
        },
        "contract",
    )
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), CONTRACT_DECISION, "contract decision")
    content_id = _sha256_bytes(canonical_json_bytes(contract))
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")

    bindings = _mapping(contract.get("input_bindings"), "input bindings")
    expected_bindings = {
        "external_time_batch_decision": "D-20260805-235",
        "external_time_batch_contract_sha256": D235_CONTRACT_SHA256,
        "external_time_batch_contract_content_id": D235_CONTRACT_CONTENT_ID,
        "external_time_batch_proof_content_id": D235_PROOF_CONTENT_ID,
        "external_time_batch_proof_manifest_sha256": D235_PROOF_MANIFEST_SHA256,
    }
    _equal(dict(bindings), expected_bindings, "D235 bindings")

    profile = _mapping(contract.get("request_profile"), "request profile")
    _equal(profile.get("version"), 1, "request version")
    _equal(profile.get("message_imprint_hash_algorithm_oid"), SHA256_OID, "SHA-256 OID")
    _equal(
        profile.get("generated_sha2_algorithm_identifier_parameters"),
        "ABSENT",
        "generated parameters",
    )
    _equal(
        profile.get("d237_exact_generated_request_replay_parameters"),
        ["ABSENT"],
        "exact replay parameters",
    )
    _equal(
        profile.get("general_rfc5754_interoperable_parser_must_accept_parameters"),
        ["ABSENT", "NULL"],
        "general interoperable parameters",
    )
    _equal(profile.get("timestamped_datum_size_bytes"), 65, "datum size")
    _equal(profile.get("hashed_message_must_equal_sha256_of_timestamped_datum"), True, "datum hash binding")
    _equal(profile.get("hashed_message_must_equal_d235_merkle_root"), True, "root binding")
    _equal(profile.get("cert_req"), True, "certReq")
    _equal(profile.get("extensions_allowed"), False, "extensions")
    _equal(profile.get("maximum_request_der_bytes"), _MAX_REQUEST_BYTES, "request ceiling")

    fixture = _mapping(contract.get("synthetic_fixture"), "synthetic fixture")
    expected_fixture = {
        "evidence_class": "SYNTHETIC_CONFORMANCE_VECTOR_ONLY",
        "tree_size": 4,
        "left_subtree_hash_hex": SYNTHETIC_LEFT_HASH,
        "right_subtree_hash_hex": SYNTHETIC_RIGHT_HASH,
        "final_node_preimage_hex": SYNTHETIC_PREIMAGE_HEX,
        "merkle_root_sha256": SYNTHETIC_ROOT,
        "request_policy_oid": SYNTHETIC_POLICY_OID,
        "request_policy_is_synthetic_placeholder": True,
        "nonce_decimal": SYNTHETIC_NONCE_DECIMAL,
        "nonce_is_synthetic_fixed_value": True,
        "request_der_hex": SYNTHETIC_REQUEST_DER_HEX,
        "request_der_size_bytes": 89,
        "request_der_sha256": SYNTHETIC_REQUEST_DER_SHA256,
    }
    _equal(dict(fixture), expected_fixture, "synthetic fixture")
    synthetic = build_and_validate_synthetic_fixture()
    _equal(synthetic["final_node_preimage_sha256"], SYNTHETIC_ROOT, "preimage root")

    capabilities = _mapping(
        contract.get("runtime_capability_preflight"),
        "runtime capability preflight",
    )
    _equal(capabilities.get("current_project_dependency"), "cryptography==47.0.0", "dependency")
    for claim in (
        "cryptography_pkcs7_certificate_extraction_is_token_verification",
        "candidate_parser_admitted_by_d237",
        "candidate_end_to_end_validator_admitted_by_d237",
        "real_token_acceptance_with_current_stack",
        "shell_openssl_allowed",
    ):
        _equal(capabilities.get(claim), False, claim)

    execution = _mapping(contract.get("execution_budget"), "execution budget")
    if not execution or any(type(value) is not int or value != 0 for value in execution.values()):
        raise Rfc3161RequestError("all execution counters must be integer zero")

    authority = _mapping(contract.get("authority_boundary"), "authority boundary")
    for claim, value in authority.items():
        expected = claim == "synthetic_request_der_verified"
        if type(value) is not bool or value is not expected:
            raise Rfc3161RequestError(f"authority claim {claim} differs")

    _equal(
        contract.get("standards"),
        [
            "https://www.rfc-editor.org/rfc/rfc3161#section-2.4.1",
            "https://www.rfc-editor.org/rfc/rfc5754#section-2",
            "https://www.rfc-editor.org/rfc/rfc5652#section-5.4",
            "https://www.rfc-editor.org/rfc/rfc5816",
        ],
        "standards",
    )
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "synthetic_fixture": synthetic,
        "real_token_acceptance_with_current_stack": False,
        "timestamp_authority_requests": 0,
        "network_calls": 0,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_d235_parent(*, contract_path: Path, proof_manifest_path: Path) -> dict[str, object]:
    """Validate exact D235 bytes and its local zero-execution proof."""

    contract_raw = read_stable_single_link_file(
        contract_path,
        label="D235 contract",
        max_bytes=_MAX_CONTRACT_BYTES,
    )
    if not hmac.compare_digest(_sha256_bytes(contract_raw), D235_CONTRACT_SHA256):
        raise Rfc3161RequestError("D235 contract SHA-256 differs")
    d235_contract = _strict_json(contract_raw, "D235 contract")
    _equal(_sha256_bytes(canonical_json_bytes(d235_contract)), D235_CONTRACT_CONTENT_ID, "D235 content ID")

    manifest_raw = read_stable_single_link_file(
        proof_manifest_path,
        label="D235 proof manifest",
        max_bytes=_MAX_CONTRACT_BYTES,
    )
    if not hmac.compare_digest(_sha256_bytes(manifest_raw), D235_PROOF_MANIFEST_SHA256):
        raise Rfc3161RequestError("D235 proof manifest SHA-256 differs")
    manifest = _strict_json(manifest_raw, "D235 proof manifest")
    _equal(manifest.get("content_id"), D235_PROOF_CONTENT_ID, "D235 proof content ID")
    execution = _mapping(manifest.get("execution"), "D235 execution")
    if any(type(value) is not int or value != 0 for value in execution.values()):
        raise Rfc3161RequestError("D235 proof is not zero execution")
    authorities = _mapping(manifest.get("authorities"), "D235 authorities")
    for claim, value in authorities.items():
        expected = claim == "synthetic_merkle_fixture_verified"
        if type(value) is not bool or value is not expected:
            raise Rfc3161RequestError(f"D235 authority claim {claim} differs")
    return {
        "d235_contract_content_id": D235_CONTRACT_CONTENT_ID,
        "d235_proof_content_id": D235_PROOF_CONTENT_ID,
        "databricks_statements": 0,
        "network_calls": 0,
        "h_drive_accesses": 0,
        "timestamp_authority_requests": 0,
    }


def _proof_json_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _write_once(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        existing = read_stable_single_link_file(
            path,
            label=f"existing D237 artifact {path.name}",
            max_bytes=2_000_000,
        )
        if not hmac.compare_digest(existing, payload):
            raise Rfc3161RequestError(
                f"existing D237 artifact differs: {path.name}"
            ) from None


def materialize_proof(*, output_root: Path) -> Path:
    """Materialize one deterministic content-addressed local D237 proof."""

    repository_root = assert_absolute_path_has_no_links(Path.cwd().resolve())
    expected_root = Path(r"C:\Users\jbattaglia\PFC_LT")
    if repository_root != expected_root:
        raise Rfc3161RequestError("cwd must be the canonical workspace")
    git_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repository_root,
            text=True,
        ).strip()
    ).resolve()
    if git_root != repository_root:
        raise Rfc3161RequestError("Git root must be the canonical workspace")
    output = assert_absolute_path_has_no_links(output_root.resolve())
    try:
        output.relative_to(repository_root / "build")
    except ValueError as exc:
        raise Rfc3161RequestError("D237 output must remain below build") from exc

    planning = repository_root / ".planning" / "phases" / "14-lt-audit-remediation"
    contract_path = planning / "EEX-ENTSOE-RFC3161-REQUEST-DER-PREFLIGHT-CONTRACT-V1.json"
    d235_contract_path = planning / "EEX-ENTSOE-EXTERNAL-TIME-BATCH-CONTRACT-V1.json"
    d235_manifest_path = (
        repository_root
        / "build"
        / "databricks-eex-daily"
        / "2026-08-05"
        / "eex-entsoe-external-time-batch-proofs"
        / D235_PROOF_CONTENT_ID
        / "manifest.json"
    )
    contract, contract_raw = load_contract(contract_path)
    contract_assessment = validate_contract(contract)
    parent_validation = validate_d235_parent(
        contract_path=d235_contract_path,
        proof_manifest_path=d235_manifest_path,
    )

    source_paths = {
        "contract": contract_path,
        "validator": Path(__file__).resolve(),
        "tests": repository_root / "tests" / "test_eex_entsoe_rfc3161_request_der.py",
    }
    source_hashes = {
        name: _sha256_bytes(
            read_stable_single_link_file(
                path,
                label=f"D237 {name}",
                max_bytes=2_000_000,
            )
        )
        for name, path in source_paths.items()
    }
    execution = {
        "databricks_connections": 0,
        "databricks_statements": 0,
        "warehouse_starts": 0,
        "network_calls": 0,
        "timestamp_authority_requests": 0,
        "h_drive_accesses": 0,
        "remote_writes": 0,
        "real_dsse_envelopes_opened": 0,
        "real_timestamp_tokens_opened": 0,
        "market_value_rows_opened": 0,
    }
    authorities = {
        "synthetic_request_der_verified": True,
        "approved_real_tsa_policy_present": False,
        "real_unpredictable_nonce_generated": False,
        "real_request_authorized_for_submission": False,
        "real_request_submitted": False,
        "real_timestamp_token_present": False,
        "rfc3161_token_cryptographically_verified": False,
        "independently_trusted_time_verified": False,
        "point_in_time_availability_verified": False,
        "model_input_authorized": False,
        "candidate_authorized": False,
        "production_authorized": False,
    }
    assessment = {
        "schema_version": _PROOF_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "contract_assessment": contract_assessment,
        "parent_validation": parent_validation,
        "synthetic_request": build_and_validate_synthetic_fixture(),
        "source_hashes": source_hashes,
        "execution": execution,
        "authorities": authorities,
    }
    assessment_bytes = _proof_json_bytes(assessment)
    content_id = _sha256_bytes(assessment_bytes)
    proof_dir = output / content_id
    proof_dir.mkdir(parents=True, exist_ok=True)
    _write_once(proof_dir / "assessment.json", assessment_bytes)
    _write_once(proof_dir / "contract.json", contract_raw)

    manifest = {
        "schema_version": _PROOF_SCHEMA,
        "content_id": content_id,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "artifacts": {
            "assessment.json": {
                "sha256": _sha256_bytes(assessment_bytes),
                "size_bytes": len(assessment_bytes),
            },
            "contract.json": {
                "sha256": _sha256_bytes(contract_raw),
                "size_bytes": len(contract_raw),
            },
        },
        "source_hashes": source_hashes,
        "execution": execution,
        "authorities": authorities,
    }
    _write_once(proof_dir / "manifest.json", _proof_json_bytes(manifest))
    return proof_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(materialize_proof(output_root=arguments.output_root))
    return 0


__all__ = [
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "Rfc3161RequestError",
    "SYNTHETIC_NONCE_DECIMAL",
    "SYNTHETIC_PREIMAGE_HEX",
    "SYNTHETIC_REQUEST_DER_HEX",
    "SYNTHETIC_REQUEST_DER_SHA256",
    "SYNTHETIC_ROOT",
    "build_and_validate_synthetic_fixture",
    "build_timestamp_request_der",
    "canonical_json_bytes",
    "load_contract",
    "materialize_proof",
    "parse_timestamp_request_der",
    "validate_contract",
    "validate_d235_parent",
]


if __name__ == "__main__":
    raise SystemExit(main())
