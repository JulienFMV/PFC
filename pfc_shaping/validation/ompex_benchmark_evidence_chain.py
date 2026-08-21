"""Non-production reference verifier for the OMPEX benchmark evidence chain.

The reference wire proves that three independently signed receipts can bind a
candidate freeze, an at-origin OMPEX observation and post-delivery realised
truth with strict chronology.  Its schemas and signing domains are deliberately
incompatible with production.  A locally valid chain never makes an origin
countable and never authorizes model selection, promotion or production.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.pipeline.strict_structured_data import load_strict_json

ASSESSMENT_SCHEMA_VERSION = "ompex_benchmark_evidence_chain_assessment.reference.v1"
CANDIDATE_FREEZE_SCHEMA = "ompex_benchmark_candidate_freeze.reference.v1"
OMPEX_AVAILABILITY_SCHEMA = "ompex_benchmark_availability.reference.v1"
TRUTH_PUBLICATION_SCHEMA = "ompex_benchmark_truth_publication.reference.v1"
AUTHORITY_CLASSIFICATION = "NON_PRODUCTION_LOCAL_REFERENCE_NEVER_COUNTABLE"
SIGNATURE_ALGORITHM = "Ed25519"

CANDIDATE_DOMAIN = b"FMV_OMPEX_BENCHMARK_CANDIDATE_FREEZE_REFERENCE_V1"
OMPEX_DOMAIN = b"FMV_OMPEX_BENCHMARK_AVAILABILITY_REFERENCE_V1"
TRUTH_DOMAIN = b"FMV_OMPEX_BENCHMARK_TRUTH_PUBLICATION_REFERENCE_V1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024

_COMMON_CORE_FIELDS = {
    "schema_version",
    "authority_classification",
    "evidence_role",
    "test_fixture",
    "origin_id",
    "origin_utc",
    "target_start_utc",
    "target_end_utc",
    "signer_organization_role",
}
_SIGNED_FIELDS = {"receipt_id", "signature"}
_CANDIDATE_FIELDS = _COMMON_CORE_FIELDS | _SIGNED_FIELDS | {
    "candidate_commitment_id",
    "candidate_commitment_sha256",
    "candidate_artifact_sha256",
    "candidate_configuration_manifest_sha256",
    "candidate_input_manifest_sha256",
    "data_cutoff_utc",
    "frozen_at_utc",
    "ompex_opened_before_freeze",
    "truth_opened_before_freeze",
}
_OMPEX_FIELDS = _COMMON_CORE_FIELDS | _SIGNED_FIELDS | {
    "candidate_freeze_receipt_id",
    "filename",
    "source_size_bytes",
    "ompex_sha256",
    "first_available_at_utc",
    "observed_at_utc",
    "opened_at_utc",
    "timestamp_semantics_contract_sha256",
    "asserted_hour_end_semantics_authenticated",
    "asserted_availability_at_origin_authenticated",
    "truth_opened_before_observation",
}
_TRUTH_FIELDS = _COMMON_CORE_FIELDS | _SIGNED_FIELDS | {
    "candidate_freeze_receipt_id",
    "ompex_availability_receipt_id",
    "truth_source_id",
    "source_size_bytes",
    "truth_sha256",
    "native_resolution",
    "timezone",
    "currency_unit",
    "published_at_utc",
    "opened_at_utc",
    "independent_of_candidate",
    "independent_of_ompex",
    "revision_status",
}


class OmpexBenchmarkEvidenceChainError(ValueError):
    """Raised when a reference evidence chain is ambiguous or invalid."""


def verify_reference_evidence_chain(
    *,
    candidate_receipt_payload: bytes,
    ompex_receipt_payload: bytes,
    truth_receipt_payload: bytes,
    candidate_public_key: Ed25519PublicKey,
    ompex_public_key: Ed25519PublicKey,
    truth_public_key: Ed25519PublicKey,
    expected_candidate_receipt_sha256: str,
    expected_ompex_receipt_sha256: str,
    expected_truth_receipt_sha256: str,
    expected_candidate_commitment_id: str,
    expected_candidate_commitment_sha256: str,
    expected_ompex_sha256: str,
    expected_timestamp_semantics_contract_sha256: str,
    expected_truth_sha256: str,
    expected_origin_id: str,
    expected_origin_utc: str,
    expected_target_start_utc: str,
    expected_target_end_utc: str,
) -> dict[str, object]:
    """Verify signatures, bindings and chronology without opening price data."""

    expected_hashes = {
        "candidate receipt": expected_candidate_receipt_sha256,
        "OMPEX receipt": expected_ompex_receipt_sha256,
        "truth receipt": expected_truth_receipt_sha256,
        "candidate commitment ID": expected_candidate_commitment_id,
        "candidate commitment": expected_candidate_commitment_sha256,
        "OMPEX artifact": expected_ompex_sha256,
        "OMPEX timestamp semantics contract": (
            expected_timestamp_semantics_contract_sha256
        ),
        "truth artifact": expected_truth_sha256,
        "origin ID": expected_origin_id,
    }
    for label, value in expected_hashes.items():
        _require_sha256(value, label=label)
    expected_origin = _utc(expected_origin_utc, label="expected origin")
    expected_start = _utc(expected_target_start_utc, label="expected target start")
    expected_end = _utc(expected_target_end_utc, label="expected target end")
    if not expected_origin < expected_start < expected_end:
        raise OmpexBenchmarkEvidenceChainError(
            "expected origin/target chronology is invalid"
        )

    keys = (candidate_public_key, ompex_public_key, truth_public_key)
    if any(not isinstance(key, Ed25519PublicKey) for key in keys):
        raise OmpexBenchmarkEvidenceChainError("all trust keys must be Ed25519")
    key_ids = tuple(public_key_id(key) for key in keys)
    if len(set(key_ids)) != 3:
        raise OmpexBenchmarkEvidenceChainError(
            "candidate, OMPEX and truth authorities must use distinct keys"
        )

    candidate = _verify_receipt(
        candidate_receipt_payload,
        expected_sha256=expected_candidate_receipt_sha256,
        expected_fields=_CANDIDATE_FIELDS,
        schema=CANDIDATE_FREEZE_SCHEMA,
        evidence_role="candidate_freeze",
        signer_role="independent_candidate_registry_reference",
        domain=CANDIDATE_DOMAIN,
        public_key=candidate_public_key,
        label="candidate freeze receipt",
    )
    ompex = _verify_receipt(
        ompex_receipt_payload,
        expected_sha256=expected_ompex_receipt_sha256,
        expected_fields=_OMPEX_FIELDS,
        schema=OMPEX_AVAILABILITY_SCHEMA,
        evidence_role="ompex_availability",
        signer_role="desk_or_vendor_availability_reference",
        domain=OMPEX_DOMAIN,
        public_key=ompex_public_key,
        label="OMPEX availability receipt",
    )
    truth = _verify_receipt(
        truth_receipt_payload,
        expected_sha256=expected_truth_receipt_sha256,
        expected_fields=_TRUTH_FIELDS,
        schema=TRUTH_PUBLICATION_SCHEMA,
        evidence_role="truth_publication",
        signer_role="independent_truth_publication_reference",
        domain=TRUTH_DOMAIN,
        public_key=truth_public_key,
        label="truth publication receipt",
    )

    for label, document in (
        ("candidate", candidate),
        ("OMPEX", ompex),
        ("truth", truth),
    ):
        if (
            document.get("origin_id") != expected_origin_id
            or document.get("origin_utc") != expected_origin_utc
            or document.get("target_start_utc") != expected_target_start_utc
            or document.get("target_end_utc") != expected_target_end_utc
        ):
            raise OmpexBenchmarkEvidenceChainError(
                f"{label} origin/target binding differs"
            )
    if not (
        candidate.get("test_fixture")
        is ompex.get("test_fixture")
        is truth.get("test_fixture")
    ):
        raise OmpexBenchmarkEvidenceChainError(
            "receipt fixture classifications differ"
        )

    _require_equal(
        candidate.get("candidate_commitment_id"),
        expected_candidate_commitment_id,
        label="candidate commitment ID binding",
    )
    _require_equal(
        candidate.get("candidate_commitment_sha256"),
        expected_candidate_commitment_sha256,
        label="candidate commitment binding",
    )
    for field in (
        "candidate_artifact_sha256",
        "candidate_configuration_manifest_sha256",
        "candidate_input_manifest_sha256",
    ):
        _require_sha256(candidate.get(field), label=field)
    if (
        candidate.get("ompex_opened_before_freeze") is not False
        or candidate.get("truth_opened_before_freeze") is not False
    ):
        raise OmpexBenchmarkEvidenceChainError(
            "candidate freeze is contaminated by benchmark or truth access"
        )
    data_cutoff = _utc(str(candidate.get("data_cutoff_utc")), label="candidate data cutoff")
    frozen_at = _utc(str(candidate.get("frozen_at_utc")), label="candidate frozen-at")
    if data_cutoff > expected_origin or data_cutoff > frozen_at or frozen_at > expected_start:
        raise OmpexBenchmarkEvidenceChainError(
            "candidate data cutoff/freeze chronology is invalid"
        )

    _require_equal(
        ompex.get("candidate_freeze_receipt_id"),
        candidate.get("receipt_id"),
        label="OMPEX candidate-freeze receipt binding",
    )
    _require_equal(ompex.get("ompex_sha256"), expected_ompex_sha256, label="OMPEX binding")
    _require_source_size(ompex.get("source_size_bytes"), label="OMPEX source size")
    filename = str(ompex.get("filename"))
    if (
        not filename.lower().endswith(".xlsx")
        or filename in {"", ".", ".."}
        or "/" in filename
        or "\\" in filename
        or ":" in filename
    ):
        raise OmpexBenchmarkEvidenceChainError("OMPEX filename is invalid")
    _require_equal(
        ompex.get("timestamp_semantics_contract_sha256"),
        expected_timestamp_semantics_contract_sha256,
        label="OMPEX timestamp semantics contract binding",
    )
    if (
        ompex.get("asserted_hour_end_semantics_authenticated") is not True
        or ompex.get("asserted_availability_at_origin_authenticated") is not True
        or ompex.get("truth_opened_before_observation") is not False
    ):
        raise OmpexBenchmarkEvidenceChainError(
            "OMPEX semantic/availability assertions are incomplete"
        )
    first_available = _utc(
        str(ompex.get("first_available_at_utc")), label="OMPEX first available"
    )
    observed_at = _utc(str(ompex.get("observed_at_utc")), label="OMPEX observed-at")
    opened_at = _utc(str(ompex.get("opened_at_utc")), label="OMPEX opened-at")
    if (
        first_available > expected_origin
        or observed_at != opened_at
        or opened_at < frozen_at
        or opened_at < first_available
    ):
        raise OmpexBenchmarkEvidenceChainError("OMPEX observation chronology is invalid")

    _require_equal(
        truth.get("candidate_freeze_receipt_id"),
        candidate.get("receipt_id"),
        label="truth candidate-freeze receipt binding",
    )
    _require_equal(
        truth.get("ompex_availability_receipt_id"),
        ompex.get("receipt_id"),
        label="truth OMPEX receipt binding",
    )
    _require_equal(truth.get("truth_sha256"), expected_truth_sha256, label="truth binding")
    _require_source_size(truth.get("source_size_bytes"), label="truth source size")
    if _SOURCE_ID.fullmatch(str(truth.get("truth_source_id"))) is None:
        raise OmpexBenchmarkEvidenceChainError("truth source ID is invalid")
    if (
        truth.get("native_resolution") != "PT1H"
        or truth.get("timezone") != "UTC"
        or truth.get("currency_unit") != "EUR/MWh"
        or truth.get("revision_status") != "FROZEN_REGISTERED_REVISION"
        or truth.get("independent_of_candidate") is not True
        or truth.get("independent_of_ompex") is not True
    ):
        raise OmpexBenchmarkEvidenceChainError("truth semantic boundary is invalid")
    published_at = _utc(str(truth.get("published_at_utc")), label="truth published-at")
    truth_opened_at = _utc(str(truth.get("opened_at_utc")), label="truth opened-at")
    if (
        published_at < expected_end
        or truth_opened_at < published_at
        or truth_opened_at < opened_at
    ):
        raise OmpexBenchmarkEvidenceChainError("truth publication chronology is invalid")

    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": "PASS_LOCAL_REFERENCE_STRUCTURE_ONLY_NO_COUNTABLE_ORIGIN",
        "authority_classification": AUTHORITY_CLASSIFICATION,
        "origin_id": expected_origin_id,
        "origin_utc": expected_origin_utc,
        "target_start_utc": expected_target_start_utc,
        "target_end_utc": expected_target_end_utc,
        "receipt_bindings": {
            "candidate": _binding(candidate_receipt_payload, candidate, key_ids[0]),
            "ompex": _binding(ompex_receipt_payload, ompex, key_ids[1]),
            "truth": _binding(truth_receipt_payload, truth, key_ids[2]),
        },
        "chronology": {
            "data_cutoff_not_after_origin": True,
            "candidate_frozen_before_target": True,
            "candidate_frozen_before_ompex_open": True,
            "ompex_asserted_available_at_origin": True,
            "truth_published_after_target_end": True,
            "truth_opened_after_publication": True,
        },
        "role_keys_distinct": True,
        "price_values_opened": False,
        "countable_origin_count": 0,
        "external_authority_verified": False,
        "scientific_scoring_authorized": False,
        "model_selection_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "blockers": [
            "reference schemas and local keys are deliberately incompatible with external production authority",
            "external trust-anchor registry and trusted-time/WORM receipts are absent",
            "no real candidate, OMPEX vintage or truth receipt has been admitted",
        ],
    }


def canonical_json(value: object) -> bytes:
    """Return the exact ASCII canonical JSON used by the reference wire."""

    _validate_no_float(value, label="canonical JSON")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise OmpexBenchmarkEvidenceChainError("canonical JSON value is invalid") from exc


def receipt_id(*, domain: bytes, core: Mapping[str, object]) -> str:
    """Compute one domain-separated reference receipt identity."""

    return hashlib.sha256(domain + b"\x00" + canonical_json(core)).hexdigest()


def signature_payload(*, domain: bytes, unsigned_receipt: Mapping[str, object]) -> bytes:
    """Return the domain-separated bytes signed by a reference authority."""

    return domain + b"\x00" + canonical_json(unsigned_receipt)


def public_key_id(public_key: Ed25519PublicKey) -> str:
    """Return the SHA-256 identity of an Ed25519 raw public key."""

    if not isinstance(public_key, Ed25519PublicKey):
        raise OmpexBenchmarkEvidenceChainError("public key is not Ed25519")
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _verify_receipt(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_fields: set[str],
    schema: str,
    evidence_role: str,
    signer_role: str,
    domain: bytes,
    public_key: Ed25519PublicKey,
    label: str,
) -> dict[str, object]:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise OmpexBenchmarkEvidenceChainError(f"{label} byte size is invalid")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise OmpexBenchmarkEvidenceChainError(f"{label} SHA-256 mismatch")
    document = _strict_mapping(payload, label=label)
    if canonical_json(document) != payload:
        raise OmpexBenchmarkEvidenceChainError(f"{label} is not exact canonical JSON")
    if set(document) != expected_fields:
        raise OmpexBenchmarkEvidenceChainError(f"{label} fields are not exact")
    if (
        document.get("schema_version") != schema
        or document.get("authority_classification") != AUTHORITY_CLASSIFICATION
        or document.get("evidence_role") != evidence_role
        or document.get("signer_organization_role") != signer_role
        or type(document.get("test_fixture")) is not bool
    ):
        raise OmpexBenchmarkEvidenceChainError(f"{label} authority identity is invalid")
    _require_sha256(document.get("origin_id"), label=f"{label} origin ID")
    _utc(str(document.get("origin_utc")), label=f"{label} origin")
    _utc(str(document.get("target_start_utc")), label=f"{label} target start")
    _utc(str(document.get("target_end_utc")), label=f"{label} target end")

    signature = document.get("signature")
    unsigned = dict(document)
    unsigned.pop("signature")
    core = dict(unsigned)
    claimed_id = core.pop("receipt_id", None)
    if claimed_id != receipt_id(domain=domain, core=core):
        raise OmpexBenchmarkEvidenceChainError(f"{label} receipt ID mismatch")
    _verify_signature(
        signature,
        signature_payload(domain=domain, unsigned_receipt=unsigned),
        public_key,
        label=label,
    )
    return document


def _verify_signature(
    signature: object,
    payload: bytes,
    public_key: Ed25519PublicKey,
    *,
    label: str,
) -> None:
    if not isinstance(signature, Mapping) or set(signature) != {
        "algorithm",
        "key_id",
        "value_base64",
    }:
        raise OmpexBenchmarkEvidenceChainError(f"{label} signature block is invalid")
    if (
        signature.get("algorithm") != SIGNATURE_ALGORITHM
        or signature.get("key_id") != public_key_id(public_key)
    ):
        raise OmpexBenchmarkEvidenceChainError(f"{label} signature trust binding is invalid")
    try:
        value = base64.b64decode(str(signature.get("value_base64")), validate=True)
        public_key.verify(value, payload)
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise OmpexBenchmarkEvidenceChainError(f"{label} signature is invalid") from exc


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise OmpexBenchmarkEvidenceChainError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, Mapping):
        raise OmpexBenchmarkEvidenceChainError(f"{label} must be a JSON object")
    return dict(parsed)


def _utc(value: str, *, label: str) -> datetime:
    if _UTC.fullmatch(value) is None:
        raise OmpexBenchmarkEvidenceChainError(f"{label} must be exact UTC seconds")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise OmpexBenchmarkEvidenceChainError(f"{label} is invalid") from exc


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OmpexBenchmarkEvidenceChainError(f"{label} SHA-256 is invalid")


def _require_source_size(value: object, *, label: str) -> None:
    if type(value) is not int or not 0 < value <= _MAX_SOURCE_BYTES:
        raise OmpexBenchmarkEvidenceChainError(f"{label} is invalid")


def _require_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise OmpexBenchmarkEvidenceChainError(f"{label} differs")


def _validate_no_float(value: object, *, label: str) -> None:
    if isinstance(value, float):
        raise OmpexBenchmarkEvidenceChainError(f"{label} contains a float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise OmpexBenchmarkEvidenceChainError(f"{label} contains a non-string key")
            _validate_no_float(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_no_float(item, label=label)


def _binding(
    payload: bytes, document: Mapping[str, object], key_id: str
) -> dict[str, object]:
    return {
        "receipt_id": document["receipt_id"],
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "signer_key_id": key_id,
        "schema_version": document["schema_version"],
    }


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "AUTHORITY_CLASSIFICATION",
    "CANDIDATE_DOMAIN",
    "CANDIDATE_FREEZE_SCHEMA",
    "OMPEX_AVAILABILITY_SCHEMA",
    "OMPEX_DOMAIN",
    "SIGNATURE_ALGORITHM",
    "TRUTH_DOMAIN",
    "TRUTH_PUBLICATION_SCHEMA",
    "OmpexBenchmarkEvidenceChainError",
    "canonical_json",
    "public_key_id",
    "receipt_id",
    "signature_payload",
    "verify_reference_evidence_chain",
]
