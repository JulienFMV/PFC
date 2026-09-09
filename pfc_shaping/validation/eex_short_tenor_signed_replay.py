"""Path-safe signed replay for structural CH EEX short-tenor receipts.

The adapter authenticates local byte integrity for two D227 receipts and two
execution-time observations. Trust anchors are supplied by the caller. It does
not establish who owns those keys, whether the time source is independent, or
whether any EEX/ENTSO-E input was available point in time. A passing replay is
therefore deliberately non-authoritative.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
)
from pfc_shaping.validation.eex_short_tenor_receipts import (
    CONTRACT_CONTENT_ID as RECEIPT_CONTRACT_CONTENT_ID,
)
from pfc_shaping.validation.eex_short_tenor_receipts import (
    RECEIPT_CONTRACT_SCHEMA,
    SELECTION_RECEIPT_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    ShortTenorReceiptError,
    load_strict_receipt_json,
    validate_structural_selection_receipt,
    validate_structural_training_receipt,
)

SIGNED_REPLAY_CONTRACT_SCHEMA = (
    "fmv-eex-ch-short-tenor-signed-replay-contract-v1"
)
SIGNED_REPLAY_CONTRACT_STATUS = (
    "PASS_LOCAL_DSSE_AND_TIME_LINK_REPLAY_ONLY_NO_EXTERNAL_TRUST"
)
SIGNED_REPLAY_CONTRACT_CONTENT_ID = (
    "09dfd36125132a1a1c8089f603a381414853c393d23dda633354938843382606"
)
RECEIPT_CONTRACT_SHA256 = (
    "3168d75e245f033daa44f4c6dab9061f51e06f2b993797b7fb7d9c46dbe6b665"
)

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SLSA_PREDICATE_TYPE = "https://slsa.dev/provenance/v1"
TRUSTED_TIME_PREDICATE_TYPE = (
    "urn:fmv:pfc:trusted-execution-time-observation:v1"
)
SHORT_TENOR_BUILD_TYPE = (
    "urn:fmv:pfc:eex-ch-short-tenor-receipt-replay:v1"
)
DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json"
SIGNATURE_ALGORITHM = "Ed25519"
KEY_ID_RULE = "sha256_subject_public_key_info_der"

_MAX_ENVELOPE_BYTES = 512 * 1024
_MAX_PUBLIC_KEY_BYTES = 4 * 1024
_MAX_STATEMENT_BYTES = 320 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")

_AUTHENTICATED_USE_BLOCKERS = (
    "externally governed trust-anchor registry with owner and role identity",
    "key lifecycle issuance rotation revocation and compromise policy",
    "independently verified trusted-time authority and source acquisition receipts",
    "same-snapshot replay of every bound EEX and ENTSO-E PIT artifact",
    "append-only or WORM receipt retention",
    "new independently frozen future holdout distinct from sealed T057",
)


class ShortTenorSignedReplayError(ValueError):
    """Raised when a signed receipt replay is ambiguous or invalid."""


def validate_signed_replay_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact non-authoritative D228 signed-replay profile."""

    mapping = _mapping(document, "signed replay contract")
    _equal(
        mapping.get("schema_version"),
        SIGNED_REPLAY_CONTRACT_SCHEMA,
        "signed replay contract schema",
    )
    _equal(
        mapping.get("status"),
        SIGNED_REPLAY_CONTRACT_STATUS,
        "signed replay contract status",
    )
    selected = _mapping(mapping.get("selected_boundary"), "selected boundary")
    _equal(
        selected.get("receipt_contract_schema"),
        RECEIPT_CONTRACT_SCHEMA,
        "receipt contract schema binding",
    )
    _equal(
        selected.get("receipt_contract_content_id"),
        RECEIPT_CONTRACT_CONTENT_ID,
        "receipt contract content binding",
    )
    _equal(
        selected.get("receipt_contract_sha256"),
        RECEIPT_CONTRACT_SHA256,
        "receipt contract byte binding",
    )
    profile = _mapping(
        mapping.get("in_toto_slsa_profile"),
        "in-toto SLSA profile",
    )
    _equal(profile.get("statement_type"), STATEMENT_TYPE, "statement type")
    _equal(
        profile.get("predicate_type"),
        SLSA_PREDICATE_TYPE,
        "SLSA predicate type",
    )
    _equal(profile.get("build_type"), SHORT_TENOR_BUILD_TYPE, "build type")
    time_profile = _mapping(
        mapping.get("trusted_time_link_profile"),
        "trusted-time link profile",
    )
    _equal(
        time_profile.get("predicate_type"),
        TRUSTED_TIME_PREDICATE_TYPE,
        "trusted-time predicate type",
    )
    _equal(
        mapping.get("required_before_authenticated_use"),
        list(_AUTHENTICATED_USE_BLOCKERS),
        "authenticated-use blockers",
    )
    content_id = _sha256_json(mapping)
    _equal(
        content_id,
        SIGNED_REPLAY_CONTRACT_CONTENT_ID,
        "signed replay contract content id",
    )
    return {
        "schema_version": SIGNED_REPLAY_CONTRACT_SCHEMA,
        "status": SIGNED_REPLAY_CONTRACT_STATUS,
        "contract_content_id": content_id,
        "receipt_contract_content_id": RECEIPT_CONTRACT_CONTENT_ID,
        "authenticated_use_blocker_count": len(_AUTHENTICATED_USE_BLOCKERS),
        "external_authority_verified": False,
        "independently_trusted_time_verified": False,
        "training_authorized": False,
        "selection_authorized": False,
        "production_authorized": False,
    }


def verify_signed_short_tenor_receipt_pair_paths(
    *,
    training_envelope_path: str | Path,
    selection_envelope_path: str | Path,
    training_execution_public_key_path: str | Path,
    selection_governance_public_key_path: str | Path,
    trusted_time_public_key_path: str | Path,
    training_time_envelope_path: str | Path,
    selection_time_envelope_path: str | Path,
    expected_training_envelope_sha256: str,
    expected_selection_envelope_sha256: str,
    expected_training_execution_public_key_sha256: str,
    expected_selection_governance_public_key_sha256: str,
    expected_trusted_time_public_key_sha256: str,
    expected_training_time_envelope_sha256: str,
    expected_selection_time_envelope_sha256: str,
    expected_trusted_time_source_id: str,
) -> dict[str, object]:
    """Replay seven exact, caller-anchored files without opening market data."""

    expected = {
        "training envelope": _sha(
            expected_training_envelope_sha256,
            "expected training envelope",
        ),
        "selection envelope": _sha(
            expected_selection_envelope_sha256,
            "expected selection envelope",
        ),
        "training execution public key": _sha(
            expected_training_execution_public_key_sha256,
            "expected training execution public key",
        ),
        "selection governance public key": _sha(
            expected_selection_governance_public_key_sha256,
            "expected selection governance public key",
        ),
        "trusted-time public key": _sha(
            expected_trusted_time_public_key_sha256,
            "expected trusted-time public key",
        ),
        "training time envelope": _sha(
            expected_training_time_envelope_sha256,
            "expected training time envelope",
        ),
        "selection time envelope": _sha(
            expected_selection_time_envelope_sha256,
            "expected selection time envelope",
        ),
    }
    time_source_id = _safe_id(
        expected_trusted_time_source_id,
        "expected trusted-time source id",
    )
    paths = {
        "training envelope": _absolute(
            training_envelope_path,
            "training envelope",
        ),
        "selection envelope": _absolute(
            selection_envelope_path,
            "selection envelope",
        ),
        "training execution public key": _absolute(
            training_execution_public_key_path,
            "training execution public key",
        ),
        "selection governance public key": _absolute(
            selection_governance_public_key_path,
            "selection governance public key",
        ),
        "trusted-time public key": _absolute(
            trusted_time_public_key_path,
            "trusted-time public key",
        ),
        "training time envelope": _absolute(
            training_time_envelope_path,
            "training time envelope",
        ),
        "selection time envelope": _absolute(
            selection_time_envelope_path,
            "selection time envelope",
        ),
    }
    _require_pairwise_distinct_paths(paths)
    payloads = {
        label: _read_exact(
            path,
            expected_sha256=expected[label],
            max_bytes=(
                _MAX_PUBLIC_KEY_BYTES
                if "public key" in label
                else _MAX_ENVELOPE_BYTES
            ),
            label=label,
        )
        for label, path in paths.items()
    }

    training_key, execution_key_id = _load_public_key(
        payloads["training execution public key"],
        label="training execution public key",
    )
    selection_key, governance_key_id = _load_public_key(
        payloads["selection governance public key"],
        label="selection governance public key",
    )
    time_key, trusted_time_key_id = _load_public_key(
        payloads["trusted-time public key"],
        label="trusted-time public key",
    )
    _require_distinct_key_ids(
        {
            "execution": execution_key_id,
            "governance": governance_key_id,
            "trusted-time": trusted_time_key_id,
        }
    )

    training_receipt, training_binding = _verify_receipt_envelope(
        payloads["training envelope"],
        public_key=training_key,
        public_key_id_value=execution_key_id,
        expected_role="TRAINING",
        expected_schema=TRAINING_RECEIPT_SCHEMA,
        label="training envelope",
    )
    selection_receipt, selection_binding = _verify_receipt_envelope(
        payloads["selection envelope"],
        public_key=selection_key,
        public_key_id_value=governance_key_id,
        expected_role="SELECTION",
        expected_schema=SELECTION_RECEIPT_SCHEMA,
        label="selection envelope",
    )

    try:
        training = validate_structural_training_receipt(training_receipt)
        selection = validate_structural_selection_receipt(
            selection_receipt,
            training_receipt=training_receipt,
        )
    except ShortTenorReceiptError as exc:
        raise ShortTenorSignedReplayError(
            "signed receipt fails D227 structural replay"
        ) from exc

    training_authority = _authority_context(training_receipt, "training authority")
    selection_authority = _authority_context(
        selection_receipt,
        "selection authority",
    )
    _verify_cross_receipt_authority_sets(
        training_authority,
        selection_authority,
    )
    _equal(
        training_authority["execution_key_id"],
        execution_key_id,
        "training execution key binding",
    )
    _equal(
        training_authority["governance_key_id"],
        governance_key_id,
        "selection governance key binding",
    )
    _equal(
        training_authority["trusted_time_key_id"],
        trusted_time_key_id,
        "trusted-time key binding",
    )

    training_time = _verify_time_envelope(
        payloads["training time envelope"],
        public_key=time_key,
        public_key_id_value=trusted_time_key_id,
        receipt=training_receipt,
        expected_role="TRAINING",
        expected_time_source_id=time_source_id,
        label="training time envelope",
    )
    selection_time = _verify_time_envelope(
        payloads["selection time envelope"],
        public_key=time_key,
        public_key_id_value=trusted_time_key_id,
        receipt=selection_receipt,
        expected_role="SELECTION",
        expected_time_source_id=time_source_id,
        label="selection time envelope",
    )
    if (
        training_time["execution_attestation_sha256"]
        == selection_time["execution_attestation_sha256"]
    ):
        raise ShortTenorSignedReplayError(
            "training and selection execution attestations must be distinct"
        )
    selection_started = _receipt_run_time(
        selection_receipt,
        "started_at_utc",
        "selection start",
    )
    if training_time["observed_at"] > selection_started:
        raise ShortTenorSignedReplayError(
            "training time observation is later than selection start"
        )

    return {
        "schema_version": "fmv_eex_ch_short_tenor_signed_replay_assessment.v1",
        "status": SIGNED_REPLAY_CONTRACT_STATUS,
        "signed_replay_contract_content_id": SIGNED_REPLAY_CONTRACT_CONTENT_ID,
        "receipt_contract_content_id": RECEIPT_CONTRACT_CONTENT_ID,
        "outer_origin_id": training["outer_origin_id"],
        "outer_origin_as_of_utc": training["outer_origin_as_of_utc"],
        "training_receipt_id": training["receipt_id"],
        "selection_receipt_id": selection["receipt_id"],
        "selected_candidate_config_sha256": selection[
            "selected_candidate_config_sha256"
        ],
        "receipt_bindings": {
            "training": {
                **training_binding,
                "envelope_sha256": expected["training envelope"],
                "public_key_file_sha256": expected[
                    "training execution public key"
                ],
                "signer_role": "EXECUTION",
            },
            "selection": {
                **selection_binding,
                "envelope_sha256": expected["selection envelope"],
                "public_key_file_sha256": expected[
                    "selection governance public key"
                ],
                "signer_role": "GOVERNANCE",
            },
        },
        "time_observation_bindings": {
            "training": {
                **_public_time_binding(training_time),
                "envelope_sha256": expected["training time envelope"],
            },
            "selection": {
                **_public_time_binding(selection_time),
                "envelope_sha256": expected["selection time envelope"],
            },
            "public_key_file_sha256": expected["trusted-time public key"],
            "time_source_id": time_source_id,
        },
        "local_dsse_signature_integrity_verified": True,
        "in_toto_slsa_bindings_verified": True,
        "d227_structural_replay_verified": True,
        "training_selection_link_verified": True,
        "execution_time_observation_links_verified": True,
        "execution_attestations_distinct": True,
        "role_keys_pairwise_distinct": True,
        "authority_sets_consistent_and_globally_disjoint": True,
        "receipt_materials_opened": False,
        "receipt_subject_artifacts_opened": False,
        "source_trusted_time_receipts_opened": False,
        "price_values_opened": False,
        "external_key_owner_identity_verified": False,
        "key_lifecycle_verified": False,
        "independently_trusted_time_verified": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "authenticated_use_blockers": list(_AUTHENTICATED_USE_BLOCKERS),
        "databricks_request_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "remote_write_count": 0,
    }


def dsse_pae(payload_type: str, payload: bytes) -> bytes:
    """Return DSSE v1 pre-authentication encoding for exact payload bytes."""

    if payload_type != DSSE_PAYLOAD_TYPE:
        raise ShortTenorSignedReplayError("DSSE payload type is invalid")
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_STATEMENT_BYTES:
        raise ShortTenorSignedReplayError("DSSE payload must be bounded bytes")
    encoded_type = payload_type.encode("utf-8")
    return b" ".join(
        (
            b"DSSEv1",
            str(len(encoded_type)).encode("ascii"),
            encoded_type,
            str(len(payload)).encode("ascii"),
            payload,
        )
    )


def public_key_id(public_key: Ed25519PublicKey) -> str:
    """Return the SHA-256 ID of SubjectPublicKeyInfo DER bytes."""

    if not isinstance(public_key, Ed25519PublicKey):
        raise ShortTenorSignedReplayError("public key must be Ed25519")
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Encode the strict local canonical JSON profile without a trailing LF."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ShortTenorSignedReplayError(
            "document is not canonical-JSON compatible"
        ) from exc


def trusted_time_observation_statement(
    *,
    receipt_role: str,
    execution_attestation_sha256: str,
    observed_at_utc: str,
    time_source_id: str,
) -> dict[str, object]:
    """Build the deterministic value-free time observation payload."""

    role = str(receipt_role)
    if role not in {"TRAINING", "SELECTION"}:
        raise ShortTenorSignedReplayError("trusted-time receipt role is invalid")
    execution_sha = _sha(
        execution_attestation_sha256,
        "execution attestation",
    )
    observed = _utc_text(observed_at_utc, "trusted-time observation")
    source_id = _safe_id(time_source_id, "trusted-time source id")
    core = {
        "receipt_role": role,
        "execution_attestation_sha256": execution_sha,
        "observed_at_utc": observed,
        "time_source_id": source_id,
    }
    predicate = {**core, "observation_id": _sha256_json(core)}
    return {
        "_type": STATEMENT_TYPE,
        "subject": [
            {
                "name": f"{role.lower()}-execution-attestation",
                "digest": {"sha256": execution_sha},
            }
        ],
        "predicateType": TRUSTED_TIME_PREDICATE_TYPE,
        "predicate": predicate,
    }


def _verify_receipt_envelope(
    envelope_bytes: bytes,
    *,
    public_key: Ed25519PublicKey,
    public_key_id_value: str,
    expected_role: str,
    expected_schema: str,
    label: str,
) -> tuple[Mapping[str, object], dict[str, str]]:
    payload = _verify_dsse_envelope(
        envelope_bytes,
        public_key=public_key,
        public_key_id_value=public_key_id_value,
        label=label,
    )
    receipt, receipt_sha256 = _verify_slsa_statement(
        payload,
        expected_role=expected_role,
        expected_schema=expected_schema,
        label=label,
    )
    return receipt, {
        "payload_type": DSSE_PAYLOAD_TYPE,
        "statement_sha256": hashlib.sha256(payload).hexdigest(),
        "receipt_sha256": receipt_sha256,
        "signing_key_id": public_key_id_value,
    }


def _verify_dsse_envelope(
    envelope_bytes: bytes,
    *,
    public_key: Ed25519PublicKey,
    public_key_id_value: str,
    label: str,
) -> bytes:
    envelope = _strict_json_mapping(envelope_bytes, label)
    if canonical_json_bytes(envelope) != envelope_bytes:
        raise ShortTenorSignedReplayError(f"{label} must use canonical JSON bytes")
    _exact_keys(envelope, {"payloadType", "payload", "signatures"}, label)
    _equal(envelope.get("payloadType"), DSSE_PAYLOAD_TYPE, f"{label} payload type")
    payload = _strict_base64(envelope.get("payload"), f"{label} payload")
    if len(payload) > _MAX_STATEMENT_BYTES:
        raise ShortTenorSignedReplayError(f"{label} payload exceeds maximum size")
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise ShortTenorSignedReplayError(f"{label} must have exactly one signature")
    signature = _mapping(signatures[0], f"{label} signature")
    _exact_keys(signature, {"keyid", "sig"}, f"{label} signature")
    signature_bytes = _strict_base64(signature.get("sig"), f"{label} signature")
    if len(signature_bytes) != 64:
        raise ShortTenorSignedReplayError(f"{label} Ed25519 signature size is invalid")
    try:
        public_key.verify(signature_bytes, dsse_pae(DSSE_PAYLOAD_TYPE, payload))
    except InvalidSignature as exc:
        raise ShortTenorSignedReplayError(f"{label} signature is invalid") from exc
    # DSSE keyid is an unauthenticated hint. The caller-supplied role key is
    # selected first and performs verification; keyid is only a consistency
    # check and never drives trust-anchor selection.
    _equal(signature.get("keyid"), public_key_id_value, f"{label} key id")
    return payload


def _verify_slsa_statement(
    payload: bytes,
    *,
    expected_role: str,
    expected_schema: str,
    label: str,
) -> tuple[Mapping[str, object], str]:
    statement = _strict_json_mapping(payload, f"{label} statement")
    if canonical_json_bytes(statement) != payload:
        raise ShortTenorSignedReplayError(
            f"{label} statement must use canonical JSON bytes"
        )
    _exact_keys(
        statement,
        {"_type", "subject", "predicateType", "predicate"},
        f"{label} statement",
    )
    _equal(statement.get("_type"), STATEMENT_TYPE, f"{label} statement type")
    _equal(
        statement.get("predicateType"),
        SLSA_PREDICATE_TYPE,
        f"{label} SLSA predicate type",
    )
    predicate = _mapping(statement.get("predicate"), f"{label} predicate")
    _exact_keys(
        predicate,
        {"buildDefinition", "runDetails"},
        f"{label} predicate",
    )
    build = _mapping(
        predicate.get("buildDefinition"),
        f"{label} build definition",
    )
    _exact_keys(
        build,
        {
            "buildType",
            "externalParameters",
            "internalParameters",
            "resolvedDependencies",
        },
        f"{label} build definition",
    )
    _equal(build.get("buildType"), SHORT_TENOR_BUILD_TYPE, f"{label} build type")
    _equal(build.get("internalParameters"), {}, f"{label} internal parameters")
    external = _mapping(
        build.get("externalParameters"),
        f"{label} external parameters",
    )
    _exact_keys(
        external,
        {
            "receipt_contract_content_id",
            "receipt_role",
            "receipt_schema",
            "receipt_sha256",
            "receipt",
        },
        f"{label} external parameters",
    )
    _equal(
        external.get("receipt_contract_content_id"),
        RECEIPT_CONTRACT_CONTENT_ID,
        f"{label} receipt contract binding",
    )
    _equal(external.get("receipt_role"), expected_role, f"{label} receipt role")
    _equal(
        external.get("receipt_schema"),
        expected_schema,
        f"{label} receipt schema",
    )
    receipt = _mapping(external.get("receipt"), f"{label} receipt")
    receipt_bytes = canonical_json_bytes(receipt)
    try:
        strict_receipt = load_strict_receipt_json(receipt_bytes)
    except ShortTenorReceiptError as exc:
        raise ShortTenorSignedReplayError(f"{label} receipt JSON is invalid") from exc
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    _equal(
        external.get("receipt_sha256"),
        receipt_sha256,
        f"{label} receipt digest",
    )
    _verify_output_subject(
        statement.get("subject"),
        receipt=strict_receipt,
        label=label,
    )
    _verify_resolved_dependencies(
        build.get("resolvedDependencies"),
        receipt=strict_receipt,
        label=label,
    )
    _verify_run_details(
        predicate.get("runDetails"),
        receipt=strict_receipt,
        label=label,
    )
    return strict_receipt, receipt_sha256


def _verify_output_subject(
    value: object,
    *,
    receipt: Mapping[str, object],
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != 1:
        raise ShortTenorSignedReplayError(f"{label} must have exactly one subject")
    subject = _mapping(value[0], f"{label} subject")
    _exact_keys(subject, {"name", "digest"}, f"{label} subject")
    receipt_subject = _mapping(receipt.get("subject"), f"{label} receipt subject")
    _equal(subject.get("name"), receipt_subject.get("name"), f"{label} subject name")
    digest = _mapping(subject.get("digest"), f"{label} subject digest")
    _exact_keys(digest, {"sha256"}, f"{label} subject digest")
    _equal(
        digest.get("sha256"),
        receipt_subject.get("sha256"),
        f"{label} subject digest",
    )


def _verify_resolved_dependencies(
    value: object,
    *,
    receipt: Mapping[str, object],
    label: str,
) -> None:
    materials = receipt.get("materials")
    if isinstance(materials, list) and materials:
        expected: list[dict[str, object]] = []
        for index, raw in enumerate(materials):
            material = _mapping(raw, f"{label} material {index}")
            expected.append(
                {
                    "name": material.get("name"),
                    "digest": {"sha256": material.get("sha256")},
                }
            )
    elif receipt.get("receipt_role") == "SELECTION":
        expected = [
            {
                "name": "candidate-grid",
                "digest": {"sha256": receipt.get("candidate_grid_sha256")},
            },
            {
                "name": "fold-inventory",
                "digest": {"sha256": receipt.get("fold_inventory_sha256")},
            },
            {
                "name": "inner-loss-commitment",
                "digest": {
                    "sha256": receipt.get("inner_loss_commitment_sha256")
                },
            },
            {
                "name": "selection-policy",
                "digest": {"sha256": receipt.get("policy_content_id")},
            },
            {
                "name": "successor-core",
                "digest": {"sha256": receipt.get("successor_core_id")},
            },
            {
                "name": "training-receipt",
                "digest": {"sha256": receipt.get("training_receipt_sha256")},
            },
        ]
    else:
        raise ShortTenorSignedReplayError(
            f"{label} role-specific dependencies are invalid"
        )
    _equal(value, expected, f"{label} resolved dependencies")


def _verify_run_details(
    value: object,
    *,
    receipt: Mapping[str, object],
    label: str,
) -> None:
    run = _mapping(value, f"{label} SLSA run details")
    _exact_keys(run, {"builder", "metadata", "byproducts"}, f"{label} SLSA run details")
    receipt_run = _mapping(receipt.get("run_details"), f"{label} receipt run details")
    _equal(
        run.get("builder"),
        {"id": receipt_run.get("builder_id")},
        f"{label} builder",
    )
    _equal(
        run.get("metadata"),
        {
            "invocationId": receipt_run.get("invocation_id"),
            "startedOn": receipt_run.get("started_at_utc"),
            "finishedOn": receipt_run.get("finished_at_utc"),
        },
        f"{label} run metadata",
    )
    byproduct_hashes = _mapping(
        receipt_run.get("byproduct_sha256s"),
        f"{label} receipt byproducts",
    )
    expected_byproducts = [
        {"name": name, "digest": {"sha256": byproduct_hashes[name]}}
        for name in sorted(byproduct_hashes)
    ]
    expected_byproducts.extend(
        [
            {
                "name": "execution-attestation",
                "digest": {
                    "sha256": receipt_run.get("execution_attestation_sha256")
                },
            },
            {
                "name": "trusted-time-receipt",
                "digest": {
                    "sha256": receipt_run.get("trusted_time_receipt_sha256")
                },
            },
        ]
    )
    _equal(run.get("byproducts"), expected_byproducts, f"{label} byproducts")


def _verify_time_envelope(
    envelope_bytes: bytes,
    *,
    public_key: Ed25519PublicKey,
    public_key_id_value: str,
    receipt: Mapping[str, object],
    expected_role: str,
    expected_time_source_id: str,
    label: str,
) -> dict[str, object]:
    payload = _verify_dsse_envelope(
        envelope_bytes,
        public_key=public_key,
        public_key_id_value=public_key_id_value,
        label=label,
    )
    statement = _strict_json_mapping(payload, f"{label} statement")
    if canonical_json_bytes(statement) != payload:
        raise ShortTenorSignedReplayError(
            f"{label} statement must use canonical JSON bytes"
        )
    _exact_keys(
        statement,
        {"_type", "subject", "predicateType", "predicate"},
        f"{label} statement",
    )
    _equal(statement.get("_type"), STATEMENT_TYPE, f"{label} statement type")
    _equal(
        statement.get("predicateType"),
        TRUSTED_TIME_PREDICATE_TYPE,
        f"{label} predicate type",
    )
    receipt_run = _mapping(receipt.get("run_details"), f"{label} receipt run")
    execution_sha = _sha(
        receipt_run.get("execution_attestation_sha256"),
        f"{label} execution attestation",
    )
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or len(subjects) != 1:
        raise ShortTenorSignedReplayError(f"{label} must have exactly one subject")
    subject = _mapping(subjects[0], f"{label} subject")
    _exact_keys(subject, {"name", "digest"}, f"{label} subject")
    _equal(
        subject.get("name"),
        f"{expected_role.lower()}-execution-attestation",
        f"{label} subject name",
    )
    digest = _mapping(subject.get("digest"), f"{label} subject digest")
    _exact_keys(digest, {"sha256"}, f"{label} subject digest")
    _equal(digest.get("sha256"), execution_sha, f"{label} subject digest")

    predicate = _mapping(statement.get("predicate"), f"{label} predicate")
    _exact_keys(
        predicate,
        {
            "receipt_role",
            "execution_attestation_sha256",
            "observed_at_utc",
            "time_source_id",
            "observation_id",
        },
        f"{label} predicate",
    )
    _equal(predicate.get("receipt_role"), expected_role, f"{label} receipt role")
    _equal(
        predicate.get("execution_attestation_sha256"),
        execution_sha,
        f"{label} execution attestation binding",
    )
    _equal(
        predicate.get("time_source_id"),
        expected_time_source_id,
        f"{label} time source",
    )
    core = {
        "receipt_role": expected_role,
        "execution_attestation_sha256": execution_sha,
        "observed_at_utc": predicate.get("observed_at_utc"),
        "time_source_id": expected_time_source_id,
    }
    _equal(
        predicate.get("observation_id"),
        _sha256_json(core),
        f"{label} observation id",
    )
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    _equal(
        receipt_run.get("trusted_time_receipt_sha256"),
        payload_sha256,
        f"{label} receipt payload binding",
    )
    observed_at = _utc(
        predicate.get("observed_at_utc"),
        f"{label} observed time",
    )
    finished_at = _utc(
        receipt_run.get("finished_at_utc"),
        f"{label} run finish",
    )
    outer_origin = _utc(
        receipt.get("outer_origin_as_of_utc"),
        f"{label} outer origin",
    )
    if observed_at < finished_at:
        raise ShortTenorSignedReplayError(
            f"{label} observation precedes run finish"
        )
    if observed_at > outer_origin:
        raise ShortTenorSignedReplayError(
            f"{label} observation is later than outer origin"
        )
    return {
        "payload_sha256": payload_sha256,
        "statement_sha256": payload_sha256,
        "execution_attestation_sha256": execution_sha,
        "observation_id": str(predicate["observation_id"]),
        "observed_at_utc": str(predicate["observed_at_utc"]),
        "observed_at": observed_at,
        "signing_key_id": public_key_id_value,
    }


def _public_time_binding(value: Mapping[str, object]) -> dict[str, str]:
    return {
        key: str(value[key])
        for key in (
            "payload_sha256",
            "statement_sha256",
            "execution_attestation_sha256",
            "observation_id",
            "observed_at_utc",
            "signing_key_id",
        )
    }


def _receipt_run_time(
    receipt: Mapping[str, object],
    field: str,
    label: str,
) -> datetime:
    run = _mapping(receipt.get("run_details"), f"{label} run details")
    return _utc(run.get(field), label)


def _load_public_key(
    payload: bytes,
    *,
    label: str,
) -> tuple[Ed25519PublicKey, str]:
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise ShortTenorSignedReplayError(f"{label} is not valid PEM") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ShortTenorSignedReplayError(f"{label} must contain an Ed25519 key")
    canonical = key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if canonical != payload:
        raise ShortTenorSignedReplayError(f"{label} PEM bytes are not canonical")
    return key, public_key_id(key)


def _authority_context(
    receipt: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    authority = _mapping(receipt.get("authority"), label)
    return {
        "governance_key_id": _sha(
            authority.get("governance_key_id"),
            f"{label} governance key",
        ),
        "execution_key_id": _sha(
            authority.get("execution_key_id"),
            f"{label} execution key",
        ),
        "trusted_time_key_id": _sha(
            authority.get("trusted_time_key_id"),
            f"{label} trusted-time key",
        ),
        "source_acquisition_key_ids": tuple(
            _sha_list(
                authority.get("source_acquisition_key_ids"),
                f"{label} source acquisition keys",
            )
        ),
        "source_trusted_time_key_ids": tuple(
            _sha_list(
                authority.get("source_trusted_time_key_ids"),
                f"{label} source trusted-time keys",
            )
        ),
    }


def _verify_cross_receipt_authority_sets(
    training: Mapping[str, object],
    selection: Mapping[str, object],
) -> None:
    for field in (
        "governance_key_id",
        "execution_key_id",
        "trusted_time_key_id",
        "source_acquisition_key_ids",
        "source_trusted_time_key_ids",
    ):
        _equal(selection[field], training[field], f"cross-receipt {field}")
    non_time = {
        str(training["governance_key_id"]),
        str(training["execution_key_id"]),
        *map(str, training["source_acquisition_key_ids"]),
    }
    time_keys = {
        str(training["trusted_time_key_id"]),
        *map(str, training["source_trusted_time_key_ids"]),
    }
    if non_time.intersection(time_keys):
        raise ShortTenorSignedReplayError(
            "cross-receipt non-time and trusted-time key sets overlap"
        )


def _require_distinct_key_ids(values: Mapping[str, str]) -> None:
    if len(set(values.values())) != len(values):
        raise ShortTenorSignedReplayError(
            "governance execution and trusted-time role keys must be distinct"
        )


def _absolute(value: str | Path, label: str) -> Path:
    try:
        return assert_absolute_path_has_no_links(value)
    except (OSError, TypeError, ValueError) as exc:
        raise ShortTenorSignedReplayError(
            f"{label} path must be absolute and link-free"
        ) from exc


def _require_pairwise_distinct_paths(paths: Mapping[str, Path]) -> None:
    entries = list(paths.items())
    for index, (left_label, left) in enumerate(entries):
        for right_label, right in entries[index + 1 :]:
            try:
                same = os.path.samefile(left, right)
            except OSError as exc:
                raise ShortTenorSignedReplayError(
                    f"cannot establish {left_label}/{right_label} path identity"
                ) from exc
            if same:
                raise ShortTenorSignedReplayError(
                    "signed replay input files must be pairwise distinct"
                )


def _read_exact(
    path: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    label: str,
) -> bytes:
    try:
        payload = read_stable_single_link_file(
            path,
            label=label,
            max_bytes=max_bytes,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ShortTenorSignedReplayError(f"{label} cannot be read safely") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ShortTenorSignedReplayError(f"{label} differs from caller-held hash")
    return payload


def _strict_json_mapping(payload: bytes, label: str) -> Mapping[str, object]:
    if not isinstance(payload, bytes) or not payload:
        raise ShortTenorSignedReplayError(f"{label} must be non-empty bytes")
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, StrictStructuredDataError) as exc:
        raise ShortTenorSignedReplayError(f"{label} is not strict UTF-8 JSON") from exc
    return _mapping(value, label)


def _strict_base64(value: object, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ShortTenorSignedReplayError(f"{label} must be non-empty base64 text")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ShortTenorSignedReplayError(f"{label} is not strict base64") from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ShortTenorSignedReplayError(f"{label} is not canonical base64")
    return decoded


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ShortTenorSignedReplayError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ShortTenorSignedReplayError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha(value: object, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ShortTenorSignedReplayError(f"{label} must be lowercase SHA-256")
    return text


def _sha_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ShortTenorSignedReplayError(f"{label} must be non-empty")
    output = [_sha(item, label) for item in value]
    if output != sorted(set(output)):
        raise ShortTenorSignedReplayError(f"{label} must be sorted and unique")
    return output


def _safe_id(value: object, label: str) -> str:
    text = str(value)
    if _SAFE_ID.fullmatch(text) is None:
        raise ShortTenorSignedReplayError(f"{label} is invalid")
    return text


def _utc(value: object, label: str) -> datetime:
    text = _utc_text(value, label)
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ShortTenorSignedReplayError(
            f"{label} must be canonical UTC seconds"
        ) from exc


def _utc_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ShortTenorSignedReplayError(f"{label} must be text")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ShortTenorSignedReplayError(
            f"{label} must be canonical UTC seconds"
        ) from exc
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ShortTenorSignedReplayError(f"{label} differs")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "DSSE_PAYLOAD_TYPE",
    "KEY_ID_RULE",
    "RECEIPT_CONTRACT_SHA256",
    "SHORT_TENOR_BUILD_TYPE",
    "SIGNATURE_ALGORITHM",
    "SIGNED_REPLAY_CONTRACT_CONTENT_ID",
    "SIGNED_REPLAY_CONTRACT_SCHEMA",
    "SIGNED_REPLAY_CONTRACT_STATUS",
    "SLSA_PREDICATE_TYPE",
    "STATEMENT_TYPE",
    "TRUSTED_TIME_PREDICATE_TYPE",
    "ShortTenorSignedReplayError",
    "canonical_json_bytes",
    "dsse_pae",
    "public_key_id",
    "trusted_time_observation_statement",
    "validate_signed_replay_contract",
    "verify_signed_short_tenor_receipt_pair_paths",
]
