"""Validate bounded ENTSO-E proposals and synthetic receipts without execution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from uuid import UUID

from pfc_shaping.path_safety import read_stable_single_link_file

CONTRACT_SCHEMA = (
    "fmv_entsoe_bounded_execution_parameter_receipt_contract.v1"
)
CONTRACT_STATUS = (
    "STATIC_OFFLINE_PARAMETER_AND_RECEIPT_CONTRACT_NO_EXECUTION_AUTHORITY"
)
CONTRACT_FILE_SHA256 = (
    "9ae529cb242ea79d97a8deae32317c8d0fc2522c7afd216863c9f7c885a6c7b2"
)
CONTRACT_CONTENT_ID = (
    "834cbba541395072886e1ba80012a76cc55a0a8990b6d354f6d1bfe110a12156"
)
GOVERNED_PACKAGE_CONTRACT_SHA256 = (
    "97edaf93c639953365d20d7539380a706f0f8fdbec9476af94262fb1d4b2204d"
)
GOVERNED_PACKAGE_CONTRACT_CONTENT_ID = (
    "4f6e644b15593e8d4ae2ca196e1d52a002d4255bdf42e7a137982edb854f0fa0"
)
GOVERNED_PACKAGE_PROOF_CONTENT_ID = (
    "314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53"
)
GOVERNED_PACKAGE_PROOF_MANIFEST_SHA256 = (
    "d7f6ad3af60d2efd087718e0515c8b73a6534387ab12c7c7a5f88f42aeadf2b4"
)
MAPPING_COMPILER_CONTRACT_SHA256 = (
    "f33bc696d8e7a535f7700b40c13dc90d73e2526073400edcd2588e57432a9b3a"
)
MAPPING_COMPILER_CONTRACT_CONTENT_ID = (
    "5dc43e18dcf107020e24638307a707d8c7afbf894ca1131e62907f8680ad0951"
)
MAPPING_COMPILER_PROOF_CONTENT_ID = (
    "bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766"
)
MAPPING_COMPILER_PROOF_MANIFEST_SHA256 = (
    "7d86330e2143e2009947b486eb1e8428dd6022735ebc33c9aa4cfc4b871c5268"
)

PROPOSAL_SCHEMA = "fmv_entsoe_bounded_capture_proposal.v1"
RECEIPT_SCHEMA = "fmv_entsoe_bounded_capture_receipt.v1"
ROLE_ORDER = ("series_dimension", "latest_values", "vintage_values")
ROLE_PARAMETERS = {
    "series_dimension": (),
    "latest_values": ("start_utc", "end_utc"),
    "vintage_values": ("start_utc", "end_utc", "as_of_cutoff_utc"),
}
ROLE_LIMITS = {
    "series_dimension": 100_001,
    "latest_values": 1_000_001,
    "vintage_values": 1_000_001,
}
ROLE_BYTE_CAPS = {
    "series_dimension": 268_435_456,
    "latest_values": 17_179_869_184,
    "vintage_values": 50_465_865_728,
}
MAX_TOTAL_LOCAL_EXPORT_BYTES = 68_719_476_736
MAX_WINDOW_DAYS = 31
STATEMENT_TIMEOUT_SECONDS = 300
BATCH_TIMEOUT_SECONDS = 900

PROPOSAL_FIELD_ORDER = (
        "schema_version",
        "evidence_class",
        "capture_batch_id",
        "created_at_utc",
        "snapshot_reference_time_utc",
        "challenge_id",
        "challenge_nonce_sha256",
        "mapping_content_id",
        "metadata_schema_fingerprint",
        "metadata_payload_sha256",
        "metadata_statement_id",
        "compiled_template_sha256",
        "execution_parameters",
        "cost_fence",
        "authorization",
)
PROPOSAL_FIELDS = frozenset(PROPOSAL_FIELD_ORDER)
COST_FENCE_FIELDS = frozenset(
    {
        "currency",
        "estimated_cost_amount",
        "hard_cost_cap_amount",
        "estimate_method",
        "warehouse_size",
        "warehouse_auto_stop_minutes",
        "max_warehouse_starts",
        "max_read_statements",
        "statement_timeout_seconds",
        "batch_timeout_seconds",
        "max_retries",
        "max_total_local_export_bytes",
        "per_role_max_rows",
        "per_role_max_bytes",
        "limit_hit_requires_rejection",
        "cap_breach_requires_rejection",
        "retry_after_failure_authorized",
    }
)
RECEIPT_FIELD_ORDER = (
        "schema_version",
        "evidence_class",
        "proposal_content_id",
        "capture_batch_id",
        "snapshot_reference_time_utc",
        "capture_started_at_utc",
        "capture_completed_at_utc",
        "warehouse_start_count",
        "read_statement_count",
        "retry_count",
        "total_runtime_seconds",
        "total_local_export_bytes",
        "statements",
        "authorities",
)
RECEIPT_FIELDS = frozenset(RECEIPT_FIELD_ORDER)
STATEMENT_FIELD_ORDER = (
        "role",
        "template_sha256",
        "parameter_sha256",
        "statement_id_sha256",
        "statement_state",
        "statement_class",
        "statement_started_at_utc",
        "statement_completed_at_utc",
        "runtime_seconds",
        "row_count",
        "size_bytes",
        "result_truncated",
        "limit_hit",
        "read_only_statement_verified",
        "remote_write_count",
        "artifact_relative_path",
        "artifact_sha256",
)
STATEMENT_FIELDS = frozenset(STATEMENT_FIELD_ORDER)
AUTHORIZATION = {
    "explicit_user_authorization_present": False,
    "authorization_id": None,
    "execution_authorized": False,
}
AUTHORITIES = {
    "real_proposal_authorized": False,
    "real_capture_receipt_admitted": False,
    "artifact_hashes_verified": False,
    "artifact_values_opened": False,
    "point_in_time_availability_proven": False,
    "training_authorized": False,
    "selection_authorized": False,
    "model_input_authorized": False,
    "candidate_assembly_authorized": False,
    "promotion_authorized": False,
    "production_authorized": False,
}
EXPECTED_COST_FENCE = {
    "warehouse_auto_stop_minutes": 5,
    "max_warehouse_starts": 1,
    "max_read_statements": 3,
    "statement_timeout_seconds": STATEMENT_TIMEOUT_SECONDS,
    "batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
    "max_retries": 0,
    "max_total_local_export_bytes": MAX_TOTAL_LOCAL_EXPORT_BYTES,
    "per_role_max_rows": ROLE_LIMITS,
    "per_role_max_bytes": ROLE_BYTE_CAPS,
    "limit_hit_requires_rejection": True,
    "cap_breach_requires_rejection": True,
    "retry_after_failure_authorized": False,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECOND = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_DECIMAL_AMOUNT = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d{1,6})?$")
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. /()+-]{0,127}$")


class EntsoeBoundedExecutionReceiptError(ValueError):
    """Raised when an offline proposal or synthetic receipt is unsafe."""


def validate_bounded_execution_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D236 static zero-execution contract."""

    contract = _mapping(document, "bounded execution contract")
    _equal(
        set(contract),
        {
            "schema_version",
            "status",
            "purpose",
            "input_bindings",
            "proposal_manifest",
            "parameter_policy",
            "cost_fence",
            "synthetic_receipt_contract",
            "execution_policy",
            "authorities",
        },
        "contract fields",
    )
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    bindings = _mapping(contract.get("input_bindings"), "input bindings")
    _equal(
        bindings,
        {
            "governed_acquisition_package_decision": "D-20260805-233",
            "governed_acquisition_package_contract_content_id": (
                GOVERNED_PACKAGE_CONTRACT_CONTENT_ID
            ),
            "governed_acquisition_package_proof_content_id": (
                GOVERNED_PACKAGE_PROOF_CONTENT_ID
            ),
            "physical_mapping_compiler_decision": "D-20260805-234",
            "physical_mapping_compiler_contract_content_id": (
                MAPPING_COMPILER_CONTRACT_CONTENT_ID
            ),
            "physical_mapping_compiler_proof_content_id": (
                MAPPING_COMPILER_PROOF_CONTENT_ID
            ),
        },
        "input bindings",
    )
    proposal = _mapping(contract.get("proposal_manifest"), "proposal manifest")
    _equal(proposal.get("schema_version"), PROPOSAL_SCHEMA, "proposal schema")
    _equal(
        proposal.get("exact_top_level_fields"),
        list(PROPOSAL_FIELD_ORDER),
        "proposal field declaration",
    )
    _equal(
        proposal.get("allowed_evidence_classes"),
        ["SYNTHETIC_TEST_ONLY", "REAL_UNAUTHORIZED_PROPOSAL"],
        "proposal evidence classes",
    )
    _equal(proposal.get("authorization"), AUTHORIZATION, "authorization")
    policy = _mapping(contract.get("parameter_policy"), "parameter policy")
    _equal(policy.get("target_window_minimum_days"), 1, "minimum window")
    _equal(policy.get("target_window_maximum_days"), MAX_WINDOW_DAYS, "maximum window")
    _equal(
        policy.get("series_dimension_parameters"),
        list(ROLE_PARAMETERS["series_dimension"]),
        "dimension parameters",
    )
    _equal(
        policy.get("latest_values_parameters"),
        list(ROLE_PARAMETERS["latest_values"]),
        "latest parameters",
    )
    _equal(
        policy.get("vintage_values_parameters"),
        list(ROLE_PARAMETERS["vintage_values"]),
        "vintage parameters",
    )
    cost = _mapping(contract.get("cost_fence"), "contract cost fence")
    for field, expected in EXPECTED_COST_FENCE.items():
        _equal(cost.get(field), expected, f"contract cost fence {field}")
    receipt = _mapping(
        contract.get("synthetic_receipt_contract"),
        "synthetic receipt contract",
    )
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(
        receipt.get("exact_top_level_fields"),
        list(RECEIPT_FIELD_ORDER),
        "receipt fields",
    )
    _equal(
        receipt.get("exact_statement_fields"),
        list(STATEMENT_FIELD_ORDER),
        "statement fields",
    )
    _equal(receipt.get("statement_roles"), list(ROLE_ORDER), "statement roles")
    execution = _mapping(contract.get("execution_policy"), "execution policy")
    for field in (
        "current_databricks_connection_budget",
        "current_databricks_statement_budget",
        "current_warehouse_start_budget",
        "current_network_call_budget",
        "current_remote_write_budget",
    ):
        _equal(execution.get(field), 0, f"execution policy {field}")
    _equal(execution.get("connector_code_allowed"), False, "connector authority")
    _equal(
        execution.get("proposal_execution_authorized"),
        False,
        "proposal execution authority",
    )
    _equal(contract.get("authorities"), AUTHORITIES, "contract authorities")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "databricks_connection_budget": 0,
        "databricks_statement_budget": 0,
        "warehouse_start_budget": 0,
        "execution_authorized": False,
        "production_authorized": False,
    }


def validate_bounded_capture_proposal(
    document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Validate a proposal as structurally ready but never executable."""

    proposal = _mapping(document, "bounded capture proposal")
    _exact_fields(proposal, PROPOSAL_FIELDS, "bounded capture proposal")
    _equal(proposal.get("schema_version"), PROPOSAL_SCHEMA, "proposal schema")
    evidence_class = proposal.get("evidence_class")
    if evidence_class not in {"SYNTHETIC_TEST_ONLY", "REAL_UNAUTHORIZED_PROPOSAL"}:
        raise EntsoeBoundedExecutionReceiptError("proposal evidence class is invalid")
    _uuid(proposal.get("capture_batch_id"), "capture batch ID")
    _uuid(proposal.get("challenge_id"), "challenge ID")
    _uuid(proposal.get("metadata_statement_id"), "metadata statement ID")
    for field in (
        "challenge_nonce_sha256",
        "mapping_content_id",
        "metadata_schema_fingerprint",
        "metadata_payload_sha256",
    ):
        _sha256(proposal.get(field), field)
    created = _parse_utc_second(proposal.get("created_at_utc"), "created_at_utc")
    snapshot = _parse_utc_second(
        proposal.get("snapshot_reference_time_utc"),
        "snapshot_reference_time_utc",
    )
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    if snapshot > created or created > reference:
        raise EntsoeBoundedExecutionReceiptError(
            "proposal timestamps must satisfy snapshot <= created <= reference"
        )
    templates = _mapping(
        proposal.get("compiled_template_sha256"),
        "compiled template hashes",
    )
    _exact_fields(templates, frozenset(ROLE_ORDER), "compiled template hashes")
    for role in ROLE_ORDER:
        _sha256(templates.get(role), f"compiled template hash {role}")
    parameters = _validate_execution_parameters(
        proposal.get("execution_parameters"),
        snapshot=snapshot,
    )
    _validate_cost_fence(proposal.get("cost_fence"))
    _equal(proposal.get("authorization"), AUTHORIZATION, "proposal authorization")
    content_id = canonical_sha256(proposal)
    return {
        "schema_version": PROPOSAL_SCHEMA,
        "status": "PASS_OFFLINE_PROPOSAL_VALIDATION_EXECUTION_NOT_AUTHORIZED",
        "proposal_content_id": content_id,
        "evidence_class": evidence_class,
        "capture_batch_id": proposal["capture_batch_id"],
        "snapshot_reference_time_utc": proposal["snapshot_reference_time_utc"],
        "window_days": parameters["window_days"],
        "cost_fence_validated": True,
        "proposal_execution_authorized": False,
        "databricks_connection_count": 0,
        "databricks_statement_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "remote_write_count": 0,
        "production_authorized": False,
    }


def validate_synthetic_capture_receipt(
    document: Mapping[str, object],
    proposal_document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Validate receipt structure only; never treat it as execution evidence."""

    proposal_result = validate_bounded_capture_proposal(
        proposal_document,
        reference_time_utc=reference_time_utc,
    )
    if proposal_result["evidence_class"] != "SYNTHETIC_TEST_ONLY":
        raise EntsoeBoundedExecutionReceiptError(
            "a synthetic receipt requires a synthetic proposal"
        )
    receipt = _mapping(document, "synthetic capture receipt")
    _exact_fields(receipt, RECEIPT_FIELDS, "synthetic capture receipt")
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(receipt.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "receipt evidence")
    _equal(
        receipt.get("proposal_content_id"),
        proposal_result["proposal_content_id"],
        "proposal content ID",
    )
    _equal(
        receipt.get("capture_batch_id"),
        proposal_result["capture_batch_id"],
        "capture batch ID",
    )
    _equal(
        receipt.get("snapshot_reference_time_utc"),
        proposal_result["snapshot_reference_time_utc"],
        "snapshot reference",
    )
    proposal_created = _parse_utc_second(
        proposal_document.get("created_at_utc"),
        "proposal created_at_utc",
    )
    started = _parse_utc_second(
        receipt.get("capture_started_at_utc"),
        "capture_started_at_utc",
    )
    completed = _parse_utc_second(
        receipt.get("capture_completed_at_utc"),
        "capture_completed_at_utc",
    )
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    if started < proposal_created or completed < started or completed > reference:
        raise EntsoeBoundedExecutionReceiptError(
            "receipt timestamps must satisfy proposal <= start <= complete <= reference"
        )
    _equal(receipt.get("warehouse_start_count"), 0, "synthetic warehouse starts")
    _equal(receipt.get("read_statement_count"), len(ROLE_ORDER), "statement count")
    _equal(receipt.get("retry_count"), 0, "retry count")
    total_runtime = _non_negative_int(
        receipt.get("total_runtime_seconds"),
        "total runtime seconds",
    )
    if total_runtime > BATCH_TIMEOUT_SECONDS:
        raise EntsoeBoundedExecutionReceiptError("batch timeout exceeded")
    total_bytes = _non_negative_int(
        receipt.get("total_local_export_bytes"),
        "total local export bytes",
    )
    if total_bytes > MAX_TOTAL_LOCAL_EXPORT_BYTES:
        raise EntsoeBoundedExecutionReceiptError("total local export cap exceeded")
    statements = receipt.get("statements")
    if not isinstance(statements, list) or len(statements) != len(ROLE_ORDER):
        raise EntsoeBoundedExecutionReceiptError("receipt statements differ")
    template_hashes = _mapping(
        proposal_document.get("compiled_template_sha256"),
        "proposal template hashes",
    )
    execution_parameters = _mapping(
        proposal_document.get("execution_parameters"),
        "proposal execution parameters",
    )
    artifact_paths: set[str] = set()
    observed_runtime = 0
    observed_bytes = 0
    previous_completed = started
    for role, raw_statement in zip(ROLE_ORDER, statements, strict=True):
        statement = _mapping(raw_statement, f"statement {role}")
        _exact_fields(statement, STATEMENT_FIELDS, f"statement {role}")
        _equal(statement.get("role"), role, f"statement role {role}")
        _equal(
            statement.get("template_sha256"),
            template_hashes[role],
            f"template hash {role}",
        )
        _equal(
            statement.get("parameter_sha256"),
            canonical_sha256(execution_parameters[role]),
            f"parameter hash {role}",
        )
        _sha256(statement.get("statement_id_sha256"), f"statement ID hash {role}")
        _equal(statement.get("statement_state"), "SUCCEEDED", f"state {role}")
        _equal(statement.get("statement_class"), "SELECT", f"class {role}")
        statement_started = _parse_utc_second(
            statement.get("statement_started_at_utc"),
            f"statement start {role}",
        )
        statement_completed = _parse_utc_second(
            statement.get("statement_completed_at_utc"),
            f"statement complete {role}",
        )
        if (
            statement_started < previous_completed
            or statement_completed < statement_started
            or statement_completed > completed
        ):
            raise EntsoeBoundedExecutionReceiptError(
                f"statement interval is invalid or overlaps: {role}"
            )
        runtime = _non_negative_int(statement.get("runtime_seconds"), f"runtime {role}")
        actual_runtime = int((statement_completed - statement_started).total_seconds())
        if runtime != actual_runtime or runtime > STATEMENT_TIMEOUT_SECONDS:
            raise EntsoeBoundedExecutionReceiptError(f"statement runtime differs: {role}")
        row_count = _non_negative_int(statement.get("row_count"), f"row count {role}")
        if row_count >= ROLE_LIMITS[role]:
            raise EntsoeBoundedExecutionReceiptError(f"row limit reached: {role}")
        size_bytes = _non_negative_int(statement.get("size_bytes"), f"size {role}")
        if size_bytes > ROLE_BYTE_CAPS[role]:
            raise EntsoeBoundedExecutionReceiptError(f"byte cap exceeded: {role}")
        _equal(statement.get("result_truncated"), False, f"truncation {role}")
        _equal(statement.get("limit_hit"), False, f"limit hit {role}")
        _equal(statement.get("read_only_statement_verified"), True, f"read-only {role}")
        _equal(statement.get("remote_write_count"), 0, f"remote writes {role}")
        artifact = _artifact_relative_path(
            statement.get("artifact_relative_path"),
            role,
        )
        if artifact in artifact_paths:
            raise EntsoeBoundedExecutionReceiptError("artifact paths must be distinct")
        artifact_paths.add(artifact)
        _sha256(statement.get("artifact_sha256"), f"artifact hash {role}")
        observed_runtime += runtime
        observed_bytes += size_bytes
        previous_completed = statement_completed
    _equal(total_runtime, observed_runtime, "receipt total runtime")
    _equal(total_bytes, observed_bytes, "receipt total export bytes")
    _equal(receipt.get("authorities"), AUTHORITIES, "receipt authorities")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS_SYNTHETIC_RECEIPT_STRUCTURE_ONLY_NO_EXECUTION_EVIDENCE",
        "proposal_content_id": proposal_result["proposal_content_id"],
        "receipt_content_id": canonical_sha256(receipt),
        "synthetic_receipt_structure_verified": True,
        "artifact_hashes_verified": False,
        "artifact_values_opened": False,
        "real_capture_receipt_admitted": False,
        "synthetic_receipt_is_execution_evidence": False,
        "databricks_connection_count": 0,
        "databricks_statement_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "remote_write_count": 0,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
    }


def verify_bounded_execution_paths(
    *,
    contract_path: str | Path,
    governed_package_contract_path: str | Path,
    governed_package_proof_manifest_path: str | Path,
    mapping_compiler_contract_path: str | Path,
    mapping_compiler_proof_manifest_path: str | Path,
    proposal_path: str | Path,
    receipt_path: str | Path,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Bind exact local D233, D234 and D236 artifacts without a connector."""

    contract = _read_hashed_json(
        contract_path,
        label="D236 bounded execution contract",
        expected_sha256=CONTRACT_FILE_SHA256,
    )
    contract_result = validate_bounded_execution_contract(contract)
    governed_contract = _read_hashed_json(
        governed_package_contract_path,
        label="D233 governed package contract",
        expected_sha256=GOVERNED_PACKAGE_CONTRACT_SHA256,
    )
    _equal(
        canonical_sha256(governed_contract),
        GOVERNED_PACKAGE_CONTRACT_CONTENT_ID,
        "D233 contract content ID",
    )
    governed_proof = _read_hashed_json(
        governed_package_proof_manifest_path,
        label="D233 proof manifest",
        expected_sha256=GOVERNED_PACKAGE_PROOF_MANIFEST_SHA256,
    )
    _equal(
        governed_proof.get("content_id"),
        GOVERNED_PACKAGE_PROOF_CONTENT_ID,
        "D233 proof content ID",
    )
    mapping_contract = _read_hashed_json(
        mapping_compiler_contract_path,
        label="D234 mapping compiler contract",
        expected_sha256=MAPPING_COMPILER_CONTRACT_SHA256,
    )
    _equal(
        canonical_sha256(mapping_contract),
        MAPPING_COMPILER_CONTRACT_CONTENT_ID,
        "D234 contract content ID",
    )
    mapping_proof = _read_hashed_json(
        mapping_compiler_proof_manifest_path,
        label="D234 proof manifest",
        expected_sha256=MAPPING_COMPILER_PROOF_MANIFEST_SHA256,
    )
    _equal(
        mapping_proof.get("content_id"),
        MAPPING_COMPILER_PROOF_CONTENT_ID,
        "D234 proof content ID",
    )
    proposal = _read_json(proposal_path, "bounded capture proposal")
    receipt = _read_json(receipt_path, "synthetic capture receipt")
    receipt_result = validate_synthetic_capture_receipt(
        receipt,
        proposal,
        reference_time_utc=reference_time_utc,
    )
    return {
        **receipt_result,
        "contract_content_id": contract_result["contract_content_id"],
        "governed_package_proof_content_id": GOVERNED_PACKAGE_PROOF_CONTENT_ID,
        "mapping_compiler_proof_content_id": MAPPING_COMPILER_PROOF_CONTENT_ID,
    }


def canonical_sha256(value: object) -> str:
    """Return deterministic JSON SHA-256 for public fixture construction."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_execution_parameters(
    raw: object,
    *,
    snapshot: datetime,
) -> dict[str, object]:
    parameters = _mapping(raw, "execution parameters")
    _exact_fields(parameters, frozenset(ROLE_ORDER), "execution parameters")
    parsed: dict[str, dict[str, datetime]] = {}
    for role in ROLE_ORDER:
        role_parameters = _mapping(parameters.get(role), f"parameters {role}")
        _exact_fields(
            role_parameters,
            frozenset(ROLE_PARAMETERS[role]),
            f"parameters {role}",
        )
        parsed[role] = {
            name: _parse_utc_second(role_parameters[name], f"{role}.{name}")
            for name in ROLE_PARAMETERS[role]
        }
    latest = parsed["latest_values"]
    vintage = parsed["vintage_values"]
    _equal(latest["start_utc"], vintage["start_utc"], "target window start")
    _equal(latest["end_utc"], vintage["end_utc"], "target window end")
    start = latest["start_utc"]
    end = latest["end_utc"]
    if any((start.hour, start.minute, start.second, end.hour, end.minute, end.second)):
        raise EntsoeBoundedExecutionReceiptError(
            "target window boundaries must be UTC midnight"
        )
    window_seconds = int((end - start).total_seconds())
    if window_seconds <= 0 or window_seconds % 86_400:
        raise EntsoeBoundedExecutionReceiptError("target window must contain whole days")
    window_days = window_seconds // 86_400
    if not 1 <= window_days <= MAX_WINDOW_DAYS:
        raise EntsoeBoundedExecutionReceiptError("target window exceeds 1..31 days")
    cutoff = vintage["as_of_cutoff_utc"]
    if cutoff < start or cutoff > snapshot:
        raise EntsoeBoundedExecutionReceiptError(
            "vintage cutoff must be between target start and snapshot reference"
        )
    return {"window_days": window_days}


def _validate_cost_fence(raw: object) -> None:
    cost = _mapping(raw, "proposal cost fence")
    _exact_fields(cost, COST_FENCE_FIELDS, "proposal cost fence")
    currency = cost.get("currency")
    if not isinstance(currency, str) or _CURRENCY.fullmatch(currency) is None:
        raise EntsoeBoundedExecutionReceiptError("currency must be ISO-like uppercase")
    estimated = _decimal_amount(cost.get("estimated_cost_amount"), "estimated cost")
    hard_cap = _decimal_amount(cost.get("hard_cost_cap_amount"), "hard cost cap")
    if estimated > hard_cap:
        raise EntsoeBoundedExecutionReceiptError("estimated cost exceeds hard cap")
    for field in ("estimate_method", "warehouse_size"):
        value = cost.get(field)
        if not isinstance(value, str) or _SAFE_TEXT.fullmatch(value) is None:
            raise EntsoeBoundedExecutionReceiptError(f"{field} format is invalid")
    for field, expected in EXPECTED_COST_FENCE.items():
        _equal(cost.get(field), expected, f"proposal cost fence {field}")


def _artifact_relative_path(value: object, role: str) -> str:
    if not isinstance(value, str) or "\\" in value or ":" in value or "//" in value:
        raise EntsoeBoundedExecutionReceiptError(f"artifact path is invalid: {role}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EntsoeBoundedExecutionReceiptError(f"artifact path is unsafe: {role}")
    if path.parts[:2] != ("build", "governed-source-intake") or len(path.parts) < 3:
        raise EntsoeBoundedExecutionReceiptError(f"artifact path prefix differs: {role}")
    return value


def _read_hashed_json(
    raw_path: str | Path,
    *,
    label: str,
    expected_sha256: str,
) -> Mapping[str, object]:
    payload = _read_path(raw_path, label=label, max_bytes=100_000)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise EntsoeBoundedExecutionReceiptError(f"{label} SHA-256 differs")
    return _strict_json(payload, label)


def _read_json(raw_path: str | Path, label: str) -> Mapping[str, object]:
    return _strict_json(_read_path(raw_path, label=label, max_bytes=100_000), label)


def _read_path(raw_path: str | Path, *, label: str, max_bytes: int) -> bytes:
    path = Path(raw_path)
    if not path.is_absolute():
        raise EntsoeBoundedExecutionReceiptError(f"{label} path must be absolute")
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise EntsoeBoundedExecutionReceiptError(f"{label} path read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EntsoeBoundedExecutionReceiptError(
                    f"{label} duplicate key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeBoundedExecutionReceiptError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeBoundedExecutionReceiptError(f"{label} must be a mapping")
    return value


def _exact_fields(
    document: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(document) != set(expected):
        raise EntsoeBoundedExecutionReceiptError(f"{label} fields differ")


def _parse_utc_second(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECOND.fullmatch(value) is None:
        raise EntsoeBoundedExecutionReceiptError(
            f"{label} must be UTC at whole seconds"
        )
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise EntsoeBoundedExecutionReceiptError(f"{label} is invalid") from exc


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeBoundedExecutionReceiptError(f"{label} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoeBoundedExecutionReceiptError(f"{label} must be timezone-aware UTC")
    return value


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EntsoeBoundedExecutionReceiptError(f"{label} must be a UUID")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise EntsoeBoundedExecutionReceiptError(f"{label} must be a UUID") from exc
    if str(parsed) != value:
        raise EntsoeBoundedExecutionReceiptError(f"{label} must be canonical lowercase")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EntsoeBoundedExecutionReceiptError(f"{label} must be lowercase SHA-256")
    return value


def _decimal_amount(value: object, label: str) -> Decimal:
    if not isinstance(value, str) or _DECIMAL_AMOUNT.fullmatch(value) is None:
        raise EntsoeBoundedExecutionReceiptError(
            f"{label} must be a non-negative fixed decimal"
        )
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise EntsoeBoundedExecutionReceiptError(f"{label} is invalid") from exc


def _non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EntsoeBoundedExecutionReceiptError(
            f"{label} must be a non-negative integer"
        )
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeBoundedExecutionReceiptError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )
