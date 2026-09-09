"""Fail-closed protocol draft for externally registered CH LT origins.

The protocol defines monthly cadence slots, exact request/receipt bindings and
the compare-and-append properties required from an external registry.  It does
not implement that authority and therefore cannot make a local structural
origin countable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import ch_lt_dependence_power_design as power_design
from pfc_shaping.validation import ch_lt_estimand_contract as estimand
from pfc_shaping.validation import ch_lt_origin_target_mask_inventory as inventory
from pfc_shaping.validation import ch_lt_successor_candidate_core_v3 as core_v3
from pfc_shaping.validation import ch_market_time_regime as market_time

SCHEMA_VERSION = "ch_lt_origin_registry_protocol.v2"
ASSESSMENT_SCHEMA_VERSION = "ch_lt_origin_registry_protocol_assessment.v2"
STATUS = "LOCAL_OUTCOME_BLIND_ORIGIN_REGISTRY_PROTOCOL_DRAFT_INCOMPLETE_NO_GO"
PROTOCOL_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V2-20260730.json"
)
PROTOCOL_SHA256 = "6ea896ccdb35414b52237f2bcf1065755c3c10444b308ce905b60f472e68c697"
PROTOCOL_ID = "0fcf13246c50f6bc79d203437f7c5294495dc233f93873abbe2aeaf3dc282204"
STRUCTURAL_INVENTORY_RELATIVE_PATH = (
    "build/ch-lt-origin-target-mask-inventories/"
    "structural-dry-run-20260730T120000Z-schema-v2.json"
)
STRUCTURAL_INVENTORY_SHA256 = (
    "0dcc3411cb962e5ba4df2e36ea7bf67d97e0c56d1dcf3cb32a2684e26016c7d1"
)

REQUIRED_EXTERNAL_EVIDENCE = (
    "FMV_APPROVED_ISSUANCE_CADENCE_AND_EXACT_SCHEDULE",
    "OFFICIAL_VERSIONED_EEX_TRADING_CALENDAR_AND_SETTLEMENT_EVENT",
    "POWER_DERIVED_REQUIRED_ORIGIN_COUNTS",
    "TRUSTED_TIME_RECEIPT_AUTHORITY",
    "ORIGIN_REQUEST_AND_REGISTRY_SERVICE_IDENTITIES_ACLS_AND_KEYRINGS",
    "BUILDER_INACCESSIBLE_EXTERNAL_CAS_WORM_AND_FRESH_HEAD",
    "EXTERNAL_REGISTRY_REQUEST_RECEIPT_SCHEMA_AND_CROSS_FIELD_CONFORMANCE",
    "ORIGIN_AVAILABLE_EEX_PRODUCT_INVENTORY_AND_PROVIDER_ATTESTATION",
    "SEALED_PREDICTION_SCENARIO_STRUCTURAL_UNIVERSE_AND_EX_ANTE_MASK_RULE_COMMITMENTS",
    "QUALIFIED_RUNTIME_SOURCE_CONFIG_AND_CANDIDATE_IDENTITIES",
    "INDEPENDENT_ADMISSION_AND_SECURITY_ITOPS_QUANT_REVIEWS",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_name",
    "lifecycle",
    "market_scope",
    "cadence_proposal",
    "external_registry_domain",
    "registration_request_contract",
    "registration_receipt_contract",
    "origin_admission_policy",
    "truth_firewall",
    "scientific_use_policy",
    "bindings",
    "required_external_evidence",
    "protocol_complete",
    "registry_implemented",
    "countable_prospective_origin_count",
    "evidence_slot_satisfied",
    "truth_open_authorized",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
    "protocol_id",
}
_AUTHORITY_FIELDS = (
    "protocol_complete",
    "registry_implemented",
    "evidence_slot_satisfied",
    "truth_open_authorized",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
)
_SECTION_SHA256 = {
    "lifecycle": "c1b2d52141335dedb18c34a9181ffc29af59f6d12c00ed7b75c21e0c8cdc1cc1",
    "market_scope": "d06ddb0b65a488d2f714025359ef31c94a1692c0a1d02e2737fe52bc7dc36628",
    "cadence_proposal": "037619f50cd882a7c65227876c95af3d2433f6536c118ce1e43e86fe359779d6",
    "external_registry_domain": (
        "6d5bd9fce2c5eeb0e4a6213ec26afba4960089c57aaa1f1f50e96746770b04a7"
    ),
    "registration_request_contract": (
        "9133a235a3bde823ba28a49cb4e0e9b77f49f95527b286ba3db62e42bc60aab9"
    ),
    "registration_receipt_contract": (
        "d35444ef6fd085f539353f0bd6064587ec9a6ca493e673ac182966df71c2759e"
    ),
    "origin_admission_policy": (
        "78c9a30ae719480111276387ca34936d60534bd0029190cee782fe332d43bab2"
    ),
    "truth_firewall": (
        "1bd5e7a404857c8c6ce7e4ade2927e40eb8ca2d610070860b922f629aff9c7d8"
    ),
    "scientific_use_policy": (
        "7b61fb73fa315d96261064c3ede17c66894aad3ed6463432a5c7aac28a3f5432"
    ),
    "bindings": "905947cd2ee65a55e0d7277b7976c5436038beed89d8dbb20f970daa6799ba51",
    "required_external_evidence": (
        "0f6643e075a5f86dcaba7f7cd68881c3a7047d4ea56d539e06da53484b1430ac"
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000


class ChLtOriginRegistryProtocolError(ValueError):
    """Raised when protocol bytes, semantics or bound evidence diverge."""


def compute_protocol_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("protocol_id", None)
    return _semantic_hash(unsigned)


def verify_origin_registry_protocol(
    *, repo_root: str | Path, protocol_path: str | Path, expected_protocol_sha256: str
) -> dict[str, object]:
    """Verify the exact local protocol and its outcome-blind governance chain."""

    _require_sha256(expected_protocol_sha256, label="expected protocol")
    if expected_protocol_sha256 != PROTOCOL_SHA256:
        raise ChLtOriginRegistryProtocolError(
            "caller-held protocol SHA-256 does not match canonical identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    if not root.is_dir() or not (root / ".git").exists():
        raise ChLtOriginRegistryProtocolError("canonical repository root is unavailable")
    if _path_key(Path.cwd()) != _path_key(root):
        raise ChLtOriginRegistryProtocolError("cwd must equal canonical repository root")
    canonical = _repo_path(root, PROTOCOL_RELATIVE_PATH)
    if _path_key(_absolute(protocol_path)) != _path_key(canonical):
        raise ChLtOriginRegistryProtocolError("origin registry protocol path is not canonical")
    payload = read_stable_single_link_file(
        canonical, label="CH LT origin registry protocol", max_bytes=_MAX_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ChLtOriginRegistryProtocolError("origin registry protocol SHA-256 mismatch")
    document = _strict_mapping(payload, label="origin registry protocol")
    result = assess_origin_registry_protocol(document, document_bytes=payload)
    bound = _verify_bound_evidence(root)
    return {**result, **bound, "governance_bindings_verified": True}


def assess_origin_registry_protocol(
    document: Mapping[str, object], *, document_bytes: bytes
) -> dict[str, object]:
    """Validate protocol semantics without accessing future or T057 outcomes."""

    parsed = _strict_mapping(document_bytes, label="origin registry protocol bytes")
    _equal(parsed, document, "protocol mapping/bytes")
    if hashlib.sha256(document_bytes).hexdigest() != PROTOCOL_SHA256:
        raise ChLtOriginRegistryProtocolError("protocol document SHA-256 mismatch")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="origin registry protocol")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(
        document.get("protocol_name"),
        "ch_lt_origin_registry_protocol_20260730_v2",
        "protocol name",
    )
    _equal(document.get("protocol_id"), PROTOCOL_ID, "protocol id")
    _equal(compute_protocol_id(document), PROTOCOL_ID, "computed protocol id")
    for section, expected_sha256 in _SECTION_SHA256.items():
        if _semantic_hash(document.get(section)) != expected_sha256:
            raise ChLtOriginRegistryProtocolError(
                f"origin registry protocol semantic section mismatch: {section}"
            )
    _verify_cadence(document.get("cadence_proposal"))
    _verify_external_registry(document.get("external_registry_domain"))
    _verify_registration_contracts(
        document.get("registration_request_contract"),
        document.get("registration_receipt_contract"),
        document.get("origin_admission_policy"),
    )
    _verify_truth_firewall(document.get("truth_firewall"))
    _equal(
        document.get("required_external_evidence"),
        [
            {"requirement": item, "status": "MISSING", "path": None, "sha256": None}
            for item in REQUIRED_EXTERNAL_EVIDENCE
        ],
        "required external evidence",
    )
    _equal(document.get("countable_prospective_origin_count"), 0, "origin count")
    for field in _AUTHORITY_FIELDS:
        _equal(document.get(field), False, field)
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": STATUS,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash_closed_locally": True,
        "protocol_externally_frozen": False,
        "fmv_cadence_approved": False,
        "statistical_cadence": "MONTHLY",
        "exact_schedule_frozen": False,
        "registry_implemented": False,
        "global_origin_uniqueness_enforced": False,
        "countable_prospective_origin_count": 0,
        "missing_external_evidence": list(REQUIRED_EXTERNAL_EVIDENCE),
        "governance_bindings_verified": False,
        "evidence_slot_satisfied": False,
        "truth_open_authorized": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }


def _verify_bound_evidence(root: Path) -> dict[str, object]:
    structural_path = _repo_path(root, STRUCTURAL_INVENTORY_RELATIVE_PATH)
    structural = inventory.verify_structural_inventory(
        repo_root=root,
        inventory_path=structural_path,
        expected_inventory_sha256=STRUCTURAL_INVENTORY_SHA256,
    )
    _equal(
        structural.get("countable_prospective_origin"),
        False,
        "structural origin countability",
    )
    power = power_design.verify_dependence_power_design(
        repo_root=root,
        design_path=_repo_path(root, power_design.DESIGN_RELATIVE_PATH),
        expected_design_sha256=power_design.DESIGN_SHA256,
    )
    design_payload = read_stable_single_link_file(
        _repo_path(root, power_design.DESIGN_RELATIVE_PATH),
        label="CH LT dependence power design",
        max_bytes=_MAX_BYTES,
    )
    design = _strict_mapping(design_payload, label="dependence power design")
    geometry = _mapping(design.get("contrast_geometry"), label="contrast geometry")
    _equal(geometry.get("origin_cadence"), "MONTHLY", "power design cadence")
    _equal(power.get("power_design_complete"), False, "power design completion")

    bindings = _expected_bindings()
    protocol_payload = read_stable_single_link_file(
        _repo_path(root, PROTOCOL_RELATIVE_PATH),
        label="CH LT origin registry protocol",
        max_bytes=_MAX_BYTES,
    )
    protocol = _strict_mapping(protocol_payload, label="origin registry protocol")
    _equal(protocol.get("bindings"), bindings, "governance bindings")
    return {
        "structural_inventory_id": structural.get("inventory_id"),
        "structural_origin_id": structural.get("origin_id"),
        "structural_origin_as_of_utc": structural.get("origin_as_of_utc"),
        "structural_origin_countable": False,
        "power_design_complete": False,
    }


def _expected_bindings() -> dict[str, object]:
    return {
        "estimand_contract": {
            "path": estimand.CONTRACT_RELATIVE_PATH,
            "sha256": estimand.DOCUMENT_SHA256,
            "contract_id": estimand.CONTRACT_ID,
        },
        "candidate_core_v3": {
            "path": core_v3.CORE_RELATIVE_PATH,
            "sha256": core_v3.CORE_SHA256,
            "core_id": core_v3.CORE_ID,
        },
        "dependence_power_design": {
            "path": power_design.DESIGN_RELATIVE_PATH,
            "sha256": power_design.DESIGN_SHA256,
            "design_id": power_design.DESIGN_ID,
        },
        "market_time_regime": {
            "path": market_time.CONTRACT_RELATIVE_PATH,
            "sha256": market_time.DOCUMENT_SHA256,
            "contract_id": market_time.CONTRACT_ID,
            "transition_admitted": False,
        },
        "structural_inventory_dry_run": {
            "path": STRUCTURAL_INVENTORY_RELATIVE_PATH,
            "sha256": STRUCTURAL_INVENTORY_SHA256,
            "inventory_id": "5fff80e90abe874a6e85e4d3e1cc667bfeb2da9c1f827fc51c0523a7d62b0ed0",
            "countable_prospective_origin": False,
        },
    }


def _verify_cadence(value: object) -> None:
    cadence = _mapping(value, label="cadence proposal")
    _equal(cadence.get("statistical_cadence"), "MONTHLY", "statistical cadence")
    _equal(cadence.get("maximum_countable_origins_per_slot"), 1, "slot origin limit")
    _equal(cadence.get("fmv_approval_state"), "MISSING", "FMV approval state")
    _equal(cadence.get("exact_schedule_manifest_path"), None, "schedule path")
    _equal(cadence.get("exact_schedule_manifest_sha256"), None, "schedule SHA-256")
    _equal(
        cadence.get("schedule_materialization"),
        "EXPLICIT_CANONICAL_UTC_TIMESTAMPS_NO_RUNTIME_CALENDAR_INFERENCE",
        "schedule materialization",
    )
    _equal(
        cadence.get("origin_time_window_invariant"),
        "CAPTURE_WINDOW_OPEN_UTC_LE_ORIGIN_AS_OF_UTC_LE_CAPTURE_WINDOW_CLOSE_UTC",
        "origin time window",
    )
    _equal(
        cadence.get("registry_commit_window_invariant"),
        "ORIGIN_AS_OF_UTC_LE_COMMITTED_AT_UTC_LE_LATEST_EXTERNAL_REGISTRY_COMMIT_UTC_LT_FIRST_TARGET_DELIVERY_START_UTC",
        "registry commit window",
    )
    _equal(
        cadence.get("missed_slot_policy"),
        "MISSED_NOT_SHIFTED_NOT_BACKFILLED_NOT_REWEIGHTED",
        "missed slot policy",
    )


def _verify_external_registry(value: object) -> None:
    registry = _mapping(value, label="external registry domain")
    _equal(
        registry.get("backend_requirement"),
        "AUTHENTICATED_REMOTE_COMPARE_AND_APPEND_WITH_IMMUTABLE_OPERATION_LOOKUP",
        "registry backend",
    )
    _equal(
        registry.get("local_filesystem_or_sqlite_role"),
        "TEST_REFERENCE_ONLY_UNSUPPORTED_AS_AUTHORITY",
        "local registry authority",
    )
    _equal(registry.get("strictly_contiguous_sequence"), True, "registry sequence")
    _equal(
        registry.get("duplicate_origin_or_slot"),
        "COMPARE_AND_APPEND_REJECT_NO_SECOND_COUNTABLE_ORIGIN",
        "duplicate origin policy",
    )
    _equal(
        registry.get("exact_retry"),
        "IDEMPOTENT_RETURNS_IDENTICAL_RECEIPT_ONLY_FOR_IDENTICAL_REQUEST_ID_AND_REQUEST_SHA256",
        "retry",
    )


def _verify_registration_contracts(
    request_value: object, receipt_value: object, admission_value: object
) -> None:
    request = _mapping(request_value, label="registration request contract")
    receipt = _mapping(receipt_value, label="registration receipt contract")
    admission = _mapping(admission_value, label="origin admission policy")
    _equal(request.get("schema"), "ch_lt_origin_registration_request.v2", "request schema")
    _equal(
        request.get("request_id_derivation"),
        "SHA256_DOMAIN_SEPARATED_CANONICAL_REQUEST_CORE_EXCLUDING_REQUEST_ID_AND_REQUEST_SIGNATURE",
        "request id derivation",
    )
    request_fields = set(_sequence(request.get("required_fields"), label="request fields"))
    required_request_fields = {
        "request_id",
        "request_signature",
        "protocol_sha256",
        "protocol_id",
        "exact_schedule_entry_id",
        "exact_schedule_entry_sha256",
        "capture_window_open_utc",
        "capture_window_close_utc",
        "latest_external_registry_commit_utc",
        "origin_as_of_utc",
        "trusted_origin_time_utc",
        "trusted_origin_time_receipt_sha256",
        "candidate_hypothesis_and_procedure_manifest_sha256",
        "structural_target_universe_commitment_sha256",
        "ex_ante_evaluation_mask_rule_commitment_sha256",
    }
    if not required_request_fields.issubset(request_fields):
        raise ChLtOriginRegistryProtocolError("request fields omit critical bindings")
    if "final_mask_commitment_sha256" in request_fields:
        raise ChLtOriginRegistryProtocolError("realized final mask is forbidden at origin")
    request_invariants = set(
        _sequence(request.get("cross_field_invariants"), label="request invariants")
    )
    for required in (
        "CADENCE_SLOT_ID_SELECTS_EXACTLY_ONE_HASH_MATCHING_SCHEDULE_ENTRY",
        "CAPTURE_WINDOW_OPEN_UTC_LE_ORIGIN_AS_OF_UTC_LE_CAPTURE_WINDOW_CLOSE_UTC",
        "TRUSTED_ORIGIN_TIME_UTC_EQ_ORIGIN_AS_OF_UTC_AND_RECEIPT_VERIFIES_BOTH",
        "REALIZED_MATURITY_OR_TRUTH_MASK_IS_ABSENT_AT_ORIGIN",
    ):
        if required not in request_invariants:
            raise ChLtOriginRegistryProtocolError("request cross-field invariant is missing")
    _equal(
        request.get("commit_deadline"),
        "COMMITTED_AT_UTC_LE_SELECTED_SCHEDULE_LATEST_EXTERNAL_REGISTRY_COMMIT_UTC_AND_LT_FIRST_TARGET_DELIVERY_START_UTC",
        "request commit deadline",
    )
    _equal(receipt.get("schema"), "ch_lt_origin_registration_receipt.v2", "receipt schema")
    receipt_fields = set(_sequence(receipt.get("required_fields"), label="receipt fields"))
    if not {
        "protocol_sha256",
        "protocol_id",
        "request_id",
        "request_sha256",
        "exact_schedule_entry_id",
        "exact_schedule_entry_sha256",
        "latest_external_registry_commit_utc",
        "origin_as_of_utc",
        "committed_at_utc",
    }.issubset(receipt_fields):
        raise ChLtOriginRegistryProtocolError("receipt fields omit critical bindings")
    receipt_invariants = set(
        _sequence(receipt.get("cross_field_invariants"), label="receipt invariants")
    )
    if (
        "ORIGIN_AS_OF_UTC_LE_COMMITTED_AT_UTC_LE_LATEST_EXTERNAL_REGISTRY_COMMIT_UTC_LT_FIRST_TARGET_DELIVERY_START_UTC"
        not in receipt_invariants
    ):
        raise ChLtOriginRegistryProtocolError("receipt commit-window invariant is missing")
    _equal(
        admission.get("realized_maturity_or_truth_mask_at_origin"),
        "FORBIDDEN_UNKNOWN_UNTIL_SEPARATE_EXTERNAL_MATURITY_EVENT",
        "realized mask at origin",
    )


def _verify_truth_firewall(value: object) -> None:
    firewall = _mapping(value, label="truth firewall")
    _equal(firewall.get("outcome_bearing_t057_registry"), "FORBIDDEN", "T057 registry")
    _equal(firewall.get("t057_outcomes_or_scores"), "FORBIDDEN", "T057 outcomes")
    _equal(
        firewall.get(
            "prediction_scenario_structural_universe_and_ex_ante_mask_rule_sealed_before_truth_open"
        ),
        True,
        "seal before truth",
    )
    _equal(firewall.get("future_holdout_consumed"), False, "future holdout")
    _equal(firewall.get("t057_consumed"), False, "T057 consumed")


def _semantic_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtOriginRegistryProtocolError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtOriginRegistryProtocolError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChLtOriginRegistryProtocolError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtOriginRegistryProtocolError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtOriginRegistryProtocolError(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtOriginRegistryProtocolError("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    return selected if selected.is_absolute() else Path(os.path.abspath(selected))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtOriginRegistryProtocolError(f"{label} SHA-256 is invalid")


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "PROTOCOL_ID",
    "PROTOCOL_RELATIVE_PATH",
    "PROTOCOL_SHA256",
    "REQUIRED_EXTERNAL_EVIDENCE",
    "SCHEMA_VERSION",
    "STATUS",
    "STRUCTURAL_INVENTORY_RELATIVE_PATH",
    "STRUCTURAL_INVENTORY_SHA256",
    "ChLtOriginRegistryProtocolError",
    "assess_origin_registry_protocol",
    "compute_protocol_id",
    "verify_origin_registry_protocol",
]
