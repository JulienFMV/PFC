"""Relational validator for one Swiss LT compute-runtime manifest.

The validator proves closed local structure and hash-bound receipt semantics.
It does not verify an independent signature, admit an evaluation, or grant
production authority.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.ch_lt_compute_runtime import (
    EXPECTED_AUTHORITY_BOUNDARY,
    EXPECTED_DETERMINISM_POLICY,
    EXPECTED_RUNTIME_MANIFEST_SCHEMA,
    assess_compute_runtime,
    canonical_json_bytes,
)

SCHEMA_VERSION = "ch_lt_compute_runtime_manifest.v1"
AUDIT_SCHEMA_VERSION = "ch_lt_compute_runtime_manifest_audit.v1"
RECEIPT_SCHEMA_VERSION = "ch_lt_compute_runtime_evidence_receipt.v1"
SCENARIO_DESIGN_SCHEMA_VERSION = "ch_lt_probabilistic_scenario_mc_design.v1"
MONTE_CARLO_STUDY_SCHEMA_VERSION = "ch_lt_monte_carlo_error_study.v1"
DESIGN_FREEZE_RECEIPT_SCHEMA_VERSION = "ch_lt_scenario_design_freeze_receipt.v1"
STATUS = "VALID_CLOSED_RUNTIME_MANIFEST_AND_LOCAL_EVIDENCE_BYTES_NOT_AUTHORITY"
MAX_MANIFEST_BYTES = 512 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_BOUND_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_FROZEN_WEIGHTS_BYTES = 64 * 1024 * 1024

INPUT_HASH_ROLES = (
    "origin_target_mask_and_inner_fold_inventory",
    "input_manifests_and_available_at_times",
    "candidate_baseline_and_hyperparameter_grid",
    "training_cache_manifest",
    "probabilistic_scenario_and_mc_design_manifest",
)
CODE_AND_WHEEL_HASH_ROLES = (
    "governed_lt_wheel",
    "dependency_lock",
    "evaluation_code",
    "runtime_environment",
    "numpy_wheel",
    "torch_wheel",
)
VERSION_FIELDS = ("python", "numpy", "torch")
CUDA_FIELDS = ("cuda_build", "cuda_runtime", "cudnn", "nvidia_driver")
HOST_FIELDS = ("os", "gpu_uuid", "gpu_model", "compute_capability")

EXPECTED_DETERMINISM_FLAGS: dict[str, object] = {
    key: EXPECTED_DETERMINISM_POLICY[key]
    for key in (
        "dtype",
        "amp",
        "tf32",
        "torch_compile",
        "cublas_workspace_config",
        "torch_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_tf32",
        "cudnn_tf32",
        "dataloader_workers",
        "clean_process_repeats",
        "fresh_process_bootstrap_required",
        "cublas_env_set_before_torch_import_and_cuda_initialization",
        "no_cuda_capability_query_before_bootstrap",
        "flags_attested_after_cuda_initialization",
        "abort_on_nondeterministic_warning_or_error",
    )
}
EXPECTED_SEED_POLICY: dict[str, object] = {
    "generator": "NUMPY_PCG64_ON_CPU_ONLY",
    "bit_generator": "PCG64",
    "seed_preimage_encoding": "UTF8_U32BE_LENGTH_PREFIXED_FIELDS_V1",
    "seed_digest_to_integer": "SHA256_FIRST_128_BITS_BIG_ENDIAN_UNSIGNED",
    "shock_dtype": "LITTLE_ENDIAN_FLOAT64",
    "shock_array_layout": "C_CONTIGUOUS",
    "shock_semantic_hash_scheme": "SHA256_DTYPE_ENDIAN_SHAPE_AND_C_ORDER_BYTES_V1",
    "shock_stream_inventory_sha256": None,
    "shock_bytes_sha256": None,
    "cuda_rng_used": False,
    "candidate_id_in_standardized_shock_seed": False,
    "common_random_numbers": True,
}
ROW_ORDER_FIELDS = (
    "row_ids_sha256",
    "scenario_ids_sha256",
    "target_mask_sha256",
    "weights_sha256",
    "chunk_plan_sha256",
    "row_count",
    "scenario_count",
    "chunk_count",
    "serialization",
)
BACKEND_FIELDS = (
    "workload_scope",
    "backend",
    "fallback_used",
    "fallback_reason",
    "gpu_fit_attempted",
    "mixed_backend",
    "rng_started_before_fallback",
    "output_started_before_fallback",
)
RECEIPT_HASH_FIELDS = {
    "parity_qualification_report": "parity_qualification_report_sha256",
    "fresh_process_bootstrap": "fresh_process_bootstrap_sha256",
    "deterministic_operation_inventory": "deterministic_operation_inventory_sha256",
    "cpu_oracle_implementation": "cpu_oracle_implementation_sha256",
    "cpu_oracle_runtime_fingerprint": "cpu_oracle_runtime_fingerprint_sha256",
    "prediction_seal": "prediction_seal_sha256",
    "scenario_seal": "scenario_seal_sha256",
    "truth_open_receipt": "truth_open_receipt_sha256",
}
RECEIPT_STATUS = {
    "parity_qualification_report": "PASS",
    "fresh_process_bootstrap": "PASS",
    "deterministic_operation_inventory": "PASS",
    "cpu_oracle_implementation": "VERIFIED",
    "cpu_oracle_runtime_fingerprint": "VERIFIED",
    "prediction_seal": "SEALED",
    "scenario_seal": "SEALED",
    "truth_open_receipt": "OPENED_AFTER_SEALS_VERIFIED",
}
RECEIPT_PAYLOAD_ROLES = {
    "parity_qualification_report": ("parity_report",),
    "fresh_process_bootstrap": ("bootstrap_report",),
    "deterministic_operation_inventory": ("operation_inventory",),
    "cpu_oracle_implementation": ("cpu_oracle_source",),
    "cpu_oracle_runtime_fingerprint": ("cpu_oracle_runtime",),
    "prediction_seal": ("predictions",),
    "scenario_seal": ("scenarios",),
    "truth_open_receipt": (
        "prediction_seal_receipt",
        "scenario_seal_receipt",
        "truth_dataset",
    ),
}
RAW_BOUND_PAYLOAD_ROLES = tuple(
    sorted(
        {
            payload_role
            for roles in RECEIPT_PAYLOAD_ROLES.values()
            for payload_role in roles
            if payload_role not in {"prediction_seal_receipt", "scenario_seal_receipt"}
        }
    )
)
EXECUTION_CONTEXT_EXCLUDED_FIELDS = frozenset(RECEIPT_HASH_FIELDS.values())
SCENARIO_DESIGN_ROLE = "probabilistic_scenario_and_mc_design_manifest"
MONTE_CARLO_STUDY_ROLE = "monte_carlo_error_study"
DESIGN_FREEZE_RECEIPT_ROLE = "design_freeze_receipt"
SCENARIO_DESIGN_FIELDS = (
    "schema_version",
    "compute_runtime_contract_id",
    "compute_runtime_document_sha256",
    "plan_id",
    "comparison_family_id",
    "scenario_count",
    "chunk_count",
    "chunk_plan_sha256",
    "shock_stream_inventory_sha256",
    "monte_carlo_error_study_sha256",
    "design_freeze_receipt_sha256",
    "frozen_positive_reference_scale",
    "qualification_data",
    "frozen_before_holdout",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
)
MONTE_CARLO_STUDY_FIELDS = (
    "schema_version",
    "compute_runtime_contract_id",
    "compute_runtime_document_sha256",
    "plan_id",
    "comparison_family_id",
    "qualification_data",
    "estimator",
    "common_random_numbers",
    "candidate_scenario_counts",
    "selected_scenario_count",
    "selected_chunk_count",
    "selected_chunk_plan_sha256",
    "shock_stream_inventory_sha256",
    "reference_scale_eur_mwh",
    "positive_margin_eur_mwh",
    "positive_margin_contrast_half_width_eur_mwh",
    "zero_margin_one_sided_half_width_eur_mwh",
    "confidence_level",
    "future_holdout_used",
    "frozen_before_holdout",
    "status",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
)
DESIGN_FREEZE_RECEIPT_FIELDS = (
    "schema_version",
    "compute_runtime_contract_id",
    "compute_runtime_document_sha256",
    "plan_id",
    "comparison_family_id",
    "scenario_design_core_sha256",
    "monte_carlo_error_study_sha256",
    "status",
    "frozen_before_holdout",
    "external_signature_verified",
    "trusted_time_verified",
    "ledger_sequence_verified",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
)
BLOCKERS = (
    "LOCAL_RECEIPTS_NOT_INDEPENDENTLY_SIGNED_OR_ATTESTED",
    "INDEPENDENT_RUNTIME_ADMISSION_ENVELOPE_ABSENT",
    "COMPUTE_CONTRACT_REMAINS_NON_EXECUTABLE",
    "PRODUCTION_AUTHORITY_FORBIDDEN",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ChLtComputeRuntimeManifestError(ValueError):
    """Raised when a runtime manifest or bound receipt is ambiguous."""


@dataclass(frozen=True)
class ComputeRuntimeManifestAssessment:
    manifest_sha256: str
    compute_runtime_contract_id: str
    compute_runtime_document_sha256: str
    validator_policy_sha256: str
    validator_implementation_sha256: str
    verified_evidence_bytes_sha256: Mapping[str, str]
    blockers: tuple[str, ...] = BLOCKERS

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": STATUS,
            "manifest_sha256": self.manifest_sha256,
            "compute_runtime_contract_id": self.compute_runtime_contract_id,
            "compute_runtime_document_sha256": self.compute_runtime_document_sha256,
            "validator_policy_sha256": self.validator_policy_sha256,
            "validator_implementation_sha256": self.validator_implementation_sha256,
            "verified_evidence_bytes_sha256": dict(self.verified_evidence_bytes_sha256),
            "blockers": list(self.blockers),
            "structure_valid": True,
            "receipt_and_bound_payload_bytes_verified": True,
            "execution_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
        }


def validator_policy_sha256() -> str:
    policy = {
        "schema_version": SCHEMA_VERSION,
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "scenario_design_schema_version": SCENARIO_DESIGN_SCHEMA_VERSION,
        "monte_carlo_study_schema_version": MONTE_CARLO_STUDY_SCHEMA_VERSION,
        "design_freeze_receipt_schema_version": DESIGN_FREEZE_RECEIPT_SCHEMA_VERSION,
        "manifest_schema": EXPECTED_RUNTIME_MANIFEST_SCHEMA,
        "input_hash_roles": INPUT_HASH_ROLES,
        "code_and_wheel_hash_roles": CODE_AND_WHEEL_HASH_ROLES,
        "version_fields": VERSION_FIELDS,
        "cuda_fields": CUDA_FIELDS,
        "host_fields": HOST_FIELDS,
        "determinism_flags": EXPECTED_DETERMINISM_FLAGS,
        "seed_policy": EXPECTED_SEED_POLICY,
        "row_order_fields": ROW_ORDER_FIELDS,
        "backend_fields": BACKEND_FIELDS,
        "receipt_hash_fields": RECEIPT_HASH_FIELDS,
        "receipt_status": RECEIPT_STATUS,
        "receipt_payload_roles": RECEIPT_PAYLOAD_ROLES,
        "raw_bound_payload_roles": RAW_BOUND_PAYLOAD_ROLES,
        "execution_context_excluded_fields": sorted(EXECUTION_CONTEXT_EXCLUDED_FIELDS),
        "scenario_design_fields": SCENARIO_DESIGN_FIELDS,
        "monte_carlo_study_fields": MONTE_CARLO_STUDY_FIELDS,
        "design_freeze_receipt_fields": DESIGN_FREEZE_RECEIPT_FIELDS,
        "blockers": BLOCKERS,
    }
    return hashlib.sha256(canonical_json_bytes(policy)).hexdigest()


def validator_implementation_sha256() -> str:
    payload = read_stable_single_link_file(
        Path(__file__),
        label="compute runtime manifest validator implementation",
        max_bytes=2 * 1024 * 1024,
    )
    return hashlib.sha256(payload).hexdigest()


def compute_execution_context_sha256(manifest: Mapping[str, object]) -> str:
    """Bind every execution fact while excluding cyclic receipt-file hashes."""

    context = {
        key: manifest[key]
        for key in sorted(manifest)
        if key not in EXECUTION_CONTEXT_EXCLUDED_FIELDS
    }
    return hashlib.sha256(canonical_json_bytes(context)).hexdigest()


def compute_scenario_design_core_sha256(design: Mapping[str, object]) -> str:
    core = dict(design)
    core.pop("design_freeze_receipt_sha256", None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def load_compute_runtime_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, object], bytes]:
    _require_sha256(expected_sha256, label="compute runtime manifest")
    payload = read_stable_single_link_file(
        path,
        label="CH LT compute runtime manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ChLtComputeRuntimeManifestError("compute runtime manifest SHA-256 mismatch")
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtComputeRuntimeManifestError(
            "compute runtime manifest is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ChLtComputeRuntimeManifestError("compute runtime manifest must be a mapping")
    return parsed, payload


def assess_compute_runtime_manifest(
    manifest: Mapping[str, object],
    *,
    manifest_bytes: bytes,
    compute_contract: Mapping[str, object],
    compute_contract_bytes: bytes,
    evidence_root: Path,
    evidence_paths: Mapping[str, Path],
) -> ComputeRuntimeManifestAssessment:
    contract_assessment = assess_compute_runtime(
        compute_contract,
        document_bytes=compute_contract_bytes,
    )
    _rebind_manifest_bytes(manifest, manifest_bytes)
    expected_fields = EXPECTED_RUNTIME_MANIFEST_SCHEMA["field_types"]
    if not isinstance(expected_fields, Mapping) or set(manifest) != set(expected_fields):
        raise ChLtComputeRuntimeManifestError("runtime manifest top-level keys are not exact")
    _require_exact(manifest.get("schema_version"), SCHEMA_VERSION, "manifest schema version")
    _require_exact(
        manifest.get("compute_runtime_contract_id"),
        contract_assessment.contract_id,
        "compute runtime contract ID",
    )
    _require_exact(
        manifest.get("compute_runtime_document_sha256"),
        contract_assessment.document_sha256,
        "compute runtime document SHA-256",
    )
    _validate_scalar_fields(manifest)
    _validate_hash_mapping(manifest.get("input_hashes"), INPUT_HASH_ROLES, "input hashes")
    code_hashes = _validate_hash_mapping(
        manifest.get("code_and_wheel_hashes"),
        CODE_AND_WHEEL_HASH_ROLES,
        "code and wheel hashes",
    )
    _require_exact(manifest.get("lock_hash"), code_hashes["dependency_lock"], "lock hash")
    _validate_string_mapping(
        manifest.get("python_numpy_torch_versions"), VERSION_FIELDS, "runtime versions"
    )
    _validate_string_mapping(
        manifest.get("cuda_build_runtime_cudnn_driver"), CUDA_FIELDS, "CUDA versions"
    )
    _validate_string_mapping(
        manifest.get("os_gpu_uuid_model_compute_capability"), HOST_FIELDS, "host identity"
    )
    _require_exact(
        manifest.get("dtype_and_determinism_flags"),
        EXPECTED_DETERMINISM_FLAGS,
        "determinism flags",
    )
    _validate_seed_policy(manifest.get("seed_tree_and_shock_hashes"))
    _validate_row_order(manifest.get("row_scenario_and_chunk_order"))
    workload_scope = _validate_backend_policy(manifest.get("backend_and_fallback_reason"))
    _validate_frozen_weights_field(manifest, workload_scope=workload_scope)
    verified = _verify_evidence(
        manifest,
        evidence_root=evidence_root,
        evidence_paths=evidence_paths,
        workload_scope=workload_scope,
    )
    return ComputeRuntimeManifestAssessment(
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        compute_runtime_contract_id=contract_assessment.contract_id,
        compute_runtime_document_sha256=contract_assessment.document_sha256,
        validator_policy_sha256=validator_policy_sha256(),
        validator_implementation_sha256=validator_implementation_sha256(),
        verified_evidence_bytes_sha256=verified,
    )


def _validate_scalar_fields(manifest: Mapping[str, object]) -> None:
    for field in (
        "parity_qualification_report_sha256",
        "fresh_process_bootstrap_sha256",
        "deterministic_operation_inventory_sha256",
        "cpu_oracle_implementation_sha256",
        "cpu_oracle_runtime_fingerprint_sha256",
        "prediction_seal_sha256",
        "scenario_seal_sha256",
        "truth_open_receipt_sha256",
        "plan_id",
    ):
        _require_sha256(manifest.get(field), label=field)
    for field in ("attempt_id", "origin_id", "candidate_id"):
        _require_nonempty_string(manifest.get(field), label=field)


def _validate_hash_mapping(
    value: object,
    roles: Sequence[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(roles):
        raise ChLtComputeRuntimeManifestError(f"{label} keys are not exact")
    for role in roles:
        _require_sha256(value[role], label=f"{label}.{role}")
    return value


def _validate_string_mapping(value: object, fields: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ChLtComputeRuntimeManifestError(f"{label} keys are not exact")
    for field in fields:
        _require_nonempty_string(value[field], label=f"{label}.{field}")


def _validate_seed_policy(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_SEED_POLICY):
        raise ChLtComputeRuntimeManifestError("seed and shock mapping keys are not exact")
    for field, expected in EXPECTED_SEED_POLICY.items():
        if expected is None:
            _require_sha256(value[field], label=f"seed policy.{field}")
        else:
            _require_exact(value[field], expected, f"seed policy.{field}")


def _validate_row_order(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(ROW_ORDER_FIELDS):
        raise ChLtComputeRuntimeManifestError("row/scenario/chunk mapping keys are not exact")
    for field in ROW_ORDER_FIELDS[:5]:
        _require_sha256(value[field], label=f"row order.{field}")
    for field in ("row_count", "scenario_count", "chunk_count"):
        count = value[field]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ChLtComputeRuntimeManifestError(f"row order.{field} must be a positive integer")
    _require_exact(
        value["serialization"],
        "DTYPE_ENDIAN_SHAPE_AND_C_ORDER_BYTES_V1",
        "row order.serialization",
    )


def _validate_backend_policy(value: object) -> str:
    if not isinstance(value, Mapping) or set(value) != set(BACKEND_FIELDS):
        raise ChLtComputeRuntimeManifestError("backend/fallback mapping keys are not exact")
    workload_scope = _require_nonempty_string(value["workload_scope"], label="workload scope")
    allowed_scopes = set(EXPECTED_AUTHORITY_BOUNDARY["cpu_float64_oracle_scopes"]) | set(
        EXPECTED_AUTHORITY_BOUNDARY["gpu_allowed_scopes_v1"]
    )
    if workload_scope not in allowed_scopes:
        raise ChLtComputeRuntimeManifestError("workload scope is outside compute authority")
    backend = value["backend"]
    if backend not in ("CPU", "GPU"):
        raise ChLtComputeRuntimeManifestError("backend must be CPU or GPU")
    for field in (
        "fallback_used",
        "gpu_fit_attempted",
        "mixed_backend",
        "rng_started_before_fallback",
        "output_started_before_fallback",
    ):
        if not isinstance(value[field], bool):
            raise ChLtComputeRuntimeManifestError(f"backend policy.{field} must be boolean")
    if value["gpu_fit_attempted"] is not False or value["mixed_backend"] is not False:
        raise ChLtComputeRuntimeManifestError("GPU fit and mixed backend are forbidden")
    fallback_reason = _require_nonempty_string(
        value["fallback_reason"], label="fallback reason"
    )
    if value["fallback_used"] is True:
        if (
            backend != "CPU"
            or fallback_reason == "NOT_APPLICABLE"
            or workload_scope
            not in set(EXPECTED_AUTHORITY_BOUNDARY["gpu_allowed_scopes_v1"])
        ):
            raise ChLtComputeRuntimeManifestError("fallback relation is invalid")
        if value["rng_started_before_fallback"] or value["output_started_before_fallback"]:
            raise ChLtComputeRuntimeManifestError("fallback occurred after RNG or output")
    elif (
        fallback_reason != "NOT_APPLICABLE"
        or value["rng_started_before_fallback"]
        or value["output_started_before_fallback"]
    ):
        raise ChLtComputeRuntimeManifestError(
            "unused fallback fields must be NOT_APPLICABLE and false"
        )
    if backend == "GPU" and workload_scope not in set(
        EXPECTED_AUTHORITY_BOUNDARY["gpu_allowed_scopes_v1"]
    ):
        raise ChLtComputeRuntimeManifestError("GPU backend is forbidden for this workload scope")
    return workload_scope


def _validate_frozen_weights_field(
    manifest: Mapping[str, object],
    *,
    workload_scope: str,
) -> None:
    value = manifest.get("frozen_shaping_weights_sha256")
    if workload_scope == "SHAPING_INFERENCE_FROM_FROZEN_WEIGHTS":
        _require_sha256(value, label="frozen shaping weights")
    elif value != "NOT_APPLICABLE":
        raise ChLtComputeRuntimeManifestError(
            "frozen shaping weights must be NOT_APPLICABLE outside shaping inference"
        )


def _verify_evidence(
    manifest: Mapping[str, object],
    *,
    evidence_root: Path,
    evidence_paths: Mapping[str, Path],
    workload_scope: str,
) -> dict[str, str]:
    root = assert_absolute_path_has_no_links(evidence_root)
    if not root.is_dir():
        raise ChLtComputeRuntimeManifestError("evidence root is unavailable")
    required_roles = (
        set(RECEIPT_HASH_FIELDS)
        | set(RAW_BOUND_PAYLOAD_ROLES)
        | {SCENARIO_DESIGN_ROLE, MONTE_CARLO_STUDY_ROLE, DESIGN_FREEZE_RECEIPT_ROLE}
    )
    if workload_scope == "SHAPING_INFERENCE_FROM_FROZEN_WEIGHTS":
        required_roles.add("frozen_shaping_weights")
    if set(evidence_paths) != required_roles:
        raise ChLtComputeRuntimeManifestError("evidence path role inventory is not exact")
    checked_paths: list[Path] = []
    captured: dict[str, bytes] = {}
    for role in sorted(required_roles):
        path = assert_absolute_path_has_no_links(evidence_paths[role])
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ChLtComputeRuntimeManifestError(
                f"evidence path escapes evidence root: {role}"
            ) from exc
        for prior in checked_paths:
            try:
                if os.path.samefile(path, prior):
                    raise ChLtComputeRuntimeManifestError(
                        "evidence paths must be physically role-distinct"
                    )
            except OSError as exc:
                raise ChLtComputeRuntimeManifestError(
                    "evidence path identity cannot be established"
                ) from exc
        checked_paths.append(path)
        if role == "frozen_shaping_weights":
            payload = read_stable_single_link_file(
                path,
                label="frozen shaping weights",
                max_bytes=MAX_FROZEN_WEIGHTS_BYTES,
            )
            expected = str(manifest["frozen_shaping_weights_sha256"])
        elif role == SCENARIO_DESIGN_ROLE:
            payload = read_stable_single_link_file(
                path,
                label="probabilistic scenario and Monte Carlo design manifest",
                max_bytes=MAX_RECEIPT_BYTES,
            )
            input_hashes = manifest["input_hashes"]
            assert isinstance(input_hashes, Mapping)
            expected = str(input_hashes[SCENARIO_DESIGN_ROLE])
        elif role in {MONTE_CARLO_STUDY_ROLE, DESIGN_FREEZE_RECEIPT_ROLE}:
            payload = read_stable_single_link_file(
                path,
                label=f"probabilistic scenario design evidence {role}",
                max_bytes=MAX_RECEIPT_BYTES,
            )
            expected = "DEFERRED_TO_SCENARIO_DESIGN"
        elif role in RAW_BOUND_PAYLOAD_ROLES:
            payload = read_stable_single_link_file(
                path,
                label=f"compute runtime bound payload {role}",
                max_bytes=MAX_BOUND_PAYLOAD_BYTES,
            )
            expected = "DEFERRED_TO_RECEIPT"
        else:
            payload = read_stable_single_link_file(
                path,
                label=f"compute runtime {role} receipt",
                max_bytes=MAX_RECEIPT_BYTES,
            )
            expected = str(manifest[RECEIPT_HASH_FIELDS[role]])
        actual = hashlib.sha256(payload).hexdigest()
        if expected not in {"DEFERRED_TO_RECEIPT", "DEFERRED_TO_SCENARIO_DESIGN"} and (
            actual != expected
        ):
            raise ChLtComputeRuntimeManifestError(f"evidence SHA-256 mismatch: {role}")
        captured[role] = payload
    scenario_design = _validate_scenario_design(
        captured[SCENARIO_DESIGN_ROLE], manifest=manifest
    )
    scenario_support_hash_fields = {
        MONTE_CARLO_STUDY_ROLE: "monte_carlo_error_study_sha256",
        DESIGN_FREEZE_RECEIPT_ROLE: "design_freeze_receipt_sha256",
    }
    for role, field in scenario_support_hash_fields.items():
        if hashlib.sha256(captured[role]).hexdigest() != scenario_design[field]:
            raise ChLtComputeRuntimeManifestError(f"scenario design evidence SHA-256 mismatch: {role}")
    _validate_monte_carlo_study(
        captured[MONTE_CARLO_STUDY_ROLE],
        design=scenario_design,
        manifest=manifest,
    )
    _validate_design_freeze_receipt(
        captured[DESIGN_FREEZE_RECEIPT_ROLE],
        design=scenario_design,
        manifest=manifest,
    )
    expected_payload_hashes: dict[str, str] = {}
    for role in RECEIPT_HASH_FIELDS:
        bound = _validate_receipt(captured[role], role=role, manifest=manifest)
        for payload_role, expected_hash in bound.items():
            if payload_role in {"prediction_seal_receipt", "scenario_seal_receipt"}:
                continue
            existing = expected_payload_hashes.get(payload_role)
            if existing is not None and existing != expected_hash:
                raise ChLtComputeRuntimeManifestError(
                    f"bound payload hash conflicts across receipts: {payload_role}"
                )
            expected_payload_hashes[payload_role] = expected_hash
    if set(expected_payload_hashes) != set(RAW_BOUND_PAYLOAD_ROLES):
        raise ChLtComputeRuntimeManifestError("bound payload role inventory is not exact")
    verified: dict[str, str] = {}
    for role, payload in captured.items():
        actual = hashlib.sha256(payload).hexdigest()
        if role in expected_payload_hashes and actual != expected_payload_hashes[role]:
            raise ChLtComputeRuntimeManifestError(f"bound payload SHA-256 mismatch: {role}")
        verified[role] = actual
    return verified


def _validate_scenario_design(
    payload: bytes, *, manifest: Mapping[str, object]
) -> Mapping[str, object]:
    try:
        design = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtComputeRuntimeManifestError(
            "scenario/Monte Carlo design is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(design, Mapping) or set(design) != set(SCENARIO_DESIGN_FIELDS):
        raise ChLtComputeRuntimeManifestError("scenario/Monte Carlo design keys are not exact")
    _require_exact(
        design["schema_version"],
        SCENARIO_DESIGN_SCHEMA_VERSION,
        "scenario/Monte Carlo design schema",
    )
    for field in (
        "compute_runtime_contract_id",
        "compute_runtime_document_sha256",
        "plan_id",
    ):
        _require_exact(design[field], manifest[field], f"scenario design.{field}")
    _require_nonempty_string(
        design["comparison_family_id"], label="scenario design.comparison_family_id"
    )
    row_order = manifest["row_scenario_and_chunk_order"]
    seed_policy = manifest["seed_tree_and_shock_hashes"]
    assert isinstance(row_order, Mapping)
    assert isinstance(seed_policy, Mapping)
    for field in ("scenario_count", "chunk_count", "chunk_plan_sha256"):
        _require_exact(design[field], row_order[field], f"scenario design.{field}")
    _require_exact(
        design["shock_stream_inventory_sha256"],
        seed_policy["shock_stream_inventory_sha256"],
        "scenario design.shock_stream_inventory_sha256",
    )
    for field in ("monte_carlo_error_study_sha256", "design_freeze_receipt_sha256"):
        _require_sha256(design[field], label=f"scenario design.{field}")
    reference_scale = design["frozen_positive_reference_scale"]
    if (
        not isinstance(reference_scale, (int, float))
        or isinstance(reference_scale, bool)
        or not math.isfinite(float(reference_scale))
        or reference_scale <= 0
    ):
        raise ChLtComputeRuntimeManifestError(
            "scenario design frozen reference scale must be finite and positive"
        )
    _require_exact(
        design["qualification_data"],
        "ADMITTED_NON_HOLDOUT_ONLY",
        "scenario design.qualification_data",
    )
    for field, expected in (
        ("frozen_before_holdout", True),
        ("execution_authorized", False),
        ("production_authorization", False),
        ("promotion_gate", False),
    ):
        _require_exact(design[field], expected, f"scenario design.{field}")
    return design


def _validate_monte_carlo_study(
    payload: bytes,
    *,
    design: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    try:
        study = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtComputeRuntimeManifestError(
            "Monte Carlo error study is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(study, Mapping) or set(study) != set(MONTE_CARLO_STUDY_FIELDS):
        raise ChLtComputeRuntimeManifestError("Monte Carlo error study keys are not exact")
    _require_exact(
        study["schema_version"],
        MONTE_CARLO_STUDY_SCHEMA_VERSION,
        "Monte Carlo study schema",
    )
    for field in ("compute_runtime_contract_id", "compute_runtime_document_sha256", "plan_id"):
        _require_exact(study[field], manifest[field], f"Monte Carlo study.{field}")
    _require_exact(
        study["comparison_family_id"],
        design["comparison_family_id"],
        "Monte Carlo study.comparison_family_id",
    )
    _require_exact(
        study["qualification_data"],
        "ADMITTED_NON_HOLDOUT_ONLY",
        "Monte Carlo study.qualification_data",
    )
    _require_exact(
        study["estimator"],
        "PAIRED_COMMON_RANDOM_NUMBER_CONTRAST",
        "Monte Carlo study.estimator",
    )
    _require_exact(study["common_random_numbers"], True, "Monte Carlo study CRN")
    counts = study["candidate_scenario_counts"]
    if (
        not isinstance(counts, list)
        or len(counts) < 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts)
        or counts != sorted(set(counts))
    ):
        raise ChLtComputeRuntimeManifestError(
            "Monte Carlo candidate scenario counts must be sorted unique positive integers"
        )
    _require_exact(
        study["selected_scenario_count"],
        design["scenario_count"],
        "Monte Carlo study selected scenario count",
    )
    if study["selected_scenario_count"] not in counts:
        raise ChLtComputeRuntimeManifestError("selected scenario count is absent from MC candidates")
    for study_field, design_field in (
        ("selected_chunk_count", "chunk_count"),
        ("selected_chunk_plan_sha256", "chunk_plan_sha256"),
        ("shock_stream_inventory_sha256", "shock_stream_inventory_sha256"),
    ):
        _require_exact(
            study[study_field], design[design_field], f"Monte Carlo study.{study_field}"
        )
    reference_scale = _require_finite_number(
        study["reference_scale_eur_mwh"],
        label="Monte Carlo reference scale",
        minimum=0.0,
        strictly_greater=True,
    )
    _require_exact(
        reference_scale,
        design["frozen_positive_reference_scale"],
        "Monte Carlo frozen reference scale",
    )
    positive_margin = _require_finite_number(
        study["positive_margin_eur_mwh"],
        label="Monte Carlo positive margin",
        minimum=0.0,
        strictly_greater=True,
    )
    positive_half_width = _require_finite_number(
        study["positive_margin_contrast_half_width_eur_mwh"],
        label="Monte Carlo positive-margin half-width",
        minimum=0.0,
    )
    zero_half_width = _require_finite_number(
        study["zero_margin_one_sided_half_width_eur_mwh"],
        label="Monte Carlo zero-margin half-width",
        minimum=0.0,
    )
    if positive_half_width > min(0.10 * positive_margin, 0.001 * reference_scale):
        raise ChLtComputeRuntimeManifestError("positive-margin Monte Carlo precision fails")
    if zero_half_width > 0.001 * reference_scale:
        raise ChLtComputeRuntimeManifestError("zero-margin Monte Carlo precision fails")
    _require_exact(study["confidence_level"], 0.95, "Monte Carlo confidence level")
    for field, expected in (
        ("future_holdout_used", False),
        ("frozen_before_holdout", True),
        ("status", "PASS_FROZEN_DESIGN_SUPPORTED"),
        ("execution_authorized", False),
        ("production_authorization", False),
        ("promotion_gate", False),
    ):
        _require_exact(study[field], expected, f"Monte Carlo study.{field}")


def _validate_design_freeze_receipt(
    payload: bytes,
    *,
    design: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    try:
        receipt = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtComputeRuntimeManifestError(
            "scenario design freeze receipt is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(receipt, Mapping) or set(receipt) != set(DESIGN_FREEZE_RECEIPT_FIELDS):
        raise ChLtComputeRuntimeManifestError("scenario design freeze receipt keys are not exact")
    _require_exact(
        receipt["schema_version"],
        DESIGN_FREEZE_RECEIPT_SCHEMA_VERSION,
        "scenario design freeze receipt schema",
    )
    for field in ("compute_runtime_contract_id", "compute_runtime_document_sha256", "plan_id"):
        _require_exact(receipt[field], manifest[field], f"design freeze receipt.{field}")
    _require_exact(
        receipt["comparison_family_id"],
        design["comparison_family_id"],
        "design freeze receipt.comparison_family_id",
    )
    _require_exact(
        receipt["scenario_design_core_sha256"],
        compute_scenario_design_core_sha256(design),
        "design freeze receipt scenario core",
    )
    _require_exact(
        receipt["monte_carlo_error_study_sha256"],
        design["monte_carlo_error_study_sha256"],
        "design freeze receipt Monte Carlo study",
    )
    for field, expected in (
        ("status", "LOCAL_FREEZE_RECEIPT_NOT_TRUSTED_TIME"),
        ("frozen_before_holdout", True),
        ("external_signature_verified", False),
        ("trusted_time_verified", False),
        ("ledger_sequence_verified", False),
        ("execution_authorized", False),
        ("production_authorization", False),
        ("promotion_gate", False),
    ):
        _require_exact(receipt[field], expected, f"design freeze receipt.{field}")


def _validate_receipt(
    payload: bytes,
    *,
    role: str,
    manifest: Mapping[str, object],
) -> Mapping[str, object]:
    try:
        receipt = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtComputeRuntimeManifestError(f"{role} receipt is not strict UTF-8 JSON") from exc
    expected_keys = {
        "schema_version",
        "role",
        "compute_runtime_contract_id",
        "compute_runtime_document_sha256",
        "plan_id",
        "attempt_id",
        "origin_id",
        "candidate_id",
        "workload_scope",
        "execution_context_sha256",
        "status",
        "bound_payload_hashes",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise ChLtComputeRuntimeManifestError(f"{role} receipt keys are not exact")
    for field in (
        "compute_runtime_contract_id",
        "compute_runtime_document_sha256",
        "plan_id",
        "attempt_id",
        "origin_id",
        "candidate_id",
    ):
        _require_exact(receipt[field], manifest[field], f"{role} receipt.{field}")
    workload_scope = manifest["backend_and_fallback_reason"]
    assert isinstance(workload_scope, Mapping)
    _require_exact(
        receipt["workload_scope"], workload_scope["workload_scope"], f"{role} workload"
    )
    _require_exact(receipt["schema_version"], RECEIPT_SCHEMA_VERSION, f"{role} schema")
    _require_exact(receipt["role"], role, f"{role} role")
    _require_exact(
        receipt["execution_context_sha256"],
        compute_execution_context_sha256(manifest),
        f"{role} execution context",
    )
    _require_exact(receipt["status"], RECEIPT_STATUS[role], f"{role} status")
    payload_hashes = _validate_hash_mapping(
        receipt["bound_payload_hashes"],
        RECEIPT_PAYLOAD_ROLES[role],
        f"{role} bound payload hashes",
    )
    if role == "truth_open_receipt":
        _require_exact(
            payload_hashes["prediction_seal_receipt"],
            manifest["prediction_seal_sha256"],
            "truth-open prediction seal binding",
        )
        _require_exact(
            payload_hashes["scenario_seal_receipt"],
            manifest["scenario_seal_sha256"],
            "truth-open scenario seal binding",
        )
    return payload_hashes


def _rebind_manifest_bytes(manifest: Mapping[str, object], payload: bytes) -> None:
    try:
        rebound = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtComputeRuntimeManifestError("manifest bytes are not strict UTF-8 JSON") from exc
    if not isinstance(rebound, Mapping) or not _strict_equal(rebound, manifest):
        raise ChLtComputeRuntimeManifestError("manifest bytes do not bind assessed mapping")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ChLtComputeRuntimeManifestError(f"{label} must be lowercase SHA-256")
    return value


def _require_nonempty_string(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ChLtComputeRuntimeManifestError(f"{label} must be a clean non-empty string")
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    minimum: float,
    strictly_greater: bool = False,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ChLtComputeRuntimeManifestError(f"{label} must be finite numeric")
    numeric = float(value)
    if (strictly_greater and numeric <= minimum) or (
        not strictly_greater and numeric < minimum
    ):
        comparator = ">" if strictly_greater else ">="
        raise ChLtComputeRuntimeManifestError(f"{label} must be {comparator} {minimum}")
    return numeric


def _require_exact(actual: object, expected: object, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ChLtComputeRuntimeManifestError(f"{label} is not exact")


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return False
        return all(_strict_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
            return False
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)
