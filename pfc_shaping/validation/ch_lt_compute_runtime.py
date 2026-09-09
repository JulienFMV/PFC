"""Structural CPU/GPU contract for future Swiss LT evaluation.

This module validates policy bytes only.  A valid document is never runtime
attestation, evaluation admission, candidate evidence, or production authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_compute_runtime.v1"
AUDIT_SCHEMA_VERSION = "ch_lt_compute_runtime_audit.v1"
LIFECYCLE_STATE = "DRAFT_LOCAL_OBSERVATION_NOT_RUNTIME_ATTESTATION"
STATUS = "VALID_STRUCTURAL_COMPUTE_POLICY_NOT_EXECUTION_AUTHORITY"
MAX_DOCUMENT_BYTES = 512 * 1024

EXPECTED_AUTHORITY_BOUNDARY: dict[str, object] = {
    "cpu_float64_oracle_scopes": [
        "MONTHLY_SOLVER",
        "EEX_REPRICING",
        "CASCADE_AND_QUOTE_SENSITIVITY_GATES",
        "ENSEMBLE_MONTHLY_FORWARD_CONSISTENCY",
        "FINAL_MARKET_COHERENCE_PROJECTION",
        "ALL_HARD_GATES",
    ],
    "gpu_allowed_scopes_v1": [
        "SCENARIO_TRANSFORM",
        "SCENARIO_SCORING",
        "SHAPING_INFERENCE_FROM_FROZEN_WEIGHTS",
    ],
    "gpu_forbidden_scopes_v1": [
        "MONTHLY_SOLVER",
        "EEX_REPRICING",
        "CASCADE_CALIBRATION",
        "ACQUISITION_ADMISSION",
        "PUBLICATION",
        "PROMOTION",
        "SHAPING_FIT_OR_MODEL_SELECTION",
    ],
    "scenario_level_policy": (
        "ENERGY_WEIGHTED_ENSEMBLE_EXPECTATION_EQUALS_MONTHLY_SOLVER_FORWARD"
    ),
    "scenario_specific_monthly_level_risk": (
        "ALLOWED_ONLY_WITH_ZERO_MEAN_ENSEMBLE_AND_FROZEN_COVARIANCE"
    ),
    "pathwise_fixed_monthly_level": "SEPARATE_SHAPE_ONLY_DIAGNOSTIC_PRODUCT",
}

EXPECTED_DETERMINISM_POLICY: dict[str, object] = {
    "dtype": "FLOAT64",
    "amp": False,
    "tf32": False,
    "torch_compile": False,
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "matmul_tf32": False,
    "cudnn_tf32": False,
    "dataloader_workers": 0,
    "exact_row_and_scenario_order": True,
    "clean_process_repeats": 3,
    "intra_backend_semantic_hashes_bitwise_equal": True,
    "fresh_process_bootstrap_required": True,
    "fresh_process_bootstrap_sha256_required": True,
    "cublas_env_set_before_torch_import_and_cuda_initialization": True,
    "no_cuda_capability_query_before_bootstrap": True,
    "flags_attested_after_cuda_initialization": True,
    "deterministic_operation_inventory_sha256_required": True,
    "abort_on_nondeterministic_warning_or_error": True,
}

EXPECTED_RNG_POLICY: dict[str, object] = {
    "standardized_shock_generator": "NUMPY_PCG64_ON_CPU_ONLY",
    "numpy_bit_generator": "PCG64",
    "seed_fields": [
        "PLAN_ID",
        "ORIGIN_ID",
        "COMPARISON_FAMILY_OR_SHOCK_STREAM_ID",
        "SCENARIO_ID",
        "DOMAIN",
    ],
    "candidate_id_in_standardized_shock_seed": False,
    "seed_preimage_encoding": "UTF8_U32BE_LENGTH_PREFIXED_FIELDS_V1",
    "seed_digest_to_integer": "SHA256_FIRST_128_BITS_BIG_ENDIAN_UNSIGNED",
    "shock_dtype": "LITTLE_ENDIAN_FLOAT64",
    "shock_array_layout": "C_CONTIGUOUS",
    "shock_semantic_hash": "SHA256_DTYPE_ENDIAN_SHAPE_AND_C_ORDER_BYTES_V1",
    "cuda_rng_forbidden_v1": True,
    "cpu_gpu_shock_bytes_identical": True,
    "common_random_numbers_across_baseline_challenger_and_backend": True,
    "shock_bytes_hashed_before_device_copy": True,
}

EXPECTED_PARITY_POLICY: dict[str, object] = {
    "qualification_data": "ADMITTED_NON_HOLDOUT_ONLY",
    "identical_row_ids_masks_order_and_shocks": True,
    "shaping_factor_max_abs": 1e-7,
    "price_and_scenario_max_abs_eur_mwh": 1e-5,
    "price_and_scenario_rmse_eur_mwh": 1e-6,
    "score_max_abs_eur_mwh": 1e-6,
    "score_max_relative": 1e-7,
    "probability_and_coverage_max_abs": 1e-8,
    "all_values_finite": True,
    "exact_shape_row_order_weights_and_mask": True,
    "semantic_serialization": "DTYPE_ENDIAN_SHAPE_AND_C_ORDER_BYTES_V1",
    "relative_reference_epsilon": 1e-12,
    "absolute_relative_combination": (
        "ABSOLUTE_ALWAYS_AND_RELATIVE_WHEN_ABS_CPU_REFERENCE_GE_EPSILON"
    ),
    "below_relative_epsilon": "ABSOLUTE_CRITERION_ONLY",
    "score_weighting_and_recomputation": "CPU_FLOAT64_CANONICAL_IMPLEMENTATION",
    "decisions_rankings_and_unsupported_bitwise_equal": True,
    "monthly_solver_tolerance_eur_mwh": 1e-9,
    "cascade_tolerance_eur_mwh": 1e-8,
    "eex_repricing_tolerance_eur_mwh": 1e-6,
    "ensemble_monthly_forward_tolerance_eur_mwh": 1e-9,
    "quantile_crossing_after_cpu_projection_eur_mwh": 0.0,
    "post_holdout_tolerance_widening": "FORBIDDEN",
}

EXPECTED_FALLBACK_POLICY: dict[str, object] = {
    "any_gpu_fit_attempt_v1": "FORBIDDEN_ABORT_ATTEMPT",
    "cpu_fit_is_separate_candidate_identity": True,
    "scenario_scoring_or_frozen_inference_cpu_fallback": (
        "ALLOWED_ONLY_BEFORE_FIRST_RNG_OR_OUTPUT"
    ),
    "partial_cpu_gpu_mix": "FORBIDDEN",
    "oom_runtime_changes": [
        "NO_SCENARIO_COUNT_REDUCTION",
        "NO_BATCH_OR_CHUNK_CHANGE",
        "NO_DTYPE_CHANGE",
        "NO_HYPERPARAMETER_CHANGE",
    ],
    "fallback_reason_and_backend_bound_in_attempt_manifest": True,
}

EXPECTED_MONTE_CARLO_POLICY: dict[str, object] = {
    "scenario_count": None,
    "status": "REQUIRES_PRE_FREEZE_NON_HOLDOUT_MC_ERROR_STUDY",
    "common_random_numbers": True,
    "contrast_half_width_confidence": 0.95,
    "positive_margin_superiority_rule": (
        "TWO_SIDED_95_HALF_WIDTH_LE_MIN_0.10_MARGIN_AND_0.001_POSITIVE_FROZEN_REFERENCE_SCALE"
    ),
    "zero_margin_noninferiority_rule": (
        "ONE_SIDED_95_MC_CI_AND_HALF_WIDTH_LE_0.001_POSITIVE_FROZEN_REFERENCE_SCALE"
    ),
    "reference_scale": "POSITIVE_PREFROZEN_SCORE_SCALE_REQUIRED",
    "missing_positive_reference_scale": "UNSUPPORTED",
    "freeze_count_and_chunk_plan_before_holdout": True,
    "runtime_count_or_chunk_change": "FORBIDDEN",
}

EXPECTED_ANTI_LEAKAGE_POLICY: dict[str, object] = {
    "required_hashes": [
        "ORIGIN_TARGET_MASK_AND_INNER_FOLD_INVENTORY",
        "INPUT_MANIFESTS_AND_AVAILABLE_AT_TIMES",
        "CODE_WHEELS_LOCK_AND_RUNTIME",
        "CANDIDATE_BASELINE_AND_HYPERPARAMETER_GRID",
    ],
    "fit_inside_each_origin": [
        "SCALER",
        "IMPUTATION",
        "FEATURE_SELECTION",
        "HYPERPARAMETERS",
        "EARLY_STOPPING",
        "PROBABILISTIC_CALIBRATION",
        "RESIDUAL_CONSTRUCTION",
        "COVARIANCE",
        "COPULA_OR_DEPENDENCE_MODEL",
        "REGIME_MODEL",
    ],
    "cache_scope": "ORIGIN_SCOPED_TRAINING_DATA_CUTOFF_LE_ORIGIN",
    "cache_manifest_required_fields": [
        "ORIGIN_ID",
        "TRAINING_DATA_CUTOFF",
        "EXACT_TRAINING_ROW_HASHES",
        "FIT_SCOPE",
        "CODE_HASH",
        "CONFIG_HASH",
    ],
    "global_cross_origin_cache": (
        "FORBIDDEN_UNLESS_EXACT_TRAINING_ROWS_HAVE_AVAILABLE_AT_LE_EACH_CONSUMING_ORIGIN"
    ),
    "post_origin_truth_loaded_during_fit_or_generation": False,
    "fit_and_generation_filesystem_access_to_post_origin_truth": "DENIED",
    "predictions_and_scenarios_sealed_before_truth_scoring": True,
    "truth_open_requires_bound_seal_receipt": True,
    "future_data_injection_test_required": True,
}

EXPECTED_RUNTIME_MANIFEST_SCHEMA: dict[str, object] = {
    "additional_properties": False,
    "field_types": {
        "schema_version": "NONEMPTY_STRING",
        "compute_runtime_contract_id": "LOWERCASE_SHA256",
        "compute_runtime_document_sha256": "LOWERCASE_SHA256",
        "parity_qualification_report_sha256": "LOWERCASE_SHA256",
        "fresh_process_bootstrap_sha256": "LOWERCASE_SHA256",
        "deterministic_operation_inventory_sha256": "LOWERCASE_SHA256",
        "cpu_oracle_implementation_sha256": "LOWERCASE_SHA256",
        "cpu_oracle_runtime_fingerprint_sha256": "LOWERCASE_SHA256",
        "frozen_shaping_weights_sha256": "LOWERCASE_SHA256_OR_NOT_APPLICABLE",
        "prediction_seal_sha256": "LOWERCASE_SHA256",
        "scenario_seal_sha256": "LOWERCASE_SHA256",
        "truth_open_receipt_sha256": "LOWERCASE_SHA256",
        "plan_id": "LOWERCASE_SHA256",
        "attempt_id": "NONEMPTY_STRING",
        "origin_id": "NONEMPTY_STRING",
        "candidate_id": "NONEMPTY_STRING",
        "input_hashes": "CLOSED_ROLE_TO_LOWERCASE_SHA256_MAPPING",
        "code_and_wheel_hashes": "CLOSED_ROLE_TO_LOWERCASE_SHA256_MAPPING",
        "lock_hash": "LOWERCASE_SHA256",
        "python_numpy_torch_versions": "CLOSED_NONEMPTY_STRING_MAPPING",
        "cuda_build_runtime_cudnn_driver": "CLOSED_NONEMPTY_STRING_MAPPING",
        "os_gpu_uuid_model_compute_capability": "CLOSED_NONEMPTY_STRING_MAPPING",
        "dtype_and_determinism_flags": "CLOSED_TYPED_MAPPING",
        "seed_tree_and_shock_hashes": "CLOSED_TYPED_MAPPING",
        "row_scenario_and_chunk_order": "CLOSED_TYPED_MAPPING",
        "backend_and_fallback_reason": "CLOSED_TYPED_MAPPING",
    },
}

BLOCKERS = (
    "STRUCTURAL_DRAFT_NOT_EXTERNAL_RUNTIME_ATTESTATION",
    "LOCAL_CAPABILITY_AND_DETERMINISM_ATTESTATION_ABSENT",
    "GPU_PARITY_QUALIFICATION_NOT_RUN",
    "MONTE_CARLO_ERROR_STUDY_NOT_FROZEN",
    "EXTERNAL_RUNTIME_MANIFEST_AND_ADMISSION_ABSENT",
    "PREREGISTRATION_V1_PATHWISE_LEVEL_POLICY_SUPERSEDED",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ChLtComputeRuntimeError(ValueError):
    """Raised when a compute-runtime policy is ambiguous or weakened."""


@dataclass(frozen=True)
class ComputeRuntimeAssessment:
    contract_id: str
    document_sha256: str
    blockers: tuple[str, ...] = BLOCKERS

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": STATUS,
            "contract_id": self.contract_id,
            "document_sha256": self.document_sha256,
            "blockers": list(self.blockers),
            "execution_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_contract_id(document: Mapping[str, object]) -> str:
    core = dict(document)
    core.pop("contract_id", None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def derive_standardized_shock_seed(
    *,
    plan_id: str,
    origin_id: str,
    shock_stream_id: str,
    scenario_id: str,
    domain: str,
) -> int:
    """Derive the candidate-independent CRN seed specified by runtime v1."""

    _require_sha256(plan_id, label="plan_id")
    fields = (plan_id, origin_id, shock_stream_id, scenario_id, domain)
    encoded = bytearray()
    for field in fields:
        if not isinstance(field, str) or not field:
            raise ChLtComputeRuntimeError("shock seed fields must be non-empty strings")
        payload = field.encode("utf-8")
        if len(payload) > 2**32 - 1:
            raise ChLtComputeRuntimeError("shock seed field is too large")
        encoded.extend(len(payload).to_bytes(4, "big"))
        encoded.extend(payload)
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big", signed=False)


def load_compute_runtime_contract(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, object], bytes]:
    _require_sha256(expected_sha256, label="compute runtime document")
    payload = read_stable_single_link_file(
        path,
        label="CH LT compute runtime document",
        max_bytes=MAX_DOCUMENT_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ChLtComputeRuntimeError("compute runtime document SHA-256 mismatch")
    parsed = load_strict_json(payload.decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise ChLtComputeRuntimeError("compute runtime document must be a mapping")
    return parsed, payload


def assess_compute_runtime(
    document: Mapping[str, object],
    *,
    document_bytes: bytes | None = None,
) -> ComputeRuntimeAssessment:
    expected_keys = {
        "schema_version",
        "contract_id",
        "lifecycle_state",
        "scope",
        "authority_boundary",
        "determinism_policy",
        "rng_policy",
        "parity_policy",
        "fallback_policy",
        "monte_carlo_policy",
        "anti_leakage_policy",
        "runtime_manifest_schema",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
    }
    if set(document) != expected_keys:
        raise ChLtComputeRuntimeError("compute runtime top-level keys are not exact")
    _require_exact(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _require_exact(document.get("lifecycle_state"), LIFECYCLE_STATE, "lifecycle state")
    _require_exact(document.get("scope"), "SWISS_LT_NON_PRODUCTION_EVALUATION", "scope")
    _require_exact(document.get("authority_boundary"), EXPECTED_AUTHORITY_BOUNDARY, "authority")
    _require_exact(
        document.get("determinism_policy"), EXPECTED_DETERMINISM_POLICY, "determinism"
    )
    _require_exact(document.get("rng_policy"), EXPECTED_RNG_POLICY, "RNG policy")
    _require_exact(document.get("parity_policy"), EXPECTED_PARITY_POLICY, "parity policy")
    _require_exact(document.get("fallback_policy"), EXPECTED_FALLBACK_POLICY, "fallback")
    _require_exact(
        document.get("monte_carlo_policy"), EXPECTED_MONTE_CARLO_POLICY, "Monte Carlo"
    )
    _require_exact(
        document.get("anti_leakage_policy"), EXPECTED_ANTI_LEAKAGE_POLICY, "anti-leakage"
    )
    _require_exact(
        document.get("runtime_manifest_schema"),
        EXPECTED_RUNTIME_MANIFEST_SCHEMA,
        "runtime manifest schema",
    )
    for field in ("execution_authorized", "production_authorization", "promotion_gate"):
        if document.get(field) is not False:
            raise ChLtComputeRuntimeError(f"{field} must be exactly false")
    contract_id = str(document.get("contract_id"))
    _require_sha256(contract_id, label="contract_id")
    if contract_id != compute_contract_id(document):
        raise ChLtComputeRuntimeError("contract_id does not bind canonical content")
    payload = document_bytes if document_bytes is not None else canonical_json_bytes(document)
    if document_bytes is not None:
        try:
            rebound = load_strict_json(document_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ChLtComputeRuntimeError("document bytes are not strict UTF-8 JSON") from exc
        if not isinstance(rebound, Mapping) or not _strict_equal(rebound, document):
            raise ChLtComputeRuntimeError("document bytes do not bind the assessed mapping")
    return ComputeRuntimeAssessment(
        contract_id=contract_id,
        document_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ChLtComputeRuntimeError(f"{label} must be lowercase SHA-256")
    return value


def _require_exact(actual: object, expected: object, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ChLtComputeRuntimeError(f"{label} is not the frozen compute policy")


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
