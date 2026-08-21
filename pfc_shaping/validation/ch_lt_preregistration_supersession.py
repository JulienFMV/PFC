"""Fail-closed verifier for the superseded CH LT preregistration v1."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.ch_lt_successor_readiness import (
    CONTRACT_ID as READINESS_CONTRACT_ID,
)
from pfc_shaping.validation.ch_lt_successor_readiness import (
    CONTRACT_RELATIVE_PATH as READINESS_RELATIVE_PATH,
)
from pfc_shaping.validation.ch_lt_successor_readiness import (
    CONTRACT_SHA256 as READINESS_SHA256,
)
from pfc_shaping.validation.ch_lt_successor_readiness import (
    STATUS as READINESS_STATUS,
)
from pfc_shaping.validation.ch_lt_successor_readiness import (
    verify_successor_readiness,
)

SCHEMA_VERSION = "ch_lt_pit_preregistration_supersession_audit.v1"
REGISTRY_SCHEMA_VERSION = "ch_lt_pit_preregistration_supersession.v1"
STATUS = "ACTIVE_FAIL_CLOSED_V1_SUPERSEDED_NO_EXECUTABLE_SUCCESSOR"

REGISTRY_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json"
)
REGISTRY_SHA256 = "76dba0b05948d336268b0a50c16df82ecd1c5138626c3fa87ee219148cb8e5e8"
PREREGISTRATION_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-PROBABILISTIC-PREREGISTRATION-DRAFT-20260724.json"
)
PREREGISTRATION_SHA256 = "aba798530084b7031a0ac38b1c48b20cff575d6082edbcf37c9a04528900ba61"
PREREGISTRATION_PLAN_ID = "ae5557fd7e58a6ee4164e7f8a949cb379fc2d8ac23766e17a1873c4de420c5f6"
ESTIMAND_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json"
)
ESTIMAND_SHA256 = "4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931"
ESTIMAND_CONTRACT_ID = "da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344"
COMPUTE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/CH-LT-COMPUTE-RUNTIME-DRAFT-20260727.json"
)
COMPUTE_SHA256 = "b231345e96e7664ae02b7dbf3514af87d47ded7783034eaab1f8d449a28fe96f"
COMPUTE_CONTRACT_ID = "d06710ba8ebee2364b81930fce51d17768f206521edfe62336ff2abdef60930a"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000


class ChLtPreregistrationSupersessionError(ValueError):
    """Raised when the supersession chain is incomplete or has been rebound."""


def verify_preregistration_supersession(
    *,
    repo_root: str | Path,
    registry_path: str | Path,
    expected_registry_sha256: str,
) -> dict[str, object]:
    """Verify exact bytes and the semantic conflict without authorizing execution."""

    _require_sha256(expected_registry_sha256, label="expected registry")
    if expected_registry_sha256 != REGISTRY_SHA256:
        raise ChLtPreregistrationSupersessionError(
            "caller-held registry SHA-256 does not match frozen canonical identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    if not root.is_dir():
        raise ChLtPreregistrationSupersessionError("repository root is unavailable")
    canonical_registry = _repo_path(root, REGISTRY_RELATIVE_PATH)
    supplied_registry = _absolute(registry_path)
    if _path_key(supplied_registry) != _path_key(canonical_registry):
        raise ChLtPreregistrationSupersessionError("registry path is not canonical")

    expected = {
        REGISTRY_RELATIVE_PATH: REGISTRY_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_SHA256,
        ESTIMAND_RELATIVE_PATH: ESTIMAND_SHA256,
        COMPUTE_RELATIVE_PATH: COMPUTE_SHA256,
        READINESS_RELATIVE_PATH: READINESS_SHA256,
    }
    payloads: dict[str, bytes] = {}
    paths: list[Path] = []
    for relative_path, expected_sha256 in expected.items():
        path = _repo_path(root, relative_path)
        payload = read_stable_single_link_file(
            path,
            label=f"CH LT supersession artifact {relative_path}",
            max_bytes=_MAX_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ChLtPreregistrationSupersessionError(
                f"artifact SHA-256 mismatch: {relative_path}"
            )
        payloads[relative_path] = payload
        paths.append(path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if os.path.samefile(left, right):
                raise ChLtPreregistrationSupersessionError(
                    "supersession roles must be physically distinct files"
                )

    registry = _strict_mapping(payloads[REGISTRY_RELATIVE_PATH], label="registry")
    preregistration = _strict_mapping(
        payloads[PREREGISTRATION_RELATIVE_PATH], label="superseded preregistration"
    )
    estimand = _strict_mapping(payloads[ESTIMAND_RELATIVE_PATH], label="estimand draft")
    compute = _strict_mapping(payloads[COMPUTE_RELATIVE_PATH], label="compute draft")
    _verify_registry(registry)
    _verify_bound_documents(preregistration, estimand, compute)
    readiness_result = verify_successor_readiness(
        repo_root=root,
        contract_path=_repo_path(root, READINESS_RELATIVE_PATH),
        expected_contract_sha256=READINESS_SHA256,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "registry_path": REGISTRY_RELATIVE_PATH,
        "registry_sha256": REGISTRY_SHA256,
        "superseded_plan_id": PREREGISTRATION_PLAN_ID,
        "superseded_document_sha256": PREREGISTRATION_SHA256,
        "successor_exists": False,
        "readiness_contract_sha256": READINESS_SHA256,
        "readiness_contract_id": READINESS_CONTRACT_ID,
        "readiness_status": readiness_result["status"],
        "candidate_core_sha256": readiness_result["candidate_core_sha256"],
        "candidate_core_authored": readiness_result["candidate_core_authored"],
        "candidate_core_hash_closed_locally": readiness_result[
            "candidate_core_hash_closed_locally"
        ],
        "candidate_core_frozen": readiness_result["candidate_core_frozen"],
        "candidate_core_externally_frozen": readiness_result[
            "candidate_core_externally_frozen"
        ],
        "candidate_core_admitted": readiness_result["candidate_core_admitted"],
        "readiness_update_sha256": readiness_result["readiness_update_sha256"],
        "readiness_complete": False,
        "readiness_blockers": readiness_result["blockers"],
        "superseded_v1_usable": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }


def _verify_registry(registry: Mapping[str, object]) -> None:
    _exact_keys(
        registry,
        {
            "schema_version",
            "status",
            "superseded",
            "non_authoritative_clarifications",
            "policy_conflict",
            "successor",
            "forbidden_uses_of_superseded_v1",
            "scientific_admission",
            "execution_authorized",
            "production_authorization",
            "promotion_gate",
            "future_holdout_consumed",
            "t057_consumed",
        },
        label="registry",
    )
    _equal(registry.get("schema_version"), REGISTRY_SCHEMA_VERSION, "registry schema")
    _equal(registry.get("status"), STATUS, "registry status")
    for field in (
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        _equal(registry.get(field), False, f"registry {field}")

    superseded = _mapping(registry.get("superseded"), label="superseded")
    expected_superseded = {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "sha256": PREREGISTRATION_SHA256,
        "schema_version": "ch_lt_pit_probabilistic_preregistration.v1",
        "plan_id": PREREGISTRATION_PLAN_ID,
        "protocol_name": "ch_lt_pit_outer24_future4_20260724_v1",
        "reason": (
            "The v1 pathwise fixed-monthly-level scenario policy conflicts with the "
            "corrected full delivered-curve estimand and would suppress scenario-specific "
            "monthly level risk."
        ),
    }
    _equal(dict(superseded), expected_superseded, "superseded binding")

    clarifications = registry.get("non_authoritative_clarifications")
    expected_clarifications = [
        {
            "role": "corrected_estimand_draft",
            "path": ESTIMAND_RELATIVE_PATH,
            "sha256": ESTIMAND_SHA256,
            "schema_version": "ch_lt_estimand_and_economic_design.v1",
            "contract_id": ESTIMAND_CONTRACT_ID,
        },
        {
            "role": "compute_runtime_draft",
            "path": COMPUTE_RELATIVE_PATH,
            "sha256": COMPUTE_SHA256,
            "schema_version": "ch_lt_compute_runtime.v1",
            "contract_id": COMPUTE_CONTRACT_ID,
        },
        {
            "role": "successor_readiness_contract",
            "path": READINESS_RELATIVE_PATH,
            "sha256": READINESS_SHA256,
            "schema_version": "ch_lt_pit_successor_readiness_contract.v1",
            "contract_id": READINESS_CONTRACT_ID,
        },
    ]
    _equal(clarifications, expected_clarifications, "clarification bindings")
    _equal(
        registry.get("policy_conflict"),
        {
            "superseded_policy": "EVERY_SCENARIO_PROJECTED_PATHWISE_TO_THE_MONTHLY_SOLVER_FORWARD",
            "corrected_ensemble_policy": (
                "ENERGY_WEIGHTED_ENSEMBLE_EXPECTATION_EQUALS_MONTHLY_SOLVER_FORWARD"
            ),
            "scenario_specific_monthly_level_risk": (
                "ALLOWED_ONLY_WITH_ZERO_MEAN_ENSEMBLE_AND_FROZEN_COVARIANCE"
            ),
            "pathwise_fixed_monthly_level": "SEPARATE_SHAPE_ONLY_DIAGNOSTIC_PRODUCT",
            "monthly_solver_level_authority_preserved": True,
        },
        "policy conflict",
    )
    _equal(
        registry.get("successor"),
        {
            "state": "NOT_CREATED",
            "schema_version": None,
            "plan_id": None,
            "execution_capability": None,
            "required_before_creation": [
                "GOVERNED_POINT_IN_TIME_CH_AND_EEX_EVIDENCE",
                "EXACT_ORIGIN_TARGET_MASK_AND_INNER_FOLD_INVENTORIES",
                "FROZEN_DEPENDENCE_POWER_AND_MULTIPLICITY_DESIGN",
                "FROZEN_PROBABILISTIC_SCENARIO_AND_MONTE_CARLO_DESIGN",
                "FMV_RISK_APPROVED_ECONOMIC_MATERIALITY",
                "INDEPENDENT_SECURITY_ITOPS_AND_QUANT_REVIEWS",
                "EXTERNAL_ADMISSION_ENVELOPE_AND_ONE_SHOT_LEDGER",
            ],
        },
        "successor",
    )
    _equal(
        registry.get("forbidden_uses_of_superseded_v1"),
        [
            "EVALUATION_EXECUTION_ADMISSION",
            "MODEL_OR_HYPERPARAMETER_SELECTION",
            "FUTURE_HOLDOUT_CONSUMPTION",
            "SCIENTIFIC_CONFIRMATION",
            "PRODUCTION_PROMOTION",
        ],
        "forbidden uses",
    )
    _equal(
        READINESS_STATUS,
        (
            "CANDIDATE_CORE_V3_CONTRAST_AWARE_LOCAL_HASH_CLOSED_NOT_EXTERNALLY_FROZEN_"
            "NOT_ADMITTED_SUCCESSOR_NOT_CREATED"
        ),
        "readiness status",
    )


def _verify_bound_documents(
    preregistration: Mapping[str, object],
    estimand: Mapping[str, object],
    compute: Mapping[str, object],
) -> None:
    _equal(
        preregistration.get("schema_version"),
        "ch_lt_pit_probabilistic_preregistration.v1",
        "preregistration schema",
    )
    _equal(preregistration.get("plan_id"), PREREGISTRATION_PLAN_ID, "plan id")
    _equal(preregistration.get("production_authorization"), False, "prereg production")
    _equal(preregistration.get("promotion_gate"), False, "prereg promotion")
    old_policy = _mapping(
        preregistration.get("probabilistic_policy"), label="old probabilistic policy"
    )
    _equal(old_policy.get("pathwise_market_product_identity_required"), True, "old pathwise")
    _equal(
        old_policy.get("quantile_market_identity_policy"),
        "MARGINAL_QUANTILES_NOT_SEPARATELY_REPRICED_SCENARIOS_PROJECTED_PATHWISE",
        "old scenario projection",
    )

    _equal(estimand.get("contract_id"), ESTIMAND_CONTRACT_ID, "estimand contract id")
    new_policy = _mapping(estimand.get("probabilistic_policy"), label="new estimand policy")
    _equal(
        new_policy.get("ensemble_monthly_forward_consistency"),
        "ENERGY_WEIGHTED_SCENARIO_EXPECTATION_EQUALS_MONTHLY_SOLVER_FORWARD",
        "ensemble consistency",
    )
    _equal(
        new_policy.get("scenario_specific_monthly_level_uncertainty"),
        "ALLOWED_ONLY_WITH_ENSEMBLE_ZERO_MEAN_AND_FROZEN_COVARIANCE_DESIGN",
        "scenario level uncertainty",
    )
    _equal(
        new_policy.get("shape_only_fixed_level_scenarios"),
        "SEPARATE_DIAGNOSTIC_PRODUCT_ONLY",
        "shape-only scenario role",
    )

    _equal(compute.get("contract_id"), COMPUTE_CONTRACT_ID, "compute contract id")
    boundary = _mapping(compute.get("authority_boundary"), label="compute authority boundary")
    _equal(
        boundary.get("scenario_level_policy"),
        "ENERGY_WEIGHTED_ENSEMBLE_EXPECTATION_EQUALS_MONTHLY_SOLVER_FORWARD",
        "compute ensemble consistency",
    )
    _equal(
        boundary.get("scenario_specific_monthly_level_risk"),
        "ALLOWED_ONLY_WITH_ZERO_MEAN_ENSEMBLE_AND_FROZEN_COVARIANCE",
        "compute scenario level risk",
    )
    _equal(
        boundary.get("pathwise_fixed_monthly_level"),
        "SEPARATE_SHAPE_ONLY_DIAGNOSTIC_PRODUCT",
        "compute shape-only role",
    )
    for document, label in ((estimand, "estimand"), (compute, "compute")):
        controls = document.get("execution_controls")
        if label == "compute":
            _equal(document.get("execution_authorized"), False, "compute execution")
            _equal(document.get("production_authorization"), False, "compute production")
            _equal(document.get("promotion_gate"), False, "compute promotion")
        else:
            mapping = _mapping(controls, label="estimand execution controls")
            _equal(mapping.get("schema_v1_authorizes_execution"), False, "estimand execution")
            _equal(mapping.get("production_authorization"), False, "estimand production")
            _equal(mapping.get("promotion_gate"), False, "estimand promotion")


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtPreregistrationSupersessionError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtPreregistrationSupersessionError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtPreregistrationSupersessionError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtPreregistrationSupersessionError(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtPreregistrationSupersessionError("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    lexical = Path(path).expanduser()
    return lexical if lexical.is_absolute() else Path(os.path.abspath(lexical))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtPreregistrationSupersessionError(f"{label} SHA-256 is invalid")


__all__ = [
    "ChLtPreregistrationSupersessionError",
    "REGISTRY_RELATIVE_PATH",
    "REGISTRY_SHA256",
    "STATUS",
    "verify_preregistration_supersession",
]
