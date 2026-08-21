"""Fail-closed, outcome-blind CH LT dependence and power design draft."""

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
from pfc_shaping.validation import ch_lt_successor_candidate_core_v3 as core_v3

SCHEMA_VERSION = "ch_lt_dependence_power_design.v1"
ASSESSMENT_SCHEMA_VERSION = "ch_lt_dependence_power_design_assessment.v1"
STATUS = "LOCAL_OUTCOME_BLIND_POWER_DESIGN_DRAFT_INCOMPLETE_NOT_EXTERNALLY_FROZEN_NO_GO"
DESIGN_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-DEPENDENCE-POWER-DESIGN-DRAFT-V1-20260730.json"
)
DESIGN_SHA256 = "005b8655b817db10e7f3c227b1c5912d545305b68e262ab82d9aa0b5817a6a91"
DESIGN_ID = "2147af9d2fd277a43df6f88f10dfa6c4061a7184c7b84ca0b09af5e66fa120c9"
REQUIRED_INPUTS = (
    "EXACT_FROZEN_PRIMARY_HYPOTHESIS_FAMILY",
    "DIRECT_CH_ADMITTED_DEVELOPMENT_PAIRED_LOSS_PANEL",
    "FMV_RISK_APPROVED_CONTRAST_MARGINS",
    "EXACT_ORIGIN_TARGET_MASK_AND_DELIVERY_MONTH_INVENTORY",
    "QUALIFIED_SIMULATION_MC_AND_CPU_GPU_PARITY_MANIFEST",
    "EXTERNAL_FREEZE_SIGNATURE_TRUSTED_TIME_CAS_WORM_AND_FRESH_HEAD",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "design_name",
    "lifecycle",
    "candidate_core_binding",
    "outcome_blind_firewall",
    "hypothesis_family",
    "contrast_geometry",
    "dependence_estimation",
    "power_calibration",
    "future_evaluation",
    "required_inputs",
    "design_complete",
    "evidence_slot_satisfied",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
    "design_id",
}
_SECTION_SHA256 = {
    "lifecycle": "aaa79d4e199b0b06a93b2d963b017871c0e23c9f0ae0090d3e3564830761a153",
    "candidate_core_binding": (
        "79b2be11d4e062d7b6cf75b35a2f95f92c4fee6ffa2034e7cc8b8c2ddbb40871"
    ),
    "outcome_blind_firewall": (
        "fcf80e34edf600810b7514764a8ece3f7ba8e9ec31cdea6f5824a581b2946fb3"
    ),
    "hypothesis_family": (
        "95f18d4a98dffcd9b8da4cdf22e5209e46c80a205ee31f2e986c7ad6f9d4c8e6"
    ),
    "contrast_geometry": (
        "2a95285807f818bb5cd985f75fe63ba9d47ec3c1f7548828eaf21200751855b3"
    ),
    "dependence_estimation": (
        "054ffd0eab5204eddada00dc8871612d2e35ab01c3fe2a078e7389cf6c673b8d"
    ),
    "power_calibration": (
        "348e9be84e01d84fc89e5011003bce0a0d6a4eb6aff5a12eea6e32b610659a96"
    ),
    "future_evaluation": (
        "f8421b0224c4a9ed8c96b7db8f3a3af6b6458a13858bae5c3f5fd3609f0e1311"
    ),
    "required_inputs": (
        "9183cbbc7bbf7394cde7ff5f346bbd312b3c765ffb61b8de21b3d772dac7de22"
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000


class ChLtDependencePowerDesignError(ValueError):
    """Raised when the power design bytes or policy are invalid."""


def compute_design_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("design_id", None)
    return _semantic_hash(unsigned)


def verify_dependence_power_design(
    *, repo_root: str | Path, design_path: str | Path, expected_design_sha256: str
) -> dict[str, object]:
    """Verify the exact draft and v3 chain without opening any outcome bytes."""

    _require_sha256(expected_design_sha256, label="expected design")
    if expected_design_sha256 != DESIGN_SHA256:
        raise ChLtDependencePowerDesignError(
            "caller-held design SHA-256 does not match canonical identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    if not root.is_dir():
        raise ChLtDependencePowerDesignError("repository root is unavailable")
    canonical = _repo_path(root, DESIGN_RELATIVE_PATH)
    if _path_key(_absolute(design_path)) != _path_key(canonical):
        raise ChLtDependencePowerDesignError("power design path is not canonical")
    payload = read_stable_single_link_file(
        canonical, label="CH LT dependence power design", max_bytes=_MAX_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != DESIGN_SHA256:
        raise ChLtDependencePowerDesignError("power design SHA-256 mismatch")
    document = _strict_mapping(payload, label="power design")
    result = assess_dependence_power_design(document, document_bytes=payload)
    core = core_v3.verify_candidate_core_v3(
        repo_root=root,
        core_path=_repo_path(root, core_v3.CORE_RELATIVE_PATH),
        expected_core_sha256=core_v3.CORE_SHA256,
    )
    _equal(core.get("core_id"), core_v3.CORE_ID, "verified core id")
    _equal(core.get("candidate_core_admitted"), False, "verified core admission")
    _equal(
        result.get("contrast_overlap_geometry"),
        core.get("contrast_overlap_geometry"),
        "design/core contrast geometry",
    )
    return result


def assess_dependence_power_design(
    document: Mapping[str, object], *, document_bytes: bytes
) -> dict[str, object]:
    parsed = _strict_mapping(document_bytes, label="power design bytes")
    _equal(parsed, document, "power design mapping/bytes")
    if hashlib.sha256(document_bytes).hexdigest() != DESIGN_SHA256:
        raise ChLtDependencePowerDesignError("power design document SHA-256 mismatch")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="power design")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("design_name"), "ch_lt_dependence_power_design_20260730_v1", "design name")
    _equal(document.get("design_id"), DESIGN_ID, "design id")
    _equal(compute_design_id(document), DESIGN_ID, "computed design id")
    for section, expected_sha256 in _SECTION_SHA256.items():
        if _semantic_hash(document.get(section)) != expected_sha256:
            raise ChLtDependencePowerDesignError(
                f"power design semantic section mismatch: {section}"
            )
    binding = _mapping(document.get("candidate_core_binding"), label="core binding")
    _equal(binding.get("path"), core_v3.CORE_RELATIVE_PATH, "core path")
    _equal(binding.get("sha256"), core_v3.CORE_SHA256, "core SHA-256")
    _equal(binding.get("core_id"), core_v3.CORE_ID, "core id")
    geometry = _verify_geometry(document.get("contrast_geometry"))
    _verify_power_calibration(document.get("power_calibration"))
    _equal(
        document.get("required_inputs"),
        [
            {"requirement": item, "status": "MISSING", "path": None, "sha256": None}
            for item in REQUIRED_INPUTS
        ],
        "required inputs",
    )
    for field in (
        "design_complete",
        "evidence_slot_satisfied",
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        _equal(document.get(field), False, field)
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": STATUS,
        "design_path": DESIGN_RELATIVE_PATH,
        "design_sha256": DESIGN_SHA256,
        "design_id": DESIGN_ID,
        "candidate_core_sha256": core_v3.CORE_SHA256,
        "candidate_core_id": core_v3.CORE_ID,
        "power_design_authored": True,
        "power_design_hash_closed_locally": True,
        "power_design_externally_frozen": False,
        "power_design_complete": False,
        "evidence_slot_satisfied": False,
        "contrast_overlap_geometry": geometry,
        "missing_inputs": list(REQUIRED_INPUTS),
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def _verify_geometry(value: object) -> list[dict[str, object]]:
    geometry = _mapping(value, label="contrast geometry")
    _equal(geometry.get("origin_cadence"), "MONTHLY", "origin cadence")
    _equal(
        geometry.get("one_global_36_origin_rule_for_all_contrasts"),
        "FORBIDDEN",
        "global 36-origin rule",
    )
    rows = _sequence(geometry.get("rows"), label="contrast geometry rows")
    expected = (
        ("FIXED_LEAD_MONTH", None, None, 1, 0, 1, True),
        ("M01_M06", 1, 6, 6, 5, 6, True),
        ("M07_M12", 7, 12, 6, 5, 6, True),
        ("M13_M24", 13, 24, 12, 11, 12, True),
        ("M25_M36", 25, 36, 12, 11, 12, True),
        ("M01_M36_FULL_HORIZON_AGGREGATE", 1, 36, 36, 35, 36, False),
    )
    if len(rows) != len(expected):
        raise ChLtDependencePowerDesignError("contrast geometry count is invalid")
    result: list[dict[str, object]] = []
    for raw, target in zip(rows, expected, strict=True):
        row = _mapping(raw, label="contrast geometry row")
        observed = (
            row.get("contrast_id"),
            row.get("minimum_lead_month"),
            row.get("maximum_lead_month"),
            row.get("support_width_months"),
            row.get("maximum_mechanical_overlap_lag_origins"),
            row.get("mechanical_expected_mean_block_lower_bound_origins"),
            row.get("primary_confirmatory_eligible"),
        )
        _equal(observed, target, f"contrast geometry {target[0]}")
        if target[1] is not None and target[2] is not None:
            derived = core_v3.derive_monthly_overlap_geometry(
                minimum_lead_month=target[1], maximum_lead_month=target[2]
            )
            _equal(
                tuple(derived.values()),
                target[3:6],
                f"derived contrast geometry {target[0]}",
            )
        result.append(
            {
                "contrast_id": target[0],
                "support_width_months": target[3],
                "maximum_mechanical_overlap_lag_origins": target[4],
                "mechanical_expected_mean_block_lower_bound_origins": target[5],
                "primary_confirmatory_eligible": target[6],
            }
        )
    return result


def _verify_power_calibration(value: object) -> None:
    calibration = _mapping(value, label="power calibration")
    _equal(
        calibration.get("candidate_origin_count_grid"),
        "COMMON_ASCENDING_UNIQUE_ORIGIN_COUNTS_SHARED_BY_ALL_CONTRASTS",
        "common origin count grid",
    )
    _equal(
        calibration.get("contrast_specific_block_lengths_applied_on_common_grid"),
        True,
        "contrast-specific blocks on common grid",
    )
    _equal(
        calibration.get("alternative_effects_distinct_from_null_margins"),
        True,
        "distinct alternative effects",
    )
    _equal(
        calibration.get("conjunctive_power_acceptance"),
        "COMPLETE_FMV_GATEKEEPING_DECISION_ONE_SIDED_95_POWER_LOWER_CONFIDENCE_"
        "BOUND_GE_0.80",
        "conjunctive gatekeeping power",
    )
    _equal(
        calibration.get("required_n_selection"),
        "SMALLEST_COMMON_N_PASSING_STRONG_FWER_CONJUNCTIVE_POWER_MARGINAL_FLOORS_"
        "EFFECTIVE_CLUSTERS_AND_COVERAGE",
        "common required N selection",
    )


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
        raise ChLtDependencePowerDesignError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtDependencePowerDesignError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChLtDependencePowerDesignError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtDependencePowerDesignError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtDependencePowerDesignError(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtDependencePowerDesignError("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    return selected if selected.is_absolute() else Path(os.path.abspath(selected))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtDependencePowerDesignError(f"{label} SHA-256 is invalid")


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "DESIGN_ID",
    "DESIGN_RELATIVE_PATH",
    "DESIGN_SHA256",
    "REQUIRED_INPUTS",
    "STATUS",
    "ChLtDependencePowerDesignError",
    "assess_dependence_power_design",
    "compute_design_id",
    "verify_dependence_power_design",
]
