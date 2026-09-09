"""Outcome-blind, contrast-aware closure for the CH LT successor core v3.

V3 supersedes the v2 global 36-origin block rule.  Mechanical delivery
overlap is a property of each predeclared horizon contrast, not a universal
property of every loss series.  This verifier never opens the outcome-bearing
T057 registry or any referenced result.
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
from pfc_shaping.validation import ch_lt_successor_candidate_core_v2 as v2

SCHEMA_VERSION = "ch_lt_successor_candidate_core.v3"
ASSESSMENT_SCHEMA_VERSION = "ch_lt_successor_candidate_core_assessment.v3"
STATUS = (
    "LOCAL_HASH_CLOSED_OUTCOME_BLIND_CONTRAST_AWARE_CORE_"
    "NOT_EXTERNALLY_FROZEN_NOT_ADMITTED"
)
CORE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-SUCCESSOR-CANDIDATE-CORE-V3-20260730.json"
)
CORE_SHA256 = "e2c2a6f9d3ca677991f3f76f03dec1328492e2f60a4015b7796a2ff6aa6f905a"
CORE_ID = "d86dab183dd95a2ca3cda0862f2c9f50cec01716a37c6cdc1c3a15763254dcc3"

REQUIRED_EVIDENCE = v2.REQUIRED_EVIDENCE
_TOP_LEVEL_KEYS = {
    "schema_version",
    "core_name",
    "lifecycle",
    "policy_scaffold",
    "outcome_blind_t057_tombstone",
    "supersession",
    "outcome_blind_firewall",
    "selection_and_hypothesis_freeze",
    "overlap_aware_dependence_and_power",
    "native_regime_scoring_graphs",
    "monte_carlo_design_correction",
    "conditioning_and_stability_design",
    "future_evaluation_design",
    "required_evidence_slots",
    "admission_boundary",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
    "core_id",
}
_SECTION_SHA256 = {
    "lifecycle": "610c45c0fe0f952bce13eefbab59f0fa8fc1b204cc9d675295e55a31f80ec97f",
    "policy_scaffold": "f8863820024035efea31df92a844b2a745801273857f09db738c64de0039a164",
    "outcome_blind_t057_tombstone": (
        "0b6a84adb15abecda66d4fe20cc3e45adc483c930ca907d8567676aa23f6363a"
    ),
    "supersession": "7df44a8f0f369cf652567788119675b8b66e6e648e614829a70eddbe34aefaa7",
    "outcome_blind_firewall": (
        "a7233ea7d1dc3702244ac0a50c9cd9e192fc0932d3cfd366b25f155a6a581d00"
    ),
    "selection_and_hypothesis_freeze": (
        "d089a0fb8f41e48246f5e5bec97cfcea388edcdcb37cdcd638a436fd70d7af8e"
    ),
    "overlap_aware_dependence_and_power": (
        "2bd4a158f1f6f1ce3dcad9e59a59dc8695e5f1dac694ea92ae4910766dcdb425"
    ),
    "native_regime_scoring_graphs": (
        "6b419dafbcbf1425aabb9e176c2747a210f820afe40e073e56f38a6c0b2e08be"
    ),
    "monte_carlo_design_correction": (
        "19e34b0a25e3261a1151ac77294dcdb22964dcd45bce7fa0ccdcd3875ab576ad"
    ),
    "conditioning_and_stability_design": (
        "e1f1497e17dcbbcc673b3ece66a34b1d5e1424bd943ef4bd904049497f9b8f52"
    ),
    "future_evaluation_design": (
        "b74b9f6d9efe5e73bf26c253669af6283c816c83e1c62f6ae4825ce871b7a846"
    ),
    "required_evidence_slots": (
        "069b09cafb8c11d18971e9a114af04733a6ecef6aedec0947b3e0c3ba4b3139f"
    ),
    "admission_boundary": (
        "f53585a42e90c1c0b9ff3794310654896fd5a0601d533152549558f478a5059e"
    ),
}
_EXPECTED_GEOMETRY = (
    ("FIXED_LEAD_MONTH", None, None, 1, 0, 1, True),
    ("M01_M06", 1, 6, 6, 5, 6, True),
    ("M07_M12", 7, 12, 6, 5, 6, True),
    ("M13_M24", 13, 24, 12, 11, 12, True),
    ("M25_M36", 25, 36, 12, 11, 12, True),
    ("M01_M36_FULL_HORIZON_AGGREGATE", 1, 36, 36, 35, 36, False),
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000


class ChLtSuccessorCandidateCoreV3Error(ValueError):
    """Raised when the v3 outcome-blind core chain is invalid."""


def compute_core_id(document: Mapping[str, object]) -> str:
    core = dict(document)
    core.pop("core_id", None)
    return _semantic_hash(core)


def derive_monthly_overlap_geometry(
    *, minimum_lead_month: int, maximum_lead_month: int
) -> dict[str, int]:
    """Return exact mechanical overlap for a contiguous monthly bucket."""

    for value, label in (
        (minimum_lead_month, "minimum lead month"),
        (maximum_lead_month, "maximum lead month"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ChLtSuccessorCandidateCoreV3Error(f"{label} must be a positive integer")
    if maximum_lead_month < minimum_lead_month:
        raise ChLtSuccessorCandidateCoreV3Error(
            "maximum lead month must not precede minimum lead month"
        )
    width = maximum_lead_month - minimum_lead_month + 1
    return {
        "support_width_months": width,
        "maximum_mechanical_overlap_lag_origins": width - 1,
        "mechanical_expected_mean_block_lower_bound_origins": width,
    }


def verify_candidate_core_v3(
    *,
    repo_root: str | Path,
    core_path: str | Path,
    expected_core_sha256: str,
) -> dict[str, object]:
    """Verify v3 and its outcome-blind v2 scaffold without opening T057 truth."""

    _require_sha256(expected_core_sha256, label="expected core")
    if expected_core_sha256 != CORE_SHA256:
        raise ChLtSuccessorCandidateCoreV3Error(
            "caller-held core SHA-256 does not match frozen canonical identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    if not root.is_dir():
        raise ChLtSuccessorCandidateCoreV3Error("repository root is unavailable")
    canonical_core = _repo_path(root, CORE_RELATIVE_PATH)
    if _path_key(_absolute(core_path)) != _path_key(canonical_core):
        raise ChLtSuccessorCandidateCoreV3Error("candidate core path is not canonical")
    payload = read_stable_single_link_file(
        canonical_core,
        label="CH LT successor core v3",
        max_bytes=_MAX_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != CORE_SHA256:
        raise ChLtSuccessorCandidateCoreV3Error("candidate core v3 SHA-256 mismatch")
    document = _strict_mapping(payload, label="candidate core v3")
    result = assess_candidate_core_v3(document, document_bytes=payload)

    v2_result = v2.verify_candidate_core_v2(
        repo_root=root,
        core_path=_repo_path(root, v2.CORE_RELATIVE_PATH),
        expected_core_sha256=v2.CORE_SHA256,
    )
    for field in (
        "candidate_core_admitted",
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        if v2_result.get(field) is not False:
            raise ChLtSuccessorCandidateCoreV3Error(
                f"v2 policy scaffold unexpectedly authorizes {field}"
            )
    return result


def assess_candidate_core_v3(
    document: Mapping[str, object],
    *,
    document_bytes: bytes,
) -> dict[str, object]:
    parsed = _strict_mapping(document_bytes, label="candidate core v3 bytes")
    _equal(parsed, document, "candidate core v3 mapping/bytes")
    if hashlib.sha256(document_bytes).hexdigest() != CORE_SHA256:
        raise ChLtSuccessorCandidateCoreV3Error("candidate core v3 document SHA-256 mismatch")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="candidate core v3")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("core_name"), "ch_lt_successor_candidate_core_20260730_v3", "core name")
    _equal(document.get("core_id"), CORE_ID, "core id")
    _equal(compute_core_id(document), CORE_ID, "computed core id")
    for section, expected_sha256 in _SECTION_SHA256.items():
        if _semantic_hash(document.get(section)) != expected_sha256:
            raise ChLtSuccessorCandidateCoreV3Error(
                f"candidate core v3 semantic section mismatch: {section}"
            )
    _verify_overlap_geometry(document.get("overlap_aware_dependence_and_power"))
    _equal(
        document.get("required_evidence_slots"),
        [
            {"requirement": item, "status": "MISSING", "path": None, "sha256": None}
            for item in REQUIRED_EVIDENCE
        ],
        "required evidence slots",
    )
    for field in (
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
        "core_path": CORE_RELATIVE_PATH,
        "core_sha256": CORE_SHA256,
        "core_id": CORE_ID,
        "candidate_core_authored": True,
        "candidate_core_hash_closed_locally": True,
        "candidate_core_frozen": False,
        "candidate_core_externally_frozen": False,
        "candidate_core_admitted": False,
        "successor_exists": False,
        "readiness_complete": False,
        "superseded_core_versions": ["v1", "v2"],
        "contrast_overlap_geometry": [
            {
                "contrast_id": item[0],
                "support_width_months": item[3],
                "maximum_mechanical_overlap_lag_origins": item[4],
                "mechanical_expected_mean_block_lower_bound_origins": item[5],
                "primary_confirmatory_eligible": item[6],
            }
            for item in _EXPECTED_GEOMETRY
        ],
        "missing_evidence": list(REQUIRED_EVIDENCE),
        "legacy_t057_outcome_metadata_exposed": True,
        "legacy_t057_confirmation_reuse": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def _verify_overlap_geometry(value: object) -> None:
    policy = _mapping(value, label="dependence and power policy")
    _equal(policy.get("origin_cadence"), "MONTHLY", "origin cadence")
    _equal(
        policy.get("common_global_mechanical_block_lower_bound_for_all_contrasts"),
        "FORBIDDEN",
        "global block lower bound",
    )
    _equal(
        policy.get("stationary_bootstrap_block_parameter"),
        "EXPECTED_MEAN_GEOMETRIC_BLOCK_LENGTH_ORIGINS_NOT_FIXED_MINIMUM_BLOCK_SIZE",
        "stationary bootstrap parameter",
    )
    rows = _sequence(policy.get("mechanical_overlap_geometry"), label="overlap geometry")
    if len(rows) != len(_EXPECTED_GEOMETRY):
        raise ChLtSuccessorCandidateCoreV3Error("overlap geometry count is invalid")
    for raw, expected in zip(rows, _EXPECTED_GEOMETRY, strict=True):
        row = _mapping(raw, label="overlap geometry row")
        observed = (
            row.get("contrast_id"),
            row.get("minimum_lead_month"),
            row.get("maximum_lead_month"),
            row.get("support_width_months"),
            row.get("maximum_mechanical_overlap_lag_origins"),
            row.get("mechanical_expected_mean_block_lower_bound_origins"),
            row.get("primary_confirmatory_eligible"),
        )
        _equal(observed, expected, f"overlap geometry {expected[0]}")
        if expected[1] is not None and expected[2] is not None:
            derived = derive_monthly_overlap_geometry(
                minimum_lead_month=expected[1], maximum_lead_month=expected[2]
            )
            _equal(
                derived,
                {
                    "support_width_months": expected[3],
                    "maximum_mechanical_overlap_lag_origins": expected[4],
                    "mechanical_expected_mean_block_lower_bound_origins": expected[5],
                },
                f"derived overlap geometry {expected[0]}",
            )
    simulation = _mapping(policy.get("power_simulation"), label="power simulation")
    _equal(
        simulation.get("candidate_origin_counts"),
        "COMMON_ASCENDING_UNIQUE_ORIGIN_COUNT_GRID_SHARED_BY_ALL_CONTRASTS",
        "common origin count grid",
    )
    _equal(
        simulation.get("direction_oriented_null_boundary_rule"),
        "EACH_CONTRAST_SIMULATED_AT_ITS_EXACT_LEAST_FAVOURABLE_NULL_BOUNDARY_"
        "FROM_DIRECTION_AND_FROZEN_HYPOTHESIS_MANIFEST",
        "direction-oriented null boundary rule",
    )
    _equal(
        simulation.get("alternative_power_calibration"),
        "COMPLETE_FMV_GATEKEEPING_DECISION_ONE_SIDED_95_POWER_LOWER_CONFIDENCE_"
        "BOUND_GE_0.80",
        "conjunctive power calibration",
    )
    _equal(
        simulation.get("required_n_selection"),
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
        raise ChLtSuccessorCandidateCoreV3Error(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtSuccessorCandidateCoreV3Error(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChLtSuccessorCandidateCoreV3Error(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtSuccessorCandidateCoreV3Error(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtSuccessorCandidateCoreV3Error(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtSuccessorCandidateCoreV3Error("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    lexical = Path(path).expanduser()
    return lexical if lexical.is_absolute() else Path(os.path.abspath(lexical))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtSuccessorCandidateCoreV3Error(f"{label} SHA-256 is invalid")


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "CORE_ID",
    "CORE_RELATIVE_PATH",
    "CORE_SHA256",
    "REQUIRED_EVIDENCE",
    "STATUS",
    "ChLtSuccessorCandidateCoreV3Error",
    "assess_candidate_core_v3",
    "compute_core_id",
    "derive_monthly_overlap_geometry",
    "verify_candidate_core_v3",
]
