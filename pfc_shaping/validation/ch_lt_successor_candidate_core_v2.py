"""Historical outcome-blind closure for CH LT core v2, superseded by v3.

V2 remains verifiable as a policy scaffold but its global 36-origin block rule
must not be used for execution or admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import ch_lt_successor_candidate_core as v1

SCHEMA_VERSION = "ch_lt_successor_candidate_core.v2"
ASSESSMENT_SCHEMA_VERSION = "ch_lt_successor_candidate_core_assessment.v2"
STATUS = "LOCAL_HASH_CLOSED_OUTCOME_BLIND_CORE_SUPERSEDED_BY_V3_NOT_ADMITTED"
CORE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-SUCCESSOR-CANDIDATE-CORE-V2-20260730.json"
)
CORE_SHA256 = "a2ec1d758043b7ed4bf111e99ef4f87d814ad292b7b3c115c36de30d1da4011e"
CORE_ID = "7042b909a1ca0aeb63f26f76fccdea7256b526651685e4e1db713ca469cc3065"
TOMBSTONE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/T057-OUTCOME-BLIND-TOMBSTONE-20260730.json"
)
TOMBSTONE_SHA256 = "e7b7524375d431004c2dfe1aff9d7fea10a9bf7742c4044f53a9cda3cdb594ea"
TOMBSTONE_ID = "9d2afbe3593c5525bf67f16a0d4482fa24304d547f3649b84b74c7549daa3380"

REQUIRED_EVIDENCE = v1.REQUIRED_EVIDENCE
_NON_T057_BINDINGS = {
    v1.READINESS_V1_RELATIVE_PATH: (
        "734a7824ec747c829526774b346da441b45fff7cff5c9eb79ffc653ac78c7b8e"
    ),
    v1.PREREGISTRATION_SUPERSESSION_RELATIVE_PATH: (
        "76dba0b05948d336268b0a50c16df82ecd1c5138626c3fa87ee219148cb8e5e8"
    ),
    v1.ESTIMAND_RELATIVE_PATH: (
        "4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931"
    ),
    v1.COMPUTE_RELATIVE_PATH: (
        "b231345e96e7664ae02b7dbf3514af87d47ded7783034eaab1f8d449a28fe96f"
    ),
    v1.DEPENDENCE_SUPERSESSION_RELATIVE_PATH: (
        "9fd9cf706c768716a42967962527a49a9f92c7a16d1e60de0b3d910565312b72"
    ),
    v1.MARKET_TIME_RELATIVE_PATH: (
        "71711ae80b64556b8deab88e70581e1c7a0ef7c684d672b2cb550f2058c19c25"
    ),
    v1.LITERATURE_RELATIVE_PATH: (
        "aa67e9e9a54c30f8973533fd1c385917529e1299560b3d779c18d0b6e7589465"
    ),
}
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
    "policy_scaffold": "f3ee1524735cd5650d1825ca46c87ed7c08027255ec70476a4e018e4b3c88921",
    "outcome_blind_t057_tombstone": (
        "0b6a84adb15abecda66d4fe20cc3e45adc483c930ca907d8567676aa23f6363a"
    ),
    "supersession": "2f3ebede1b334ef2e502aa9e2860d499ada3efbfeeaddef6b9544ff52111e005",
    "outcome_blind_firewall": (
        "a7233ea7d1dc3702244ac0a50c9cd9e192fc0932d3cfd366b25f155a6a581d00"
    ),
    "selection_and_hypothesis_freeze": (
        "1fe49553e5c439ed0d9f71f1ac4cd063ad0f8969ec450bd62f241bc9443fe484"
    ),
    "overlap_aware_dependence_and_power": (
        "fd36e71c617d839d190edd5206c86e181d12c5e43bb15a41ed3ddc20c2688395"
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
        "ae670e61505ae7d6b56a4b390c4b8eb6303e860511224676777327939a34a1f3"
    ),
    "required_evidence_slots": (
        "069b09cafb8c11d18971e9a114af04733a6ecef6aedec0947b3e0c3ba4b3139f"
    ),
    "admission_boundary": (
        "f53585a42e90c1c0b9ff3794310654896fd5a0601d533152549558f478a5059e"
    ),
}
_TOMBSTONE_KEYS = {
    "schema_version",
    "tombstone_name",
    "superseded_registry_identity",
    "access_policy",
    "forbidden_input_classes",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "t057_consumed",
    "tombstone_id",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000


class ChLtSuccessorCandidateCoreV2Error(ValueError):
    """Raised when the v2 outcome-blind core chain is invalid."""


def compute_core_id(document: Mapping[str, object]) -> str:
    core = dict(document)
    core.pop("core_id", None)
    return _semantic_hash(core)


def verify_candidate_core_v2(
    *,
    repo_root: str | Path,
    core_path: str | Path,
    expected_core_sha256: str,
) -> dict[str, object]:
    """Verify v2 without reading the outcome-bearing T057 registry."""

    _require_sha256(expected_core_sha256, label="expected core")
    if expected_core_sha256 != CORE_SHA256:
        raise ChLtSuccessorCandidateCoreV2Error(
            "caller-held core SHA-256 does not match frozen canonical identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    if not root.is_dir():
        raise ChLtSuccessorCandidateCoreV2Error("repository root is unavailable")
    canonical_core = _repo_path(root, CORE_RELATIVE_PATH)
    if _path_key(_absolute(core_path)) != _path_key(canonical_core):
        raise ChLtSuccessorCandidateCoreV2Error("candidate core path is not canonical")

    expected = {
        CORE_RELATIVE_PATH: CORE_SHA256,
        v1.CORE_RELATIVE_PATH: v1.CORE_SHA256,
        TOMBSTONE_RELATIVE_PATH: TOMBSTONE_SHA256,
        **_NON_T057_BINDINGS,
    }
    payloads: dict[str, bytes] = {}
    paths: list[Path] = []
    for relative_path, expected_sha256 in expected.items():
        if relative_path == v1.T057_REGISTRY_RELATIVE_PATH:
            raise ChLtSuccessorCandidateCoreV2Error(
                "outcome-bearing T057 registry entered the v2 read set"
            )
        path = _repo_path(root, relative_path)
        payload = read_stable_single_link_file(
            path,
            label=f"CH LT successor core v2 artifact {relative_path}",
            max_bytes=_MAX_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ChLtSuccessorCandidateCoreV2Error(
                f"artifact SHA-256 mismatch: {relative_path}"
            )
        payloads[relative_path] = payload
        paths.append(path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if os.path.samefile(left, right):
                raise ChLtSuccessorCandidateCoreV2Error(
                    "candidate core v2 roles must be physically distinct files"
                )

    document = _strict_mapping(payloads[CORE_RELATIVE_PATH], label="candidate core v2")
    result = assess_candidate_core_v2(document, document_bytes=payloads[CORE_RELATIVE_PATH])
    v1_document = _strict_mapping(
        payloads[v1.CORE_RELATIVE_PATH], label="candidate core v1 policy scaffold"
    )
    v1.assess_candidate_core(v1_document, document_bytes=payloads[v1.CORE_RELATIVE_PATH])
    tombstone = _strict_mapping(payloads[TOMBSTONE_RELATIVE_PATH], label="T057 tombstone")
    _verify_tombstone(tombstone)
    _verify_non_t057_bindings(payloads)
    return result


def assess_candidate_core_v2(
    document: Mapping[str, object],
    *,
    document_bytes: bytes,
) -> dict[str, object]:
    parsed = _strict_mapping(document_bytes, label="candidate core v2 bytes")
    _equal(parsed, document, "candidate core v2 mapping/bytes")
    if hashlib.sha256(document_bytes).hexdigest() != CORE_SHA256:
        raise ChLtSuccessorCandidateCoreV2Error("candidate core v2 document SHA-256 mismatch")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="candidate core v2")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("core_name"), "ch_lt_successor_candidate_core_20260730_v2", "core name")
    _equal(document.get("core_id"), CORE_ID, "core id")
    _equal(compute_core_id(document), CORE_ID, "computed core id")
    for section, expected_sha256 in _SECTION_SHA256.items():
        if _semantic_hash(document.get(section)) != expected_sha256:
            raise ChLtSuccessorCandidateCoreV2Error(
                f"candidate core v2 semantic section mismatch: {section}"
            )
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
        "current_policy_core": False,
        "superseded_by_v3": True,
        "successor_exists": False,
        "readiness_complete": False,
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


def _verify_tombstone(document: Mapping[str, object]) -> None:
    _exact_keys(document, _TOMBSTONE_KEYS, label="T057 tombstone")
    _equal(document.get("schema_version"), "t057_outcome_blind_tombstone.v1", "tombstone schema")
    _equal(document.get("tombstone_id"), TOMBSTONE_ID, "tombstone id")
    core = dict(document)
    core.pop("tombstone_id", None)
    _equal(_semantic_hash(core), TOMBSTONE_ID, "computed tombstone id")
    _equal(
        document.get("superseded_registry_identity"),
        {
            "sha256": "0efec38b768b5e14add6cbc35c9b0cf9f10eb23f2cbf040813e68fe734ca4cf6",
            "role": "OUTCOME_BEARING_REGISTRY_IDENTITY_ONLY_SOURCE_BYTES_FORBIDDEN",
        },
        "superseded registry tombstone",
    )
    access = _mapping(document.get("access_policy"), label="tombstone access")
    _equal(access.get("allowed_successor_access"), "THIS_TOMBSTONE_ONLY", "tombstone access")
    for field in (
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "t057_consumed",
    ):
        _equal(document.get(field), False, f"tombstone {field}")


def _verify_non_t057_bindings(payloads: Mapping[str, bytes]) -> None:
    readiness = _strict_mapping(payloads[v1.READINESS_V1_RELATIVE_PATH], label="readiness v1")
    prereg = _strict_mapping(
        payloads[v1.PREREGISTRATION_SUPERSESSION_RELATIVE_PATH],
        label="preregistration supersession",
    )
    estimand = _strict_mapping(payloads[v1.ESTIMAND_RELATIVE_PATH], label="estimand")
    compute = _strict_mapping(payloads[v1.COMPUTE_RELATIVE_PATH], label="compute")
    dependence = _strict_mapping(
        payloads[v1.DEPENDENCE_SUPERSESSION_RELATIVE_PATH], label="dependence supersession"
    )
    market_time = _strict_mapping(payloads[v1.MARKET_TIME_RELATIVE_PATH], label="market time")
    _equal(
        readiness.get("contract_id"),
        "5e655cab1ca090cd067100dc8eb06161811b77991132e7c299883a7eb249e706",
        "readiness v1 id",
    )
    _equal(_mapping(prereg.get("successor"), label="successor").get("state"), "NOT_CREATED", "successor state")
    _equal(estimand.get("contract_id"), "da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344", "estimand id")
    _equal(compute.get("contract_id"), "d06710ba8ebee2364b81930fce51d17768f206521edfe62336ff2abdef60930a", "compute id")
    mc = _mapping(compute.get("monte_carlo_policy"), label="compute MC policy")
    _equal(
        mc.get("positive_margin_superiority_rule"),
        "TWO_SIDED_95_HALF_WIDTH_LE_MIN_0.10_MARGIN_AND_0.001_POSITIVE_FROZEN_REFERENCE_SCALE",
        "positive-margin MC rule",
    )
    _equal(
        mc.get("zero_margin_noninferiority_rule"),
        "ONE_SIDED_95_MC_CI_AND_HALF_WIDTH_LE_0.001_POSITIVE_FROZEN_REFERENCE_SCALE",
        "zero-margin MC rule",
    )
    _equal(dependence.get("scientific_admission"), False, "dependence admission")
    controls = _mapping(market_time.get("execution_controls"), label="market time controls")
    _equal(controls.get("contract_authorizes_holdout_consumption"), False, "holdout authority")


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
        raise ChLtSuccessorCandidateCoreV2Error(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtSuccessorCandidateCoreV2Error(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtSuccessorCandidateCoreV2Error(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtSuccessorCandidateCoreV2Error(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtSuccessorCandidateCoreV2Error("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    lexical = Path(path).expanduser()
    return lexical if lexical.is_absolute() else Path(os.path.abspath(lexical))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtSuccessorCandidateCoreV2Error(f"{label} SHA-256 is invalid")


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "CORE_ID",
    "CORE_RELATIVE_PATH",
    "CORE_SHA256",
    "REQUIRED_EVIDENCE",
    "STATUS",
    "TOMBSTONE_ID",
    "TOMBSTONE_RELATIVE_PATH",
    "TOMBSTONE_SHA256",
    "ChLtSuccessorCandidateCoreV2Error",
    "assess_candidate_core_v2",
    "compute_core_id",
    "verify_candidate_core_v2",
]
