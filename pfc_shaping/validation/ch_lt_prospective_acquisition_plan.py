"""Fail-closed contract for the next fresh CH LT data acquisition slice."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_prospective_data_acquisition_plan.v1"
AUDIT_SCHEMA_VERSION = "ch_lt_prospective_data_acquisition_plan_audit.v1"
LIFECYCLE_STATE = "DRAFT_BLOCKED_EXTERNAL_AUTHORITIES_AND_LICENSED_FEEDS_ABSENT"
STATUS = "STRUCTURE_VALID_BLOCKED_NOT_EXECUTABLE"
PLAN_ID = "a0faf97bc10b4d51e23b21c200a73cdd95f92ffa9f8ea7976bc4a10859a8688f"
DOCUMENT_SHA256 = "99361b488e5a031f66c64866c7d69f21a7d8d031e30cf37b097a452bb5627c5a"
PLAN_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PROSPECTIVE-DATA-ACQUISITION-PLAN-20260727.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000
_TOP_LEVEL_KEYS = {
    "schema_version",
    "plan_id",
    "lifecycle_state",
    "as_of_utc",
    "scope",
    "market",
    "timezone",
    "scientific_boundary",
    "acquisition_tracks",
    "admission_controls",
    "primary_source_observations",
    "next_actions",
    "execution_authorized",
    "scientific_admission",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
}


class ChLtProspectiveAcquisitionPlanError(ValueError):
    """Raised when the acquisition plan can be weakened or rebound."""


@dataclass(frozen=True)
class ProspectiveAcquisitionAssessment:
    plan_id: str
    document_sha256: str
    status: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "document_sha256": self.document_sha256,
            "status": self.status,
            "execution_authorized": False,
            "scientific_admission": False,
            "production_authorization": False,
            "promotion_gate": False,
            "future_holdout_consumed": False,
            "t057_consumed": False,
            "blockers": list(self.blockers),
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_plan_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("plan_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def load_prospective_acquisition_plan(
    path: str | Path, *, expected_sha256: str
) -> tuple[dict[str, object], bytes]:
    _require_sha256(expected_sha256, label="expected plan")
    if expected_sha256 != DOCUMENT_SHA256:
        raise ChLtProspectiveAcquisitionPlanError(
            "caller-held plan SHA-256 does not match frozen canonical identity"
        )
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path(os.path.abspath(selected))
    canonical = Path(__file__).resolve().parents[2].joinpath(
        *PLAN_RELATIVE_PATH.split("/")
    )
    if _path_key(selected) != _path_key(canonical):
        raise ChLtProspectiveAcquisitionPlanError("plan path is not canonical")
    payload = read_stable_single_link_file(
        selected,
        label="CH LT prospective acquisition plan",
        max_bytes=_MAX_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != DOCUMENT_SHA256:
        raise ChLtProspectiveAcquisitionPlanError("plan SHA-256 mismatch")
    document = _strict_mapping(payload, label="plan")
    return dict(document), payload


def assess_prospective_acquisition_plan(
    document: Mapping[str, object], *, document_bytes: bytes | None = None
) -> ProspectiveAcquisitionAssessment:
    if document_bytes is None:
        raise ChLtProspectiveAcquisitionPlanError(
            "captured canonical plan bytes are required"
        )
    if hashlib.sha256(document_bytes).hexdigest() != DOCUMENT_SHA256:
        raise ChLtProspectiveAcquisitionPlanError(
            "captured plan bytes do not match frozen canonical identity"
        )
    captured = _strict_mapping(document_bytes, label="captured plan")
    if not _strict_equal(captured, document):
        raise ChLtProspectiveAcquisitionPlanError(
            "plan mapping differs from captured bytes"
        )
    _exact_keys(document, _TOP_LEVEL_KEYS, label="plan")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("lifecycle_state"), LIFECYCLE_STATE, "lifecycle")
    _equal(document.get("as_of_utc"), "2026-07-27T00:00:00Z", "as-of")
    _equal(
        document.get("scope"),
        "FRESH_CH_LT_LEVEL_HOURLY_AND_QUARTER_HOURLY_EVIDENCE",
        "scope",
    )
    _equal(document.get("market"), "CH", "market")
    _equal(document.get("timezone"), "Europe/Zurich", "timezone")
    for field in (
        "execution_authorized",
        "scientific_admission",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        _equal(document.get(field), False, field)
    plan_id = _sha256(document.get("plan_id"), label="plan_id")
    if plan_id != PLAN_ID:
        raise ChLtProspectiveAcquisitionPlanError(
            "plan_id does not match frozen canonical identity"
        )
    if plan_id != compute_plan_id(document):
        raise ChLtProspectiveAcquisitionPlanError("plan_id does not bind plan content")

    _verify_scientific_boundary(document.get("scientific_boundary"))
    blockers = _verify_tracks(document.get("acquisition_tracks"))
    _verify_admission_controls(document.get("admission_controls"))
    _verify_source_observations(document.get("primary_source_observations"))
    _verify_next_actions(document.get("next_actions"))
    return ProspectiveAcquisitionAssessment(
        plan_id=plan_id,
        document_sha256=DOCUMENT_SHA256,
        status=STATUS,
        blockers=tuple(blockers),
    )


def _verify_scientific_boundary(value: object) -> None:
    boundary = _mapping(value, label="scientific boundary")
    expected = {
        "monthly_level_authority": "SOLVER_WITH_HARD_GOVERNED_CH_EEX_FORWARD_QUOTES",
        "hourly_shape_truth": "CH_DAY_AHEAD_AUCTION_PRICE_NATIVE_HOURLY",
        "quarter_hour_shape_truth_status": (
            "UNSUPPORTED_15MIN_UNTIL_CH_EXPLICIT_AUCTION_GO_LIVE_AND_PRODUCT_IDENTITY_ARE_PROVEN"
        ),
        "future_quarter_hour_shape_truth": (
            "CH_EXPLICIT_DAY_AHEAD_15MIN_AUCTION_PRODUCT_TO_BE_FROZEN_AFTER_CONFIRMED_GO_LIVE"
        ),
        "intraday_continuous_15min_role": (
            "SEPARATE_DIAGNOSTIC_PRODUCT_NEVER_DAY_AHEAD_TRUTH"
        ),
        "cross_product_absolute_level_pooling": False,
        "hourly_duplication_as_quarter_hour_truth": False,
        "neighbor_proxy_as_direct_ch_truth": False,
        "swissgrid_imbalance_as_spot_truth": False,
        "ompex_role": "DIAGNOSTIC_BENCHMARK_ONLY",
    }
    _equal(dict(boundary), expected, "scientific boundary")


def _verify_tracks(value: object) -> list[str]:
    tracks = _sequence(value, label="acquisition tracks")
    if len(tracks) != 4:
        raise ChLtProspectiveAcquisitionPlanError("acquisition track count is invalid")
    by_id: dict[str, Mapping[str, object]] = {}
    for raw in tracks:
        track = _mapping(raw, label="acquisition track")
        track_id = str(track.get("track_id", ""))
        if not track_id or track_id in by_id:
            raise ChLtProspectiveAcquisitionPlanError("acquisition track identity is invalid")
        by_id[track_id] = track
        _equal(track.get("current_admissible"), False, f"{track_id} admissibility")
        blockers = _sequence(track.get("blockers"), label=f"{track_id} blockers")
        if not blockers or not all(isinstance(item, str) and item for item in blockers):
            raise ChLtProspectiveAcquisitionPlanError(f"{track_id} blockers are invalid")
    _equal(
        set(by_id),
        {
            "CH_DAY_AHEAD_HOURLY_ENERGY_CHARTS",
            "CH_INTRADAY_CONTINUOUS_15MIN_EPEX",
            "CH_EXPLICIT_AUCTION_15MIN_GO_LIVE_MONITOR",
            "CH_EEX_FORWARD_POINT_IN_TIME",
        },
        "track identities",
    )
    hourly = by_id["CH_DAY_AHEAD_HOURLY_ENERGY_CHARTS"]
    _equal(hourly.get("source_system"), "ENERGY_CHARTS_API", "hourly source")
    _equal(hourly.get("source_product"), "DAY_AHEAD_PRICE_CH", "hourly product")
    _equal(hourly.get("raw_cadence_seconds"), 3600, "hourly cadence")
    _equal(hourly.get("expected_source_observation_count"), 720, "hourly count")
    _equal(
        hourly.get("native_quarter_hour_truth_eligible"), False, "hourly QH eligibility"
    )
    _equal(
        hourly.get("native_hourly_truth_eligible_after_external_admission"),
        True,
        "hourly eligibility",
    )
    quarter = by_id["CH_INTRADAY_CONTINUOUS_15MIN_EPEX"]
    for field, expected in {
        "source_system": "EPEX_SPOT_LICENSED_MARKET_DATA",
        "market_area": "CH",
        "trading_modality": "CONTINUOUS",
        "product": "15_MIN",
        "price_statistic": "VOLUME_WEIGHTED_AVERAGE_EUR_MWH",
        "volume_statistic": "TRADED_VOLUME_MWH",
        "raw_cadence_seconds": 900,
        "positive_volume_required": True,
        "zero_or_missing_volume_policy": "UNSUPPORTED_NEVER_PRICE_ZERO_OR_IMPUTED",
        "complete_four_quarters_per_local_hour_required": True,
        "current_repo_adapter_implemented": False,
        "licensed_provider_access_bound": False,
        "product_identity_manifest_frozen": False,
        "native_quarter_hour_product_observation_eligible_after_external_admission": True,
        "authoritative_ch_day_ahead_quarter_hour_truth_eligible": False,
    }.items():
        _equal(quarter.get(field), expected, f"quarter-hour {field}")
    monitor = by_id["CH_EXPLICIT_AUCTION_15MIN_GO_LIVE_MONITOR"]
    for field, expected in {
        "source_system": "EPEX_SPOT_AND_SWISSGRID_OFFICIAL_NOTICES",
        "target_product": "CH_EXPLICIT_DAY_AHEAD_15MIN_AUCTION",
        "planned_go_live_window": "Q3_2026",
        "confirmed_go_live": False,
        "first_delivery_date": None,
        "official_notice_bytes_sha256": None,
        "licensed_market_data_schema_sha256": None,
        "current_day_ahead_reference_cadence_seconds": 3600,
        "future_target_cadence_seconds": 900,
        "current_status": "PLANNED_NOT_CONFIRMED_UNSUPPORTED_15MIN",
    }.items():
        _equal(monitor.get(field), expected, f"go-live monitor {field}")
    forwards = by_id["CH_EEX_FORWARD_POINT_IN_TIME"]
    _equal(
        forwards.get("source_product_family"),
        "CH_BASE_PEAK_OFFPEAK_MONTH_QUARTER_CALENDAR",
        "forward products",
    )
    _equal(forwards.get("monthly_solver_authority_input"), True, "forward authority")
    _equal(forwards.get("current_repo_source_kind"), "TEST_FIXTURE", "forward current source")
    return [
        blocker
        for track in tracks
        for blocker in _sequence(
            _mapping(track, label="track").get("blockers"), label="track blockers"
        )
    ]


def _verify_admission_controls(value: object) -> None:
    controls = _mapping(value, label="admission controls")
    expected_keys = {
        "exact_provider_raw_bytes",
        "source_specific_replay_equality",
        "available_at_complete_and_le_origin",
        "independent_trusted_time_receipt",
        "independent_signature",
        "builder_inaccessible_immutable_namespace",
        "external_cas_receipt_and_fresh_authenticated_head",
        "no_link_reparse_or_hardlink_alias",
        "prediction_and_scenario_sealed_before_truth_open",
        "one_shot_future_holdout_ledger",
    }
    _exact_keys(controls, expected_keys, label="admission controls")
    if any(value is not True for value in controls.values()):
        raise ChLtProspectiveAcquisitionPlanError("every admission control must be required")


def _verify_source_observations(value: object) -> None:
    observations = _sequence(value, label="primary source observations")
    if len(observations) != 3:
        raise ChLtProspectiveAcquisitionPlanError("primary source observation count is invalid")
    expected_sources = {
        "ENERGY_CHARTS_API_V1_6",
        "EPEX_SPOT_MARKET_RESULTS",
        "SWISSGRID_BALANCING_ROADMAP_2026_2030",
    }
    observed_sources: set[str] = set()
    for raw in observations:
        item = _mapping(raw, label="primary source observation")
        observed_sources.add(str(item.get("source")))
        _equal(item.get("retrieved_on"), "2026-07-27", "source retrieval date")
        _equal(item.get("content_bytes_captured"), False, "source byte capture")
        _equal(item.get("admission_evidence"), False, "source admission")
    _equal(observed_sources, expected_sources, "primary source identities")


def _verify_next_actions(value: object) -> None:
    actions = _sequence(value, label="next actions")
    if len(actions) != 7:
        raise ChLtProspectiveAcquisitionPlanError("next action count is invalid")
    if [item.get("order") for item in map(lambda raw: _mapping(raw, label="action"), actions)] != list(
        range(1, 8)
    ):
        raise ChLtProspectiveAcquisitionPlanError("next action order is invalid")
    for raw in actions:
        _equal(_mapping(raw, label="action").get("non_production_only"), True, "action scope")


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtProspectiveAcquisitionPlanError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtProspectiveAcquisitionPlanError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChLtProspectiveAcquisitionPlanError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtProspectiveAcquisitionPlanError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtProspectiveAcquisitionPlanError(f"{label} is invalid")


def _sha256(value: object, *, label: str) -> str:
    _require_sha256(value, label=label)
    assert isinstance(value, str)
    return value


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtProspectiveAcquisitionPlanError(f"{label} SHA-256 is invalid")


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DOCUMENT_SHA256",
    "PLAN_RELATIVE_PATH",
    "PLAN_ID",
    "STATUS",
    "ChLtProspectiveAcquisitionPlanError",
    "assess_prospective_acquisition_plan",
    "compute_plan_id",
    "load_prospective_acquisition_plan",
]
