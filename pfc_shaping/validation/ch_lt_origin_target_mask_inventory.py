"""Outcome-blind structural origin/target/mask inventory for Swiss LT evaluation.

The inventory fixes the exact 36 full local delivery months implied by one
caller-supplied UTC origin.  It deliberately contains neither predictions nor
post-origin truth and therefore cannot satisfy the corresponding scientific
evidence slot by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import ch_lt_dependence_power_design as power_design
from pfc_shaping.validation import ch_lt_estimand_contract as estimand
from pfc_shaping.validation import ch_lt_successor_candidate_core_v3 as core_v3
from pfc_shaping.validation import ch_market_time_regime as market_time

SCHEMA_VERSION = "ch_lt_origin_target_mask_inventory.v2"
ASSESSMENT_SCHEMA_VERSION = "ch_lt_origin_target_mask_inventory_assessment.v2"
STATUS = "LOCAL_OUTCOME_BLIND_STRUCTURAL_INVENTORY_NOT_EXTERNALLY_FROZEN_NO_GO"
OUTPUT_RELATIVE_DIRECTORY = "build/ch-lt-origin-target-mask-inventories"
HORIZON_MONTHS = 36

_TZ = ZoneInfo("Europe/Zurich")
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 4 * 1024 * 1024
_TOP_LEVEL_KEYS = {
    "schema_version",
    "inventory_name",
    "lifecycle",
    "bindings",
    "origin_policy",
    "mask_policy",
    "origin",
    "targets",
    "inventory_id",
    "evidence_slot_satisfied",
    "predictions_sealed",
    "outcomes_opened",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
}
_TARGET_KEYS = {
    "target_id",
    "lead_month",
    "lead_bucket",
    "delivery_month",
    "delivery_local_start",
    "delivery_local_end",
    "delivery_start_utc",
    "delivery_end_utc",
    "expected_native_hourly_interval_count",
    "expected_quarter_hour_interval_count",
    "full_local_delivery_month",
    "structural_masks",
    "origin_available_eex_product_inventory_state",
    "prediction_state",
    "truth_state",
    "evaluation_eligible",
}
_AUTHORITY_FIELDS = (
    "evidence_slot_satisfied",
    "predictions_sealed",
    "outcomes_opened",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
)
_ORIGIN_POLICY = {
    "market": "CH",
    "timezone": "Europe/Zurich",
    "origin_timestamp_axis": "UTC",
    "origin_timestamp_source": "CALLER_SUPPLIED_NEVER_INFERRED",
    "issuance_cadence": "UNSET_REQUIRES_FMV_APPROVAL",
    "issuance_calendar": "UNSET_REQUIRES_FMV_APPROVAL",
    "duplicate_origin_as_of_utc": "FORBIDDEN",
    "origin_order": "STRICTLY_INCREASING_WHEN_MULTIPLE_ORIGINS_ARE_COMBINED",
    "delivery_support": "FULL_LOCAL_DELIVERY_MONTHS_ONLY",
    "minimum_lead_month": 1,
    "maximum_lead_month": 36,
    "primary_weighting_unit": "UNIQUE_ORIGIN_AS_OF_UTC",
    "outcome_access_during_construction": "FORBIDDEN",
    "artifact_role": "STRUCTURAL_DRY_RUN_NOT_A_COUNTABLE_PROSPECTIVE_ORIGIN",
}
_MASK_POLICY = {
    "market_consistency": "PENDING_ORIGIN_AVAILABLE_CH_EEX_PRODUCT_INVENTORY",
    "hourly_shape": "PENDING_POST_DELIVERY_DIRECT_NATIVE_CH_HOURLY_OR_FINER_TRUTH",
    "quarter_hourly_shape": "UNSUPPORTED_MARKET_TRANSITION_NOT_ADMITTED",
    "probabilistic_scenarios": "UNSUPPORTED_SCENARIO_DESIGN_NOT_ADMITTED",
    "transition_crossing_month": (
        "UNSUPPORTED_UNTIL_EXACT_GO_LIVE_BOUNDARY_AND_SEPARATE_SCORING_GRAPH_ARE_FROZEN"
    ),
    "missing_or_nonfinite": "UNSUPPORTED_NEVER_ZERO_OR_WIN",
    "final_complete_case_mask": (
        "MUST_BE_SEALED_BEFORE_ANY_POST_ORIGIN_OUTCOME_IS_OPENED"
    ),
    "hourly_duplication_as_quarter_hour_truth": "FORBIDDEN",
}
_STRUCTURAL_MASKS = {
    "MARKET_CONSISTENCY": "PENDING_ORIGIN_AVAILABLE_PRODUCT_INVENTORY",
    "HOURLY_SHAPE": "PENDING_POST_DELIVERY_NATIVE_TRUTH",
    "QUARTER_HOURLY_SHAPE": "UNSUPPORTED_MARKET_TRANSITION_NOT_ADMITTED",
    "PROBABILISTIC_SCENARIOS": "UNSUPPORTED_SCENARIO_DESIGN_NOT_ADMITTED",
}


class ChLtOriginTargetMaskInventoryError(ValueError):
    """Raised when structural inventory bytes or their bindings are invalid."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compute_inventory_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("inventory_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def build_structural_inventory(
    *, repo_root: str | Path, origin_as_of_utc: str
) -> dict[str, object]:
    """Build one deterministic structural inventory without reading outcomes."""

    root = _verified_repo_root(repo_root)
    _verify_bound_governance(root)
    origin_utc = _parse_origin(origin_as_of_utc)
    origin_local = origin_utc.astimezone(_TZ)
    origin_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "market": "CH",
                "as_of_utc": origin_as_of_utc,
                "timezone": "Europe/Zurich",
            }
        )
    ).hexdigest()
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "inventory_name": (
            "ch_lt_origin_target_mask_inventory_"
            + origin_as_of_utc.replace(":", "").replace("-", "")
            + "_v2"
        ),
        "lifecycle": {
            "state": "LOCAL_STRUCTURAL_DRY_RUN_NOT_EXTERNALLY_FROZEN",
            "externally_frozen": False,
            "external_signature": None,
            "trusted_time_receipt": None,
            "external_cas_receipt": None,
        },
        "bindings": _expected_bindings(),
        "origin_policy": _ORIGIN_POLICY,
        "mask_policy": _MASK_POLICY,
        "origin": {
            "origin_id": origin_id,
            "as_of_utc": origin_as_of_utc,
            "as_of_local": origin_local.isoformat(timespec="seconds"),
            "origin_local_year": origin_local.year,
            "origin_local_month": origin_local.month,
        },
        "targets": [
            _target_row(origin_local=origin_local, origin_id=origin_id, lead_month=lead)
            for lead in range(1, HORIZON_MONTHS + 1)
        ],
        "evidence_slot_satisfied": False,
        "predictions_sealed": False,
        "outcomes_opened": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }
    document["inventory_id"] = compute_inventory_id(document)
    assess_structural_inventory(document, document_bytes=canonical_json_bytes(document) + b"\n")
    return document


def write_structural_inventory(
    *, document: Mapping[str, object], output_path: str | Path, repo_root: str | Path
) -> dict[str, object]:
    """Write immutable canonical bytes only in the governed build namespace."""

    root = _verified_repo_root(repo_root)
    target = _validated_output_path(root, output_path)
    payload = canonical_json_bytes(document) + b"\n"
    assessment = assess_structural_inventory(document, document_bytes=payload)
    _verify_bound_governance(root)
    write_durable_exact_bytes(target, payload, label="CH LT origin target mask inventory")
    return {
        **assessment,
        "inventory_path": str(target),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "inventory_size_bytes": len(payload),
        "governance_bindings_verified": True,
    }


def verify_structural_inventory(
    *, repo_root: str | Path, inventory_path: str | Path, expected_inventory_sha256: str
) -> dict[str, object]:
    """Verify caller-held bytes, canonical encoding and frozen governance bindings."""

    _require_sha256(expected_inventory_sha256, label="expected inventory")
    root = _verified_repo_root(repo_root)
    target = _validated_output_path(root, inventory_path)
    payload = read_stable_single_link_file(
        target, label="CH LT origin target mask inventory", max_bytes=_MAX_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != expected_inventory_sha256:
        raise ChLtOriginTargetMaskInventoryError("inventory SHA-256 mismatch")
    document = _strict_mapping(payload, label="inventory")
    result = assess_structural_inventory(document, document_bytes=payload)
    _verify_bound_governance(root)
    return {
        **result,
        "inventory_path": str(target),
        "inventory_sha256": expected_inventory_sha256,
        "inventory_size_bytes": len(payload),
        "governance_bindings_verified": True,
    }


def assess_structural_inventory(
    document: Mapping[str, object], *, document_bytes: bytes
) -> dict[str, object]:
    """Validate semantics without consulting predictions or outcome-bearing truth."""

    parsed = _strict_mapping(document_bytes, label="inventory bytes")
    _equal(parsed, document, "inventory mapping/bytes")
    if document_bytes != canonical_json_bytes(document) + b"\n":
        raise ChLtOriginTargetMaskInventoryError("inventory bytes are not canonical")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="inventory")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    inventory_id = _sha256(document.get("inventory_id"), label="inventory id")
    _equal(inventory_id, compute_inventory_id(document), "computed inventory id")
    _equal(document.get("bindings"), _expected_bindings(), "governance bindings")
    _equal(document.get("origin_policy"), _ORIGIN_POLICY, "origin policy")
    _equal(document.get("mask_policy"), _MASK_POLICY, "mask policy")
    _verify_lifecycle(document.get("lifecycle"))

    origin = _mapping(document.get("origin"), label="origin")
    _exact_keys(
        origin,
        {
            "origin_id",
            "as_of_utc",
            "as_of_local",
            "origin_local_year",
            "origin_local_month",
        },
        label="origin",
    )
    origin_utc = _parse_origin(_text(origin.get("as_of_utc"), label="origin as_of_utc"))
    origin_local = origin_utc.astimezone(_TZ)
    _equal(
        origin.get("as_of_local"), origin_local.isoformat(timespec="seconds"), "origin local"
    )
    _equal(origin.get("origin_local_year"), origin_local.year, "origin local year")
    _equal(origin.get("origin_local_month"), origin_local.month, "origin local month")
    expected_origin_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "market": "CH",
                "as_of_utc": origin.get("as_of_utc"),
                "timezone": "Europe/Zurich",
            }
        )
    ).hexdigest()
    _equal(origin.get("origin_id"), expected_origin_id, "origin id")
    expected_name = (
        "ch_lt_origin_target_mask_inventory_"
        + _text(origin.get("as_of_utc"), label="origin as_of_utc")
        .replace(":", "")
        .replace("-", "")
        + "_v2"
    )
    _equal(document.get("inventory_name"), expected_name, "inventory name")

    targets = _sequence(document.get("targets"), label="targets")
    if len(targets) != HORIZON_MONTHS:
        raise ChLtOriginTargetMaskInventoryError("target count must equal 36")
    for lead_month, raw in enumerate(targets, start=1):
        observed = _mapping(raw, label=f"target M{lead_month:02d}")
        _exact_keys(observed, _TARGET_KEYS, label=f"target M{lead_month:02d}")
        expected = _target_row(
            origin_local=origin_local,
            origin_id=expected_origin_id,
            lead_month=lead_month,
        )
        _equal(observed, expected, f"target M{lead_month:02d}")
    for field in _AUTHORITY_FIELDS:
        _equal(document.get(field), False, field)
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": STATUS,
        "inventory_id": inventory_id,
        "origin_id": expected_origin_id,
        "origin_as_of_utc": origin.get("as_of_utc"),
        "target_count": HORIZON_MONTHS,
        "minimum_lead_month": 1,
        "maximum_lead_month": HORIZON_MONTHS,
        "outcome_blind": True,
        "structural_dry_run": True,
        "countable_prospective_origin": False,
        "governance_bindings_verified": False,
        "evidence_slot_satisfied": False,
        "predictions_sealed": False,
        "outcomes_opened": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }


def _target_row(*, origin_local: datetime, origin_id: str, lead_month: int) -> dict[str, object]:
    year, month = _add_months(origin_local.year, origin_local.month, lead_month)
    next_year, next_month = _add_months(year, month, 1)
    local_start = datetime(year, month, 1, tzinfo=_TZ)
    local_end = datetime(next_year, next_month, 1, tzinfo=_TZ)
    utc_start = local_start.astimezone(UTC)
    utc_end = local_end.astimezone(UTC)
    seconds = int((utc_end - utc_start).total_seconds())
    if seconds % 3600 != 0:
        raise ChLtOriginTargetMaskInventoryError("delivery month is not hour aligned")
    hours = seconds // 3600
    bucket = _lead_bucket(lead_month)
    delivery_month = f"{year:04d}-{month:02d}"
    return {
        "target_id": f"{origin_id}:M{lead_month:02d}:{delivery_month}",
        "lead_month": lead_month,
        "lead_bucket": bucket,
        "delivery_month": delivery_month,
        "delivery_local_start": local_start.isoformat(timespec="seconds"),
        "delivery_local_end": local_end.isoformat(timespec="seconds"),
        "delivery_start_utc": utc_start.strftime(_UTC_FORMAT),
        "delivery_end_utc": utc_end.strftime(_UTC_FORMAT),
        "expected_native_hourly_interval_count": hours,
        "expected_quarter_hour_interval_count": hours * 4,
        "full_local_delivery_month": True,
        "structural_masks": _STRUCTURAL_MASKS,
        "origin_available_eex_product_inventory_state": "NOT_ATTACHED",
        "prediction_state": "NOT_CAPTURED_STRUCTURAL_DRY_RUN",
        "truth_state": "NOT_AVAILABLE_OR_NOT_BOUND_NOT_READ",
        "evaluation_eligible": False,
    }


def _verify_bound_governance(root: Path) -> None:
    estimand_path = _repo_path(root, estimand.CONTRACT_RELATIVE_PATH)
    document, payload = estimand.load_estimand_contract(
        estimand_path, expected_sha256=estimand.DOCUMENT_SHA256
    )
    assessment = estimand.assess_estimand_contract(document, document_bytes=payload)
    _equal(assessment.contract_id, estimand.CONTRACT_ID, "estimand contract id")
    _equal(
        estimand.EXPECTED_HORIZON_POLICY.get("eligible_lead_month_min"),
        1,
        "estimand minimum lead month",
    )
    _equal(
        estimand.EXPECTED_HORIZON_POLICY.get("eligible_lead_month_max"),
        HORIZON_MONTHS,
        "estimand maximum lead month",
    )
    _equal(
        estimand.EXPECTED_HORIZON_POLICY.get("buckets"),
        [
            {"bucket_id": "M01_M06", "minimum_lead_month": 1, "maximum_lead_month": 6},
            {"bucket_id": "M07_M12", "minimum_lead_month": 7, "maximum_lead_month": 12},
            {"bucket_id": "M13_M24", "minimum_lead_month": 13, "maximum_lead_month": 24},
            {"bucket_id": "M25_M36", "minimum_lead_month": 25, "maximum_lead_month": 36},
        ],
        "estimand lead buckets",
    )
    _equal(
        estimand.EXPECTED_ORIGIN_DESIGN_POLICY.get("primary_observation_and_weighting_unit"),
        "UNIQUE_ORIGIN_AS_OF_UTC",
        "estimand origin unit",
    )
    _equal(
        estimand.EXPECTED_ORIGIN_DESIGN_POLICY.get("duplicate_origin_as_of_utc"),
        "FORBIDDEN",
        "estimand duplicate origin policy",
    )
    core_result = core_v3.verify_candidate_core_v3(
        repo_root=root,
        core_path=_repo_path(root, core_v3.CORE_RELATIVE_PATH),
        expected_core_sha256=core_v3.CORE_SHA256,
    )
    _equal(core_result.get("core_id"), core_v3.CORE_ID, "candidate core id")
    design_result = power_design.verify_dependence_power_design(
        repo_root=root,
        design_path=_repo_path(root, power_design.DESIGN_RELATIVE_PATH),
        expected_design_sha256=power_design.DESIGN_SHA256,
    )
    _equal(design_result.get("design_id"), power_design.DESIGN_ID, "power design id")
    market_document, market_payload = market_time.load_market_time_regime_contract(
        _repo_path(root, market_time.CONTRACT_RELATIVE_PATH),
        expected_sha256=market_time.DOCUMENT_SHA256,
        repo_root=root,
    )
    market_assessment = market_time.assess_market_time_regime(
        market_document,
        document_bytes=market_payload,
        repo_root=root,
        verify_evidence_files=True,
    )
    _equal(market_assessment.contract_id, market_time.CONTRACT_ID, "market-time id")
    _equal(market_assessment.status, market_time.STATUS, "market-time status")
    market_result = market_assessment.to_dict()
    _equal(
        market_result.get("native_day_ahead_market_time_unit_minutes"),
        60,
        "market-time native day-ahead resolution",
    )
    _equal(market_result.get("transition_admitted"), False, "market-time transition")


def _expected_bindings() -> dict[str, object]:
    return {
        "estimand_contract": {
            "path": estimand.CONTRACT_RELATIVE_PATH,
            "sha256": estimand.DOCUMENT_SHA256,
            "contract_id": estimand.CONTRACT_ID,
        },
        "candidate_core": {
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
            "confirmed_go_live": False,
            "transition_admitted": False,
        },
    }


def _validated_output_path(root: Path, path: str | Path) -> Path:
    selected = _absolute(path)
    output_root = assert_absolute_path_has_no_links(
        _repo_path(root, OUTPUT_RELATIVE_DIRECTORY)
    )
    if not output_root.is_dir():
        raise ChLtOriginTargetMaskInventoryError("inventory output directory is unavailable")
    selected = assert_absolute_path_has_no_links(selected)
    if selected.parent != output_root or selected.suffix != ".json":
        raise ChLtOriginTargetMaskInventoryError(
            "inventory path must be a direct JSON child of the governed build namespace"
        )
    return selected


def _verified_repo_root(path: str | Path) -> Path:
    root = assert_absolute_path_has_no_links(_absolute(path))
    if not root.is_dir() or not (root / ".git").exists():
        raise ChLtOriginTargetMaskInventoryError("canonical repository root is unavailable")
    if os.path.normcase(str(Path.cwd().resolve(strict=True))) != os.path.normcase(str(root)):
        raise ChLtOriginTargetMaskInventoryError("cwd must equal the canonical repository root")
    return root


def _parse_origin(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ChLtOriginTargetMaskInventoryError("origin as_of_utc must use canonical UTC Z")
    try:
        parsed = datetime.strptime(value, _UTC_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ChLtOriginTargetMaskInventoryError(
            "origin as_of_utc must use YYYY-MM-DDTHH:MM:SSZ"
        ) from exc
    if parsed.strftime(_UTC_FORMAT) != value:
        raise ChLtOriginTargetMaskInventoryError("origin as_of_utc is not canonical")
    return parsed


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    zero_based = year * 12 + (month - 1) + delta
    return divmod(zero_based, 12)[0], divmod(zero_based, 12)[1] + 1


def _lead_bucket(lead_month: int) -> str:
    if 1 <= lead_month <= 6:
        return "M01_M06"
    if 7 <= lead_month <= 12:
        return "M07_M12"
    if 13 <= lead_month <= 24:
        return "M13_M24"
    if 25 <= lead_month <= 36:
        return "M25_M36"
    raise ChLtOriginTargetMaskInventoryError("lead month is outside M01..M36")


def _verify_lifecycle(value: object) -> None:
    lifecycle = _mapping(value, label="lifecycle")
    _equal(
        lifecycle,
        {
            "state": "LOCAL_STRUCTURAL_DRY_RUN_NOT_EXTERNALLY_FROZEN",
            "externally_frozen": False,
            "external_signature": None,
            "trusted_time_receipt": None,
            "external_cas_receipt": None,
        },
        "lifecycle",
    )


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtOriginTargetMaskInventoryError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtOriginTargetMaskInventoryError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChLtOriginTargetMaskInventoryError(f"{label} must be an array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChLtOriginTargetMaskInventoryError(f"{label} must be non-empty text")
    return value


def _sha256(value: object, *, label: str) -> str:
    _require_sha256(value, label=label)
    return str(value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtOriginTargetMaskInventoryError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtOriginTargetMaskInventoryError(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtOriginTargetMaskInventoryError("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    return selected if selected.is_absolute() else Path(os.path.abspath(selected))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtOriginTargetMaskInventoryError(f"{label} SHA-256 is invalid")


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "HORIZON_MONTHS",
    "OUTPUT_RELATIVE_DIRECTORY",
    "SCHEMA_VERSION",
    "STATUS",
    "ChLtOriginTargetMaskInventoryError",
    "assess_structural_inventory",
    "build_structural_inventory",
    "canonical_json_bytes",
    "compute_inventory_id",
    "verify_structural_inventory",
    "write_structural_inventory",
]
