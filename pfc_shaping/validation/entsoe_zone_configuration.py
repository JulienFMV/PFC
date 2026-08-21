"""Validate temporal ENTSO-E area-code mappings for coupled-zone PFC inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from urllib.parse import urlsplit

from pfc_shaping.validation.entsoe_pfc_data_coverage import COUPLED_PRICE_ZONES

REGISTRY_SCHEMA = "fmv_entsoe_zone_configuration_registry.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_zone_configuration_assessment.v1"
STATUS = (
    "PASS_TEMPORAL_ZONE_MAPPING_STRUCTURE_OWNER_ASSERTED_"
    "OFFICIAL_REGISTRY_UNVERIFIED_NO_MODEL_AUTHORITY"
)
EVIDENCE_CLASSES = frozenset({"REAL_OWNER_ASSERTED", "SYNTHETIC_TEST_ONLY"})
REQUIRED_LOGICAL_ZONES = tuple(COUPLED_PRICE_ZONES)
SOURCE_IDENTIFIER_SCHEMES = frozenset({"EIC_Y_CODE", "OWNER_CANONICAL_LABEL"})
DOMAIN_KINDS = frozenset(
    {"BIDDING_ZONE", "CONTROL_AREA", "COUNTRY", "MARKET_BALANCE_AREA"}
)
SOURCE_STATUSES = frozenset({"ACTIVE", "DEACTIVATED", "SUPERSEDED"})
SOURCE_DOCUMENT_ID_KINDS = frozenset(
    {"OFFICIAL_EIC_REGISTRY_ENTRY_ID", "GOVERNED_EIC_REGISTRY_SNAPSHOT_ID"}
)
AUTHORITIES = {
    "external_registry_owner_identity_verified": False,
    "official_eic_registry_membership_verified": False,
    "official_eic_check_digit_verified": False,
    "current_exact_dev_zone_coverage_proven": False,
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "production_authorized": False,
}

_EIC = re.compile(r"^[0-9A-Z-]{16}$")
_OWNER = re.compile(r"^[A-Za-z0-9_.@ -]{3,128}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class EntsoeZoneConfigurationError(ValueError):
    """Raised when a zone mapping is incomplete, overlapping, or ambiguous."""


def assess_zone_configuration(document: Mapping[str, object]) -> dict[str, object]:
    """Validate an owner registry and report temporal bidding-zone coverage."""

    registry = _mapping(document, "zone registry")
    _exact(
        registry,
        {
            "schema_version",
            "evidence_class",
            "registry_owner",
            "registered_at_utc",
            "coverage_window_start_utc",
            "coverage_window_end_utc",
            "entries",
            "authorities",
        },
        "zone registry",
    )
    _equal(registry["schema_version"], REGISTRY_SCHEMA, "registry schema")
    evidence_class = registry["evidence_class"]
    if evidence_class not in EVIDENCE_CLASSES:
        raise EntsoeZoneConfigurationError("registry evidence class is invalid")
    owner = registry["registry_owner"]
    _clean_string(owner, "registry owner")
    if not isinstance(owner, str) or _OWNER.fullmatch(owner) is None:
        raise EntsoeZoneConfigurationError("registry owner format is invalid")
    if evidence_class == "REAL_OWNER_ASSERTED" and owner.casefold().startswith(
        "synthetic"
    ):
        raise EntsoeZoneConfigurationError("real registry owner is synthetic")
    _utc(registry["registered_at_utc"], "registered_at_utc")
    window_start = _utc(
        registry["coverage_window_start_utc"], "coverage_window_start_utc"
    )
    window_end = _utc(
        registry["coverage_window_end_utc"], "coverage_window_end_utc"
    )
    if window_start >= window_end:
        raise EntsoeZoneConfigurationError("coverage window is empty or reversed")
    _equal(registry["authorities"], AUTHORITIES, "registry authorities")

    raw_entries = registry["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise EntsoeZoneConfigurationError("zone entries must be a non-empty list")
    if len(raw_entries) > 10_000:
        raise EntsoeZoneConfigurationError("zone entry limit exceeded")

    entries: list[dict[str, object]] = []
    content_ids: set[str] = set()
    for position, raw_entry in enumerate(raw_entries, start=1):
        entry = _validate_entry(raw_entry, position)
        content_id = canonical_sha256(raw_entry)
        if content_id in content_ids:
            raise EntsoeZoneConfigurationError("zone entry is duplicated")
        content_ids.add(content_id)
        entries.append(entry)

    _reject_overlaps(entries)
    gaps = _coverage_gaps(entries, window_start, window_end)
    complete = all(not zone_gaps for zone_gaps in gaps.values())
    changes = _code_change_counts(entries, window_start, window_end)
    real_owner_asserted = evidence_class == "REAL_OWNER_ASSERTED"
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": STATUS,
        "registry_content_id": canonical_sha256(registry),
        "evidence_class": evidence_class,
        "entry_count": len(entries),
        "required_logical_zones": list(REQUIRED_LOGICAL_ZONES),
        "coverage_window_start_utc": registry["coverage_window_start_utc"],
        "coverage_window_end_utc": registry["coverage_window_end_utc"],
        "missing_intervals_by_logical_zone": gaps,
        "bidding_zone_code_change_count_by_logical_zone": changes,
        "eic_y_code_format_verified": True,
        "temporal_overlap_absent": True,
        "synthetic_or_owner_asserted_window_complete": complete,
        "owner_asserted_window_complete": real_owner_asserted and complete,
        "external_registry_owner_identity_verified": False,
        "official_eic_registry_membership_verified": False,
        "official_eic_check_digit_verified": False,
        "zone_configuration_verified": False,
        "current_exact_dev_zone_coverage_proven": False,
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "production_authorized": False,
        "validator_databricks_request_count": 0,
        "validator_warehouse_start_count": 0,
        "validator_remote_write_count": 0,
    }


def resolve_owner_asserted_logical_zone(
    document: Mapping[str, object],
    *,
    source_identifier_scheme: str,
    source_identifier: str,
    domain_kind: str,
    at_utc: str,
) -> str:
    """Resolve one exact source identifier after full registry validation."""

    assess_zone_configuration(document)
    instant = _utc(at_utc, "resolution instant")
    matches: list[str] = []
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):  # narrowed by the assessment
        raise EntsoeZoneConfigurationError("zone entries must be a list")
    for position, raw_entry in enumerate(raw_entries, start=1):
        entry = _validate_entry(raw_entry, position)
        if (
            entry["source_identifier_scheme"] == source_identifier_scheme
            and entry["source_identifier"] == source_identifier
            and entry["domain_kind"] == domain_kind
            and entry["valid_from_utc_parsed"] <= instant
            < entry["valid_to_utc_parsed"]
        ):
            matches.append(str(entry["logical_zone"]))
    if len(matches) != 1:
        raise EntsoeZoneConfigurationError(
            "source identifier has no unique owner-asserted zone mapping at instant"
        )
    return matches[0]


def resolve_owner_asserted_zone_interval(
    document: Mapping[str, object],
    *,
    source_identifier_scheme: str,
    source_identifier: str,
    domain_kind: str,
    valid_from_utc: str,
    valid_to_utc: str,
) -> dict[str, str]:
    """Resolve one interval only when a single registry entry contains it."""

    assessment = assess_zone_configuration(document)
    start = _utc(valid_from_utc, "binding valid_from_utc")
    end = _utc(valid_to_utc, "binding valid_to_utc")
    if start >= end:
        raise EntsoeZoneConfigurationError("binding interval is empty or reversed")
    window_start = _utc(
        assessment["coverage_window_start_utc"], "registry coverage start"
    )
    window_end = _utc(
        assessment["coverage_window_end_utc"], "registry coverage end"
    )
    if start < window_start or end > window_end:
        raise EntsoeZoneConfigurationError(
            "binding interval exceeds the registry coverage window"
        )
    matches: list[tuple[str, str]] = []
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise EntsoeZoneConfigurationError("zone entries must be a list")
    for position, raw_entry in enumerate(raw_entries, start=1):
        entry = _validate_entry(raw_entry, position)
        if (
            entry["source_identifier_scheme"] == source_identifier_scheme
            and entry["source_identifier"] == source_identifier
            and entry["domain_kind"] == domain_kind
            and entry["valid_from_utc_parsed"] <= start
            and end <= entry["valid_to_utc_parsed"]
        ):
            matches.append(
                (str(entry["logical_zone"]), canonical_sha256(raw_entry))
            )
    if len(matches) != 1:
        raise EntsoeZoneConfigurationError(
            "binding interval has no unique containing owner-asserted zone entry"
        )
    logical_zone, entry_id = matches[0]
    return {"logical_zone": logical_zone, "zone_entry_id": entry_id}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_entry(raw_entry: object, position: int) -> dict[str, object]:
    entry = _mapping(raw_entry, f"zone entry {position}")
    _exact(
        entry,
        {
            "source_identifier_scheme",
            "source_identifier",
            "logical_zone",
            "domain_kind",
            "valid_from_utc",
            "valid_to_utc",
            "source_status",
            "source_document_id_kind",
            "source_document_id",
            "source_endpoint",
            "owner_semantics_confirmed",
            "reason",
        },
        f"zone entry {position}",
    )
    scheme = entry["source_identifier_scheme"]
    if scheme not in SOURCE_IDENTIFIER_SCHEMES:
        raise EntsoeZoneConfigurationError("source identifier scheme is invalid")
    identifier = entry["source_identifier"]
    _clean_string(identifier, f"zone entry {position} source identifier")
    if scheme == "EIC_Y_CODE":
        if (
            not isinstance(identifier, str)
            or _EIC.fullmatch(identifier) is None
            or identifier[2] != "Y"
        ):
            raise EntsoeZoneConfigurationError(
                "EIC area identifier must be a 16-character type-Y code"
            )
    logical_zone = entry["logical_zone"]
    if logical_zone not in REQUIRED_LOGICAL_ZONES:
        raise EntsoeZoneConfigurationError("logical zone is invalid")
    domain_kind = entry["domain_kind"]
    if domain_kind not in DOMAIN_KINDS:
        raise EntsoeZoneConfigurationError("domain kind is invalid")
    start = _utc(entry["valid_from_utc"], f"zone entry {position} valid_from_utc")
    end = _utc(entry["valid_to_utc"], f"zone entry {position} valid_to_utc")
    if start >= end:
        raise EntsoeZoneConfigurationError("zone validity interval is empty or reversed")
    status = entry["source_status"]
    if status not in SOURCE_STATUSES:
        raise EntsoeZoneConfigurationError("source status is invalid")
    if entry["source_document_id_kind"] not in SOURCE_DOCUMENT_ID_KINDS:
        raise EntsoeZoneConfigurationError("source document ID kind is invalid")
    _clean_string(entry["source_document_id"], f"zone entry {position} document ID")
    _https_url(entry["source_endpoint"], f"zone entry {position} source endpoint")
    if entry["owner_semantics_confirmed"] is not True:
        raise EntsoeZoneConfigurationError(
            "zone entry requires owner-confirmed semantics"
        )
    _clean_string(entry["reason"], f"zone entry {position} reason")
    return {
        **entry,
        "valid_from_utc_parsed": start,
        "valid_to_utc_parsed": end,
    }


def _reject_overlaps(entries: list[dict[str, object]]) -> None:
    by_source: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    by_logical_eic: dict[tuple[str, str], list[dict[str, object]]] = {}
    for entry in entries:
        source_key = (
            str(entry["source_identifier_scheme"]),
            str(entry["source_identifier"]),
            str(entry["domain_kind"]),
        )
        by_source.setdefault(source_key, []).append(entry)
        if entry["source_identifier_scheme"] == "EIC_Y_CODE":
            logical_key = (str(entry["logical_zone"]), str(entry["domain_kind"]))
            by_logical_eic.setdefault(logical_key, []).append(entry)
    for values in by_source.values():
        _reject_interval_overlap(values, "source identifier")
    for values in by_logical_eic.values():
        _reject_interval_overlap(values, "logical-zone EIC")


def _reject_interval_overlap(values: list[dict[str, object]], label: str) -> None:
    ordered = sorted(values, key=lambda item: item["valid_from_utc_parsed"])
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current["valid_from_utc_parsed"] < previous["valid_to_utc_parsed"]:
            raise EntsoeZoneConfigurationError(f"{label} validity intervals overlap")


def _coverage_gaps(
    entries: list[dict[str, object]],
    window_start: datetime,
    window_end: datetime,
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for zone in REQUIRED_LOGICAL_ZONES:
        relevant = sorted(
            (
                entry
                for entry in entries
                if entry["source_identifier_scheme"] == "EIC_Y_CODE"
                and entry["domain_kind"] == "BIDDING_ZONE"
                and entry["logical_zone"] == zone
                and entry["valid_to_utc_parsed"] > window_start
                and entry["valid_from_utc_parsed"] < window_end
            ),
            key=lambda item: item["valid_from_utc_parsed"],
        )
        cursor = window_start
        gaps: list[dict[str, str]] = []
        for entry in relevant:
            start = max(cursor, entry["valid_from_utc_parsed"])
            end = min(window_end, entry["valid_to_utc_parsed"])
            if start > cursor:
                gaps.append({"start_utc": _utc_text(cursor), "end_utc": _utc_text(start)})
            cursor = max(cursor, end)
        if cursor < window_end:
            gaps.append(
                {"start_utc": _utc_text(cursor), "end_utc": _utc_text(window_end)}
            )
        result[zone] = gaps
    return result


def _code_change_counts(
    entries: list[dict[str, object]],
    window_start: datetime,
    window_end: datetime,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in entries:
        if (
            entry["source_identifier_scheme"] == "EIC_Y_CODE"
            and entry["domain_kind"] == "BIDDING_ZONE"
            and entry["valid_to_utc_parsed"] > window_start
            and entry["valid_from_utc_parsed"] < window_end
        ):
            counts[str(entry["logical_zone"])] += 1
    return {zone: max(0, counts[zone] - 1) for zone in REQUIRED_LOGICAL_ZONES}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeZoneConfigurationError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeZoneConfigurationError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeZoneConfigurationError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _clean_string(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CONTROL.search(value)
    ):
        raise EntsoeZoneConfigurationError(f"{label} must be a clean string")


def _https_url(value: object, label: str) -> None:
    _clean_string(value, label)
    if not isinstance(value, str):
        raise EntsoeZoneConfigurationError(f"{label} must be HTTPS")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise EntsoeZoneConfigurationError(f"{label} must be a clean HTTPS URL")


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EntsoeZoneConfigurationError(f"{label} must be UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeZoneConfigurationError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EntsoeZoneConfigurationError(f"{label} must be UTC")
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "REGISTRY_SCHEMA",
    "REQUIRED_LOGICAL_ZONES",
    "STATUS",
    "EntsoeZoneConfigurationError",
    "assess_zone_configuration",
    "canonical_sha256",
    "resolve_owner_asserted_logical_zone",
    "resolve_owner_asserted_zone_interval",
]
