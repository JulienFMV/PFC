"""Bind exact ENTSO-E series signatures to temporal coupled-zone identities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone

from pfc_shaping.validation.entsoe_series_family_mapping import (
    DIRECTIONAL_FAMILIES,
    assess_series_family_mapping,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    assess_series_inventory,
    canonical_sha256,
    series_signature_id,
)
from pfc_shaping.validation.entsoe_zone_configuration import (
    DOMAIN_KINDS,
    SOURCE_IDENTIFIER_SCHEMES,
    assess_zone_configuration,
    resolve_owner_asserted_zone_interval,
)

BINDING_SCHEMA = "fmv_entsoe_series_zone_binding.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_series_zone_binding_assessment.v1"
STATUS = (
    "PASS_TEMPORAL_SERIES_ZONE_BINDING_OWNER_ASSERTED_"
    "NO_VALUE_OR_MODEL_AUTHORITY"
)
EVIDENCE_CLASSES = frozenset({"REAL_OWNER_ASSERTED", "SYNTHETIC_TEST_ONLY"})
SOURCE_ZONE_FIELDS = frozenset({"from_zone", "to_zone"})
ZONE_REQUIREMENTS = {
    "day_ahead_prices": ("CH", "DE_LU", "FR", "IT_NORTH", "AT"),
    "actual_load": ("CH", "DE_LU", "FR", "IT_NORTH", "AT"),
    "load_forecast": ("CH", "DE_LU", "FR", "IT_NORTH", "AT"),
    "generation_actual": ("CH", "DE_LU", "FR", "IT_NORTH", "AT"),
    "generation_forecast": ("CH", "DE_LU", "FR", "IT_NORTH", "AT"),
    "renewables_forecast_day_ahead": ("CH", "DE_LU", "FR", "IT_NORTH", "AT"),
    "renewables_forecast_intraday": ("CH", "DE_LU", "FR", "IT_NORTH", "AT"),
    "hydro_reservoir_storage": ("CH", "FR", "IT_NORTH", "AT"),
}
AUTHORITIES = {
    "external_binding_owner_identity_verified": False,
    "official_zone_registry_verified": False,
    "real_fact_delivery_coverage_proven": False,
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "production_authorized": False,
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_OWNER = re.compile(r"^[A-Za-z0-9_.@ -]{3,128}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class EntsoeSeriesZoneBindingError(ValueError):
    """Raised when a series-to-zone temporal binding is unsafe or incomplete."""


def assess_series_zone_bindings(
    inventory: Mapping[str, object],
    family_mapping: Mapping[str, object],
    zone_registry: Mapping[str, object],
    binding_document: Mapping[str, object],
) -> dict[str, object]:
    """Compose D253 and D254 without granting data or model authority."""

    inventory_assessment = assess_series_inventory(inventory)
    family_assessment = assess_series_family_mapping(inventory, family_mapping)
    zone_assessment = assess_zone_configuration(zone_registry)
    document = _mapping(binding_document, "series-zone binding document")
    _exact(
        document,
        {
            "schema_version",
            "evidence_class",
            "inventory_content_id",
            "family_mapping_content_id",
            "zone_registry_content_id",
            "binding_owner",
            "bound_at_utc",
            "bindings",
            "authorities",
        },
        "series-zone binding document",
    )
    _equal(document["schema_version"], BINDING_SCHEMA, "binding schema")
    evidence_class = document["evidence_class"]
    if evidence_class not in EVIDENCE_CLASSES:
        raise EntsoeSeriesZoneBindingError("binding evidence class is invalid")
    _equal(
        document["inventory_content_id"],
        inventory_assessment["inventory_content_id"],
        "inventory content ID",
    )
    _equal(
        document["family_mapping_content_id"],
        family_assessment["mapping_content_id"],
        "family mapping content ID",
    )
    _equal(
        document["zone_registry_content_id"],
        zone_assessment["registry_content_id"],
        "zone registry content ID",
    )
    _equal(document["authorities"], AUTHORITIES, "binding authorities")
    owner = document["binding_owner"]
    _clean_string(owner, "binding owner")
    if not isinstance(owner, str) or _OWNER.fullmatch(owner) is None:
        raise EntsoeSeriesZoneBindingError("binding owner format is invalid")
    if evidence_class == "REAL_OWNER_ASSERTED" and owner.casefold().startswith(
        "synthetic"
    ):
        raise EntsoeSeriesZoneBindingError("real binding owner is synthetic")
    bound_at = _utc(document["bound_at_utc"], "bound_at_utc")
    if bound_at < _utc(family_mapping["mapped_at_utc"], "family mapped_at_utc"):
        raise EntsoeSeriesZoneBindingError("binding predates the family mapping")
    if bound_at < _utc(zone_registry["registered_at_utc"], "registry registered_at_utc"):
        raise EntsoeSeriesZoneBindingError("binding predates the zone registry")

    family_evidence = family_assessment["evidence_class"]
    zone_evidence = zone_assessment["evidence_class"]
    if evidence_class == "SYNTHETIC_TEST_ONLY":
        if family_evidence != "SYNTHETIC_TEST_ONLY" or zone_evidence != "SYNTHETIC_TEST_ONLY":
            raise EntsoeSeriesZoneBindingError(
                "synthetic binding requires synthetic family and zone evidence"
            )
    elif family_evidence != "REAL_OWNER_REVIEWED" or zone_evidence != "REAL_OWNER_ASSERTED":
        raise EntsoeSeriesZoneBindingError(
            "real binding requires owner-asserted family and zone evidence"
        )

    inventory_by_id = _inventory_rows_by_signature(inventory)
    family_by_id = _mapped_families_by_signature(family_mapping)
    zone_scoped_ids = {
        signature_id
        for signature_id, family in family_by_id.items()
        if family in ZONE_REQUIREMENTS
    }

    raw_bindings = document["bindings"]
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise EntsoeSeriesZoneBindingError("bindings must be a non-empty list")
    if len(raw_bindings) > 50_000:
        raise EntsoeSeriesZoneBindingError("binding entry limit exceeded")

    intervals_by_signature: dict[str, list[tuple[datetime, datetime]]] = {}
    intervals_by_family_zone: dict[
        tuple[str, str], list[tuple[datetime, datetime]]
    ] = {}
    binding_ids: set[str] = set()
    bound_signatures: set[str] = set()
    for position, raw_binding in enumerate(raw_bindings, start=1):
        binding = _mapping(raw_binding, f"binding {position}")
        _exact(
            binding,
            {
                "signature_id",
                "source_zone_field",
                "source_identifier_scheme",
                "domain_kind",
                "series_valid_from_utc",
                "series_valid_to_utc",
                "expected_logical_zone",
                "zone_entry_id",
                "reason",
            },
            f"binding {position}",
        )
        binding_id = canonical_sha256(binding)
        if binding_id in binding_ids:
            raise EntsoeSeriesZoneBindingError("binding entry is duplicated")
        binding_ids.add(binding_id)
        signature_id = binding["signature_id"]
        _sha(signature_id, f"binding {position} signature ID")
        if signature_id not in family_by_id:
            raise EntsoeSeriesZoneBindingError(
                "binding references an unknown or non-mapped signature"
            )
        family = family_by_id[str(signature_id)]
        if family in DIRECTIONAL_FAMILIES:
            raise EntsoeSeriesZoneBindingError(
                "directional family must use border mapping, not zone binding"
            )
        if family not in ZONE_REQUIREMENTS:
            raise EntsoeSeriesZoneBindingError(
                "binding family is not in the coupled-zone requirement matrix"
            )
        source_field = binding["source_zone_field"]
        if source_field not in SOURCE_ZONE_FIELDS:
            raise EntsoeSeriesZoneBindingError("source zone field is invalid")
        raw_identifier = inventory_by_id[str(signature_id)][str(source_field)]
        if not isinstance(raw_identifier, str) or not raw_identifier:
            raise EntsoeSeriesZoneBindingError(
                "selected source zone field is null or invalid in the inventory"
            )
        scheme = binding["source_identifier_scheme"]
        if scheme not in SOURCE_IDENTIFIER_SCHEMES:
            raise EntsoeSeriesZoneBindingError("source identifier scheme is invalid")
        domain_kind = binding["domain_kind"]
        if domain_kind not in DOMAIN_KINDS:
            raise EntsoeSeriesZoneBindingError("domain kind is invalid")
        start = _utc(binding["series_valid_from_utc"], "series_valid_from_utc")
        end = _utc(binding["series_valid_to_utc"], "series_valid_to_utc")
        if start >= end:
            raise EntsoeSeriesZoneBindingError(
                "series binding interval is empty or reversed"
            )
        expected_zone = binding["expected_logical_zone"]
        if expected_zone not in ZONE_REQUIREMENTS[family]:
            raise EntsoeSeriesZoneBindingError(
                f"logical zone is not required for family {family}"
            )
        _sha(binding["zone_entry_id"], f"binding {position} zone entry ID")
        _clean_string(binding["reason"], f"binding {position} reason")
        try:
            resolved = resolve_owner_asserted_zone_interval(
                zone_registry,
                source_identifier_scheme=str(scheme),
                source_identifier=raw_identifier,
                domain_kind=str(domain_kind),
                valid_from_utc=str(binding["series_valid_from_utc"]),
                valid_to_utc=str(binding["series_valid_to_utc"]),
            )
        except ValueError as exc:
            raise EntsoeSeriesZoneBindingError(
                "series interval does not resolve through the bound zone registry"
            ) from exc
        _equal(resolved["logical_zone"], expected_zone, "resolved logical zone")
        _equal(resolved["zone_entry_id"], binding["zone_entry_id"], "zone entry ID")
        intervals_by_signature.setdefault(str(signature_id), []).append((start, end))
        intervals_by_family_zone.setdefault((family, str(expected_zone)), []).append(
            (start, end)
        )
        bound_signatures.add(str(signature_id))

    for intervals in intervals_by_signature.values():
        _reject_overlapping_intervals(intervals)
    missing_signature_ids = sorted(zone_scoped_ids.difference(bound_signatures))
    window_start = _utc(
        zone_registry["coverage_window_start_utc"], "registry coverage start"
    )
    window_end = _utc(
        zone_registry["coverage_window_end_utc"], "registry coverage end"
    )
    missing_intervals = {
        family: {
            zone: _interval_gaps(
                intervals_by_family_zone.get((family, zone), []),
                window_start,
                window_end,
            )
            for zone in zones
        }
        for family, zones in ZONE_REQUIREMENTS.items()
    }
    baseline_price_complete = all(
        not gaps for gaps in missing_intervals["day_ahead_prices"].values()
    )
    all_requirements_complete = all(
        not gaps
        for family_gaps in missing_intervals.values()
        for gaps in family_gaps.values()
    )
    real_owner_asserted = evidence_class == "REAL_OWNER_ASSERTED"
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": STATUS,
        "binding_content_id": canonical_sha256(document),
        "inventory_content_id": inventory_assessment["inventory_content_id"],
        "family_mapping_content_id": family_assessment["mapping_content_id"],
        "zone_registry_content_id": zone_assessment["registry_content_id"],
        "binding_entry_count": len(raw_bindings),
        "zone_scoped_mapped_signature_count": len(zone_scoped_ids),
        "bound_signature_count": len(bound_signatures),
        "missing_binding_signature_ids": missing_signature_ids,
        "missing_intervals_by_family_and_logical_zone": missing_intervals,
        "synthetic_or_owner_asserted_day_ahead_price_window_complete": (
            baseline_price_complete
        ),
        "synthetic_or_owner_asserted_all_coupled_requirements_complete": (
            all_requirements_complete
        ),
        "owner_asserted_day_ahead_price_window_complete": (
            real_owner_asserted and baseline_price_complete
        ),
        "external_binding_owner_identity_verified": False,
        "official_zone_registry_verified": False,
        "real_fact_delivery_coverage_proven": False,
        "series_zone_binding_verified": False,
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "production_authorized": False,
        "validator_databricks_request_count": 0,
        "validator_warehouse_start_count": 0,
        "validator_remote_write_count": 0,
    }


def _inventory_rows_by_signature(
    inventory: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_rows = inventory["rows"]
    if not isinstance(raw_rows, list):
        raise EntsoeSeriesZoneBindingError("inventory rows must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for raw_row in raw_rows:
        row = _mapping(raw_row, "inventory row")
        result[series_signature_id(row)] = row
    return result


def _mapped_families_by_signature(
    family_mapping: Mapping[str, object],
) -> dict[str, str]:
    raw_entries = family_mapping["entries"]
    if not isinstance(raw_entries, list):
        raise EntsoeSeriesZoneBindingError("family mapping entries must be a list")
    result: dict[str, str] = {}
    for raw_entry in raw_entries:
        entry = _mapping(raw_entry, "family mapping entry")
        if entry["disposition"] == "MAPPED":
            result[str(entry["signature_id"])] = str(entry["logical_family"])
    return result


def _reject_overlapping_intervals(intervals: list[tuple[datetime, datetime]]) -> None:
    ordered = sorted(intervals)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current[0] < previous[1]:
            raise EntsoeSeriesZoneBindingError(
                "one series signature has overlapping zone-binding intervals"
            )


def _interval_gaps(
    intervals: list[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, str]]:
    clipped = sorted(
        (max(start, window_start), min(end, window_end))
        for start, end in intervals
        if end > window_start and start < window_end
    )
    cursor = window_start
    gaps: list[dict[str, str]] = []
    for start, end in clipped:
        if start > cursor:
            gaps.append({"start_utc": _utc_text(cursor), "end_utc": _utc_text(start)})
        cursor = max(cursor, end)
    if cursor < window_end:
        gaps.append({"start_utc": _utc_text(cursor), "end_utc": _utc_text(window_end)})
    return gaps


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeSeriesZoneBindingError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeSeriesZoneBindingError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeSeriesZoneBindingError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _clean_string(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CONTROL.search(value)
    ):
        raise EntsoeSeriesZoneBindingError(f"{label} must be a clean string")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EntsoeSeriesZoneBindingError(f"{label} must be lowercase SHA-256")


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EntsoeSeriesZoneBindingError(f"{label} must be UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeSeriesZoneBindingError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EntsoeSeriesZoneBindingError(f"{label} must be UTC")
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "BINDING_SCHEMA",
    "STATUS",
    "ZONE_REQUIREMENTS",
    "EntsoeSeriesZoneBindingError",
    "assess_series_zone_bindings",
]
