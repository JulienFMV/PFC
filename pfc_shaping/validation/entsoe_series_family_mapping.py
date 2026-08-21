"""Map exact ENTSO-E dimension signatures to PFC data families offline."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone

from pfc_shaping.validation.entsoe_normalized_quality import (
    CROSSBORDER_GROUPS,
    REQUIRED_BORDERS,
    REQUIRED_GENERATION_FIELDS,
    REQUIRED_GROUPS,
    SIGNED_SEMANTICS,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    ADDITIONAL_HIGH_VALUE_GROUPS,
    assess_series_inventory,
    canonical_sha256,
    series_signature_id,
)

MAPPING_SCHEMA = "fmv_entsoe_series_family_mapping.v2"
ASSESSMENT_SCHEMA = "fmv_entsoe_series_family_mapping_assessment.v2"
STATUS = (
    "PASS_EXACT_MAPPING_STRUCTURE_ASSESSED_OWNER_IDENTITY_UNVERIFIED_"
    "NO_VALUE_OR_MODEL_AUTHORITY"
)
EVIDENCE_CLASSES = frozenset({"REAL_OWNER_REVIEWED", "SYNTHETIC_TEST_ONLY"})
DISPOSITIONS = frozenset({"MAPPED", "AMBIGUOUS", "OUT_OF_SCOPE"})
EXPLORATORY_HIGH_VALUE_GROUPS = (
    "actual_generation_per_unit",
    "available_generation_capacity",
    "generation_curtailment",
    "load_forecast_margin",
    "flow_based_capacity_parameters",
    "allocated_crosszonal_balancing_capacity",
)
LOGICAL_FAMILIES = frozenset(
    {
        *REQUIRED_GROUPS,
        *ADDITIONAL_HIGH_VALUE_GROUPS,
        *EXPLORATORY_HIGH_VALUE_GROUPS,
    }
)
RESOLUTIONS = frozenset({"PT15M", "PT30M", "PT60M", "PT1H", "P1D", "P7D"})
CANONICAL_UNITS = frozenset(
    {"MW", "MWh", "EUR/MWh", "EUR/MW", "PERCENT", "SOURCE_UNIT_DOCUMENTED"}
)
NON_CROSSBORDER_SIGNS = frozenset(
    {"NON_NEGATIVE_PHYSICAL_QUANTITY", "PRICE_CAN_BE_NEGATIVE"}
)
RENEWABLE_FIELDS = frozenset({"solar", "wind"})
TECHNOLOGY_FAMILIES = frozenset(
    {
        "generation_actual",
        "generation_forecast",
        "renewables_forecast_day_ahead",
        "renewables_forecast_intraday",
        "installed_generation_capacity",
        "available_generation_capacity",
        "actual_generation_per_unit",
        "generation_curtailment",
    }
)
PRICE_FAMILIES = frozenset(
    {
        "day_ahead_prices",
        "balancing_energy_prices",
        "imbalance_price",
        "reserve_capacity_prices",
    }
)
DIRECTIONAL_FAMILIES = frozenset(
    {*CROSSBORDER_GROUPS, "allocated_crosszonal_balancing_capacity"}
)
DIRECTION_COVERAGES = frozenset(
    {
        "CH_TO_NEIGHBOR_ONLY",
        "NEIGHBOR_TO_CH_ONLY",
        "BOTH_DIRECTIONS_IN_ONE_SIGNED_SERIES",
    }
)
REQUIRED_DIRECTION_COVERAGES = frozenset(
    {"CH_TO_NEIGHBOR_ONLY", "NEIGHBOR_TO_CH_ONLY"}
)
SINGLE_SIGNED_BIDIRECTIONAL_FAMILIES = frozenset(
    {"crossborder_physical_flows"}
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_OWNER = re.compile(r"^[A-Za-z0-9_.@ -]{3,128}$")


class EntsoeSeriesFamilyMappingError(ValueError):
    """Raised when an owner mapping is ambiguous or inconsistent."""


def assess_series_family_mapping(
    inventory: Mapping[str, object],
    mapping_document: Mapping[str, object],
) -> dict[str, object]:
    """Validate an exact owner mapping and report value-free family coverage."""

    inventory_assessment = assess_series_inventory(inventory)
    rows = inventory["rows"]
    if not isinstance(rows, list):  # already checked; keeps type narrowing explicit
        raise EntsoeSeriesFamilyMappingError("inventory rows must be a list")
    inventory_by_id: dict[str, Mapping[str, object]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise EntsoeSeriesFamilyMappingError("inventory row must be an object")
        signature_id = series_signature_id(raw_row)
        if signature_id in inventory_by_id:
            raise EntsoeSeriesFamilyMappingError("inventory signature ID is duplicated")
        inventory_by_id[signature_id] = raw_row

    document = _mapping(mapping_document, "mapping document")
    _exact(
        document,
        {
            "schema_version",
            "evidence_class",
            "inventory_content_id",
            "mapping_owner",
            "mapped_at_utc",
            "entries",
            "authorities",
        },
        "mapping document",
    )
    _equal(document["schema_version"], MAPPING_SCHEMA, "mapping schema")
    evidence_class = document["evidence_class"]
    if evidence_class not in EVIDENCE_CLASSES:
        raise EntsoeSeriesFamilyMappingError("mapping evidence class is invalid")
    _equal(
        document["inventory_content_id"],
        inventory_assessment["inventory_content_id"],
        "inventory content ID",
    )
    owner = document["mapping_owner"]
    _clean_string(owner, "mapping owner")
    if not isinstance(owner, str) or _OWNER.fullmatch(owner) is None:
        raise EntsoeSeriesFamilyMappingError("mapping owner format is invalid")
    if evidence_class == "REAL_OWNER_REVIEWED" and owner.casefold().startswith(
        "synthetic"
    ):
        raise EntsoeSeriesFamilyMappingError("real mapping owner is synthetic")
    mapped_at = _utc(document["mapped_at_utc"], "mapped_at_utc")
    captured_at = _utc(inventory["captured_at_utc"], "inventory captured_at_utc")
    if mapped_at < captured_at:
        raise EntsoeSeriesFamilyMappingError(
            "mapped_at_utc precedes the bound inventory capture"
        )
    _validate_false_authorities(document["authorities"])

    entries = document["entries"]
    if not isinstance(entries, list) or not entries:
        raise EntsoeSeriesFamilyMappingError("mapping entries must be a non-empty list")

    seen: set[str] = set()
    disposition_counts: Counter[str] = Counter()
    family_signature_counts: Counter[str] = Counter()
    family_series_counts: Counter[str] = Counter()
    fields_by_family: dict[str, set[str]] = {}
    borders_by_family: dict[str, set[str]] = {}
    directions_by_family_border: dict[str, dict[str, set[str]]] = {}
    price_unit_mismatches: list[str] = []

    for position, raw_entry in enumerate(entries, start=1):
        entry = _mapping(raw_entry, f"mapping entry {position}")
        _exact(
            entry,
            {
                "signature_id",
                "disposition",
                "logical_family",
                "logical_field",
                "border",
                "direction_coverage",
                "canonical_unit",
                "native_resolution",
                "sign_semantics",
                "source_semantics_confirmed",
                "reason",
            },
            f"mapping entry {position}",
        )
        signature_id = entry["signature_id"]
        _sha(signature_id, f"entry {position} signature ID")
        if signature_id in seen:
            raise EntsoeSeriesFamilyMappingError("mapping signature ID is duplicated")
        seen.add(signature_id)
        if signature_id not in inventory_by_id:
            raise EntsoeSeriesFamilyMappingError("mapping references an unknown signature ID")
        disposition = entry["disposition"]
        if disposition not in DISPOSITIONS:
            raise EntsoeSeriesFamilyMappingError("mapping disposition is invalid")
        disposition_counts[str(disposition)] += 1
        _clean_string(entry["reason"], f"entry {position} reason")

        if disposition != "MAPPED":
            for field in (
                "logical_family",
                "logical_field",
                "border",
                "direction_coverage",
                "canonical_unit",
                "native_resolution",
                "sign_semantics",
            ):
                if entry[field] is not None:
                    raise EntsoeSeriesFamilyMappingError(
                        f"{disposition} entry must leave normalized fields null"
                    )
            if entry["source_semantics_confirmed"] is not False:
                raise EntsoeSeriesFamilyMappingError(
                    f"{disposition} entry cannot confirm source semantics"
                )
            continue

        family = entry["logical_family"]
        if family not in LOGICAL_FAMILIES:
            raise EntsoeSeriesFamilyMappingError("logical family is invalid")
        logical_field = entry["logical_field"]
        if logical_field is not None:
            _identifier(logical_field, f"entry {position} logical field")
        if family in TECHNOLOGY_FAMILIES and logical_field is None:
            raise EntsoeSeriesFamilyMappingError(
                f"technology family {family} requires a logical field"
            )
        unit = entry["canonical_unit"]
        if unit not in CANONICAL_UNITS:
            raise EntsoeSeriesFamilyMappingError("canonical unit is invalid")
        if family == "day_ahead_prices" and unit != "EUR/MWh":
            price_unit_mismatches.append(str(signature_id))
        resolution = entry["native_resolution"]
        if resolution not in RESOLUTIONS:
            raise EntsoeSeriesFamilyMappingError("native resolution is invalid")
        if entry["source_semantics_confirmed"] is not True:
            raise EntsoeSeriesFamilyMappingError(
                "mapped entry requires owner-confirmed source semantics"
            )

        border = entry["border"]
        direction = entry["direction_coverage"]
        sign = entry["sign_semantics"]
        if family in DIRECTIONAL_FAMILIES:
            if border not in REQUIRED_BORDERS:
                raise EntsoeSeriesFamilyMappingError(
                    f"cross-border family {family} requires a canonical CH border"
                )
            if sign not in SIGNED_SEMANTICS:
                raise EntsoeSeriesFamilyMappingError(
                    f"cross-border family {family} requires documented sign semantics"
                )
            if direction not in DIRECTION_COVERAGES:
                raise EntsoeSeriesFamilyMappingError(
                    f"cross-border family {family} requires direction coverage"
                )
            if (
                direction == "CH_TO_NEIGHBOR_ONLY"
                and sign != "SIGNED_POSITIVE_CH_TO_NEIGHBOR"
            ):
                raise EntsoeSeriesFamilyMappingError(
                    "CH-to-neighbor direction and sign semantics differ"
                )
            if (
                direction == "NEIGHBOR_TO_CH_ONLY"
                and sign != "SIGNED_POSITIVE_NEIGHBOR_TO_CH"
            ):
                raise EntsoeSeriesFamilyMappingError(
                    "neighbor-to-CH direction and sign semantics differ"
                )
            if direction == "BOTH_DIRECTIONS_IN_ONE_SIGNED_SERIES":
                if family not in SINGLE_SIGNED_BIDIRECTIONAL_FAMILIES:
                    raise EntsoeSeriesFamilyMappingError(
                        f"family {family} cannot claim both directions in one series"
                    )
                if sign not in {
                    "SIGNED_POSITIVE_CH_TO_NEIGHBOR",
                    "SIGNED_POSITIVE_NEIGHBOR_TO_CH",
                }:
                    raise EntsoeSeriesFamilyMappingError(
                        "bidirectional series requires an explicit positive direction"
                    )
            borders_by_family.setdefault(str(family), set()).add(str(border))
            normalized_directions = (
                set(REQUIRED_DIRECTION_COVERAGES)
                if direction == "BOTH_DIRECTIONS_IN_ONE_SIGNED_SERIES"
                else {str(direction)}
            )
            directions_by_family_border.setdefault(str(family), {}).setdefault(
                str(border), set()
            ).update(normalized_directions)
        else:
            if border is not None:
                raise EntsoeSeriesFamilyMappingError(
                    "non-cross-border family cannot declare a border"
                )
            if direction is not None:
                raise EntsoeSeriesFamilyMappingError(
                    "non-cross-border family cannot declare direction coverage"
                )
            expected_sign = (
                "PRICE_CAN_BE_NEGATIVE"
                if family in PRICE_FAMILIES
                else "NON_NEGATIVE_PHYSICAL_QUANTITY"
            )
            if sign != expected_sign or sign not in NON_CROSSBORDER_SIGNS:
                raise EntsoeSeriesFamilyMappingError(
                    f"sign semantics differ for family {family}"
                )

        raw_series_count = inventory_by_id[str(signature_id)]["series_count"]
        if isinstance(raw_series_count, bool) or not isinstance(raw_series_count, int):
            raise EntsoeSeriesFamilyMappingError("inventory series count is invalid")
        family_signature_counts[str(family)] += 1
        family_series_counts[str(family)] += raw_series_count
        if logical_field is not None:
            fields_by_family.setdefault(str(family), set()).add(str(logical_field))

    unmapped_ids = sorted(set(inventory_by_id).difference(seen))
    missing_required_groups = sorted(set(REQUIRED_GROUPS).difference(family_series_counts))
    missing_generation_actual_fields = sorted(
        REQUIRED_GENERATION_FIELDS.difference(fields_by_family.get("generation_actual", set()))
    )
    missing_renewable_fields = {
        family: sorted(RENEWABLE_FIELDS.difference(fields_by_family.get(family, set())))
        for family in (
            "renewables_forecast_day_ahead",
            "renewables_forecast_intraday",
        )
    }
    missing_borders = {
        family: sorted(set(REQUIRED_BORDERS).difference(borders_by_family.get(family, set())))
        for family in sorted(CROSSBORDER_GROUPS)
    }
    missing_directions = {
        family: {
            border: sorted(
                REQUIRED_DIRECTION_COVERAGES.difference(
                    directions_by_family_border.get(family, {}).get(border, set())
                )
            )
            for border in REQUIRED_BORDERS
        }
        for family in sorted(CROSSBORDER_GROUPS)
    }
    all_required_coverage_mapped = not any(
        (
            unmapped_ids,
            disposition_counts["AMBIGUOUS"],
            missing_required_groups,
            missing_generation_actual_fields,
            *(missing_renewable_fields.values()),
            *(missing_borders.values()),
            *(
                missing
                for family_missing in missing_directions.values()
                for missing in family_missing.values()
            ),
            price_unit_mismatches,
        )
    )
    real_owner_reviewed = evidence_class == "REAL_OWNER_REVIEWED"
    family_mapping_verified = (
        real_owner_reviewed
        and not unmapped_ids
        and disposition_counts["AMBIGUOUS"] == 0
    )
    owner_asserted_required_coverage_complete = (
        family_mapping_verified and all_required_coverage_mapped
    )
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": STATUS,
        "mapping_content_id": canonical_sha256(document),
        "inventory_content_id": inventory_assessment["inventory_content_id"],
        "evidence_class": evidence_class,
        "signature_count": len(inventory_by_id),
        "mapped_signature_count": disposition_counts["MAPPED"],
        "ambiguous_signature_count": disposition_counts["AMBIGUOUS"],
        "out_of_scope_signature_count": disposition_counts["OUT_OF_SCOPE"],
        "unmapped_signature_count": len(unmapped_ids),
        "unmapped_signature_ids": unmapped_ids,
        "mapped_signature_count_by_family": dict(sorted(family_signature_counts.items())),
        "mapped_series_count_by_family": dict(sorted(family_series_counts.items())),
        "mapped_fields_by_family": {
            family: sorted(fields) for family, fields in sorted(fields_by_family.items())
        },
        "missing_required_groups": missing_required_groups,
        "missing_generation_actual_fields": missing_generation_actual_fields,
        "missing_renewable_fields_by_family": missing_renewable_fields,
        "missing_borders_by_crossborder_family": missing_borders,
        "missing_directions_by_crossborder_family_and_border": missing_directions,
        "day_ahead_price_unit_mismatch_signature_ids": sorted(price_unit_mismatches),
        "all_required_coverage_mapped": all_required_coverage_mapped,
        "real_owner_reviewed": real_owner_reviewed,
        "owner_asserted_family_mapping_complete": family_mapping_verified,
        "owner_asserted_required_coverage_complete": (
            owner_asserted_required_coverage_complete
        ),
        "external_mapping_owner_identity_verified": False,
        "family_mapping_verified": False,
        "value_quality_proven": False,
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "production_authorized": False,
        "validator_databricks_request_count": 0,
        "validator_warehouse_start_count": 0,
        "validator_remote_write_count": 0,
    }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeSeriesFamilyMappingError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeSeriesFamilyMappingError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeSeriesFamilyMappingError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _clean_string(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CONTROL.search(value)
    ):
        raise EntsoeSeriesFamilyMappingError(f"{label} must be a clean string")


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise EntsoeSeriesFamilyMappingError(f"{label} must be a normalized identifier")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EntsoeSeriesFamilyMappingError(f"{label} must be lowercase SHA-256")


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EntsoeSeriesFamilyMappingError(f"{label} must be UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeSeriesFamilyMappingError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EntsoeSeriesFamilyMappingError(f"{label} must be UTC")
    return parsed


def _validate_false_authorities(value: object) -> None:
    authorities = _mapping(value, "mapping authorities")
    expected = {
        "value_quality",
        "point_in_time_availability",
        "model_input",
        "production",
    }
    if set(authorities) != expected or any(item is not False for item in authorities.values()):
        raise EntsoeSeriesFamilyMappingError(
            "mapping authorities must be exact and false"
        )


__all__ = [
    "ASSESSMENT_SCHEMA",
    "EXPLORATORY_HIGH_VALUE_GROUPS",
    "MAPPING_SCHEMA",
    "STATUS",
    "EntsoeSeriesFamilyMappingError",
    "assess_series_family_mapping",
]
