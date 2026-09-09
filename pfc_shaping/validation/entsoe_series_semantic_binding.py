"""Bind physical ENTSO-E series IDs to exact owner-reviewed semantics offline."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone

import pandas as pd

from pfc_shaping.validation.entsoe_normalized_quality import ROLE_FIELDS
from pfc_shaping.validation.entsoe_series_family_mapping import (
    DIRECTIONAL_FAMILIES,
    MAPPING_SCHEMA,
    assess_series_family_mapping,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    SEMANTIC_FIELDS,
    assess_series_inventory,
    canonical_sha256,
    series_signature_id,
)

BINDING_SCHEMA = "fmv_entsoe_series_semantic_binding.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_series_semantic_binding_assessment.v1"
STATUS = "PASS_SERIES_TO_SIGNATURE_STRUCTURE_NO_VALUE_OR_MODEL_AUTHORITY"
EVIDENCE_CLASSES = frozenset({"SYNTHETIC_TEST_ONLY", "REAL_OWNER_ASSERTED"})
AUTHORITIES = {
    "external_binding_owner_identity_verified": False,
    "real_series_inventory_coverage_proven": False,
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "candidate_assembly_authorized": False,
    "production_authorized": False,
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_OWNER = re.compile(r"^[A-Za-z0-9_.@ -]{3,128}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class EntsoeSeriesSemanticBindingError(ValueError):
    """Raised when a physical-series semantic join is incomplete or ambiguous."""


def dimension_content_id(series_dimension: pd.DataFrame) -> str:
    """Return a deterministic identity for the exact normalized dimension rows."""

    dimension = _validate_dimension(series_dimension)
    return canonical_sha256(
        {
            "columns": list(ROLE_FIELDS["series_dimension"]),
            "rows": [
                [str(row[column]) for column in ROLE_FIELDS["series_dimension"]]
                for row in dimension.to_dict(orient="records")
            ],
        }
    )


def assess_series_semantic_binding(
    series_dimension: pd.DataFrame,
    inventory: Mapping[str, object],
    family_mapping: Mapping[str, object],
    binding_document: Mapping[str, object],
) -> dict[str, object]:
    """Prove the one-to-one series-ID join without granting data authority."""

    dimension = _validate_dimension(series_dimension)
    inventory_assessment = assess_series_inventory(inventory)
    mapping_assessment = assess_series_family_mapping(inventory, family_mapping)
    document = _mapping(binding_document, "series semantic binding")
    _exact(
        document,
        {
            "schema_version",
            "evidence_class",
            "base_dimension_content_id",
            "inventory_content_id",
            "family_mapping_content_id",
            "binding_owner",
            "bound_at_utc",
            "entries",
            "authorities",
        },
        "series semantic binding",
    )
    _equal(document["schema_version"], BINDING_SCHEMA, "binding schema")
    evidence_class = document["evidence_class"]
    if evidence_class not in EVIDENCE_CLASSES:
        raise EntsoeSeriesSemanticBindingError("binding evidence class is invalid")
    _equal(
        document["base_dimension_content_id"],
        dimension_content_id(dimension),
        "base dimension content ID",
    )
    _equal(
        document["inventory_content_id"],
        inventory_assessment["inventory_content_id"],
        "inventory content ID",
    )
    _equal(
        document["family_mapping_content_id"],
        mapping_assessment["mapping_content_id"],
        "family mapping content ID",
    )
    _equal(document["authorities"], AUTHORITIES, "binding authorities")
    owner = document["binding_owner"]
    _clean(owner, "binding owner")
    if not isinstance(owner, str) or _OWNER.fullmatch(owner) is None:
        raise EntsoeSeriesSemanticBindingError("binding owner format is invalid")
    if evidence_class == "REAL_OWNER_ASSERTED" and owner.casefold().startswith(
        "synthetic"
    ):
        raise EntsoeSeriesSemanticBindingError("real binding owner is synthetic")
    mapped_at = _utc(family_mapping["mapped_at_utc"], "family mapped_at_utc")
    bound_at = _utc(document["bound_at_utc"], "bound_at_utc")
    if bound_at < mapped_at:
        raise EntsoeSeriesSemanticBindingError("binding predates family mapping")
    mapping_evidence = mapping_assessment["evidence_class"]
    if evidence_class == "SYNTHETIC_TEST_ONLY":
        if mapping_evidence != "SYNTHETIC_TEST_ONLY":
            raise EntsoeSeriesSemanticBindingError(
                "synthetic binding requires synthetic family mapping"
            )
    elif mapping_evidence != "REAL_OWNER_REVIEWED":
        raise EntsoeSeriesSemanticBindingError(
            "real binding requires owner-reviewed family mapping"
        )

    inventory_by_signature = _inventory_by_signature(inventory)
    mapping_by_signature = _mapping_by_signature(family_mapping)
    dimension_by_series = dimension.set_index("series_id").to_dict(orient="index")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise EntsoeSeriesSemanticBindingError("binding entries must be non-empty")
    if len(raw_entries) > 100_000:
        raise EntsoeSeriesSemanticBindingError("binding entry limit exceeded")

    observed_series: set[str] = set()
    observed_signature_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for position, raw_entry in enumerate(raw_entries, start=1):
        entry = _mapping(raw_entry, f"binding entry {position}")
        _exact(
            entry,
            {"series_id", "signature_id", "source_semantics", "reason"},
            f"binding entry {position}",
        )
        series_id = entry["series_id"]
        _clean(series_id, f"binding entry {position} series ID")
        if not isinstance(series_id, str):
            raise EntsoeSeriesSemanticBindingError("series ID must be a string")
        normalized_series = series_id.casefold()
        if normalized_series in observed_series:
            raise EntsoeSeriesSemanticBindingError("binding series ID is duplicated")
        observed_series.add(normalized_series)
        if series_id not in dimension_by_series:
            raise EntsoeSeriesSemanticBindingError(
                "binding references an unknown base-dimension series ID"
            )
        signature_id = entry["signature_id"]
        _sha(signature_id, f"binding entry {position} signature ID")
        semantics = _mapping(entry["source_semantics"], "source semantics")
        _exact(semantics, set(SEMANTIC_FIELDS), "source semantics")
        computed_signature = series_signature_id(semantics)
        _equal(signature_id, computed_signature, "computed signature ID")
        if signature_id not in inventory_by_signature:
            raise EntsoeSeriesSemanticBindingError(
                "binding signature is absent from the bound inventory"
            )
        logical = mapping_by_signature.get(str(signature_id))
        if logical is None or logical["disposition"] != "MAPPED":
            raise EntsoeSeriesSemanticBindingError(
                "binding signature is not owner-mapped"
            )
        _clean(entry["reason"], f"binding entry {position} reason")
        _validate_logical_dimension_row(
            dimension_by_series[series_id],
            semantics,
            logical,
        )
        observed_signature_counts[str(signature_id)] += 1
        family_counts[str(logical["logical_family"])] += 1

    expected_series = {str(value).casefold() for value in dimension["series_id"]}
    if observed_series != expected_series:
        raise EntsoeSeriesSemanticBindingError(
            "base-dimension series coverage differs from semantic bindings"
        )
    expected_signature_counts = {
        signature_id: int(row["series_count"])
        for signature_id, row in inventory_by_signature.items()
        if mapping_by_signature.get(signature_id, {}).get("disposition") == "MAPPED"
    }
    if dict(observed_signature_counts) != expected_signature_counts:
        raise EntsoeSeriesSemanticBindingError(
            "per-signature physical series cardinality differs from inventory"
        )

    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": STATUS,
        "binding_content_id": canonical_sha256(document),
        "base_dimension_content_id": dimension_content_id(dimension),
        "inventory_content_id": inventory_assessment["inventory_content_id"],
        "family_mapping_content_id": mapping_assessment["mapping_content_id"],
        "evidence_class": evidence_class,
        "physical_series_count": len(observed_series),
        "mapped_signature_count": len(observed_signature_counts),
        "physical_series_count_by_logical_family": dict(sorted(family_counts.items())),
        "all_base_series_bound_once": True,
        "all_mapped_signature_cardinalities_match": True,
        "external_binding_owner_identity_verified": False,
        "real_series_inventory_coverage_proven": False,
        "series_semantic_binding_verified": False,
        "temporal_zone_binding_proven": False,
        "effective_cadence_package_binding_proven": False,
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "production_authorized": False,
        "validator_databricks_request_count": 0,
        "validator_warehouse_start_count": 0,
        "validator_remote_write_count": 0,
        "real_value_rows_opened": 0,
    }


def _validate_dimension(frame: pd.DataFrame) -> pd.DataFrame:
    expected = list(ROLE_FIELDS["series_dimension"])
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise EntsoeSeriesSemanticBindingError(
            "series dimension must be a non-empty DataFrame"
        )
    if list(frame.columns) != expected or frame.columns.has_duplicates:
        raise EntsoeSeriesSemanticBindingError("series dimension fields differ")
    if bool(frame.isna().any().any()):
        raise EntsoeSeriesSemanticBindingError("series dimension contains nulls")
    copy = frame.copy(deep=True)
    for column in expected:
        for value in copy[column]:
            _clean(value, f"series dimension {column}")
    normalized = copy["series_id"].str.casefold()
    if bool(normalized.duplicated(keep=False).any()):
        raise EntsoeSeriesSemanticBindingError(
            "series dimension IDs are duplicated after casefold"
        )
    return copy


def _inventory_by_signature(
    inventory: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    rows = inventory["rows"]
    if not isinstance(rows, list):
        raise EntsoeSeriesSemanticBindingError("inventory rows must be a list")
    return {
        series_signature_id(_mapping(row, "inventory row")): _mapping(
            row, "inventory row"
        )
        for row in rows
    }


def _mapping_by_signature(
    family_mapping: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    _equal(family_mapping["schema_version"], MAPPING_SCHEMA, "mapping schema")
    entries = family_mapping["entries"]
    if not isinstance(entries, list):
        raise EntsoeSeriesSemanticBindingError("family mapping entries must be a list")
    return {
        str(entry["signature_id"]): _mapping(entry, "family mapping entry")
        for entry in entries
        if isinstance(entry, Mapping)
    }


def _validate_logical_dimension_row(
    dimension_row: Mapping[str, object],
    source_semantics: Mapping[str, object],
    logical: Mapping[str, object],
) -> None:
    family = logical["logical_family"]
    _equal(dimension_row["group_name"], family, "normalized logical family")
    expected_field = logical["logical_field"] or source_semantics["field_name"]
    if expected_field is None:
        raise EntsoeSeriesSemanticBindingError(
            "mapped series has no normalized field identity"
        )
    _equal(dimension_row["field_name"], expected_field, "normalized logical field")
    _equal(dimension_row["unit"], logical["canonical_unit"], "normalized unit")
    _equal(
        dimension_row["native_resolution"],
        logical["native_resolution"],
        "declared current resolution",
    )
    if family not in DIRECTIONAL_FAMILIES:
        return
    border = str(logical["border"])
    neighbor = border.split("-", maxsplit=1)[1]
    direction = logical["direction_coverage"]
    if direction == "CH_TO_NEIGHBOR_ONLY":
        expected = ("CH", neighbor)
    elif direction == "NEIGHBOR_TO_CH_ONLY":
        expected = (neighbor, "CH")
    elif logical["sign_semantics"] == "SIGNED_POSITIVE_CH_TO_NEIGHBOR":
        expected = ("CH", neighbor)
    else:
        expected = (neighbor, "CH")
    _equal(
        (dimension_row["from_zone"], dimension_row["to_zone"]),
        expected,
        "normalized border direction",
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeSeriesSemanticBindingError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeSeriesSemanticBindingError(f"{label} fields differ")


def _clean(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CONTROL.search(value)
    ):
        raise EntsoeSeriesSemanticBindingError(f"{label} must be a clean string")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EntsoeSeriesSemanticBindingError(f"{label} must be lowercase SHA-256")


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EntsoeSeriesSemanticBindingError(f"{label} must be UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeSeriesSemanticBindingError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EntsoeSeriesSemanticBindingError(f"{label} must be UTC")
    return parsed


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeSeriesSemanticBindingError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "BINDING_SCHEMA",
    "STATUS",
    "EntsoeSeriesSemanticBindingError",
    "assess_series_semantic_binding",
    "dimension_content_id",
]
