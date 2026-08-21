from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation import entsoe_lt_model_input_admission as module
from pfc_shaping.validation import entsoe_normalized_quality as d239
from pfc_shaping.validation import entsoe_series_family_mapping as d253
from pfc_shaping.validation import entsoe_series_inventory as d243
from pfc_shaping.validation import entsoe_series_semantic_binding as d272
from pfc_shaping.validation import entsoe_series_zone_binding as d255
from pfc_shaping.validation import entsoe_zone_configuration as d254
from pfc_shaping.validation.entsoe_lt_model_input_admission import (
    EntsoeLtModelInputAdmissionError,
    feature_record_id,
    validate_model_input_admission_contract,
    verify_synthetic_entsoe_lt_model_input_admission,
)
from tests.test_entsoe_cadence_package_binding import (
    cadence_sidecar_factory as _d270_cadence_sidecar_factory,
)
from tests.test_entsoe_incremental_quality import (
    quality_package_factory as _d245_quality_package_factory,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = module.CONTRACT_PATH
REFERENCE = datetime(2026, 8, 6, 2, 0, tzinfo=timezone.utc)
START = "2026-08-01T00:00:00Z"
END = "2026-08-02T00:00:00Z"
ZONES = ("CH", "DE_LU", "FR", "IT_NORTH", "AT")
GENERATION_FIELDS = (
    "solar",
    "wind",
    "nuclear",
    "hydro_run_of_river",
    "hydro_reservoir",
    "hydro_pumped_storage",
)


@pytest.fixture
def quality_package_factory():
    yield from _d245_quality_package_factory.__wrapped__()


@pytest.fixture
def cadence_sidecar_factory():
    yield from _d270_cadence_sidecar_factory.__wrapped__()


def _complete_dimension() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(
        group: str,
        field: str,
        *,
        zone: str | None = None,
        border: str | None = None,
        direction: str = "CH_TO_NEIGHBOR_ONLY",
        unit: str = "MW",
    ) -> None:
        if zone is not None:
            from_zone = to_zone = zone
            suffix = zone.casefold()
        elif border is not None:
            neighbor = border.split("-", maxsplit=1)[1]
            if direction == "NEIGHBOR_TO_CH_ONLY":
                from_zone, to_zone = neighbor, "CH"
                suffix = f"{neighbor.casefold()}_to_ch"
            else:
                from_zone, to_zone = "CH", neighbor
                suffix = f"ch_to_{neighbor.casefold()}"
        else:  # pragma: no cover - helper misuse guard
            raise AssertionError("zone or border required")
        series_id = f"s{len(rows) + 1:03d}_{group}_{field}_{suffix}"
        rows.append(
            {
                "series_id": series_id,
                "group_name": group,
                "field_name": field,
                "from_zone": from_zone,
                "to_zone": to_zone,
                "unit": unit,
                "native_resolution": "PT1H",
                "source_endpoint": f"https://example.invalid/entsoe/{group}",
                "document_id": f"SYNTHETIC/{series_id}",
            }
        )

    for zone in ZONES:
        add("actual_load", "load", zone=zone)
        add("day_ahead_prices", "price", zone=zone, unit="EUR/MWh")
        add("load_forecast", "load", zone=zone)
        add("generation_forecast", "generation", zone=zone)
    for field in GENERATION_FIELDS:
        add("generation_actual", field, zone="CH")
    for zone in ZONES[1:]:
        add("generation_actual", "generation", zone=zone)
    for family in (
        "renewables_forecast_day_ahead",
        "renewables_forecast_intraday",
    ):
        add(family, "solar", zone="CH")
        add(family, "wind", zone="CH")
        for zone in ZONES[1:]:
            add(family, "renewables", zone=zone)
    for zone in ("CH", "FR", "IT_NORTH", "AT"):
        add("hydro_reservoir_storage", "reservoir", zone=zone, unit="MWh")
    for family in sorted(d239.CROSSBORDER_GROUPS):
        for border in d239.REQUIRED_BORDERS:
            add(family, "flow", border=border)
            if family != "crossborder_physical_flows":
                add(
                    family,
                    "flow",
                    border=border,
                    direction="NEIGHBOR_TO_CH_ONLY",
                )
    return pd.DataFrame(rows, columns=d239.ROLE_FIELDS["series_dimension"])


def _complete_fixture() -> dict[str, object]:
    dimension = _complete_dimension()
    targets = pd.date_range(START, periods=24, freq="1h", tz="UTC")
    latest_rows: list[dict[str, object]] = []
    vintage_rows: list[dict[str, object]] = []
    signs: dict[str, str] = {}
    backfill: dict[str, str] = {}
    for metadata in dimension.itertuples(index=False):
        if metadata.group_name == "day_ahead_prices":
            signs[metadata.series_id] = "PRICE_CAN_BE_NEGATIVE"
        elif metadata.group_name in d239.CROSSBORDER_GROUPS:
            signs[metadata.series_id] = (
                "SIGNED_POSITIVE_CH_TO_NEIGHBOR"
                if metadata.from_zone == "CH"
                else "SIGNED_POSITIVE_NEIGHBOR_TO_CH"
            )
        else:
            signs[metadata.series_id] = "NON_NEGATIVE_PHYSICAL_QUANTITY"
        backfill[metadata.series_id] = "NATIVE_VINTAGES"
        for index, target in enumerate(targets, start=1):
            is_pre_target = metadata.group_name in d239.PRE_TARGET_GROUPS
            as_of = (
                target - pd.Timedelta(hours=1) if is_pre_target else target + pd.Timedelta(hours=1)
            )
            common = {
                "series_id": metadata.series_id,
                "target_ts_utc": target,
                "value": float(index),
                "quality_flag": "OK",
                "revision_number": 1,
                "meta_load_timestamp_utc": as_of,
            }
            latest_rows.append({**common, "is_historical": True})
            vintage_rows.append({**common, "as_of_utc": as_of})
    latest = pd.DataFrame(latest_rows)[list(d239.ROLE_FIELDS["latest_values"])]
    vintages = pd.DataFrame(vintage_rows)[list(d239.ROLE_FIELDS["vintage_values"])]
    return {
        "series_dimension": dimension,
        "latest_values": latest,
        "vintage_values": vintages,
        "sign_semantics_by_series": signs,
        "backfill_status_by_series": backfill,
    }


def _inventory(dimension: pd.DataFrame) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    count = len(dimension)
    rows: list[dict[str, object]] = []
    raw_by_series: dict[str, dict[str, object]] = {}
    for position, metadata in enumerate(dimension.itertuples(index=False), start=1):
        row = {
            "group_name": metadata.group_name,
            "field_name": metadata.field_name,
            "document_type": f"SYN{position:04d}",
            "business_type": "A04",
            "process_type": "A16",
            "psr_type": None,
            "from_zone": metadata.from_zone,
            "to_zone": metadata.to_zone,
            "unit": metadata.unit,
            "series_count": 1,
            "min_meta_load_timestamp_utc": "2026-08-01T00:00:00Z",
            "max_meta_load_timestamp_utc": "2026-08-02T00:00:00Z",
            "total_signature_count": count,
            "total_series_count": count,
        }
        rows.append(row)
        raw_by_series[str(metadata.series_id)] = row
    document = {
        "schema_version": d243.INVENTORY_SCHEMA,
        "evidence_class": "REAL_DIMENSION_INVENTORY_OWNER_ASSERTED",
        "query_sha256": d243.QUERY_SHA256,
        "statement_id": "123e4567-e89b-12d3-a456-426614174000",
        "captured_at_utc": "2026-08-06T00:00:00Z",
        "europe_zurich_capture_date": "2026-08-06",
        "daily_capture_reservation_content_id": "a" * 64,
        "catalog": "dev",
        "schema": "gold",
        "table": "dimentsoeseries",
        "columns": list(d243.ROW_FIELDS_IN_ORDER),
        "rows": rows,
        "execution": dict(d243.EXECUTION),
        "authorities": dict(d243.AUTHORITIES),
    }
    return document, raw_by_series


def _border_and_direction(metadata: object) -> tuple[str, str, str]:
    from_zone = str(metadata.from_zone)
    to_zone = str(metadata.to_zone)
    neighbor = to_zone if from_zone == "CH" else from_zone
    border = f"CH-{neighbor}"
    if from_zone == "CH":
        return border, "CH_TO_NEIGHBOR_ONLY", "SIGNED_POSITIVE_CH_TO_NEIGHBOR"
    return border, "NEIGHBOR_TO_CH_ONLY", "SIGNED_POSITIVE_NEIGHBOR_TO_CH"


def _family_mapping(
    dimension: pd.DataFrame,
    inventory: dict[str, object],
    raw_by_series: dict[str, dict[str, object]],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for metadata in dimension.itertuples(index=False):
        family = str(metadata.group_name)
        border = direction = None
        sign = (
            "PRICE_CAN_BE_NEGATIVE"
            if family == "day_ahead_prices"
            else "NON_NEGATIVE_PHYSICAL_QUANTITY"
        )
        if family in d253.DIRECTIONAL_FAMILIES:
            border, direction, sign = _border_and_direction(metadata)
            if family == "crossborder_physical_flows":
                direction = "BOTH_DIRECTIONS_IN_ONE_SIGNED_SERIES"
        entries.append(
            {
                "signature_id": d243.series_signature_id(raw_by_series[str(metadata.series_id)]),
                "disposition": "MAPPED",
                "logical_family": family,
                "logical_field": str(metadata.field_name),
                "border": border,
                "direction_coverage": direction,
                "canonical_unit": str(metadata.unit),
                "native_resolution": str(metadata.native_resolution),
                "sign_semantics": sign,
                "source_semantics_confirmed": True,
                "reason": "Synthetic complete family mapping.",
            }
        )
    return {
        "schema_version": d253.MAPPING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "inventory_content_id": d243.assess_series_inventory(inventory)["inventory_content_id"],
        "mapping_owner": "Synthetic composite owner",
        "mapped_at_utc": "2026-08-06T00:15:00Z",
        "entries": entries,
        "authorities": {
            "value_quality": False,
            "point_in_time_availability": False,
            "model_input": False,
            "production": False,
        },
    }


def _semantic_binding(
    dimension: pd.DataFrame,
    inventory: dict[str, object],
    mapping: dict[str, object],
    raw_by_series: dict[str, dict[str, object]],
) -> dict[str, object]:
    entries = []
    for series_id in dimension["series_id"]:
        raw = raw_by_series[str(series_id)]
        semantics = {field: raw[field] for field in d243.SEMANTIC_FIELDS}
        entries.append(
            {
                "series_id": str(series_id),
                "signature_id": d243.series_signature_id(semantics),
                "source_semantics": semantics,
                "reason": "Synthetic exact physical semantic binding.",
            }
        )
    return {
        "schema_version": d272.BINDING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "base_dimension_content_id": d272.dimension_content_id(dimension),
        "inventory_content_id": d243.assess_series_inventory(inventory)["inventory_content_id"],
        "family_mapping_content_id": d253.assess_series_family_mapping(inventory, mapping)[
            "mapping_content_id"
        ],
        "binding_owner": "Synthetic semantic owner",
        "bound_at_utc": "2026-08-06T00:30:00Z",
        "entries": entries,
        "authorities": dict(d272.AUTHORITIES),
    }


def _zone_entry(zone: str) -> dict[str, object]:
    return {
        "source_identifier_scheme": "OWNER_CANONICAL_LABEL",
        "source_identifier": zone,
        "logical_zone": zone,
        "domain_kind": "BIDDING_ZONE",
        "valid_from_utc": START,
        "valid_to_utc": END,
        "source_status": "ACTIVE",
        "source_document_id_kind": "GOVERNED_EIC_REGISTRY_SNAPSHOT_ID",
        "source_document_id": f"SYNTHETIC-ZONE-{zone}",
        "source_endpoint": "https://example.invalid/governed-zone-registry",
        "owner_semantics_confirmed": True,
        "reason": "Synthetic exact temporal zone identity.",
    }


def _zone_documents(
    dimension: pd.DataFrame,
    inventory: dict[str, object],
    mapping: dict[str, object],
    raw_by_series: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    registry_entries = [_zone_entry(zone) for zone in ZONES]
    registry = {
        "schema_version": d254.REGISTRY_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "registry_owner": "Synthetic zone owner",
        "registered_at_utc": "2026-08-06T00:20:00Z",
        "coverage_window_start_utc": START,
        "coverage_window_end_utc": END,
        "entries": registry_entries,
        "authorities": dict(d254.AUTHORITIES),
    }
    entry_id_by_zone = {
        str(entry["logical_zone"]): d254.canonical_sha256(entry) for entry in registry_entries
    }
    bindings = []
    for metadata in dimension.itertuples(index=False):
        if metadata.group_name not in d255.ZONE_REQUIREMENTS:
            continue
        raw = raw_by_series[str(metadata.series_id)]
        zone = str(raw["from_zone"])
        bindings.append(
            {
                "signature_id": d243.series_signature_id(raw),
                "source_zone_field": "from_zone",
                "source_identifier_scheme": "OWNER_CANONICAL_LABEL",
                "domain_kind": "BIDDING_ZONE",
                "series_valid_from_utc": START,
                "series_valid_to_utc": END,
                "expected_logical_zone": zone,
                "zone_entry_id": entry_id_by_zone[zone],
                "reason": "Synthetic complete series-zone binding.",
            }
        )
    binding = {
        "schema_version": d255.BINDING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "inventory_content_id": d243.assess_series_inventory(inventory)["inventory_content_id"],
        "family_mapping_content_id": d253.assess_series_family_mapping(inventory, mapping)[
            "mapping_content_id"
        ],
        "zone_registry_content_id": d254.assess_zone_configuration(registry)["registry_content_id"],
        "binding_owner": "Synthetic zone binding owner",
        "bound_at_utc": "2026-08-06T00:35:00Z",
        "bindings": bindings,
        "authorities": dict(d255.AUTHORITIES),
    }
    return registry, binding


def _feature_records() -> list[dict[str, object]]:
    common = {
        "source_issue_utc": None,
        "operational_horizon_end_utc": None,
        "training_cutoff_utc": None,
        "support_state": "SUPPORTED",
        "missing_value_policy": "PRESERVE_NULL",
        "monthly_effect_policy": "ZERO_MEAN_WITHIN_SOLVER_MONTH",
    }
    return [
        {
            **common,
            "feature_id": "actual_load_training",
            "source_family": "actual_load",
            "information_role": "REALIZED_ACTUAL",
            "requested_use": "TRAINING_COVARIATE",
            "target_start_utc": "2026-08-01T00:00:00Z",
            "target_end_utc": "2026-08-01T01:00:00Z",
            "origin_utc": "2026-08-01T03:00:00Z",
            "available_at_utc": "2026-08-01T02:00:00Z",
            "dependency_roles": [],
            "dependency_available_at_utc": [],
        },
        {
            **common,
            "feature_id": "load_forecast_training",
            "source_family": "load_forecast",
            "information_role": "OPERATIONAL_FORECAST",
            "requested_use": "TRAINING_COVARIATE",
            "target_start_utc": "2026-08-01T02:00:00Z",
            "target_end_utc": "2026-08-01T03:00:00Z",
            "origin_utc": "2026-08-01T04:00:00Z",
            "available_at_utc": "2026-08-01T01:05:00Z",
            "source_issue_utc": "2026-08-01T01:00:00Z",
            "operational_horizon_end_utc": "2026-08-01T04:00:00Z",
            "dependency_roles": [],
            "dependency_available_at_utc": [],
        },
        {
            **common,
            "feature_id": "flow_lagged",
            "source_family": "crossborder_physical_flows",
            "information_role": "LAGGED_REALIZED",
            "requested_use": "LAGGED_COVARIATE",
            "target_start_utc": "2026-08-01T04:00:00Z",
            "target_end_utc": "2026-08-01T05:00:00Z",
            "origin_utc": "2026-08-01T07:00:00Z",
            "available_at_utc": "2026-08-01T06:00:00Z",
            "dependency_roles": ["REALIZED_ACTUAL"],
            "dependency_available_at_utc": ["2026-08-01T06:00:00Z"],
        },
        {
            **common,
            "feature_id": "known_calendar",
            "source_family": "calendar",
            "information_role": "CALENDAR_KNOWN",
            "requested_use": "PREDICTION_COVARIATE",
            "target_start_utc": "2026-08-01T06:00:00Z",
            "target_end_utc": "2026-08-01T07:00:00Z",
            "origin_utc": "2026-08-01T05:00:00Z",
            "available_at_utc": "2020-01-01T00:00:00Z",
            "dependency_roles": [],
            "dependency_available_at_utc": [],
            "monthly_effect_policy": "NOT_APPLICABLE",
        },
    ]


def _selection(
    dimension: pd.DataFrame,
    semantic: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    series_by_family = {
        family: str(dimension.loc[dimension["group_name"].eq(family), "series_id"].iloc[0])
        for family in (
            "actual_load",
            "load_forecast",
            "crossborder_physical_flows",
        )
    }
    entries = []
    for record in records:
        role = str(record["information_role"])
        if role not in module.DIRECT_ROLES:
            continue
        family = str(record["source_family"])
        entries.append(
            {
                "feature_record_id": feature_record_id(record),
                "feature_id_sha256": hashlib.sha256(
                    str(record["feature_id"]).encode("utf-8")
                ).hexdigest(),
                "series_id": series_by_family[family],
                "source_family": family,
                "information_role": role,
                "requested_use": record["requested_use"],
                "reason": "Synthetic direct physical feature selection.",
            }
        )
    return {
        "schema_version": module.SELECTION_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "base_dimension_content_id": d272.dimension_content_id(dimension),
        "semantic_binding_content_id": d272.canonical_sha256(semantic),
        "feature_records_content_id": module.canonical_sha256(records),
        "selected_at_utc": "2026-08-06T00:45:00Z",
        "entries": entries,
        "authorities": dict(module.SELECTION_AUTHORITIES),
    }


def _documents(dimension: pd.DataFrame) -> dict[str, object]:
    inventory, raw_by_series = _inventory(dimension)
    mapping = _family_mapping(dimension, inventory, raw_by_series)
    semantic = _semantic_binding(dimension, inventory, mapping, raw_by_series)
    registry, zone_binding = _zone_documents(
        dimension,
        inventory,
        mapping,
        raw_by_series,
    )
    records = _feature_records()
    return {
        "inventory": inventory,
        "mapping": mapping,
        "semantic": semantic,
        "registry": registry,
        "zone_binding": zone_binding,
        "records": records,
        "selection": _selection(dimension, semantic, records),
    }


@pytest.fixture
def complete_composite(quality_package_factory, cadence_sidecar_factory):
    complete = _complete_fixture()

    def replace(fixture: dict[str, object]) -> None:
        fixture.clear()
        fixture.update(copy.deepcopy(complete))

    package, context, fixture = quality_package_factory(fixture_mutator=replace)
    sidecar = cadence_sidecar_factory(package, context, fixture)
    return {
        "package": package,
        "context": context,
        "sidecar": sidecar,
        "fixture": fixture,
        **_documents(fixture["series_dimension"]),
    }


def _verify(composite: dict[str, object]):
    return verify_synthetic_entsoe_lt_model_input_admission(
        contract_path=CONTRACT_PATH,
        package_manifest_path=composite["package"],
        quality_context_path=composite["context"],
        cadence_manifest_path=composite["sidecar"],
        inventory_document=composite["inventory"],
        family_mapping_document=composite["mapping"],
        semantic_binding_document=composite["semantic"],
        zone_registry_document=composite["registry"],
        zone_binding_document=composite["zone_binding"],
        feature_records=composite["records"],
        feature_selection_document=composite["selection"],
        reference_time_utc=REFERENCE,
    )


def _reselect(composite: dict[str, object]) -> None:
    composite["selection"] = _selection(
        composite["fixture"]["series_dimension"],
        composite["semantic"],
        composite["records"],
    )


def test_contract_is_exact_zero_query_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    result = validate_model_input_admission_contract(document)

    assert result["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert result["real_model_input_admission_implemented"] is False
    assert result["databricks_connection_count"] == 0
    assert result["production_authorized"] is False


def test_complete_cross_gate_composite_passes_without_real_authority(
    complete_composite,
) -> None:
    result = _verify(complete_composite)
    assessment = result.assessment

    assert assessment["status"] == module.PASS_STATUS
    assert assessment["blocker_codes"] == []
    assert assessment["all_synthetic_structural_gates_pass"] is True
    assert assessment["counts"]["physical_series"] == 82
    assert assessment["counts"]["cadence_expected_targets"] == 1_968
    assert assessment["counts"]["cadence_missing_targets"] == 0
    assert assessment["counts"]["selected_direct_features"] == 3
    assert assessment["counts"]["nonphysical_model_records"] == 1
    assert assessment["d245_incremental_quality_passed"] is True
    assert assessment["required_family_direction_unit_coverage_complete"] is True
    assert assessment["coupled_zone_coverage_complete"] is True
    assert assessment["feature_pit_structure_complete"] is True
    assert assessment["real_model_input_authorized"] is False
    assert assessment["raw_values_persisted_in_assessment"] is False
    assert assessment["clear_series_or_feature_ids_persisted_in_assessment"] is False
    assert all(value is False for value in assessment["authorities"].values())
    assert all(value == 0 for value in module.EXECUTION.values())
    assert all(
        profile["raw_ids_persisted"] is False
        and "series_id" not in profile
        and "feature_id" not in profile
        for profile in assessment["selected_feature_profiles"]
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("base_dimension_content_id", "selection dimension content ID"),
        ("semantic_binding_content_id", "selection semantic binding content ID"),
        ("feature_records_content_id", "feature records content ID"),
    ],
)
def test_selection_is_content_bound_to_dimension_semantics_and_records(
    complete_composite,
    field: str,
    message: str,
) -> None:
    complete_composite["selection"][field] = "f" * 64

    with pytest.raises(EntsoeLtModelInputAdmissionError, match=message):
        _verify(complete_composite)


def test_wrong_family_or_duplicate_direct_selection_fails_closed(
    complete_composite,
) -> None:
    dimension = complete_composite["fixture"]["series_dimension"]
    price_series = str(
        dimension.loc[dimension["group_name"].eq("day_ahead_prices"), "series_id"].iloc[0]
    )
    complete_composite["selection"]["entries"][0]["series_id"] = price_series
    with pytest.raises(EntsoeLtModelInputAdmissionError, match="series logical family"):
        _verify(complete_composite)

    complete_composite["selection"] = _selection(
        dimension,
        complete_composite["semantic"],
        complete_composite["records"],
    )
    complete_composite["selection"]["entries"].append(
        copy.deepcopy(complete_composite["selection"]["entries"][0])
    )
    with pytest.raises(EntsoeLtModelInputAdmissionError, match="selection is duplicated"):
        _verify(complete_composite)


def test_nonphysical_record_is_allowed_but_cannot_enter_physical_selection(
    complete_composite,
) -> None:
    calendar = complete_composite["records"][-1]
    dimension = complete_composite["fixture"]["series_dimension"]
    actual_series = str(
        dimension.loc[dimension["group_name"].eq("actual_load"), "series_id"].iloc[0]
    )
    complete_composite["selection"]["entries"].append(
        {
            "feature_record_id": feature_record_id(calendar),
            "feature_id_sha256": hashlib.sha256(
                str(calendar["feature_id"]).encode("utf-8")
            ).hexdigest(),
            "series_id": actual_series,
            "source_family": calendar["source_family"],
            "information_role": calendar["information_role"],
            "requested_use": calendar["requested_use"],
            "reason": "Invalid nonphysical selection roast.",
        }
    )

    with pytest.raises(
        EntsoeLtModelInputAdmissionError,
        match="nonphysical or derived feature",
    ):
        _verify(complete_composite)


def test_selected_target_must_be_inside_cadence_window(complete_composite) -> None:
    record = complete_composite["records"][0]
    record.update(
        {
            "target_start_utc": "2026-07-31T20:00:00Z",
            "target_end_utc": "2026-07-31T21:00:00Z",
            "origin_utc": "2026-07-31T23:00:00Z",
            "available_at_utc": "2026-07-31T22:00:00Z",
        }
    )
    _reselect(complete_composite)

    with pytest.raises(EntsoeLtModelInputAdmissionError, match="cadence assessment window"):
        _verify(complete_composite)


@pytest.mark.parametrize(
    ("target_start", "target_end", "message"),
    [
        (
            "2026-08-01T00:15:00Z",
            "2026-08-01T01:15:00Z",
            "absolute UTC native grid",
        ),
        (
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:30:00Z",
            "duration differs from native cadence",
        ),
    ],
)
def test_selected_interval_must_match_effective_native_cadence(
    complete_composite,
    target_start: str,
    target_end: str,
    message: str,
) -> None:
    record = complete_composite["records"][0]
    record.update(
        {
            "target_start_utc": target_start,
            "target_end_utc": target_end,
            "origin_utc": "2026-08-01T03:00:00Z",
            "available_at_utc": "2026-08-01T02:00:00Z",
        }
    )
    _reselect(complete_composite)

    with pytest.raises(EntsoeLtModelInputAdmissionError, match=message):
        _verify(complete_composite)


def test_selected_origin_cannot_follow_selection_or_reference(
    complete_composite,
) -> None:
    record = complete_composite["records"][0]
    record["origin_utc"] = "2026-08-07T00:00:00Z"
    _reselect(complete_composite)

    with pytest.raises(EntsoeLtModelInputAdmissionError, match="origin follows"):
        _verify(complete_composite)


def test_selection_cannot_predate_any_predecessor_evidence(
    complete_composite,
) -> None:
    complete_composite["selection"]["selected_at_utc"] = "2026-08-06T00:34:59Z"

    with pytest.raises(EntsoeLtModelInputAdmissionError, match="predates predecessor"):
        _verify(complete_composite)


def test_missing_direct_selection_is_a_composite_blocker(complete_composite) -> None:
    complete_composite["selection"]["entries"].pop()

    result = _verify(complete_composite)

    assert result.assessment["status"] == module.FAIL_STATUS
    assert result.assessment["blocker_codes"] == ["DIRECT_FEATURE_SELECTION_INCOMPLETE"]


def test_rejected_pit_record_blocks_even_when_not_selected(complete_composite) -> None:
    rejected = complete_composite["records"][0]
    rejected["origin_utc"] = "2026-08-01T00:30:00Z"
    _reselect(complete_composite)
    rejected_id = feature_record_id(rejected)
    complete_composite["selection"]["entries"] = [
        entry
        for entry in complete_composite["selection"]["entries"]
        if entry["feature_record_id"] != rejected_id
    ]

    result = _verify(complete_composite)

    assert result.assessment["status"] == module.FAIL_STATUS
    assert result.assessment["blocker_codes"] == ["FEATURE_PIT_STRUCTURE_REJECTED"]


def test_zone_binding_gap_survives_composition_as_a_blocker(
    complete_composite,
) -> None:
    complete_composite["zone_binding"]["bindings"].pop()

    result = _verify(complete_composite)

    assert result.assessment["status"] == module.FAIL_STATUS
    assert result.assessment["blocker_codes"] == ["COUPLED_ZONE_BINDING_INCOMPLETE"]


def test_cadence_gap_survives_composition_as_a_blocker(
    complete_composite,
    cadence_sidecar_factory,
) -> None:
    complete_composite["sidecar"] = cadence_sidecar_factory(
        complete_composite["package"],
        complete_composite["context"],
        complete_composite["fixture"],
        assessment_start="2026-07-31T23:00:00Z",
        assessment_end="2026-08-02T01:00:00Z",
    )

    result = _verify(complete_composite)

    assert result.assessment["status"] == module.FAIL_STATUS
    assert result.assessment["blocker_codes"] == ["CADENCE_PACKAGE_INCOMPLETE"]


def test_derived_forecast_error_requires_separate_lineage(complete_composite) -> None:
    derived = {
        **complete_composite["records"][0],
        "feature_id": "lagged_load_forecast_error",
        "information_role": "FORECAST_ERROR",
        "requested_use": "LAGGED_COVARIATE",
        "target_start_utc": "2026-08-01T08:00:00Z",
        "target_end_utc": "2026-08-01T09:00:00Z",
        "origin_utc": "2026-08-01T12:00:00Z",
        "available_at_utc": "2026-08-01T11:00:00Z",
        "dependency_roles": ["OPERATIONAL_FORECAST", "REALIZED_ACTUAL"],
        "dependency_available_at_utc": [
            "2026-08-01T07:00:00Z",
            "2026-08-01T11:00:00Z",
        ],
    }
    complete_composite["records"].append(derived)
    _reselect(complete_composite)

    result = _verify(complete_composite)

    assert result.assessment["status"] == module.FAIL_STATUS
    assert result.assessment["blocker_codes"] == ["DERIVED_TRANSFORM_LINEAGE_REQUIRED"]


def test_cadence_manifest_reread_cannot_change_after_d270(
    complete_composite,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = module.d270.verify_synthetic_entsoe_cadence_package

    def mutate_after_verification(**kwargs):
        result = original(**kwargs)
        path = Path(kwargs["cadence_manifest_path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        document["created_at_utc"] = "2026-08-02T03:00:01Z"
        path.write_text(json.dumps(document), encoding="utf-8")
        return result

    monkeypatch.setattr(
        module.d270,
        "verify_synthetic_entsoe_cadence_package",
        mutate_after_verification,
    )

    with pytest.raises(
        EntsoeLtModelInputAdmissionError,
        match="cadence manifest reread content ID",
    ):
        _verify(complete_composite)


def test_duplicate_json_key_in_cadence_manifest_is_rejected(
    complete_composite,
) -> None:
    path = Path(complete_composite["sidecar"])
    text = path.read_text(encoding="utf-8")
    duplicate = f'{{"schema_version":"{module.d270.MANIFEST_SCHEMA}",' + text[1:]
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(EntsoeLtModelInputAdmissionError, match="duplicate keys"):
        _verify(complete_composite)
