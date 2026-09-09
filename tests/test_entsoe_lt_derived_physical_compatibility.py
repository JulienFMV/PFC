from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pfc_shaping.validation import entsoe_lt_derived_feature_lineage as d281
from pfc_shaping.validation import entsoe_lt_derived_physical_compatibility as module
from pfc_shaping.validation import entsoe_lt_model_input_admission as d280
from pfc_shaping.validation.entsoe_lt_derived_physical_compatibility import (
    EntsoeLtDerivedPhysicalCompatibilityError,
    validate_derived_physical_compatibility_contract,
    validate_derived_physical_compatibility_contract_path,
    verify_synthetic_entsoe_lt_derived_physical_compatibility,
)
from tests.test_entsoe_lt_model_input_admission import (
    _selection as _d280_selection,
)
from tests.test_entsoe_lt_model_input_admission import (
    cadence_sidecar_factory as _d280_cadence_sidecar_factory,
)
from tests.test_entsoe_lt_model_input_admission import (
    complete_composite as _d280_complete_composite,
)
from tests.test_entsoe_lt_model_input_admission import (
    quality_package_factory as _d280_quality_package_factory,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = module.CONTRACT_PATH
REFERENCE = datetime(2026, 8, 6, 2, 0, tzinfo=timezone.utc)


@pytest.fixture
def quality_package_factory():
    yield from _d280_quality_package_factory.__wrapped__()


@pytest.fixture
def cadence_sidecar_factory():
    yield from _d280_cadence_sidecar_factory.__wrapped__()


def _add_forecast_error(composite: dict[str, object]) -> None:
    records = composite["records"]
    actual = records[0]
    forecast = records[1]
    forecast.update(
        {
            "target_start_utc": actual["target_start_utc"],
            "target_end_utc": actual["target_end_utc"],
            "origin_utc": actual["origin_utc"],
            "available_at_utc": "2026-07-31T23:30:00Z",
            "source_issue_utc": "2026-07-31T23:00:00Z",
            "operational_horizon_end_utc": actual["target_end_utc"],
        }
    )
    output = {
        "feature_id": "load_forecast_error_lagged_training",
        "source_family": "load_forecast_error",
        "information_role": "FORECAST_ERROR",
        "requested_use": "LAGGED_COVARIATE",
        "target_start_utc": actual["target_start_utc"],
        "target_end_utc": actual["target_end_utc"],
        "origin_utc": actual["origin_utc"],
        "available_at_utc": actual["available_at_utc"],
        "source_issue_utc": None,
        "operational_horizon_end_utc": None,
        "training_cutoff_utc": None,
        "dependency_roles": ["OPERATIONAL_FORECAST", "REALIZED_ACTUAL"],
        "dependency_available_at_utc": [
            forecast["available_at_utc"],
            actual["available_at_utc"],
        ],
        "support_state": "SUPPORTED",
        "missing_value_policy": "PRESERVE_NULL",
        "monthly_effect_policy": "ZERO_MEAN_WITHIN_SOLVER_MONTH",
    }
    records.append(output)
    _rebind_selection(composite)
    composite["lineage"] = _lineage(records, actual, forecast, output)


def _lineage(
    records: list[dict[str, object]],
    actual: dict[str, object],
    forecast: dict[str, object],
    output: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": d281.LINEAGE_SCHEMA,
        "evidence_class": d281.EVIDENCE_CLASS,
        "feature_availability_contract_content_id": (
            "c7826b4ad2fa5cdb6baff5d077f00ee5fd8d98108cebef9920c147d787df2ab0"
        ),
        "feature_records_content_id": d281.canonical_sha256(records),
        "declared_at_utc": "2026-08-06T01:00:00Z",
        "transforms": [
            {
                "transform_id": "load_forecast_error_lagged_training",
                "transform_kind": d281.TRANSFORM_KIND,
                "output_feature_record_id": d281.feature_record_id(output),
                "input_feature_record_ids_in_formula_order": [
                    d281.feature_record_id(actual),
                    d281.feature_record_id(forecast),
                ],
                "formula": d281.FORMULA,
                "calculation_created_at_utc": actual["available_at_utc"],
                "missing_value_policy": d281.MISSING_POLICY,
                "output_sign_convention": d281.OUTPUT_SIGN,
                "semantic_unit_zone_and_cadence_binding": d281.PHYSICAL_BINDING_STATUS,
                "reason": "Synthetic exact D282 physical compatibility roast.",
            }
        ],
        "authorities": dict(d281.LINEAGE_AUTHORITIES),
    }


def _rebind_selection(composite: dict[str, object]) -> None:
    composite["selection"] = _d280_selection(
        composite["fixture"]["series_dimension"],
        composite["semantic"],
        composite["records"],
    )


@pytest.fixture
def compatible_composite(quality_package_factory, cadence_sidecar_factory):
    composite = _d280_complete_composite.__wrapped__(
        quality_package_factory,
        cadence_sidecar_factory,
    )
    _add_forecast_error(composite)
    return composite


def _verify(composite: dict[str, object]) -> dict[str, object]:
    return verify_synthetic_entsoe_lt_derived_physical_compatibility(
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
        lineage_document=composite["lineage"],
        reference_time_utc=REFERENCE,
    )


def _predecessors(composite: dict[str, object]):
    d280_result = d280.verify_synthetic_entsoe_lt_model_input_admission(
        contract_path=d280.CONTRACT_PATH,
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
    d281_result = d281.assess_derived_feature_lineage(
        composite["records"],
        composite["lineage"],
        reference_time_utc=REFERENCE,
    )
    tracked = module._tracked_content_ids(
        composite["records"],
        composite["selection"],
        composite["mapping"],
        composite["semantic"],
        composite["registry"],
        composite["zone_binding"],
        composite["lineage"],
    )
    return d280_result, d281_result, tracked


def test_contract_is_exact_zero_query_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    direct = validate_derived_physical_compatibility_contract(document)
    stable = validate_derived_physical_compatibility_contract_path(CONTRACT_PATH)

    assert direct == stable
    assert direct["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert direct["real_or_feature_values_opened"] is False
    assert direct["derived_model_input_authorized"] is False
    assert all(value == 0 for value in module.EXECUTION.values())
    assert all(value is False for value in module.AUTHORITIES.values())


def test_complete_d280_d281_composite_proves_physical_compatibility_only(
    compatible_composite,
) -> None:
    result = _verify(compatible_composite)

    assert result["status"] == module.PASS_STATUS
    assert result["transform_count"] == 1
    assert result["all_primitives_selected_once"] is True
    assert result["all_temporal_zone_bindings_complete"] is True
    assert result["all_logical_zones_match"] is True
    assert result["all_canonical_units_are_mw"] is True
    assert result["all_effective_native_cadences_match"] is True
    assert result["d280_lineage_blocker_structurally_discharged"] is True
    assert result["derived_model_input_authorized"] is False
    assert result["real_or_feature_values_opened"] is False
    assert result["arithmetic_recomputed"] is False
    assert all(value is False for value in result["authorities"].values())
    assert all(value == 0 for value in module.EXECUTION.values())
    profile = result["profiles"][0]
    assert profile["canonical_unit"] == "MW"
    assert profile["effective_native_resolution"] == "PT1H"
    assert profile["target_duration_seconds"] == 3600
    assert profile["raw_ids_persisted"] is False
    assert result["reference_time_utc"] == "2026-08-06T02:00:00Z"
    serialized = json.dumps(result, sort_keys=True)
    for clear_id in (
        "actual_load_training",
        "load_forecast_training",
        "load_forecast_error_lagged_training",
    ):
        assert clear_id not in serialized


def test_different_actual_and_forecast_logical_zones_fail_closed(
    compatible_composite,
) -> None:
    dimension = compatible_composite["fixture"]["series_dimension"]
    de_forecast = str(
        dimension.loc[
            dimension["group_name"].eq("load_forecast") & dimension["from_zone"].eq("DE_LU"),
            "series_id",
        ].iloc[0]
    )
    forecast_id = d281.feature_record_id(compatible_composite["records"][1])
    forecast_selection = next(
        entry
        for entry in compatible_composite["selection"]["entries"]
        if entry["feature_record_id"] == forecast_id
    )
    forecast_selection["series_id"] = de_forecast

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="logical zones differ",
    ):
        _verify(compatible_composite)


def test_lineage_primitive_must_be_selected_by_d280(compatible_composite) -> None:
    d280_result, d281_result, tracked = _predecessors(compatible_composite)
    selection = copy.deepcopy(compatible_composite["selection"])
    actual_id = d281.feature_record_id(compatible_composite["records"][0])
    selection["entries"] = [
        entry for entry in selection["entries"] if entry["feature_record_id"] != actual_id
    ]
    tracked["feature_selection"] = module.canonical_sha256(selection)
    assessment = copy.deepcopy(d280_result.assessment)
    assessment["bindings"]["feature_selection_content_id"] = tracked["feature_selection"]

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="not selected exactly once",
    ):
        module._compose_verified_predecessors(
            assessment,
            module.canonical_sha256(assessment),
            d281_result,
            compatible_composite["records"],
            selection,
            compatible_composite["mapping"],
            compatible_composite["semantic"],
            compatible_composite["zone_binding"],
            tracked,
            REFERENCE,
        )


def test_any_non_lineage_d280_blocker_prevents_composition(compatible_composite) -> None:
    flow_id = d281.feature_record_id(compatible_composite["records"][2])
    compatible_composite["selection"]["entries"] = [
        entry
        for entry in compatible_composite["selection"]["entries"]
        if entry["feature_record_id"] != flow_id
    ]

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="exactly the derived-lineage blocker",
    ):
        _verify(compatible_composite)


def test_d281_formula_and_operand_binding_remain_mandatory(compatible_composite) -> None:
    compatible_composite["lineage"]["transforms"][0]["formula"] = (
        "OPERATIONAL_FORECAST_MINUS_REALIZED_ACTUAL"
    )

    with pytest.raises(d281.EntsoeLtDerivedFeatureLineageError, match="formula differs"):
        _verify(compatible_composite)


def test_input_mutation_during_composition_fails_closed(
    compatible_composite,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = d281.assess_derived_feature_lineage

    def mutate_after_assessment(feature_records, lineage_document, **kwargs):
        result = original(feature_records, lineage_document, **kwargs)
        feature_records[0]["feature_id"] = "mutated_during_composition"
        return result

    monkeypatch.setattr(module.d281, "assess_derived_feature_lineage", mutate_after_assessment)

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="inputs changed during verification",
    ):
        _verify(compatible_composite)


@pytest.mark.parametrize("predecessor", ["D280", "D281"])
def test_predecessor_authority_escalation_fails_closed(
    compatible_composite,
    predecessor: str,
) -> None:
    d280_result, d281_result, tracked = _predecessors(compatible_composite)
    d280_assessment = copy.deepcopy(d280_result.assessment)
    d281_assessment = copy.deepcopy(d281_result)
    if predecessor == "D280":
        d280_assessment["authorities"]["production"] = True
    else:
        d281_assessment["authorities"]["production"] = True

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match=f"{predecessor} authorities differ",
    ):
        module._compose_verified_predecessors(
            d280_assessment,
            module.canonical_sha256(d280_assessment),
            d281_assessment,
            compatible_composite["records"],
            compatible_composite["selection"],
            compatible_composite["mapping"],
            compatible_composite["semantic"],
            compatible_composite["zone_binding"],
            tracked,
            REFERENCE,
        )


@pytest.mark.parametrize(
    "binding_name",
    [
        "feature_records_content_id",
        "feature_selection_content_id",
        "family_mapping_content_id",
        "semantic_binding_content_id",
        "zone_registry_content_id",
        "zone_binding_content_id",
        "lineage_content_id",
    ],
)
def test_every_composite_content_binding_is_exact(
    compatible_composite,
    binding_name: str,
) -> None:
    d280_result, d281_result, tracked = _predecessors(compatible_composite)
    tracked[binding_name.removesuffix("_content_id")] = "f" * 64

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="tracked .* content ID differs",
    ):
        module._compose_verified_predecessors(
            d280_result.assessment,
            d280_result.assessment_content_id,
            d281_result,
            compatible_composite["records"],
            compatible_composite["selection"],
            compatible_composite["mapping"],
            compatible_composite["semantic"],
            compatible_composite["zone_binding"],
            tracked,
            REFERENCE,
        )


def test_canonical_mw_is_checked_again_at_composition(compatible_composite) -> None:
    d280_result, d281_result, tracked = _predecessors(compatible_composite)
    mapping = copy.deepcopy(compatible_composite["mapping"])
    actual_series = compatible_composite["selection"]["entries"][0]["series_id"]
    signature = next(
        entry["signature_id"]
        for entry in compatible_composite["semantic"]["entries"]
        if entry["series_id"] == actual_series
    )
    next(entry for entry in mapping["entries"] if entry["signature_id"] == signature)[
        "canonical_unit"
    ] = "MWh"
    tracked["family_mapping"] = module.canonical_sha256(mapping)
    assessment = copy.deepcopy(d280_result.assessment)
    assessment["bindings"]["family_mapping_content_id"] = tracked["family_mapping"]

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="canonical MW",
    ):
        module._compose_verified_predecessors(
            assessment,
            module.canonical_sha256(assessment),
            d281_result,
            compatible_composite["records"],
            compatible_composite["selection"],
            mapping,
            compatible_composite["semantic"],
            compatible_composite["zone_binding"],
            tracked,
            REFERENCE,
        )


def test_target_requires_one_covering_temporal_zone_binding(
    compatible_composite,
) -> None:
    d280_result, d281_result, tracked = _predecessors(compatible_composite)
    zone_binding = copy.deepcopy(compatible_composite["zone_binding"])
    actual_series = compatible_composite["selection"]["entries"][0]["series_id"]
    signature = next(
        entry["signature_id"]
        for entry in compatible_composite["semantic"]["entries"]
        if entry["series_id"] == actual_series
    )
    zone_binding["bindings"] = [
        entry for entry in zone_binding["bindings"] if entry["signature_id"] != signature
    ]
    tracked["zone_binding"] = module.canonical_sha256(zone_binding)
    assessment = copy.deepcopy(d280_result.assessment)
    assessment["bindings"]["zone_binding_content_id"] = tracked["zone_binding"]

    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="no unique temporal zone binding",
    ):
        module._compose_verified_predecessors(
            assessment,
            module.canonical_sha256(assessment),
            d281_result,
            compatible_composite["records"],
            compatible_composite["selection"],
            compatible_composite["mapping"],
            compatible_composite["semantic"],
            zone_binding,
            tracked,
            REFERENCE,
        )


def test_unsupported_common_target_duration_fails_closed(
    compatible_composite,
) -> None:
    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="supported effective native cadence",
    ):
        module._resolution_for_seconds(1200)


def test_contract_hash_and_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document["status"] = "FORGED"
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        EntsoeLtDerivedPhysicalCompatibilityError,
        match="contract SHA-256 differs",
    ):
        validate_derived_physical_compatibility_contract_path(forged.resolve())

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    original_hash = module.CONTRACT_FILE_SHA256
    try:
        module.CONTRACT_FILE_SHA256 = hashlib.sha256(duplicate.read_bytes()).hexdigest()
        with pytest.raises(
            EntsoeLtDerivedPhysicalCompatibilityError,
            match="duplicate keys",
        ):
            validate_derived_physical_compatibility_contract_path(duplicate.resolve())
    finally:
        module.CONTRACT_FILE_SHA256 = original_hash
