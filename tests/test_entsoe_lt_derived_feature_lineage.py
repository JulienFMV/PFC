from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pfc_shaping.validation import entsoe_lt_derived_feature_lineage as module
from pfc_shaping.validation.entsoe_lt_derived_feature_lineage import (
    EntsoeLtDerivedFeatureLineageError,
    assess_derived_feature_lineage,
    assess_derived_feature_lineage_path,
    feature_record_id,
    validate_derived_feature_lineage_contract,
    validate_derived_feature_lineage_contract_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-LT-DERIVED-FEATURE-LINEAGE-CONTRACT-V1.json"
)
REFERENCE = datetime(2026, 8, 1, 4, 0, tzinfo=timezone.utc)


def _common() -> dict[str, object]:
    return {
        "target_start_utc": "2026-08-01T00:00:00Z",
        "target_end_utc": "2026-08-01T01:00:00Z",
        "origin_utc": "2026-08-01T03:00:00Z",
        "source_issue_utc": None,
        "operational_horizon_end_utc": None,
        "training_cutoff_utc": None,
        "support_state": "SUPPORTED",
        "missing_value_policy": "PRESERVE_NULL",
        "monthly_effect_policy": "ZERO_MEAN_WITHIN_SOLVER_MONTH",
    }


def _records() -> list[dict[str, object]]:
    common = _common()
    actual = {
        **common,
        "feature_id": "load_actual_20260801",
        "source_family": "actual_load",
        "information_role": "REALIZED_ACTUAL",
        "requested_use": "TRAINING_COVARIATE",
        "available_at_utc": "2026-08-01T02:00:00Z",
        "dependency_roles": [],
        "dependency_available_at_utc": [],
    }
    forecast = {
        **common,
        "feature_id": "load_forecast_20260801",
        "source_family": "load_forecast",
        "information_role": "OPERATIONAL_FORECAST",
        "requested_use": "TRAINING_COVARIATE",
        "available_at_utc": "2026-08-01T01:30:00Z",
        "source_issue_utc": "2026-08-01T00:00:00Z",
        "operational_horizon_end_utc": "2026-08-01T02:00:00Z",
        "dependency_roles": [],
        "dependency_available_at_utc": [],
    }
    output = {
        **common,
        "feature_id": "load_forecast_error_lagged_20260801",
        "source_family": "load_forecast_error",
        "information_role": "FORECAST_ERROR",
        "requested_use": "LAGGED_COVARIATE",
        "available_at_utc": "2026-08-01T02:00:00Z",
        "dependency_roles": ["OPERATIONAL_FORECAST", "REALIZED_ACTUAL"],
        "dependency_available_at_utc": [
            "2026-08-01T01:30:00Z",
            "2026-08-01T02:00:00Z",
        ],
    }
    return [actual, forecast, output]


def _lineage(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": module.LINEAGE_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "feature_availability_contract_content_id": (
            "c7826b4ad2fa5cdb6baff5d077f00ee5fd8d98108cebef9920c147d787df2ab0"
        ),
        "feature_records_content_id": module.canonical_sha256(records),
        "declared_at_utc": "2026-08-01T03:30:00Z",
        "transforms": [
            {
                "transform_id": "load_forecast_error_lagged",
                "transform_kind": "LOAD_FORECAST_ERROR",
                "output_feature_record_id": feature_record_id(records[2]),
                "input_feature_record_ids_in_formula_order": [
                    feature_record_id(records[0]),
                    feature_record_id(records[1]),
                ],
                "formula": "REALIZED_ACTUAL_MINUS_OPERATIONAL_FORECAST",
                "calculation_created_at_utc": "2026-08-01T02:00:00Z",
                "missing_value_policy": "PRESERVE_NULL",
                "output_sign_convention": "POSITIVE_WHEN_ACTUAL_ABOVE_FORECAST",
                "semantic_unit_zone_and_cadence_binding": ("REQUIRES_LATER_D280_COMPOSITE"),
                "reason": "Synthetic exact primitive dependency lineage.",
            }
        ],
        "authorities": dict(module.LINEAGE_AUTHORITIES),
    }


def _assess(
    records: list[dict[str, object]],
    lineage: dict[str, object],
):
    return assess_derived_feature_lineage(
        records,
        lineage,
        reference_time_utc=REFERENCE,
    )


def _rebind(lineage: dict[str, object], records: list[dict[str, object]]) -> None:
    lineage["feature_records_content_id"] = module.canonical_sha256(records)
    transform = lineage["transforms"][0]
    transform["output_feature_record_id"] = feature_record_id(records[2])
    transform["input_feature_record_ids_in_formula_order"] = [
        feature_record_id(records[0]),
        feature_record_id(records[1]),
    ]


def test_contract_is_exact_zero_cost_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    direct = validate_derived_feature_lineage_contract(document)
    bound = validate_derived_feature_lineage_contract_path(CONTRACT_PATH)

    assert direct == bound
    assert direct["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert direct["semantic_unit_zone_cadence_proven"] is False
    assert direct["arithmetic_recomputed"] is False
    assert direct["model_input_authorized"] is False
    assert direct["databricks_statement_count"] == 0
    assert all(value is False for value in document["authorities"].values())


def test_exact_load_forecast_error_lineage_passes_without_model_authority() -> None:
    records = _records()
    result = _assess(records, _lineage(records))

    assert result["status"] == module.PASS_STATUS
    assert result["transform_count"] == 1
    assert result["forecast_error_record_count"] == 1
    assert result["all_forecast_error_records_bound_once"] is True
    assert result["exact_input_identity_and_availability_bound"] is True
    assert result["semantic_unit_zone_and_cadence_proven"] is False
    assert result["arithmetic_recomputed"] is False
    assert result["d280_derived_model_input_admitted"] is False
    assert result["databricks_connection_count"] == 0
    assert result["h_drive_access_count"] == 0
    assert all(value is False for value in result["authorities"].values())
    serialized = json.dumps(result, sort_keys=True)
    assert "load_actual_20260801" not in serialized
    assert "load_forecast_20260801" not in serialized
    assert "load_forecast_error_lagged_20260801" not in serialized


def test_formula_operand_order_and_source_families_fail_closed() -> None:
    records = _records()
    swapped = _lineage(records)
    swapped["transforms"][0]["input_feature_record_ids_in_formula_order"].reverse()
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="input role"):
        _assess(records, swapped)

    records = _records()
    records[0]["source_family"] = "generation_actual"
    lineage = _lineage(records)
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="source family"):
        _assess(records, lineage)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("target_start_utc", "actual/output target_start_utc differs"),
        ("target_end_utc", "actual/output target_end_utc differs"),
        ("origin_utc", "actual/output origin_utc differs"),
    ],
)
def test_target_and_origin_must_match_every_primitive(
    field: str,
    expected: str,
) -> None:
    records = _records()
    replacements = {
        "target_start_utc": "2026-07-31T23:00:00Z",
        "target_end_utc": "2026-08-01T00:30:00Z",
        "origin_utc": "2026-08-01T03:15:00Z",
    }
    records[0][field] = replacements[field]
    lineage = _lineage(records)

    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match=expected):
        _assess(records, lineage)


def test_output_dependency_timestamps_bind_the_exact_input_records() -> None:
    records = _records()
    records[2]["dependency_available_at_utc"] = [
        "2026-08-01T02:00:00Z",
        "2026-08-01T01:30:00Z",
    ]
    lineage = _lineage(records)

    with pytest.raises(
        EntsoeLtDerivedFeatureLineageError,
        match="output dependency availability differs",
    ):
        _assess(records, lineage)


def test_calculation_and_declaration_chronology_fail_closed() -> None:
    records = _records()
    early = _lineage(records)
    early["transforms"][0]["calculation_created_at_utc"] = "2026-08-01T01:59:59Z"
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="predates"):
        _assess(records, early)

    late = _lineage(records)
    late["transforms"][0]["calculation_created_at_utc"] = "2026-08-01T03:31:00Z"
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="follows"):
        _assess(records, late)

    future = _lineage(records)
    future["declared_at_utc"] = "2026-08-01T04:00:01Z"
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="reference"):
        _assess(records, future)


def test_missing_duplicate_and_self_dependencies_fail_closed() -> None:
    records = _records()
    missing = _lineage(records)
    missing["transforms"] = []
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="transforms"):
        _assess(records, missing)

    duplicate = _lineage(records)
    duplicate["transforms"].append(copy.deepcopy(duplicate["transforms"][0]))
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="duplicated"):
        _assess(records, duplicate)

    self_link = _lineage(records)
    output_id = self_link["transforms"][0]["output_feature_record_id"]
    self_link["transforms"][0]["input_feature_record_ids_in_formula_order"][0] = output_id
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="self"):
        _assess(records, self_link)


def test_every_forecast_error_record_requires_exactly_one_transform() -> None:
    records = _records()
    second = copy.deepcopy(records[2])
    second["feature_id"] = "second_load_forecast_error"
    records.append(second)
    lineage = _lineage(records)

    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="exactly once"):
        _assess(records, lineage)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("formula", "OPERATIONAL_FORECAST_MINUS_REALIZED_ACTUAL", "formula differs"),
        ("missing_value_policy", "ZERO", "missing policy differs"),
        ("output_sign_convention", "UNSPECIFIED", "output sign differs"),
        (
            "semantic_unit_zone_and_cadence_binding",
            "PROVEN",
            "physical binding status differs",
        ),
    ],
)
def test_formula_missingness_sign_and_physical_authority_cannot_drift(
    field: str,
    value: str,
    expected: str,
) -> None:
    records = _records()
    lineage = _lineage(records)
    lineage["transforms"][0][field] = value

    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match=expected):
        _assess(records, lineage)


def test_content_detachment_and_authority_escalation_fail_closed() -> None:
    records = _records()
    detached = _lineage(records)
    detached["feature_records_content_id"] = "f" * 64
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="content ID"):
        _assess(records, detached)

    escalated = _lineage(records)
    escalated["authorities"]["model_input"] = True
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="authorities"):
        _assess(records, escalated)


def test_rejected_d273_input_never_gains_lineage_authority() -> None:
    records = _records()
    records[0]["available_at_utc"] = "2026-08-01T03:01:00Z"
    lineage = _lineage(records)
    records[2]["dependency_available_at_utc"][1] = "2026-08-01T03:01:00Z"
    records[2]["available_at_utc"] = "2026-08-01T03:01:00Z"
    _rebind(lineage, records)

    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="not structurally eligible"):
        _assess(records, lineage)


def test_mutated_contract_and_duplicate_json_key_are_rejected(tmp_path: Path) -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document["authorities"]["model_input"] = True
    mutated = tmp_path / "mutated-contract.json"
    mutated.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="SHA-256"):
        validate_derived_feature_lineage_contract_path(mutated.resolve())

    duplicate = tmp_path / "duplicate-contract.json"
    raw = CONTRACT_PATH.read_text(encoding="utf-8")
    duplicate.write_text(raw.replace("{", '{"schema_version":"duplicate",', 1), encoding="utf-8")
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="SHA-256"):
        validate_derived_feature_lineage_contract_path(duplicate.resolve())


def test_lineage_path_is_stable_and_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    records = _records()
    lineage = _lineage(records)
    valid = tmp_path / "lineage.json"
    valid.write_text(json.dumps(lineage), encoding="utf-8")

    result = assess_derived_feature_lineage_path(
        records,
        valid.resolve(),
        reference_time_utc=REFERENCE,
    )
    assert result["lineage_content_id"] == module.canonical_sha256(lineage)

    duplicate = tmp_path / "duplicate-lineage.json"
    raw = valid.read_text(encoding="utf-8")
    duplicate.write_text(
        raw.replace("{", '{"schema_version":"duplicate",', 1),
        encoding="utf-8",
    )
    with pytest.raises(EntsoeLtDerivedFeatureLineageError, match="JSON is invalid"):
        assess_derived_feature_lineage_path(
            records,
            duplicate.resolve(),
            reference_time_utc=REFERENCE,
        )


def test_source_has_no_ct_databricks_h_or_value_runtime_path() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8").casefold()

    assert "pfc_shaping.ct" not in source
    assert "databricks.sql" not in source
    assert "\\h:" not in source
    assert 'record["value"]' not in source
