from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pfc_shaping.validation import entsoe_lt_derived_arithmetic_execution as module
from pfc_shaping.validation import entsoe_lt_derived_physical_compatibility as d282
from pfc_shaping.validation.entsoe_lt_derived_arithmetic_execution import (
    EntsoeLtDerivedArithmeticExecutionError,
    execute_synthetic_load_forecast_error,
    validate_derived_arithmetic_execution_contract,
    validate_derived_arithmetic_execution_contract_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = module.CONTRACT_PATH
REFERENCE = datetime(2026, 8, 6, 3, 0, tzinfo=timezone.utc)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _profile(position: int) -> dict[str, object]:
    return {
        "transform_id_sha256": _hash(f"transform-{position}"),
        "output_feature_record_id": _hash(f"output-{position}"),
        "input_feature_record_ids_in_formula_order": [
            _hash(f"actual-record-{position}"),
            _hash(f"forecast-record-{position}"),
        ],
        "actual_series_id_sha256": _hash(f"actual-series-{position}"),
        "forecast_series_id_sha256": _hash(f"forecast-series-{position}"),
        "logical_zone_sha256": _hash("CH"),
        "canonical_unit": "MW",
        "effective_native_resolution": "PT1H",
        "target_duration_seconds": 3600,
        "temporal_zone_binding_complete": True,
        "derived_output_inherits_physical_profile": True,
        "raw_ids_persisted": False,
        "raw_or_feature_values_opened": False,
    }


def _d282_assessment(count: int = 2) -> dict[str, object]:
    profiles = [_profile(position) for position in range(count)]
    return {
        "schema_version": d282.ASSESSMENT_SCHEMA,
        "status": d282.PASS_STATUS,
        "contract_content_id": d282.CONTRACT_CONTENT_ID,
        "reference_time_utc": "2026-08-06T02:00:00Z",
        "bindings": {
            name: _hash(name)
            for name in (
                "d280_assessment_content_id",
                "d281_assessment_content_id",
                "feature_records_content_id",
                "feature_selection_content_id",
                "family_mapping_content_id",
                "semantic_binding_content_id",
                "zone_registry_content_id",
                "zone_binding_content_id",
                "lineage_content_id",
            )
        },
        "transform_count": count,
        "all_primitives_selected_once": True,
        "all_temporal_zone_bindings_complete": True,
        "all_logical_zones_match": True,
        "all_canonical_units_are_mw": True,
        "all_effective_native_cadences_match": True,
        "d280_lineage_blocker_structurally_discharged": True,
        "profiles": profiles,
        "clear_series_or_feature_ids_persisted": False,
        "real_or_feature_values_opened": False,
        "arithmetic_recomputed": False,
        "derived_model_input_authorized": False,
        "authorities": dict(d282.AUTHORITIES),
        **module.EXECUTION,
    }


def _artifact(
    assessment: dict[str, object],
    role: str,
    values: list[str | None],
) -> dict[str, object]:
    rows = []
    for position, (profile, value) in enumerate(zip(assessment["profiles"], values, strict=True)):
        is_actual = role == module.ACTUAL_ROLE
        record_position = 0 if is_actual else 1
        start_hour = position
        rows.append(
            {
                "feature_record_id": profile["input_feature_record_ids_in_formula_order"][
                    record_position
                ],
                "target_start_utc": f"2026-08-01T{start_hour:02d}:00:00Z",
                "target_end_utc": f"2026-08-01T{start_hour + 1:02d}:00:00Z",
                "series_id_sha256": profile[
                    "actual_series_id_sha256" if is_actual else "forecast_series_id_sha256"
                ],
                "logical_zone_sha256": profile["logical_zone_sha256"],
                "canonical_unit": profile["canonical_unit"],
                "effective_native_resolution": profile["effective_native_resolution"],
                "value_decimal_mw": value,
            }
        )
    return {
        "schema_version": module.OPERAND_SCHEMA,
        "evidence_class": module.EVIDENCE_CLASS,
        "role": role,
        "created_at_utc": "2026-08-06T02:30:00Z",
        "d282_assessment_content_id": module.canonical_sha256(assessment),
        "rows": rows,
        "authorities": dict(module.ARTIFACT_AUTHORITIES),
    }


@pytest.fixture
def arithmetic_case() -> dict[str, object]:
    assessment = _d282_assessment()
    return {
        "assessment": assessment,
        "actual": _artifact(assessment, module.ACTUAL_ROLE, ["1250.25", None]),
        "forecast": _artifact(
            assessment,
            module.FORECAST_ROLE,
            ["1248", "900.5"],
        ),
    }


def _execute(case: dict[str, object]):
    return execute_synthetic_load_forecast_error(
        contract_path=CONTRACT_PATH,
        d282_assessment=case["assessment"],
        actual_operand_artifact=case["actual"],
        forecast_operand_artifact=case["forecast"],
        reference_time_utc=REFERENCE,
    )


def test_contract_is_exact_zero_query_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    direct = validate_derived_arithmetic_execution_contract(document)
    stable = validate_derived_arithmetic_execution_contract_path(CONTRACT_PATH)

    assert direct == stable
    assert direct["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert direct["algorithm_content_id"] == module.ALGORITHM_CONTENT_ID
    assert direct["real_arithmetic_authorized"] is False
    assert direct["model_input_authorized"] is False
    assert all(value == 0 for value in module.EXECUTION.values())
    assert all(value is False for value in module.AUTHORITIES.values())


def test_exact_decimal_subtraction_and_null_propagation_are_content_addressed(
    arithmetic_case,
) -> None:
    result = _execute(arithmetic_case)

    assert result.assessment["status"] == module.PASS_STATUS
    assert result.assessment["computed_value_count"] == 1
    assert result.assessment["null_propagated_count"] == 1
    assert result.assessment["canonical_decimal_arithmetic_executed"] is True
    assert result.assessment["real_arithmetic_authorized"] is False
    assert result.assessment["model_input_authorized"] is False
    values = [row["value_decimal_mw"] for row in result.output_artifact["rows"]]
    assert sorted(values, key=lambda value: value is None) == ["2.25", None]
    null_row = next(
        row for row in result.output_artifact["rows"] if row["value_decimal_mw"] is None
    )
    assert null_row["input_null_propagated"] is True
    assert result.output_artifact_content_id == module.canonical_sha256(result.output_artifact)
    assert result.assessment_content_id == module.canonical_sha256(result.assessment)
    serialized_assessment = json.dumps(result.assessment, sort_keys=True)
    assert "1250.25" not in serialized_assessment
    assert "1248" not in serialized_assessment
    assert "2.25" not in serialized_assessment


def test_replay_is_deterministic_and_output_order_is_canonical(arithmetic_case) -> None:
    arithmetic_case["actual"]["rows"].reverse()
    arithmetic_case["forecast"]["rows"].reverse()

    first = _execute(arithmetic_case)
    second = _execute(arithmetic_case)

    assert first == second
    output_ids = [row["output_feature_record_id"] for row in first.output_artifact["rows"]]
    assert output_ids == sorted(output_ids)


def test_sign_convention_is_actual_minus_forecast() -> None:
    assessment = _d282_assessment(1)
    case = {
        "assessment": assessment,
        "actual": _artifact(assessment, module.ACTUAL_ROLE, ["5"]),
        "forecast": _artifact(assessment, module.FORECAST_ROLE, ["7.5"]),
    }

    result = _execute(case)

    assert result.output_artifact["rows"][0]["value_decimal_mw"] == "-2.5"
    assert result.assessment["output_sign_convention"] == "POSITIVE_WHEN_ACTUAL_ABOVE_FORECAST"


def test_negative_zero_is_canonicalized() -> None:
    assessment = _d282_assessment(1)
    case = {
        "assessment": assessment,
        "actual": _artifact(assessment, module.ACTUAL_ROLE, ["0"]),
        "forecast": _artifact(assessment, module.FORECAST_ROLE, ["0"]),
    }

    result = _execute(case)

    assert result.output_artifact["rows"][0]["value_decimal_mw"] == "0"


@pytest.mark.parametrize(
    "bad_value",
    [1.5, True, "01", "+1", "1.0", "1e3", "NaN", "-0"],
)
def test_noncanonical_or_binary_numeric_inputs_fail_closed(
    arithmetic_case,
    bad_value,
) -> None:
    arithmetic_case["actual"]["rows"][0]["value_decimal_mw"] = bad_value

    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="canonical base-10|not canonical",
    ):
        _execute(arithmetic_case)


@pytest.mark.parametrize(
    ("bad_value", "expected"),
    [
        ("1.1234567", "scale exceeds"),
        ("1234567890123456789", "significant digits exceed"),
        ("1000000001", "magnitude is invalid"),
    ],
)
def test_decimal_scale_precision_and_magnitude_are_bounded(
    arithmetic_case,
    bad_value: str,
    expected: str,
) -> None:
    arithmetic_case["actual"]["rows"][0]["value_decimal_mw"] = bad_value

    with pytest.raises(EntsoeLtDerivedArithmeticExecutionError, match=expected):
        _execute(arithmetic_case)


@pytest.mark.parametrize(
    ("artifact_name", "field", "value", "expected"),
    [
        ("actual", "series_id_sha256", "f" * 64, "actual series hash differs"),
        ("forecast", "logical_zone_sha256", "f" * 64, "logical zone hash differs"),
        ("actual", "canonical_unit", "MWh", "actual unit differs"),
        (
            "forecast",
            "effective_native_resolution",
            "PT15M",
            "native resolution differs",
        ),
    ],
)
def test_physical_profile_drift_fails_closed(
    arithmetic_case,
    artifact_name: str,
    field: str,
    value: str,
    expected: str,
) -> None:
    arithmetic_case[artifact_name]["rows"][0][field] = value

    with pytest.raises(EntsoeLtDerivedArithmeticExecutionError, match=expected):
        _execute(arithmetic_case)


def test_target_interval_and_duration_must_match(arithmetic_case) -> None:
    mismatch = copy.deepcopy(arithmetic_case)
    mismatch["forecast"]["rows"][0]["target_end_utc"] = "2026-08-01T02:00:00Z"
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="target_end_utc differs",
    ):
        _execute(mismatch)

    wrong_duration = copy.deepcopy(arithmetic_case)
    wrong_duration["assessment"]["profiles"][0]["target_duration_seconds"] = 1800
    _rebind_artifacts(wrong_duration)
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="target duration differs",
    ):
        _execute(wrong_duration)


def _rebind_artifacts(case: dict[str, object]) -> None:
    content_id = module.canonical_sha256(case["assessment"])
    case["actual"]["d282_assessment_content_id"] = content_id
    case["forecast"]["d282_assessment_content_id"] = content_id


def test_missing_duplicate_and_orphan_operand_rows_fail_closed(arithmetic_case) -> None:
    missing = copy.deepcopy(arithmetic_case)
    missing["actual"]["rows"].pop()
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="operand row is missing",
    ):
        _execute(missing)

    duplicate = copy.deepcopy(arithmetic_case)
    duplicate["actual"]["rows"].append(copy.deepcopy(duplicate["actual"]["rows"][0]))
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="duplicated",
    ):
        _execute(duplicate)

    orphan = copy.deepcopy(arithmetic_case)
    extra = copy.deepcopy(orphan["actual"]["rows"][0])
    extra["feature_record_id"] = "f" * 64
    orphan["actual"]["rows"].append(extra)
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="orphan or substituted",
    ):
        _execute(orphan)


def test_operand_artifacts_are_bound_to_exact_d282_content(arithmetic_case) -> None:
    arithmetic_case["actual"]["d282_assessment_content_id"] = "f" * 64

    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="operand D282 assessment content ID differs",
    ):
        _execute(arithmetic_case)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arithmetic_recomputed", True),
        ("derived_model_input_authorized", True),
        ("real_or_feature_values_opened", True),
    ],
)
def test_d282_authority_or_value_escalation_fails_closed(
    arithmetic_case,
    field: str,
    value: bool,
) -> None:
    arithmetic_case["assessment"][field] = value
    _rebind_artifacts(arithmetic_case)

    with pytest.raises(EntsoeLtDerivedArithmeticExecutionError, match=f"D282 {field}"):
        _execute(arithmetic_case)


def test_future_operand_and_future_d282_reference_fail_closed(arithmetic_case) -> None:
    future_operand = copy.deepcopy(arithmetic_case)
    future_operand["actual"]["created_at_utc"] = "2026-08-06T03:00:01Z"
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="created after reference",
    ):
        _execute(future_operand)

    future_d282 = copy.deepcopy(arithmetic_case)
    future_d282["assessment"]["reference_time_utc"] = "2026-08-06T03:00:01Z"
    _rebind_artifacts(future_d282)
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="predates D282",
    ):
        _execute(future_d282)


def test_input_mutation_during_execution_fails_closed(
    arithmetic_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = module._validate_operand_artifact
    calls = 0

    def mutate_after_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            arithmetic_case["actual"]["rows"][0]["value_decimal_mw"] = "1250.5"
        return result

    monkeypatch.setattr(module, "_validate_operand_artifact", mutate_after_validation)
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="actual operand artifact changed",
    ):
        _execute(arithmetic_case)


def test_contract_hash_and_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document["status"] = "FORGED"
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        EntsoeLtDerivedArithmeticExecutionError,
        match="contract SHA-256 differs",
    ):
        validate_derived_arithmetic_execution_contract_path(forged.resolve())

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    original_hash = module.CONTRACT_FILE_SHA256
    try:
        module.CONTRACT_FILE_SHA256 = hashlib.sha256(duplicate.read_bytes()).hexdigest()
        with pytest.raises(
            EntsoeLtDerivedArithmeticExecutionError,
            match="duplicate keys",
        ):
            validate_derived_arithmetic_execution_contract_path(duplicate.resolve())
    finally:
        module.CONTRACT_FILE_SHA256 = original_hash


def test_runtime_has_no_ct_databricks_h_or_binary_float_path() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "pfc_shaping.ct" not in source
    assert "import databricks" not in source.lower()
    assert "from databricks" not in source.lower()
    assert ".connect(" not in source
    assert "H:\\" not in source
    assert "H:/" not in source
    assert "float(" not in source
    assert "Decimal" in source
    assert "PRESERVE_NULL" in json.dumps(module.ALGORITHM_SPEC)
