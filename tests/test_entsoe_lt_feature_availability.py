from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pfc_shaping.validation.entsoe_lt_feature_availability import (
    CONTRACT_CONTENT_ID,
    CONTRACT_STATUS,
    EntsoeLtFeatureAvailabilityError,
    assess_feature_availability,
    validate_feature_availability_contract,
    validate_feature_availability_contract_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-LT-FEATURE-AVAILABILITY-CONTRACT-V1.json"
)


def _record(role: str, **overrides: object) -> dict[str, object]:
    dependency_count = {
        "FORECAST_ERROR": 2,
        "LAGGED_REALIZED": 1,
    }.get(role, 0)
    dependencies = [
        "2026-01-01T10:00:00Z",
        "2026-01-01T11:00:00Z",
    ][:dependency_count]
    dependency_roles = {
        "FORECAST_ERROR": ["OPERATIONAL_FORECAST", "REALIZED_ACTUAL"],
        "LAGGED_REALIZED": ["REALIZED_ACTUAL"],
    }.get(role, [])
    record: dict[str, object] = {
        "feature_id": f"feature_{role.casefold()}",
        "source_family": "actual_load",
        "information_role": role,
        "requested_use": "PREDICTION_COVARIATE",
        "target_start_utc": "2026-01-01T00:00:00Z",
        "target_end_utc": "2026-01-01T01:00:00Z",
        "origin_utc": "2026-01-02T00:00:00Z",
        "available_at_utc": dependencies[-1] if dependencies else "2026-01-01T02:00:00Z",
        "source_issue_utc": None,
        "operational_horizon_end_utc": None,
        "training_cutoff_utc": None,
        "dependency_roles": dependency_roles,
        "dependency_available_at_utc": dependencies,
        "support_state": "SUPPORTED",
        "missing_value_policy": "PRESERVE_NULL",
        "monthly_effect_policy": "ZERO_MEAN_WITHIN_SOLVER_MONTH",
    }
    record.update(overrides)
    return record


def _eligible_records() -> list[dict[str, object]]:
    return [
        _record(
            "REALIZED_ACTUAL",
            requested_use="TRAINING_COVARIATE",
        ),
        _record(
            "OPERATIONAL_FORECAST",
            feature_id="feature_operational_forecast",
            target_start_utc="2026-01-03T00:00:00Z",
            target_end_utc="2026-01-03T01:00:00Z",
            origin_utc="2026-01-02T00:00:00Z",
            available_at_utc="2026-01-01T13:05:00Z",
            source_issue_utc="2026-01-01T13:00:00Z",
            operational_horizon_end_utc="2026-01-04T00:00:00Z",
        ),
        _record(
            "FORECAST_ERROR",
            requested_use="LAGGED_COVARIATE",
        ),
        _record("LAGGED_REALIZED"),
        _record(
            "CALENDAR_KNOWN",
            target_start_utc="2029-01-01T00:00:00Z",
            target_end_utc="2029-01-01T01:00:00Z",
            available_at_utc="2020-01-01T00:00:00Z",
            monthly_effect_policy="NOT_APPLICABLE",
        ),
        _record(
            "ORIGIN_FROZEN_CLIMATOLOGY",
            target_start_utc="2029-01-01T00:00:00Z",
            target_end_utc="2029-01-01T01:00:00Z",
            available_at_utc="2026-01-01T00:00:00Z",
            training_cutoff_utc="2025-12-31T23:59:59Z",
        ),
        _record(
            "GOVERNED_SCENARIO_SHAPE",
            target_start_utc="2029-01-01T00:00:00Z",
            target_end_utc="2029-01-01T01:00:00Z",
            available_at_utc="2025-12-01T00:00:00Z",
            source_issue_utc="2025-12-01T00:00:00Z",
        ),
    ]


def test_contract_and_bound_path_are_exact_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    direct = validate_feature_availability_contract(document)
    bound = validate_feature_availability_contract_path(CONTRACT_PATH)

    assert direct == bound
    assert direct["status"] == CONTRACT_STATUS
    assert direct["contract_content_id"] == CONTRACT_CONTENT_ID
    assert direct["information_role_count"] == 7
    assert direct["model_input_authorized"] is False
    assert direct["databricks_statement_count"] == 0


def test_contract_rejects_authority_and_actual_future_use_escalation() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    authority = copy.deepcopy(document)
    authority["authorities"]["model_input"] = True
    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="authorities differs"):
        validate_feature_availability_contract(authority)

    future_actual = copy.deepcopy(document)
    future_actual["information_roles"][0]["future_target_use"] = "ALLOWED"
    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="information roles"):
        validate_feature_availability_contract(future_actual)


def test_all_seven_roles_can_pass_their_exact_structural_use() -> None:
    result = assess_feature_availability(_eligible_records())

    assert result["record_count"] == 7
    assert result["status"] == "PASS_VALUE_FREE_PIT_STRUCTURE_NO_MODEL_AUTHORITY"
    assert result["rejected_model_request_count"] == 0
    assert result["status_counts"] == {
        "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY": 7
    }
    assert result["severity_counts"] == {"INFO": 7}
    assert result["raw_values_opened"] is False
    assert result["databricks_connection_count"] == 0
    assert all(value is False for value in result["authorities"].values())


def test_future_same_target_actual_is_rejected_as_critical_leakage() -> None:
    record = _record(
        "REALIZED_ACTUAL",
        target_start_utc="2026-01-03T00:00:00Z",
        target_end_utc="2026-01-03T01:00:00Z",
        available_at_utc="2026-01-03T02:00:00Z",
    )

    result = assess_feature_availability([record])
    reasons = set(result["outcomes"][0]["reasons"])

    assert result["status"] == "FAIL_VALUE_FREE_FEATURE_LEAKAGE_OR_UNSUPPORTED"
    assert "ACTUAL_TARGET_NOT_ENDED_AT_ORIGIN" in reasons
    assert "AVAILABLE_AFTER_ORIGIN" in reasons
    assert "ACTUAL_REQUESTED_AS_PREDICTION_OR_LAGGED_FEATURE" in reasons
    assert result["outcomes"][0]["severity"] == "CRITICAL"
    assert result["outcomes"][0]["confidence"] == "HIGH"


def test_operational_forecast_must_be_issued_and_cover_the_target_by_origin() -> None:
    record = _record(
        "OPERATIONAL_FORECAST",
        target_start_utc="2026-01-03T00:00:00Z",
        target_end_utc="2026-01-03T01:00:00Z",
        origin_utc="2026-01-02T00:00:00Z",
        available_at_utc="2026-01-02T00:05:00Z",
        source_issue_utc="2026-01-03T00:30:00Z",
        operational_horizon_end_utc="2026-01-03T00:30:00Z",
    )

    result = assess_feature_availability([record])
    reasons = set(result["outcomes"][0]["reasons"])

    assert "AVAILABLE_AFTER_ORIGIN" in reasons
    assert "FORECAST_ISSUED_AFTER_ORIGIN" in reasons
    assert "FORECAST_ISSUED_AFTER_TARGET_START" in reasons
    assert "TARGET_OUTSIDE_OPERATIONAL_FORECAST_HORIZON" in reasons


def test_n_plus_three_target_cannot_use_a_short_operational_forecast_horizon() -> None:
    record = _record(
        "OPERATIONAL_FORECAST",
        target_start_utc="2029-01-01T00:00:00Z",
        target_end_utc="2029-01-01T01:00:00Z",
        available_at_utc="2026-01-01T00:05:00Z",
        source_issue_utc="2026-01-01T00:00:00Z",
        operational_horizon_end_utc="2026-01-03T00:00:00Z",
    )

    result = assess_feature_availability([record])

    assert result["reason_counts"] == {
        "TARGET_OUTSIDE_OPERATIONAL_FORECAST_HORIZON": 1
    }


@pytest.mark.parametrize(
    ("role", "overrides"),
    [
        (
            "OPERATIONAL_FORECAST",
            {
                "source_issue_utc": "2026-01-01T00:00:00Z",
                "operational_horizon_end_utc": "2026-01-04T00:00:00Z",
            },
        ),
        (
            "CALENDAR_KNOWN",
            {"monthly_effect_policy": "NOT_APPLICABLE"},
        ),
        (
            "ORIGIN_FROZEN_CLIMATOLOGY",
            {"training_cutoff_utc": "2025-12-31T23:59:59Z"},
        ),
    ],
)
def test_future_rows_cannot_enter_the_fold_as_training_covariates(
    role: str, overrides: dict[str, object]
) -> None:
    record = _record(
        role,
        requested_use="TRAINING_COVARIATE",
        target_start_utc="2026-01-03T00:00:00Z",
        target_end_utc="2026-01-03T01:00:00Z",
        **overrides,
    )

    reasons = set(assess_feature_availability([record])["outcomes"][0]["reasons"])

    assert "TRAINING_TARGET_NOT_ENDED_AT_ORIGIN" in reasons


@pytest.mark.parametrize(
    ("role", "overrides"),
    [
        (
            "OPERATIONAL_FORECAST",
            {
                "source_issue_utc": "2026-01-01T00:00:00Z",
                "operational_horizon_end_utc": "2026-01-04T00:00:00Z",
            },
        ),
        (
            "CALENDAR_KNOWN",
            {"monthly_effect_policy": "NOT_APPLICABLE"},
        ),
        (
            "ORIGIN_FROZEN_CLIMATOLOGY",
            {"training_cutoff_utc": "2025-12-31T23:59:59Z"},
        ),
        (
            "GOVERNED_SCENARIO_SHAPE",
            {"source_issue_utc": "2025-12-01T00:00:00Z"},
        ),
    ],
)
def test_contemporaneous_prediction_cannot_start_before_the_origin(
    role: str, overrides: dict[str, object]
) -> None:
    record = _record(
        role,
        target_start_utc="2026-01-01T23:00:00Z",
        target_end_utc="2026-01-03T01:00:00Z",
        **overrides,
    )

    reasons = set(assess_feature_availability([record])["outcomes"][0]["reasons"])

    assert "PREDICTION_TARGET_STARTED_BEFORE_ORIGIN" in reasons


def test_lagged_realized_target_is_expected_to_precede_the_origin() -> None:
    result = assess_feature_availability([_record("LAGGED_REALIZED")])

    assert result["reason_counts"] == {}
    assert result["all_model_requests_structurally_pit_safe"] is True


def test_lagged_label_cannot_hide_a_future_operational_forecast() -> None:
    record = _record(
        "OPERATIONAL_FORECAST",
        requested_use="LAGGED_COVARIATE",
        target_start_utc="2026-01-03T00:00:00Z",
        target_end_utc="2026-01-03T01:00:00Z",
        source_issue_utc="2026-01-01T00:00:00Z",
        operational_horizon_end_utc="2026-01-04T00:00:00Z",
    )

    result = assess_feature_availability([record])

    assert result["reason_counts"] == {"LAGGED_TARGET_NOT_ENDED_AT_ORIGIN": 1}


def test_forecast_error_is_only_lagged_after_truth_and_forecast_availability() -> None:
    record = _record(
        "FORECAST_ERROR",
        target_start_utc="2026-01-03T00:00:00Z",
        target_end_utc="2026-01-03T01:00:00Z",
        available_at_utc="2026-01-01T10:00:00Z",
        dependency_available_at_utc=[
            "2026-01-01T10:00:00Z",
            "2026-01-03T02:00:00Z",
        ],
    )

    reasons = set(assess_feature_availability([record])["outcomes"][0]["reasons"])

    assert "DERIVED_AVAILABILITY_DIFFERS_FROM_DEPENDENCIES" in reasons
    assert "FORECAST_ERROR_TARGET_NOT_ENDED_AT_ORIGIN" in reasons
    assert "FORECAST_ERROR_NOT_LAGGED_OR_DIAGNOSTIC" in reasons


def test_dependency_roles_cannot_disguise_two_forecasts_as_forecast_error() -> None:
    record = _record(
        "FORECAST_ERROR",
        dependency_roles=["OPERATIONAL_FORECAST", "OPERATIONAL_FORECAST"],
    )

    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="dependency roles"):
        assess_feature_availability([record])


@pytest.mark.parametrize(
    ("role", "overrides", "expected_reason"),
    [
        (
            "REALIZED_ACTUAL",
            {
                "requested_use": "TRAINING_COVARIATE",
                "available_at_utc": "2026-01-01T00:30:00Z",
            },
            "ACTUAL_AVAILABLE_BEFORE_TARGET_END",
        ),
        (
            "LAGGED_REALIZED",
            {
                "available_at_utc": "2026-01-01T00:30:00Z",
                "dependency_available_at_utc": ["2026-01-01T00:30:00Z"],
            },
            "LAGGED_REALIZED_AVAILABLE_BEFORE_TARGET_END",
        ),
        (
            "OPERATIONAL_FORECAST",
            {
                "available_at_utc": "2026-01-01T09:00:00Z",
                "source_issue_utc": "2026-01-01T10:00:00Z",
                "operational_horizon_end_utc": "2026-01-02T00:00:00Z",
            },
            "FORECAST_AVAILABLE_BEFORE_ISSUE",
        ),
    ],
)
def test_impossible_information_chronology_is_rejected(
    role: str,
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    record = _record(role, **overrides)

    reasons = set(assess_feature_availability([record])["outcomes"][0]["reasons"])

    assert expected_reason in reasons


def test_backfill_after_origin_never_gains_retroactive_feature_availability() -> None:
    record = _record(
        "REALIZED_ACTUAL",
        requested_use="TRAINING_COVARIATE",
        available_at_utc="2026-01-02T00:00:01Z",
    )

    result = assess_feature_availability([record])

    assert result["reason_counts"] == {"AVAILABLE_AFTER_ORIGIN": 1}


@pytest.mark.parametrize("policy", ["ZERO_FILL", "NEUTRAL_FILL"])
def test_missing_feature_cannot_be_silently_encoded_as_a_number(policy: str) -> None:
    record = _record(
        "CALENDAR_KNOWN",
        target_start_utc="2026-01-03T00:00:00Z",
        target_end_utc="2026-01-03T01:00:00Z",
        support_state="MISSING",
        missing_value_policy=policy,
        monthly_effect_policy="NOT_APPLICABLE",
    )

    result = assess_feature_availability([record])
    reasons = set(result["outcomes"][0]["reasons"])

    assert "SUPPORT_MISSING" in reasons
    assert f"MISSING_POLICY_{policy}_FORBIDDEN" in reasons
    assert result["severity_counts"] == {"HIGH": 1}


def test_non_calendar_effect_must_remain_zero_mean_inside_solver_month() -> None:
    record = _record(
        "ORIGIN_FROZEN_CLIMATOLOGY",
        target_start_utc="2029-01-01T00:00:00Z",
        target_end_utc="2029-01-01T01:00:00Z",
        training_cutoff_utc="2025-12-31T23:59:59Z",
        monthly_effect_policy="NOT_APPLICABLE",
    )

    result = assess_feature_availability([record])

    assert result["reason_counts"] == {"NON_CALENDAR_EFFECT_NOT_ZERO_MEAN": 1}


def test_climatology_must_be_frozen_strictly_before_each_origin() -> None:
    record = _record(
        "ORIGIN_FROZEN_CLIMATOLOGY",
        target_start_utc="2029-01-01T00:00:00Z",
        target_end_utc="2029-01-01T01:00:00Z",
        available_at_utc="2026-01-02T00:00:00Z",
        training_cutoff_utc="2026-01-02T00:00:00Z",
    )

    result = assess_feature_availability([record])

    assert result["reason_counts"] == {"CLIMATOLOGY_CUTOFF_NOT_BEFORE_ORIGIN": 1}


def test_scenario_shape_published_after_origin_or_used_for_training_is_rejected() -> None:
    record = _record(
        "GOVERNED_SCENARIO_SHAPE",
        requested_use="TRAINING_COVARIATE",
        target_start_utc="2029-01-01T00:00:00Z",
        target_end_utc="2029-01-01T01:00:00Z",
        available_at_utc="2026-01-02T00:00:01Z",
        source_issue_utc="2026-01-02T00:00:01Z",
    )

    reasons = set(assess_feature_availability([record])["outcomes"][0]["reasons"])

    assert "AVAILABLE_AFTER_ORIGIN" in reasons
    assert "SCENARIO_ISSUED_AFTER_ORIGIN" in reasons
    assert "SCENARIO_USE_INVALID" in reasons


def test_malformed_dependency_count_extra_value_field_and_duplicate_key_fail_closed() -> None:
    count = _record("FORECAST_ERROR", dependency_available_at_utc=[])
    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="dependency.*count"):
        assess_feature_availability([count])

    extra = _record("CALENDAR_KNOWN", monthly_effect_policy="NOT_APPLICABLE")
    extra["value"] = 42.0
    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="fields differ"):
        assess_feature_availability([extra])

    duplicate = _record("CALENDAR_KNOWN", monthly_effect_policy="NOT_APPLICABLE")
    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="key is duplicated"):
        assess_feature_availability([duplicate, copy.deepcopy(duplicate)])


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-01-01 00:00:00Z",
        "2026-01-01T00:00:00+00:00",
        "2026-1-1T00:00:00Z",
        "not-a-time",
    ],
)
def test_noncanonical_timestamp_is_rejected(timestamp: str) -> None:
    record = _record(
        "CALENDAR_KNOWN",
        origin_utc=timestamp,
        monthly_effect_policy="NOT_APPLICABLE",
    )

    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="canonical UTC Z"):
        assess_feature_availability([record])


def test_output_hashes_feature_identity_and_never_contains_values() -> None:
    record = _record("CALENDAR_KNOWN", monthly_effect_policy="NOT_APPLICABLE")

    result = assess_feature_availability([record])
    outcome = result["outcomes"][0]

    assert outcome["feature_id_sha256"] != record["feature_id"]
    assert "feature_id" not in outcome
    assert "value" not in outcome
    assert outcome["raw_feature_id_persisted"] is False
    assert outcome["raw_value_opened"] is False


def test_contract_path_rejects_a_mutated_copy(tmp_path: Path) -> None:
    mutated = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutated["missingness_and_shape_policy"]["neutral_fill_authorized"] = True
    path = tmp_path / "mutated-contract.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(EntsoeLtFeatureAvailabilityError, match="SHA-256 differs"):
        validate_feature_availability_contract_path(path.resolve())
