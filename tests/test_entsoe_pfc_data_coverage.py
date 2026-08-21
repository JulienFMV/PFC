from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pfc_shaping.validation.entsoe_pfc_data_coverage import (
    CONTRACT_CONTENT_ID,
    CONTRACT_PATH,
    EntsoePfcDataCoverageError,
    assess_entsoe_pfc_data_coverage,
    load_and_assess_entsoe_pfc_data_coverage,
)


def _document() -> dict[str, object]:
    return json.loads(Path(CONTRACT_PATH).read_text(encoding="utf-8"))


def test_exact_audit_reports_partial_dev_coverage_without_authority() -> None:
    result = load_and_assess_entsoe_pfc_data_coverage()

    assert result["contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["observed_macro_family_count"] == 11
    assert result["baseline_required_group_count"] == 13
    assert result["baseline_unproven_dimension_count"] == 10
    assert result["coupled_zone_requirement_count"] == 5
    assert result["additional_high_value_family_count"] == 14
    assert result["exploratory_family_count"] == 6
    assert result["newly_identified_useful_family_count"] == 9
    assert result["current_exact_inventory_proven"] is False
    assert result["all_required_data_types_proven_in_dev"] is False
    assert result["model_input_authorized"] is False


def test_current_inventory_and_d243_cannot_be_self_declared_complete() -> None:
    for field in (
        "current_exact_dimension_inventory_captured",
        "d243_inventory_executed",
        "coverage_can_be_classified_present_or_absent",
    ):
        document = _document()
        document["evidence_state"][field] = True
        with pytest.raises(EntsoePfcDataCoverageError, match=field):
            assess_entsoe_pfc_data_coverage(document)


def test_model_and_data_authority_escalation_is_rejected() -> None:
    document = _document()
    document["authorities"]["model_input_authorized"] = True

    with pytest.raises(EntsoePfcDataCoverageError, match="authorities"):
        assess_entsoe_pfc_data_coverage(document)


def test_coupled_price_zones_include_ch_and_all_direct_neighbours() -> None:
    document = _document()
    first = document["coupled_zone_requirements"][0]

    assert first["families"] == ["day_ahead_prices"]
    assert first["logical_zones"] == ["CH", "DE_LU", "FR", "IT_NORTH", "AT"]
    assert first["priority"] == "BASELINE_REQUIRED"

    changed = copy.deepcopy(document)
    changed["coupled_zone_requirements"][0]["logical_zones"].remove("FR")
    with pytest.raises(EntsoePfcDataCoverageError, match="coupled price zones"):
        assess_entsoe_pfc_data_coverage(changed)


def test_new_families_cover_intraday_storage_fcr_adequacy_and_zone_history() -> None:
    document = _document()
    names = {
        entry["family"] for entry in document["newly_identified_useful_families"]
    }

    assert {
        "intraday_energy_prices",
        "energy_storage_actual_and_capacity",
        "fcr_capacity_and_capacity_prices",
        "short_term_adequacy_forecasts",
        "bidding_zone_and_eic_configuration_history",
    } <= names


def test_only_zone_configuration_history_is_a_new_baseline_blocker() -> None:
    document = _document()
    blockers = [
        entry["family"]
        for entry in document["newly_identified_useful_families"]
        if entry["baseline_blocker"]
    ]

    assert blockers == ["bidding_zone_and_eic_configuration_history"]


def test_optional_family_cannot_be_promoted_to_baseline_blocker() -> None:
    document = _document()
    document["newly_identified_useful_families"][0]["baseline_blocker"] = True

    with pytest.raises(EntsoePfcDataCoverageError, match="optional new families"):
        assess_entsoe_pfc_data_coverage(document)


def test_source_binding_tamper_is_rejected() -> None:
    document = _document()
    document["source_bindings"]["d243_inventory_contract_sha256"] = "f" * 64

    with pytest.raises(EntsoePfcDataCoverageError, match="source binding"):
        assess_entsoe_pfc_data_coverage(document)


def test_execution_is_zero_cost_and_real_rows_remain_unopened() -> None:
    result = load_and_assess_entsoe_pfc_data_coverage()

    assert set(result["execution"].values()) == {0}

    document = _document()
    document["execution"]["databricks_statement_count"] = 1
    with pytest.raises(EntsoePfcDataCoverageError, match="execution counters"):
        assess_entsoe_pfc_data_coverage(document)


def test_validator_has_no_databricks_or_pandas_dependency() -> None:
    source = Path(
        "pfc_shaping/validation/entsoe_pfc_data_coverage.py"
    ).read_text(encoding="utf-8")

    assert "import pandas" not in source
    assert "databricks.sql" not in source
    assert "requests." not in source
