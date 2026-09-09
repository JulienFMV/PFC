from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pfc_shaping.validation.eex_entsoe_governed_acquisition_package import (
    ACQUISITION_PACKAGE_CONTRACT_CONTENT_ID,
    ACQUISITION_PACKAGE_CONTRACT_FILE_SHA256,
    GovernedAcquisitionPackageError,
    build_zero_execution_preflight,
    load_strict_json,
    validate_governed_acquisition_package_contract,
    validate_local_boundaries,
    validate_zero_execution_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-ENTSOE-GOVERNED-ACQUISITION-PACKAGE-CONTRACT-V1.json"
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _artifact(document: dict[str, object], role: str) -> dict[str, object]:
    inventory = document["artifact_inventory"]
    assert isinstance(inventory, list)
    return next(item for item in inventory if item["role"] == role)


def test_contract_exact_bytes_and_content_validate() -> None:
    payload = CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == ACQUISITION_PACKAGE_CONTRACT_FILE_SHA256
    result = validate_governed_acquisition_package_contract(_contract())
    assert result["contract_content_id"] == ACQUISITION_PACKAGE_CONTRACT_CONTENT_ID
    assert result["this_contract_databricks_statements"] == 0
    assert result["model_input_authorized"] is False


def test_contract_carries_all_current_eex_products_and_eur_per_mwh() -> None:
    result = validate_governed_acquisition_package_contract(_contract())
    assert result["eex_product_families"] == [
        "DAY",
        "WEEK",
        "WEEKEND",
        "MONTH",
        "QUARTER",
        "YEAR",
    ]
    assert result["eex_price_unit"] == "EUR/MWh"
    assert result["entsoe_day_ahead_price_unit"] == "EUR/MWh"


def test_contract_preserves_monthly_solver_level_authority() -> None:
    scope = _contract()["source_scope"]
    assert scope["monthly_level_authority"] == "solver"  # type: ignore[index]
    assert scope["short_tenor_shape_must_be_zero_mean_within_solver_month"] is True  # type: ignore[index]


def test_contract_does_not_call_historical_hourly_prices_a_15_minute_defect() -> None:
    result = validate_governed_acquisition_package_contract(_contract())
    assert result[
        "historical_hourly_swiss_prices_are_not_a_missing_15_minute_data_defect"
    ] is True


def test_preflight_is_deterministic_and_zero_execution() -> None:
    first = build_zero_execution_preflight(_contract())
    second = build_zero_execution_preflight(_contract())
    assert first == second
    assert all(value == 0 for value in first["execution"].values())  # type: ignore[union-attr]
    assert validate_zero_execution_preflight(first) == first


def test_existing_eex_is_hash_verified_without_parsing_price_rows() -> None:
    local = validate_local_boundaries(ROOT)
    assert local["status"] == "PASS_LOCAL_HASHES_ONLY_ZERO_EXECUTION"
    assert local["existing_eex_rows_parsed"] == 0
    assert local["databricks_statements"] == 0
    assert local["warehouse_starts"] == 0


def test_future_ceiling_is_once_daily_but_inactive() -> None:
    budget = build_zero_execution_preflight(_contract())["future_daily_budget"]
    assert budget["max_capture_batches_per_europe_zurich_day"] == 1  # type: ignore[index]
    assert budget["max_warehouse_starts_per_capture_batch"] == 1  # type: ignore[index]
    assert budget["max_entsoe_read_only_statements_per_capture_batch"] == 3  # type: ignore[index]
    assert budget["ceiling_active"] is False  # type: ignore[index]
    assert budget["new_explicit_user_authorization_required"] is True  # type: ignore[index]


def test_four_source_envelope_roles_are_exact() -> None:
    preflight = build_zero_execution_preflight(_contract())
    assert preflight["source_roles"] == [
        "eex_acquisition",
        "eex_source_trusted_time",
        "entsoe_acquisition",
        "entsoe_source_trusted_time",
    ]


@pytest.mark.parametrize("product", ["DAY", "WEEK", "WEEKEND"])
def test_missing_short_tenor_product_fails_closed(product: str) -> None:
    document = _contract()
    products = _artifact(document, "eex_daily_snapshot")["required_product_families"]
    products.remove(product)  # type: ignore[union-attr]
    with pytest.raises(GovernedAcquisitionPackageError, match="EEX products differs"):
        validate_governed_acquisition_package_contract(document)


def test_eex_unit_downgrade_fails_closed() -> None:
    document = _contract()
    _artifact(document, "eex_daily_snapshot")["price_unit"] = "EUR"
    with pytest.raises(GovernedAcquisitionPackageError, match="EEX price unit differs"):
        validate_governed_acquisition_package_contract(document)


def test_short_tenor_level_authority_fails_closed() -> None:
    document = _contract()
    document["source_scope"]["monthly_level_authority"] = "short_tenor"  # type: ignore[index]
    with pytest.raises(GovernedAcquisitionPackageError, match="monthly level authority"):
        validate_governed_acquisition_package_contract(document)


def test_daily_capture_or_warehouse_ceiling_cannot_increase() -> None:
    for field in (
        "future_max_capture_batches_per_europe_zurich_day",
        "future_max_warehouse_starts_per_capture_batch",
    ):
        document = _contract()
        document["execution_and_cost_fence"][field] = 2  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError):
            validate_governed_acquisition_package_contract(document)


def test_databricks_execution_or_write_enablement_fails_closed() -> None:
    for field in (
        "this_contract_execution_enabled",
        "entsoe_value_query_generation_authorized",
        "ddl_dml_merge_copy_unload_or_temp_objects_authorized",
    ):
        document = _contract()
        document["execution_and_cost_fence"][field] = True  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError):
            validate_governed_acquisition_package_contract(document)


def test_backfill_cannot_be_promoted_to_historical_pit_truth() -> None:
    document = _contract()
    document["source_scope"][  # type: ignore[index]
        "historical_backfill_may_reconstruct_past_information_sets"
    ] = True
    with pytest.raises(GovernedAcquisitionPackageError, match="backfill"):
        validate_governed_acquisition_package_contract(document)


def test_eex_reuse_must_not_add_a_statement() -> None:
    document = _contract()
    _artifact(document, "eex_daily_snapshot")["new_databricks_statement_count"] = 1
    with pytest.raises(GovernedAcquisitionPackageError, match="EEX new statements"):
        validate_governed_acquisition_package_contract(document)


def test_external_time_is_existence_bound_not_exact_signature_time() -> None:
    external = _contract()["external_time_batch_contract"]
    assert external[  # type: ignore[index]
        "proves_existence_no_later_than_external_time_not_exact_signature_creation_time"
    ] is True
    assert external[  # type: ignore[index]
        "historical_rows_preceding_challenge_do_not_gain_retroactive_point_in_time_proof"
    ] is True


def test_timestamp_profile_mixing_fails_closed() -> None:
    document = _contract()
    document["external_time_batch_contract"]["profile_mixing_authorized"] = True  # type: ignore[index]
    with pytest.raises(GovernedAcquisitionPackageError, match="profile mixing"):
        validate_governed_acquisition_package_contract(document)


def test_any_nonzero_preflight_execution_fails_closed() -> None:
    baseline = build_zero_execution_preflight(_contract())
    for field in baseline["execution"]:  # type: ignore[union-attr]
        preflight = copy.deepcopy(baseline)
        preflight["execution"][field] = 1  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError, match="must be integer zero"):
            validate_zero_execution_preflight(preflight)


def test_preflight_authority_escalation_fails_closed() -> None:
    baseline = build_zero_execution_preflight(_contract())
    for field in baseline["authorities"]:  # type: ignore[union-attr]
        preflight = copy.deepcopy(baseline)
        preflight["authorities"][field] = True  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError, match="authority"):
            validate_zero_execution_preflight(preflight)


def test_preflight_evidence_escalation_fails_closed() -> None:
    baseline = build_zero_execution_preflight(_contract())
    for field in baseline["evidence"]:  # type: ignore[union-attr]
        preflight = copy.deepcopy(baseline)
        preflight["evidence"][field] = True  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError, match="evidence"):
            validate_zero_execution_preflight(preflight)


def test_local_hash_validation_can_be_bound_into_preflight() -> None:
    local = validate_local_boundaries(ROOT)
    for field in (
        "databricks_connections",
        "databricks_statements",
        "warehouse_starts",
        "network_calls",
        "h_drive_accesses",
        "remote_writes",
    ):
        assert local[field] == 0
    preflight = build_zero_execution_preflight(_contract(), local_validation=local)
    assert preflight["local_boundary_integrity_verified"] is True
    assert preflight["existing_eex_bytes_hash_verified"] is True
    validate_zero_execution_preflight(preflight)


def test_unknown_top_level_contract_field_fails_closed() -> None:
    document = _contract()
    document["future_escape_hatch"] = False
    with pytest.raises(GovernedAcquisitionPackageError, match="contract fields differ"):
        validate_governed_acquisition_package_contract(document)


@pytest.mark.parametrize(
    "field",
    [
        "zero_query_plan_sha256",
        "entsoe_metadata_admission_contract_sha256",
        "receipt_contract_sha256",
        "trust_registry_contract_sha256",
        "registered_signer_binding_contract_sha256",
        "entsoe_intake_contract_sha256",
        "eex_normalizer_module_sha256",
    ],
)
def test_any_bound_contract_hash_change_fails_closed(field: str) -> None:
    document = _contract()
    document["selected_boundaries"][field] = "0" * 64  # type: ignore[index]
    with pytest.raises(GovernedAcquisitionPackageError, match="boundary"):
        validate_governed_acquisition_package_contract(document)


@pytest.mark.parametrize(
    ("role", "field"),
    [
        ("eex_daily_snapshot", "FactLoadTimestampUtc"),
        ("entsoe_series_dimension", "native_resolution"),
        ("entsoe_latest_values", "meta_load_timestamp_utc"),
        ("entsoe_vintage_values", "as_of_utc"),
    ],
)
def test_missing_normalized_field_fails_closed(role: str, field: str) -> None:
    document = _contract()
    fields = _artifact(document, role)["required_fields"]
    fields.remove(field)  # type: ignore[union-attr]
    with pytest.raises(GovernedAcquisitionPackageError, match="fields differs"):
        validate_governed_acquisition_package_contract(document)


def test_cost_fence_rejects_bool_as_zero_counter() -> None:
    document = _contract()
    document["execution_and_cost_fence"][  # type: ignore[index]
        "this_contract_h_drive_accesses"
    ] = False
    with pytest.raises(GovernedAcquisitionPackageError, match="integer zero"):
        validate_governed_acquisition_package_contract(document)


def test_cost_fence_rejects_unbounded_or_increased_limits() -> None:
    mutations = (
        ("future_max_entsoe_read_only_statements_per_capture_batch", 4),
        ("future_max_statement_runtime_seconds", 301),
        ("future_max_batch_runtime_seconds", 901),
        ("future_max_retries", 1),
        ("max_total_local_export_bytes", 68_719_476_737),
    )
    for field, value in mutations:
        document = _contract()
        document["execution_and_cost_fence"][field] = value  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError):
            validate_governed_acquisition_package_contract(document)


def test_per_artifact_row_or_byte_cap_cannot_increase() -> None:
    for field in ("max_rows", "max_bytes"):
        document = _contract()
        cap = document["execution_and_cost_fence"]["per_artifact_hard_caps"][  # type: ignore[index]
            "entsoe_vintage_values"
        ]
        cap[field] += 1  # type: ignore[index,operator]
        with pytest.raises(GovernedAcquisitionPackageError, match="vintage_values"):
            validate_governed_acquisition_package_contract(document)


def test_h_drive_or_remote_write_authorization_fails_closed() -> None:
    for field in ("h_drive_access_authorized", "remote_write_authorized"):
        document = _contract()
        document["delivery_boundary"][field] = True  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError, match="delivery"):
            validate_governed_acquisition_package_contract(document)


def test_manifest_must_retain_statement_class_and_lineage_fields() -> None:
    document = _contract()
    document["manifest_contract"]["artifact_required_fields"].remove(  # type: ignore[index]
        "statement_class"
    )
    with pytest.raises(GovernedAcquisitionPackageError, match="artifact manifest fields"):
        validate_governed_acquisition_package_contract(document)


def test_source_envelopes_bind_same_challenge_and_registry_roles() -> None:
    for field in (
        "all_payloads_bind_same_capture_batch_id_and_challenge_nonce_sha256",
        "signer_key_must_resolve_to_same_named_d229_registry_role",
    ):
        document = _contract()
        document["source_envelope_contract"][field] = False  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError, match="source envelopes"):
            validate_governed_acquisition_package_contract(document)


def test_batch_timestamp_requires_inclusion_and_causal_separation() -> None:
    for field in (
        "one_inclusion_proof_per_exact_envelope_leaf_required",
        "source_role_registry_keys_must_be_valid_for_entire_challenge_to_external_time_bracket",
        "future_training_and_selection_envelopes_require_a_separate_causally_later_external_time_batch",
    ):
        document = _contract()
        document["external_time_batch_contract"][field] = False  # type: ignore[index]
        with pytest.raises(GovernedAcquisitionPackageError, match="external time"):
            validate_governed_acquisition_package_contract(document)


def test_external_time_cannot_claim_exact_signature_time() -> None:
    document = _contract()
    document["external_time_batch_contract"][  # type: ignore[index]
        "proves_existence_no_later_than_external_time_not_exact_signature_creation_time"
    ] = False
    with pytest.raises(GovernedAcquisitionPackageError, match="external time"):
        validate_governed_acquisition_package_contract(document)


def test_strict_json_rejects_duplicate_fields_and_non_finite_tokens() -> None:
    with pytest.raises(GovernedAcquisitionPackageError, match="duplicate JSON field"):
        load_strict_json(b'{"x":1,"x":2}')
    with pytest.raises(GovernedAcquisitionPackageError, match="non-finite JSON token"):
        load_strict_json(b'{"x":NaN}')
