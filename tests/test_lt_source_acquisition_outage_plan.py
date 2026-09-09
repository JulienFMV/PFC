from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "LT-SOURCE-ACQUISITION-OUTAGE-PLAN-20260902.json"
)
SELECTION_REQUEST_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-REQUEST-V1-20260903.json"
)
SELECTION_EVIDENCE_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DAY-AHEAD-EFFECTIVE-SERIES-SELECTION-EVIDENCE-V1-20260903.json"
)
EXPORT_PREFLIGHT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DAY-AHEAD-REALIZED-FINAL-EXPORT-PREFLIGHT-V1-20260904.json"
)
FRESHNESS_COST_CHECK_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DAY-AHEAD-FRESHNESS-COST-CHECK-V1-20260904.json"
)
JULY_CANDIDATE_CAPTURE_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DAY-AHEAD-JULY-CANDIDATE-CAPTURE-V1-20260904.json"
)
JULY_LATEST_CANDIDATE_REPLAY_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DAY-AHEAD-JULY-LATEST-CANDIDATE-REPLAY-V1-20260904.json"
)
JULY_LSEG_RECONCILIATION_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "LSEG-ENTSOE-JULY-LATEST-RECONCILIATION-V1-20260904.json"
)


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _lane(name: str) -> dict[str, object]:
    lanes = _plan()["lanes"]
    assert isinstance(lanes, dict)
    lane = lanes[name]
    assert isinstance(lane, dict)
    return lane


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_request() -> dict[str, object]:
    return json.loads(SELECTION_REQUEST_PATH.read_text(encoding="utf-8"))


def _selection_evidence() -> dict[str, object]:
    return json.loads(SELECTION_EVIDENCE_PATH.read_text(encoding="utf-8"))


def _export_preflight() -> dict[str, object]:
    return json.loads(EXPORT_PREFLIGHT_PATH.read_text(encoding="utf-8"))


def _freshness_cost_check() -> dict[str, object]:
    return json.loads(FRESHNESS_COST_CHECK_PATH.read_text(encoding="utf-8"))


def _july_candidate_capture() -> dict[str, object]:
    return json.loads(JULY_CANDIDATE_CAPTURE_PATH.read_text(encoding="utf-8"))


def _july_latest_candidate_replay() -> dict[str, object]:
    return json.loads(JULY_LATEST_CANDIDATE_REPLAY_PATH.read_text(encoding="utf-8"))


def _july_lseg_reconciliation() -> dict[str, object]:
    return json.loads(JULY_LSEG_RECONCILIATION_PATH.read_text(encoding="utf-8"))


def _canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_plan_records_bounded_comparison_and_coordination_with_negative_authority() -> None:
    plan = _plan()

    execution = plan["execution"]
    authorities = plan["authorities"]
    assert isinstance(execution, dict)
    assert isinstance(authorities, dict)
    assert execution == {
        "databricks_api_requests": 101,
        "databricks_statements": 8,
        "warehouse_starts": 0,
        "warehouse_resizes": 0,
        "warehouse_creates": 0,
        "warehouse_stop_requests": 1,
        "warehouse_stop_successes": 0,
        "aggregate_business_value_comparisons": 2,
        "raw_business_rows_returned": 21_755,
        "remote_writes": 4,
    }
    assert authorities and all(value is False for value in authorities.values())
    assert plan["status"] == "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS"


def test_public_outage_observation_grants_no_evidence_or_execution() -> None:
    observation = _plan()["public_service_observation"]
    assert isinstance(observation, dict)
    assert observation["status"] == "EXTERNAL_CORROBORATION_NOT_GOVERNANCE_EVIDENCE"
    for field in (
        "proves_internal_silver_unavailability",
        "proves_source_freshness",
        "proves_original_publication",
        "proves_finality",
        "authorizes_execution",
    ):
        assert observation[field] is False


def test_existing_eex_capture_matches_the_bound_local_bytes() -> None:
    eex = _lane("eex")
    for role in ("snapshot", "manifest"):
        artifact = eex[role]
        assert isinstance(artifact, dict)
        path = ROOT / str(artifact["path"])
        if not path.exists():
            pytest.skip("bound local EEX capture bytes are absent from this checkout")
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]

    assert eex["reuse_only"] is True
    assert eex["new_databricks_statement_authorized"] is False


def test_eex_query_provenance_completion_does_not_close_remaining_authority_gaps() -> None:
    eex = _lane("eex")
    provenance = eex["bound_query_provenance"]
    assert isinstance(provenance, dict)
    assert provenance == {
        "status": "VERIFIED_LOCALLY_WITHOUT_OPENING_BUSINESS_ROWS",
        "decision_id": "D-20260903-280",
        "query_size_bytes": 625,
        "query_sha256": "54a2e7e1752af4506673d2b5cbc2666f0deea45ec96e6d82561e4b265c78797a",
        "manifest_sha256": eex["manifest"]["sha256"],
        "artifact_sha256": eex["snapshot"]["sha256"],
    }
    assert eex["remaining_evidence"] == [
        "INDEPENDENT_SOURCE_TIME_EVIDENCE",
        "SIGNED_ENVELOPES_AND_EXTERNAL_TIME",
        "CONVERSION_TO_EXISTING_SIGNED_EEX_VINTAGE_CATALOG",
    ]


def test_entsoe_sql_bindings_match_and_no_warehouse_start_is_authorized() -> None:
    entsoe = _lane("entsoe")
    bindings = entsoe["sql_bindings"]
    assert isinstance(bindings, dict)
    assert (
        _sha256(ROOT / "docs/data/sql/databricks_prd_entsoe_day_ahead_causal_export_v2.sql")
        == bindings["causal_asof_sha256"]
    )
    assert (
        _sha256(ROOT / "docs/data/sql/databricks_prd_entsoe_day_ahead_realized_export_v2.sql")
        == bindings["realized_final_sha256"]
    )
    assert entsoe["local_cost_preflight"] == (
        "PASS_USER_ACCEPTED_1548216106_BYTE_HARD_STORAGE_BOUND_ACTUAL_EXPORT_READ_31967426_BYTES"
    )
    assert entsoe["internal_silver_snapshot_condition"] == (
        "JULY_EXACT_FULL_WINDOW_COVERAGE_VERIFIED_AND_LATEST_REVISION_CANDIDATE_CAPTURED"
    )
    assert entsoe["warehouse_start_authorized"] is False


def test_first_entsoe_request_is_realized_smoke_not_holdout_or_model_input() -> None:
    request = _lane("entsoe")["first_delivery_request"]
    assert isinstance(request, dict)
    assert request["usage"] == "realized_final"
    assert request["window_start_utc"] == "2026-06-30T22:00:00Z"
    assert request["window_end_utc"] == "2026-07-31T22:00:00Z"
    assert request["utc_partitions"] == ["2026-06", "2026-07"]
    assert request["requires_source_refresh"] is False
    assert request["requires_value_bound_finality_evidence"] is True
    assert request["holdout_authorized"] is False
    assert request["model_input_authorized"] is False
    assert request["execution_authorized_by_this_plan"] is False


def test_entsoe_selection_request_is_exactly_bound_and_authority_negative() -> None:
    plan = _plan()
    lanes = plan["lanes"]
    assert isinstance(lanes, dict)
    entsoe = lanes["entsoe"]
    assert isinstance(entsoe, dict)
    plan_request = entsoe["first_delivery_request"]
    assert isinstance(plan_request, dict)
    binding = plan_request["series_selection_request"]
    assert isinstance(binding, dict)

    request = _selection_request()
    assert binding["path"] == str(SELECTION_REQUEST_PATH.relative_to(ROOT)).replace("\\", "/")
    assert binding["canonical_json_sha256"] == _canonical_json_sha256(request)
    assert binding["owner_response_received"] is False
    assert binding["selection_authorized"] is False
    assert plan_request["series_selection_status"] == (
        "CONSTRUCTION_REFERENCE_SELECTION_FROZEN_FROM_EXACT_LSEG_PARITY"
    )
    assert request["status"] == (
        "REQUEST_PREPARED_RESPONSE_NOT_RECEIVED_NO_EXECUTION_OR_MODEL_AUTHORITY"
    )
    assert plan["completed_local_steps"] == [
        "REUSE_AND_BIND_EXISTING_EEX_CAPTURE_WITHOUT_NEW_QUERY",
        "PREPARE_ENTSOE_EFFECTIVE_DATED_SERIES_SELECTION_REQUEST",
        "RESOLVE_ENTSOE_AT_AND_DE_LU_CONSTRUCTION_REFERENCE_BY_EXACT_LSEG_PARITY",
        "PREPARE_EXACT_ENTSOE_JULY_REALIZED_EXPORT_AND_RUN_METADATA_COST_PREFLIGHT",
        "OPEN_PLATFORM_COST_QUOTE_REQUEST_WITHOUT_EXECUTION_AUTHORITY",
        "RUN_AUTHORIZED_VALUE_BLIND_FRESHNESS_AND_COST_CHECK_AND_REPORT_CH_GAP",
        "VERIFY_EXACT_JULY_COVERAGE_AND_CAPTURE_QUARANTINED_LATEST_REVISION_CANDIDATE",
        "BUILD_AND_VERIFY_AUTHORITY_NEGATIVE_REALIZED_LATEST_CANDIDATE_REPLAY",
        "RECONCILE_JULY_ENTSOE_LATEST_CANDIDATE_EXACTLY_AGAINST_BOUNDED_LSEG_LATEST",
    ]
    assert plan["execution_order"][0] == (
        "TRACK_SEPTEMBER_CH_BACKFILL_SEPARATELY_ON_FMVSA_OPENDATA_LAKEHOUSE_ISSUE_4"
    )
    assert plan["execution_order"][1] == (
        "PROMOTE_THE_SAME_CAPTURED_BYTES_TO_REALIZED_FINAL_ONLY_IF_EXTERNAL_"
        "EVIDENCE_BINDS_EXACT_SEMANTIC_HASH_WINDOW_AND_SERIES_KEYS"
    )

    scope = request["request_scope"]
    assert isinstance(scope, dict)
    assert scope["window_start_utc"] == plan_request["window_start_utc"]
    assert scope["window_end_utc"] == plan_request["window_end_utc"]
    assert scope["utc_partitions"] == plan_request["utc_partitions"]
    assert scope["metadata_only"] is True
    assert scope["business_values_requested"] is False

    execution = request["execution"]
    authorities = request["authorities"]
    assert isinstance(execution, dict)
    assert isinstance(authorities, dict)
    assert execution and all(type(value) is int and value == 0 for value in execution.values())
    assert authorities and all(value is False for value in authorities.values())


def test_entsoe_selection_evidence_freezes_sequence_one_from_exact_lseg_parity() -> None:
    plan_request = _lane("entsoe")["first_delivery_request"]
    assert isinstance(plan_request, dict)
    binding = plan_request["series_selection_evidence"]
    assert isinstance(binding, dict)

    evidence = _selection_evidence()
    assert binding["path"] == str(SELECTION_EVIDENCE_PATH.relative_to(ROOT)).replace("\\", "/")
    assert binding["canonical_json_sha256"] == _canonical_json_sha256(evidence)
    assert plan_request["series_selection_status"] == (
        "CONSTRUCTION_REFERENCE_SELECTION_FROZEN_FROM_EXACT_LSEG_PARITY"
    )
    assert binding["construction_smoke_export_selection_frozen"] is True
    assert binding["governed_model_selection_authorized"] is False

    selected = evidence["construction_reference_selection"]
    assert isinstance(selected, dict)
    assert selected == {
        "at_price": "day_ahead_prices||at_price||1",
        "de_lu_price": "day_ahead_prices||de_lu_price||1",
        "selection_basis": "EXACT_FULL_WINDOW_LSEG_EPEX_ACTUAL_PRICE_PARITY",
        "in_window_change_observed": False,
        "construction_smoke_export_selection_frozen": True,
    }

    metrics = evidence["comparison_metrics"]
    assert isinstance(metrics, list)
    assert len(metrics) == 4
    by_identity = {(row["field_name"], row["classification_sequence"]): row for row in metrics}
    for field in ("at_price", "de_lu_price"):
        selected_metrics = by_identity[(field, "1")]
        assert selected_metrics["entsoe_quarter_hours"] == 2976
        assert selected_metrics["matched_lseg_quarter_hours"] == 2976
        assert selected_metrics["missing_lseg_quarter_hours"] == 0
        assert selected_metrics["overlapping_quarter_hours"] == 0
        assert selected_metrics["mae_eur_per_mwh"] == 0.0
        assert selected_metrics["rmse_eur_per_mwh"] == 0.0
        assert selected_metrics["correlation"] == 1.0
        assert selected_metrics["equal_to_half_cent_quarter_hours"] == 2976
        assert by_identity[(field, "2")]["mae_eur_per_mwh"] > 9.0

    limitations = evidence["limitations"]
    execution = evidence["comparison_execution"]
    authorities = evidence["authorities"]
    assert isinstance(limitations, dict)
    assert isinstance(execution, dict)
    assert isinstance(authorities, dict)
    assert execution["state"] == "SUCCEEDED"
    assert execution["statement_count"] == 1
    assert execution["result_row_count"] == 4
    assert execution["result_truncated"] is False
    assert execution["warehouse_state_before"] == "RUNNING"
    assert execution["warehouse_start_count"] == 0
    assert execution["raw_price_rows_returned"] == 0
    assert execution["read_bytes"] == 9_364_142_086
    assert execution["write_remote_bytes"] == 0
    assert limitations["owner_response_received"] is False
    assert limitations["executed_query_text_hash_verified"] is False
    assert limitations["realized_finality_proven"] is False
    assert authorities and all(value is False for value in authorities.values())


def test_realized_export_preflight_is_exact_stopped_and_authority_negative() -> None:
    plan_request = _lane("entsoe")["first_delivery_request"]
    plan_preflight = _lane("entsoe")["realized_final_export_preflight"]
    preflight = _export_preflight()
    assert isinstance(plan_request, dict)
    assert isinstance(plan_preflight, dict)

    assert plan_preflight["path"] == str(EXPORT_PREFLIGHT_PATH.relative_to(ROOT)).replace("\\", "/")
    assert plan_preflight["canonical_json_sha256"] == _canonical_json_sha256(preflight)
    assert preflight["status"] == "STOP_NO_ACTIVE_WAREHOUSE_AND_SCAN_BOUND_UNPROVEN"
    assert preflight["request_scope"]["window_start_utc"] == plan_request["window_start_utc"]
    assert preflight["request_scope"]["window_end_utc"] == plan_request["window_end_utc"]
    assert preflight["request_scope"]["utc_partitions"] == plan_request["utc_partitions"]
    assert (
        preflight["query_binding"]["sha256"]
        == _lane("entsoe")["sql_bindings"]["realized_final_sha256"]
    )
    assert preflight["series_selection"] == {
        "ch_price": "day_ahead_prices||ch_price",
        "at_price": "day_ahead_prices||at_price||1",
        "de_lu_price": "day_ahead_prices||de_lu_price||1",
        "fr_price": "day_ahead_prices||fr_price",
        "it_nord_price": "day_ahead_prices||it_nord_price",
    }

    live = preflight["live_metadata_preflight"]
    cost = preflight["cost_fence"]
    execution = live["execution"]
    assert live["warehouse"]["state"] == "STOPPED"
    assert live["warehouse"]["started_for_request"] is False
    assert live["source_table"]["partition_columns"] == ["_year", "_month"]
    assert execution == {
        "control_plane_get_count": 3,
        "databricks_statement_count": 0,
        "warehouse_start_count": 0,
        "warehouse_resize_count": 0,
        "warehouse_create_count": 0,
        "business_row_count_opened": 0,
        "remote_write_count": 0,
    }
    assert cost["physical_partition_pruning_proven"] is False
    assert cost["scan_upper_bound_is_hard"] is False
    assert cost["estimated_scan_upper_bound_bytes"] is None
    assert cost["maximum_scan_bytes_approved"] is None
    assert cost["prohibited_comparison_reference"]["must_not_be_repeated"] is True
    assert preflight["authorities"] and all(
        value is False for value in preflight["authorities"].values()
    )


def test_platform_cost_request_is_coordination_only_and_grants_no_execution() -> None:
    entsoe = _lane("entsoe")
    coordination = entsoe["platform_coordination"]
    assert entsoe["status"] == (
        "JULY_LATEST_CANDIDATE_RECONCILED_EXACTLY_PENDING_OPTIONAL_FINALITY_AND_"
        "SEPTEMBER_CH_BACKFILL"
    )
    assert coordination == {
        "repository": "FMVSA/opendata-lakehouse",
        "issue_number": 4,
        "url": "https://github.com/FMVSA/opendata-lakehouse/issues/4",
        "title": "[LT] Cost quote for bounded July 2026 ENTSO-E realized_final export",
        "created_at_utc": "2026-09-04T07:38:22Z",
        "created_by": "JulienFMV",
        "state": "OPEN",
        "assignee_count": 0,
        "comment_count_at_verification": 3,
        "latest_comment_url": (
            "https://github.com/FMVSA/opendata-lakehouse/issues/4#issuecomment-5538187585"
        ),
        "request_scope": (
            "OPTIONAL_VALUE_BOUND_FINALITY_PROMOTION_FOR_CAPTURED_JULY_CANDIDATE_"
            "AND_SEPARATE_SEPTEMBER_CH_BACKFILL_STATUS"
        ),
        "warehouse_start_authorized": False,
        "sql_execution_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def test_freshness_cost_check_binds_low_cost_and_visible_ch_gap() -> None:
    entsoe = _lane("entsoe")
    binding = entsoe["freshness_cost_check"]
    check = _freshness_cost_check()
    assert binding["path"] == str(FRESHNESS_COST_CHECK_PATH.relative_to(ROOT)).replace("\\", "/")
    assert binding["canonical_json_sha256"] == _canonical_json_sha256(check)
    assert check["status"] == (
        "TABLE_UPDATED_TODAY_CH_DELIVERY_GAP_OBSERVED_WAREHOUSE_STOP_FORBIDDEN"
    )
    assert check["delta_detail"]["last_modified_utc"] == "2026-09-04T06:42:56Z"
    assert check["delta_detail"]["size_in_bytes"] == 1_548_216_106
    assert check["delta_detail"]["num_files"] == 137
    assert check["bounded_freshness_query"]["read_bytes"] == 7_798_287
    assert check["bounded_freshness_query"]["price_columns_selected"] is False
    assert check["bounded_freshness_query"]["write_remote_bytes"] == 0

    watermarks = {row["field_name"]: row for row in check["selected_series_watermarks"]}
    assert watermarks["ch_price"]["max_interval_end_utc"] == "2026-09-02T22:00:00Z"
    for field in ("at_price", "de_lu_price", "fr_price", "it_nord_price"):
        assert watermarks[field]["max_interval_end_utc"] == "2026-09-04T22:00:00Z"
    assert all(row["dq_failed_count"] == 0 for row in watermarks.values())

    execution = check["execution"]
    assert execution["databricks_statement_count"] == 2
    assert execution["warehouse_start_request_count"] == 0
    assert execution["warehouse_stop_request_count"] == 1
    assert execution["warehouse_stop_success_count"] == 0
    assert execution["business_price_rows_returned"] == 0
    assert check["warehouse"]["stop_request_status"] == "REJECTED_HTTP_403"
    assert check["authorities"] and all(value is False for value in check["authorities"].values())


def test_july_candidate_capture_is_complete_quarantined_and_finality_negative() -> None:
    entsoe = _lane("entsoe")
    binding = entsoe["july_candidate_capture"]
    capture = _july_candidate_capture()
    assert binding["path"] == str(JULY_CANDIDATE_CAPTURE_PATH.relative_to(ROOT)).replace("\\", "/")
    assert binding["canonical_json_sha256"] == _canonical_json_sha256(capture)
    assert capture["status"] == (
        "PASS_QUARANTINED_LATEST_REVISION_CANDIDATE_PENDING_VALUE_BOUND_FINALITY"
    )
    assert capture["scope"]["expected_quarter_hours_per_series"] == 2_976
    assert capture["scope"]["series_selection"] == {
        "ch_price": "day_ahead_prices||ch_price",
        "at_price": "day_ahead_prices||at_price||1",
        "de_lu_price": "day_ahead_prices||de_lu_price||1",
        "fr_price": "day_ahead_prices||fr_price",
        "it_nord_price": "day_ahead_prices||it_nord_price",
    }

    coverage = capture["normalized_coverage"]
    assert coverage["status"] == "PASS_EXACT_FULL_WINDOW_VALUE_BLIND_COVERAGE"
    assert coverage["business_value_columns_opened"] == 0
    assert coverage["query_metrics"]["read_bytes"] == 65_771_005
    assert len(coverage["series"]) == 5
    for series in coverage["series"]:
        assert series["expanded_quarter_hour_count"] == 2_976
        assert series["missing_quarter_hour_count"] == 0
        assert series["overlap_quarter_hour_count"] == 0
        assert series["invalid_native_row_count"] == 0

    candidate = capture["candidate_export"]
    assert candidate["status"] == "QUARANTINED_LATEST_REVISION_CANDIDATE_PENDING_FINALITY"
    assert candidate["query_metrics"]["read_bytes"] == 31_967_426
    assert candidate["raw_row_count"] == 12_083
    assert candidate["parquet_size_bytes"] == 564_727
    assert candidate["parquet_sha256"] == (
        "046ae86ab84a72c61ea44547cc386f85e49d37d325352f4bfca208a08e5a9baa"
    )
    assert candidate["semantic_sha256"] == (
        "5a6d72b5531e843dc4e7920c519cc3d93189e70d91277662a763f6575967374e"
    )
    assert candidate["price_values_persisted_below_build_only"] is True
    assert candidate["price_values_printed_or_committed"] is False

    validation = capture["local_validation"]
    assert validation["status"] == ("PASS_RAW_EXPORT_CONTRACT_BLOCKED_ONLY_ON_FINALITY_EVIDENCE")
    assert validation["blocking_error"] == ("evidence does not exactly cover selected SeriesKeys")
    assert validation["realized_final_validation_complete"] is False
    assert (
        capture["next_required_evidence"]["must_bind_candidate_semantic_sha256"]
        == (candidate["semantic_sha256"])
    )
    assert capture["next_required_evidence"]["it_north_requires_platform_authority"] is True
    assert capture["authorities"] and all(
        value is False for value in capture["authorities"].values()
    )


def test_july_latest_candidate_replay_advances_without_claiming_finality() -> None:
    entsoe = _lane("entsoe")
    binding = entsoe["july_latest_candidate_replay"]
    replay = _july_latest_candidate_replay()

    assert binding["path"] == str(JULY_LATEST_CANDIDATE_REPLAY_PATH.relative_to(ROOT)).replace(
        "\\", "/"
    )
    assert binding["canonical_json_sha256"] == _canonical_json_sha256(replay)
    assert replay["status"] == (
        "PASS_VERIFIED_REALIZED_LATEST_CANDIDATE_REPLAY_NOT_FINAL_NOT_MODEL_AUTHORITY"
    )
    assert replay["source_contract_audit"]["signed_finality_service_found"] is False
    assert replay["captured_source"]["warehouse_rerun_required"] is False
    assert replay["local_replay"]["usage"] == "realized_latest_candidate"
    assert replay["local_replay"]["verification_status"] == (
        "VERIFIED_SELF_CONTAINED_DAY_AHEAD_EXPORT_REPLAY"
    )
    assert replay["local_replay"]["raw_row_count"] == 12_083
    assert replay["local_replay"]["consumer_row_count"] == 12_083
    assert replay["local_replay"]["is_final"] is False
    assert replay["realized_final_promotion"]["required_for_local_replay"] is False
    assert (
        replay["realized_final_promotion"]["external_value_bound_finality_evidence_still_required"]
        is True
    )
    assert replay["execution"]["remote_writes"] == 1
    assert all(value == 0 for key, value in replay["execution"].items() if key != "remote_writes")
    assert replay["authorities"] and all(value is False for value in replay["authorities"].values())


def test_entsoe_selection_request_matches_the_runtime_candidate_inventory() -> None:
    from pfc_shaping.validation import entsoe_day_ahead_export

    request = _selection_request()
    decisions = request["owner_decisions_required"]
    fixed = request["fixed_series"]
    assert isinstance(decisions, list)
    assert isinstance(fixed, dict)

    requested_candidates = {
        decision["field_name"]: set(decision["candidate_series_keys"]) for decision in decisions
    }
    assert requested_candidates == {
        "at_price": set(entsoe_day_ahead_export._SERIES_CANDIDATES["at_price"]),
        "de_lu_price": set(entsoe_day_ahead_export._SERIES_CANDIDATES["de_lu_price"]),
    }
    assert fixed == {
        field: next(iter(entsoe_day_ahead_export._SERIES_CANDIDATES[field]))
        for field in ("ch_price", "fr_price", "it_nord_price")
    }

    response = request["required_response_contract"]
    assert isinstance(response, dict)
    assert response["exactly_one_rule_per_field"] is True
    assert response["each_rule_must_equal_request_window"] is True
    assert response["in_window_series_change_authorized"] is False
    assert response["exact_window_coverage_required"] is True
    assert response["gaps_authorized"] is False
    assert response["overlaps_authorized"] is False
    assert response["implicit_default_authorized"] is False
    assert response["candidate_averaging_authorized"] is False
    assert response["consumer_inference_authorized"] is False


def test_each_requested_entsoe_candidate_pair_builds_the_exact_bounded_parameters() -> None:
    from itertools import product

    from pfc_shaping.validation.entsoe_day_ahead_export import (
        build_realized_export_parameters,
    )

    request = _selection_request()
    scope = request["request_scope"]
    fixed = request["fixed_series"]
    decisions = request["owner_decisions_required"]
    assert isinstance(scope, dict)
    assert isinstance(fixed, dict)
    assert isinstance(decisions, list)

    candidate_sets = [decision["candidate_series_keys"] for decision in decisions]
    for at_key, de_lu_key in product(*candidate_sets):
        selection = {**fixed, "at_price": at_key, "de_lu_price": de_lu_key}
        parameters = build_realized_export_parameters(
            series_selection=selection,
            window_start_utc=scope["window_start_utc"],
            window_end_utc=scope["window_end_utc"],
            assessed_at_utc="2026-09-03T00:00:00Z",
        )
        assert parameters == {
            "start_utc": scope["window_start_utc"],
            "end_utc": scope["window_end_utc"],
            "partition_start_year": 2026,
            "partition_start_month": 6,
            "partition_end_year": 2026,
            "partition_end_month": 7,
            "assessed_at_utc": "2026-09-03T00:00:00Z",
            "ch_series_key": fixed["ch_price"],
            "at_series_key": at_key,
            "de_lu_series_key": de_lu_key,
            "fr_series_key": fixed["fr_price"],
            "it_nord_series_key": fixed["it_nord_price"],
        }


def test_historical_backfill_cannot_be_relabelled_as_causal_truth() -> None:
    causal = _lane("entsoe")["causal_history"]
    assert isinstance(causal, dict)
    assert causal["status"] == "BLOCKED_NO_RETROACTIVE_CAUSALITY"
    assert causal["july_2026_backfill_may_be_used_as_causal_truth"] is False
    assert causal["future_causal_capture_requires_producer_recovery"] is True
    assert causal["future_causal_capture_requires_frozen_origin_protocol"] is True


def test_lseg_reconciliation_passed_without_expanding_its_authority() -> None:
    lseg = _lane("lseg")
    assert lseg["status"] == "JULY_LATEST_RECONCILIATION_PASS_EXACT_NO_MODEL_AUTHORITY"
    assert lseg["new_query_authorized"] is False
    assert lseg["silent_source_substitution_authorized"] is False
    assert lseg["it_north_covered_by_lseg"] is False
    assert lseg["point_in_time_extract_used"] is False
    assert lseg["realized_finality_proven"] is False


def test_july_lseg_reconciliation_is_hash_bound_exact_and_authority_negative() -> None:
    entsoe = _lane("entsoe")
    binding = entsoe["july_lseg_reconciliation"]
    evidence = _july_lseg_reconciliation()
    assert isinstance(binding, dict)

    assert binding["path"] == str(JULY_LSEG_RECONCILIATION_PATH.relative_to(ROOT)).replace(
        "\\", "/"
    )
    assert binding["canonical_json_sha256"] == _canonical_json_sha256(evidence)
    assert evidence["status"] == (
        "PASS_EXACT_LATEST_SOURCE_RECONCILIATION_NOT_FINALITY_OR_MODEL_AUTHORITY"
    )
    assert evidence["query_contract"]["sha256"] == _sha256(
        ROOT / evidence["query_contract"]["path"]
    )

    preflight = evidence["preflight"]
    assert preflight["rejected_vintage_table"]["statistics_total_size_bytes"] == 11_544_235_073
    assert preflight["rejected_vintage_table"]["queried"] is False
    assert preflight["selected_latest_table"]["statistics_total_size_bytes"] == 16_758_198
    assert preflight["maximum_scan_bytes"] == 33_554_432

    successful = evidence["capture_execution"]["successful_statement"]
    assert successful["read_bytes"] == 13_141_107
    assert successful["rows_produced_count"] == 9_672
    assert successful["write_remote_bytes"] == 0
    assert evidence["capture_execution"]["warehouse_start_count"] == 0

    policy = evidence["reconciliation_policy"]
    assert policy["frozen_before_business_values_were_compared"] is True
    assert policy["thresholds_changed_after_result"] is False
    assert policy["minimum_matched_hours_per_zone"] == 744
    assert policy["minimum_hourly_overlap_ratio"] == 1.0

    result = evidence["reconciliation_result"]
    for metrics in result["per_zone"].values():
        assert metrics["matched_hours"] == 744
        assert metrics["entsoe_missing_hours"] == 0
        assert metrics["lseg_missing_hours"] == 0
        assert metrics["hourly_overlap_ratio"] == 1.0
        assert metrics["maximum_abs_difference_eur_per_mwh"] == 0.0
    assert result["it_north"]["entsoe_complete_hours"] == 744
    assert result["it_north"]["lseg_crosscheck_available"] is False

    assert evidence["authorities"]["source_reconciliation_validated"] is True
    for name, value in evidence["authorities"].items():
        if name != "source_reconciliation_validated":
            assert value is False
