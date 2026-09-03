from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def _canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_plan_is_authority_and_execution_negative() -> None:
    plan = _plan()

    execution = plan["execution"]
    authorities = plan["authorities"]
    assert isinstance(execution, dict)
    assert isinstance(authorities, dict)
    assert execution and all(type(value) is int and value == 0 for value in execution.values())
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
    assert entsoe["local_cost_preflight"] == "STOP_NO_ACTIVE_WAREHOUSE"
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
    assert plan_request["series_selection_status"] == "REQUEST_PREPARED_RESPONSE_NOT_RECEIVED"
    assert request["status"] == (
        "REQUEST_PREPARED_RESPONSE_NOT_RECEIVED_NO_EXECUTION_OR_MODEL_AUTHORITY"
    )
    assert plan["completed_local_steps"] == [
        "REUSE_AND_BIND_EXISTING_EEX_CAPTURE_WITHOUT_NEW_QUERY",
        "PREPARE_ENTSOE_EFFECTIVE_DATED_SERIES_SELECTION_REQUEST",
    ]
    assert plan["execution_order"][0] == (
        "TRANSMIT_REQUEST_AND_OBTAIN_ENTSOE_EFFECTIVE_DATED_SERIES_SELECTION_EVIDENCE"
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


def test_entsoe_selection_request_matches_the_runtime_candidate_inventory() -> None:
    from pfc_shaping.validation import entsoe_day_ahead_export

    request = _selection_request()
    decisions = request["owner_decisions_required"]
    fixed = request["fixed_series"]
    assert isinstance(decisions, list)
    assert isinstance(fixed, dict)

    requested_candidates = {
        decision["field_name"]: set(decision["candidate_series_keys"])
        for decision in decisions
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


def test_lseg_waits_for_exact_entsoe_frame_without_substitution() -> None:
    lseg = _lane("lseg")
    assert lseg["status"] == "WAIT_FOR_EXACT_ENTSOE_REALIZED_FRAME"
    assert lseg["new_query_authorized"] is False
    assert lseg["silent_source_substitution_authorized"] is False
    assert lseg["it_north_covered_by_lseg"] is False
