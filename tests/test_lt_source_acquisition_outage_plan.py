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
