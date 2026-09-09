from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

import pfc_shaping.validation.databricks_pfc_source_cost_preflight as subject
from pfc_shaping.validation.databricks_pfc_source_cost_preflight import (
    AUTHORITIES,
    CONTRACT_CONTENT_ID,
    CONTRACT_PATH,
    EVIDENCE_CLASS,
    EXECUTION,
    GO_AUTHORITY,
    PROFILE_BINDINGS,
    READY_STATUS,
    RECEIPT_SCHEMA,
    RUNNING_STATE,
    STOPPED_STATUS,
    UNCAPPED_STATUS,
    DatabricksPfcSourceCostPreflightError,
    canonical_sha256,
    derive_snapshot_id,
    validate_cost_preflight_contract,
    validate_cost_preflight_contract_path,
    validate_cost_preflight_receipt,
    validate_cost_preflight_receipt_path,
)

REFERENCE = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)


def _receipt() -> dict[str, object]:
    profiles = [
        {
            "profile_role": binding["profile_role"],
            "source_table": binding["source_table"],
            "query_sha256": binding["query_sha256"],
            "delta_size_bytes": 1_000,
            "delta_file_count": 2,
            "partition_columns": ["delivery_date"],
            "estimate_method": "DELTA_METADATA_UPPER_BOUND",
            "estimated_scan_bytes_lower_bound": 500,
            "estimated_scan_bytes_upper_bound": 1_000,
            "upper_bound_is_hard_cap": True,
            "pruning_proven": True,
            "planned_partition_predicate": "delivery_date >= 2026-01-01",
        }
        for binding in PROFILE_BINDINGS
    ]
    without_snapshot: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "created_at_utc": "2026-08-06T14:30:00Z",
        "request_sha256": subject.REQUEST_SHA256,
        "warehouse": {
            "warehouse_id_sha256": "a" * 64,
            "state_at_preflight": RUNNING_STATE,
            "warehouse_size": "2X-Small",
            "warehouse_type": "CLASSIC",
            "started_for_request": False,
            "auto_start_triggered": False,
        },
        "execution": {
            "control_plane_get_count": 4,
            "metadata_statement_count": 0,
            "business_profile_statement_count": 0,
            "business_row_count_opened": 0,
            "warehouse_start_count": 0,
            "databricks_write_count": 0,
            "retry_count": 0,
        },
        "profiles": profiles,
        "cost_review": {
            "cost_unit": "DBU",
            "estimated_maximum_dbu": "1",
            "proposed_maximum_dbu": "2",
            "maximum_runtime_seconds": 300,
            "cost_owner_decision": "NOT_REVIEWED",
        },
        "go_authority": dict(GO_AUTHORITY),
    }
    return {**without_snapshot, "snapshot_id": derive_snapshot_id(without_snapshot)}


def _resign(receipt: dict[str, object]) -> dict[str, object]:
    receipt["snapshot_id"] = derive_snapshot_id(
        {key: value for key, value in receipt.items() if key != "snapshot_id"}
    )
    return receipt


def _validate(receipt: dict[str, object]):
    return validate_cost_preflight_receipt(receipt, reference_time_utc=REFERENCE)


def test_contract_is_content_addressed_and_zero_remote() -> None:
    summary = validate_cost_preflight_contract_path()

    assert summary["contract_content_id"] == CONTRACT_CONTENT_ID
    assert summary["exact_profile_count"] == 4
    assert summary["phase_a_go_authorized"] is False
    assert all(value == 0 for value in EXECUTION.values())


def test_contract_rejects_semantic_mutation() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["authorities"]["phase_a_execution"] = True

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="authorities"):
        validate_cost_preflight_contract(contract)


def test_ready_receipt_still_requires_human_review_and_grants_no_go() -> None:
    result = _validate(_receipt())
    assessment = result.assessment

    assert assessment["status"] == READY_STATUS
    assert assessment["profile_count"] == 4
    assert assessment["total_delta_size_bytes"] == 4_000
    assert assessment["total_estimated_scan_bytes_upper_bound"] == 4_000
    assert assessment["human_cost_review_required"] is True
    assert assessment["phase_a_go_authorized"] is False
    assert assessment["phase_b_go_authorized"] is False
    assert assessment["model_input_authorized"] is False
    assert assessment["authorities"] == AUTHORITIES
    assert all(assessment[key] == 0 for key in EXECUTION)


def test_stopped_warehouse_returns_stop_without_starting_it() -> None:
    receipt = _receipt()
    receipt["warehouse"]["state_at_preflight"] = "STOPPED"

    result = _validate(_resign(receipt))

    assert result.assessment["status"] == STOPPED_STATUS
    assert result.assessment["warehouse_started_for_request"] is False
    assert result.assessment["phase_a_go_authorized"] is False


def test_uncapped_scan_returns_stop_and_cannot_carry_cost_claim() -> None:
    receipt = _receipt()
    receipt["profiles"][0]["upper_bound_is_hard_cap"] = False
    receipt["cost_review"]["estimated_maximum_dbu"] = None
    receipt["cost_review"]["proposed_maximum_dbu"] = None
    receipt["cost_review"]["maximum_runtime_seconds"] = 0

    result = _validate(_resign(receipt))

    assert result.assessment["status"] == UNCAPPED_STATUS
    assert result.assessment["all_scan_upper_bounds_hard_capped"] is False
    assert result.assessment["phase_a_go_authorized"] is False


def test_assessment_is_deterministic_and_content_addressed() -> None:
    first = _validate(_receipt())
    second = _validate(_receipt())

    assert first.assessment == second.assessment
    assert first.assessment_content_id == second.assessment_content_id
    assert canonical_sha256(first.assessment) == first.assessment_content_id


def test_snapshot_id_binds_every_receipt_field() -> None:
    receipt = _receipt()
    receipt["profiles"][0]["delta_size_bytes"] += 1

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="snapshot derivation"):
        _validate(receipt)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_sha256", "b" * 64, "request binding"),
        ("schema_version", "wrong", "receipt schema"),
        ("evidence_class", "BUSINESS_ROWS", "evidence class"),
    ],
)
def test_receipt_identity_mutations_fail_closed(field, value, message) -> None:
    receipt = _receipt()
    receipt[field] = value

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match=message):
        _validate(_resign(receipt))


def test_stale_future_and_naive_reference_times_fail_closed() -> None:
    stale = _receipt()
    stale["created_at_utc"] = "2026-08-05T14:59:59Z"
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="stale"):
        _validate(_resign(stale))

    future = _receipt()
    future["created_at_utc"] = "2026-08-06T15:00:01Z"
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="future"):
        _validate(_resign(future))

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="timezone-aware"):
        validate_cost_preflight_receipt(
            _receipt(),
            reference_time_utc=datetime(2026, 8, 6, 15, 0),
        )


def test_profile_inventory_missing_duplicate_and_extra_fail_closed() -> None:
    missing = _receipt()
    missing["profiles"].pop()
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="exactly four"):
        _validate(_resign(missing))

    duplicate = _receipt()
    duplicate["profiles"][1] = copy.deepcopy(duplicate["profiles"][0])
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="roles differ or repeat"):
        _validate(_resign(duplicate))

    extra = _receipt()
    extra["profiles"][0]["unexpected"] = 1
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="fields differ"):
        _validate(_resign(extra))


def test_profile_source_and_query_drift_fail_closed() -> None:
    for field, value, message in (
        ("source_table", "prd.secret.other", "source table"),
        ("query_sha256", "b" * 64, "query SHA-256"),
    ):
        receipt = _receipt()
        receipt["profiles"][0][field] = value
        with pytest.raises(DatabricksPfcSourceCostPreflightError, match=message):
            _validate(_resign(receipt))


def test_delta_size_file_and_scan_bounds_fail_closed() -> None:
    receipt = _receipt()
    receipt["profiles"][0]["delta_size_bytes"] = 0
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="size/file count conflict"):
        _validate(_resign(receipt))

    receipt = _receipt()
    receipt["profiles"][0]["estimated_scan_bytes_lower_bound"] = 1_001
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="bounds invert"):
        _validate(_resign(receipt))

    receipt = _receipt()
    receipt["profiles"][0]["estimated_scan_bytes_upper_bound"] = 0
    receipt["profiles"][0]["estimated_scan_bytes_lower_bound"] = 0
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="cap is zero"):
        _validate(_resign(receipt))


def test_unavailable_estimate_has_one_canonical_representation() -> None:
    receipt = _receipt()
    profile = receipt["profiles"][0]
    profile["estimate_method"] = "UNAVAILABLE"
    profile["estimated_scan_bytes_lower_bound"] = 0
    profile["estimated_scan_bytes_upper_bound"] = 0
    profile["upper_bound_is_hard_cap"] = False
    profile["pruning_proven"] = False
    profile["planned_partition_predicate"] = None
    receipt["cost_review"] = {
        "cost_unit": "DBU",
        "estimated_maximum_dbu": None,
        "proposed_maximum_dbu": None,
        "maximum_runtime_seconds": 0,
        "cost_owner_decision": "NOT_REVIEWED",
    }
    assert _validate(_resign(receipt)).assessment["status"] == UNCAPPED_STATUS

    profile["upper_bound_is_hard_cap"] = True
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="fields conflict"):
        _validate(_resign(receipt))


def test_pruning_requires_partition_and_exact_predicate_semantics() -> None:
    receipt = _receipt()
    receipt["profiles"][0]["partition_columns"] = []
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="lacks partitions"):
        _validate(_resign(receipt))

    receipt = _receipt()
    receipt["profiles"][0]["pruning_proven"] = False
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="must not declare"):
        _validate(_resign(receipt))

    receipt = _receipt()
    receipt["profiles"][0]["partition_columns"] = ["date", "date"]
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="partitions repeat"):
        _validate(_resign(receipt))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("metadata_statement_count", 5, "too many metadata"),
        ("business_profile_statement_count", 1, "must be zero"),
        ("business_row_count_opened", 1, "must be zero"),
        ("warehouse_start_count", 1, "must be zero"),
        ("databricks_write_count", 1, "must be zero"),
        ("retry_count", 1, "must be zero"),
    ],
)
def test_execution_escalations_fail_closed(field, value, message) -> None:
    receipt = _receipt()
    receipt["execution"][field] = value

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match=message):
        _validate(_resign(receipt))


def test_warehouse_start_auto_start_and_state_drift_fail_closed() -> None:
    for field, value, message in (
        ("started_for_request", True, "started for request"),
        ("auto_start_triggered", True, "auto-start"),
        ("state_at_preflight", "STARTING", "state differs"),
    ):
        receipt = _receipt()
        receipt["warehouse"][field] = value
        with pytest.raises(DatabricksPfcSourceCostPreflightError, match=message):
            _validate(_resign(receipt))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("estimated_maximum_dbu", "1.0", "not canonical"),
        ("estimated_maximum_dbu", "1e0", "canonical decimal"),
        ("proposed_maximum_dbu", "0.5", "below estimate"),
        ("maximum_runtime_seconds", 901, "maximum runtime differs"),
        ("cost_owner_decision", "APPROVED", "cost owner decision"),
    ],
)
def test_cost_claim_mutations_fail_closed(field, value, message) -> None:
    receipt = _receipt()
    receipt["cost_review"][field] = value

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match=message):
        _validate(_resign(receipt))


def test_go_authority_escalation_fails_closed() -> None:
    receipt = _receipt()
    receipt["go_authority"]["phase_a_go_authorized"] = True

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="GO authority"):
        _validate(_resign(receipt))


def test_receipt_path_is_build_local_strict_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_receipt(), indent=2) + "\n", encoding="utf-8")

    first = validate_cost_preflight_receipt_path(
        path.resolve(),
        reference_time_utc=REFERENCE,
    )
    second = validate_cost_preflight_receipt_path(
        path.resolve(),
        reference_time_utc=REFERENCE,
    )

    assert first.assessment_content_id == second.assessment_content_id


def test_receipt_path_rejects_outside_build_and_duplicate_json(tmp_path: Path) -> None:
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="repo-local build"):
        validate_cost_preflight_receipt_path(
            CONTRACT_PATH,
            reference_time_utc=REFERENCE,
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"x","schema_version":"y"}',
        encoding="utf-8",
    )
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="duplicate"):
        validate_cost_preflight_receipt_path(
            duplicate.resolve(),
            reference_time_utc=REFERENCE,
        )


def test_receipt_path_rejects_hardlink(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    original.write_text(json.dumps(_receipt()), encoding="utf-8")
    linked = tmp_path / "linked.json"
    os.link(original, linked)

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="read failed"):
        validate_cost_preflight_receipt_path(
            linked.resolve(),
            reference_time_utc=REFERENCE,
        )


def test_receipt_mid_validation_mutation_fails_closed(tmp_path: Path, monkeypatch) -> None:
    path = (tmp_path / "receipt.json").resolve()
    payload = (json.dumps(_receipt()) + "\n").encode()
    path.write_bytes(payload)
    original_read = subject._read
    receipt_reads = 0

    def changing_read(candidate: Path, label: str, maximum_bytes: int) -> bytes:
        nonlocal receipt_reads
        value = original_read(candidate, label, maximum_bytes)
        if candidate == path:
            receipt_reads += 1
            if receipt_reads == 2:
                return value + b" "
        return value

    monkeypatch.setattr(subject, "_read", changing_read)
    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="changed during"):
        validate_cost_preflight_receipt_path(path, reference_time_utc=REFERENCE)


def test_mutated_contract_path_fails_before_receipt_review(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["receipt_policy"]["phase_a_go_authorized"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(DatabricksPfcSourceCostPreflightError, match="SHA-256 differs"):
        validate_cost_preflight_receipt(
            _receipt(),
            reference_time_utc=REFERENCE,
            contract_path=path.resolve(),
        )


def test_module_has_no_ct_or_databricks_client_import() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")

    assert "pfc_shaping.ct" not in source
    assert "databricks.sql" not in source
    assert "requests" not in source
