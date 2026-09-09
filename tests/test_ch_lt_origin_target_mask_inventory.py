from __future__ import annotations

import json
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory as packaged_cli
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_origin_target_mask_inventory import (
    ASSESSMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    STATUS,
    ChLtOriginTargetMaskInventoryError,
    assess_structural_inventory,
    build_structural_inventory,
    canonical_json_bytes,
    compute_inventory_id,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "2026-07-30T12:00:00Z"


@pytest.fixture(scope="module")
def inventory() -> dict[str, object]:
    return build_structural_inventory(repo_root=ROOT, origin_as_of_utc=ORIGIN)


def _rebound(document: dict[str, object]) -> tuple[dict[str, object], bytes]:
    document["inventory_id"] = compute_inventory_id(document)
    payload = canonical_json_bytes(document) + b"\n"
    return document, payload


def test_inventory_fixes_exact_full_local_months_and_dst_counts(
    inventory: dict[str, object],
) -> None:
    result = assess_structural_inventory(
        inventory, document_bytes=canonical_json_bytes(inventory) + b"\n"
    )
    assert inventory["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == "ch_lt_origin_target_mask_inventory.v2"
    assert result["schema_version"] == ASSESSMENT_SCHEMA_VERSION
    assert inventory["inventory_name"].endswith("_v2")
    assert result["status"] == STATUS
    assert result["origin_as_of_utc"] == ORIGIN
    assert result["target_count"] == 36
    assert result["countable_prospective_origin"] is False
    targets = inventory["targets"]
    assert isinstance(targets, list)
    assert targets[0]["delivery_month"] == "2026-08"
    assert targets[-1]["delivery_month"] == "2029-07"
    assert [targets[index]["lead_bucket"] for index in (0, 5, 6, 11, 12, 23, 24, 35)] == [
        "M01_M06",
        "M01_M06",
        "M07_M12",
        "M07_M12",
        "M13_M24",
        "M13_M24",
        "M25_M36",
        "M25_M36",
    ]
    march_2027 = targets[7]
    october_2027 = targets[14]
    assert march_2027["delivery_month"] == "2027-03"
    assert march_2027["expected_native_hourly_interval_count"] == 743
    assert march_2027["expected_quarter_hour_interval_count"] == 2972
    assert october_2027["delivery_month"] == "2027-10"
    assert october_2027["expected_native_hourly_interval_count"] == 745
    assert october_2027["expected_quarter_hour_interval_count"] == 2980


def test_inventory_is_outcome_blind_and_cannot_claim_authority(
    inventory: dict[str, object],
) -> None:
    targets = inventory["targets"]
    assert isinstance(targets, list)
    assert all(
        row["truth_state"] == "NOT_AVAILABLE_OR_NOT_BOUND_NOT_READ" for row in targets
    )
    assert all(
        row["prediction_state"] == "NOT_CAPTURED_STRUCTURAL_DRY_RUN" for row in targets
    )
    assert all(row["evaluation_eligible"] is False for row in targets)
    assert all(
        row["structural_masks"]["QUARTER_HOURLY_SHAPE"]
        == "UNSUPPORTED_MARKET_TRANSITION_NOT_ADMITTED"
        for row in targets
    )
    for field in (
        "evidence_slot_satisfied",
        "predictions_sealed",
        "outcomes_opened",
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        assert inventory[field] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dst_count", "target M08"),
        ("qh_upgrade", "target M01"),
        ("authority", "scientific_admission"),
        ("truth_sealed", "target M01"),
        ("extra_outcome", "target M01 keys are not exact"),
    ],
)
def test_rehashed_semantic_attacks_fail_closed(
    inventory: dict[str, object], mutation: str, message: str
) -> None:
    attacked = json.loads(json.dumps(inventory))
    targets = attacked["targets"]
    assert isinstance(targets, list) and isinstance(targets[0], dict)
    if mutation == "dst_count":
        targets[7]["expected_native_hourly_interval_count"] = 744
    elif mutation == "qh_upgrade":
        targets[0]["structural_masks"]["QUARTER_HOURLY_SHAPE"] = "ELIGIBLE"
    elif mutation == "authority":
        attacked["scientific_admission"] = True
    elif mutation == "truth_sealed":
        targets[0]["truth_state"] = "SEALED_UNOPENED_NOT_READ"
    else:
        targets[0]["outcome_price"] = 42.0
    attacked, payload = _rebound(attacked)
    with pytest.raises(ChLtOriginTargetMaskInventoryError, match=message):
        assess_structural_inventory(attacked, document_bytes=payload)


def test_noncanonical_bytes_and_invalid_origin_fail_closed(
    inventory: dict[str, object],
) -> None:
    noncanonical = json.dumps(inventory, indent=2).encode("utf-8")
    with pytest.raises(ChLtOriginTargetMaskInventoryError, match="not canonical"):
        assess_structural_inventory(inventory, document_bytes=noncanonical)
    for invalid in (
        "2026-07-30T12:00:00+00:00",
        "2026-07-30T12:00Z",
        "2026-07-30",
    ):
        with pytest.raises(ChLtOriginTargetMaskInventoryError, match="origin as_of_utc"):
            build_structural_inventory(repo_root=ROOT, origin_as_of_utc=invalid)


def test_retired_v1_schema_fails_closed(inventory: dict[str, object]) -> None:
    retired = json.loads(json.dumps(inventory))
    retired["schema_version"] = "ch_lt_origin_target_mask_inventory.v1"
    retired, payload = _rebound(retired)
    with pytest.raises(ChLtOriginTargetMaskInventoryError, match="schema version"):
        assess_structural_inventory(retired, document_bytes=payload)


def test_packaged_cli_rejects_unsealed_checkout_before_parsing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _reject() -> None:
        raise ReleaseCliIdentityError("checkout runtime is not sealed")

    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", _reject)
    assert packaged_cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    result = json.loads(captured.err)
    assert result["status"] == "INVALID_GOVERNED_LT_RUNTIME"
    assert result["promotion_gate"] is False


def test_packaged_cli_failure_is_structured(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", lambda: None)
    assert packaged_cli.main(
        [
            "--evidence-root",
            str(ROOT),
            "audit",
            "--inventory",
            str(ROOT / "build" / "ch-lt-origin-target-mask-inventories" / "missing.json"),
            "--expected-inventory-sha256",
            "0" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    result = json.loads(captured.err)
    assert result["status"] == "INVALID_CH_LT_ORIGIN_TARGET_MASK_INVENTORY_NO_GO"
    assert result["evidence_slot_satisfied"] is False
