from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pfc_shaping.validation.ch_lt_prospective_acquisition_plan import (
    DOCUMENT_SHA256,
    PLAN_ID,
    STATUS,
    ChLtProspectiveAcquisitionPlanError,
    assess_prospective_acquisition_plan,
    compute_plan_id,
    load_prospective_acquisition_plan,
)
from scripts.audit_ch_lt_prospective_acquisition_plan import main

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-PROSPECTIVE-DATA-ACQUISITION-PLAN-20260727.json"
)


def _document() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _rehash(document: dict[str, object]) -> None:
    document["plan_id"] = compute_plan_id(document)


def _mutated_bytes(document: dict[str, object]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_canonical_plan_is_hash_bound_and_explicitly_blocked() -> None:
    payload = PLAN.read_bytes()
    document = json.loads(payload)

    result = assess_prospective_acquisition_plan(document, document_bytes=payload)

    assert result.plan_id == PLAN_ID
    assert result.document_sha256 == hashlib.sha256(payload).hexdigest() == DOCUMENT_SHA256
    assert result.status == STATUS
    assert result.blockers
    audit = result.to_dict()
    assert audit["execution_authorized"] is False
    assert audit["scientific_admission"] is False
    assert audit["production_authorization"] is False
    assert audit["promotion_gate"] is False


@pytest.mark.parametrize(
    ("track_id", "field", "value"),
    [
        (
            "CH_DAY_AHEAD_HOURLY_ENERGY_CHARTS",
            "native_quarter_hour_truth_eligible",
            True,
        ),
        (
            "CH_INTRADAY_CONTINUOUS_15MIN_EPEX",
            "price_statistic",
            "LAST_EUR_MWH",
        ),
        (
            "CH_INTRADAY_CONTINUOUS_15MIN_EPEX",
            "positive_volume_required",
            False,
        ),
        (
            "CH_EEX_FORWARD_POINT_IN_TIME",
            "monthly_solver_authority_input",
            False,
        ),
        (
            "CH_EXPLICIT_AUCTION_15MIN_GO_LIVE_MONITOR",
            "confirmed_go_live",
            True,
        ),
    ],
)
def test_rehashed_policy_weakening_fails_closed(
    track_id: str, field: str, value: object
) -> None:
    document = copy.deepcopy(_document())
    tracks = document["acquisition_tracks"]
    assert isinstance(tracks, list)
    track = next(item for item in tracks if item["track_id"] == track_id)
    track[field] = value
    _rehash(document)

    with pytest.raises(
        ChLtProspectiveAcquisitionPlanError, match="frozen canonical identity"
    ):
        assess_prospective_acquisition_plan(
            document, document_bytes=_mutated_bytes(document)
        )


def test_rehashed_cross_product_level_pooling_is_rejected() -> None:
    document = _document()
    boundary = document["scientific_boundary"]
    assert isinstance(boundary, dict)
    boundary["cross_product_absolute_level_pooling"] = True
    _rehash(document)

    with pytest.raises(
        ChLtProspectiveAcquisitionPlanError, match="frozen canonical identity"
    ):
        assess_prospective_acquisition_plan(
            document, document_bytes=_mutated_bytes(document)
        )


def test_plan_mapping_must_rebind_to_captured_bytes() -> None:
    document = _document()
    document["execution_authorized"] = True

    with pytest.raises(ChLtProspectiveAcquisitionPlanError, match="differs from captured"):
        assess_prospective_acquisition_plan(document, document_bytes=PLAN.read_bytes())


def test_loader_requires_caller_held_file_hash() -> None:
    with pytest.raises(
        ChLtProspectiveAcquisitionPlanError, match="frozen canonical identity"
    ):
        load_prospective_acquisition_plan(PLAN, expected_sha256="0" * 64)


def test_loader_rejects_noncanonical_path_before_read() -> None:
    copied = ROOT / "CH-LT-PROSPECTIVE-DATA-ACQUISITION-PLAN-shadow.json"
    with pytest.raises(ChLtProspectiveAcquisitionPlanError, match="not canonical"):
        load_prospective_acquisition_plan(copied, expected_sha256=DOCUMENT_SHA256)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_authority_key",
        "continuous_allowed_and_forbidden_uses",
        "reuse_t057_action",
        "source_observation",
    ],
)
def test_rehashed_shadow_plan_policy_smuggling_fails_canonical_identity(
    mutation: str,
) -> None:
    document = copy.deepcopy(_document())
    tracks = document["acquisition_tracks"]
    actions = document["next_actions"]
    observations = document["primary_source_observations"]
    assert isinstance(tracks, list)
    assert isinstance(actions, list)
    assert isinstance(observations, list)
    continuous = next(
        item
        for item in tracks
        if item["track_id"] == "CH_INTRADAY_CONTINUOUS_15MIN_EPEX"
    )
    if mutation == "unknown_authority_key":
        continuous["authoritative_ch_day_ahead_truth_override"] = True
    elif mutation == "continuous_allowed_and_forbidden_uses":
        continuous["allowed_uses_after_external_admission"] = [
            "DAY_AHEAD_MODEL_SELECTION"
        ]
        continuous["forbidden_uses"] = []
    elif mutation == "reuse_t057_action":
        actions[6]["action"] = "REUSE_T057_AND_DE_LU_PILOT_FOR_MODEL_SELECTION"
        actions[6]["route"] = "T057_AND_PILOT"
    else:
        observations[0]["observation"] = "AUTHORITATIVE_NATIVE_15MIN_CH_TRUTH"
    _rehash(document)

    with pytest.raises(
        ChLtProspectiveAcquisitionPlanError, match="frozen canonical identity"
    ):
        assess_prospective_acquisition_plan(
            document, document_bytes=_mutated_bytes(document)
        )


def test_cli_validates_structure_without_authorizing_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--plan",
                str(PLAN),
                "--expected-plan-sha256",
                DOCUMENT_SHA256,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == STATUS
    assert result["execution_authorized"] is False
