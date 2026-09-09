from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import pfc_shaping.validation.ch_lt_pit_preregistration as preregistration_module
from pfc_shaping.validation.ch_lt_pit_preregistration import (
    DRAFT_STATUS,
    FROZEN_BLOCKED_STATUS,
    FROZEN_STATE,
    REGIMES,
    UNRESOLVED_EXECUTION_BOUNDARY_BLOCKERS,
    ChLtPitPreregistrationError,
    assess_preregistration,
    canonical_json_bytes,
    canonical_sha256,
    compute_plan_id,
    load_preregistration,
)
from scripts.audit_ch_lt_pit_preregistration import (
    _workspace_output,
    audit_preregistration,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DRAFT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-PIT-PROBABILISTIC-PREREGISTRATION-DRAFT-20260724.json"
)


def _draft() -> dict[str, object]:
    return json.loads(CANONICAL_DRAFT.read_text(encoding="utf-8"))


def _rehash(document: dict[str, object]) -> dict[str, object]:
    document["plan_id"] = compute_plan_id(document)
    return document


def _frozen_plan() -> dict[str, object]:
    document = copy.deepcopy(_draft())
    document["lifecycle"] = {
        "state": FROZEN_STATE,
        "frozen_at_utc": "2026-07-24T12:00:00Z",
        "trusted_time_receipt_sha256": "1" * 64,
        "governance_receipt_sha256": "2" * 64,
        "candidate_grid_sha256": "3" * 64,
    }
    data = document["data_admission"]
    assert isinstance(data, dict)
    data["admitted_role_manifests"] = [
        {
            "role": role,
            "manifest_sha256": character * 64,
            "trusted_time_receipt_sha256": "b" * 64,
            "external_cas_receipt_sha256": "c" * 64,
            "independent_signature_verified": True,
            "immutable_builder_inaccessible": True,
            "available_at_complete": True,
            "point_in_time_replay_pass": True,
        }
        for role, character in zip(
            ("eex_forward_source", "eex_forwards_history", "epex_ch"),
            ("a", "b", "c"),
        )
    ]

    origins: list[dict[str, object]] = []
    for timestamp in pd.date_range("2024-01-01", periods=24, freq="MS", tz="UTC"):
        local_month = timestamp.tz_convert("Europe/Zurich").month
        season = (
            "WINTER"
            if local_month in {12, 1, 2}
            else "SPRING"
            if local_month in {3, 4, 5}
            else "SUMMER"
            if local_month in {6, 7, 8}
            else "AUTUMN"
        )
        text = timestamp.isoformat().replace("+00:00", "Z")
        origins.append(
            {
                "origin_id": f"origin-{text[:10]}",
                "as_of_utc": text,
                "season": season,
                "regimes": [
                    "HYDRO_LOW" if len(origins) % 2 == 0 else "HYDRO_HIGH",
                    REGIMES[2 + len(origins) % 4],
                ],
            }
        )
    design = document["origin_design"]
    assert isinstance(design, dict)
    outer = design["retrospective_outer"]
    assert isinstance(outer, dict)
    outer["exact_origins"] = origins
    outer["origin_inventory_sha256"] = canonical_sha256(origins)
    outer["regime_definition_manifest_sha256"] = "d" * 64

    episodes = [
        {
            "episode_id": "future-winter",
            "start_utc": "2026-12-01T00:00:00Z",
            "end_utc": "2026-12-15T00:00:00Z",
            "season": "WINTER",
        },
        {
            "episode_id": "future-spring",
            "start_utc": "2027-03-01T00:00:00Z",
            "end_utc": "2027-03-15T00:00:00Z",
            "season": "SPRING",
        },
        {
            "episode_id": "future-summer",
            "start_utc": "2027-06-01T00:00:00Z",
            "end_utc": "2027-06-15T00:00:00Z",
            "season": "SUMMER",
        },
        {
            "episode_id": "future-autumn",
            "start_utc": "2027-09-01T00:00:00Z",
            "end_utc": "2027-09-15T00:00:00Z",
            "season": "AUTUMN",
        },
    ]
    future = design["future_holdout"]
    assert isinstance(future, dict)
    future["exact_episodes"] = episodes
    future["episode_inventory_sha256"] = canonical_sha256(episodes)

    document["economic_materiality"] = {
        "metric": "PROFILE_CAPTURE_VALUE_ERROR_EUR_MWH",
        "population": "FMV_PREDEFINED_FIL_ACC_AND_DISPATCH_PROFILES",
        "minimum_capture_value_improvement_eur_mwh": 0.1,
        "transaction_and_imbalance_assumptions_sha256": "e" * 64,
        "fmv_risk_approval_receipt_sha256": "f" * 64,
        "status": "RISK_APPROVED_FROZEN",
    }
    document["independent_review"] = {
        "security_receipt_sha256": "4" * 64,
        "it_operations_receipt_sha256": "5" * 64,
        "quant_data_receipt_sha256": "6" * 64,
        "required_no_open_severity": ["P0", "P1"],
    }
    return _rehash(document)


def test_canonical_draft_is_hash_bound_and_explicitly_not_executable() -> None:
    payload = CANONICAL_DRAFT.read_bytes()
    document = json.loads(payload)

    assessment = assess_preregistration(document, document_bytes=payload)

    assert assessment.status == DRAFT_STATUS
    assert assessment.execution_authorized is False
    assert assessment.production_authorization is False
    assert assessment.promotion_gate is False
    assert assessment.document_sha256 == hashlib.sha256(payload).hexdigest()
    assert assessment.plan_id == compute_plan_id(document)
    assert set(assessment.blockers) == {
        "PLAN_NOT_FROZEN_BY_INDEPENDENT_TRUSTED_TIME_AND_GOVERNANCE",
        "GOVERNED_PIT_DATA_ROLE_MANIFESTS_ABSENT",
        "EXACT_24_ORIGIN_INVENTORY_AND_REGIME_DEFINITIONS_ABSENT",
        "EXACT_FOUR_SEASONAL_FUTURE_EPISODES_ABSENT",
        "FMV_CAPTURE_VALUE_MATERIALITY_THRESHOLD_NOT_RISK_APPROVED",
        "INDEPENDENT_SECURITY_ITOPS_QUANT_REVIEW_RECEIPTS_ABSENT",
        *UNRESOLVED_EXECUTION_BOUNDARY_BLOCKERS,
    }


def test_self_declared_frozen_structure_never_authorizes_evaluation() -> None:
    assert FROZEN_STATE == "FROZEN_STRUCTURE_UNVERIFIED_EXTERNAL_ADMISSION_REQUIRED"

    assessment = assess_preregistration(_frozen_plan())

    assert assessment.status == FROZEN_BLOCKED_STATUS
    assert assessment.execution_authorized is False
    assert assessment.production_authorization is False
    assert assessment.promotion_gate is False
    assert "PATH_VERIFIED_PLAN_GOVERNANCE_AND_TRUSTED_TIME_ABSENT" in assessment.blockers
    assert "DURABLE_ONE_SHOT_ATTEMPT_SEAL_AND_LEDGER_NOT_IMPLEMENTED" in assessment.blockers


def test_rehashed_threshold_weakening_still_fails_frozen_policy() -> None:
    document = _draft()
    deterministic = document["deterministic_policy"]
    assert isinstance(deterministic, dict)
    deterministic["minimum_relative_mae_improvement"] = 0.0
    _rehash(document)

    with pytest.raises(ChLtPitPreregistrationError, match="deterministic policy"):
        assess_preregistration(document)


def test_python_bool_integer_equivalence_cannot_bypass_policy() -> None:
    document = _draft()
    controls = document["execution_controls"]
    assert isinstance(controls, dict)
    controls["one_shot_future_holdout_namespace"] = 1
    _rehash(document)

    with pytest.raises(ChLtPitPreregistrationError, match="execution controls"):
        assess_preregistration(document)


def test_current_local_evidence_cannot_be_relabelled_as_admissible() -> None:
    document = _draft()
    data = document["data_admission"]
    assert isinstance(data, dict)
    evidence = data["current_evidence"]
    assert isinstance(evidence, list)
    assert isinstance(evidence[0], dict)
    evidence[0]["admissible_for_training"] = True
    _rehash(document)

    with pytest.raises(ChLtPitPreregistrationError, match="admissible_for_training"):
        assess_preregistration(document)


def test_frozen_plan_rejects_wrong_season_even_when_inventory_is_rehashed() -> None:
    document = _frozen_plan()
    design = document["origin_design"]
    assert isinstance(design, dict)
    outer = design["retrospective_outer"]
    assert isinstance(outer, dict)
    origins = outer["exact_origins"]
    assert isinstance(origins, list)
    assert isinstance(origins[0], dict)
    origins[0]["season"] = "SUMMER"
    outer["origin_inventory_sha256"] = canonical_sha256(origins)
    _rehash(document)

    with pytest.raises(ChLtPitPreregistrationError, match="season does not match"):
        assess_preregistration(document)


def test_frozen_plan_rejects_mutually_exclusive_regime_labels() -> None:
    document = _frozen_plan()
    design = document["origin_design"]
    assert isinstance(design, dict)
    outer = design["retrospective_outer"]
    assert isinstance(outer, dict)
    origins = outer["exact_origins"]
    assert isinstance(origins, list)
    assert isinstance(origins[0], dict)
    origins[0]["regimes"] = ["HYDRO_LOW", "HYDRO_HIGH", "NEGATIVE_PRICE"]
    outer["origin_inventory_sha256"] = canonical_sha256(origins)
    _rehash(document)

    with pytest.raises(ChLtPitPreregistrationError, match="both HYDRO_LOW"):
        assess_preregistration(document)


def test_frozen_plan_rejects_shared_manifest_hash_across_roles() -> None:
    document = _frozen_plan()
    data = document["data_admission"]
    assert isinstance(data, dict)
    manifests = data["admitted_role_manifests"]
    assert isinstance(manifests, list)
    assert isinstance(manifests[0], dict)
    assert isinstance(manifests[1], dict)
    manifests[1]["manifest_sha256"] = manifests[0]["manifest_sha256"]
    _rehash(document)

    with pytest.raises(ChLtPitPreregistrationError, match="cannot share"):
        assess_preregistration(document)


def test_future_episode_cannot_cross_season_boundary() -> None:
    document = _frozen_plan()
    design = document["origin_design"]
    assert isinstance(design, dict)
    future = design["future_holdout"]
    assert isinstance(future, dict)
    episodes = future["exact_episodes"]
    assert isinstance(episodes, list)
    assert isinstance(episodes[0], dict)
    episodes[0] = {
        "episode_id": "future-winter",
        "start_utc": "2027-02-28T00:00:00Z",
        "end_utc": "2027-03-14T00:00:00Z",
        "season": "WINTER",
    }
    future["episode_inventory_sha256"] = canonical_sha256(episodes)
    _rehash(document)

    with pytest.raises(ChLtPitPreregistrationError, match="crosses a season"):
        assess_preregistration(document)


def test_assessment_rejects_document_bytes_rebinding() -> None:
    frozen = _frozen_plan()

    with pytest.raises(ChLtPitPreregistrationError, match="differs from captured"):
        assess_preregistration(frozen, document_bytes=CANONICAL_DRAFT.read_bytes())


def test_loader_preserves_lexical_path_for_link_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical = ROOT / "would-be-link" / "plan.json"
    seen: list[Path] = []

    def _capture(path: Path, **kwargs: object) -> bytes:
        seen.append(path)
        return CANONICAL_DRAFT.read_bytes()

    monkeypatch.setattr(
        preregistration_module,
        "read_stable_single_link_file",
        _capture,
    )

    document, _ = load_preregistration(lexical)

    assert document["plan_id"] == _draft()["plan_id"]
    assert seen == [lexical]


def test_strict_loader_rejects_duplicate_json_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preregistration_module,
        "read_stable_single_link_file",
        lambda *args, **kwargs: b'{"schema_version":"x","schema_version":"y"}',
    )

    with pytest.raises(ChLtPitPreregistrationError, match="plan JSON is invalid"):
        load_preregistration(ROOT / "unused.json")


def test_audit_requires_exact_plan_file_hash() -> None:
    with pytest.raises(ChLtPitPreregistrationError, match="differ from expected"):
        audit_preregistration(
            plan_path=CANONICAL_DRAFT,
            expected_plan_sha256="0" * 64,
        )


def test_cli_validate_draft_passes_but_admit_execution_fails_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = hashlib.sha256(CANONICAL_DRAFT.read_bytes()).hexdigest()

    assert (
        main(
            [
                "--plan",
                str(CANONICAL_DRAFT),
                "--expected-plan-sha256",
                digest,
                "--mode",
                "validate-draft",
            ]
        )
        == 0
    )
    draft_result = json.loads(capsys.readouterr().out)
    assert draft_result["status"] == DRAFT_STATUS

    assert (
        main(
            [
                "--plan",
                str(CANONICAL_DRAFT),
                "--expected-plan-sha256",
                digest,
                "--mode",
                "admit-execution",
            ]
        )
        == 3
    )
    admission_result = json.loads(capsys.readouterr().out)
    assert admission_result["execution_authorized"] is False


def test_validate_draft_mode_does_not_greenlight_frozen_structure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frozen = _frozen_plan()
    payload = canonical_json_bytes(frozen)
    monkeypatch.setattr(
        preregistration_module,
        "read_stable_single_link_file",
        lambda *args, **kwargs: payload,
    )

    assert (
        main(
            [
                "--plan",
                str(ROOT / "unused-frozen.json"),
                "--expected-plan-sha256",
                hashlib.sha256(payload).hexdigest(),
                "--mode",
                "validate-draft",
            ]
        )
        == 4
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == FROZEN_BLOCKED_STATUS
    assert result["execution_authorized"] is False


def test_audit_output_is_restricted_to_canonical_namespace() -> None:
    with pytest.raises(ValueError, match="canonical audit namespace"):
        _workspace_output(ROOT / "arbitrary-audit.json")


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})
