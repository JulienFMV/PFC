from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_ch_eex_current_local_capture import (
    REGISTRY_RELATIVE_PATH,
    REGISTRY_SHA256,
)
from scripts.audit_ch_eex_current_repricing_sensitivity import (
    STATUS,
    audit_current_eex_repricing_sensitivity,
    evaluate_current_eex_repricing_mechanics,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT.joinpath(*REGISTRY_RELATIVE_PATH.split("/"))
CURRENT_SOURCE = (
    ROOT
    / "build"
    / "eex-forward-local-captures"
    / "eex-ch-20260730-v1"
    / "source.xlsx"
)
LOCAL_EVIDENCE_PRESENT = REGISTRY.is_file() and CURRENT_SOURCE.is_file()


def _workbook(path: Path) -> bytes:
    frame = pd.DataFrame(
        [
            [
                None,
                "M07_2026_BASE",
                "Y01_2027_BASE",
                "Q01_2027_BASE",
                "Q02_2027_BASE",
                "Q03_2027_BASE",
                "Q04_2027_BASE",
                "Y01_2027_PEAK",
                "Q01_2027_PEAK",
                "Q02_2027_PEAK",
                "Q03_2027_PEAK",
                "Q04_2027_PEAK",
            ],
            [None, "I0", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10"],
            ["Date", None, None, None, None, None, None, None, None, None, None, None],
            [
                "29.07.2026",
                70.0,
                80.001,
                80.0,
                80.0,
                80.0,
                80.0,
                95.0,
                95.0,
                95.0,
                95.0,
                95.0,
            ],
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="CH", index=False, header=False)
    return path.read_bytes()


def test_local_mechanics_separates_active_exactness_from_source_point_conflict(
    tmp_path: Path,
) -> None:
    payload = _workbook(tmp_path / "quotes.xlsx")
    import hashlib

    result = evaluate_current_eex_repricing_mechanics(
        source_payload=payload,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_semantics_attested=False,
    )

    assert result["status"] == STATUS
    assert result["local_base_mechanics_status"] == (
        "PASS_ACTIVE_BASE_CONSTRAINTS_AND_SENSITIVITY"
    )
    assert result["quote_values_disclosed"] is False
    assert result["source_quote_counts"] == {
        "total": 11,
        "base_total": 6,
        "peak_total": 5,
        "future_base_solver_domain": 5,
        "future_peak_joint_oracle_domain": 5,
        "explicit_offpeak_total": 0,
        "excluded_started_base": 1,
        "excluded_started_peak": 0,
    }
    repricing = result["base_repricing"]
    assert repricing["active_independent_points_exact"] is True
    assert repricing["all_source_points_exact"] is False
    assert repricing["redundant_consistency_only_quote_count"] == 1
    assert repricing["redundant_consistency_only_products"] == ["2027"]
    assert repricing["all_source_points_max_abs_error_eur_mwh"] > 1e-9
    assert repricing["rounding_or_tick_interval_pass_claimed"] is False
    joint = result["joint_hourly_mechanical_oracle"]
    assert joint["constraint_system_feasible"] is True
    assert joint["active_constraint_rows_exact"] is True
    assert joint["peak_repricing"]["active_independent_points_exact"] is True
    assert joint["peak_repricing"]["all_source_points_exact"] is True
    assert joint["implied_offpeak_repricing"]["active_shared_points_exact"] is True
    assert joint["curve_values_disclosed"] is False
    gates = {gate["gate_id"]: gate["status"] for gate in result["gates"]}
    assert gates["active_independent_base_repricing_exact"] == "PASS"
    assert gates["all_future_base_source_points_reprice_exactly"] == "FAIL"
    assert gates["source_tick_rounding_and_settlement_semantics_attested"] == "FAIL"
    assert gates["reference_scale_sensitivity"] == "PASS"
    assert gates["cross_check_scale_sensitivity"] == "PASS"
    assert gates["cross_scale_derivative_invariance"] == "PASS"
    assert gates["active_independent_peak_repricing_exact"] == "PASS"
    assert gates["active_shared_implied_offpeak_identity_exact"] == "PASS"
    assert gates["peak_hourly_mechanical_oracle_evaluated"] == "PASS"
    assert gates["final_delivered_hourly_candidate_repricing_evaluated"] == "FAIL"
    rendered = json.dumps(result, sort_keys=True)
    assert "quote_price_eur_mwh" not in rendered
    assert "monthly_curve_eur_mwh" not in rendered
    assert '"80.001"' not in rendered
    assert result["scientific_admission"] is False
    assert result["production_authorization"] is False


def test_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = _workbook(tmp_path / "quotes.xlsx")

    with pytest.raises(ValueError, match="does not bind"):
        evaluate_current_eex_repricing_mechanics(
            source_payload=payload,
            source_sha256="0" * 64,
            source_semantics_attested=False,
        )


def test_source_semantics_attestation_flag_must_be_boolean(tmp_path: Path) -> None:
    payload = _workbook(tmp_path / "quotes.xlsx")
    import hashlib

    with pytest.raises(ValueError, match="must be boolean"):
        evaluate_current_eex_repricing_mechanics(
            source_payload=payload,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            source_semantics_attested=1,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(
    not LOCAL_EVIDENCE_PRESENT,
    reason="repo-local current EEX capture is unavailable",
)
def test_selected_current_capture_reports_real_point_conflict_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)

    result = audit_current_eex_repricing_sensitivity(
        registry_path=REGISTRY,
        expected_registry_sha256=REGISTRY_SHA256,
    )

    assert result["status"] == STATUS
    assert result["capture_id"] == "eex-ch-20260730-v1"
    assert result["snapshot_date"] == "2026-07-29"
    assert result["source_quote_counts"]["future_base_solver_domain"] == 19
    assert result["base_repricing"]["active_independent_quote_count"] == 17
    assert result["base_repricing"]["redundant_consistency_only_quote_count"] == 2
    assert result["base_repricing"]["redundant_consistency_only_products"] == [
        "2026-Q4",
        "2027",
    ]
    assert result["base_repricing"]["active_independent_points_exact"] is True
    assert result["base_repricing"]["all_source_points_exact"] is False
    joint = result["joint_hourly_mechanical_oracle"]
    assert joint["constraint_system_feasible"] is True
    assert joint["active_constraint_rows_exact"] is True
    assert joint["base_repricing"]["active_independent_quote_count"] == 17
    assert joint["base_repricing"]["redundant_consistency_only_products"] == [
        "2026-Q4",
        "2027",
    ]
    assert joint["peak_repricing"]["active_independent_quote_count"] == 17
    assert joint["peak_repricing"]["active_independent_points_exact"] is True
    assert joint["implied_offpeak_repricing"]["active_shared_points_exact"] is True
    assert result["sensitivity"]["reference_status"].startswith("PASS_")
    assert result["sensitivity"]["cross_check_status"].startswith("PASS_")
    assert result["monthly_solver_hard_level_eligible"] is False
    assert result["t057_consumed"] is False
    assert result["production_authorization"] is False
