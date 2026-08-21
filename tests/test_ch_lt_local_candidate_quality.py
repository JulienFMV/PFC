from __future__ import annotations

import json
from pathlib import Path

import pytest

import pfc_shaping.validation.ch_lt_local_candidate_quality as local_quality
from pfc_shaping.validation.ch_lt_local_candidate_quality import (
    CANONICAL_COMPLETION_RELATIVE_PATH,
    CANONICAL_COMPLETION_SHA256,
    STATUS,
    ChLtLocalCandidateQualityError,
    build_local_candidate_quality_report,
)

ROOT = Path(__file__).resolve().parents[1]
COMPLETION = ROOT.joinpath(*CANONICAL_COMPLETION_RELATIVE_PATH.split("/"))


def _build(tmp_path: Path) -> dict[str, object]:
    return build_local_candidate_quality_report(
        repo_root=ROOT,
        completion_manifest=COMPLETION,
        expected_completion_sha256=CANONICAL_COMPLETION_SHA256,
        output_json=tmp_path / "quality.json",
        output_markdown=tmp_path / "quality.md",
    )


def test_canonical_local_candidate_report_is_structurally_green_but_production_no_go(
    tmp_path: Path,
) -> None:
    report = _build(tmp_path)

    assert report["schema_version"] == "ch_lt_local_candidate_quality.v2"
    assert report["status"] == STATUS
    assert report["production_authorization"] is False
    assert report["promotion_gate"] is False
    assert report["future_holdout_consumed"] is False
    assert report["t057_consumed"] is False
    assert report["t057_supersession_metadata_read"] is True
    assert report["t057_reused_for_selection"] is False
    gates = report["gates"]
    assert gates["monthly_level_preserved_within_1e_9_eur_mwh"] is True
    assert gates["monthly_solver_constraint_residual_within_1e_9_eur_mwh"] is True
    assert gates["governed_fresh_hard_ch_quotes"] is False
    assert gates["direct_governed_ch_15min_truth"] is False
    assert gates["probabilistic_quantiles_calibrated"] is False
    assert gates["promotion_eligible"] is False
    assert report["curve"]["scenario_summaries"]["central"]["rows"] == 155_616
    assert report["curve"]["scenario_summaries"]["central"]["p10_non_null_rows"] == 0
    assert report["intraday"]["rolling_origin"]["candidate_accepted"] is False
    assert (
        report["intraday"]["rolling_origin"]["aggregate_metrics_source"]
        == "RECOMPUTED_FROM_HASH_BOUND_FOLD_ROWS"
    )
    assert "scenario_weights" not in report["curve"]
    assert "structural_display_weights_not_probabilities" in report["curve"]
    assert report["t057_corrected_evidence"]["historical_distinct_fold_count"] == 1
    assert json.loads((tmp_path / "quality.json").read_text(encoding="utf-8")) == report


def test_current_source_drift_is_reported_not_hidden(tmp_path: Path) -> None:
    report = _build(tmp_path)
    state = report["source"]["input_state"]

    assert state["exact_run_code_bytes_captured"] is False
    assert state["reproducible_execution_proven"] is False
    assert state["source_drift_count"] >= 1
    assert any(
        item["role"] == "local_runner_source_observed_after_import"
        for item in state["source_drift"]
    )


def test_wrong_caller_held_completion_hash_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ChLtLocalCandidateQualityError, match="caller-held"):
        build_local_candidate_quality_report(
            repo_root=ROOT,
            completion_manifest=COMPLETION,
            expected_completion_sha256="0" * 64,
            output_json=tmp_path / "quality.json",
            output_markdown=tmp_path / "quality.md",
        )


def test_quality_output_cannot_modify_immutable_run15() -> None:
    with pytest.raises(ChLtLocalCandidateQualityError, match="outside immutable run15"):
        build_local_candidate_quality_report(
            repo_root=ROOT,
            completion_manifest=COMPLETION,
            expected_completion_sha256=CANONICAL_COMPLETION_SHA256,
            output_json=COMPLETION.parent / "quality.json",
            output_markdown=COMPLETION.parent / "quality.md",
        )


def test_cumulative_output_byte_budget_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(local_quality, "_MAX_TOTAL_OUTPUT_BYTES", 1)
    with pytest.raises(ChLtLocalCandidateQualityError, match="cumulative byte budget"):
        _build(tmp_path)
