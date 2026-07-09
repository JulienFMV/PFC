from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plan_epex_lab_locked_holdout import build_plan


def test_plan_epex_lab_locked_holdout_binds_selected_no_ompex_candidate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted)
    adjusted_sha = _sha256(adjusted)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "replacement_verdict": {"replace_incumbent": True},
                "selected_adjusted_csv_sha256": adjusted_sha,
                "selected_trial": {"adjusted_csv_sha256": adjusted_sha},
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
            }
        ),
        encoding="utf-8",
    )
    lab_manifest = tmp_path / "lab.json"
    lab_manifest.write_text(json.dumps({"config": {"weekend_intensity": 0.75}}), encoding="utf-8")

    plan = build_plan(
        baseline_csv=baseline,
        adjusted_csv=adjusted,
        selection_summary=selection,
        lab_manifest=lab_manifest,
        output=tmp_path / "plan.json",
        frozen_at_utc="2026-07-09T00:00:00Z",
        holdout_start_utc="2026-07-10T00:00:00Z",
        holdout_end_utc="2026-07-24T00:00:00Z",
        min_holdout_hours=300,
    )

    assert plan["schema_version"] == "epex_lab_locked_holdout_plan.v1"
    assert plan["benchmark_policy"] == "locked_future_no_ompex_holdout"
    assert plan["production_approved"] is False
    assert plan["adjusted_csv_sha256"] == adjusted_sha
    assert plan["candidate_timestamp_identity"]["timestamp_sets_identical"] is True
    assert plan["candidate_timestamp_identity"]["candidate_timestamp_count"] == 4
    assert plan["pass_criteria"]["candidate_timestamp_set_sha256"] == plan["candidate_timestamp_identity"][
        "candidate_timestamp_set_sha256"
    ]
    assert plan["selection_policy"]["pass"] is True
    assert plan["locked_lab_config"] == {"weekend_intensity": 0.75}
    assert plan["pass_criteria"]["min_holdout_hours"] == 300
    assert "OMPEX" in plan["forbidden_inputs"]
    assert "scripts/run_epex_lab_locked_holdout.py" in plan["commands"]["run_locked_holdout_template"]
    assert "--expected-plan-sha256 <T057_PLAN_JSON_SHA256>" in plan["commands"]["run_locked_holdout_template"]
    assert "--spot-parquet <FUTURE_SPOT_PARQUET>" in plan["commands"]["run_locked_holdout_template"]
    assert "compare_hpfc_ompex" not in plan["commands"]["run_future_backtest_template"]
    assert "compare_hpfc_ompex" not in plan["commands"]["run_locked_holdout_template"]
    assert (tmp_path / "plan.json").exists()


def test_plan_epex_lab_locked_holdout_resolves_relative_paths(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "space dir"
    work.mkdir()
    baseline = work / "baseline.csv"
    adjusted = work / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted)
    selection = _write_selection_summary(work, adjusted)
    lab_manifest = _write_lab_manifest(work)
    monkeypatch.chdir(work)

    plan = build_plan(
        baseline_csv=Path("baseline.csv"),
        adjusted_csv=Path("adjusted.csv"),
        selection_summary=selection.relative_to(work),
        lab_manifest=lab_manifest.relative_to(work),
        output=Path("plan.json"),
        frozen_at_utc="2026-07-09T00:00:00Z",
        holdout_start_utc="2026-07-10T00:00:00Z",
        holdout_end_utc="2026-07-24T00:00:00Z",
    )

    assert plan["baseline_csv"] == str(baseline.resolve())
    assert plan["adjusted_csv"] == str(adjusted.resolve())
    assert f'--baseline-csv "{baseline.resolve()}"' in plan["commands"]["run_future_backtest_template"]
    assert f'--adjusted-csv "{adjusted.resolve()}"' in plan["commands"]["run_future_backtest_template"]
    assert (work / "plan.json").exists()


def test_plan_epex_lab_locked_holdout_rejects_missing_selection_summary(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted)
    lab_manifest = _write_lab_manifest(tmp_path)

    with pytest.raises(ValueError, match="selection_summary is required"):
        build_plan(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            lab_manifest=lab_manifest,
            frozen_at_utc="2026-07-09T00:00:00Z",
            holdout_start_utc="2026-07-10T00:00:00Z",
            holdout_end_utc="2026-07-24T00:00:00Z",
        )


def test_plan_epex_lab_locked_holdout_rejects_missing_lab_manifest(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted)
    selection = _write_selection_summary(tmp_path, adjusted)

    with pytest.raises(ValueError, match="lab_manifest is required"):
        build_plan(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            selection_summary=selection,
            frozen_at_utc="2026-07-09T00:00:00Z",
            holdout_start_utc="2026-07-10T00:00:00Z",
            holdout_end_utc="2026-07-24T00:00:00Z",
        )


def test_plan_epex_lab_locked_holdout_rejects_unbound_selection(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted)
    selection = tmp_path / "selection.json"
    lab_manifest = _write_lab_manifest(tmp_path)
    selection.write_text(
        json.dumps(
            {
                "replacement_verdict": {"replace_incumbent": True},
                "selected_adjusted_csv_sha256": "wrong",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection summary is not bound"):
        build_plan(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            selection_summary=selection,
            lab_manifest=lab_manifest,
            frozen_at_utc="2026-07-09T00:00:00Z",
            holdout_start_utc="2026-07-10T00:00:00Z",
            holdout_end_utc="2026-07-24T00:00:00Z",
        )


def test_plan_epex_lab_locked_holdout_rejects_timestamp_set_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted, periods=5)
    selection = _write_selection_summary(tmp_path, adjusted)
    lab_manifest = _write_lab_manifest(tmp_path)

    with pytest.raises(ValueError, match="timestamp identities"):
        build_plan(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            selection_summary=selection,
            lab_manifest=lab_manifest,
            frozen_at_utc="2026-07-09T00:00:00Z",
            holdout_start_utc="2026-07-10T00:00:00Z",
            holdout_end_utc="2026-07-24T00:00:00Z",
        )


def _write_candidate_csv(path: Path, *, periods: int = 4) -> None:
    rows = ["timestamp_ch,utc_offset_ch"]
    for hour in range(periods):
        rows.append(f"10.07.2026 {hour + 2:02d}:00,UTC+02:00")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_selection_summary(tmp_path: Path, adjusted: Path) -> Path:
    adjusted_sha = _sha256(adjusted)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "replacement_verdict": {"replace_incumbent": True},
                "selected_adjusted_csv_sha256": adjusted_sha,
                "selected_trial": {"adjusted_csv_sha256": adjusted_sha},
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
            }
        ),
        encoding="utf-8",
    )
    return selection


def _write_lab_manifest(tmp_path: Path) -> Path:
    lab_manifest = tmp_path / "lab.json"
    lab_manifest.write_text(json.dumps({"config": {"weekend_intensity": 0.75}}), encoding="utf-8")
    return lab_manifest


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
