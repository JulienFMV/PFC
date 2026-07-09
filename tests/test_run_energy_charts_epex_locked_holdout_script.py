from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import run_energy_charts_epex_locked_holdout as script


def test_run_energy_charts_locked_holdout_rejects_plan_hash_before_fetch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(script, "fetch_hourly_spot", lambda **kwargs: calls.append("fetch"))

    summary = script.run_energy_charts_locked_holdout(
        plan_json=plan,
        expected_plan_sha256="wrong",
        output_dir=tmp_path / "run",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_PLAN_HASH_MISMATCH"
    assert summary["spot_fetch_ran"] is False
    assert summary["locked_holdout_ran"] is False
    assert calls == []
    assert Path(summary["run_summary"]).exists()


def test_run_energy_charts_locked_holdout_waits_on_partial_spot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    plan_sha = _sha256(plan)
    runner_calls: list[str] = []

    def fake_fetch(**kwargs):
        Path(kwargs["summary_json"]).write_text('{"status":"PARTIAL_COVERAGE"}\n', encoding="utf-8")
        return {
            "status": "PARTIAL_COVERAGE",
            "full_window_covered": False,
            "observed_hour_count": 1,
            "expected_hour_count": 2,
            "next_action": "Refresh later.",
        }

    monkeypatch.setattr(script, "fetch_hourly_spot", fake_fetch)
    monkeypatch.setattr(script, "run_locked_holdout", lambda **kwargs: runner_calls.append("run"))

    summary = script.run_energy_charts_locked_holdout(
        plan_json=plan,
        expected_plan_sha256=plan_sha,
        output_dir=tmp_path / "run",
        as_of_utc="2026-07-10T03:00:00Z",
    )

    assert summary["status"] == "LOCKED_HOLDOUT_SPOT_WAITING"
    assert summary["spot_fetch_ran"] is True
    assert summary["spot_fetch"]["observed_hour_count"] == 1
    assert summary["locked_holdout_ran"] is False
    assert runner_calls == []
    assert summary["next_action"] == "Refresh later."


def test_run_energy_charts_locked_holdout_waits_before_window_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    plan_sha = _sha256(plan)
    calls: list[str] = []
    monkeypatch.setattr(script, "fetch_hourly_spot", lambda **kwargs: calls.append("fetch"))
    monkeypatch.setattr(script, "run_locked_holdout", lambda **kwargs: calls.append("run"))

    summary = script.run_energy_charts_locked_holdout(
        plan_json=plan,
        expected_plan_sha256=plan_sha,
        output_dir=tmp_path / "run",
        as_of_utc="2026-07-10T00:30:00Z",
    )

    assert summary["status"] == "LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE"
    assert summary["spot_fetch_ran"] is False
    assert summary["locked_holdout_ran"] is False
    assert summary["as_of_utc"] == "2026-07-10T00:30:00Z"
    assert summary["latest_required_holdout_utc"] == "2026-07-10T01:00:00Z"
    assert calls == []


def test_run_energy_charts_locked_holdout_runs_locked_runner_when_full(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    plan_sha = _sha256(plan)

    def fake_fetch(**kwargs):
        Path(kwargs["output_parquet"]).write_text("spot", encoding="utf-8")
        Path(kwargs["summary_json"]).write_text('{"status":"OK"}\n', encoding="utf-8")
        return {
            "status": "OK",
            "full_window_covered": True,
            "observed_hour_count": 2,
            "expected_hour_count": 2,
        }

    def fake_run(**kwargs):
        run_summary = Path(kwargs["output_dir"]) / "locked_holdout_run_summary.json"
        run_summary.parent.mkdir(parents=True, exist_ok=True)
        run_summary.write_text('{"status":"LOCKED_HOLDOUT_PASS"}\n', encoding="utf-8")
        return {
            "status": "LOCKED_HOLDOUT_PASS",
            "holdout_pass": True,
            "run_summary": str(run_summary),
            "next_action": "Review holdout evidence.",
        }

    monkeypatch.setattr(script, "fetch_hourly_spot", fake_fetch)
    monkeypatch.setattr(script, "run_locked_holdout", fake_run)

    summary = script.run_energy_charts_locked_holdout(
        plan_json=plan,
        expected_plan_sha256=plan_sha,
        output_dir=tmp_path / "run",
        as_of_utc="2026-07-10T03:00:00Z",
    )

    assert summary["status"] == "LOCKED_HOLDOUT_PASS"
    assert summary["holdout_pass"] is True
    assert summary["spot_fetch_ran"] is True
    assert summary["locked_holdout_ran"] is True
    assert summary["locked_holdout_run_summary_sha256"]
    assert Path(summary["run_summary"]).exists()


def test_main_returns_nonzero_until_locked_holdout_pass(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(tmp_path)
    monkeypatch.setattr(
        script,
        "run_energy_charts_locked_holdout",
        lambda **kwargs: {"status": "LOCKED_HOLDOUT_SPOT_WAITING", "holdout_pass": False},
    )

    code = script.main(
        [
            "--plan-json",
            str(plan),
            "--expected-plan-sha256",
            _sha256(plan),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert code == 1


def _write_plan(tmp_path: Path) -> Path:
    plan = tmp_path / "locked_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "plan_id": "test_holdout",
                "holdout_start_utc": "2026-07-10T00:00:00Z",
                "holdout_end_utc": "2026-07-10T02:00:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
