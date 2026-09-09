from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_epex_lab_locked_holdout_queue import audit_queue, main


def test_audit_locked_holdout_queue_waits_before_future_windows(tmp_path: Path) -> None:
    t057 = _write_plan(
        tmp_path / "locked_holdout_plan_t057.json",
        plan_id="t057_locked_t056_future_holdout",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    t061 = _write_plan(
        tmp_path / "locked_holdout_plan_t061.json",
        plan_id="t061_locked_t060_future_holdout",
        start="2026-07-24T00:00:00Z",
        end="2026-08-07T00:00:00Z",
    )

    summary = audit_queue(
        plan_jsons=[t061, t057],
        output=tmp_path / "queue.json",
        as_of_utc="2026-07-09T00:00:00Z",
        search_roots=[tmp_path / "output"],
        output_root=tmp_path / "phase14",
    )

    assert summary["schema_version"] == "epex_lab_locked_holdout_queue_audit.v1"
    assert summary["status"] == "WAITING_FOR_FUTURE_HOLDOUT_WINDOWS"
    assert summary["plan_count"] == 2
    assert summary["future_window_count"] == 2
    assert summary["spot_refresh_due_count"] == 0
    assert summary["duplicate_plan_id_count"] == 0
    assert summary["overlapping_window_count"] == 0
    assert summary["queue_issues"] == []
    assert [row["plan_id"] for row in summary["plans"]] == [
        "t057_locked_t056_future_holdout",
        "t061_locked_t060_future_holdout",
    ]
    first = summary["plans"][0]
    assert first["plan_json_sha256"] == _sha256(t057)
    assert first["temporal_status"] == "WAITING_FOR_HOLDOUT_START"
    assert first["blocking_stage"] == "wait_for_holdout_window_start"
    assert first["ready_for_spot_refresh"] is False
    assert first["promotion_gate"] is False
    assert first["production_approved"] is False
    assert first["artifact_checks"]["baseline_csv_sha256_bound"] is True
    assert first["artifact_checks"]["adjusted_csv_sha256_bound"] is True
    assert first["artifact_checks"]["lab_manifest_sha256_bound"] is True
    assert first["artifact_checks"]["selection_summary_sha256_bound"] is True
    assert "run_energy_charts_epex_locked_holdout.py" in first["recommended_commands"][
        "run_energy_charts_locked_holdout"
    ]
    assert "--expected-plan-sha256" in first["recommended_commands"]["run_energy_charts_locked_holdout"]
    assert "discover_epex_spot_parquet_candidates.py" in first["recommended_commands"]["discover_spot_parquet"]
    assert (tmp_path / "queue.json").exists()


def test_audit_locked_holdout_queue_marks_completed_window_due(tmp_path: Path) -> None:
    t057 = _write_plan(
        tmp_path / "t057.json",
        plan_id="t057",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    t061 = _write_plan(
        tmp_path / "t061.json",
        plan_id="t061",
        start="2026-07-24T00:00:00Z",
        end="2026-08-07T00:00:00Z",
    )

    summary = audit_queue(
        plan_jsons=[t057, t061],
        as_of_utc="2026-07-25T00:00:00Z",
    )

    by_id = {row["plan_id"]: row for row in summary["plans"]}
    assert summary["status"] == "WAITING_FOR_SPOT_REFRESH_AND_LOCKED_HOLDOUT_RUN"
    assert summary["spot_refresh_due_count"] == 1
    assert summary["active_window_count"] == 1
    assert by_id["t057"]["temporal_status"] == "HOLDOUT_WINDOW_COMPLETE"
    assert by_id["t057"]["blocking_stage"] == "refresh_spot_and_run_locked_holdout"
    assert by_id["t057"]["ready_for_spot_refresh"] is True
    assert by_id["t061"]["temporal_status"] == "IN_HOLDOUT_WINDOW"
    assert by_id["t061"]["blocking_stage"] == "wait_for_holdout_window_completion"


def test_audit_locked_holdout_queue_blocks_policy_invalid_plan(tmp_path: Path) -> None:
    plan = _write_plan(
        tmp_path / "bad.json",
        plan_id="bad",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
        ompex_used_in_selection=True,
    )

    summary = audit_queue(plan_jsons=[plan], as_of_utc="2026-07-09T00:00:00Z")

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_QUEUE_INVALID"
    assert summary["policy_invalid_plan_count"] == 1
    assert summary["plans"][0]["blocking_stage"] == "locked_holdout_plan_policy_invalid"
    assert summary["plans"][0]["policy_checks"]["ompex_not_selection"] is False


def test_audit_locked_holdout_queue_blocks_missing_or_tampered_artifact(tmp_path: Path) -> None:
    plan = _write_plan(
        tmp_path / "bad_artifact.json",
        plan_id="bad_artifact",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    payload = json.loads(plan.read_text(encoding="utf-8"))
    Path(payload["adjusted_csv"]).write_text("tampered\n", encoding="utf-8")

    summary = audit_queue(plan_jsons=[plan], as_of_utc="2026-07-09T00:00:00Z")

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_QUEUE_INVALID"
    assert summary["artifact_invalid_plan_count"] == 1
    assert summary["plans"][0]["blocking_stage"] == "locked_holdout_artifact_missing_or_hash_mismatch"
    assert summary["plans"][0]["artifact_checks"]["adjusted_csv_sha256_bound"] is False
    assert summary["plans"][0]["next_required_step"] == (
        "restore_or_regenerate_locked_candidate_artifacts_without_retuning_holdout"
    )


def test_audit_locked_holdout_queue_cli_returns_nonzero_for_invalid_policy(tmp_path: Path) -> None:
    plan = _write_plan(
        tmp_path / "bad.json",
        plan_id="bad",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
        production_approved=True,
    )

    code = main(
        [
            "--plan-json",
            str(plan),
            "--output",
            str(tmp_path / "queue.json"),
            "--as-of-utc",
            "2026-07-09T00:00:00Z",
        ]
    )

    assert code == 1
    summary = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert summary["policy_invalid_plan_count"] == 1
    assert summary["plans"][0]["policy_checks"]["production_approved_false"] is False


def test_audit_locked_holdout_queue_cli_discovers_plans_by_glob(tmp_path: Path) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    _write_plan(
        plans / "locked_holdout_plan_t057.json",
        plan_id="t057",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    _write_plan(
        plans / "locked_holdout_plan_t061.json",
        plan_id="t061",
        start="2026-07-24T00:00:00Z",
        end="2026-08-07T00:00:00Z",
    )

    code = main(
        [
            "--plan-glob",
            str(plans / "locked_holdout_plan_*.json"),
            "--output",
            str(tmp_path / "queue.json"),
            "--as-of-utc",
            "2026-07-09T00:00:00Z",
        ]
    )

    assert code == 0
    summary = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert summary["plan_count"] == 2
    assert [row["plan_id"] for row in summary["plans"]] == ["t057", "t061"]
    assert summary["artifact_invalid_plan_count"] == 0


def test_audit_locked_holdout_queue_cli_returns_nonzero_for_tampered_artifact(tmp_path: Path) -> None:
    plan = _write_plan(
        tmp_path / "bad_artifact.json",
        plan_id="bad_artifact",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    payload = json.loads(plan.read_text(encoding="utf-8"))
    Path(payload["lab_manifest"]).write_text("tampered\n", encoding="utf-8")

    code = main(
        [
            "--plan-json",
            str(plan),
            "--output",
            str(tmp_path / "queue.json"),
            "--as-of-utc",
            "2026-07-09T00:00:00Z",
        ]
    )

    assert code == 1
    summary = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert summary["artifact_invalid_plan_count"] == 1
    assert summary["plans"][0]["artifact_checks"]["lab_manifest_sha256_bound"] is False


def test_audit_locked_holdout_queue_blocks_duplicate_plan_id(tmp_path: Path) -> None:
    first = _write_plan(
        tmp_path / "first.json",
        plan_id="duplicate",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    second = _write_plan(
        tmp_path / "second.json",
        plan_id="duplicate",
        start="2026-07-24T00:00:00Z",
        end="2026-08-07T00:00:00Z",
    )

    summary = audit_queue(plan_jsons=[first, second], as_of_utc="2026-07-09T00:00:00Z")

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_QUEUE_INVALID"
    assert summary["duplicate_plan_id_count"] == 1
    assert summary["queue_issues"][0]["issue"] == "duplicate_plan_id"
    assert summary["queue_issues"][0]["plan_id"] == "duplicate"


def test_audit_locked_holdout_queue_blocks_overlapping_windows(tmp_path: Path) -> None:
    first = _write_plan(
        tmp_path / "first.json",
        plan_id="first",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    second = _write_plan(
        tmp_path / "second.json",
        plan_id="second",
        start="2026-07-23T00:00:00Z",
        end="2026-08-06T00:00:00Z",
    )

    summary = audit_queue(plan_jsons=[first, second], as_of_utc="2026-07-09T00:00:00Z")

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_QUEUE_INVALID"
    assert summary["overlapping_window_count"] == 1
    assert summary["queue_issues"][0]["issue"] == "overlapping_holdout_window"
    assert summary["queue_issues"][0]["left_plan_id"] == "first"
    assert summary["queue_issues"][0]["right_plan_id"] == "second"


def test_audit_locked_holdout_queue_cli_returns_nonzero_for_queue_issue(tmp_path: Path) -> None:
    first = _write_plan(
        tmp_path / "first.json",
        plan_id="first",
        start="2026-07-10T00:00:00Z",
        end="2026-07-24T00:00:00Z",
    )
    second = _write_plan(
        tmp_path / "second.json",
        plan_id="second",
        start="2026-07-23T00:00:00Z",
        end="2026-08-06T00:00:00Z",
    )

    code = main(
        [
            "--plan-json",
            str(first),
            "--plan-json",
            str(second),
            "--output",
            str(tmp_path / "queue.json"),
            "--as-of-utc",
            "2026-07-09T00:00:00Z",
        ]
    )

    assert code == 1
    summary = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert summary["overlapping_window_count"] == 1


def _write_plan(
    path: Path,
    *,
    plan_id: str,
    start: str,
    end: str,
    ompex_used_in_selection: bool = False,
    production_approved: bool = False,
) -> Path:
    artifacts = _write_artifacts(path.parent)
    path.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "plan_id": plan_id,
                "read_only": True,
                "promotion_gate": False,
                "production_approved": production_approved,
                "benchmark_policy": "locked_future_no_ompex_holdout",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": ompex_used_in_selection,
                "ompex_used_in_backtest": False,
                "frozen_at_utc": "2026-07-09T00:00:00Z",
                "holdout_start_utc": start,
                "holdout_end_utc": end,
                "baseline_csv": str(artifacts["baseline_csv"]),
                "baseline_csv_sha256": "baseline-sha",
                "adjusted_csv": str(artifacts["adjusted_csv"]),
                "adjusted_csv_sha256": artifacts["adjusted_csv_sha256"],
                "lab_manifest": str(artifacts["lab_manifest"]),
                "lab_manifest_sha256": artifacts["lab_manifest_sha256"],
                "selection_summary": str(artifacts["selection_summary"]),
                "selection_summary_sha256": artifacts["selection_summary_sha256"],
                "selection_policy": {"pass": True},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["baseline_csv_sha256"] = artifacts["baseline_csv_sha256"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_artifacts(tmp_path: Path) -> dict[str, str | Path]:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    lab_manifest = tmp_path / "lab_manifest.json"
    selection_summary = tmp_path / "selection_summary.json"
    baseline.write_text("baseline\n", encoding="utf-8")
    adjusted.write_text("adjusted-sha", encoding="utf-8")
    lab_manifest.write_text('{"config":{}}\n', encoding="utf-8")
    selection_summary.write_text('{"replacement_verdict":{"replace_incumbent":true}}\n', encoding="utf-8")
    return {
        "baseline_csv": baseline,
        "baseline_csv_sha256": _sha256(baseline),
        "adjusted_csv": adjusted,
        "adjusted_csv_sha256": _sha256(adjusted),
        "lab_manifest": lab_manifest,
        "lab_manifest_sha256": _sha256(lab_manifest),
        "selection_summary": selection_summary,
        "selection_summary_sha256": _sha256(selection_summary),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
