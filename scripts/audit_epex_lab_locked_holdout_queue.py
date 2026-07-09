"""Audit the queue of locked EPEX lab holdout plans.

This read-only helper summarizes multiple locked holdout plans so operators can
see which future window is active, which plan SHA must be used, and whether the
next action is to wait, refresh spot data, or fix an invalid plan. It does not
fetch spot data, run backtests, or approve production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "epex_lab_locked_holdout_queue_audit.v1"
PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
LOCKED_HOLDOUT_POLICY = "locked_future_no_ompex_holdout"
DEFAULT_SEARCH_ROOT = "output/phase14"


def audit_queue(
    *,
    plan_jsons: list[Path],
    output: Path | None = None,
    as_of_utc: str | None = None,
    search_roots: list[Path] | None = None,
    output_root: Path = Path("output/phase14"),
) -> dict[str, Any]:
    if not plan_jsons:
        raise ValueError("at least one plan JSON is required")
    as_of = _to_utc(as_of_utc) if as_of_utc else pd.Timestamp.now(tz="UTC")
    roots = search_roots or [Path(DEFAULT_SEARCH_ROOT)]
    rows = [
        _plan_status(
            plan_json=plan_json,
            as_of=as_of,
            search_roots=roots,
            output_root=output_root,
        )
        for plan_json in plan_jsons
    ]
    rows.sort(key=lambda row: (row.get("holdout_start_utc") or "", row.get("plan_id") or ""))
    invalid_count = sum(1 for row in rows if row["blocking_stage"] == "locked_holdout_plan_invalid")
    policy_invalid_count = sum(1 for row in rows if row["blocking_stage"] == "locked_holdout_plan_policy_invalid")
    artifact_invalid_count = sum(1 for row in rows if row["blocking_stage"] == "locked_holdout_artifact_missing_or_hash_mismatch")
    due_count = sum(1 for row in rows if row["blocking_stage"] == "refresh_spot_and_run_locked_holdout")
    active_count = sum(1 for row in rows if row["temporal_status"] == "IN_HOLDOUT_WINDOW")
    future_count = sum(1 for row in rows if row["temporal_status"] == "WAITING_FOR_HOLDOUT_START")
    status = (
        "NO_GO_LOCKED_HOLDOUT_QUEUE_INVALID"
        if invalid_count or policy_invalid_count or artifact_invalid_count
        else "WAITING_FOR_SPOT_REFRESH_AND_LOCKED_HOLDOUT_RUN"
        if due_count
        else "WAITING_FOR_HOLDOUT_WINDOW_COMPLETION"
        if active_count
        else "WAITING_FOR_FUTURE_HOLDOUT_WINDOWS"
        if future_count
        else "NO_GO_LOCKED_HOLDOUT_QUEUE_EMPTY"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "as_of_utc": _iso(as_of),
        "status": status,
        "plan_count": len(rows),
        "invalid_plan_count": invalid_count,
        "policy_invalid_plan_count": policy_invalid_count,
        "artifact_invalid_plan_count": artifact_invalid_count,
        "future_window_count": future_count,
        "active_window_count": active_count,
        "spot_refresh_due_count": due_count,
        "search_roots": [str(_resolved_path(root)) for root in roots],
        "plans": rows,
        "next_actions": _next_actions(rows),
    }
    if output is not None:
        output = _resolved_path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _plan_status(
    *,
    plan_json: Path,
    as_of: pd.Timestamp,
    search_roots: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    plan_json = _resolved_path(plan_json)
    plan_sha = _sha256(plan_json) if plan_json.exists() else None
    plan = _read_json(plan_json) if plan_json.exists() else {}
    plan_id = str(plan.get("plan_id") or plan_json.stem)
    schema_ok = plan.get("schema_version") == PLAN_SCHEMA_VERSION
    start = _to_utc(plan.get("holdout_start_utc")) if plan.get("holdout_start_utc") else None
    end = _to_utc(plan.get("holdout_end_utc")) if plan.get("holdout_end_utc") else None
    expected = (
        pd.date_range(start, end, freq="h", inclusive="left")
        if schema_ok and start is not None and end is not None and end > start
        else pd.DatetimeIndex([])
    )
    latest_required = expected[-1] if len(expected) else None
    policy_checks = _policy_checks(plan)
    policy_ok = all(policy_checks.values())
    artifact_status = _artifact_status(plan, plan_json=plan_json)
    artifact_checks = {
        key: value
        for key, value in artifact_status.items()
        if key.endswith("_present") or key.endswith("_exists") or key.endswith("_sha256_bound")
    }
    artifacts_ok = all(artifact_checks.values())
    temporal_status = _temporal_status(as_of=as_of, start=start, latest_required=latest_required)
    if not schema_ok or start is None or end is None or end <= start:
        blocking_stage = "locked_holdout_plan_invalid"
        next_required_step = "fix_locked_holdout_plan_schema_or_window"
    elif not policy_ok:
        blocking_stage = "locked_holdout_plan_policy_invalid"
        next_required_step = "fix_locked_holdout_plan_policy_before_waiting_or_running"
    elif not artifacts_ok:
        blocking_stage = "locked_holdout_artifact_missing_or_hash_mismatch"
        next_required_step = "restore_or_regenerate_locked_candidate_artifacts_without_retuning_holdout"
    elif temporal_status == "WAITING_FOR_HOLDOUT_START":
        blocking_stage = "wait_for_holdout_window_start"
        next_required_step = "wait_without_retuning_candidate"
    elif temporal_status == "IN_HOLDOUT_WINDOW":
        blocking_stage = "wait_for_holdout_window_completion"
        next_required_step = "wait_for_full_window_then_refresh_spot"
    else:
        blocking_stage = "refresh_spot_and_run_locked_holdout"
        next_required_step = "refresh_spot_then_run_exact_sha_locked_holdout"
    commands = _recommended_commands(
        plan_json=plan_json,
        plan_sha=plan_sha,
        plan_id=plan_id,
        search_roots=search_roots,
        output_root=output_root,
    )
    return {
        "plan_json": str(plan_json),
        "plan_json_sha256": plan_sha,
        "plan_id": plan_id,
        "schema_version": plan.get("schema_version"),
        "schema_ok": schema_ok,
        "benchmark_policy": plan.get("benchmark_policy"),
        "frozen_at_utc": plan.get("frozen_at_utc"),
        "holdout_start_utc": _iso(start),
        "holdout_end_utc": _iso(end),
        "latest_required_holdout_utc": _iso(latest_required),
        "expected_holdout_hours": int(len(expected)),
        "baseline_csv_sha256": plan.get("baseline_csv_sha256"),
        "adjusted_csv_sha256": plan.get("adjusted_csv_sha256"),
        "selection_policy_pass": (plan.get("selection_policy") or {}).get("pass"),
        "temporal_status": temporal_status,
        "blocking_stage": blocking_stage,
        "next_required_step": next_required_step,
        "ready_for_spot_refresh": blocking_stage == "refresh_spot_and_run_locked_holdout",
        "production_approved": plan.get("production_approved"),
        "promotion_gate": plan.get("promotion_gate"),
        "policy_checks": policy_checks,
        "artifact_status": artifact_status,
        "artifact_checks": artifact_checks,
        "recommended_commands": commands,
    }


def _policy_checks(plan: dict[str, Any]) -> dict[str, bool]:
    return {
        "schema_version": plan.get("schema_version") == PLAN_SCHEMA_VERSION,
        "benchmark_policy_locked": plan.get("benchmark_policy") == LOCKED_HOLDOUT_POLICY,
        "read_only": plan.get("read_only") is True,
        "promotion_gate_false": plan.get("promotion_gate") is False,
        "production_approved_false": plan.get("production_approved") is False,
        "ompex_not_model": plan.get("ompex_used_in_model") is False,
        "ompex_not_selection": plan.get("ompex_used_in_selection") is False,
        "ompex_not_backtest": plan.get("ompex_used_in_backtest") is False,
        "adjusted_csv_sha256_present": bool(str(plan.get("adjusted_csv_sha256") or "").strip()),
        "baseline_csv_sha256_present": bool(str(plan.get("baseline_csv_sha256") or "").strip()),
    }


def _temporal_status(
    *,
    as_of: pd.Timestamp,
    start: pd.Timestamp | None,
    latest_required: pd.Timestamp | None,
) -> str:
    if start is None or latest_required is None:
        return "INVALID_WINDOW"
    if as_of < start:
        return "WAITING_FOR_HOLDOUT_START"
    if as_of <= latest_required:
        return "IN_HOLDOUT_WINDOW"
    return "HOLDOUT_WINDOW_COMPLETE"


def _recommended_commands(
    *,
    plan_json: Path,
    plan_sha: str | None,
    plan_id: str,
    search_roots: list[Path],
    output_root: Path,
) -> dict[str, str | None]:
    output_dir = output_root / plan_id
    discovery_json = output_dir / "spot_parquet_discovery.json"
    energy_charts_dir = output_dir / "energy_charts_locked_holdout"
    commands: dict[str, str | None] = {
        "discover_spot_parquet": _join_command(
            [
                "python",
                "scripts/discover_epex_spot_parquet_candidates.py",
                "--plan-json",
                str(plan_json),
                *[
                    item
                    for root in search_roots
                    for item in ["--search-root", str(root)]
                ],
                "--output-json",
                str(discovery_json),
                "--max-candidates",
                "5",
            ]
        ),
        "run_energy_charts_locked_holdout": None,
    }
    if plan_sha:
        commands["run_energy_charts_locked_holdout"] = _join_command(
            [
                "python",
                "scripts/run_energy_charts_epex_locked_holdout.py",
                "--plan-json",
                str(plan_json),
                "--expected-plan-sha256",
                plan_sha,
                "--output-dir",
                str(energy_charts_dir),
            ]
        )
    return commands


def _artifact_status(plan: dict[str, Any], *, plan_json: Path) -> dict[str, Any]:
    baseline = _file_binding_status(plan, plan_json=plan_json, path_key="baseline_csv")
    adjusted = _file_binding_status(plan, plan_json=plan_json, path_key="adjusted_csv")
    lab_manifest = _file_binding_status(plan, plan_json=plan_json, path_key="lab_manifest")
    selection_summary = _file_binding_status(plan, plan_json=plan_json, path_key="selection_summary")
    return {
        "baseline_csv": baseline,
        "adjusted_csv": adjusted,
        "lab_manifest": lab_manifest,
        "selection_summary": selection_summary,
        "baseline_csv_present": baseline["path_present"],
        "baseline_csv_exists": baseline["file_exists"],
        "baseline_csv_sha256_bound": baseline["sha256_bound"],
        "adjusted_csv_present": adjusted["path_present"],
        "adjusted_csv_exists": adjusted["file_exists"],
        "adjusted_csv_sha256_bound": adjusted["sha256_bound"],
        "lab_manifest_present": lab_manifest["path_present"],
        "lab_manifest_exists": lab_manifest["file_exists"],
        "lab_manifest_sha256_bound": lab_manifest["sha256_bound"],
        "selection_summary_present": selection_summary["path_present"],
        "selection_summary_exists": selection_summary["file_exists"],
        "selection_summary_sha256_bound": selection_summary["sha256_bound"],
    }


def _file_binding_status(plan: dict[str, Any], *, plan_json: Path, path_key: str) -> dict[str, Any]:
    raw_path = plan.get(path_key)
    expected_sha = plan.get(f"{path_key}_sha256")
    resolved = _resolve_plan_path(raw_path, plan_json=plan_json) if raw_path else None
    exists = bool(resolved is not None and resolved.exists())
    actual_sha = _sha256(resolved) if resolved is not None and exists else None
    return {
        "path": str(raw_path) if raw_path else None,
        "resolved_path": str(resolved) if resolved is not None else None,
        "path_present": bool(str(raw_path or "").strip()),
        "expected_sha256": expected_sha,
        "expected_sha256_present": bool(str(expected_sha or "").strip()),
        "file_exists": exists,
        "actual_sha256": actual_sha,
        "sha256_bound": bool(expected_sha and actual_sha and expected_sha == actual_sha),
    }


def _resolve_plan_path(value: Any, *, plan_json: Path) -> Path:
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return _resolved_path(raw)
    candidates = []
    repo_root = _repo_root_for_plan(plan_json)
    if repo_root is not None:
        candidates.append(repo_root / raw)
    candidates.extend([Path.cwd() / raw, plan_json.parent / raw, raw])
    for candidate in candidates:
        resolved = _resolved_path(candidate)
        if resolved.exists():
            return resolved
    return _resolved_path(candidates[0])


def _repo_root_for_plan(plan_json: Path) -> Path | None:
    for parent in [plan_json.parent, *plan_json.parents]:
        if (parent / ".git").exists() or (parent / "AGENTS.md").exists():
            return parent
    return None


def _next_actions(rows: list[dict[str, Any]]) -> list[dict[str, str | bool | None]]:
    return [
        {
            "plan_id": row["plan_id"],
            "blocking_stage": row["blocking_stage"],
            "next_required_step": row["next_required_step"],
            "ready_for_spot_refresh": row["ready_for_spot_refresh"],
            "holdout_start_utc": row["holdout_start_utc"],
            "holdout_end_utc": row["holdout_end_utc"],
            "plan_json_sha256": row["plan_json_sha256"],
        }
        for row in rows
    ]


def _to_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return _to_utc(value).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _join_command(parts: list[str]) -> str:
    return " ".join(_quote(part) for part in parts)


def _quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of-utc", default=None)
    parser.add_argument("--search-root", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("output/phase14"))
    args = parser.parse_args(argv)
    summary = audit_queue(
        plan_jsons=args.plan_json,
        output=args.output,
        as_of_utc=args.as_of_utc,
        search_roots=args.search_root,
        output_root=args.output_root,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["invalid_plan_count"] == 0 and summary["policy_invalid_plan_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
