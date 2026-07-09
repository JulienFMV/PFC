"""Pre-register a locked future no-OMPEX holdout for an EPEX lab candidate.

The plan freezes an already selected adjusted CSV and records the future spot
window that must be evaluated later. It does not read OMPEX, does not run a
model, and does not approve production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from export_local_test_ch_hourly_csv import _parse_timestamp_ch


SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
BENCHMARK_POLICY = "locked_future_no_ompex_holdout"
CANDIDATE_TIMESTAMP_COLUMN = "timestamp_ch"
CANDIDATE_UTC_OFFSET_COLUMN = "utc_offset_ch"


def build_plan(
    *,
    baseline_csv: Path,
    adjusted_csv: Path,
    output: Path | None = None,
    frozen_at_utc: str,
    holdout_start_utc: str,
    holdout_end_utc: str,
    plan_id: str = "t057_locked_t056_future_holdout",
    selection_summary: Path | None = None,
    lab_manifest: Path | None = None,
    valuation_timestamp_utc: str | None = None,
    embargo_days: int = 1,
    eval_days: int | None = None,
    min_holdout_hours: int = 168,
    min_residual_mae_improvement_eur_mwh: float = 0.0,
) -> dict[str, Any]:
    baseline_csv = _resolved_path(baseline_csv)
    adjusted_csv = _resolved_path(adjusted_csv)
    output = _resolved_path(output) if output is not None else None
    selection_summary = _resolved_path(selection_summary) if selection_summary is not None else None
    lab_manifest = _resolved_path(lab_manifest) if lab_manifest is not None else None
    frozen_at = _utc_text(frozen_at_utc)
    holdout_start = _utc_text(holdout_start_utc)
    holdout_end = _utc_text(holdout_end_utc)
    if holdout_start <= frozen_at:
        raise ValueError("holdout_start_utc must be after frozen_at_utc")
    if holdout_end <= holdout_start:
        raise ValueError("holdout_end_utc must be after holdout_start_utc")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    if min_holdout_hours <= 0:
        raise ValueError("min_holdout_hours must be positive")

    baseline_sha = _sha256(baseline_csv)
    adjusted_sha = _sha256(adjusted_csv)
    candidate_timestamp_identity = _candidate_timestamp_identity(
        baseline_csv=baseline_csv,
        adjusted_csv=adjusted_csv,
    )
    if selection_summary is None:
        raise ValueError("selection_summary is required for a locked holdout plan")
    if lab_manifest is None:
        raise ValueError("lab_manifest is required for a locked holdout plan")
    selection = _load_json(selection_summary) if selection_summary is not None else None
    selection_policy = _selection_policy(selection, adjusted_sha) if selection is not None else None
    if selection_policy is not None and selection_policy["pass"] is not True:
        raise ValueError("selection summary is not bound to the adjusted CSV or is not no-OMPEX replacement-approved")

    valuation = _utc_text(valuation_timestamp_utc or frozen_at)
    eval_days_value = int(eval_days) if eval_days is not None else _days_between(holdout_start, holdout_end)
    if eval_days_value <= 0:
        raise ValueError("eval_days must be positive")

    lab = _load_json(lab_manifest) if lab_manifest is not None else None
    plan = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "read_only": True,
        "activation_status": "locked_future_holdout",
        "promotion_gate": False,
        "production_approved": False,
        "benchmark_policy": BENCHMARK_POLICY,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "forbidden_inputs": ["OMPEX", "HFC_OMPEX", "external_HPFC_benchmark"],
        "frozen_at_utc": frozen_at,
        "holdout_start_utc": holdout_start,
        "holdout_end_utc": holdout_end,
        "baseline_csv": str(baseline_csv),
        "baseline_csv_sha256": baseline_sha,
        "adjusted_csv": str(adjusted_csv),
        "adjusted_csv_sha256": adjusted_sha,
        "candidate_timestamp_identity": candidate_timestamp_identity,
        "selection_summary": str(selection_summary) if selection_summary is not None else None,
        "selection_summary_sha256": _sha256(selection_summary) if selection_summary is not None else None,
        "selection_policy": selection_policy,
        "lab_manifest": str(lab_manifest) if lab_manifest is not None else None,
        "lab_manifest_sha256": _sha256(lab_manifest) if lab_manifest is not None else None,
        "locked_lab_config": (lab or {}).get("config"),
        "backtest": {
            "valuation_timestamp_utc": valuation,
            "embargo_days": int(embargo_days),
            "eval_days": eval_days_value,
            "min_eval_hours": 24,
            "lookback_years": 2,
            "max_auto_folds": 12,
        },
        "pass_criteria": {
            "adjusted_csv_sha256": adjusted_sha,
            "baseline_csv_sha256": baseline_sha,
            "candidate_timestamp_count": candidate_timestamp_identity["candidate_timestamp_count"],
            "candidate_timestamp_set_sha256": candidate_timestamp_identity["candidate_timestamp_set_sha256"],
            "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
            "strict_lab_gate_pass": True,
            "min_holdout_hours": int(min_holdout_hours),
            "min_residual_mae_improvement_eur_mwh": float(min_residual_mae_improvement_eur_mwh),
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
        },
        "commands": {
            "run_locked_holdout_template": _join_command(
                [
                    "python",
                    "scripts/run_epex_lab_locked_holdout.py",
                    "--plan-json",
                    str(output or "<T057_PLAN_JSON>"),
                    "--expected-plan-sha256",
                    "<T057_PLAN_JSON_SHA256>",
                    "--spot-parquet",
                    "<FUTURE_SPOT_PARQUET>",
                    "--output-dir",
                    "<T057_HOLDOUT_OUTPUT_DIR>",
                ]
            ),
            "run_future_backtest_template": _join_command(
                [
                    "python",
                    "scripts/backtest_epex_shape_lab_against_spot.py",
                    "--baseline-csv",
                    str(baseline_csv),
                    "--adjusted-csv",
                    str(adjusted_csv),
                    "--spot-parquet",
                    "<FUTURE_SPOT_PARQUET>",
                    "--output-dir",
                    "<T057_HOLDOUT_OUTPUT_DIR>",
                    "--valuation-timestamp",
                    valuation,
                    "--lookback-years",
                    "2",
                    "--eval-days",
                    str(eval_days_value),
                    "--embargo-days",
                    str(int(embargo_days)),
                    "--max-auto-folds",
                    "12",
                    "--min-eval-hours",
                    "24",
                ]
            ),
            "audit_future_holdout_template": _join_command(
                [
                    "python",
                    "scripts/audit_epex_lab_locked_holdout.py",
                    "--plan-json",
                    str(output or "<T057_PLAN_JSON>"),
                    "--spot-backtest-summary",
                    "<T057_HOLDOUT_OUTPUT_DIR>/spot_backtest_summary.json",
                    "--output",
                    "<T057_HOLDOUT_OUTPUT_DIR>/locked_holdout_audit.json",
                ]
            ),
        },
        "note": (
            "This pre-registration freezes the selected T056 candidate before "
            "future spot rows are available. It must not be edited after the "
            "holdout window starts; create a new plan instead."
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def _selection_policy(selection: dict[str, Any], adjusted_sha: str) -> dict[str, Any]:
    selected_trial = selection.get("selected_trial") if isinstance(selection.get("selected_trial"), dict) else {}
    verdict = selection.get("replacement_verdict") if isinstance(selection.get("replacement_verdict"), dict) else {}
    selected_shas = {
        str(value)
        for value in [
            selection.get("selected_adjusted_csv_sha256"),
            selected_trial.get("adjusted_csv_sha256"),
        ]
        if value
    }
    value = {
        "selection_summary_present": True,
        "adjusted_csv_sha256": adjusted_sha,
        "replace_incumbent": verdict.get("replace_incumbent"),
        "selected_adjusted_csv_sha256": selection.get("selected_adjusted_csv_sha256"),
        "selected_trial_adjusted_csv_sha256": selected_trial.get("adjusted_csv_sha256"),
        "ompex_used_in_model": selection.get("ompex_used_in_model"),
        "ompex_used_in_selection": selection.get("ompex_used_in_selection"),
        "ompex_used_in_backtest": selection.get("ompex_used_in_backtest"),
    }
    value["pass"] = bool(
        adjusted_sha in selected_shas
        and verdict.get("replace_incumbent") is True
        and selection.get("ompex_used_in_model") is False
        and selection.get("ompex_used_in_selection") is False
        and selection.get("ompex_used_in_backtest") is False
    )
    return value


def _candidate_timestamp_identity(*, baseline_csv: Path, adjusted_csv: Path) -> dict[str, Any]:
    baseline = _candidate_timestamp_status(baseline_csv)
    adjusted = _candidate_timestamp_status(adjusted_csv)
    timestamp_sets_identical = bool(
        baseline["timestamp_set_sha256"]
        and adjusted["timestamp_set_sha256"]
        and baseline["timestamp_set_sha256"] == adjusted["timestamp_set_sha256"]
    )
    checks = {
        "baseline_required_columns_present": baseline["required_columns_present"],
        "baseline_timestamps_parseable": baseline["timestamps_parseable"],
        "baseline_no_duplicate_timestamps": baseline["no_duplicate_timestamps"],
        "adjusted_required_columns_present": adjusted["required_columns_present"],
        "adjusted_timestamps_parseable": adjusted["timestamps_parseable"],
        "adjusted_no_duplicate_timestamps": adjusted["no_duplicate_timestamps"],
        "timestamp_sets_identical": timestamp_sets_identical,
    }
    if not all(checks.values()):
        raise ValueError("baseline and adjusted CSV timestamp identities must be parseable, unique, and identical")
    return {
        "baseline_timestamp_count": baseline["timestamp_count"],
        "baseline_timestamp_min_utc": baseline["timestamp_min_utc"],
        "baseline_timestamp_max_utc": baseline["timestamp_max_utc"],
        "baseline_timestamp_set_sha256": baseline["timestamp_set_sha256"],
        "adjusted_timestamp_count": adjusted["timestamp_count"],
        "adjusted_timestamp_min_utc": adjusted["timestamp_min_utc"],
        "adjusted_timestamp_max_utc": adjusted["timestamp_max_utc"],
        "adjusted_timestamp_set_sha256": adjusted["timestamp_set_sha256"],
        "candidate_timestamp_count": baseline["timestamp_count"],
        "candidate_timestamp_min_utc": baseline["timestamp_min_utc"],
        "candidate_timestamp_max_utc": baseline["timestamp_max_utc"],
        "candidate_timestamp_set_sha256": baseline["timestamp_set_sha256"],
        "timestamp_sets_identical": timestamp_sets_identical,
        "checks": checks,
    }


def _candidate_timestamp_status(path: Path) -> dict[str, Any]:
    failed = {
        "required_columns_present": False,
        "timestamps_parseable": False,
        "no_duplicate_timestamps": False,
        "timestamp_count": 0,
        "timestamp_min_utc": None,
        "timestamp_max_utc": None,
        "timestamp_set_sha256": None,
    }
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return failed
    required = {CANDIDATE_TIMESTAMP_COLUMN, CANDIDATE_UTC_OFFSET_COLUMN}
    if not required.issubset(frame.columns):
        return failed
    status = {**failed, "required_columns_present": True}
    try:
        parsed = _parse_timestamp_ch(frame[CANDIDATE_TIMESTAMP_COLUMN], frame[CANDIDATE_UTC_OFFSET_COLUMN])
        idx = pd.DatetimeIndex(parsed).tz_convert("UTC")
    except Exception:  # noqa: BLE001 - fail closed for malformed timestamp evidence
        return status
    status["timestamps_parseable"] = True
    status["no_duplicate_timestamps"] = not idx.has_duplicates
    observed_unique = pd.DatetimeIndex(idx.unique()).sort_values()
    status["timestamp_count"] = int(len(observed_unique))
    status["timestamp_min_utc"] = _utc_text(observed_unique[0]) if len(observed_unique) else None
    status["timestamp_max_utc"] = _utc_text(observed_unique[-1]) if len(observed_unique) else None
    status["timestamp_set_sha256"] = _timestamp_index_sha256(observed_unique)
    return status


def _timestamp_index_sha256(index: pd.DatetimeIndex) -> str:
    digest = hashlib.sha256()
    for item in index:
        digest.update(_utc_text(item).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _utc_text(value: str) -> str:
    from pandas import Timestamp

    ts = Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _days_between(start_utc: str, end_utc: str) -> int:
    from pandas import Timestamp

    start = Timestamp(start_utc)
    end = Timestamp(end_utc)
    hours = (end - start).total_seconds() / 3600.0
    return max(1, int(round(hours / 24.0)))


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _join_command(parts: list[str]) -> str:
    return " ".join(_quote(part) for part in parts)


def _quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--adjusted-csv", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--lab-manifest", type=Path, required=True)
    parser.add_argument("--plan-id", default="t057_locked_t056_future_holdout")
    parser.add_argument("--frozen-at-utc", required=True)
    parser.add_argument("--holdout-start-utc", required=True)
    parser.add_argument("--holdout-end-utc", required=True)
    parser.add_argument("--valuation-timestamp-utc", default=None)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--eval-days", type=int, default=None)
    parser.add_argument("--min-holdout-hours", type=int, default=168)
    parser.add_argument("--min-residual-mae-improvement-eur-mwh", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan(
        baseline_csv=args.baseline_csv,
        adjusted_csv=args.adjusted_csv,
        selection_summary=args.selection_summary,
        lab_manifest=args.lab_manifest,
        output=args.output,
        frozen_at_utc=args.frozen_at_utc,
        holdout_start_utc=args.holdout_start_utc,
        holdout_end_utc=args.holdout_end_utc,
        valuation_timestamp_utc=args.valuation_timestamp_utc,
        embargo_days=args.embargo_days,
        eval_days=args.eval_days,
        min_holdout_hours=args.min_holdout_hours,
        min_residual_mae_improvement_eur_mwh=args.min_residual_mae_improvement_eur_mwh,
        plan_id=args.plan_id,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
