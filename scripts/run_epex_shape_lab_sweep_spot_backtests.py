"""Run no-OMPEX spot backtests for eligible EPEX shape-lab sweep trials.

This is a lab-only orchestration helper. It reads a pre-registered sweep plan
and an executed sweep summary, runs ``backtest_epex_shape_lab_against_spot`` for
eligible trials only, and can optionally write the weak-bucket selection
summary. It never reads OMPEX/HFC benchmark files and never approves
production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_epex_shape_lab_against_spot import backtest_against_spot
from scripts.summarize_epex_shape_lab_spot_backtests import summarize_spot_backtests


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_sweep_spot_backtests(
    *,
    plan_json: Path,
    sweep_summary: Path,
    output_root: Path,
    output_summary: Path,
    incumbent_backtest: Path | None = None,
    selection_output_dir: Path | None = None,
    resume: bool = True,
    lookback_years: int = 2,
    eval_days: int = 30,
    embargo_days: int = 1,
    max_auto_folds: int = 12,
    min_eval_hours: int = 24,
) -> dict[str, Any]:
    plan_json = _resolve_cli_path(plan_json)
    sweep_summary = _resolve_cli_path(sweep_summary)
    output_root = _resolve_cli_path(output_root)
    output_summary = _resolve_cli_path(output_summary)
    incumbent_backtest = _resolve_cli_path(incumbent_backtest) if incumbent_backtest is not None else None
    selection_output_dir = _resolve_cli_path(selection_output_dir) if selection_output_dir is not None else None
    plan = _read_json(plan_json)
    sweep = _read_json(sweep_summary)
    _validate_plan(plan)
    _validate_sweep(sweep)

    candidate_csv = _resolve_recorded_path(plan["candidate_csv"])
    spot_parquet = _resolve_recorded_path(plan["spot_parquet"])
    _verify_hash(candidate_csv, str(plan["candidate_csv_sha256"]), "candidate_csv")
    _verify_hash(spot_parquet, str(plan["spot_parquet_sha256"]), "spot_parquet")

    trial_map = {str(trial["trial_id"]): trial for trial in plan["trials"]}
    eligible_trial_ids = _eligible_trial_ids(sweep)
    rows = []
    for trial_id in eligible_trial_ids:
        if trial_id not in trial_map:
            raise ValueError(f"eligible trial {trial_id!r} is missing from the plan")
        trial = trial_map[trial_id]
        adjusted_csv = _resolve_recorded_path(trial["output_dir"]) / "candidate_epex_shape_lab_adjusted.csv"
        if not adjusted_csv.exists():
            raise FileNotFoundError(f"adjusted CSV missing for trial {trial_id}: {adjusted_csv}")
        trial_output = output_root / trial_id
        summary_json = trial_output / "spot_backtest_summary.json"
        reused = bool(
            resume
            and _backtest_matches(
                summary_json=summary_json,
                baseline_csv=candidate_csv,
                adjusted_csv=adjusted_csv,
                spot_parquet=spot_parquet,
            )
        )
        if not reused:
            backtest_against_spot(
                baseline_csv=candidate_csv,
                adjusted_csv=adjusted_csv,
                spot_parquet=spot_parquet,
                output_dir=trial_output,
                valuation_timestamp=str(plan["valuation_timestamp"]),
                lookback_years=int(lookback_years),
                eval_days=int(eval_days),
                embargo_days=int(embargo_days),
                max_auto_folds=int(max_auto_folds),
                min_eval_hours=int(min_eval_hours),
            )
        summary = _read_json(summary_json)
        rows.append(
            {
                "trial_id": trial_id,
                "summary_json": str(summary_json),
                "reused_existing": reused,
                "status": summary.get("status"),
                "strict_lab_gate_pass": bool(summary.get("strict_lab_gate_pass")),
                "adjusted_csv_sha256": (summary.get("source_hashes") or {}).get("adjusted_csv"),
            }
        )

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    run_csv = output_summary.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(run_csv, index=False)
    selection_summary = None
    if selection_output_dir is not None:
        selection_summary = summarize_spot_backtests(
            sweep_summary=sweep_summary,
            backtest_root=output_root,
            output_dir=selection_output_dir,
            incumbent_backtest=incumbent_backtest,
        )

    out = {
        "schema_version": "epex_shape_lab_sweep_spot_backtest_run.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "benchmark_policy": "eligible_sweep_trials_spot_backtest_no_ompex",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "plan_json": str(plan_json),
        "sweep_summary": str(sweep_summary),
        "output_root": str(output_root),
        "trial_count_eligible": int(len(eligible_trial_ids)),
        "trial_count_backtested": int(len(rows)),
        "reused_existing_count": int(sum(1 for row in rows if row["reused_existing"])),
        "run_csv": str(run_csv),
        "selection_summary": selection_summary,
    }
    output_summary.write_text(json.dumps(_jsonable(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("activation_status") != "lab_only":
        raise ValueError("sweep plan must be lab_only")
    if plan.get("benchmark_policy") != "pre_registered_independent_no_ompex":
        raise ValueError("sweep plan must be pre_registered_independent_no_ompex")
    if plan.get("ompex_used_in_model") is not False or plan.get("ompex_used_in_selection") is not False:
        raise ValueError("sweep plan must not use OMPEX")
    if plan.get("production_approved") is not False:
        raise ValueError("sweep plan must not be production approved")
    if not isinstance(plan.get("trials"), list) or not plan["trials"]:
        raise ValueError("sweep plan must contain trials")


def _validate_sweep(sweep: dict[str, Any]) -> None:
    if sweep.get("benchmark_policy") != "executed_independent_no_ompex":
        raise ValueError("sweep summary must be executed_independent_no_ompex")
    if sweep.get("ompex_used_in_model") is not False or sweep.get("ompex_used_in_selection") is not False:
        raise ValueError("sweep summary must not use OMPEX")
    if sweep.get("production_approved") is not False:
        raise ValueError("sweep summary must not be production approved")


def _eligible_trial_ids(sweep: dict[str, Any]) -> list[str]:
    ranking = pd.read_csv(_resolve_recorded_path(sweep["ranking_csv"]))
    if "trial_id" not in ranking.columns or "eligible_for_selection" not in ranking.columns:
        raise ValueError("sweep ranking CSV must include trial_id and eligible_for_selection")
    eligible = ranking[_bool_series(ranking["eligible_for_selection"])]
    return [str(value) for value in eligible["trial_id"].tolist()]


def _backtest_matches(*, summary_json: Path, baseline_csv: Path, adjusted_csv: Path, spot_parquet: Path) -> bool:
    if not summary_json.exists():
        return False
    try:
        summary = _read_json(summary_json)
    except (OSError, json.JSONDecodeError):
        return False
    hashes = summary.get("source_hashes") or {}
    return (
        summary.get("benchmark_policy") == "rolling_origin_epex_spot_no_ompex_lab_only"
        and summary.get("ompex_used_in_model") is False
        and summary.get("ompex_used_in_selection") is False
        and summary.get("ompex_used_in_backtest") is False
        and summary.get("production_approved") is False
        and summary.get("promotion_gate") is False
        and hashes.get("baseline_csv") == _sha256(baseline_csv)
        and hashes.get("adjusted_csv") == _sha256(adjusted_csv)
        and hashes.get("spot_parquet") == _sha256(spot_parquet)
    )


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {actual}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_cli_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (Path.cwd() / path).resolve(strict=False)


def _resolve_recorded_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return bool(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--incumbent-backtest", type=Path)
    parser.add_argument("--selection-output-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--lookback-years", type=int, default=2)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--max-auto-folds", type=int, default=12)
    parser.add_argument("--min-eval-hours", type=int, default=24)
    args = parser.parse_args(argv)
    summary = run_sweep_spot_backtests(
        plan_json=args.plan_json,
        sweep_summary=args.sweep_summary,
        output_root=args.output_root,
        output_summary=args.output_summary,
        incumbent_backtest=args.incumbent_backtest,
        selection_output_dir=args.selection_output_dir,
        resume=not args.no_resume,
        lookback_years=args.lookback_years,
        eval_days=args.eval_days,
        embargo_days=args.embargo_days,
        max_auto_folds=args.max_auto_folds,
        min_eval_hours=args.min_eval_hours,
    )
    print(
        json.dumps(
            {
                "trial_count_backtested": summary["trial_count_backtested"],
                "reused_existing_count": summary["reused_existing_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
