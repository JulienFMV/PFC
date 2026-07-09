"""Summarize no-OMPEX spot backtests for EPEX shape-lab sweep trials.

This script is read-only. It ranks already-backtested lab trials on realized
EPEX spot bucket evidence, optionally compares them to an incumbent lab
candidate, and never approves production promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


NIGHT_KEY = "residual_level:night_00_05"
RAMP_KEY = "hourly_ramp:all"
OVERALL_KEY = "residual_level:all"
CORE_BUCKETS = {
    "evening_mae_improvement_eur_mwh": "residual_level:evening_ramp_17_21",
    "solar_tail_mae_improvement_eur_mwh": "residual_level:solar_tail_mar_oct_10_16",
    "weekend_mae_improvement_eur_mwh": "residual_level:weekend",
}
CORE_REPLACEMENT_KEYS = [
    "overall_mae_improvement_eur_mwh",
    "evening_mae_improvement_eur_mwh",
    "solar_tail_mae_improvement_eur_mwh",
    "weekend_mae_improvement_eur_mwh",
    "post_valuation_mae_improvement_eur_mwh",
]


def summarize_spot_backtests(
    *,
    sweep_summary: Path,
    backtest_root: Path,
    output_dir: Path,
    incumbent_backtest: Path | None = None,
    min_night_positive_folds: int = 8,
    min_ramp_positive_folds: int = 8,
    max_ramp_regression_eur_mwh: float = 0.002,
) -> dict[str, Any]:
    sweep = _read_json(sweep_summary)
    _validate_sweep(sweep)
    incumbent = _incumbent_row(incumbent_backtest) if incumbent_backtest is not None else None
    rows = []
    eligible_trial_ids = _eligible_trial_ids(sweep)
    for trial_id in eligible_trial_ids:
        summary_path = backtest_root / trial_id / "spot_backtest_summary.json"
        if not summary_path.exists():
            rows.append(_missing_row(trial_id, summary_path))
            continue
        rows.append(
            _trial_row(
                trial_id,
                _read_json(summary_path),
                summary_path=summary_path,
                incumbent=incumbent,
                min_night_positive_folds=min_night_positive_folds,
                min_ramp_positive_folds=min_ramp_positive_folds,
                max_ramp_regression_eur_mwh=max_ramp_regression_eur_mwh,
            )
        )

    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            [
                "weak_bucket_candidate",
                "night_mae_improvement_eur_mwh",
                "ramp_mae_improvement_eur_mwh",
                "overall_mae_improvement_eur_mwh",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        ranking.insert(0, "spot_rank", range(1, len(ranking) + 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_csv = output_dir / "spot_backtest_trial_ranking.csv"
    ranking.to_csv(ranking_csv, index=False)
    best_weak = _first_row(ranking[ranking["weak_bucket_candidate"].astype(bool)]) if not ranking.empty else None
    best_replacement = (
        _first_row(ranking[ranking["replacement_candidate"].astype(bool)])
        if not ranking.empty and "replacement_candidate" in ranking.columns
        else None
    )
    selected = best_replacement or best_weak
    best_overall = _first_row(ranking.sort_values("overall_mae_improvement_eur_mwh", ascending=False)) if not ranking.empty else None
    replacement = _replacement_verdict(selected, incumbent)
    summary = {
        "schema_version": "epex_shape_lab_spot_backtest_summary.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "benchmark_policy": "summarized_eligible_trials_no_ompex",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "sweep_summary": str(sweep_summary),
        "backtest_root": str(backtest_root),
        "trial_count_from_sweep": int(len(eligible_trial_ids)),
        "trial_count_summarized": int(len(rows)),
        "strict_pass_count": int(ranking["strict_lab_gate_pass"].sum()) if not ranking.empty else 0,
        "weak_bucket_candidate_count": int(ranking["weak_bucket_candidate"].sum()) if not ranking.empty else 0,
        "replacement_candidate_count": int(ranking["replacement_candidate"].sum()) if not ranking.empty else 0,
        "selected_trial": selected,
        "selected_adjusted_csv_sha256": selected.get("adjusted_csv_sha256") if selected is not None else None,
        "best_weak_bucket_trial": best_weak,
        "best_replacement_trial": best_replacement,
        "best_overall_trial": best_overall,
        "incumbent": incumbent,
        "replacement_verdict": replacement,
        "selection_thresholds": {
            "min_night_positive_folds": int(min_night_positive_folds),
            "min_ramp_positive_folds": int(min_ramp_positive_folds),
            "max_ramp_regression_eur_mwh": float(max_ramp_regression_eur_mwh),
        },
        "ranking_csv": str(ranking_csv),
    }
    (output_dir / "spot_backtest_selection_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _validate_sweep(sweep: dict[str, Any]) -> None:
    if sweep.get("benchmark_policy") != "executed_independent_no_ompex":
        raise ValueError("sweep summary must be executed_independent_no_ompex")
    if sweep.get("ompex_used_in_model") is not False or sweep.get("ompex_used_in_selection") is not False:
        raise ValueError("sweep summary must not use OMPEX")
    if sweep.get("production_approved") is not False:
        raise ValueError("sweep summary must be lab-only, not production approved")


def _eligible_trial_ids(sweep: dict[str, Any]) -> list[str]:
    ranking_csv = Path(str(sweep["ranking_csv"]))
    ranking = pd.read_csv(ranking_csv)
    if "eligible_for_selection" not in ranking.columns or "trial_id" not in ranking.columns:
        raise ValueError("sweep ranking CSV must include eligible_for_selection and trial_id")
    eligible = ranking[_bool_series(ranking["eligible_for_selection"])]
    return [str(value) for value in eligible["trial_id"].tolist()]


def _trial_row(
    trial_id: str,
    summary: dict[str, Any],
    *,
    summary_path: Path,
    incumbent: dict[str, Any] | None,
    min_night_positive_folds: int,
    min_ramp_positive_folds: int,
    max_ramp_regression_eur_mwh: float,
) -> dict[str, Any]:
    _validate_backtest(summary, summary_path=summary_path)
    row = {
        "trial_id": trial_id,
        "summary_json": str(summary_path),
        "status": str(summary.get("status")),
        "strict_lab_gate_pass": bool(summary.get("strict_lab_gate_pass")),
        "adjusted_csv_sha256": str((summary.get("source_hashes") or {}).get("adjusted_csv", "")),
        "overall_mae_improvement_eur_mwh": _overall(summary),
        "night_mae_improvement_eur_mwh": _bucket(summary, NIGHT_KEY, "mean_mae_improvement_eur_mwh"),
        "night_positive_folds": int(_bucket(summary, NIGHT_KEY, "positive_mae_improvement_fold_count")),
        "ramp_mae_improvement_eur_mwh": _bucket(summary, RAMP_KEY, "mean_mae_improvement_eur_mwh"),
        "ramp_positive_folds": int(_bucket(summary, RAMP_KEY, "positive_mae_improvement_fold_count")),
        "post_valuation_mae_improvement_eur_mwh": _post_valuation(summary),
    }
    for output_key, bucket_key in CORE_BUCKETS.items():
        row[output_key] = _bucket(summary, bucket_key, "mean_mae_improvement_eur_mwh")
    row["night_beats_incumbent"] = (
        incumbent is None or row["night_mae_improvement_eur_mwh"] > incumbent["night_mae_improvement_eur_mwh"]
    )
    row["ramp_within_incumbent_tolerance"] = (
        incumbent is None
        or row["ramp_mae_improvement_eur_mwh"]
        >= incumbent["ramp_mae_improvement_eur_mwh"] - float(max_ramp_regression_eur_mwh)
    )
    row["weak_bucket_candidate"] = bool(
        row["strict_lab_gate_pass"]
        and row["night_positive_folds"] >= int(min_night_positive_folds)
        and row["ramp_positive_folds"] >= int(min_ramp_positive_folds)
        and row["night_beats_incumbent"]
        and row["ramp_within_incumbent_tolerance"]
    )
    row["degraded_vs_incumbent"] = ",".join(_degraded_vs_incumbent(row, incumbent))
    row["replacement_candidate"] = bool(
        incumbent is not None and row["weak_bucket_candidate"] and not row["degraded_vs_incumbent"]
    )
    return row


def _incumbent_row(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    summary = _read_json(path)
    _validate_backtest(summary, summary_path=path)
    row = {
        "summary_json": str(path),
        "adjusted_csv_sha256": str((summary.get("source_hashes") or {}).get("adjusted_csv", "")),
        "overall_mae_improvement_eur_mwh": _overall(summary),
        "night_mae_improvement_eur_mwh": _bucket(summary, NIGHT_KEY, "mean_mae_improvement_eur_mwh"),
        "night_positive_folds": int(_bucket(summary, NIGHT_KEY, "positive_mae_improvement_fold_count")),
        "ramp_mae_improvement_eur_mwh": _bucket(summary, RAMP_KEY, "mean_mae_improvement_eur_mwh"),
        "ramp_positive_folds": int(_bucket(summary, RAMP_KEY, "positive_mae_improvement_fold_count")),
        "post_valuation_mae_improvement_eur_mwh": _post_valuation(summary),
    }
    for output_key, bucket_key in CORE_BUCKETS.items():
        row[output_key] = _bucket(summary, bucket_key, "mean_mae_improvement_eur_mwh")
    return row


def _replacement_verdict(best_weak: dict[str, Any] | None, incumbent: dict[str, Any] | None) -> dict[str, Any]:
    if best_weak is None:
        return {"status": "NO_WEAK_BUCKET_CANDIDATE", "replace_incumbent": False}
    if incumbent is None:
        return {"status": "NO_INCUMBENT_PROVIDED", "replace_incumbent": False}
    degraded = _degraded_vs_incumbent(best_weak, incumbent)
    if degraded:
        return {
            "status": "WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS",
            "replace_incumbent": False,
            "degraded_vs_incumbent": degraded,
        }
    return {"status": "WEAK_BUCKET_AND_CORE_METRICS_BEAT_INCUMBENT", "replace_incumbent": True}


def _degraded_vs_incumbent(row: dict[str, Any], incumbent: dict[str, Any] | None) -> list[str]:
    if incumbent is None:
        return []
    degraded = []
    for key in CORE_REPLACEMENT_KEYS:
        if float(row[key]) < float(incumbent[key]):
            degraded.append(key)
    return degraded


def _validate_backtest(summary: dict[str, Any], *, summary_path: Path) -> None:
    if summary.get("benchmark_policy") != "rolling_origin_epex_spot_no_ompex_lab_only":
        raise ValueError(f"{summary_path} is not a no-OMPEX rolling spot backtest")
    if (
        summary.get("ompex_used_in_model") is not False
        or summary.get("ompex_used_in_selection") is not False
        or summary.get("ompex_used_in_backtest") is not False
    ):
        raise ValueError(f"{summary_path} uses OMPEX")
    if summary.get("production_approved") is not False or summary.get("promotion_gate") is not False:
        raise ValueError(f"{summary_path} must remain lab-only")


def _bucket(summary: dict[str, Any], key: str, field: str) -> float:
    bucket = (summary.get("rolling_bucket_metrics") or {}).get(key)
    if not isinstance(bucket, dict) or field not in bucket:
        raise ValueError(f"spot backtest missing bucket metric {key}.{field}")
    return float(bucket[field])


def _overall(summary: dict[str, Any]) -> float:
    metrics = summary.get("rolling_metrics") or {}
    return float(metrics["mean_profile_mae_improvement_eur_mwh"])


def _post_valuation(summary: dict[str, Any]) -> float:
    metrics = summary.get("post_valuation_metrics") or {}
    return float(metrics.get("residual_mae_improvement_eur_mwh", 0.0))


def _missing_row(trial_id: str, summary_path: Path) -> dict[str, Any]:
    row = {
        "trial_id": trial_id,
        "summary_json": str(summary_path),
        "status": "MISSING_BACKTEST",
        "strict_lab_gate_pass": False,
        "adjusted_csv_sha256": "",
        "overall_mae_improvement_eur_mwh": 0.0,
        "night_mae_improvement_eur_mwh": 0.0,
        "night_positive_folds": 0,
        "ramp_mae_improvement_eur_mwh": 0.0,
        "ramp_positive_folds": 0,
        "post_valuation_mae_improvement_eur_mwh": 0.0,
        "night_beats_incumbent": False,
        "ramp_within_incumbent_tolerance": False,
        "weak_bucket_candidate": False,
        "degraded_vs_incumbent": "",
        "replacement_candidate": False,
    }
    for output_key in CORE_BUCKETS:
        row[output_key] = 0.0
    return row


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_row(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    return _jsonable(frame.iloc[0].to_dict())


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(val) for val in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--backtest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incumbent-backtest", type=Path)
    parser.add_argument("--min-night-positive-folds", type=int, default=8)
    parser.add_argument("--min-ramp-positive-folds", type=int, default=8)
    parser.add_argument("--max-ramp-regression-eur-mwh", type=float, default=0.002)
    args = parser.parse_args(argv)
    summary = summarize_spot_backtests(
        sweep_summary=args.sweep_summary,
        backtest_root=args.backtest_root,
        output_dir=args.output_dir,
        incumbent_backtest=args.incumbent_backtest,
        min_night_positive_folds=args.min_night_positive_folds,
        min_ramp_positive_folds=args.min_ramp_positive_folds,
        max_ramp_regression_eur_mwh=args.max_ramp_regression_eur_mwh,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
