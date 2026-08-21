"""Analyze no-OMPEX EPEX shape-lab sweep parameter sensitivity.

This diagnostic is lab-only. It reads a pre-registered sweep plan and a
spot-backtest selection summary, joins trial parameters with realized metrics,
and writes parameter-response tables. It does not approve production and does
not read OMPEX/HFC benchmark files.
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


REPO_ROOT = Path(__file__).resolve().parent.parent
IMPROVEMENT_COLUMNS = [
    "overall_mae_improvement_eur_mwh",
    "post_valuation_mae_improvement_eur_mwh",
    "evening_mae_improvement_eur_mwh",
    "solar_tail_mae_improvement_eur_mwh",
    "weekend_mae_improvement_eur_mwh",
    "night_mae_improvement_eur_mwh",
    "ramp_mae_improvement_eur_mwh",
]
BOOLEAN_COLUMNS = ["weak_bucket_candidate", "replacement_candidate"]


def analyze_sweep_sensitivity(
    *,
    plan_json: Path,
    selection_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    plan_json = _resolve_cli_path(plan_json)
    selection_summary = _resolve_cli_path(selection_summary)
    output_dir = _resolve_cli_path(output_dir)
    plan = _read_json(plan_json)
    selection = _read_json(selection_summary)
    _validate_plan(plan)
    _validate_selection(selection)
    ranking_csv = _resolve_recorded_path(selection["ranking_csv"])
    ranking = pd.read_csv(ranking_csv)
    params = _trial_parameters(plan)
    merged = ranking.merge(params, on="trial_id", how="left", validate="one_to_one")
    missing_params = sorted(merged.loc[merged["parameter_set_missing"], "trial_id"].astype(str).tolist())
    if missing_params:
        raise ValueError(f"ranking includes trials missing from plan: {missing_params}")
    for col in BOOLEAN_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].map(_bool_value).astype(bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = output_dir / "trial_parameter_metrics.csv"
    parameter_csv = output_dir / "parameter_sensitivity.csv"
    correlation_csv = output_dir / "parameter_metric_correlations.csv"
    merged.to_csv(merged_csv, index=False)
    parameter_sensitivity = _parameter_sensitivity(merged, params)
    parameter_sensitivity.to_csv(parameter_csv, index=False)
    correlations = _parameter_metric_correlations(merged, params)
    correlations.to_csv(correlation_csv, index=False)
    summary_json = output_dir / "sweep_sensitivity_summary.json"
    out = {
        "schema_version": "epex_shape_lab_sweep_sensitivity.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "benchmark_policy": "parameter_sensitivity_no_ompex_lab_only",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "plan_json": str(plan_json),
        "plan_json_sha256": _sha256(plan_json),
        "selection_summary": str(selection_summary),
        "selection_summary_sha256": _sha256(selection_summary),
        "ranking_csv": str(ranking_csv),
        "trial_count": int(len(merged)),
        "strict_pass_count": int(_bool_series(merged.get("strict_lab_gate_pass", pd.Series(dtype=bool))).sum()),
        "weak_bucket_candidate_count": int(_bool_series(merged.get("weak_bucket_candidate", pd.Series(dtype=bool))).sum()),
        "replacement_candidate_count": int(
            _bool_series(merged.get("replacement_candidate", pd.Series(dtype=bool))).sum()
        ),
        "replacement_verdict": selection.get("replacement_verdict"),
        "incumbent": selection.get("incumbent"),
        "best_by_metric": _best_by_metric(merged),
        "parameter_sensitivity_csv": str(parameter_csv),
        "parameter_metric_correlations_csv": str(correlation_csv),
        "trial_parameter_metrics_csv": str(merged_csv),
        "summary_json": str(summary_json),
        "next_hypothesis_hint": _next_hypothesis_hint(selection),
    }
    summary_json.write_text(json.dumps(_jsonable(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("activation_status") != "lab_only":
        raise ValueError("sweep plan must be lab_only")
    if plan.get("benchmark_policy") != "pre_registered_independent_no_ompex":
        raise ValueError("sweep plan must be pre_registered_independent_no_ompex")
    if plan.get("production_approved") is not False:
        raise ValueError("sweep plan must not be production approved")
    if plan.get("ompex_used_in_model") is not False or plan.get("ompex_used_in_selection") is not False:
        raise ValueError("sweep plan must not use OMPEX")
    if not isinstance(plan.get("trials"), list) or not plan["trials"]:
        raise ValueError("sweep plan must contain trials")


def _validate_selection(selection: dict[str, Any]) -> None:
    if selection.get("benchmark_policy") != "summarized_eligible_trials_no_ompex":
        raise ValueError("selection summary must be summarized_eligible_trials_no_ompex")
    if selection.get("production_approved") is not False or selection.get("promotion_gate") is not False:
        raise ValueError("selection summary must not approve production")
    if (
        selection.get("ompex_used_in_model") is not False
        or selection.get("ompex_used_in_selection") is not False
        or selection.get("ompex_used_in_backtest") is not False
    ):
        raise ValueError("selection summary must not use OMPEX")
    if "ranking_csv" not in selection:
        raise ValueError("selection summary must include ranking_csv")


def _trial_parameters(plan: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for trial in plan["trials"]:
        params = dict(trial.get("parameters") or {})
        rows.append({"trial_id": str(trial["trial_id"]), **params, "parameter_set_missing": False})
    return pd.DataFrame(rows)


def _parameter_sensitivity(merged: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parameter_columns = [col for col in params.columns if col not in {"trial_id", "parameter_set_missing"}]
    for parameter in parameter_columns:
        for value, group in merged.groupby(parameter, dropna=False):
            row: dict[str, Any] = {
                "parameter": parameter,
                "value": value,
                "trial_count": int(len(group)),
                "strict_pass_count": int(_bool_series(group.get("strict_lab_gate_pass", pd.Series(dtype=bool))).sum()),
                "weak_bucket_candidate_count": int(
                    _bool_series(group.get("weak_bucket_candidate", pd.Series(dtype=bool))).sum()
                ),
                "replacement_candidate_count": int(
                    _bool_series(group.get("replacement_candidate", pd.Series(dtype=bool))).sum()
                ),
            }
            for metric in IMPROVEMENT_COLUMNS:
                if metric not in group.columns:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_max"] = float(values.max())
                best_idx = values.idxmax()
                row[f"{metric}_best_trial_id"] = str(group.loc[best_idx, "trial_id"])
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["parameter", "value"]).reset_index(drop=True)


def _parameter_metric_correlations(merged: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    rows = []
    parameter_columns = [col for col in params.columns if col not in {"trial_id", "parameter_set_missing"}]
    for parameter in parameter_columns:
        x = pd.to_numeric(merged[parameter], errors="coerce")
        if x.nunique(dropna=True) < 2:
            continue
        for metric in IMPROVEMENT_COLUMNS:
            if metric not in merged.columns:
                continue
            y = pd.to_numeric(merged[metric], errors="coerce")
            valid = x.notna() & y.notna()
            corr = float(x[valid].corr(y[valid])) if int(valid.sum()) >= 2 else None
            rows.append({"parameter": parameter, "metric": metric, "pearson_correlation": corr})
    if not rows:
        return pd.DataFrame(columns=["parameter", "metric", "pearson_correlation"])
    return pd.DataFrame(rows).sort_values(["parameter", "metric"]).reset_index(drop=True)


def _best_by_metric(merged: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for metric in IMPROVEMENT_COLUMNS:
        if metric not in merged.columns:
            continue
        values = pd.to_numeric(merged[metric], errors="coerce")
        if values.notna().sum() == 0:
            continue
        idx = values.idxmax()
        out[metric] = {
            "trial_id": str(merged.loc[idx, "trial_id"]),
            "value": float(values.loc[idx]),
            "replacement_candidate": bool(_bool_value(merged.loc[idx].get("replacement_candidate", False))),
            "weak_bucket_candidate": bool(_bool_value(merged.loc[idx].get("weak_bucket_candidate", False))),
        }
    return out


def _next_hypothesis_hint(selection: dict[str, Any]) -> str:
    verdict = selection.get("replacement_verdict") or {}
    if verdict.get("replace_incumbent") is True:
        return "candidate_replacement_requires_full_strict_and_future_holdout_path"
    degraded = verdict.get("degraded_vs_incumbent") or []
    if "post_valuation_mae_improvement_eur_mwh" in degraded:
        return "protect_post_valuation_before_expanding_weak_bucket_gains"
    return "pre_register_next_no_ompex_line_before_any_additional_tuning"


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
    if pd.isna(value):
        return False
    return bool(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = analyze_sweep_sensitivity(
        plan_json=args.plan_json,
        selection_summary=args.selection_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps({"trial_count": summary["trial_count"], "summary_json": summary["summary_json"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
