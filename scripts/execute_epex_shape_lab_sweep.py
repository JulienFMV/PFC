"""Execute a pre-registered EPEX shape-lab sweep without OMPEX.

The executor reads a plan produced by ``plan_epex_shape_lab_sweep.py``, verifies
the input hashes, runs each trial through the EPEX lab, compares baseline vs
adjusted with independent diagnostics, audits lab governance without OMPEX, and
writes a no-OMPEX ranking summary.
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

from scripts.audit_epex_shape_lab_governance import audit_governance
from scripts.compare_epex_shape_lab_ab import compare_ab
from scripts.run_epex_shape_lab_ab import run_ab


def execute_sweep(
    *,
    plan_json: Path,
    output_summary: Path,
    max_trials: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    _validate_plan(plan, sweep_root=plan_json.resolve().parent)
    if max_trials is not None and max_trials < 0:
        raise ValueError("--max-trials must be non-negative")
    candidate_csv = Path(plan["candidate_csv"])
    spot_parquet = Path(plan["spot_parquet"])
    _verify_hash(candidate_csv, plan["candidate_csv_sha256"], "candidate_csv")
    _verify_hash(spot_parquet, plan["spot_parquet_sha256"], "spot_parquet")

    rows: list[dict[str, Any]] = []
    trials = list(plan["trials"])
    if max_trials is not None:
        trials = trials[: int(max_trials)]

    for trial in trials:
        trial_dir = Path(trial["output_dir"])
        comparison_dir = trial_dir / "independent_ab_comparison"
        governance_json = trial_dir / "governance_audit" / "epex_shape_lab_governance_audit.json"
        manifest_json = trial_dir / "ab_lab_manifest.json"
        adjusted_csv = trial_dir / "candidate_epex_shape_lab_adjusted.csv"
        independent_json = comparison_dir / "ab_comparison_summary.json"

        manifest_current = resume and _manifest_matches_plan(
            manifest_json=manifest_json,
            adjusted_csv=adjusted_csv,
            candidate_csv=candidate_csv,
            spot_parquet=spot_parquet,
            plan=plan,
            trial=trial,
        )
        reran_manifest = False
        if not manifest_current:
            run_ab(
                candidate_csv=candidate_csv,
                spot_parquet=spot_parquet,
                output_dir=trial_dir,
                valuation_timestamp=str(plan["valuation_timestamp"]),
                weekend_intensity=float(trial["parameters"]["weekend_intensity"]),
                low_tail_intensity=float(trial["parameters"]["low_tail_intensity"]),
                peak_subshape_intensity=float(trial["parameters"]["peak_subshape_intensity"]),
                evening_recovery_intensity=float(trial["parameters"].get("evening_recovery_intensity", 0.0)),
                max_abs_delta_eur_mwh=float(trial["parameters"]["max_abs_delta_eur_mwh"]),
                night_intensity=float(trial["parameters"].get("night_intensity", 0.0)),
                ramp_intensity=float(trial["parameters"].get("ramp_intensity", 0.0)),
                lookback_years=int(trial["parameters"].get("lookback_years", plan.get("lookback_years", 5))),
            )
            reran_manifest = True

        comparison_current = (
            not reran_manifest
            and resume
            and _comparison_matches_plan(
                independent_json=independent_json,
                candidate_csv=candidate_csv,
                adjusted_csv=adjusted_csv,
            )
        )
        reran_comparison = False
        if not comparison_current:
            compare_ab(
                baseline_csv=candidate_csv,
                adjusted_csv=adjusted_csv,
                output_dir=comparison_dir,
            )
            reran_comparison = True

        governance_current = (
            not reran_manifest
            and not reran_comparison
            and resume
            and _governance_matches_plan(
                governance_json=governance_json,
                manifest_json=manifest_json,
                independent_json=independent_json,
            )
        )
        if not governance_current:
            audit_governance(
                lab_manifest=manifest_json,
                independent_summary=independent_json,
                output_json=governance_json,
            )

        lab = json.loads(manifest_json.read_text(encoding="utf-8"))
        independent = json.loads(independent_json.read_text(encoding="utf-8"))
        governance = json.loads(governance_json.read_text(encoding="utf-8"))
        calendar = _read_calendar_summary(comparison_dir / "calendar_delta_summary.csv")
        annual = _read_annual_summary(comparison_dir / "annual_summary.csv")
        rows.append(
            _trial_row(
                trial,
                lab,
                independent,
                governance,
                calendar,
                annual,
                selection_thresholds=_selection_thresholds(plan),
                scoring_policy=_scoring_policy(plan),
            )
        )

    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            [
                "eligible_for_selection",
                "independent_shape_score",
                "max_abs_monthly_mean_delta_eur_mwh",
                "ramp_abs_p99_increase_eur_mwh",
            ],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    ranking_csv = output_summary.with_suffix(".csv")
    ranking.to_csv(ranking_csv, index=False)
    summary = {
        "plan_id": plan["plan_id"],
        "benchmark_policy": "executed_independent_no_ompex",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "production_approved": False,
        "trial_count_planned": int(plan["trial_count"]),
        "trial_count_executed": int(len(rows)),
        "eligible_count": int(ranking["eligible_for_selection"].sum()) if not ranking.empty else 0,
        "ranking_csv": str(ranking_csv),
        "best_trial": _best_eligible_row(ranking),
        "selection_thresholds": _selection_thresholds(plan),
        "scoring_policy": _scoring_policy(plan),
        "selection_basis": [
            "governance PASS",
            "finite and quantile ordered adjusted candidate",
            "monthly/fan drift within thresholds",
            "no weighted negative hours",
            "EPEX spot freshness and coverage thresholds when pre-registered",
            "ramp and negative price thresholds when pre-registered",
            "night/ramp components when pre-registered",
            "evening recovery component when pre-registered",
            "higher independent duck/shape score",
            "lower ramp penalty as tie-break",
        ],
        "forbidden_selection_inputs": plan["forbidden_selection_inputs"],
    }
    output_summary.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _validate_plan(plan: dict[str, Any], *, sweep_root: Path) -> None:
    required_top_level = {
        "plan_id",
        "activation_status",
        "production_approved",
        "benchmark_policy",
        "ompex_used_in_model",
        "ompex_used_in_selection",
        "candidate_csv",
        "spot_parquet",
        "candidate_csv_sha256",
        "spot_parquet_sha256",
        "valuation_timestamp",
        "forbidden_selection_inputs",
        "trial_count",
        "trials",
    }
    missing_top_level = sorted(required_top_level - set(plan))
    if missing_top_level:
        raise ValueError(f"sweep plan missing keys: {missing_top_level}")
    if plan.get("activation_status") != "lab_only":
        raise ValueError("sweep plan must be lab_only")
    if plan.get("production_approved") is not False:
        raise ValueError("sweep plan must not be production approved")
    if plan.get("benchmark_policy") != "pre_registered_independent_no_ompex":
        raise ValueError("sweep plan must be pre_registered_independent_no_ompex")
    if plan.get("ompex_used_in_model") is not False or plan.get("ompex_used_in_selection") is not False:
        raise ValueError("sweep plan must not use OMPEX for model or selection")
    forbidden = set(plan.get("forbidden_selection_inputs") or [])
    required_forbidden = {"OMPEX", "HFC_OMPEX", "external_HPFC_benchmark"}
    if not required_forbidden.issubset(forbidden):
        raise ValueError("sweep plan must explicitly forbid OMPEX/HFC selection inputs")

    trials = plan.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("sweep plan must contain at least one trial")
    if int(plan.get("trial_count")) != len(trials):
        raise ValueError("sweep plan trial_count does not match trials length")
    _selection_thresholds(plan)
    _scoring_policy(plan)

    seen_trial_ids: set[str] = set()
    seen_output_dirs: set[Path] = set()
    required_params = {"weekend_intensity", "low_tail_intensity", "peak_subshape_intensity", "max_abs_delta_eur_mwh"}
    for trial in trials:
        trial_id = str(trial.get("trial_id", ""))
        if not trial_id:
            raise ValueError("sweep trial missing trial_id")
        if trial_id in seen_trial_ids:
            raise ValueError(f"duplicate sweep trial_id: {trial_id}")
        seen_trial_ids.add(trial_id)

        output_dir = Path(str(trial.get("output_dir", ""))).resolve()
        if output_dir in seen_output_dirs:
            raise ValueError(f"duplicate sweep output_dir: {output_dir}")
        if not _is_relative_to(output_dir, sweep_root):
            raise ValueError(f"sweep trial output_dir outside sweep root: {output_dir}")
        seen_output_dirs.add(output_dir)

        params = trial.get("parameters")
        if not isinstance(params, dict):
            raise ValueError(f"sweep trial {trial_id} missing parameters")
        missing_params = sorted(required_params - set(params))
        if missing_params:
            raise ValueError(f"sweep trial {trial_id} missing parameters: {missing_params}")


def _selection_thresholds(plan: dict[str, Any]) -> dict[str, float | None]:
    raw = plan.get("selection_thresholds") or {}
    allowed = {
        "max_epex_spot_age_days",
        "min_epex_fit_coverage_days",
        "max_ramp_p99_increase_eur_mwh",
        "min_adjusted_price_eur_mwh",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown selection thresholds: {unknown}")
    return {key: _optional_threshold(raw.get(key)) for key in allowed}


def _scoring_policy(plan: dict[str, Any]) -> dict[str, float]:
    raw = plan.get("scoring_policy") or {}
    defaults = {
        "duck_weight": 1.0,
        "solar_tail_weight": 1.0,
        "weekend_weight": 1.0,
        "midday_weight": 1.0,
        "night_weight": 0.5,
        "ramp_penalty_weight": 0.25,
    }
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ValueError(f"unknown scoring policy keys: {unknown}")
    return {key: float(raw.get(key, default)) for key, default in defaults.items()}


def _optional_threshold(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _manifest_matches_plan(
    *,
    manifest_json: Path,
    adjusted_csv: Path,
    candidate_csv: Path,
    spot_parquet: Path,
    plan: dict[str, Any],
    trial: dict[str, Any],
) -> bool:
    if not manifest_json.exists():
        return False
    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    source_hashes = manifest.get("source_hashes") or {}
    config = manifest.get("config") or {}
    outputs = manifest.get("outputs") or {}
    params = trial["parameters"]
    return (
        manifest.get("activation_status") == "lab_only"
        and manifest.get("production_approved") is False
        and manifest.get("ompex_used_in_selection") is False
        and _same_path(manifest.get("candidate_csv"), candidate_csv)
        and _same_path(manifest.get("spot_parquet"), spot_parquet)
        and source_hashes.get("candidate_csv") == plan["candidate_csv_sha256"]
        and source_hashes.get("spot_parquet") == plan["spot_parquet_sha256"]
        and config.get("valuation_timestamp") == plan["valuation_timestamp"]
        and _float_equal(config.get("weekend_intensity"), params["weekend_intensity"])
        and _float_equal(config.get("low_tail_intensity"), params["low_tail_intensity"])
        and _float_equal(config.get("peak_subshape_intensity"), params["peak_subshape_intensity"])
        and _float_equal(config.get("evening_recovery_intensity", 0.0), params.get("evening_recovery_intensity", 0.0))
        and _float_equal(config.get("night_intensity", 0.0), params.get("night_intensity", 0.0))
        and _float_equal(config.get("ramp_intensity", 0.0), params.get("ramp_intensity", 0.0))
        and _float_equal(config.get("max_abs_delta_eur_mwh"), params["max_abs_delta_eur_mwh"])
        and int(config.get("lookback_years", -1)) == int(params.get("lookback_years", plan.get("lookback_years", 5)))
        and _same_path(outputs.get("adjusted_csv"), adjusted_csv)
    )


def _comparison_matches_plan(*, independent_json: Path, candidate_csv: Path, adjusted_csv: Path) -> bool:
    if not independent_json.exists():
        return False
    try:
        comparison = json.loads(independent_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        comparison.get("benchmark_policy") == "independent_no_ompex"
        and comparison.get("ompex_used_in_model") is False
        and comparison.get("ompex_used_in_selection") is False
        and _same_path(comparison.get("baseline_csv"), candidate_csv)
        and _same_path(comparison.get("adjusted_csv"), adjusted_csv)
        and "max_abs_monthly_mean_delta_eur_mwh" in comparison
        and "max_abs_width_delta_eur_mwh" in comparison
    )


def _governance_matches_plan(*, governance_json: Path, manifest_json: Path, independent_json: Path) -> bool:
    if not governance_json.exists():
        return False
    try:
        governance = json.loads(governance_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    inputs = governance.get("inputs") or {}
    return (
        governance.get("audit") == "epex_shape_lab_governance"
        and _same_path(inputs.get("lab_manifest"), manifest_json)
        and _same_path(inputs.get("independent_summary"), independent_json)
        and inputs.get("ompex_metrics") is None
        and "status" in governance
    )


def _best_eligible_row(ranking: pd.DataFrame) -> dict[str, Any] | None:
    if ranking.empty:
        return None
    eligible = ranking[ranking["eligible_for_selection"]]
    if eligible.empty:
        return None
    return eligible.iloc[0].to_dict()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _same_path(left: Any, right: Path) -> bool:
    if left is None:
        return False
    try:
        return Path(str(left)).resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


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


def _read_calendar_summary(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    out: dict[str, float] = {}
    for _, row in frame.iterrows():
        out[str(row["bucket"])] = float(row["mean_delta_eur_mwh"])
    return out


def _read_annual_summary(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return {
        "evening_minus_midday_change_mean_eur_mwh": float(frame["evening_minus_midday_change_eur_mwh"].mean()),
        "weekend_minus_weekday_change_mean_eur_mwh": float(frame["weekend_minus_weekday_change_eur_mwh"].mean()),
    }


def _trial_row(
    trial: dict[str, Any],
    lab: dict[str, Any],
    independent: dict[str, Any],
    governance: dict[str, Any],
    calendar: dict[str, float],
    annual: dict[str, float],
    *,
    selection_thresholds: dict[str, float | None],
    scoring_policy: dict[str, float],
) -> dict[str, Any]:
    lab_audit = lab.get("audit") or {}
    fit_info = lab.get("fit_info") or {}
    epex_age_days = _age_days(
        valuation_timestamp=fit_info.get("valuation_timestamp_utc")
        or (lab.get("config") or {}).get("valuation_timestamp"),
        max_timestamp=fit_info.get("max_timestamp_utc"),
    )
    epex_coverage_days = _coverage_days(
        min_timestamp=fit_info.get("min_timestamp_utc"),
        max_timestamp=fit_info.get("max_timestamp_utc"),
    )
    ramp_increase = float(independent["ramp_abs_p99_adjusted_eur_mwh"]) - float(independent["ramp_abs_p99_baseline_eur_mwh"])
    monthly_drift = float(independent["max_abs_monthly_mean_delta_eur_mwh"])
    width_drift = float(independent["max_abs_width_delta_eur_mwh"])
    min_adjusted_price = float(independent["min_adjusted_price_eur_mwh"])
    eligible = (
        governance.get("status") == "PASS"
        and independent.get("finite_adjusted_ok") is True
        and independent.get("quantile_order_adjusted_ok") is True
        and int(independent.get("weighted_negative_hours_adjusted", -1)) == 0
        and monthly_drift <= 1e-5
        and width_drift <= 1e-9
        and _threshold_max_ok(epex_age_days, selection_thresholds["max_epex_spot_age_days"])
        and _threshold_min_ok(epex_coverage_days, selection_thresholds["min_epex_fit_coverage_days"])
        and _threshold_max_ok(ramp_increase, selection_thresholds["max_ramp_p99_increase_eur_mwh"])
        and _threshold_min_ok(min_adjusted_price, selection_thresholds["min_adjusted_price_eur_mwh"])
    )
    duck_change = float(annual.get("evening_minus_midday_change_mean_eur_mwh", 0.0))
    solar_tail_delta = float(calendar.get("solar_tail_mar_oct_10_16", 0.0))
    midday_delta = float(calendar.get("midday_11_15", 0.0))
    weekend_delta = float(calendar.get("weekend", 0.0))
    night_delta = float(calendar.get("night_00_05", 0.0))
    shape_score = (
        scoring_policy["duck_weight"] * duck_change
        + scoring_policy["solar_tail_weight"] * max(0.0, -solar_tail_delta)
        + scoring_policy["midday_weight"] * max(0.0, -midday_delta)
        + scoring_policy["weekend_weight"] * max(0.0, -weekend_delta)
        + scoring_policy["night_weight"] * abs(night_delta)
        - scoring_policy["ramp_penalty_weight"] * max(0.0, ramp_increase)
    )
    return {
        "trial_id": trial["trial_id"],
        "eligible_for_selection": bool(eligible),
        "governance_status": governance.get("status"),
        "weekend_intensity": float(trial["parameters"]["weekend_intensity"]),
        "low_tail_intensity": float(trial["parameters"]["low_tail_intensity"]),
        "peak_subshape_intensity": float(trial["parameters"]["peak_subshape_intensity"]),
        "evening_recovery_intensity": float(trial["parameters"].get("evening_recovery_intensity", 0.0)),
        "night_intensity": float(trial["parameters"].get("night_intensity", 0.0)),
        "ramp_intensity": float(trial["parameters"].get("ramp_intensity", 0.0)),
        "max_abs_delta_eur_mwh": float(independent["max_abs_delta_eur_mwh"]),
        "max_abs_monthly_mean_delta_eur_mwh": monthly_drift,
        "max_abs_width_delta_eur_mwh": width_drift,
        "weighted_negative_hours_adjusted": int(independent["weighted_negative_hours_adjusted"]),
        "min_adjusted_price_eur_mwh": min_adjusted_price,
        "epex_spot_age_days": epex_age_days,
        "epex_fit_coverage_days": epex_coverage_days,
        "duck_change_mean_eur_mwh": duck_change,
        "solar_tail_mean_delta_eur_mwh": solar_tail_delta,
        "midday_mean_delta_eur_mwh": midday_delta,
        "weekend_mean_delta_eur_mwh": weekend_delta,
        "night_mean_delta_eur_mwh": night_delta,
        "ramp_abs_p99_increase_eur_mwh": ramp_increase,
        "independent_shape_score": float(shape_score),
        "lab_max_after_constraint_abs_error_eur_mwh": _optional_float(
            lab_audit.get("max_after_constraint_abs_error_eur_mwh")
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _age_days(*, valuation_timestamp: Any, max_timestamp: Any) -> float | None:
    if valuation_timestamp is None or max_timestamp is None:
        return None
    valuation = pd.Timestamp(valuation_timestamp)
    maximum = pd.Timestamp(max_timestamp)
    if valuation.tzinfo is None:
        valuation = valuation.tz_localize("UTC")
    if maximum.tzinfo is None:
        maximum = maximum.tz_localize("UTC")
    return (valuation.tz_convert("UTC") - maximum.tz_convert("UTC")).total_seconds() / 86400.0


def _coverage_days(*, min_timestamp: Any, max_timestamp: Any) -> float | None:
    if min_timestamp is None or max_timestamp is None:
        return None
    minimum = pd.Timestamp(min_timestamp)
    maximum = pd.Timestamp(max_timestamp)
    if minimum.tzinfo is None:
        minimum = minimum.tz_localize("UTC")
    if maximum.tzinfo is None:
        maximum = maximum.tz_localize("UTC")
    return (maximum.tz_convert("UTC") - minimum.tz_convert("UTC")).total_seconds() / 86400.0


def _threshold_max_ok(value: float | None, threshold: float | None) -> bool:
    if threshold is None:
        return True
    return value is not None and value <= threshold


def _threshold_min_ok(value: float | None, threshold: float | None) -> bool:
    if threshold is None:
        return True
    return value is not None and value >= threshold


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
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    summary = execute_sweep(
        plan_json=args.plan_json,
        output_summary=args.output_summary,
        max_trials=args.max_trials,
        resume=not args.no_resume,
    )
    print(json.dumps({"trial_count_executed": summary["trial_count_executed"], "eligible_count": summary["eligible_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
