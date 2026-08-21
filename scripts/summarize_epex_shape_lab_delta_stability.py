"""Summarize cross-date stability of frozen EPEX shape-lab delta fields.

This read-only diagnostic compares independent no-OMPEX A/B aligned outputs.
It checks whether the hourly adjustment field itself is stable across dates,
not only whether aggregate gates pass.  It does not read OMPEX and is not a
production promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import LOCAL_TZ, _parse_timestamp_ch  # noqa: E402


DELTA = "delta_eur_mwh"
CONFIG_KEYS = [
    "variant",
    "weekend_intensity",
    "low_tail_intensity",
    "peak_subshape_intensity",
    "max_abs_delta_eur_mwh",
    "negative_price_floor",
    "max_weighted_negative_hours",
    "lookback_years",
    "require_monthly_base_constraints",
]


def summarize_delta_stability(
    *,
    cases: list[tuple[str, Path, Path, Path]],
    output_dir: Path,
    min_timestamp_delta_correlation: float = 0.999,
    max_timestamp_delta_mae: float = 0.05,
    max_timestamp_delta_abs: float = 0.25,
    max_month_hour_delta_mae: float = 0.05,
    max_month_hour_delta_abs: float = 0.25,
    max_boundary_jump_delta_abs: float = 0.25,
) -> dict[str, Any]:
    if len(cases) < 2:
        raise ValueError("delta stability requires at least two cases")
    _case_pass.thresholds = {
        "min_timestamp_delta_correlation": float(min_timestamp_delta_correlation),
        "max_timestamp_delta_mae": float(max_timestamp_delta_mae),
        "max_timestamp_delta_abs": float(max_timestamp_delta_abs),
        "max_month_hour_delta_mae": float(max_month_hour_delta_mae),
        "max_month_hour_delta_abs": float(max_month_hour_delta_abs),
        "max_boundary_jump_delta_abs": float(max_boundary_jump_delta_abs),
    }
    loaded = [_load_case(label, aligned, summary, lab) for label, aligned, summary, lab in cases]
    reference = loaded[0]
    rows = []
    for case in loaded[1:]:
        rows.append(_compare_case(reference, case))

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_csv = output_dir / "delta_stability_comparisons.csv"
    frame.to_csv(comparisons_csv, index=False)
    config_hashes = sorted({case["config_hash"] for case in loaded})
    failed = frame[~frame["case_pass"].astype(bool)] if not frame.empty else frame
    decision = {
        "schema_version": "epex_shape_lab_delta_stability_summary.v1",
        "read_only": True,
        "promotion_gate": False,
        "benchmark_policy": "cross_date_independent_no_ompex_delta_stability",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "reference_label": reference["label"],
        "case_count": int(len(loaded)),
        "comparison_count": int(len(frame)),
        "passed_comparison_count": int(frame["case_pass"].sum()) if not frame.empty else 0,
        "failed_comparison_count": int(len(failed)),
        "config_hashes": config_hashes,
        "config_consistent": len(config_hashes) == 1,
        "status": "PASS" if len(frame) > 0 and failed.empty and len(config_hashes) == 1 else "FAIL",
        "thresholds": {
            "min_timestamp_delta_correlation": float(min_timestamp_delta_correlation),
            "max_timestamp_delta_mae": float(max_timestamp_delta_mae),
            "max_timestamp_delta_abs": float(max_timestamp_delta_abs),
            "max_month_hour_delta_mae": float(max_month_hour_delta_mae),
            "max_month_hour_delta_abs": float(max_month_hour_delta_abs),
            "max_boundary_jump_delta_abs": float(max_boundary_jump_delta_abs),
        },
        "outputs": {"comparisons_csv": str(comparisons_csv)},
        "note": (
            "This is cross-date delta-field stability evidence only. It does "
            "not approve production promotion and does not use OMPEX."
        ),
    }
    summary_path = output_dir / "delta_stability_summary.json"
    summary_path.write_text(json.dumps(_jsonable(decision), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def _load_case(label: str, aligned_csv: Path, summary_json: Path, lab_manifest: Path) -> dict[str, Any]:
    summary = _read_json(summary_json)
    lab = _read_json(lab_manifest)
    frame = pd.read_csv(aligned_csv)
    if "timestamp_ch" not in frame.columns or DELTA not in frame.columns:
        raise ValueError(f"{aligned_csv} must contain timestamp_ch and {DELTA}")
    ts = _parse_timestamps(frame["timestamp_ch"])
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(ts)
    frame[DELTA] = pd.to_numeric(frame[DELTA], errors="raise")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{aligned_csv} has duplicate timestamps")
    config = {key: (lab.get("config") or {}).get(key) for key in CONFIG_KEYS}
    return {
        "label": label,
        "aligned_csv": aligned_csv,
        "summary_json": summary_json,
        "lab_manifest": lab_manifest,
        "frame": frame,
        "summary": summary,
        "lab": lab,
        "config": config,
        "config_hash": _stable_hash(config),
    }


def _compare_case(reference: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    ref = reference["frame"]
    cur = case["frame"]
    joined = ref[[DELTA]].join(cur[[DELTA]], how="inner", lsuffix="_reference", rsuffix="_case")
    if joined.empty:
        raise ValueError(f"{reference['label']} and {case['label']} have no overlapping timestamps")
    diff = joined[f"{DELTA}_case"] - joined[f"{DELTA}_reference"]
    month_hour = _profile_compare(ref, cur, ["month", "hour"])
    boundary = _boundary_compare(ref, cur)
    row = {
        "reference_label": reference["label"],
        "case_label": case["label"],
        "reference_aligned_csv": str(reference["aligned_csv"]),
        "case_aligned_csv": str(case["aligned_csv"]),
        "reference_lab_manifest": str(reference["lab_manifest"]),
        "case_lab_manifest": str(case["lab_manifest"]),
        "reference_config_hash": reference["config_hash"],
        "case_config_hash": case["config_hash"],
        "config_match": reference["config_hash"] == case["config_hash"],
        "reference_benchmark_policy": reference["summary"].get("benchmark_policy"),
        "case_benchmark_policy": case["summary"].get("benchmark_policy"),
        "reference_ompex_used_in_model": bool(reference["summary"].get("ompex_used_in_model")),
        "case_ompex_used_in_model": bool(case["summary"].get("ompex_used_in_model")),
        "reference_ompex_used_in_selection": bool(reference["summary"].get("ompex_used_in_selection")),
        "case_ompex_used_in_selection": bool(case["summary"].get("ompex_used_in_selection")),
        "reference_lab_activation_status": reference["lab"].get("activation_status"),
        "case_lab_activation_status": case["lab"].get("activation_status"),
        "reference_lab_production_approved": bool(reference["lab"].get("production_approved")),
        "case_lab_production_approved": bool(case["lab"].get("production_approved")),
        "reference_hours": int(len(ref)),
        "case_hours": int(len(cur)),
        "common_hours": int(len(joined)),
        "missing_in_reference": int(len(cur.index.difference(ref.index))),
        "missing_in_case": int(len(ref.index.difference(cur.index))),
        "timestamp_delta_correlation": _corr(joined[f"{DELTA}_reference"], joined[f"{DELTA}_case"]),
        "timestamp_delta_diff_mean_eur_mwh": float(diff.mean()),
        "timestamp_delta_diff_mae_eur_mwh": float(diff.abs().mean()),
        "timestamp_delta_diff_rmse_eur_mwh": float(np.sqrt(np.mean(np.square(diff.to_numpy(dtype=float))))),
        "timestamp_delta_diff_max_abs_eur_mwh": float(diff.abs().max()),
        "month_hour_delta_correlation": month_hour["correlation"],
        "month_hour_delta_diff_mae_eur_mwh": month_hour["mae"],
        "month_hour_delta_diff_rmse_eur_mwh": month_hour["rmse"],
        "month_hour_delta_diff_max_abs_eur_mwh": month_hour["max_abs"],
        "boundary_jump_diff_mae_eur_mwh": boundary["mae"],
        "boundary_jump_diff_rmse_eur_mwh": boundary["rmse"],
        "boundary_jump_diff_max_abs_eur_mwh": boundary["max_abs"],
    }
    row["case_pass"] = _case_pass(row)
    return row


def _case_pass(row: dict[str, Any]) -> bool:
    thresholds = _case_pass.thresholds
    return bool(
        row["config_match"] is True
        and row["reference_benchmark_policy"] == "independent_no_ompex"
        and row["case_benchmark_policy"] == "independent_no_ompex"
        and row["reference_ompex_used_in_model"] is False
        and row["case_ompex_used_in_model"] is False
        and row["reference_ompex_used_in_selection"] is False
        and row["case_ompex_used_in_selection"] is False
        and row["reference_lab_activation_status"] == "lab_only"
        and row["case_lab_activation_status"] == "lab_only"
        and row["reference_lab_production_approved"] is False
        and row["case_lab_production_approved"] is False
        and row["missing_in_reference"] == 0
        and row["missing_in_case"] == 0
        and row["timestamp_delta_correlation"] >= thresholds["min_timestamp_delta_correlation"]
        and row["timestamp_delta_diff_mae_eur_mwh"] <= thresholds["max_timestamp_delta_mae"]
        and row["timestamp_delta_diff_max_abs_eur_mwh"] <= thresholds["max_timestamp_delta_abs"]
        and row["month_hour_delta_diff_mae_eur_mwh"] <= thresholds["max_month_hour_delta_mae"]
        and row["month_hour_delta_diff_max_abs_eur_mwh"] <= thresholds["max_month_hour_delta_abs"]
        and row["boundary_jump_diff_max_abs_eur_mwh"] <= thresholds["max_boundary_jump_delta_abs"]
    )


_case_pass.thresholds = {
    "min_timestamp_delta_correlation": 0.999,
    "max_timestamp_delta_mae": 0.05,
    "max_timestamp_delta_abs": 0.25,
    "max_month_hour_delta_mae": 0.05,
    "max_month_hour_delta_abs": 0.25,
    "max_boundary_jump_delta_abs": 0.25,
}


def _profile_compare(reference: pd.DataFrame, case: pd.DataFrame, keys: list[str]) -> dict[str, float]:
    ref = _with_calendar(reference).groupby(keys, sort=True)[DELTA].mean()
    cur = _with_calendar(case).groupby(keys, sort=True)[DELTA].mean()
    joined = pd.concat([ref, cur], axis=1, keys=["reference", "case"]).dropna()
    diff = joined["case"] - joined["reference"]
    return {
        "correlation": _corr(joined["reference"], joined["case"]),
        "mae": float(diff.abs().mean()) if not diff.empty else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(diff.to_numpy(dtype=float))))) if not diff.empty else 0.0,
        "max_abs": float(diff.abs().max()) if not diff.empty else 0.0,
    }


def _boundary_compare(reference: pd.DataFrame, case: pd.DataFrame) -> dict[str, float]:
    ref = _boundary_jumps(reference)
    cur = _boundary_jumps(case)
    joined = pd.concat([ref, cur], axis=1, keys=["reference", "case"]).dropna()
    diff = joined["case"] - joined["reference"]
    return {
        "mae": float(diff.abs().mean()) if not diff.empty else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(diff.to_numpy(dtype=float))))) if not diff.empty else 0.0,
        "max_abs": float(diff.abs().max()) if not diff.empty else 0.0,
    }


def _with_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    if {"month", "hour"}.issubset(frame.columns):
        return frame
    out = frame.copy()
    out["month"] = out.index.month.astype(int)
    out["hour"] = out.index.hour.astype(int)
    return out


def _boundary_jumps(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_index().copy()
    ordered["year_month"] = pd.PeriodIndex(ordered.index.strftime("%Y-%m"), freq="M").astype(str)
    ordered["previous_year_month"] = ordered["year_month"].shift(1)
    ordered["previous_delta"] = ordered[DELTA].shift(1)
    boundaries = ordered.loc[ordered["year_month"].ne(ordered["previous_year_month"])].copy()
    if len(boundaries) <= 1:
        return pd.Series(dtype=float)
    boundaries = boundaries.iloc[1:].copy()
    boundaries["jump"] = boundaries[DELTA] - boundaries["previous_delta"]
    return boundaries.groupby("year_month", sort=True)["jump"].mean()


def _corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2:
        return 1.0
    value = left.corr(right)
    return 0.0 if pd.isna(value) else float(value)


def _parse_case(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split("|")
    if len(parts) != 4:
        raise ValueError("case must use LABEL|ALIGNED_AB_CSV|AB_SUMMARY_JSON|LAB_MANIFEST_JSON")
    label, aligned, summary, lab = parts
    if not label:
        raise ValueError("case label must not be empty")
    return label, Path(aligned), Path(summary), Path(lab)


def _parse_timestamps(values: pd.Series) -> pd.Series:
    try:
        return _parse_timestamp_ch(values, None)
    except ValueError:
        return pd.to_datetime(values.astype(str), utc=True).dt.tz_convert(LOCAL_TZ)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case tuple as LABEL|ALIGNED_AB_CSV|AB_SUMMARY_JSON|LAB_MANIFEST_JSON. Repeat for each date.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-timestamp-delta-correlation", type=float, default=0.999)
    parser.add_argument("--max-timestamp-delta-mae", type=float, default=0.05)
    parser.add_argument("--max-timestamp-delta-abs", type=float, default=0.25)
    parser.add_argument("--max-month-hour-delta-mae", type=float, default=0.05)
    parser.add_argument("--max-month-hour-delta-abs", type=float, default=0.25)
    parser.add_argument("--max-boundary-jump-delta-abs", type=float, default=0.25)
    args = parser.parse_args(argv)
    _case_pass.thresholds = {
        "min_timestamp_delta_correlation": float(args.min_timestamp_delta_correlation),
        "max_timestamp_delta_mae": float(args.max_timestamp_delta_mae),
        "max_timestamp_delta_abs": float(args.max_timestamp_delta_abs),
        "max_month_hour_delta_mae": float(args.max_month_hour_delta_mae),
        "max_month_hour_delta_abs": float(args.max_month_hour_delta_abs),
        "max_boundary_jump_delta_abs": float(args.max_boundary_jump_delta_abs),
    }
    decision = summarize_delta_stability(
        cases=[_parse_case(item) for item in args.case],
        output_dir=args.output_dir,
        min_timestamp_delta_correlation=args.min_timestamp_delta_correlation,
        max_timestamp_delta_mae=args.max_timestamp_delta_mae,
        max_timestamp_delta_abs=args.max_timestamp_delta_abs,
        max_month_hour_delta_mae=args.max_month_hour_delta_mae,
        max_month_hour_delta_abs=args.max_month_hour_delta_abs,
        max_boundary_jump_delta_abs=args.max_boundary_jump_delta_abs,
    )
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))
    return 0 if decision["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
