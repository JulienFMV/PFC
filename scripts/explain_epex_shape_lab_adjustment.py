"""Explain an EPEX shape-lab adjustment without using OMPEX.

This diagnostic is lab-only.  It reconstructs raw component contributions from
the EPEX shape templates and the frozen lab configuration, compares them with
the final projected candidate delta, and summarizes the change by common shape
buckets.  It does not approve production.
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

from pfc_shaping.lt.model.shape_constraints import eex_peak_mask  # noqa: E402
from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch  # noqa: E402


PRICE = "price_weighted_mean_eur_mwh"
TEMPLATE_COMPONENTS = {
    "weekend": "weekend_delta_eur_mwh",
    "low_tail": "low_tail_delta_eur_mwh",
    "peak_subshape": "peak_subshape_delta_eur_mwh",
    "evening_recovery": "evening_recovery_delta_eur_mwh",
    "night": "night_delta_eur_mwh",
    "ramp": "ramp_delta_eur_mwh",
}
INTENSITY_KEYS = {
    "weekend": "weekend_intensity",
    "low_tail": "low_tail_intensity",
    "peak_subshape": "peak_subshape_intensity",
    "evening_recovery": "evening_recovery_intensity",
    "night": "night_intensity",
    "ramp": "ramp_intensity",
}


def explain_adjustment(
    *,
    baseline_csv: Path,
    adjusted_csv: Path,
    lab_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _load_json(lab_manifest)
    config = _lab_config(manifest)
    templates_path = _templates_path(manifest, lab_manifest=lab_manifest)

    baseline = _load_candidate(baseline_csv, label="baseline")
    adjusted = _load_candidate(adjusted_csv, label="adjusted")
    joined = _join_candidates(baseline, adjusted)
    templates = _load_templates(templates_path)
    explained = _attach_component_contributions(joined, templates, config)

    by_bucket = _bucket_summary(explained)
    by_component = _component_summary(explained)
    by_month = _group_summary(explained, ["year_month"])
    by_hour = _group_summary(explained, ["hour"])
    conservation = _conservation_summary(explained)
    strict_checks = _strict_checks(explained, conservation)

    output_dir.mkdir(parents=True, exist_ok=True)
    explained_csv = output_dir / "timestamp_component_contributions.csv"
    bucket_csv = output_dir / "bucket_delta_summary.csv"
    component_csv = output_dir / "component_delta_summary.csv"
    month_csv = output_dir / "month_delta_summary.csv"
    hour_csv = output_dir / "hour_delta_summary.csv"
    explained.to_csv(explained_csv, index=False)
    by_bucket.to_csv(bucket_csv, index=False)
    by_component.to_csv(component_csv, index=False)
    by_month.to_csv(month_csv, index=False)
    by_hour.to_csv(hour_csv, index=False)

    summary = {
        "schema_version": "epex_shape_lab_adjustment_explainability.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "benchmark_policy": "shape_explainability_no_ompex_diagnostic_only",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "baseline_csv": str(baseline_csv),
        "adjusted_csv": str(adjusted_csv),
        "lab_manifest": str(lab_manifest),
        "templates_csv": str(templates_path),
        "source_hashes": {
            "baseline_csv": _sha256(baseline_csv),
            "adjusted_csv": _sha256(adjusted_csv),
            "lab_manifest": _sha256(lab_manifest),
            "templates_csv": _sha256(templates_path),
        },
        "candidate_hours": int(len(explained)),
        "config": {key: float(config.get(key, 0.0)) for key in sorted(INTENSITY_KEYS.values())},
        "strict_checks": strict_checks,
        "conservation": conservation,
        "component_totals": _component_totals(by_component),
        "bucket_delta_summary": _bucket_totals(by_bucket),
        "outputs": {
            "timestamp_component_contributions_csv": str(explained_csv),
            "bucket_delta_summary_csv": str(bucket_csv),
            "component_delta_summary_csv": str(component_csv),
            "month_delta_summary_csv": str(month_csv),
            "hour_delta_summary_csv": str(hour_csv),
        },
        "output_hashes": {
            "timestamp_component_contributions_csv": _sha256(explained_csv),
            "bucket_delta_summary_csv": _sha256(bucket_csv),
            "component_delta_summary_csv": _sha256(component_csv),
            "month_delta_summary_csv": _sha256(month_csv),
            "hour_delta_summary_csv": _sha256(hour_csv),
        },
        "status": "DIAGNOSTIC_PASS" if all(strict_checks.values()) else "DIAGNOSTIC_FAIL",
        "note": (
            "Explains the final EPEX shape-lab delta using no-OMPEX template "
            "components. Projection/capping/floor effects appear in "
            "projection_residual_eur_mwh."
        ),
    }
    summary_path = output_dir / "shape_explainability_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_candidate(path: Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted({PRICE, "timestamp_ch"} - set(frame.columns))
    if missing:
        raise ValueError(f"{label} candidate missing required columns: {missing}")
    out = frame.copy()
    out[PRICE] = pd.to_numeric(out[PRICE], errors="raise")
    out.index = pd.DatetimeIndex(_parse_timestamp_ch(out["timestamp_ch"], out.get("utc_offset_ch")))
    if out.index.has_duplicates:
        raise ValueError(f"{label} candidate has duplicate timestamps")
    out = out.sort_index()
    return out[[PRICE]]


def _join_candidates(baseline: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    joined = baseline.join(adjusted, how="inner", lsuffix="_baseline", rsuffix="_adjusted")
    if len(joined) != len(baseline) or len(joined) != len(adjusted):
        raise ValueError(
            "baseline and adjusted candidates must have identical timestamp sets: "
            f"baseline={len(baseline)}, adjusted={len(adjusted)}, overlap={len(joined)}"
        )
    out = joined.copy()
    idx = out.index
    out["timestamp_ch"] = idx.astype(str)
    out["timestamp_utc"] = idx.tz_convert("UTC").astype(str)
    out["year_month"] = pd.PeriodIndex(idx.strftime("%Y-%m"), freq="M").astype(str)
    out["month"] = idx.month.astype(int)
    out["hour"] = idx.hour.astype(int)
    out["weekday"] = idx.weekday.astype(int)
    out["is_weekend"] = idx.weekday >= 5
    out["is_peak_like"] = pd.Series(eex_peak_mask(idx.tz_convert("UTC"), country="CH"), index=out.index)
    out["is_solar_tail"] = out["month"].between(3, 10) & out["hour"].between(10, 16)
    out["is_midday"] = out["hour"].between(11, 15)
    out["is_evening_ramp"] = out["hour"].between(17, 21)
    out["is_night"] = out["hour"].between(0, 5)
    out["actual_delta_eur_mwh"] = out[f"{PRICE}_adjusted"] - out[f"{PRICE}_baseline"]
    return out.reset_index(drop=True)


def _load_templates(path: Path) -> pd.DataFrame:
    templates = pd.read_csv(path)
    required = {"month", "hour", "is_weekend", *TEMPLATE_COMPONENTS.values()}
    missing = sorted(required - set(templates.columns))
    if missing:
        raise ValueError(f"templates CSV missing required columns: {missing}")
    out = templates.copy()
    out["month"] = out["month"].astype(int)
    out["hour"] = out["hour"].astype(int)
    out["is_weekend"] = out["is_weekend"].astype(bool)
    return out


def _attach_component_contributions(
    joined: pd.DataFrame,
    templates: pd.DataFrame,
    config: dict[str, float],
) -> pd.DataFrame:
    merged = joined.merge(templates, on=["month", "hour", "is_weekend"], how="left", validate="many_to_one")
    if merged[list(TEMPLATE_COMPONENTS.values())].isna().any().any():
        raise ValueError("templates do not cover every candidate month/hour/weekend cell")
    component_cols = []
    for component, template_col in TEMPLATE_COMPONENTS.items():
        intensity = float(config.get(INTENSITY_KEYS[component], 0.0))
        contribution_col = f"{component}_raw_contribution_eur_mwh"
        merged[contribution_col] = intensity * merged[template_col].astype(float)
        component_cols.append(contribution_col)
    merged["raw_total_delta_eur_mwh"] = merged[component_cols].sum(axis=1)
    merged["projection_residual_eur_mwh"] = merged["actual_delta_eur_mwh"] - merged["raw_total_delta_eur_mwh"]
    ordered = [
        "timestamp_ch",
        "timestamp_utc",
        "year_month",
        "month",
        "hour",
        "weekday",
        "is_weekend",
        "is_peak_like",
        "is_solar_tail",
        "is_midday",
        "is_evening_ramp",
        "is_night",
        f"{PRICE}_baseline",
        f"{PRICE}_adjusted",
        "actual_delta_eur_mwh",
        *component_cols,
        "raw_total_delta_eur_mwh",
        "projection_residual_eur_mwh",
    ]
    return merged[ordered]


def _bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "all": pd.Series(True, index=frame.index),
        "weekday": ~frame["is_weekend"].astype(bool),
        "weekend": frame["is_weekend"].astype(bool),
        "peak_like_eex": frame["is_peak_like"].astype(bool),
        "offpeak_like": ~frame["is_peak_like"].astype(bool),
        "solar_tail_mar_oct_10_16": frame["is_solar_tail"].astype(bool),
        "midday_11_15": frame["is_midday"].astype(bool),
        "evening_ramp_17_21": frame["is_evening_ramp"].astype(bool),
        "night_00_05": frame["is_night"].astype(bool),
    }
    rows = []
    for bucket, mask in masks.items():
        rows.append(_summary_row(frame.loc[mask], {"bucket": bucket}))
    return pd.DataFrame(rows)


def _component_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component in TEMPLATE_COMPONENTS:
        col = f"{component}_raw_contribution_eur_mwh"
        values = frame[col].astype(float)
        rows.append(
            {
                "component": component,
                "mean_contribution_eur_mwh": _mean(values),
                "mean_abs_contribution_eur_mwh": _mean(values.abs()),
                "p95_abs_contribution_eur_mwh": _quantile(values.abs(), 0.95),
                "max_abs_contribution_eur_mwh": _max(values.abs()),
                "nonzero_hours": int((values.abs() > 1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def _group_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        rows.append(_summary_row(group, dict(zip(keys, values, strict=True))))
    return pd.DataFrame(rows)


def _summary_row(frame: pd.DataFrame, labels: dict[str, Any]) -> dict[str, Any]:
    actual = frame["actual_delta_eur_mwh"].astype(float)
    raw = frame["raw_total_delta_eur_mwh"].astype(float)
    projection = frame["projection_residual_eur_mwh"].astype(float)
    row = {
        **labels,
        "hours": int(len(frame)),
        "mean_actual_delta_eur_mwh": _mean(actual),
        "mean_abs_actual_delta_eur_mwh": _mean(actual.abs()),
        "p95_abs_actual_delta_eur_mwh": _quantile(actual.abs(), 0.95),
        "max_abs_actual_delta_eur_mwh": _max(actual.abs()),
        "mean_raw_total_delta_eur_mwh": _mean(raw),
        "mean_projection_residual_eur_mwh": _mean(projection),
        "mean_abs_projection_residual_eur_mwh": _mean(projection.abs()),
    }
    for component in TEMPLATE_COMPONENTS:
        col = f"{component}_raw_contribution_eur_mwh"
        row[f"mean_{component}_raw_contribution_eur_mwh"] = _mean(frame[col].astype(float))
    return row


def _conservation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    base = frame.groupby("year_month")["actual_delta_eur_mwh"].mean().abs()
    peak = frame.loc[frame["is_peak_like"].astype(bool)].groupby("year_month")["actual_delta_eur_mwh"].mean().abs()
    return {
        "max_monthly_base_mean_abs_delta_eur_mwh": _max(base),
        "max_monthly_peak_mean_abs_delta_eur_mwh": _max(peak) if len(peak) else 0.0,
        "monthly_base_bucket_count": int(len(base)),
        "monthly_peak_bucket_count": int(len(peak)),
    }


def _strict_checks(frame: pd.DataFrame, conservation: dict[str, Any]) -> dict[str, bool]:
    numeric = frame[
        [
            "actual_delta_eur_mwh",
            "raw_total_delta_eur_mwh",
            "projection_residual_eur_mwh",
            *[f"{component}_raw_contribution_eur_mwh" for component in TEMPLATE_COMPONENTS],
        ]
    ].to_numpy(dtype=float)
    return {
        "finite_contributions_ok": bool(np.isfinite(numeric).all()),
        "candidate_hours_positive": bool(len(frame) > 0),
        "monthly_base_delta_conserved": bool(conservation["max_monthly_base_mean_abs_delta_eur_mwh"] <= 1e-5),
        "monthly_peak_delta_conserved": bool(conservation["max_monthly_peak_mean_abs_delta_eur_mwh"] <= 1e-5),
        "ompex_not_used": True,
    }


def _lab_config(manifest: dict[str, Any]) -> dict[str, float]:
    raw = manifest.get("config") or manifest.get("epex_lab_config")
    if not isinstance(raw, dict):
        raise ValueError("lab manifest missing config/epex_lab_config")
    config: dict[str, float] = {}
    for key in INTENSITY_KEYS.values():
        config[key] = float(raw.get(key, 0.0))
    return config


def _templates_path(manifest: dict[str, Any], *, lab_manifest: Path) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get("templates_csv"):
        raise ValueError("lab manifest missing outputs.templates_csv")
    path = Path(str(outputs["templates_csv"]))
    if not path.is_absolute():
        candidate = lab_manifest.parent / path
        path = candidate if candidate.exists() else path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_totals(component_summary: pd.DataFrame) -> dict[str, Any]:
    return {
        str(row["component"]): {
            "mean_abs_contribution_eur_mwh": float(row["mean_abs_contribution_eur_mwh"]),
            "p95_abs_contribution_eur_mwh": float(row["p95_abs_contribution_eur_mwh"]),
            "nonzero_hours": int(row["nonzero_hours"]),
        }
        for row in component_summary.to_dict(orient="records")
    }


def _bucket_totals(bucket_summary: pd.DataFrame) -> dict[str, Any]:
    return {
        str(row["bucket"]): {
            "hours": int(row["hours"]),
            "mean_abs_actual_delta_eur_mwh": float(row["mean_abs_actual_delta_eur_mwh"]),
            "mean_abs_projection_residual_eur_mwh": float(row["mean_abs_projection_residual_eur_mwh"]),
        }
        for row in bucket_summary.to_dict(orient="records")
    }


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _max(values: pd.Series) -> float:
    return float(values.max()) if len(values) else float("nan")


def _quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if len(values) else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--adjusted-csv", required=True, type=Path)
    parser.add_argument("--lab-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = explain_adjustment(
        baseline_csv=args.baseline_csv,
        adjusted_csv=args.adjusted_csv,
        lab_manifest=args.lab_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0 if summary["status"] == "DIAGNOSTIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
