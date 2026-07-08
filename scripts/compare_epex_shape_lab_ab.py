"""Compare baseline vs adjusted EPEX shape-lab candidates without OMPEX.

This diagnostic is intentionally independent from OMPEX/HFC.  It checks that
an adjusted lab candidate keeps timestamps aligned, preserves monthly levels,
does not rebuild fan width unexpectedly, and records the calendar shape effects
introduced by the A/B delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch  # noqa: E402


PRICE = "price_weighted_mean_eur_mwh"
PRICE_COLUMNS = [
    "price_slow_eur_mwh",
    "price_central_eur_mwh",
    "price_fast_eur_mwh",
    PRICE,
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]


def compare_ab(
    *,
    baseline_csv: Path,
    adjusted_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    baseline = _load_hourly(baseline_csv)
    adjusted = _load_hourly(adjusted_csv)
    joined = baseline.join(adjusted, how="inner", lsuffix="_baseline", rsuffix="_adjusted")
    if len(joined) != len(baseline) or len(joined) != len(adjusted):
        raise ValueError(
            "baseline and adjusted candidates must have identical timestamp sets: "
            f"baseline={len(baseline)}, adjusted={len(adjusted)}, overlap={len(joined)}"
        )
    joined = _add_calendar(joined)
    joined["delta_eur_mwh"] = joined[f"{PRICE}_adjusted"] - joined[f"{PRICE}_baseline"]
    joined["width_delta_eur_mwh"] = (
        joined["structural_width_eur_mwh_adjusted"] - joined["structural_width_eur_mwh_baseline"]
    )

    monthly = _monthly_summary(joined)
    annual = _annual_summary(joined)
    calendar = _calendar_summary(joined)
    summary = _summary(joined, monthly)

    output_dir.mkdir(parents=True, exist_ok=True)
    joined.reset_index(names="timestamp_ch").to_csv(output_dir / "aligned_baseline_adjusted.csv", index=False)
    monthly.to_csv(output_dir / "monthly_summary.csv", index=False)
    annual.to_csv(output_dir / "annual_summary.csv", index=False)
    calendar.to_csv(output_dir / "calendar_delta_summary.csv", index=False)
    summary.update(
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "baseline_csv": str(baseline_csv),
            "adjusted_csv": str(adjusted_csv),
            "outputs": {
                "aligned_csv": str(output_dir / "aligned_baseline_adjusted.csv"),
                "monthly_summary_csv": str(output_dir / "monthly_summary.csv"),
                "annual_summary_csv": str(output_dir / "annual_summary.csv"),
                "calendar_delta_summary_csv": str(output_dir / "calendar_delta_summary.csv"),
            },
        }
    )
    (output_dir / "ab_comparison_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_hourly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(["timestamp_ch", *PRICE_COLUMNS]) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    ts = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    out = frame.copy()
    for column in PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out.index = pd.DatetimeIndex(ts)
    if out.index.has_duplicates:
        raise ValueError(f"{path} has duplicate timestamps")
    return out[PRICE_COLUMNS].sort_index()


def _add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = out.index
    out["year"] = idx.year.astype(int)
    out["month"] = idx.month.astype(int)
    out["year_month"] = pd.PeriodIndex(idx.strftime("%Y-%m"), freq="M").astype(str)
    out["hour"] = idx.hour.astype(int)
    out["weekday"] = idx.weekday.astype(int)
    out["is_weekend"] = out["weekday"] >= 5
    out["is_solar_tail"] = out["month"].between(3, 10) & out["hour"].between(10, 16)
    out["is_evening_ramp"] = out["hour"].between(17, 21)
    out["is_midday"] = out["hour"].between(11, 15)
    return out


def _monthly_summary(joined: pd.DataFrame) -> pd.DataFrame:
    grouped = joined.groupby("year_month", sort=True)
    rows = []
    for month, group in grouped:
        rows.append(
            {
                "year_month": month,
                "rows": int(len(group)),
                "baseline_mean_eur_mwh": float(group[f"{PRICE}_baseline"].mean()),
                "adjusted_mean_eur_mwh": float(group[f"{PRICE}_adjusted"].mean()),
                "mean_delta_eur_mwh": float(group["delta_eur_mwh"].mean()),
                "max_abs_delta_eur_mwh": float(group["delta_eur_mwh"].abs().max()),
                "width_delta_mean_eur_mwh": float(group["width_delta_eur_mwh"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _annual_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in joined.groupby("year", sort=True):
        baseline_midday = float(group.loc[group["is_midday"], f"{PRICE}_baseline"].mean())
        adjusted_midday = float(group.loc[group["is_midday"], f"{PRICE}_adjusted"].mean())
        baseline_evening = float(group.loc[group["is_evening_ramp"], f"{PRICE}_baseline"].mean())
        adjusted_evening = float(group.loc[group["is_evening_ramp"], f"{PRICE}_adjusted"].mean())
        baseline_weekend = float(group.loc[group["is_weekend"], f"{PRICE}_baseline"].mean())
        adjusted_weekend = float(group.loc[group["is_weekend"], f"{PRICE}_adjusted"].mean())
        baseline_weekday = float(group.loc[~group["is_weekend"], f"{PRICE}_baseline"].mean())
        adjusted_weekday = float(group.loc[~group["is_weekend"], f"{PRICE}_adjusted"].mean())
        rows.append(
            {
                "year": int(year),
                "baseline_evening_minus_midday_eur_mwh": baseline_evening - baseline_midday,
                "adjusted_evening_minus_midday_eur_mwh": adjusted_evening - adjusted_midday,
                "evening_minus_midday_change_eur_mwh": (adjusted_evening - adjusted_midday)
                - (baseline_evening - baseline_midday),
                "baseline_weekend_minus_weekday_eur_mwh": baseline_weekend - baseline_weekday,
                "adjusted_weekend_minus_weekday_eur_mwh": adjusted_weekend - adjusted_weekday,
                "weekend_minus_weekday_change_eur_mwh": (adjusted_weekend - adjusted_weekday)
                - (baseline_weekend - baseline_weekday),
            }
        )
    return pd.DataFrame(rows)


def _calendar_summary(joined: pd.DataFrame) -> pd.DataFrame:
    buckets = {
        "all": pd.Series(True, index=joined.index),
        "weekend": joined["is_weekend"],
        "weekday": ~joined["is_weekend"],
        "solar_tail_mar_oct_10_16": joined["is_solar_tail"],
        "evening_ramp_17_21": joined["is_evening_ramp"],
        "midday_11_15": joined["is_midday"],
    }
    rows = []
    for name, mask in buckets.items():
        group = joined.loc[mask.to_numpy(dtype=bool)]
        if group.empty:
            continue
        rows.append(
            {
                "bucket": name,
                "rows": int(len(group)),
                "mean_delta_eur_mwh": float(group["delta_eur_mwh"].mean()),
                "p05_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.05)),
                "p50_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.50)),
                "p95_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.95)),
                "max_abs_delta_eur_mwh": float(group["delta_eur_mwh"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _summary(joined: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, Any]:
    numeric = joined[[f"{col}_adjusted" for col in PRICE_COLUMNS if col != "structural_width_eur_mwh"]]
    quantile_order = bool(
        (
            (joined["structural_p10_eur_mwh_adjusted"] <= joined["structural_p50_eur_mwh_adjusted"])
            & (joined["structural_p50_eur_mwh_adjusted"] <= joined["structural_p90_eur_mwh_adjusted"])
        ).all()
    )
    ramp_baseline = joined[f"{PRICE}_baseline"].diff().abs().dropna()
    ramp_adjusted = joined[f"{PRICE}_adjusted"].diff().abs().dropna()
    return {
        "n_hours": int(len(joined)),
        "finite_adjusted_ok": bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
        "quantile_order_adjusted_ok": quantile_order,
        "weighted_negative_hours_adjusted": int((joined[f"{PRICE}_adjusted"] < 0.0).sum()),
        "min_adjusted_price_eur_mwh": float(numeric.min().min()),
        "max_abs_delta_eur_mwh": float(joined["delta_eur_mwh"].abs().max()),
        "mean_delta_eur_mwh": float(joined["delta_eur_mwh"].mean()),
        "max_abs_monthly_mean_delta_eur_mwh": float(monthly["mean_delta_eur_mwh"].abs().max()),
        "max_abs_width_delta_eur_mwh": float(joined["width_delta_eur_mwh"].abs().max()),
        "ramp_abs_p99_baseline_eur_mwh": float(ramp_baseline.quantile(0.99)),
        "ramp_abs_p99_adjusted_eur_mwh": float(ramp_adjusted.quantile(0.99)),
        "ramp_abs_max_baseline_eur_mwh": float(ramp_baseline.max()),
        "ramp_abs_max_adjusted_eur_mwh": float(ramp_adjusted.max()),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--adjusted-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = compare_ab(
        baseline_csv=args.baseline_csv,
        adjusted_csv=args.adjusted_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
