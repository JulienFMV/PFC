"""Diagnose parity between a raw LT fan parquet and an audited hourly CH CSV.

This script is read-only with respect to model inputs.  It converts the fan
parquet using the same lightweight helper used by staging, aligns it with an
already exported hourly CSV, and writes explainable deltas.  Optional product
normalization audits can be run on both sides to show whether the fan-derived
hourly artifact is promotion-facing.
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

from scripts.audit_ch_product_normalization import run_audit  # noqa: E402
from scripts.export_local_test_ch_hourly_csv import (  # noqa: E402
    PRICE_COLUMNS,
    _eex_peak_mask,
    _parse_timestamp_ch,
    to_hourly_csv_frame,
)


PRICE = "price_weighted_mean_eur_mwh"


def diagnose_parity(
    *,
    fan_parquet: Path,
    reference_csv: Path,
    local_start_date: str,
    local_end_date: str,
    output_dir: Path,
    forwards_path: Path | None = None,
    required_forward_date: str | None = None,
    source_hierarchy_policy: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fan = pd.read_parquet(fan_parquet)
    fan_hourly = to_hourly_csv_frame(
        fan,
        local_start_date=local_start_date,
        local_end_date=local_end_date,
    )
    fan_hourly_path = output_dir / "fan_hourly_converted.csv"
    fan_hourly.to_csv(fan_hourly_path, index=False)

    fan_loaded = _load_hourly(fan_hourly_path)
    reference_loaded = _load_hourly(reference_csv)
    missing_in_fan = reference_loaded.index.difference(fan_loaded.index)
    missing_in_reference = fan_loaded.index.difference(reference_loaded.index)
    common_columns = [column for column in PRICE_COLUMNS if column in fan_loaded.columns and column in reference_loaded.columns]
    missing_columns = sorted(set(PRICE_COLUMNS) - set(common_columns))
    joined = reference_loaded[common_columns].join(
        fan_loaded[common_columns],
        how="inner",
        lsuffix="_reference",
        rsuffix="_fan",
    )
    if joined.empty:
        raise ValueError("fan-derived hourly CSV and reference CSV have no overlapping timestamps")
    joined = _add_calendar(joined)
    for column in common_columns:
        joined[f"{column}_delta_fan_minus_reference"] = joined[f"{column}_fan"] - joined[f"{column}_reference"]

    column_summary = _column_summary(joined, common_columns)
    monthly_summary = _monthly_summary(joined)
    load_type_summary = _load_type_summary(joined)
    boundary_summary = _boundary_summary(joined)

    aligned_path = output_dir / "aligned_fan_reference.csv"
    column_path = output_dir / "column_delta_summary.csv"
    monthly_path = output_dir / "monthly_delta_summary.csv"
    load_type_path = output_dir / "load_type_delta_summary.csv"
    boundary_path = output_dir / "boundary_delta_summary.csv"
    joined.reset_index(names="timestamp_ch").to_csv(aligned_path, index=False)
    column_summary.to_csv(column_path, index=False)
    monthly_summary.to_csv(monthly_path, index=False)
    load_type_summary.to_csv(load_type_path, index=False)
    boundary_summary.to_csv(boundary_path, index=False)

    product_audits: dict[str, Any] = {}
    if forwards_path is not None:
        product_audits = _run_product_audits(
            fan_hourly_path=fan_hourly_path,
            reference_csv=reference_csv,
            forwards_path=forwards_path,
            required_forward_date=required_forward_date,
            source_hierarchy_policy=source_hierarchy_policy,
            output_dir=output_dir,
        )

    weighted_delta = (
        joined[f"{PRICE}_delta_fan_minus_reference"]
        if f"{PRICE}_delta_fan_minus_reference" in joined.columns
        else pd.Series(dtype=float)
    )
    monthly_weighted = (
        monthly_summary["weighted_mean_delta_fan_minus_reference_eur_mwh"]
        if "weighted_mean_delta_fan_minus_reference_eur_mwh" in monthly_summary.columns
        else pd.Series(dtype=float)
    )
    summary: dict[str, Any] = {
        "schema_version": "fan_to_hourly_parity_diagnostic.v1",
        "read_only": True,
        "promotion_gate": False,
        "fan_parquet": str(fan_parquet),
        "reference_csv": str(reference_csv),
        "local_start_date": local_start_date,
        "local_end_date": local_end_date,
        "fan_rows_15min": int(len(fan)),
        "fan_hourly_rows": int(len(fan_loaded)),
        "reference_rows": int(len(reference_loaded)),
        "aligned_rows": int(len(joined)),
        "missing_timestamps_in_fan": int(len(missing_in_fan)),
        "missing_timestamps_in_reference": int(len(missing_in_reference)),
        "fan_coverage_ratio": float(len(joined) / len(fan_loaded)) if len(fan_loaded) else 0.0,
        "reference_coverage_ratio": float(len(joined) / len(reference_loaded)) if len(reference_loaded) else 0.0,
        "coverage_status": (
            "PASS"
            if len(missing_in_fan) == 0 and len(missing_in_reference) == 0
            else "PARTIAL_OVERLAP"
        ),
        "common_price_columns": common_columns,
        "missing_price_columns": missing_columns,
        "max_abs_weighted_delta_eur_mwh": _safe_max_abs(weighted_delta),
        "mean_abs_weighted_delta_eur_mwh": _safe_mean_abs(weighted_delta),
        "max_abs_monthly_weighted_delta_eur_mwh": _safe_max_abs(monthly_weighted),
        "outputs": {
            "fan_hourly_converted_csv": str(fan_hourly_path),
            "aligned_csv": str(aligned_path),
            "column_delta_summary_csv": str(column_path),
            "monthly_delta_summary_csv": str(monthly_path),
            "load_type_delta_summary_csv": str(load_type_path),
            "boundary_delta_summary_csv": str(boundary_path),
        },
        "product_audits": product_audits,
        "note": (
            "This diagnostic explains fan-to-hourly parity only. It does not "
            "approve fan-derived artifacts for promotion."
        ),
    }
    summary_path = output_dir / "fan_to_hourly_parity_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_hourly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp_ch" not in frame.columns:
        raise ValueError(f"{path} missing timestamp_ch")
    ts = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    out = frame.copy()
    out.index = pd.DatetimeIndex(ts)
    if out.index.has_duplicates:
        raise ValueError(f"{path} has duplicate timestamps")
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="raise")
    return out.sort_index()


def _add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = out.index
    out["year"] = idx.year.astype(int)
    out["month"] = idx.month.astype(int)
    out["year_month"] = pd.PeriodIndex(idx.strftime("%Y-%m"), freq="M").astype(str)
    out["hour"] = idx.hour.astype(int)
    out["weekday"] = idx.weekday.astype(int)
    out["is_eex_peak"] = _eex_peak_mask(pd.Series(idx, index=out.index), country="CH").to_numpy(dtype=bool)
    return out


def _column_summary(joined: pd.DataFrame, common_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in common_columns:
        delta = joined[f"{column}_delta_fan_minus_reference"]
        rows.append(
            {
                "column": column,
                "rows": int(delta.notna().sum()),
                "reference_mean_eur_mwh": float(joined[f"{column}_reference"].mean()),
                "fan_mean_eur_mwh": float(joined[f"{column}_fan"].mean()),
                "mean_delta_fan_minus_reference_eur_mwh": float(delta.mean()),
                "mean_abs_delta_eur_mwh": float(delta.abs().mean()),
                "p95_abs_delta_eur_mwh": float(delta.abs().quantile(0.95)),
                "max_abs_delta_eur_mwh": float(delta.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _monthly_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if f"{PRICE}_delta_fan_minus_reference" not in joined.columns:
        return pd.DataFrame()
    rows = []
    for year_month, group in joined.groupby("year_month", sort=True):
        delta = group[f"{PRICE}_delta_fan_minus_reference"]
        rows.append(
            {
                "year_month": str(year_month),
                "rows": int(len(group)),
                "reference_weighted_mean_eur_mwh": float(group[f"{PRICE}_reference"].mean()),
                "fan_weighted_mean_eur_mwh": float(group[f"{PRICE}_fan"].mean()),
                "weighted_mean_delta_fan_minus_reference_eur_mwh": float(delta.mean()),
                "weighted_mean_abs_delta_eur_mwh": float(delta.abs().mean()),
                "weighted_max_abs_delta_eur_mwh": float(delta.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _load_type_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if f"{PRICE}_delta_fan_minus_reference" not in joined.columns:
        return pd.DataFrame()
    buckets = {
        "BASE": pd.Series(True, index=joined.index),
        "PEAK": joined["is_eex_peak"],
        "OFFPEAK": ~joined["is_eex_peak"],
    }
    rows = []
    for load_type, mask in buckets.items():
        group = joined.loc[mask.to_numpy(dtype=bool)]
        if group.empty:
            continue
        delta = group[f"{PRICE}_delta_fan_minus_reference"]
        rows.append(
            {
                "load_type": load_type,
                "rows": int(len(group)),
                "reference_weighted_mean_eur_mwh": float(group[f"{PRICE}_reference"].mean()),
                "fan_weighted_mean_eur_mwh": float(group[f"{PRICE}_fan"].mean()),
                "weighted_mean_delta_fan_minus_reference_eur_mwh": float(delta.mean()),
                "weighted_mean_abs_delta_eur_mwh": float(delta.abs().mean()),
                "weighted_max_abs_delta_eur_mwh": float(delta.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _boundary_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if f"{PRICE}_delta_fan_minus_reference" not in joined.columns:
        return pd.DataFrame()
    ordered = joined.sort_index().copy()
    ordered["previous_year_month"] = ordered["year_month"].shift(1)
    ordered["previous_delta"] = ordered[f"{PRICE}_delta_fan_minus_reference"].shift(1)
    ordered["is_boundary"] = ordered["year_month"].ne(ordered["previous_year_month"])
    ordered.iloc[0, ordered.columns.get_loc("is_boundary")] = False
    boundaries = ordered.loc[ordered["is_boundary"]].copy()
    if boundaries.empty:
        return pd.DataFrame()
    boundaries["delta_jump_fan_minus_reference_eur_mwh"] = (
        boundaries[f"{PRICE}_delta_fan_minus_reference"] - boundaries["previous_delta"]
    )
    boundaries["is_year_boundary"] = boundaries.index.month == 1
    return boundaries.reset_index(names="timestamp_ch")[
        [
            "timestamp_ch",
            "year_month",
            "previous_year_month",
            "is_year_boundary",
            f"{PRICE}_delta_fan_minus_reference",
            "previous_delta",
            "delta_jump_fan_minus_reference_eur_mwh",
        ]
    ]


def _run_product_audits(
    *,
    fan_hourly_path: Path,
    reference_csv: Path,
    forwards_path: Path,
    required_forward_date: str | None,
    source_hierarchy_policy: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for label, csv_path in {"fan": fan_hourly_path, "reference": reference_csv}.items():
        gates, summary = run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date=required_forward_date,
            source_hierarchy_policy_path=source_hierarchy_policy,
            command_argv=None,
        )
        audit_dir = output_dir / f"product_audit_{label}"
        audit_dir.mkdir(parents=True, exist_ok=True)
        gates_path = audit_dir / "gates.csv"
        summary_path = audit_dir / "summary.json"
        gates.to_csv(gates_path, index=False)
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        audits[label] = {
            "gates_csv": str(gates_path),
            "summary_json": str(summary_path),
            "all_gates_pass": bool(summary.get("all_gates_pass")),
            "critical_count": int(summary.get("critical_count", 0)),
            "delivered_curve_drift_count": int(summary.get("delivered_curve_drift_count", 0)),
            "blocking_quote_conflict_count": int(summary.get("blocking_quote_conflict_count", 0)),
            "supported_hard_gate_max_abs_residual_eur_mwh": summary.get(
                "supported_hard_gate_max_abs_residual_eur_mwh"
            ),
        }
    return audits


def _safe_max_abs(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.abs().max())


def _safe_mean_abs(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.abs().mean())


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
    parser.add_argument("--fan-parquet", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--local-start-date", required=True)
    parser.add_argument("--local-end-date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forwards", type=Path)
    parser.add_argument("--required-forward-date")
    parser.add_argument("--source-hierarchy-policy", type=Path)
    args = parser.parse_args(argv)
    summary = diagnose_parity(
        fan_parquet=args.fan_parquet,
        reference_csv=args.reference_csv,
        local_start_date=args.local_start_date,
        local_end_date=args.local_end_date,
        output_dir=args.output_dir,
        forwards_path=args.forwards,
        required_forward_date=args.required_forward_date,
        source_hierarchy_policy=args.source_hierarchy_policy,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
