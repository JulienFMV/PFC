"""Calibrate LT quant-shaping smoothness lambdas on a candidate curve.

This is a diagnostic/governance harness, not a production generator.  It
evaluates a declared lambda grid for the equality-constrained LT quant
optimizer, writes objective/shape metrics, and emits a short L-curve report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets
from pfc_shaping.lt.model.quant_shape_optimizer import QuantShapeOptimizer
from pfc_shaping.lt.model.shape_constraints import build_base_peak_offpeak_constraint_system
from scripts.export_local_test_ch_hourly_csv import (
    _latest_eex_prices_by_load_type,
    _parse_timestamp_ch,
)


PRICE = "price_weighted_mean_eur_mwh"


@dataclass(frozen=True)
class Candidate:
    label: str
    lambda_smooth_h: float
    lambda_smooth_m: float
    lambda_seam: float
    family: str = "grid"


def _default_candidates() -> list[Candidate]:
    base = [
        Candidate("projection_only", 0.0, 0.0, 0.0, "baseline"),
        Candidate("h_10", 10.0, 0.0, 0.0),
        Candidate("h_30", 30.0, 0.0, 0.0),
        Candidate("h_100", 100.0, 0.0, 0.0),
        Candidate("h0_m1000_s1000", 0.0, 1000.0, 1000.0),
        Candidate("h0_m10000_s10000", 0.0, 10000.0, 10000.0),
        Candidate("h30_m1000_s1000", 30.0, 1000.0, 1000.0),
        Candidate("h100_m1000_s1000", 100.0, 1000.0, 1000.0),
        Candidate("h30_m10000_s10000", 30.0, 10000.0, 10000.0),
    ]
    return base


def _candidate_perturbations(candidate: Candidate) -> list[Candidate]:
    perturbed: list[Candidate] = []
    for field in ("lambda_smooth_h", "lambda_smooth_m", "lambda_seam"):
        value = getattr(candidate, field)
        if value <= 0.0:
            continue
        for factor in (0.75, 1.25):
            values = {
                "lambda_smooth_h": candidate.lambda_smooth_h,
                "lambda_smooth_m": candidate.lambda_smooth_m,
                "lambda_seam": candidate.lambda_seam,
            }
            values[field] = value * factor
            perturbed.append(
                Candidate(
                    f"{candidate.label}_{field}_x{factor:.2f}",
                    values["lambda_smooth_h"],
                    values["lambda_smooth_m"],
                    values["lambda_seam"],
                    "perturbation",
                )
            )
    return perturbed


def _read_prior(csv_path: Path, *, price_column: str, start: str | None, end: str | None) -> pd.Series:
    df = pd.read_csv(csv_path)
    if price_column not in df.columns:
        raise ValueError(f"{csv_path} is missing price column {price_column!r}")
    if "timestamp_ch" not in df.columns:
        raise ValueError(f"{csv_path} is missing timestamp_ch")
    ts = _parse_timestamp_ch(df["timestamp_ch"], df.get("utc_offset_ch"))
    values = pd.to_numeric(df[price_column], errors="raise").astype(float)
    prior = pd.Series(values.to_numpy(dtype=float), index=ts, name=price_column)
    if start is not None:
        prior = prior.loc[prior.index >= pd.Timestamp(start, tz="Europe/Zurich")]
    if end is not None:
        prior = prior.loc[prior.index <= pd.Timestamp(end, tz="Europe/Zurich")]
    if prior.empty:
        raise ValueError("selected calibration window is empty")
    if not prior.index.is_monotonic_increasing:
        prior = prior.sort_index()
    return prior


def _product_year(product: str) -> int | None:
    match = re.match(r"^(\d{4})", str(product))
    return int(match.group(1)) if match else None


def _filter_prices_to_index(prices: dict[str, float], index: pd.DatetimeIndex) -> dict[str, float]:
    years = set(int(year) for year in index.tz_convert("Europe/Zurich").year)
    return {
        str(product): float(price)
        for product, price in prices.items()
        if (year := _product_year(str(product))) in years
    }


def _month_means(curve: pd.Series) -> pd.Series:
    local = curve.copy()
    local.index = curve.index.tz_convert("Europe/Zurich").tz_localize(None)
    return local.groupby(local.index.to_period("M")).mean()


def _curve_metrics(
    curve: pd.Series,
    prior: pd.Series,
    *,
    base_forward_prices: dict[str, float],
) -> dict[str, float | str]:
    diff = curve.to_numpy(dtype=float) - prior.to_numpy(dtype=float)
    d2 = np.diff(curve.to_numpy(dtype=float), n=2)
    boundary_idx = [
        i for i in range(1, len(curve)) if curve.index[i].month != curve.index[i - 1].month
    ]
    boundary_jumps = np.array(
        [abs(float(curve.iloc[i] - curve.iloc[i - 1])) for i in boundary_idx],
        dtype=float,
    )
    periods = _month_means(curve)
    month_diffs = periods.diff().dropna()
    month_d2 = periods.diff().diff().dropna()
    local_series = pd.Series(curve.index.tz_convert("Europe/Zurich"), index=range(len(curve)))
    buckets, _ = calibration_buckets(local_series, base_forward_prices)
    bucket_values = buckets.to_numpy(dtype=object)
    residual_boundary_jumps = []
    for i in boundary_idx:
        prev_bucket = str(bucket_values[i - 1])
        next_bucket = str(bucket_values[i])
        if prev_bucket.endswith("-RESIDUAL") or next_bucket.endswith("-RESIDUAL"):
            residual_boundary_jumps.append(abs(float(curve.iloc[i] - curve.iloc[i - 1])))
    residual_boundary = np.asarray(residual_boundary_jumps, dtype=float)
    return {
        "prior_rmse_eur_mwh": float(np.sqrt(np.mean(diff * diff))),
        "prior_mae_eur_mwh": float(np.mean(np.abs(diff))),
        "hourly_curvature_rms": float(np.sqrt(np.mean(d2 * d2))) if len(d2) else 0.0,
        "hourly_curvature_abs_p95": float(np.percentile(np.abs(d2), 95.0)) if len(d2) else 0.0,
        "boundary_jump_abs_p95_eur_mwh": float(np.percentile(boundary_jumps, 95.0)) if len(boundary_jumps) else 0.0,
        "boundary_jump_abs_max_eur_mwh": float(np.max(boundary_jumps)) if len(boundary_jumps) else 0.0,
        "residual_boundary_jump_abs_max_eur_mwh": float(np.max(residual_boundary)) if len(residual_boundary) else 0.0,
        "month_adjacent_jump_abs_p95_eur_mwh": float(np.percentile(np.abs(month_diffs), 95.0)) if len(month_diffs) else 0.0,
        "month_adjacent_jump_abs_max_eur_mwh": float(np.max(np.abs(month_diffs))) if len(month_diffs) else 0.0,
        "month_curvature_abs_p95_eur_mwh": float(np.percentile(np.abs(month_d2), 95.0)) if len(month_d2) else 0.0,
        "month_curvature_abs_max_eur_mwh": float(np.max(np.abs(month_d2))) if len(month_d2) else 0.0,
    }


def _spot_month_benchmark(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    spot = pd.read_parquet(path)
    if isinstance(spot, pd.DataFrame):
        if "price_eur_mwh" not in spot.columns:
            return {}
        series = spot["price_eur_mwh"].astype(float)
    else:
        series = spot.astype(float)
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None:
        return {}
    local = series.copy()
    local.index = local.index.tz_convert("Europe/Zurich")
    monthly = local.groupby([local.index.year, local.index.month]).mean()
    diffs = monthly.groupby(level=0).diff().dropna().abs()
    if diffs.empty:
        return {}
    return {
        "spot_month_adjacent_jump_abs_p50_eur_mwh": float(diffs.quantile(0.50)),
        "spot_month_adjacent_jump_abs_p95_eur_mwh": float(diffs.quantile(0.95)),
        "spot_month_adjacent_jump_abs_max_eur_mwh": float(diffs.max()),
    }


def _year_folds(prior: pd.Series) -> list[tuple[str, pd.Series]]:
    local_year = prior.index.tz_convert("Europe/Zurich").year
    folds = []
    for year in sorted(set(int(v) for v in local_year)):
        fold = prior.loc[local_year == year]
        if len(fold) >= 24 * 28:
            folds.append((str(year), fold))
    if not folds:
        folds.append(("selected_window", prior))
    return folds


def _evaluate_candidate(
    candidate: Candidate,
    *,
    folds: list[tuple[str, pd.Series]],
    base_prices_all: dict[str, float],
    peak_prices_all: dict[str, float],
    forward_snapshot_date: str,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    fold_rows: list[dict[str, float | str]] = []
    for fold_name, fold_prior in folds:
        base_prices = _filter_prices_to_index(base_prices_all, fold_prior.index)
        peak_prices = _filter_prices_to_index(peak_prices_all, fold_prior.index)
        constraints = build_base_peak_offpeak_constraint_system(
            fold_prior.index.tz_convert("UTC"),
            base_prices,
            peak_prices,
            country="CH",
        )
        result = QuantShapeOptimizer(
            lambda_smooth_h=candidate.lambda_smooth_h,
            lambda_smooth_m=candidate.lambda_smooth_m,
            lambda_seam=candidate.lambda_seam,
            epsilon_ridge=1e-9,
            feasibility_tol=1e-6,
            stationarity_tol=1e-6,
        ).solve(fold_prior, constraints)
        residuals = result.constraint_residuals
        row: dict[str, float | str] = {
            "label": candidate.label,
            "family": candidate.family,
            "fold": fold_name,
            "forward_snapshot_date": forward_snapshot_date,
            "lambda_smooth_h": candidate.lambda_smooth_h,
            "lambda_smooth_m": candidate.lambda_smooth_m,
            "lambda_seam": candidate.lambda_seam,
            "max_constraint_abs_error_eur_mwh": float(residuals["abs_error"].max()) if not residuals.empty else 0.0,
            "kkt_primal_inf": float(result.kkt.primal_inf),
            "kkt_stationarity_inf": float(result.kkt.stationarity_inf),
            "objective_value": float(result.objective_value),
        }
        row.update(_curve_metrics(result.curve, fold_prior, base_forward_prices=base_prices))
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    aggregate: dict[str, float | str] = {
        "label": candidate.label,
        "family": candidate.family,
        "fold": "all",
        "forward_snapshot_date": forward_snapshot_date,
        "lambda_smooth_h": candidate.lambda_smooth_h,
        "lambda_smooth_m": candidate.lambda_smooth_m,
        "lambda_seam": candidate.lambda_seam,
        "fold_count": float(len(fold_df)),
    }
    for column in fold_df.columns:
        if column in aggregate or column in {"label", "family", "fold", "forward_snapshot_date"}:
            continue
        if pd.api.types.is_numeric_dtype(fold_df[column]):
            values = fold_df[column].astype(float)
            if column == "max_constraint_abs_error_eur_mwh":
                aggregate[column] = float(values.max())
            else:
                aggregate[column] = float(values.mean())
            aggregate[f"{column}_max_fold"] = float(values.max())
    return aggregate, fold_df


def _normalised(values: pd.Series) -> pd.Series:
    v = values.astype(float)
    lo = float(v.min())
    hi = float(v.max())
    if np.isclose(lo, hi):
        return pd.Series(0.0, index=v.index)
    return (v - lo) / (hi - lo)


def _select_lcurve_candidate(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    grid = results.loc[
        results["family"].isin(["baseline", "grid"]) & (results["fold"] == "all")
    ].copy()
    feasible = grid.loc[grid["max_constraint_abs_error_eur_mwh"] <= 1e-7].copy()
    if feasible.empty:
        feasible = grid.copy()
    roughness = (
        np.log1p(feasible["hourly_curvature_rms"].astype(float))
        + np.log1p(feasible["boundary_jump_abs_p95_eur_mwh"].astype(float))
        + np.log1p(feasible["month_curvature_abs_p95_eur_mwh"].astype(float))
    )
    feasible["lcurve_prior_score"] = _normalised(np.log1p(feasible["prior_rmse_eur_mwh"].astype(float)))
    feasible["lcurve_roughness_score"] = _normalised(roughness)
    feasible["lcurve_selection_score"] = (
        0.45 * feasible["lcurve_prior_score"] + 0.55 * feasible["lcurve_roughness_score"]
    )
    selected = str(feasible.sort_values(["lcurve_selection_score", "prior_rmse_eur_mwh"]).iloc[0]["label"])
    scored = results.merge(
        feasible[["label", "lcurve_prior_score", "lcurve_roughness_score", "lcurve_selection_score"]],
        on="label",
        how="left",
    )
    return selected, scored


def _write_report(
    path: Path,
    *,
    csv_path: Path,
    output_csv: Path,
    selected: pd.Series,
    spot_benchmark: dict[str, float],
    perturbations: pd.DataFrame,
) -> None:
    lines = [
        "# LT Quant Smoothness Lambda Calibration",
        "",
        f"* source_csv: `{csv_path}`",
        f"* metrics_csv: `{output_csv}`",
        "* status: diagnostic calibration evidence, not production sign-off",
        "",
        "## Selected L-Curve Candidate",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "label",
        "lambda_smooth_h",
        "lambda_smooth_m",
        "lambda_seam",
        "prior_rmse_eur_mwh",
        "boundary_jump_abs_p95_eur_mwh",
        "residual_boundary_jump_abs_max_eur_mwh",
        "month_adjacent_jump_abs_p95_eur_mwh",
        "month_curvature_abs_p95_eur_mwh",
        "max_constraint_abs_error_eur_mwh",
        "lcurve_selection_score",
    ]:
        value = selected.get(key, "")
        lines.append(f"| {key} | {value} |")
    if spot_benchmark:
        lines.extend(["", "## Historical Spot Monthly-Shape Benchmark", "", "| metric | value |", "|---|---:|"])
        for key, value in spot_benchmark.items():
            lines.append(f"| {key} | {value:.6f} |")
    if not perturbations.empty:
        selected_month = float(selected["month_adjacent_jump_abs_p95_eur_mwh"])
        selected_residual = float(selected["residual_boundary_jump_abs_max_eur_mwh"])
        perturbations = perturbations.copy()
        perturbations["month_p95_change_vs_selected"] = (
            perturbations["month_adjacent_jump_abs_p95_eur_mwh"].astype(float) - selected_month
        )
        perturbations["residual_seam_change_vs_selected"] = (
            perturbations["residual_boundary_jump_abs_max_eur_mwh"].astype(float) - selected_residual
        )
        lines.extend(
            [
                "",
                "## +/-25% Stability Perturbations",
                "",
                "| label | month_p95_change | residual_seam_change | prior_rmse |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in perturbations.iterrows():
            lines.append(
                "| "
                f"{row['label']} | "
                f"{row['month_p95_change_vs_selected']:.6f} | "
                f"{row['residual_seam_change_vs_selected']:.6f} | "
                f"{row['prior_rmse_eur_mwh']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Hard EEX constraints remain non-negotiable; candidates with non-zero residuals are invalid.",
            "- The L-curve score is a documented heuristic over distance-to-prior and shape roughness.",
            "- Desk/independent quant approval is still required before using these lambdas in production.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Candidate hourly CSV used as prior.")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--spot", default="data/epex_hourly.parquet")
    parser.add_argument("--price-column", default=PRICE)
    parser.add_argument("--start", default="2028-01-01")
    parser.add_argument("--end", default="2030-12-31 23:00")
    parser.add_argument("--output", default="output/lt_quant_lambda_calibration/lambda_sensitivity_report.csv")
    parser.add_argument("--report", default=".planning/phases/14-lt-audit-remediation/LT-QUANT-LAMBDA-CALIBRATION.md")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    output_path = Path(args.output)
    report_path = Path(args.report)
    prior = _read_prior(csv_path, price_column=args.price_column, start=args.start, end=args.end)
    latest, prices_by_load_type = _latest_eex_prices_by_load_type(Path(args.forwards), market="CH")
    base_prices_all = prices_by_load_type.get("BASE", {})
    peak_prices_all = prices_by_load_type.get("PEAK", {})
    if not base_prices_all:
        raise ValueError(f"no CH BASE forwards found in {args.forwards}")
    folds = _year_folds(prior)
    forward_snapshot_date = str(pd.Timestamp(latest).date())

    rows: list[dict[str, float | str]] = []
    fold_detail_rows: list[pd.DataFrame] = []
    candidates = _default_candidates()
    for candidate in candidates:
        row, fold_df = _evaluate_candidate(
            candidate,
            folds=folds,
            base_prices_all=base_prices_all,
            peak_prices_all=peak_prices_all,
            forward_snapshot_date=forward_snapshot_date,
        )
        rows.append(row)
        fold_detail_rows.append(fold_df)
    raw_results = pd.DataFrame(rows)
    selected_label, scored_results = _select_lcurve_candidate(raw_results)
    selected_candidate = next(candidate for candidate in candidates if candidate.label == selected_label)

    perturbation_rows: list[dict[str, float | str]] = []
    perturbation_fold_rows: list[pd.DataFrame] = []
    for candidate in _candidate_perturbations(selected_candidate):
        row, fold_df = _evaluate_candidate(
            candidate,
            folds=folds,
            base_prices_all=base_prices_all,
            peak_prices_all=peak_prices_all,
            forward_snapshot_date=forward_snapshot_date,
        )
        perturbation_rows.append(row)
        perturbation_fold_rows.append(fold_df)
    perturbations = pd.DataFrame(perturbation_rows)
    detail = pd.concat(fold_detail_rows + perturbation_fold_rows, ignore_index=True, sort=False)
    output = pd.concat([scored_results, perturbations, detail], ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    selected_row = output.loc[output["label"] == selected_label].iloc[0]
    spot_benchmark = _spot_month_benchmark(Path(args.spot) if args.spot else None)
    _write_report(
        report_path,
        csv_path=csv_path,
        output_csv=output_path,
        selected=selected_row,
        spot_benchmark=spot_benchmark,
        perturbations=perturbations,
    )
    print(
        "[lambda-calibration] "
        f"selected={selected_label} "
        f"lambda_h={selected_candidate.lambda_smooth_h:g} "
        f"lambda_m={selected_candidate.lambda_smooth_m:g} "
        f"lambda_seam={selected_candidate.lambda_seam:g} "
        f"metrics={output_path} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
