"""Audit CH HFC/PFC seasonal coherence and duck-curve diagnostics.

The audit is intentionally additive: it does not change generated curves.  It
flags cases where the final hourly curve or the internal base component imply a
questionable monthly shape, especially for annual-only forward years where the
model must synthesize missing monthly products.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets
from scripts.export_local_test_ch_hourly_csv import _eex_peak_mask, _parse_timestamp_ch

PRICE = "price_weighted_mean_eur_mwh"
LOCAL_TZ = "Europe/Zurich"


def _load_hourly_csv(path: Path, *, price_column: str = PRICE) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp_ch", price_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["ts"] = _parse_timestamp_ch(df["timestamp_ch"], df.get("utc_offset_ch"))
    df["year"] = df["ts"].dt.year.astype(int)
    df["month"] = df["ts"].dt.month.astype(int)
    df["quarter"] = df["ts"].dt.quarter.astype(int)
    df["hour"] = df["ts"].dt.hour.astype(int)
    df["weekday"] = df["ts"].dt.weekday.astype(int)
    df[price_column] = pd.to_numeric(df[price_column], errors="raise")
    return df


def _latest_forwards(path: Path, *, market: str) -> tuple[pd.Timestamp, pd.DataFrame]:
    df = pd.read_parquet(path)
    sub = df[df["market"].astype(str) == market].copy()
    if sub.empty:
        raise ValueError(f"no forwards for market={market!r} in {path}")
    sub["date"] = pd.to_datetime(sub["date"])
    sub["load_type"] = sub["load_type"].astype(str).str.upper()
    base = sub[sub["load_type"] == "BASE"].copy()
    if base.empty:
        raise ValueError(f"no BASE forwards for market={market!r} in {path}")
    latest = pd.Timestamp(base["date"].max())
    snap = sub[sub["date"] == latest].copy()
    snap["product"] = snap["product"].astype(str)
    snap["price"] = snap["price"].astype(float)
    return latest, snap


def _product_mask(df: pd.DataFrame, product: str) -> pd.Series:
    if "-Q" in product:
        year = int(product[:4])
        quarter = int(product[-1])
        return (df["year"] == year) & (df["quarter"] == quarter)
    if "-" in product:
        year = int(product[:4])
        month = int(product[5:7])
        return (df["year"] == year) & (df["month"] == month)
    if product.isdigit():
        return df["year"] == int(product)
    return pd.Series(False, index=df.index)


def quoted_product_residuals(
    hourly: pd.DataFrame,
    forwards: pd.DataFrame,
    *,
    price_column: str = PRICE,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in forwards.sort_values("product").iterrows():
        product = str(row["product"])
        load_type = str(row.get("load_type", "BASE")).upper()
        mask = _product_mask(hourly, product)
        if load_type == "PEAK":
            mask = mask & _eex_peak_mask(hourly["ts"], country="CH")
        if not bool(mask.any()):
            continue
        mean = float(hourly.loc[mask, price_column].mean())
        target = float(row["price"])
        rows.append(
            {
                "load_type": load_type,
                "product": product,
                "target_eex_eur_mwh": target,
                "curve_mean_eur_mwh": mean,
                "residual_eur_mwh": mean - target,
                "abs_residual_eur_mwh": abs(mean - target),
                "rows": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["load_type", "product"]).reset_index(drop=True)


def monthly_means(
    hourly: pd.DataFrame,
    *,
    price_column: str = PRICE,
) -> pd.DataFrame:
    return (
        hourly.groupby(["year", "month"], as_index=False)
        .agg(
            mean_eur_mwh=(price_column, "mean"),
            min_eur_mwh=(price_column, "min"),
            max_eur_mwh=(price_column, "max"),
            rows=(price_column, "size"),
        )
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )


def hourly_month_matrix(
    hourly: pd.DataFrame,
    *,
    price_column: str = PRICE,
) -> pd.DataFrame:
    return (
        hourly.groupby(["year", "month", "hour"], as_index=False)
        .agg(mean_eur_mwh=(price_column, "mean"))
        .sort_values(["year", "month", "hour"])
        .reset_index(drop=True)
    )


def _coverage_for_year(forwards: pd.DataFrame, year: int) -> str:
    products = set(forwards["product"].astype(str))
    months = {f"{year}-{month:02d}" for month in range(1, 13)}
    quarters = {f"{year}-Q{quarter}" for quarter in range(1, 5)}
    if months & products:
        return "monthly_or_mixed"
    if quarters.issubset(products):
        return "full_quarter"
    if quarters & products:
        return "partial_quarter"
    if str(year) in products:
        return "annual_only"
    return "unquoted"


def _quarter_months(year: int, quarter: int) -> list[int]:
    first = (int(quarter) - 1) * 3 + 1
    return [first, first + 1, first + 2]


def _month_counts(hourly: pd.DataFrame, *, year: int, months: list[int], peak_only: bool = False) -> dict[int, int]:
    frame = hourly[(hourly["year"] == int(year)) & (hourly["month"].isin(months))]
    if peak_only:
        frame = frame[_eex_peak_mask(frame["ts"], country="CH")]
    return {month: int((frame["month"] == month).sum()) for month in months}


def _weighted_parent_mean(values: dict[int, float], counts: dict[int, int]) -> float | None:
    total = sum(int(counts.get(month, 0)) for month in values)
    if total <= 0:
        return None
    return float(sum(float(values[month]) * int(counts.get(month, 0)) for month in values) / total)


def monthly_split_checks(
    hourly: pd.DataFrame,
    ch_forwards: pd.DataFrame,
    neighbor_forwards: pd.DataFrame,
    *,
    price_column: str = PRICE,
    neighbor_market: str = "DE",
) -> pd.DataFrame:
    """Compare unquoted CH monthly splits with neighbor monthly market shape."""
    rows: list[dict[str, object]] = []
    ch_products_by_load = {
        str(load_type).upper(): set(group["product"].astype(str))
        for load_type, group in ch_forwards.groupby("load_type")
    }
    by_neighbor = {
        (str(row["load_type"]).upper(), str(row["product"])): float(row["price"])
        for _, row in neighbor_forwards.iterrows()
    }
    for year in sorted(hourly["year"].unique()):
        for quarter in range(1, 5):
            months = _quarter_months(int(year), quarter)
            if not bool(((hourly["year"] == int(year)) & (hourly["month"].isin(months))).any()):
                continue
            for load_type in ["BASE", "PEAK"]:
                ch_products = ch_products_by_load.get(load_type, set())
                direct_ch_months = {f"{int(year)}-{month:02d}" for month in months} & ch_products
                if len(direct_ch_months) == len(months):
                    continue
                parent = f"{int(year)}-Q{quarter}"
                parent_key = (load_type, parent)
                month_keys = {month: (load_type, f"{int(year)}-{month:02d}") for month in months}
                if parent_key not in by_neighbor or any(key not in by_neighbor for key in month_keys.values()):
                    continue
                peak_only = load_type == "PEAK"
                counts = _month_counts(hourly, year=int(year), months=months, peak_only=peak_only)
                if any(counts[month] <= 0 for month in months):
                    continue
                frame = hourly[(hourly["year"] == int(year)) & (hourly["month"].isin(months))]
                if peak_only:
                    frame = frame[_eex_peak_mask(frame["ts"], country="CH")]
                ch_month = frame.groupby("month")[price_column].mean().to_dict()
                ch_parent = _weighted_parent_mean({m: float(ch_month[m]) for m in months}, counts)
                neighbor_month = {m: by_neighbor[month_keys[m]] for m in months}
                neighbor_parent = _weighted_parent_mean(neighbor_month, counts)
                if ch_parent is None or neighbor_parent is None:
                    continue
                ch_dev = np.array([float(ch_month[m]) - ch_parent for m in months], dtype=float)
                nb_dev = np.array([float(neighbor_month[m]) - neighbor_parent for m in months], dtype=float)
                corr = float(np.corrcoef(ch_dev, nb_dev)[0, 1]) if np.std(ch_dev) > 1e-9 and np.std(nb_dev) > 1e-9 else np.nan
                nb_range = float(np.max(nb_dev) - np.min(nb_dev))
                ch_range = float(np.max(ch_dev) - np.min(ch_dev))
                amplitude_ratio = ch_range / nb_range if nb_range > 1e-9 else np.nan
                max_abs_dev_diff = float(np.max(np.abs(ch_dev - nb_dev)))
                level_leak = (
                    float(np.median([abs(float(ch_month[m]) - float(neighbor_month[m])) for m in months])) < 0.02
                    and abs(float(ch_parent) - float(neighbor_parent)) > 0.5
                )
                severity = "ok"
                reason = "neighbor monthly split plausible"
                if level_leak:
                    severity = "critical"
                    reason = "unquoted CH monthly levels appear copied from neighbor levels"
                elif np.isfinite(corr) and corr < 0.20:
                    severity = "critical"
                    reason = "CH monthly split is weakly aligned with neighbor shape"
                elif max_abs_dev_diff > 45.0:
                    severity = "critical"
                    reason = "CH monthly deviations are too far from neighbor deviations"
                elif nb_range >= 5.0 and np.isfinite(amplitude_ratio) and (amplitude_ratio < 0.33 or amplitude_ratio > 3.0):
                    severity = "critical"
                    reason = "CH monthly split amplitude is outside robust neighbor range"
                elif np.isfinite(corr) and corr < 0.45:
                    severity = "warning"
                    reason = "CH monthly split correlation vs neighbor is low"
                elif max_abs_dev_diff > 35.0:
                    severity = "warning"
                    reason = "CH monthly deviations are wide vs neighbor deviations"
                elif nb_range >= 5.0 and np.isfinite(amplitude_ratio) and (amplitude_ratio < 0.50 or amplitude_ratio > 2.5):
                    severity = "warning"
                    reason = "CH monthly split amplitude is marginal vs neighbor"
                rows.append(
                    {
                        "year": int(year),
                        "quarter": int(quarter),
                        "load_type": load_type,
                        "neighbor_market": neighbor_market,
                        "ch_parent_mean_eur_mwh": float(ch_parent),
                        "neighbor_parent_mean_eur_mwh": float(neighbor_parent),
                        "split_corr": corr,
                        "ch_range_eur_mwh": ch_range,
                        "neighbor_range_eur_mwh": nb_range,
                        "amplitude_ratio": amplitude_ratio,
                        "max_abs_dev_diff_eur_mwh": max_abs_dev_diff,
                        "level_leak": bool(level_leak),
                        "severity": severity,
                        "reason": reason,
                        **{f"ch_m{m}_dev_eur_mwh": float(ch_dev[pos]) for pos, m in enumerate(months)},
                        **{f"neighbor_m{m}_dev_eur_mwh": float(nb_dev[pos]) for pos, m in enumerate(months)},
                    }
                )
    return pd.DataFrame(rows)


def calendar_coherence_checks(hourly: pd.DataFrame, *, price_column: str = PRICE) -> pd.DataFrame:
    """Audit month-level week/weekend shape and week-to-week roughness."""
    rows: list[dict[str, object]] = []
    frame = hourly.copy()
    iso = frame["ts"].dt.isocalendar()
    frame["iso_year"] = iso.year.astype(int)
    frame["iso_week"] = iso.week.astype(int)
    for (year, month), group in frame.groupby(["year", "month"]):
        weekday_mean = float(group.loc[group["weekday"] < 5, price_column].mean())
        weekend_mean = float(group.loc[group["weekday"] >= 5, price_column].mean())
        weekend_minus_weekday = weekend_mean - weekday_mean
        weekly = group.groupby(["iso_year", "iso_week"])[price_column].mean().sort_index()
        max_week_to_week = float(weekly.diff().abs().max()) if len(weekly) > 1 else 0.0
        severity = "ok"
        reason = "calendar shape plausible"
        if weekend_minus_weekday > 10.0:
            severity = "critical"
            reason = "weekend premium over weekday is implausibly high"
        elif weekend_minus_weekday > 5.0:
            severity = "warning"
            reason = "weekend premium over weekday is high"
        elif max_week_to_week > 35.0:
            severity = "warning"
            reason = "large week-to-week mean jump inside month"
        rows.append(
            {
                "year": int(year),
                "month": int(month),
                "weekday_mean_eur_mwh": weekday_mean,
                "weekend_mean_eur_mwh": weekend_mean,
                "weekend_minus_weekday_eur_mwh": weekend_minus_weekday,
                "max_week_to_week_delta_eur_mwh": max_week_to_week,
                "severity": severity,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def monthly_path_checks(
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
    ch_forwards: pd.DataFrame,
    neighbor_forwards: pd.DataFrame | None = None,
    *,
    neighbor_market: str = "DE",
) -> pd.DataFrame:
    """Audit synthetic month paths and seam jumps around synthetic EEX buckets."""
    base_prices = {
        str(row["product"]): float(row["price"])
        for _, row in ch_forwards[ch_forwards["load_type"].astype(str).str.upper() == "BASE"].iterrows()
    }
    if not base_prices:
        return pd.DataFrame()
    buckets, bucket_targets = calibration_buckets(hourly["ts"], base_prices)
    bucket_frame = hourly[["year", "month"]].copy()
    bucket_frame["bucket"] = buckets.to_numpy()
    bucket_by_month = (
        bucket_frame.groupby(["year", "month"])["bucket"]
        .agg(lambda s: str(s.mode().iloc[0]) if not s.mode().empty else "")
        .reset_index()
    )
    path = monthly.merge(bucket_by_month, on=["year", "month"], how="left").sort_values(["year", "month"])
    rows: list[dict[str, object]] = []
    neighbor_months: dict[tuple[int, int], float] = {}
    if neighbor_forwards is not None and not neighbor_forwards.empty:
        nb_base = neighbor_forwards[neighbor_forwards["load_type"].astype(str).str.upper() == "BASE"].copy()
        for _, row in nb_base.iterrows():
            product = str(row["product"])
            try:
                period = pd.Period(product, freq="M")
            except ValueError:
                continue
            if len(product) == 7 and product[4] == "-":
                neighbor_months[(int(period.year), int(period.month))] = float(row["price"])

    def synthetic_or_unquoted(bucket: str) -> bool:
        return bucket.endswith("-RESIDUAL") or bucket.isdigit() or "-Q" in bucket

    def quote_constrained(bucket: str) -> bool:
        return bucket in base_prices or bucket.endswith("-RESIDUAL")

    for year, group in path.groupby("year", sort=True):
        group = group.reset_index(drop=True)
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]
            if str(prev["bucket"]) != str(curr["bucket"]):
                continue
            bucket = str(curr["bucket"])
            synthetic = bucket.endswith("-RESIDUAL") or bucket.isdigit()
            if not synthetic:
                continue
            delta = float(curr["mean_eur_mwh"] - prev["mean_eur_mwh"])
            severity = "ok"
            reason = "month-to-month path plausible"
            if abs(delta) > 25.0:
                severity = "critical"
                reason = "large adjacent monthly jump inside synthetic bucket"
            elif abs(delta) > 18.0:
                severity = "warning"
                reason = "elevated adjacent monthly jump inside synthetic bucket"
            rows.append(
                {
                    "year": int(year),
                    "bucket": bucket,
                    "from_month": int(prev["month"]),
                    "to_month": int(curr["month"]),
                    "from_mean_eur_mwh": float(prev["mean_eur_mwh"]),
                    "to_mean_eur_mwh": float(curr["mean_eur_mwh"]),
                    "delta_eur_mwh": delta,
                    "check_type": "adjacent_delta",
                    "severity": severity,
                    "reason": reason,
                }
            )
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]
            prev_bucket = str(prev["bucket"])
            curr_bucket = str(curr["bucket"])
            if prev_bucket == curr_bucket:
                continue
            if not (synthetic_or_unquoted(prev_bucket) or synthetic_or_unquoted(curr_bucket)):
                continue
            delta = float(curr["mean_eur_mwh"] - prev["mean_eur_mwh"])
            year_i = int(curr["year"])
            prev_month = int(prev["month"])
            curr_month = int(curr["month"])
            neighbor_delta = np.nan
            excess_vs_neighbor = np.nan
            residual_edge_deviation = np.nan
            nb_prev = neighbor_months.get((year_i, prev_month))
            nb_curr = neighbor_months.get((year_i, curr_month))
            if nb_prev is not None and nb_curr is not None:
                neighbor_delta = float(nb_curr - nb_prev)
                excess_vs_neighbor = float(abs(delta - neighbor_delta))
            severity = "ok"
            reason = "inter-bucket monthly seam plausible"
            if quote_constrained(prev_bucket) and quote_constrained(curr_bucket):
                residual_bucket = curr_bucket if curr_bucket.endswith("-RESIDUAL") else prev_bucket if prev_bucket.endswith("-RESIDUAL") else ""
                if residual_bucket:
                    residual_target = bucket_targets.get(residual_bucket)
                    if residual_target is not None:
                        residual_month_mean = float(curr["mean_eur_mwh"] if curr_bucket == residual_bucket else prev["mean_eur_mwh"])
                        residual_edge_deviation = float(residual_month_mean - residual_target)
                        if abs(residual_edge_deviation) > 15.0:
                            severity = "critical"
                            reason = "edge month is materially away from CH residual target"
                        elif abs(residual_edge_deviation) > 10.0:
                            severity = "warning"
                            reason = "edge month is away from CH residual target"
                if severity == "ok":
                    reason = "inter-bucket seam is constrained by CH quoted/implied forwards"
            elif np.isfinite(excess_vs_neighbor):
                if abs(delta) > 30.0 and excess_vs_neighbor > 12.0:
                    severity = "critical"
                    reason = "inter-bucket monthly seam is materially wider than neighbor market"
                elif abs(delta) > 22.0 and excess_vs_neighbor > 8.0:
                    severity = "warning"
                    reason = "inter-bucket monthly seam is wider than neighbor market"
            elif abs(delta) > 35.0:
                severity = "critical"
                reason = "large unverified inter-bucket monthly seam"
            elif abs(delta) > 25.0:
                severity = "warning"
                reason = "elevated unverified inter-bucket monthly seam"
            rows.append(
                {
                    "year": year_i,
                    "bucket": f"{prev_bucket}->{curr_bucket}",
                    "from_bucket": prev_bucket,
                    "to_bucket": curr_bucket,
                    "from_month": prev_month,
                    "to_month": curr_month,
                    "from_mean_eur_mwh": float(prev["mean_eur_mwh"]),
                    "to_mean_eur_mwh": float(curr["mean_eur_mwh"]),
                    "delta_eur_mwh": delta,
                    "neighbor_market": neighbor_market,
                    "neighbor_delta_eur_mwh": neighbor_delta,
                    "excess_vs_neighbor_eur_mwh": excess_vs_neighbor,
                    "residual_edge_deviation_eur_mwh": residual_edge_deviation,
                    "check_type": "inter_bucket_seam",
                    "severity": severity,
                    "reason": reason,
                }
            )
        for idx in range(1, len(group) - 1):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]
            nxt = group.iloc[idx + 1]
            if len({str(prev["bucket"]), str(curr["bucket"]), str(nxt["bucket"])}) != 1:
                continue
            bucket = str(curr["bucket"])
            synthetic = bucket.endswith("-RESIDUAL") or bucket.isdigit()
            if not synthetic:
                continue
            left = float(curr["mean_eur_mwh"] - prev["mean_eur_mwh"])
            right = float(nxt["mean_eur_mwh"] - curr["mean_eur_mwh"])
            if left * right >= 0.0:
                continue
            reversal = abs(left) + abs(right)
            severity = "ok"
            reason = "month-to-month reversal plausible"
            if reversal > 42.0 and min(abs(left), abs(right)) > 12.0:
                severity = "critical"
                reason = "sharp local monthly reversal inside synthetic bucket"
            elif reversal > 30.0 and min(abs(left), abs(right)) > 10.0:
                severity = "warning"
                reason = "elevated local monthly reversal inside synthetic bucket"
            rows.append(
                {
                    "year": int(year),
                    "bucket": bucket,
                    "from_month": int(prev["month"]),
                    "to_month": int(nxt["month"]),
                    "from_mean_eur_mwh": float(prev["mean_eur_mwh"]),
                    "to_mean_eur_mwh": float(nxt["mean_eur_mwh"]),
                    "delta_eur_mwh": reversal,
                    "check_type": "local_reversal",
                    "severity": severity,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def _month_bucket_frame(hourly: pd.DataFrame, base_prices: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
    buckets, bucket_targets = calibration_buckets(hourly["ts"], base_prices)
    bucket_frame = hourly[["year", "month"]].copy()
    bucket_frame["bucket"] = buckets.to_numpy()
    bucket_by_month = (
        bucket_frame.groupby(["year", "month"])["bucket"]
        .agg(lambda s: str(s.mode().iloc[0]) if not s.mode().empty else "")
        .reset_index()
    )
    return bucket_by_month, bucket_targets


def cross_year_month_shape_checks(
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
    ch_forwards: pd.DataFrame,
) -> pd.DataFrame:
    """Audit cross-year month consistency against implied parent bucket levels.

    This catches a defect that within-year path checks miss: two adjacent
    synthetic buckets can both preserve their EEX means while assigning the
    parent spread to implausible parts of the seasonal shape.
    """
    base_prices = {
        str(row["product"]): float(row["price"])
        for _, row in ch_forwards[ch_forwards["load_type"].astype(str).str.upper() == "BASE"].iterrows()
    }
    if not base_prices:
        return pd.DataFrame()
    bucket_by_month, bucket_targets = _month_bucket_frame(hourly, base_prices)
    path = monthly.merge(bucket_by_month, on=["year", "month"], how="left").sort_values(["year", "month"])
    by_key = {
        (int(row["year"]), int(row["month"])): row
        for _, row in path.iterrows()
        if str(row.get("bucket", ""))
    }

    def comparable(bucket: str) -> bool:
        return bool(bucket) and (bucket.endswith("-RESIDUAL") or bucket.isdigit())

    def row_for(year: int, month: int) -> pd.Series | None:
        return by_key.get((int(year), int(month)))

    rows: list[dict[str, object]] = []
    years = sorted(int(year) for year in path["year"].unique())
    for from_year, to_year in zip(years, years[1:]):
        common_months = sorted(
            set(path.loc[path["year"] == from_year, "month"].astype(int))
            & set(path.loc[path["year"] == to_year, "month"].astype(int))
        )
        for month in common_months:
            left = row_for(from_year, month)
            right = row_for(to_year, month)
            if left is None or right is None:
                continue
            from_bucket = str(left["bucket"])
            to_bucket = str(right["bucket"])
            if not (comparable(from_bucket) and comparable(to_bucket)):
                continue
            if from_bucket not in bucket_targets or to_bucket not in bucket_targets:
                continue
            parent_spread = float(bucket_targets[to_bucket] - bucket_targets[from_bucket])
            same_month_spread = float(right["mean_eur_mwh"] - left["mean_eur_mwh"])
            spread_error = float(same_month_spread - parent_spread)
            severity = "ok"
            reason = "same-month cross-year spread is coherent with parent bucket spread"
            if abs(parent_spread) > 2.0 and same_month_spread * parent_spread < 0.0 and abs(spread_error) > 5.0:
                severity = "critical"
                reason = "same-month cross-year spread has opposite sign to parent bucket spread"
            elif abs(spread_error) > 12.0:
                severity = "critical"
                reason = "same-month cross-year spread is materially inconsistent with parent bucket spread"
            elif abs(parent_spread) > 1.5 and abs(same_month_spread) < 0.25:
                severity = "warning"
                reason = "same-month values are near-cloned despite a non-zero parent bucket spread"
            elif abs(spread_error) > 8.0:
                severity = "warning"
                reason = "same-month cross-year spread is wide versus parent bucket spread"
            rows.append(
                {
                    "from_year": int(from_year),
                    "to_year": int(to_year),
                    "month": int(month),
                    "from_bucket": from_bucket,
                    "to_bucket": to_bucket,
                    "from_parent_target_eur_mwh": float(bucket_targets[from_bucket]),
                    "to_parent_target_eur_mwh": float(bucket_targets[to_bucket]),
                    "parent_spread_eur_mwh": parent_spread,
                    "from_month_mean_eur_mwh": float(left["mean_eur_mwh"]),
                    "to_month_mean_eur_mwh": float(right["mean_eur_mwh"]),
                    "same_month_spread_eur_mwh": same_month_spread,
                    "spread_error_eur_mwh": spread_error,
                    "check_type": "same_month_parent_spread",
                    "severity": severity,
                    "reason": reason,
                }
            )

        spring_month = 4
        winter_month = 12
        left_spring = row_for(from_year, spring_month)
        left_winter = row_for(from_year, winter_month)
        right_spring = row_for(to_year, spring_month)
        right_winter = row_for(to_year, winter_month)
        if left_spring is None or left_winter is None or right_spring is None or right_winter is None:
            continue
        from_bucket_spring = str(left_spring["bucket"])
        from_bucket_winter = str(left_winter["bucket"])
        to_bucket_spring = str(right_spring["bucket"])
        to_bucket_winter = str(right_winter["bucket"])
        if (
            from_bucket_spring != from_bucket_winter
            or to_bucket_spring != to_bucket_winter
            or not comparable(from_bucket_spring)
            or not comparable(to_bucket_spring)
        ):
            continue
        from_slope = float(left_winter["mean_eur_mwh"] - left_spring["mean_eur_mwh"])
        to_slope = float(right_winter["mean_eur_mwh"] - right_spring["mean_eur_mwh"])
        slope_delta = float(to_slope - from_slope)
        severity = "ok"
        reason = "cross-year Apr-Dec seasonal slope is coherent"
        if abs(slope_delta) > 12.0:
            severity = "critical"
            reason = "cross-year Apr-Dec seasonal slope changes materially without a quoted monthly signal"
        elif abs(slope_delta) > 8.0:
            severity = "warning"
            reason = "cross-year Apr-Dec seasonal slope changes sharply"
        rows.append(
            {
                "from_year": int(from_year),
                "to_year": int(to_year),
                "month": "04-12",
                "from_bucket": from_bucket_spring,
                "to_bucket": to_bucket_spring,
                "from_parent_target_eur_mwh": float(bucket_targets.get(from_bucket_spring, np.nan)),
                "to_parent_target_eur_mwh": float(bucket_targets.get(to_bucket_spring, np.nan)),
                "parent_spread_eur_mwh": float(
                    bucket_targets.get(to_bucket_spring, np.nan) - bucket_targets.get(from_bucket_spring, np.nan)
                ),
                "from_month_mean_eur_mwh": float(left_spring["mean_eur_mwh"]),
                "to_month_mean_eur_mwh": float(right_spring["mean_eur_mwh"]),
                "from_winter_minus_spring_eur_mwh": from_slope,
                "to_winter_minus_spring_eur_mwh": to_slope,
                "slope_delta_eur_mwh": slope_delta,
                "check_type": "seasonal_slope_delta",
                "severity": severity,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _month_value(months: pd.DataFrame, year: int, month: int) -> float | None:
    sub = months[(months["year"] == year) & (months["month"] == month)]
    if sub.empty:
        return None
    return float(sub["mean_eur_mwh"].iloc[0])


def seasonal_coherence_checks(
    monthly: pd.DataFrame,
    forwards: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in sorted(monthly["year"].unique()):
        jan = _month_value(monthly, int(year), 1)
        feb = _month_value(monthly, int(year), 2)
        sep = _month_value(monthly, int(year), 9)
        octo = _month_value(monthly, int(year), 10)
        nov = _month_value(monthly, int(year), 11)
        dec = _month_value(monthly, int(year), 12)
        if jan is None or octo is None:
            continue

        q1_values = [value for value in [jan, feb, _month_value(monthly, int(year), 3)] if value is not None]
        q4_values = [value for value in [octo, nov, dec] if value is not None]
        autumn_values = [value for value in [sep, octo, nov] if value is not None]
        winter_values = [value for value in [jan, feb, dec] if value is not None]
        jan_minus_oct = jan - octo
        q1_minus_q4 = float(np.mean(q1_values) - np.mean(q4_values)) if q1_values and q4_values else np.nan
        winter_minus_autumn = (
            float(np.mean(winter_values) - np.mean(autumn_values)) if winter_values and autumn_values else np.nan
        )
        coverage = _coverage_for_year(forwards, int(year))

        severity = "ok"
        reason = "seasonal ordering plausible"
        if coverage == "annual_only" and jan_minus_oct < -5.0:
            severity = "critical"
            reason = "annual-only synthetic shape has January materially below October"
        elif coverage == "annual_only" and jan_minus_oct < 0.0:
            severity = "warning"
            reason = "annual-only synthetic shape has January below October"
        elif coverage == "partial_quarter" and q1_minus_q4 < -5.0:
            severity = "warning"
            reason = "partial-quarter year has Q1 materially below Q4 after completion"

        rows.append(
            {
                "year": int(year),
                "forward_coverage": coverage,
                "jan_mean_eur_mwh": jan,
                "oct_mean_eur_mwh": octo,
                "jan_minus_oct_eur_mwh": jan_minus_oct,
                "q1_minus_q4_eur_mwh": q1_minus_q4,
                "winter_minus_autumn_eur_mwh": winter_minus_autumn,
                "severity": severity,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def component_monthly_means(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("component parquet index must be a DatetimeIndex")
    if frame.index.tz is None:
        frame.index = pd.to_datetime(frame.index, utc=True)
    else:
        frame.index = frame.index.tz_convert("UTC")
    local = frame.tz_convert(LOCAL_TZ)
    numeric_cols = [col for col in ["price_shape", "B", "f_S", "f_W", "f_H", "f_Q", "f_WV", "f_bridge"] if col in local]
    hourly = local[numeric_cols].resample("h").mean()
    hourly["year"] = hourly.index.year.astype(int)
    hourly["month"] = hourly.index.month.astype(int)
    out = hourly.groupby(["year", "month"], as_index=False)[numeric_cols].mean()
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def component_jan_oct_checks(component_monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric_cols = [c for c in component_monthly.columns if c not in {"year", "month"}]
    for year in sorted(component_monthly["year"].unique()):
        jan = component_monthly[(component_monthly["year"] == year) & (component_monthly["month"] == 1)]
        octo = component_monthly[(component_monthly["year"] == year) & (component_monthly["month"] == 10)]
        if jan.empty or octo.empty:
            continue
        row: dict[str, object] = {"year": int(year)}
        for col in numeric_cols:
            row[f"{col}_jan_minus_oct"] = float(jan[col].iloc[0] - octo[col].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    lines = [
        "| " + " | ".join(str(c) for c in view.columns) + " |",
        "|" + "|".join("---" for _ in view.columns) + "|",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"| ... | {' | '.join([''] * (len(view.columns) - 1))} |")
    return "\n".join(lines)


def _severity_counts(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty or "severity" not in frame:
        return 0, 0
    return int((frame["severity"] == "critical").sum()), int((frame["severity"] == "warning").sum())


def write_report(
    path: Path,
    *,
    csv_path: Path,
    forwards_path: Path,
    latest_forward_date: pd.Timestamp,
    neighbor_market: str,
    quoted_residuals: pd.DataFrame,
    seasonal_checks: pd.DataFrame,
    monthly_split_checks_frame: pd.DataFrame,
    calendar_checks: pd.DataFrame,
    monthly_path_checks_frame: pd.DataFrame,
    cross_year_checks: pd.DataFrame,
    monthly: pd.DataFrame,
    hour_month: pd.DataFrame,
    component_checks: pd.DataFrame | None,
) -> None:
    seasonal_critical, seasonal_warnings = _severity_counts(seasonal_checks)
    split_critical, split_warnings = _severity_counts(monthly_split_checks_frame)
    calendar_critical, calendar_warnings = _severity_counts(calendar_checks)
    path_critical, path_warnings = _severity_counts(monthly_path_checks_frame)
    cross_year_critical, cross_year_warnings = _severity_counts(cross_year_checks)
    critical = seasonal_critical + split_critical + calendar_critical + path_critical + cross_year_critical
    warnings = seasonal_warnings + split_warnings + calendar_warnings + path_warnings + cross_year_warnings
    max_quoted_residual = (
        float(quoted_residuals["abs_residual_eur_mwh"].max()) if not quoted_residuals.empty else float("nan")
    )
    jan_oct_rows = seasonal_checks[seasonal_checks["jan_minus_oct_eur_mwh"] < 0.0]
    lines = [
        "# CH HFC Seasonal Coherence Audit",
        "",
        f"* CSV: `{csv_path}`",
        f"* forwards: `{forwards_path}`",
        f"* latest EEX CH BASE date: `{latest_forward_date.date()}`",
        "* scope: `local/test diagnostics`",
        "* production approval: `NO`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total critical flags | {critical} |",
        f"| total warning flags | {warnings} |",
        f"| seasonal critical flags | {seasonal_critical} |",
        f"| seasonal warning flags | {seasonal_warnings} |",
        f"| monthly split critical flags | {split_critical} |",
        f"| monthly split warning flags | {split_warnings} |",
        f"| calendar critical flags | {calendar_critical} |",
        f"| calendar warning flags | {calendar_warnings} |",
        f"| monthly path critical flags | {path_critical} |",
        f"| monthly path warning flags | {path_warnings} |",
        f"| cross-year month shape critical flags | {cross_year_critical} |",
        f"| cross-year month shape warning flags | {cross_year_warnings} |",
        f"| max quoted-product residual EUR/MWh | {max_quoted_residual:.6f} |",
        f"| years with January below October | {len(jan_oct_rows)} |",
        "",
        "## Seasonal Checks",
        "",
        _md_table(seasonal_checks),
        "",
        "## Unquoted Monthly Split Checks",
        "",
        f"Neighbor market guide: `{neighbor_market}`. The test compares CH month-vs-parent deviations, not absolute levels.",
        "",
        _md_table(monthly_split_checks_frame),
        "",
        "## Monthly Path Checks",
        "",
        "These checks evaluate adjacent months inside synthetic annual/residual buckets and "
        "month-to-month seams crossing into or out of synthetic/unquoted buckets.",
        "",
        _md_table(monthly_path_checks_frame),
        "",
        "## Cross-Year Month Shape Checks",
        "",
        "These checks compare homologous months and Apr-Dec seasonal slopes across adjacent synthetic "
        "BASE buckets, using the implied parent bucket targets as the economic reference.",
        "",
        _md_table(cross_year_checks),
        "",
        "## Calendar Coherence Checks",
        "",
        _md_table(calendar_checks),
        "",
        "## Quoted EEX Product Residuals",
        "",
        _md_table(quoted_residuals),
        "",
        "## Monthly Means",
        "",
        _md_table(monthly),
        "",
        "## Hour x Month Duck Diagnostics",
        "",
        "_Full table is written to the CSV sidecar; first rows shown below._",
        "",
        _md_table(hour_month, max_rows=48),
        "",
    ]
    if component_checks is not None:
        lines.extend(
            [
                "## Component Jan-Oct Checks",
                "",
                _md_table(component_checks),
                "",
                "Interpretation: if `B_jan_minus_oct` is already negative, the inversion is in the "
                "forward cascade/base reconstruction, not in the intraday duck curve.",
                "",
            ]
        )
    lines.extend(
        [
            "## Gate Interpretation",
            "",
            "* `critical` means annual-only synthetic monthly completion contradicts the expected Swiss winter premium.",
            "* Monthly split flags are evaluated on unquoted CH months only; quoted EEX monthly products remain market signals.",
            "* Monthly path flags catch sharp adjacent jumps and local reversals inside annual/residual synthetic buckets, "
            "plus cross-bucket monthly seams that are materially wider than the neighbor market guide.",
            "* Cross-year month shape flags catch cases where adjacent synthetic buckets preserve EEX means but allocate "
            "the parent spread to implausible months or seasonal slopes.",
            "* Calendar flags catch implausible weekend premiums and week-to-week jumps inside a generated month.",
            "* This is a model-quality gate, not a market override: if quoted monthly/quarterly EEX products imply the inversion, the gate should not force a correction.",
            "* The current local/test curve should be considered seasonally suspect if any critical flag remains.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(
    csv_path: Path,
    forwards_path: Path,
    *,
    component_parquet: Path | None = None,
    market: str = "CH",
    neighbor_market: str = "DE",
    price_column: str = PRICE,
) -> dict[str, pd.DataFrame | pd.Timestamp]:
    hourly = _load_hourly_csv(csv_path, price_column=price_column)
    latest, forwards = _latest_forwards(forwards_path, market=market)
    base_forwards = forwards[forwards["load_type"] == "BASE"].copy()
    if base_forwards.empty:
        raise ValueError(f"no BASE forwards for market={market!r} in {forwards_path}")
    monthly = monthly_means(hourly, price_column=price_column)
    hour_month = hourly_month_matrix(hourly, price_column=price_column)
    residuals = quoted_product_residuals(hourly, forwards, price_column=price_column)
    seasonal = seasonal_coherence_checks(monthly, base_forwards)
    neighbor_forwards = pd.DataFrame()
    try:
        _, neighbor_forwards = _latest_forwards(forwards_path, market=neighbor_market)
        monthly_splits = monthly_split_checks(
            hourly,
            forwards,
            neighbor_forwards,
            price_column=price_column,
            neighbor_market=neighbor_market,
        )
    except ValueError:
        monthly_splits = pd.DataFrame()
    monthly_path = monthly_path_checks(
        monthly,
        hourly,
        forwards,
        neighbor_forwards if not neighbor_forwards.empty else None,
        neighbor_market=neighbor_market,
    )
    cross_year = cross_year_month_shape_checks(monthly, hourly, forwards)
    calendar_checks = calendar_coherence_checks(hourly, price_column=price_column)
    component_checks = None
    if component_parquet is not None:
        component_checks = component_jan_oct_checks(component_monthly_means(component_parquet))
    return {
        "latest_forward_date": latest,
        "quoted_residuals": residuals,
        "seasonal_checks": seasonal,
        "monthly_path_checks": monthly_path,
        "cross_year_month_shape_checks": cross_year,
        "monthly_split_checks": monthly_splits,
        "calendar_checks": calendar_checks,
        "monthly": monthly,
        "hour_month": hour_month,
        "component_checks": component_checks if component_checks is not None else pd.DataFrame(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--component-parquet", default=None)
    parser.add_argument("--market", default="CH")
    parser.add_argument("--neighbor-market", default="DE")
    parser.add_argument("--price-column", default=PRICE)
    parser.add_argument("--report", required=True)
    parser.add_argument("--monthly-output", default=None)
    parser.add_argument("--hour-month-output", default=None)
    parser.add_argument("--monthly-split-output", default=None)
    parser.add_argument("--monthly-path-output", default=None)
    parser.add_argument("--cross-year-output", default=None)
    parser.add_argument("--calendar-output", default=None)
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args(argv)

    result = audit(
        Path(args.csv),
        Path(args.forwards),
        component_parquet=Path(args.component_parquet) if args.component_parquet else None,
        market=args.market,
        neighbor_market=args.neighbor_market,
        price_column=args.price_column,
    )
    monthly = result["monthly"]
    hour_month = result["hour_month"]
    monthly_path = result["monthly_path_checks"]
    cross_year = result["cross_year_month_shape_checks"]
    monthly_splits = result["monthly_split_checks"]
    calendar_checks = result["calendar_checks"]
    if args.monthly_output:
        Path(args.monthly_output).parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(args.monthly_output, index=False)
    if args.hour_month_output:
        Path(args.hour_month_output).parent.mkdir(parents=True, exist_ok=True)
        hour_month.to_csv(args.hour_month_output, index=False)
    if args.monthly_split_output:
        Path(args.monthly_split_output).parent.mkdir(parents=True, exist_ok=True)
        monthly_splits.to_csv(args.monthly_split_output, index=False)
    if args.monthly_path_output:
        Path(args.monthly_path_output).parent.mkdir(parents=True, exist_ok=True)
        monthly_path.to_csv(args.monthly_path_output, index=False)
    if args.cross_year_output:
        Path(args.cross_year_output).parent.mkdir(parents=True, exist_ok=True)
        cross_year.to_csv(args.cross_year_output, index=False)
    if args.calendar_output:
        Path(args.calendar_output).parent.mkdir(parents=True, exist_ok=True)
        calendar_checks.to_csv(args.calendar_output, index=False)
    write_report(
        Path(args.report),
        csv_path=Path(args.csv),
        forwards_path=Path(args.forwards),
        latest_forward_date=result["latest_forward_date"],  # type: ignore[arg-type]
        neighbor_market=args.neighbor_market,
        quoted_residuals=result["quoted_residuals"],  # type: ignore[arg-type]
        seasonal_checks=result["seasonal_checks"],  # type: ignore[arg-type]
        monthly_split_checks_frame=monthly_splits,  # type: ignore[arg-type]
        calendar_checks=calendar_checks,  # type: ignore[arg-type]
        monthly_path_checks_frame=monthly_path,  # type: ignore[arg-type]
        cross_year_checks=cross_year,  # type: ignore[arg-type]
        monthly=monthly,  # type: ignore[arg-type]
        hour_month=hour_month,  # type: ignore[arg-type]
        component_checks=result["component_checks"],  # type: ignore[arg-type]
    )
    seasonal = result["seasonal_checks"]  # type: ignore[assignment]
    critical = sum(
        _severity_counts(frame)[0]
        for frame in [seasonal, monthly_splits, monthly_path, cross_year, calendar_checks]  # type: ignore[list-item]
    )
    warning = sum(
        _severity_counts(frame)[1]
        for frame in [seasonal, monthly_splits, monthly_path, cross_year, calendar_checks]  # type: ignore[list-item]
    )
    print(f"[seasonal-audit] critical={critical} warning={warning}")
    print(f"[seasonal-audit] report -> {args.report}")
    if args.fail_on_critical and critical:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
