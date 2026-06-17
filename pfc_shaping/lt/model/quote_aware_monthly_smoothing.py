"""Quote-aware completion smoothing for local/test LT HFC curves.

The smoother first solves monthly additive targets in the null-space of EEX
bucket constraints.  It then turns those monthly targets into a continuous
hourly profile and projects that profile back onto the exact BASE/PEAK EEX
constraints.  This keeps quoted Month/Quarter/Cal means unchanged while
avoiding artificial month-end steps in unquoted completions.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets


SCENARIO_PRICE_COLUMNS = {
    "slow": "price_slow_eur_mwh",
    "central": "price_central_eur_mwh",
    "fast": "price_fast_eur_mwh",
}

PRICE_COLUMNS = [
    "price_slow_eur_mwh",
    "price_central_eur_mwh",
    "price_fast_eur_mwh",
    "price_weighted_mean_eur_mwh",
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]


def apply_quote_aware_monthly_smoothing(
    hourly: pd.DataFrame,
    *,
    ts_ch: pd.Series,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float] | None,
    peak_mask: pd.Series | None,
    weights: Mapping[str, float],
    intensity: float = 1.0,
    smoothness_lambda: float = 8.0,
    max_unquoted_month_delta_eur_mwh: float = 18.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smooth implicit monthly completions without changing EEX bucket means.

    Deltas are solved per delivery year, not independently per EEX bucket, so
    the objective can smooth transitions across Quarter/Cal bucket boundaries.
    The solved monthly deltas are used as targets for a continuous hourly
    profile.  A final Euclidean projection enforces zero total energy in every
    quote-aware BASE bucket and, when PEAK products are provided, zero total
    PEAK energy in every PEAK bucket.  Directly quoted months are excluded from
    the free profile and remain pointwise unchanged.
    """

    if intensity < 0.0 or not np.isfinite(float(intensity)):
        raise ValueError("monthly smoothing intensity must be finite and non-negative")
    if smoothness_lambda < 0.0 or not np.isfinite(float(smoothness_lambda)):
        raise ValueError("monthly smoothing lambda must be finite and non-negative")
    if max_unquoted_month_delta_eur_mwh < 0.0 or not np.isfinite(float(max_unquoted_month_delta_eur_mwh)):
        raise ValueError("max unquoted monthly delta must be finite and non-negative")

    out = hourly.copy()
    if float(intensity) == 0.0 or float(smoothness_lambda) == 0.0 or float(max_unquoted_month_delta_eur_mwh) == 0.0:
        return out, pd.DataFrame()
    if len(ts_ch) != len(out):
        raise ValueError(f"monthly smoothing timestamp length mismatch: {len(ts_ch)} != {len(out)}")
    ts = pd.Series(np.asarray(ts_ch), index=out.index)
    if ts.isna().any():
        raise ValueError("monthly smoothing requires finite local timestamps")
    missing_cols = [col for col in SCENARIO_PRICE_COLUMNS.values() if col not in out.columns]
    if missing_cols:
        raise ValueError(f"monthly smoothing missing scenario columns: {missing_cols}")

    base_buckets, base_targets = calibration_buckets(ts, base_forward_prices)
    if base_buckets.isna().any():
        missing = int(base_buckets.isna().sum())
        raise ValueError(f"monthly smoothing found {missing} rows without an EEX BASE bucket")

    peak_buckets = pd.Series(dtype=object)
    if peak_forward_prices and peak_mask is not None:
        if len(peak_mask) != len(out):
            raise ValueError(f"monthly smoothing peak mask length mismatch: {len(peak_mask)} != {len(out)}")
        peak = pd.Series(np.asarray(peak_mask), index=out.index).astype(bool)
        if bool(peak.any()):
            peak_buckets, _ = calibration_buckets(ts.loc[peak], peak_forward_prices)
            if peak_buckets.isna().any():
                missing = int(peak_buckets.isna().sum())
                raise ValueError(f"monthly smoothing found {missing} EEX peak rows without a PEAK bucket")

    month_key = pd.Series(pd.PeriodIndex(ts.dt.strftime("%Y-%m"), freq="M"), index=out.index)
    weighted_before = out["price_weighted_mean_eur_mwh"].astype(float)
    audit_inputs: list[dict[str, object]] = []
    monthly_delta = pd.Series(0.0, index=pd.Index(sorted(month_key.unique()), name="month"), dtype=float)
    direct_month_quotes = {str(key) for key in base_forward_prices if _is_month_product(str(key))}
    if peak_forward_prices:
        direct_month_quotes.update(str(key) for key in peak_forward_prices if _is_month_product(str(key)))
    fixed_months_all = {month for month in monthly_delta.index if str(month) in direct_month_quotes}

    years = sorted(int(year) for year in ts.dt.year.dropna().unique())
    for year in years:
        year_idx = pd.Index(ts.index[ts.dt.year == year])
        months = pd.Index(sorted(month_key.loc[year_idx].unique()), name="month")
        if len(months) < 3:
            continue
        old_means = np.array(
            [float(weighted_before.loc[year_idx[month_key.loc[year_idx] == month]].mean()) for month in months]
        )
        if not np.isfinite(old_means).all():
            raise ValueError(f"monthly smoothing non-finite monthly means in year {year}")

        constraints: list[np.ndarray] = []
        base_bucket_by_month: dict[pd.Period, str] = {}
        for bucket, bucket_idx in base_buckets.loc[year_idx].groupby(base_buckets.loc[year_idx], dropna=True).groups.items():
            if bucket is None or str(bucket) not in base_targets:
                continue
            bucket_idx = pd.Index(bucket_idx)
            coeff = np.array([float((month_key.loc[bucket_idx] == month).sum()) for month in months], dtype=float)
            if coeff.sum() > 0.0:
                constraints.append(coeff)
            for month in months[coeff > 0.0]:
                base_bucket_by_month[month] = str(bucket)

        if not peak_buckets.empty:
            peak_in_year = year_idx.intersection(peak_buckets.index)
            if len(peak_in_year) > 0:
                for _, peak_idx in peak_buckets.loc[peak_in_year].groupby(peak_buckets.loc[peak_in_year]).groups.items():
                    peak_idx = pd.Index(peak_idx)
                    coeff = np.array([float((month_key.loc[peak_idx] == month).sum()) for month in months], dtype=float)
                    if coeff.sum() > 0.0:
                        constraints.append(coeff)

        delta = _solve_bucket_delta(
            old_means,
            constraints=np.vstack(constraints),
            smoothness_lambda=float(smoothness_lambda),
        )
        delta = delta * float(intensity)
        max_abs = float(np.max(np.abs(delta))) if len(delta) else 0.0
        if max_abs > float(max_unquoted_month_delta_eur_mwh) > 0.0:
            delta = delta * (float(max_unquoted_month_delta_eur_mwh) / max_abs)
        elif float(max_unquoted_month_delta_eur_mwh) == 0.0:
            delta = delta * 0.0

        constraint_matrix = np.vstack(constraints)
        residual = constraint_matrix @ delta
        max_residual = float(np.max(np.abs(residual)))
        max_peak_residual = float(np.max(np.abs(residual[1:]))) if len(residual) > 1 else 0.0
        if max_residual > 1e-7:
            raise ValueError(f"monthly smoothing constraint residual too large for {year}: {max_residual}")

        for pos, month in enumerate(months):
            monthly_delta.loc[month] += float(delta[pos])

        fixed_months = {month for month in months if str(month) in direct_month_quotes}
        peak_in_year = year_idx.intersection(peak_buckets.index) if not peak_buckets.empty else pd.Index([])
        audit_inputs.append(
            {
                "year": year,
                "year_idx": year_idx,
                "months": months,
                "old_means": old_means,
                "delta": delta,
                "base_bucket_by_month": base_bucket_by_month,
                "base_constraint_residual_mwh": float(residual[0]),
                "max_peak_constraint_residual_mwh": max_peak_residual,
                "max_constraint_residual_mwh": max_residual,
                "peak_in_year": peak_in_year,
                "fixed_months": fixed_months,
            }
        )

    if float(np.abs(monthly_delta).sum()) <= 1e-12:
        row_delta = pd.Series(0.0, index=out.index, dtype=float)
    else:
        step_delta = pd.Series([float(monthly_delta.loc[month]) for month in month_key], index=out.index, dtype=float)
        row_delta = _continuous_projected_row_delta(
            ts=ts,
            month_key=month_key,
            monthly_delta=monthly_delta,
            step_delta=step_delta,
            fixed_months=fixed_months_all,
            base_buckets=base_buckets,
            peak_buckets=peak_buckets if not peak_buckets.empty else None,
        )

    audit_rows: list[dict[str, object]] = []
    for info in audit_inputs:
        year = int(info["year"])
        year_idx = pd.Index(info["year_idx"])
        months = pd.Index(info["months"])
        old_means = np.asarray(info["old_means"], dtype=float)
        delta = np.asarray(info["delta"], dtype=float)
        peak_in_year = pd.Index(info["peak_in_year"])
        year_row_delta = row_delta.loc[year_idx]
        actual_month_delta = year_row_delta.groupby(month_key.loc[year_idx]).mean()
        base_projected_residuals = _bucket_sum_residuals(year_row_delta, base_buckets.loc[year_idx])
        peak_projected_residuals = (
            _bucket_sum_residuals(year_row_delta.loc[peak_in_year], peak_buckets.loc[peak_in_year])
            if len(peak_in_year)
            else np.array([], dtype=float)
        )
        max_projected_base_residual = (
            float(np.max(np.abs(base_projected_residuals))) if len(base_projected_residuals) else 0.0
        )
        max_projected_peak_residual = (
            float(np.max(np.abs(peak_projected_residuals))) if len(peak_projected_residuals) else 0.0
        )
        max_projected_residual = max(max_projected_base_residual, max_projected_peak_residual)
        if max_projected_residual > 1e-7:
            raise ValueError(f"monthly smoothing projected residual too large for {year}: {max_projected_residual}")

        base_bucket_by_month = dict(info["base_bucket_by_month"])
        for pos, month in enumerate(months):
            applied_delta = float(actual_month_delta.get(month, 0.0))
            audit_rows.append(
                {
                    "bucket": base_bucket_by_month.get(month, ""),
                    "year": int(month.year),
                    "month": int(month.month),
                    "is_direct_month_quote": str(month) in direct_month_quotes,
                    "mean_before_eur_mwh": float(old_means[pos]),
                    "target_delta_eur_mwh": float(delta[pos]),
                    "delta_eur_mwh": applied_delta,
                    "mean_after_eur_mwh": float(old_means[pos] + applied_delta),
                    "base_constraint_residual_mwh": float(info["base_constraint_residual_mwh"]),
                    "max_peak_constraint_residual_mwh": float(info["max_peak_constraint_residual_mwh"]),
                    "max_constraint_residual_mwh": float(info["max_constraint_residual_mwh"]),
                    "max_projected_base_residual_mwh": max_projected_base_residual,
                    "max_projected_peak_residual_mwh": max_projected_peak_residual,
                    "max_projected_constraint_residual_mwh": max_projected_residual,
                }
            )

    for col in SCENARIO_PRICE_COLUMNS.values():
        out[col] = out[col].astype(float) + row_delta
    out = _recompute_weighted_fan_columns(out, weights)

    min_price = float(out[[*SCENARIO_PRICE_COLUMNS.values(), "price_weighted_mean_eur_mwh"]].min().min())
    if min_price < float(negative_price_floor) - 1e-9:
        raise ValueError(
            f"monthly smoothing breached negative price floor: min={min_price:.6f}, "
            f"floor={float(negative_price_floor):.6f}"
        )
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "monthly smoothing produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )

    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(["year", "month", "bucket"]).reset_index(drop=True)
    return out, audit


def _solve_bucket_delta(
    monthly_means: np.ndarray,
    *,
    constraints: np.ndarray,
    smoothness_lambda: float,
) -> np.ndarray:
    n = int(len(monthly_means))
    if n < 3 or smoothness_lambda == 0.0:
        return np.zeros(n, dtype=float)
    d2 = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0

    penalty = d2.T @ d2
    hessian = np.eye(n, dtype=float) + float(smoothness_lambda) * penalty
    rhs = -float(smoothness_lambda) * penalty @ monthly_means.astype(float)

    a = _independent_constraint_rows(np.asarray(constraints, dtype=float))
    if a.size == 0:
        return np.linalg.solve(hessian, rhs)
    kkt = np.block(
        [
            [hessian, a.T],
            [a, np.zeros((a.shape[0], a.shape[0]), dtype=float)],
        ]
    )
    kkt_rhs = np.concatenate([rhs, np.zeros(a.shape[0], dtype=float)])
    try:
        solution = np.linalg.solve(kkt, kkt_rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(kkt, kkt_rhs, rcond=None)[0]
    return solution[:n]


def _continuous_projected_row_delta(
    *,
    ts: pd.Series,
    month_key: pd.Series,
    monthly_delta: pd.Series,
    step_delta: pd.Series,
    fixed_months: set[pd.Period],
    base_buckets: pd.Series,
    peak_buckets: pd.Series | None,
) -> pd.Series:
    """Build a smooth hourly correction and enforce exact bucket sums."""

    if float(np.abs(monthly_delta).sum()) <= 1e-12:
        return pd.Series(0.0, index=ts.index, dtype=float)
    raw = _smoothstep_monthly_profile(ts=ts, month_key=month_key, monthly_delta=monthly_delta)
    fixed_mask = month_key.isin(fixed_months).to_numpy(dtype=bool)
    raw.loc[fixed_mask] = 0.0
    free_index = raw.index[~fixed_mask]
    if len(free_index) == 0:
        if float(np.abs(step_delta).sum()) > 1e-8:
            raise ValueError("monthly smoothing has no free rows for a non-zero correction")
        return pd.Series(0.0, index=ts.index, dtype=float)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    free_pos = pd.Series(np.arange(len(free_index), dtype=int), index=free_index)

    def add_bucket_constraints(buckets: pd.Series) -> None:
        for _, bucket_idx in buckets.groupby(buckets, dropna=True).groups.items():
            bucket_idx = pd.Index(bucket_idx)
            target = float(step_delta.loc[bucket_idx.intersection(step_delta.index)].sum())
            bucket_free = bucket_idx.intersection(free_index)
            if len(bucket_free) == 0:
                if abs(target) > 1e-8:
                    raise ValueError("monthly smoothing fixed rows cannot satisfy a non-zero bucket target")
                continue
            row = np.zeros(len(free_index), dtype=float)
            row[free_pos.loc[bucket_free].to_numpy(dtype=int)] = 1.0
            rows.append(row)
            rhs.append(target)

    add_bucket_constraints(base_buckets)
    if peak_buckets is not None and not peak_buckets.empty:
        add_bucket_constraints(peak_buckets)

    projected = raw.copy()
    if rows:
        a, b = _independent_constraint_rows_with_rhs(np.vstack(rows), np.asarray(rhs, dtype=float))
        free_values = raw.loc[free_index].to_numpy(dtype=float)
        projected.loc[free_index] = _project_values_to_constraints(free_values, a, b)
    projected.loc[fixed_mask] = 0.0
    return projected.astype(float)


def _smoothstep_monthly_profile(
    *,
    ts: pd.Series,
    month_key: pd.Series,
    monthly_delta: pd.Series,
) -> pd.Series:
    centers: list[float] = []
    values: list[float] = []
    for month in monthly_delta.index:
        idx = month_key.index[month_key == month]
        if len(idx) == 0:
            continue
        ns = pd.Series(ts.loc[idx]).astype("int64").to_numpy(dtype=float)
        centers.append(float(ns.mean()))
        values.append(float(monthly_delta.loc[month]))
    if not centers:
        return pd.Series(0.0, index=ts.index, dtype=float)
    if len(centers) == 1:
        return pd.Series(values[0], index=ts.index, dtype=float)

    x = np.asarray(centers, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    t = pd.Series(ts).astype("int64").to_numpy(dtype=float)
    out = np.empty(len(t), dtype=float)
    out[t <= x[0]] = y[0]
    out[t >= x[-1]] = y[-1]
    middle = (t > x[0]) & (t < x[-1])
    interval = np.searchsorted(x, t[middle], side="right") - 1
    interval = np.clip(interval, 0, len(x) - 2)
    left = x[interval]
    right = x[interval + 1]
    u = (t[middle] - left) / (right - left)
    smooth = u * u * (3.0 - 2.0 * u)
    out[middle] = y[interval] * (1.0 - smooth) + y[interval + 1] * smooth
    return pd.Series(out, index=ts.index, dtype=float)


def _project_values_to_constraints(values: np.ndarray, constraints: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if constraints.size == 0:
        return values
    residual = constraints @ values - rhs
    gram = constraints @ constraints.T
    try:
        multipliers = np.linalg.solve(gram, residual)
    except np.linalg.LinAlgError:
        multipliers = np.linalg.lstsq(gram, residual, rcond=None)[0]
    return values - constraints.T @ multipliers


def _independent_constraint_rows_with_rhs(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    values: list[float] = []
    current = np.empty((0, matrix.shape[1]), dtype=float)
    rank = 0
    for row, value in zip(matrix, rhs, strict=True):
        if not np.isfinite(row).all() or float(np.linalg.norm(row)) <= tol:
            continue
        candidate = np.vstack([current, row])
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=tol))
        if candidate_rank > rank:
            rows.append(row)
            values.append(float(value))
            current = candidate
            rank = candidate_rank
    if not rows:
        return np.empty((0, matrix.shape[1]), dtype=float), np.empty(0, dtype=float)
    return np.vstack(rows), np.asarray(values, dtype=float)


def _bucket_sum_residuals(row_delta: pd.Series, buckets: pd.Series) -> np.ndarray:
    residuals: list[float] = []
    for _, bucket_idx in buckets.groupby(buckets, dropna=True).groups.items():
        idx = pd.Index(bucket_idx).intersection(row_delta.index)
        if len(idx):
            residuals.append(float(row_delta.loc[idx].sum()))
    return np.asarray(residuals, dtype=float)


def _is_month_product(product: str) -> bool:
    try:
        pd.Period(product, freq="M")
    except ValueError:
        return False
    return len(product) == 7 and product[4] == "-"


def _independent_constraint_rows(matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    rows: list[np.ndarray] = []
    current = np.empty((0, matrix.shape[1]), dtype=float)
    rank = 0
    for row in matrix:
        if not np.isfinite(row).all() or float(np.linalg.norm(row)) <= tol:
            continue
        candidate = np.vstack([current, row])
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=tol))
        if candidate_rank > rank:
            rows.append(row)
            current = candidate
            rank = candidate_rank
    if not rows:
        return np.empty((0, matrix.shape[1]), dtype=float)
    return np.vstack(rows)


def _recompute_weighted_fan_columns(out: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    labels = tuple(SCENARIO_PRICE_COLUMNS)
    matrix = np.column_stack([out[SCENARIO_PRICE_COLUMNS[label]].to_numpy(dtype=float) for label in labels])
    w = np.array([float(weights[label]) for label in labels], dtype=float)
    out["price_weighted_mean_eur_mwh"] = matrix @ w
    out["structural_p10_eur_mwh"] = [_weighted_quantile_row(row, w, 0.10) for row in matrix]
    out["structural_p50_eur_mwh"] = [_weighted_quantile_row(row, w, 0.50) for row in matrix]
    out["structural_p90_eur_mwh"] = [_weighted_quantile_row(row, w, 0.90) for row in matrix]
    out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
    return out


def _weighted_quantile_row(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    xs = values[order]
    ws = weights[order]
    cumulative = np.cumsum(ws) / float(np.sum(ws))
    return float(xs[np.searchsorted(cumulative, q, side="left")])
