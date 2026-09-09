"""Cross-year seasonal shaping under hard EEX constraints.

The optimizer works on monthly additive deltas for synthetic BASE buckets
such as ``YYYY`` and ``YYYY-RESIDUAL``.  It preserves active CH BASE/PEAK
bucket means exactly while penalizing implausible cross-year month spreads.
Neighbor markets enter only through recentered shape guides, never through
absolute levels.
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


def apply_cross_year_seasonal_shape_optimizer(
    hourly: pd.DataFrame,
    *,
    ts_ch: pd.Series,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float] | None,
    peak_mask: pd.Series | None,
    guide_month_prices: Mapping[str, float] | None,
    weights: Mapping[str, float],
    intensity: float = 1.0,
    lambda_prior: float = 1.0,
    lambda_month_smooth: float = 1.0,
    lambda_guide: float = 2.0,
    lambda_cross_month: float = 8.0,
    lambda_cross_slope: float = 20.0,
    max_month_delta_eur_mwh: float = 12.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile synthetic monthly shapes across adjacent years.

    The hard constraints are monthly-delta sums per BASE and PEAK bucket.
    Directly quoted months and quarters are not free variables; the solver only
    adjusts annual and annual-residual synthetic buckets.
    """

    for name, value in {
        "intensity": intensity,
        "lambda_prior": lambda_prior,
        "lambda_month_smooth": lambda_month_smooth,
        "lambda_guide": lambda_guide,
        "lambda_cross_month": lambda_cross_month,
        "lambda_cross_slope": lambda_cross_slope,
        "max_month_delta_eur_mwh": max_month_delta_eur_mwh,
    }.items():
        if float(value) < 0.0 or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite and non-negative")

    out = hourly.copy()
    if float(intensity) == 0.0:
        return out, pd.DataFrame()
    if len(ts_ch) != len(out):
        raise ValueError(f"cross-year seasonal timestamp length mismatch: {len(ts_ch)} != {len(out)}")
    missing_cols = [col for col in SCENARIO_PRICE_COLUMNS.values() if col not in out.columns]
    if missing_cols:
        raise ValueError(f"cross-year seasonal optimizer missing scenario columns: {missing_cols}")

    ts = pd.Series(np.asarray(ts_ch), index=out.index)
    if ts.isna().any():
        raise ValueError("cross-year seasonal optimizer requires finite local timestamps")
    base_buckets, base_targets = calibration_buckets(ts, base_forward_prices)
    if base_buckets.isna().any():
        missing = int(base_buckets.isna().sum())
        raise ValueError(f"cross-year seasonal optimizer found {missing} rows without an EEX BASE bucket")

    peak_buckets = pd.Series(dtype=object)
    peak = pd.Series(False, index=out.index)
    if peak_forward_prices and peak_mask is not None:
        if len(peak_mask) != len(out):
            raise ValueError(f"cross-year seasonal peak mask length mismatch: {len(peak_mask)} != {len(out)}")
        peak = pd.Series(np.asarray(peak_mask), index=out.index).astype(bool)
        if bool(peak.any()):
            peak_buckets, _ = calibration_buckets(ts.loc[peak], peak_forward_prices)
            if peak_buckets.isna().any():
                missing = int(peak_buckets.isna().sum())
                raise ValueError(f"cross-year seasonal optimizer found {missing} PEAK rows without a PEAK bucket")

    month_key = pd.Series(pd.PeriodIndex(ts.dt.strftime("%Y-%m"), freq="M"), index=out.index)
    current_monthly = out.groupby(month_key)["price_weighted_mean_eur_mwh"].mean().astype(float)
    base_month = _bucket_by_month(month_key, base_buckets)
    direct_month_quotes = {pd.Period(str(key), freq="M") for key in base_forward_prices if _is_month_product(str(key))}
    if peak_forward_prices:
        direct_month_quotes.update(pd.Period(str(key), freq="M") for key in peak_forward_prices if _is_month_product(str(key)))

    free_months = [
        month
        for month in current_monthly.index
        if month not in direct_month_quotes and _is_synthetic_bucket(str(base_month.get(month, "")))
    ]
    if len(free_months) < 2:
        return out, pd.DataFrame()
    free_pos = {month: pos for pos, month in enumerate(free_months)}
    n = len(free_months)

    objective_rows: list[np.ndarray] = []
    objective_rhs: list[float] = []

    def add_objective(coeff: Mapping[pd.Period, float], rhs: float, weight: float) -> None:
        if weight <= 0.0:
            return
        row = np.zeros(n, dtype=float)
        for month, value in coeff.items():
            pos = free_pos.get(month)
            if pos is not None:
                row[pos] += float(value)
        if not np.any(np.abs(row) > 0.0):
            return
        scale = float(np.sqrt(weight))
        objective_rows.append(row * scale)
        objective_rhs.append(float(rhs) * scale)

    for month in free_months:
        add_objective({month: 1.0}, 0.0, float(lambda_prior))

    all_months = list(current_monthly.index.sort_values())
    for left, mid, right in zip(all_months, all_months[1:], all_months[2:]):
        current_curvature = (
            float(current_monthly.loc[left])
            - 2.0 * float(current_monthly.loc[mid])
            + float(current_monthly.loc[right])
        )
        add_objective(
            {left: 1.0, mid: -2.0, right: 1.0},
            -current_curvature,
            float(lambda_month_smooth),
        )

    guide_targets = _recentered_guide_targets(
        months=free_months,
        month_key=month_key,
        base_buckets=base_buckets,
        base_targets=base_targets,
        guide_month_prices=guide_month_prices or {},
    )
    for month, target in guide_targets.items():
        add_objective(
            {month: 1.0},
            float(target) - float(current_monthly.loc[month]),
            float(lambda_guide),
        )

    for from_year, to_year in _adjacent_years(free_months):
        for month_number in range(1, 13):
            left = pd.Period(f"{from_year}-{month_number:02d}", freq="M")
            right = pd.Period(f"{to_year}-{month_number:02d}", freq="M")
            if left not in free_pos or right not in free_pos:
                continue
            left_bucket = str(base_month.get(left, ""))
            right_bucket = str(base_month.get(right, ""))
            if left_bucket not in base_targets or right_bucket not in base_targets:
                continue
            parent_spread = float(base_targets[right_bucket]) - float(base_targets[left_bucket])
            current_spread = float(current_monthly.loc[right]) - float(current_monthly.loc[left])
            add_objective(
                {right: 1.0, left: -1.0},
                parent_spread - current_spread,
                float(lambda_cross_month),
            )

        left_apr = pd.Period(f"{from_year}-04", freq="M")
        left_dec = pd.Period(f"{from_year}-12", freq="M")
        right_apr = pd.Period(f"{to_year}-04", freq="M")
        right_dec = pd.Period(f"{to_year}-12", freq="M")
        if all(month in free_pos for month in [left_apr, left_dec, right_apr, right_dec]):
            current_slope_delta = (
                float(current_monthly.loc[right_dec] - current_monthly.loc[right_apr])
                - float(current_monthly.loc[left_dec] - current_monthly.loc[left_apr])
            )
            add_objective(
                {right_dec: 1.0, right_apr: -1.0, left_dec: -1.0, left_apr: 1.0},
                -current_slope_delta,
                float(lambda_cross_slope),
            )

    if not objective_rows:
        return out, pd.DataFrame()

    constraints = _delta_constraint_matrix(
        free_months=free_months,
        month_key=month_key,
        base_buckets=base_buckets,
        peak_buckets=peak_buckets if not peak_buckets.empty else None,
        peak_index=peak.index[peak.to_numpy(dtype=bool)],
    )
    delta = _solve_equality_constrained_ls(
        np.vstack(objective_rows),
        np.asarray(objective_rhs, dtype=float),
        constraints,
    )
    delta = delta * float(intensity)
    max_abs = float(np.max(np.abs(delta))) if len(delta) else 0.0
    if max_abs > float(max_month_delta_eur_mwh) > 0.0:
        delta = delta * (float(max_month_delta_eur_mwh) / max_abs)
    elif float(max_month_delta_eur_mwh) == 0.0:
        delta = delta * 0.0

    month_delta = pd.Series(delta, index=pd.Index(free_months, name="month"), dtype=float)
    row_delta = pd.Series([float(month_delta.get(month, 0.0)) for month in month_key], index=out.index, dtype=float)
    row_delta = _line_search_negative_floor(
        out,
        row_delta,
        negative_price_floor=float(negative_price_floor),
        max_weighted_negative_hours=int(max_weighted_negative_hours),
    )
    if float(np.abs(row_delta).sum()) <= 1e-12:
        return out, pd.DataFrame()

    for col in SCENARIO_PRICE_COLUMNS.values():
        out[col] = out[col].astype(float) + row_delta
    out = _recompute_weighted_fan_columns(out, weights)
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)

    actual_delta = row_delta.groupby(month_key).mean()
    base_residual = _bucket_sum_residuals(row_delta, base_buckets)
    peak_residual = (
        _bucket_sum_residuals(row_delta.loc[peak.index[peak.to_numpy(dtype=bool)]], peak_buckets)
        if not peak_buckets.empty
        else np.array([], dtype=float)
    )
    max_constraint_residual = max(
        float(np.max(np.abs(base_residual))) if len(base_residual) else 0.0,
        float(np.max(np.abs(peak_residual))) if len(peak_residual) else 0.0,
    )
    if max_constraint_residual > 1e-6:
        raise ValueError(f"cross-year seasonal optimizer constraint residual too large: {max_constraint_residual}")

    after_monthly = out.groupby(month_key)["price_weighted_mean_eur_mwh"].mean().astype(float)
    audit_rows = []
    for month in free_months:
        audit_rows.append(
            {
                "year": int(month.year),
                "month": int(month.month),
                "bucket": str(base_month.get(month, "")),
                "mean_before_eur_mwh": float(current_monthly.loc[month]),
                "guide_target_eur_mwh": float(guide_targets.get(month, np.nan)),
                "delta_eur_mwh": float(actual_delta.get(month, 0.0)),
                "mean_after_eur_mwh": float(after_monthly.loc[month]),
                "max_constraint_residual_eur_mwh": max_constraint_residual,
            }
        )
    return out, pd.DataFrame(audit_rows).sort_values(["year", "month"]).reset_index(drop=True)


def _bucket_by_month(month_key: pd.Series, buckets: pd.Series) -> dict[pd.Period, str]:
    frame = pd.DataFrame({"month": month_key, "bucket": buckets})
    grouped = frame.groupby("month")["bucket"].agg(lambda s: str(s.mode().iloc[0]) if not s.mode().empty else "")
    return {pd.Period(month, freq="M"): str(bucket) for month, bucket in grouped.items()}


def _is_synthetic_bucket(bucket: str) -> bool:
    return bucket.endswith("-RESIDUAL") or bucket.isdigit()


def _is_month_product(product: str) -> bool:
    try:
        pd.Period(product, freq="M")
    except ValueError:
        return False
    return len(product) == 7 and product[4] == "-"


def _adjacent_years(months: list[pd.Period]) -> list[tuple[int, int]]:
    years = sorted({int(month.year) for month in months})
    return [(left, right) for left, right in zip(years, years[1:]) if right == left + 1]


def _recentered_guide_targets(
    *,
    months: list[pd.Period],
    month_key: pd.Series,
    base_buckets: pd.Series,
    base_targets: Mapping[str, float],
    guide_month_prices: Mapping[str, float],
) -> dict[pd.Period, float]:
    raw = {pd.Period(str(key), freq="M"): float(value) for key, value in guide_month_prices.items()}
    free = set(months)
    out: dict[pd.Period, float] = {}
    frame = pd.DataFrame({"month": month_key, "bucket": base_buckets})
    counts = frame.groupby(["bucket", "month"]).size()
    for bucket, group in frame.groupby("bucket"):
        bucket = str(bucket)
        if bucket not in base_targets:
            continue
        bucket_months = [pd.Period(month, freq="M") for month in group["month"].unique()]
        candidate = [month for month in bucket_months if month in free and month in raw]
        if not candidate:
            continue
        total = float(sum(int(counts.loc[(bucket, month)]) for month in candidate))
        if total <= 0.0:
            continue
        raw_mean = sum(float(raw[month]) * float(counts.loc[(bucket, month)]) for month in candidate) / total
        shift = float(base_targets[bucket]) - raw_mean
        for month in candidate:
            out[month] = float(raw[month]) + shift
    return out


def _delta_constraint_matrix(
    *,
    free_months: list[pd.Period],
    month_key: pd.Series,
    base_buckets: pd.Series,
    peak_buckets: pd.Series | None,
    peak_index: pd.Index,
) -> np.ndarray:
    free_pos = {month: pos for pos, month in enumerate(free_months)}
    rows: list[np.ndarray] = []

    def add_rows(buckets: pd.Series, months: pd.Series) -> None:
        for _, idx in buckets.groupby(buckets, dropna=True).groups.items():
            idx = pd.Index(idx)
            row = np.zeros(len(free_months), dtype=float)
            for month, count in months.loc[idx].value_counts().items():
                pos = free_pos.get(pd.Period(month, freq="M"))
                if pos is not None:
                    row[pos] += float(count)
            if np.any(np.abs(row) > 0.0):
                rows.append(row)

    add_rows(base_buckets, month_key)
    if peak_buckets is not None and not peak_buckets.empty and len(peak_index) > 0:
        add_rows(peak_buckets, month_key.loc[peak_index])
    if not rows:
        return np.empty((0, len(free_months)), dtype=float)
    return _independent_rows(np.vstack(rows))


def _solve_equality_constrained_ls(objective: np.ndarray, rhs: np.ndarray, constraints: np.ndarray) -> np.ndarray:
    hessian = objective.T @ objective + 1e-9 * np.eye(objective.shape[1], dtype=float)
    linear = objective.T @ rhs
    if constraints.size == 0:
        return np.linalg.solve(hessian, linear)
    a = _independent_rows(constraints)
    kkt = np.block([[hessian, a.T], [a, np.zeros((a.shape[0], a.shape[0]), dtype=float)]])
    kkt_rhs = np.concatenate([linear, np.zeros(a.shape[0], dtype=float)])
    try:
        solution = np.linalg.solve(kkt, kkt_rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(kkt, kkt_rhs, rcond=None)[0]
    return solution[: objective.shape[1]]


def _independent_rows(matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
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


def _line_search_negative_floor(
    hourly: pd.DataFrame,
    row_delta: pd.Series,
    *,
    negative_price_floor: float,
    max_weighted_negative_hours: int,
) -> pd.Series:
    scale = 1.0
    scenario_cols = list(SCENARIO_PRICE_COLUMNS.values())
    while scale >= 1e-6:
        candidate = row_delta * scale
        values = hourly[scenario_cols].astype(float).add(candidate, axis=0)
        weighted = hourly["price_weighted_mean_eur_mwh"].astype(float) + candidate
        min_price = float(pd.concat([values, weighted.rename("price_weighted_mean_eur_mwh")], axis=1).min().min())
        weighted_negative_hours = int((weighted < 0.0).sum())
        if min_price >= float(negative_price_floor) - 1e-9 and weighted_negative_hours <= int(max_weighted_negative_hours):
            return candidate
        scale *= 0.5
    raise ValueError("cross-year seasonal optimizer cannot satisfy negative floor with bounded line search")


def _bucket_sum_residuals(row_delta: pd.Series, buckets: pd.Series) -> np.ndarray:
    residuals: list[float] = []
    for _, idx in buckets.groupby(buckets, dropna=True).groups.items():
        idx = pd.Index(idx).intersection(row_delta.index)
        if len(idx):
            residuals.append(float(row_delta.loc[idx].sum()))
    return np.asarray(residuals, dtype=float)


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
