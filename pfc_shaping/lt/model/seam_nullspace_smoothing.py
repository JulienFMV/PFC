"""Local seam smoothing in the nullspace of EEX average constraints."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets
from pfc_shaping.lt.model.shape_constraints import eex_peak_mask, interval_hours, validate_utc_index
from pfc_shaping.lt.model.structural_scenario_bracket import (
    PRICE_COLUMNS,
    SCENARIO_PRICE_COLUMNS,
    recompute_structural_scenario_bracket,
)


def apply_seam_nullspace_smoothing(
    hourly: pd.DataFrame,
    *,
    ts_ch: pd.Series,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float] | None,
    weights: Mapping[str, float],
    seam_window_hours: int = 168,
    target_boundary_jump_eur_mwh: float = 18.0,
    min_boundary_jump_eur_mwh: float = 20.0,
    max_abs_delta_eur_mwh: float = 12.0,
    intensity: float = 1.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce month-boundary cliffs without changing EEX BASE/PEAK means.

    The smoother builds a bounded additive correction from local seam ramps,
    then projects that correction into the nullspace of all active BASE and
    PEAK average constraints.  The same correction is applied to all scenario
    price columns so fan spreads and quantile ordering are preserved.
    """

    if seam_window_hours < 2:
        raise ValueError("seam_window_hours must be at least 2")
    for name, value in {
        "target_boundary_jump_eur_mwh": target_boundary_jump_eur_mwh,
        "min_boundary_jump_eur_mwh": min_boundary_jump_eur_mwh,
        "max_abs_delta_eur_mwh": max_abs_delta_eur_mwh,
        "intensity": intensity,
    }.items():
        if float(value) < 0.0 or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite and non-negative")

    out = hourly.copy()
    if (
        float(intensity) == 0.0
        or float(max_abs_delta_eur_mwh) == 0.0
        or float(target_boundary_jump_eur_mwh) >= float(min_boundary_jump_eur_mwh)
    ):
        return out, pd.DataFrame()
    missing_cols = [col for col in SCENARIO_PRICE_COLUMNS.values() if col not in out.columns]
    if missing_cols:
        raise ValueError(f"seam smoothing missing scenario columns: {missing_cols}")
    if "price_weighted_mean_eur_mwh" not in out.columns:
        raise ValueError("seam smoothing requires price_weighted_mean_eur_mwh")
    if len(ts_ch) != len(out):
        raise ValueError(f"seam smoothing timestamp length mismatch: {len(ts_ch)} != {len(out)}")

    ts = pd.Series(np.asarray(ts_ch), index=out.index)
    if ts.isna().any():
        raise ValueError("seam smoothing requires finite local timestamps")
    index_utc = pd.DatetimeIndex(ts).tz_convert("UTC")
    validate_utc_index(index_utc)

    price = out["price_weighted_mean_eur_mwh"].astype(float).to_numpy(dtype=float)
    raw_delta, seam_rows = _raw_seam_delta(
        price,
        pd.DatetimeIndex(ts),
        seam_window_hours=int(seam_window_hours),
        target_boundary_jump=float(target_boundary_jump_eur_mwh),
        min_boundary_jump=float(min_boundary_jump_eur_mwh),
        intensity=float(intensity),
    )
    if not seam_rows:
        return out, pd.DataFrame()

    constraints = _constraint_matrix(
        index_utc,
        ts,
        base_forward_prices=base_forward_prices,
        peak_forward_prices=peak_forward_prices or {},
    )
    projected_delta = _project_to_zero_constraints(raw_delta, constraints)
    max_projected = float(np.max(np.abs(projected_delta))) if len(projected_delta) else 0.0
    if max_projected > float(max_abs_delta_eur_mwh) > 0.0:
        projected_delta = projected_delta * (float(max_abs_delta_eur_mwh) / max_projected)

    out_before = out.copy()
    for column in SCENARIO_PRICE_COLUMNS.values():
        out[column] = out[column].astype(float).to_numpy(dtype=float) + projected_delta
    out = _recompute_weighted_fan_columns(out, weights)

    min_price = float(out[[*SCENARIO_PRICE_COLUMNS.values(), "price_weighted_mean_eur_mwh"]].min().min())
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if min_price < float(negative_price_floor) - 1e-9 or weighted_negative_hours > int(max_weighted_negative_hours):
        scale = _max_feasible_scale(
            out_before,
            projected_delta,
            weights=weights,
            negative_price_floor=float(negative_price_floor),
            max_weighted_negative_hours=int(max_weighted_negative_hours),
        )
        projected_delta = projected_delta * scale
        out = out_before.copy()
        for column in SCENARIO_PRICE_COLUMNS.values():
            out[column] = out[column].astype(float).to_numpy(dtype=float) + projected_delta
        out = _recompute_weighted_fan_columns(out, weights)

    min_price = float(out[[*SCENARIO_PRICE_COLUMNS.values(), "price_weighted_mean_eur_mwh"]].min().min())
    if min_price < float(negative_price_floor) - 1e-9:
        raise ValueError(
            f"seam smoothing breached negative price floor: min={min_price:.6f}, "
            f"floor={float(negative_price_floor):.6f}"
        )
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "seam smoothing produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )

    before = out_before["price_weighted_mean_eur_mwh"].astype(float).to_numpy(dtype=float)
    after = out["price_weighted_mean_eur_mwh"].astype(float).to_numpy(dtype=float)
    audit_rows = []
    for row in seam_rows:
        i = int(row["row"])
        before_jump = float(before[i] - before[i - 1])
        after_jump = float(after[i] - after[i - 1])
        audit_rows.append(
            {
                **row,
                "jump_before_eur_mwh": before_jump,
                "jump_after_eur_mwh": after_jump,
                "abs_jump_before_eur_mwh": abs(before_jump),
                "abs_jump_after_eur_mwh": abs(after_jump),
                "reduction_eur_mwh": abs(before_jump) - abs(after_jump),
                "delta_left_eur_mwh": float(projected_delta[i - 1]),
                "delta_right_eur_mwh": float(projected_delta[i]),
                "max_abs_delta_eur_mwh": float(np.max(np.abs(projected_delta))) if len(projected_delta) else 0.0,
                "max_constraint_residual_eur_mwh": (
                    float(np.max(np.abs(constraints @ projected_delta))) if constraints.size else 0.0
                ),
            }
        )

    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values("abs_jump_before_eur_mwh", ascending=False).reset_index(drop=True)
    return out, audit


def _raw_seam_delta(
    price: np.ndarray,
    ts: pd.DatetimeIndex,
    *,
    seam_window_hours: int,
    target_boundary_jump: float,
    min_boundary_jump: float,
    intensity: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    delta = np.zeros(len(price), dtype=float)
    rows: list[dict[str, object]] = []
    half = int(seam_window_hours)
    for i in range(1, len(price)):
        t = ts[i]
        if int(t.day) != 1 or int(t.hour) != 0:
            continue
        jump = float(price[i] - price[i - 1])
        abs_jump = abs(jump)
        if abs_jump <= float(min_boundary_jump):
            continue
        desired_jump = float(np.sign(jump) * target_boundary_jump)
        reduction = (jump - desired_jump) * float(intensity)
        if abs(reduction) <= 1e-12:
            continue
        left_start = max(0, i - half)
        left_stop = i
        right_start = i
        right_stop = min(len(price), i + half)
        if left_stop <= left_start or right_stop <= right_start:
            continue
        left_u = np.linspace(0.0, 1.0, left_stop - left_start, endpoint=True)
        right_u = np.linspace(1.0, 0.0, right_stop - right_start, endpoint=True)
        left_taper = _smoothstep(left_u)
        right_taper = _smoothstep(right_u)
        delta[left_start:left_stop] += 0.5 * reduction * left_taper
        delta[right_start:right_stop] -= 0.5 * reduction * right_taper
        rows.append({"timestamp_ch": str(t), "row": int(i), "raw_jump_eur_mwh": jump})
    return delta, rows


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def _constraint_matrix(
    index_utc: pd.DatetimeIndex,
    ts_ch: pd.Series,
    *,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float],
) -> np.ndarray:
    hours = interval_hours(index_utc)
    rows: list[np.ndarray] = []
    base_buckets, _ = calibration_buckets(ts_ch, base_forward_prices)
    for _, idx in base_buckets.groupby(base_buckets, dropna=True).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        denom = float(hours[idx_arr].sum())
        if denom <= 0.0:
            continue
        row = np.zeros(len(index_utc), dtype=float)
        row[idx_arr] = hours[idx_arr] / denom
        rows.append(row)

    if peak_forward_prices:
        peak_mask = pd.Series(eex_peak_mask(index_utc, country="CH"), index=ts_ch.index)
        if bool(peak_mask.any()):
            peak_buckets, _ = calibration_buckets(ts_ch.loc[peak_mask], peak_forward_prices)
            for _, idx in peak_buckets.groupby(peak_buckets, dropna=True).groups.items():
                idx_arr = np.asarray(list(idx), dtype=int)
                denom = float(hours[idx_arr].sum())
                if denom <= 0.0:
                    continue
                row = np.zeros(len(index_utc), dtype=float)
                row[idx_arr] = hours[idx_arr] / denom
                rows.append(row)

    if not rows:
        return np.zeros((0, len(index_utc)), dtype=float)
    return _independent_rows(np.vstack(rows))


def _project_to_zero_constraints(values: np.ndarray, constraints: np.ndarray) -> np.ndarray:
    if constraints.size == 0:
        return np.asarray(values, dtype=float)
    residual = constraints @ values
    gram = constraints @ constraints.T
    try:
        multipliers = np.linalg.solve(gram, residual)
    except np.linalg.LinAlgError:
        multipliers = np.linalg.lstsq(gram, residual, rcond=None)[0]
    return values - constraints.T @ multipliers


def _independent_rows(matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    rows: list[np.ndarray] = []
    current = np.empty((0, matrix.shape[1]), dtype=float)
    rank = 0
    for row in matrix:
        trial = np.vstack([current, row])
        trial_rank = int(np.linalg.matrix_rank(trial, tol=tol))
        if trial_rank > rank:
            rows.append(row)
            current = trial
            rank = trial_rank
    if not rows:
        return np.zeros((0, matrix.shape[1]), dtype=float)
    return np.vstack(rows)


def _recompute_weighted_fan_columns(hourly: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    return recompute_structural_scenario_bracket(hourly, weights, include_legacy_aliases=True)


def _max_feasible_scale(
    hourly_before: pd.DataFrame,
    delta: np.ndarray,
    *,
    weights: Mapping[str, float],
    negative_price_floor: float,
    max_weighted_negative_hours: int,
) -> float:
    lo = 0.0
    hi = 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        trial = hourly_before.copy()
        for column in SCENARIO_PRICE_COLUMNS.values():
            trial[column] = trial[column].astype(float).to_numpy(dtype=float) + mid * delta
        trial = _recompute_weighted_fan_columns(trial, weights)
        min_price = float(trial[[*SCENARIO_PRICE_COLUMNS.values(), "price_weighted_mean_eur_mwh"]].min().min())
        weighted_negative_hours = int((trial["price_weighted_mean_eur_mwh"] < 0.0).sum())
        if min_price >= negative_price_floor - 1e-9 and weighted_negative_hours <= max_weighted_negative_hours:
            lo = mid
        else:
            hi = mid
    return lo
