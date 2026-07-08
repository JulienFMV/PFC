"""Experimental EPEX-only hourly shape lab for LT A/B candidates.

This module is intentionally off the production path.  It builds additive
hourly shape deltas from historical EPEX spot residuals, then projects those
deltas into the nullspace of active EEX BASE/PEAK constraints before applying
them to a candidate curve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.shape_constraints import build_base_peak_offpeak_constraint_system, validate_utc_index


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


@dataclass(frozen=True)
class ABShapeLabConfig:
    """Configuration for an advisory, off-by-default EPEX shape experiment."""

    variant: str = "epex_weekend_tail_peak_v1"
    lookback_years: int = 5
    valuation_timestamp: str | pd.Timestamp | None = None
    weekend_intensity: float = 0.0
    low_tail_intensity: float = 0.0
    peak_subshape_intensity: float = 0.0
    night_intensity: float = 0.0
    ramp_intensity: float = 0.0
    max_abs_delta_eur_mwh: float = 8.0
    negative_price_floor: float = -30.0
    max_weighted_negative_hours: int = 0
    min_sample_hours: int = 24
    require_monthly_base_constraints: bool = True
    country: str = "CH"
    calendar_tz: str = "Europe/Zurich"
    price_col: str = "price_eur_mwh"


def fit_epex_shape_templates(
    spot: pd.DataFrame,
    config: ABShapeLabConfig,
    *,
    valuation_timestamp: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit month/hour templates from point-in-time EPEX spot residuals.

    The fit is deliberately independent from OMPEX or any external HPFC
    benchmark.  Only rows strictly before the valuation timestamp are eligible.
    Residuals are computed versus their local calendar-month mean so the
    resulting templates are shape-only rather than level priors.
    """

    if config.lookback_years <= 0:
        raise ValueError("lookback_years must be positive")
    if config.min_sample_hours < 0:
        raise ValueError("min_sample_hours must be non-negative")
    if config.price_col not in spot.columns:
        raise ValueError(f"spot frame missing price column {config.price_col!r}")

    ts_utc = _extract_spot_utc_index(spot)
    price = pd.Series(spot[config.price_col].astype(float).to_numpy(dtype=float), index=ts_utc)
    finite = np.isfinite(price.to_numpy(dtype=float))
    ts_utc = ts_utc[finite]
    price = price.iloc[np.flatnonzero(finite)]

    valuation = _coerce_optional_timestamp(
        valuation_timestamp if valuation_timestamp is not None else config.valuation_timestamp,
    )
    if valuation is None:
        raise ValueError("EPEX shape lab requires an explicit valuation_timestamp for point-in-time fitting")
    mask = ts_utc < valuation
    lookback_start = valuation - pd.DateOffset(years=int(config.lookback_years))
    mask &= ts_utc >= lookback_start
    ts_utc = ts_utc[mask]
    price = price.loc[mask]

    if len(price) < max(24, config.min_sample_hours):
        raise ValueError("not enough point-in-time EPEX spot rows to fit shape templates")

    local = ts_utc.tz_convert(config.calendar_tz)
    features = pd.DataFrame(
        {
            "price": price.to_numpy(dtype=float),
            "month": local.month.astype(int),
            "hour": local.hour.astype(int),
            "is_weekend": (local.weekday >= 5).astype(bool),
            "is_peak_like": ((local.weekday < 5) & (local.hour >= 8) & (local.hour <= 19)).astype(bool),
            "is_solar_tail": ((local.month >= 3) & (local.month <= 10) & (local.hour >= 10) & (local.hour <= 16)).astype(bool),
        },
        index=ts_utc,
    )
    month_period = pd.PeriodIndex(local.strftime("%Y-%m"), freq="M")
    month_mean = features.groupby(month_period)["price"].transform("mean")
    features["residual"] = features["price"] - month_mean

    month_hour_median = features.groupby(["month", "hour"])["residual"].median()
    month_hour_q25 = features.groupby(["month", "hour"])["residual"].quantile(0.25)
    month_peak_mean = features.loc[features["is_peak_like"]].groupby("month")["residual"].median()
    non_night_mean = features.loc[~features["hour"].between(0, 5)].groupby("month")["residual"].median()
    cell_median = features.groupby(["month", "hour", "is_weekend"])["residual"].median()
    cell_count = features.groupby(["month", "hour", "is_weekend"])["residual"].size()
    ramp_smoothing: dict[tuple[int, int], float] = {}
    for month in range(1, 13):
        for hour in range(24):
            current = float(month_hour_median.get((month, hour), 0.0))
            previous_value = float(month_hour_median.get((month, (hour - 1) % 24), current))
            next_value = float(month_hour_median.get((month, (hour + 1) % 24), current))
            ramp_smoothing[(month, hour)] = 0.5 * (previous_value + next_value) - current

    rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        for hour in range(24):
            mh_median = float(month_hour_median.get((month, hour), 0.0))
            mh_q25 = float(month_hour_q25.get((month, hour), mh_median))
            non_night_ref = float(non_night_mean.get(month, 0.0))
            for is_weekend in (False, True):
                cell_key = (month, hour, is_weekend)
                n_obs = int(cell_count.get(cell_key, 0))
                shrink = _sample_shrink(n_obs, config.min_sample_hours)
                cell_residual = float(cell_median.get(cell_key, mh_median))
                weekend_delta = (cell_residual - mh_median) * shrink

                is_solar_tail = 3 <= month <= 10 and 10 <= hour <= 16
                low_tail_delta = min(0.0, mh_q25 - mh_median) * shrink if is_solar_tail else 0.0

                is_peak_like = (not is_weekend) and 8 <= hour <= 19
                peak_ref = float(month_peak_mean.get(month, 0.0))
                peak_subshape_delta = (mh_median - peak_ref) * shrink if is_peak_like else 0.0
                night_delta = (mh_median - non_night_ref) * shrink if 0 <= hour <= 5 else 0.0
                ramp_delta = ramp_smoothing[(month, hour)] * shrink

                rows.append(
                    {
                        "month": month,
                        "hour": hour,
                        "is_weekend": bool(is_weekend),
                        "weekend_delta_eur_mwh": float(weekend_delta),
                        "low_tail_delta_eur_mwh": float(low_tail_delta),
                        "peak_subshape_delta_eur_mwh": float(peak_subshape_delta),
                        "night_delta_eur_mwh": float(night_delta),
                        "ramp_delta_eur_mwh": float(ramp_delta),
                        "n_obs": n_obs,
                        "sample_shrink": float(shrink),
                    }
                )

    templates = pd.DataFrame(rows)
    info: dict[str, Any] = {
        "variant": config.variant,
        "benchmark_policy": "advisory_only_ompex_forbidden_as_input_target_loss_or_gate",
        "source": "EPEX_CH_spot_history",
        "n_input_rows": int(len(price)),
        "min_timestamp_utc": str(ts_utc.min()) if len(ts_utc) else None,
        "max_timestamp_utc": str(ts_utc.max()) if len(ts_utc) else None,
        "valuation_timestamp_utc": str(valuation) if valuation is not None else None,
        "lookback_years": int(config.lookback_years),
        "price_col": config.price_col,
    }
    return templates, info


def apply_epex_ab_shape_lab(
    hourly: pd.DataFrame,
    templates: pd.DataFrame,
    *,
    ts_ch: pd.Series | pd.DatetimeIndex,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float] | None,
    weights: Mapping[str, float],
    config: ABShapeLabConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fitted EPEX shape templates without changing EEX average constraints."""

    _validate_config(config)
    missing_cols = [col for col in SCENARIO_PRICE_COLUMNS.values() if col not in hourly.columns]
    if missing_cols:
        raise ValueError(f"EPEX shape lab missing scenario columns: {missing_cols}")
    if "price_weighted_mean_eur_mwh" not in hourly.columns:
        raise ValueError("EPEX shape lab requires price_weighted_mean_eur_mwh")
    if len(ts_ch) != len(hourly):
        raise ValueError(f"EPEX shape lab timestamp length mismatch: {len(ts_ch)} != {len(hourly)}")

    total_intensity = (
        float(config.weekend_intensity)
        + float(config.low_tail_intensity)
        + float(config.peak_subshape_intensity)
        + float(config.night_intensity)
        + float(config.ramp_intensity)
    )
    out = hourly.copy()
    if total_intensity == 0.0 or float(config.max_abs_delta_eur_mwh) == 0.0:
        return out, pd.DataFrame()

    ts = _coerce_local_index(ts_ch, config.calendar_tz)
    if config.require_monthly_base_constraints:
        _require_monthly_base_constraints(ts, base_forward_prices)
    template = _validate_templates(templates)
    feature = pd.DataFrame(
        {
            "row": np.arange(len(ts), dtype=int),
            "month": ts.month.astype(int),
            "hour": ts.hour.astype(int),
            "is_weekend": (ts.weekday >= 5).astype(bool),
        }
    )
    mapped = feature.merge(template, on=["month", "hour", "is_weekend"], how="left", validate="many_to_one")
    mapped = mapped.sort_values("row")
    for col in [
        "weekend_delta_eur_mwh",
        "low_tail_delta_eur_mwh",
        "peak_subshape_delta_eur_mwh",
        "night_delta_eur_mwh",
        "ramp_delta_eur_mwh",
    ]:
        mapped[col] = mapped[col].fillna(0.0).astype(float)

    raw_delta = (
        float(config.weekend_intensity) * mapped["weekend_delta_eur_mwh"].to_numpy(dtype=float)
        + float(config.low_tail_intensity) * mapped["low_tail_delta_eur_mwh"].to_numpy(dtype=float)
        + float(config.peak_subshape_intensity) * mapped["peak_subshape_delta_eur_mwh"].to_numpy(dtype=float)
        + float(config.night_intensity) * mapped["night_delta_eur_mwh"].to_numpy(dtype=float)
        + float(config.ramp_intensity) * mapped["ramp_delta_eur_mwh"].to_numpy(dtype=float)
    )
    if not np.isfinite(raw_delta).all():
        raise ValueError("EPEX shape lab produced non-finite raw deltas")

    projected_delta, constraint_residual = project_delta_to_base_peak_nullspace(
        raw_delta,
        ts_ch=ts,
        base_forward_prices=base_forward_prices,
        peak_forward_prices=peak_forward_prices or {},
        country=config.country,
        require_monthly_base_constraints=config.require_monthly_base_constraints,
    )
    max_projected = float(np.max(np.abs(projected_delta))) if len(projected_delta) else 0.0
    if max_projected > float(config.max_abs_delta_eur_mwh) > 0.0:
        projected_delta = projected_delta * (float(config.max_abs_delta_eur_mwh) / max_projected)
        _, constraint_residual = project_delta_to_base_peak_nullspace(
            projected_delta,
            ts_ch=ts,
            base_forward_prices=base_forward_prices,
            peak_forward_prices=peak_forward_prices or {},
            country=config.country,
            require_monthly_base_constraints=config.require_monthly_base_constraints,
        )

    before = out.copy()
    for column in SCENARIO_PRICE_COLUMNS.values():
        out[column] = out[column].astype(float).to_numpy(dtype=float) + projected_delta
    out = _recompute_weighted_fan_columns(out, weights, base_hourly=before, delta=projected_delta)
    out, projected_delta = _enforce_floor_and_negative_hour_cap(
        before,
        projected_delta,
        weights=weights,
        negative_price_floor=float(config.negative_price_floor),
        max_weighted_negative_hours=int(config.max_weighted_negative_hours),
    )

    min_price = float(out[[*SCENARIO_PRICE_COLUMNS.values(), "price_weighted_mean_eur_mwh"]].min().min())
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if min_price < float(config.negative_price_floor) - 1e-9:
        raise ValueError(
            f"EPEX shape lab breached negative price floor: min={min_price:.6f}, "
            f"floor={float(config.negative_price_floor):.6f}"
        )
    if weighted_negative_hours > int(config.max_weighted_negative_hours):
        raise ValueError(
            "EPEX shape lab produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(config.max_weighted_negative_hours)}"
        )

    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)

    after_constraint_error = _max_after_constraint_error(
        out["price_weighted_mean_eur_mwh"].to_numpy(dtype=float),
        ts_ch=ts,
        base_forward_prices=base_forward_prices,
        peak_forward_prices=peak_forward_prices or {},
        country=config.country,
    )
    audit = pd.DataFrame(
        [
            {
                "variant": config.variant,
                "benchmark_policy": "advisory_only_ompex_forbidden_as_input_target_loss_or_gate",
                "n_hours": int(len(out)),
                "weekend_intensity": float(config.weekend_intensity),
                "low_tail_intensity": float(config.low_tail_intensity),
                "peak_subshape_intensity": float(config.peak_subshape_intensity),
                "night_intensity": float(config.night_intensity),
                "ramp_intensity": float(config.ramp_intensity),
                "max_abs_raw_delta_eur_mwh": float(np.max(np.abs(raw_delta))) if len(raw_delta) else 0.0,
                "max_abs_projected_delta_eur_mwh": float(np.max(np.abs(projected_delta))) if len(projected_delta) else 0.0,
                "mean_projected_delta_eur_mwh": float(np.mean(projected_delta)) if len(projected_delta) else 0.0,
                "max_delta_constraint_residual_eur_mwh": float(constraint_residual),
                "max_after_constraint_abs_error_eur_mwh": float(after_constraint_error),
                "min_price_eur_mwh": min_price,
                "weighted_negative_hours": weighted_negative_hours,
                "nonzero_delta_hours": int(np.sum(np.abs(projected_delta) > 1e-12)),
            }
        ]
    )
    return out, audit


def project_delta_to_base_peak_nullspace(
    delta: np.ndarray,
    *,
    ts_ch: pd.Series | pd.DatetimeIndex,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float] | None = None,
    country: str = "CH",
    require_monthly_base_constraints: bool = True,
) -> tuple[np.ndarray, float]:
    """Project an additive delta onto the zero-residual BASE/PEAK nullspace."""

    values = np.asarray(delta, dtype=float)
    if values.ndim != 1:
        raise ValueError("delta must be a one-dimensional array")
    ts = _coerce_local_index(ts_ch, "Europe/Zurich")
    if len(ts) != len(values):
        raise ValueError(f"delta length mismatch: {len(values)} != {len(ts)}")
    if require_monthly_base_constraints:
        _require_monthly_base_constraints(ts, base_forward_prices)
    index_utc = validate_utc_index(ts.tz_convert("UTC"))
    system = build_base_peak_offpeak_constraint_system(
        index_utc,
        base_forward_prices,
        peak_forward_prices or {},
        country=country,
    )
    constraints = system.matrix
    projected = _project_to_zero_constraints(values, constraints)
    residual = float(np.max(np.abs(constraints @ projected))) if constraints.size else 0.0
    return projected, residual


def build_ab_lab_manifest(
    config: ABShapeLabConfig,
    *,
    fit_info: Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str] | None = None,
    audit: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a small governance manifest for an EPEX A/B shape-lab run."""

    manifest: dict[str, Any] = {
        "ab_shape_lab": "epex_only_off_by_default",
        "activation_status": "lab_only",
        "production_approved": False,
        "config": _jsonable(asdict(config)),
        "benchmark_policy": "advisory_only_ompex_forbidden_as_input_target_loss_or_gate",
        "forbidden_inputs": ["OMPEX", "HFC_OMPEX", "external_HPFC_benchmark"],
        "ompex_used_in_selection": False,
        "selection_basis": "pre_registered_independent_epex_spot_residual_ab_only",
        "source_hashes_required_for_promotion": True,
        "source_hashes": _jsonable(dict(source_hashes or {})),
        "fit_info": _jsonable(dict(fit_info or {})),
    }
    if audit is not None and not audit.empty:
        manifest["audit"] = _jsonable(audit.iloc[0].to_dict())
    return manifest


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _max_after_constraint_error(
    values: np.ndarray,
    *,
    ts_ch: pd.DatetimeIndex,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float],
    country: str,
) -> float:
    system = build_base_peak_offpeak_constraint_system(
        ts_ch.tz_convert("UTC"),
        base_forward_prices,
        peak_forward_prices,
        country=country,
    )
    residuals = system.residuals(np.asarray(values, dtype=float))
    return float(residuals["abs_error"].max()) if not residuals.empty else 0.0


def _extract_spot_utc_index(spot: pd.DataFrame) -> pd.DatetimeIndex:
    if "timestamp_utc" in spot.columns:
        return pd.DatetimeIndex(pd.to_datetime(spot["timestamp_utc"], utc=True))
    if "timestamp" in spot.columns:
        parsed = pd.to_datetime(spot["timestamp"], utc=False)
        index = pd.DatetimeIndex(parsed)
        if index.tz is None:
            raise ValueError("spot timestamp column must be timezone-aware; use timestamp_utc for UTC data")
        return index.tz_convert("UTC")
    if isinstance(spot.index, pd.DatetimeIndex):
        index = spot.index
        if index.tz is None:
            raise ValueError("spot DatetimeIndex must be timezone-aware")
        return index.tz_convert("UTC")
    raise ValueError("spot frame requires timestamp_utc, timestamp, or a DatetimeIndex")


def _coerce_optional_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _coerce_local_index(values: pd.Series | pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        index = values
        if index.tz is None:
            index = index.tz_localize(tz)
        return index.tz_convert(tz)
    raw = pd.Series(values)
    if _series_has_explicit_timezone(raw):
        index = pd.DatetimeIndex(pd.to_datetime(raw, utc=True))
        return index.tz_convert(tz)
    parsed = pd.to_datetime(raw, utc=False)
    index = pd.DatetimeIndex(parsed)
    if index.tz is None:
        index = index.tz_localize(tz)
    else:
        index = index.tz_convert(tz)
    return index


def _series_has_explicit_timezone(values: pd.Series) -> bool:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return True
    if not values.dtype == object:
        return False
    pattern = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
    for value in values.dropna().head(20):
        if pattern.search(str(value).strip()):
            return True
    return False


def _require_monthly_base_constraints(ts: pd.DatetimeIndex, base_forward_prices: Mapping[str, float]) -> None:
    quoted_months = {str(key) for key in base_forward_prices if _is_month_key(str(key))}
    delivered_months = set(pd.PeriodIndex(ts.strftime("%Y-%m"), freq="M").astype(str))
    missing = sorted(delivered_months - quoted_months)
    if missing:
        raise ValueError(
            "EPEX shape lab requires monthly BASE constraints for every delivered month "
            "when monthly solver is the level authority; missing "
            f"{missing[:6]}{'...' if len(missing) > 6 else ''}"
        )


def _is_month_key(key: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}", key))


def _sample_shrink(n_obs: int, min_sample_hours: int) -> float:
    if n_obs <= 0:
        return 0.0
    if min_sample_hours <= 0:
        return 1.0
    return float(n_obs / (n_obs + min_sample_hours))


def _validate_config(config: ABShapeLabConfig) -> None:
    for name in [
        "weekend_intensity",
        "low_tail_intensity",
        "peak_subshape_intensity",
        "night_intensity",
        "ramp_intensity",
        "max_abs_delta_eur_mwh",
    ]:
        value = float(getattr(config, name))
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    if int(config.max_weighted_negative_hours) < 0:
        raise ValueError("max_weighted_negative_hours must be non-negative")
    if not np.isfinite(float(config.negative_price_floor)):
        raise ValueError("negative_price_floor must be finite")


def _validate_templates(templates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "month",
        "hour",
        "is_weekend",
        "weekend_delta_eur_mwh",
        "low_tail_delta_eur_mwh",
        "peak_subshape_delta_eur_mwh",
        "night_delta_eur_mwh",
        "ramp_delta_eur_mwh",
    }
    missing = sorted(required - set(templates.columns))
    if missing:
        raise ValueError(f"EPEX shape lab templates missing columns: {missing}")
    out = templates.copy()
    out["month"] = out["month"].astype(int)
    out["hour"] = out["hour"].astype(int)
    out["is_weekend"] = out["is_weekend"].astype(bool)
    return out


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
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=float)
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


def _recompute_weighted_fan_columns(
    hourly: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    base_hourly: pd.DataFrame | None = None,
    delta: np.ndarray | None = None,
) -> pd.DataFrame:
    out = hourly.copy()
    labels = tuple(SCENARIO_PRICE_COLUMNS)
    missing_weights = [label for label in labels if label not in weights]
    if missing_weights:
        raise ValueError(f"EPEX shape lab missing scenario weights: {missing_weights}")
    matrix = np.column_stack([out[SCENARIO_PRICE_COLUMNS[label]].to_numpy(dtype=float) for label in labels])
    weight_vector = np.array([float(weights[label]) for label in labels], dtype=float)
    if not np.isclose(float(weight_vector.sum()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("scenario weights must sum to 1")
    if base_hourly is not None and delta is not None and "price_weighted_mean_eur_mwh" in base_hourly.columns:
        delta_arr = np.asarray(delta, dtype=float)
        if delta_arr.shape != (len(out),):
            raise ValueError("delta length mismatch while recomputing weighted fan columns")
        out["price_weighted_mean_eur_mwh"] = (
            base_hourly["price_weighted_mean_eur_mwh"].astype(float).to_numpy(dtype=float) + delta_arr
        )
        shifted_quantiles = True
        for column in ["structural_p10_eur_mwh", "structural_p50_eur_mwh", "structural_p90_eur_mwh"]:
            if column not in base_hourly.columns:
                shifted_quantiles = False
                break
            out[column] = base_hourly[column].astype(float).to_numpy(dtype=float) + delta_arr
        if shifted_quantiles and "structural_width_eur_mwh" in base_hourly.columns:
            out["structural_width_eur_mwh"] = base_hourly["structural_width_eur_mwh"].astype(float).to_numpy(dtype=float)
        elif shifted_quantiles:
            out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
        else:
            out["structural_p10_eur_mwh"] = np.quantile(matrix, 0.10, axis=1)
            out["structural_p50_eur_mwh"] = np.quantile(matrix, 0.50, axis=1)
            out["structural_p90_eur_mwh"] = np.quantile(matrix, 0.90, axis=1)
            out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
    else:
        out["price_weighted_mean_eur_mwh"] = matrix @ weight_vector
        out["structural_p10_eur_mwh"] = np.quantile(matrix, 0.10, axis=1)
        out["structural_p50_eur_mwh"] = np.quantile(matrix, 0.50, axis=1)
        out["structural_p90_eur_mwh"] = np.quantile(matrix, 0.90, axis=1)
        out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
    return out


def _enforce_floor_and_negative_hour_cap(
    hourly_before: pd.DataFrame,
    delta: np.ndarray,
    *,
    weights: Mapping[str, float],
    negative_price_floor: float,
    max_weighted_negative_hours: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    trial = hourly_before.copy()
    for column in SCENARIO_PRICE_COLUMNS.values():
        trial[column] = trial[column].astype(float).to_numpy(dtype=float) + delta
    trial = _recompute_weighted_fan_columns(trial, weights, base_hourly=hourly_before, delta=delta)
    min_price = float(trial[[*SCENARIO_PRICE_COLUMNS.values(), "price_weighted_mean_eur_mwh"]].min().min())
    weighted_negative_hours = int((trial["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if min_price >= negative_price_floor - 1e-9 and weighted_negative_hours <= max_weighted_negative_hours:
        return trial, delta

    lo = 0.0
    hi = 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        candidate = hourly_before.copy()
        for column in SCENARIO_PRICE_COLUMNS.values():
            candidate[column] = candidate[column].astype(float).to_numpy(dtype=float) + mid * delta
        candidate = _recompute_weighted_fan_columns(candidate, weights, base_hourly=hourly_before, delta=mid * delta)
        min_candidate = float(candidate[[*SCENARIO_PRICE_COLUMNS.values(), "price_weighted_mean_eur_mwh"]].min().min())
        neg_candidate = int((candidate["price_weighted_mean_eur_mwh"] < 0.0).sum())
        if min_candidate >= negative_price_floor - 1e-9 and neg_candidate <= max_weighted_negative_hours:
            lo = mid
        else:
            hi = mid

    scaled_delta = delta * lo
    out = hourly_before.copy()
    for column in SCENARIO_PRICE_COLUMNS.values():
        out[column] = out[column].astype(float).to_numpy(dtype=float) + scaled_delta
    return _recompute_weighted_fan_columns(out, weights, base_hourly=hourly_before, delta=scaled_delta), scaled_delta
