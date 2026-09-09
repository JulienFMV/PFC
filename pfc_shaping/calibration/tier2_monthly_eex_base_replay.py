"""Target-blind deterministic BASE-only core for Tier 2 monthly EEX replay.

This module does not authenticate artifacts and does not create campaign
evidence. It is the pure generation core that a later path-only verifier can
call after reconstructing exact permitted rows from signed lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    product_periods,
    solve_monthly_forward_curve_from_constraints,
)
from pfc_shaping.calibration.tier2_monthly_eex_evaluation import _sha256_json


BASE_REPLAY_CONFIG_SCHEMA = "tier2_monthly_eex_base_candidate_config.v1"
BASE_REPLAY_OUTPUT_SCHEMA = "tier2_monthly_eex_base_model_output.v1"
BASE_REPLAY_STATUS = "MODEL_OUTPUT_REPLAYED_BASE_ONLY"
BASE_INPUT_SET_SCHEMA = "tier2_monthly_eex_base_input_set.v1"
BASELINE_ID = "market_constrained_seasonal.v1"
JOINT_SEGMENT_UNSUPPORTED = "UNSUPPORTED_REQUIRES_JOINT_PEAK_OFFPEAK_SOLVER"
BASELINE_UNSUPPORTED = "UNSUPPORTED_BASELINE"
MAX_KKT_CONDITION_NUMBER = 1e12
MAX_OBJECTIVE_WEIGHT = 1e12

_MONTHLY_CONFIG_FIELDS = {
    "lambda_prior",
    "lambda_smooth_month",
    "lambda_smooth_yoy",
    "lambda_shape",
    "neighbor_shrinkage",
    "robust_panel_quantile",
    "min_history_snapshots",
    "max_prior_residual_eur_mwh",
    "constraint_tolerance",
    "quote_conflict_tolerance",
    "stationarity_tolerance",
}
_RETAINED_COLUMNS = {"market", "load_type", "product", "price", "quote_id"}
_HISTORY_COLUMNS = {
    "date",
    "available_at",
    "market",
    "load_type",
    "product",
    "price",
    "snapshot_id",
    "row_hash",
}


def _base_input_set_sha256(
    base_history_row_hashes: Sequence[str],
    base_retained_row_hashes: Sequence[str],
) -> str:
    history = tuple(base_history_row_hashes)
    retained = tuple(base_retained_row_hashes)
    for label, values in (("history", history), ("retained", retained)):
        if any(type(value) is not str or not value for value in values):
            raise Tier2MonthlyBaseReplayContractError(
                f"BASE {label} row hash inventory is invalid"
            )
        if len(values) != len(set(values)):
            raise Tier2MonthlyBaseReplayContractError(
                f"BASE {label} row hash inventory contains duplicates"
            )
    if set(history).intersection(retained):
        raise Tier2MonthlyBaseReplayContractError(
            "BASE historical and retained row inventories overlap"
        )
    return _sha256_json(
        {
            "schema_version": BASE_INPUT_SET_SCHEMA,
            "base_history_row_hashes": list(history),
            "base_retained_row_hashes": list(retained),
        }
    )


class Tier2MonthlyBaseReplayError(ValueError):
    """Raised when the BASE-only replay contract cannot be satisfied."""


class Tier2MonthlyBaseReplayUnsupported(Tier2MonthlyBaseReplayError):
    """Raised for preregistered scientific coverage limitations."""


class Tier2MonthlyBaseReplayContractError(Tier2MonthlyBaseReplayError):
    """Raised when inputs violate the deterministic replay contract."""


class Tier2MonthlyBaseReplayNumericalError(Tier2MonthlyBaseReplayError):
    """Raised when the governed solve is numerically unacceptable."""


@dataclass(frozen=True)
class Tier2MonthlyBaseReplayOutput:
    schema_version: str
    status: str
    scope: str
    candidate_config_sha256: str
    baseline_id: str
    delivery_months: tuple[str, ...]
    candidate_monthly_base: tuple[float, ...]
    baseline_monthly_base: tuple[float, ...]
    candidate_max_abs_constraint_residual: float
    baseline_max_abs_constraint_residual: float

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)


def validate_base_candidate_config(
    value: Mapping[str, object],
) -> MonthlyCurveConfig:
    """Validate one exhaustive candidate config without implicit defaults."""

    if type(value) is not dict:
        raise Tier2MonthlyBaseReplayContractError(
            "candidate config must be a plain JSON object"
        )
    if _MONTHLY_CONFIG_FIELDS != {field.name for field in fields(MonthlyCurveConfig)}:
        raise Tier2MonthlyBaseReplayContractError(
            "candidate config schema is out of sync with MonthlyCurveConfig"
        )
    _exact_keys(value, {"schema_version", "monthly_curve"}, label="candidate config")
    if value.get("schema_version") != BASE_REPLAY_CONFIG_SCHEMA:
        raise Tier2MonthlyBaseReplayContractError("candidate config schema mismatch")
    raw = value.get("monthly_curve")
    if type(raw) is not dict:
        raise Tier2MonthlyBaseReplayContractError(
            "candidate monthly_curve must be a plain JSON object"
        )
    _exact_keys(raw, _MONTHLY_CONFIG_FIELDS, label="candidate monthly_curve")

    config = MonthlyCurveConfig(
        lambda_prior=_bounded_positive_float(raw["lambda_prior"], label="lambda_prior"),
        lambda_smooth_month=_bounded_nonnegative_float(
            raw["lambda_smooth_month"], label="lambda_smooth_month"
        ),
        lambda_smooth_yoy=_bounded_nonnegative_float(
            raw["lambda_smooth_yoy"], label="lambda_smooth_yoy"
        ),
        lambda_shape=_bounded_nonnegative_float(
            raw["lambda_shape"], label="lambda_shape"
        ),
        neighbor_shrinkage=_nonnegative_float(
            raw["neighbor_shrinkage"], label="neighbor_shrinkage"
        ),
        robust_panel_quantile=_unit_float(
            raw["robust_panel_quantile"], label="robust_panel_quantile"
        ),
        min_history_snapshots=_integer_at_least(
            raw["min_history_snapshots"], 24, label="min_history_snapshots"
        ),
        max_prior_residual_eur_mwh=_optional_positive_float(
            raw["max_prior_residual_eur_mwh"],
            label="max_prior_residual_eur_mwh",
        ),
        constraint_tolerance=_positive_float(
            raw["constraint_tolerance"], label="constraint_tolerance"
        ),
        quote_conflict_tolerance=_positive_float(
            raw["quote_conflict_tolerance"], label="quote_conflict_tolerance"
        ),
        stationarity_tolerance=_positive_float(
            raw["stationarity_tolerance"], label="stationarity_tolerance"
        ),
    )
    if config.neighbor_shrinkage != 0.0:
        raise Tier2MonthlyBaseReplayContractError(
            "BASE replay forbids neighbor inputs; neighbor_shrinkage must be zero"
        )
    if config.robust_panel_quantile != 0.5:
        raise Tier2MonthlyBaseReplayContractError(
            "BASE replay robust_panel_quantile must be exactly 0.5"
        )
    if config.max_prior_residual_eur_mwh is not None:
        raise Tier2MonthlyBaseReplayContractError(
            "BASE replay v1 requires max_prior_residual_eur_mwh=null"
        )
    if config.min_history_snapshots != 24:
        raise Tier2MonthlyBaseReplayContractError(
            "BASE replay min_history_snapshots must be exactly 24"
        )
    if config.constraint_tolerance != 1e-9:
        raise Tier2MonthlyBaseReplayContractError(
            "candidate constraint_tolerance must be exactly 1e-9"
        )
    if config.quote_conflict_tolerance != 0.01:
        raise Tier2MonthlyBaseReplayContractError(
            "candidate quote_conflict_tolerance must be exactly 0.01"
        )
    if config.stationarity_tolerance != 1e-7:
        raise Tier2MonthlyBaseReplayContractError(
            "candidate stationarity_tolerance must be exactly 1e-7"
        )
    return config


def _replay_monthly_base_outputs(
    *,
    delivery_months: Sequence[str],
    retained_quotes: pd.DataFrame,
    fold_historical_context: pd.DataFrame,
    candidate_config: Mapping[str, object],
) -> Tier2MonthlyBaseReplayOutput:
    """Generate candidate and canonical baseline without target or diagnostics."""

    months = _delivery_grid(delivery_months)
    config_payload = _candidate_config_snapshot(candidate_config)
    config_hash = _sha256_json(config_payload)
    config = validate_base_candidate_config(config_payload)
    retained = _validated_retained_quotes(retained_quotes)
    history = _validated_history(fold_historical_context)
    try:
        constraints = build_monthly_constraint_system(
            months,
            _market_quotes(retained),
            timezone="Europe/Zurich",
            market="CH",
            load_type="BASE",
            constraint_tolerance=config.constraint_tolerance,
            quote_conflict_tolerance=config.quote_conflict_tolerance,
        )
    except (TypeError, ValueError) as exc:
        raise Tier2MonthlyBaseReplayContractError(
            "retained quote constraint construction failed"
        ) from exc
    _require_represented_month_per_delivery_year(constraints)
    seasonal_prior = _market_constrained_seasonal_prior(
        history,
        constraints=constraints,
        min_observations=24,
    )
    candidate = _solve_checked(
        constraints,
        config=config,
        shape_prior=seasonal_prior,
        label="candidate",
    )
    baseline_config = MonthlyCurveConfig(
        lambda_prior=1e-6,
        lambda_smooth_month=0.0,
        lambda_smooth_yoy=0.0,
        lambda_shape=1.0,
        neighbor_shrinkage=0.0,
        robust_panel_quantile=0.5,
        min_history_snapshots=config.min_history_snapshots,
        max_prior_residual_eur_mwh=None,
        constraint_tolerance=config.constraint_tolerance,
        quote_conflict_tolerance=config.quote_conflict_tolerance,
        stationarity_tolerance=config.stationarity_tolerance,
    )
    baseline = _solve_checked(
        constraints,
        config=baseline_config,
        shape_prior=seasonal_prior,
        label="baseline",
    )
    candidate_residual = _validate_solve(candidate, config=config, label="candidate")
    baseline_residual = _validate_solve(
        baseline, config=baseline_config, label="baseline"
    )
    _validate_retained_repricing(
        retained,
        candidate.monthly_curve,
        constraints=constraints,
        label="candidate",
    )
    _validate_retained_repricing(
        retained,
        baseline.monthly_curve,
        constraints=constraints,
        label="baseline",
    )
    candidate_values = tuple(float(value) for value in candidate.monthly_curve)
    baseline_values = tuple(float(value) for value in baseline.monthly_curve)
    return Tier2MonthlyBaseReplayOutput(
        schema_version=BASE_REPLAY_OUTPUT_SCHEMA,
        status=BASE_REPLAY_STATUS,
        scope="MONTHLY_EEX_BASE_MASKED_QUOTE_ONLY",
        candidate_config_sha256=config_hash,
        baseline_id=BASELINE_ID,
        delivery_months=tuple(str(month) for month in months),
        candidate_monthly_base=candidate_values,
        baseline_monthly_base=baseline_values,
        candidate_max_abs_constraint_residual=candidate_residual,
        baseline_max_abs_constraint_residual=baseline_residual,
    )


def _market_constrained_seasonal_prior(
    history: pd.DataFrame,
    *,
    constraints: Any,
    min_observations: int,
) -> pd.Series:
    ch = history[
        history["market"].eq("CH") & history["load_type"].eq("BASE")
    ].copy()
    deviations: dict[int, dict[str, float]] = {
        month: {} for month in range(1, 13)
    }
    for snapshot_id, snapshot in ch.groupby("snapshot_id", sort=True):
        if snapshot["product"].duplicated().any():
            raise Tier2MonthlyBaseReplayContractError(
                "historical snapshot contains duplicate CH BASE products"
            )
        prices = dict(
            zip(snapshot["product"].astype(str), snapshot["price"].astype(float))
        )
        snapshot_values: dict[int, list[float]] = {
            month: [] for month in range(1, 13)
        }
        for product, price in prices.items():
            if len(product) != 7 or product[4] != "-":
                continue
            try:
                year = int(product[:4])
                month = int(product[5:7])
            except ValueError:
                continue
            if not 1 <= month <= 12 or str(year) not in prices:
                continue
            snapshot_values[month].append(
                float(price) - float(prices[str(year)])
            )
        for month, values in snapshot_values.items():
            if values:
                deviations[month][str(snapshot_id)] = float(
                    np.median(np.asarray(values, dtype=np.float64))
                )
    medians: dict[int, float] = {}
    for month, by_snapshot in deviations.items():
        if len(by_snapshot) < min_observations:
            raise Tier2MonthlyBaseReplayUnsupported(
                f"{BASELINE_UNSUPPORTED}: month={month:02d}, "
                f"snapshots={len(by_snapshot)}, required={min_observations}"
            )
        medians[month] = float(
            np.median(np.asarray(list(by_snapshot.values()), dtype=np.float64))
        )
    raw = pd.Series(
        {
            month: medians[int(month.month)]
            for month in constraints.delivery_grid.months
        },
        dtype=np.float64,
    )
    return raw


def _require_represented_month_per_delivery_year(constraints: Any) -> None:
    months = constraints.delivery_grid.months
    represented = constraints.month_buckets.notna()
    for year in sorted(set(int(month.year) for month in months)):
        in_year = np.asarray([int(month.year) == year for month in months])
        if not bool(represented.to_numpy()[in_year].any()):
            raise Tier2MonthlyBaseReplayUnsupported(
                f"UNSUPPORTED_NO_RETAINED_BASE_LEVEL_ANCHOR: year={year}"
            )


def _validated_retained_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _RETAINED_COLUMNS, label="retained quotes")
    retained = frame.copy()
    retained["market"] = retained["market"].astype(str).str.upper()
    retained["load_type"] = retained["load_type"].astype(str).str.upper()
    _require_nonempty_string_series(retained["product"], label="retained product")
    _require_nonempty_string_series(retained["quote_id"], label="retained quote_id")
    if retained.empty:
        raise Tier2MonthlyBaseReplayUnsupported("retained quote set is empty")
    if set(retained["market"]) != {"CH"} or set(retained["load_type"]) != {"BASE"}:
        raise Tier2MonthlyBaseReplayUnsupported(JOINT_SEGMENT_UNSUPPORTED)
    retained["price"] = pd.to_numeric(retained["price"], errors="coerce")
    if not np.isfinite(retained["price"].to_numpy(dtype=float)).all():
        raise Tier2MonthlyBaseReplayContractError(
            "retained quotes contain non-finite prices"
        )
    if retained["product"].astype(str).duplicated().any():
        raise Tier2MonthlyBaseReplayContractError(
            "retained quotes contain duplicate economic products"
        )
    if retained["quote_id"].astype(str).duplicated().any():
        raise Tier2MonthlyBaseReplayContractError(
            "retained quotes contain duplicate quote identities"
        )
    return retained.sort_values(["product", "quote_id"], kind="mergesort")


def _validated_history(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _HISTORY_COLUMNS, label="fold historical context")
    history = frame.copy()
    history["market"] = history["market"].astype(str).str.upper()
    history["load_type"] = history["load_type"].astype(str).str.upper()
    for column in ("product", "snapshot_id", "row_hash"):
        _require_nonempty_string_series(
            history[column], label=f"fold historical {column}"
        )
    history["price"] = pd.to_numeric(history["price"], errors="coerce")
    if not np.isfinite(history["price"].to_numpy(dtype=float)).all():
        raise Tier2MonthlyBaseReplayContractError(
            "fold historical context contains non-finite prices"
        )
    if history["row_hash"].astype(str).duplicated().any():
        raise Tier2MonthlyBaseReplayContractError(
            "fold historical context contains duplicate row hashes"
        )
    return history.sort_values(
        ["available_at", "snapshot_id", "market", "load_type", "product"],
        kind="mergesort",
    )


def _market_quotes(frame: pd.DataFrame) -> tuple[MarketQuote, ...]:
    return tuple(
        MarketQuote(
            market="CH",
            product=str(row.product),
            load_type="BASE",
            price=float(row.price),
            quote_id=str(row.quote_id),
        )
        for row in frame.itertuples(index=False)
    )


def _delivery_grid(values: Sequence[str]) -> pd.PeriodIndex:
    try:
        months = pd.PeriodIndex(values, freq="M")
    except (TypeError, ValueError) as exc:
        raise Tier2MonthlyBaseReplayContractError("delivery months are invalid") from exc
    if not len(months) or months.has_duplicates or not months.is_monotonic_increasing:
        raise Tier2MonthlyBaseReplayContractError(
            "delivery months must be non-empty, unique and chronological"
        )
    expected = pd.period_range(months[0], months[-1], freq="M")
    if not months.equals(expected):
        raise Tier2MonthlyBaseReplayContractError("delivery months must be contiguous")
    return months


def _validate_solve(result: Any, *, config: MonthlyCurveConfig, label: str) -> float:
    values = result.monthly_curve.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} produced non-finite monthly values"
        )
    residual = float(result.kkt["max_abs_constraint_residual"])
    stationarity = float(result.kkt["stationarity_residual"])
    condition = float(result.kkt["condition_number"])
    if not math.isfinite(residual) or not math.isfinite(stationarity):
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} KKT residual diagnostics are non-finite"
        )
    if residual > config.constraint_tolerance:
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} hard-constraint residual exceeds tolerance"
        )
    if stationarity > config.stationarity_tolerance:
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} stationarity residual exceeds tolerance"
        )
    if bool(result.kkt["solved_by_lstsq"]):
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} required a least-squares KKT fallback"
        )
    if bool(result.kkt["ridge_used"]):
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} used an ungoverned numerical ridge"
        )
    if not math.isfinite(condition) or condition > MAX_KKT_CONDITION_NUMBER:
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} KKT condition number is unacceptable"
        )
    return residual


def _solve_checked(
    constraints: Any,
    *,
    config: MonthlyCurveConfig,
    shape_prior: pd.Series,
    label: str,
) -> Any:
    try:
        return solve_monthly_forward_curve_from_constraints(
            constraints,
            config=config,
            shape_prior=shape_prior,
        )
    except (ArithmeticError, ValueError) as exc:
        raise Tier2MonthlyBaseReplayNumericalError(
            f"{label} monthly solver failed closed"
        ) from exc


def _validate_retained_repricing(
    retained: pd.DataFrame,
    curve: pd.Series,
    *,
    constraints: Any,
    label: str,
) -> None:
    hours = constraints.delivery_grid.month_hours
    for row in retained.itertuples(index=False):
        months = product_periods(str(row.product))
        values = curve.reindex(months)
        weights = hours.reindex(months)
        if values.isna().any() or weights.isna().any():
            raise Tier2MonthlyBaseReplayContractError(
                f"{label} retained product escaped the delivery grid"
            )
        predicted = float(np.average(values.to_numpy(dtype=float), weights=weights))
        if abs(predicted - float(row.price)) > 1e-9:
            raise Tier2MonthlyBaseReplayNumericalError(
                f"{label} does not reprice every retained quote"
            )


def _candidate_config_snapshot(value: Mapping[str, object]) -> dict[str, object]:
    if type(value) is not dict or type(value.get("monthly_curve")) is not dict:
        raise Tier2MonthlyBaseReplayContractError(
            "candidate config must contain plain JSON objects"
        )
    return {
        "schema_version": value.get("schema_version"),
        "monthly_curve": dict(value["monthly_curve"]),  # type: ignore[arg-type]
        **{
            str(key): item
            for key, item in value.items()
            if key not in {"schema_version", "monthly_curve"}
        },
    }


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise Tier2MonthlyBaseReplayContractError(
            f"{label} fields mismatch: expected={sorted(expected)}, actual={sorted(value)}"
        )


def _require_columns(frame: pd.DataFrame, expected: set[str], *, label: str) -> None:
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise Tier2MonthlyBaseReplayContractError(
            f"{label} missing columns: {missing}"
        )


def _require_nonempty_string_series(series: pd.Series, *, label: str) -> None:
    if series.map(lambda value: type(value) is not str or not value.strip()).any():
        raise Tier2MonthlyBaseReplayContractError(
            f"{label} must contain non-empty JSON strings"
        )


def _number(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise Tier2MonthlyBaseReplayContractError(f"{label} must be a JSON float")
    number = value
    if not math.isfinite(number):
        raise Tier2MonthlyBaseReplayContractError(f"{label} must be finite")
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise Tier2MonthlyBaseReplayContractError(f"{label} cannot be negative zero")
    return number


def _nonnegative_float(value: object, *, label: str) -> float:
    number = _number(value, label=label)
    if number < 0.0:
        raise Tier2MonthlyBaseReplayContractError(f"{label} must be non-negative")
    return number


def _positive_float(value: object, *, label: str) -> float:
    number = _number(value, label=label)
    if number <= 0.0:
        raise Tier2MonthlyBaseReplayContractError(f"{label} must be positive")
    return number


def _bounded_nonnegative_float(value: object, *, label: str) -> float:
    number = _nonnegative_float(value, label=label)
    if number > MAX_OBJECTIVE_WEIGHT:
        raise Tier2MonthlyBaseReplayContractError(
            f"{label} must be <= {MAX_OBJECTIVE_WEIGHT:.0e}"
        )
    return number


def _bounded_positive_float(value: object, *, label: str) -> float:
    number = _positive_float(value, label=label)
    if number > MAX_OBJECTIVE_WEIGHT:
        raise Tier2MonthlyBaseReplayContractError(
            f"{label} must be <= {MAX_OBJECTIVE_WEIGHT:.0e}"
        )
    return number


def _unit_float(value: object, *, label: str) -> float:
    number = _number(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise Tier2MonthlyBaseReplayContractError(f"{label} must be in [0, 1]")
    return number


def _integer_at_least(value: object, minimum: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Tier2MonthlyBaseReplayContractError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _optional_positive_float(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, label=label)


__all__ = [
    "Tier2MonthlyBaseReplayError",
    "Tier2MonthlyBaseReplayUnsupported",
    "Tier2MonthlyBaseReplayContractError",
    "Tier2MonthlyBaseReplayNumericalError",
    "Tier2MonthlyBaseReplayOutput",
    "validate_base_candidate_config",
]
