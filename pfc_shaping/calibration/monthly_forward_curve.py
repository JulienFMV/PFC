"""Monthly BASE forward curve constraints and data contracts.

This module is the first layer of the monthly forward curve reform.  It builds
an explicit month-level constraint system from EEX-style calendar, quarter and
monthly BASE quotes.  The constraints are hour-weighted with the delivery
timezone, so leap years and daylight-saving transitions are represented before
any optimizer is introduced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd

from pfc_shaping.calibration.constraints import (
    ConstraintRow,
    ConstraintSystem,
    FeasibilityReport,
)


MONTHLY_CURVE_SCHEMA_VERSION = "monthly_curve_constraints_v1"

_MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")
_QUARTER_RE = re.compile(r"^(?P<year>\d{4})-Q(?P<quarter>[1-4])$")
_YEAR_RE = re.compile(r"^(?P<year>\d{4})$")


@dataclass(frozen=True)
class MonthlyCurveConfig:
    lambda_prior: float = 1e-6
    lambda_smooth_month: float = 1.0
    lambda_smooth_yoy: float = 0.25
    lambda_shape: float = 1.0
    neighbor_shrinkage: float = 0.5
    robust_panel_quantile: float = 0.5
    min_history_snapshots: int = 24
    max_prior_residual_eur_mwh: float | None = None
    constraint_tolerance: float = 1e-9
    stationarity_tolerance: float = 1e-7


@dataclass(frozen=True)
class DeliveryGrid:
    months: pd.PeriodIndex
    timezone: str
    month_hours: pd.Series
    calendar: str = "CH"


@dataclass(frozen=True)
class MarketQuote:
    market: str
    product: str
    load_type: str
    price: float
    snapshot_date: pd.Timestamp | None = None
    source: str = ""
    available_at: pd.Timestamp | None = None

    def key(self) -> str:
        return f"{self.market.upper()}:{self.load_type.upper()}:{self.product}"


@dataclass(frozen=True)
class MonthlyCurveInputs:
    delivery_grid: DeliveryGrid
    own_quotes: tuple[MarketQuote, ...]
    neighbor_quotes: tuple[MarketQuote, ...]
    eex_history: pd.DataFrame
    run_timestamp: pd.Timestamp
    config: MonthlyCurveConfig
    source_hashes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MonthlyConstraintSystem:
    schema_version: str
    delivery_grid: DeliveryGrid
    constraints: ConstraintSystem
    rows: pd.DataFrame
    month_buckets: pd.Series
    bucket_targets: Mapping[str, float]
    quote_diagnostics: pd.DataFrame

    @property
    def matrix(self) -> np.ndarray:
        return self.constraints.matrix

    @property
    def targets(self) -> np.ndarray:
        return self.constraints.targets

    @property
    def names(self) -> list[str]:
        return self.constraints.names

    def residuals(self, values: np.ndarray) -> pd.DataFrame:
        return self.constraints.residuals(values)

    def feasibility_report(self, *, tolerance: float = 1e-9) -> FeasibilityReport:
        return self.constraints.feasibility_report(tolerance=tolerance)


@dataclass(frozen=True)
class MonthlyCurveResult:
    monthly_curve_schema_version: str
    monthly_curve: pd.Series
    constraints: pd.DataFrame
    residuals: pd.DataFrame
    priors: pd.DataFrame
    diagnostics: pd.DataFrame
    kkt: dict[str, float | int | bool]


def build_delivery_grid(
    delivery_months: Sequence[str | pd.Period] | pd.PeriodIndex,
    *,
    timezone: str = "Europe/Zurich",
    calendar: str = "CH",
) -> DeliveryGrid:
    months = _normalize_months(delivery_months)
    return DeliveryGrid(
        months=months,
        timezone=timezone,
        month_hours=month_delivery_hours(months, timezone=timezone),
        calendar=calendar,
    )


def month_delivery_hours(
    delivery_months: Sequence[str | pd.Period] | pd.PeriodIndex,
    *,
    timezone: str = "Europe/Zurich",
) -> pd.Series:
    """Return actual delivery hours for local calendar months."""

    months = _normalize_months(delivery_months)
    hours: list[float] = []
    for month in months:
        start = month.to_timestamp(how="start").tz_localize(timezone)
        end = (month + 1).to_timestamp(how="start").tz_localize(timezone)
        delta = end.tz_convert("UTC") - start.tz_convert("UTC")
        hours.append(float(delta / pd.Timedelta(hours=1)))
    return pd.Series(hours, index=months, dtype=float, name="month_hours")


def build_monthly_constraint_system(
    delivery_months: Sequence[str | pd.Period] | pd.PeriodIndex,
    own_quotes: Mapping[str, float] | Sequence[MarketQuote],
    *,
    timezone: str = "Europe/Zurich",
    market: str = "CH",
    load_type: str = "BASE",
    constraint_tolerance: float = 1e-9,
) -> MonthlyConstraintSystem:
    """Build hour-weighted non-overlapping monthly constraints.

    Products are selected by priority ``Month > Quarter > Calendar``.  Coarser
    products that overlap finer products create explicit residual buckets; fully
    covered coarser products are retained in diagnostics and must be consistent
    with the active rows.
    """

    grid = build_delivery_grid(delivery_months, timezone=timezone, calendar=market)
    quotes = _normalize_quotes(
        own_quotes,
        market=market,
        load_type=load_type,
        tolerance=constraint_tolerance,
    )
    quote_by_product = {
        quote.product: quote
        for quote in quotes
        if quote.load_type.upper() == load_type.upper()
    }

    month_buckets = pd.Series([pd.NA] * len(grid.months), index=grid.months, dtype=object, name="bucket")
    bucket_targets: dict[str, float] = {}
    bucket_parent: dict[str, str] = {}
    bucket_sources: dict[str, tuple[str, ...]] = {}
    quote_diag: list[dict[str, object]] = []

    for product in sorted(quote_by_product, key=_product_sort_key):
        quote = quote_by_product[product]
        product_months = _product_periods(product)
        overlap_mask = grid.months.isin(product_months)
        if not bool(overlap_mask.any()):
            quote_diag.append(_quote_diag_row(quote, active=False, dropped_reason="outside_delivery_grid"))
            continue
        _raise_on_partial_product_grid(product, grid.months)

        free_mask = overlap_mask & month_buckets.isna().to_numpy()
        if not bool(free_mask.any()):
            quote_diag.append(_quote_diag_row(quote, active=False, dropped_reason="redundant_consistent"))
            continue

        known_mask = overlap_mask & month_buckets.notna().to_numpy()
        bucket = product if not bool(known_mask.any()) else f"{product}-RESIDUAL"
        target = _residual_target(
            parent_product=product,
            parent_target=quote.price,
            parent_mask=overlap_mask,
            free_mask=free_mask,
            known_mask=known_mask,
            month_buckets=month_buckets,
            month_hours=grid.month_hours,
            bucket_targets=bucket_targets,
        )
        month_buckets.loc[grid.months[free_mask]] = bucket
        bucket_targets[bucket] = target
        bucket_parent[bucket] = product
        bucket_sources[bucket] = (quote.key(),)
        quote_diag.append(_quote_diag_row(quote, active=True, dropped_reason=""))

    _validate_all_quotes(
        quotes=tuple(quote_by_product.values()),
        grid=grid,
        month_buckets=month_buckets,
        bucket_targets=bucket_targets,
        tolerance=constraint_tolerance,
    )

    rows: list[ConstraintRow] = []
    row_meta: list[dict[str, object]] = []
    active_buckets = [bucket for bucket in month_buckets.dropna().drop_duplicates().tolist()]
    for row_index, bucket in enumerate(active_buckets):
        mask = (month_buckets == bucket).to_numpy()
        weights = np.zeros(len(grid.months), dtype=float)
        bucket_hours = float(grid.month_hours.loc[grid.months[mask]].sum())
        weights[mask] = grid.month_hours.loc[grid.months[mask]].to_numpy(dtype=float) / bucket_hours
        row_name = f"{load_type.upper()}:{bucket}"
        rows.append(
            ConstraintRow(
                name=row_name,
                target=bucket_targets[bucket],
                weights=weights,
                kind=load_type.upper(),
                metadata={
                    "bucket": bucket,
                    "parent_product": bucket_parent.get(bucket, bucket),
                    "source_quote_keys": bucket_sources.get(bucket, tuple()),
                    "hours": bucket_hours,
                },
            )
        )
        row_meta.append(
            {
                "row_index": row_index,
                "constraint_name": row_name,
                "product": bucket,
                "load_type": load_type.upper(),
                "target": bucket_targets[bucket],
                "active": True,
                "is_residual": bucket.endswith("-RESIDUAL"),
                "parent_product": bucket_parent.get(bucket, bucket),
                "source_quote_keys": "|".join(bucket_sources.get(bucket, tuple())),
                "dropped_reason": "",
                "active_row_indices": (row_index,),
                "n_months": int(mask.sum()),
                "hours": bucket_hours,
            }
        )

    row_index_by_bucket = {row["product"]: int(row["row_index"]) for row in row_meta}
    for diag in quote_diag:
        if diag["active"]:
            continue
        product = str(diag["product"])
        if diag["dropped_reason"] == "redundant_consistent":
            product_months = _product_periods(product)
            mask = grid.months.isin(product_months)
            covering = sorted(
                {
                    row_index_by_bucket[str(bucket)]
                    for bucket in month_buckets.loc[grid.months[mask]].dropna().unique()
                    if str(bucket) in row_index_by_bucket
                }
            )
            diag["active_row_indices"] = tuple(covering)

    system = ConstraintSystem(tuple(rows), n_variables=len(grid.months))
    report = system.feasibility_report(tolerance=constraint_tolerance)
    if not report.feasible:
        raise ValueError(
            "monthly constraint system is infeasible: "
            f"rank(A)={report.rank_a}, rank([A|q])={report.rank_augmented}, "
            f"inf_residual={report.infeasibility_inf:.12g}"
        )

    return MonthlyConstraintSystem(
        schema_version=MONTHLY_CURVE_SCHEMA_VERSION,
        delivery_grid=grid,
        constraints=system,
        rows=pd.DataFrame(row_meta),
        month_buckets=month_buckets,
        bucket_targets=dict(bucket_targets),
        quote_diagnostics=pd.DataFrame(quote_diag),
    )


def solve_monthly_forward_curve(
    *,
    market: str,
    delivery_months: Sequence[str | pd.Period] | pd.PeriodIndex,
    own_quotes: Mapping[str, float] | Sequence[MarketQuote],
    eex_history: pd.DataFrame | None = None,
    neighbor_markets: Sequence[str] = ("DE", "FR", "AT", "IT"),
    neighbor_quotes: Sequence[MarketQuote] | None = None,
    run_timestamp: pd.Timestamp | None = None,
    config: MonthlyCurveConfig | None = None,
    shape_prior: pd.Series | object | None = None,
    timezone: str = "Europe/Zurich",
) -> MonthlyCurveResult:
    """Solve the monthly BASE curve under hard own-market EEX constraints."""

    del eex_history, neighbor_markets, neighbor_quotes, run_timestamp
    constraints = build_monthly_constraint_system(
        delivery_months,
        own_quotes,
        timezone=timezone,
        market=market,
        load_type="BASE",
        constraint_tolerance=(config or MonthlyCurveConfig()).constraint_tolerance,
    )
    return solve_monthly_forward_curve_from_constraints(
        constraints,
        config=config,
        shape_prior=shape_prior,
    )


def solve_monthly_forward_curve_from_inputs(
    inputs: MonthlyCurveInputs,
    *,
    shape_prior: pd.Series | object | None = None,
) -> MonthlyCurveResult:
    """Contract-based entry point used by production and local export."""

    return solve_monthly_forward_curve_from_constraints(
        build_monthly_constraint_system(
            inputs.delivery_grid.months,
            inputs.own_quotes,
            timezone=inputs.delivery_grid.timezone,
            market=inputs.delivery_grid.calendar,
            load_type="BASE",
            constraint_tolerance=inputs.config.constraint_tolerance,
        ),
        config=inputs.config,
        shape_prior=shape_prior,
    )


def solve_monthly_forward_curve_from_constraints(
    monthly_constraints: MonthlyConstraintSystem,
    *,
    config: MonthlyCurveConfig | None = None,
    shape_prior: pd.Series | object | None = None,
) -> MonthlyCurveResult:
    """Solve a small equality-constrained quadratic monthly curve problem."""

    cfg = config or MonthlyCurveConfig()
    n = len(monthly_constraints.delivery_grid.months)
    a = monthly_constraints.matrix
    q = monthly_constraints.targets
    parent_flat = _parent_flat_baseline(monthly_constraints)
    parent_operator = _parent_mean_operator(monthly_constraints)
    shape_operator = np.eye(n) - parent_operator
    shape_target = _shape_prior_vector(shape_prior, monthly_constraints)
    shape_target = _recenter_vector_by_parent(shape_target, monthly_constraints)

    terms: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    ridge_used = False
    if cfg.lambda_prior > 0.0:
        terms.append(("parent_flat_prior", float(cfg.lambda_prior), np.eye(n), parent_flat))
    else:
        ridge_used = True
        terms.append(("numerical_parent_flat_ridge", 1e-10, np.eye(n), parent_flat))

    d2 = _second_difference_operator(monthly_constraints)
    if cfg.lambda_smooth_month > 0.0 and d2.size:
        terms.append(("smooth_month", float(cfg.lambda_smooth_month), d2, np.zeros(d2.shape[0])))

    dyoy, dropped_yoy = _yoy_shape_operator(monthly_constraints, shape_operator)
    if cfg.lambda_smooth_yoy > 0.0 and dyoy.size:
        terms.append(("smooth_yoy_shape", float(cfg.lambda_smooth_yoy), dyoy, np.zeros(dyoy.shape[0])))

    if cfg.lambda_shape > 0.0 and shape_prior is not None:
        terms.append(("shape_prior", float(cfg.lambda_shape), shape_operator, shape_target))

    q_matrix = np.zeros((n, n), dtype=float)
    c_vector = np.zeros(n, dtype=float)
    for _, weight, operator, target in terms:
        q_matrix += weight * (operator.T @ operator)
        c_vector -= weight * (operator.T @ target)

    kkt = np.block(
        [
            [q_matrix, a.T],
            [a, np.zeros((a.shape[0], a.shape[0]), dtype=float)],
        ]
    )
    rhs = np.concatenate([-c_vector, q])
    try:
        solution = np.linalg.solve(kkt, rhs)
        solved_by_lstsq = False
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        solved_by_lstsq = True

    x = solution[:n]
    multipliers = solution[n:]
    active_residual = a @ x - q
    stationarity = q_matrix @ x + c_vector + a.T @ multipliers
    objective_terms = {
        name: float(0.5 * weight * np.sum((operator @ x - target) ** 2))
        for name, weight, operator, target in terms
    }
    residuals = monthly_constraints.residuals(x)
    priors = pd.DataFrame(
        {
            "month": [str(month) for month in monthly_constraints.delivery_grid.months],
            "parent_bucket": monthly_constraints.month_buckets.astype(str).to_list(),
            "parent_flat_eur_mwh": parent_flat,
            "shape_prior_eur_mwh": shape_target,
            "monthly_curve_eur_mwh": x,
        }
    )
    diagnostics = pd.DataFrame(
        [
            {"metric": "max_abs_constraint_residual", "value": float(np.max(np.abs(active_residual))) if len(q) else 0.0},
            {"metric": "stationarity_residual", "value": float(np.max(np.abs(stationarity))) if len(stationarity) else 0.0},
            {"metric": "condition_number", "value": _condition_number(kkt)},
            {"metric": "active_constraint_rank", "value": float(np.linalg.matrix_rank(a)) if a.size else 0.0},
            {"metric": "nullspace_dimension", "value": float(n - (np.linalg.matrix_rank(a) if a.size else 0))},
            {"metric": "dropped_yoy_rows", "value": float(dropped_yoy)},
        ]
        + [{"metric": f"objective_{name}", "value": value} for name, value in objective_terms.items()]
    )
    return MonthlyCurveResult(
        monthly_curve_schema_version=MONTHLY_CURVE_SCHEMA_VERSION,
        monthly_curve=pd.Series(x, index=monthly_constraints.delivery_grid.months, name="monthly_base_eur_mwh"),
        constraints=monthly_constraints.rows.copy(),
        residuals=residuals,
        priors=priors,
        diagnostics=diagnostics,
        kkt={
            "max_abs_constraint_residual": float(np.max(np.abs(active_residual))) if len(q) else 0.0,
            "stationarity_residual": float(np.max(np.abs(stationarity))) if len(stationarity) else 0.0,
            "condition_number": _condition_number(kkt),
            "active_constraint_rank": int(np.linalg.matrix_rank(a)) if a.size else 0,
            "nullspace_dimension": int(n - (np.linalg.matrix_rank(a) if a.size else 0)),
            "ridge_used": bool(ridge_used),
            "solved_by_lstsq": bool(solved_by_lstsq),
            "dropped_yoy_rows": int(dropped_yoy),
        },
    )


def _normalize_months(
    delivery_months: Sequence[str | pd.Period] | pd.PeriodIndex,
) -> pd.PeriodIndex:
    if isinstance(delivery_months, pd.PeriodIndex):
        months = delivery_months
    else:
        months = pd.PeriodIndex(delivery_months, freq="M")
    months = months.asfreq("M")
    if months.has_duplicates:
        raise ValueError("delivery months must be unique")
    if not months.is_monotonic_increasing:
        months = months.sort_values()
    return months


def _normalize_quotes(
    own_quotes: Mapping[str, float] | Sequence[MarketQuote],
    *,
    market: str,
    load_type: str,
    tolerance: float,
) -> tuple[MarketQuote, ...]:
    if isinstance(own_quotes, Mapping):
        quotes = tuple(
            MarketQuote(
                market=market,
                product=str(product),
                load_type=load_type,
                price=float(price),
                source="mapping",
            )
            for product, price in own_quotes.items()
        )
    else:
        quotes = tuple(own_quotes)

    by_product: dict[tuple[str, str], MarketQuote] = {}
    for quote in quotes:
        if quote.market.upper() != market.upper():
            raise ValueError(
                f"unexpected quote market {quote.market!r} for target market {market!r}"
            )
        _parse_product(quote.product)
        price = float(quote.price)
        if not np.isfinite(price):
            raise ValueError(f"non-finite quote price for {quote.key()}")
        key = (quote.load_type.upper(), quote.product)
        previous = by_product.get(key)
        if previous is not None:
            if abs(float(previous.price) - price) > tolerance:
                raise ValueError(
                    f"conflicting duplicate quote for {quote.product}: "
                    f"{previous.price} vs {quote.price}"
                )
            continue
        by_product[key] = quote
    return tuple(by_product.values())


def _parse_product(product: str) -> tuple[str, int, int | None]:
    product = str(product)
    month_match = _MONTH_RE.match(product)
    if month_match:
        month = int(month_match.group("month"))
        if month < 1 or month > 12:
            raise ValueError(f"unsupported EEX month product {product!r}")
        return ("month", int(month_match.group("year")), month)
    quarter_match = _QUARTER_RE.match(product)
    if quarter_match:
        return ("quarter", int(quarter_match.group("year")), int(quarter_match.group("quarter")))
    year_match = _YEAR_RE.match(product)
    if year_match:
        return ("year", int(year_match.group("year")), None)
    raise ValueError(f"unsupported EEX product key {product!r}")


def _product_sort_key(product: str) -> tuple[int, int, int]:
    kind, year, part = _parse_product(product)
    priority = {"month": 0, "quarter": 1, "year": 2}[kind]
    return (priority, year, int(part or 0))


def _product_periods(product: str) -> pd.PeriodIndex:
    kind, year, part = _parse_product(product)
    if kind == "month":
        return pd.PeriodIndex([f"{year}-{int(part):02d}"], freq="M")
    if kind == "quarter":
        first = (int(part) - 1) * 3 + 1
        return pd.PeriodIndex([f"{year}-{month:02d}" for month in range(first, first + 3)], freq="M")
    return pd.PeriodIndex([f"{year}-{month:02d}" for month in range(1, 13)], freq="M")


def product_periods(product: str) -> pd.PeriodIndex:
    """Return the full monthly delivery set covered by an EEX product key."""

    return _product_periods(product)


def _raise_on_partial_product_grid(product: str, delivery_months: pd.PeriodIndex) -> None:
    product_months = _product_periods(product)
    product_set = set(product_months)
    grid_set = set(delivery_months)
    overlap = product_set & grid_set
    if overlap and not product_set <= grid_set:
        missing = sorted(str(month) for month in product_set - grid_set)
        raise ValueError(f"partial delivery grid for quoted product {product}: missing {missing}")


def _residual_target(
    *,
    parent_product: str,
    parent_target: float,
    parent_mask: np.ndarray,
    free_mask: np.ndarray,
    known_mask: np.ndarray,
    month_buckets: pd.Series,
    month_hours: pd.Series,
    bucket_targets: Mapping[str, float],
) -> float:
    if not bool(known_mask.any()):
        return float(parent_target)

    known_buckets = month_buckets.loc[month_buckets.index[known_mask]]
    known_targets = known_buckets.map(lambda bucket: bucket_targets[str(bucket)]).astype(float)
    known_hours = month_hours.loc[month_hours.index[known_mask]].astype(float)
    parent_hours = float(month_hours.loc[month_hours.index[parent_mask]].sum())
    free_hours = float(month_hours.loc[month_hours.index[free_mask]].sum())
    if free_hours <= 0.0:
        raise ValueError(f"empty residual EEX bucket under {parent_product}")
    known_energy = float((known_targets.to_numpy(dtype=float) * known_hours.to_numpy(dtype=float)).sum())
    residual_energy = float(parent_target) * parent_hours - known_energy
    target = residual_energy / free_hours
    if not np.isfinite(target):
        raise ValueError(f"non-finite residual EEX target under {parent_product}")
    return float(target)


def _validate_all_quotes(
    *,
    quotes: tuple[MarketQuote, ...],
    grid: DeliveryGrid,
    month_buckets: pd.Series,
    bucket_targets: Mapping[str, float],
    tolerance: float,
) -> None:
    for quote in quotes:
        product_months = _product_periods(quote.product)
        mask = grid.months.isin(product_months)
        if not bool(mask.any()):
            continue
        if bool(month_buckets.loc[grid.months[mask]].isna().any()):
            raise ValueError(f"quoted product {quote.product} is not fully represented by active rows")
        month_targets = (
            month_buckets.loc[grid.months[mask]]
            .map(lambda bucket: bucket_targets[str(bucket)])
            .astype(float)
        )
        hours = grid.month_hours.loc[grid.months[mask]].astype(float)
        implied = float((month_targets.to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / hours.sum())
        if abs(implied - float(quote.price)) > tolerance:
            raise ValueError(
                f"inconsistent quoted product {quote.product}: target={float(quote.price):.12g}, "
                f"implied={implied:.12g}, diff={implied - float(quote.price):.12g}"
            )


def _quote_diag_row(quote: MarketQuote, *, active: bool, dropped_reason: str) -> dict[str, object]:
    return {
        "quote_key": quote.key(),
        "market": quote.market.upper(),
        "load_type": quote.load_type.upper(),
        "product": quote.product,
        "target": float(quote.price),
        "active": bool(active),
        "dropped_reason": dropped_reason,
        "active_row_indices": tuple(),
    }


def _parent_flat_baseline(monthly_constraints: MonthlyConstraintSystem) -> np.ndarray:
    values = np.zeros(len(monthly_constraints.delivery_grid.months), dtype=float)
    for i, month in enumerate(monthly_constraints.delivery_grid.months):
        bucket = monthly_constraints.month_buckets.loc[month]
        if pd.isna(bucket):
            continue
        values[i] = float(monthly_constraints.bucket_targets[str(bucket)])
    missing = monthly_constraints.month_buckets.isna().to_numpy()
    if bool(missing.any()) and bool((~missing).any()):
        hours = monthly_constraints.delivery_grid.month_hours.to_numpy(dtype=float)
        represented_hours = hours[~missing]
        represented_values = values[~missing]
        total_hours = float(represented_hours.sum())
        if total_hours > 0.0:
            values[missing] = float((represented_values * represented_hours).sum() / total_hours)
    return values


def _parent_mean_operator(monthly_constraints: MonthlyConstraintSystem) -> np.ndarray:
    n = len(monthly_constraints.delivery_grid.months)
    operator = np.eye(n, dtype=float)
    for bucket in monthly_constraints.month_buckets.dropna().drop_duplicates():
        mask = monthly_constraints.month_buckets.eq(bucket).to_numpy()
        idx = np.flatnonzero(mask)
        hours = monthly_constraints.delivery_grid.month_hours.iloc[idx].to_numpy(dtype=float)
        total = float(hours.sum())
        if total <= 0.0:
            continue
        operator[idx, :] = 0.0
        operator[np.ix_(idx, idx)] = np.repeat((hours / total).reshape(1, -1), len(idx), axis=0)
    return operator


def _shape_prior_vector(
    shape_prior: pd.Series | object | None,
    monthly_constraints: MonthlyConstraintSystem,
) -> np.ndarray:
    n = len(monthly_constraints.delivery_grid.months)
    if shape_prior is None:
        return np.zeros(n, dtype=float)
    if isinstance(shape_prior, pd.Series):
        series = shape_prior
    elif hasattr(shape_prior, "shape"):
        series = getattr(shape_prior, "shape")
        if not isinstance(series, pd.Series):
            raise TypeError("shape_prior.shape must be a pandas Series")
    else:
        raise TypeError("shape_prior must be a pandas Series or an object exposing a Series named 'shape'")
    return series.reindex(monthly_constraints.delivery_grid.months).fillna(0.0).to_numpy(dtype=float)


def _recenter_vector_by_parent(
    values: np.ndarray,
    monthly_constraints: MonthlyConstraintSystem,
) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for bucket in monthly_constraints.month_buckets.dropna().drop_duplicates():
        mask = monthly_constraints.month_buckets.eq(bucket).to_numpy()
        idx = np.flatnonzero(mask)
        hours = monthly_constraints.delivery_grid.month_hours.iloc[idx].to_numpy(dtype=float)
        total = float(hours.sum())
        if total <= 0.0:
            continue
        mean = float((out[idx] * hours).sum() / total)
        out[idx] -= mean
    return out


def _second_difference_operator(monthly_constraints: MonthlyConstraintSystem) -> np.ndarray:
    rows: list[np.ndarray] = []
    months = monthly_constraints.delivery_grid.months
    month_buckets = monthly_constraints.month_buckets.reindex(months)
    n = len(months)
    for i in range(1, n - 1):
        if months[i - 1] + 1 != months[i] or months[i] + 1 != months[i + 1]:
            continue
        buckets = {
            str(month_buckets.iloc[i - 1]),
            str(month_buckets.iloc[i]),
            str(month_buckets.iloc[i + 1]),
        }
        if len(buckets) != 1:
            continue
        row = np.zeros(n, dtype=float)
        row[i - 1] = 1.0
        row[i] = -2.0
        row[i + 1] = 1.0
        rows.append(row)
    if not rows:
        return np.zeros((0, n), dtype=float)
    return np.vstack(rows)


def _yoy_shape_operator(
    monthly_constraints: MonthlyConstraintSystem,
    shape_operator: np.ndarray,
) -> tuple[np.ndarray, int]:
    months = monthly_constraints.delivery_grid.months
    month_to_pos = {month: i for i, month in enumerate(months)}
    parent_types = _parent_block_types(monthly_constraints)
    rows: list[np.ndarray] = []
    dropped = 0
    for i, month in enumerate(months):
        previous = month - 12
        if previous not in month_to_pos:
            continue
        j = month_to_pos[previous]
        if parent_types[i] != parent_types[j]:
            comparable = _residual_calendar_yoy_shape_row(
                monthly_constraints,
                current_pos=i,
                previous_pos=j,
                parent_types=parent_types,
            )
            if comparable is None:
                dropped += 1
                continue
            rows.append(comparable)
            continue
        rows.append(shape_operator[i, :] - shape_operator[j, :])
    if not rows:
        return np.zeros((0, len(months)), dtype=float), dropped
    return np.vstack(rows), dropped


def _residual_calendar_yoy_shape_row(
    monthly_constraints: MonthlyConstraintSystem,
    *,
    current_pos: int,
    previous_pos: int,
    parent_types: list[str],
) -> np.ndarray | None:
    type_current = parent_types[current_pos]
    type_previous = parent_types[previous_pos]
    if {type_current, type_previous} != {"residual", "calendar"}:
        return None

    months = monthly_constraints.delivery_grid.months
    residual_pos = current_pos if type_current == "residual" else previous_pos
    residual_bucket = monthly_constraints.month_buckets.iloc[residual_pos]
    if pd.isna(residual_bucket):
        return None
    residual_mask = monthly_constraints.month_buckets.eq(residual_bucket).to_numpy()
    residual_indices = np.flatnonzero(residual_mask)
    residual_month_numbers = {int(months[pos].month) for pos in residual_indices}
    if not residual_month_numbers:
        return None

    def comparable_block(position: int) -> np.ndarray | None:
        year = int(months[position].year)
        if parent_types[position] == "residual":
            bucket = monthly_constraints.month_buckets.iloc[position]
            mask = monthly_constraints.month_buckets.eq(bucket).to_numpy()
            return np.flatnonzero(mask)
        if parent_types[position] == "calendar":
            indices = np.array(
                [
                    idx
                    for idx, month in enumerate(months)
                    if int(month.year) == year and int(month.month) in residual_month_numbers
                ],
                dtype=int,
            )
            if len(indices) != len(residual_month_numbers):
                return None
            return indices
        return None

    current_block = comparable_block(current_pos)
    previous_block = comparable_block(previous_pos)
    if current_block is None or previous_block is None or len(current_block) == 0 or len(previous_block) == 0:
        return None

    row = np.zeros(len(months), dtype=float)
    row[current_pos] += 1.0
    row[previous_pos] -= 1.0
    row -= _weighted_mean_row(monthly_constraints, current_block)
    row += _weighted_mean_row(monthly_constraints, previous_block)
    return row


def _weighted_mean_row(monthly_constraints: MonthlyConstraintSystem, indices: np.ndarray) -> np.ndarray:
    row = np.zeros(len(monthly_constraints.delivery_grid.months), dtype=float)
    hours = monthly_constraints.delivery_grid.month_hours.iloc[indices].to_numpy(dtype=float)
    total = float(hours.sum())
    if total <= 0.0:
        return row
    row[indices] = hours / total
    return row


def _parent_block_types(monthly_constraints: MonthlyConstraintSystem) -> list[str]:
    parent_by_bucket: dict[str, str] = {}
    if not monthly_constraints.rows.empty:
        parent_by_bucket = {
            str(row["product"]): str(row["parent_product"])
            for _, row in monthly_constraints.rows.iterrows()
        }
    out: list[str] = []
    for month in monthly_constraints.delivery_grid.months:
        bucket = monthly_constraints.month_buckets.loc[month]
        if pd.isna(bucket):
            out.append("unassigned")
            continue
        bucket_text = str(bucket)
        parent = parent_by_bucket.get(bucket_text, bucket_text)
        if bucket_text.endswith("-RESIDUAL"):
            out.append("residual")
        elif "-Q" in parent:
            out.append("quarter")
        elif parent.isdigit():
            out.append("calendar")
        elif _MONTH_RE.match(parent):
            out.append("month")
        else:
            out.append("unknown")
    return out


def _condition_number(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")
