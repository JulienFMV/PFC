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
    lambda_prior: float = 0.0
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
