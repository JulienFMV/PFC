"""Shape priors for the monthly forward curve solver.

All priors in this module are expressed as EUR/MWh deviations from the CH
parent block represented by ``MonthlyConstraintSystem.month_buckets``.  They
must therefore have an hour-weighted mean of zero inside each active CH parent
block.  This is the no-level-leakage contract: external markets and history can
shape unquoted degrees of freedom, but they cannot set the CH level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyConstraintSystem,
    month_delivery_hours,
)


DEFAULT_CH_STRUCTURAL_MONTHLY_RATIOS: dict[int, float] = {
    1: 1.20,
    2: 1.18,
    3: 1.05,
    4: 0.95,
    5: 0.85,
    6: 0.82,
    7: 0.85,
    8: 0.85,
    9: 1.00,
    10: 1.05,
    11: 1.10,
    12: 1.15,
}

_MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")
_QUARTER_RE = re.compile(r"^(?P<year>\d{4})-Q(?P<quarter>[1-4])$")
_YEAR_RE = re.compile(r"^(?P<year>\d{4})$")


@dataclass(frozen=True)
class MonthlyShapePrior:
    """A monthly zero-mean shape prior plus diagnostics."""

    shape: pd.Series
    diagnostics: pd.DataFrame
    contributions: pd.DataFrame
    status: str


def recenter_shape_by_parent(
    shape: pd.Series | Mapping[pd.Period, float],
    constraints: MonthlyConstraintSystem,
) -> pd.Series:
    """Recenter a shape so each active CH parent bucket has zero weighted mean."""

    months = constraints.delivery_grid.months
    out = pd.Series(shape, dtype=float).reindex(months).fillna(0.0).astype(float)
    out.index = months
    for bucket in constraints.month_buckets.dropna().drop_duplicates():
        mask = constraints.month_buckets.eq(bucket)
        idx = constraints.month_buckets.index[mask]
        hours = constraints.delivery_grid.month_hours.loc[idx].astype(float)
        total = float(hours.sum())
        if total <= 0.0:
            continue
        mean = float((out.loc[idx].to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / total)
        out.loc[idx] = out.loc[idx] - mean
    out.name = "shape_deviation_eur_mwh"
    return out


def _parent_mean_diagnostics(
    values: pd.Series,
    constraints: MonthlyConstraintSystem,
) -> pd.DataFrame:
    months = constraints.delivery_grid.months
    series = pd.Series(values, dtype=float).reindex(months).fillna(0.0)
    rows: list[dict[str, object]] = []
    for month in months:
        bucket = constraints.month_buckets.loc[month]
        if pd.isna(bucket):
            rows.append(
                {
                    "month": str(month),
                    "parent_bucket": "",
                    "parent_hours": np.nan,
                    "parent_weighted_mean_eur_mwh": np.nan,
                }
            )
            continue
        mask = constraints.month_buckets.eq(bucket)
        idx = constraints.month_buckets.index[mask]
        hours = constraints.delivery_grid.month_hours.loc[idx].astype(float)
        total = float(hours.sum())
        if total <= 0.0:
            mean = np.nan
        else:
            mean = float((series.loc[idx].to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / total)
        rows.append(
            {
                "month": str(month),
                "parent_bucket": str(bucket),
                "parent_hours": total,
                "parent_weighted_mean_eur_mwh": mean,
            }
        )
    return pd.DataFrame(rows).set_index("month")


def _cap_zero_mean_by_parent(
    shape: pd.Series,
    constraints: MonthlyConstraintSystem,
    cap_eur_mwh: float,
) -> pd.Series:
    """Apply an absolute cap while preserving each parent-block weighted mean."""

    cap = abs(float(cap_eur_mwh))
    months = constraints.delivery_grid.months
    out = pd.Series(shape, dtype=float).reindex(months).fillna(0.0).astype(float)
    out.index = months
    for bucket in constraints.month_buckets.dropna().drop_duplicates():
        mask = constraints.month_buckets.eq(bucket)
        idx = constraints.month_buckets.index[mask]
        values = out.loc[idx].to_numpy(dtype=float)
        hours = constraints.delivery_grid.month_hours.loc[idx].to_numpy(dtype=float)
        if len(values) == 0 or float(hours.sum()) <= 0.0:
            continue
        if np.nanmax(np.abs(values)) <= cap:
            continue

        def weighted_mean(shift: float) -> float:
            clipped = np.clip(values - shift, -cap, cap)
            return float(np.average(clipped, weights=hours))

        lo = float(np.nanmin(values) - cap - 1.0)
        hi = float(np.nanmax(values) + cap + 1.0)
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if weighted_mean(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        out.loc[idx] = np.clip(values - ((lo + hi) / 2.0), -cap, cap)
    out.name = "shape_deviation_eur_mwh"
    return out


def quote_coverage_by_horizon(
    eex_history: pd.DataFrame,
    *,
    load_type: str = "BASE",
) -> pd.DataFrame:
    """Count monthly EEX quotes by market and delivery-year horizon."""

    if eex_history.empty:
        return pd.DataFrame(columns=["market", "horizon_bucket", "monthly_quote_count"])
    _require_columns(eex_history, {"date", "product", "load_type", "market"})
    df = eex_history.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["market"] = df["market"].astype(str).str.upper()
    df = df[df["load_type"].astype(str).str.upper().eq(str(load_type).upper())]
    products = df["product"].astype(str)
    month_mask = products.str.match(_MONTH_RE)
    df = df[month_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["market", "horizon_bucket", "monthly_quote_count"])
    df["delivery_year"] = df["product"].astype(str).str.slice(0, 4).astype(int)
    df["horizon"] = df["delivery_year"] - df["date"].dt.year.astype(int)
    df["horizon_bucket"] = df["horizon"].map(_horizon_bucket)
    markets = sorted(df["market"].dropna().astype(str).str.upper().unique())
    buckets = ["h+0", "h+1", "h+2", "h+3+"]
    index = pd.MultiIndex.from_product([markets, buckets], names=["market", "horizon_bucket"])
    out = (
        df.groupby(["market", "horizon_bucket"], sort=True)
        .size()
        .reindex(index, fill_value=0)
        .rename("monthly_quote_count")
        .reset_index()
    )
    return out[["market", "horizon_bucket", "monthly_quote_count"]]


def build_neighbor_panel_shape_prior(
    constraints: MonthlyConstraintSystem,
    neighbor_quotes: Mapping[str, Mapping[str, float]] | Sequence[MarketQuote],
    *,
    neighbor_markets: Sequence[str] = ("DE", "FR", "AT", "IT"),
    load_type: str = "BASE",
    neighbor_shrinkage: float = 0.5,
    run_timestamp: pd.Timestamp | None = None,
) -> MonthlyShapePrior:
    """Build a robust zero-mean shape prior from neighboring EEX quotes."""

    prices_by_market = _prices_by_market(neighbor_quotes, load_type=load_type)
    market_shapes: dict[str, pd.Series] = {}
    diagnostic_rows: list[dict[str, object]] = []
    for market in [str(m).upper() for m in neighbor_markets]:
        prices = prices_by_market.get(market, {})
        if not prices:
            diagnostic_rows.append(_market_diag(market, status="NO_QUOTES"))
            continue
        raw, evidence = _raw_neighbor_month_values(constraints, prices)
        shape = _recenter_market_raw_values(raw, constraints)
        usable = int(shape.notna().sum())
        month_count = int((evidence == "month").sum())
        quarter_count = int((evidence == "quarter").sum())
        calendar_count = int((evidence == "calendar").sum())
        block_count = int(((evidence == "quarter") | (evidence == "calendar")).sum())
        horizon_counts = _evidence_counts_by_horizon(
            evidence,
            constraints.delivery_grid.months,
            run_timestamp=run_timestamp,
        )
        if usable > 0 and month_count == 0 and quarter_count == 0:
            diagnostic_rows.append(
                _market_diag(
                    market,
                    status="CALENDAR_LEVEL_ONLY_UNSUPPORTED",
                    covered_months=0,
                    direct_month_quotes=0,
                    block_shape_months=0,
                    horizon_months=len(constraints.delivery_grid.months),
                    quarter_quotes=0,
                    calendar_quotes=calendar_count,
                    **horizon_counts,
                )
            )
            continue
        if usable == 0:
            diagnostic_rows.append(
                _market_diag(
                    market,
                    status="UNUSABLE_PARENT_COVERAGE",
                    covered_months=0,
                    direct_month_quotes=month_count,
                    block_shape_months=block_count,
                    horizon_months=len(constraints.delivery_grid.months),
                    quarter_quotes=quarter_count,
                    calendar_quotes=calendar_count,
                    **horizon_counts,
                )
            )
            continue
        market_shapes[market] = shape
        status = "MONTH_SHAPE_USED" if month_count > 0 else "BLOCK_SHAPE_USED"
        diagnostic_rows.append(
            _market_diag(
                market,
                status=status,
                covered_months=usable,
                direct_month_quotes=month_count,
                block_shape_months=block_count,
                horizon_months=len(constraints.delivery_grid.months),
                quarter_quotes=quarter_count,
                calendar_quotes=calendar_count,
                **horizon_counts,
            )
        )

    contributions = pd.DataFrame(market_shapes, index=constraints.delivery_grid.months)
    if contributions.empty:
        zero = pd.Series(0.0, index=constraints.delivery_grid.months, name="shape_deviation_eur_mwh")
        return MonthlyShapePrior(zero, pd.DataFrame(diagnostic_rows), contributions, "UNSUPPORTED")

    usable_markets = [col for col in contributions.columns if contributions[col].notna().any()]
    monthly_markets = [
        str(row["market"])
        for row in diagnostic_rows
        if row.get("status") == "MONTH_SHAPE_USED" and str(row["market"]) in usable_markets
    ]
    block_markets = [
        str(row["market"])
        for row in diagnostic_rows
        if row.get("status") == "BLOCK_SHAPE_USED" and str(row["market"]) in usable_markets
    ]
    combined = contributions.median(axis=1, skipna=True).fillna(0.0)
    full_monthly_markets = [
        str(row["market"])
        for row in diagnostic_rows
        if row.get("status") == "MONTH_SHAPE_USED"
        and int(row.get("direct_month_quotes", 0)) >= len(constraints.delivery_grid.months)
        and str(row["market"]) in usable_markets
    ]
    if len(full_monthly_markets) >= 2:
        status = "PANEL_MULTI_MARKET"
    elif len(monthly_markets) >= 2:
        keep = 1.0 - min(1.0, max(0.0, float(neighbor_shrinkage))) * 0.5
        combined = combined * keep
        status = "PARTIAL_MONTHLY_PANEL"
    elif len(monthly_markets) == 1:
        keep = 1.0 - min(1.0, max(0.0, float(neighbor_shrinkage)))
        combined = combined * keep
        status = "DE_SINGLE_MARKET_MONTHLY" if monthly_markets[0] == "DE" else "SINGLE_MARKET_MONTHLY"
    elif block_markets:
        keep = 1.0 - min(1.0, max(0.0, float(neighbor_shrinkage)))
        combined = combined * keep
        status = "PANEL_BLOCK_SHAPE"
    else:
        status = "UNSUPPORTED"

    shape = recenter_shape_by_parent(combined, constraints)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["prior_status"] = status
    prior_far_status = _far_horizon_monthly_evidence_status(diagnostics)
    diagnostics["prior_far_horizon_monthly_evidence"] = prior_far_status
    diagnostics["market_far_horizon_monthly_evidence"] = diagnostics.apply(
        _market_far_horizon_monthly_evidence_status,
        axis=1,
    )
    diagnostics["far_horizon_monthly_evidence"] = prior_far_status
    return MonthlyShapePrior(shape, diagnostics, contributions, status)


def build_history_shape_prior(
    constraints: MonthlyConstraintSystem,
    eex_history: pd.DataFrame,
    *,
    market: str = "CH",
    load_type: str = "BASE",
    run_timestamp: pd.Timestamp | None = None,
    min_snapshots: int = 24,
    lookback_years: int | None = None,
) -> MonthlyShapePrior:
    """Estimate CH forward-market month-vs-parent deviations from history."""

    months = constraints.delivery_grid.months
    if eex_history.empty:
        return _unsupported_prior(months, "UNSUPPORTED")
    _require_columns(eex_history, {"date", "product", "load_type", "market", "price"})
    hist = _prepare_history(
        eex_history,
        market=market,
        load_type=load_type,
        run_timestamp=run_timestamp,
        lookback_years=lookback_years,
    )
    if hist.empty:
        return _unsupported_prior(months, "UNSUPPORTED")

    snapshot_prices = {
        date: dict(zip(group["product"].astype(str), group["price"].astype(float)))
        for date, group in hist.groupby("date", sort=True)
    }
    bucket_parent = _bucket_parent_map(constraints)
    rows: list[dict[str, object]] = []
    values: dict[pd.Period, float] = {}
    for month in months:
        bucket = constraints.month_buckets.loc[month]
        if pd.isna(bucket):
            rows.append(_history_diag(month, "", "UNSUPPORTED", 0, np.nan, np.nan))
            values[month] = 0.0
            continue
        parent_product = bucket_parent.get(str(bucket), str(bucket))
        deviations = _historical_deviations_for_month(
            month=month,
            parent_bucket=str(bucket),
            parent_product=parent_product,
            constraints=constraints,
            snapshot_prices=snapshot_prices,
        )
        n = len(deviations)
        if n < int(min_snapshots):
            rows.append(_history_diag(month, str(bucket), "UNSUPPORTED", n, np.nan, np.nan))
            values[month] = 0.0
            continue
        arr = np.asarray(deviations, dtype=float)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        rows.append(_history_diag(month, str(bucket), "USED", n, median, mad))
        values[month] = median

    diagnostics = pd.DataFrame(rows)
    shape = recenter_shape_by_parent(pd.Series(values), constraints)
    if (diagnostics["status"] == "USED").all():
        status = "HISTORY_FORWARD"
    elif (diagnostics["status"] == "USED").any():
        status = "PARTIAL_HISTORY_FORWARD"
    else:
        status = "UNSUPPORTED"
    return MonthlyShapePrior(shape, diagnostics, pd.DataFrame(index=months), status)


def build_structural_monthly_shape_prior(
    constraints: MonthlyConstraintSystem,
    *,
    monthly_ratios: Mapping[int, float] | None = None,
    amplitude_eur_mwh: float = 20.0,
    cap_eur_mwh: float | None = None,
    shrinkage: float = 0.0,
    fallback_reason: str | None = None,
    history_counts: Mapping[int, int] | None = None,
) -> MonthlyShapePrior:
    """Build an explicit template structural prior in zero-mean space."""

    amplitude = float(amplitude_eur_mwh)
    if not np.isfinite(amplitude):
        raise ValueError("amplitude_eur_mwh must be finite")
    if cap_eur_mwh is not None:
        cap_value = float(cap_eur_mwh)
        if not np.isfinite(cap_value) or cap_value <= 0.0:
            raise ValueError("cap_eur_mwh must be positive and finite")
    else:
        cap_value = np.nan
    shrink = float(shrinkage)
    if not np.isfinite(shrink) or shrink < 0.0 or shrink > 1.0:
        raise ValueError("shrinkage must be finite and in [0, 1]")
    ratios = dict(monthly_ratios or DEFAULT_CH_STRUCTURAL_MONTHLY_RATIOS)
    for month_number, ratio in ratios.items():
        if not np.isfinite(float(ratio)):
            raise ValueError(f"monthly ratio for month {month_number!r} must be finite")
    counts = {int(k): int(v) for k, v in dict(history_counts or {}).items()}
    months = constraints.delivery_grid.months
    raw = pd.Series(
        {
            month: (float(ratios.get(int(month.month), 1.0)) - 1.0) * amplitude
            for month in months
        },
        index=months,
        dtype=float,
        name="raw_deviation_eur_mwh",
    )
    raw_parent = _parent_mean_diagnostics(raw, constraints)
    recentered = recenter_shape_by_parent(raw, constraints).rename("recentered_deviation_eur_mwh")
    pre_cap = (recentered * (1.0 - shrink)).rename("pre_cap_deviation_eur_mwh")
    if cap_eur_mwh is None:
        shape = pre_cap.rename("shape_deviation_eur_mwh")
    else:
        shape = _cap_zero_mean_by_parent(pre_cap, constraints, cap_value)
    shape_parent = _parent_mean_diagnostics(shape, constraints)
    was_capped = shape.ne(pre_cap)
    max_abs_parent_mean_residual = float(
        np.nanmax(np.abs(shape_parent["parent_weighted_mean_eur_mwh"].to_numpy(dtype=float)))
    )
    diagnostics = pd.DataFrame(
        {
            "source": "template_structural_monthly_ratios",
            "month": [str(month) for month in months],
            "month_number": [int(month.month) for month in months],
            "parent_bucket": [str(raw_parent.loc[str(month), "parent_bucket"]) for month in months],
            "parent_hours": [float(raw_parent.loc[str(month), "parent_hours"]) for month in months],
            "ratio": [float(ratios.get(int(month.month), 1.0)) for month in months],
            "amplitude_eur_mwh": amplitude,
            "raw_deviation_eur_mwh": [float(raw.loc[month]) for month in months],
            "pre_recenter_parent_mean_eur_mwh": [
                float(raw_parent.loc[str(month), "parent_weighted_mean_eur_mwh"]) for month in months
            ],
            "recentered_deviation_eur_mwh": [float(recentered.loc[month]) for month in months],
            "recenter_adjustment_eur_mwh": [float(recentered.loc[month] - raw.loc[month]) for month in months],
            "shrinkage": shrink,
            "pre_cap_deviation_eur_mwh": [float(pre_cap.loc[month]) for month in months],
            "cap_eur_mwh": cap_value,
            "was_capped": [bool(was_capped.loc[month]) for month in months],
            "cap_adjustment_eur_mwh": [float(shape.loc[month] - pre_cap.loc[month]) for month in months],
            "shape_deviation_eur_mwh": [float(shape.loc[month]) for month in months],
            "shape_parent_mean_eur_mwh": [
                float(shape_parent.loc[str(month), "parent_weighted_mean_eur_mwh"]) for month in months
            ],
            "max_abs_parent_mean_residual": max_abs_parent_mean_residual,
            "zero_mean_parent_space": True,
            "fallback_reason": fallback_reason or "",
            "n_history": [int(counts.get(int(month.month), 0)) for month in months],
            "status": "STRUCTURAL_TEMPLATE",
        }
    )
    contributions = pd.DataFrame(
        {
            "raw_deviation_eur_mwh": raw,
            "recentered_deviation_eur_mwh": recentered,
            "pre_cap_deviation_eur_mwh": pre_cap,
            "shape_deviation_eur_mwh": shape,
        },
        index=months,
    )
    return MonthlyShapePrior(
        shape,
        diagnostics,
        contributions,
        "STRUCTURAL_TEMPLATE",
    )


def build_structural_monthly_shape_prior_from_history(
    constraints: MonthlyConstraintSystem,
    eex_history: pd.DataFrame,
    *,
    market: str = "CH",
    load_type: str = "BASE",
    run_timestamp: pd.Timestamp | None = None,
    min_snapshots: int = 24,
    lookback_years: int | None = None,
    fallback_to_template: bool = False,
    fallback_amplitude_eur_mwh: float = 20.0,
) -> MonthlyShapePrior:
    """Derive CH structural monthly shape from forward month-vs-CAL history."""

    months = constraints.delivery_grid.months
    if eex_history.empty:
        if fallback_to_template:
            return build_structural_monthly_shape_prior(
                constraints,
                amplitude_eur_mwh=fallback_amplitude_eur_mwh,
                fallback_reason="empty_history",
                history_counts={month_number: 0 for month_number in range(1, 13)},
            )
        return _unsupported_prior(months, "UNSUPPORTED")
    _require_columns(eex_history, {"date", "product", "load_type", "market", "price"})
    hist = _prepare_history(
        eex_history,
        market=market,
        load_type=load_type,
        run_timestamp=run_timestamp,
        lookback_years=lookback_years,
    )
    rows: list[dict[str, object]] = []
    if not hist.empty:
        snapshot_prices = {
            date: dict(zip(group["product"].astype(str), group["price"].astype(float)))
            for date, group in hist.groupby("date", sort=True)
        }
        for prices in snapshot_prices.values():
            for year in _candidate_years(prices):
                cal_key = str(year)
                if cal_key not in prices:
                    continue
                cal_price = float(prices[cal_key])
                for month_number in range(1, 13):
                    month_key = f"{year}-{month_number:02d}"
                    if month_key not in prices:
                        continue
                    rows.append(
                        {
                            "month_number": month_number,
                            "deviation": float(prices[month_key]) - cal_price,
                        }
                    )
    if not rows:
        if fallback_to_template:
            return build_structural_monthly_shape_prior(
                constraints,
                amplitude_eur_mwh=fallback_amplitude_eur_mwh,
                fallback_reason="no_monthly_cal_history",
                history_counts={month_number: 0 for month_number in range(1, 13)},
            )
        return _unsupported_prior(months, "UNSUPPORTED")

    observations = pd.DataFrame(rows)
    grouped = observations.groupby("month_number")["deviation"]
    medians = grouped.median()
    counts = grouped.size()
    if any(int(counts.get(month_number, 0)) < int(min_snapshots) for month_number in range(1, 13)):
        if fallback_to_template:
            return build_structural_monthly_shape_prior(
                constraints,
                amplitude_eur_mwh=fallback_amplitude_eur_mwh,
                fallback_reason="insufficient_monthly_history",
                history_counts={month_number: int(counts.get(month_number, 0)) for month_number in range(1, 13)},
            )
        diagnostics = pd.DataFrame(
            {
                "month_number": list(range(1, 13)),
                "n_history": [int(counts.get(month_number, 0)) for month_number in range(1, 13)],
                "status": "UNSUPPORTED",
            }
        )
        return MonthlyShapePrior(
            pd.Series(0.0, index=months, name="shape_deviation_eur_mwh"),
            diagnostics,
            pd.DataFrame(index=months),
            "UNSUPPORTED",
        )

    raw = {
        month: float(medians.loc[int(month.month)])
        for month in constraints.delivery_grid.months
    }
    shape = recenter_shape_by_parent(pd.Series(raw), constraints)
    diagnostics = pd.DataFrame(
        {
            "month": [str(month) for month in constraints.delivery_grid.months],
            "month_number": [int(month.month) for month in constraints.delivery_grid.months],
            "median_deviation": [float(medians.loc[int(month.month)]) for month in constraints.delivery_grid.months],
            "n_history": [int(counts.loc[int(month.month)]) for month in constraints.delivery_grid.months],
            "status": "USED",
        }
    )
    return MonthlyShapePrior(
        shape,
        diagnostics,
        pd.DataFrame(index=constraints.delivery_grid.months),
        "STRUCTURAL_FORWARD_CLIMATOLOGY",
    )


def build_fused_shape_prior(
    constraints: MonthlyConstraintSystem,
    *,
    panel_prior: MonthlyShapePrior | None = None,
    history_prior: MonthlyShapePrior | None = None,
    structural_prior: MonthlyShapePrior | None = None,
    weights: Mapping[str, float] | None = None,
) -> MonthlyShapePrior:
    """Fuse available zero-mean priors into one shape object."""

    weights = dict(weights or {"panel": 1.0, "history": 1.0, "structural": 0.5})
    pieces: list[tuple[str, MonthlyShapePrior, float]] = []
    for name, prior in (
        ("panel", panel_prior),
        ("history", history_prior),
        ("structural", structural_prior),
    ):
        if prior is None or prior.status == "UNSUPPORTED":
            continue
        weight = float(weights.get(name, 0.0))
        if weight > 0.0:
            pieces.append((name, prior, weight))
    if not pieces:
        return _unsupported_prior(constraints.delivery_grid.months, "UNSUPPORTED")

    total_weight = sum(weight for _, _, weight in pieces)
    combined = sum(prior.shape * (weight / total_weight) for _, prior, weight in pieces)
    shape = recenter_shape_by_parent(combined, constraints)
    status = _fused_status([prior.status for _, prior, _ in pieces])
    diagnostics = pd.DataFrame(
        {
            "source": [name for name, _, _ in pieces],
            "source_status": [prior.status for _, prior, _ in pieces],
            "weight": [weight for _, _, weight in pieces],
        }
    )
    contributions = pd.DataFrame(
        {name: prior.shape for name, prior, _ in pieces},
        index=constraints.delivery_grid.months,
    )
    return MonthlyShapePrior(shape, diagnostics, contributions, status)


def _prices_by_market(
    quotes: Mapping[str, Mapping[str, float]] | Sequence[MarketQuote],
    *,
    load_type: str,
) -> dict[str, dict[str, float]]:
    if isinstance(quotes, Mapping):
        return {
            str(market).upper(): {str(product): float(price) for product, price in prices.items()}
            for market, prices in quotes.items()
        }
    out: dict[str, dict[str, float]] = {}
    for quote in quotes:
        if quote.load_type.upper() != str(load_type).upper():
            continue
        out.setdefault(quote.market.upper(), {})[quote.product] = float(quote.price)
    return out


def _raw_neighbor_month_values(
    constraints: MonthlyConstraintSystem,
    prices: Mapping[str, float],
) -> tuple[pd.Series, pd.Series]:
    values: dict[pd.Period, float] = {}
    evidence: dict[pd.Period, str] = {}
    for month in constraints.delivery_grid.months:
        keys = (
            (str(month), "month"),
            (f"{month.year}-Q{((month.month - 1) // 3) + 1}", "quarter"),
            (str(month.year), "calendar"),
        )
        for key, kind in keys:
            if key in prices:
                values[month] = float(prices[key])
                evidence[month] = kind
                break
        else:
            values[month] = np.nan
            evidence[month] = "missing"
    return pd.Series(values, dtype=float), pd.Series(evidence, dtype=object)


def _recenter_market_raw_values(
    raw: pd.Series,
    constraints: MonthlyConstraintSystem,
) -> pd.Series:
    out = pd.Series(np.nan, index=constraints.delivery_grid.months, dtype=float)
    for bucket in constraints.month_buckets.dropna().drop_duplicates():
        mask = constraints.month_buckets.eq(bucket)
        idx = constraints.month_buckets.index[mask]
        block = raw.loc[idx]
        if block.isna().any():
            continue
        hours = constraints.delivery_grid.month_hours.loc[idx].astype(float)
        mean = float((block.to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / hours.sum())
        out.loc[idx] = block - mean
    return out


def _prepare_history(
    eex_history: pd.DataFrame,
    *,
    market: str,
    load_type: str,
    run_timestamp: pd.Timestamp | None,
    lookback_years: int | None,
) -> pd.DataFrame:
    df = eex_history.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[
        (df["market"].astype(str).str.upper() == str(market).upper())
        & (df["load_type"].astype(str).str.upper() == str(load_type).upper())
    ].copy()
    if run_timestamp is not None:
        run_date = pd.Timestamp(run_timestamp).tz_localize(None).normalize()
        df = df[df["date"] <= run_date]
        if lookback_years is not None:
            df = df[df["date"] >= run_date - pd.DateOffset(years=int(lookback_years))]
    return df


def _historical_deviations_for_month(
    *,
    month: pd.Period,
    parent_bucket: str,
    parent_product: str,
    constraints: MonthlyConstraintSystem,
    snapshot_prices: Mapping[pd.Timestamp, Mapping[str, float]],
) -> list[float]:
    deviations: list[float] = []
    month_number = int(month.month)
    for prices in snapshot_prices.values():
        for hist_year in _candidate_years(prices):
            month_key = f"{hist_year}-{month_number:02d}"
            if month_key not in prices:
                continue
            parent_value = _historical_parent_value(
                hist_year=hist_year,
                parent_bucket=parent_bucket,
                parent_product=parent_product,
                constraints=constraints,
                prices=prices,
            )
            if parent_value is None:
                continue
            deviations.append(float(prices[month_key]) - float(parent_value))
    return deviations


def _historical_parent_value(
    *,
    hist_year: int,
    parent_bucket: str,
    parent_product: str,
    constraints: MonthlyConstraintSystem,
    prices: Mapping[str, float],
) -> float | None:
    if parent_bucket.endswith("-RESIDUAL") and _YEAR_RE.match(parent_product):
        target_bucket_mask = constraints.month_buckets.eq(parent_bucket)
        residual_month_numbers = sorted(int(m.month) for m in constraints.month_buckets.index[target_bucket_mask])
        if residual_month_numbers == list(range(4, 13)):
            cal_key = str(hist_year)
            q1_key = f"{hist_year}-Q1"
            if cal_key not in prices or q1_key not in prices:
                return None
            months = pd.period_range(f"{hist_year}-01", f"{hist_year}-12", freq="M")
            hours = month_delivery_hours(months, timezone=constraints.delivery_grid.timezone)
            q1_hours = hours.loc[pd.period_range(f"{hist_year}-01", f"{hist_year}-03", freq="M")].sum()
            residual_hours = hours.sum() - q1_hours
            return (float(prices[cal_key]) * float(hours.sum()) - float(prices[q1_key]) * float(q1_hours)) / float(
                residual_hours
            )
        return None

    if _QUARTER_RE.match(parent_product):
        key = f"{hist_year}-Q{_QUARTER_RE.match(parent_product).group('quarter')}"
        return float(prices[key]) if key in prices else None
    if _YEAR_RE.match(parent_product):
        key = str(hist_year)
        return float(prices[key]) if key in prices else None
    if _MONTH_RE.match(parent_product):
        return None
    return None


def _candidate_years(prices: Mapping[str, float]) -> list[int]:
    years: set[int] = set()
    for product in prices:
        try:
            first = int(str(product)[:4])
        except ValueError:
            continue
        if _YEAR_RE.match(str(product)) or _QUARTER_RE.match(str(product)) or _MONTH_RE.match(str(product)):
            years.add(first)
    return sorted(years)


def _bucket_parent_map(constraints: MonthlyConstraintSystem) -> dict[str, str]:
    if constraints.rows.empty:
        return {}
    return {
        str(row["product"]): str(row["parent_product"])
        for _, row in constraints.rows.iterrows()
        if bool(row.get("active", True))
    }


def _unsupported_prior(months: pd.PeriodIndex, status: str) -> MonthlyShapePrior:
    shape = pd.Series(0.0, index=months, name="shape_deviation_eur_mwh")
    diagnostics = pd.DataFrame({"month": [str(month) for month in months], "status": status})
    return MonthlyShapePrior(shape, diagnostics, pd.DataFrame(index=months), status)


def _history_diag(
    month: pd.Period,
    parent_block_id: str,
    status: str,
    n_history: int,
    median_deviation: float,
    mad_deviation: float,
) -> dict[str, object]:
    return {
        "month": str(month),
        "month_number": int(month.month),
        "parent_block_id": parent_block_id,
        "status": status,
        "n_history": int(n_history),
        "median_deviation": median_deviation,
        "mad_deviation": mad_deviation,
    }


def _market_diag(
    market: str,
    *,
    status: str,
    covered_months: int = 0,
    direct_month_quotes: int = 0,
    block_shape_months: int = 0,
    horizon_months: int = 0,
    quarter_quotes: int = 0,
    calendar_quotes: int = 0,
    **extra: object,
) -> dict[str, object]:
    quote_share = float(direct_month_quotes) / float(horizon_months) if int(horizon_months) > 0 else 0.0
    row = {
        "market": market,
        "status": status,
        "covered_months": int(covered_months),
        "direct_month_quotes": int(direct_month_quotes),
        "direct_month_quote_share": quote_share,
        "block_shape_months": int(block_shape_months),
        "horizon_months": int(horizon_months),
        "quarter_quotes": int(quarter_quotes),
        "calendar_quotes": int(calendar_quotes),
    }
    row.update(extra)
    return row


def _evidence_counts_by_horizon(
    evidence: pd.Series,
    months: pd.PeriodIndex,
    *,
    run_timestamp: pd.Timestamp | None,
) -> dict[str, int | str]:
    buckets = ("h+0", "h+1", "h+2", "h+3+")
    out: dict[str, int | str] = {}
    if run_timestamp is None:
        out["horizon_reference"] = "UNSPECIFIED"
        for bucket in buckets:
            out[f"direct_month_quotes_{bucket}"] = 0
            out[f"block_shape_months_{bucket}"] = 0
            out[f"covered_months_{bucket}"] = 0
        return out

    out["horizon_reference"] = str(pd.Timestamp(run_timestamp).tz_localize(None).normalize().date())
    run_year = int(pd.Timestamp(run_timestamp).tz_localize(None).year)
    for bucket in buckets:
        month_mask = [_horizon_bucket(int(month.year) - run_year) == bucket for month in months]
        if not any(month_mask):
            out[f"direct_month_quotes_{bucket}"] = 0
            out[f"block_shape_months_{bucket}"] = 0
            out[f"covered_months_{bucket}"] = 0
            continue
        bucket_evidence = evidence.loc[months[month_mask]]
        out[f"direct_month_quotes_{bucket}"] = int(bucket_evidence.eq("month").sum())
        out[f"block_shape_months_{bucket}"] = int(bucket_evidence.isin(["quarter", "calendar"]).sum())
        out[f"covered_months_{bucket}"] = int(bucket_evidence.isin(["month", "quarter", "calendar"]).sum())
    return out


def _far_horizon_monthly_evidence_status(diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty:
        return "UNSUPPORTED"
    required = {"direct_month_quotes_h+2", "direct_month_quotes_h+3+"}
    if not required <= set(diagnostics.columns):
        return "UNSPECIFIED_HORIZON"
    far_direct = int(
        pd.to_numeric(diagnostics["direct_month_quotes_h+2"], errors="coerce").fillna(0).sum()
        + pd.to_numeric(diagnostics["direct_month_quotes_h+3+"], errors="coerce").fillna(0).sum()
    )
    if far_direct <= 0:
        return "NO_FAR_HORIZON_MONTHLY_EVIDENCE"
    markets = diagnostics[
        (
            pd.to_numeric(diagnostics["direct_month_quotes_h+2"], errors="coerce").fillna(0)
            + pd.to_numeric(diagnostics["direct_month_quotes_h+3+"], errors="coerce").fillna(0)
        )
        > 0
    ]["market"].astype(str).str.upper().tolist()
    if markets == ["DE"]:
        return "DE_FAR_HORIZON_MONTHLY_EVIDENCE"
    if "DE" in markets:
        return "MULTI_MARKET_FAR_HORIZON_MONTHLY_WITH_DE"
    return "NON_DE_FAR_HORIZON_MONTHLY_EVIDENCE"


def _market_far_horizon_monthly_evidence_status(row: pd.Series) -> str:
    direct = _safe_int(row.get("direct_month_quotes_h+2", 0)) + _safe_int(row.get("direct_month_quotes_h+3+", 0))
    if direct <= 0:
        return "NO_FAR_HORIZON_MONTHLY_EVIDENCE"
    market = str(row.get("market", "")).upper()
    if market == "DE":
        return "DE_FAR_HORIZON_MONTHLY_EVIDENCE"
    return "MARKET_FAR_HORIZON_MONTHLY_EVIDENCE"


def _safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _horizon_bucket(horizon: int) -> str:
    if horizon <= 0:
        return "h+0"
    if horizon == 1:
        return "h+1"
    if horizon == 2:
        return "h+2"
    return "h+3+"


def _fused_status(statuses: list[str]) -> str:
    if "PANEL_MULTI_MARKET" in statuses:
        return "PANEL_MULTI_MARKET"
    if "PARTIAL_MONTHLY_PANEL" in statuses:
        return "PARTIAL_MONTHLY_PANEL"
    if "PARTIAL_PANEL_MULTI_MARKET" in statuses:
        return "PARTIAL_PANEL_MULTI_MARKET"
    if "DE_SINGLE_MARKET_MONTHLY" in statuses:
        return "DE_SINGLE_MARKET_MONTHLY"
    if "DE_SINGLE_MARKET" in statuses:
        return "DE_SINGLE_MARKET"
    if "SINGLE_MARKET_MONTHLY" in statuses:
        return "SINGLE_MARKET_MONTHLY"
    if "PANEL_BLOCK_SHAPE" in statuses:
        return "PANEL_BLOCK_SHAPE"
    if "HISTORY_FORWARD" in statuses or "PARTIAL_HISTORY_FORWARD" in statuses:
        return "HISTORY_FORWARD"
    if "STRUCTURAL_FORWARD_CLIMATOLOGY" in statuses:
        return "STRUCTURAL_FORWARD_CLIMATOLOGY"
    if "STRUCTURAL_TEMPLATE" in statuses:
        return "STRUCTURAL_TEMPLATE"
    return "UNSUPPORTED"


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required EEX history columns: {missing}")
