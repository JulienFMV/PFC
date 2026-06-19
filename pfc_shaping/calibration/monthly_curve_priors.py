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
        status = "PARTIAL_PANEL_MULTI_MARKET"
    elif len(monthly_markets) == 1:
        keep = 1.0 - min(1.0, max(0.0, float(neighbor_shrinkage)))
        combined = combined * keep
        status = "DE_SINGLE_MARKET" if monthly_markets[0] == "DE" else "SINGLE_MARKET"
    elif block_markets:
        keep = 1.0 - min(1.0, max(0.0, float(neighbor_shrinkage)))
        combined = combined * keep
        status = "PANEL_BLOCK_SHAPE"
    else:
        status = "UNSUPPORTED"

    shape = recenter_shape_by_parent(combined, constraints)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["prior_status"] = status
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
) -> MonthlyShapePrior:
    """Build an explicit template structural prior in zero-mean space."""

    ratios = dict(monthly_ratios or DEFAULT_CH_STRUCTURAL_MONTHLY_RATIOS)
    raw = {
        month: (float(ratios.get(int(month.month), 1.0)) - 1.0) * float(amplitude_eur_mwh)
        for month in constraints.delivery_grid.months
    }
    shape = recenter_shape_by_parent(pd.Series(raw), constraints)
    diagnostics = pd.DataFrame(
        {
            "month": [str(month) for month in constraints.delivery_grid.months],
            "month_number": [int(month.month) for month in constraints.delivery_grid.months],
            "ratio": [float(ratios.get(int(month.month), 1.0)) for month in constraints.delivery_grid.months],
            "status": "STRUCTURAL_TEMPLATE",
        }
    )
    return MonthlyShapePrior(
        shape,
        diagnostics,
        pd.DataFrame(index=constraints.delivery_grid.months),
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
) -> dict[str, object]:
    quote_share = float(direct_month_quotes) / float(horizon_months) if int(horizon_months) > 0 else 0.0
    return {
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
    if "PARTIAL_PANEL_MULTI_MARKET" in statuses:
        return "PARTIAL_PANEL_MULTI_MARKET"
    if "DE_SINGLE_MARKET" in statuses:
        return "DE_SINGLE_MARKET"
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
