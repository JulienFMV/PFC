"""Yearly portfolio analytics for the Deviwa pool dashboard.

State-of-the-art utility ERM metrics, computed per **delivery year**:

  - **FMV-share hedge ratio** = Σ FMV-deal_volume[Y] / Σ programme[Y]
    Replaces the conceptually wrong "% of deals with future delivery" formula
    used in the legacy `portfolio_analytics.compute_portfolio_metrics`.
  - **Industry hedge corridor** by Y-N maturity (E.ON / EnBW / Axpo / EFET /
    Eurelectric reference). A ratio inside the corridor → green; below → amber
    or red; above → "over-hedged", flag for discussion.
  - **Hedge ladder by trade date** : cumulative hedge volume + volume-weighted
    average price as the deals were executed. Useful overlaid on Cal-Y forward
    history to answer "did we hedge well?".
  - **Budget projection** : locked-in cost (notional of executed deals) +
    at-risk band (open exposure × current year forward proxy, ± P10/P90
    from a 20 %-annualized volatility assumption — POC simplification).
    The year price proxy uses the Cal-Y contract when liquid, falling back
    to a quarter-hour-weighted average of remaining ``Q*_Y`` contracts
    once Cal-Y enters delivery (= current calendar year case).

**IMPORTANT — semantic scope of the hedge ratio :**

  Deviwa.xlsx contains ONLY the deals between each actor and FMV. The actors
  (RELL, EVTL, EW Binn, EDSH) typically have additional supplier contracts
  (BKW, EnAlpin, EWZ, ...) NOT in this dataset. So ``hedge_ratio_pct``
  represents the **FMV-share** of the actor's hedging programme, not the
  total external hedge. A ratio > 100 % means the actor has bought more
  from FMV than its forecast load — a legitimate strategy (e.g. resale,
  layered tranching).

  EDSH is special : its PRO_REAL sheet exposes the per-supplier
  decomposition (BKW + EnAlpin + EWZ + FMV + Spot). The dashboard uses
  that for the *total* hedge view of EDSH. The other 3 actors are limited
  to the FMV-share view from this module.

Sign / scope conventions (post `_invert_to_client_perspective` in deviwa_parser) :

  Original FMV view → after flip → client view :

  - FMV ``Withdrawal`` (FMV delivers energy) → client ``Intake``
    = client receives energy = client BUYS → **hedge of the load (+1)**
  - FMV ``Intake`` (FMV receives energy) → client ``Withdrawal``
    = client delivers energy = client SELLS → **un-hedges / long position (-1)**
  - ``volume_sum`` is absolute (always ≥ 0)
  - ``notional_sum`` / ``pnl_sum`` are signed post-flip ; an Intake (client
    buys) typically has notional_sum < 0 (client paid out).

Year accounting :

  - Each row in the deals frame represents one (deal × month) tuple. Year is
    derived from ``month``. A 4-year contract spans 48 rows (one per month).
  - Programme is hourly. Year volume in MWh = sum of programme_mw over the
    hours of the year (each hourly row contributes 1 MWh per MW). DST is
    correctly handled by the upstream cache layer (autumn DST day has 25
    distinct UTC hours preserved).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Industry hedge corridors (POC defaults — calibrated to public utility ERM
# disclosures, can be overridden at runtime if FMV/Deviwa publishes its own
# policy targets later)
# ---------------------------------------------------------------------------
#
# Sources :
#   - EFET (European Federation of Energy Traders) risk management guidelines
#   - Eurelectric Gas & Power Risk Management Best Practices
#   - Public hedging disclosures from E.ON / EnBW / RWE (DE) and Axpo / Alpiq /
#     BKW (CH) annual reports — typical staggered curve
#
# The mapping is by delivery-year offset from "today":
#   Y-1  = the next calendar year      (e.g. 2027 viewed from 2026)
#   Y-2  = two calendar years out      (e.g. 2028 viewed from 2026)
#   Y-3  = three years out
#   Y-4+ = four+ years out
#
# A hedge ratio inside (low_pct, high_pct) is "in_corridor" → green.
# Below low_pct → "below" (amber/red).
# Above high_pct → "above" (potentially over-hedged → flag for review).
# The current calendar year (offset 0) is in delivery and treated separately.

INDUSTRY_HEDGE_TARGETS: dict[int, tuple[int, int]] = {
    1: (80, 100),
    2: (50, 80),
    3: (20, 50),
    4: (0, 30),
}

# POC volatility assumption for budget P10/P90 band.
# 20 % annualized log-volatility of CH base forwards is a conservative central
# estimate that fits 2024-2026 EEX history. Replace with bootstrapped historical
# vol in a future iteration if accuracy matters.
VOLATILITY_ANNUAL_PCT: float = 20.0

# z-score for a one-tailed 80 % CI (10/90 percentiles).
# norm.ppf(0.90) ≈ 1.2816.
Z_P10_P90: float = 1.2816


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YearlyHedgeMetrics:
    """All KPIs for a given delivery year × actor (or pool).

    Attributes:
        year: delivery calendar year (e.g. 2027).
        programme_mwh: forecast net consumption for the year, in MWh.
            When ``programme_is_extrapolated`` is True, this value comes from
            carry-forward of the most recent year with actual data (industry-
            standard EFET / Eurelectric convention for stable load profiles).
        hedged_mwh: net hedge volume from deals (Σ Withdrawal − Σ Intake).
            Positive = client has bought net energy (= hedged the load).
            Negative = client has net sold (= long position, unusual for a GRD).
        open_mwh: signed open exposure = programme_mwh − hedged_mwh.
            Positive = under-hedged (still need to buy); negative = over-hedged.
        hedge_ratio_pct: 100 × hedged / programme. NaN if programme = 0.
            Not capped — values > 100 % indicate over-hedge.
        avg_hedge_price_eur_mwh: volume-weighted average price of all deals
            with delivery in the year. NaN if no deals.
        notional_eur: total contract value of deals delivering in the year
            (absolute EUR — the cost the client has committed to).
        pnl_eur: realized + unrealized P&L of those deals (signed).
        n_deals: number of distinct deal IDs touching the year.
        target_low_pct, target_high_pct: industry corridor for this delivery
            year vs `today`. None if no rule applies (e.g. delivery year < today).
        target_status: one of
            "in_corridor"    → ratio inside [low, high]
            "below"          → ratio < low (under-hedged)
            "above"          → ratio > high (potential over-hedge)
            "no_data"        → ratio is NaN (no programme to compare)
            "n/a"            → no industry corridor for this year offset
        programme_is_extrapolated: True if ``programme_mwh`` was carried
            forward from an earlier year because the actor's forecast did
            not extend to this delivery year. Dashboards should surface
            this with a "Hochrechnung" badge so the user knows the
            programme baseline is not a fresh client forecast.
    """
    year: int
    programme_mwh: float
    hedged_mwh: float
    open_mwh: float
    hedge_ratio_pct: float
    avg_hedge_price_eur_mwh: float
    notional_eur: float
    pnl_eur: float
    n_deals: int
    target_low_pct: int | None
    target_high_pct: int | None
    target_status: str
    programme_is_extrapolated: bool = False


@dataclass(frozen=True)
class BudgetProjection:
    """Total expected energy cost for a delivery year × actor.

    Decomposition :
        - locked_eur     = notional already committed via deals (paid forward)
        - open_mwh       = residual exposure (= programme − hedged)
        - at_risk_eur    = open_mwh × current Cal-Y forward price
        - central_eur    = locked + at_risk (point estimate)
        - p10_eur, p90_eur = the 10th and 90th percentile of the **cost
          distribution**. Under a log-normal price assumption with annualized
          volatility ``VOL_ANNUAL_PCT`` :
            σ_cost  = |open_mwh| × forward × VOL × √(years_to_delivery)
            p10_eur = central − Z × σ_cost   (= best-case cost outcome)
            p90_eur = central + Z × σ_cost   (= worst-case cost outcome)

    The labels are interpreted as *cost percentiles* — ``p10_eur`` is the
    best 10 % of cost outcomes (lowest cost) and ``p90_eur`` the worst 10 %
    (highest cost). For an under-hedged year (open_mwh > 0) the best cost
    is achieved when forward prices fall ; for an over-hedged year
    (open_mwh < 0, client has surplus to resell) the best cost is achieved
    when forward prices rise. The σ band is symmetric around central in
    EUR magnitude — the direction of correlation cost↔forward flips with
    the sign of open_mwh, but the cost CDF passes through (p10, p90) at
    the 10/90 percentile points regardless.
    """
    year: int
    locked_eur: float
    open_mwh: float
    forward_price_eur_mwh: float
    central_eur: float
    p10_eur: float
    p90_eur: float
    sigma_assumed_pct: float


# ---------------------------------------------------------------------------
# Industry corridor helpers
# ---------------------------------------------------------------------------

def get_target_corridor(
    delivery_year: int,
    today: pd.Timestamp | None = None,
) -> tuple[int, int] | None:
    """Return ``(low_pct, high_pct)`` from ``INDUSTRY_HEDGE_TARGETS`` for the
    given delivery year vs today, or ``None`` if no rule applies.

    Args:
        delivery_year: the calendar year of delivery (e.g. 2027).
        today: reference date. Defaults to current time in Europe/Zurich.
    """
    if today is None:
        today = pd.Timestamp.now(tz="Europe/Zurich")
    offset = delivery_year - today.year
    if offset < 1:
        return None  # past or current year — different rules apply (already in delivery)
    return INDUSTRY_HEDGE_TARGETS.get(offset)


def classify_hedge_status(
    ratio_pct: float,
    corridor: tuple[int, int] | None,
) -> str:
    """Map a hedge ratio against an industry corridor to a status string."""
    if not np.isfinite(ratio_pct):
        return "no_data"
    if corridor is None:
        return "n/a"
    low, high = corridor
    if ratio_pct < low:
        return "below"
    if ratio_pct > high:
        return "above"
    return "in_corridor"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_actor(df: pd.DataFrame, actor: str | None, actor_col: str = "actor") -> pd.DataFrame:
    """If ``actor`` is None, return ``df`` unchanged (= pool aggregate view).
    Otherwise return a defensive **copy** of rows matching ``actor`` exactly.

    The copy avoids ``SettingWithCopyWarning`` if the caller mutates the
    returned frame (negligible perf cost on a 240-row deals frame).
    """
    if df is None or df.empty:
        return df
    if actor is None:
        return df
    if actor_col not in df.columns:
        return df.iloc[0:0]
    return df[df[actor_col] == actor].copy()


def _scope_signed_volume(deals: pd.DataFrame) -> pd.Series:
    """Return signed hedge volume per row in **client view** :
        + ``volume_sum`` for ``Intake``     (= client receives = buys = HEDGE)
        − ``volume_sum`` for ``Withdrawal`` (= client delivers = sells = un-hedge)
        0 for any other / missing scope.

    Confirmed on real Deviwa.xlsx data : an original FMV ``Withdrawal``
    (FMV delivers energy to client) flips to client ``Intake`` after
    `_invert_to_client_perspective`, and the post-flip ``notional_sum``
    is negative (client paid out cash for energy), confirming this IS
    a hedge from the client's load perspective.

    Robust to capitalization and whitespace in ``scope``. Unknown scopes
    (or missing column) contribute 0 — deliberate over fail-loud so a
    stray row doesn't crash the dashboard.
    """
    if "volume_sum" not in deals.columns:
        return pd.Series(0.0, index=deals.index)
    vol = pd.to_numeric(deals["volume_sum"], errors="coerce").fillna(0.0).abs()
    if "scope" not in deals.columns:
        return pd.Series(0.0, index=deals.index)
    scope = deals["scope"].astype(str).str.lower()
    sign = pd.Series(0.0, index=deals.index)
    sign[scope.str.contains("intake", na=False)] = 1.0
    sign[scope.str.contains("withdrawal", na=False)] = -1.0
    return vol * sign


def _volume_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Volume-weighted average. NaN if no positive weights or no valid values."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").abs()
    valid = v.notna() & w.notna() & (w > 0)
    if not valid.any():
        return float("nan")
    total_w = float(w[valid].sum())
    if total_w <= 0:
        return float("nan")
    return float((v[valid] * w[valid]).sum() / total_w)


# ---------------------------------------------------------------------------
# compute_hedge_by_year — the central function
# ---------------------------------------------------------------------------

def compute_hedge_by_year(
    deals: pd.DataFrame,
    programme: pd.DataFrame,
    actor: str | None = None,
    today: pd.Timestamp | None = None,
    *,
    extrapolate_missing_programme: bool = True,
    include_years: list[int] | None = None,
) -> dict[int, YearlyHedgeMetrics]:
    """Compute the true hedge ratio per delivery year for one actor or the pool.

    Args:
        deals: long-format deals frame with at minimum these columns :
            ``actor, scope, month, volume_sum, notional_sum, deal``.
            ``scope`` must already be flipped to client view.
            ``month`` is parseable as a date (used to derive delivery year).
        programme: long-format programme frame with at minimum :
            ``actor, timestamp, programme_mw``.
            ``timestamp`` is tz-aware Europe/Zurich (start-of-interval).
        actor: actor name to filter by, or ``None`` for the consolidated pool.
        today: reference date for industry corridor. Defaults to now (CH).
        extrapolate_missing_programme: when True (default), missing programme
            years are filled with carry-forward from the most recent year that
            has actual data. The corresponding ``YearlyHedgeMetrics`` carries
            ``programme_is_extrapolated=True`` so the dashboard can surface a
            "Hochrechnung" badge. When False, missing years stay at 0 and
            ``hedge_ratio_pct`` will be NaN. Carry-forward is the standard
            EFET / Eurelectric assumption for relatively stable load profiles
            (most Swiss DSOs grow ≤ 2 % / year).
        include_years: explicit list of years that must appear in the output
            dict. Years without any data get ``programme_mwh=0,
            hedged_mwh=0, hedge_ratio_pct=0.0, target_status="below"`` —
            useful for the cockpit to always show 4 year cards regardless
            of data availability.

    Returns:
        Dict mapping ``delivery_year → YearlyHedgeMetrics``. By default
        contains years where deals OR programme data exists. Use
        ``include_years`` to force a specific set.
    """
    if today is None:
        today = pd.Timestamp.now(tz="Europe/Zurich")

    # Filter actor (None → pool aggregation: no filter, all actors summed)
    deals_f = _filter_actor(deals, actor, "actor")
    prog_f = _filter_actor(programme, actor, "actor")

    # ── Programme aggregation per year ──────────────────────────────────────
    # Each programme row represents one hour. Sum of programme_mw × 1h = MWh.
    prog_yearly: dict[int, float] = {}
    if (
        prog_f is not None
        and not prog_f.empty
        and "timestamp" in prog_f.columns
        and "programme_mw" in prog_f.columns
    ):
        prog_y = prog_f[["timestamp", "programme_mw"]].copy()
        prog_y["_year"] = prog_y["timestamp"].dt.year
        agg = prog_y.groupby("_year")["programme_mw"].sum()
        prog_yearly = {int(y): float(v) for y, v in agg.items() if pd.notna(v)}

    # ── Deals aggregation per year ──────────────────────────────────────────
    # Year is derived from `month`. Each row already represents the contribution
    # of one deal to one month, so per-year aggregation is a simple sum.
    deal_yearly: dict[int, dict] = {}
    if (
        deals_f is not None
        and not deals_f.empty
        and "month" in deals_f.columns
    ):
        d = deals_f.copy()
        d["_year"] = pd.to_datetime(d["month"], errors="coerce").dt.year
        d = d[d["_year"].notna()].copy()
        if not d.empty:
            d["_year"] = d["_year"].astype(int)
            d["_signed_vol"] = _scope_signed_volume(d)
            d["_abs_vol"] = pd.to_numeric(d["volume_sum"], errors="coerce").fillna(0.0).abs()
            d["_notional_abs"] = (
                pd.to_numeric(d.get("notional_sum"), errors="coerce")
                .fillna(0.0)
                .abs()
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                d["_deal_price"] = np.where(
                    d["_abs_vol"] > 0,
                    d["_notional_abs"] / d["_abs_vol"].replace(0, np.nan),
                    np.nan,
                )

            for year, g in d.groupby("_year"):
                deal_yearly[int(year)] = {
                    "hedged_mwh": float(g["_signed_vol"].sum()),
                    "abs_vol": float(g["_abs_vol"].sum()),
                    "notional": float(g["_notional_abs"].sum()),
                    "pnl": float(
                        pd.to_numeric(g.get("pnl_sum", 0), errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                    "avg_price": _volume_weighted_average(g["_deal_price"], g["_abs_vol"]),
                    "n_deals": int(g["deal"].nunique()) if "deal" in g.columns else len(g),
                }

    # ── Determine the set of years to emit ──────────────────────────────────
    # Default : years with either programme or deals data.
    # If include_years is given, ALL of them get an entry (filled with zero
    # if no data and extrapolation cannot fill).
    base_years = set(prog_yearly.keys()) | set(deal_yearly.keys())
    if include_years is not None:
        all_years = sorted(set(include_years) | base_years)
    else:
        all_years = sorted(base_years)

    # ── Optional carry-forward extrapolation of programme ───────────────────
    # Carry-forward is applied ONLY to years where the actor has at least
    # one deal — those are the years with explicit commitment, where a
    # programme baseline is needed to evaluate the open exposure.
    #
    # For years with no deals AND no programme, we deliberately do NOT
    # invent a programme : extrapolating RELL's 62 240 MWh 2026 forecast
    # to 2027-2029 would inject ~190 GWh of phantom long exposure into
    # the pool risk and inflate VaR by ~15× without any underlying
    # commercial truth (RELL hasn't told us they need that energy).
    extrapolated_years: set[int] = set()
    if extrapolate_missing_programme:
        years_with_real_prog = sorted(
            y for y, v in prog_yearly.items() if v and v > 0
        )
        years_with_deals = set(deal_yearly.keys())
        for y in all_years:
            if prog_yearly.get(y, 0.0) > 0:
                continue  # actual data, no extrapolation needed
            if y not in years_with_deals:
                continue  # no commitment from the actor → don't invent a programme
            sources = [s for s in years_with_real_prog if s < y]
            if sources:
                src_year = max(sources)
                prog_yearly[y] = prog_yearly[src_year]
                extrapolated_years.add(y)

    # ── Combine and emit YearlyHedgeMetrics per year ────────────────────────
    out: dict[int, YearlyHedgeMetrics] = {}
    for y in all_years:
        prog_mwh = prog_yearly.get(y, 0.0)
        d_data = deal_yearly.get(y, {})
        hedged_mwh = float(d_data.get("hedged_mwh", 0.0))
        notional = float(d_data.get("notional", 0.0))
        pnl = float(d_data.get("pnl", 0.0))
        avg_price = float(d_data.get("avg_price", float("nan")))
        n_deals = int(d_data.get("n_deals", 0))

        open_mwh = prog_mwh - hedged_mwh
        ratio = (100.0 * hedged_mwh / prog_mwh) if prog_mwh > 0 else float("nan")

        corridor = get_target_corridor(y, today)
        status = classify_hedge_status(ratio, corridor)

        out[y] = YearlyHedgeMetrics(
            year=y,
            programme_mwh=prog_mwh,
            hedged_mwh=hedged_mwh,
            open_mwh=open_mwh,
            hedge_ratio_pct=ratio,
            avg_hedge_price_eur_mwh=avg_price,
            notional_eur=notional,
            pnl_eur=pnl,
            n_deals=n_deals,
            target_low_pct=corridor[0] if corridor else None,
            target_high_pct=corridor[1] if corridor else None,
            target_status=status,
            programme_is_extrapolated=(y in extrapolated_years),
        )

    return out


# ---------------------------------------------------------------------------
# Hedge ladder
# ---------------------------------------------------------------------------

def compute_hedge_ladder(
    deals: pd.DataFrame,
    delivery_year: int,
    actor: str | None = None,
) -> pd.DataFrame:
    """Cumulative hedge build-up by trade_date for a given delivery year.

    Designed to be overlaid on a Cal-Y price chart in the dashboard so the
    user can see *when* deals were struck relative to the market.

    Args:
        deals: long-format deals frame (same as ``compute_hedge_by_year``).
        delivery_year: filter to deals whose ``month`` is in that year.
        actor: filter by actor; ``None`` for pool aggregate.

    Returns:
        DataFrame with columns :
            trade_date          : day a deal was traded (Timestamp)
            hedge_volume_mwh    : signed daily hedge (+ intake, − withdrawal)
            cum_hedge_mwh       : running cumulative net hedge
            deal_price_eur_mwh  : volume-weighted price of that day's deals
            cum_avg_price       : running volume-weighted average price

        Empty DataFrame (with same columns) if no deals match.

    Note on monotonicity :
        ``cum_hedge_mwh`` is monotonically increasing only when all deals
        for the (actor × year) selection have the same scope (typically
        ``Intake`` only for a load-following GRD). In the pool view, or
        for an actor mixing Intake and Withdrawal, the cumulative curve
        can dip — this is real and meaningful (it shows the days when
        positions were unwound). The chart should treat dips as valid
        events, not interpolate over them.
    """
    columns = [
        "trade_date",
        "hedge_volume_mwh",
        "cum_hedge_mwh",
        "deal_price_eur_mwh",
        "cum_avg_price",
    ]
    if deals is None or deals.empty:
        return pd.DataFrame(columns=columns)

    deals_f = _filter_actor(deals, actor, "actor")
    if deals_f is None or deals_f.empty:
        return pd.DataFrame(columns=columns)

    if "month" not in deals_f.columns or "trade_date" not in deals_f.columns:
        return pd.DataFrame(columns=columns)

    d = deals_f.copy()
    d["_year"] = pd.to_datetime(d["month"], errors="coerce").dt.year
    d = d[d["_year"] == delivery_year]
    if d.empty:
        return pd.DataFrame(columns=columns)

    d["_trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d = d.dropna(subset=["_trade_date"])
    if d.empty:
        return pd.DataFrame(columns=columns)

    d["_signed_vol"] = _scope_signed_volume(d)
    d["_abs_vol"] = pd.to_numeric(d["volume_sum"], errors="coerce").fillna(0.0).abs()
    d["_notional_abs"] = (
        pd.to_numeric(d.get("notional_sum"), errors="coerce").fillna(0.0).abs()
    )

    g = (
        d.groupby("_trade_date")
        .agg(
            hedge_volume_mwh=("_signed_vol", "sum"),
            abs_vol=("_abs_vol", "sum"),
            notional=("_notional_abs", "sum"),
        )
        .reset_index()
        .rename(columns={"_trade_date": "trade_date"})
        .sort_values("trade_date")
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        g["deal_price_eur_mwh"] = np.where(
            g["abs_vol"] > 0,
            g["notional"] / g["abs_vol"].replace(0, np.nan),
            np.nan,
        )
    g["cum_hedge_mwh"] = g["hedge_volume_mwh"].cumsum()
    cum_notional = g["notional"].cumsum()
    cum_abs_vol = g["abs_vol"].cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        g["cum_avg_price"] = np.where(
            cum_abs_vol > 0,
            cum_notional / cum_abs_vol.replace(0, np.nan),
            np.nan,
        )

    return g[columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forward price lookup
# ---------------------------------------------------------------------------

def _quarter_hours(year: int, quarter: int) -> int:
    """Number of hours in a calendar quarter (handles 28/29-Feb)."""
    if quarter == 1:
        start, end = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=4, day=1)
    elif quarter == 2:
        start, end = pd.Timestamp(year=year, month=4, day=1), pd.Timestamp(year=year, month=7, day=1)
    elif quarter == 3:
        start, end = pd.Timestamp(year=year, month=7, day=1), pd.Timestamp(year=year, month=10, day=1)
    elif quarter == 4:
        start, end = pd.Timestamp(year=year, month=10, day=1), pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        raise ValueError(f"quarter must be 1..4, got {quarter}")
    return int((end - start).total_seconds() // 3600)


def _latest_tenor_price(
    forwards: pd.DataFrame,
    pattern: str,
) -> float:
    """Latest settle for the first product whose code contains ``pattern``."""
    df = forwards[forwards["product"].astype(str).str.contains(pattern, na=False, regex=False)]
    df = df.dropna(subset=["price"])
    if df.empty:
        return float("nan")
    if "date" in df.columns:
        df = df.sort_values("date")
    return float(df["price"].iloc[-1])


def get_forward_year_proxy(
    forwards: pd.DataFrame,
    delivery_year: int,
    market: str = "CH",
    load_type: str = "base",
) -> float:
    """Best available forward proxy for the **whole calendar year**.

    EEX retires the Cal-Y (``Y01_{YEAR}``) contract once the year enters
    delivery, so for the current year only quarterlies/monthlies remain.
    This function uses a layered fallback :

      1. If ``Y01_{year}`` Cal contract exists → use its latest settle.
      2. Else : volume-weighted average of available ``Q01..Q04_{year}``
         contracts, weighted by the number of calendar hours in each
         quarter (handles leap years correctly via :func:`_quarter_hours`).
         Quarters without an available forward are silently dropped from
         the average (typical mid-year case : Q1, Q2 already settled, only
         Q3 & Q4 remain liquid).
      3. Else : NaN.

    The function does NOT use realized spot for already-elapsed sub-periods
    (Jan-Apr 2026 in the example case), to keep the proxy purely
    forward-based for the budget projection. The dashboard layer can add
    a "current year" callout that combines spot YTD + this forward proxy
    if higher accuracy is needed for headline numbers.

    Args:
        forwards: long-format with columns ``date, market, product, load_type, price``.
        delivery_year: e.g. 2026 (current) or 2027 (future).
        market: market filter (default "CH").
        load_type: "base" or "peak" (default "base").

    Returns:
        Best-available forward price as a float, NaN if no Cal/Q tenor present.
    """
    if forwards is None or forwards.empty:
        return float("nan")

    df = forwards
    if "market" in df.columns:
        df = df[df["market"].astype(str).str.upper() == market.upper()]
    if "load_type" in df.columns:
        df = df[df["load_type"].astype(str).str.lower() == load_type.lower()]
    if df.empty or "product" not in df.columns or "price" not in df.columns:
        return float("nan")

    # 1. Cal-Y if available
    cal = _latest_tenor_price(df, f"Y01_{delivery_year}")
    if np.isfinite(cal):
        return cal

    # 2. Weighted average of remaining quarterlies
    weighted_sum = 0.0
    total_hours = 0
    for q in (1, 2, 3, 4):
        q_price = _latest_tenor_price(df, f"Q0{q}_{delivery_year}")
        if not np.isfinite(q_price):
            continue
        h = _quarter_hours(delivery_year, q)
        weighted_sum += q_price * h
        total_hours += h
    if total_hours > 0:
        return weighted_sum / total_hours

    # 3. Nothing available
    return float("nan")


def get_forward_cal_price(
    forwards: pd.DataFrame,
    delivery_year: int,
    market: str = "CH",
    load_type: str = "base",
) -> float:
    """Latest **calendar-year** forward price for the given delivery year.

    EEX product codes used by FMV's `Price_Report_EEX_Yearly.xlsx` follow
    the pattern ``Y01_{YYYY}_{BASE|PEAK}`` for the calendar year contract,
    next to ``Q01..Q04_{YYYY}`` quarters and ``M01..M12_{YYYY}`` months.
    A naive substring match on the year would silently return whichever
    tenor happens to sort last (e.g. Q01 = 120 EUR vs Cal = 90 EUR — the
    budget projection would be off by 30 % for no visible reason).

    This function filters explicitly on the ``Y01_{year}`` pattern, which
    is unique to the calendar-year contract.

    Args:
        forwards: long-format frame with columns
            ``date, market, product, load_type, price``.
        delivery_year: e.g. 2027.
        market: filter (default "CH").
        load_type: "base" or "peak" (case-insensitive, default "base").

    Returns:
        Most recent settlement price as a float, or NaN if no Cal-Y row found.
    """
    if forwards is None or forwards.empty:
        return float("nan")

    df = forwards
    if "market" in df.columns:
        df = df[df["market"].astype(str).str.upper() == market.upper()]
    if "load_type" in df.columns:
        df = df[df["load_type"].astype(str).str.lower() == load_type.lower()]
    if df.empty or "product" not in df.columns or "price" not in df.columns:
        return float("nan")

    # Explicit Cal-Y pattern: Y01_{YEAR} appears in the EEX product code.
    cal_pattern = f"Y01_{delivery_year}"
    df = df[df["product"].astype(str).str.contains(cal_pattern, na=False, regex=False)]
    df = df.dropna(subset=["price"])
    if df.empty:
        return float("nan")

    if "date" in df.columns:
        df = df.sort_values("date")
    return float(df["price"].iloc[-1])


# ---------------------------------------------------------------------------
# Budget projection
# ---------------------------------------------------------------------------

def compute_budget_projection(
    deals: pd.DataFrame,
    programme: pd.DataFrame,
    forwards: pd.DataFrame,
    actor: str | None = None,
    today: pd.Timestamp | None = None,
    *,
    extrapolate_missing_programme: bool = True,
    include_years: list[int] | None = None,
) -> dict[int, BudgetProjection]:
    """Per delivery year : locked-in cost + at-risk band on residual exposure.

    The decomposition is :

        locked_eur     = Σ |notional_sum| of deals delivering in year Y
        open_mwh       = programme_mwh − hedged_mwh
        at_risk_eur    = open_mwh × current Cal-Y forward
        central_eur    = locked + at_risk
        σ_eur          = |open_mwh| × forward × VOL_ANNUAL × √(years_to_delivery)
        p10/p90_eur    = central ∓ Z × σ_eur

    Where Z = 1.2816 (one-tailed 80 % CI cutoff). Years-to-delivery is computed
    against the year midpoint (1 July) and floored at 0.25y to avoid σ → 0
    for the current delivery year (still has time variance until last day).

    NaN forward price → all dependent fields are NaN; ``locked_eur`` and
    ``open_mwh`` remain valid since they don't depend on the forward.
    """
    if today is None:
        today = pd.Timestamp.now(tz="Europe/Zurich")

    metrics = compute_hedge_by_year(
        deals, programme, actor=actor, today=today,
        extrapolate_missing_programme=extrapolate_missing_programme,
        include_years=include_years,
    )
    out: dict[int, BudgetProjection] = {}

    for y, m in metrics.items():
        # Year price proxy : Cal-Y for future years, quarterly-weighted average
        # for the current year (Cal-Y already in delivery, no longer tradeable).
        fwd_price = get_forward_year_proxy(forwards, y)

        # Mid-year delivery anchor (1 July of year Y in the same tz as today)
        delivery_mid = pd.Timestamp(year=y, month=7, day=1, tz=today.tz)
        years_to_delivery = max(
            (delivery_mid - today).total_seconds() / (365.25 * 86400),
            0.25,  # floor: even mid-delivery year still has price uncertainty
        )

        if not np.isfinite(fwd_price):
            out[y] = BudgetProjection(
                year=y,
                locked_eur=m.notional_eur,
                open_mwh=m.open_mwh,
                forward_price_eur_mwh=float("nan"),
                central_eur=float("nan"),
                p10_eur=float("nan"),
                p90_eur=float("nan"),
                sigma_assumed_pct=VOLATILITY_ANNUAL_PCT,
            )
            continue

        at_risk_central = m.open_mwh * fwd_price
        central = m.notional_eur + at_risk_central

        # σ scales with √T under the standard log-normal assumption.
        sigma_eur = (
            abs(m.open_mwh)
            * fwd_price
            * (VOLATILITY_ANNUAL_PCT / 100.0)
            * float(np.sqrt(years_to_delivery))
        )
        p10 = central - Z_P10_P90 * sigma_eur
        p90 = central + Z_P10_P90 * sigma_eur

        out[y] = BudgetProjection(
            year=y,
            locked_eur=m.notional_eur,
            open_mwh=m.open_mwh,
            forward_price_eur_mwh=fwd_price,
            central_eur=central,
            p10_eur=p10,
            p90_eur=p90,
            sigma_assumed_pct=VOLATILITY_ANNUAL_PCT,
        )

    return out
