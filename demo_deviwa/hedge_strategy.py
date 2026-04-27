"""GRD-facing hedge strategy : engine + recommendation rules.

Two pure functions, no Streamlit state, no plotting :

  - :func:`derive_open_position_per_year` turns an actor's
    ``programme − hedged`` into one hourly MWh series per delivery year,
    so the rest of the dashboard can treat an open position exactly
    like an uploaded load profile (same heatmaps, same valuation, same
    stress sensitivity).

  - :func:`recommend_hedge_blotter` produces a small, defensible action
    list for a single delivery year and a given annual MWh volume :
    "buy X MWh of Cal-Base @ today's forward, layer Y MWh of Q-Peak if
    the profile is peak-heavy". Built for the FMV-to-GRD use case where
    the customer is a small Swiss DSO, not a trading desk.

The recommendation logic is deliberately conservative for the Swiss
market : Cal-Base is the primary instrument (= the only liquid one),
Q-strips only layer on top when the profile shape calls for it, and
M-contracts are excluded (liquidity is too thin for a GRD-sized
ticket). Corridors come from :data:`INDUSTRY_HEDGE_TARGETS` — same
table the cockpit year cards already use, so a "buy this" suggested
here lands the actor squarely in the green zone there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .portfolio_yearly import (
        INDUSTRY_HEDGE_TARGETS,
        _filter_actor,
        _latest_tenor_price,
        _quarter_hours,
        _quarter_peak_hours,
        _scope_signed_volume,
        get_forward_year_proxy,
    )
except ImportError:
    from portfolio_yearly import (  # type: ignore[no-redef]
        INDUSTRY_HEDGE_TARGETS,
        _filter_actor,
        _latest_tenor_price,
        _quarter_hours,
        _quarter_peak_hours,
        _scope_signed_volume,
        get_forward_year_proxy,
    )


# ---------------------------------------------------------------------------
# Open-position curve
# ---------------------------------------------------------------------------

def derive_open_position_per_year(
    programme: pd.DataFrame,
    deals: pd.DataFrame,
    actor: str | None,
    today: pd.Timestamp | None = None,
    horizon_years: int = 4,
) -> dict[int, pd.Series]:
    """Hourly open-position curve per delivery year for one actor or the pool.

    For each delivery year ``y`` in ``[today.year, today.year + horizon_years)``
    the function returns a tz-aware hourly Series in MWh, computed as
    ``programme_mw − hedged_mw_at_hour``. Cal-Base / Q-Base / M-Base
    contracts deliver flat, so the per-hour hedge is the per-month deal
    volume divided by the number of hours in that month — the standard
    convention used by Volue / KYOS / EEX for shaping a strip into an
    hourly curve.

    The Series is **not** clipped to positive values : a negative open
    means the actor is over-hedged on that hour (selling surplus at
    spot), which the heatmap should surface with a different colour.

    Args:
        programme: long-format with ``timestamp, actor, programme_mw``.
        deals: long-format with ``actor, deal, month, scope, volume_sum``.
        actor: actor name, or ``None`` for the consolidated pool aggregate.
        today: reference date ; defaults to current Europe/Zurich time.
        horizon_years: how many years of curves to emit, starting at
            ``today.year``. Default 4 to match the cockpit horizon.

    Returns:
        ``{year: hourly_series_mwh}``. Missing years (no programme rows)
        are absent from the dict.
    """
    if today is None:
        today = pd.Timestamp.now(tz="Europe/Zurich")

    prog_f = _filter_actor(programme, actor, "actor")
    deals_f = _filter_actor(deals, actor, "actor")

    out: dict[int, pd.Series] = {}
    if prog_f is None or prog_f.empty or "timestamp" not in prog_f.columns:
        return out

    target_years = list(range(today.year, today.year + horizon_years))

    # Pre-aggregate deal volumes per (year, month) once, so the loop below
    # is cheap. Sign convention follows _scope_signed_volume (Intake = +,
    # Withdrawal = −) so an over-hedged actor surfaces as negative open.
    deal_by_month: dict[tuple[int, int], float] = {}
    if (
        deals_f is not None
        and not deals_f.empty
        and "month" in deals_f.columns
    ):
        d = deals_f.copy()
        d["_month"] = pd.to_datetime(d["month"], errors="coerce")
        d = d.dropna(subset=["_month"])
        if not d.empty:
            d["_signed"] = _scope_signed_volume(d)
            for (y, m), vol in d.groupby([d["_month"].dt.year, d["_month"].dt.month])["_signed"].sum().items():
                deal_by_month[(int(y), int(m))] = float(vol)

    for y in target_years:
        slice_ = prog_f[prog_f["timestamp"].dt.year == y].copy()
        if slice_.empty:
            continue
        slice_ = slice_.sort_values("timestamp")
        # Hourly programme series (already MWh per hour since each row is
        # one hour at programme_mw, and 1 MW × 1 h = 1 MWh).
        prog = pd.Series(
            pd.to_numeric(slice_["programme_mw"], errors="coerce").fillna(0.0).values,
            index=pd.DatetimeIndex(slice_["timestamp"]),
        )
        # Build a hedged-MWh-per-hour series : for each (year, month) bucket
        # we have a total signed volume ; spread it flat across all hours
        # of that month.
        hedged = pd.Series(0.0, index=prog.index)
        if deal_by_month:
            month_index = prog.index.month
            for m in range(1, 13):
                vol = deal_by_month.get((y, m), 0.0)
                if vol == 0.0:
                    continue
                mask = month_index == m
                n_hours = int(mask.sum())
                if n_hours == 0:
                    continue
                hedged.loc[mask] = vol / n_hours

        out[y] = (prog - hedged).rename("open_mwh")

    return out


# ---------------------------------------------------------------------------
# Hedge blotter recommendation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HedgeAction:
    """One row of the suggested-trades blotter."""
    instrument: str               # e.g. "Cal-2027 BASE"
    delivery_year: int
    volume_mwh: float
    reference_price_eur_mwh: float
    notional_eur: float
    rationale: str


def _peak_hours_in_year(year: int) -> int:
    """EEX Peak hours (Mon-Fri 08:00–20:00) in the calendar year."""
    idx = pd.date_range(
        f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h"
    )
    return int(((idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 20)).sum())


def _q1_q4_peak_price(
    forwards: pd.DataFrame,
    year: int,
    market: str = "CH",
) -> float:
    """Volume-weighted average of ``Q01_{year}`` and ``Q04_{year}`` PEAK
    settlements, weighted by EEX peak hours in each quarter.

    Used by the blotter when the recommended ticket is the **winter
    PEAK strip** : the displayed instrument is "Q1+Q4 {year} PEAK", so
    the reference price must reflect the actual two-quarter average,
    not whatever ``Y01_{year}_PEAK`` happens to settle at. Returns
    NaN if either quarter is missing — the caller falls back to the
    full-year peak proxy in that case.
    """
    if forwards is None or forwards.empty:
        return float("nan")
    df = forwards
    if "market" in df.columns:
        df = df[df["market"].astype(str).str.upper() == market.upper()]
    if "load_type" in df.columns:
        df = df[df["load_type"].astype(str).str.lower() == "peak"]
    if df.empty or "product" not in df.columns or "price" not in df.columns:
        return float("nan")
    q1 = _latest_tenor_price(df, f"Q01_{year}")
    q4 = _latest_tenor_price(df, f"Q04_{year}")
    if not (np.isfinite(q1) and np.isfinite(q4)):
        return float("nan")
    h1 = _quarter_peak_hours(year, 1)
    h4 = _quarter_peak_hours(year, 4)
    if h1 + h4 <= 0:
        return float("nan")
    return (q1 * h1 + q4 * h4) / (h1 + h4)


def _peak_share_pct(profile_mwh: pd.Series) -> float:
    """Volume share of EEX-peak hours in the profile (0–100 %).

    Uses absolute values so the figure stays interpretable on
    open-position curves that mix positive (under-hedged) and negative
    (over-hedged) hours. The peak share is the fraction of the
    profile's *energy magnitude* that falls in EEX peak hours, which
    is what drives the "layer a peak strip on top" decision.
    """
    if profile_mwh.empty:
        return float("nan")
    idx = profile_mwh.index
    abs_total = float(profile_mwh.abs().sum())
    if abs_total <= 0:
        return float("nan")
    is_peak = (idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 20)
    abs_peak = float(profile_mwh[is_peak].abs().sum())
    return 100.0 * abs_peak / abs_total


def recommend_hedge_blotter(
    profile_mwh: pd.Series,
    delivery_year: int,
    today: pd.Timestamp,
    forwards: pd.DataFrame,
) -> tuple[list[HedgeAction], dict]:
    """GRD-facing hedge blotter for one delivery year.

    Returns a small list of concrete trade rows ("buy X MWh Cal-2027
    BASE @ 90.79 EUR/MWh") plus a metadata dict with the inputs that
    drove the decision (corridor, target ratio, peak share, status,
    etc.) so the UI can render an explainer alongside the table.

    The function is **sign-aware** : the input profile carries the
    delivery direction the GRD needs to hedge.

      - **Net positive** (``Σ profile > +ε``) — load to procure or
        net-short open position. The function recommends a *buy*
        plan : Cal-Base + optional Q1+Q4 PEAK winter strip if the
        profile is peak-heavy.
      - **Net negative** (``Σ profile < -ε``) — net-long open
        position (the actor has sold/locked more than its load).
        No buy ticket is recommended ; ``meta["status"] =
        "over_hedged"`` and the UI surfaces an "unwind via FMV
        Trading" hint instead. Recommending an additional buy here
        would be directionally wrong and is an old bug worth pinning.
      - **Near zero** (``|Σ profile| ≤ ε``) — already balanced ;
        ``meta["status"] = "balanced"``.

    ``ε`` is set to 0.5 % of ``|Σ profile|`` so a tiny shape
    imbalance with a near-zero net doesn't trigger a buy.

    Strategy (FMV / Swiss-DSO oriented) :

      1. Corridor mid-point from :data:`INDUSTRY_HEDGE_TARGETS` for
         the year offset ``y - today.year`` (Y+1 → 90 %, Y+2 → 65 %,
         Y+3 → 35 %, Y+4 → 15 %). Current delivery year is not
         recommended here — that horizon belongs to FMV's intraday
         desk, not the GRD's hedge plan.
      2. Cal-Base is the primary instrument — only liquid Swiss
         tenor for a GRD-sized ticket.
      3. If the profile is **peak-heavy** (peak share > 40 %), layer
         a Q-strip on top : 70 % via Cal-Base, 30 % via Q1 + Q4 PEAK
         where the winter peak premium is largest. The PEAK price is
         the **weighted average of Q01 and Q04 settlements** (peak-
         hour weights), not the full-year Cal-Y PEAK proxy.
      4. M-contracts are deliberately excluded — Swiss M liquidity
         is too thin to absorb GRD volumes without spread cost.

    Args:
        profile_mwh: hourly load curve (MWh per hour, signed) for one
            calendar year.
        delivery_year: Y for which to recommend the hedge.
        today: reference date.
        forwards: long-format with ``date, market, product, load_type,
            price``.

    Returns:
        ``(actions, meta)`` where ``actions`` is the list of
        :class:`HedgeAction` rows (empty for net-long, balanced, or
        unpriceable horizons) and ``meta`` carries :
        ``annual_volume_mwh`` (signed), ``annual_volume_abs_mwh``,
        ``peak_share_pct``, ``year_offset``, ``corridor``,
        ``target_ratio_pct``, ``q_strip_active``, ``status``
        ('buy' / 'over_hedged' / 'balanced' / 'no_corridor' /
        'no_forward').
    """
    annual_signed = float(profile_mwh.sum())
    annual_abs = float(profile_mwh.abs().sum())
    peak_share = _peak_share_pct(profile_mwh)
    offset = delivery_year - today.year
    corridor = INDUSTRY_HEDGE_TARGETS.get(offset)
    meta: dict = {
        "annual_volume_mwh": annual_signed,
        "annual_volume_abs_mwh": annual_abs,
        "peak_share_pct": peak_share,
        "year_offset": offset,
        "corridor": corridor,
        "target_ratio_pct": None,
        "q_strip_active": False,
        "status": None,
    }

    # No corridor (offset 0 = current year, or > 4) → no recommendation.
    if corridor is None or offset <= 0:
        meta["status"] = "no_corridor"
        return [], meta

    # Direction guard. ``ε`` = 0.5 % of |Σ| keeps tiny shape imbalances
    # with near-zero net from triggering a buy ticket.
    epsilon = max(0.005 * annual_abs, 1.0)
    if annual_signed < -epsilon:
        meta["status"] = "over_hedged"
        return [], meta
    if annual_signed <= epsilon:
        meta["status"] = "balanced"
        return [], meta

    # From here on : the profile is meaningfully positive → buy plan.
    target_ratio = 0.5 * (corridor[0] + corridor[1]) / 100.0
    meta["target_ratio_pct"] = target_ratio * 100.0
    target_volume = annual_signed * target_ratio  # use signed sum, now > 0

    cal_base_price = get_forward_year_proxy(
        forwards, delivery_year, market="CH", load_type="base"
    )
    if not np.isfinite(cal_base_price):
        meta["status"] = "no_forward"
        return [], meta

    actions: list[HedgeAction] = []
    is_peak_heavy = np.isfinite(peak_share) and peak_share > 40.0
    meta["q_strip_active"] = is_peak_heavy
    meta["status"] = "buy"

    if is_peak_heavy:
        # 70 % via Cal-Base, 30 % via Q1+Q4 PEAK winter strip
        cal_volume = target_volume * 0.70
        q_volume = target_volume * 0.30

        actions.append(HedgeAction(
            instrument=f"Cal-{delivery_year} BASE",
            delivery_year=delivery_year,
            volume_mwh=cal_volume,
            reference_price_eur_mwh=cal_base_price,
            notional_eur=cal_volume * cal_base_price,
            rationale=(
                f"Grundlast-Anteil ({100.0 - peak_share:.0f} %) auf Cal-Base — "
                f"liquidstes Schweizer Tenor."
            ),
        ))

        # Winter strip price : true Q01+Q04 average (peak-hour-weighted),
        # NOT the full-year Cal-Y PEAK proxy — those products differ.
        winter_price = _q1_q4_peak_price(forwards, delivery_year, market="CH")
        if np.isfinite(winter_price):
            actions.append(HedgeAction(
                instrument=f"Q1+Q4 {delivery_year} PEAK",
                delivery_year=delivery_year,
                volume_mwh=q_volume,
                reference_price_eur_mwh=winter_price,
                notional_eur=q_volume * winter_price,
                rationale=(
                    f"Spitzenlast-Anteil ({peak_share:.0f} % > 40 %) — "
                    f"Q1 + Q4 PEAK fängt die Winter-Spitzen ab "
                    f"(Volumen-gewichteter Durchschnitt der beiden Quartale)."
                ),
            ))
        else:
            # No Q1 or Q4 PEAK available → fall back to all-Cal-Base.
            actions = [HedgeAction(
                instrument=f"Cal-{delivery_year} BASE",
                delivery_year=delivery_year,
                volume_mwh=target_volume,
                reference_price_eur_mwh=cal_base_price,
                notional_eur=target_volume * cal_base_price,
                rationale=(
                    f"Profil ist spitzenlastig ({peak_share:.0f} %) aber "
                    f"Q1/Q4 PEAK-Settlements unvollständig — komplette "
                    f"Volume auf Cal-Base."
                ),
            )]
    else:
        actions.append(HedgeAction(
            instrument=f"Cal-{delivery_year} BASE",
            delivery_year=delivery_year,
            volume_mwh=target_volume,
            reference_price_eur_mwh=cal_base_price,
            notional_eur=target_volume * cal_base_price,
            rationale=(
                f"Grundlastprofil "
                f"({peak_share:.0f} % Spitze < 40 %) — komplette Volume "
                f"auf Cal-Base."
            ),
        ))

    return actions, meta
