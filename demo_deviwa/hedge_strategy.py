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
        _quarter_hours,
        _scope_signed_volume,
        get_forward_year_proxy,
    )
except ImportError:
    from portfolio_yearly import (  # type: ignore[no-redef]
        INDUSTRY_HEDGE_TARGETS,
        _filter_actor,
        _quarter_hours,
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
    drove the decision (corridor, target ratio, peak share, etc.) so
    the UI can render an explainer alongside the table.

    Strategy (FMV / Swiss-DSO oriented) :

      1. Pick the corridor mid-point from
         :data:`INDUSTRY_HEDGE_TARGETS` for the year offset
         ``y - today.year`` (Y+1 → 90 %, Y+2 → 65 %, Y+3 → 35 %,
         Y+4 → 15 %). The current delivery year is *not* recommended
         here — at that horizon the action belongs to FMV's intraday
         desk, not the GRD's hedge plan.

      2. Cal-Base is the primary instrument (the only liquid Swiss
         tenor for a GRD-sized ticket).

      3. If the profile is **peak-heavy** (peak share > 40 %), layer
         a Q-strip on top : 70 % of the recommended volume on Cal-Base,
         30 % on the **winter** quarters (Q1 + Q4) where the peak
         premium is largest. The Q-strip uses the latest available
         peak forward, weighted by peak hours per quarter.

      4. M-contracts are deliberately excluded — Swiss M liquidity
         is too thin to absorb GRD volumes without spread cost.

    Args:
        profile_mwh: hourly load curve (MWh per hour, one calendar year).
        delivery_year: Y for which to recommend the hedge.
        today: reference date.
        forwards: long-format forward frame with ``date, market,
            product, load_type, price``.

    Returns:
        ``(actions, meta)`` where ``actions`` is the list of
        :class:`HedgeAction` rows (possibly empty if no forward is
        available) and ``meta`` carries the explanatory variables :
        ``corridor``, ``target_ratio_pct``, ``peak_share_pct``,
        ``annual_volume_mwh``, ``q_strip_active``.
    """
    annual_volume = float(profile_mwh.sum())
    peak_share = _peak_share_pct(profile_mwh)
    offset = delivery_year - today.year
    corridor = INDUSTRY_HEDGE_TARGETS.get(offset)
    meta: dict = {
        "annual_volume_mwh": annual_volume,
        "peak_share_pct": peak_share,
        "year_offset": offset,
        "corridor": corridor,
        "target_ratio_pct": None,
        "q_strip_active": False,
    }

    # No corridor (offset 0 = current year, or > 4) → no recommendation.
    if corridor is None or offset <= 0:
        return [], meta

    target_ratio = 0.5 * (corridor[0] + corridor[1]) / 100.0
    meta["target_ratio_pct"] = target_ratio * 100.0
    target_volume = annual_volume * target_ratio
    if target_volume <= 0:
        return [], meta

    # Cal-Base price for the year (or Q-strip-weighted proxy if Cal-Y
    # has retired — covered by get_forward_year_proxy).
    cal_base_price = get_forward_year_proxy(
        forwards, delivery_year, market="CH", load_type="base"
    )
    if not np.isfinite(cal_base_price):
        return [], meta

    actions: list[HedgeAction] = []

    # Decide Cal-Base vs Cal-Base + Q-Peak split.
    is_peak_heavy = np.isfinite(peak_share) and peak_share > 40.0
    meta["q_strip_active"] = is_peak_heavy

    if is_peak_heavy:
        # 70 % via Cal-Base, 30 % via Q-Peak winter (Q1 + Q4)
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

        # Q-Peak winter strip : Q1 and Q4 of the same year, peak load
        peak_proxy = get_forward_year_proxy(
            forwards, delivery_year, market="CH", load_type="peak"
        )
        if np.isfinite(peak_proxy):
            actions.append(HedgeAction(
                instrument=f"Q1+Q4 {delivery_year} PEAK",
                delivery_year=delivery_year,
                volume_mwh=q_volume,
                reference_price_eur_mwh=peak_proxy,
                notional_eur=q_volume * peak_proxy,
                rationale=(
                    f"Spitzenlast-Anteil ({peak_share:.0f} % > 40 %) — "
                    f"Q1 + Q4 PEAK fängt die Winter-Spitzen ab."
                ),
            ))
        else:
            # Peak forward not available → fall back to all Cal-Base
            actions = []
            actions.append(HedgeAction(
                instrument=f"Cal-{delivery_year} BASE",
                delivery_year=delivery_year,
                volume_mwh=target_volume,
                reference_price_eur_mwh=cal_base_price,
                notional_eur=target_volume * cal_base_price,
                rationale=(
                    f"Profil ist spitzenlastig ({peak_share:.0f} %) aber "
                    f"PEAK-Forward nicht verfügbar — komplette Volume auf "
                    f"Cal-Base."
                ),
            ))
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
