"""Unit tests for ``demo_deviwa.hedge_strategy``.

Pins the FMV-to-GRD hedge engine :

  - Open-position curve = programme − flat-spread hedge per delivery
    month, sign-aware (Intake +, Withdrawal −) so an over-hedged
    actor surfaces as negative open MWh.
  - Blotter recommendation : Cal-Base only for baseload-heavy
    profiles ; Cal-Base + Q-Peak split for peak-heavy profiles ;
    nothing recommended for the current delivery year (FMV intraday
    desk territory).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "demo_deviwa"))

from hedge_strategy import (  # noqa: E402
    HedgeAction,
    derive_open_position_per_year,
    recommend_hedge_blotter,
)
from portfolio_yearly import INDUSTRY_HEDGE_TARGETS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def programme_flat_2027():
    """0.5 MW flat hourly programme for 2027 (8760 hours, no leap)."""
    ts = pd.date_range(
        "2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich"
    )
    return pd.DataFrame({
        "actor": "TEST",
        "timestamp": ts,
        "programme_mw": np.full(len(ts), 0.5),
        "actual_mw": np.nan,
        "delta_mw": np.nan,
    })


@pytest.fixture
def programme_peak_heavy_2027():
    """Peak-heavy 2027 programme : 1.0 MW during Mon-Fri 08-20, 0.2 MW otherwise."""
    ts = pd.date_range(
        "2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich"
    )
    is_peak = (ts.dayofweek < 5) & (ts.hour >= 8) & (ts.hour < 20)
    mw = np.where(is_peak, 1.0, 0.2)
    return pd.DataFrame({
        "actor": "TEST", "timestamp": ts,
        "programme_mw": mw, "actual_mw": np.nan, "delta_mw": np.nan,
    })


@pytest.fixture
def deals_simple_2027():
    """100 MWh Intake hedge per month for 2027 = 1200 MWh annual."""
    rows = []
    for month in range(1, 13):
        rows.append({
            "actor": "TEST",
            "deal": f"D_2027_M{month:02d}",
            "month": pd.Timestamp(year=2027, month=month, day=1),
            "scope": "Intake",
            "volume_sum": 100.0,
            "notional_sum": -8000.0,
            "pnl_sum": 0.0,
            "trade_date": pd.Timestamp("2026-03-15"),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def forwards_with_peak():
    """CH BASE + PEAK forwards for Y01_2027 + Q01..Q04 PEAK 2027."""
    rows = [
        ("Y01_2027_BASE", "base", 90.0),
        ("Y01_2027_PEAK", "peak", 110.0),
        ("Q01_2027_PEAK", "peak", 130.0),  # winter peak high
        ("Q02_2027_PEAK", "peak", 95.0),
        ("Q03_2027_PEAK", "peak", 100.0),
        ("Q04_2027_PEAK", "peak", 125.0),
    ]
    return pd.DataFrame([
        {"date": pd.Timestamp("2026-04-23"), "market": "CH",
         "product": p, "load_type": lt, "price": pr}
        for (p, lt, pr) in rows
    ])


# ---------------------------------------------------------------------------
# derive_open_position_per_year
# ---------------------------------------------------------------------------

def test_open_position_no_deals_equals_programme(programme_flat_2027):
    """No deals → open MWh per hour = programme MWh per hour, exactly."""
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    out = derive_open_position_per_year(
        programme_flat_2027, pd.DataFrame(), actor="TEST", today=today
    )
    assert 2027 in out
    s = out[2027]
    assert len(s) == 8760
    assert s.iloc[0] == pytest.approx(0.5)
    assert float(s.sum()) == pytest.approx(0.5 * 8760)


def test_open_position_intake_reduces_open(programme_flat_2027, deals_simple_2027):
    """1200 MWh Intake hedge / 4380 MWh programme → 3180 MWh open annually."""
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    out = derive_open_position_per_year(
        programme_flat_2027, deals_simple_2027, actor="TEST", today=today
    )
    s = out[2027]
    expected_annual_open = 0.5 * 8760 - 1200.0
    assert float(s.sum()) == pytest.approx(expected_annual_open, rel=1e-9)


def test_open_position_withdrawal_widens_open(programme_flat_2027):
    """Withdrawal deals are *negative* signed volume — they make the
    actor *shorter*, so the open MWh widens (the actor buys *more* at
    market because it has already sold some forward)."""
    deals = pd.DataFrame([{
        "actor": "TEST", "deal": "D_WD",
        "month": pd.Timestamp(2027, 6, 1),
        "scope": "Withdrawal",
        "volume_sum": 200.0,
        "notional_sum": 16000.0,
        "pnl_sum": 0.0,
        "trade_date": pd.Timestamp("2026-03-15"),
    }])
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    out = derive_open_position_per_year(
        programme_flat_2027, deals, actor="TEST", today=today
    )
    june_hours = ((out[2027].index.month == 6)).sum()
    expected_annual = 0.5 * 8760 - (-200.0)  # selling 200 → open widens by 200
    assert float(out[2027].sum()) == pytest.approx(expected_annual, rel=1e-9)
    # June hours should reflect the per-hour bump: programme 0.5 + 200/N_hours_june
    june_open = out[2027].loc[out[2027].index.month == 6]
    expected_per_june_hour = 0.5 + 200.0 / june_hours
    assert (june_open.round(6) == round(expected_per_june_hour, 6)).all()


def test_open_position_overhedged_goes_negative(programme_flat_2027):
    """A deal larger than the monthly programme produces negative open MWh
    in that month — the heatmap should colour it differently. Pin the sign."""
    deals = pd.DataFrame([{
        "actor": "TEST", "deal": "D_BIG",
        "month": pd.Timestamp(2027, 7, 1),
        "scope": "Intake",
        "volume_sum": 1000.0,         # July has only 0.5 MW × 744 h = 372 MWh prog
        "notional_sum": -80000.0, "pnl_sum": 0.0,
        "trade_date": pd.Timestamp("2026-03-15"),
    }])
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    out = derive_open_position_per_year(
        programme_flat_2027, deals, actor="TEST", today=today
    )
    july_open = out[2027].loc[out[2027].index.month == 7]
    assert (july_open < 0).all()


# ---------------------------------------------------------------------------
# recommend_hedge_blotter
# ---------------------------------------------------------------------------

def test_blotter_baseload_profile_recommends_cal_base_only(
    programme_flat_2027, forwards_with_peak
):
    """Flat 0.5 MW profile → 0 % peak share → 100 % Cal-Base, no Q-strip."""
    profile = pd.Series(
        programme_flat_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_flat_2027["timestamp"]),
    )
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    actions, meta = recommend_hedge_blotter(profile, 2027, today, forwards_with_peak)
    assert len(actions) == 1
    assert actions[0].instrument == "Cal-2027 BASE"
    assert meta["q_strip_active"] is False
    # Y+1 2027 from 2026 → corridor (80, 100), midpoint = 90 %
    assert meta["target_ratio_pct"] == pytest.approx(90.0)
    expected_volume = float(profile.sum()) * 0.90
    assert actions[0].volume_mwh == pytest.approx(expected_volume, rel=1e-9)


def test_blotter_peak_heavy_splits_cal_and_q(
    programme_peak_heavy_2027, forwards_with_peak
):
    """Peak-heavy profile → 70 % Cal-Base + 30 % Q-Peak winter."""
    profile = pd.Series(
        programme_peak_heavy_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_peak_heavy_2027["timestamp"]),
    )
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    actions, meta = recommend_hedge_blotter(profile, 2027, today, forwards_with_peak)
    assert meta["q_strip_active"] is True
    # Peak share for this profile: peak_hours × 1.0 / total_volume.
    # 261 weekdays × 12 h × 1.0 = 3132 MWh peak
    # Total = 3132 + (8760 - 3132) × 0.2 = 3132 + 1125.6 = 4257.6 MWh
    # share = 3132 / 4257.6 ≈ 73.6 %
    assert 70.0 < meta["peak_share_pct"] < 80.0
    assert len(actions) == 2
    assert actions[0].instrument == "Cal-2027 BASE"
    assert actions[1].instrument == "Q1+Q4 2027 PEAK"
    total_target = float(profile.sum()) * 0.90  # corridor mid
    assert actions[0].volume_mwh == pytest.approx(total_target * 0.70, rel=1e-9)
    assert actions[1].volume_mwh == pytest.approx(total_target * 0.30, rel=1e-9)


def test_blotter_current_year_returns_empty(programme_flat_2027, forwards_with_peak):
    """Current delivery year is intraday-desk territory — no GRD action."""
    profile = pd.Series(
        programme_flat_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_flat_2027["timestamp"]),
    )
    today = pd.Timestamp("2027-04-27", tz="Europe/Zurich")  # already in 2027
    actions, meta = recommend_hedge_blotter(profile, 2027, today, forwards_with_peak)
    assert actions == []
    assert meta["year_offset"] == 0


def test_blotter_no_forward_returns_empty_with_meta(programme_flat_2027):
    """Empty forwards → no actions but metadata still populated for UI."""
    profile = pd.Series(
        programme_flat_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_flat_2027["timestamp"]),
    )
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    actions, meta = recommend_hedge_blotter(profile, 2027, today, pd.DataFrame())
    assert actions == []
    assert meta["annual_volume_mwh"] > 0
    assert meta["target_ratio_pct"] == pytest.approx(90.0)


def test_peak_share_robust_to_mixed_sign_positions(forwards_with_peak):
    """Open-position curves can mix positive (under-hedged) and negative
    (over-hedged) hours. The peak share must stay between 0-100 % and
    correctly reflect the fraction of the position's energy magnitude
    that falls in EEX peak hours.
    """
    from hedge_strategy import _peak_share_pct
    ts = pd.date_range("2027-01-05", periods=24, freq="h", tz="Europe/Zurich")
    # Mon 05 Jan 2027 : peak hours 08-19 = 12 hours.
    # Set peak hours to +2 MW, off-peak to -1 MW.
    is_peak = (ts.dayofweek < 5) & (ts.hour >= 8) & (ts.hour < 20)
    values = np.where(is_peak, 2.0, -1.0)
    profile = pd.Series(values, index=ts)
    # |peak| sum = 12 × 2 = 24 ; |total| sum = 24 + 12 × 1 = 36 ; share = 24/36 = 66.67 %
    share = _peak_share_pct(profile)
    assert 0.0 <= share <= 100.0
    assert share == pytest.approx(100.0 * 24 / 36, rel=1e-9)


def test_blotter_corridor_ramps_down_with_horizon(
    programme_flat_2027, forwards_with_peak
):
    """Y+1 should recommend more volume than Y+3 — corridor enforces this."""
    profile = pd.Series(
        programme_flat_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_flat_2027["timestamp"]),
    )
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    rows = [
        {"date": pd.Timestamp("2026-04-23"), "market": "CH",
         "product": f"Y01_{y}_BASE", "load_type": "base", "price": 90.0}
        for y in (2027, 2028, 2029, 2030)
    ]
    fwd = pd.concat([forwards_with_peak, pd.DataFrame(rows)], ignore_index=True)

    vols = []
    for y in (2027, 2028, 2029, 2030):
        actions, _ = recommend_hedge_blotter(profile, y, today, fwd)
        vols.append(actions[0].volume_mwh if actions else 0.0)

    # Strictly decreasing : Y+1 > Y+2 > Y+3 > Y+4
    assert vols[0] > vols[1] > vols[2] > vols[3] > 0


def test_blotter_net_long_position_does_not_recommend_buy(
    programme_flat_2027, forwards_with_peak
):
    """Net-long (over-hedged) open position must NOT receive a buy
    recommendation. This is a regression guard : an earlier version of
    the page absoluted the curve before passing it to the engine, so
    a -892 MWh net-long actor was getting a "buy 1880 MWh Cal-2027
    BASE" — the directional opposite of what they need.
    """
    # Negate the flat 2027 programme so the profile is uniformly net long
    profile = -1.0 * pd.Series(
        programme_flat_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_flat_2027["timestamp"]),
    )
    assert float(profile.sum()) < 0  # fixture sanity
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    actions, meta = recommend_hedge_blotter(profile, 2027, today, forwards_with_peak)
    assert actions == []
    assert meta["status"] == "over_hedged"
    assert meta["annual_volume_mwh"] < 0  # signed sum preserved
    assert meta["annual_volume_abs_mwh"] > 0


def test_blotter_balanced_position_returns_no_action(forwards_with_peak):
    """A profile with offsetting positive and negative hours but a near-
    zero net falls into the ``balanced`` branch — no action ticket.
    """
    ts = pd.date_range(
        "2027-01-01 00:00", periods=24, freq="h", tz="Europe/Zurich"
    )
    # 12 hours of +1 MW, 12 hours of -1 MW → net 0 over the day.
    values = np.array([1.0 if i < 12 else -1.0 for i in range(24)])
    profile = pd.Series(values, index=ts)
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    actions, meta = recommend_hedge_blotter(profile, 2027, today, forwards_with_peak)
    assert actions == []
    assert meta["status"] == "balanced"


def test_blotter_q1q4_peak_price_uses_winter_average_not_caly_peak(
    programme_peak_heavy_2027, forwards_with_peak
):
    """The peak-heavy branch must price the ``Q1+Q4 PEAK`` ticket with
    a true Q01+Q04 weighted average, not whatever ``Y01_{year}_PEAK``
    settles at. The fixture has Y01_PEAK=110 vs Q01_PEAK=130, Q04_PEAK
    =125 — picking the Cal-Y proxy would under-price the winter strip.
    """
    profile = pd.Series(
        programme_peak_heavy_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_peak_heavy_2027["timestamp"]),
    )
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    actions, _ = recommend_hedge_blotter(profile, 2027, today, forwards_with_peak)
    # Two actions : Cal-Base + Q1+Q4 PEAK
    q_action = next(a for a in actions if a.instrument.startswith("Q1+Q4"))

    # Closed-form expectation : peak hours per quarter × price, averaged.
    from hedge_strategy import _quarter_peak_hours
    h1 = _quarter_peak_hours(2027, 1)
    h4 = _quarter_peak_hours(2027, 4)
    expected = (130.0 * h1 + 125.0 * h4) / (h1 + h4)
    assert q_action.reference_price_eur_mwh == pytest.approx(expected, rel=1e-9)
    # And it is decisively NOT the Y01_2027_PEAK fixture price (= 110.0)
    assert abs(q_action.reference_price_eur_mwh - 110.0) > 5.0


def test_blotter_q1q4_peak_falls_back_to_cal_base_when_strip_missing(
    programme_peak_heavy_2027,
):
    """If Q01 or Q04 PEAK is missing from the cache, the engine must
    NOT silently price the winter ticket with Cal-Y PEAK. Instead it
    falls back to all-Cal-Base for the full target volume, with a
    rationale that names the missing input.
    """
    # Forwards with Cal-Base + Cal-Peak only (no Q-PEAK)
    fwd = pd.DataFrame([
        {"date": pd.Timestamp("2026-04-23"), "market": "CH",
         "product": "Y01_2027_BASE", "load_type": "base", "price": 90.0},
        {"date": pd.Timestamp("2026-04-23"), "market": "CH",
         "product": "Y01_2027_PEAK", "load_type": "peak", "price": 110.0},
    ])
    profile = pd.Series(
        programme_peak_heavy_2027["programme_mw"].values,
        index=pd.DatetimeIndex(programme_peak_heavy_2027["timestamp"]),
    )
    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    actions, meta = recommend_hedge_blotter(profile, 2027, today, fwd)
    assert len(actions) == 1
    assert actions[0].instrument == "Cal-2027 BASE"
    # Full target volume on Cal-Base since the winter strip couldn't price
    target = float(profile.sum()) * 0.90
    assert actions[0].volume_mwh == pytest.approx(target, rel=1e-9)
