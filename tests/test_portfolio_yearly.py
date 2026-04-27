"""
Unit tests for ``demo_deviwa.portfolio_yearly``.

Locks the two behaviours flagged by the Phase 2 audit:

  1. ``Y01_{year}`` filter must select calendar-year strips for future years
     and reject Q* / M* tenors that contain the same year string.
  2. Current delivery year must NOT silently degrade to a NaN budget when
     ``Y01_{year}`` has already retired — the quarterly-weighted proxy in
     ``get_forward_year_proxy`` is the canonical fallback.

Plus a few sanity checks on the central hedge-ratio formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Locate `demo_deviwa` regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent / "demo_deviwa"))

from portfolio_yearly import (  # noqa: E402
    INDUSTRY_HEDGE_TARGETS,
    Z_P10_P90,
    build_qstrip_year_series,
    classify_hedge_status,
    compute_budget_projection,
    compute_hedge_by_year,
    get_forward_cal_price,
    get_forward_year_proxy,
    get_target_corridor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forwards_mixed():
    """Single-day forward sheet with all of Y01, Q01-Q04, M05-M06 for 2026 + 2027.

    Mirrors the real `Price_Report_EEX_Yearly.xlsx` schema after the
    extract pipeline (long-format with date / market / product /
    load_type / price columns).
    """
    last_date = pd.Timestamp("2026-04-23")
    rows = [
        # 2026 — Cal already retired (no Y01_2026), only Q3, Q4 and a few months
        ("CH000EEX_Q032026_BASE Q03_2026_BASE", "base", 84.45),
        ("CH000EEX_Q042026_BASE Q04_2026_BASE", "base", 120.45),
        ("CH000EEX_M052026_BASE M05_2026_BASE", "base", 78.06),
        ("CH000EEX_M062026_BASE M06_2026_BASE", "base", 74.96),
        # 2027 — full Cal + quarterlies (different prices to detect mismatch)
        ("CH000EEX_Y012027_BASE Y01_2027_BASE", "base", 90.79),
        ("CH000EEX_Q012027_BASE Q01_2027_BASE", "base", 120.43),  # would pollute
        ("CH000EEX_Q022027_BASE Q02_2027_BASE", "base", 64.55),
        ("CH000EEX_Q032027_BASE Q03_2027_BASE", "base", 67.58),
        ("CH000EEX_Q042027_BASE Q04_2027_BASE", "base", 110.98),
        # 2027 PEAK product (must NOT be picked when load_type='base')
        ("CH000EEX_Y012027_PEAK Y01_2027_PEAK", "peak", 99.99),
    ]
    return pd.DataFrame(
        [
            {"date": last_date, "market": "CH", "product": p, "load_type": lt, "price": v}
            for (p, lt, v) in rows
        ]
    )


@pytest.fixture
def deals_minimal():
    """One actor "TEST" with 12 monthly Intake rows for 2027 = 1200 MWh hedged."""
    rows = []
    for month in range(1, 13):
        rows.append(
            {
                "actor": "TEST",
                "deal": f"TEST_2027_M{month:02d}",
                "month": pd.Timestamp(year=2027, month=month, day=1),
                "scope": "Intake",
                "volume_sum": 100.0,  # 100 MWh per month
                "notional_sum": -8000.0,  # client pays out 8000 EUR (price = 80 EUR/MWh)
                "pnl_sum": 0.0,
                "trade_date": pd.Timestamp("2026-03-15"),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def programme_minimal():
    """One actor "TEST" with 0.2 MW flat programme for 2027 (= 1752 MWh / year)."""
    ts = pd.date_range("2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich")
    return pd.DataFrame(
        {
            "actor": "TEST",
            "timestamp": ts,
            "programme_mw": np.full(len(ts), 0.2),
            "actual_mw": np.nan,
            "delta_mw": np.nan,
        }
    )


# ---------------------------------------------------------------------------
# Behaviour 1 : Y01_{year} must be unambiguous
# ---------------------------------------------------------------------------

def test_get_forward_cal_price_picks_y01_for_future_year(forwards_mixed):
    """For 2027, must return Y01 (90.79), NOT Q01 (120.43) or others."""
    p = get_forward_cal_price(forwards_mixed, 2027, market="CH", load_type="base")
    assert p == pytest.approx(90.79)


def test_get_forward_cal_price_returns_nan_for_year_without_cal(forwards_mixed):
    """For 2026 there is no Y01_2026 in the fixture → NaN (strict Cal lookup)."""
    p = get_forward_cal_price(forwards_mixed, 2026)
    assert np.isnan(p)


def test_get_forward_cal_price_respects_load_type(forwards_mixed):
    """Y01_2027_PEAK must not be returned when caller asked for base."""
    p_base = get_forward_cal_price(forwards_mixed, 2027, load_type="base")
    p_peak = get_forward_cal_price(forwards_mixed, 2027, load_type="peak")
    assert p_base == pytest.approx(90.79)
    assert p_peak == pytest.approx(99.99)


# ---------------------------------------------------------------------------
# Behaviour 2 : current-year proxy via Q* aggregation
# ---------------------------------------------------------------------------

def test_get_forward_year_proxy_uses_cal_when_available(forwards_mixed):
    """For future year 2027, must use the Y01 directly (= Cal price)."""
    p = get_forward_year_proxy(forwards_mixed, 2027)
    assert p == pytest.approx(90.79)


def test_get_forward_year_proxy_falls_back_to_quarters(forwards_mixed):
    """For current year 2026 (no Y01), must average available Q3 + Q4 weighted
    by quarter hours.

    Q3_2026: 84.45 EUR over 92 days × 24 = 2208 hours
    Q4_2026: 120.45 EUR over 92 days × 24 = 2208 hours
    Weighted avg = (84.45 × 2208 + 120.45 × 2208) / (2208 × 2) = 102.45
    """
    p = get_forward_year_proxy(forwards_mixed, 2026)
    expected = (84.45 * 2208 + 120.45 * 2208) / (2208 * 2)
    assert p == pytest.approx(expected, rel=1e-6)


def test_get_forward_year_proxy_returns_nan_when_nothing(forwards_mixed):
    """No tenor for 2030 → NaN (no fallback can save this)."""
    p = get_forward_year_proxy(forwards_mixed, 2030)
    assert np.isnan(p)


# ---------------------------------------------------------------------------
# Behaviour 2b : synthetic Q-strip series for the current delivery year
# ---------------------------------------------------------------------------

def test_build_qstrip_series_yields_one_row_per_settlement_date():
    """The Q-strip series is one weighted-average row per settlement date,
    not one row per quarter-day. Pinned because the Marktüberblick page
    plots ``date → price`` directly and would render a stair-step if the
    weighting was applied per-row instead of per-day.
    """
    rows = []
    # Two settlement dates, all four quarters present on each
    for d in (pd.Timestamp("2026-04-22"), pd.Timestamp("2026-04-23")):
        for q, price in zip(range(1, 5), [120.0, 60.0, 80.0, 100.0]):
            rows.append({
                "date": d, "market": "CH",
                "product": f"Q0{q}_2026_BASE",
                "load_type": "base", "price": price,
            })
    forwards = pd.DataFrame(rows)
    series = build_qstrip_year_series(forwards, 2026, market="CH", load_type="base")
    assert len(series) == 2
    # 2026 is non-leap. Q1=2160h, Q2=2184h, Q3=2208h, Q4=2208h → 8760h total.
    expected = (120.0 * 2160 + 60.0 * 2184 + 80.0 * 2208 + 100.0 * 2208) / 8760
    assert (series["price"].round(4) == round(expected, 4)).all()
    assert (series["product"] == "Q-Strip 2026 (BASE)").all()


def test_build_qstrip_series_drops_dates_with_no_quarters():
    """When the source frame has no Q* contracts for the year, return an
    empty frame — no synthetic row should be invented from thin air."""
    rows = [
        {"date": pd.Timestamp("2026-04-23"), "market": "CH",
         "product": "Y01_2027_BASE", "load_type": "base", "price": 90.0},
    ]
    forwards = pd.DataFrame(rows)
    series = build_qstrip_year_series(forwards, 2026, market="CH", load_type="base")
    assert series.empty


def test_build_qstrip_series_handles_partial_quarter_coverage():
    """Half-way through 2026 the tail of the strip is Q3 + Q4 only — must
    average those two and ignore the missing Q1/Q2 rather than returning
    NaN. Mirrors the real EEX cache state in spring of the current year."""
    rows = [
        {"date": pd.Timestamp("2026-07-01"), "market": "CH",
         "product": "Q03_2026_BASE", "load_type": "base", "price": 84.0},
        {"date": pd.Timestamp("2026-07-01"), "market": "CH",
         "product": "Q04_2026_BASE", "load_type": "base", "price": 120.0},
    ]
    forwards = pd.DataFrame(rows)
    series = build_qstrip_year_series(forwards, 2026, market="CH", load_type="base")
    assert len(series) == 1
    expected = (84.0 * 2208 + 120.0 * 2208) / (2208 + 2208)
    assert series["price"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_build_qstrip_series_peak_weights_by_peak_hours_not_calendar():
    """PEAK strip valuation must weight quarters by EEX peak hours
    (Mon-Fri 08:00-20:00), not calendar hours.

    Pinned because the original implementation reused ``_quarter_hours``
    for both BASE and PEAK, biasing the synthetic Cal-Peak series. In
    practice the two weight sets are numerically close on any single
    calendar year (peak fraction ≈ 36 % of every quarter), so this test
    locks the *formula* rather than chasing a magnitude gap : the
    helper output must match the closed-form peak-hour weighted mean
    to numerical precision.
    """
    from portfolio_yearly import _quarter_peak_hours
    q_peak = {q: _quarter_peak_hours(2026, q) for q in (1, 2, 3, 4)}

    rows = []
    prices = [180.0, 90.0, 110.0, 160.0]
    for q, price in zip(range(1, 5), prices):
        rows.append({
            "date": pd.Timestamp("2026-04-23"), "market": "CH",
            "product": f"Q0{q}_2026_PEAK",
            "load_type": "peak", "price": price,
        })
    forwards = pd.DataFrame(rows)
    series = build_qstrip_year_series(forwards, 2026, market="CH", load_type="peak")
    assert len(series) == 1
    expected_peak = (
        sum(p * q_peak[q] for q, p in zip(range(1, 5), prices))
        / sum(q_peak.values())
    )
    assert series["price"].iloc[0] == pytest.approx(expected_peak, rel=1e-12)
    assert (series["product"] == "Q-Strip 2026 (PEAK)").all()


def test_quarter_peak_hours_matches_explicit_count():
    """Sanity check the peak-hour helper against an independent count.

    Q1 2026 peak hours = 12 hours/day × number of weekdays in Jan-Mar
    2026. Computed twice : once via the production helper, once via
    a direct ``date_range`` filtered on weekday + hour-of-day.
    """
    from portfolio_yearly import _quarter_peak_hours
    naive = pd.date_range(
        "2026-01-01 00:00", "2026-03-31 23:00", freq="h"
    )
    explicit = ((naive.dayofweek < 5) & (naive.hour >= 8) & (naive.hour < 20)).sum()
    assert _quarter_peak_hours(2026, 1) == int(explicit)


def test_get_forward_year_proxy_peak_uses_peak_hours():
    """Same fix mirrored on the latest-tenor proxy used by the budget
    projection. A PEAK current-year proxy must weight available
    quarters by peak hours, not calendar hours."""
    from portfolio_yearly import _quarter_peak_hours
    rows = []
    for q, price in zip(range(1, 5), [180.0, 90.0, 110.0, 160.0]):
        rows.append({
            "date": pd.Timestamp("2026-04-23"), "market": "CH",
            "product": f"Q0{q}_2026_PEAK",
            "load_type": "peak", "price": price,
        })
    fwd = pd.DataFrame(rows)
    p = get_forward_year_proxy(fwd, 2026, market="CH", load_type="peak")
    q_peak = {q: _quarter_peak_hours(2026, q) for q in (1, 2, 3, 4)}
    expected = sum(pr * q_peak[q] for q, pr in zip(range(1, 5), [180.0, 90.0, 110.0, 160.0])) / sum(q_peak.values())
    assert p == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Behaviour 3 : current-year budget must NOT be all-NaN
# ---------------------------------------------------------------------------

def test_compute_budget_projection_current_year_has_value(deals_minimal, forwards_mixed):
    """For current year 2026, the budget proxy uses Q-weighted avg, so
    central_eur must be a finite number (not NaN).

    Constructs deals + programme for 2026 directly (using `tz_localize`
    with explicit DST handling, since a naïve `−1y` would hit the spring
    forward gap on 2026-03-29 02:00).
    """
    deals_2026 = deals_minimal.copy()
    deals_2026["month"] = deals_2026["month"].apply(lambda t: t.replace(year=2026))
    deals_2026["deal"] = deals_2026["deal"].str.replace("2027", "2026")

    ts_2026 = pd.date_range(
        "2026-01-01 00:00", "2026-12-31 23:00", freq="h", tz="Europe/Zurich"
    )
    prog_2026 = pd.DataFrame(
        {
            "actor": "TEST",
            "timestamp": ts_2026,
            "programme_mw": np.full(len(ts_2026), 0.2),
            "actual_mw": np.nan,
            "delta_mw": np.nan,
        }
    )

    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_budget_projection(
        deals_2026, prog_2026, forwards_mixed, actor="TEST", today=today
    )
    bp = out[2026]
    assert np.isfinite(bp.forward_price_eur_mwh), "current-year forward proxy must not be NaN when Q tenors exist"
    assert np.isfinite(bp.central_eur)
    assert np.isfinite(bp.p10_eur)
    assert np.isfinite(bp.p90_eur)
    assert bp.p10_eur < bp.central_eur < bp.p90_eur


def test_compute_budget_projection_current_year_uses_residual_volume(
    deals_minimal, forwards_mixed
):
    """For the current delivery year, the at-risk leg must price the
    residual open volume — not the full year — and the programme and
    hedged horizons must be aligned at the same calendar boundary.

    Setup (today = 2026-04-26, cutoff = 2026-05-01) :
      - programme : 0.2 MW flat for 2026
      - deals     : 12 × 100 MWh Intake for 2026 (Jan-Dec)
      - residual programme = 0.2 × hours(2026-05-01 → 2026-12-31 23:00)
                           = 0.2 × 5881         (DST autumn adds 1h)
                           = 1176.2 MWh
      - residual hedged    = 8 × 100 = 800 MWh   (May-Dec only)
      - residual open      = 1176.2 − 800 = 376.2 MWh
    """
    deals_2026 = deals_minimal.copy()
    deals_2026["month"] = deals_2026["month"].apply(lambda t: t.replace(year=2026))
    deals_2026["deal"] = deals_2026["deal"].str.replace("2027", "2026")

    ts_2026 = pd.date_range(
        "2026-01-01 00:00", "2026-12-31 23:00", freq="h", tz="Europe/Zurich"
    )
    prog_2026 = pd.DataFrame(
        {
            "actor": "TEST",
            "timestamp": ts_2026,
            "programme_mw": np.full(len(ts_2026), 0.2),
            "actual_mw": np.nan,
            "delta_mw": np.nan,
        }
    )

    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_budget_projection(
        deals_2026, prog_2026, forwards_mixed, actor="TEST", today=today
    )
    bp = out[2026]

    # Residual semantics flag is set
    assert bp.is_current_year_residual is True

    # Residual hours from 2026-05-01 00:00 to 2026-12-31 23:00 inclusive,
    # accounting for the autumn DST repeat of the 02:00 hour on Oct 25.
    cutoff = pd.Timestamp("2026-05-01 00:00", tz="Europe/Zurich")
    residual_hours = ((ts_2026 >= cutoff) & (ts_2026.year == 2026)).sum()
    expected_open = 0.2 * residual_hours - 8 * 100.0
    assert bp.open_mwh == pytest.approx(expected_open, rel=1e-6), (
        f"open_mwh must reflect the residual May-Dec window exactly ; "
        f"got {bp.open_mwh:.3f}, expected {expected_open:.3f}"
    )

    # locked + at_risk decomposition unchanged
    expected_locked = 12 * 8000.0
    assert bp.locked_eur == pytest.approx(expected_locked, rel=1e-6)
    expected_central = expected_locked + bp.open_mwh * bp.forward_price_eur_mwh
    assert bp.central_eur == pytest.approx(expected_central, rel=1e-6)


def test_compute_budget_projection_aligns_horizons_at_month_boundary(
    forwards_mixed,
):
    """Pins the audit follow-up : programme and hedge horizons must be
    aligned at the same calendar boundary.

    A naive split would keep April deals (``month_end > today``) while
    only keeping ~3 days of April programme (``timestamp > today``),
    pricing the residual ~3-day April programme against the **full**
    April hedge volume — under-stating open exposure by the elapsed
    share of the current month.

    Construct the worst-case fixture : April + May deals only, no
    other months. With today = 2026-04-27, the residual must be **May
    only** for both programme and hedge, not "tail of April + May
    programme vs April + May hedge".
    """
    rows = []
    for month, hedge_mwh in [(4, 200.0), (5, 200.0)]:
        rows.append({
            "actor": "TEST", "deal": f"D_{month:02d}",
            "month": pd.Timestamp(year=2026, month=month, day=1),
            "scope": "Intake", "volume_sum": hedge_mwh,
            "notional_sum": -hedge_mwh * 80.0, "pnl_sum": 0.0,
            "trade_date": pd.Timestamp("2026-03-15"),
        })
    deals = pd.DataFrame(rows)

    ts_2026 = pd.date_range(
        "2026-04-01 00:00", "2026-05-31 23:00", freq="h", tz="Europe/Zurich"
    )
    prog = pd.DataFrame({
        "actor": "TEST", "timestamp": ts_2026,
        "programme_mw": np.full(len(ts_2026), 1.0),  # 1 MW flat
        "actual_mw": np.nan, "delta_mw": np.nan,
    })

    today = pd.Timestamp("2026-04-27", tz="Europe/Zurich")
    bp = compute_budget_projection(
        deals, prog, forwards_mixed, actor="TEST", today=today
    )[2026]

    # Cutoff = 2026-05-01. Both programme and hedge restricted to May.
    cutoff = pd.Timestamp("2026-05-01 00:00", tz="Europe/Zurich")
    may_hours = ((ts_2026 >= cutoff) & (ts_2026.year == 2026)).sum()
    expected_open = 1.0 * may_hours - 200.0  # 1 MW × May hours − May hedge

    assert bp.open_mwh == pytest.approx(expected_open, rel=1e-6), (
        f"Misaligned horizons : got open_mwh={bp.open_mwh:.3f}, "
        f"expected {expected_open:.3f} (May programme only vs May hedge only)"
    )


def test_compute_budget_projection_future_year_uses_full_year_volume(
    deals_minimal, programme_minimal, forwards_mixed
):
    """For a *future* delivery year the at-risk leg must keep using the
    full-year open volume — the residual semantics only kick in for the
    current year. Pins the regression boundary.
    """
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_budget_projection(
        deals_minimal, programme_minimal, forwards_mixed, actor="TEST", today=today
    )
    bp = out[2027]

    assert bp.is_current_year_residual is False
    full_year_open = 1752.0 - 1200.0
    assert bp.open_mwh == pytest.approx(full_year_open)
    # Future year uses Y01_2027_BASE = 90.79
    assert bp.forward_price_eur_mwh == pytest.approx(90.79)
    assert bp.central_eur == pytest.approx(
        12 * 8000.0 + full_year_open * 90.79, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Sanity checks on hedge ratio core formula
# ---------------------------------------------------------------------------

def test_hedge_ratio_simple_intake_only(deals_minimal, programme_minimal):
    """1200 MWh hedged / 1752 MWh programme = 68.49 % (well-formed Intake)."""
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(deals_minimal, programme_minimal, actor="TEST", today=today)
    m = out[2027]
    assert m.hedged_mwh == pytest.approx(1200.0)
    assert m.programme_mwh == pytest.approx(1752.0)
    assert m.hedge_ratio_pct == pytest.approx(100 * 1200 / 1752, rel=1e-9)
    assert m.open_mwh == pytest.approx(1752 - 1200)
    assert m.n_deals == 12


def test_hedge_ratio_intake_minus_withdrawal(programme_minimal):
    """1 Withdrawal of 200 MWh + 1 Intake of 600 MWh → net hedge = 400 MWh."""
    deals = pd.DataFrame(
        [
            {
                "actor": "TEST",
                "deal": "WD_1",
                "month": pd.Timestamp("2027-01-01"),
                "scope": "Withdrawal",
                "volume_sum": 200.0,
                "notional_sum": 16000.0,
                "pnl_sum": 0.0,
                "trade_date": pd.Timestamp("2026-01-10"),
            },
            {
                "actor": "TEST",
                "deal": "IN_1",
                "month": pd.Timestamp("2027-01-01"),
                "scope": "Intake",
                "volume_sum": 600.0,
                "notional_sum": -48000.0,
                "pnl_sum": 0.0,
                "trade_date": pd.Timestamp("2026-02-10"),
            },
        ]
    )
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(deals, programme_minimal, actor="TEST", today=today)
    m = out[2027]
    assert m.hedged_mwh == pytest.approx(400.0)  # 600 Intake − 200 Withdrawal


def test_hedge_ratio_no_programme_returns_nan(deals_minimal):
    """Deals exist but no programme → ratio NaN, no crash."""
    empty_prog = pd.DataFrame(columns=["actor", "timestamp", "programme_mw"])
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(deals_minimal, empty_prog, actor="TEST", today=today)
    m = out[2027]
    assert np.isnan(m.hedge_ratio_pct)
    assert m.programme_mwh == 0.0
    assert m.hedged_mwh == 1200.0  # still computed


def test_hedge_ratio_no_deals_returns_zero(programme_minimal):
    """Programme exists but no deals → hedged=0, ratio=0, open=full programme."""
    empty_deals = pd.DataFrame(columns=["actor", "scope", "month", "volume_sum", "notional_sum", "pnl_sum", "deal", "trade_date"])
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(empty_deals, programme_minimal, actor="TEST", today=today)
    m = out[2027]
    assert m.hedged_mwh == 0.0
    assert m.hedge_ratio_pct == pytest.approx(0.0)
    assert m.open_mwh == pytest.approx(1752.0)


# ---------------------------------------------------------------------------
# Industry corridor classification
# ---------------------------------------------------------------------------

def test_industry_corridor_lookups():
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    assert get_target_corridor(2026, today) is None  # offset 0 → in delivery
    assert get_target_corridor(2027, today) == (80, 100)
    assert get_target_corridor(2028, today) == (50, 80)
    assert get_target_corridor(2029, today) == (20, 50)
    assert get_target_corridor(2030, today) == (0, 30)
    assert get_target_corridor(2031, today) is None  # beyond Y-4


def test_classify_hedge_status_branches():
    assert classify_hedge_status(float("nan"), (50, 80)) == "no_data"
    assert classify_hedge_status(75.0, None) == "n/a"
    assert classify_hedge_status(40.0, (50, 80)) == "below"
    assert classify_hedge_status(75.0, (50, 80)) == "in_corridor"
    assert classify_hedge_status(95.0, (50, 80)) == "above"


# ---------------------------------------------------------------------------
# Programme carry-forward extrapolation
# ---------------------------------------------------------------------------

def test_extrapolation_carries_last_programme_forward(deals_minimal):
    """When deal exists for a year without programme, carry-forward 2027 → 2028."""
    # Programme covers 2027 only ; deals_minimal has 12 monthly Intake rows for 2027.
    # Add a single Intake deal for 2028.
    extra = pd.DataFrame([
        {
            "actor": "TEST",
            "deal": "TEST_2028_M01",
            "month": pd.Timestamp(year=2028, month=1, day=1),
            "scope": "Intake",
            "volume_sum": 50.0,
            "notional_sum": -4000.0,
            "pnl_sum": 0.0,
            "trade_date": pd.Timestamp("2026-03-15"),
        },
    ])
    deals_with_2028 = pd.concat([deals_minimal, extra], ignore_index=True)
    ts = pd.date_range(
        "2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich"
    )
    prog_2027 = pd.DataFrame(
        {
            "actor": "TEST",
            "timestamp": ts,
            "programme_mw": np.full(len(ts), 0.2),
            "actual_mw": np.nan,
            "delta_mw": np.nan,
        }
    )
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(
        deals_with_2028, prog_2027, actor="TEST", today=today,
        extrapolate_missing_programme=True,
    )
    # 2027 : real programme, no flag
    assert out[2027].programme_mwh == pytest.approx(1752.0)
    assert out[2027].programme_is_extrapolated is False
    # 2028 : carry-forward from 2027, flag set
    assert out[2028].programme_mwh == pytest.approx(1752.0)
    assert out[2028].programme_is_extrapolated is True
    # Ratio with 2028 hedge of 50 MWh on programme 1752 ≈ 2.85 %
    assert out[2028].hedge_ratio_pct == pytest.approx(100 * 50 / 1752, rel=1e-6)


def test_extrapolation_off_keeps_zero_programme(deals_minimal):
    """When extrapolation is disabled, missing year has programme=0 and ratio=NaN."""
    extra = pd.DataFrame([
        {
            "actor": "TEST",
            "deal": "TEST_2028_M01",
            "month": pd.Timestamp(year=2028, month=1, day=1),
            "scope": "Intake",
            "volume_sum": 50.0,
            "notional_sum": -4000.0,
            "pnl_sum": 0.0,
            "trade_date": pd.Timestamp("2026-03-15"),
        },
    ])
    deals_with_2028 = pd.concat([deals_minimal, extra], ignore_index=True)
    ts = pd.date_range(
        "2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich"
    )
    prog_2027 = pd.DataFrame(
        {
            "actor": "TEST",
            "timestamp": ts,
            "programme_mw": np.full(len(ts), 0.2),
            "actual_mw": np.nan,
            "delta_mw": np.nan,
        }
    )
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(
        deals_with_2028, prog_2027, actor="TEST", today=today,
        extrapolate_missing_programme=False,
    )
    assert out[2028].programme_mwh == 0.0
    assert np.isnan(out[2028].hedge_ratio_pct)
    assert out[2028].programme_is_extrapolated is False


def test_include_years_forces_card_for_actor_without_data():
    """include_years=[2027, 2028] gives entries even when no deals/programme."""
    empty_deals = pd.DataFrame(columns=["actor", "scope", "month", "volume_sum", "notional_sum", "pnl_sum", "deal", "trade_date"])
    empty_prog = pd.DataFrame(columns=["actor", "timestamp", "programme_mw"])
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(
        empty_deals, empty_prog, actor="UNKNOWN", today=today,
        include_years=[2027, 2028],
    )
    assert set(out.keys()) == {2027, 2028}
    assert out[2027].programme_mwh == 0.0
    assert out[2027].hedged_mwh == 0.0


def test_extrapolation_does_not_extrapolate_backward(deals_minimal):
    """Programme 2028 should NOT be extrapolated to fill 2026 (only forward)."""
    ts = pd.date_range(
        "2028-01-01 00:00", "2028-12-31 23:00", freq="h", tz="Europe/Zurich"
    )
    prog_2028 = pd.DataFrame(
        {
            "actor": "TEST",
            "timestamp": ts,
            "programme_mw": np.full(len(ts), 0.2),
            "actual_mw": np.nan,
            "delta_mw": np.nan,
        }
    )
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    out = compute_hedge_by_year(
        deals_minimal, prog_2028, actor="TEST", today=today,
        extrapolate_missing_programme=True,
        include_years=[2026, 2027, 2028],
    )
    # 2026, 2027 are BEFORE the only data year (2028) → no source to carry forward
    assert out[2026].programme_mwh == 0.0
    assert out[2026].programme_is_extrapolated is False
    assert out[2027].programme_mwh == 0.0
    assert out[2027].programme_is_extrapolated is False
    # 2028 has actual data
    assert out[2028].programme_mwh > 0
    assert out[2028].programme_is_extrapolated is False
