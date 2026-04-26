"""
Unit tests for ``demo_deviwa.pool_diversification``.

Locks the SOTA properties of the diversification module :

  - Sub-additivity of VaR : Σ VaR_i ≥ VaR_pool, with equality iff all
    actors share the same exposure sign.
  - Z-score table & ES factor are correct for the standard CL levels.
  - Forward proxy / hedge metrics are properly composed (uses Phase 2
    machinery — covered separately by test_portfolio_yearly).
  - Edge cases : empty pool, single actor, all-zero exposures, missing
    forward, all-same-sign vs mixed-sign.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "demo_deviwa"))

from pool_diversification import (  # noqa: E402
    Z_BY_CONF,
    ActorRiskMetrics,
    PoolRiskMetrics,
    _es_factor,
    _years_to_delivery,
    _z_score,
    compute_actor_risk,
    compute_load_correlation_matrix,
    compute_pool_diversification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = pd.Timestamp("2026-04-26", tz="Europe/Zurich")


@pytest.fixture
def forwards_simple():
    """Single-day forward sheet with Cal-2027 + Cal-2028 + a Q* fallback for 2026."""
    last_date = pd.Timestamp("2026-04-23")
    return pd.DataFrame(
        [
            # 2026 has no Cal — must fall back to Q-weighted average
            {"date": last_date, "market": "CH", "product": "Q03_2026_BASE", "load_type": "base", "price": 80.0},
            {"date": last_date, "market": "CH", "product": "Q04_2026_BASE", "load_type": "base", "price": 100.0},
            # 2027 / 2028 have Cal
            {"date": last_date, "market": "CH", "product": "Y01_2027_BASE", "load_type": "base", "price": 90.0},
            {"date": last_date, "market": "CH", "product": "Y01_2028_BASE", "load_type": "base", "price": 80.0},
        ]
    )


def _make_actor_row(actor: str, year: int, scope: str, volume: float, price: float = 80.0) -> dict:
    return {
        "actor": actor,
        "deal": f"{actor}_{year}_{scope[:2]}",
        "month": pd.Timestamp(year=year, month=6, day=1),
        "scope": scope,
        "volume_sum": volume,
        "notional_sum": -volume * price if scope == "Intake" else volume * price,
        "pnl_sum": 0.0,
        "trade_date": pd.Timestamp("2026-01-15"),
    }


@pytest.fixture
def deals_two_actors():
    """A is long (under-hedged), B is short (over-hedged) on the same year."""
    return pd.DataFrame(
        [
            _make_actor_row("A", 2027, "Intake", 100.0),  # 100 MWh hedged
            _make_actor_row("B", 2027, "Intake", 800.0),  # 800 MWh hedged (over-hedge if prog=500)
        ]
    )


@pytest.fixture
def programme_two_actors():
    """A: 0.05 MW flat (= 438 MWh/year). B: 0.057 MW flat (= 500 MWh/year)."""
    ts = pd.date_range("2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich")
    return pd.DataFrame(
        [
            *[
                {"actor": "A", "timestamp": t, "programme_mw": 0.05, "actual_mw": np.nan, "delta_mw": np.nan}
                for t in ts
            ],
            *[
                {"actor": "B", "timestamp": t, "programme_mw": 0.0571, "actual_mw": np.nan, "delta_mw": np.nan}
                for t in ts
            ],
        ]
    )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def test_z_table_known_values():
    """z-scores for the standard 1-tailed VaR confidence levels."""
    assert Z_BY_CONF[0.90] == pytest.approx(1.2816, abs=1e-4)
    assert Z_BY_CONF[0.95] == pytest.approx(1.6449, abs=1e-4)
    assert Z_BY_CONF[0.975] == pytest.approx(1.96, abs=1e-4)
    assert Z_BY_CONF[0.99] == pytest.approx(2.3263, abs=1e-4)


def test_z_score_interpolation_in_table_range():
    """For an unsupported level, _z_score interpolates between the bracketing entries."""
    z = _z_score(0.96)  # between 0.95 (1.6449) and 0.975 (1.96)
    assert 1.6449 < z < 1.96


def test_z_score_clips_outside_table():
    """For levels below or above the table extremes, return the bracket value."""
    assert _z_score(0.5) == Z_BY_CONF[0.90]
    assert _z_score(0.9999) == Z_BY_CONF[0.999]


def test_z_score_999_in_table():
    """0.999 is now in the table → exact value, no silent under-estimation."""
    assert _z_score(0.999) == pytest.approx(3.0902, abs=1e-4)


def test_es_factor_for_normal_at_95pct():
    """ES factor = φ(z) / (1 − α). For α=0.95, z=1.6449 → factor ≈ 2.0626."""
    z = _z_score(0.95)
    factor = _es_factor(z, 0.95)
    expected = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / 0.05
    assert factor == pytest.approx(expected, rel=1e-9)
    assert factor == pytest.approx(2.0626, abs=1e-3)


def test_es_factor_greater_than_z_for_normal():
    """For a normal distribution, ES > VaR at the same confidence level."""
    for cl in (0.90, 0.95, 0.975, 0.99):
        z = _z_score(cl)
        assert _es_factor(z, cl) > z, f"ES factor must exceed z at {cl}"


def test_years_to_delivery_floor():
    """Even mid-delivery years return at least 0.25 years horizon."""
    today = pd.Timestamp("2026-06-15", tz="Europe/Zurich")
    h = _years_to_delivery(2026, today)
    assert h == pytest.approx(0.25)


def test_years_to_delivery_normal():
    today = pd.Timestamp("2026-04-26", tz="Europe/Zurich")
    h = _years_to_delivery(2028, today)
    # 1 July 2028 is ~2.18 years from 26 April 2026
    assert 2.0 < h < 2.3


# ---------------------------------------------------------------------------
# Per-actor risk
# ---------------------------------------------------------------------------

def test_actor_risk_under_hedged_is_long_exposure(deals_two_actors, programme_two_actors, forwards_simple):
    """A has 100 MWh hedged of 438 programme → open = +338 MWh = long."""
    r = compute_actor_risk("A", 2027, deals_two_actors, programme_two_actors, forwards_simple, today=TODAY)
    assert r is not None
    assert r.open_mwh > 0
    assert r.exposure_eur > 0
    assert r.var_eur > 0
    assert r.es_eur > r.var_eur


def test_actor_risk_over_hedged_is_short_exposure(deals_two_actors, programme_two_actors, forwards_simple):
    """B has 800 MWh hedged of 500 programme → open = −300 MWh = short."""
    r = compute_actor_risk("B", 2027, deals_two_actors, programme_two_actors, forwards_simple, today=TODAY)
    assert r is not None
    assert r.open_mwh < 0
    assert r.exposure_eur < 0
    assert r.var_eur > 0  # VaR is always ≥ 0 (uses |exposure|)
    assert r.es_eur > r.var_eur


def test_actor_risk_no_data_returns_none(deals_two_actors, programme_two_actors, forwards_simple):
    """An actor with no data on the requested year returns None."""
    r = compute_actor_risk("ZZZ", 2027, deals_two_actors, programme_two_actors, forwards_simple, today=TODAY)
    assert r is None


def test_actor_risk_no_forward_returns_nan_metrics(deals_two_actors, programme_two_actors):
    """If forward is unavailable, the risk metrics are NaN but open_mwh stays."""
    empty_forwards = pd.DataFrame(columns=["date", "market", "product", "load_type", "price"])
    r = compute_actor_risk("A", 2027, deals_two_actors, programme_two_actors, empty_forwards, today=TODAY)
    assert r is not None
    assert np.isnan(r.forward_price_eur_mwh)
    assert np.isnan(r.var_eur)
    assert r.open_mwh != 0  # the underlying hedge metric still computed


# ---------------------------------------------------------------------------
# Pool diversification — the central SOTA property
# ---------------------------------------------------------------------------

def test_pool_subadditivity_holds_with_mixed_signs(deals_two_actors, programme_two_actors, forwards_simple):
    """Sub-additivity is STRICT when signs differ → diversification benefit > 0."""
    p = compute_pool_diversification(2027, deals_two_actors, programme_two_actors, forwards_simple, today=TODAY)
    assert p.pool_var_eur < p.sum_individual_var_eur, "VaR_pool must be strictly less than Σ VaR_i for mixed signs"
    assert p.diversification_benefit_eur > 0
    assert 0 < p.diversification_pct < 100


def test_pool_no_benefit_when_all_same_sign(forwards_simple):
    """Two long actors → exposures sum without offset → benefit = 0."""
    deals = pd.DataFrame(
        [
            _make_actor_row("X", 2027, "Intake", 100.0),
            _make_actor_row("Y", 2027, "Intake", 200.0),
        ]
    )
    ts = pd.date_range("2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich")
    programme = pd.DataFrame(
        [
            *[{"actor": "X", "timestamp": t, "programme_mw": 0.5, "actual_mw": np.nan, "delta_mw": np.nan} for t in ts],
            *[{"actor": "Y", "timestamp": t, "programme_mw": 0.7, "actual_mw": np.nan, "delta_mw": np.nan} for t in ts],
        ]
    )
    p = compute_pool_diversification(2027, deals, programme, forwards_simple, today=TODAY)
    assert p.pool_var_eur == pytest.approx(p.sum_individual_var_eur, rel=1e-9)
    assert p.diversification_benefit_eur == pytest.approx(0.0, abs=1e-6)


def test_pool_full_offset_when_exposures_cancel(forwards_simple):
    """Two actors with exact opposite exposures → pool VaR ≈ 0, benefit ≈ Σ VaR."""
    deals = pd.DataFrame(
        [
            _make_actor_row("LONG", 2027, "Intake", 100.0),  # 100 MWh hedge → open = -100 vs prog=0
            _make_actor_row("SHRT", 2027, "Withdrawal", 100.0),  # negative hedge → +100 open
        ]
    )
    # Programme: just enough so that LONG ends up over-hedged by 100 vs SHRT under-hedged by 100
    ts = pd.date_range("2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich")
    programme = pd.DataFrame(
        [
            # LONG programme 0 → open = 0 - 100 = -100 (over-hedged)
            *[{"actor": "LONG", "timestamp": t, "programme_mw": 0.0, "actual_mw": np.nan, "delta_mw": np.nan} for t in ts],
            # SHRT programme 0 → open = 0 - (-100) = +100
            *[{"actor": "SHRT", "timestamp": t, "programme_mw": 0.0, "actual_mw": np.nan, "delta_mw": np.nan} for t in ts],
        ]
    )
    p = compute_pool_diversification(2027, deals, programme, forwards_simple, today=TODAY)
    # Net pool exposure = -100×fwd + 100×fwd = 0
    assert abs(p.net_exposure_eur) < 1e-3
    assert p.pool_var_eur == pytest.approx(0.0, abs=1e-3)
    assert p.diversification_benefit_eur == pytest.approx(p.sum_individual_var_eur, rel=1e-9)
    assert p.diversification_pct == pytest.approx(100.0, rel=1e-9)


def test_pool_single_actor_zero_benefit(forwards_simple):
    """A single actor in the pool → benefit = 0 (nothing to diversify against)."""
    deals = pd.DataFrame([_make_actor_row("ONLY", 2027, "Intake", 100.0)])
    ts = pd.date_range("2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich")
    programme = pd.DataFrame(
        [{"actor": "ONLY", "timestamp": t, "programme_mw": 0.5, "actual_mw": np.nan, "delta_mw": np.nan} for t in ts]
    )
    p = compute_pool_diversification(2027, deals, programme, forwards_simple, today=TODAY)
    assert p.pool_var_eur == pytest.approx(p.sum_individual_var_eur, rel=1e-9)
    assert p.diversification_benefit_eur == pytest.approx(0.0, abs=1e-6)


def test_pool_var_scales_with_z_score(deals_two_actors, programme_two_actors, forwards_simple):
    """Higher confidence → higher VaR (linear in z)."""
    p_95 = compute_pool_diversification(2027, deals_two_actors, programme_two_actors, forwards_simple, today=TODAY, confidence_level=0.95)
    p_99 = compute_pool_diversification(2027, deals_two_actors, programme_two_actors, forwards_simple, today=TODAY, confidence_level=0.99)
    ratio = p_99.pool_var_eur / p_95.pool_var_eur
    expected_ratio = Z_BY_CONF[0.99] / Z_BY_CONF[0.95]
    assert ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_pool_es_greater_than_var(deals_two_actors, programme_two_actors, forwards_simple):
    """For normal returns, ES > VaR at the same confidence level."""
    p = compute_pool_diversification(2027, deals_two_actors, programme_two_actors, forwards_simple, today=TODAY)
    assert p.pool_es_eur > p.pool_var_eur
    assert p.sum_individual_es_eur > p.sum_individual_var_eur


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def test_correlation_matrix_diagonal_is_one(programme_two_actors):
    """Self-correlation must be 1 by definition."""
    corr = compute_load_correlation_matrix(programme_two_actors, freq="D")
    if corr.empty:
        pytest.skip("not enough data for correlation in fixture")
    for actor in corr.index:
        assert corr.loc[actor, actor] == pytest.approx(1.0, abs=1e-9)


def test_correlation_matrix_is_symmetric(programme_two_actors):
    corr = compute_load_correlation_matrix(programme_two_actors, freq="D")
    if corr.empty:
        pytest.skip("not enough data")
    for a in corr.index:
        for b in corr.columns:
            if pd.notna(corr.loc[a, b]) and pd.notna(corr.loc[b, a]):
                assert corr.loc[a, b] == pytest.approx(corr.loc[b, a], abs=1e-9)


def test_correlation_matrix_empty_input():
    """Empty programme → empty matrix (no crash)."""
    empty = pd.DataFrame(columns=["actor", "timestamp", "programme_mw"])
    corr = compute_load_correlation_matrix(empty)
    assert corr.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_pool_no_forward_propagates_nan(deals_two_actors, programme_two_actors):
    """Empty forwards → "cannot price risk", surface as NaN.

    This is semantically different from "no risk" (= 0). The dashboard
    layer must distinguish the two states (e.g. show "—" / "Daten nicht
    verfügbar" instead of "0 EUR Risiko, 0 % Diversifikation").
    """
    empty_forwards = pd.DataFrame(columns=["date", "market", "product", "load_type", "price"])
    p = compute_pool_diversification(2027, deals_two_actors, programme_two_actors, empty_forwards, today=TODAY)
    assert np.isnan(p.pool_var_eur)
    assert np.isnan(p.sum_individual_var_eur)
    assert np.isnan(p.diversification_benefit_eur)
    assert np.isnan(p.diversification_pct)
    assert np.isnan(p.net_exposure_eur)
    assert np.isnan(p.pool_es_eur)


def test_pool_benefit_clamps_floating_point_noise():
    """In a same-sign pool, residual FP noise must clamp to 0 (no negative benefits)."""
    # Two long actors with hedge=programme exactly (open=0 for both)
    deals = pd.DataFrame(
        [
            _make_actor_row("X", 2027, "Intake", 100.0),
            _make_actor_row("Y", 2027, "Intake", 200.0),
        ]
    )
    ts = pd.date_range("2027-01-01 00:00", "2027-12-31 23:00", freq="h", tz="Europe/Zurich")
    programme = pd.DataFrame(
        [
            *[{"actor": "X", "timestamp": t, "programme_mw": 0.05, "actual_mw": np.nan, "delta_mw": np.nan} for t in ts],
            *[{"actor": "Y", "timestamp": t, "programme_mw": 0.10, "actual_mw": np.nan, "delta_mw": np.nan} for t in ts],
        ]
    )
    forwards_simple = pd.DataFrame([
        {"date": pd.Timestamp("2026-04-23"), "market": "CH", "product": "Y01_2027_BASE", "load_type": "base", "price": 90.0},
    ])
    p = compute_pool_diversification(2027, deals, programme, forwards_simple, today=TODAY)
    # benefit must be exactly 0.0, not e.g. -1.45e-11
    assert p.diversification_benefit_eur == 0.0
    assert p.diversification_pct == 0.0


def test_pool_actor_universe_filter(deals_two_actors, programme_two_actors, forwards_simple):
    """Restricting to one actor must drop the other from the pool."""
    p = compute_pool_diversification(
        2027, deals_two_actors, programme_two_actors, forwards_simple,
        actors=["A"], today=TODAY,
    )
    assert len(p.actors) == 1
    assert p.actors[0].actor == "A"
