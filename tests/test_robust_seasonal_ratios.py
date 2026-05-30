"""Tests for the SOTA regime-aware seasonal-ratios estimator.

Fast unit tests on synthetic fixtures (no real EPEX builds). The full A/B
benchmark on real data lives in `scripts/run_perfect_foresight.py --ab`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pfc_shaping.calibration.robust_seasonal_ratios import (
    CH_MONTHLY_PRIOR,
    CH_QUARTERLY_PRIOR,
    RegimeAwareSeasonalRatios,
    _regime_weights,
    _renormalise_ratios,
    _weighted_median,
)

TZ = "Europe/Zurich"


# ---------------------------------------------------------------------------
# Synthetic-data fixtures
# ---------------------------------------------------------------------------
def _hourly_2yr(scale_per_month: dict[int, float], year_factor: dict[int, float]) -> pd.Series:
    """Build a 2-year hourly UTC series with deterministic seasonal structure.

    Each (year, month) cell takes a constant price = ``year_factor[year] *
    scale_per_month[month]``. The local-tz year boundary is respected via the
    same Europe/Zurich convention the module uses.
    """
    idx = pd.date_range("2022-12-15", "2025-01-15", freq="1h", inclusive="left", tz="UTC")
    local = idx.tz_convert(TZ)
    yr = local.year.values
    mo = local.month.values
    price = np.array([year_factor.get(int(y), 1.0) * scale_per_month.get(int(m), 1.0)
                      for y, m in zip(yr, mo)], dtype=float)
    return pd.Series(price, index=idx, name="price_eur_mwh")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_renormalise_keeps_average_at_one():
    out = _renormalise_ratios({1: 2.0, 2: 4.0})  # mean 3 → normalize
    assert_allclose(np.mean(list(out.values())), 1.0, atol=1e-12)


def test_renormalise_handles_empty_and_zero():
    assert _renormalise_ratios({}) == {}
    out = _renormalise_ratios({1: 0.0, 2: 0.0})
    assert out == {1: 0.0, 2: 0.0}


def test_weighted_median_against_unweighted():
    """Equal weights → weighted median equals the unweighted median."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones_like(v)
    assert_allclose(_weighted_median(v, w), 3.0, atol=1e-12)


def test_weighted_median_shifts_with_weights():
    """Heavy weight on a lower value pulls the weighted median down."""
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([10.0, 1.0, 1.0])  # the 1.0 carries most mass
    assert _weighted_median(v, w) == 1.0


# ---------------------------------------------------------------------------
# Regime weights — Marcjasz et al. 2025 anti-crisis kernel
# ---------------------------------------------------------------------------
def test_regime_weights_normal_year_close_to_one():
    level = pd.Series({2023: 80.0, 2024: 75.0})  # median 77.5, distance 2.5
    w = _regime_weights(level, scale=30.0)
    assert w[2023] > 0.9 and w[2024] > 0.9


def test_regime_weights_soft_exclude_crisis():
    """A crisis-level year (e.g. 230 EUR/MWh) is heavily down-weighted."""
    level = pd.Series({2022: 230.0, 2023: 80.0, 2024: 75.0})  # median 80
    w = _regime_weights(level, scale=30.0)
    assert w[2022] < 0.01      # crisis soft-excluded (exp(-150/30) ≈ 0.007)
    assert w[2023] > 0.95      # at median: weight 1.0
    # 2024 at distance 5 from median: exp(-5/30) ≈ 0.85 — still close to 1
    assert w[2024] > 0.7


# ---------------------------------------------------------------------------
# Estimator: end-to-end on a synthetic 2-year series
# ---------------------------------------------------------------------------
def test_estimator_recovers_synthetic_seasonality():
    """On a clean 2-year synthetic series, the estimator recovers the shape."""
    # Build a series where summer (Jul=0.7) and winter (Jan=1.3) are perfectly
    # constant within each (year, month). Annual mean = 1.0 by construction.
    truth = {1: 1.3, 2: 1.2, 3: 1.05, 4: 0.95, 5: 0.85, 6: 0.80,
             7: 0.70, 8: 0.80, 9: 0.95, 10: 1.05, 11: 1.15, 12: 1.20}
    s = _hourly_2yr(truth, {2023: 1.0, 2024: 1.0})
    est = RegimeAwareSeasonalRatios().fit(s.to_frame("price_eur_mwh"))
    m = est.seasonal_ratios_["month"]
    # The recovered ratios should rank-order match the truth (a partial-year
    # 2022-12 and 2025-01 add a tiny shrinkage, plus the literature prior
    # contributes 0.5/(2+0.5) ≈ 20% — exact recovery is not expected).
    # Rank-correlation > 0.9 is a tight check that the SHAPE is recovered.
    from scipy.stats import spearmanr
    rho, _ = spearmanr(list(m.values()), [truth[k] for k in m])
    assert rho > 0.9, f"Spearman rank corr too low: {rho:.3f}"


def test_estimator_shrinks_to_prior_with_no_data():
    """An almost-empty series collapses to the CH-physical prior."""
    # Single-day series (insufficient observations per month) → most cells
    # fall through to the prior fallback.
    idx = pd.date_range("2024-06-01", "2024-06-02", freq="1h", inclusive="left", tz="UTC")
    s = pd.Series([100.0] * len(idx), index=idx)
    est = RegimeAwareSeasonalRatios().fit(s.to_frame("price_eur_mwh"))
    m = est.seasonal_ratios_["month"]
    # The 11 months with no data must equal the renormalised prior; the only
    # observed month (June) gets ratio ≈ 1 / (yearly_mean = 100) = 1.0, then
    # shrunk and renormalised. We assert the OVERALL SHAPE matches the prior.
    from scipy.stats import spearmanr
    rho, _ = spearmanr(list(m.values()),
                       [CH_MONTHLY_PRIOR[k] for k in m])
    assert rho > 0.8


def test_estimator_regime_downweights_crisis_year():
    """A crisis year embedded in the training data must be soft-excluded.

    Uses realistic price scales (~50-250 EUR/MWh) so the default regime_scale=30
    parameter is meaningful. The crisis year (200 EUR/MWh) is ~4× the median.
    """
    s_normal = _hourly_2yr({m: 50.0 for m in range(1, 13)},
                           {2023: 1.0, 2024: 1.0})
    s_crisis = _hourly_2yr({m: 50.0 for m in range(1, 13)},
                           {2023: 4.0, 2024: 1.0})  # 2023 = 200 EUR/MWh (crisis)
    est_n = RegimeAwareSeasonalRatios().fit(s_normal.to_frame("price_eur_mwh"))
    est_c = RegimeAwareSeasonalRatios().fit(s_crisis.to_frame("price_eur_mwh"))
    rw_crisis = est_c.diagnostics_["regime_weights_by_year"]
    assert rw_crisis[2023] < 0.05, f"crisis year not down-weighted: {rw_crisis}"
    # The fit should be stable across normal/crisis training (both have a
    # normal 2024, so 2024 carries the load when 2023 is down-weighted).
    m_n = est_n.seasonal_ratios_["month"]; m_c = est_c.seasonal_ratios_["month"]
    diffs = np.array([m_n[k] - m_c[k] for k in m_n])
    assert np.abs(diffs).max() < 0.5  # crisis didn't blow up the estimate


def test_estimator_returns_drop_in_compatible_shape():
    """The output must match the baseline fitter's contract."""
    s = _hourly_2yr({m: 1.0 + 0.1 * (m - 6) for m in range(1, 13)},
                    {2023: 1.0, 2024: 1.0})
    est = RegimeAwareSeasonalRatios().fit(s.to_frame("price_eur_mwh"))
    sr = est.seasonal_ratios_
    assert set(sr.keys()) == {"quarter", "month"}
    assert set(sr["quarter"].keys()) == {1, 2, 3, 4}
    assert set(sr["month"].keys()) == {m for m in range(1, 13)}
    assert all(isinstance(v, float) for v in sr["month"].values())
    # Trends are None under this estimator (regime-stationary assumption).
    assert est.seasonal_trends_ is None


def test_estimator_validates_inputs():
    with pytest.raises(ValueError, match="empty"):
        RegimeAwareSeasonalRatios().fit(pd.DataFrame({"price_eur_mwh": []}))
    df = pd.DataFrame({"foo": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    with pytest.raises(ValueError, match="price_eur_mwh"):
        RegimeAwareSeasonalRatios().fit(df)
    df2 = pd.DataFrame({"price_eur_mwh": [50.0]},
                       index=pd.date_range("2024-01-01", periods=1))  # tz-naive
    with pytest.raises(ValueError, match="timezone"):
        RegimeAwareSeasonalRatios().fit(df2)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
def test_ch_priors_are_normalised():
    # Monthly prior mean should be close to 1 (within 5% — small deviations
    # are fine since downstream renormalisation handles it).
    assert 0.95 < float(np.mean(list(CH_MONTHLY_PRIOR.values()))) < 1.05
    assert 0.95 < float(np.mean(list(CH_QUARTERLY_PRIOR.values()))) < 1.05


def test_ch_monthly_prior_winter_higher_than_summer():
    """CH-physical prior must reflect winter > summer (hydro storage + heating)."""
    winter = np.mean([CH_MONTHLY_PRIOR[m] for m in (12, 1, 2)])
    summer = np.mean([CH_MONTHLY_PRIOR[m] for m in (6, 7, 8)])
    assert winter > summer + 0.2  # at least 20% premium
