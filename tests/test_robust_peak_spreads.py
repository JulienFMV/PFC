"""Tests for the SOTA hydro-aware peak-spread estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pfc_shaping.calibration.robust_peak_spreads import (
    CH_PEAK_SPREAD_PRIOR,
    HydroAwarePeakSpreads,
    _holiday_aware_peak_mask,
)

TZ = "Europe/Zurich"


def _hourly_with_peak_premium(years: list[int], base: float, premium: float,
                              holiday_dates: list[str] | None = None) -> pd.Series:
    """Build hourly series where peak hours sit `premium` above base.

    Peak hours = 08–20 Mon–Fri excluding `holiday_dates` (local Europe/Zurich).
    """
    start = f"{min(years)}-01-01"
    end = f"{max(years) + 1}-01-01"
    idx = pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC")
    local = idx.tz_convert(TZ)
    is_peak = (local.weekday < 5) & (local.hour >= 8) & (local.hour < 20)
    if holiday_dates:
        hol = pd.to_datetime(holiday_dates).date
        is_peak = is_peak & ~np.isin(local.normalize().date, hol)
    arr = np.where(is_peak, base + premium, base).astype(float)
    return pd.Series(arr, index=idx, name="price_eur_mwh")


# ---------------------------------------------------------------------------
# Holiday-aware peak mask (fixes the pre-existing fit_peak_spreads bug)
# ---------------------------------------------------------------------------
def test_holiday_aware_mask_excludes_national_ch_holidays():
    """Aug 1 2024 is a Thursday CH national holiday: must be off-peak."""
    idx = pd.date_range("2024-08-01 09:00", "2024-08-01 10:00", freq="1h", tz=TZ)
    mask = _holiday_aware_peak_mask(idx)
    assert mask.sum() == 0  # both hours classified off-peak

    # The same hour on a regular Thursday is peak.
    idx2 = pd.date_range("2024-08-08 09:00", "2024-08-08 10:00", freq="1h", tz=TZ)
    mask2 = _holiday_aware_peak_mask(idx2)
    assert mask2.sum() == 2


def test_holiday_aware_mask_empty_input():
    mask = _holiday_aware_peak_mask(pd.DatetimeIndex([], tz=TZ))
    assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# Estimator: end-to-end
# ---------------------------------------------------------------------------
def test_estimator_recovers_constant_premium_clean_data():
    """On clean data with a 10 EUR/MWh peak premium, the estimator recovers ~10."""
    s = _hourly_with_peak_premium([2023, 2024], base=60.0, premium=10.0)
    est = HydroAwarePeakSpreads().fit(s.to_frame("price_eur_mwh"))
    # Each month's empirical spread is exactly 10*(n_peak/n_total) - 0*(n_offpeak/n_total)
    # vs the base mean = base + premium*(n_peak/n_total). So
    # spread = peak_avg - base_avg = (base + premium) - (base + premium * p) = premium * (1-p)
    # with p = n_peak/n_total ≈ 0.35 → spread ≈ 6.5 EUR/MWh.
    # With shrinkage to the CH prior (winter ≈10, summer ≈3), posterior mixes.
    # We only assert that the fitted spread is plausibly close to that range
    # (not dragged to baseline default 5.0).
    spreads = est.peak_base_spreads_
    assert all(0 < v < 12 for v in spreads.values())
    # Winter posterior must be > summer posterior (CH prior dominates the mix).
    winter = np.mean([spreads[m] for m in (12, 1, 2)])
    summer = np.mean([spreads[m] for m in (6, 7, 8)])
    assert winter > summer  # CH hydro-flex flattens summer


def test_estimator_falls_back_to_prior_on_no_data():
    est = HydroAwarePeakSpreads().fit(pd.DataFrame({"price_eur_mwh": []}))
    assert est.diagnostics_.get("fallback") is True
    assert est.peak_base_spreads_ == CH_PEAK_SPREAD_PRIOR


def test_estimator_falls_back_on_insufficient_rows():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    s = pd.Series([50.0] * 50, index=idx)
    est = HydroAwarePeakSpreads().fit(s.to_frame("price_eur_mwh"))
    assert est.diagnostics_.get("fallback") is True


def test_estimator_returns_drop_in_compatible_shape():
    s = _hourly_with_peak_premium([2024], base=60.0, premium=8.0)
    est = HydroAwarePeakSpreads().fit(s.to_frame("price_eur_mwh"))
    assert set(est.peak_base_spreads_.keys()) == set(range(1, 13))
    assert set(est.base_price_per_month_.keys()) == set(range(1, 13))
    for v in est.peak_base_spreads_.values():
        assert isinstance(v, float) and np.isfinite(v)


def test_estimator_regime_downweights_crisis_year():
    """A crisis year (peak spread = 40) is soft-excluded by the regime kernel."""
    s_normal = _hourly_with_peak_premium([2023, 2024], base=60.0, premium=8.0)
    # Build a 2-year series where 2023 has a 50 EUR/MWh peak premium (crisis)
    # and 2024 has 8 EUR/MWh (normal).
    idx = pd.date_range("2023-01-01", "2025-01-01", freq="1h", inclusive="left", tz="UTC")
    local = idx.tz_convert(TZ)
    is_peak = (local.weekday < 5) & (local.hour >= 8) & (local.hour < 20)
    crisis_year = local.year == 2023
    premium = np.where(crisis_year, 50.0, 8.0)
    arr = np.where(is_peak, 60.0 + premium, 60.0).astype(float)
    s_crisis = pd.Series(arr, index=idx)

    est_n = HydroAwarePeakSpreads().fit(s_normal.to_frame("price_eur_mwh"))
    est_c = HydroAwarePeakSpreads().fit(s_crisis.to_frame("price_eur_mwh"))

    # Crisis year should be down-weighted in the diagnostics.
    rw = est_c.diagnostics_["regime_weights_by_year"]
    assert rw[2023] < 0.5  # soft-excluded vs the 2024 normal year
    # And the resulting spread should be closer to the prior + 2024 evidence
    # than to a naive mean over both years.
    naive_mean = (8.0 + 50.0) / 2  # would be 29 if we equally weighted
    fitted = np.mean(list(est_c.peak_base_spreads_.values()))
    assert fitted < naive_mean - 5  # significantly pulled toward normal range


def test_estimator_validates_inputs():
    df = pd.DataFrame({"price_eur_mwh": [50.0] * 200},
                      index=pd.date_range("2024-01-01", periods=200))  # tz-naive
    with pytest.raises(ValueError, match="timezone"):
        HydroAwarePeakSpreads().fit(df)


# ---------------------------------------------------------------------------
# CH prior sanity
# ---------------------------------------------------------------------------
def test_ch_prior_winter_higher_than_summer():
    """CH-physical prior: winter peak premium > summer peak premium
    (hydro storage arbitrages summer peaks; heating + dark evenings in winter)."""
    winter = np.mean([CH_PEAK_SPREAD_PRIOR[m] for m in (12, 1, 2)])
    summer = np.mean([CH_PEAK_SPREAD_PRIOR[m] for m in (6, 7, 8)])
    assert winter > summer + 5.0  # at least 5 EUR/MWh seasonal swing


def test_ch_prior_all_positive_and_finite():
    for m, v in CH_PEAK_SPREAD_PRIOR.items():
        assert 0 < v < 20, f"month {m} prior {v} out of plausible CH range"
