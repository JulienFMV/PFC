import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.signed_benchmark import calendar_cell_reference, closed_month_targets


def history():
    idx = pd.date_range("2024-01-02", "2025-01-01", freq="h", inclusive="left", tz="Europe/Zurich").tz_convert("UTC")
    values = 50 + 100 * np.cos(idx.tz_convert("Europe/Zurich").hour.to_numpy() * np.pi / 12)
    return pd.Series(values, index=idx)


def test_closed_months_preserve_negatives_and_exclude_open_and_partial_months():
    prices = history()
    origin = pd.Timestamp("2024-12-31T12:00Z")
    result = closed_month_targets(prices, origin)
    months = result.index.tz_convert("Europe/Zurich").strftime("%Y-%m")
    assert set(months) == {f"2024-{m:02d}" for m in range(2, 12)}
    assert result.price_eur_mwh.lt(0).any() and result.signed_target.lt(0).any()
    assert result.ratio_target.dropna().between(.2, 3).all()
    assert result.signed_target.groupby(months).mean().abs().max() < 1e-9
    changed = prices.copy()
    changed.loc[changed.index >= pd.Timestamp("2024-12-01", tz="Europe/Zurich")] = -9999.
    pd.testing.assert_frame_equal(result, closed_month_targets(changed, origin))


def test_low_mean_days_remain_in_signed_target_only():
    prices = history()
    day = prices.index.tz_convert("Europe/Zurich").strftime("%Y-%m-%d") == "2024-07-01"
    prices.loc[day] -= 70
    result = closed_month_targets(prices, pd.Timestamp("2025-01-01T01:00Z"))
    selected = result.loc[result.index.tz_convert("Europe/Zurich").strftime("%Y-%m-%d") == "2024-07-01"]
    assert len(selected) == 24 and selected.ratio_target.isna().all()
    assert selected.signed_target.notna().all()


def test_both_dst_clock_hours_remain_distinct():
    result = closed_month_targets(history(), pd.Timestamp("2025-01-01T01:00Z"))
    autumn = result.index.tz_convert("Europe/Zurich").strftime("%Y-%m-%d %H") == "2024-10-27 02"
    assert autumn.sum() == 2


def test_interior_gap_and_naive_origin_fail():
    prices = history()
    missing = prices.drop(pd.Timestamp("2024-06-15T12:00Z"))
    with pytest.raises(ValueError, match="complete consecutive"):
        closed_month_targets(missing, pd.Timestamp("2025-01-01T01:00Z"))
    with pytest.raises(ValueError, match="timezone-aware"):
        closed_month_targets(prices, pd.Timestamp("2025-01-01"))


def test_reference_keeps_signed_units_and_accounts_for_every_prediction():
    target = pd.Series([-40., -80.], index=pd.DatetimeIndex(["2024-01-02T12:00Z", "2024-01-03T12:00Z"]))
    delivery = pd.date_range("2025-07-01T12:00Z", periods=24, freq="h")
    raw, counts = calendar_cell_reference(target, delivery)
    assert np.all(raw == -60) and sum(counts.values()) == len(delivery)
    shifted, _ = calendar_cell_reference(target + 100, delivery)
    np.testing.assert_allclose(shifted, raw + 100)
    with pytest.raises(ValueError, match="precede delivery"):
        calendar_cell_reference(target, target.index)
