import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.signed_benchmark import calendar_cell_reference, closed_month_targets


def example():
    return pd.Series([-20., 40.], index=pd.DatetimeIndex(["2024-01-02T12:00Z", "2024-01-03T12:00Z"]))


def test_weighted_mean_and_unweighted_default():
    target = example()
    delivery = pd.date_range("2025-07-01T12:00Z", periods=24, freq="h")
    default, counts = calendar_cell_reference(target, delivery)
    np.testing.assert_array_equal(default, np.full(24, 10.))
    weights = pd.Series([1., 3.], index=target.index)
    weighted, new_counts = calendar_cell_reference(target, delivery, sample_weight=weights)
    np.testing.assert_allclose(weighted, 25.)
    assert counts == new_counts
    scaled, _ = calendar_cell_reference(target, delivery, sample_weight=weights * 1e100)
    np.testing.assert_allclose(weighted, scaled)
    shifted, _ = calendar_cell_reference(target + 100, delivery, sample_weight=weights)
    np.testing.assert_allclose(shifted, weighted + 100)


@pytest.mark.parametrize("bad", ["shift", "reorder", "missing", "nan", "inf", "zero", "negative", "boolean", "string"])
def test_bad_weights_rejected(bad):
    target = example()
    weights = pd.Series([1., 3.], index=target.index)
    if bad == "shift": weights.index += pd.Timedelta(hours=1)
    elif bad == "reorder": weights = weights.iloc[::-1]
    elif bad == "missing": weights = weights.iloc[:-1]
    elif bad == "nan": weights.iloc[0] = np.nan
    elif bad == "inf": weights.iloc[0] = np.inf
    elif bad == "zero": weights.iloc[0] = 0
    elif bad == "negative": weights.iloc[0] = -1
    elif bad == "boolean": weights = weights.gt(0)
    elif bad == "string": weights = weights.astype(str)
    with pytest.raises(ValueError, match="sample weights"):
        calendar_cell_reference(target, pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC"), sample_weight=weights)


def test_future_weighted_history_rejected():
    target = example()
    with pytest.raises(ValueError, match="precede delivery"):
        calendar_cell_reference(target, target.index, sample_weight=pd.Series(1., index=target.index))


def test_uniform_weight_parity_on_leap_and_dst_history():
    index = pd.date_range("2024-01-01", "2025-01-01", freq="h", inclusive="left", tz="Europe/Zurich").tz_convert("UTC")
    prices = pd.Series(40 + 60*np.cos(np.arange(len(index))/24), index=index)
    targets = closed_month_targets(prices, pd.Timestamp("2025-01-01T00:00Z"))
    assert len(targets) == 366*24
    delivery = pd.date_range("2025-03-01", "2025-04-01", freq="h", inclusive="left", tz="Europe/Zurich").tz_convert("UTC")
    a, ca = calendar_cell_reference(targets.signed_target, delivery)
    b, cb = calendar_cell_reference(targets.signed_target, delivery, sample_weight=pd.Series(1., index=targets.index))
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    assert ca == cb and len(a) == 31*24-1


def test_error_decomposition_and_ramp_boundaries():
    from scripts.run_lt_hourly_recency import diagnostic_frame
    index = pd.date_range("2024-02-01", "2024-04-01", freq="h", inclusive="left", tz="Europe/Zurich").tz_convert("UTC")
    truth = pd.Series(np.sin(np.arange(len(index))) * 40, index=index)
    prediction = truth * .5 + 20
    frame = diagnostic_frame(prediction, truth)
    np.testing.assert_allclose(frame.full_error, frame.shape_error + frame.level_error, atol=1e-12)
    assert frame.ramp_error.isna().sum() == 2
    np.testing.assert_allclose(np.mean(frame.full_error**2), np.mean(frame.shape_error**2)+np.mean(frame.level_error**2), atol=1e-10)
