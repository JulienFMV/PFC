import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from tests.test_signed_hourly_assembly import fixture


@pytest.mark.parametrize("month", ["2024-02", "2024-03", "2024-10", "2032-10"])
@pytest.mark.parametrize("price", [-40., 0., .01, 80.])
def test_signed_conditioning_conserves_hours_and_solver(month, price):
    assembly, kwargs = fixture(month, price)
    baseline = assembly.build(**kwargs)
    grid = baseline.index
    calendar = enrich_15min_index(grid)
    model = ShapeIntraday()
    for row in calendar.itertuples():
        model.base_factors_[(row.saison, row.type_jour, row.heure_hce)] = np.array([.7, .9, 1.1, 1.3])
    hourly = baseline.price_shape.resample("h").mean()
    delta = model.price_conditioned_residual(hourly, grid, calendar, reference_date=pd.Timestamp("2023-01-01", tz="UTC"))
    assert delta.groupby(grid.floor("h")).mean().abs().max() < 1e-12
    composed = assembly.build(**kwargs, signed_intraday_shape=delta)
    np.testing.assert_allclose(composed.price_shape.resample("h").mean(), hourly, atol=1e-9, rtol=0)
    np.testing.assert_allclose(composed.price_shape - baseline.price_shape, delta, atol=1e-9, rtol=0)
    assert abs(composed.price_shape.mean() - price) < 1e-9


@pytest.mark.parametrize("bad", ["missing", "duplicate", "offset", "naive", "nan", "boolean", "shift", "calendar"])
def test_reject_invalid_conditioning(bad):
    grid = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    prices = pd.Series([-10., 0.], index=grid[::4])
    calendar = enrich_15min_index(grid)
    if bad == "missing": grid = grid[:-1]
    elif bad == "duplicate": grid = grid.insert(0, grid[0])
    elif bad == "offset": grid += pd.Timedelta(minutes=1)
    elif bad == "naive": grid = grid.tz_localize(None)
    elif bad == "nan": prices.iloc[0] = np.nan
    elif bad == "boolean": prices = prices.gt(0)
    elif bad == "shift": prices.index += pd.Timedelta(hours=1)
    elif bad == "calendar": calendar = calendar.iloc[::-1]
    with pytest.raises(ValueError, match="price conditioning"):
        ShapeIntraday().price_conditioned_residual(prices, grid, calendar, reference_date=pd.Timestamp("2025-01-01", tz="UTC"))


def test_zero_and_negative_parent_orientation():
    grid = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    model = ShapeIntraday()
    model.apply = lambda timestamps, *args, **kwargs: pd.Series(np.tile([.7, .9, 1.1, 1.3], 3), index=timestamps)
    delta = model.price_conditioned_residual(pd.Series([-10., 0., 10.], index=grid[::4]), grid, enrich_15min_index(grid), reference_date=grid[0])
    np.testing.assert_allclose(delta, [3, 1, -1, -3, 0, 0, 0, 0, -3, -1, 1, 3], atol=1e-14)
