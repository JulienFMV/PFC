"""Composition preserves signed prices, parent hours and solver levels."""
import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.water_value import WaterValueCorrection
from pfc_shaping.lt.signed_intraday import intraday_cell_reference
from tests.test_signed_hourly_assembly import fixture


def context(kwargs):
    idx = kwargs["delivery_index"]
    hydro = pd.DataFrame({"fill_deviation": np.linspace(-.2, .3, len(idx))}, index=idx)
    residual = pd.Series(np.tile([-4., -2., 1., 5.], len(idx) // 4), index=idx)
    return hydro, residual


@pytest.mark.parametrize("month", ["2024-02", "2024-03", "2024-10"])
@pytest.mark.parametrize("base", [-50., 0., 50.])
def test_additive_composition_keeps_intrahour_orientation_and_market_levels(month, base):
    assembler, kwargs = fixture(month, base)
    kwargs["base_prices"][month + "-Peak"] = base + 20
    kwargs["quoted_keys"].add(month + "-Peak")
    hydro, residual = context(kwargs)
    baseline = assembler.build(**kwargs)
    composed = assembler.build(**kwargs, signed_intraday_shape=residual)
    np.testing.assert_allclose(composed.price_shape - baseline.price_shape, residual, atol=1e-9, rtol=0)
    np.testing.assert_allclose(composed.price_shape.resample("h").mean(), baseline.price_shape.resample("h").mean(), atol=1e-9, rtol=0)
    assembler.wv = WaterValueCorrection()
    watered = assembler.build(**kwargs, hydro_forecast=hydro)
    both = assembler.build(**kwargs, hydro_forecast=hydro, signed_intraday_shape=residual)
    np.testing.assert_allclose(both.price_shape - watered.price_shape, residual, atol=1e-9, rtol=0)
    np.testing.assert_allclose(both.delta_wv, watered.delta_wv, atol=0, rtol=0)
    assert abs(both.price_shape.mean() - base) < 1e-9
    assert abs(both.delta_wv.mean()) < 1e-9
    assert both.attrs["composition_contract"] == "signed-additive-composition.v1"
    if base == 0:
        assert both.delta_wv.eq(0).all()
    else:
        assert both.delta_wv.abs().max() > 0


@pytest.mark.parametrize("change", ["shift", "missing", "nan", "inf", "mean", "bool", "complex", "string", "no_hourly"])
def test_bad_quarter_shape_rejected(change):
    assembler, kwargs = fixture()
    _, residual = context(kwargs)
    if change == "shift": residual.index += pd.Timedelta(minutes=15)
    elif change == "missing": residual = residual.iloc[1:]
    elif change == "nan": residual.iloc[0] = np.nan
    elif change == "inf": residual.iloc[0] = np.inf
    elif change == "mean": residual.iloc[0] += 1
    elif change == "bool": residual = residual.gt(0)
    elif change == "complex": residual = residual.astype(complex)
    elif change == "string": residual = residual.astype(str)
    elif change == "no_hourly": kwargs.pop("signed_hourly_shape")
    with pytest.raises(ValueError, match="signed intraday"):
        assembler.build(**kwargs, signed_intraday_shape=residual)


@pytest.mark.parametrize("change", ["missing_model", "missing_forecast", "late_start", "early_end", "nan", "naive", "unsorted", "floor", "wrong_delta", "unaligned_delta"])
def test_invalid_hydro_composition_rejected(change):
    model = WaterValueCorrection()
    assembler, kwargs = fixture(water_value=model)
    hydro, _ = context(kwargs)
    if change == "missing_model": assembler.wv = None
    elif change == "missing_forecast": hydro = None
    elif change == "late_start": hydro = hydro.iloc[1:]
    elif change == "early_end": hydro = hydro.iloc[:-1]
    elif change == "nan": hydro.iloc[0, 0] = np.nan
    elif change == "naive": hydro = hydro.tz_localize(None)
    elif change == "unsorted": hydro = hydro.iloc[::-1]
    elif change == "floor": model.enforce_floor = True
    elif change == "wrong_delta": model.compute_delta_wv = lambda base, **kw: pd.Series(1., index=base.index)
    elif change == "unaligned_delta": model.compute_delta_wv = lambda base, **kw: base.iloc[1:]
    with pytest.raises(ValueError, match="signed hourly"):
        assembler.build(**kwargs, hydro_forecast=hydro)


def test_additive_reference_handles_negative_zero_prices_and_unseen_season():
    index = pd.date_range("2024-03-01", periods=24 * 4 * 30, freq="15min", tz="UTC")
    residual = np.tile([-4., -2., 1., 5.], len(index) // 4)
    levels = np.repeat(np.resize([-100., 0., 100.], len(index) // 4), 4)
    future = pd.date_range("2024-10-26T22:00Z", periods=25 * 4, freq="15min")
    prediction, counts = intraday_cell_reference(pd.Series(levels + residual, index=index), future)
    np.testing.assert_allclose(prediction, np.tile([-4., -2., 1., 5.], 25), atol=1e-12)
    assert prediction.groupby(future.floor("h")).mean().eq(0).all()
    assert sum(counts.values()) == len(future)


@pytest.mark.parametrize("bad", ["missing_quarter", "offset_grid", "nan", "future_training", "duplicate"])
def test_reference_rejects_incomplete_or_future_history(bad):
    idx = pd.date_range("2024-01-01", periods=48 * 4, freq="15min", tz="UTC")
    train = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    future = pd.date_range("2024-02-01", periods=4, freq="15min", tz="UTC")
    if bad == "missing_quarter": train = train.iloc[1:]
    elif bad == "offset_grid": train.index += pd.Timedelta(minutes=1)
    elif bad == "nan": train.iloc[0] = np.nan
    elif bad == "future_training": future = idx[:4]
    elif bad == "duplicate": train = pd.concat([train, train.iloc[-1:]])
    with pytest.raises(ValueError):
        intraday_cell_reference(train, future)
