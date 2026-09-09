"""Signed hourly prices use the existing assembler and hard product projection."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape


class UnusedHourly(ShapeHourlyMLP):
    def apply(self, timestamps, calendar_df, reference_date=None, outages_forecast=None):
        raise AssertionError("signed input must replace native hourly and weekly shaping")


def fixture(month="2024-03", base=0.0, **overrides):
    first = pd.Period(month, freq="M")
    start = first.start_time.tz_localize("Europe/Zurich")
    end = (first + 1).start_time.tz_localize("Europe/Zurich")
    hourly = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    quarters = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
    local = hourly.tz_convert("Europe/Zurich")
    shape = center_signed_hourly_shape(pd.Series(
        100 * np.cos(local.hour.to_numpy() * np.pi / 12) + 30 * (local.dayofweek >= 5),
        index=hourly,
    ))
    model = UnusedHourly()
    model.f_W_ = {"Ouvrable": 999.0, "Samedi": -100.0}
    settings = dict(shape_hourly=model, shape_intraday=ShapeIntraday(),
        monthly_level_authority="solver", skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True, allow_negative_prices=True)
    settings.update(overrides)
    template = PFCAssembler(**settings)
    kwargs = dict(base_prices={month: base}, quoted_keys={month},
        delivery_index=quarters, reference_date=start - pd.Timedelta(days=1),
        signed_hourly_shape=shape)
    return template, kwargs


@pytest.mark.parametrize("month,expected_hours", [("2024-02", 696), ("2024-03", 743), ("2024-10", 745)])
@pytest.mark.parametrize("base", [-50., 0., 50.])
def test_signed_prices_conserve_levels_and_native_hours(month, expected_hours, base):
    template, kwargs = fixture(month, base)
    original = kwargs["signed_hourly_shape"].copy(deep=True)
    frame = template.build(**kwargs)
    pd.testing.assert_series_equal(original, kwargs["signed_hourly_shape"])
    assert len(frame) == 4 * expected_hours
    assert abs(frame.price_shape.mean() - base) < 1e-9
    np.testing.assert_allclose(frame.price_shape.resample("h").mean(), original + base, atol=1e-9, rtol=0)
    assert frame.price_shape.min() < 0 < frame.price_shape.max()
    assert frame[["f_W", "f_H", "f_Q", "f_bridge"]].eq(1).all().all()
    assert frame[["p10", "p90"]].isna().all().all()
    assert frame.attrs["hourly_shape_contract"] == "signed-hourly-eur-mwh.v1"


def test_supported_peak_and_base_reprice_after_signed_prior():
    template, kwargs = fixture("2024-10", 0.)
    kwargs["base_prices"]["2024-10-Peak"] = 20.
    kwargs["quoted_keys"].add("2024-10-Peak")
    frame = template.build(**kwargs)
    local = frame.index.tz_convert("Europe/Zurich")
    peak = (local.dayofweek < 5) & (local.hour >= 8) & (local.hour < 20)
    assert abs(frame.price_shape.mean()) < 1e-9
    assert abs(frame.price_shape.loc[peak].mean() - 20) < 1e-9
    assert frame.delta_final_product_projection.abs().max() > 1
    np.testing.assert_allclose(frame.price_shape, frame.price_pre_final_projection + frame.delta_final_product_projection)
    assert template.final_product_projection_report_["max_abs_error_eur_mwh"] < 1e-9


@pytest.mark.parametrize("change", ["nonzero_mean", "missing_hour", "duplicate", "unsorted", "naive", "nan", "infinity", "bool", "complex", "string", "wrong_month", "quarter_grid"])
def test_invalid_signed_shapes_fail(change):
    template, kwargs = fixture()
    shape = kwargs["signed_hourly_shape"]
    if change == "nonzero_mean": shape = shape + 1
    elif change == "missing_hour": shape = shape.drop(shape.index[50])
    elif change == "duplicate": shape = pd.concat([shape, shape.iloc[-1:]])
    elif change == "unsorted": shape = shape.iloc[::-1]
    elif change == "naive": shape = shape.tz_localize(None)
    elif change == "nan": shape.iloc[30] = np.nan
    elif change == "infinity": shape.iloc[30] = np.inf
    elif change == "bool": shape = shape.gt(0)
    elif change == "complex": shape = shape.astype(complex) + 1j
    elif change == "string": shape = shape.astype(str)
    elif change == "wrong_month": kwargs["delivery_index"] = kwargs["delivery_index"] + pd.Timedelta(days=1)
    elif change == "quarter_grid": shape = pd.Series(np.repeat(shape.to_numpy(), 4), index=kwargs["delivery_index"])
    kwargs["signed_hourly_shape"] = shape
    with pytest.raises((ValueError, TypeError)):
        template.build(**kwargs)


@pytest.mark.parametrize("overrides", [
    {"monthly_level_authority": "legacy"}, {"skip_legacy_level_cascade": False},
    {"skip_legacy_base_smoothing": False}, {"enforce_positivity": True},
    {"enforce_m_factor_floor": True}, {"enforce_floor": True},
    {"enable_solar_modulation": True}, {"enable_electrification_shape": True},
    {"enable_intraday_amplitude_shrinkage": True},
    {"water_value": SimpleNamespace(enforce_floor=False)}, {"uncertainty": object()},
])
def test_incompatible_components_are_explicitly_rejected(overrides):
    template, kwargs = fixture(**overrides)
    with pytest.raises(ValueError, match="signed hourly shape"):
        template.build(**kwargs)


@pytest.mark.parametrize("field", ["entso_forecast", "hydro_forecast", "outages_forecast"])
def test_no_context_silently_ignored(field):
    template, kwargs = fixture()
    kwargs[field] = pd.DataFrame()
    with pytest.raises(ValueError, match="signed hourly shape"):
        template.build(**kwargs)


def test_non_neutral_intraday_rejected():
    class Intraday:
        def apply(self, idx, *args, **kwargs):
            return pd.Series(1 + .1 * (idx.minute.to_numpy() // 15 - 1.5), index=idx)
    template, kwargs = fixture(shape_intraday=Intraday())
    with pytest.raises(ValueError, match="neutral intraday"):
        template.build(**kwargs)


def test_missing_solver_month_and_foreign_calendar_rejected():
    template, kwargs = fixture()
    kwargs["base_prices"] = {"2024": 0.}
    with pytest.raises(ValueError, match="explicit solver BASE"):
        template.build(**kwargs)
    kwargs["country"] = "DE"
    with pytest.raises(ValueError, match="CH monthly solver"):
        template.build(**kwargs)


def test_multiple_months_keep_separate_solver_levels_without_hidden_damping():
    template, first = fixture("2024-02", -20.)
    _, second = fixture("2024-03", 30.)
    first["base_prices"].update(second["base_prices"])
    first["quoted_keys"].update(second["quoted_keys"])
    first["delivery_index"] = first["delivery_index"].append(second["delivery_index"])
    first["signed_hourly_shape"] = pd.concat([first["signed_hourly_shape"], second["signed_hourly_shape"]])
    frame = template.build(**first)
    keys = frame.index.tz_convert("Europe/Zurich").strftime("%Y-%m")
    np.testing.assert_allclose(frame.price_shape.groupby(keys).mean(), [-20., 30.], atol=1e-9, rtol=0)
    first["reference_date"] -= pd.DateOffset(years=5)
    distant = template.build(**first)
    pd.testing.assert_series_equal(frame.price_shape, distant.price_shape)
