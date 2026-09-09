from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.intraday_amplitude import (
    compress_intraday_peak_amplitude,
    compress_price_peak_amplitude,
)
from pfc_shaping.lt.model.shape_intraday import (
    MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR,
    ShapeIntraday,
)


def _hourly_fixture():
    local = pd.date_range(
        "2025-07-01",
        "2025-07-03",
        freq="1h",
        inclusive="left",
        tz="Europe/Zurich",
    )
    idx = local.tz_convert("UTC")
    cal = enrich_15min_index(idx, country="CH")
    is_peak = (
        (cal["type_jour"].astype(str) == "Ouvrable")
        & (cal["heure_hce"].astype(int) >= 8)
        & (cal["heure_hce"].astype(int) < 20)
    )
    f_h = pd.Series(np.where(is_peak.to_numpy(), 1.1, 0.9), index=idx, name="f_H")
    base = pd.Series(100.0, index=idx, name="B")
    return idx, cal, f_h, base, is_peak


def test_compresses_peak_offpeak_contrast_to_target():
    _, cal, f_h, base, is_peak = _hourly_fixture()
    out = compress_intraday_peak_amplitude(f_h, cal, base, {7: 5.0})

    spread = float(out[is_peak].mean() - out[~is_peak].mean())
    assert spread == pytest.approx(0.05, abs=1e-12)


def test_preserves_local_day_mean():
    idx, cal, f_h, base, _ = _hourly_fixture()
    out = compress_intraday_peak_amplitude(f_h, cal, base, {7: 5.0})
    local = idx.tz_convert("Europe/Zurich")
    day = pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in local])

    day_means = out.groupby(day).mean().to_numpy()
    assert np.max(np.abs(day_means - 1.0)) < 1e-12


def test_missing_spreads_is_identity():
    _, cal, f_h, base, _ = _hourly_fixture()
    out = compress_intraday_peak_amplitude(f_h, cal, base, None)
    pd.testing.assert_series_equal(out, f_h)


def test_no_compression_when_target_is_wider_than_current():
    _, cal, f_h, base, _ = _hourly_fixture()
    out = compress_intraday_peak_amplitude(f_h, cal, base, {7: 30.0})
    pd.testing.assert_series_equal(out, f_h)


def test_price_compression_preserves_month_mean_and_targets_spread():
    _, cal, _, _, is_peak = _hourly_fixture()
    price = pd.Series(np.where(is_peak.to_numpy(), 110.0, 90.0), index=cal.index, name="price_shape")

    out = compress_price_peak_amplitude(price, cal, {7: 5.0})

    assert float(out.mean()) == pytest.approx(float(price.mean()), abs=1e-12)
    spread = float(out[is_peak].mean() - out[~is_peak].mean())
    assert spread == pytest.approx(5.0, abs=1e-12)


def test_price_compression_missing_spreads_is_identity():
    _, cal, _, _, is_peak = _hourly_fixture()
    price = pd.Series(np.where(is_peak.to_numpy(), 110.0, 90.0), index=cal.index, name="price_shape")
    out = compress_price_peak_amplitude(price, cal, None)
    pd.testing.assert_series_equal(out, price)


def test_intraday_amplitude_flag_defaults_off():
    import inspect

    from pfc_shaping.lt.model.assembler import PFCAssembler

    default = inspect.signature(PFCAssembler).parameters["enable_intraday_amplitude_shrinkage"].default
    assert default is False


def test_horizon_flattening_preserves_exact_parent_hour_neutrality():
    index = pd.date_range("2026-09-01T00:00:00Z", periods=4, freq="15min")
    calendar = enrich_15min_index(index, country="DE")
    first = calendar.iloc[0]
    key = (first["saison"], first["type_jour"], int(first["heure_hce"]))
    model = ShapeIntraday(enable_sparse_support_regularization=True)
    model.base_factors_[key] = np.array([0.8, 0.95, 1.05, 1.2])

    factors = model.apply(
        index,
        calendar,
        reference_date=pd.Timestamp("2026-01-01T00:00:00Z"),
    )

    assert float(factors.mean()) == pytest.approx(1.0, abs=1e-15)


def test_sparse_base_fit_uses_stable_price_space_regression_near_zero():
    parent_hours = pd.date_range(
        "2026-04-01T00:00:00Z",
        periods=MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR,
        freq="D",
    )
    index = pd.DatetimeIndex(
        [hour + pd.Timedelta(minutes=15 * quarter) for hour in parent_hours for quarter in range(4)]
    )
    hourly_means = np.linspace(-20.0, 20.0, len(parent_hours))
    hourly_means[len(parent_hours) // 2] = 0.001
    expected = np.array([0.7, 0.9, 1.1, 1.3])
    prices = np.concatenate([mean * expected for mean in hourly_means])
    hour_data = pd.DataFrame(
        {
            "price_eur_mwh": prices,
            "quart": np.tile(np.arange(1, 5), len(parent_hours)),
            "_weight": np.ones(len(index)),
        },
        index=index,
    )

    fitted = ShapeIntraday(enable_sparse_support_regularization=True)._fit_base(hour_data)

    assert fitted is not None
    np.testing.assert_allclose(fitted, expected, rtol=0.0, atol=1e-12)
    assert float(fitted.mean()) == pytest.approx(1.0, abs=1e-15)


def test_sparse_profiles_are_contracted_to_data_derived_dense_support():
    model = ShapeIntraday(enable_sparse_support_regularization=True)
    dense_key = ("Hiver", "Ouvrable", 8)
    sparse_key = ("Printemps", "Samedi", 8)
    model.base_factors_ = {
        dense_key: np.array([0.8, 0.95, 1.05, 1.2]),
        sparse_key: np.array([0.1, 0.7, 1.2, 2.0]),
    }
    model.n_obs_ = {
        dense_key: (MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR + 1) * 4,
        sparse_key: MIN_PARENT_HOURS_FOR_RATIO_ESTIMATOR * 4,
    }

    model._constrain_sparse_cells_to_dense_support()

    assert model.sparse_support_envelope_ == pytest.approx(0.2)
    assert model.sparse_support_constrained_cells_ == 1
    assert np.max(np.abs(model.base_factors_[sparse_key] - 1.0)) == pytest.approx(0.2)
    assert float(model.base_factors_[sparse_key].mean()) == pytest.approx(1.0, abs=1e-15)


def test_sparse_support_regularization_is_experimental_and_disabled_by_default():
    assert ShapeIntraday().enable_sparse_support_regularization is False


def test_sparse_support_diagnostics_survive_save_load(tmp_path):
    model = ShapeIntraday(enable_sparse_support_regularization=True)
    key = ("Hiver", "Ouvrable", 8)
    model.base_factors_[key] = np.array([0.8, 0.95, 1.05, 1.2])
    model.n_obs_[key] = 100
    model.sparse_support_envelope_ = 0.2
    model.sparse_support_constrained_cells_ = 1
    path = tmp_path / "intraday.parquet"

    model.save(path)
    loaded = ShapeIntraday.load(path)

    assert loaded.enable_sparse_support_regularization is True
    assert loaded.sparse_support_envelope_ == pytest.approx(0.2)
    assert loaded.sparse_support_constrained_cells_ == 1
    np.testing.assert_array_equal(loaded.base_factors_[key], model.base_factors_[key])
