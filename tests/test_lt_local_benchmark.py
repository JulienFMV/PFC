from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pfc_shaping.lt.model.shape_hourly_mlp as native
from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.local_benchmark import (
    CHALLENGERS, complete_training_days, fit_challenger, parameter_grid,
    postprocess, prediction_matrix, score_curves, seasonal_reference, training_matrix,
)
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP


def grid(start, end):
    return pd.date_range(start, end, freq="15min", inclusive="left",
                         tz="Europe/Zurich").tz_convert("UTC")


def hydro():
    index = pd.date_range("2019-01-07", "2020-12-21", freq="W-MON", tz="Europe/Zurich")
    return pd.DataFrame({"fill_pct": 50 + 20 * np.sin(np.arange(len(index)))},
                        index=index.tz_convert("UTC"))


@pytest.mark.parametrize("start,end,expected", [("2020-03-29", "2020-03-30", 92),
                                               ("2020-10-25", "2020-10-26", 100)])
def test_complete_training_days_preserves_dst_and_removes_partial_days(start, end, expected):
    index = grid(start, end)
    prices = pd.DataFrame({"price_eur_mwh": 50.0}, index=index)
    origin = index[-1] + pd.Timedelta(minutes=15)
    assert len(complete_training_days(prices, origin)) == expected
    with pytest.raises(ValueError, match="no complete"):
        complete_training_days(prices.iloc[1:], origin)
    with pytest.raises(ValueError, match="no complete"):
        complete_training_days(prices, origin - pd.Timedelta(minutes=15))


def test_matrix_and_postprocess_match_corrected_native_including_autumn_hour(monkeypatch):
    index = grid("2020-10-24", "2020-10-27")
    prices = pd.DataFrame({"price_eur_mwh": 50 + index.hour + index.minute / 10}, index=index)
    source_hydro = hydro().loc[lambda f: f.index < index[0]]
    origin = pd.Timestamp("2020-12-31T12:00:00Z")
    captured = {}

    class Capture:
        def __init__(self, **kwargs):
            self.n_iter_, self.loss_ = 1, 0.0

        def fit(self, x, y):
            captured["x"], captured["y"] = x.copy(), y.copy()
            return self

        def predict(self, x):
            captured["prediction_x"] = x.copy()
            return .7 + .6 * x[:, 0] + .2 * x[:, 7]

    monkeypatch.setattr(native, "MLPRegressor", Capture)
    incumbent = HydroAlignedShapeHourlyMLP().fit(prices, enrich_15min_index(index), source_hydro)
    x, y, times, mapper = training_matrix(prices, source_hydro, origin)
    np.testing.assert_allclose(x, captured["x"][:, :9], rtol=0, atol=2e-15)
    np.testing.assert_allclose(y, captured["y"], rtol=0, atol=2e-15)
    assert len(times) == 72  # Native intentionally merges repeated autumn hour.
    future = grid("2021-03-28", "2021-03-30")
    expected = incumbent.apply(future, enrich_15min_index(future), reference_date=origin)
    xp = prediction_matrix(future, origin, mapper)
    np.testing.assert_allclose(xp, captured["prediction_x"][:, :9], rtol=0, atol=1e-15)
    np.testing.assert_allclose(postprocess(incumbent.mlp_.predict(captured["prediction_x"]), future),
                               expected.to_numpy(), rtol=0, atol=1e-15)


def test_future_hydro_or_prices_cannot_enter_training():
    index = grid("2020-12-20", "2020-12-22")
    prices = pd.DataFrame({"price_eur_mwh": 50.0}, index=index)
    origin = pd.Timestamp("2020-12-21T00:00Z")
    with pytest.raises(ValueError, match="precede origin"):
        training_matrix(prices, hydro(), origin)
    with pytest.raises(ValueError, match="precede origin"):
        training_matrix(prices.iloc[:4], hydro(), pd.Timestamp("2020-12-20T23:00Z"))


def test_grids_are_exact_and_real_execution_does_not_use_synthetic_containers(monkeypatch):
    import pfc_shaping.lt.evaluation_challengers as lab

    def forbidden(*args, **kwargs):
        raise AssertionError("real execution called synthetic API")

    monkeypatch.setattr(lab, "SyntheticTrainingSet", forbidden)
    monkeypatch.setattr(lab, "fit_synthetic_challenger", forbidden)
    assert [len(parameter_grid(c)) for c in CHALLENGERS] == [1, 5, 15, 16]
    x = np.random.default_rng(42).normal(size=(40, 9))
    y = .5 + x[:, 0] * .1
    times = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    model = fit_challenger("ridge-linear", x, y, times, pd.Timestamp("2020-02-01T00:00Z"), {"alpha": "1"})
    assert np.isfinite(model.predict(x)).all()
    with pytest.raises(ValueError, match="outside"):
        fit_challenger("ridge-linear", x, y, times, times[-1], {"alpha": "123"})
    with pytest.raises(ValueError, match="precede origin"):
        fit_challenger("ridge-linear", x, y, times, times[-1], {"alpha": "1"})


def test_seasonal_reference_preserves_constant_and_reports_fallbacks():
    train = grid("2020-01-06", "2020-01-07")
    future = grid("2021-07-04", "2021-07-05")
    values, counts = seasonal_reference(np.full(len(train), 1.3), train, future)
    np.testing.assert_allclose(values, 1.0)
    assert counts["heure_hce"] == len(future)
    assert sum(counts.values()) == len(future)


def test_scores_center_months_before_segments_and_count_native_hours():
    index = grid("2021-03-01", "2021-05-01")
    actual = pd.Series(50 + 20 * np.sin(np.arange(len(index)) // 4), index=index)
    error = np.where(index.tz_convert("Europe/Zurich").month == 3, 100.0, -30.0)
    predictions = pd.DataFrame({"a": actual + error, "b": actual}, index=index)
    scores, errors, coverage = score_curves(predictions, actual, pd.Timestamp("2020-12-31T12:00Z"))
    assert errors.shape == (1463, 2)
    assert np.abs(errors.to_numpy()).max() < 1e-12
    assert all(r["eligible"] for r in coverage)
    assert scores.query("segment == 'M25_M36'")["status"].eq("UNSUPPORTED_NO_ROWS").all()
    assert scores.query("segment == 'ALL'")["hours"].eq(1463).all()
    broken_truth = actual.drop(index[10])
    _, remaining, cov = score_curves(predictions, broken_truth, pd.Timestamp("2020-12-31T12:00Z"))
    assert len(remaining) == 720
    assert cov[0]["eligible"] is False
    predictions.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="mask"):
        score_curves(predictions, actual, pd.Timestamp("2020-12-31T12:00Z"))


def test_negative_truth_stays_in_scores():
    index = grid("2021-01-01", "2021-02-01")
    truth = pd.Series(np.where(index.hour < 12, -20.0, 40.0), index=index)
    scores, _, _ = score_curves(pd.DataFrame({"model": 20.0}, index=index), truth,
                                pd.Timestamp("2020-12-31T12:00Z"))
    negative = scores.query("segment == 'NEGATIVE_TRUTH'").iloc[0]
    assert negative.hours == 372
    assert negative.mae == pytest.approx(30)
