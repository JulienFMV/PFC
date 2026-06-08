import numpy as np
import pandas as pd

from pfc_shaping.ct.model.foundation_forecaster import FoundationForecaster
from pfc_shaping.ct.model.lear_forecaster import LEARForecaster


class _FakeChronos2Pipeline:
    def __init__(self):
        self.df = None
        self.future_df = None
        self.kwargs = None

    def predict_df(self, df, future_df=None, **kwargs):
        self.df = df.copy()
        self.future_df = None if future_df is None else future_df.copy()
        self.kwargs = kwargs
        horizon = int(kwargs["prediction_length"])
        return pd.DataFrame({
            "0.1": np.full(horizon, 9.0),
            "0.5": np.full(horizon, 10.0),
            "0.9": np.full(horizon, 11.0),
        })


def _forecaster_with_fake_pipeline() -> tuple[FoundationForecaster, _FakeChronos2Pipeline]:
    pipe = _FakeChronos2Pipeline()
    fm = FoundationForecaster()
    fm._loaded = True
    fm._backend = "chronos2"
    fm._pipeline = pipe
    return fm, pipe


def test_chronos2_future_covariates_use_synthetic_future_grid():
    fm, pipe = _forecaster_with_fake_pipeline()
    horizon = 5
    n_context = 60
    prices = pd.Series(np.arange(n_context, dtype=float))
    future_covariates = {
        "cal_is_weekend": pd.Series(np.arange(n_context + horizon, dtype=float)),
    }

    result = fm.forecast(
        prices,
        horizon=horizon,
        future_covariates=future_covariates,
    )

    assert result is not None
    assert pipe.future_df is not None
    assert "target" not in pipe.future_df.columns
    assert pipe.future_df.columns.tolist() == ["timestamp", "item_id", "cal_is_weekend"]
    expected_ts = pd.date_range(
        start=pipe.df["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )
    pd.testing.assert_index_equal(pd.DatetimeIndex(pipe.future_df["timestamp"]), expected_ts, check_names=False)
    np.testing.assert_array_equal(pipe.future_df["cal_is_weekend"].to_numpy(), np.arange(n_context, n_context + horizon))
    np.testing.assert_array_equal(pipe.df["cal_is_weekend"].to_numpy(), np.arange(n_context))


def test_chronos2_skips_short_future_covariate_without_crashing():
    fm, pipe = _forecaster_with_fake_pipeline()
    prices = pd.Series(np.arange(60, dtype=float))

    result = fm.forecast(
        prices,
        horizon=5,
        future_covariates={"too_short": pd.Series([1.0, 2.0])},
    )

    assert result is not None
    assert pipe.future_df is None
    assert "too_short" not in pipe.df.columns


def test_lear_future_covariates_default_off_and_calendar_shape():
    context = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    future = pd.date_range(context[-1] + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC")

    off_model = LEARForecaster(use_foundation_model=False)
    assert off_model._build_future_covariates(context, future, cutoff_utc=context[-1]) is None

    on_model = LEARForecaster(
        use_foundation_model=False,
        use_future_covariates=True,
    )
    covariates = on_model._build_future_covariates(context, future, cutoff_utc=context[-1])

    assert covariates is not None
    assert "cal_hour_sin" in covariates
    assert "cal_is_weekend" in covariates
    assert "forecast_wind_de_mw" not in covariates
    assert all(len(series) == len(context) + len(future) for series in covariates.values())
    assert all(np.isfinite(series.to_numpy()).all() for series in covariates.values())


def test_lear_de_renewable_future_is_separately_gated():
    context = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    future = pd.date_range(context[-1] + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC")
    combined = context.append(future)

    calendar_only = LEARForecaster(
        use_foundation_model=False,
        use_future_covariates=True,
        use_de_renewable_future=False,
    )
    calendar_only._de_renewable_forecast_series_h_ = {
        "forecast_wind_de_mw": pd.Series(np.arange(len(combined), dtype=float), index=combined),
    }
    assert "forecast_wind_de_mw" not in calendar_only._build_future_covariates(context, future, cutoff_utc=context[-1])

    with_de = LEARForecaster(
        use_foundation_model=False,
        use_future_covariates=True,
        use_de_renewable_future=True,
    )
    with_de._de_renewable_forecast_series_h_ = {
        "forecast_wind_de_mw": pd.Series(np.arange(len(combined), dtype=float), index=combined),
    }
    covariates = with_de._build_future_covariates(context, future, cutoff_utc=context[-1])

    assert covariates is not None
    assert "forecast_wind_de_mw" in covariates
    np.testing.assert_array_equal(covariates["forecast_wind_de_mw"].iloc[-len(future):].to_numpy(), np.arange(60, 84))
