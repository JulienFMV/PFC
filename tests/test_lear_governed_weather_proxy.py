from __future__ import annotations

import pandas as pd

from pfc_shaping.model.lear_forecaster import LEARForecaster


def test_weather_proxy_fills_missing_governed_solar_and_wind_forecasts() -> None:
    idx = pd.date_range("2026-05-01 00:00:00+00:00", periods=24 * 10, freq="h")
    exog = pd.DataFrame(index=idx)
    exog["wx_de_munich_shortwave_radiation"] = [0.0 if ts.hour < 6 or ts.hour > 18 else 100.0 for ts in idx]
    exog["wx_de_hamburg_wind_speed_10m"] = 10.0
    exog["forecast_solar_de_mw"] = pd.Series(
        [0.0 if ts.hour < 6 or ts.hour > 18 else 200.0 for ts in idx[:24 * 5]],
        index=idx[:24 * 5],
    )
    exog["forecast_wind_de_mw"] = pd.Series(50.0, index=idx[:24 * 5])

    model = LEARForecaster(use_governed_forecast_features=True)
    model._forecast_feature_pivots = {}

    model._fill_governed_forecast_gaps_from_weather(exog)

    future_midday = idx[-13]
    future_night = idx[-1]
    assert exog.loc[future_midday, "forecast_solar_de_mw"] == 200.0
    assert exog.loc[future_night, "forecast_solar_de_mw"] == 0.0
    assert exog.loc[future_midday, "forecast_wind_de_mw"] == 50.0
    assert "forecast_solar_de_mw" in model._forecast_feature_pivots
    assert "forecast_wind_de_mw" in model._forecast_feature_pivots
