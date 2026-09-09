"""Sparse intraday corrections must survive the Parquet round trip."""

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.shape_intraday import ShapeIntraday


def test_sparse_corrections_round_trip_preserves_factors(tmp_path):
    model = ShapeIntraday()
    keys = [("hiver", "ouvrable", 18), ("hiver", "ouvrable", 19)]
    for key in keys:
        model.base_factors_[key] = np.array([0.94, 0.98, 1.02, 1.06])
    model.corrections_[keys[0]] = {"b_solar_q1": 0.02, "intercept_q1": 0.01}
    model.corrections_[keys[1]] = {"b_load_q2": -0.03, "b_flow_q3": 0.04}
    timestamps = pd.date_range("2026-11-02T18:00:00Z", periods=8, freq="15min")
    calendar = pd.DataFrame(
        {"saison": "hiver", "type_jour": "ouvrable", "heure_hce": [18] * 4 + [19] * 4,
         "quart": [1, 2, 3, 4] * 2}, index=timestamps,
    )
    physical = pd.DataFrame(
        {"solar_regime": 1.5, "load_deviation": 0.2, "flow_deviation": -0.4},
        index=timestamps,
    )
    expected = model.apply(timestamps, calendar, physical, timestamps[0])
    path = tmp_path / "intraday.parquet"
    model.save(path)
    loaded = ShapeIntraday.load(path)
    actual = loaded.apply(timestamps, calendar, physical, timestamps[0])

    assert np.isfinite(actual).all()
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    assert loaded.corrections_ == model.corrections_
    for key in keys:
        np.testing.assert_array_equal(loaded.get(*key), model.get(*key))
