from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.intraday_amplitude import (
    compress_intraday_peak_amplitude,
    compress_price_peak_amplitude,
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
