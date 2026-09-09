from __future__ import annotations

import pandas as pd

from scripts.audit_ch_hfc_vs_spot_shape import audit


def test_hfc_vs_spot_audit_compares_normalized_shapes(tmp_path):
    timestamps = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h", tz="Europe/Zurich")
    base = []
    for ts in timestamps:
        value = 80.0
        if ts.month in (12, 1, 2):
            value += 20.0
        if 17 <= ts.hour <= 21:
            value += 15.0
        if 11 <= ts.hour <= 15:
            value -= 10.0
        if ts.weekday() >= 5:
            value -= 5.0
        base.append(value)
    hfc = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + pd.Index(timestamps.strftime("%z")).str.slice(0, 3) + ":00",
            "price_weighted_mean_eur_mwh": base,
            "price_fast_eur_mwh": [x - 5.0 for x in base],
            "structural_p10_eur_mwh": [x - 5.0 for x in base],
        }
    )
    csv = tmp_path / "hfc.csv"
    hfc.to_csv(csv, index=False)

    spot_ts = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    spot_values = []
    for ts_utc in spot_ts:
        ts = ts_utc.tz_convert("Europe/Zurich")
        value = 70.0
        if ts.month in (12, 1, 2):
            value += 18.0
        if 17 <= ts.hour <= 21:
            value += 13.0
        if 11 <= ts.hour <= 15:
            value -= 9.0
        if ts.weekday() >= 5:
            value -= 4.0
        spot_values.append(value)
    spot = pd.DataFrame({"price_eur_mwh": spot_values}, index=spot_ts)
    spot_path = tmp_path / "spot.parquet"
    spot.to_parquet(spot_path)

    metrics, profile, negatives, _, summary = audit(csv, spot_path)

    assert not metrics.empty
    assert not profile.empty
    assert not negatives.empty
    assert summary["latest_hfc_peak_offpeak_spread_eur_mwh"] > 0.0
    assert summary["latest_hfc_evening_midday_spread_eur_mwh"] > 0.0
    assert summary["latest_hfc_shape_corr_vs_spot"] > 0.8
