from __future__ import annotations

import pandas as pd

from scripts.audit_ch_hfc_seasonal_coherence import audit


def test_seasonal_audit_flags_annual_only_january_below_october(tmp_path):
    rows = []
    for month, price in [(1, 70.0), (10, 85.0)]:
        for hour in range(24):
            ts = pd.Timestamp(year=2030, month=month, day=1, hour=hour)
            rows.append(
                {
                    "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                    "utc_offset_ch": "UTC+01:00" if month == 1 else "UTC+02:00",
                    "timestamp_utc": ts.strftime("%d.%m.%Y %H:%M"),
                    "price_weighted_mean_eur_mwh": price,
                }
            )
    csv = tmp_path / "curve.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH"],
            "load_type": ["BASE"],
            "date": [pd.Timestamp("2026-06-12")],
            "product": ["2030"],
            "price": [77.5],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    seasonal = result["seasonal_checks"]
    assert seasonal.loc[seasonal["year"] == 2030, "severity"].iloc[0] == "critical"
    assert seasonal.loc[seasonal["year"] == 2030, "jan_minus_oct_eur_mwh"].iloc[0] == -15.0


def test_seasonal_audit_keeps_quoted_quarter_inversion_as_market_signal(tmp_path):
    rows = []
    for month, price in [(1, 70.0), (10, 85.0)]:
        for hour in range(24):
            ts = pd.Timestamp(year=2030, month=month, day=1, hour=hour)
            rows.append(
                {
                    "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                    "utc_offset_ch": "UTC+01:00" if month == 1 else "UTC+02:00",
                    "timestamp_utc": ts.strftime("%d.%m.%Y %H:%M"),
                    "price_weighted_mean_eur_mwh": price,
                }
            )
    csv = tmp_path / "curve.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH"] * 4,
            "load_type": ["BASE"] * 4,
            "date": [pd.Timestamp("2026-06-12")] * 4,
            "product": ["2030-Q1", "2030-Q2", "2030-Q3", "2030-Q4"],
            "price": [70.0, 60.0, 65.0, 85.0],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    seasonal = result["seasonal_checks"]
    assert seasonal.loc[seasonal["year"] == 2030, "forward_coverage"].iloc[0] == "full_quarter"
    assert seasonal.loc[seasonal["year"] == 2030, "severity"].iloc[0] == "ok"
