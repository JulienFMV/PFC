from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_ch_hfc_seasonal_coherence import audit


def _offset_for_month(month: int) -> str:
    return "UTC+02:00" if 4 <= int(month) <= 10 else "UTC+01:00"


def _write_monthly_curve(tmp_path, monthly_values: dict[tuple[int, int], float]) -> Path:
    rows = []
    for (year, month), price in sorted(monthly_values.items()):
        for hour in range(24):
            ts = pd.Timestamp(year=year, month=month, day=1, hour=hour)
            rows.append(
                {
                    "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                    "utc_offset_ch": _offset_for_month(month),
                    "timestamp_utc": ts.strftime("%d.%m.%Y %H:%M"),
                    "price_weighted_mean_eur_mwh": price,
                }
            )
    csv = tmp_path / "curve.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


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


def test_seasonal_audit_flags_far_horizon_amplitude_collapse(tmp_path):
    rows = []
    prices = {
        (2028, 1): 120.0,
        (2028, 2): 115.0,
        (2028, 3): 100.0,
        (2028, 4): 78.0,
        (2028, 5): 64.0,
        (2028, 6): 62.0,
        (2028, 7): 64.0,
        (2028, 8): 67.0,
        (2028, 9): 70.0,
        (2028, 10): 75.0,
        (2028, 11): 79.0,
        (2028, 12): 82.0,
        **{(2029, month): 70.0 + (month % 3) for month in range(1, 13)},
    }
    for (year, month), price in prices.items():
        for hour in range(24):
            ts = pd.Timestamp(year=year, month=month, day=1, hour=hour)
            rows.append(
                {
                    "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                    "utc_offset_ch": _offset_for_month(month),
                    "timestamp_utc": ts.strftime("%d.%m.%Y %H:%M"),
                    "price_weighted_mean_eur_mwh": price,
                }
            )
    csv = tmp_path / "curve.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH", "CH"],
            "load_type": ["BASE", "BASE", "BASE"],
            "date": [pd.Timestamp("2026-06-12")] * 3,
            "product": ["2028", "2028-Q1", "2029"],
            "price": [80.0, 110.0, 72.0],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    seasonal = result["seasonal_checks"]
    row = seasonal.loc[seasonal["year"] == 2029].iloc[0]
    assert row["severity"] == "critical"
    assert "amplitude collapses" in row["reason"]
    assert row["annual_monthly_range_eur_mwh"] < 0.50 * row["previous_year_monthly_range_eur_mwh"]


def test_monthly_split_audit_flags_unquoted_ch_months_inverted_vs_neighbor(tmp_path):
    rows = []
    for month, price in [(1, 80.0), (2, 100.0), (3, 120.0)]:
        for day in [1, 2]:
            for hour in range(24):
                ts = pd.Timestamp(year=2030, month=month, day=day, hour=hour)
                rows.append(
                    {
                        "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                        "utc_offset_ch": "UTC+01:00",
                        "timestamp_utc": ts.strftime("%d.%m.%Y %H:%M"),
                        "price_weighted_mean_eur_mwh": price,
                    }
                )
    csv = tmp_path / "curve.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "DE", "DE", "DE", "DE"],
            "load_type": ["BASE"] * 5,
            "date": [pd.Timestamp("2026-06-12")] * 5,
            "product": ["2030-Q1", "2030-Q1", "2030-01", "2030-02", "2030-03"],
            "price": [100.0, 100.0, 120.0, 100.0, 80.0],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    splits = result["monthly_split_checks"]
    q1_base = splits[(splits["year"] == 2030) & (splits["quarter"] == 1) & (splits["load_type"] == "BASE")]
    assert q1_base["severity"].iloc[0] == "critical"
    assert q1_base["split_corr"].iloc[0] < 0.0


def test_calendar_audit_flags_weekend_premium(tmp_path):
    rows = []
    for day in range(1, 8):
        ts_day = pd.Timestamp(year=2030, month=1, day=day)
        price = 120.0 if ts_day.weekday() >= 5 else 90.0
        for hour in range(24):
            ts = ts_day + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                    "utc_offset_ch": "UTC+01:00",
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
            "price": [100.0],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    calendar = result["calendar_checks"]
    jan = calendar[(calendar["year"] == 2030) & (calendar["month"] == 1)]
    assert jan["severity"].iloc[0] == "critical"
    assert jan["weekend_minus_weekday_eur_mwh"].iloc[0] == 30.0


def test_monthly_path_audit_flags_jump_inside_annual_synthetic_bucket(tmp_path):
    rows = []
    for month, price in [(1, 100.0), (2, 102.0), (3, 70.0)]:
        for day in [1, 2]:
            for hour in range(24):
                ts = pd.Timestamp(year=2030, month=month, day=day, hour=hour)
                rows.append(
                    {
                        "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                        "utc_offset_ch": "UTC+01:00",
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
            "price": [90.0],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    path = result["monthly_path_checks"]
    assert "critical" in set(path["severity"])
    critical = path[path["severity"] == "critical"].iloc[0]
    assert critical["check_type"] == "adjacent_delta"
    assert critical["delta_eur_mwh"] == -32.0


def test_monthly_path_audit_flags_inter_bucket_seam_vs_neighbor(tmp_path):
    rows = []
    for month, price in [(1, 110.0), (2, 105.0), (3, 100.0), (4, 60.0)]:
        for day in [1, 2]:
            for hour in range(24):
                ts = pd.Timestamp(year=2030, month=month, day=day, hour=hour)
                rows.append(
                    {
                        "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                        "utc_offset_ch": "UTC+01:00" if month <= 3 else "UTC+02:00",
                        "timestamp_utc": ts.strftime("%d.%m.%Y %H:%M"),
                        "price_weighted_mean_eur_mwh": price,
                    }
                )
    csv = tmp_path / "curve.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH", "DE", "DE"],
            "load_type": ["BASE"] * 4,
            "date": [pd.Timestamp("2026-06-12")] * 4,
            "product": ["2030", "2030-Q1", "2030-03", "2030-04"],
            "price": [85.0, 105.0, 90.0, 80.0],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    path = result["monthly_path_checks"]
    seam = path[path["check_type"] == "inter_bucket_seam"].iloc[0]
    assert seam["severity"] == "critical"
    assert seam["from_bucket"] == "2030-Q1"
    assert seam["to_bucket"] == "2030-RESIDUAL"
    assert seam["delta_eur_mwh"] == -40.0
    assert seam["neighbor_delta_eur_mwh"] == -10.0


def test_monthly_path_audit_does_not_fail_directly_quoted_ch_seam_vs_neighbor(tmp_path):
    rows = []
    for month, price in [(1, 120.0), (2, 115.0), (3, 100.0), (4, 70.0), (5, 72.0), (6, 74.0)]:
        for day in [1, 2]:
            for hour in range(24):
                ts = pd.Timestamp(year=2030, month=month, day=day, hour=hour)
                rows.append(
                    {
                        "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
                        "utc_offset_ch": "UTC+01:00" if month <= 3 else "UTC+02:00",
                        "timestamp_utc": ts.strftime("%d.%m.%Y %H:%M"),
                        "price_weighted_mean_eur_mwh": price,
                    }
                )
    csv = tmp_path / "curve.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH", "DE", "DE"],
            "load_type": ["BASE"] * 4,
            "date": [pd.Timestamp("2026-06-12")] * 4,
            "product": ["2030-Q1", "2030-Q2", "2030-03", "2030-04"],
            "price": [111.666667, 72.0, 90.0, 85.0],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    path = result["monthly_path_checks"]
    seam = path[path["check_type"] == "inter_bucket_seam"].iloc[0]
    assert seam["from_bucket"] == "2030-Q1"
    assert seam["to_bucket"] == "2030-Q2"
    assert seam["delta_eur_mwh"] == -30.0
    assert seam["severity"] == "ok"
    assert seam["reason"] == "inter-bucket seam is constrained by CH quoted/implied forwards"


def test_cross_year_month_shape_flags_residual_vs_next_annual_incoherence(tmp_path):
    values = {}
    values.update({(2028, 1): 117.0, (2028, 2): 115.0, (2028, 3): 98.0})
    values.update(
        {
            (2028, 4): 62.0,
            (2028, 5): 61.0,
            (2028, 6): 62.0,
            (2028, 7): 66.0,
            (2028, 8): 65.0,
            (2028, 9): 72.0,
            (2028, 10): 78.0,
            (2028, 11): 85.0,
            (2028, 12): 85.0,
        }
    )
    values.update(
        {
            (2029, 1): 82.0,
            (2029, 2): 75.0,
            (2029, 3): 68.0,
            (2029, 4): 62.05,
            (2029, 5): 58.0,
            (2029, 6): 57.0,
            (2029, 7): 59.0,
            (2029, 8): 64.0,
            (2029, 9): 72.0,
            (2029, 10): 81.0,
            (2029, 11): 90.0,
            (2029, 12): 99.0,
        }
    )
    csv = _write_monthly_curve(tmp_path, values)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH", "CH"],
            "load_type": ["BASE", "BASE", "BASE"],
            "date": [pd.Timestamp("2026-06-17")] * 3,
            "product": ["2028", "2028-Q1", "2029"],
            "price": [80.40, 109.97, 72.41],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    checks = result["cross_year_month_shape_checks"]
    assert "critical" in set(checks["severity"])
    slope = checks[checks["check_type"] == "seasonal_slope_delta"].iloc[0]
    assert slope["severity"] == "critical"
    april = checks[(checks["check_type"] == "same_month_parent_spread") & (checks["month"] == 4)].iloc[0]
    assert april["severity"] == "warning"
    assert april["reason"] == "same-month values are near-cloned despite a non-zero parent bucket spread"


def test_cross_year_month_shape_accepts_parallel_parent_spread(tmp_path):
    values = {}
    values.update({(2028, 1): 117.0, (2028, 2): 115.0, (2028, 3): 98.0})
    residual_2028 = {
        4: 62.0,
        5: 61.0,
        6: 62.0,
        7: 66.0,
        8: 65.0,
        9: 72.0,
        10: 78.0,
        11: 85.0,
        12: 85.0,
    }
    values.update({(2028, month): value for month, value in residual_2028.items()})
    parent_spread = 2.0
    values.update({(2029, month): value + parent_spread for month, value in residual_2028.items()})
    values.update({(2029, 1): 119.0, (2029, 2): 117.0, (2029, 3): 100.0})
    csv = _write_monthly_curve(tmp_path, values)

    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH", "CH"],
            "load_type": ["BASE", "BASE", "BASE"],
            "date": [pd.Timestamp("2026-06-17")] * 3,
            "product": ["2028", "2028-Q1", "2029"],
            "price": [80.40, 109.97, 72.62],
        }
    ).to_parquet(forwards, index=False)

    result = audit(csv, forwards)

    checks = result["cross_year_month_shape_checks"]
    assert not checks.empty
    assert set(checks["severity"]) == {"ok"}
