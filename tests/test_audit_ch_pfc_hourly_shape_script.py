from __future__ import annotations

import pandas as pd

from scripts.audit_ch_pfc_hourly_shape import audit


def test_audit_scores_well_shaped_hourly_csv(tmp_path):
    ts = pd.date_range("2030-01-01", periods=24 * 14, freq="h", tz="Europe/Zurich")
    rows = []
    for t in ts:
        price = 80.0
        if 11 <= t.hour <= 15:
            price -= 10.0
        if 17 <= t.hour <= 21:
            price += 12.0
        if t.weekday() >= 5:
            price -= 4.0
        width = 20.0 if 17 <= t.hour <= 21 else 6.0
        rows.append(
            {
                "timestamp_ch": t.strftime("%Y-%m-%d %H:%M:%S%z"),
                "timestamp_utc": t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
                "price_slow_eur_mwh": price - width / 2.0,
                "price_central_eur_mwh": price,
                "price_fast_eur_mwh": price + width / 2.0,
                "price_weighted_mean_eur_mwh": price,
                "structural_p10_eur_mwh": price - width / 2.0,
                "structural_p50_eur_mwh": price,
                "structural_p90_eur_mwh": price + width / 2.0,
                "structural_width_eur_mwh": width,
            }
        )
    csv = tmp_path / "pfc.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fw = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH"],
            "load_type": ["BASE"],
            "date": [pd.Timestamp("2026-06-11")],
            "product": ["2030"],
            "price": [pd.read_csv(csv)["price_weighted_mean_eur_mwh"].mean()],
        }
    ).to_parquet(fw, index=False)

    annual, residuals, metrics = audit(csv, fw)

    assert not annual.empty
    assert residuals.loc[0, "abs_error_eur_mwh"] == 0.0
    assert metrics["score_10"] >= 8.5


def test_audit_uses_quote_aware_residual_buckets(tmp_path):
    ts = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h", tz="Europe/Zurich")
    q1 = ts.quarter == 1
    price = pd.Series(69.800824, index=ts)
    price.loc[q1] = 110.76
    csv = tmp_path / "pfc.csv"
    pd.DataFrame(
        {
            "timestamp_ch": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": price - 5.0,
            "price_central_eur_mwh": price,
            "price_fast_eur_mwh": price + 5.0,
            "price_weighted_mean_eur_mwh": price,
            "structural_p10_eur_mwh": price - 5.0,
            "structural_p50_eur_mwh": price,
            "structural_p90_eur_mwh": price + 5.0,
            "structural_width_eur_mwh": [10.0] * len(ts),
        }
    ).to_csv(csv, index=False)
    cal_target = float(price.mean())
    fw = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH"],
            "load_type": ["BASE", "BASE"],
            "date": [pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-12")],
            "product": ["2030", "2030-Q1"],
            "price": [cal_target, 110.76],
        }
    ).to_parquet(fw, index=False)

    _, residuals, metrics = audit(csv, fw)

    assert set(residuals["product"]) == {"2030-Q1", "2030-RESIDUAL"}
    assert metrics["max_eex_error_eur_mwh"] < 1e-5
