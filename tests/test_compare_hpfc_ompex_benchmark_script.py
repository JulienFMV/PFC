from __future__ import annotations

import pandas as pd

from scripts.compare_hpfc_ompex_benchmark import compare


def test_compare_selects_hour_ending_alignment_and_marks_advisory(tmp_path) -> None:
    ts = pd.date_range("2026-07-01 00:00", periods=6, freq="h")
    hpfc = pd.DataFrame(
        {
            "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC+02:00",
            "price_weighted_mean_eur_mwh": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            "structural_p10_eur_mwh": [48.0, 49.0, 50.0, 51.0, 52.0, 53.0],
            "structural_p90_eur_mwh": [52.0, 53.0, 54.0, 55.0, 56.0, 57.0],
        }
    )
    hpfc_path = tmp_path / "hpfc.csv"
    hpfc.to_csv(hpfc_path, index=False)

    # OMPEX rows are deliberately timestamped as hour-ending.
    ompex = pd.DataFrame(
        {
            "Date": (ts + pd.Timedelta(hours=1)).strftime("%d.%m.%Y %H:%M"),
            "EUR/MWh": hpfc["price_weighted_mean_eur_mwh"],
        }
    )
    ompex_path = tmp_path / "HFC_Ompex_test.xlsx"
    ompex.to_excel(ompex_path, index=False)

    summary = compare(
        hpfc_csv=hpfc_path,
        ompex_xlsx=ompex_path,
        output_dir=tmp_path / "benchmark",
        write_plots=False,
    )

    assert summary["alignment"] == "ompex_minus_1h_hourending"
    assert summary["mae"] == 0.0
    assert summary["benchmark_policy"] == "advisory"
    assert summary["read_only"] is True
    assert summary["promotion_gate"] is False
    assert summary["production_approved"] is False
    assert summary["ompex_used_in_model"] is False
    assert summary["ompex_used_in_selection"] is False
    assert summary["ompex_used_in_backtest"] is False
    assert "imperfect benchmark" in str(summary["ompex_quality_caveat"])
    assert (tmp_path / "benchmark" / "alignment_sensitivity.csv").exists()
