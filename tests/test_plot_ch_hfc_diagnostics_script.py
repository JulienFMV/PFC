from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_ch_hfc_diagnostics import build_plots


def _sample_hfc_csv(path: Path, *, baseline: bool = False) -> None:
    parts = []
    for year in (2027, 2028, 2030):
        for month in range(1, 13):
            parts.append(pd.date_range(f"{year}-{month:02d}-01", periods=24, freq="h", tz="Europe/Zurich"))
    ts = parts[0].append(parts[1:])
    seasonal = 25.0 * np.cos((ts.month.to_numpy() - 1.0) / 12.0 * 2.0 * np.pi)
    intraday = 8.0 * np.sin((ts.hour.to_numpy() - 7.0) / 24.0 * 2.0 * np.pi)
    price = 80.0 + seasonal + intraday
    if baseline:
        price = price - 1.5 * np.sin(ts.month.to_numpy() / 12.0 * 2.0 * np.pi)
    frame = pd.DataFrame(
        {
            "timestamp_ch": ts.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + ts.strftime("%z").str.slice(0, 3) + ":" + ts.strftime("%z").str.slice(3, 5),
            "timestamp_utc": ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": price - 5.0,
            "price_central_eur_mwh": price,
            "price_fast_eur_mwh": price + 5.0,
            "price_weighted_mean_eur_mwh": price,
            "structural_p10_eur_mwh": price - 5.0,
            "structural_p50_eur_mwh": price,
            "structural_p90_eur_mwh": price + 5.0,
            "structural_width_eur_mwh": 10.0,
        }
    )
    frame.to_csv(path, index=False)


def test_plot_ch_hfc_diagnostics_builds_governance_pack(tmp_path):
    csv = tmp_path / "hfc.csv"
    baseline = tmp_path / "hfc_no_smoothing.csv"
    _sample_hfc_csv(csv)
    _sample_hfc_csv(baseline, baseline=True)
    forwards = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH", "CH", "CH"],
            "load_type": ["BASE", "BASE", "BASE", "PEAK"],
            "date": [pd.Timestamp("2026-06-16")] * 4,
            "product": ["2027", "2028", "2030", "2030"],
            "price": [88.0, 80.0, 74.0, 84.0],
        }
    ).to_parquet(forwards, index=False)
    output = tmp_path / "diagnostics"

    paths = build_plots(csv, forwards, output, baseline_csv=baseline)

    names = {path.name for path in paths}
    assert "07_eex_residuals_by_product.png" in names
    assert "08_boundary_delta_jumps.png" in names
    assert "09_executive_qa_summary.png" in names
    assert (output / "monthly_diagnostics.csv").exists()
    residuals = pd.read_csv(output / "eex_residual_diagnostics.csv")
    assert set(residuals["load_type"]) == {"BASE", "PEAK"}
