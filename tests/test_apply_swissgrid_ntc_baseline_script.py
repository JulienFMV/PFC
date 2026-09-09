from __future__ import annotations

import pandas as pd

from pfc_shaping.data.electrification_scenarios import RECOMMENDED_COLUMNS
from scripts.apply_swissgrid_ntc_baseline import (
    COMPONENT_ID,
    apply_ntc_baseline,
    load_swissgrid_ntc_baseline,
)


def _scenario_frame() -> pd.DataFrame:
    row = {column: pd.NA for column in RECOMMENDED_COLUMNS}
    row.update(
        {
            "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
            "measurement_date": pd.NaT,
            "track": "scenario",
            "source": "unit",
            "component_ids": "unit",
            "scenario_edition": "unit",
            "scenario": "central",
            "country": "CH",
            "delivery_year": 2030,
            "delivery_month": None,
            "scenario_weight": 1.0,
            "quality_flag": "official_partial_proxy",
            "ingested_at_utc": pd.Timestamp("2026-01-02", tz="UTC"),
            "demand_twh": 60.0,
            "ntc_ch_at_gw": pd.NA,
            "ntc_ch_de_gw": pd.NA,
            "ntc_ch_fr_gw": pd.NA,
            "ntc_ch_it_gw": pd.NA,
        }
    )
    return pd.DataFrame([row])


def test_load_swissgrid_ntc_baseline_parses_signed_directional_columns(tmp_path):
    csv_path = tmp_path / "swissgrid.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date Time;B:Intraday NTC Schweiz > Österreich [MW];C:Intraday NTC Österreich > Schweiz [MW];"
                "D:Intraday NTC Schweiz > Deutschland [MW];E:Intraday NTC Deutschland > Schweiz [MW];"
                "F:Intraday NTC Schweiz > Frankreich [MW];G:Intraday NTC Frankreich > Schweiz [MW];"
                "H:Intraday NTC Schweiz > Italien [MW];I:Intraday NTC Italien > Schweiz [MW]",
                "01.01.2026 00:00;1200;-900;4000;-800;1300;-3200;1516;-1910",
                "01.01.2026 00:15;1200;-1000;4200;-900;1400;-3300;1600;-1900",
            ]
        ),
        encoding="utf-8",
    )

    baseline = load_swissgrid_ntc_baseline(csv_path)

    assert set(baseline["ntc_field"]) == {
        "ntc_ch_at_gw",
        "ntc_ch_de_gw",
        "ntc_ch_fr_gw",
        "ntc_ch_it_gw",
    }
    values = baseline.set_index("ntc_field")["symmetric_baseline_gw"].to_dict()
    assert values["ntc_ch_at_gw"] == 0.95
    assert values["ntc_ch_de_gw"] == 0.85
    assert values["ntc_ch_fr_gw"] == 1.35
    assert values["ntc_ch_it_gw"] == 1.558


def test_apply_ntc_baseline_fills_only_missing_ntc_fields():
    baseline = pd.DataFrame(
        [
            {
                "ntc_field": "ntc_ch_de_gw",
                "border": "CH-DE",
                "symmetric_baseline_gw": 0.85,
                "method": "unit",
            }
        ]
    )
    frame = _scenario_frame()
    frame["ntc_ch_at_gw"] = 1.0

    out, changes = apply_ntc_baseline(frame, baseline)

    row = out.iloc[0]
    assert float(row["ntc_ch_de_gw"]) == 0.85
    assert float(row["ntc_ch_at_gw"]) == 1.0
    assert pd.isna(row["hydro_twh"])
    assert COMPONENT_ID in row["component_ids"]
    assert "swissgrid_ntc_baseline_proxy" in row["quality_flag"]
    assert changes["column"].tolist() == ["ntc_ch_de_gw"]
