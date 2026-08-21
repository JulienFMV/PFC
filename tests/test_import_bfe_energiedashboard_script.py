from __future__ import annotations

import pandas as pd

from scripts.import_bfe_energiedashboard import (
    FEEDS,
    build_bfe_energiedashboard_daily,
    main,
)


def _write_sources(path):
    samples = {
        "ogd104_stromproduktion_swissgrid.csv": "Datum,Energietraeger,Produktion_GWh\n2026-01-01,Flusskraft,22.6\n2026-01-01,Kernkraft,79.6\n",
        "ogd103_stromverbrauch_swissgrid_lv_und_endv.csv": "Datum,Landesverbrauch_GWh,Endverbrauch_GWh\n2026-01-01,170,150\n",
        "ogd107_strom_import_export.csv": "Date,AT_CH_GWh,DE_CH_GWh,FR_CH_GWh,IT_CH_GWh,CH_AT_GWh,CH_DE_GWh,CH_FR_GWh,CH_IT_GWh,Nettoimport\n2026-01-01,1,2,3,4,5,6,7,8,-16\n",
        "ogd106_preise_strom_boerse.csv": "Datum,Baseload_EUR_MWh\n2026-01-01,48\n",
        "ogd110_strom_verbrauch_prognose.csv": "Datum,Endverbrauch_Prognose_Min_GWh,Endverbrauch_Prognose_Mittel_GWh,Endverbrauch_Prognose_Max_GWh\n2026-01-02,147,156,165\n",
        "ogd103_stromverbrauch_geschaetzt_swissgrid.csv": "Datum,Landesverbrauch_geschaetzt_SG_GWh\n2026-01-01,183\n",
    }
    for feed in FEEDS:
        filename = feed.url.rsplit("/", 1)[-1]
        (path / filename).write_text(samples[filename], encoding="utf-8")


def test_build_bfe_energiedashboard_daily_normalizes_local_csvs(tmp_path):
    _write_sources(tmp_path)

    out = build_bfe_energiedashboard_daily(
        source_dir=tmp_path,
        ingested_at_utc="2026-06-12T00:00:00Z",
    )

    assert {"date_utc", "dataset_id", "track", "variable", "value", "unit"} <= set(out.columns)
    assert "production_hydro_ror_gwh" in set(out["variable"])
    assert "net_import_gwh" in set(out["variable"])
    assert "final_consumption_forecast_mean_gwh" in set(out["variable"])
    forecast = out[out["track"] == "forecast"].iloc[0]
    assert pd.Timestamp(forecast["as_of_utc"]) == pd.Timestamp("2026-06-12T00:00:00Z")


def test_import_bfe_energiedashboard_script_writes_output_and_report(tmp_path):
    _write_sources(tmp_path)
    output = tmp_path / "bfe.parquet"
    report = tmp_path / "report.md"

    rc = main(
        [
            "--raw-dir",
            str(tmp_path),
            "--no-download",
            "--output",
            str(output),
            "--report",
            str(report),
            "--ingested-at-utc",
            "2026-06-12T00:00:00Z",
        ]
    )

    assert rc == 0
    assert len(pd.read_parquet(output)) > 0
    assert "BFE Energiedashboard Import" in report.read_text(encoding="utf-8")
