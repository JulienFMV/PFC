from __future__ import annotations

import zipfile

import pandas as pd

from scripts.import_bfe_structural_actuals import (
    build_installed_capacity_summary,
    build_production_plants,
    build_reservoir_weekly,
    build_structural_actuals,
    build_wasta_hydro,
    main,
)


def _write_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def _write_sources(path):
    (path / "ogd17_fuellungsgrad_speicherseen.csv").write_text(
        "\n".join(
            [
                "Datum,Wallis_speicherinhalt_gwh,Graubuenden_speicherinhalt_gwh,Tessin_speicherinhalt_gwh,UebrigCH_speicherinhalt_gwh,TotalCH_speicherinhalt_gwh,Wallis_max_speicherinhalt_gwh,Graubuenden_max_speicherinhalt_gwh,Tessin_max_speicherinhalt_gwh,UebrigCH_max_speicherinhalt_gwh,TotalCH_max_speicherinhalt_gwh",
                "2026-01-04,10,20,30,40,100,20,40,60,80,200",
            ]
        ),
        encoding="utf-8",
    )
    _write_zip(
        path / "ch.bfe.elektrizitaetsproduktionsanlagen.zip",
        {
            "ElectricityProductionPlant.csv": "\n".join(
                [
                    "xtf_id,Address,PostCode,Municipality,Canton,BeginningOfOperation,InitialPower,TotalPower,MainCategory,SubCategory,PlantCategory,_x,_y",
                    "1,,1000,Bern,BE,2020-01-01,1000,2000,maincat_2,subcat_2,plantcat_2,,",
                    "2,,1000,Bern,BE,2020-01-01,3000,4000,maincat_3,subcat_3,plantcat_3,,",
                ]
            ),
            "MainCategoryCatalogue.csv": "Catalogue_id,de,fr,it,en\nmaincat_2,Andere,,,Other renewable energies\nmaincat_3,Kernkraft,,,Nuclear energy\n",
            "SubCategoryCatalogue.csv": "Catalogue_id,de,fr,it,en\nsubcat_2,PV,,,Photovoltaic\nsubcat_3,Kernkraft,,,Nuclear energy\n",
            "PlantCategoryCatalogue.csv": "Catalogue_id,de,fr,it,en\nplantcat_2,Angebaut,,,Attached\nplantcat_3,Kernkraftwerk,,,Nuclear power plant\n",
            "OrientationCatalogue.csv": "Catalogue_id,de,fr,it,en\norient_1,Norden,,,North\n",
            "PlantDetail.csv": "xtf_id,Date,Power,Inclination,Orientation,PlantCategory,ElectricityProductionPlantR\n1,2020-01-01,1,5,orient_1,plantcat_2,1\n",
        },
    )
    _write_zip(
        path / "statistik-wasserkraftanlagen_2056.csv.zip",
        {
            "HydropowerPlant.csv": "WASTANumber,Name,Location,Canton,TypeCode,BeginningOfOperation,FallHeight,_x,_y\n10,Hydro A,Bern,BE,t1,1980,100,,\n",
            "TechnicalSpecification.csv": "xtf_id,Year,hydropowerPlantR,PerformanceGeneratorMaximum,PerformanceTurbineMaximum,ProductionExpected,PumpsPowerInputMaximum,EnginePowerDemand,OperationalStatusCode,DateOfStatistic,FallHeight\n100,2025,10,2000,2100,5000,300,0,os1,2025-12-31,100\n",
            "HydropowerPlantTypeCatalogue.csv": "OBJECTID,DE,FR,IT,ID\n1,Laufkraftwerk,,,t1\n",
            "HydropowerPlantOperationalStatusCatalogue.csv": "OBJECTID,DE,FR,IT,ID\n1,im Normalbetrieb,,,os1\n",
        },
    )


def test_build_bfe_structural_actuals_normalizes_sources(tmp_path):
    _write_sources(tmp_path)

    reservoir = build_reservoir_weekly(tmp_path, ingested_at_utc="2026-06-12T00:00:00Z")
    plants = build_production_plants(tmp_path, ingested_at_utc="2026-06-12T00:00:00Z")
    capacity = build_installed_capacity_summary(plants)
    _, wasta_summary = build_wasta_hydro(tmp_path, ingested_at_utc="2026-06-12T00:00:00Z")
    structural = build_structural_actuals(reservoir, capacity, wasta_summary)

    assert {"hydro_reservoir_content_gwh", "hydro_reservoir_fill_ratio"} <= set(reservoir["variable"])
    assert {"pv", "nuclear"} <= set(plants["technology"])
    assert {"pv_gw", "nuclear_gw"} <= set(capacity["variable"])
    assert "hydro_capacity_gw" in set(structural["variable"])
    assert "hydro_reservoir_twh" in set(structural["variable"])


def test_import_bfe_structural_actuals_script_writes_outputs(tmp_path):
    _write_sources(tmp_path)
    report = tmp_path / "report.md"
    output_args = [
        "--raw-dir",
        str(tmp_path),
        "--no-download",
        "--reservoir-output",
        str(tmp_path / "reservoir.parquet"),
        "--plants-output",
        str(tmp_path / "plants.parquet"),
        "--capacity-output",
        str(tmp_path / "capacity.parquet"),
        "--wasta-plants-output",
        str(tmp_path / "wasta_plants.parquet"),
        "--wasta-summary-output",
        str(tmp_path / "wasta_summary.parquet"),
        "--structural-output",
        str(tmp_path / "structural.parquet"),
        "--report",
        str(report),
        "--ingested-at-utc",
        "2026-06-12T00:00:00Z",
    ]

    rc = main(output_args)

    assert rc == 0
    assert len(pd.read_parquet(tmp_path / "structural.parquet")) > 0
    assert "BFE Structural Actuals Import" in report.read_text(encoding="utf-8")
