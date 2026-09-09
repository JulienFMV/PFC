from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.data.ep2050 import derive_ep2050_annual_scenarios


def test_derive_ep2050_annual_scenarios_converts_gwh_to_twh():
    hourly = pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2022-10-01", tz="UTC"),
                "source": "unit",
                "scenario": "ZERO_Basis",
                "year": 2030,
                "technology": "total_consumption",
                "gwh": 700.0,
            },
            {
                "publication_date": pd.Timestamp("2022-10-01", tz="UTC"),
                "source": "unit",
                "scenario": "ZERO_Basis",
                "year": 2030,
                "technology": "total_consumption",
                "gwh": 900.0,
            },
            {
                "publication_date": pd.Timestamp("2022-10-01", tz="UTC"),
                "source": "unit",
                "scenario": "ZERO_Basis",
                "year": 2030,
                "technology": "pv",
                "gwh": 100.0,
            },
            {
                "publication_date": pd.Timestamp("2022-10-01", tz="UTC"),
                "source": "unit",
                "scenario": "ZERO_Basis",
                "year": 2030,
                "technology": "pv",
                "gwh": 200.0,
            },
            {
                "publication_date": pd.Timestamp("2022-10-01", tz="UTC"),
                "source": "unit",
                "scenario": "ZERO_Basis",
                "year": 2030,
                "technology": "imports",
                "gwh": 80.0,
            },
            {
                "publication_date": pd.Timestamp("2022-10-01", tz="UTC"),
                "source": "unit",
                "scenario": "ZERO_Basis",
                "year": 2030,
                "technology": "exports",
                "gwh": 30.0,
            },
        ]
    )

    out = derive_ep2050_annual_scenarios(hourly).iloc[0]

    assert float(out["demand_twh"]) == pytest.approx(1.6)
    assert float(out["peak_load_gw"]) == pytest.approx(900.0)
    assert float(out["pv_twh"]) == pytest.approx(0.3)
    assert float(out["pv_gw"]) == pytest.approx(200.0)
    assert float(out["import_twh"]) == pytest.approx(0.08)
    assert float(out["export_twh"]) == pytest.approx(0.03)
    assert float(out["net_import_twh"]) == pytest.approx(0.05)
