from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.build_ep2050_multi_scenario_pfc as builder
from scripts.build_ep2050_multi_scenario_pfc import (
    _fit_components,
    build_weighted_fan_chart,
    derive_slow_central_fast_inventory,
)


def _enriched_inventory() -> pd.DataFrame:
    rows = []
    for scenario, demand_twh, pv_twh, ev_twh in [
        ("WWB", 70.0, 6.0, 0.10),
        ("ZERO_Basis", 74.0, 14.0, 1.00),
    ]:
        rows.append(
            {
                "publication_date": pd.Timestamp("2026-06-05", tz="UTC"),
                "source": "OFEN_EP2050_hourly_11142+ch_first_pfc_proxy_v0",
                "scenario": scenario,
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": np.nan,
                "scenario_weight": 0.5,
                "quality_flag": "official_proxy_enriched",
                "demand_twh": demand_twh,
                "peak_load_gw": 15.0,
                "winter_demand_twh": 20.0,
                "pv_gw": pv_twh / 1.5,
                "pv_twh": pv_twh,
                "wind_gw": 0.2,
                "wind_twh": 0.4,
                "battery_power_gw": 2.0,
                "battery_energy_gwh": 5.0,
                "ev_twh": ev_twh,
                "heatpump_twh": 4.0,
                "managed_charging_share": 0.35,
                "hydro_twh": 40.0,
                "hydro_capacity_gw": 15.0,
                "hydro_reservoir_twh": 4.0,
                "nuclear_gw": 1.2,
                "dispatchable_gw": 0.0,
                "gas_gw": 0.0,
                "coal_gw": 0.0,
                "import_twh": 16.0,
                "export_twh": 8.0,
                "net_import_twh": 8.0,
                "ntc_ch_de_gw": 4.0,
                "ntc_ch_fr_gw": 4.0,
                "ntc_ch_it_gw": 3.0,
            }
        )
    return pd.DataFrame(rows)


def test_derive_slow_central_fast_inventory_is_bounded_and_governed():
    out = derive_slow_central_fast_inventory(
        _enriched_inventory(),
        weights={"slow": 0.25, "central": 0.50, "fast": 0.25},
        mapping_publication_date="2026-06-05",
    )

    assert list(out["scenario"]) == ["slow", "central", "fast"]
    assert dict(zip(out["scenario"], out["scenario_weight"])) == {
        "slow": 0.25,
        "central": 0.50,
        "fast": 0.25,
    }
    central = out[out["scenario"] == "central"].iloc[0]
    assert float(central["demand_twh"]) == pytest.approx(72.0)
    assert float(central["pv_twh"]) == pytest.approx(10.0)
    assert float(central["ev_twh"]) == pytest.approx(0.55)
    assert central["quality_flag"] == "internal_midpoint_proxy_enriched"
    assert str(central["original_scenario"]) == "midpoint(WWB,ZERO_Basis)"
    assert out["publication_date"].max() == pd.Timestamp("2026-06-05", tz="UTC")


def test_derive_slow_central_fast_inventory_fails_if_source_bound_missing():
    frame = _enriched_inventory()
    frame = frame[frame["scenario"] != "WWB"]

    with pytest.raises(KeyError, match="missing source scenarios"):
        derive_slow_central_fast_inventory(frame)


def test_build_weighted_fan_chart_writes_ordered_structural_columns():
    idx = pd.date_range("2030-01-01", periods=3, freq="1h", tz="UTC")
    frames = {
        "slow": pd.DataFrame({"price_shape": [60.0, 62.0, 64.0]}, index=idx),
        "central": pd.DataFrame({"price_shape": [65.0, 67.0, 69.0]}, index=idx),
        "fast": pd.DataFrame({"price_shape": [70.0, 72.0, 74.0]}, index=idx),
    }

    fan = build_weighted_fan_chart(
        frames,
        weights={"slow": 0.25, "central": 0.50, "fast": 0.25},
    )

    assert list(fan.columns) == [
        "curve_slow",
        "curve_central",
        "curve_fast",
        "weighted_mean",
        "structural_scenario_low",
        "structural_scenario_central",
        "structural_scenario_high",
        "structural_scenario_spread",
    ]
    assert float(fan["weighted_mean"].iloc[0]) == pytest.approx(65.0)
    assert np.all(fan["structural_scenario_low"] <= fan["structural_scenario_central"])
    assert np.all(fan["structural_scenario_central"] <= fan["structural_scenario_high"])
    assert float(fan["structural_scenario_spread"].mean()) > 0.0


def test_fit_components_uses_distinct_cutoff_intraday_source(monkeypatch):
    hourly_index = pd.date_range("2025-09-30T20:00:00Z", periods=4, freq="1h")
    intraday_index = pd.date_range("2025-09-30T23:30:00Z", periods=6, freq="15min")
    hourly = pd.DataFrame({"price_eur_mwh": np.arange(4.0)}, index=hourly_index)
    intraday = pd.DataFrame({"price_eur_mwh": np.arange(6.0)}, index=intraday_index)
    calls: dict[str, object] = {"calendars": []}

    def fake_calendar(index, *, country):
        calls["calendars"].append((index.copy(), country))
        return pd.DataFrame(index=index)

    class FakeHourly:
        def __init__(self, **_kwargs):
            pass

        def fit(self, frame, calendar):
            calls["hourly"] = (frame.copy(), calendar.copy())
            return self

    class FakeIntraday:
        def fit(self, frame, _entso, calendar):
            calls["intraday"] = (frame.copy(), calendar.copy())
            return self

    class FakeCascader:
        def __init__(self, **_kwargs):
            pass

        def fit_seasonal_ratios(self, _frame):
            return None

        def fit_peak_spreads(self, _frame):
            return None

    monkeypatch.setattr(builder, "enrich_15min_index", fake_calendar)
    monkeypatch.setattr(builder, "ShapeHourly", FakeHourly)
    monkeypatch.setattr(builder, "ShapeIntraday", FakeIntraday)
    monkeypatch.setattr(builder, "ContractCascader", FakeCascader)
    monkeypatch.setattr(builder, "ArbitrageFreeCalibrator", lambda **_kwargs: object())

    _fit_components(
        hourly,
        "CH",
        intraday_epex=intraday,
        intraday_market="DE",
        intraday_cutoff="2025-10-01T00:00:00Z",
    )

    fitted_intraday = calls["intraday"][0]
    assert fitted_intraday.index.min() == pd.Timestamp("2025-10-01T00:00:00Z")
    assert len(fitted_intraday) == 4
    assert [country for _, country in calls["calendars"]] == ["CH", "DE"]


def test_fit_components_rejects_empty_intraday_after_cutoff():
    index = pd.date_range("2025-09-30T20:00:00Z", periods=4, freq="15min")
    frame = pd.DataFrame({"price_eur_mwh": np.arange(4.0)}, index=index)

    with pytest.raises(ValueError, match="intraday EPEX calibration frame is empty"):
        _fit_components(
            frame,
            "CH",
            intraday_epex=frame,
            intraday_market="DE",
            intraday_cutoff="2025-10-01T00:00:00Z",
        )


def test_fit_components_rejects_non_quarter_hour_intraday_grid():
    hourly_index = pd.date_range("2025-10-01T00:00:00Z", periods=4, freq="1h")
    hourly = pd.DataFrame({"price_eur_mwh": np.arange(4.0)}, index=hourly_index)

    with pytest.raises(ValueError, match="contiguous 15-minute UTC"):
        _fit_components(
            hourly,
            "CH",
            intraday_epex=hourly,
            intraday_market="DE",
            intraday_cutoff="2025-10-01T00:00:00Z",
        )


def test_legacy_direct_publication_entrypoint_is_disabled():
    with pytest.raises(RuntimeError, match="direct multi-scenario publication is disabled"):
        builder.main([])
