from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.electrification_shape import (
    ElectrificationFHCorrection,
    ElectrificationScenarioStore,
    apply_electrification_scenarios,
    assert_production_scenario_inventory,
    electrification_modulate,
    structural_fan_chart,
)


VINTAGE = pd.Timestamp("2026-01-15", tz="UTC")


def _hourly_fixture(start: str = "2030-07-01", days: int = 2):
    local = pd.date_range(
        start,
        periods=24 * days,
        freq="1h",
        tz="Europe/Zurich",
    )
    idx = local.tz_convert("UTC")
    cal = enrich_15min_index(idx, country="CH")
    f_h = pd.Series(1.0, index=idx, name="f_H")
    return idx, cal, f_h


def _scenario_frame(
    *,
    pv_gw: float,
    battery_energy_gwh: float = 0.0,
    battery_power_gw: float = 0.0,
    ev_twh: float = 0.0,
    managed_charging_share: float = 0.0,
    wind_gw: float = 0.0,
    hydro_twh: float = 0.0,
    hydro_capacity_gw: float = 0.0,
    hydro_reservoir_twh: float = 0.0,
):
    return pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": np.nan,
                "demand_twh": 60.0,
                "pv_gw": pv_gw,
                "wind_gw": wind_gw,
                "battery_energy_gwh": battery_energy_gwh,
                "battery_power_gw": battery_power_gw,
                "ev_twh": ev_twh,
                "managed_charging_share": managed_charging_share,
                "heatpump_twh": 0.0,
                "hydro_twh": hydro_twh,
                "hydro_capacity_gw": hydro_capacity_gw,
                "hydro_reservoir_twh": hydro_reservoir_twh,
            }
        ]
    )


def _multi_scenario_frame():
    rows = []
    for year, pv_slow, pv_fast in [(2027, 10.0, 12.0), (2030, 12.0, 60.0)]:
        for scenario, pv, weight in [
            ("slow", pv_slow, 0.25),
            ("central", 0.5 * (pv_slow + pv_fast), 0.50),
            ("fast", pv_fast, 0.25),
        ]:
            rows.append(
                {
                    "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                    "source": "unit",
                    "scenario": scenario,
                    "country": "CH",
                    "delivery_year": year,
                    "delivery_month": np.nan,
                    "scenario_weight": weight,
                    "demand_twh": 60.0,
                    "pv_gw": pv,
                    "battery_energy_gwh": 0.0 if (year == 2027 or scenario == "slow") else 10.0,
                    "battery_power_gw": 0.0 if (year == 2027 or scenario == "slow") else 3.0,
                    "ev_twh": 0.0,
                    "heatpump_twh": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _production_frame():
    rows = []
    for country in ["CH", "DE", "FR", "IT", "AT"]:
        rows.append(
            {
                "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                "measurement_date": pd.NaT,
                "track": "scenario",
                "source": "unit",
                "scenario_edition": "unit-2026",
                "scenario": "central",
                "country": country,
                "delivery_year": 2030,
                "delivery_month": np.nan,
                "scenario_weight": 1.0,
                "quality_flag": "official",
                "ingested_at_utc": pd.Timestamp("2026-01-01", tz="UTC"),
                "demand_twh": 60.0,
                "peak_load_gw": 10.0,
                "winter_demand_twh": 18.0,
                "pv_gw": 20.0,
                "pv_twh": 10.0,
                "wind_gw": 5.0,
                "wind_twh": 8.0,
                "battery_power_gw": 3.0,
                "battery_energy_gwh": 10.0,
                "ev_twh": 2.0,
                "heatpump_twh": 3.0,
                "managed_charging_share": 0.4,
                "hydro_twh": 35.0,
                "hydro_capacity_gw": 12.0,
                "hydro_reservoir_twh": 4.0,
                "nuclear_gw": 1.0,
                "dispatchable_gw": 2.0,
                "gas_gw": 1.0,
                "coal_gw": 0.5,
                "import_twh": 8.0,
                "export_twh": 4.0,
                "net_import_twh": 4.0,
                "ntc_ch_de_gw": 4.0,
                "ntc_ch_fr_gw": 4.0,
                "ntc_ch_it_gw": 3.0,
                "ntc_ch_at_gw": 1.0,
                "gas_eur_mwh": 30.0,
                "coal_eur_mwh": 12.0,
                "co2_eur_t": 80.0,
                "electrolysis_twh": 1.0,
                "p2x_gw": 0.5,
                "dsm_gw": 0.3,
            }
        )
    return pd.DataFrame(rows)


def _block_means(series: pd.Series, cal: pd.DataFrame) -> dict[str, float]:
    hour = cal["heure_hce"].astype(int)
    return {
        "midday": float(series[(hour >= 10) & (hour <= 15)].mean()),
        "evening": float(series[(hour >= 18) & (hour <= 21)].mean()),
        "night": float(series[(hour <= 5) | (hour >= 22)].mean()),
    }


def test_scenario_store_asof_excludes_future_publications():
    frame = pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "pv_gw": 10.0,
            },
            {
                "publication_date": pd.Timestamp("2026-02-01", tz="UTC"),
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "pv_gw": 99.0,
            },
        ]
    )
    row = ElectrificationScenarioStore(frame=frame).asof(VINTAGE).get(
        country="CH",
        scenario="central",
        delivery_period=pd.Period("2030-07", "M"),
    )
    assert float(row["pv_gw"]) == 10.0


def test_scenario_store_raises_when_only_future_publication_exists():
    frame = pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2026-02-01", tz="UTC"),
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "pv_gw": 99.0,
            }
        ]
    )
    store = ElectrificationScenarioStore(frame=frame).asof(VINTAGE)
    with pytest.raises(KeyError):
        store.get(country="CH", scenario="central", delivery_period=pd.Period("2030-07", "M"))


def test_missing_required_columns_fail_fast():
    with pytest.raises(KeyError):
        ElectrificationScenarioStore(frame=pd.DataFrame([{"scenario": "central"}]))


def test_empty_store_is_identity():
    _, cal, f_h = _hourly_fixture()
    out = electrification_modulate(
        f_h,
        cal,
        shape_hourly=None,
        vintage=VINTAGE,
        scenario_store=ElectrificationScenarioStore(frame=pd.DataFrame(columns=[
            "publication_date",
            "scenario",
            "country",
            "delivery_year",
        ])),
    )
    pd.testing.assert_series_equal(out, f_h)


def test_empty_store_raises_when_scenario_data_required():
    _, cal, f_h = _hourly_fixture()
    with pytest.raises(FileNotFoundError, match="required"):
        electrification_modulate(
            f_h,
            cal,
            shape_hourly=None,
            vintage=VINTAGE,
            scenario_store=ElectrificationScenarioStore(frame=pd.DataFrame(columns=[
                "publication_date",
                "scenario",
                "country",
                "delivery_year",
            ])),
            require_scenario_data=True,
        )


def test_pv_deepens_midday_bowl_and_preserves_local_day_mean():
    idx, cal, f_h = _hourly_fixture()
    store = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=20.0))
    out = ElectrificationFHCorrection(scenario_store=store).apply(f_h, cal, vintage=VINTAGE)

    means = _block_means(out, cal)
    assert means["midday"] < means["night"]

    local = idx.tz_convert("Europe/Zurich")
    day = pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in local])
    day_means = out.groupby(day).mean().to_numpy()
    assert np.max(np.abs(day_means - 1.0)) < 1e-12


def test_pv_energy_share_deepens_midday_without_capacity_column():
    _, cal, f_h = _hourly_fixture()
    frame = pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "demand_twh": 60.0,
                "pv_twh": 12.0,
            }
        ]
    )
    store = ElectrificationScenarioStore(frame=frame)

    out = ElectrificationFHCorrection(scenario_store=store).apply(f_h, cal, vintage=VINTAGE)

    means = _block_means(out, cal)
    assert means["midday"] < means["night"]


def test_preserves_sign_of_negative_fh_values():
    _, cal, f_h = _hourly_fixture()
    f_h.iloc[12] = -0.5
    store = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=20.0))

    out = ElectrificationFHCorrection(scenario_store=store).apply(f_h, cal, vintage=VINTAGE)

    assert out.iloc[12] < 0.0


def test_preserves_input_day_mean_on_dst_short_day():
    local = pd.date_range(
        "2030-03-31",
        "2030-04-01",
        freq="1h",
        inclusive="left",
        tz="Europe/Zurich",
    )
    assert len(local) == 23
    idx = local.tz_convert("UTC")
    cal = enrich_15min_index(idx, country="CH")
    f_h = pd.Series(1.0048309, index=idx, name="f_H")
    store = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=20.0))

    out = ElectrificationFHCorrection(scenario_store=store).apply(f_h, cal, vintage=VINTAGE)

    assert float(out.mean()) == pytest.approx(float(f_h.mean()), abs=1e-12)


def test_battery_refills_midday_and_compresses_evening_vs_pv_only():
    _, cal, f_h = _hourly_fixture()
    pv_store = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=20.0))
    battery_store = ElectrificationScenarioStore(
        frame=_scenario_frame(pv_gw=20.0, battery_energy_gwh=20.0, battery_power_gw=5.0)
    )

    pv = ElectrificationFHCorrection(scenario_store=pv_store).apply(f_h, cal, vintage=VINTAGE)
    battery = ElectrificationFHCorrection(scenario_store=battery_store).apply(f_h, cal, vintage=VINTAGE)

    pv_means = _block_means(pv, cal)
    battery_means = _block_means(battery, cal)
    assert battery_means["midday"] > pv_means["midday"]
    assert battery_means["evening"] < pv_means["evening"]


def test_wind_lowers_winter_night_block_vs_no_wind():
    _, cal, f_h = _hourly_fixture(start="2030-01-02", days=2)
    no_wind = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=0.0, wind_gw=0.0))
    wind = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=0.0, wind_gw=25.0))

    base = ElectrificationFHCorrection(scenario_store=no_wind).apply(f_h, cal, vintage=VINTAGE)
    out = ElectrificationFHCorrection(scenario_store=wind).apply(f_h, cal, vintage=VINTAGE)

    assert _block_means(out, cal)["night"] < _block_means(base, cal)["night"]


def test_hydro_flexibility_compresses_winter_evening_ramp():
    _, cal, f_h = _hourly_fixture(start="2030-01-02", days=2)
    no_hydro = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=0.0))
    hydro = ElectrificationScenarioStore(
        frame=_scenario_frame(
            pv_gw=0.0,
            hydro_twh=35.0,
            hydro_capacity_gw=12.0,
            hydro_reservoir_twh=8.0,
        )
    )

    base = ElectrificationFHCorrection(scenario_store=no_hydro).apply(f_h, cal, vintage=VINTAGE)
    out = ElectrificationFHCorrection(scenario_store=hydro).apply(f_h, cal, vintage=VINTAGE)

    assert _block_means(out, cal)["evening"] < _block_means(base, cal)["evening"]


def test_managed_charging_reduces_ev_evening_uplift():
    _, cal, f_h = _hourly_fixture(start="2030-01-02", days=2)
    unmanaged = ElectrificationScenarioStore(frame=_scenario_frame(pv_gw=0.0, ev_twh=10.0))
    managed = ElectrificationScenarioStore(
        frame=_scenario_frame(pv_gw=0.0, ev_twh=10.0, managed_charging_share=1.0)
    )

    unmanaged_out = ElectrificationFHCorrection(scenario_store=unmanaged).apply(
        f_h, cal, vintage=VINTAGE
    )
    managed_out = ElectrificationFHCorrection(scenario_store=managed).apply(
        f_h, cal, vintage=VINTAGE
    )

    assert _block_means(managed_out, cal)["evening"] < _block_means(unmanaged_out, cal)["evening"]


def test_structural_fan_chart_weighted_quantiles_are_ordered():
    idx = pd.date_range("2030-01-01", periods=3, freq="1h", tz="UTC")
    curves = {
        "slow": pd.Series([1.0, 2.0, 3.0], index=idx),
        "central": pd.Series([2.0, 3.0, 4.0], index=idx),
        "fast": pd.Series([4.0, 5.0, 6.0], index=idx),
    }

    fan = structural_fan_chart(curves, weights={"slow": 0.25, "central": 0.5, "fast": 0.25})

    assert list(fan.columns) == ["weighted_mean", "q10", "q50", "q90", "structural_width"]
    assert np.all(fan["q10"] <= fan["q50"])
    assert np.all(fan["q50"] <= fan["q90"])
    assert float(fan["weighted_mean"].iloc[0]) == pytest.approx(2.25)


def test_scenario_divergence_widens_from_2027_to_2030():
    idx_2027, cal_2027, f_2027 = _hourly_fixture(start="2027-07-01", days=1)
    idx_2030, cal_2030, f_2030 = _hourly_fixture(start="2030-07-01", days=1)
    store = ElectrificationScenarioStore(frame=_multi_scenario_frame())
    scenarios = ["slow", "central", "fast"]
    weights = {"slow": 0.25, "central": 0.50, "fast": 0.25}

    curves_2027 = apply_electrification_scenarios(
        f_2027,
        cal_2027,
        shape_hourly=None,
        vintage=VINTAGE,
        scenarios=scenarios,
        scenario_store=store,
    )
    curves_2030 = apply_electrification_scenarios(
        f_2030,
        cal_2030,
        shape_hourly=None,
        vintage=VINTAGE,
        scenarios=scenarios,
        scenario_store=store,
    )
    fan_2027 = structural_fan_chart(curves_2027, weights=weights)
    fan_2030 = structural_fan_chart(curves_2030, weights=weights)

    h_2027 = cal_2027["heure_hce"].astype(int)
    h_2030 = cal_2030["heure_hce"].astype(int)
    midday_2027 = (h_2027 >= 10) & (h_2027 <= 15)
    midday_2030 = (h_2030 >= 10) & (h_2030 <= 15)
    assert float(fan_2030.loc[midday_2030.to_numpy(), "structural_width"].mean()) > float(
        fan_2027.loc[midday_2027.to_numpy(), "structural_width"].mean()
    )
    assert idx_2027.tz is not None
    assert idx_2030.tz is not None


def test_production_scenario_inventory_accepts_governed_multi_country_rows():
    assert_production_scenario_inventory(_production_frame())
    assert_production_scenario_inventory(
        _production_frame(),
        scenarios=["central"],
        years=[2030],
    )


def test_production_scenario_inventory_rejects_missing_country_scenario_year():
    frame = _production_frame()
    frame.loc[frame["country"] == "DE", "delivery_year"] = 2029

    with pytest.raises(ValueError, match="country/scenario/year coverage"):
        assert_production_scenario_inventory(frame, scenarios=["central"], years=[2030])


def test_production_scenario_inventory_rejects_missing_critical_values():
    frame = _production_frame()
    frame.loc[0, "demand_twh"] = pd.NA

    with pytest.raises(ValueError, match="missing critical values"):
        assert_production_scenario_inventory(frame, scenarios=["central"], years=[2030])


def test_production_scenario_inventory_rejects_proxy_quality_flags():
    frame = _production_frame()
    frame.loc[0, "quality_flag"] = "official_proxy_enriched"

    with pytest.raises(ValueError, match="non-production quality"):
        assert_production_scenario_inventory(frame)


def test_electrification_modulate_rejects_proxy_when_production_required():
    _, cal, f_h = _hourly_fixture()
    frame = _scenario_frame(pv_gw=20.0)
    frame["quality_flag"] = "official_proxy_enriched"

    with pytest.raises(ValueError, match="production electrification scenarios"):
        electrification_modulate(
            f_h,
            cal,
            shape_hourly=None,
            vintage=VINTAGE,
            scenario_store=ElectrificationScenarioStore(frame=frame),
            require_production_scenario_data=True,
        )


def test_electrification_shape_flag_defaults_off():
    from pfc_shaping.lt.model.assembler import PFCAssembler

    params = inspect.signature(PFCAssembler).parameters
    assert params["enable_electrification_shape"].default is False
    assert params["electrification_scenario"].default is None
    assert params["electrification_scenario_path"].default is None
    assert params["require_electrification_scenarios"].default is False
    assert params["require_production_electrification_scenarios"].default is False
