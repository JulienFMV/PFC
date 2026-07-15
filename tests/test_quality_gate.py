from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.pipeline.quality_gate import QualityGateError, validate_input_frame
from pfc_shaping.pipeline.production_phases import (
    _validate_production_input_freshness,
    _validate_strict_production_freshness_policy,
)


def _frame(latest: str) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp(latest), periods=4, freq="h")
    return pd.DataFrame({"value": 1.0}, index=index)


def _dense_recent_frame() -> pd.DataFrame:
    index = pd.date_range(
        "2026-07-03T00:00:00Z",
        "2026-07-10T00:00:00Z",
        freq="15min",
    )
    return pd.DataFrame({"value": 1.0}, index=index)


def _daily_recent_hydro() -> pd.DataFrame:
    index = pd.date_range("2026-07-03T00:00:00Z", "2026-07-10T00:00:00Z", freq="1D")
    return pd.DataFrame({"fill_deviation": 0.0}, index=index)


def test_stale_input_is_warning_by_default() -> None:
    report = validate_input_frame(
        _frame("2026-07-01T00:00:00Z"),
        name="spot",
        required_columns=["value"],
        min_rows=1,
        max_age_days=2,
        reference_timestamp="2026-07-10T00:00:00Z",
    )

    assert len(report.warnings) == 1
    assert "stale" in report.warnings[0]


def test_duplicate_timestamps_are_rejected() -> None:
    frame = _dense_recent_frame()
    duplicated = pd.concat([frame, frame.iloc[[-1]]]).sort_index()

    with pytest.raises(QualityGateError, match="duplicate timestamps"):
        validate_input_frame(
            duplicated,
            name="spot",
            required_columns=["value"],
            min_rows=1,
            max_age_days=2,
            reference_timestamp="2026-07-10T00:00:00Z",
            fail_on_stale=True,
        )


def test_stale_input_is_blocking_in_strict_mode() -> None:
    with pytest.raises(QualityGateError, match="stale"):
        validate_input_frame(
            _frame("2026-07-01T00:00:00Z"),
            name="spot",
            required_columns=["value"],
            min_rows=1,
            max_age_days=2,
            reference_timestamp="2026-07-10T00:00:00Z",
            fail_on_stale=True,
        )


def test_freshness_boundary_is_inclusive() -> None:
    report = validate_input_frame(
        _frame("2026-07-08T00:00:00Z"),
        name="spot",
        required_columns=["value"],
        min_rows=1,
        max_age_days=2,
        reference_timestamp="2026-07-10T00:00:00Z",
        fail_on_stale=True,
    )

    assert report.errors == []
    assert report.warnings == []


def test_future_observation_is_always_blocking() -> None:
    with pytest.raises(QualityGateError, match="after reference timestamp"):
        validate_input_frame(
            _frame("2026-07-11T00:00:00Z"),
            name="spot",
            required_columns=["value"],
            min_rows=1,
            max_age_days=2,
            reference_timestamp="2026-07-10T00:00:00Z",
        )


def test_fresh_index_does_not_hide_stale_critical_feature() -> None:
    index = pd.DatetimeIndex(
        ["2026-07-06T00:00:00Z", "2026-07-07T00:00:00Z", "2026-07-09T00:00:00Z", "2026-07-10T00:00:00Z"]
    )
    frame = pd.DataFrame({"value": [1.0, 1.0, float("nan"), float("nan")]}, index=index)

    with pytest.raises(QualityGateError, match="spot.value.*stale"):
        validate_input_frame(
            frame,
            name="spot",
            required_columns=["value"],
            min_rows=1,
            max_age_days=2,
            reference_timestamp="2026-07-10T00:00:00Z",
            fail_on_stale=True,
        )


def test_finite_coverage_threshold_is_blocking() -> None:
    frame = _frame("2026-07-10T00:00:00Z")
    frame.iloc[:2, frame.columns.get_loc("value")] = float("nan")

    with pytest.raises(QualityGateError, match="finite coverage"):
        validate_input_frame(
            frame,
            name="spot",
            required_columns=["value"],
            min_rows=1,
            max_age_days=2,
            reference_timestamp="2026-07-10T00:00:00Z",
            fail_on_stale=True,
            min_finite_fraction=0.75,
        )


def test_one_fresh_point_does_not_hide_a_recent_observation_gap() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-07-03T00:00:00Z",
            "2026-07-03T01:00:00Z",
            "2026-07-10T00:00:00Z",
        ]
    )
    frame = pd.DataFrame({"value": [1.0, 1.0, 1.0]}, index=index)

    with pytest.raises(QualityGateError, match="observation gap"):
        validate_input_frame(
            frame,
            name="spot",
            required_columns=["value"],
            min_rows=1,
            max_age_days=2,
            reference_timestamp="2026-07-10T00:00:00Z",
            fail_on_stale=True,
            recent_window_days=7,
            min_recent_finite_fraction=0.95,
            max_finite_gap_days=1 / 24,
        )


def test_production_freshness_blocks_one_stale_core_dataset() -> None:
    fresh = _dense_recent_frame()
    epex = fresh.rename(columns={"value": "price_eur_mwh"})
    entso = pd.DataFrame(
        {"load_mw": 1.0, "solar_mw": 1.0, "wind_mw": 1.0},
        index=fresh.index,
    )
    stale_hydro = pd.DataFrame(
        {"fill_deviation": 0.0},
        index=pd.date_range(end="2026-06-01T00:00:00Z", periods=4, freq="1D"),
    )
    config = {
        "quality": {
            "freshness": {
                "fail_on_stale_inputs": True,
                "epex_max_age_hours": 48,
                "entso_max_age_hours": 72,
                "hydro_max_age_days": 10,
                "outages_enabled": False,
            }
        }
    }

    with pytest.raises(RuntimeError, match="hydro.*stale"):
        _validate_production_input_freshness(
            epex_ch=epex,
            epex_de=epex,
            entso=entso,
            hydro=stale_hydro,
            outages=None,
            config=config,
            reference_timestamp="2026-07-10T00:00:00Z",
        )


def test_production_freshness_requires_enabled_outages() -> None:
    fresh = _dense_recent_frame()
    epex = fresh.rename(columns={"value": "price_eur_mwh"})
    entso = pd.DataFrame(
        {"load_mw": 1.0, "solar_mw": 1.0, "wind_mw": 1.0},
        index=fresh.index,
    )
    hydro = _daily_recent_hydro()
    config = {
        "quality": {
            "freshness": {
                "outages_enabled": True,
            }
        }
    }

    with pytest.raises(RuntimeError, match="dataset is missing"):
        _validate_production_input_freshness(
            epex_ch=epex,
            epex_de=epex,
            entso=entso,
            hydro=hydro,
            outages=None,
            config=config,
            reference_timestamp="2026-07-10T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("start", "frequency", "expected_error"),
    [
        ("2026-07-03T00:00:00Z", "1h", "grid_coverage"),
        ("2026-07-03T00:07:00Z", "15min", "grid_phase_mismatch"),
    ],
)
def test_production_freshness_rejects_invalid_neighbor_grid(
    start: str,
    frequency: str,
    expected_error: str,
) -> None:
    fresh = _dense_recent_frame()
    epex = fresh.rename(columns={"value": "price_eur_mwh"})
    entso = pd.DataFrame(
        {"load_mw": 1.0, "solar_mw": 1.0, "wind_mw": 1.0},
        index=fresh.index,
    )
    neighbor_index = pd.date_range(start, "2026-07-10T00:00:00Z", freq=frequency)
    neighbor = pd.DataFrame({"price_eur_mwh": 80.0}, index=neighbor_index)

    with pytest.raises(RuntimeError, match=expected_error):
        _validate_production_input_freshness(
            epex_ch=epex,
            epex_de=epex,
            entso=entso,
            hydro=_daily_recent_hydro(),
            outages=None,
            config={"quality": {"freshness": {"outages_enabled": False}}},
            reference_timestamp="2026-07-10T00:00:00Z",
            neighbor_prices_15min={"at": neighbor},
        )


def test_production_freshness_policy_cannot_be_disabled() -> None:
    fresh = _dense_recent_frame()
    epex = fresh.rename(columns={"value": "price_eur_mwh"})
    entso = pd.DataFrame(
        {"load_mw": 1.0, "solar_mw": 1.0, "wind_mw": 1.0},
        index=fresh.index,
    )
    hydro = pd.DataFrame({"fill_deviation": 0.0}, index=fresh.index)

    with pytest.raises(RuntimeError, match="cannot disable"):
        _validate_production_input_freshness(
            epex_ch=epex,
            epex_de=epex,
            entso=entso,
            hydro=hydro,
            outages=None,
            config={"quality": {"freshness": {"fail_on_stale_inputs": False}}},
            reference_timestamp="2026-07-10T00:00:00Z",
        )


def test_production_freshness_policy_cannot_relax_sanctioned_thresholds() -> None:
    fresh = _dense_recent_frame()
    epex = fresh.rename(columns={"value": "price_eur_mwh"})
    entso = pd.DataFrame(
        {"load_mw": 1.0, "solar_mw": 1.0, "wind_mw": 1.0},
        index=fresh.index,
    )
    hydro = pd.DataFrame({"fill_deviation": 0.0}, index=fresh.index)

    with pytest.raises(RuntimeError, match="exceeds sanctioned maximum"):
        _validate_production_input_freshness(
            epex_ch=epex,
            epex_de=epex,
            entso=entso,
            hydro=hydro,
            outages=None,
            config={"quality": {"freshness": {"epex_max_age_hours": 999}}},
            reference_timestamp="2026-07-10T00:00:00Z",
        )


@pytest.mark.parametrize(
    "key",
    [
        "min_finite_fraction",
        "min_recent_finite_fraction",
        "recent_window_days",
        "epex_max_age_hours",
        "entso_max_age_hours",
        "hydro_max_age_days",
        "epex_max_gap_hours",
        "entso_max_gap_hours",
        "hydro_max_gap_days",
    ],
)
def test_production_freshness_policy_rejects_non_finite_values(key: str) -> None:
    with pytest.raises(QualityGateError, match=f"{key} must be finite"):
        _validate_strict_production_freshness_policy(
            {key: float("nan")},
            QualityGateError,
        )
