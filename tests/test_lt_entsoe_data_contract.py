from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.data.lt_input_sources import LTInputPaths
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
from pfc_shaping.pipeline import production_phases


def test_shape_hourly_mlp_has_no_hidden_outage_file_read() -> None:
    source = inspect.getsource(ShapeHourlyMLP.fit)

    assert "read_parquet" not in source
    assert "_try_load_outages" not in source


def test_lt_orchestrator_injects_governed_outages_into_mlp() -> None:
    source = inspect.getsource(production_phases.run_long_term_phase)

    assert 'sh_fit_kwargs["outages_df"] = inputs.outages_all' in source


def test_lt_orchestrator_fails_closed_before_intraday_fit_without_product_identity() -> None:
    source = inspect.getsource(production_phases.run_long_term_phase)
    frame = pd.DataFrame(
        {"price_eur_mwh": [40.0, 41.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC"),
    )

    assert "_require_production_intraday_price_truth(inputs.epex_de)" in source
    with pytest.raises(ValueError, match="resolution provenance is missing"):
        production_phases._require_production_intraday_price_truth(frame)


def test_lt_horizon_is_anchored_to_explicit_valuation_not_runtime_clock() -> None:
    source = inspect.getsource(production_phases.run_long_term_phase)

    assert "Timestamp.utcnow" not in source
    assert "_next_delivery_start_date" in source


def test_lt_horizon_uses_next_swiss_calendar_day() -> None:
    assert production_phases._next_delivery_start_date(
        pd.Timestamp("2026-07-13T23:30:00Z"),
        timezone="Europe/Zurich",
    ) == "2026-07-15"


def test_lt_horizon_uses_next_swiss_calendar_day_across_dst_fallback() -> None:
    assert production_phases._next_delivery_start_date(
        pd.Timestamp("2026-10-25T12:00:00Z"),
        timezone="Europe/Zurich",
    ) == "2026-10-26"


def test_pit_role_failure_occurs_before_any_parquet_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    available_at = "2026-07-13T10:00:00+00:00"
    roles = {
        role: {
            "path": f"{role}.parquet",
            "calibration_eligible": role != "epex_ch",
            "available_at_utc": available_at,
        }
        for role in ("epex_ch", "epex_de", "entso", "hydro")
    }
    paths = LTInputPaths(
        root=tmp_path,
        snapshot_root=tmp_path / "snapshot",
        layout="external_v2",
        generation_id="generation-001",
        calibration_eligible=True,
        available_at_utc=available_at,
        files={role: tmp_path / f"{role}.parquet" for role in roles},
        expected_files=roles,
        schema_version="lt_input_snapshot.v3",
    )
    config = {
        "forwards": {"monthly_curve_solver": {"enabled": False}},
        "quality": {"freshness": {"outages_enabled": False}},
    }
    monkeypatch.setattr(
        production_phases,
        "read_yaml_snapshot",
        lambda _path: (config, None),
    )
    monkeypatch.setattr(
        production_phases,
        "resolve_lt_input_paths",
        lambda *_args, **_kwargs: paths,
    )

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("Parquet I/O occurred before PIT role validation")

    monkeypatch.setattr(production_phases, "read_parquet_snapshot", forbidden_read)

    with pytest.raises(ValueError, match="not calibration eligible: epex_ch"):
        production_phases.load_inputs(
            str(tmp_path),
            production_phases.logging.getLogger("test"),
            reference_timestamp="2026-07-13T11:00:00Z",
        )
