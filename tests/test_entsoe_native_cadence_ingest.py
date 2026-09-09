from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data import ingest_entso


def test_hourly_series_remains_hourly_and_is_normalized_to_utc() -> None:
    index = pd.date_range(
        "2026-01-01T01:00:00",
        periods=3,
        freq="h",
        tz="Europe/Zurich",
    )
    source = pd.Series([10.0, 20.0, 30.0], index=index)

    observed = ingest_entso._normalize_native_series(source, "load_mw")

    assert observed.name == "load_mw"
    assert len(observed) == 3
    assert observed.index.tz is not None
    assert str(observed.index.tz) == "UTC"
    assert observed.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_generation_nulls_and_absent_technologies_are_not_zero_filled() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    source = pd.DataFrame(
        {
            ("Solar", "Actual Aggregated"): [10.0, np.nan, 30.0],
        },
        index=index,
    )

    solar = ingest_entso._extract_generation_column(source, "Solar")
    wind = ingest_entso._extract_generation_column(source, "Wind Onshore")

    assert solar.iloc[0] == 10.0
    assert pd.isna(solar.iloc[1])
    assert wind.index.equals(index)
    assert wind.isna().all()


def test_load_selection_rejects_ambiguous_fallback_columns() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    ambiguous = pd.DataFrame(
        {"Forecasted Load": [1.0, 2.0], "Other": [3.0, 4.0]},
        index=index,
    )

    with pytest.raises(ValueError, match="no unambiguous Actual Load"):
        ingest_entso._extract_actual_load(ambiguous, "load_mw")


def test_duplicate_source_timestamps_fail_closed() -> None:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    source = pd.Series([1.0, 2.0], index=pd.DatetimeIndex([timestamp, timestamp]))

    with pytest.raises(ValueError, match="timestamps must be unique"):
        ingest_entso._normalize_native_series(source, "load_mw")


def test_border_directions_remain_separate_and_net_requires_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    t1 = pd.Timestamp("2026-01-01T01:00:00Z")
    t2 = pd.Timestamp("2026-01-01T02:00:00Z")
    supplied = {
        "scheduled_export_ch_de_mw": pd.Series([100.0, 120.0], index=[t0, t1]),
        "scheduled_import_ch_de_mw": pd.Series([20.0, 30.0], index=[t1, t2]),
        "scheduled_export_ch_fr_mw": pd.Series([40.0], index=[t0]),
    }

    def fake_query(*_args, name: str, **_kwargs) -> pd.Series:
        return supplied.get(name, pd.Series(dtype=float, name=name)).rename(name)

    class BorderClient:
        query_scheduled_exchanges = object()
        query_crossborder_flows = object()
        query_net_transfer_capacity_dayahead = object()

    monkeypatch.setattr(ingest_entso, "_query_series_or_empty", fake_query)

    observed = ingest_entso._load_swiss_border_features(
        BorderClient(),
        t0,
        pd.Timestamp("2026-01-02T00:00:00Z"),
    )

    assert "scheduled_export_ch_de_mw" in observed
    assert "scheduled_import_ch_de_mw" in observed
    assert "scheduled_net_export_ch_de_mw" in observed
    assert observed.loc[t1, "scheduled_net_export_ch_de_mw"] == 100.0
    assert pd.isna(observed.loc[t0, "scheduled_net_export_ch_de_mw"])
    assert pd.isna(observed.loc[t2, "scheduled_net_export_ch_de_mw"])
    assert "scheduled_export_ch_fr_mw" in observed
    assert "scheduled_net_export_ch_fr_mw" not in observed


def test_neighbor_mixed_native_grids_are_not_forward_filled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hourly = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    quarter_hourly = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")

    class FakeClient:
        def query_load(self, zone: str, **_kwargs) -> pd.DataFrame:
            if zone != "DE_LU":
                raise RuntimeError("not supplied")
            return pd.DataFrame({"Actual Load": [100.0, 110.0]}, index=hourly)

        def query_generation(self, zone: str, **_kwargs) -> pd.DataFrame:
            if zone != "DE_LU":
                raise RuntimeError("not supplied")
            return pd.DataFrame(
                {
                    "Solar": [10.0, 11.0, 12.0, 13.0, 14.0],
                    "Wind Onshore": [20.0, 21.0, 22.0, 23.0, 24.0],
                    "Wind Offshore": [5.0, 5.0, 5.0, 5.0, 5.0],
                },
                index=quarter_hourly,
            )

    monkeypatch.setattr(
        ingest_entso,
        "_retry",
        lambda func, *args, **kwargs: func(*args, **kwargs),
    )

    observed = ingest_entso._load_neighbor_power_features(
        FakeClient(),
        hourly[0],
        hourly[-1] + pd.Timedelta(hours=1),
    )

    assert len(observed) == 5
    assert pd.isna(observed.loc[quarter_hourly[1], "load_de_mw"])
    assert pd.isna(observed.loc[quarter_hourly[1], "residual_load_de_mw"])
    assert observed.loc[hourly[0], "residual_load_de_mw"] == 65.0
    assert observed.loc[hourly[1], "residual_load_de_mw"] == 67.0


def test_ingest_source_contains_no_implicit_upsampling_or_zero_fill() -> None:
    source = inspect.getsource(ingest_entso)

    assert ".resample(" not in source
    assert ".ffill(" not in source
    assert ".fillna(0" not in source
    assert ".iloc[:, -1]" not in source


def test_existing_legacy_cache_is_rejected_before_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "entso-existing.parquet"
    existing.write_bytes(b"untrusted legacy cache")

    def forbidden_network(*_args, **_kwargs) -> pd.DataFrame:
        raise AssertionError("network access occurred before legacy-cache rejection")

    monkeypatch.setattr(ingest_entso, "load_from_api", forbidden_network)

    with pytest.raises(FileExistsError, match="unproven native cadence"):
        ingest_entso.fetch_and_cache(
            "2026-01-01",
            "2026-01-02",
            parquet_path=existing,
        )
