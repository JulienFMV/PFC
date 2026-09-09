from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pfc_shaping.data.lt_replay_transforms as replay_transforms_module
from pfc_shaping.data import ingest_entso, ingest_epex, ingest_hydro
from pfc_shaping.data.lt_input_replay import approved_parser_payload_for_role
from pfc_shaping.data.lt_replay_transforms import (
    build_entso_features,
    build_hydro_water_value,
    clean_epex,
)


def test_portable_epex_transform_matches_acquisition_transform() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {"price_eur_mwh": [40.0, 41.0, 39.0, 42.0, 38.0, 400.0] * 2},
        index=index,
    )

    pd.testing.assert_frame_equal(clean_epex(frame), ingest_epex._clean(frame))


def test_portable_entso_transform_matches_acquisition_transform() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    values = np.arange(12, dtype=float)
    frame = pd.DataFrame(
        {
            "solar_mw": values,
            "load_mw": 100.0 + values,
            "cross_border_mw": -5.0 + values,
            "scheduled_net_export_ch_de_mw": values * 2,
            "ntc_total_ch_de_mw": 50.0 + values,
            "ntc_net_ch_de_mw": values - 3.0,
            "ntc_export_ch_de_mw": 20.0 + values,
            "ntc_import_ch_de_mw": 10.0 + values,
            "fr_nuclear_unavailable_mw": values * 1000.0,
        },
        index=index,
    )

    pd.testing.assert_frame_equal(
        build_entso_features(frame),
        ingest_entso.build_features(frame),
    )


def test_portable_hydro_transform_matches_acquisition_transform() -> None:
    index = pd.to_datetime(
        [
            "2022-01-05T00:00:00Z",
            "2023-01-04T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2025-01-01T00:00:00Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "fill_pct": [45.0, 50.0, 47.0, 55.0],
            "fill_gwh": [450.0, 500.0, 470.0, 550.0],
            "max_capacity_gwh": [1000.0] * 4,
        },
        index=index,
    )

    pd.testing.assert_frame_equal(
        build_hydro_water_value(frame),
        ingest_hydro.build_water_value(frame),
    )


def test_feature_parser_identity_is_readable_from_zipimport_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = Path(str(replay_transforms_module.__file__)).read_bytes()
    monkeypatch.setattr(
        replay_transforms_module,
        "__file__",
        str(tmp_path / "publisher.pyz" / "pfc_shaping" / "data" / "lt_replay_transforms.py"),
    )

    assert approved_parser_payload_for_role("epex_ch") == expected


def test_epex_feature_engineering_is_prefix_invariant() -> None:
    index = pd.date_range("2026-06-01", periods=24, freq="15min", tz="UTC")
    prefix = pd.DataFrame(
        {"price_eur_mwh": [50.0 + float(position) for position in range(20)]},
        index=index[:20],
    )
    future = pd.DataFrame(
        {"price_eur_mwh": [10000.0, -10000.0, 20000.0, -20000.0]},
        index=index[20:],
    )

    expected = clean_epex(prefix)
    observed = clean_epex(pd.concat([prefix, future])).loc[prefix.index]

    pd.testing.assert_frame_equal(expected, observed, check_freq=False)


def test_entso_feature_engineering_is_prefix_invariant() -> None:
    index = pd.date_range("2026-06-01", periods=24, freq="15min", tz="UTC")
    prefix_values = np.arange(20, dtype=float)
    prefix = pd.DataFrame(
        {
            "load_mw": 7000.0 + prefix_values,
            "solar_mw": 100.0 + prefix_values,
            "wind_mw": 500.0 + prefix_values,
            "cross_border_mw": prefix_values - 10.0,
            "fr_nuclear_unavailable_mw": prefix_values * 10.0,
        },
        index=index[:20],
    )
    future = pd.DataFrame(
        {
            "load_mw": [1e9] * 4,
            "solar_mw": [1e9] * 4,
            "wind_mw": [1e9] * 4,
            "cross_border_mw": [-1e9] * 4,
            "fr_nuclear_unavailable_mw": [1e9] * 4,
        },
        index=index[20:],
    )

    expected = build_entso_features(prefix)
    observed = build_entso_features(pd.concat([prefix, future])).loc[prefix.index]

    pd.testing.assert_frame_equal(expected, observed, check_freq=False)


def test_delivery_month_uses_swiss_local_time() -> None:
    index = pd.DatetimeIndex(
        ["2026-03-31T21:45:00Z", "2026-03-31T22:00:00Z"],
        tz="UTC",
    )

    months = replay_transforms_module._delivery_month(index)

    assert list(map(str, months)) == ["2026-03", "2026-04"]


def test_hydro_support_requires_five_prior_same_week_observations_and_scale() -> None:
    short_index = pd.date_range("2025-01-06", periods=2, freq="W-MON", tz="UTC")
    short = pd.DataFrame(
        {
            "fill_pct": [40.0, 41.0],
            "fill_gwh": [4000.0, 4100.0],
            "max_capacity_gwh": [10000.0, 10000.0],
        },
        index=short_index,
    )
    assert not bool(build_hydro_water_value(short)["water_value_supported"].any())

    long_index = pd.date_range("2020-01-06", periods=312, freq="W-MON", tz="UTC")
    long = pd.DataFrame(
        {
            "fill_pct": [
                40.0 + float(position % 52) * 0.2 + float(position // 52)
                for position in range(len(long_index))
            ],
            "fill_gwh": 5000.0,
            "max_capacity_gwh": 10000.0,
        },
        index=long_index,
    )
    transformed = build_hydro_water_value(long)

    assert bool(
        np.isfinite(
            transformed[
                [
                    "water_value_reference_mean_pct",
                    "water_value_reference_std_pct",
                    "water_value_reference_se_pct",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    assert bool(transformed["water_value_supported"].tail(8).all())
    assert bool((transformed["water_value_history_count"].tail(8) >= 5).all())
    assert bool((transformed["water_value_reference_std_pct"].tail(8) > 0).all())

    zero_scale = pd.DataFrame(
        {
            "fill_pct": 50.0,
            "fill_gwh": 5000.0,
            "max_capacity_gwh": 10000.0,
        },
        index=pd.DatetimeIndex(
            [
                "2019-12-30",
                "2021-01-04",
                "2022-01-03",
                "2023-01-02",
                "2024-01-01",
                "2024-12-30",
            ],
            tz="Europe/Zurich",
        ).tz_convert("UTC"),
    )
    zero_scale.iloc[-1, zero_scale.columns.get_loc("fill_pct")] = 60.0
    unsupported = build_hydro_water_value(zero_scale).iloc[-1]
    assert not bool(unsupported["water_value_supported"])
    assert unsupported["water_value_history_count"] >= 5
