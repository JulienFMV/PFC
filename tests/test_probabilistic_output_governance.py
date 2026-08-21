from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pfc_shaping.lt.model.pfc_flavors import PFCFlavors
from pfc_shaping.pipeline.export_euler import export_csv
from pfc_shaping.pipeline.probabilistic_output_governance import (
    deterministic_only_frame,
    deterministic_pfc_export_frame,
    governed_intervals_available,
    quarantine_legacy_intervals,
)
from pfc_shaping.pipeline.production_phases import _save_market_artifacts
from pfc_shaping.pipeline.strict_structured_data import load_strict_yaml


def _pfc_with_advisory_intervals() -> pd.DataFrame:
    index = pd.date_range("2027-01-01", periods=4, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "price_shape": [80.0, 81.0, 82.0, 83.0],
            "p10": [70.0, 71.0, 72.0, 73.0],
            "p90": [90.0, 91.0, 92.0, 93.0],
            "profile_type": ["M+1..M+6"] * 4,
            "confidence": [1.0] * 4,
            "calibrated": [True] * 4,
        },
        index=index,
    )


def test_euler_export_excludes_advisory_interval_columns(tmp_path) -> None:
    destination = export_csv(_pfc_with_advisory_intervals(), tmp_path / "pfc.csv")

    exported = pd.read_csv(destination, sep=";")
    assert list(exported.columns) == [
        "timestamp_local",
        "price_shape",
        "profile_type",
        "confidence",
    ]


def test_deterministic_flavors_drop_interval_columns() -> None:
    flavors = PFCFlavors().generate(_pfc_with_advisory_intervals())

    assert set(flavors) == {"mid_market", "client", "production"}
    assert all("p10" not in frame and "p90" not in frame for frame in flavors.values())


def test_probabilistic_client_floor_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="rolling-origin evidence"):
        PFCFlavors(client_percentile="p90")


@pytest.mark.parametrize(
    "alias",
    ["q10", "quantile_90", "lower", "upper", "structural_p10_eur_mwh"],
)
def test_deterministic_schema_rejects_interval_aliases(alias: str) -> None:
    frame = pd.DataFrame({"price_shape": [80.0], alias: [70.0]})

    with pytest.raises(ValueError, match="outside the deterministic LT schema"):
        deterministic_only_frame(frame, context="test")


def test_legacy_interval_quarantine_drops_bounds_and_classifies() -> None:
    frame = pd.DataFrame({"price_shape": [80.0], "p10": [70.0], "p90": [90.0]})

    quarantined = quarantine_legacy_intervals(frame, context="legacy-test")

    assert list(quarantined.columns) == ["price_shape"]
    assert quarantined.attrs["probabilistic_status"] == "QUARANTINED_UNCALIBRATED"
    assert quarantined.attrs["quarantined_interval_columns"] == ["p10", "p90"]


@pytest.mark.parametrize(
    "aliases",
    [
        ("q10", "q90"),
        ("Q 05", "Q 95"),
        ("lower", "upper"),
        ("interval_lower", "interval_upper"),
        ("quantile_10", "percentile_90"),
        ("structural_p10_eur_mwh", "price_P90"),
        ("quantile10", "Percentile90"),
        ("lowerBound", "UPPERBOUND"),
        ("structuralP10EurMwh", "priceQuantile90"),
        ("scenario_low", "scenario_high"),
        ("ci80Low", "ci80High"),
        ("lower95", "upper95"),
        ("10thPercentile", "90thPercentile"),
        ("pctl10", "pctl90"),
    ],
)
def test_legacy_interval_quarantine_drops_all_known_aliases(
    aliases: tuple[str, str],
) -> None:
    frame = pd.DataFrame(
        {"price_shape": [80.0], aliases[0]: [70.0], aliases[1]: [90.0]}
    )

    quarantined = quarantine_legacy_intervals(frame, context="legacy-alias-test")

    assert list(quarantined.columns) == ["price_shape"]
    assert quarantined.attrs["probabilistic_status"] == "QUARANTINED_UNCALIBRATED"
    assert quarantined.attrs["quarantined_interval_columns"] == list(aliases)


@pytest.mark.parametrize(
    ("p10", "p90"),
    [
        (70.0, 90.0),
        (float("nan"), 90.0),
        (float("inf"), 90.0),
        (70.0, float("-inf")),
        ("not-numeric", 90.0),
        (True, 90.0),
        (90.0, 70.0),
    ],
)
def test_self_declared_interval_status_never_bootstraps_availability(
    p10: object,
    p90: object,
) -> None:
    frame = pd.DataFrame({"price_shape": [80.0], "p10": [p10], "p90": [p90]})
    frame.attrs["probabilistic_status"] = "CALIBRATED_ROLLING_ORIGIN"

    assert governed_intervals_available(frame) is False
    frame.attrs["probabilistic_status"] = "QUARANTINED_UNCALIBRATED"
    assert governed_intervals_available(frame) is False


@pytest.mark.parametrize("status", [None, "", "UNKNOWN", "CALIBRATED"])
def test_governed_interval_availability_requires_allowlisted_status(status: object) -> None:
    frame = pd.DataFrame({"price_shape": [80.0], "p10": [70.0], "p90": [90.0]})
    if status is not None:
        frame.attrs["probabilistic_status"] = status

    assert governed_intervals_available(frame) is False


def test_governed_interval_availability_rejects_partial_invalid_series() -> None:
    frame = pd.DataFrame(
        {"price_shape": [80.0, 81.0], "p10": [70.0, float("nan")], "p90": [90.0, 91.0]}
    )
    frame.attrs["probabilistic_status"] = "CALIBRATED_ROLLING_ORIGIN"

    assert governed_intervals_available(frame) is False


def test_governed_interval_availability_rejects_duplicate_labels() -> None:
    frame = pd.DataFrame([[100.0, 101.0, 50.0]], columns=["p10", "p10", "p90"])
    frame.attrs["probabilistic_status"] = "CALIBRATED_ROLLING_ORIGIN"

    assert governed_intervals_available(frame) is False


def test_deterministic_only_frame_rejects_duplicate_labels() -> None:
    frame = pd.DataFrame([[80.0, 70.0]], columns=["price_shape", "price_shape"])

    with pytest.raises(ValueError, match="duplicate column labels"):
        deterministic_only_frame(frame, context="test deterministic sink")


@pytest.mark.parametrize(
    "aliases",
    [
        ("forecast_floor", "forecast_ceiling"),
        ("prediction_min", "prediction_max"),
        ("interval_lwr", "interval_upr"),
        ("borne_basse", "borne_haute"),
        ("perc10", "perc90"),
    ],
)
def test_dashboard_pfc_export_uses_positive_schema(aliases: tuple[str, str]) -> None:
    frame = pd.DataFrame(
        {
            "price_shape": [80.0],
            "profile_type": ["M+1"],
            "confidence": [1.0],
            aliases[0]: [70.0],
            aliases[1]: [90.0],
        }
    )

    exported = deterministic_pfc_export_frame(frame, context="test dashboard export")

    assert list(exported.columns) == ["price_shape", "profile_type", "confidence"]


def test_dashboard_pfc_export_rejects_duplicate_labels() -> None:
    frame = pd.DataFrame([[80.0, 70.0]], columns=["price_shape", "price_shape"])

    with pytest.raises(ValueError, match="duplicate column labels"):
        deterministic_pfc_export_frame(frame, context="test dashboard export")


def test_direct_lt_output_sink_rejects_populated_bounds(tmp_path: Path) -> None:
    frame = _pfc_with_advisory_intervals()
    artifact = SimpleNamespace(
        code="DE",
        pfc=frame,
        out_base=str(tmp_path / "pfc_de_15min"),
        sh=object(),
        wv=None,
    )

    with pytest.raises(RuntimeError, match="Direct LT artifact publication is disabled"):
        _save_market_artifacts(artifact, str(tmp_path), logging.getLogger("test"))

    assert not (tmp_path / "pfc_de_15min.parquet").exists()
    assert not (tmp_path / "pfc_de_15min.csv").exists()


def test_runtime_export_config_contains_no_lt_intervals() -> None:
    config_path = Path(__file__).resolve().parents[1] / "pfc_shaping" / "config.yaml"
    config = load_strict_yaml(config_path.read_text(encoding="utf-8"))

    assert set(config["export"]["columns"]).isdisjoint(
        {"p10", "p90", "q10", "quantile_90", "lower", "upper"}
    )
