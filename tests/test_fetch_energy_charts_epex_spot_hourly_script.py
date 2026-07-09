from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import fetch_energy_charts_epex_spot_hourly as script


def test_build_hourly_spot_aggregates_only_observed_points() -> None:
    raw = pd.DataFrame(
        {"price_eur_mwh": [10.0, 14.0, 18.0, 20.0, 30.0]},
        index=pd.to_datetime(
            [
                "2026-07-10T00:00:00Z",
                "2026-07-10T00:15:00Z",
                "2026-07-10T00:30:00Z",
                "2026-07-10T00:45:00Z",
                "2026-07-10T01:00:00Z",
            ],
            utc=True,
        ),
    )

    hourly, coverage = script._build_hourly_spot(
        raw,
        start_utc=pd.Timestamp("2026-07-10T00:00:00Z"),
        end_utc=pd.Timestamp("2026-07-10T03:00:00Z"),
    )

    assert hourly.index.tolist() == [
        pd.Timestamp("2026-07-10T00:00:00Z"),
        pd.Timestamp("2026-07-10T01:00:00Z"),
    ]
    assert hourly["price_eur_mwh"].tolist() == [15.5, 30.0]
    assert coverage["expected_hour_count"] == 3
    assert coverage["observed_hour_count"] == 2
    assert coverage["missing_hour_count"] == 1
    assert coverage["first_missing_utc"] == "2026-07-10T02:00:00Z"
    assert coverage["spot_max_utc"] == "2026-07-10T01:00:00Z"
    assert coverage["full_window_covered"] is False


def test_fetch_hourly_spot_fails_closed_without_writing_partial_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = pd.DataFrame(
        {"price_eur_mwh": [10.0]},
        index=pd.to_datetime(["2026-07-10T00:00:00Z"], utc=True),
    )
    monkeypatch.setattr(script, "_fetch_raw_prices", lambda *, start, end, bzn: raw)
    output = tmp_path / "spot.parquet"
    summary_path = tmp_path / "summary.json"

    summary = script.fetch_hourly_spot(
        start="2026-07-10",
        end="2026-07-10T02:00:00Z",
        bzn="CH",
        output_parquet=output,
        summary_json=summary_path,
    )

    assert summary["status"] == "PARTIAL_COVERAGE"
    assert summary["full_window_covered"] is False
    assert summary["missing_hour_count"] == 1
    assert not output.exists()
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["output_parquet_sha256"] is None


def test_main_writes_complete_hourly_parquet(tmp_path: Path, monkeypatch) -> None:
    raw = pd.DataFrame(
        {"price_eur_mwh": [10.0, 20.0]},
        index=pd.to_datetime(["2026-07-10T00:00:00Z", "2026-07-10T01:00:00Z"], utc=True),
    )
    monkeypatch.setattr(script, "_fetch_raw_prices", lambda *, start, end, bzn: raw)
    output = tmp_path / "spot.parquet"
    summary_path = tmp_path / "summary.json"

    code = script.main(
        [
            "--start",
            "2026-07-10",
            "--end",
            "2026-07-10T02:00:00Z",
            "--output-parquet",
            str(output),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert code == 0
    hourly = pd.read_parquet(output)
    assert hourly.index.tolist() == [
        pd.Timestamp("2026-07-10T00:00:00Z"),
        pd.Timestamp("2026-07-10T01:00:00Z"),
    ]
    assert hourly["price_eur_mwh"].tolist() == [10.0, 20.0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "OK"
    assert summary["output_parquet_sha256"]


def test_main_returns_nonzero_for_partial_coverage(tmp_path: Path, monkeypatch) -> None:
    raw = pd.DataFrame(
        {"price_eur_mwh": [10.0]},
        index=pd.to_datetime(["2026-07-10T00:00:00Z"], utc=True),
    )
    monkeypatch.setattr(script, "_fetch_raw_prices", lambda *, start, end, bzn: raw)

    code = script.main(
        [
            "--start",
            "2026-07-10",
            "--end",
            "2026-07-10T02:00:00Z",
            "--output-parquet",
            str(tmp_path / "spot.parquet"),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert code == 1
