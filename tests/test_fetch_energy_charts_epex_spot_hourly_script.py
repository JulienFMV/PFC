from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import fetch_energy_charts_epex_spot_hourly as script


def test_build_hourly_spot_excludes_incomplete_quarter_hour_hours() -> None:
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

    assert hourly.index.tolist() == [pd.Timestamp("2026-07-10T00:00:00Z")]
    assert hourly["price_eur_mwh"].tolist() == [15.5]
    assert coverage["expected_hour_count"] == 3
    assert coverage["observed_hour_count"] == 1
    assert coverage["missing_hour_count"] == 2
    assert coverage["first_missing_utc"] == "2026-07-10T01:00:00Z"
    assert coverage["spot_max_utc"] == "2026-07-10T00:00:00Z"
    assert coverage["full_window_covered"] is False
    assert coverage["source_cadence_seconds"] == 900
    assert coverage["incomplete_source_observation_count"] == 1


def test_raw_replay_rejects_partial_qh_second_hour_as_full_coverage() -> None:
    timestamps = [
        int(value.timestamp())
        for value in pd.date_range(
            "2026-07-10T00:00:00Z",
            periods=5,
            freq="15min",
        )
    ]
    payload = json.dumps(
        {"unix_seconds": timestamps, "price": [10, 11, 12, 13, 14]}
    ).encode("ascii")

    hourly, coverage = script.replay_provider_raw_hourly_bytes(
        payload,
        start="2026-07-10T00:00:00Z",
        end="2026-07-10T02:00:00Z",
    )

    assert hourly.index.tolist() == [pd.Timestamp("2026-07-10T00:00:00Z")]
    assert coverage["full_window_covered"] is False
    assert coverage["missing_hour_count"] == 1
    assert coverage["first_missing_utc"] == "2026-07-10T01:00:00Z"


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


def test_fetch_hourly_spot_uses_energy_charts_date_params(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_fetch(*, start, end, bzn):
        calls.append((start, end, bzn))
        return pd.DataFrame(
            {"price_eur_mwh": [10.0]},
            index=pd.to_datetime(["2026-07-10T01:00:00Z"], utc=True),
        )

    monkeypatch.setattr(script, "_fetch_raw_prices", fake_fetch)

    summary = script.fetch_hourly_spot(
        start="2026-07-10T01:00:00Z",
        end="2026-07-10T02:00:00Z",
        bzn="CH",
        output_parquet=tmp_path / "spot.parquet",
        summary_json=tmp_path / "summary.json",
    )

    assert calls == [("2026-07-10", "2026-07-11", "CH")]
    assert summary["api_start"] == "2026-07-10"
    assert summary["api_end"] == "2026-07-11"


def test_fetch_hourly_spot_records_fetch_error_without_traceback(tmp_path: Path, monkeypatch) -> None:
    def fake_fetch(*, start, end, bzn):
        raise RuntimeError("not published")

    monkeypatch.setattr(script, "_fetch_raw_prices", fake_fetch)
    output = tmp_path / "spot.parquet"
    summary_path = tmp_path / "summary.json"

    summary = script.fetch_hourly_spot(
        start="2026-07-10",
        end="2026-07-10T02:00:00Z",
        bzn="CH",
        output_parquet=output,
        summary_json=summary_path,
    )

    assert summary["status"] == "SPOT_FETCH_ERROR"
    assert summary["full_window_covered"] is False
    assert summary["expected_hour_count"] == 2
    assert summary["missing_hour_count"] == 2
    assert "RuntimeError: not published" == summary["fetch_error"]
    assert not output.exists()
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "SPOT_FETCH_ERROR"


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


def test_fetch_hourly_spot_persists_exact_provider_raw_bytes(tmp_path: Path, monkeypatch) -> None:
    payload = b'{"unix_seconds":[1783641600],"price":[42.5],"metadata":{"revision":"pit"}}'
    raw = pd.DataFrame(
        {"price_eur_mwh": [42.5]},
        index=pd.to_datetime([1783641600], unit="s", utc=True),
    )
    raw.attrs["provider_raw_response_bytes"] = payload
    monkeypatch.setattr(script, "_fetch_raw_prices", lambda *, start, end, bzn: raw)
    raw_path = tmp_path / "provider-raw.json"

    summary = script.fetch_hourly_spot(
        start="2026-07-10T00:00:00Z",
        end="2026-07-10T01:00:00Z",
        bzn="CH",
        output_parquet=tmp_path / "spot.parquet",
        summary_json=tmp_path / "summary.json",
        provider_raw_json=raw_path,
    )

    assert summary["status"] == "OK"
    assert raw_path.read_bytes() == payload
    assert summary["provider_raw_json_sha256"] == script._sha256(raw_path)
