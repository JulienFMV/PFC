from __future__ import annotations

import json
from pathlib import Path

import pytest

from pfc_shaping.data.governed_lt_acquisition import ENERGY_CHARTS_PRICE_URL
from scripts.capture_public_energy_charts_lt import capture_energy_charts_price

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "governed_lt_acquisition"
    / "energy_charts_price_ch.json"
)


class _Response:
    status_code = 200
    url = (
        ENERGY_CHARTS_PRICE_URL
        + "?bzn=CH&start=2026-07-01T00%3A00Z&end=2026-07-01T23%3A00Z"
    )
    headers = {
        "content-type": "application/json",
        "content-length": str(FIXTURE.stat().st_size),
        "etag": '"fixture"',
    }

    def iter_content(self, *, chunk_size: int):
        assert chunk_size > 0
        yield FIXTURE.read_bytes()


class _Session:
    def __init__(self) -> None:
        self.call: dict[str, object] | None = None

    def get(self, url: str, **kwargs):
        self.call = {"url": url, **kwargs}
        return _Response()


def test_public_capture_writes_exact_builder_spec_and_negative_authority(tmp_path):
    session = _Session()
    output = tmp_path / "capture"

    summary_path = capture_energy_charts_price(
        role="epex_ch",
        start_utc="2026-07-01T00:00:00Z",
        end_utc="2026-07-02T00:00:00Z",
        raw_cadence_minutes=60,
        acquisition_id="capture-test-001",
        output_directory=output,
        session=session,
        ca_bundle=None,
    )

    assert summary_path == output / "capture-summary.json"
    assert (output / "provider-response.json").read_bytes() == FIXTURE.read_bytes()
    spec = json.loads((output / "capture-spec.json").read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert spec["schema_version"] == "lt_provider_capture_spec.v1"
    assert spec["documents"][0]["body_path"] == str(output / "provider-response.json")
    assert spec["documents"][0]["request_parameters"] == {
        "bzn": "CH",
        "start": "2026-07-01T00:00Z",
        "end": "2026-07-01T23:00Z",
    }
    assert summary["status"] == "COMPLETE_LOCAL_UNTRUSTED_CAPTURE"
    assert summary["schema_version"] == "local_public_provider_capture.v2"
    assert summary["production_authorization"] is False
    assert summary["promotion_gate"] is False
    assert summary["timestamp_authority"] == "LOCAL_WORKSTATION_CLOCK_UNTRUSTED"
    assert summary["validated_bronze_rows"] == 96
    assert summary["resolution_provenance"] == {
        "schema_version": "lt_observation_resolution_provenance.v2",
        "role": "epex_ch",
        "source_cadence_seconds": 3600,
        "output_cadence_seconds": 900,
        "source_observation_count": 24,
        "output_observation_count": 96,
        "resampling_method": "FORWARD_FILL_WITHIN_SOURCE_INTERVAL",
        "output_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
        "native_quarter_hour_cadence_eligible": False,
        "native_hourly_cadence_eligible": True,
        "hourly_aggregation_eligible": True,
        "native_quarter_hour_truth_eligible": False,
        "product_identity_status": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED",
        "quarter_hour_truth_blocker": "SOURCE_CADENCE_IS_HOURLY",
        "scientific_use_class": "NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY",
    }
    assert session.call is not None
    assert session.call["allow_redirects"] is False
    assert session.call["stream"] is True


def test_public_capture_fails_before_spec_when_declared_cadence_differs(tmp_path):
    output = tmp_path / "capture"

    with pytest.raises(ValueError, match="provider request does not match the allow-list"):
        capture_energy_charts_price(
            role="epex_ch",
            start_utc="2026-07-01T00:00:00Z",
            end_utc="2026-07-02T00:00:00Z",
            raw_cadence_minutes=15,
            acquisition_id="capture-test-002",
            output_directory=output,
            session=_Session(),
            ca_bundle=None,
        )

    assert (output / "capture-attempt.json").exists()
    assert not (output / "capture-spec.json").exists()
    assert not (output / "capture-summary.json").exists()


def test_public_capture_refuses_existing_output_directory(tmp_path):
    output = tmp_path / "capture"
    output.mkdir()

    with pytest.raises(FileExistsError):
        capture_energy_charts_price(
            role="epex_ch",
            start_utc="2026-07-01T00:00:00Z",
            end_utc="2026-07-02T00:00:00Z",
            raw_cadence_minutes=60,
            acquisition_id="capture-test-003",
            output_directory=output,
            session=_Session(),
            ca_bundle=None,
        )
