from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

import pfc_shaping.cli.governed_acquisition_builder as builder_module
import pfc_shaping.data.governed_lt_acquisition as acquisition_module
import pfc_shaping.durable_artifact as durable_module
from pfc_shaping.cli.governed_acquisition_builder import build_acquisition
from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_PRICE_URL,
    ENERGY_CHARTS_SOURCE_SYSTEM,
    ENTSOE_API_URL,
    ENTSOE_SOURCE_SYSTEM,
    SFOE_OGD17_URL,
    SFOE_SOURCE_SYSTEM,
    CapturedProviderDocument,
    GovernedLTAcquisitionError,
    approved_provider_parser_payload_for_role,
    build_provider_transform_config,
    build_raw_envelope,
    dataframe_semantic_sha256,
    energy_price_resolution_provenance,
    require_native_quarter_hour_price_truth,
    transform_provider_raw,
    validate_provider_document_inventory,
    validate_raw_envelope,
    verify_provider_raw_replay,
)

FIXTURES = Path(__file__).parent / "fixtures" / "governed_lt_acquisition"
START = "2026-07-01T00:00:00Z"
END = "2026-07-01T04:00:00Z"
EPEX_END = "2026-07-02T00:00:00Z"
RECEIVED = "2026-07-01T04:05:00Z"


def test_energy_charts_exact_json_replays_to_complete_15_minute_bronze() -> None:
    envelope = _energy_charts_envelope()
    parsed = validate_raw_envelope(envelope)

    assert parsed["source_system"] == ENERGY_CHARTS_SOURCE_SYSTEM
    assert parsed["documents"][0]["body_sha256"]

    frame = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=build_provider_transform_config(
            role="epex_ch", start_utc=START, end_utc=EPEX_END
        ),
    )

    assert list(frame.columns) == ["price_eur_mwh"]
    assert len(frame) == 96
    assert frame.index[0] == pd.Timestamp(START)
    assert frame.index[-1] == pd.Timestamp("2026-07-01T23:45:00Z")
    assert frame.iloc[:4, 0].tolist() == [54.25] * 4
    assert frame.iloc[-4:, 0].tolist() == [55.0] * 4
    assert energy_price_resolution_provenance(frame, role="epex_ch") == {
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
    with pytest.raises(
        GovernedLTAcquisitionError,
        match="not native quarter-hour truth",
    ):
        require_native_quarter_hour_price_truth(frame, role="epex_ch")


def test_legacy_energy_charts_loader_is_explicitly_non_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pfc_shaping.data import ingest_energy_charts as legacy

    payload = json.loads(
        (FIXTURES / "energy_charts_price_ch.json").read_text(encoding="utf-8")
    )

    class _Response:
        @staticmethod
        def json() -> dict[str, object]:
            return payload

    monkeypatch.setattr(legacy, "_retry_get", lambda *_args, **_kwargs: _Response())
    frame = legacy.load_prices("2026-07-01", "2026-07-02", bzn="CH")

    status = frame.attrs[legacy.LEGACY_RESOLUTION_STATUS_ATTR]
    assert status["production_authorization"] is False
    assert status["native_quarter_hour_truth_eligible"] is False
    with pytest.raises(
        GovernedLTAcquisitionError,
        match="resolution provenance is missing or ambiguous",
    ):
        require_native_quarter_hour_price_truth(frame, role="epex_ch")


def test_energy_charts_quarter_hour_cadence_still_requires_product_identity() -> None:
    timestamps = [
        int(value.timestamp())
        for value in pd.date_range(START, periods=96, freq="15min")
    ]
    body = json.dumps(
        {
            "deprecated": False,
            "license_info": "CC BY 4.0 test fixture",
            "price": [50.0 + position / 100.0 for position in range(96)],
            "unit": "EUR / MWh",
            "unix_seconds": timestamps,
        },
        separators=(",", ":"),
    ).encode("ascii")
    envelope = build_raw_envelope(
        acquisition_id="acquisition-qh-001",
        source_role="epex_ch",
        source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
        source_locator=ENERGY_CHARTS_PRICE_URL,
        provider_id=ENERGY_CHARTS_SOURCE_SYSTEM,
        received_at_utc=RECEIVED,
        documents=[
            CapturedProviderDocument(
                document_id="day_ahead_price",
                request_url=ENERGY_CHARTS_PRICE_URL,
                request_parameters={
                    "bzn": "CH",
                    "start": "2026-07-01T00:00Z",
                    "end": "2026-07-01T23:45Z",
                },
                response_media_type="application/json",
                body=body,
            )
        ],
    )

    frame = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=build_provider_transform_config(
            role="epex_ch", start_utc=START, end_utc=EPEX_END
        ),
    )
    provenance = energy_price_resolution_provenance(frame, role="epex_ch")

    assert provenance["source_cadence_seconds"] == 900
    assert provenance["resampling_method"] == "NONE"
    assert provenance["native_quarter_hour_cadence_eligible"] is True
    assert provenance["native_hourly_cadence_eligible"] is False
    assert provenance["hourly_aggregation_eligible"] is True
    assert provenance["native_quarter_hour_truth_eligible"] is False
    assert provenance["quarter_hour_truth_blocker"] == (
        "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_REQUIRED"
    )
    assert frame["price_eur_mwh"].nunique() == 96
    with pytest.raises(
        GovernedLTAcquisitionError,
        match="EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_REQUIRED",
    ):
        require_native_quarter_hour_price_truth(frame, role="epex_ch")


def test_energy_charts_day_ahead_lead_accepts_exact_48_hour_boundary() -> None:
    frame = transform_provider_raw(
        envelope_payload=_energy_charts_envelope(
            received_at_utc="2026-06-30T00:00:00Z"
        ),
        parser_config=build_provider_transform_config(
            role="epex_ch",
            start_utc=START,
            end_utc=EPEX_END,
        ),
    )

    assert len(frame) == 96


def test_energy_charts_day_ahead_lead_rejects_one_second_over_48_hours() -> None:
    with pytest.raises(
        GovernedLTAcquisitionError,
        match="day-ahead window exceeds the received_at_utc lead ceiling",
    ):
        transform_provider_raw(
            envelope_payload=_energy_charts_envelope(
                received_at_utc="2026-06-29T23:59:59Z"
            ),
            parser_config=build_provider_transform_config(
                role="epex_ch",
                start_utc=START,
                end_utc=EPEX_END,
            ),
        )


def test_named_entsoe_xml_bundle_replays_load_solar_and_wind() -> None:
    envelope = build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="entso",
        source_system=ENTSOE_SOURCE_SYSTEM,
        source_locator=ENTSOE_API_URL,
        provider_id=ENTSOE_SOURCE_SYSTEM,
        received_at_utc=RECEIVED,
        documents=[
            CapturedProviderDocument(
                document_id="actual_load_ch",
                request_url=ENTSOE_API_URL,
                request_parameters=_entsoe_parameters("A65", "outBiddingZone_Domain"),
                response_media_type="application/xml",
                body=(FIXTURES / "entsoe_actual_load_ch.xml").read_bytes(),
                credential_reference="env:ENTSOE_API_KEY",
            ),
            CapturedProviderDocument(
                document_id="actual_generation_ch",
                request_url=ENTSOE_API_URL,
                request_parameters=_entsoe_parameters("A75", "in_Domain"),
                response_media_type="application/xml",
                body=(FIXTURES / "entsoe_actual_generation_ch.xml").read_bytes(),
                credential_reference="env:ENTSOE_API_KEY",
            ),
        ],
    )

    frame = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=build_provider_transform_config(
            role="entso", start_utc=START, end_utc=END
        ),
    )

    assert list(frame.columns) == ["load_mw", "solar_mw", "wind_mw"]
    assert len(frame) == 16
    assert frame.loc[pd.Timestamp("2026-07-01T01:45:00Z"), "load_mw"] == 7100.0
    assert frame.loc[pd.Timestamp("2026-07-01T02:15:00Z"), "solar_mw"] == 80.0
    assert frame.loc[pd.Timestamp("2026-07-01T03:45:00Z"), "wind_mw"] == 90.0


def test_entsoe_actuals_reject_received_at_before_window_end() -> None:
    with pytest.raises(
        GovernedLTAcquisitionError,
        match="actual window extends beyond received_at_utc",
    ):
        transform_provider_raw(
            envelope_payload=_entsoe_envelope(
                received_at_utc="2026-06-30T00:00:00Z"
            ),
            parser_config=build_provider_transform_config(
                role="entso",
                start_utc=START,
                end_utc=END,
            ),
        )


def test_entsoe_two_chunk_inventory_replays_contiguous_window() -> None:
    documents = [
        *_entsoe_chunk_documents(
            sequence=1,
            start="2026-07-01T00:00:00Z",
            end="2026-07-01T02:00:00Z",
            base=7000.0,
        ),
        *_entsoe_chunk_documents(
            sequence=2,
            start="2026-07-01T02:00:00Z",
            end="2026-07-01T04:00:00Z",
            base=7200.0,
        ),
    ]
    envelope = build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="entso",
        source_system=ENTSOE_SOURCE_SYSTEM,
        source_locator=ENTSOE_API_URL,
        provider_id=ENTSOE_SOURCE_SYSTEM,
        received_at_utc=RECEIVED,
        documents=documents,
    )

    frame = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=build_provider_transform_config(
            role="entso", start_utc=START, end_utc=END
        ),
    )

    assert len(frame) == 16
    assert frame.index.is_monotonic_increasing
    assert frame.loc[pd.Timestamp("2026-07-01T01:45:00Z"), "load_mw"] == 7100.0
    assert frame.loc[pd.Timestamp("2026-07-01T02:00:00Z"), "load_mw"] == 7200.0


def test_entsoe_chunk_gap_fails_closed_before_body_replay() -> None:
    documents = [
        *_entsoe_chunk_documents(
            sequence=1,
            start="2026-07-01T00:00:00Z",
            end="2026-07-01T02:00:00Z",
            base=7000.0,
        ),
        *_entsoe_chunk_documents(
            sequence=2,
            start="2026-07-01T02:15:00Z",
            end="2026-07-01T04:00:00Z",
            base=7200.0,
        ),
    ]
    envelope = build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="entso",
        source_system=ENTSOE_SOURCE_SYSTEM,
        source_locator=ENTSOE_API_URL,
        provider_id=ENTSOE_SOURCE_SYSTEM,
        received_at_utc=RECEIVED,
        documents=documents,
    )

    with pytest.raises(GovernedLTAcquisitionError, match="gap, overlap"):
        transform_provider_raw(
            envelope_payload=envelope,
            parser_config=build_provider_transform_config(
                role="entso", start_utc=START, end_utc=END
            ),
        )


def test_entsoe_chunk_inventory_rejects_reordered_pair() -> None:
    with pytest.raises(GovernedLTAcquisitionError, match="inventory mismatch"):
        validate_provider_document_inventory(
            "entso",
            (
                "actual_generation_ch_0001",
                "actual_load_ch_0001",
            ),
        )


def test_sfoe_exact_csv_replays_without_cache_or_clipping() -> None:
    envelope = build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="hydro",
        source_system=SFOE_SOURCE_SYSTEM,
        source_locator=SFOE_OGD17_URL,
        provider_id=SFOE_SOURCE_SYSTEM,
        received_at_utc=RECEIVED,
        documents=[
            CapturedProviderDocument(
                document_id="ogd17_reservoir_levels",
                request_url=SFOE_OGD17_URL,
                request_parameters={},
                response_media_type="text/csv",
                body=(FIXTURES / "sfoe_ogd17.csv").read_bytes(),
            )
        ],
    )

    frame = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=build_provider_transform_config(
            role="hydro",
            start_utc="2026-06-14T00:00:00Z",
            end_utc="2026-07-02T00:00:00Z",
        ),
    )

    assert frame["fill_pct"].tolist() == [70.0, 72.0, 74.0]
    assert frame["fill_gwh"].tolist() == [7000.0, 7200.0, 7400.0]
    assert frame["wallis_gwh"].tolist() == [3200.0, 3300.0, 3400.0]
    assert frame.index[0] == pd.Timestamp("2026-06-15", tz="Europe/Zurich").tz_convert(
        "UTC"
    )


def test_sfoe_rejects_observation_date_after_received_at() -> None:
    envelope = build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="hydro",
        source_system=SFOE_SOURCE_SYSTEM,
        source_locator=SFOE_OGD17_URL,
        provider_id=SFOE_SOURCE_SYSTEM,
        received_at_utc="2026-06-20T00:00:00Z",
        documents=[
            CapturedProviderDocument(
                document_id="ogd17_reservoir_levels",
                request_url=SFOE_OGD17_URL,
                request_parameters={},
                response_media_type="text/csv",
                body=(FIXTURES / "sfoe_ogd17.csv").read_bytes(),
            )
        ],
    )

    with pytest.raises(
        GovernedLTAcquisitionError,
        match="observation date extends beyond received_at_utc",
    ):
        transform_provider_raw(
            envelope_payload=envelope,
            parser_config=build_provider_transform_config(
                role="hydro",
                start_utc="2026-06-14T00:00:00Z",
                end_utc="2026-07-02T00:00:00Z",
            ),
        )


@pytest.mark.parametrize(
    "dates",
    [
        ("2026-03-23", "2026-03-30", "2026-04-06"),
        ("2026-10-19", "2026-10-26", "2026-11-02"),
    ],
)
def test_sfoe_weekly_civil_grid_survives_both_swiss_dst_transitions(
    dates: tuple[str, str, str],
) -> None:
    original = (FIXTURES / "sfoe_ogd17.csv").read_text(encoding="utf-8")
    for old, new in zip(("2026-06-15", "2026-06-22", "2026-06-29"), dates):
        original = original.replace(old, new)
    local_start = pd.Timestamp(dates[0], tz="Europe/Zurich")
    local_end = pd.Timestamp(dates[-1], tz="Europe/Zurich") + pd.Timedelta(days=1)
    envelope = build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="hydro",
        source_system=SFOE_SOURCE_SYSTEM,
        source_locator=SFOE_OGD17_URL,
        provider_id=SFOE_SOURCE_SYSTEM,
        received_at_utc=local_end.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        documents=[
            CapturedProviderDocument(
                document_id="ogd17_reservoir_levels",
                request_url=SFOE_OGD17_URL,
                request_parameters={},
                response_media_type="text/csv",
                body=original.encode("utf-8"),
            )
        ],
    )
    frame = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=build_provider_transform_config(
                role="hydro",
                start_utc=(local_start - pd.Timedelta(days=1))
                .tz_convert("UTC")
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                end_utc=local_end.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )

    local_index = frame.index.tz_convert("Europe/Zurich")
    assert all(timestamp.hour == 0 and timestamp.weekday() == 0 for timestamp in local_index)
    assert all(
        delta == pd.Timedelta(days=7)
        for delta in local_index.tz_localize(None).to_series().diff().dropna()
    )


def test_provider_config_rejects_sub_grid_seconds() -> None:
    with pytest.raises(GovernedLTAcquisitionError, match="15-minute UTC grid"):
        build_provider_transform_config(
            role="epex_ch",
            start_utc="2026-07-01T00:00:30Z",
            end_utc="2026-07-02T00:00:30Z",
        )


def test_envelope_rejects_sensitive_request_parameter() -> None:
    with pytest.raises(GovernedLTAcquisitionError, match="sensitive parameter"):
        build_raw_envelope(
            acquisition_id="acquisition-001",
            source_role="entso",
            source_system=ENTSOE_SOURCE_SYSTEM,
            source_locator=ENTSOE_API_URL,
            provider_id=ENTSOE_SOURCE_SYSTEM,
            received_at_utc=RECEIVED,
            documents=[
                CapturedProviderDocument(
                    document_id="actual_load_ch",
                    request_url=ENTSOE_API_URL,
                    request_parameters={"securityToken": "must-not-be-captured"},
                    response_media_type="application/xml",
                    body=b"<root />",
                )
            ],
        )


def test_envelope_rejects_body_mutation_even_when_json_is_recanonicalized() -> None:
    parsed = json.loads(_energy_charts_envelope())
    parsed["documents"][0]["body_base64"] = "e30="
    tampered = json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")

    with pytest.raises(GovernedLTAcquisitionError, match="body size mismatch"):
        validate_raw_envelope(tampered)


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"content-length": "1"}, "Content-Length does not match"),
        ({"content-type": "text/csv"}, "Content-Type does not match"),
    ],
)
def test_envelope_rejects_response_header_body_mismatch(
    headers: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(GovernedLTAcquisitionError, match=message):
        _energy_charts_envelope(response_headers=headers)


def test_provider_transform_config_rejects_unapproved_role() -> None:
    with pytest.raises(GovernedLTAcquisitionError, match="role is not allow-listed"):
        build_provider_transform_config(
            role="commodities",
            start_utc=START,
            end_utc=END,
        )


def test_provider_parser_identity_is_readable_from_zipimport_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = Path(str(acquisition_module.__file__)).read_bytes()
    monkeypatch.setattr(
        acquisition_module,
        "__file__",
        str(tmp_path / "publisher.pyz" / "pfc_shaping" / "data" / "governed_lt_acquisition.py"),
    )

    assert approved_provider_parser_payload_for_role("epex_ch") == expected


def test_entsoe_rejects_duplicate_load_time_series() -> None:
    original = (FIXTURES / "entsoe_actual_load_ch.xml").read_text(encoding="utf-8")
    time_series = "<TimeSeries>" + original.split("<TimeSeries>", 1)[1].split(
        "</TimeSeries>", 1
    )[0] + "</TimeSeries>"
    duplicated = original.replace("</GL_MarketDocument>", f"{time_series}</GL_MarketDocument>")

    with pytest.raises(GovernedLTAcquisitionError, match="exactly one load_mw"):
        transform_provider_raw(
            envelope_payload=_entsoe_envelope(load_body=duplicated.encode("utf-8")),
            parser_config=build_provider_transform_config(
                role="entso", start_utc=START, end_utc=END
            ),
        )


def test_entsoe_rejects_negative_physical_quantity() -> None:
    original = (FIXTURES / "entsoe_actual_load_ch.xml").read_text(encoding="utf-8")
    negative = original.replace("<quantity>7000", "<quantity>-1", 1)

    with pytest.raises(GovernedLTAcquisitionError, match="quantity is negative"):
        transform_provider_raw(
            envelope_payload=_entsoe_envelope(load_body=negative.encode("utf-8")),
            parser_config=build_provider_transform_config(
                role="entso", start_utc=START, end_utc=END
            ),
        )


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        (
            "urn:iec62325.351:tc57wg16:451-6:load-document:3:0",
            "urn:attacker:unrelated",
            "namespace/root mismatch",
        ),
        ("10YCH-SWISSGRIDZ", "10Y1001A1001A83F", "Swiss domain mismatch"),
        (">MAW<", ">KWH<", "quantity unit is not MW"),
    ],
)
def test_entsoe_rejects_semantic_identity_confusion(
    needle: str,
    replacement: str,
    message: str,
) -> None:
    original = (FIXTURES / "entsoe_actual_load_ch.xml").read_text(encoding="utf-8")
    mutated = original.replace(needle, replacement, 1)

    with pytest.raises(GovernedLTAcquisitionError, match=message):
        transform_provider_raw(
            envelope_payload=_entsoe_envelope(load_body=mutated.encode("utf-8")),
            parser_config=build_provider_transform_config(
                role="entso", start_utc=START, end_utc=END
            ),
        )


def test_entsoe_rejects_foreign_namespace_semantic_descendant() -> None:
    original = (FIXTURES / "entsoe_actual_load_ch.xml").read_text(encoding="utf-8")
    mutated = original.replace(
        "<type>A65</type>",
        '<evil:type xmlns:evil="urn:attacker:shadow">A65</evil:type><type>A65</type>',
        1,
    )

    with pytest.raises(GovernedLTAcquisitionError, match="foreign-namespace"):
        transform_provider_raw(
            envelope_payload=_entsoe_envelope(load_body=mutated.encode("utf-8")),
            parser_config=build_provider_transform_config(
                role="entso", start_utc=START, end_utc=END
            ),
        )


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("<businessType>A04</businessType>", "<businessType>A01</businessType>", "business type"),
        (
            "<objectAggregation>A01</objectAggregation>",
            "<objectAggregation>A08</objectAggregation>",
            "object aggregation",
        ),
    ],
)
def test_entsoe_rejects_wrong_time_series_business_identity(
    needle: str,
    replacement: str,
    message: str,
) -> None:
    original = (FIXTURES / "entsoe_actual_load_ch.xml").read_text(encoding="utf-8")
    mutated = original.replace(needle, replacement, 1)

    with pytest.raises(GovernedLTAcquisitionError, match=message):
        transform_provider_raw(
            envelope_payload=_entsoe_envelope(load_body=mutated.encode("utf-8")),
            parser_config=build_provider_transform_config(
                role="entso", start_utc=START, end_utc=END
            ),
        )


def test_entsoe_rejects_same_namespace_nested_identity_shadow() -> None:
    original = (FIXTURES / "entsoe_actual_load_ch.xml").read_text(encoding="utf-8")
    mutated = original.replace(
        "<type>A65</type>",
        "<extension><type>A65</type></extension><type>A75</type>",
        1,
    )

    with pytest.raises(GovernedLTAcquisitionError, match="document type mismatch"):
        transform_provider_raw(
            envelope_payload=_entsoe_envelope(load_body=mutated.encode("utf-8")),
            parser_config=build_provider_transform_config(
                role="entso", start_utc=START, end_utc=END
            ),
        )


def test_sfoe_rejects_extra_schema_column() -> None:
    original = (FIXTURES / "sfoe_ogd17.csv").read_text(encoding="utf-8")
    lines = original.splitlines()
    mutated = "\n".join([f"{lines[0]},unexpected", *[f"{line},1" for line in lines[1:]]])
    envelope = build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="hydro",
        source_system=SFOE_SOURCE_SYSTEM,
        source_locator=SFOE_OGD17_URL,
        provider_id=SFOE_SOURCE_SYSTEM,
        received_at_utc=RECEIVED,
        documents=[
            CapturedProviderDocument(
                document_id="ogd17_reservoir_levels",
                request_url=SFOE_OGD17_URL,
                request_parameters={},
                response_media_type="text/csv",
                body=(mutated + "\n").encode("utf-8"),
            )
        ],
    )

    with pytest.raises(GovernedLTAcquisitionError, match="schema is not exact"):
        transform_provider_raw(
            envelope_payload=envelope,
            parser_config=build_provider_transform_config(
                role="hydro",
                start_utc="2026-06-14T00:00:00Z",
                end_utc="2026-07-02T00:00:00Z",
            ),
        )


def test_energy_charts_preserves_negative_prices() -> None:
    body = json.loads((FIXTURES / "energy_charts_price_ch.json").read_bytes())
    body["price"][0] = -75.0
    frame = transform_provider_raw(
        envelope_payload=_energy_charts_envelope(
            body=json.dumps(body, separators=(",", ":")).encode("ascii")
        ),
        parser_config=build_provider_transform_config(
            role="epex_ch", start_utc=START, end_utc=EPEX_END
        ),
    )

    assert frame.iloc[:4, 0].tolist() == [-75.0] * 4


def test_provider_replay_rejects_parquet_row_bomb_before_dataframe_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"price_eur_mwh": 50.0},
        index=pd.date_range(START, periods=4_097, freq="15min"),
    )
    buffer = BytesIO()
    frame.to_parquet(buffer, index=True)
    parser_payload = approved_provider_parser_payload_for_role("epex_ch")
    monkeypatch.setattr(
        acquisition_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DataFrame allocation must not occur")
        ),
    )

    with pytest.raises(GovernedLTAcquisitionError, match="parquet row budget"):
        verify_provider_raw_replay(
            role="epex_ch",
            source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
            envelope_payload=_energy_charts_envelope(),
            bronze_payload=buffer.getvalue(),
            parser_payload=parser_payload,
            parser_code_sha256=hashlib.sha256(parser_payload).hexdigest(),
            parser_config=build_provider_transform_config(
                role="epex_ch", start_utc=START, end_utc=EPEX_END
            ),
        )


def test_provider_replay_rejects_relabelled_hourly_data_as_native_quarter_hour() -> None:
    envelope = _energy_charts_envelope()
    config = build_provider_transform_config(
        role="epex_ch", start_utc=START, end_utc=EPEX_END
    )
    frame = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=config,
    )
    original_hash = dataframe_semantic_sha256(frame)
    relabelled = frame.copy()
    relabelled.attrs["lt_observation_resolution_provenance"] = {
        **frame.attrs["lt_observation_resolution_provenance"],
        "native_quarter_hour_truth_eligible": True,
    }
    assert dataframe_semantic_sha256(relabelled) != original_hash
    buffer = BytesIO()
    relabelled.to_parquet(buffer, index=True)
    parser_payload = approved_provider_parser_payload_for_role("epex_ch")

    with pytest.raises(
        GovernedLTAcquisitionError,
        match="resolution provenance is inconsistent",
    ):
        verify_provider_raw_replay(
            role="epex_ch",
            source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
            envelope_payload=envelope,
            bronze_payload=buffer.getvalue(),
            parser_payload=parser_payload,
            parser_code_sha256=hashlib.sha256(parser_payload).hexdigest(),
            parser_config=config,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unit", "CHF / MWh", "unit is not EUR/MWh"),
        ("deprecated", True, "endpoint is deprecated"),
    ],
)
def test_energy_charts_rejects_wrong_semantic_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    body = json.loads((FIXTURES / "energy_charts_price_ch.json").read_bytes())
    body[field] = value

    with pytest.raises(GovernedLTAcquisitionError, match=message):
        transform_provider_raw(
            envelope_payload=_energy_charts_envelope(
                body=json.dumps(body, separators=(",", ":")).encode("ascii")
            ),
            parser_config=build_provider_transform_config(
                role="epex_ch", start_utc=START, end_utc=EPEX_END
            ),
        )


def test_provider_replay_rejects_runtime_fingerprint_drift() -> None:
    config = build_provider_transform_config(
        role="epex_ch", start_utc=START, end_utc=EPEX_END
    )
    config["runtime_fingerprint"] = {
        **config["runtime_fingerprint"],
        "python": {"implementation": "mutated", "version": "0", "cache_tag": "none"},
    }

    with pytest.raises(GovernedLTAcquisitionError, match="runtime fingerprint mismatch"):
        transform_provider_raw(
            envelope_payload=_energy_charts_envelope(),
            parser_config=config,
        )


def test_energy_charts_rejects_an_incomplete_provider_window() -> None:
    body = json.dumps(
        {
            "deprecated": False,
            "license_info": "CC BY 4.0 test fixture",
            "unix_seconds": [1782864000, 1782867600, 1782874800],
            "price": [54.25, 52.0, 51.5],
            "unit": "EUR / MWh",
        },
        separators=(",", ":"),
    ).encode("ascii")
    envelope = _energy_charts_envelope(body=body)

    with pytest.raises(GovernedLTAcquisitionError, match="cadence is unsupported or incomplete"):
        transform_provider_raw(
            envelope_payload=envelope,
            parser_config=build_provider_transform_config(
                role="epex_ch", start_utc=START, end_utc=EPEX_END
            ),
        )


def test_separate_unsigned_builder_emits_replayable_artifacts_only(
    tmp_path: Path,
) -> None:
    capture_spec = {
        "schema_version": "lt_provider_capture_spec.v1",
        "acquisition_id": "acquisition-001",
        "source_role": "epex_ch",
        "source_system": ENERGY_CHARTS_SOURCE_SYSTEM,
        "source_locator": ENERGY_CHARTS_PRICE_URL,
        "provider_id": ENERGY_CHARTS_SOURCE_SYSTEM,
        "received_at_utc": RECEIVED,
        "start_utc": START,
        "end_utc": EPEX_END,
        "documents": [
            {
                "document_id": "day_ahead_price",
                "body_path": str((FIXTURES / "energy_charts_price_ch.json").resolve()),
                "request_url": ENERGY_CHARTS_PRICE_URL,
                "request_parameters": {
                    "bzn": "CH",
                    "start": "2026-07-01T00:00Z",
                    "end": "2026-07-01T23:00Z",
                },
                "credential_reference": None,
                "response_status_code": 200,
                "response_media_type": "application/json",
                "response_content_encoding": "identity",
                "response_headers": {"etag": '"fixture-v1"'},
            }
        ],
    }
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(capture_spec, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    output = tmp_path / "unsigned-build"

    manifest_path = build_acquisition(
        capture_spec_path=spec_path,
        output_directory=output,
    )

    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["schema_version"] == "lt_provider_acquisition_build.v2"
    assert manifest["signing_status"] == "UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES"
    assert manifest["publication_status"] == "NOT_PUBLISHED"
    assert manifest["namespace_security_status"] == (
        "BUILDER_MUTABLE_UNTRUSTED_QUARANTINE"
    )
    assert manifest["authority_handoff_status"] == (
        "REQUIRES_BUILDER_INACCESSIBLE_COPY_AND_INDEPENDENT_SIGNATURE"
    )
    assert set(manifest["artifacts"]) == {
        "bronze.parquet",
        "provider-parser.py",
        "provider-raw-envelope.json",
        "provider-transform-config.json",
    }
    bronze = pd.read_parquet(output / "bronze.parquet")
    assert len(bronze) == 96
    assert manifest["resolution_provenance"] == energy_price_resolution_provenance(
        bronze,
        role="epex_ch",
    )
    assert manifest["resolution_provenance"][
        "native_quarter_hour_truth_eligible"
    ] is False
    assert not (output / "lt_input_snapshot.json").exists()
    assert (
        build_acquisition(
            capture_spec_path=spec_path,
            output_directory=output,
        )
        == manifest_path
    )


def test_unsigned_builder_resumes_its_exact_staging_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(
            _energy_capture_spec(),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    output = tmp_path / "unsigned-build"
    original_write = builder_module._write_new
    writes = 0

    def interrupt(path: Path, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("simulated interruption")
        original_write(path, payload)

    monkeypatch.setattr(builder_module, "_write_new", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        build_acquisition(capture_spec_path=spec_path, output_directory=output)
    assert not output.exists()

    monkeypatch.setattr(builder_module, "_write_new", original_write)
    manifest = build_acquisition(capture_spec_path=spec_path, output_directory=output)
    assert manifest.is_file()
    assert not list(tmp_path.glob(".unsigned-build.staging-*"))


def test_unsigned_builder_recovers_exact_hardlink_residue_after_process_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    output = tmp_path / "unsigned-build"
    original_rename = builder_module._rename_pinned_child

    def kill_before_rename(**_kwargs: object) -> None:
        raise OSError("simulated process kill before staging rename")

    monkeypatch.setattr(builder_module, "_rename_pinned_child", kill_before_rename)
    with pytest.raises(OSError, match="simulated process kill"):
        build_acquisition(capture_spec_path=spec_path, output_directory=output)
    staging_entries = list(tmp_path.glob(".unsigned-build.staging-*"))
    assert len(staging_entries) == 1
    staging = staging_entries[0]
    target = staging / "bronze.parquet"
    residue = staging / (
        f"{durable_module._temporary_prefix(target)}simulated-process-kill.tmp"
    )
    os.link(target, residue)
    assert target.stat(follow_symlinks=False).st_nlink == 2

    monkeypatch.setattr(builder_module, "_rename_pinned_child", original_rename)
    manifest = build_acquisition(capture_spec_path=spec_path, output_directory=output)

    assert manifest.is_file()
    assert not residue.exists()
    assert (output / "bronze.parquet").stat(follow_symlinks=False).st_nlink == 1


def test_exclusive_recovery_never_unlinks_divergent_hardlink_residue(
    tmp_path: Path,
) -> None:
    target = tmp_path / "bronze.parquet"
    target.write_bytes(b"divergent")
    residue = tmp_path / (
        f"{durable_module._temporary_prefix(target)}simulated-process-kill.tmp"
    )
    os.link(target, residue)

    with pytest.raises(ValueError, match="divergent hardlink residue"):
        durable_module.recover_durable_exact_bytes_residue(
            target,
            b"expected!",
            label="LT acquisition bronze",
            exclusive_writer_lease_proven=True,
        )

    assert target.read_bytes() == b"divergent"
    assert residue.exists()
    assert target.stat(follow_symlinks=False).st_nlink == 2


def test_unsigned_builder_rejects_oversized_body_before_loading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _energy_capture_spec()
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder_module, "MAX_PROVIDER_DOCUMENT_BYTES", 16)

    with pytest.raises(ValueError, match="exceeds the maximum size"):
        build_acquisition(
            capture_spec_path=spec_path,
            output_directory=tmp_path / "unsigned-build",
        )


def test_unsigned_builder_rejects_cumulative_body_budget_before_loading_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _energy_capture_spec()
    spec["documents"] = [dict(spec["documents"][0]), dict(spec["documents"][0])]
    spec["documents"][1]["document_id"] = "second_body"
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder_module, "MAX_PROVIDER_DOCUMENT_TOTAL_BYTES", 1)
    monkeypatch.setattr(
        builder_module,
        "validate_provider_document_inventory",
        lambda *_args, **_kwargs: (),
    )
    original_read = builder_module.read_stable_single_link_file
    body_reads: list[str] = []

    def record_read(path: Path, *, label: str, max_bytes: int | None = None) -> bytes:
        if label.startswith("provider body"):
            body_reads.append(label)
        return original_read(path, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(builder_module, "read_stable_single_link_file", record_read)

    with pytest.raises(ValueError, match="cumulative body budget"):
        build_acquisition(
            capture_spec_path=spec_path,
            output_directory=tmp_path / "unsigned-build",
        )
    assert body_reads == []


def test_unsigned_builder_validates_output_chain_before_first_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    output = tmp_path / "untrusted-parent" / "unsigned-build"
    original_assert = builder_module.assert_absolute_path_has_no_links

    def reject_output(path: Path) -> Path:
        selected = Path(path)
        if selected == output.absolute():
            raise ValueError("simulated parent junction")
        return original_assert(selected)

    monkeypatch.setattr(builder_module, "assert_absolute_path_has_no_links", reject_output)

    with pytest.raises(ValueError, match="simulated parent junction"):
        build_acquisition(capture_spec_path=spec_path, output_directory=output)
    assert not output.parent.exists()


def test_unsigned_builder_rejects_relative_capture_spec_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="capture specification path must be absolute"):
        build_acquisition(
            capture_spec_path=Path("capture-spec.json"),
            output_directory=tmp_path / "unsigned-build",
        )
    assert not (tmp_path / "unsigned-build").exists()


def test_unsigned_builder_rejects_relative_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="output directory must be absolute"):
        build_acquisition(
            capture_spec_path=spec_path,
            output_directory=Path("unsigned-build"),
        )
    assert not (tmp_path / "unsigned-build").exists()


def test_unsigned_builder_requires_preprovisioned_output_parent(tmp_path: Path) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    output = tmp_path / "missing-parent" / "unsigned-build"

    with pytest.raises(ValueError, match="output parent must be a pre-provisioned directory"):
        build_acquisition(capture_spec_path=spec_path, output_directory=output)
    assert not output.parent.exists()


@pytest.mark.parametrize("body_path", [r"~\body.json", r"C:body.json", r"\body.json"])
def test_unsigned_builder_rejects_lexically_relative_body_path(
    tmp_path: Path,
    body_path: str,
) -> None:
    spec = _energy_capture_spec()
    documents = spec["documents"]
    assert isinstance(documents, list)
    assert isinstance(documents[0], dict)
    documents[0]["body_path"] = body_path
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="body_path must be absolute"):
        build_acquisition(
            capture_spec_path=spec_path,
            output_directory=tmp_path / "unsigned-build",
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows path grammar contract")
@pytest.mark.parametrize(
    ("body_path", "message"),
    [
        (
            f"{(FIXTURES / 'energy_charts_price_ch.json').resolve()}:payload",
            "alternate data stream",
        ),
        (r"\\server\share\body.json", "local fixed drive"),
        (
            f"{(FIXTURES / 'energy_charts_price_ch.json').resolve()}.",
            "trailing-dot/space",
        ),
    ],
)
def test_unsigned_builder_rejects_windows_namespace_aliases(
    tmp_path: Path,
    body_path: str,
    message: str,
) -> None:
    spec = _energy_capture_spec()
    documents = spec["documents"]
    assert isinstance(documents, list)
    assert isinstance(documents[0], dict)
    documents[0]["body_path"] = body_path
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        build_acquisition(
            capture_spec_path=spec_path,
            output_directory=tmp_path / "unsigned-build",
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows DOS-device contract")
def test_unsigned_builder_rejects_subst_or_dos_device_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder_module,
        "_windows_drive_device_target",
        lambda _drive: r"\??\C:\redirected",
    )

    with pytest.raises(ValueError, match="directly to a physical local volume"):
        build_acquisition(
            capture_spec_path=spec_path,
            output_directory=tmp_path / "unsigned-build",
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows device-name contract")
@pytest.mark.parametrize(
    "component",
    ["CON", "nul.txt", "LPT9.audit", "COM¹.txt", "lpt³.audit"],
)
def test_unsigned_builder_rejects_reserved_windows_device_names(
    tmp_path: Path,
    component: str,
) -> None:
    with pytest.raises(ValueError, match="reserved Windows device name"):
        builder_module._assert_windows_local_canonical_path(
            tmp_path / component,
            label="test path",
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows control-character contract")
def test_unsigned_builder_rejects_windows_control_characters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Windows control character"):
        builder_module._assert_windows_local_canonical_path(
            tmp_path / "unsafe\x01name",
            label="test path",
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows filename grammar contract")
@pytest.mark.parametrize(
    ("component", "message"),
    [
        ("CAPTUR~1", "8.3 alias marker"),
        ("unsafe<name", "forbidden Windows filename character"),
        ("unsafe?name", "forbidden Windows filename character"),
    ],
)
def test_unsigned_builder_rejects_windows_alias_and_forbidden_characters(
    tmp_path: Path,
    component: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        builder_module._assert_windows_local_canonical_path(
            tmp_path / component,
            label="test path",
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-share contract")
def test_output_parent_handle_blocks_namespace_replacement(tmp_path: Path) -> None:
    parent = tmp_path / "quarantine"
    parent.mkdir()
    replacement_name = tmp_path / "quarantine-evicted"
    identity = builder_module._directory_identity(parent)

    with builder_module._pin_output_parent(parent, expected=identity):
        with pytest.raises(OSError):
            parent.rename(replacement_name)
    parent.rename(replacement_name)
    assert replacement_name.is_dir()


def test_staging_pin_proves_exclusive_writer_lease(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    identity = builder_module._directory_identity(staging)

    with builder_module._pin_directory(
        staging,
        expected=identity,
        delete_access=True,
    ) as first_pin:
        assert first_pin.exclusive_writer_lease is True
        with pytest.raises(OSError):
            with builder_module._pin_directory(
                staging,
                expected=identity,
                delete_access=True,
            ):
                pytest.fail("a second writer lease must not be admitted")


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-share contract")
def test_staging_handle_blocks_substitution_before_handle_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    output = tmp_path / "unsigned-build"
    evicted = tmp_path / "evicted-staging"
    original_rename = builder_module._rename_pinned_child
    attack_attempted = False

    def attempt_substitution(**kwargs: object) -> None:
        nonlocal attack_attempted
        attack_attempted = True
        child_pin = kwargs["child_pin"]
        assert isinstance(child_pin, builder_module._PinnedDirectory)
        with pytest.raises(OSError):
            child_pin.path.rename(evicted)
        original_rename(**kwargs)

    monkeypatch.setattr(builder_module, "_rename_pinned_child", attempt_substitution)
    manifest = build_acquisition(
        capture_spec_path=spec_path,
        output_directory=output,
    )

    assert attack_attempted is True
    assert manifest.is_file()
    assert not evicted.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-relative rename contract")
def test_handle_relative_rename_never_resolves_destination_from_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    output_parent = tmp_path / "quarantine"
    output_parent.mkdir()
    attacker_cwd = tmp_path / "attacker-cwd"
    attacker_cwd.mkdir()
    monkeypatch.chdir(attacker_cwd)

    manifest = build_acquisition(
        capture_spec_path=spec_path,
        output_directory=output_parent / "unsigned-build",
    )

    assert manifest.is_file()
    assert manifest.parent == output_parent / "unsigned-build"
    assert not (attacker_cwd / "unsigned-build").exists()


def test_builder_cli_emits_structured_fail_closed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "invalid-spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(builder_module, "assert_installed_runtime_sealed", lambda: None)

    rc = builder_module.main(
        [
            "--capture-spec",
            str(spec_path),
            "--output-directory",
            str(tmp_path / "unsigned-build"),
        ]
    )

    assert rc == 50
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "pfc-lt-build-acquisition"
    assert error["status"] == "INTEGRITY_FAILURE"
    assert error["retry_disposition"] == "QUARANTINE_OR_EXACT_RETRY_AFTER_REPAIR"
    assert error["production_authorization"] is False


@pytest.mark.slow
def test_admitted_wheel_builder_entrypoint_emits_unsigned_exact_artifacts(
    tmp_path: Path,
) -> None:
    wheel_value = os.environ.get("PFC_TEST_LT_WHEEL", "").strip()
    dependency_value = os.environ.get(
        "PFC_TEST_PUBLISHER_DEPENDENCY_ROOT",
        "",
    ).strip()
    if not wheel_value or not dependency_value:
        pytest.skip("an audited LT wheel and exact dependency root are required")
    wheel = Path(wheel_value)
    dependency_root = Path(dependency_value)
    install_root = tmp_path / "installed-wheel"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-compile",
            "--target",
            str(install_root),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert install.returncode == 0, install.stderr
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(_energy_capture_spec(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    output = tmp_path / "unsigned-build"
    isolated_cwd = tmp_path / "isolated-cwd"
    isolated_cwd.mkdir()
    code = (
        "import sys;"
        "sys.path[:0]=[sys.argv[1],sys.argv[2]];"
        "from pfc_shaping.cli.governed_acquisition_builder import main;"
        "raise SystemExit(main(sys.argv[3:]))"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(install_root),
            str(dependency_root),
            "--capture-spec",
            str(spec_path),
            "--output-directory",
            str(output),
        ],
        cwd=isolated_cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    manifest_path = Path(result.stdout.strip())
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["signing_status"] == "UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES"
    assert manifest["publication_status"] == "NOT_PUBLISHED"
    assert manifest["source_role"] == "epex_ch"


def _energy_charts_envelope(
    *,
    body: bytes | None = None,
    response_headers: dict[str, str] | None = None,
    received_at_utc: str = RECEIVED,
) -> bytes:
    return build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="epex_ch",
        source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
        source_locator=ENERGY_CHARTS_PRICE_URL,
        provider_id=ENERGY_CHARTS_SOURCE_SYSTEM,
        received_at_utc=received_at_utc,
        documents=[
            CapturedProviderDocument(
                document_id="day_ahead_price",
                request_url=ENERGY_CHARTS_PRICE_URL,
                request_parameters={
                    "bzn": "CH",
                    "start": "2026-07-01T00:00Z",
                    "end": "2026-07-01T23:00Z",
                },
                response_media_type="application/json",
                body=body or (FIXTURES / "energy_charts_price_ch.json").read_bytes(),
                response_headers=response_headers or {},
            )
        ],
    )


def _energy_capture_spec() -> dict[str, object]:
    return {
        "schema_version": "lt_provider_capture_spec.v1",
        "acquisition_id": "acquisition-001",
        "source_role": "epex_ch",
        "source_system": ENERGY_CHARTS_SOURCE_SYSTEM,
        "source_locator": ENERGY_CHARTS_PRICE_URL,
        "provider_id": ENERGY_CHARTS_SOURCE_SYSTEM,
        "received_at_utc": RECEIVED,
        "start_utc": START,
        "end_utc": EPEX_END,
        "documents": [
            {
                "document_id": "day_ahead_price",
                "body_path": str(
                    (FIXTURES / "energy_charts_price_ch.json").resolve()
                ),
                "request_url": ENERGY_CHARTS_PRICE_URL,
                "request_parameters": {
                    "bzn": "CH",
                    "start": "2026-07-01T00:00Z",
                    "end": "2026-07-01T23:00Z",
                },
                "credential_reference": None,
                "response_status_code": 200,
                "response_media_type": "application/json",
                "response_content_encoding": "identity",
                "response_headers": {"etag": '"fixture-v1"'},
            }
        ],
    }


def _entsoe_envelope(
    *,
    load_body: bytes | None = None,
    received_at_utc: str = RECEIVED,
) -> bytes:
    return build_raw_envelope(
        acquisition_id="acquisition-001",
        source_role="entso",
        source_system=ENTSOE_SOURCE_SYSTEM,
        source_locator=ENTSOE_API_URL,
        provider_id=ENTSOE_SOURCE_SYSTEM,
        received_at_utc=received_at_utc,
        documents=[
            CapturedProviderDocument(
                document_id="actual_load_ch",
                request_url=ENTSOE_API_URL,
                request_parameters=_entsoe_parameters(
                    "A65", "outBiddingZone_Domain"
                ),
                response_media_type="application/xml",
                body=load_body
                or (FIXTURES / "entsoe_actual_load_ch.xml").read_bytes(),
                credential_reference="env:ENTSOE_API_KEY",
            ),
            CapturedProviderDocument(
                document_id="actual_generation_ch",
                request_url=ENTSOE_API_URL,
                request_parameters=_entsoe_parameters("A75", "in_Domain"),
                response_media_type="application/xml",
                body=(FIXTURES / "entsoe_actual_generation_ch.xml").read_bytes(),
                credential_reference="env:ENTSOE_API_KEY",
            ),
        ],
    )


def _entsoe_parameters(document_type: str, domain_parameter: str) -> dict[str, str]:
    return {
        "documentType": document_type,
        "processType": "A16",
        domain_parameter: "10YCH-SWISSGRIDZ",
        "periodStart": "202607010000",
        "periodEnd": "202607010400",
    }


def _entsoe_chunk_documents(
    *,
    sequence: int,
    start: str,
    end: str,
    base: float,
) -> list[CapturedProviderDocument]:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    period_start = start_timestamp.strftime("%Y%m%d%H%M")
    period_end = end_timestamp.strftime("%Y%m%d%H%M")
    suffix = f"{sequence:04d}"
    return [
        CapturedProviderDocument(
            document_id=f"actual_load_ch_{suffix}",
            request_url=ENTSOE_API_URL,
            request_parameters={
                "documentType": "A65",
                "processType": "A16",
                "outBiddingZone_Domain": "10YCH-SWISSGRIDZ",
                "periodStart": period_start,
                "periodEnd": period_end,
            },
            response_media_type="application/xml",
            body=_entsoe_chunk_xml(
                start=start_timestamp,
                end=end_timestamp,
                generation=False,
                base=base,
            ),
            credential_reference="env:ENTSOE_API_KEY",
        ),
        CapturedProviderDocument(
            document_id=f"actual_generation_ch_{suffix}",
            request_url=ENTSOE_API_URL,
            request_parameters={
                "documentType": "A75",
                "processType": "A16",
                "in_Domain": "10YCH-SWISSGRIDZ",
                "periodStart": period_start,
                "periodEnd": period_end,
            },
            response_media_type="application/xml",
            body=_entsoe_chunk_xml(
                start=start_timestamp,
                end=end_timestamp,
                generation=True,
                base=base,
            ),
            credential_reference="env:ENTSOE_API_KEY",
        ),
    ]


def _entsoe_chunk_xml(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    generation: bool,
    base: float,
) -> bytes:
    start_text = start.strftime("%Y-%m-%dT%H:%MZ")
    end_text = end.strftime("%Y-%m-%dT%H:%MZ")

    def series(mrid: str, values: list[float], psr_type: str | None = None) -> str:
        psr = (
            f"<MktPSRType><psrType>{psr_type}</psrType></MktPSRType>"
            if psr_type is not None
            else ""
        )
        points = "".join(
            f"<Point><position>{position}</position><quantity>{value}</quantity></Point>"
            for position, value in enumerate(values, start=1)
        )
        return (
            f"<TimeSeries><mRID>{mrid}</mRID>"
            f"<businessType>{'A01' if generation else 'A04'}</businessType>"
            f"<objectAggregation>{'A08' if generation else 'A01'}</objectAggregation>"
            "<curveType>A01</curveType>"
            "<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>"
            f"{psr}<Period><timeInterval><start>{start_text}</start>"
            f"<end>{end_text}</end></timeInterval><resolution>PT60M</resolution>"
            f"{points}</Period></TimeSeries>"
        )

    if generation:
        namespace = "urn:iec62325.351:tc57wg16:451-6:generation-document:3:0"
        identity = (
            "<type>A75</type><process.processType>A16</process.processType>"
            "<in_Domain.mRID>10YCH-SWISSGRIDZ</in_Domain.mRID>"
        )
        time_series = series("solar-series", [10.0, 20.0], "B16") + series(
            "wind-series", [100.0, 90.0], "B19"
        )
    else:
        namespace = "urn:iec62325.351:tc57wg16:451-6:load-document:3:0"
        identity = (
            "<type>A65</type><process.processType>A16</process.processType>"
            "<outBiddingZone_Domain.mRID>10YCH-SWISSGRIDZ</outBiddingZone_Domain.mRID>"
        )
        time_series = series("load-series", [base, base + 100.0])
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<GL_MarketDocument xmlns="{namespace}"><mRID>fixture</mRID>'
        f"{identity}{time_series}</GL_MarketDocument>"
    ).encode("utf-8")
